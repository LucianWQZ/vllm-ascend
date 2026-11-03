# Speculative Decoding KV Sliding Window

Speculative decoding draft models are typically trained on short contexts
(e.g. 4-8k tokens). At inference time, when the running context grows far
beyond the training length, the draft's attention over the full KV cache goes
out-of-distribution and acceptance collapses: on GLM-5.2 with an early DSpark
draft, the mean acceptance length dropped from ~5 to ~1 at 32k context, making
speculation a net loss.

The KV sliding window caps the **draft model's attention** to the most recent
`draft_window_size` tokens. The target keeps its original attention and
verification semantics, including any native SWA; only eligible Draft reads
are shortened. On acceptance-collapsed drafts this
restores acceptance (the same GLM-5.2 draft recovered from ~1.0 to ~4.7-5.5
at 32k, up to +381% end-to-end throughput in the original evaluation).

## Configuration

The window is enabled through `additional_config` for supported
EAGLE3 / DFlash / DSpark draft consumers. The MRV2 default entry requires an
explicitly eager Draft model; limited, default-off Graph routes are
described below. Target Graph selection remains independent. See the MRV2
consumer and validation limits below before choosing a configuration.

```json
{"draft_window_size": 2048}
```

It is disabled by default (key absent). On MRV2, MTP and the config-identified
built-in co-trained DeepSeek-V4 DSpark category ignore the window with a
warning while retaining their original speculative method. Eagle3 follows
the switch, including co-trained Eagle3 models. Checkpoint path names do not
identify the V4 exception. The existing MRV1 adapter and its policy remain
separate from the MRV2 implementation described below.

## Window math

Let `W` be the window size, `b` the kernel block size, `K = future_offset`
the number of draft query positions lying beyond the `seq_lens` value the
adapter sees. The MRV1 adapter computes, entirely on device:

```text
start        = floor(clamp(seq_lens + K - W, min=0) / b) * b   # block-aligned window start
windowed_len = seq_lens - start                                # <= W + b - 1
start_block  = start // b
needed       = ceil(windowed_len / b)  (capped at max_window_blocks = W//b + 1)
```

`future_offset` differs by draft family because their `seq_lens` semantics
differ:

| Method | future_offset | Why |
| --- | --- | --- |
| EAGLE3 | `num_speculative_tokens` | `seq_lens` is context-only; the K draft positions lie beyond it, so the window end must cover `seq_lens + K`. |
| DFlash / DSpark | `0` | the input-preparation kernel already bakes the query stretch (bonus + mask tokens) into `seq_lens`, so `final = seq_lens`. |

The block table is cropped by gathering columns
`start_block[i] + [0, max_window_blocks)` from the full table into a
zero-padded clone; columns past `needed` (and past the full-table width) are
zeroed. Zeroed columns are never read because every FIA KV-length input is
capped at the same `windowed_len`.

Two invariants hold throughout:

- **slot_mapping is never windowed.** Draft KV writes still go through the
  full block table at absolute positions, so the cache stays coherent for the
  target model and for window sizes changed between steps.
- **Only what FIA reads is capped.** The FIA kernel's KV-length inputs are the
  NPU `seq_lens` tensor plus CPU mirrors (`seq_lens_cpu`, `_seq_lens_cpu`,
  `seq_lens_cpu_upper_bound`); all of them are windowed with the same
  arithmetic so no path can read past the window.

## MRV1 implementation

`SlidingWindowAdapter` (`vllm_ascend/worker/v1/spec_decode/utils.py`) owns the math
above and a persistent `[max_num_reqs, max_window_blocks]` clone buffer. It is
constructed in `AscendSpecDecodeBaseProposer.__init__`
(`vllm_ascend/worker/v1/spec_decode/llm_base_proposer.py`) when `draft_window_size` is
set, then applied to the draft's `common_attn_metadata` **before** the
per-layer metadata is built.

There are two apply points because of how the drafts construct metadata:

- EAGLE3 / DFlash: applied once in `_propose` on the shared
  `common_attn_metadata` (`llm_base_proposer.py`).
- DSpark: that path is bypassed - the dspark branch of
  `build_draft_attn_metadata` overwrites
  `common_attn_metadata.block_table_tensor` with a per-group full table before
  building. The adapter is therefore applied **after** the per-group overwrite
  and before `builder.build_for_drafting`, so FIA reads the windowed clone of
  the per-group table instead of the full one.

## MRV2 implementation

MRV2 resolves one method policy before allocating a Window owner. MTP and the
config-identified built-in DeepSeek-V4 DSpark category force Window OFF while
preserving their original speculative method. Plain Eagle and dynamic
speculative length are not supported with an extra Window. Checkpoint names
and missing configuration fields do not establish a supported consumer.

### Execution and support boundaries

The ordinary entry uses `speculative_config.enforce_eager=true`. Target Graph
is independent of that Draft setting. To select an admitted Draft Graph route,
set `VLLM_ASCEND_EXPERIMENTAL_DRAFT_WINDOW_GRAPH=1` and Draft
`enforce_eager=false`. Keep the Target in `FULL_DECODE_ONLY`. An absent
`draft_window_size` disables the extra Window; it does not disable native SWA.
No unsupported route silently falls back to eager.

The experimental Graph contract checks the actual normalized model,
projection/auxiliary geometry, backend, topology and capture capacity. It
applies to both additional Window OFF and ON when that experimental route is
selected. Ordinary Window-OFF eager creates no Window runtime and bypasses
these new Graph checks. Historical single/two-request, recorder, oracle and
matched-eager experiment flags are not product APIs.

| Qualified Draft consumer | Target TP / parent DP | Local request capacity | Query / extra Window |
| --- | --- | --- | --- |
| Qwen3.6 eight-layer Full DFlash | TP4 / DP1 | 1 or 2 | static K7/Q8; OFF or W32768 |
| Qwen3.6 eight-layer Full DFlash | TP4 / DP2 | 1 per replica | static K7/Q8; OFF or W32768 |
| Qwen3.6 Eagle3 | TP4 / DP1 | 1 or 2 | static K3; first Q4, AR Q1; OFF or W1024 |
| Qwen3.6 Eagle3 | TP4 / DP2 | 1 per replica | same first/AR geometry |
| GLM DSpark / GLM DFlash | TP16 | qualified model/executor configuration | actual prepared query width |
| Qwen3.8 DFlash2 | TP2 / DP1 | qualified static capture geometry | static K7/Q8; native P2048, OFF or W1024 |

This table does not qualify arbitrary checkpoints sharing a method name.
The Full DFlash and Eagle guards retain the known multi-request and padding
limits. Preempted/resumed requests and unqualified shapes are rejected before
Graph state changes. The specified six-layer mixed-SWA DFlash checkpoint is
blocked for an unresolved valid-output counterexample; the native-SWA
performance warning does not override this protection.

Request limits use actual per-DP active rows, capture padding, model capacity
and static-query lookahead. Former fixed prompt/output lengths are validation
points, not user APIs. In parent DP2, each replica owns independent requests,
Full KV and banks; the independent Draft descriptor remains TP4/DP1. Idle
replicas execute the original Full/PAD dummy for collectives without creating
a business Window proposal. Partial static-K Eagle Graph proposals are not
qualified: zero proposals still execute the first attention consumer, while
a full proposal uses its configured AR consumers. This is not dynamic-K support.

Actual applicable features are used together. Target SP/FlashComm does not
turn an independent dense Draft into a sharded reader. GLM legacy DSA-CP does
not apply to Qwen consumers. Target CP compatibility retains its padded
model-input count and separate actual query/write coordinates; no sparse
Draft SFA/indexer/DSA provider is added by this implementation. Unsupported
compressed, packed, transferred, offloaded or context-parallel Draft layouts
fail explicitly at provider initialization.

### Canonical reads and method preparation

The existing runner owns canonical Full tables, initialized block counts,
request slots, positions and write slots. Its normal preparation gathers the
read source into current batch order. Ordinary reads project that freshly
prepared table; Graph banks borrow `BlockTables.input_block_tables`. Initialized
counts are gathered from the canonical CPU state in persistent request-slot
order with that same current mapping, independent of rotating UVA transport.
No read projection clips a previous Window result.

`kernel_block_sizes[gid]` supplies the logical span of a table column, which
can differ from physical allocation blocks. One shared scalar implementation
computes the historical block-aligned start. Every participating group must
represent that start exactly; a new LCM alignment is not substituted.

DFlash and DSpark consume their actual prepared query width and current exact
end. Eagle3 freezes its proposal horizon and start at the first pass, then
updates the end through rejection-corrected AR preparation. Ordinary Eagle
requires `W >= 2*K+1` to cover its query width and configured rejection bound.
An unfinished prefill keeps Full reads. Prompt progress is computed once per
proposal from computed prompt tokens and scheduled tokens excluding Draft
placeholders.

Ordinary Eagle first-pass native extent includes rejected query padding. The
logical proposal end is the last valid absolute query position plus one; it
must not be replaced by that padded extent. The Runtime snapshots the live/native pair
from prepared positions and lengths together. Later AR consumers use their
current exact end and the frozen start. Query layout, absolute positions,
Full block writes and Target metadata retain their original semantics.

### Ordinary preparation and reader ownership

One proposal scope owns preparation, per-group metadata and the final reader
completion event. A single callback modifies fresh common metadata before the
existing Ascend builder. Host/native projections bind their read tables and
exact lengths. Dense preparation instead attaches its slot/mask read state,
clears Common CPU exact fields and exposes only workspace capacity to the
builder. Query/write operands remain unchanged. Enabled Eagle metadata is not
subsequently overwritten by the baseline Full-length AR update. Window OFF
keeps the original path.

All ordinary layouts use a current, completed exact snapshot. DSpark MLA can
reuse the CPU copy its existing preparation just completed; other paths may
perform a blocking device-to-CPU copy. The completed-request counts prepared
for the next Target input and optimistic scheduling uppers are not generally
the current rejection-corrected Draft lengths. For W1024/B128, E1151 starts at
zero, whereas upper1152 would incorrectly start at128.

The existing parallel selector can reuse the exact CPU list attached by the
Window preparation, avoiding a duplicate conversion. Ordinary MLA still uses
its original builder and read consumer; its mutable decode RoPE scratch is
copied before forward and retained with the proposal. This synchronization
and allocation cost remains visible in same-mode OFF/ON measurements. Async
scheduling is not disabled, and this implementation does not claim zero D2H
or complete preparation/decode overlap.

The ordinary dense reader receives explicit CPU exact ends, query ends,
frozen starts and Full-row flags from that preparation boundary. It builds
only initialized physical token addresses and the corresponding mask. Scratch
padding repeats the last valid token: masking an uninitialized KV tail alone
can still propagate NaNs. Packed/ragged queries retain explicit input and
output maps; mixed Full/Window rows use separate cohorts so short decodes do
not inherit a long prompt's allocation. Real page strides, nonzero K/V storage
offsets and non-ND format-aware gathering are preserved.

Native FP32 attention and the original fused ND gather/cast are retained for
the saved BF16 cancellation counterexample under the original tolerance.
The actual read owner selects one FP32 pair per geometry and submits the
unchanged kernel before attention. A subsequent layer reuses that pair in
stream order. This does not permit sharing one writable pair across proposals,
streams or concurrent consumers. The extra scratch and conversion cost is
part of the implementation, not an implied performance benefit.

Copy sources, pinned DMA sources, destinations and derived metadata remain
strongly held before possible submission. At pool capacity, completed owners
are reclaimed or the oldest submitted event is waited for. Free capacity
requires no global synchronization. Any uncertain preparation, record,
query or wait failure quarantines the pool and retains its backing. Host
return is not reader completion; a final event must cover every actual reader.

### Graph capture, UPDATE and replay

The existing method managers own capture descriptors, graphs, UPDATE and
replay. One provider indexes only the geometry supplied by the fixed capture
callbacks. Each actual consumer owns a persistent read table and CPU/device
exact lengths. Eagle's merged AR graph needs independent ordinal banks because
multiple consumers are staged before one replay. Those banks stay strongly
owned even though captured framework resources can be weak references.

A proposal provides its real phase rows, current exact lengths and persistent
request mapping. Preparation validates all readable intervals before modifying
a bank, zeros inactive rows, and writes directly into the final capture-owned
buffers. The original metadata builder supplies the baseline metadata;
shallow private headers preserve layer aliases and leave Target objects untouched.
Eagle capture binds its owned Full query coordinates and sets `causal=True`. Its native
first consumer retains the prepared query layout and uses `ChunkedPrefill` with
the bank's GPU lengths. DFlash preserves its per-group causality and also binds
those GPU lengths. UPDATE supplies each descriptor's cumulative query lengths.

The existing `ParamSource` resolves each consumer's table and exact-length
arguments for UPDATE. Bank addresses remain stable. Current/update-stream
ordering, ExternalEvent dependencies and the original graph completion
notification/FIFO protect reuse. UPDATE return, leaving a Host proposal scope
and HTTP completion are not substitutes for the last reader. An uncertain
partial submission quarantines the provider without dropping its banks.

Capture dummy inputs, normal padding and initialization remain part of the
framework. Normal execution needs no oracle, tensor dump, experimental worker,
source hash, private evidence file or injected business request.

### Native SWA and DFlash2

Startup detection uses the actual Draft consumer implementations, normalized
architecture/layer count and actual native window/mask. It distinguishes Full,
SWA, mixed and unknown. Full KV retention alone does not establish Full
attention; a missing field does not establish that a model has no SWA.
Unknown or conflicting semantics reject an unqualified extra Window or new
DFlash2 Graph route, while leaving the original legal Window-OFF path intact.
A narrowing extra Window produces one startup warning. It is explicit and
never enabled automatically; acceptance or throughput can decrease.

Ordinary causal native-SWA groups retain their native band and Full writes.
If an extra Window covers every possible native read, an eligible mixed-model
native group can remain unchanged. Otherwise preparation intersects its read
interval with `[S,E)` before the existing builder. Its exact completed length
is reused and any H2D source is pinned and retained. Unsupported nonresident
layouts and additional padded request rows remain rejected.

DFlash2 is `DFlash2DraftModel` with `method="dflash"`. Its candidate Top-K,
convolution, selector, path walk, sampling and probability cache retain their
original implementation. With native noncausal SWA, a query at absolute p
reads `[max(0,p-(P-1)), min(E,p+(P-1)+1))`. The loader binds the native mask;
FIA uses sparse mode4 and symmetric pre/next P-1. Window ON intersects that
range with `[S,E)`; it does not make the model causal.

The DFlash2 cache-spec hook keeps those Draft groups in Full KV storage to
avoid stale pages in native paged reads. Native P is preserved in the executing
attention implementation. Target/GDN cache specifications and unrelated
hybrid-cache policies remain unchanged. A noncausal native-only extra-Window
provider requires `W + B - 1 + Q <= P`, using its actual column span and query
width. These safeguards also apply when the new Graph path uses extra Window ON.

The DFlash2 Graph route uses static K7/Q8, Target/Draft TP2 and DP1 with the
explicit experimental switch. FlashComm, SP, CP and other unqualified variants
remain rejected. Capture reuses the original forward/convolution, context-KV
preparation, selector/walk and probability-cache operations. CPU request
preparation and parameter resolution stay outside capture. A context-only
zero-proposal call still completes its actual Full KV writes and ownership;
no nonexistent Draft attention is demanded. Eager and Graph do not co-own the
same mutable Window state.

The verified five-layer P2048 Qwen3.8 deployment retains its disclosed capacity
tradeoff: at the same 8GiB/rank KV budget, estimated cache capacity changed
from179,973 to167,573 tokens (about6.89% lower). Exact capacity depends on the
engine configuration. Extra W1024 is not currently recommended as a speedup
setting for this checkpoint. Its original finite numerical tolerance failures
remain separately recorded; successful requests do not convert them to PASS.

### Validation and measurement

Use an independent same-window reference with comparable valid rows,
positions, shape and KV prestate. Check Full writes, Target isolation, request
mapping, padding and safe reader reuse separately from service success.
Natural generation differences across scheduling or Window settings are not
by themselves a numerical counterexample, and HTTP success is not an
independent numerical proof. Source/API ordering, actual binding/ownership
and independent numerical evidence support the stated engineering scope;
hardware operand/read-set observation is not claimed.

Validate all actually applicable features together rather than their switch
permutations. Formal timing excludes intrusive numerical tools and uses the
same code, model, topology, physical cards, Graph mode, other features, inputs,
output budget, sampling, warmup and cache policy. Only extra Window OFF/ON
changes; OFF retains native SWA. Report throughput, TTFT, TPOT, per-request and
group E2E, mean acceptance length and accepted/drafted counts, including
negative results and interference. Same-candidate OFF/ON does not establish
absence of regression against an independent official baseline.

Window limits reads, not Full KV storage. Account for model/cache capacity,
private dense scratch and mixed prefill/decode allocations. A smaller read
view alone promises neither capacity nor speedup. Numerical limits, normal
service, performance and release approval remain separate conclusions.

### Related MRV2 files

- `worker/v2/spec_decode/draft_window_policy.py` and `draft_window_plan.py`: method policy and interval math.
- `draft_window_runtime.py`, `draft_window_read.py`, `draft_window_dense.py`, `draft_window_mla.py`: ordinary preparation and reads.
- `draft_window.py`, `draft_window_graph.py`, `draft_window_eager.py`: Graph source geometry, banks and proposal inputs.
- `draft_window_eagle3*` and the existing DFlash/AR managers: method-specific capture/replay integration.
- `draft_window_backends.py`, `draft_window_native_attention.py`, `draft_graph_review_limits.py`: real consumer inventory and support boundaries.
- `worker/v2/attn_utils.py`: the existing metadata-builder seam.
