#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Repair4 controlled NPU FIA microfixture; never a full-model readiness gate.

Examples (run these in the server's vLLM/Ascend Python environment)::

    python tools/p7x_graph_repair4_fixture.py run --dry-run
    python tools/p7x_graph_repair4_fixture.py run --run-id r4 --attempt 1 --output /tmp/r4-1
    python tools/p7x_graph_repair4_fixture.py check /tmp/r4-1/result.json --manifest /tmp/r4-1/manifest.json

Only ``run --dry-run`` and ``check`` are host operations. ``run`` always starts
a fresh NPU process, requires torch_npu 2.10.0.post4 / CANN 9.1, and never falls
back to CPU or a replacement graph. The child uses the real full_graph_fia,
update_full_graph_params and AscendAttentionBackendImpl.update_graph_params.
It uses P3 materialization and the P6 provider with fixture-owned FullSource.
It does NOT use the production planner, model runner, DFlash capture-input
helper, KV write path, or a target model. G4F remains BLOCKED, even on success.

The v2.10.0 branch is NOT an exact post4 source pin. Runtime evidence includes
the installed Python API source, file hash, native extension hash and FIA
schema. Native task timing is not inferred from Python call order. The measured
output is compared with an independent token-by-token full-table oracle.

G5 synchronization is diagnostic, outside capture and only in this fixture.
An ordinary Event after the last graph consumer proves its completion; a
separate ordinary Event proves update-stream completion. ExternalEvent is used
only by production capture/update. In particular it is never queried. A replay
whose update failed may be waiting forever: do not fake an event record, drain,
reset, release, or reuse it. Write an abort record and exit the disposable child
without Python/NPU teardown; the parent also enforces a timeout. Process death
is NOT reported as verified device completion.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import signal
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import regex as re

if TYPE_CHECKING:
    from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "p7x-repair4-fia-fixture-v1"
MANIFEST_SCHEMA = "p7x-repair4-fia-manifest-v1"
BLOCK_SIZE = 128
NUM_BLOCKS = 32
HEAD_SIZE = 64
QUERY_WIDTH = 7
NUM_HEADS = 2
WINDOW_SIZE = 1024
CAPACITY_TOKENS = WINDOW_SIZE + BLOCK_SIZE - 1
TABLE_WIDTH = (CAPACITY_TOKENS + BLOCK_SIZE - 1) // BLOCK_SIZE
FULL_TABLE_WIDTH = 12
ABS_TOL = 0.016
# Pre-device acceptance budget, not a claim about native FIA error: normal
# values <32 have FP16 half-ULP <=1/128 at input and output (sum <=0.015625).
# The sentinel's larger input error is diluted by 1/1025 in the sole tail case.
TORCH_NPU_VERSION = "2.10.0.post4"
LAYER = "draft.layers.0.self_attn"
SOURCE_FILES = (
    "tools/p7x_graph_repair4_fixture.py",
    "vllm_ascend/attention/attention_v1.py",
    "vllm_ascend/compilation/acl_graph.py",
    "vllm_ascend/worker/v2/spec_decode/draft_window_graph.py",
    "vllm_ascend/worker/v2/spec_decode/draft_window.py",
    "vllm_ascend/worker/v2/spec_decode/draft_window_graph_characterization.py",
    "vllm_ascend/worker/v2/spec_decode/draft_window_graph_oracle.py",
    "vllm_ascend/worker/v2/spec_decode/dflash/aclgraph.py",
)
API_REFERENCES = {
    "graphs_branch_only": "https://github.com/Ascend/pytorch/blob/v2.10.0/torch_npu/npu/graphs.py",
    "events_branch_only": "https://github.com/Ascend/pytorch/blob/v2.10.0/torch_npu/npu/streams.py",
    "cann_version": (
        "https://gitee.com/ascend/pytorch/blob/2b4108588b7bdf5b1c1c30d9b81004ba637605d1/torch_npu/npu/utils.py"
    ),
}
BLOCKERS = {
    "G4F": "BLOCKED: no target model/head/logits or controlled target execution",
    "production_capture_inputs": "PARTIAL: fixture metadata; DFlash capture-input helper/model runner not exercised",
    "production_write": "NOT_EXERCISED: prepopulated read-only KV; no reshape_and_cache or target writes",
    "native_task_timeline": "OPEN: numerical effective inputs only; no native task/update-generation trace",
    "physical_reads": "OPEN: no device memory-transaction trace",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def source_identity() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()

    return {
        "head": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        # HEAD alone does not identify a dirty checkout or this untracked tool.
        "files": {name: file_digest(ROOT / name) for name in SOURCE_FILES},
    }


def cases() -> list[dict[str, Any]]:
    """Literal device controls, deliberately not constructed by the oracle."""
    a = {
        "full_table": list(range(1, 13)),
        "exact": 1152,
        "horizon": 1152,
        "upper": 1152,
        "window": WINDOW_SIZE,
        "source_column": 1,
        "table": list(range(2, 10)) + [0],
        "length": 1024,
    }
    b = {
        "full_table": [10, 11, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
        "exact": 1280,
        "horizon": 1280,
        "upper": 1280,
        "window": WINDOW_SIZE,
        "source_column": 2,
        "table": list(range(17, 25)) + [0],
        "length": 1024,
    }
    short = {
        "full_table": list(range(1, 13)),
        "exact": 64,
        "horizon": 64,
        "upper": 64,
        "window": WINDOW_SIZE,
        "source_column": 0,
        "table": [1] + [0] * 8,
        "length": 64,
    }
    nonaligned = {
        "full_table": list(range(1, 10)) + [31, 11, 12],
        "exact": 1153,
        "horizon": 1153,
        "upper": 1153,
        "window": WINDOW_SIZE,
        "source_column": 1,
        "table": list(range(2, 10)) + [31],
        "length": 1025,
    }
    return [
        dict(control, label=label, generation=index + 1, epoch=index + 1)
        for index, (label, control) in enumerate(
            (("A", a), ("B", b), ("A_prime", a), ("short", short), ("nonaligned", nonaligned))
        )
    ]


def value_at(page: int, offset: int, channel: int) -> float:
    # High tail sentinel makes a single stale-length token observable in FP16.
    return (512 if page == 31 else page) + offset / 256 + channel / 256


def full_table_oracle(case: dict[str, Any]) -> dict[str, Any]:
    """Enumerate full logical tokens, without importing any production planner.

    Q=K=0, so noncausal softmax is uniform. V[page,offset,0,d] is
    value_at(page, offset, d), with the page-31 tail sentinel rounded to FP16
    by the device. Nonaligned uses only offset zero, for which the expected
    mean stays well within ABS_TOL despite sentinel rounding.
    This gives a numeric oracle as well as the independently derived read set.
    """
    for key in ("exact", "horizon", "upper", "window"):
        require(type(case[key]) is int, f"{key} must be an integer")
    exact, horizon, upper, window = (case[k] for k in ("exact", "horizon", "upper", "window"))
    table = case["full_table"]
    require(isinstance(table, list) and bool(table), "full table must be nonempty")
    require(all(type(p) is int and 0 <= p < NUM_BLOCKS for p in table), "invalid full-table physical page")
    require(0 < exact <= horizon and upper >= exact and window > 0, "invalid oracle horizon/length")
    start = max(horizon - window, 0) // BLOCK_SIZE * BLOCK_SIZE
    require(start < exact <= len(table) * BLOCK_SIZE, "source shortage or empty window")
    tokens = [(table[pos // BLOCK_SIZE], pos % BLOCK_SIZE) for pos in range(start, exact)]
    columns = list(range(start // BLOCK_SIZE, (exact + BLOCK_SIZE - 1) // BLOCK_SIZE))
    require(len(columns) <= TABLE_WIDTH, "window exceeds fixture table capacity")
    row = [sum(value_at(page, offset, d) for page, offset in tokens) / len(tokens) for d in range(HEAD_SIZE)]
    return {
        "start": start,
        "length": len(tokens),
        "columns": columns,
        "table": [table[c] for c in columns] + [0] * (TABLE_WIDTH - len(columns)),
        "upper": min(upper - start, len(columns) * BLOCK_SIZE),
        "output": [[list(row) for _ in range(NUM_HEADS)] for _ in range(QUERY_WIDTH)],
    }


def scope() -> dict[str, Any]:
    return {
        "device": "NPU only; explicit run command; no CPU fallback",
        "capture": "AscendAttentionBackendImpl.full_graph_fia",
        "update": "update_full_graph_params -> AscendAttentionBackendImpl.update_graph_params",
        "binding": "FullDraftKVSource -> P3.materialize -> P6.bind/consume -> build_draft_window_graph_update_binding",
        "geometry": {
            "q": [QUERY_WIDTH, NUM_HEADS, HEAD_SIZE],
            "kv": [NUM_BLOCKS, BLOCK_SIZE, 1, HEAD_SIZE],
            "table": [1, TABLE_WIDTH],
            "dtype": "torch.float16",
            "causal": False,
            "layout": "TND",
            "window": WINDOW_SIZE,
            "capacity_tokens": CAPACITY_TOKENS,
        },
        "sequence": "stage -> update.wait_stream(replay) -> replay -> consumer Event -> update -> update Event",
        "completion": "ordinary update Event and last-consumer Event synchronize+query outside capture",
        "oracle": "independent full-table token enumeration; A != B and A_prime == A",
        "numerical_acceptance": {
            "atol": ABS_TOL,
            "rtol": 0,
            "dtype": "float16",
            "reason": "predeclared input/output quantization budget; native FIA accuracy must pass on device",
        },
        "outside_window_only_perturbation": "NOT_IMPLEMENTED",
        "blockers": dict(BLOCKERS),
        "ready": False,
        "api_references": dict(API_REFERENCES),
        "source_policy": "installed post4 sources/schema/native binary hashes required; branch URL is not a post4 pin",
    }


def make_manifest(run_id: str, attempt: str, device: str, abort_at: str) -> dict[str, Any]:
    return {
        "schema": MANIFEST_SCHEMA,
        "run_id": run_id,
        "attempt": attempt,
        "nonce": str(uuid.uuid4()),
        "device": device,
        "abort_at": abort_at,
        "source": source_identity(),
        "scope": scope(),
        "cases": cases(),
    }


def validate_manifest(manifest: dict[str, Any]) -> None:
    require(manifest["schema"] == MANIFEST_SCHEMA, "wrong manifest schema")
    require(
        all(isinstance(manifest[k], str) and manifest[k] for k in ("run_id", "attempt", "nonce")),
        "missing run/attempt/challenge identity",
    )
    require(str(uuid.UUID(manifest["nonce"])) == manifest["nonce"], "invalid challenge nonce")
    require(re.fullmatch(r"npu:\d+", manifest["device"]) is not None, "device must be explicit npu:N")
    require(
        digest(manifest["cases"]) == digest(cases()) and digest(manifest["scope"]) == digest(scope()),
        "stale fixture shape/schema/controls",
    )
    require(manifest["source"] == source_identity(), "source identity mismatch, including dirty files")


@dataclass
class CompletionGuard:
    """A failed generation permanently poisons this fixture, even after a drain."""

    state: str = "idle"
    generation: int = 0
    completion_verified: bool = True

    def stage(self, generation: int) -> None:
        require(
            self.state in ("idle", "complete") and self.completion_verified,
            "unsafe reuse rejected: last consumer completion is not verified or fixture is aborted",
        )
        require(generation == self.generation + 1, "stale generation rejected")
        self.generation = generation
        self.state = "staged"
        self.completion_verified = False

    def submitted(self) -> None:
        require(self.state == "staged", "replay requires staged generation")
        self.state = "submitted"

    def updated(self) -> None:
        require(self.state == "submitted", "update requires submitted replay")
        self.state = "updated"

    def completed(self) -> None:
        require(self.state == "updated", "completion requires both update and consumer boundaries")
        self.state = "complete"
        self.completion_verified = True

    def abort(self) -> dict[str, Any]:
        previous = self.state
        self.state = "aborted"
        rejected = False
        try:
            self.stage(self.generation + 1)
        except ValueError:
            rejected = True
        return {
            "at": previous,
            "generation": self.generation,
            "completion_verified": self.completion_verified,
            "unsafe_reuse_rejected": rejected,
            "reused": False,
        }


def execute_cases(adapter: Any, manifest: dict[str, Any], document: dict[str, Any], checkpoint: Any) -> None:
    """Shared ordering/state logic; test adapters are NOT a device-run backend."""
    guard = CompletionGuard()
    completion_attempted = False
    try:
        for case in manifest["cases"]:
            completion_attempted = False
            guard.stage(case["generation"])
            adapter.stage(case)
            checkpoint("staged", guard)
            if manifest["abort_at"] == "before_replay":
                raise RuntimeError("injected abort before replay")
            # Mark before the API: a throwing replay may have partially submitted.
            guard.submitted()
            checkpoint("replay_may_be_submitted", guard)
            adapter.replay()
            if manifest["abort_at"] == "after_replay":
                raise RuntimeError("injected abort after replay; no completion claim")
            adapter.update()
            guard.updated()
            checkpoint("updated", guard)
            if manifest["abort_at"] == "after_update":
                raise RuntimeError("injected abort after update")
            completion_attempted = True
            completion = adapter.complete()
            guard.completed()
            checkpoint("last_consumer_completed", guard)
            if manifest["abort_at"] == "after_completion":
                raise RuntimeError("injected abort after completion")
            document["records"].append(adapter.snapshot(case, completion))
        document["status"] = "COMPLETE"
    except BaseException as error:
        # Only attempt completion when the backend returned through event.record.
        # Never record a substitute ExternalEvent or wait on an un-updated replay.
        if guard.state == "updated" and not completion_attempted:
            try:
                document["abort_completion"] = adapter.complete()
                guard.completed()
            except BaseException as completion_error:
                document["completion_error"] = f"{type(completion_error).__name__}: {completion_error}"
        document["status"] = "ABORTED"
        document["error"] = f"{type(error).__name__}: {error}"
        document["abort"] = guard.abort()


def tensor_spec(tensor: Any) -> dict[str, Any]:
    return {
        "address": tensor.data_ptr(),
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "offset": tensor.storage_offset(),
    }


def api_source(function: Any) -> dict[str, Any]:
    source_file = inspect.getsourcefile(function)
    if source_file is None:
        raise ValueError("API source file unavailable; exact source evidence is required")
    lines, line = inspect.getsourcelines(function)
    path = Path(source_file).resolve()
    return {
        "path": str(path),
        "line": line,
        "file_sha256": file_digest(path),
        "source": "".join(lines),
        "signature": str(inspect.signature(function)),
    }


def runtime_evidence(torch: Any, torch_npu: Any) -> dict[str, Any]:
    require(torch_npu.__version__.split("+")[0] == TORCH_NPU_VERSION, "requires torch_npu 2.10.0.post4")
    require(torch.__version__.split("+")[0] == "2.10.0", "requires torch 2.10.0")
    require(torch.npu.is_available() is True, "NPU unavailable; CPU fallback is forbidden")
    cann = torch_npu.npu.get_cann_version()
    require(isinstance(cann, str) and re.match(r"^9\.1(?:\.|$)", cann) is not None, "requires reported CANN 9.1")
    functions = {
        "graph_task_group_begin": torch.npu.graph_task_group_begin,
        "graph_task_group_end": torch.npu.graph_task_group_end,
        "graph_task_update_begin": torch.npu.graph_task_update_begin,
        "graph_task_update_end": torch.npu.graph_task_update_end,
        "NPUGraph.replay": torch.npu.NPUGraph.replay,
        "Event.record": torch.npu.Event.record,
        "Event.synchronize": torch.npu.Event.synchronize,
        "Event.query": torch.npu.Event.query,
        "ExternalEvent.wait": torch.npu.ExternalEvent.wait,
        "ExternalEvent.reset": torch.npu.ExternalEvent.reset,
        "ExternalEvent.record": torch.npu.ExternalEvent.record,
        "get_cann_version": torch_npu.npu.get_cann_version,
    }
    schema = str(torch.ops.npu.npu_fused_infer_attention_score.out._schema)
    require(
        all(name in schema for name in ("block_table", "actual_seq_lengths_kv", "workspace")),
        "unsupported FIA out schema",
    )
    return {
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "cann": cann,
        "api_sources": {name: api_source(fn) for name, fn in functions.items()},
        "native_extension_sha256": file_digest(Path(torch_npu._C.__file__)),
        "fia_schema": schema,
    }


class ProductionBinding:
    """Fixture owners, real P3/P6 code. No model/speculator owner is fabricated."""

    def __init__(self, fixture: Any) -> None:
        from vllm_ascend.attention.attention_v1 import AscendAttentionBackend, AscendAttentionMetadataBuilder
        from vllm_ascend.worker.v2.spec_decode import draft_window as window
        from vllm_ascend.worker.v2.spec_decode import draft_window_graph as graph

        torch = fixture.torch
        self.window, self.graph, self.fixture = window, graph, fixture
        self.route = graph.DraftWindowGraphRoute.DSPARK
        counts = torch.full((1, 1), FULL_TABLE_WIDTH, dtype=torch.int32)
        self.owner = SimpleNamespace(
            block_sizes=[BLOCK_SIZE],
            kernel_block_sizes=[BLOCK_SIZE],
            input_block_tables=[fixture.full_table],
            slot_mappings=fixture.slots.view(1, QUERY_WIDTH),
            num_blocks=SimpleNamespace(cpu=counts, gpu=counts.to(fixture.device)),
        )
        group = window.DraftWindowGroupInventory(
            global_group_id=0,
            local_consumer_index=0,
            layer_names=(LAYER,),
            backend_class=AscendAttentionBackend,
            builder_class=AscendAttentionMetadataBuilder,
            logical_block_size=BLOCK_SIZE,
            kernel_block_size=BLOCK_SIZE,
            block_span_tokens=BLOCK_SIZE,
            causal=False,
            layout_category=window.DraftWindowLayoutCategory.DENSE_GQA_DIRECT_PREFIX,
            capability_failures=(),
        )
        inventory = window.DraftWindowInventory(
            method="dspark",
            target_model_type="fixture-no-target",
            runtime_route="dspark",
            window_size=WINDOW_SIZE,
            groups=(group,),
            empty_global_group_ids=(),
            capability_failures=(),
        )
        full_source = window.build_full_draft_kv_source(inventory=inventory, block_tables=self.owner)
        capability = window.DraftWindowSourceCapability(
            inventory=inventory,
            full_source=full_source,
            provider=window.SingleGroupDirectPrefixProvider(
                inventory=inventory, full_source=full_source, legacy_block_size=BLOCK_SIZE, expected_future_offset=0
            ),
        )
        self.runtime = window.build_draft_window_runtime_state(capability)
        self.owner._draft_window_runtime_state = self.runtime
        self.provider = graph.install_draft_window_graph_binding_provider(self.owner)
        fingerprint = graph.DraftWindowGraphRouteFingerprint(
            method="dspark",
            route=self.route,
            graph_mode="FULL",
            backend_impl="vllm_ascend.attention.attention_v1.AscendAttentionBackendImpl",
            builder_impl="vllm_ascend.attention.attention_v1.AscendAttentionMetadataBuilder",
            global_group_id=0,
            layout=group.layout_category.value,
            capture_descriptor=f"repair4:tokens={QUERY_WIDTH},reqs=1,mode=FULL",
            capture_bucket=(1, QUERY_WIDTH),
            query_width=QUERY_WIDTH,
            padded_request_count=1,
            padded_token_count=QUERY_WIDTH,
            static_k=QUERY_WIDTH - 1,
            window_size=WINDOW_SIZE,
            logical_block_size=BLOCK_SIZE,
            kernel_block_size=BLOCK_SIZE,
            block_span_tokens=BLOCK_SIZE,
            tensor_parallel_size=1,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            sequence_parallel=False,
            dsa_context_parallel=False,
            flashcomm=False,
            async_scheduling=False,
            prefix_caching=False,
        )
        self.capture_identity = self.provider.register_capture_identity(
            fingerprint=fingerprint, execution_lane=0, consumer_ordinal=0
        )
        self.startup = self.provider.startup_capture_read_inputs(self.capture_identity)
        require(list(self.startup.window_block_table.shape) == [1, TABLE_WIDTH], "P6 capacity mismatch")
        fixture.table, fixture.length = self.startup.window_block_table, self.startup.device_window_exact
        self.mapping = torch.tensor([0], dtype=torch.int32, device=fixture.device)
        self.target_sentinel = {"fixture_isolation_sentinel": torch.tensor([-771], device=fixture.device)}
        self.write_side = graph.DraftWindowGraphWriteSideReference(
            slot_mapping=fixture.slots,
            positions=fixture.positions,
            query_start_loc=fixture.query_locs,
            actual_seq_lengths_q=(QUERY_WIDTH,),
            attn_state=fixture.metadata.attn_state,
            num_actual_tokens=QUERY_WIDTH,
            causal=False,
        )

    def build_metadata(self, read: Any) -> dict[str, Any]:
        from vllm_ascend.attention.attention_v1 import AscendMetadata

        metadata = AscendMetadata(
            block_tables=read.window_block_table,
            seq_lens=read.device_window_exact,
            seq_lens_cpu=read.cpu_window_exact,
            seq_lens_list=read.cpu_window_exact.tolist(),
            actual_seq_lengths_q=list(self.write_side.actual_seq_lengths_q),
            slot_mapping=self.write_side.slot_mapping,
            query_start_loc=self.write_side.query_start_loc,
            attn_state=self.write_side.attn_state,
            num_actual_tokens=QUERY_WIDTH,
            causal=False,
        )
        metadata.positions = self.write_side.positions
        return {LAYER: metadata}

    def bind(self, case: dict[str, Any]) -> None:
        torch = self.fixture.torch
        required = (case["length"] + BLOCK_SIZE - 1) // BLOCK_SIZE
        descriptor = self.window.DraftWindowReadDescriptor(
            global_group_id=0,
            source_start_column=case["source_column"],
            first_token_offset=0,
            required_columns=required,
            safe_extent_tokens=required * BLOCK_SIZE,
            valid_num_blocks=FULL_TABLE_WIDTH,
            source_row_width=FULL_TABLE_WIDTH,
        )
        exact = torch.tensor([case["length"]], dtype=torch.int32)
        self.inputs = self.runtime.materialize(
            mapping_epoch=case["epoch"],
            ordered_request_ids=("repair4-request",),
            idx_mapping=self.mapping,
            read_descriptors=(descriptor,),
            device_window_exact=None,
            cpu_window_exact=exact,
            cpu_window_upper=exact,
            cpu_valid_num_blocks=torch.tensor([FULL_TABLE_WIDTH], dtype=torch.int32),
            num_reqs_padded=1,
            phase=self.window.DraftWindowExecutionPhase.PROPOSAL,
            step=0,
            execution_lane=0,
            row_phase_fingerprint=("decode",),
        )
        # P3 generation IDs start at zero; artifact generations start at one.
        require(self.inputs.identity.generation_id == case["generation"] - 1, "P3 generation drift")
        self.carrier = self.provider.bind_read_carrier(
            capture_identity=self.capture_identity,
            inputs=self.inputs,
            proposal_id=case["generation"],
            metadata_builder=self.build_metadata,
            target_metadata=self.target_sentinel,
            write_side_reference=self.write_side,
        )
        self.batch = list(
            self.graph.consume_draft_window_graph_metadata(self.owner, route=self.route, expected_consumer_count=1)
        )
        self.graph.assert_draft_window_graph_metadata_batch(
            self.owner,
            route=self.route,
            draft_attn_metadatas=self.batch,
            num_reqs=1,
            num_tokens=QUERY_WIDTH,
            graph_mode="FULL",
        )
        self.update_binding = self.graph.build_draft_window_graph_update_binding(
            self.owner, route=self.route, draft_attn_metadatas=self.batch
        )
        self.fixture.metadata = self.batch[0][LAYER]
        # Exercise the real P6 early-rebind rejection while SUBMITTED. It fails
        # before copying storage. Never attempt this after UPDATE allows rebinding.
        self.reuse_rejected = False
        try:
            self.provider.bind_read_carrier(
                capture_identity=self.capture_identity,
                inputs=self.inputs,
                proposal_id=case["generation"],
                metadata_builder=self.build_metadata,
                target_metadata=self.target_sentinel,
                write_side_reference=self.write_side,
            )
        except RuntimeError as error:
            require("cannot overwrite an active" in str(error), "unexpected P6 rejection")
            self.reuse_rejected = True
        require(self.reuse_rejected, "P6 accepted unsafe early reuse")

    def retire_after_completion(self) -> dict[str, Any]:
        require(self.carrier.backend_attribution is not None, "real backend update attribution missing")
        self.runtime.retire_generation(self.inputs.identity)
        return {
            "provider_schema": self.capture_identity.provider_schema_version,
            "provider_id": id(self.provider),
            "p3_generation": self.inputs.identity.generation_id,
            "epoch": self.inputs.identity.mapping_epoch,
            "unsafe_reuse_rejected": self.reuse_rejected,
            "retired_after_consumer_completion": True,
            "update_function": self.carrier.backend_attribution.update_function_path,
        }


class NPUFixture:
    """Real production backend methods with a small, explicit fixture context."""

    def __init__(
        self,
        torch: Any,
        backend: Any,
        metadata: Any,
        config: Any,
        context: Any,
        graph_module: Any,
        manifest: dict[str, Any],
        document: dict[str, Any],
    ) -> None:
        self.torch, self.backend, self.metadata = torch, backend, metadata
        self.config, self.context, self.graph_module = config, context, graph_module
        self.manifest, self.document = manifest, document
        self.device = manifest["device"]
        torch.npu.set_device(self.device)
        self.replay_stream = torch.npu.Stream(device=self.device)
        self.update_stream = torch.npu.Stream(device=self.device)
        self.consumer_done, self.update_done = torch.npu.Event(), torch.npu.Event()
        self.graph = torch.npu.NPUGraph()
        self.q = torch.zeros((QUERY_WIDTH, NUM_HEADS, HEAD_SIZE), dtype=torch.float16, device=self.device)
        self.k = torch.zeros((NUM_BLOCKS, BLOCK_SIZE, 1, HEAD_SIZE), dtype=torch.float16, device=self.device)
        # Host construction is fixture input, never a CPU attention execution fallback.
        self.v = torch.tensor(
            [
                [[[value_at(page, offset, d) for d in range(HEAD_SIZE)]] for offset in range(BLOCK_SIZE)]
                for page in range(NUM_BLOCKS)
            ],
            dtype=torch.float16,
            device=self.device,
        )
        self.output = torch.empty_like(self.q)
        self.full_table = torch.empty((1, FULL_TABLE_WIDTH), dtype=torch.int32, device=self.device)
        self.slots = torch.arange(QUERY_WIDTH, dtype=torch.int64, device=self.device)
        self.positions = torch.arange(QUERY_WIDTH, dtype=torch.int64, device=self.device)
        self.query_locs = torch.tensor([0, QUERY_WIDTH], dtype=torch.int32, device=self.device)
        self.binding = ProductionBinding(self)
        self.table: Tensor = self.binding.startup.window_block_table
        self.length: Tensor = self.binding.startup.device_window_exact
        self.backend.key_cache, self.backend.value_cache = self.k, self.v
        metadata.block_tables, metadata.seq_lens = self.table, self.length
        metadata.actual_seq_lengths_q = [QUERY_WIDTH]
        metadata.query_start_loc, metadata.slot_mapping = self.query_locs, self.slots
        metadata.causal = False
        metadata.num_actual_tokens = QUERY_WIDTH
        self.retained = [
            self.q,
            self.k,
            self.v,
            self.table,
            self.output,
            self.full_table,
            self.length,
            self.slots,
            self.positions,
            self.query_locs,
        ]
        self.buffers = dict(
            zip(
                ("q", "k", "v", "table", "output", "full_table", "length", "slots", "positions", "query_locs"),
                self.retained,
            )
        )
        # All initialization must precede the private capture/replay stream.
        self.replay_stream.wait_stream(torch.npu.current_stream())

    def stage(self, case: dict[str, Any]) -> None:
        torch = self.torch
        with torch.npu.stream(self.replay_stream):
            self.full_table.copy_(torch.tensor([case["full_table"]], dtype=torch.int32, device=self.device))
            self.binding.bind(case)
            self.output.fill_(float("nan"))  # a replay that does no work cannot pass
        # Same pre-replay dependency as DFlashAclGraphManager.run_fullgraph.
        self.update_stream.wait_stream(self.replay_stream)

    def capture(self, torch_npu: Any) -> None:
        torch = self.torch
        with torch.npu.stream(self.replay_stream):
            # Startup has no runtime generation or lease, just the registered
            # P6 backing. A real P3 generation will be bound only after capture.
            first = self.manifest["cases"][0]
            self.table.copy_(torch.tensor([first["table"]], dtype=torch.int32, device=self.device))
            self.length.fill_(first["length"])
            self.metadata.seq_lens_list = [first["length"]]
            # Warm the real op without entering full_graph_fia's ExternalEvent wait.
            for _ in range(2):
                self.warm_output = torch_npu.npu_fused_infer_attention_score(
                    query=self.q,
                    key=self.k.view(NUM_BLOCKS, BLOCK_SIZE, HEAD_SIZE),
                    value=self.v.view(NUM_BLOCKS, BLOCK_SIZE, HEAD_SIZE),
                    block_table=self.table,
                    input_layout="TND",
                    block_size=BLOCK_SIZE,
                    actual_seq_lengths=[QUERY_WIDTH],
                    actual_seq_lengths_kv=[first["length"]],
                    num_key_value_heads=1,
                    num_heads=NUM_HEADS,
                    scale=self.backend.scale,
                    sparse_mode=0,
                )
        self.replay_stream.synchronize()
        self.update_stream.synchronize()
        with torch.npu.graph(self.graph, stream=self.replay_stream):
            self.capture_output = self.backend.full_graph_fia(self.q, self.k, self.v, self.metadata, self.output)
        params = self.graph_module.get_draft_graph_params()
        require(
            all(len(getattr(params, name)[QUERY_WIDTH]) == 1 for name in ("events", "handles", "attn_params")),
            "production capture must register exactly one FIA consumer",
        )
        self.params = params
        captured = params.attn_params[QUERY_WIDTH][0]
        require(
            captured[0].data_ptr() == self.q.data_ptr()
            and captured[3].data_ptr() == self.table.data_ptr()
            and captured[11].data_ptr() == self.output.data_ptr(),
            "FIA captured different fixture operands",
        )
        self.buffers["workspace"] = params.workspaces[QUERY_WIDTH]
        self.initial_specs = {name: tensor_spec(value) for name, value in self.buffers.items()}
        self.document["capture"] = {
            "count": 1,
            "backend": "AscendAttentionBackendImpl.full_graph_fia",
            "handle_id": id(params.handles[QUERY_WIDTH][0]),
            "external_event_id": id(params.events[QUERY_WIDTH][0]),
            "buffers": self.initial_specs,
            "provider_id": id(self.binding.provider),
            "provider_schema": 2,
        }

    def replay(self) -> None:
        with self.torch.npu.stream(self.replay_stream):
            self.graph.replay()
            # This is after the graph's last (and only) consumer, NOT after just UPDATE.
            self.consumer_done.record(self.replay_stream)

    def update(self) -> None:
        from vllm_ascend.attention.attention_v1 import AscendAttentionBackend

        self.graph_module.update_full_graph_params(
            AscendAttentionBackend,
            self.update_stream,
            self.context,
            QUERY_WIDTH,
            self.config,
            draft_attn_metadatas=self.binding.batch,
            draft_window_graph_update_binding=self.binding.update_binding,
        )
        self.update_done.record(self.update_stream)

    def complete(self) -> dict[str, Any]:
        self.update_done.synchronize()
        self.consumer_done.synchronize()
        require(
            self.update_done.query() is True and self.consumer_done.query() is True,
            "ordinary completion events did not verify both streams",
        )
        return {
            "method": "ordinary_events_after_update_and_last_consumer",
            "verified": True,
            "update_event": id(self.update_done),
            "consumer_event": id(self.consumer_done),
            "update_stream": int(self.update_stream.npu_stream),
            "consumer_stream": int(self.replay_stream.npu_stream),
        }

    def snapshot(self, case: dict[str, Any], completion: dict[str, Any]) -> dict[str, Any]:
        specs = {name: tensor_spec(value) for name, value in self.buffers.items()}
        require(specs == self.initial_specs, "capture buffer address/shape/schema changed")
        return {
            "manifest_digest": digest(self.manifest),
            "nonce": self.manifest["nonce"],
            "process_id": os.getpid(),
            "label": case["label"],
            "generation": case["generation"],
            "epoch": case["epoch"],
            "buffers": specs,
            "completion": completion,
            "full_table": self.full_table.cpu().tolist()[0],
            "table": self.table.cpu().tolist()[0],
            "length": self.length.cpu().tolist()[0],
            "host_lengths": list(self.metadata.seq_lens_list),
            "actual_seq_lengths_q": list(self.metadata.actual_seq_lengths_q),
            "binding": self.binding.retire_after_completion(),
            "output": self.output.cpu().float().tolist(),
            "controls": {
                "slots": self.slots.cpu().tolist(),
                "positions": self.positions.cpu().tolist(),
                "query_locs": self.query_locs.cpu().tolist(),
                "q": digest(self.q.cpu().tolist()),
                "k": digest(self.k.cpu().tolist()),
                "v": digest(self.v.cpu().tolist()),
            },
        }


def write_json(path: Path, value: Any) -> None:
    # Atomic checkpoint: a killed worker cannot leave a partially parsed success.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def device_worker(manifest_path: Path, output: Path) -> int:
    manifest = json.loads(manifest_path.read_text())
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "manifest_digest": digest(manifest),
        "identity": manifest,
        "process_id": os.getpid(),
        "status": "STARTING",
        "records": [],
        "blockers": dict(BLOCKERS),
        "ready": False,
    }
    try:
        validate_manifest(manifest)
        # No accelerator imports in the host checker. Avoid tools/bisect shadowing stdlib.
        sys.path = [p for p in sys.path if Path(p or ".").resolve() != ROOT / "tools"]
        sys.path.insert(0, str(ROOT))
        import torch
        import torch_npu
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.config.compilation import CompilationConfig, CUDAGraphMode
        from vllm.forward_context import ForwardContext, override_forward_context
        from vllm.v1.attention.backend import AttentionType

        from vllm_ascend.ascend_forward_context import _EXTRA_CTX
        from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl, AscendAttentionState, AscendMetadata
        from vllm_ascend.attention.utils import using_paged_attention
        from vllm_ascend.compilation import acl_graph

        document["runtime"] = runtime_evidence(torch, torch_npu)
        document["runtime"]["production_sources"] = {
            "capture": api_source(AscendAttentionBackendImpl.full_graph_fia),
            "backend_update": api_source(AscendAttentionBackendImpl.update_graph_params),
            "helper": api_source(acl_graph.update_full_graph_params),
        }
        torch.npu.set_device(manifest["device"])
        config = VllmConfig(compilation_config=CompilationConfig(cudagraph_mode=CUDAGraphMode.FULL))
        context = ForwardContext(no_compile_layers={}, attn_metadata={}, slot_mapping={})
        with set_current_vllm_config(config), override_forward_context(context), torch.no_grad():
            _EXTRA_CTX.is_draft_model = True
            _EXTRA_CTX.is_draft_model_prefill = False
            _EXTRA_CTX.sinks = False
            require(not using_paged_attention(QUERY_WIDTH, config), "backend selects PA rather than FIA")
            backend = AscendAttentionBackendImpl(
                num_heads=NUM_HEADS,
                head_size=HEAD_SIZE,
                scale=HEAD_SIZE**-0.5,
                num_kv_heads=1,
                alibi_slopes=None,
                sliding_window=None,
                kv_cache_dtype="auto",
                logits_soft_cap=None,
                attn_type=AttentionType.DECODER,
                kv_sharing_target_layer_name=None,
            )
            require(
                not backend.use_bnsd_kv_cache and not backend.enable_c8_quant,
                "fixture requires ordinary NHD FP16 KV, without quantization",
            )
            # Actual production allocation, only in this disposable process. No monkeypatches.
            acl_graph.set_draft_graph_params([QUERY_WIDTH])
            fixture = NPUFixture(
                torch,
                backend,
                AscendMetadata(attn_state=AscendAttentionState.SpecDecoding),
                config,
                context,
                acl_graph,
                manifest,
                document,
            )
            write_json(output, document)
            fixture.capture(torch_npu)

            def checkpoint(stage: str, guard: CompletionGuard) -> None:
                document["checkpoint"] = {
                    "stage": stage,
                    "generation": guard.generation,
                    "completion_verified": guard.completion_verified,
                }
                write_json(output, document)

            execute_cases(fixture, manifest, document, checkpoint)
    except BaseException as error:
        document.update(status="ABORTED", error=f"{type(error).__name__}: {error}")
        document["abort"] = CompletionGuard(state="unverified", completion_verified=False).abort()
    write_json(output, document)
    # Keep all graph/buffer references alive through this point. No destructor on failure.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if document["status"] == "COMPLETE" else 2)


def numeric_output(value: Any) -> list[float]:
    require(isinstance(value, list) and len(value) == QUERY_WIDTH, "output query shape mismatch")
    flat = []
    for token in value:
        require(isinstance(token, list) and len(token) == NUM_HEADS, "output head shape mismatch")
        for head in token:
            require(isinstance(head, list) and len(head) == HEAD_SIZE, "output channel shape mismatch")
            require(all(type(v) in (float, int) and math.isfinite(v) for v in head), "nonfinite/non-numeric output")
            flat.extend(head)
    return flat


def check_artifact(document: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "fixture": "FAIL",
        "production_integration": "PARTIAL",
        "ready": False,
        "G4F": BLOCKERS["G4F"],
        "blockers": dict(BLOCKERS),
        "errors": [],
    }
    if isinstance(document, dict) and document.get("status") != "COMPLETE":
        result["device_failure"] = {
            k: document[k] for k in ("error", "abort", "checkpoint", "process_exit") if k in document
        }
    try:
        validate_manifest(manifest)
        require(document["schema"] == SCHEMA, "wrong artifact schema")
        require(
            document["identity"] == manifest and document["manifest_digest"] == digest(manifest),
            "stale/cross-run artifact identity",
        )
        require(document["ready"] is False and document["blockers"] == BLOCKERS, "overclaimed readiness/scope")
        require(document["status"] == "COMPLETE" and "abort" not in document, "aborted/incomplete device run")
        require(manifest["abort_at"] == "none", "fault-injection run cannot pass")
        runtime = document["runtime"]
        require(runtime["torch_npu"].split("+")[0] == TORCH_NPU_VERSION, "wrong torch_npu version")
        require(runtime["torch"].split("+")[0] == "2.10.0", "wrong torch version")
        require(re.match(r"^9\.1(?:\.|$)", runtime["cann"]) is not None, "wrong CANN version")
        required_api = {
            "graph_task_group_begin",
            "graph_task_group_end",
            "graph_task_update_begin",
            "graph_task_update_end",
            "NPUGraph.replay",
            "Event.record",
            "Event.synchronize",
            "Event.query",
            "ExternalEvent.wait",
            "ExternalEvent.reset",
            "ExternalEvent.record",
            "get_cann_version",
        }
        require(
            set(runtime["api_sources"]) == required_api
            and set(runtime["production_sources"]) == {"capture", "backend_update", "helper"},
            "missing exact API sources",
        )
        for evidence in list(runtime["api_sources"].values()) + list(runtime["production_sources"].values()):
            require(
                isinstance(evidence["source"], str)
                and bool(evidence["source"])
                and type(evidence["line"]) is int
                and evidence["line"] > 0
                and len(evidence["file_sha256"]) == 64
                and bool(evidence["path"])
                and isinstance(evidence["signature"], str),
                "invalid exact API source evidence",
            )
        for name, filename in (
            ("capture", "vllm_ascend/attention/attention_v1.py"),
            ("backend_update", "vllm_ascend/attention/attention_v1.py"),
            ("helper", "vllm_ascend/compilation/acl_graph.py"),
        ):
            require(
                runtime["production_sources"][name]["file_sha256"] == manifest["source"]["files"][filename],
                "imported production API differs from manifest source",
            )
        require(
            len(runtime["native_extension_sha256"]) == 64 and "actual_seq_lengths_kv" in runtime["fia_schema"],
            "missing native binary/FIA schema identity",
        )
        capture = document["capture"]
        require(
            capture["count"] == 1 and capture["backend"] == "AscendAttentionBackendImpl.full_graph_fia",
            "not the production FIA capture seam",
        )
        records = document["records"]
        require(isinstance(records, list) and len(records) == len(cases()), "missing/extra controlled records")
        expected_shapes = {
            "q": [QUERY_WIDTH, NUM_HEADS, HEAD_SIZE],
            "output": [QUERY_WIDTH, NUM_HEADS, HEAD_SIZE],
            "k": [NUM_BLOCKS, BLOCK_SIZE, 1, HEAD_SIZE],
            "v": [NUM_BLOCKS, BLOCK_SIZE, 1, HEAD_SIZE],
            "table": [1, TABLE_WIDTH],
            "full_table": [1, FULL_TABLE_WIDTH],
            "length": [1],
            "slots": [QUERY_WIDTH],
            "positions": [QUERY_WIDTH],
            "query_locs": [2],
        }
        addresses = []
        require(set(capture["buffers"]) == set(expected_shapes) | {"workspace"}, "buffer inventory mismatch")
        for name, spec in capture["buffers"].items():
            require(type(spec["address"]) is int and spec["address"] > 0, f"invalid {name} address")
            require(spec["device"] == manifest["device"] and spec["offset"] == 0, f"invalid {name} device/offset")
            if name in expected_shapes:
                require(spec["shape"] == expected_shapes[name], f"{name} shape mismatch")
                dtype = (
                    "torch.float16"
                    if name in ("q", "k", "v", "output")
                    else ("torch.int64" if name in ("slots", "positions") else "torch.int32")
                )
                require(spec["dtype"] == dtype, f"{name} dtype mismatch")
            require(
                isinstance(spec["shape"], list)
                and bool(spec["shape"])
                and all(type(s) is int and s > 0 for s in spec["shape"]),
                "invalid buffer shape",
            )
            stride = spec["stride"]
            expected_stride: list[int] = []
            size = 1
            for dim in reversed(spec["shape"]):
                expected_stride.insert(0, size)
                size *= dim
            require(stride == expected_stride, "noncontiguous/changed buffer stride")
            addresses.append(spec["address"])
        require(len(addresses) == len(set(addresses)), "fixture buffers alias")
        outputs = []
        for case, record in zip(manifest["cases"], records):
            require(
                all(type(record[k]) is int for k in ("generation", "epoch", "length")),
                "noninteger generation/epoch/length",
            )
            require(all(record[k] == case[k] for k in ("label", "generation", "epoch")), "stale generation/epoch/order")
            require(
                record["manifest_digest"] == digest(manifest) and record["nonce"] == manifest["nonce"],
                "record challenge mismatch",
            )
            require(
                type(document["process_id"]) is int
                and document["process_id"] > 0
                and record["process_id"] == document["process_id"],
                "cross-process records",
            )
            require(record["buffers"] == capture["buffers"], "address/shape/schema changed after capture")
            binding = record["binding"]
            require(
                binding["provider_schema"] == capture["provider_schema"] == 2
                and binding["provider_id"] == capture["provider_id"]
                and type(binding["provider_id"]) is int
                and binding["provider_id"] > 0
                and binding["p3_generation"] == case["generation"] - 1
                and binding["epoch"] == case["epoch"]
                and binding["unsafe_reuse_rejected"] is True
                and binding["retired_after_consumer_completion"] is True
                and binding["update_function"] == "vllm_ascend.compilation.acl_graph.update_full_graph_params",
                "stale/missing production P3/P6 binding or unsafe reuse",
            )
            completion = record["completion"]
            require(
                completion["verified"] is True
                and completion["method"] == "ordinary_events_after_update_and_last_consumer",
                "last consumer completion not verified",
            )
            for field in ("consumer_event", "update_event", "consumer_stream", "update_stream"):
                require(type(completion[field]) is int and completion[field] > 0, "invalid completion identity")
            require(
                completion["consumer_event"] != completion["update_event"]
                and completion["consumer_stream"] != completion["update_stream"],
                "completion aliases update",
            )
            require(record["full_table"] == case["full_table"], "full source changed")
            oracle = full_table_oracle(case)
            require(record["table"] == oracle["table"], "incorrect full-table projection")
            require(
                record["length"] == oracle["length"] and record["host_lengths"] == [oracle["length"]],
                "incorrect/stale FIA length",
            )
            require(record["actual_seq_lengths_q"] == [QUERY_WIDTH], "query layout changed")
            actual, expected = numeric_output(record["output"]), numeric_output(oracle["output"])
            require(
                all(abs(a - b) <= ABS_TOL for a, b in zip(actual, expected)),
                f"{case['label']} measured output mismatches full-table oracle",
            )
            require(record["controls"] == records[0]["controls"], "read-only fixture controls changed")
            require(
                all(
                    isinstance(record["controls"][name], str)
                    and re.fullmatch(r"[a-f0-9]{64}", record["controls"][name]) is not None
                    for name in ("q", "k", "v")
                ),
                "missing Q/K/V control hashes",
            )
            require(
                record["controls"]["slots"] == list(range(QUERY_WIDTH))
                and record["controls"]["positions"] == list(range(QUERY_WIDTH))
                and record["controls"]["query_locs"] == [0, QUERY_WIDTH],
                "write/query sentinels changed",
            )
            outputs.append(actual)
        require(any(abs(a - b) > 10 * ABS_TOL for a, b in zip(outputs[0], outputs[1])), "A/B not distinguishable")
        require(all(abs(a - b) <= ABS_TOL for a, b in zip(outputs[0], outputs[2])), "A_prime did not restore A")
        result.update(
            fixture="PASS",
            effective_inputs="A/B/A_prime measured by independent numerical oracle",
            G5="PASS_DIAGNOSTIC_LAST_CONSUMER_COMPLETION",
            G4F=BLOCKERS["G4F"],
        )
    except (KeyError, TypeError, ValueError, IndexError, AttributeError, OSError) as error:
        result["errors"].append(f"{type(error).__name__}: {error}")
    return result


def launch(args: Any) -> int:
    manifest = make_manifest(args.run_id, args.attempt, args.device, args.abort_at)
    validate_manifest(manifest)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest_path, result_path = args.output / "manifest.json", args.output / "result.json"
    write_json(manifest_path, manifest)
    command = [sys.executable, str(Path(__file__).resolve()), "_device", str(manifest_path), str(result_path)]
    timed_out = False
    with (args.output / "device.log").open("w") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            child.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        except BaseException:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            raise
    if result_path.exists():
        document = json.loads(result_path.read_text())
    else:
        document = {
            "schema": SCHEMA,
            "manifest_digest": digest(manifest),
            "identity": manifest,
            "process_id": child.pid,
            "records": [],
            "ready": False,
            "blockers": dict(BLOCKERS),
        }
    if timed_out or child.returncode != 0 or document.get("status") != "COMPLETE":
        document["status"] = "ABORTED"
        document.setdefault("abort", CompletionGuard(state="unverified", completion_verified=False).abort())
        document["process_exit"] = {
            "returncode": child.returncode,
            "timeout": timed_out,
            "device_completion_inferred_from_exit": False,
        }
        write_json(result_path, document)
    report = check_artifact(document, manifest)
    write_json(args.output / "check.json", report)
    print(json.dumps(report, indent=2))
    return 0 if report["fixture"] == "PASS" else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="explicit NPU run in a disposable child; never CPU fallback")
    run.add_argument("--dry-run", action="store_true", help="print concrete controls/scope without importing torch")
    run.add_argument("--run-id")
    run.add_argument("--attempt")
    run.add_argument("--device", default="npu:0", help="NPU device, default npu:0")
    run.add_argument("--output", type=Path, help="new directory; existing directories rejected")
    run.add_argument("--timeout", type=int, default=120, help="child timeout seconds; default 120")
    run.add_argument(
        "--abort-at",
        choices=("none", "before_replay", "after_replay", "after_update", "after_completion"),
        default="none",
        help="diagnostic fault injection; always yields non-PASS",
    )
    check = sub.add_parser("check", help="host checker; requires separate expected run manifest")
    check.add_argument("artifact", type=Path)
    check.add_argument("--manifest", required=True, type=Path)
    worker = sub.add_parser("_device", help="internal isolated worker; use run")
    worker.add_argument("manifest", type=Path)
    worker.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run" and args.dry_run:
        print(
            json.dumps(
                {
                    "status": "DRY_RUN_NOT_DEVICE_EVIDENCE",
                    "scope": scope(),
                    "cases": cases(),
                    "expected": [full_table_oracle(c) for c in cases()],
                },
                indent=2,
            )
        )
        return 0
    if args.command == "_device":
        return device_worker(args.manifest, args.output)
    if args.command == "run":
        if not args.run_id or not args.attempt or args.output is None or args.timeout <= 0:
            parser.error("run requires --run-id, --attempt, --output and a positive --timeout")
        return launch(args)
    try:
        report = check_artifact(json.loads(args.artifact.read_text()), json.loads(args.manifest.read_text()))
    except (ValueError, OSError) as error:
        report = {"fixture": "FAIL", "ready": False, "errors": [str(error)]}
    print(json.dumps(report, indent=2))
    return 0 if report["fixture"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
