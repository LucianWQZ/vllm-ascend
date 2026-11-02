# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""DFlash2's fixed selector inputs and native-resident Graph source contract.

Only Host metadata is inspected here. Dispatch, request reordering and metadata
preparation remain outside capture. The inherited DFlash forward captures the
convolution, candidate TopK, selector, NPU walk and probability-cache writes;
this module neither clones those algorithms nor creates another Graph owner.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("DFlash2 Graph: " + message)


def _capture_tensor(
    parent: Any,
    name: str,
    shape: tuple[int, ...],
    dtype: str,
) -> Any:
    tensor = getattr(parent, name)
    _require(
        tuple(tensor.shape) == shape and str(tensor.dtype) == dtype,
        f"{name} shape/dtype does not match static K/selector capacity",
    )
    _require(tensor.is_contiguous() and tensor.data_ptr() != 0, f"{name} must have live contiguous persistent storage")
    return tensor


def _attention_impl_type():
    # Worker-only import; CONFIG_OFF/policy evaluation stays host-only.
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl

    return AscendAttentionBackendImpl


def validate_dflash2_graph_config(vllm_config: Any) -> None:
    """First Graph qualification is static K7 on the actual uncombined TP2 route.

    The mode and experimental gate are checked at manager construction even
    when the additional Draft Window is OFF. This check uses the real config,
    never a model name or the KV block size as a proxy for K or topology.
    """
    config = vllm_config.speculative_config
    parallel = vllm_config.parallel_config
    draft_parallel = config.draft_parallel_config
    topology = tuple(
        getattr(parallel, name, None)
        for name in (
            "tensor_parallel_size",
            "data_parallel_size",
            "pipeline_parallel_size",
            "prefill_context_parallel_size",
            "decode_context_parallel_size",
        )
    )
    _require(
        topology == (2, 1, 1, 1, 1) and getattr(draft_parallel, "tensor_parallel_size", None) == 2,
        "first Graph qualification requires actual Target/Draft TP2, DP1/PP1/PCP1/DCP1",
    )
    _require(
        type(config.num_speculative_tokens) is int
        and config.num_speculative_tokens == 7
        and not config.uses_dynamic_speculative_decoding()
        and not config.enable_adaptive_verification,
        "first Graph qualification requires static nonadaptive K7/Q8",
    )
    additional = vllm_config.additional_config or {}
    _require(
        additional.get("enable_dsa_cp", False) is False
        and additional.get("enable_flashcomm1", False) is False
        and getattr(parallel, "use_sequence_parallel_moe", None) is False
        and getattr(draft_parallel, "use_sequence_parallel_moe", None) is False,
        "first Graph qualification excludes FlashComm, SP and legacy DSA-CP",
    )


@dataclass(frozen=True, slots=True)
class DFlash2GraphContract:
    query_width: int
    max_num_reqs: int
    block_tables: Any = field(repr=False, compare=False)
    # The fixed speculator initializes these once and only updates their
    # contents. Retain the actual inputs for capture's entire lifetime, without
    # certifying the same private model structure before every proposal.
    buffers: tuple[Any, ...] = field(repr=False, compare=False)
    native_windows: tuple[tuple[str, int | None], ...]

    def validate_source(self, group: Any, *, causal: bool | Mapping[int, bool], window_size: int) -> None:
        """Check DFlash2's native read band on the shared backend inventory.

        Inventory certifies Full-retention storage and canonical block geometry;
        only the noncausal native mask and extra-Window subset are method-specific.
        """
        gid, kernel, native = group.gid, group.table_column_span, group.native_window
        _require(type(window_size) is int and window_size > 0, "positive integer Window required")
        _require(
            len(group.layers) == len(self.native_windows)
            and dict(self.native_windows) == dict.fromkeys(group.layers, native),
            "incomplete actual Draft layer inventory",
        )
        _require(
            (causal if type(causal) is bool else causal.get(gid)) is False,
            "DFlash2 Graph source requires actual noncausal attention",
        )
        table = self.block_tables.input_block_tables[gid]
        expansion = self.block_tables.blocks_per_kv_block[gid]
        _require(
            expansion == group.scheduler_block_size // kernel
            and len(table.shape) == 2
            and table.shape[0] == self.max_num_reqs
            and table.shape[1] > 0
            and table.shape[1] % expansion == 0
            and tuple(table.stride()) == (table.shape[1], 1)
            and str(table.dtype) == "torch.int32",
            "canonical source must retain the actual complete expanded int32 table rows",
        )
        if native is not None:
            _require(
                window_size + kernel - 1 + self.query_width <= native,
                "Window + block - 1 + query width exceeds the original noncausal native band",
            )


def validate_dflash2_graph_source(owner: Any) -> DFlash2GraphContract:
    """Bind one loaded DFlash2 model's existing addresses, never allocate KV.

    Called after set_attn has published the actual Draft consumers and before
    its Source inventory is admitted; Graph OFF uses the same execution check
    at manager construction, without creating a Window runtime or zero owner.
    """
    validate_dflash2_graph_config(owner.vllm_config)
    config = owner.speculative_config
    _require(
        tuple(owner.draft_model_config.architectures) == ("DFlash2DraftModel",), "actual DFlash2 architecture required"
    )
    _require(
        config.method == "dflash" and config.enforce_eager is not True,
        "this is the Graph execution contract, not a typed eager owner",
    )
    _require(
        not config.uses_dynamic_speculative_decoding() and not owner.enable_adaptive_verification,
        "first DFlash2 Graph implementation requires static nonadaptive K",
    )
    _require(owner.use_fp64_gumbel is False, "the NPU selector does not implement fp64 Gumbel")
    k, q, top_k, capacity = owner.num_speculative_steps, owner.num_query_per_req, owner.top_k, owner.max_num_reqs
    _require(
        all(type(v) is int and v > 0 for v in (k, q, top_k, capacity)) and q == k + 1,
        "static DFlash2 K/anchor query geometry differs",
    )
    _require(
        config.num_speculative_tokens == k and 0 < top_k <= owner.vocab_size,
        "selector or speculative configuration changed",
    )
    _require(getattr(owner, "_draft_window_runtime", None) is None, "typed eager and Graph owners cannot coexist")
    _require(
        getattr(owner, "_dflash2_full_kv_reference", False) is True,
        "the actual DFlash2 full-retention loader contract is missing",
    )
    sampler, model = owner.candidate_sampler, owner.model
    selector = model.model.candidate_selector
    _require(
        sampler.num_steps == k
        and sampler.top_k == selector.top_k == top_k
        and callable(model.compute_candidates)
        and callable(selector),
        "actual selector/sampler contract differs",
    )
    fields = (
        ("sample_indices", (capacity * k,), "torch.int64"),
        ("sample_pos", (capacity * k,), "torch.int64"),
        ("sample_idx_mapping", (capacity * k,), "torch.int64"),
        ("temperature", (capacity,), "torch.float32"),
        ("seeds", (capacity,), "torch.int64"),
        ("draft_tokens", (capacity, k), "torch.int64"),
    )
    buffers = (
        _capture_tensor(sampler, "scores", (capacity, k, top_k), "torch.float32"),
        _capture_tensor(sampler, "cached_candidate_ids", (capacity, k, top_k), "torch.int64"),
        *(_capture_tensor(owner, *args) for args in fields),
    )
    _require(config.draft_sample_method in ("greedy", "probabilistic"), "unknown candidate sampling mode")
    probabilistic = config.draft_sample_method == "probabilistic"
    _require(
        (owner.draft_logits is not None) is probabilistic,
        "probabilistic selector requires its existing full-vocab cache; greedy must not add one",
    )
    if probabilistic:
        buffers += (_capture_tensor(owner, "draft_logits", (capacity, k, owner.vocab_size), "torch.float32"),)
    impl_type = _attention_impl_type()
    layers = []
    context = owner.vllm_config.compilation_config.static_forward_context
    _require(
        owner.attn_vllm_config.compilation_config.static_forward_context is context,
        "Draft attention and model execution do not share the actual static context owner",
    )
    for name in sorted(owner.draft_attn_layer_names):
        impl = context[name].impl
        _require(type(impl) is impl_type, "DFlash2 native Graph requires the actual dense FIA implementation")
        native = impl.sliding_window
        _require(native is None or type(native) is int and native > 0, "invalid model native window")
        mask = getattr(impl, "_dflash2_native_swa_mask", None)
        _require(
            impl.sinks is None and impl.use_bnsd_kv_cache is False,
            "sinks/packed KV need a separate native Graph contract",
        )
        _require(
            native is None or (mask is not None and mask.data_ptr() != 0),
            "noncausal native mask was not created by the DFlash2 loader",
        )
        layers.append((name, native))
        if mask is not None:
            buffers += (mask,)
    _require(bool(layers), "no actual Draft attention layers")
    for layer in model.model.layers:
        for name in ("attention_conv", "mlp_conv"):
            conv = getattr(layer, name)
            _require(conv.block_size == q, "convolution must reset at the original anchor-plus-K query boundary")
    _require(len(model.model.layers) == len(layers), "DFlash2 convolution/attention layer inventory differs")
    return DFlash2GraphContract(q, capacity, owner.block_tables, buffers, tuple(layers))
