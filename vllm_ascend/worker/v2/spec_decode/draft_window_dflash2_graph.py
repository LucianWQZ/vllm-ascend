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
from typing import Any


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("DFlash2 Graph: " + message)


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


def validate_dflash2_graph_read_source(
    owner: Any, group: Any, *, causal: bool | Mapping[int, bool], window_size: int
) -> None:
    """Qualify the real owner and its inventory without a second buffer owner."""
    validate_dflash2_graph_source(owner)
    gid, kernel, native = group.gid, group.table_column_span, group.native_window
    _require(type(window_size) is int and window_size > 0, "positive integer Window required")
    context = owner.vllm_config.compilation_config.static_forward_context
    _require(
        len(group.layers) == len(owner.draft_attn_layer_names)
        and set(group.layers) == owner.draft_attn_layer_names
        and all(context[name].impl.sliding_window == native for name in group.layers),
        "incomplete actual Draft layer inventory",
    )
    _require(
        (causal if type(causal) is bool else causal.get(gid)) is False,
        "DFlash2 Graph source requires actual noncausal attention",
    )
    table = owner.block_tables.input_block_tables[gid]
    expansion = owner.block_tables.blocks_per_kv_block[gid]
    _require(
        expansion == group.scheduler_block_size // kernel
        and len(table.shape) == 2
        and table.shape[0] == owner.max_num_reqs
        and table.shape[1] > 0
        and table.shape[1] % expansion == 0
        and tuple(table.stride()) == (table.shape[1], 1)
        and str(table.dtype) == "torch.int32",
        "canonical source must retain the actual complete expanded int32 table rows",
    )
    if native is not None:
        _require(
            window_size + kernel - 1 + owner.num_query_per_req <= native,
            "Window + block - 1 + query width exceeds the original noncausal native band",
        )


def validate_dflash2_graph_source(owner: Any) -> None:
    """Validate the loaded owner's fixed capture inputs without allocating KV.

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
    # Fixed DFlash/DFlash2 constructors own the sampler and per-request input
    # allocations. Capture retains those original tensors; Window does not
    # replace them or maintain a second allocation/layout schema.
    _require(config.draft_sample_method in ("greedy", "probabilistic"), "unknown candidate sampling mode")
    probabilistic = config.draft_sample_method == "probabilistic"
    _require(
        (owner.draft_logits is not None) is probabilistic,
        "probabilistic selector requires its existing full-vocab cache; greedy must not add one",
    )
    impl_type = _attention_impl_type()
    context = owner.vllm_config.compilation_config.static_forward_context
    _require(
        owner.attn_vllm_config.compilation_config.static_forward_context is context,
        "Draft attention and model execution do not share the actual static context owner",
    )
    _require(bool(owner.draft_attn_layer_names), "no actual Draft attention layers")
    for name in owner.draft_attn_layer_names:
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
    for layer in model.model.layers:
        for name in ("attention_conv", "mlp_conv"):
            conv = getattr(layer, name)
            _require(conv.block_size == q, "convolution must reset at the original anchor-plus-K query boundary")
    _require(
        len(model.model.layers) == len(owner.draft_attn_layer_names),
        "DFlash2 convolution/attention layer inventory differs",
    )
