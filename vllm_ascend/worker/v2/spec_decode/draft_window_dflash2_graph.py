# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""DFlash2's fixed selector inputs and native-resident Graph source contract.

Only Host metadata is inspected here. Dispatch, request reordering and metadata
preparation remain outside capture. The inherited DFlash forward captures the
convolution, candidate TopK, selector, NPU walk and probability-cache writes;
this module neither clones those algorithms nor creates another Graph owner.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm_ascend.worker.v2.spec_decode.draft_window_graph import DraftWindowGraphTensorIdentity


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("DFlash2 Graph: " + message)


def _capture_tensor(
    parent: Any,
    name: str,
    shape: tuple[int, ...],
    dtype: str,
) -> tuple[Any, str, Any, DraftWindowGraphTensorIdentity]:
    from vllm_ascend.worker.v2.spec_decode.draft_window_graph import _tensor_identity

    tensor = getattr(parent, name)
    _require(
        tuple(tensor.shape) == shape and str(tensor.dtype) == dtype,
        f"{name} shape/dtype does not match static K/selector capacity",
    )
    _require(tensor.is_contiguous() and tensor.data_ptr() != 0, f"{name} must have live contiguous persistent storage")
    return parent, name, tensor, _tensor_identity(tensor)


@dataclass(frozen=True, slots=True)
class _NativeLayer:
    name: str
    impl: Any = field(repr=False, compare=False)
    native_window: int | None
    mask: Any = field(repr=False, compare=False)
    mask_address: int | None


def _source_types():
    # Worker-only imports; importing this contract or doing CONFIG_OFF/policy
    # resolution does not import an attention backend or inspect devices.
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    from vllm_ascend.attention.attention_v1 import (
        AscendAttentionBackend,
        AscendAttentionBackendImpl,
        AscendAttentionMetadataBuilder,
    )

    return FullAttentionSpec, AscendAttentionBackend, AscendAttentionBackendImpl, AscendAttentionMetadataBuilder


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
    num_steps: int
    query_width: int
    top_k: int
    max_num_reqs: int
    sampling_mode: str
    model: Any = field(repr=False, compare=False)
    selector: Any = field(repr=False, compare=False)
    sampler: Any = field(repr=False, compare=False)
    static_context: Any = field(repr=False, compare=False)
    block_tables: Any = field(repr=False, compare=False)
    buffers: tuple[tuple[Any, str, Any, DraftWindowGraphTensorIdentity], ...]
    layers: tuple[_NativeLayer, ...]
    convolution_owners: tuple[tuple[Any, int], ...] = field(repr=False, compare=False)

    def validate_owner(self, owner: Any) -> None:
        """Before submission, reject stale captured addresses without a D2H.

        These checks are not completion evidence. The existing DFlash stream,
        Graph bank and context-only owner contracts still protect the readers.
        """
        if (
            owner.num_speculative_steps != self.num_steps
            or owner.num_query_per_req != self.query_width
            or owner.top_k != self.top_k
            or owner.max_num_reqs != self.max_num_reqs
            or owner.speculative_config.draft_sample_method != self.sampling_mode
            or (self.sampling_mode == "greedy" and owner.draft_logits is not None)
            or owner.use_fp64_gumbel is not False
            or owner.enable_adaptive_verification
            or owner.model is not self.model
            or owner.model.model.candidate_selector is not self.selector
            or owner.candidate_sampler is not self.sampler
            or owner.vllm_config.compilation_config.static_forward_context is not self.static_context
        ):
            raise RuntimeError("DFlash2 Graph model/selector/static shape owner changed")
        from vllm_ascend.worker.v2.spec_decode.draft_window_graph import _tensor_identity

        for parent, name, tensor, captured in self.buffers:
            actual = getattr(parent, name)
            if actual is not tensor or _tensor_identity(actual) != captured:
                raise RuntimeError(f"DFlash2 Graph captured input owner changed: {name}")
        context = owner.vllm_config.compilation_config.static_forward_context
        for layer in self.layers:
            if (
                context.get(layer.name) is None
                or context[layer.name].impl is not layer.impl
                or layer.impl.sliding_window != layer.native_window
                or getattr(layer.impl, "_dflash2_native_swa_mask", None) is not layer.mask
                or (layer.mask is not None and layer.mask.data_ptr() != layer.mask_address)
            ):
                raise RuntimeError("DFlash2 Graph attention/mask owner changed")
        for convolution, block_size in self.convolution_owners:
            if convolution.block_size != block_size:
                raise RuntimeError("DFlash2 Graph convolution query grouping changed")

    def validate_source(
        self,
        attn_groups: Sequence[Sequence[Any]],
        block_tables: Any,
        causal: bool | Mapping[int, bool],
        window_size: int,
        query_width: int,
    ) -> dict[int, int | None]:
        """Validate storage addressing separately from the native read mask.

        A FullAttentionSpec with native P is full-retention storage, not Full
        attention. For noncausal native SWA the entire aligned extra-Window
        interval must fit its original symmetric P-band. No P or mask is erased.
        First integration admits one homogeneous global Draft group only.
        """
        full_spec, backend, impl_type, builder = _source_types()
        _require(block_tables is self.block_tables, "canonical BlockTables owner changed")
        _require(type(window_size) is int and window_size > 0, "positive integer Window required")
        _require(type(query_width) is int and query_width == self.query_width, "captured query width changed")
        groups = [(gid, group) for gid, consumers in enumerate(attn_groups) for group in consumers]
        _require(len(groups) == 1, "one homogeneous Draft global group is required; mixed groups are unqualified")
        gid, group = groups[0]
        _require(
            type(group.kv_cache_group_id) is int and group.kv_cache_group_id == gid,
            "AttentionGroup global id differs from its actual table owner",
        )
        _require(
            group.backend is backend and type(group.get_metadata_builder(0)) is builder,
            "only the actual dense Ascend FIA backend/builder is qualified",
        )
        spec = group.kv_cache_spec
        _require(
            type(spec) is full_spec and getattr(spec, "attention_chunk_size", None) is None,
            "native/recycling KV specs cannot masquerade as full-retention storage",
        )
        _require(
            gid < len(block_tables.block_sizes)
            and gid < len(block_tables.kernel_block_sizes)
            and gid < len(block_tables.blocks_per_kv_block)
            and gid < len(block_tables.input_block_tables),
            "missing actual block-table geometry",
        )
        logical, kernel = block_tables.block_sizes[gid], block_tables.kernel_block_sizes[gid]
        _require(
            type(logical) is int
            and type(kernel) is int
            and logical > 0
            and kernel > 0
            and logical == spec.block_size
            and logical % kernel == 0,
            "native resident Graph requires divisible actual logical/kernel block spans",
        )
        expansion = block_tables.blocks_per_kv_block[gid]
        _require(
            type(expansion) is int and expansion == logical // kernel,
            "block-table kernel column expansion disagrees with logical/kernel spans",
        )
        table = block_tables.input_block_tables[gid]
        _require(
            len(table.shape) == 2
            and table.shape[0] == self.max_num_reqs
            and table.shape[1] > 0
            and table.shape[1] % expansion == 0
            and tuple(table.stride()) == (table.shape[1], 1)
            and str(table.dtype) == "torch.int32",
            "canonical source must retain the actual complete expanded int32 table rows",
        )
        native = getattr(spec, "sliding_window", None)
        _require(native is None or type(native) is int and native > 0, "invalid native window")
        names = tuple(group.layer_names)
        _require(
            len(names) == len(set(names)) == len(self.layers) and set(names) == {layer.name for layer in self.layers},
            "incomplete actual Draft layer inventory",
        )
        group_causal = causal if type(causal) is bool else causal.get(gid)
        _require(group_causal is False, "first DFlash2 Graph source requires actual noncausal attention")
        for layer in self.layers:
            actual = layer.impl
            _require(
                type(actual) is impl_type and actual.sliding_window == layer.native_window == native,
                "model native window and full-retention KV spec disagree",
            )
            _require(
                actual.sinks is None and actual.use_bnsd_kv_cache is False,
                "sinks/packed KV need a separate native Graph contract",
            )
            mask = getattr(actual, "_dflash2_native_swa_mask", None)
            _require(mask is layer.mask, "native noncausal mask owner changed")
            if native is not None:
                _require(
                    mask is not None and mask.data_ptr() == layer.mask_address,
                    "missing or replaced native noncausal mask backing",
                )
        if native is not None:
            _require(
                window_size + kernel - 1 + query_width <= native,
                "Window + block - 1 + query width exceeds the original noncausal native band",
            )
        return {gid: native}


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
    _, _, impl_type, _ = _source_types()
    layers, convolutions = [], []
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
        _require(native is None or mask is not None, "noncausal native mask was not created by the DFlash2 loader")
        layers.append(_NativeLayer(name, impl, native, mask, None if mask is None else mask.data_ptr()))
    _require(bool(layers), "no actual Draft attention layers")
    for layer in model.model.layers:
        for name in ("attention_conv", "mlp_conv"):
            conv = getattr(layer, name)
            _require(conv.block_size == q, "convolution must reset at the original anchor-plus-K query boundary")
            convolutions.append((conv, q))
    _require(len(convolutions) == 2 * len(layers), "DFlash2 convolution/attention layer inventory differs")
    contract = DFlash2GraphContract(
        k,
        q,
        top_k,
        capacity,
        config.draft_sample_method,
        model,
        selector,
        sampler,
        context,
        owner.block_tables,
        buffers,
        tuple(layers),
        tuple(convolutions),
    )
    contract.validate_owner(owner)
    return contract
