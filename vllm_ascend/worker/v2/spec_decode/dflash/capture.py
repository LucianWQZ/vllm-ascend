# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Instance-scoped DFlash capture adapter for the Draft Window spike.

The worker patch installs :func:`ascend_prepare_dflash_inputs_to_capture` once.
Without an explicit context on the Draft speculator's private ``InputBuffers``
instance, the adapter delegates to the exact upstream helper. The experimental
Window path changes only Draft read tables and lengths; dummy full slot
mappings and query/write carriers remain owned by canonical ``BlockTables``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens
from vllm.v1.worker.gpu.cudagraph_utils import (
    AttentionState,
    BatchExecutionDescriptor,
)
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import (
    _prepare_dflash_inputs_to_capture as _ORIGINAL_PREPARE_DFLASH_INPUTS_TO_CAPTURE,
)
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.worker.v2.attn_utils import build_attn_metadata
from vllm_ascend.worker.v2.spec_decode.draft_window_graph import (
    DraftWindowGraphBindingProvider,
    DraftWindowGraphCaptureIdentity,
    DraftWindowGraphReadInputs,
    DraftWindowGraphRoute,
    resolve_parallel_draft_window_graph_route,
)

_CAPTURE_CONTEXT_ATTRIBUTE = "_ascend_draft_window_capture_context"


@dataclass(slots=True)
class _CaptureScopeState:
    active: bool = True


@dataclass(frozen=True, slots=True)
class _CaptureEntry:
    key: tuple[int, int, str]
    identity: DraftWindowGraphCaptureIdentity
    read_inputs: DraftWindowGraphReadInputs


@dataclass(frozen=True, slots=True)
class DraftWindowDFlashCaptureContext:
    """Typed startup context attached only to one Draft ``InputBuffers``."""

    provider: DraftWindowGraphBindingProvider
    route: DraftWindowGraphRoute
    input_buffers_owner_id: int
    entries: tuple[_CaptureEntry, ...]
    _state: _CaptureScopeState
    target_metadata: Mapping[str, Any] | None = None
    native_swa_plan: Any = None

    def validate_for_provider(
        self,
        provider: DraftWindowGraphBindingProvider,
    ) -> None:
        self._require_active()
        if provider is not self.provider:
            raise RuntimeError("capture context belongs to another graph provider")
        if self.native_swa_plan is not provider.native_swa_graph_plan:
            raise RuntimeError("native SWA capture program contract changed")
        if not self.entries:
            raise RuntimeError("capture context has no registered graph consumers")

    def resolve(
        self,
        num_reqs: int,
        num_tokens: int,
        skip_attn: bool,
    ) -> tuple[DraftWindowGraphCaptureIdentity, DraftWindowGraphReadInputs]:
        self._require_active()
        mode = CUDAGraphMode.PIECEWISE.name if skip_attn else CUDAGraphMode.FULL.name
        key = (num_reqs, num_tokens, mode)
        matches = tuple(entry for entry in self.entries if entry.key == key)
        if len(matches) != 1:
            raise RuntimeError(
                "Draft Window capture bucket does not resolve exactly one persistent "
                f"carrier: key={key!r} matches={len(matches)}"
            )
        entry = matches[0]
        current = self.provider.startup_capture_read_inputs(entry.identity)
        if current is not entry.read_inputs:
            raise RuntimeError("Draft Window capture backing changed after context binding")
        return entry.identity, entry.read_inputs

    def _require_active(self) -> None:
        if not self._state.active:
            raise RuntimeError("stale Draft Window capture context is no longer active")


def _normalized_request_count(
    descriptor: BatchExecutionDescriptor,
    *,
    max_num_reqs: int,
) -> int:
    return descriptor.num_reqs or min(descriptor.num_tokens, max_num_reqs)


def _descriptor_key(
    descriptor: BatchExecutionDescriptor,
    *,
    max_num_reqs: int,
) -> tuple[int, int, str]:
    return (
        _normalized_request_count(descriptor, max_num_reqs=max_num_reqs),
        descriptor.num_tokens,
        descriptor.cg_mode.name,
    )


def _build_capture_context(
    owner: Any,
    input_buffers: InputBuffers,
    capture_descs: Mapping[CUDAGraphMode, Sequence[BatchExecutionDescriptor]],
    *,
    max_num_reqs: int,
) -> DraftWindowDFlashCaptureContext | None:
    provider = getattr(owner, "_draft_window_graph_binding_provider", None)
    if provider is None:
        return None
    if not isinstance(provider, DraftWindowGraphBindingProvider):
        raise TypeError("invalid instance-owned Draft Window graph binding provider")
    if not isinstance(input_buffers, InputBuffers) or (getattr(owner, "input_buffers", None) is not input_buffers):
        raise RuntimeError(
            "P6 has no safe Ascend-only capture carrier: capture must use the speculator-owned Draft input buffers"
        )
    if getattr(owner, "target_input_buffers", None) is input_buffers:
        raise RuntimeError("Draft Window capture context cannot use Target input buffers")
    native_swa = provider.native_swa_graph_plan
    if native_swa is not None and (
        native_swa.input_owner_id != id(input_buffers)
        or native_swa.device_lengths_ptr != input_buffers.seq_lens.data_ptr()
    ):
        raise RuntimeError("native SWA capture replaced its admitted private input owner")
    route = resolve_parallel_draft_window_graph_route(owner)
    if route is None:
        raise RuntimeError("Draft Window graph provider has no parallel capture route")

    descriptors = tuple(descriptor for mode_descriptors in capture_descs.values() for descriptor in mode_descriptors)
    identities = provider.registered_capture_identities(route=route)
    if not descriptors or not identities:
        raise RuntimeError(
            "P6 has no safe Ascend-only capture carrier: capture descriptors and "
            "registered capture identities must both be explicit"
        )
    if len(descriptors) != len(identities):
        raise RuntimeError("Draft Window capture descriptor/carrier cardinality mismatch")

    entries: list[_CaptureEntry] = []
    for descriptor in descriptors:
        key = _descriptor_key(descriptor, max_num_reqs=max_num_reqs)
        matches = tuple(
            identity for identity in identities if identity.descriptor == descriptor and identity.num_reqs == key[0]
        )
        if len(matches) != 1:
            raise RuntimeError(
                "Draft Window capture descriptor does not resolve exactly one "
                f"registered carrier: descriptor={descriptor!r} matches={len(matches)}"
            )
        identity = matches[0]
        entries.append(
            _CaptureEntry(
                key=key,
                identity=identity,
                read_inputs=provider.startup_capture_read_inputs(identity),
            )
        )
    if len({entry.key for entry in entries}) != len(entries):
        raise RuntimeError("Draft Window capture descriptor mapping is ambiguous")

    return DraftWindowDFlashCaptureContext(
        provider=provider,
        route=route,
        input_buffers_owner_id=id(input_buffers),
        entries=tuple(entries),
        _state=_CaptureScopeState(),
        target_metadata=getattr(getattr(owner, "model_state", None), "attn_metadata", None),
        native_swa_plan=native_swa,
    )


@contextmanager
def draft_window_dflash_capture_scope(
    owner: Any,
    input_buffers: InputBuffers,
    capture_descs: Mapping[CUDAGraphMode, Sequence[BatchExecutionDescriptor]],
    *,
    max_num_reqs: int,
) -> Iterator[DraftWindowDFlashCaptureContext | None]:
    """Attach a capture context to one Draft-private input owner and clean it."""
    if hasattr(input_buffers, _CAPTURE_CONTEXT_ATTRIBUTE):
        raise RuntimeError("Draft Window capture context is already attached")
    context = _build_capture_context(
        owner,
        input_buffers,
        capture_descs,
        max_num_reqs=max_num_reqs,
    )
    if context is None:
        yield None
        return
    setattr(input_buffers, _CAPTURE_CONTEXT_ATTRIBUTE, context)
    try:
        yield context
    finally:
        context._state.active = False
        if hasattr(input_buffers, _CAPTURE_CONTEXT_ATTRIBUTE):
            delattr(input_buffers, _CAPTURE_CONTEXT_ATTRIBUTE)


def _seed_capture_read_inputs(
    identity: DraftWindowGraphCaptureIdentity,
    read_inputs: DraftWindowGraphReadInputs,
    input_block_tables: Sequence[torch.Tensor],
    *,
    max_model_len: int,
    provider: DraftWindowGraphBindingProvider,
) -> int:
    gid = provider.group.gid
    if not 0 <= gid < len(input_block_tables):
        raise RuntimeError("Draft Window capture gid is absent from dummy full tables")
    source = input_block_tables[gid]
    destination = read_inputs.window_block_table
    if source.ndim != 2 or destination.ndim != 2:
        raise RuntimeError("Draft Window capture tables must be two-dimensional")
    if source.shape[0] != read_inputs.padded_request_count:
        raise RuntimeError("Draft Window capture full table row count is inconsistent")
    if destination.shape[0] != read_inputs.padded_request_count:
        raise RuntimeError("Draft Window capture backing row count is inconsistent")

    capture_upper = min(
        max_model_len,
        provider.window_size + provider.group.table_column_span - 1,
    )
    valid_num_blocks = (capture_upper + provider.group.table_column_span - 1) // provider.group.table_column_span
    if valid_num_blocks > source.shape[1] or valid_num_blocks > destination.shape[1]:
        raise RuntimeError("Draft Window capture table cannot cover the Window capacity")

    destination.zero_()
    destination[:, :valid_num_blocks].copy_(source[:, :valid_num_blocks])
    read_inputs.valid_num_blocks.fill_(valid_num_blocks)
    read_inputs.device_window_exact.fill_(capture_upper)
    read_inputs.cpu_window_exact.fill_(capture_upper)
    read_inputs.cpu_window_upper.fill_(capture_upper)
    return capture_upper


def ascend_prepare_dflash_inputs_to_capture(
    num_reqs: int,
    num_tokens: int,
    input_buffers: InputBuffers,
    block_tables: BlockTables,
    attn_groups: list[list[AttentionGroup]],
    kv_cache_config: KVCacheConfig,
    max_model_len: int,
    skip_attn: bool,
    causal: bool | Mapping[int, bool],
) -> AttentionState:
    """Preserve upstream OFF behavior or inject the explicit capture carrier."""
    context = getattr(input_buffers, _CAPTURE_CONTEXT_ATTRIBUTE, None)
    if context is None:
        return _ORIGINAL_PREPARE_DFLASH_INPUTS_TO_CAPTURE(
            num_reqs,
            num_tokens,
            input_buffers,
            block_tables,
            attn_groups,
            kv_cache_config,
            max_model_len,
            skip_attn,
            causal,
        )
    if not isinstance(context, DraftWindowDFlashCaptureContext):
        raise TypeError("invalid Draft Window capture context on Draft input buffers")
    if context.input_buffers_owner_id != id(input_buffers):
        raise RuntimeError("Draft Window capture context owner mismatch")
    identity, read_inputs = context.resolve(num_reqs, num_tokens, skip_attn)

    input_batch = InputBatch.make_dummy(num_reqs, num_tokens, input_buffers)
    input_block_tables = block_tables.get_dummy_block_tables(num_reqs)
    slot_mappings = block_tables.get_dummy_slot_mappings(num_tokens)
    slot_mappings_by_layer = build_slot_mappings_by_layer(
        slot_mappings,
        kv_cache_config,
    )
    if skip_attn:
        return AttentionState(None, slot_mappings_by_layer)

    max_seq_len = _seed_capture_read_inputs(
        identity, read_inputs, input_block_tables, max_model_len=max_model_len, provider=context.provider
    )
    query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
    dcp_local_seq_lens = None
    if block_tables.cp_size > 1:
        dcp_local_seq_lens = prepare_dcp_local_seq_lens(
            input_buffers.dcp_local_seq_lens,
            read_inputs.device_window_exact,
            input_batch.num_reqs,
            block_tables.cp_size,
            block_tables.cp_rank,
            block_tables.cp_interleave,
            num_reqs_padded=input_batch.num_reqs_after_padding,
        )
    read_block_tables = list(input_block_tables)
    read_block_tables[context.provider.group.gid] = read_inputs.window_block_table
    native_swa = context.native_swa_plan
    from vllm_ascend.worker.v2.spec_decode.draft_window_eager import AscendDraftAttentionReadInputs

    graph_read_inputs = AscendDraftAttentionReadInputs(
        block_tables=tuple(read_block_tables),
        seq_lens=read_inputs.device_window_exact,
        seq_lens_cpu_exact=read_inputs.cpu_window_exact,
        seq_lens_cpu_upper_bound=read_inputs.cpu_window_upper,
        max_seq_len=max_seq_len,
        dcp_local_seq_lens=dcp_local_seq_lens,
    )
    attn_metadata = build_attn_metadata(
        attn_groups=attn_groups if native_swa is None else native_swa.filter_groups(attn_groups, native=False),
        num_reqs=num_reqs,
        num_tokens=num_tokens,
        query_start_loc_gpu=input_batch.query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        max_query_len=num_tokens // num_reqs,
        seq_lens=read_inputs.device_window_exact,
        seq_lens_cpu=read_inputs.cpu_window_exact,
        seq_lens_cpu_upper_bound=read_inputs.cpu_window_upper,
        draft_window_read_inputs=graph_read_inputs,
        dcp_local_seq_lens=dcp_local_seq_lens,
        max_seq_len=max_seq_len,
        block_tables=tuple(read_block_tables),
        slot_mappings=slot_mappings,
        kv_cache_config=kv_cache_config,
        for_cudagraph_capture=True,
        causal=causal,
    )
    window_metadata = attn_metadata
    if native_swa is not None:
        # Reproduce the upstream native read inputs, not Window lengths.
        # Its captured canonical table remains the baseline stable owner;
        # every group's original full write slots and causality are retained.
        native_metadata = build_attn_metadata(
            attn_groups=native_swa.filter_groups(attn_groups, native=True),
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=num_tokens // num_reqs,
            seq_lens=input_batch.seq_lens,
            dcp_local_seq_lens=None,
            max_seq_len=max_model_len,
            block_tables=input_block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            for_cudagraph_capture=True,
            causal=causal,
        )
        if set(native_metadata) != set(native_swa.native_layers) or set(window_metadata) != set(native_swa.full_layers):
            raise RuntimeError("native SWA capture metadata does not cover the admitted readers")
        attn_metadata = {**native_metadata, **window_metadata}
    return AttentionState(attn_metadata, slot_mappings_by_layer)
