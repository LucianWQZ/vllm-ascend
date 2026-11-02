# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Instance-scoped DFlash capture adapter for Draft Window.

The Ascend worker installs :func:`ascend_prepare_dflash_inputs_to_capture` once.
Without an explicit context on the Draft speculator's private ``InputBuffers``
instance, the adapter delegates to the exact upstream helper. The
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
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import (
    AttentionState,
    BatchExecutionDescriptor,
)
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import (
    _prepare_dflash_inputs_to_capture as _ORIGINAL_PREPARE_DFLASH_INPUTS_TO_CAPTURE,
)
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.worker.v2.spec_decode.draft_window import DraftWindowWorkspace
from vllm_ascend.worker.v2.spec_decode.draft_window_graph import (
    DraftWindowGraphBindingProvider,
    DraftWindowGraphCaptureIdentity,
    DraftWindowGraphRoute,
    resolve_parallel_draft_window_graph_route,
)

_CAPTURE_CONTEXT_ATTRIBUTE = "_ascend_draft_window_capture_context"


@dataclass(slots=True)
class DraftWindowDFlashCaptureContext:
    """Temporarily bind the real provider to its Draft-private input buffers."""

    provider: DraftWindowGraphBindingProvider
    route: DraftWindowGraphRoute
    input_buffers: InputBuffers
    captures: dict[tuple[int, int], DraftWindowGraphCaptureIdentity]
    active: bool = True

    def validate_for_provider(self, provider: DraftWindowGraphBindingProvider) -> None:
        if not self.active:
            raise RuntimeError("stale Draft Window capture context is no longer active")
        if provider is not self.provider:
            raise RuntimeError("capture context belongs to another graph provider")

    def resolve(self, num_reqs: int, num_tokens: int) -> tuple[DraftWindowGraphCaptureIdentity, DraftWindowWorkspace]:
        self.validate_for_provider(self.provider)
        try:
            identity = self.captures[num_reqs, num_tokens]
        except KeyError as error:
            raise RuntimeError("Draft Window capture has no exact persistent bank") from error
        return identity, self.provider.startup_capture_workspace(identity)


def _build_capture_context(
    owner: Any,
    input_buffers: InputBuffers,
    capture_descs: Mapping[CUDAGraphMode, Sequence[BatchExecutionDescriptor]],
) -> DraftWindowDFlashCaptureContext | None:
    provider = getattr(owner, "_draft_window_graph_binding_provider", None)
    if provider is None:
        return None
    if not isinstance(provider, DraftWindowGraphBindingProvider):
        raise TypeError("invalid instance-owned Draft Window graph binding provider")
    if not isinstance(input_buffers, InputBuffers) or getattr(owner, "input_buffers", None) is not input_buffers:
        raise RuntimeError("capture must use the speculator-owned Draft input buffers")
    if getattr(owner, "target_input_buffers", None) is input_buffers:
        raise RuntimeError("Draft Window capture context cannot use Target input buffers")
    route = resolve_parallel_draft_window_graph_route(owner)
    if route is None:
        raise RuntimeError("Draft Window graph provider has no parallel capture route")
    descriptors = tuple(
        descriptor
        for entries in capture_descs.values()
        for descriptor in entries
        if descriptor.cg_mode is CUDAGraphMode.FULL
    )
    captures = {}
    for descriptor in descriptors:
        if descriptor.num_reqs is None:
            raise RuntimeError("FULL Draft Window capture requires a request count")
        key = descriptor.num_reqs, descriptor.num_tokens
        if key in captures:
            raise RuntimeError("Draft Window capture descriptor mapping is ambiguous")
        captures[key] = provider.capture_identity_for(route=route, descriptor=descriptor)
    if not captures:
        raise RuntimeError("Draft Window capture has no FULL descriptor")
    return DraftWindowDFlashCaptureContext(provider, route, input_buffers, captures)


@contextmanager
def draft_window_dflash_capture_scope(
    owner: Any,
    input_buffers: InputBuffers,
    capture_descs: Mapping[CUDAGraphMode, Sequence[BatchExecutionDescriptor]],
) -> Iterator[DraftWindowDFlashCaptureContext | None]:
    """Attach a capture context to one Draft-private input owner and clean it."""
    if hasattr(input_buffers, _CAPTURE_CONTEXT_ATTRIBUTE):
        raise RuntimeError("Draft Window capture context is already attached")
    context = _build_capture_context(
        owner,
        input_buffers,
        capture_descs,
    )
    if context is None:
        yield None
        return
    setattr(input_buffers, _CAPTURE_CONTEXT_ATTRIBUTE, context)
    try:
        yield context
    finally:
        context.active = False
        if hasattr(input_buffers, _CAPTURE_CONTEXT_ATTRIBUTE):
            delattr(input_buffers, _CAPTURE_CONTEXT_ATTRIBUTE)


def _seed_capture_workspace(
    identity: DraftWindowGraphCaptureIdentity,
    workspace: DraftWindowWorkspace,
    input_block_tables: Sequence[torch.Tensor],
    *,
    max_model_len: int,
    provider: DraftWindowGraphBindingProvider,
) -> int:
    gid = provider.source.group.gid
    if not 0 <= gid < len(input_block_tables):
        raise RuntimeError("Draft Window capture gid is absent from dummy full tables")
    source = input_block_tables[gid]
    destination = workspace.window_block_table
    if source.ndim != 2 or destination.ndim != 2:
        raise RuntimeError("Draft Window capture tables must be two-dimensional")
    if source.shape[0] != identity.descriptor.num_reqs:
        raise RuntimeError("Draft Window capture full table row count is inconsistent")
    if destination.shape[0] != identity.descriptor.num_reqs:
        raise RuntimeError("Draft Window capture backing row count is inconsistent")

    capture_upper = min(
        max_model_len,
        provider.source.window_size + provider.source.group.table_column_span - 1,
    )
    valid_num_blocks = (
        capture_upper + provider.source.group.table_column_span - 1
    ) // provider.source.group.table_column_span
    if valid_num_blocks > source.shape[1] or valid_num_blocks > destination.shape[1]:
        raise RuntimeError("Draft Window capture table cannot cover the Window capacity")

    destination.zero_()
    destination[:, :valid_num_blocks].copy_(source[:, :valid_num_blocks])
    workspace.device_window_exact.fill_(capture_upper)
    workspace.cpu_window_exact.fill_(capture_upper)
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
    if context is None or skip_attn:
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
    if context.input_buffers is not input_buffers:
        raise RuntimeError("Draft Window capture context owner mismatch")
    identity, workspace = context.resolve(num_reqs, num_tokens)

    # Keep dummy InputBatch, Full slots, query geometry and model-specific
    # metadata construction in the fixed MRV2 helper. Only replace the actual
    # Window read operands before any capture forward is submitted.
    state = _ORIGINAL_PREPARE_DFLASH_INPUTS_TO_CAPTURE(
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
    if not state.attn_metadata:
        raise RuntimeError("Draft Window capture builder produced no attention metadata")
    tables = tuple(table[:num_reqs] for table in block_tables.input_block_tables)
    capacity = _seed_capture_workspace(
        identity, workspace, tables, max_model_len=max_model_len, provider=context.provider
    )
    for name in context.provider.source.group.layers:
        layer = state.attn_metadata[name]
        layer.block_tables = workspace.window_block_table
        layer.seq_lens = layer.seq_lens_gpu = workspace.device_window_exact
        layer.seq_lens_cpu = workspace.cpu_window_exact
        layer.seq_lens_list = workspace.cpu_window_exact.tolist()
        layer.max_seq_len = capacity
        layer.query_start_loc = input_buffers.query_start_loc[: num_reqs + 1]
    return state
