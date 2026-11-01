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
    active: bool = True

    def validate_for_provider(self, provider: DraftWindowGraphBindingProvider) -> None:
        if not self.active:
            raise RuntimeError("stale Draft Window capture context is no longer active")
        if provider is not self.provider:
            raise RuntimeError("capture context belongs to another graph provider")

    def resolve(self, num_reqs: int, num_tokens: int) -> tuple[DraftWindowGraphCaptureIdentity, DraftWindowWorkspace]:
        self.validate_for_provider(self.provider)
        matches = tuple(
            identity
            for identity in self.provider.registered_capture_identities(route=self.route)
            if identity.descriptor.num_reqs == num_reqs
            and identity.descriptor.num_tokens == num_tokens
            and identity.descriptor.cg_mode is CUDAGraphMode.FULL
        )
        if len(matches) != 1:
            raise RuntimeError(
                "Draft Window capture bucket does not resolve exactly one persistent "
                f"carrier: requests={num_reqs} tokens={num_tokens} mode=FULL matches={len(matches)}"
            )
        identity = matches[0]
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
    identities = provider.registered_capture_identities(route=route)
    if not descriptors or len(descriptors) != len(identities):
        raise RuntimeError("Draft Window capture descriptor/carrier cardinality mismatch")
    # The upstream helper receives only these three coordinates. Require that
    # they select exactly one of the provider's real, fully qualified keys.
    keys = set()
    for descriptor in descriptors:
        num_reqs = descriptor.num_reqs
        matches = [identity for identity in identities if identity.descriptor == descriptor]
        if len(matches) != 1:
            raise RuntimeError("Draft Window capture descriptor does not resolve exactly one registered carrier")
        key = (num_reqs, descriptor.num_tokens, descriptor.cg_mode)
        if key in keys:
            raise RuntimeError("Draft Window capture descriptor mapping is ambiguous")
        keys.add(key)
    return DraftWindowDFlashCaptureContext(provider, route, input_buffers)


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
    workspace.valid_num_blocks.fill_(valid_num_blocks)
    workspace.device_window_exact.fill_(capture_upper)
    workspace.cpu_window_exact.fill_(capture_upper)
    workspace.cpu_window_upper.fill_(capture_upper)
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

    input_batch = InputBatch.make_dummy(num_reqs, num_tokens, input_buffers)
    input_block_tables = block_tables.get_dummy_block_tables(num_reqs)
    slot_mappings = block_tables.get_dummy_slot_mappings(num_tokens)
    slot_mappings_by_layer = build_slot_mappings_by_layer(
        slot_mappings,
        kv_cache_config,
    )
    max_seq_len = _seed_capture_workspace(
        identity, workspace, input_block_tables, max_model_len=max_model_len, provider=context.provider
    )
    query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
    dcp_local_seq_lens = None
    if block_tables.cp_size > 1:
        dcp_local_seq_lens = prepare_dcp_local_seq_lens(
            input_buffers.dcp_local_seq_lens,
            workspace.device_window_exact,
            input_batch.num_reqs,
            block_tables.cp_size,
            block_tables.cp_rank,
            block_tables.cp_interleave,
            num_reqs_padded=input_batch.num_reqs_after_padding,
        )
    read_block_tables = list(input_block_tables)
    read_block_tables[context.provider.source.group.gid] = workspace.window_block_table
    from vllm_ascend.worker.v2.spec_decode.draft_window_eager import AscendDraftAttentionReadInputs

    graph_read_inputs = AscendDraftAttentionReadInputs(
        block_tables=tuple(read_block_tables),
        seq_lens=workspace.device_window_exact,
        seq_lens_cpu_exact=workspace.cpu_window_exact,
        seq_lens_cpu_upper_bound=workspace.cpu_window_upper,
        max_seq_len=max_seq_len,
        dcp_local_seq_lens=dcp_local_seq_lens,
    )
    attn_metadata = build_attn_metadata(
        attn_groups=attn_groups,
        num_reqs=num_reqs,
        num_tokens=num_tokens,
        query_start_loc_gpu=input_batch.query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        max_query_len=num_tokens // num_reqs,
        seq_lens=workspace.device_window_exact,
        seq_lens_cpu=workspace.cpu_window_exact,
        seq_lens_cpu_upper_bound=workspace.cpu_window_upper,
        draft_window_read_inputs=graph_read_inputs,
        dcp_local_seq_lens=dcp_local_seq_lens,
        max_seq_len=max_seq_len,
        block_tables=tuple(read_block_tables),
        slot_mappings=slot_mappings,
        kv_cache_config=kv_cache_config,
        for_cudagraph_capture=True,
        causal=causal,
    )
    return AttentionState(attn_metadata, slot_mappings_by_layer)
