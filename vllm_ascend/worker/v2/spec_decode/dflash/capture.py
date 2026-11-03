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

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import AttentionState
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import (
    _prepare_dflash_inputs_to_capture as _ORIGINAL_PREPARE_DFLASH_INPUTS_TO_CAPTURE,
)
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.worker.v2.spec_decode.draft_window_graph import (
    DraftWindowGraphBindingProvider,
    DraftWindowGraphBucket,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_graph_metadata import bind_draft_window_read

_CAPTURE_CONTEXT_ATTRIBUTE = "_ascend_draft_window_capture_context"


@contextmanager
def draft_window_dflash_capture_scope(owner: Any) -> Iterator[DraftWindowGraphBindingProvider | None]:
    """Attach the admitted provider at the actual Draft capture boundary.

    Startup registration already validates the manager's full descriptors.
    The fixed callback resolves each actual geometry before a capture forward.
    """
    provider = getattr(owner, "_draft_window_graph_binding_provider", None)
    if provider is None:
        yield None
        return
    # The fixed speculator constructs private Draft inputs and passes those
    # same inputs to its manager; Target inputs remain a separate field.
    input_buffers = owner.input_buffers
    if hasattr(input_buffers, _CAPTURE_CONTEXT_ATTRIBUTE):
        raise RuntimeError("Draft Window capture context is already attached")
    setattr(input_buffers, _CAPTURE_CONTEXT_ATTRIBUTE, provider)
    try:
        yield provider
    finally:
        delattr(input_buffers, _CAPTURE_CONTEXT_ATTRIBUTE)


def _seed_capture_workspace(
    workspace: DraftWindowGraphBucket,
    *,
    max_model_len: int,
) -> int:
    # The actual provider borrows this same canonical BlockTables owner.
    # Capture prepares its Full dummy tables before seeding the private bank.
    provider = workspace.owner
    source = provider.table[: workspace.descriptor.num_reqs]
    destination = workspace.window_block_table

    capture_upper = min(
        max_model_len,
        provider.window_size + provider.group.table_column_span - 1,
    )
    valid_num_blocks = (capture_upper + provider.group.table_column_span - 1) // provider.group.table_column_span
    if valid_num_blocks > source.shape[1] or valid_num_blocks > destination.shape[1]:
        raise RuntimeError("Draft Window capture table cannot cover the Window capacity")

    destination.zero_()
    destination[:, :valid_num_blocks].copy_(source[:, :valid_num_blocks])
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
    workspace = (
        None if context is None or skip_attn else context.capture_banks_for(num_reqs=num_reqs, num_tokens=num_tokens)[0]
    )

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
    if workspace is None:
        return state
    if not state.attn_metadata:
        raise RuntimeError("Draft Window capture builder produced no attention metadata")
    capacity = _seed_capture_workspace(workspace, max_model_len=max_model_len)
    for name in workspace.owner.group.layers:
        layer = state.attn_metadata[name]
        bind_draft_window_read(layer, workspace, capacity)
        layer.query_start_loc = input_buffers.query_start_loc[: num_reqs + 1]
    return state
