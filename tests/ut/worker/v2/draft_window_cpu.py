# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU geometry references for the two dense Draft planning leaf kernels.

Tests retain the real preparation, metadata and ownership code. These local
references replace only native plan/pack launches and are not a CPU service.
"""

from collections.abc import Callable

import pytest
import torch


def _pack_cpu(
    offsets: torch.Tensor, total: int, query_width: int, *, retain: Callable[..., None] | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    assert offsets.device.type == "cpu"
    widths = offsets.diff()
    padding = query_width - widths
    local = torch.arange(query_width, device=offsets.device)[None, :]
    packed = offsets[:-1, None] + (local - padding[:, None]).clamp(min=0)
    restored = torch.cat(
        [row * query_width + padding[row] + torch.arange(int(width)) for row, width in enumerate(widths)]
    )
    assert restored.numel() == total
    return packed, restored


def _plan_cpu(
    table: torch.Tensor,
    rows: torch.Tensor,
    exact: torch.Tensor,
    frozen: torch.Tensor | None,
    *,
    block_span: int,
    anchor_span: int,
    window: int,
    capacity: int,
    query_width: int,
    causal: bool,
    query_ends: torch.Tensor | None = None,
    full_rows: torch.Tensor | None = None,
    query_offsets: torch.Tensor | None = None,
    retain: Callable[..., None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert exact.device.type == "cpu"
    n = exact.numel()
    starts = (
        torch.div(torch.clamp(exact - window, min=0), anchor_span, rounding_mode="floor") * anchor_span
        if frozen is None
        else frozen.clone()
    )
    visible = exact - starts
    if full_rows is not None:
        starts = torch.where(full_rows, 0, starts)
        visible = exact - starts
    columns = torch.arange(capacity, device=exact.device, dtype=torch.int64)
    absolute = starts[:, None] + torch.minimum(columns[None, :], visible[:, None] - 1)
    block_columns = torch.div(absolute, block_span, rounding_mode="floor")
    table_indices = rows[:, None] * table.shape[1] + block_columns
    blocks = table.reshape(-1).index_select(0, table_indices.reshape(-1)).reshape(n, capacity)
    slots = blocks.to(torch.int64) * block_span + absolute % block_span
    mask = (columns[None, None, :] >= visible[:, None, None]).expand(n, query_width, capacity)
    if causal:
        query_ends = exact if query_ends is None else query_ends
        queries = (
            query_ends[:, None]
            - starts[:, None]
            - query_width
            + torch.arange(query_width, device=exact.device)[None, :]
        )
        mask = mask | (columns[None, None, :] > queries[:, :, None])
    if query_offsets is not None:
        padding = query_width - query_offsets.diff()
        mask = mask | (torch.arange(query_width)[None, :, None] < padding[:, None, None])
    return slots, starts, visible, mask[:, None].contiguous()


@pytest.fixture(autouse=True)
def cpu_dense_planning(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_ascend.ops.triton.v2.spec_decode import draft_window

    monkeypatch.setattr(draft_window, "pack_draft_query_indices", _pack_cpu)
    monkeypatch.setattr(draft_window, "plan_draft_window", _plan_cpu)
