# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Prepared exact reads for ordinary dense, parallel eager proposals.

Capacity controls allocation only. Gather only valid KV tokens, repeating the
last valid token into masked scratch padding: masking uninitialized cache tail
directly is unsafe because zero attention weights can still multiply NaNs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

DENSE_KV_ALIGNMENT = 32


def _apply_read_metadata(read: DenseDraftRead | DenseDraftReadBatch, common: Any) -> None:
    n = common.num_reqs
    lengths = read.seq_lens
    if n != read.num_reqs:
        lengths = read.seq_lens.new_zeros(n)
        read.retain(lengths)
        lengths[: read.num_reqs] = read.seq_lens
    common.seq_lens = lengths
    # These are explicitly capacity/workspace bounds, never CPU exact.
    common.seq_lens_cpu_upper_bound = torch.full((n,), read.capacity, dtype=torch.int32, device="cpu")
    common.seq_lens_cpu = None
    common._seq_lens_cpu = None
    common.max_seq_len = read.capacity
    common._draft_window_device_read = read


def _read_capacity_cpu(read: DenseDraftRead | DenseDraftReadBatch, common: Any) -> torch.Tensor:
    if getattr(common, "_draft_window_device_read", None) is not read:
        raise ValueError("Dense Draft read belongs to another common metadata object")
    return common.seq_lens_cpu_upper_bound


@dataclass(frozen=True)
class DenseDraftRead:
    slots: torch.Tensor
    seq_lens: torch.Tensor
    mask: torch.Tensor
    capacity: int
    query_width: int
    retain: Callable[..., None]
    may_have_empty_rows: bool = False
    _attention_workspaces: dict[tuple[int, ...], tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict, init=False, compare=False, repr=False
    )

    @property
    def num_query_tokens(self) -> int:
        return self.num_reqs * self.query_width

    @property
    def num_reqs(self) -> int:
        return self.slots.shape[0]

    apply = _apply_read_metadata
    capacity_cpu = _read_capacity_cpu


@dataclass(frozen=True)
class DenseDraftReadBatch:
    """Private reads partitioned by phase, queries and Full allocation geometry.

    An unfinished prompt can need Full KV while concurrent decodes need W.
    Separate native operands avoid padding every decode to that Full capacity
    or every decode query to the prompt's query width. This never partitions
    by device E, S, rejection, or any other value requiring a Host read.
    """

    parts: tuple[tuple[torch.Tensor, DenseDraftRead], ...]
    seq_lens: torch.Tensor
    capacity: int
    num_query_tokens: int
    retain: Callable[..., None]

    @property
    def num_reqs(self) -> int:
        return self.seq_lens.numel()

    apply = _apply_read_metadata
    capacity_cpu = _read_capacity_cpu


def _copy_owned_read_tensor(
    source: torch.Tensor, retain: Callable[..., None], *, device: torch.device | None = None
) -> torch.Tensor:
    """Own both operands before a copy can partially submit and raise."""
    destination = torch.empty_like(source, device=source.device if device is None else device)
    retain(source, destination)
    destination.copy_(source, non_blocking=True)
    return destination


def prepare_dense_draft_read(
    block_table: torch.Tensor,
    valid_columns: Sequence[int],
    block_span: int,
    *,
    exact_device: torch.Tensor,
    exact_cpu: torch.Tensor | None = None,
    query_ends_cpu: torch.Tensor | None = None,
    upper_cpu: torch.Tensor,
    window_size: int,
    anchor_span: int,
    query_width: int,
    causal: bool,
    retain: Callable[..., None],
    frozen_starts: torch.Tensor | None = None,
    query_ends: torch.Tensor | None = None,
    full_rows_cpu: torch.Tensor | None = None,
    query_start_loc_cpu: torch.Tensor | None = None,
    query_start_loc_device: torch.Tensor | None = None,
) -> DenseDraftRead | DenseDraftReadBatch:
    """Use completed current exact lengths, not optimistic scheduling bounds.

    Runtime shares one exact snapshot across groups and supplies frozen Eagle
    starts and native query ends from that same preparation. All read slots
    lie in [S,E); capacity padding repeats its last initialized token. Query
    padding and Full continuing rows retain the method's existing semantics.
    """
    n = len(valid_columns)
    if block_table.ndim != 2 or block_table.dtype != torch.int32 or block_span <= 0:
        raise ValueError("Invalid compact dense Draft table geometry")
    if block_table.shape[0] < n or any(count < 0 or count > block_table.shape[1] for count in valid_columns):
        raise ValueError("Invalid initialized compact column extent")
    if n < 1 or window_size < 1 or query_width < 1 or anchor_span < 1 or anchor_span % block_span:
        raise ValueError("Invalid dense Draft read geometry")
    if upper_cpu.device.type != "cpu" or upper_cpu.shape != (n,) or upper_cpu.dtype not in (torch.int32, torch.int64):
        raise ValueError("Missing current preparation allocation upper bounds")
    available = torch.tensor(valid_columns, dtype=torch.int64, device="cpu") * block_span
    if bool((upper_cpu <= 0).any()):
        raise ValueError("Allocation upper must be positive")
    if exact_device.ndim != 1 or exact_device.numel() < n:
        raise ValueError("Missing current device lengths")
    if exact_device.dtype not in (torch.int32, torch.int64):
        raise ValueError("Dense read lengths must be integer tensors")
    if exact_device.device != block_table.device:
        raise ValueError("Dense read operands must share their source device")
    retain(exact_device, block_table)
    # A single completed snapshot certifies this preparation. The runtime
    # shares it across groups; direct callers may request their own snapshot.
    exact_cpu = exact_device[:n].to(device="cpu", dtype=torch.int32, copy=True) if exact_cpu is None else exact_cpu
    if exact_cpu.device.type != "cpu" or exact_cpu.shape != (n,) or exact_cpu.dtype not in (torch.int32, torch.int64):
        raise ValueError("Dense Draft read requires one completed CPU exact value per request")
    if bool(((exact_cpu <= 0) | (exact_cpu > upper_cpu) | (exact_cpu > available)).any()):
        raise ValueError("Dense Draft exact lengths exceed initialized canonical columns or preparation bounds")
    if query_ends_cpu is None:
        query_ends_cpu = exact_cpu if query_ends is None else query_ends.to(device="cpu", dtype=torch.int32, copy=True)
    if (
        query_ends_cpu.device.type != "cpu"
        or query_ends_cpu.shape != (n,)
        or query_ends_cpu.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("Native query ends must match current request rows")
    if frozen_starts is not None and frozen_starts.dtype not in (torch.int32, torch.int64):
        raise ValueError("Frozen Draft anchors must contain integers")
    starts = (
        torch.div(torch.clamp(exact_cpu - window_size, min=0), anchor_span, rounding_mode="floor") * anchor_span
        if frozen_starts is None
        else frozen_starts.to(device="cpu", dtype=torch.int32)
    )
    if starts.shape != (n,) or bool(((starts < 0) | (starts > exact_cpu) | (starts % anchor_span != 0)).any()):
        raise ValueError("Frozen Draft anchors must match current request rows")
    retain(exact_cpu, starts, query_ends_cpu)
    if full_rows_cpu is not None:
        if full_rows_cpu.device.type != "cpu" or full_rows_cpu.dtype != torch.bool or full_rows_cpu.shape != (n,):
            raise ValueError("Full-read phase rows require one current CPU boolean per request")
    if (query_start_loc_cpu is None) != (query_start_loc_device is None):
        raise ValueError("Packed queries require current CPU and device boundaries")
    widths = None
    if query_start_loc_cpu is not None:
        assert query_start_loc_device is not None
        if (
            query_start_loc_cpu.device.type != "cpu"
            or query_start_loc_cpu.dtype not in (torch.int32, torch.int64)
            or query_start_loc_cpu.shape != (n + 1,)
            or query_start_loc_device.shape != (n + 1,)
            or query_start_loc_device.device != exact_device.device
            or query_start_loc_device.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("Packed query boundaries must match the active preparation")
        widths = query_start_loc_cpu.diff()
        if int(query_start_loc_cpu[0]) != 0 or bool(((widths <= 0) | (widths > query_width)).any()):
            raise ValueError("Packed query widths exceed their current allocation bound")

    def materialize(host_rows: torch.Tensor, width: int, full: bool) -> DenseDraftRead:
        ends = exact_cpu.index_select(0, host_rows)
        origins = starts.index_select(0, host_rows)
        if full:
            origins.zero_()
        visible = ends - origins
        capacity_bound = int(visible.max())
        capacity = (capacity_bound + DENSE_KV_ALIGNMENT - 1) // DENSE_KV_ALIGNMENT * DENSE_KV_ALIGNMENT
        columns = torch.arange(capacity, device="cpu", dtype=torch.int64)
        absolute = origins[:, None] + torch.minimum(columns, visible[:, None] - 1)
        address = _copy_owned_read_tensor(absolute, retain, device=block_table.device)
        logical_columns = torch.div(address, block_span, rounding_mode="floor")
        if host_rows.numel() == n:
            blocks = block_table[:n].gather(1, logical_columns)
        else:
            physical_rows = _copy_owned_read_tensor(host_rows, retain, device=block_table.device)
            if block_table.is_contiguous():
                indices = physical_rows[:, None] * block_table.shape[1] + logical_columns
                blocks = block_table.view(-1).index_select(0, indices.flatten()).reshape_as(address)
            else:
                blocks = block_table[physical_rows[:, None], logical_columns]
        slots = blocks.to(torch.int64) * block_span + address % block_span
        excluded = (columns[None, None, :] >= visible[:, None, None]).expand(-1, width, -1)
        if causal:
            query_end = query_ends_cpu.index_select(0, host_rows)
            queries = query_end[:, None] - origins[:, None] - width + torch.arange(width, device="cpu")
            excluded = excluded | (columns[None, None, :] > queries[:, :, None])
        mask = _copy_owned_read_tensor(excluded[:, None].contiguous(), retain, device=block_table.device)
        lengths = _copy_owned_read_tensor(visible, retain, device=block_table.device)
        read = DenseDraftRead(slots, lengths, mask, capacity, width, retain, causal and window_size < width)
        retain(read, block_table, address, logical_columns, blocks, slots, lengths, mask)
        return read

    if widths is None and (full_rows_cpu is None or not bool(full_rows_cpu.any())):
        return materialize(torch.arange(n, device="cpu"), query_width, False)
    groups: dict[tuple[bool, int, int], list[int]] = {}
    for row in range(n):
        full = full_rows_cpu is not None and bool(full_rows_cpu[row])
        capacity = (int(upper_cpu[row]) + DENSE_KV_ALIGNMENT - 1) // DENSE_KV_ALIGNMENT if full else 0
        key = (full, query_width if widths is None else int(widths[row]), capacity)
        groups.setdefault(key, []).append(row)
    if len(groups) == 1:
        full, width, _ = next(iter(groups))
        return materialize(torch.arange(n, device="cpu"), width, full)
    visible = torch.empty(n, dtype=exact_cpu.dtype, device=block_table.device)
    retain(visible)
    parts = []
    for (full, width, _capacity), group in groups.items():
        host_rows = torch.tensor(group, dtype=torch.int64, device="cpu")
        selected = _copy_owned_read_tensor(host_rows, retain, device=block_table.device)
        bounds = (
            selected * query_width
            if query_start_loc_device is None
            else query_start_loc_device.index_select(0, selected)
        )
        queries = (bounds[:, None] + torch.arange(width, device=selected.device)[None, :]).flatten()
        retain(bounds, queries)
        read = materialize(host_rows, width, full)
        visible.index_copy_(0, selected, read.seq_lens)
        parts.append((queries, read))
    result = DenseDraftReadBatch(
        tuple(parts), visible, max(read.capacity for _, read in parts), sum(q.numel() for q, _ in parts), retain
    )
    retain(result)
    return result


def gather_dense_draft_kv(
    read: DenseDraftRead,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    output_dtype: torch.dtype | None = None,
    output_buffers: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather ND token rows, including interleaved/padded physical KV pages."""
    if key_cache.ndim != 4 or key_cache.shape != value_cache.shape:
        raise ValueError("Dense Draft KV requires matching four-dimensional caches")
    if key_cache.dtype not in (torch.float16, torch.bfloat16) or value_cache.dtype != key_cache.dtype:
        raise ValueError("Dense Draft KV requires unquantized float16/bfloat16 storage")
    if output_dtype not in (None, torch.float32):
        raise ValueError("Dense Draft gather conversion supports only FP32 native operands")
    if key_cache.device != value_cache.device or read.slots.device != key_cache.device:
        raise ValueError("Dense Draft slots and KV caches must share their device")
    shape = (read.num_reqs, read.capacity, *key_cache.shape[2:])
    layouts = []
    for cache in (key_cache, value_cache):
        blocks, block_size, heads, dim = cache.shape
        token_size = heads * dim
        page_stride = cache.stride(0)
        if (
            cache.stride()[1:] != (token_size, dim, 1)
            or page_stride < block_size * token_size
            or page_stride % token_size
        ):
            raise ValueError("Dense Draft KV requires contiguous ND token rows and aligned physical pages")
        stride_rows = page_stride // token_size
        layouts.append((cache, blocks, block_size, heads, dim, token_size, stride_rows))
    if output_dtype == torch.float32 and key_cache.device.type == "npu":
        import torch_npu

        from vllm_ascend.ops.triton.v2.spec_decode.draft_window import gather_draft_kv_fp32

        # Raw Triton addresses require an ND physical layout. Other formats
        # retain the existing format-aware PyTorch gather/conversion path.
        if all(torch_npu.get_npu_format(cache) in (0, 2) for cache in (key_cache, value_cache)):
            keys, values = gather_draft_kv_fp32(
                read.slots,
                key_cache,
                value_cache,
                retain=read.retain,
                **({"output_buffers": output_buffers} if output_buffers is not None else {}),
            )
            read.retain(keys, values)
            return keys, values
    if output_buffers is not None:
        raise ValueError("Reusable Draft operands require the native ND FP32 gather")
    slots = read.slots.reshape(-1)
    indices_by_stride = {}
    gathered = []
    for cache, blocks, block_size, heads, dim, token_size, stride_rows in layouts:
        # Preserve the original storage offset, including the V offset in a
        # combined K/V page. This is a view, never a full-cache flatten/copy.
        rows = cache.as_strided(((blocks - 1) * stride_rows + block_size, heads, dim), (token_size, dim, 1))
        if stride_rows not in indices_by_stride:
            indices_by_stride[stride_rows] = (
                slots
                if stride_rows == block_size
                else slots + torch.div(slots, block_size, rounding_mode="floor") * (stride_rows - block_size)
            )
        operand = rows.index_select(0, indices_by_stride[stride_rows]).reshape(shape)
        read.retain(operand)
        gathered.append(operand.to(output_dtype or cache.dtype))
    keys, values = gathered
    read.retain(keys, values)
    return keys, values


def dense_draft_attention(
    read: DenseDraftRead | DenseDraftReadBatch,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Consume the completed exact read projection with native FP32 attention.

    Keep native inputs in FP32: fused BF16 attention lost small cancellation
    results in a controlled real-layer same-window comparison. The native FA
    consumer uses the device mask and padded initialized KV without explicit
    GQA expansion. All private operands and returned workspaces remain owned
    through reader completion.
    """
    # Lazy import keeps leaf planning/ownership usable in CPU-only tests.
    import torch_npu

    if isinstance(read, DenseDraftReadBatch):
        output = query.new_empty((read.num_query_tokens, *query.shape[-2:]))
        read.retain(output)
        for indices, part in read.parts:
            queries = query.index_select(0, indices)
            read.retain(queries)
            value = dense_draft_attention(part, queries, key_cache, value_cache, scale)
            output.index_copy_(0, indices, value)
        return output
    workspace_key = None
    if (
        key_cache.device.type == "npu"
        and value_cache.device == key_cache.device
        and key_cache.ndim == value_cache.ndim == 4
        and all(torch_npu.get_npu_format(cache) in (0, 2) for cache in (key_cache, value_cache))
    ):
        # Only this immediate native FA consumer reuses scratch. Its next
        # gather is queued after the preceding FA on the same NPU stream,
        # so the overwrite executes after that reader. A different stream,
        # KV geometry, read/group or proposal gets independent storage.
        # Standalone gather callers retain independent materializations.
        stream = torch.npu.current_stream(key_cache.device)
        workspace_key = (stream.npu_stream, read.num_reqs, read.capacity, *key_cache.shape[2:])
    buffers = read._attention_workspaces.get(workspace_key) if workspace_key is not None else None
    keys_fp32, values_fp32 = gather_dense_draft_kv(
        read,
        key_cache,
        value_cache,
        output_dtype=torch.float32,
        **({"output_buffers": buffers} if buffers is not None else {}),
    )
    if workspace_key is not None:
        read._attention_workspaces[workspace_key] = (keys_fp32, values_fp32)
    query = query[: read.num_query_tokens]
    query = query.reshape(read.num_reqs, read.query_width, *query.shape[-2:])
    queries_fp32 = query.float().contiguous()
    read.retain(queries_fp32, keys_fp32, values_fp32)
    outputs = torch_npu.npu_fusion_attention(
        queries_fp32,
        keys_fp32,
        values_fp32,
        head_num=query.shape[-2],
        input_layout="BSND",
        atten_mask=read.mask,
        scale=scale,
        keep_prob=1.0,
        sparse_mode=0,
        inner_precise=0,
    )
    read.retain(*outputs)
    result = outputs[0]
    if read.may_have_empty_rows:
        empty_rows = read.mask.all(-1, keepdim=True).transpose(1, 2)
        read.retain(empty_rows)
        result.masked_fill_(empty_rows, 0)
    output = result.to(query.dtype).flatten(0, 1)
    read.retain(output)
    return output
