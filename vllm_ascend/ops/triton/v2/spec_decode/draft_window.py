# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Plan private eager Draft reads from current device lengths."""

from collections.abc import Callable

import torch
from vllm.triton_utils import tl, triton

GATHER_TOKEN_BLOCK_SIZE = 8


@triton.jit
def _gather_draft_kv_fp32(
    slots,
    keys,
    values,
    out_keys,
    out_values,
    KEY_PAGE_STRIDE: tl.constexpr,
    VALUE_PAGE_STRIDE: tl.constexpr,
    BLOCK_SPAN: tl.constexpr,
    TOKEN_SIZE: tl.constexpr,
    TOKEN_PADDING: tl.constexpr,
    TOKEN_BLOCK: tl.constexpr,
    CHANNEL_BLOCK: tl.constexpr,
):
    # Infer allocation extent from the launch grid. Only the final tile's
    # padding specializes this kernel, not each Full-prefill capacity.
    total_tokens = tl.num_programs(0) * TOKEN_BLOCK - TOKEN_PADDING
    token = tl.program_id(0) * TOKEN_BLOCK + tl.arange(0, TOKEN_BLOCK)
    channel = tl.arange(0, CHANNEL_BLOCK)
    slot = tl.load(slots + token, mask=token < total_tokens, other=0).to(tl.int64)
    page, offset = slot // BLOCK_SPAN, slot % BLOCK_SPAN
    # Compute each physical token address once, then broadcast its channels.
    # The flat version repeated the slot load and page arithmetic per channel.
    valid = (token[:, None] < total_tokens) & (channel[None, :] < TOKEN_SIZE)
    output_offset = token[:, None] * TOKEN_SIZE + channel[None, :]
    key_offset = (page * KEY_PAGE_STRIDE + offset * TOKEN_SIZE)[:, None] + channel[None, :]
    key = tl.load(keys + key_offset, mask=valid, other=0)
    tl.store(out_keys + output_offset, key.to(tl.float32), mask=valid)
    value_offset = (page * VALUE_PAGE_STRIDE + offset * TOKEN_SIZE)[:, None] + channel[None, :]
    value = tl.load(values + value_offset, mask=valid, other=0)
    tl.store(out_values + output_offset, value.to(tl.float32), mask=valid)


def gather_draft_kv_fp32(
    slots: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    retain: Callable[..., None] | None = None,
    output_buffers: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather validated ND token rows directly into private native FP32 operands.

    Tensor pointers preserve their own storage offsets; K/V page strides can
    differ. The caller validates contiguous token rows and owns both outputs
    until reader completion. Slots already exclude every uninitialized tail.
    """
    if not slots.is_contiguous():
        raise ValueError("Fused Draft gather requires contiguous private slots")
    shape = (*slots.shape, *keys.shape[2:])
    if output_buffers is None:
        gathered_keys = torch.empty(shape, dtype=torch.float32, device=keys.device)
        gathered_values = torch.empty_like(gathered_keys)
    else:
        import torch_npu

        if len(output_buffers) != 2 or any(
            buffer.shape != shape
            or buffer.dtype != torch.float32
            or buffer.device != keys.device
            or not buffer.is_contiguous()
            or torch_npu.get_npu_format(buffer) not in (0, 2)
            for buffer in output_buffers
        ):
            raise ValueError("Reusable Draft gather outputs must match contiguous ND FP32 operands")
        gathered_keys, gathered_values = output_buffers
        regions = []
        for tensor in (slots, keys, values, gathered_keys, gathered_values):
            storage = tensor.untyped_storage()
            regions.append((storage.data_ptr(), storage.nbytes()))
        for index in (3, 4):
            start, size = regions[index]
            if any(start < other + count and other < start + size for other, count in regions[:index]):
                raise ValueError("Reusable Draft outputs must not alias Full KV, slots or one another")
    if retain is not None:
        retain(slots, keys, values, gathered_keys, gathered_values)
    token_size = keys.shape[2] * keys.shape[3]
    _gather_draft_kv_fp32[(triton.cdiv(slots.numel(), GATHER_TOKEN_BLOCK_SIZE),)](
        slots,
        keys,
        values,
        gathered_keys,
        gathered_values,
        keys.stride(0),
        values.stride(0),
        keys.shape[1],
        token_size,
        -slots.numel() % GATHER_TOKEN_BLOCK_SIZE,
        GATHER_TOKEN_BLOCK_SIZE,
        triton.next_power_of_2(token_size),
    )
    return gathered_keys, gathered_values
