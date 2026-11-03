# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""A single Full prompt must not expand every concurrent window operand."""

import torch

from vllm_ascend.worker.v2.spec_decode.draft_window_dense import prepare_dense_draft_read


def test_full_prompt_does_not_multiply_decode_scratch_by_full_capacity():
    n, block, full_end, full_width = 16, 128, 32767, 33
    ends = torch.tensor([1151] * (n - 1) + [full_end], dtype=torch.int32)
    offsets = torch.tensor([4 * row for row in range(n)] + [4 * (n - 1) + full_width], dtype=torch.int32)
    table = torch.arange(n * 256, dtype=torch.int32).reshape(n, 256)
    read = prepare_dense_draft_read(
        *(table[list(tuple(range(n)))], (256,) * n, block),
        exact_cpu=ends,
        query_ends_cpu=ends,
        frozen_starts=torch.where(
            torch.tensor([False] * (n - 1) + [True]),
            0,
            torch.div(torch.clamp(ends - 1024, min=0), block, rounding_mode="floor") * block,
        ),
        upper_cpu=ends + 1,
        query_width=full_width,
        causal=False,
        full_rows_cpu=torch.tensor([False] * (n - 1) + [True]),
        query_start_loc_cpu=offsets,
        query_start_loc_device=offsets.clone(),
        retain=lambda *objects: None,
    )
    private_reads = [part for _, part in read.parts] if hasattr(read, "parts") else [read]
    # Bound actual private allocation geometry, rather than a nominal estimate.
    # The original implementation allocated n*32768 KV tokens and n*33*32768
    # mask elements for this mixed request, even though 15 rows only need W.
    kv_tokens = sum(private.slots.numel() for private in private_reads)
    mask_elements = sum(private.mask.numel() for private in private_reads)
    assert kv_tokens == 15 * 1152 + 32768
    assert mask_elements == 15 * 4 * 1152 + full_width * 32768
    assert read.seq_lens.tolist() == [1151] * 15 + [full_end]
    assert (ends - read.seq_lens).tolist() == [0] * n
