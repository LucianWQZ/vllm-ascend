# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Full-read padding must follow each cohort's allocation geometry."""

import pytest
import torch

from vllm_ascend.worker.v2.spec_decode.draft_window_dense import (
    DenseDraftRead,
    prepare_dense_draft_read,
)


@pytest.mark.parametrize("order", [(0, 1, 2), (2, 0, 1), (1, 2, 0)])
def test_different_full_row_bounds_do_not_multiply_largest_workspace(order):
    exact = torch.tensor([1537, 8191, 1152], dtype=torch.int32)[list(order)]
    upper = torch.tensor([1538, 8192, 1152], dtype=torch.int32)[list(order)]
    full = torch.tensor([False, True, True])[list(order)]
    rows = torch.tensor([2, 0, 1], dtype=torch.int64)[list(order)]
    table = torch.arange(3 * 64, dtype=torch.int32).reshape(3, 64)
    original = table.clone()
    held: list[object] = []
    read = prepare_dense_draft_read(
        *(table[list(tuple(rows.tolist()))], tuple(((upper + 127) // 128).tolist()), 128),
        exact_cpu=exact,
        query_ends_cpu=exact,
        frozen_starts=torch.div(torch.clamp(exact - 1024, min=0), 128, rounding_mode="floor") * 128,
        exact_device=exact,
        upper_cpu=upper,
        window_size=1024,
        anchor_span=128,
        query_width=8,
        causal=False,
        full_rows_cpu=full,
        retain=lambda *items: held.extend(items),
    )
    expected_starts = torch.where(full, 0, (exact - 1024).clamp(min=0) // 128 * 128)
    assert torch.equal(exact - read.seq_lens, expected_starts)
    assert torch.equal(read.seq_lens, exact - expected_starts)
    assert torch.equal(table, original)
    parts = (
        ((tuple(range(3)), read),)
        if isinstance(read, DenseDraftRead)
        else tuple((tuple((indices.reshape(-1, 8)[:, 0] // 8).tolist()), part) for indices, part in read.parts)
    )
    for indices, private in parts:
        for local, request in enumerate(indices):
            absolute = torch.arange(int(expected_starts[request]), int(exact[request]))
            expected_slots = table[rows[request], absolute // 128] * 128 + absolute % 128
            assert torch.equal(private.slots[local, : absolute.numel()], expected_slots)
            assert not private.mask[local, ..., : absolute.numel()].any()
    actual_tokens = sum(len(indices) * private.capacity for indices, private in parts)
    row_bounds = exact - expected_starts
    expected_tokens = int(((row_bounds + 31) // 32 * 32).sum())
    assert actual_tokens == expected_tokens, "short Full rows inherited another request's largest capacity"


def test_equal_aligned_full_bounds_keep_one_existing_read():
    table = torch.arange(2 * 17, dtype=torch.int32).reshape(2, 17)
    read = prepare_dense_draft_read(
        *(table[list((0, 1))], (17, 17), 128),
        exact_cpu=torch.tensor([2055, 2063], dtype=torch.int32),
        query_ends_cpu=torch.tensor([2055, 2063], dtype=torch.int32),
        frozen_starts=torch.div(
            torch.clamp(torch.tensor([2055, 2063], dtype=torch.int32) - 1024, min=0), 128, rounding_mode="floor"
        )
        * 128,
        exact_device=torch.tensor([2055, 2063], dtype=torch.int32),
        upper_cpu=torch.tensor([2064, 2070], dtype=torch.int32),
        window_size=1024,
        anchor_span=128,
        query_width=8,
        causal=False,
        full_rows_cpu=torch.tensor([True, True]),
        retain=lambda *items: None,
    )
    assert isinstance(read, DenseDraftRead)
    assert read.capacity == 2080
    assert (torch.tensor([2055, 2063]) - read.seq_lens).tolist() == [0, 0]
    assert read.seq_lens.tolist() == [2055, 2063]
