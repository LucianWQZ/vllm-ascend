# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest
import torch

from vllm_ascend.worker.v2.spec_decode import draft_window_dense
from vllm_ascend.worker.v2.spec_decode.draft_window_dense import (
    DenseDraftReadBatch,
    prepare_dense_draft_read,
)


@pytest.mark.parametrize("causal", [False, True])
def test_ragged_queries_keep_full_row_and_use_device_exact_at_boundary(causal):
    table = torch.arange(32, dtype=torch.int32).reshape(2, 16)
    source = (table[list((1, 0))], (16, 16), 128)
    before = table.clone()
    offsets = torch.tensor([0, 3, 8], dtype=torch.int32)
    exact = torch.tensor([1151, 1408], dtype=torch.int32)
    held: list[object] = []
    read = prepare_dense_draft_read(
        *source,
        exact_cpu=exact,
        query_ends_cpu=exact,
        frozen_starts=torch.where(
            torch.tensor([False, True]),
            0,
            torch.div(torch.clamp(exact - 1024, min=0), 128, rounding_mode="floor") * 128,
        ),
        upper_cpu=torch.tensor([1152, 1411], dtype=torch.int32),
        query_width=5,
        causal=causal,
        retain=lambda *items: held.extend(items),
        full_rows_cpu=torch.tensor([False, True]),
        query_start_loc_cpu=offsets,
        query_start_loc_device=offsets.clone(),
    )
    assert (exact - read.seq_lens).tolist() == [0, 0]
    assert read.seq_lens.tolist() == [1151, 1408]
    assert read.num_query_tokens == 8
    assert isinstance(read, DenseDraftReadBatch)
    assert [indices.tolist() for indices, _ in read.parts] == [[0, 1, 2], [3, 4, 5, 6, 7]]
    assert [part.capacity for _, part in read.parts] == [1152, 1408]
    for row, (state_row, end, width) in enumerate(((1, 1151, 3), (0, 1408, 5))):
        expected = [int(before[state_row, pos // 128]) * 128 + pos % 128 for pos in range(end)]
        private = read.parts[row][1]
        assert private.slots[0, :end].tolist() == expected
        assert private.slots[0, end:].tolist() == [expected[-1]] * (private.capacity - end)
        for local in range(width):
            length = end - width + local + 1 if causal else end
            assert not private.mask[0, 0, local, :length].any()
            assert private.mask[0, 0, local, length:].all()
    assert torch.equal(table, before)
    offsets.zero_()
    exact.fill_(1)
    assert read.seq_lens.tolist() == [1151, 1408]
    assert read.parts[1][0].tolist() == [3, 4, 5, 6, 7]
    assert any(item is read for item in held)


def test_ragged_eagle_native_padding_preserves_frozen_anchor_and_full_continuation():
    table = torch.arange(64, dtype=torch.int32).reshape(2, 32)
    source = (table[list((0, 1))], (32, 32), 128)
    offsets = torch.tensor([0, 4, 261], dtype=torch.int32)
    read = prepare_dense_draft_read(
        *source,
        exact_cpu=torch.tensor([1795, 2177], dtype=torch.int32),
        query_ends_cpu=torch.tensor([1796, 2177], dtype=torch.int32),
        upper_cpu=torch.tensor([1800, 2180], dtype=torch.int32),
        query_width=257,
        causal=True,
        frozen_starts=torch.where(torch.tensor([False, True]), 0, torch.tensor([640, 1024], dtype=torch.int32)),
        full_rows_cpu=torch.tensor([False, True]),
        query_start_loc_cpu=offsets,
        query_start_loc_device=offsets.clone(),
        retain=lambda *items: None,
    )
    assert (torch.tensor([1795, 2177]) - read.seq_lens).tolist() == [640, 0]
    assert read.seq_lens.tolist() == [1155, 2177]
    assert read.num_query_tokens == 261
    assert isinstance(read, DenseDraftReadBatch)
    assert (~read.parts[0][1].mask[0, 0]).sum(-1).tolist() == [1153, 1154, 1155, 1155]
    assert read.parts[0][0].tolist() == [0, 1, 2, 3]
    assert read.parts[1][0].tolist() == list(range(4, 261))
    assert [part.capacity for _, part in read.parts] == [1152 + 32, 2208]


def test_uniform_full_read_rows_keep_original_query_layout():
    source = (torch.arange(24, dtype=torch.int32).reshape(2, 12)[list((0, 1))], (12, 12), 128)
    read = prepare_dense_draft_read(
        *source,
        exact_cpu=torch.tensor([1151, 1287], dtype=torch.int32),
        query_ends_cpu=torch.tensor([1151, 1287], dtype=torch.int32),
        frozen_starts=torch.where(
            torch.tensor([False, True]),
            0,
            torch.div(
                torch.clamp(torch.tensor([1151, 1287], dtype=torch.int32) - 1024, min=0), 128, rounding_mode="floor"
            )
            * 128,
        ),
        upper_cpu=torch.tensor([1152, 1290], dtype=torch.int32),
        query_width=8,
        causal=False,
        full_rows_cpu=torch.tensor([False, True]),
        retain=lambda *items: None,
    )
    assert (torch.tensor([1151, 1287]) - read.seq_lens).tolist() == [0, 0]
    assert read.num_query_tokens == 16
    assert isinstance(read, DenseDraftReadBatch)
    assert max(part.capacity for _, part in read.parts) == 1312
    assert [indices.tolist() for indices, _ in read.parts] == [list(range(8)), list(range(8, 16))]
    assert all(part.num_query_tokens == indices.numel() for indices, part in read.parts)


@pytest.mark.parametrize("causal", [False, True])
def test_decode_ragged_widths_use_same_partition_without_query_padding(causal, monkeypatch):
    row_copies = []
    copy = draft_window_dense._copy_owned_read_tensor

    def track_row_copy(source, retain, *, device=None):
        if source.ndim == 1 and source.dtype == torch.int64:
            row_copies.append(source.tolist())
        return copy(source, retain, device=device)

    monkeypatch.setattr(draft_window_dense, "_copy_owned_read_tensor", track_row_copy)
    table = torch.arange(48, dtype=torch.int32).reshape(3, 16)
    source = (table[list((2, 0, 1))], (16, 16, 16), 128)
    offsets = torch.tensor([0, 1, 4, 5], dtype=torch.int32)
    exact = torch.tensor([1151, 1291, 1408], dtype=torch.int32)
    read = prepare_dense_draft_read(
        *source,
        exact_cpu=exact,
        query_ends_cpu=exact,
        upper_cpu=exact + 3,
        query_width=3,
        causal=causal,
        frozen_starts=torch.tensor([0, 256, 384], dtype=torch.int32),
        full_rows_cpu=torch.zeros(3, dtype=torch.bool),
        query_start_loc_cpu=offsets,
        query_start_loc_device=offsets.clone(),
        retain=lambda *items: None,
    )
    assert isinstance(read, DenseDraftReadBatch)
    assert read.num_query_tokens == 5
    assert [indices.tolist() for indices, _ in read.parts] == [[0, 4], [1, 2, 3]]
    assert [part.query_width for _, part in read.parts] == [1, 3]
    assert read.seq_lens.tolist() == [1151, 1035, 1024]
    # Query placement and KV table selection share each group's one row copy.
    assert row_copies == [[0, 2], [1]]
    for indices, part in read.parts:
        assert indices.numel() == part.num_query_tokens
        for row in range(part.num_reqs):
            visible = int(part.seq_lens[row])
            allowed = [visible - part.query_width + step + 1 for step in range(part.query_width)]
            assert (~part.mask[row, 0]).sum(-1).tolist() == (allowed if causal else [visible] * part.query_width)
            assert part.slots[row, visible:].eq(part.slots[row, visible - 1]).all()
