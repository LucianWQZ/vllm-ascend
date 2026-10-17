# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Independent G3 oracle contracts for Graph characterization."""

from __future__ import annotations

import pytest

from vllm_ascend.worker.v2.spec_decode.draft_window_graph_oracle import (
    compute_graph_characterization_oracle,
)


@pytest.mark.parametrize(
    ("horizon", "exact", "upper", "window", "expected_start", "expected_ids"),
    [
        (32, 32, 32, 64, 0, (100, 101)),
        (80, 80, 80, 64, 16, (101, 102, 103, 104)),
        (81, 81, 95, 64, 16, (101, 102, 103, 104, 105)),
        (4096, 4096, 4103, 1024, 3072, tuple(range(292, 356))),
        # Future-column coverage: H crosses a block while E remains exact.
        (67, 65, 67, 64, 0, (100, 101, 102, 103, 104)),
    ],
)
def test_independent_graph_oracle_covers_base_boundaries(
    horizon: int,
    exact: int,
    upper: int,
    window: int,
    expected_start: int,
    expected_ids: tuple[int, ...],
) -> None:
    full_table = tuple(range(100, 500))
    result = compute_graph_characterization_oracle(
        full_table=full_table,
        full_horizon_end=horizon,
        full_exact_end=exact,
        full_upper_end=upper,
        window_size=window,
        legacy_block_size=16,
        block_span_tokens=16,
        query_width=7,
        valid_block_count=len(full_table),
    )
    assert result.window_start == expected_start
    assert result.visible_length == exact - expected_start
    assert result.expected_block_ids == expected_ids
    assert result.window_upper >= result.visible_length
    assert result.window_upper <= result.safe_extent_tokens


def test_independent_graph_oracle_rejects_source_shortage() -> None:
    with pytest.raises(RuntimeError, match="source shortage"):
        compute_graph_characterization_oracle(
            full_table=(1, 2, 3, 4),
            full_horizon_end=64,
            full_exact_end=64,
            full_upper_end=64,
            window_size=64,
            legacy_block_size=16,
            block_span_tokens=16,
            query_width=7,
            valid_block_count=3,
        )


def test_independent_graph_oracle_rejects_nonzero_direct_prefix_offset() -> None:
    with pytest.raises(NotImplementedError, match="zero offset"):
        compute_graph_characterization_oracle(
            full_table=tuple(range(20)),
            full_horizon_end=72,
            full_exact_end=72,
            full_upper_end=72,
            window_size=64,
            legacy_block_size=8,
            block_span_tokens=16,
            query_width=7,
            valid_block_count=20,
        )
