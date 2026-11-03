# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest

from vllm_ascend.worker.v2.spec_decode.draft_window_plan import (
    compute_draft_window_start,
    window_capacity_tokens,
)


def test_exact_window_boundaries_against_literal_token_oracle():
    for block in (1, 4, 16, 128):
        for window in (1, 7, 16, 129):
            for future in (0, min(window, 3)):
                for end in (0, 1, block - 1, block, block + 1, 255, 1025):
                    start = compute_draft_window_start(
                        full_exact_end=end, window_size=window, block_span_tokens=block, future_offset=future
                    )
                    horizon = end + future
                    # Enumerate block-aligned starts at or before the desired
                    # horizon cut rather than duplicate the production division.
                    legal_starts = range(0, max(horizon - window, 0) + 1, block)
                    expected = list(range(max(legal_starts), end))
                    assert list(range(start, end)) == expected
                    assert end - start == len(expected)
                    assert end - start <= window_capacity_tokens(window, block)


def test_upper_bound_does_not_replace_exact_length():
    exact, upper = 1025, 1088
    start = compute_draft_window_start(full_exact_end=exact, window_size=128, block_span_tokens=64)
    assert start == 896
    assert (exact - start, upper - start) == (129, 192)
    wrong_start = compute_draft_window_start(full_exact_end=upper, window_size=128, block_span_tokens=64)
    assert wrong_start != start


def test_eagle_frozen_anchor_survives_all_bounded_rejections():
    for block in (1, 4, 128):
        for end in (0, 1, 7, 8, 127, 128, 255, 1025):
            start = compute_draft_window_start(
                full_exact_end=end, future_offset=8, window_size=16, block_span_tokens=block
            )
            for rejection in range(min(end, 8) + 1):
                for step in range(9):
                    current_end = end - rejection + step
                    assert start <= current_end <= end + 8
                    assert current_end - start >= 0


def test_empty_interval_keeps_its_actual_zero_visible_length():
    start = compute_draft_window_start(full_exact_end=0, window_size=1024, block_span_tokens=128)
    assert start == 0
    assert window_capacity_tokens(1024, 128) == 1151


@pytest.mark.parametrize("exact,future,window", [(-1, 0, 128), (1024, 10, 1)])
def test_unreadable_start_is_rejected(exact, future, window):
    with pytest.raises(ValueError):
        compute_draft_window_start(full_exact_end=exact, future_offset=future, window_size=window, block_span_tokens=1)
