# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest

from vllm_ascend.worker.v2.spec_decode.draft_window_plan import (
    compute_draft_window_plan,
    freeze_eagle_window_plan,
    project_draft_window_interval,
    window_capacity_tokens,
)


def test_exact_window_boundaries_against_literal_token_oracle():
    for block in (1, 4, 16, 128):
        for window in (1, 7, 16, 129):
            for future in (0, min(window, 3)):
                for end in (0, 1, block - 1, block, block + 1, 255, 1025):
                    plan = compute_draft_window_plan(
                        full_exact_end=end, window_size=window, block_span_tokens=block, future_offset=future
                    )
                    horizon = end + future
                    # Enumerate block-aligned starts at or before the desired
                    # horizon cut rather than duplicate the production division.
                    legal_starts = range(0, max(horizon - window, 0) + 1, block)
                    expected = list(range(max(legal_starts), end))
                    assert list(range(plan.window_start, plan.full_exact_end)) == expected
                    assert plan.window_visible_length == len(expected)
                    assert plan.window_visible_length <= plan.window_capacity_tokens
                    assert plan.full_horizon_end == horizon


def test_upper_bound_does_not_replace_exact_length():
    exact, upper = 1025, 1088
    plan = compute_draft_window_plan(full_exact_end=exact, window_size=128, block_span_tokens=64)
    assert plan.window_start == 896
    assert (exact - plan.window_start, upper - plan.window_start) == (129, 192)
    wrong_plan = compute_draft_window_plan(full_exact_end=upper, window_size=128, block_span_tokens=64)
    assert wrong_plan.window_start != plan.window_start


def test_eagle_frozen_anchor_survives_all_bounded_rejections():
    for block in (1, 4, 128):
        for end in (0, 1, 7, 8, 127, 128, 255, 1025):
            plan = freeze_eagle_window_plan(
                proposal_exact_end=end, static_k=8, max_rejected_tokens=8, window_size=16, block_span_tokens=block
            )
            for rejection in range(min(end, 8) + 1):
                for step in range(9):
                    step_plan = plan.at_exact_end(end - rejection + step)
                    assert step_plan.window_start == plan.window_start
                    assert step_plan.full_horizon_end == plan.full_horizon_end
                    assert step_plan.window_visible_length >= 0


def test_eagle_invalid_configuration_rejected_before_request_execution():
    with pytest.raises(ValueError, match="maximum rejected"):
        freeze_eagle_window_plan(
            proposal_exact_end=1024, static_k=8, max_rejected_tokens=8, window_size=15, block_span_tokens=128
        )
    plan = freeze_eagle_window_plan(
        proposal_exact_end=1024, static_k=8, max_rejected_tokens=8, window_size=16, block_span_tokens=1
    )
    with pytest.raises(ValueError, match="precedes"):
        plan.at_exact_end(1015)
    with pytest.raises(ValueError, match="horizon"):
        plan.at_exact_end(1033)


def test_each_group_projects_same_logical_interval_without_lcm():
    for span in (8, 16, 32, 64):
        projected = project_draft_window_interval(window_start=96, full_exact_end=137, block_span_tokens=span)
        assert projected.source_start_column * span + projected.first_token_offset == 96
        source_blocks = [list(range(start, start + span)) for start in range(0, 192, span)]
        selected = source_blocks[projected.source_start_column : projected.source_end_column]
        tokens = [token for block in selected for token in block]
        assert tokens[projected.first_token_offset : projected.first_token_offset + 41] == list(range(96, 137))
        assert projected.safe_extent_tokens >= 41
        assert projected.safe_extent_tokens < 41 + span
    assert (projected.source_start_column, projected.first_token_offset, projected.required_columns) == (1, 32, 2)


def test_empty_request_does_not_invent_padding_or_read_columns():
    projection = project_draft_window_interval(window_start=0, full_exact_end=0, block_span_tokens=128)
    assert projection.required_columns == 0
    assert projection.safe_extent_tokens == 0
    assert window_capacity_tokens(1024, 128) == 1151


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"full_exact_end": -1}, ValueError),
        ({"full_exact_end": 1.1}, TypeError),
        ({"future_offset": -1}, ValueError),
        ({"window_size": 0}, ValueError),
        ({"window_size": True}, TypeError),
        ({"block_span_tokens": 0}, ValueError),
        ({"future_offset": 10, "window_size": 1, "block_span_tokens": 1}, ValueError),
    ],
)
def test_invalid_plan_inputs(overrides, error):
    args = dict(full_exact_end=1024, window_size=128, block_span_tokens=16)
    args.update(overrides)
    with pytest.raises(error):
        compute_draft_window_plan(**args)
