# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Real row phases and canonical count gathering without summary wrappers."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from vllm_ascend.worker.v2.spec_decode.draft_window import (
    DraftWindowSourceCapability,
    build_full_draft_kv_source,
    gather_valid_num_blocks_cpu,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_backends import (
    DraftWindowGroupRoute,
    DraftWindowProviderStatus,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_eager import DraftWindowProposalEpochSource
from vllm_ascend.worker.v2.spec_decode.draft_window_nongraph import (
    build_phase_aware_materialization_plan,
    classify_draft_window_batch_phase,
)


def _batch(rows):
    computed, prompt, scheduled = (
        (np.array(values, dtype=np.int64) for values in zip(*rows))
        if rows
        else (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
        )
    )
    return SimpleNamespace(
        num_reqs=len(rows),
        req_ids=[f"request-{row}" for row in range(len(rows))],
        idx_mapping_np=np.arange(len(rows), dtype=np.int64),
        num_computed_prefill_tokens_np=computed,
        prefill_len_np=prompt,
        num_scheduled_tokens=scheduled,
        num_draft_tokens_per_req=None,
        is_prefilling_np=computed < prompt,
        has_prefill=bool(np.any(computed < prompt)),
    )


@pytest.mark.parametrize(
    "rows,full_rows,baseline,bypass",
    [
        ([(0, 8, 2)], (True,), (0,), True),
        ([(6, 8, 2)], (False,), (), False),
        ([(0, 8, 2), (8, 8, 1)], (True, False), (0,), False),
        ([], (), (), False),
    ],
)
def test_bypass_requires_every_real_row_to_be_continuing(rows, full_rows, baseline, bypass):
    plan = classify_draft_window_batch_phase(_batch(rows))
    assert plan.full_read_rows == full_rows
    assert plan.baseline_only_rows == baseline
    assert plan.bypass_entire_batch is bypass


@pytest.mark.parametrize("field", ["req_ids", "idx_mapping_np", "num_scheduled_tokens"])
def test_epoch_still_rejects_changed_requests_mapping_or_phases(field):
    batch = _batch([(0, 8, 2)])
    owner = DraftWindowProposalEpochSource()
    epoch = owner.begin(batch)
    replacement = {"req_ids": ["replacement"], "idx_mapping_np": np.array([1]), "num_scheduled_tokens": np.array([8])}
    setattr(batch, field, replacement[field])
    with pytest.raises(RuntimeError, match="ordering|idx_mapping|row phases"):
        owner.validate(epoch, batch)


@pytest.fixture
def full_source():
    group = DraftWindowGroupRoute(0, ("draft.layer",), 4, 4, DraftWindowProviderStatus.DIRECT_DENSE)
    counts = torch.tensor([[4, 3]], dtype=torch.int32)
    owner = SimpleNamespace(
        input_block_tables=[torch.arange(8, dtype=torch.int32).reshape(2, 4)],
        num_blocks=SimpleNamespace(cpu=counts, gpu=counts.clone()),
        block_sizes=[4],
        kernel_block_sizes=[4],
    )
    return build_full_draft_kv_source(group=group, block_tables=owner)


def test_canonical_cpu_counts_keep_mapping_order_padding_and_source(full_source):
    before = full_source._persistent_num_blocks_cpu.clone()
    actual = gather_valid_num_blocks_cpu(
        full_source, global_group_id=0, idx_mapping_np=np.array([1, 0]), num_reqs=2, num_reqs_padded=3
    )
    assert actual.tolist() == [3, 4, 0]
    assert actual.data_ptr() != full_source._persistent_num_blocks_cpu.data_ptr()
    assert torch.equal(full_source._persistent_num_blocks_cpu, before)


@pytest.mark.parametrize("mapping", [np.array([0.5]), np.array([2])])
def test_canonical_count_gather_rejects_invalid_mapping(full_source, mapping):
    with pytest.raises((TypeError, IndexError)):
        gather_valid_num_blocks_cpu(
            full_source, global_group_id=0, idx_mapping_np=mapping, num_reqs=1, num_reqs_padded=1
        )


@pytest.mark.parametrize("count", [-1, 5])
def test_phase_materialization_rejects_counts_outside_canonical_row(full_source, count):
    source: Any = SimpleNamespace(group=SimpleNamespace(table_column_span=4, gid=0), full_source=full_source)
    phase = classify_draft_window_batch_phase(_batch([(0, 8, 2)]))
    with pytest.raises(RuntimeError, match="valid-block count"):
        build_phase_aware_materialization_plan(
            source=source,
            phase_plan=phase,
            cpu_full_exact=torch.tensor([2], dtype=torch.int32),
            cpu_valid_num_blocks=torch.tensor([count], dtype=torch.int32),
            actual_request_count=1,
            padded_request_count=1,
        )


@pytest.mark.parametrize("step,end", [(0, 14), (1, 13), (2, 14)])
def test_shared_row_projection_keeps_eagle_anchor_and_full_prefill(full_source, step, end):
    from vllm_ascend.worker.v2.spec_decode.draft_window_plan import compute_draft_window_plan

    group = DraftWindowGroupRoute(0, ("draft.layer",), 4, 4, DraftWindowProviderStatus.DIRECT_DENSE)
    from vllm_ascend.worker.v2.spec_decode.draft_window_policy import resolve_draft_window_policy

    policy = resolve_draft_window_policy(requested_window_size=8, method="eagle3")
    source = DraftWindowSourceCapability(
        policy=policy,
        runtime_route="eagle3",
        group=group,
        full_source=full_source,
        expected_future_offset=3,
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        tensor_parallel_rank=0,
    )
    frozen = tuple(
        compute_draft_window_plan(full_exact_end=end, future_offset=3, window_size=8, block_span_tokens=4)
        for end in (14, 2)
    )
    phases = classify_draft_window_batch_phase(_batch([(8, 8, 1), (0, 8, 2)]))
    result = build_phase_aware_materialization_plan(
        source=source,
        frozen_row_plans=frozen,
        actual_request_count=2,
        padded_request_count=3,
        phase_plan=phases,
        cpu_full_exact=torch.tensor([end, 2 + step, 0], dtype=torch.int32),
        cpu_valid_num_blocks=torch.tensor([4, 3, 0], dtype=torch.int32),
    )
    assert frozen[0].window_start == 8
    assert result.read_descriptors[0].source_start_column == 2
    assert result.cpu_window_exact.tolist() == [end - 8, 2 + step, 0]
    assert result.read_descriptors[1].source_start_column == 0
    assert result.read_descriptors[1].required_columns == 3
    assert full_source.input_block_table_row_width(0) == 4
