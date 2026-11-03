# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Real row phases and canonical count gathering without summary wrappers."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_ascend.worker.v2.spec_decode.draft_window import (
    DraftWindowGraphSource,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_backends import (
    DraftWindowGroupRoute,
    DraftWindowProviderStatus,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_runtime import continuing_prefill_rows


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
    "rows,full_rows,bypass",
    [
        ([(0, 8, 2)], (True,), True),
        ([(6, 8, 2)], (False,), False),
        ([(0, 8, 2), (8, 8, 1)], (True, False), False),
        ([], (), False),
    ],
)
def test_bypass_requires_every_real_row_to_be_continuing(rows, full_rows, bypass):
    batch = _batch(rows)
    assert tuple(continuing_prefill_rows(batch, batch.num_reqs)) == full_rows
    assert (bool(full_rows) and all(full_rows)) is bypass


def test_phase_snapshot_is_not_recomputed_inside_the_proposal():
    batch = _batch([(0, 8, 2)])
    rows = tuple(continuing_prefill_rows(batch, batch.num_reqs))
    batch.num_computed_prefill_tokens_np[0] = 8
    assert rows == (True,)
    assert tuple(continuing_prefill_rows(batch, batch.num_reqs)) == (False,)


@pytest.fixture
def full_source():
    group = DraftWindowGroupRoute(0, ("draft.layer",), 4, DraftWindowProviderStatus.DIRECT_DENSE)
    counts = torch.tensor([[4, 3]], dtype=torch.int32)
    owner = SimpleNamespace(
        input_block_tables=[torch.arange(8, dtype=torch.int32).reshape(2, 4)],
        num_blocks=SimpleNamespace(cpu=counts, gpu=counts.clone()),
        block_sizes=[4],
        kernel_block_sizes=[4],
    )
    from vllm_ascend.worker.v2.spec_decode.draft_window_policy import resolve_draft_window_policy

    return DraftWindowGraphSource(resolve_draft_window_policy(requested_window_size=8, method="eagle3"), group, owner)


def test_canonical_cpu_counts_keep_mapping_order_padding_and_source(full_source):
    before = full_source.block_tables.num_blocks.cpu.clone()
    batch = _batch([(0, 8, 2), (0, 8, 2)])
    batch.idx_mapping_np = np.array([1, 0])
    bank = _materialize(full_source, batch, [10, 14, 0], [4, 3])
    assert bank.cpu_window_exact.tolist() == [10, 14, 0]
    assert bank.window_block_table.tolist() == [[0, 1, 2, 0], [4, 5, 6, 7], [0, 0, 0, 0]]
    assert torch.equal(full_source.block_tables.num_blocks.cpu, before)


@pytest.mark.parametrize("mapping", [np.array([-1]), np.array([2])])
def test_canonical_count_gather_rejects_invalid_mapping(full_source, mapping):
    batch = _batch([(0, 8, 2)])
    batch.idx_mapping_np = mapping
    with pytest.raises(IndexError):
        _materialize(full_source, batch, [2], [4, 3])


def _materialize(full_source, batch, exact, counts, frozen=None):
    from dataclasses import replace
    from types import SimpleNamespace

    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

    from vllm_ascend.worker.v2.spec_decode.draft_window_graph import (
        DraftWindowGraphBindingProvider,
    )

    if len(exact) > full_source.table.shape[0]:
        table = torch.cat(
            (
                full_source.table,
                full_source.table.new_zeros(len(exact) - full_source.table.shape[0], full_source.table.shape[1]),
            )
        )
        full_source = replace(
            full_source,
            block_tables=SimpleNamespace(
                input_block_tables=[table], num_blocks=SimpleNamespace(cpu=torch.tensor([counts], dtype=torch.int32))
            ),
        )
    else:
        full_source.block_tables.num_blocks.cpu[0] = torch.tensor(counts, dtype=torch.int32)
    provider = DraftWindowGraphBindingProvider(source=full_source)
    descriptor = BatchExecutionDescriptor(CUDAGraphMode.FULL, 4 * len(exact), len(exact), uniform_token_count=4)
    bank = provider.register_capture_identity(descriptor=descriptor)
    provider.proposal = tuple(continuing_prefill_rows(batch, batch.num_reqs))
    return provider.prepare(
        bank,
        cpu_full_exact=torch.tensor(exact, dtype=torch.int32),
        idx_mapping_np=batch.idx_mapping_np,
        frozen_row_plans=frozen,
        step=0,
    )


@pytest.mark.parametrize("count", [-1, 5])
def test_materialization_rejects_counts_outside_canonical_row(full_source, count):
    with pytest.raises(RuntimeError, match="initialized block count"):
        _materialize(full_source, _batch([(0, 8, 2)]), [2], [count])


@pytest.mark.parametrize("step,end", [(0, 14), (1, 13), (2, 14)])
def test_actual_bank_keeps_eagle_anchor_and_full_prefill(full_source, step, end):
    from vllm_ascend.worker.v2.spec_decode.draft_window_plan import compute_draft_window_plan

    frozen = tuple(
        compute_draft_window_plan(full_exact_end=value, future_offset=3, window_size=8, block_span_tokens=4)
        for value in (14, 2)
    )
    before = full_source.table.clone()
    bank = _materialize(full_source, _batch([(8, 8, 1), (0, 8, 2)]), [end, 2 + step, 0], [4, 3, 0], frozen)
    assert frozen[0].window_start == 8
    assert bank.cpu_window_exact.tolist() == [end - 8, 2 + step, 0]
    assert bank.window_block_table.tolist() == [[2, 3, 0, 0], [4, 5, 6, 0], [0, 0, 0, 0]]
    assert torch.equal(full_source.table, before)
