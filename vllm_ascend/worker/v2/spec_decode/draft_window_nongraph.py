# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Request phases and source-owned projections used by the Graph read bank."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from vllm_ascend.worker.v2.spec_decode.draft_window_runtime import continuing_prefill_rows


@dataclass(frozen=True, slots=True)
class DraftWindowBatchPhasePlan:
    """Only continuing-prefill rows need baseline Full reads."""

    full_read_rows: tuple[bool, ...]

    def __post_init__(self) -> None:
        if any(type(full_read) is not bool for full_read in self.full_read_rows):
            raise TypeError("Full-read row markers must be Python bools")

    @property
    def baseline_only_rows(self) -> tuple[int, ...]:
        return tuple(row for row, full_read in enumerate(self.full_read_rows) if full_read)

    @property
    def bypass_entire_batch(self) -> bool:
        return bool(self.full_read_rows) and all(self.full_read_rows)


def classify_draft_window_batch_phase(input_batch: Any) -> DraftWindowBatchPhasePlan:
    """Classify every active row from pinned ``InputBatch`` source fields."""
    num_reqs = int(input_batch.num_reqs)
    # The common eager/Graph helper checks each real prompt-progress counter,
    # including speculative placeholders, once before deriving these rows.
    continuing = continuing_prefill_rows(input_batch, num_reqs)
    computed = np.asarray(input_batch.num_computed_prefill_tokens_np)[:num_reqs]
    prefill_len = np.asarray(input_batch.prefill_len_np)[:num_reqs]
    declared = np.asarray(input_batch.is_prefilling_np)
    if declared.ndim != 1 or declared.shape[0] < num_reqs:
        raise ValueError("is_prefilling does not cover active request rows")
    actual_prefill = computed < prefill_len
    if not np.array_equal(actual_prefill, declared[:num_reqs].astype(np.bool_, copy=False)):
        raise RuntimeError("InputBatch is_prefilling rows disagree with computed/prefill lengths")
    if bool(getattr(input_batch, "has_prefill", False)) != bool(np.any(actual_prefill)):
        raise RuntimeError("InputBatch has_prefill disagrees with row phase state")
    return DraftWindowBatchPhasePlan(tuple(bool(value) for value in continuing))
