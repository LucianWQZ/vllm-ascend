# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Request phases and source-owned projections used by the Graph read bank."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from vllm_ascend.worker.v2.spec_decode.draft_window import (
    DraftWindowReadDescriptor,
    DraftWindowSourceCapability,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_plan import DraftWindowPlan, compute_draft_window_plan
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


@dataclass(frozen=True, slots=True)
class DraftWindowPhaseAwareMaterializationPlan:
    read_descriptors: tuple[DraftWindowReadDescriptor, ...]
    cpu_window_exact: torch.Tensor
    cpu_window_upper: torch.Tensor
    cpu_valid_num_blocks: torch.Tensor


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


def build_phase_aware_materialization_plan(
    *,
    source: DraftWindowSourceCapability,
    phase_plan: DraftWindowBatchPhasePlan,
    cpu_full_exact: torch.Tensor,
    cpu_apply_time_full_upper: torch.Tensor,
    cpu_valid_num_blocks: torch.Tensor,
    actual_request_count: int,
    padded_request_count: int,
    frozen_row_plans: tuple[DraftWindowPlan, ...] | None = None,
) -> DraftWindowPhaseAwareMaterializationPlan:
    if len(phase_plan.full_read_rows) != actual_request_count:
        raise ValueError("row phase plan does not match active request count")
    if frozen_row_plans is not None and len(frozen_row_plans) != actual_request_count:
        raise ValueError("frozen window plans must match active request count")
    for name, tensor in (
        ("full exact", cpu_full_exact),
        ("apply-time full upper", cpu_apply_time_full_upper),
        ("valid block counts", cpu_valid_num_blocks),
    ):
        if tensor.device.type != "cpu" or tensor.dtype != torch.int32:
            raise TypeError(f"CPU {name} must be CPU int32")
        if tensor.ndim != 1 or tensor.shape[0] != padded_request_count:
            raise ValueError(f"CPU {name} must match padded request count")

    exact = torch.zeros(padded_request_count, dtype=torch.int32)
    upper = torch.zeros(padded_request_count, dtype=torch.int32)
    descriptors: list[DraftWindowReadDescriptor] = []
    span = source.group.table_column_span
    source_width = source.full_source.input_block_table_row_width(source.group.gid)
    for row, full_read in enumerate(phase_plan.full_read_rows):
        full_exact = int(cpu_full_exact[row])
        full_upper = int(cpu_apply_time_full_upper[row])
        valid = int(cpu_valid_num_blocks[row])
        if full_exact < 0 or full_upper < full_exact:
            raise ValueError("full exact/upper lengths are inconsistent")
        if valid < 0 or valid > source_width:
            raise RuntimeError("resident valid-block count is outside the canonical source row")
        if full_read:
            safe_extent = valid * span
            if safe_extent < full_exact:
                raise RuntimeError("continuing-prefill full read exceeds its resident source")
            descriptor = DraftWindowReadDescriptor(
                global_group_id=source.group.gid,
                source_start_column=0,
                first_token_offset=0,
                required_columns=valid,
                safe_extent_tokens=safe_extent,
                valid_num_blocks=valid,
                source_row_width=source_width,
            )
            visible = full_exact
            visible_upper = min(max(full_upper, visible), safe_extent)
        else:
            if frozen_row_plans is None:
                scalar = compute_draft_window_plan(
                    full_exact_end=full_exact,
                    window_size=source.window_size,
                    block_span_tokens=source.group.table_column_span,
                )
                descriptor = source.describe(scalar, valid_num_blocks=valid)
            else:
                scalar = frozen_row_plans[row]
                descriptor = source.describe_frozen_step(scalar, full_exact_end=full_exact, valid_num_blocks=valid)
            visible = full_exact - scalar.window_start
            visible_upper = min(
                max(full_upper - scalar.window_start, visible),
                descriptor.safe_extent_tokens,
            )
        if visible_upper < visible:
            raise RuntimeError("phase-aware upper length is shorter than exact")
        descriptors.append(descriptor)
        exact[row] = visible
        upper[row] = visible_upper
    return DraftWindowPhaseAwareMaterializationPlan(
        read_descriptors=tuple(descriptors),
        cpu_window_exact=exact,
        cpu_window_upper=upper,
        cpu_valid_num_blocks=cpu_valid_num_blocks,
    )
