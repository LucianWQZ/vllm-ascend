# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Request phases and source-owned projections used by the Graph read bank."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch

from vllm_ascend.worker.v2.spec_decode.draft_window import (
    DraftWindowReadDescriptor,
    FullDraftKVSource,
    SingleGroupDirectPrefixProvider,
    gather_valid_num_blocks_cpu,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_plan import compute_draft_window_plan
from vllm_ascend.worker.v2.spec_decode.draft_window_runtime import continuing_prefill_rows


class DraftWindowRowPhase(Enum):
    """Phase of one ordered request row before the current proposal."""

    PURE_DECODE = "pure_decode"
    PREFILL_CONTINUING = "prefill_continuing"
    PREFILL_TERMINAL = "prefill_terminal"


class DraftWindowBatchPhase(Enum):
    PURE_DECODE = "pure_decode"
    PURE_PREFILL_CONTINUING = "pure_prefill_continuing"
    PURE_PREFILL_TERMINAL = "pure_prefill_terminal"
    MIXED = "mixed"


@dataclass(frozen=True, slots=True)
class DraftWindowBatchPhasePlan:
    row_phases: tuple[DraftWindowRowPhase, ...]
    batch_phase: DraftWindowBatchPhase
    ordered_request_fingerprint: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.row_phases) != len(self.ordered_request_fingerprint):
            raise ValueError("row phases must match the ordered request fingerprint")
        if any(not request_id for request_id in self.ordered_request_fingerprint):
            raise ValueError("phase-plan request IDs must be non-empty")

    @property
    def row_phase_fingerprint(self) -> tuple[str, ...]:
        return tuple(phase.value for phase in self.row_phases)

    @property
    def baseline_only_rows(self) -> tuple[int, ...]:
        return tuple(
            row for row, phase in enumerate(self.row_phases) if phase is DraftWindowRowPhase.PREFILL_CONTINUING
        )

    @property
    def window_rows(self) -> tuple[int, ...]:
        return tuple(
            row for row, phase in enumerate(self.row_phases) if phase is not DraftWindowRowPhase.PREFILL_CONTINUING
        )

    @property
    def bypass_entire_batch(self) -> bool:
        return bool(self.row_phases) and not self.window_rows


@dataclass(frozen=True, slots=True)
class DraftWindowPhaseAwareMaterializationPlan:
    read_descriptors: tuple[DraftWindowReadDescriptor, ...]
    cpu_window_exact: torch.Tensor
    cpu_window_upper: torch.Tensor
    cpu_valid_num_blocks: torch.Tensor
    row_phase_fingerprint: tuple[str, ...]


@dataclass(frozen=True, slots=True, eq=False)
class DraftWindowResidentSetSnapshot:
    """Canonical prefix/residency evidence in current batch-row order."""

    global_group_id: int
    ordered_request_fingerprint: tuple[str, ...]
    mapping_fingerprint: tuple[int, ...]
    source_identity: Any
    valid_num_blocks: torch.Tensor


def classify_draft_window_batch_phase(input_batch: Any) -> DraftWindowBatchPhasePlan:
    """Classify every active row from pinned ``InputBatch`` source fields."""
    num_reqs = int(input_batch.num_reqs)
    request_ids = tuple(input_batch.req_ids[:num_reqs])
    if len(request_ids) != num_reqs or len(set(request_ids)) != num_reqs:
        raise ValueError("phase classification requires unique ordered request IDs")
    arrays = {
        "computed": np.asarray(input_batch.num_computed_prefill_tokens_np),
        "scheduled": np.asarray(input_batch.num_scheduled_tokens),
        "prefill length": np.asarray(input_batch.prefill_len_np),
        "is_prefilling": np.asarray(input_batch.is_prefilling_np),
    }
    for name, array in arrays.items():
        if array.ndim != 1 or array.shape[0] < num_reqs:
            raise ValueError(f"{name} does not cover active request rows")
    computed = arrays["computed"][:num_reqs]
    scheduled = arrays["scheduled"][:num_reqs]
    prefill_len = arrays["prefill length"][:num_reqs]
    declared_prefill = arrays["is_prefilling"][:num_reqs].astype(np.bool_, copy=False)
    if any(not np.issubdtype(array.dtype, np.integer) for array in (computed, scheduled, prefill_len)):
        raise TypeError("prefill phase counters must contain integers")
    if bool(np.any(computed < 0)) or bool(np.any(scheduled < 0)) or bool(np.any(prefill_len < 0)):
        raise ValueError("prefill phase counters must be nonnegative")
    actual_prefill = computed < prefill_len
    if not np.array_equal(actual_prefill, declared_prefill):
        raise RuntimeError("InputBatch is_prefilling rows disagree with computed/prefill lengths")
    if bool(getattr(input_batch, "has_prefill", False)) != bool(np.any(actual_prefill)):
        raise RuntimeError("InputBatch has_prefill disagrees with row phase state")
    continuing = continuing_prefill_rows(input_batch, num_reqs)

    phases: list[DraftWindowRowPhase] = []
    for row in range(num_reqs):
        if not bool(actual_prefill[row]):
            phases.append(DraftWindowRowPhase.PURE_DECODE)
        elif bool(continuing[row]):
            phases.append(DraftWindowRowPhase.PREFILL_CONTINUING)
        else:
            phases.append(DraftWindowRowPhase.PREFILL_TERMINAL)
    unique = set(phases)
    if not phases or unique == {DraftWindowRowPhase.PURE_DECODE}:
        batch_phase = DraftWindowBatchPhase.PURE_DECODE
    elif unique == {DraftWindowRowPhase.PREFILL_CONTINUING}:
        batch_phase = DraftWindowBatchPhase.PURE_PREFILL_CONTINUING
    elif unique == {DraftWindowRowPhase.PREFILL_TERMINAL}:
        batch_phase = DraftWindowBatchPhase.PURE_PREFILL_TERMINAL
    else:
        batch_phase = DraftWindowBatchPhase.MIXED
    return DraftWindowBatchPhasePlan(
        row_phases=tuple(phases),
        batch_phase=batch_phase,
        ordered_request_fingerprint=request_ids,
    )


def build_prefix_resident_snapshot(
    *,
    full_source: FullDraftKVSource,
    global_group_id: int,
    idx_mapping_np: np.ndarray,
    ordered_request_ids: Sequence[str],
    num_reqs: int,
    num_reqs_padded: int,
) -> DraftWindowResidentSetSnapshot:
    request_ids = tuple(ordered_request_ids)
    if len(request_ids) != num_reqs:
        raise ValueError("resident snapshot request IDs must match active rows")
    mapping = np.asarray(idx_mapping_np)
    if mapping.ndim != 1 or mapping.shape[0] < num_reqs:
        raise ValueError("resident snapshot mapping does not cover active rows")
    counts = gather_valid_num_blocks_cpu(
        full_source,
        global_group_id=global_group_id,
        idx_mapping_np=mapping,
        num_reqs=num_reqs,
        num_reqs_padded=num_reqs_padded,
    )
    return DraftWindowResidentSetSnapshot(
        global_group_id=global_group_id,
        ordered_request_fingerprint=request_ids,
        mapping_fingerprint=tuple(int(value) for value in mapping[:num_reqs]),
        source_identity=full_source.source_identity,
        valid_num_blocks=counts,
    )


def build_phase_aware_materialization_plan(
    *,
    provider: SingleGroupDirectPrefixProvider,
    phase_plan: DraftWindowBatchPhasePlan,
    cpu_full_exact: torch.Tensor,
    cpu_apply_time_full_upper: torch.Tensor,
    cpu_valid_num_blocks: torch.Tensor,
    actual_request_count: int,
    padded_request_count: int,
) -> DraftWindowPhaseAwareMaterializationPlan:
    if len(phase_plan.row_phases) != actual_request_count:
        raise ValueError("row phase plan does not match active request count")
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
    span = provider.block_span_tokens
    source_width = provider.full_source.input_block_table_row_width(provider.global_group_id)
    for row, phase in enumerate(phase_plan.row_phases):
        full_exact = int(cpu_full_exact[row])
        full_upper = int(cpu_apply_time_full_upper[row])
        valid = int(cpu_valid_num_blocks[row])
        if full_exact < 0 or full_upper < full_exact:
            raise ValueError("full exact/upper lengths are inconsistent")
        if valid < 0 or valid > source_width:
            raise RuntimeError("resident valid-block count is outside the canonical source row")
        if phase is DraftWindowRowPhase.PREFILL_CONTINUING:
            safe_extent = valid * span
            if safe_extent < full_exact:
                raise RuntimeError("continuing-prefill full read exceeds its resident source")
            descriptor = DraftWindowReadDescriptor(
                global_group_id=provider.global_group_id,
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
            scalar = compute_draft_window_plan(
                full_exact_end=full_exact,
                future_offset=0,
                window_size=provider.inventory.window_size,
                block_span_tokens=provider.legacy_block_size,
            )
            descriptor = provider.describe(scalar, valid_num_blocks=valid)
            visible = scalar.window_visible_length
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
        row_phase_fingerprint=phase_plan.row_phase_fingerprint,
    )
