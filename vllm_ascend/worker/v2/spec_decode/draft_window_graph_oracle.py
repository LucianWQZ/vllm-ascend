# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Independent host oracle for the experimental Graph characterization.

This module deliberately does not import or call the production Window
planner/provider.  The G3 campaign uses it as an independent integer oracle
for the exact source columns and block ids that should reach the device
operator boundary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DraftWindowGraphOracleResult:
    full_horizon_end: int
    full_exact_end: int
    full_upper_end: int
    window_start: int
    visible_length: int
    window_upper: int
    source_start_column: int
    first_token_offset: int
    required_columns: int
    safe_extent_tokens: int
    expected_columns: tuple[int, ...]
    expected_block_ids: tuple[int, ...]
    query_width: int
    valid_block_count: int


def audit_characterization_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Audit one machine artifact without importing the runtime implementation."""
    if snapshot.get("schema") != "mrv2-draft-window-graph-characterization-v2":
        raise ValueError("unsupported Graph characterization artifact schema")
    events = snapshot.get("events")
    operands = snapshot.get("operand_events")
    g4_records = snapshot.get("g4_records")
    lifecycle = snapshot.get("lifecycle_events")
    if not isinstance(events, list):
        raise TypeError("Graph characterization events must be a list")
    if not isinstance(operands, list):
        raise TypeError("Graph characterization operand events must be a list")
    if not isinstance(g4_records, list):
        raise TypeError("Graph characterization G4 records must be a list")
    if not isinstance(lifecycle, list):
        raise TypeError("Graph characterization artifact sections must be lists")

    stages = {event.get("stage") for event in events if isinstance(event, Mapping)}
    operand_stages = {event.get("stage") for event in operands if isinstance(event, Mapping)}
    lifecycle_stages = {event.get("stage") for event in lifecycle if isinstance(event, Mapping)}
    required_source_stages = {
        "capture",
        "materialized",
        "bound",
        "live",
        "replay",
        "consumer",
        "update",
    }
    g1_host_ready = {
        "capture_operator",
        "update_operator",
    }.issubset(operand_stages)
    g4_ready = bool(g4_records) and required_source_stages.issubset(stages)
    g5_event_ready = {
        "replay_submitted",
        "update_event_recorded",
        "proposal_retired",
    }.issubset(lifecycle_stages)

    bound = [event for event in events if isinstance(event, Mapping) and event.get("stage") == "bound"]
    g2_ready = False
    if len(bound) >= 3:
        first, middle, last = bound[-3:]
        g2_ready = (
            first.get("backing_table_address")
            == middle.get("backing_table_address")
            == last.get("backing_table_address")
            and first.get("table_values") == last.get("table_values")
            and first.get("table_values") != middle.get("table_values")
            and first.get("generation_id") != last.get("generation_id")
        )
    return {
        "G1_HOST_OPERATOR_BOUNDARY": "PASS" if g1_host_ready else "OPEN",
        "G1_ACTUAL_ACL_TASK_INTERNAL": "DEVICE_TRACE_REQUIRED",
        "G2_HOST_A_B_A": "PASS" if g2_ready else "OPEN",
        "G3_OPERATOR_BOUNDARY": "PASS" if g1_host_ready else "OPEN",
        "G3_PHYSICAL_MEMORY_TRANSACTION": "DEVICE_TOOL_REQUIRED",
        "G4_WRITE_TARGET_ORACLE": "PASS" if g4_ready else "OPEN",
        "G5_EVENT_INSTRUMENTATION": "PASS" if g5_event_ready else "OPEN",
        "G5_UPDATE_EFFECTIVE_GENERATION": "NPU_TASK_TRACE_REQUIRED",
        "ready_for_device_campaign": bool(g1_host_ready and g4_ready and g5_event_ready),
    }


def compute_graph_characterization_oracle(
    *,
    full_table: Sequence[int],
    full_horizon_end: int,
    full_exact_end: int,
    full_upper_end: int,
    window_size: int,
    legacy_block_size: int,
    block_span_tokens: int,
    query_width: int,
    valid_block_count: int,
) -> DraftWindowGraphOracleResult:
    """Compute ``S/E/V`` and the direct-prefix source projection independently.

    ``full_horizon_end`` is ``H`` and may include a method-specific future
    horizon.  ``full_exact_end`` is the current exact ``E``.  The upper carrier
    is projected from ``full_upper_end`` and clipped to the source span that is
    actually materialized.  Any unsupported offset or source shortage is an
    explicit error instead of a guessed result.
    """

    scalar_values = {
        "full horizon end": full_horizon_end,
        "full exact end": full_exact_end,
        "full upper end": full_upper_end,
        "window size": window_size,
        "legacy block size": legacy_block_size,
        "block span": block_span_tokens,
        "query width": query_width,
        "valid block count": valid_block_count,
    }
    for name, value in scalar_values.items():
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
    if full_horizon_end < 0 or full_exact_end < 0 or full_upper_end < 0:
        raise ValueError("Graph oracle length inputs must be non-negative")
    if window_size <= 0 or legacy_block_size <= 0 or block_span_tokens <= 0:
        raise ValueError("Graph oracle geometry must be positive")
    if query_width <= 0:
        raise ValueError("Graph oracle query width must be positive")
    if valid_block_count < 0 or valid_block_count > len(full_table):
        raise ValueError("Graph oracle valid block count exceeds the full table")
    if not (full_exact_end <= full_horizon_end):
        raise ValueError("Graph oracle exact end exceeds the frozen horizon")
    if full_upper_end < full_exact_end:
        raise ValueError("Graph oracle upper end is below the exact end")

    raw_start = max(full_horizon_end - window_size, 0)
    window_start = (raw_start // legacy_block_size) * legacy_block_size
    if window_start > full_exact_end:
        raise ValueError("Graph oracle frozen anchor exceeds the current exact end")
    visible_length = full_exact_end - window_start
    source_start_column, first_token_offset = divmod(window_start, block_span_tokens)
    if first_token_offset:
        raise NotImplementedError("Graph oracle base route requires a direct-prefix zero offset")
    required_columns = 0
    if visible_length:
        required_columns = (visible_length + block_span_tokens - 1) // block_span_tokens
    source_end_column = source_start_column + required_columns
    if source_end_column > valid_block_count:
        raise RuntimeError(
            "Graph oracle source shortage: "
            f"start={source_start_column} required={required_columns} "
            f"end={source_end_column} valid={valid_block_count}"
        )
    safe_extent_tokens = required_columns * block_span_tokens
    window_upper = min(
        max(full_upper_end - window_start, visible_length),
        safe_extent_tokens,
    )
    expected_columns = tuple(range(source_start_column, source_end_column))
    expected_block_ids = tuple(int(full_table[column]) for column in expected_columns)
    return DraftWindowGraphOracleResult(
        full_horizon_end=full_horizon_end,
        full_exact_end=full_exact_end,
        full_upper_end=full_upper_end,
        window_start=window_start,
        visible_length=visible_length,
        window_upper=window_upper,
        source_start_column=source_start_column,
        first_token_offset=first_token_offset,
        required_columns=required_columns,
        safe_extent_tokens=safe_extent_tokens,
        expected_columns=expected_columns,
        expected_block_ids=expected_block_ids,
        query_width=query_width,
        valid_block_count=valid_block_count,
    )
