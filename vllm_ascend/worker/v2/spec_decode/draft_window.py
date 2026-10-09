# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Policy and scalar planning contracts for MRV2 draft windows.

This module deliberately has no tensor, backend, or device dependencies.  P1
only decides whether a configured window is eligible and describes the scalar
read interval that later phases may materialize.
"""

from dataclasses import dataclass
from enum import Enum


class DraftWindowPolicyState(Enum):
    """Result of resolving a configured MRV2 draft-window request."""

    CONFIG_OFF = "config_off"
    FORCED_OFF = "forced_off"
    ELIGIBLE_ON = "eligible_on"


@dataclass(frozen=True, slots=True)
class DraftWindowPolicyDecision:
    """Immutable result of method/model policy resolution."""

    state: DraftWindowPolicyState
    requested_window_size: int | None
    effective_window_size: int | None
    normalized_method: str
    target_model_type: str
    reason: str | None


@dataclass(frozen=True, slots=True)
class DraftWindowScalarPlan:
    """Scalar view of one draft window; no storage is owned by this value."""

    full_exact_end: int
    full_horizon_end: int
    window_start: int
    window_visible_length: int
    window_capacity_tokens: int


@dataclass(frozen=True, slots=True)
class DirectPrefixProjection:
    """Projection of an absolute token interval onto a direct-paged row."""

    source_start_column: int
    first_token_offset: int
    required_columns: int
    safe_extent_tokens: int

    @property
    def source_end_column(self) -> int:
        """Exclusive source column after the projected prefix."""
        return self.source_start_column + self.required_columns


def _normalize_method(method: str) -> str:
    return method.strip().lower()


def _normalize_model_type(model_type: str) -> str:
    return model_type.strip().lower().replace("-", "_")


def _is_deepseek_v4_target(normalized_model_type: str) -> bool:
    """Match the explicit DeepSeek-V4 model-type family used by config."""
    return normalized_model_type.startswith("deepseek_v4")


def resolve_draft_window_policy(
    *,
    requested_window_size: int | None,
    method: str,
    target_model_type: str,
) -> DraftWindowPolicyDecision:
    """Resolve method/model policy before any numeric or runtime work.

    ``None`` is the only OFF value.  MTP and DeepSeek-V4 co-trained DSpark
    requests are forced OFF even when their configured number is otherwise
    invalid; this ordering is part of the feature contract.
    """
    normalized_method = _normalize_method(method)
    normalized_model_type = _normalize_model_type(target_model_type)

    if requested_window_size is None:
        return DraftWindowPolicyDecision(
            state=DraftWindowPolicyState.CONFIG_OFF,
            requested_window_size=None,
            effective_window_size=None,
            normalized_method=normalized_method,
            target_model_type=normalized_model_type,
            reason=None,
        )

    if normalized_method == "mtp":
        return DraftWindowPolicyDecision(
            state=DraftWindowPolicyState.FORCED_OFF,
            requested_window_size=requested_window_size,
            effective_window_size=None,
            normalized_method=normalized_method,
            target_model_type=normalized_model_type,
            reason="mtp",
        )

    if normalized_method == "dspark" and _is_deepseek_v4_target(normalized_model_type):
        return DraftWindowPolicyDecision(
            state=DraftWindowPolicyState.FORCED_OFF,
            requested_window_size=requested_window_size,
            effective_window_size=None,
            normalized_method=normalized_method,
            target_model_type=normalized_model_type,
            reason="deepseek_v4_cotrained_dspark",
        )

    if normalized_method == "eagle":
        raise NotImplementedError("draft_window_size is not supported for plain Eagle in MRV2")

    if normalized_method not in {"eagle3", "dflash", "dspark"}:
        raise NotImplementedError(
            f"draft_window_size is not supported for MRV2 speculative method {normalized_method!r}"
        )

    return DraftWindowPolicyDecision(
        state=DraftWindowPolicyState.ELIGIBLE_ON,
        requested_window_size=requested_window_size,
        effective_window_size=requested_window_size,
        normalized_method=normalized_method,
        target_model_type=normalized_model_type,
        reason=None,
    )


def resolve_static_future_offset(
    *,
    normalized_method: str,
    configured_k: int,
    uses_dynamic_k: bool,
) -> int:
    """Return the Frozen method horizon offset, rejecting dynamic K."""
    if uses_dynamic_k:
        raise NotImplementedError("draft_window_size does not support dynamic speculative decoding")
    if normalized_method == "eagle3":
        return configured_k
    if normalized_method in {"dflash", "dspark"}:
        return 0
    raise NotImplementedError(
        f"no static draft-window future offset is defined for speculative method {normalized_method!r}"
    )


def validate_eligible_window_numbers(*, window_size: int, future_offset: int) -> None:
    """Apply the interim fail-closed numeric gate for eligible routes."""
    if window_size <= 0:
        raise ValueError("draft_window_size must be greater than 0 for an eligible MRV2 route")
    if future_offset < 0:
        raise ValueError("static future offset must be greater than or equal to 0")
    if window_size < future_offset:
        raise ValueError("draft_window_size must be at least the static future offset")


def _validate_scalar_inputs(
    *,
    full_exact_end: int,
    future_offset: int,
    window_size: int,
    legacy_block_size: int,
) -> None:
    if full_exact_end < 0:
        raise ValueError("full exact end must be greater than or equal to 0")
    if future_offset < 0:
        raise ValueError("future offset must be greater than or equal to 0")
    if window_size <= 0:
        raise ValueError("window size must be greater than 0")
    if legacy_block_size <= 0:
        raise ValueError("legacy block size must be greater than 0")


def compute_legacy_anchor(
    *,
    full_context_length: int,
    future_offset: int,
    window_size: int,
    legacy_block_size: int,
) -> int:
    """Compute ``B * floor(max(L + K - W, 0) / B)`` exactly."""
    _validate_scalar_inputs(
        full_exact_end=full_context_length,
        future_offset=future_offset,
        window_size=window_size,
        legacy_block_size=legacy_block_size,
    )
    full_horizon_end = full_context_length + future_offset
    raw_start = max(full_horizon_end - window_size, 0)
    return (raw_start // legacy_block_size) * legacy_block_size


def compute_visible_length(*, full_exact_end: int, window_start: int) -> int:
    """Return the exact visible length and reject a negative interval."""
    if full_exact_end < 0:
        raise ValueError("full exact end must be greater than or equal to 0")
    if window_start < 0:
        raise ValueError("window start must be greater than or equal to 0")
    if window_start > full_exact_end:
        raise ValueError(f"window start {window_start} exceeds full exact end {full_exact_end}")
    return full_exact_end - window_start


def compute_window_capacity_tokens(*, window_size: int, legacy_block_size: int) -> int:
    """Return the Frozen scalar capacity upper bound ``W + B - 1``."""
    if window_size <= 0:
        raise ValueError("window size must be greater than 0")
    if legacy_block_size <= 0:
        raise ValueError("legacy block size must be greater than 0")
    return window_size + legacy_block_size - 1


def compute_draft_window_scalar_plan(
    *,
    full_exact_end: int,
    future_offset: int,
    window_size: int,
    legacy_block_size: int,
) -> DraftWindowScalarPlan:
    """Build an immutable scalar plan from exact, not upper-bound, length."""
    window_start = compute_legacy_anchor(
        full_context_length=full_exact_end,
        future_offset=future_offset,
        window_size=window_size,
        legacy_block_size=legacy_block_size,
    )
    return DraftWindowScalarPlan(
        full_exact_end=full_exact_end,
        full_horizon_end=full_exact_end + future_offset,
        window_start=window_start,
        window_visible_length=compute_visible_length(
            full_exact_end=full_exact_end,
            window_start=window_start,
        ),
        window_capacity_tokens=compute_window_capacity_tokens(
            window_size=window_size,
            legacy_block_size=legacy_block_size,
        ),
    )


def project_direct_prefix(
    *,
    window_start: int,
    window_visible_length: int,
    block_span_tokens: int,
) -> DirectPrefixProjection:
    """Project ``[S, S + V)`` without changing the absolute anchor."""
    if window_start < 0:
        raise ValueError("window start must be greater than or equal to 0")
    if window_visible_length < 0:
        raise ValueError("window visible length must be greater than or equal to 0")
    if block_span_tokens <= 0:
        raise ValueError("block span must be greater than 0")

    source_start_column, first_token_offset = divmod(window_start, block_span_tokens)
    required_columns = 0
    if window_visible_length:
        required_columns = (first_token_offset + window_visible_length + block_span_tokens - 1) // block_span_tokens
    safe_extent_tokens = 0
    if required_columns:
        safe_extent_tokens = required_columns * block_span_tokens - first_token_offset
    return DirectPrefixProjection(
        source_start_column=source_start_column,
        first_token_offset=first_token_offset,
        required_columns=required_columns,
        safe_extent_tokens=safe_extent_tokens,
    )


def compute_direct_prefix_capacity(
    *,
    window_size: int,
    legacy_block_size: int,
    block_span_tokens: int,
) -> int:
    """Return the zero-offset direct-prefix capacity in source columns."""
    if block_span_tokens <= 0:
        raise ValueError("block span must be greater than 0")
    capacity_tokens = compute_window_capacity_tokens(
        window_size=window_size,
        legacy_block_size=legacy_block_size,
    )
    return (capacity_tokens + block_span_tokens - 1) // block_span_tokens
