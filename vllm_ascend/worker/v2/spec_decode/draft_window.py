# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Policy, scalar planning, and read-only source contracts for draft windows.

P1 decides whether a configured window is eligible and describes its scalar
interval. P2 inventories the actual draft attention consumers and records how
that interval would project onto the canonical full block tables. It still
does not allocate a window table or connect one to attention.
"""

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    CircularBufferSpec,
    FullAttentionSpec,
    MLAAttentionSpec,
    RSWASpec,
    SlidingWindowSpec,
)
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.ascend_config import validate_additional_config_bool
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionMetadataBuilder,
)
from vllm_ascend.attention.dsa_v1 import AscendDSABackend
from vllm_ascend.attention.indexer import AscendSFAIndexerBackend
from vllm_ascend.attention.mla_v1 import AscendMLABackend
from vllm_ascend.attention.sfa_v1 import AscendSFABackend


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


class DraftWindowLayoutCategory(Enum):
    """P2 classification of the actual draft attention read layout."""

    DENSE_GQA_DIRECT_PREFIX = "dense_gqa_direct_prefix"
    MLA = "mla"
    DSA = "dsa"
    SFA = "sfa"
    NATIVE_OR_RECYCLING = "native_or_recycling"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class DraftWindowRuntimeContext:
    """Runtime modes that must be proven before a P2 candidate is usable."""

    graph_mode: str
    async_scheduling: bool
    tensor_parallel_size: int
    data_parallel_size: int
    pipeline_parallel_size: int
    prefill_context_parallel_size: int
    decode_context_parallel_size: int
    sequence_parallel: bool
    dsa_context_parallel: bool
    flashcomm: bool


@dataclass(frozen=True, slots=True)
class DraftWindowGroupInventory:
    """One executable ``AttentionGroup`` with its global cache-group id."""

    global_group_id: int
    local_consumer_index: int
    layer_names: tuple[str, ...]
    backend_class: type[Any]
    builder_class: type[Any]
    logical_block_size: int
    kernel_block_size: int
    block_span_tokens: int | None
    causal: bool
    layout_category: DraftWindowLayoutCategory
    capability_failures: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DraftWindowInventory:
    """Immutable inventory and fail-closed result for one eligible route."""

    method: str
    target_model_type: str
    runtime_route: str
    window_size: int
    groups: tuple[DraftWindowGroupInventory, ...]
    empty_global_group_ids: tuple[int, ...]
    capability_failures: tuple[str, ...]

    @property
    def consumer_global_group_ids(self) -> tuple[int, ...]:
        """Unique global gids in first-consumer order."""
        return tuple(dict.fromkeys(group.global_group_id for group in self.groups))

    def require_single_group_direct_prefix(self) -> DraftWindowGroupInventory:
        """Return the sole candidate or raise one exhaustive diagnostic."""
        failures = list(self.capability_failures)
        for group in self.groups:
            failures.extend(
                "gid={gid} backend={backend} builder={builder} layout={layout} reason={reason}".format(
                    gid=group.global_group_id,
                    backend=group.backend_class.__name__,
                    builder=group.builder_class.__name__,
                    layout=group.layout_category.value,
                    reason=reason,
                )
                for reason in group.capability_failures
            )
        if len(self.groups) != 1 and not failures:
            failures.append(f"expected one executable consumer, found {len(self.groups)}")
        if failures:
            details = "; ".join(failures)
            raise NotImplementedError(
                "MRV2 draft-window capability rejected: "
                f"method={self.method} model={self.target_model_type} "
                f"route={self.runtime_route}; {details}"
            )
        group = self.groups[0]
        if (
            group.layout_category is not DraftWindowLayoutCategory.DENSE_GQA_DIRECT_PREFIX
            or group.block_span_tokens is None
        ):
            raise RuntimeError("a failure-free inventory must be direct-prefix")
        return group


def _group_diagnostic_context(
    inventory: DraftWindowInventory,
    group: DraftWindowGroupInventory,
) -> str:
    """Return the common identity required on every P2 source failure."""
    backend_name = getattr(group.backend_class, "__name__", repr(group.backend_class))
    return (
        f"method={inventory.method} model={inventory.target_model_type} "
        f"route={inventory.runtime_route} gid={group.global_group_id} "
        f"backend={backend_name} layout={group.layout_category.value}"
    )


@dataclass(frozen=True, slots=True)
class DraftWindowSourceIdentity:
    """Stable P2 owner identity; batch freshness is intentionally not encoded.

    ``idx_mapping_ptr`` identifies one tensor allocation only. An in-place
    rewrite keeps the same pointer, so P3 must add a caller-owned monotonic
    batch/generation identity before a bound source can outlive one batch.
    """

    block_tables_owner_id: int
    input_block_table_ptrs: tuple[int, ...]
    persistent_num_blocks_ptr: int
    idx_mapping_ptr: int | None


@dataclass(frozen=True, slots=True)
class FullDraftKVSource:
    """Borrowed, read-only references to canonical full BlockTables state.

    Tensor references stay private because PyTorch has no immutable tensor
    view. Public methods expose identity and geometry only; the sole P2 tensor
    operation, :func:`gather_valid_num_blocks`, reads these references and
    writes only caller-owned output.
    """

    logical_block_sizes: tuple[int, ...]
    kernel_block_sizes: tuple[int, ...]
    source_identity: DraftWindowSourceIdentity
    _block_tables_owner: Any = field(repr=False, compare=False)
    _input_block_tables: tuple[torch.Tensor, ...] = field(
        repr=False,
        compare=False,
    )
    _persistent_num_blocks: torch.Tensor = field(repr=False, compare=False)
    _idx_mapping: torch.Tensor | None = field(repr=False, compare=False)

    @property
    def num_groups(self) -> int:
        return len(self._input_block_tables)

    def input_block_table_row_width(self, global_group_id: int) -> int:
        self._validate_global_group_id(global_group_id)
        return self._input_block_tables[global_group_id].shape[1]

    def references_input_block_table(
        self,
        global_group_id: int,
        tensor: torch.Tensor,
    ) -> bool:
        self._validate_global_group_id(global_group_id)
        return self._input_block_tables[global_group_id] is tensor

    def references_persistent_num_blocks(self, tensor: torch.Tensor) -> bool:
        return self._persistent_num_blocks is tensor

    def references_same_idx_mapping_tensor(self, tensor: torch.Tensor) -> bool:
        """Return whether ``tensor`` is the same object, not whether it is fresh."""
        return self._idx_mapping is tensor

    def bind_batch(self, idx_mapping: torch.Tensor) -> "FullDraftKVSource":
        """Return a view bound to the exact mapping used for this batch.

        DFlash and DSpark consume ``input_batch.idx_mapping`` directly instead
        of populating ``DraftModelSpeculator.idx_mapping``. Binding is therefore
        explicit and per batch; no mapping is copied or guessed at set-attn
        time. The returned view is valid only for the caller-defined current
        batch. P2 cannot detect an in-place rewrite of the mapping tensor and
        callers must not cache this view across batch/proposal generations.
        """
        _validate_idx_mapping(idx_mapping)
        return FullDraftKVSource(
            logical_block_sizes=self.logical_block_sizes,
            kernel_block_sizes=self.kernel_block_sizes,
            source_identity=DraftWindowSourceIdentity(
                block_tables_owner_id=self.source_identity.block_tables_owner_id,
                input_block_table_ptrs=self.source_identity.input_block_table_ptrs,
                persistent_num_blocks_ptr=self.source_identity.persistent_num_blocks_ptr,
                idx_mapping_ptr=idx_mapping.data_ptr(),
            ),
            _block_tables_owner=self._block_tables_owner,
            _input_block_tables=self._input_block_tables,
            _persistent_num_blocks=self._persistent_num_blocks,
            _idx_mapping=idx_mapping,
        )

    def validate_owner_bindings(self) -> None:
        """Reject replacement of a canonical owner tensor after P2 binding."""
        current_tables = tuple(self._block_tables_owner.input_block_tables)
        if len(current_tables) != len(self._input_block_tables):
            raise RuntimeError("canonical input block table group count changed")
        for global_group_id, (current, bound) in enumerate(zip(current_tables, self._input_block_tables)):
            if current is not bound:
                raise RuntimeError(
                    f"canonical input block table {global_group_id} was replaced after FullSource binding"
                )
        if self._block_tables_owner.num_blocks.gpu is not self._persistent_num_blocks:
            raise RuntimeError("canonical persistent num_blocks.gpu was replaced after FullSource binding")

    def _validate_global_group_id(self, global_group_id: int) -> None:
        if not 0 <= global_group_id < self.num_groups:
            raise IndexError(f"global group id {global_group_id} is outside [0, {self.num_groups})")


@dataclass(frozen=True, slots=True)
class DraftWindowReadDescriptor:
    """Read-only description of a row slice in the canonical full source."""

    global_group_id: int
    source_start_column: int
    first_token_offset: int
    required_columns: int
    safe_extent_tokens: int
    valid_num_blocks: int
    source_row_width: int

    @property
    def source_end_column(self) -> int:
        return self.source_start_column + self.required_columns


@dataclass(frozen=True, slots=True)
class SingleGroupDirectPrefixProvider:
    """P2-only projection provider; it never materializes a window table."""

    inventory: DraftWindowInventory
    full_source: FullDraftKVSource
    legacy_block_size: int
    expected_future_offset: int
    _group: DraftWindowGroupInventory = field(init=False, repr=False)
    _capacity_columns: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        group = self.inventory.require_single_group_direct_prefix()
        if self.legacy_block_size <= 0:
            raise ValueError("legacy block size must be greater than 0")
        if not isinstance(self.expected_future_offset, int) or isinstance(
            self.expected_future_offset,
            bool,
        ):
            raise TypeError("expected future offset must be a Python int")
        if self.expected_future_offset < 0:
            raise ValueError("expected future offset must be greater than or equal to 0")
        method_future_offset = resolve_static_future_offset(
            normalized_method=self.inventory.method,
            configured_k=self.expected_future_offset,
            uses_dynamic_k=False,
        )
        if method_future_offset != self.expected_future_offset:
            raise ValueError(
                "configured future offset does not match method semantics: "
                f"method={self.inventory.method} "
                f"offset={self.expected_future_offset} "
                f"expected={method_future_offset}"
            )
        if group.global_group_id >= self.full_source.num_groups:
            raise RuntimeError(f"inventory gid {group.global_group_id} has no canonical full source")
        self.full_source.validate_owner_bindings()
        _validate_source_geometry(
            inventory=self.inventory,
            group=group,
            logical_block_sizes=self.full_source.logical_block_sizes,
            kernel_block_sizes=self.full_source.kernel_block_sizes,
            num_groups=self.full_source.num_groups,
        )
        span = group.block_span_tokens
        assert span is not None
        try:
            capacity_columns = compute_direct_prefix_capacity(
                window_size=self.inventory.window_size,
                legacy_block_size=self.legacy_block_size,
                block_span_tokens=span,
            )
        except ValueError as error:
            raise NotImplementedError(
                "MRV2 direct-prefix geometry is not proven: "
                f"method={self.inventory.method} "
                f"model={self.inventory.target_model_type} "
                f"gid={group.global_group_id} span={span} "
                f"legacy_block_size={self.legacy_block_size}"
            ) from error
        object.__setattr__(self, "_group", group)
        object.__setattr__(self, "_capacity_columns", capacity_columns)

    def describe(
        self,
        plan: DraftWindowScalarPlan,
        *,
        valid_num_blocks: int,
    ) -> DraftWindowReadDescriptor:
        """Validate one active row and return its canonical source interval."""
        self.full_source.validate_owner_bindings()
        if not isinstance(valid_num_blocks, int) or isinstance(
            valid_num_blocks,
            bool,
        ):
            raise TypeError("valid_num_blocks must be a Python int")
        if valid_num_blocks < 0:
            raise ValueError("valid_num_blocks must be greater than or equal to 0")
        if plan.full_horizon_end < plan.full_exact_end:
            raise ValueError("scalar plan horizon cannot precede its exact end")
        future_offset = plan.full_horizon_end - plan.full_exact_end
        if future_offset != self.expected_future_offset:
            raise ValueError(
                "scalar plan future offset does not match configured static "
                "future offset: "
                f"method={self.inventory.method} "
                f"offset={future_offset} expected={self.expected_future_offset}"
            )
        expected_start = compute_legacy_anchor(
            full_context_length=plan.full_exact_end,
            future_offset=future_offset,
            window_size=self.inventory.window_size,
            legacy_block_size=self.legacy_block_size,
        )
        if plan.window_start != expected_start:
            raise ValueError(
                f"scalar plan does not use the canonical P1 anchor: {plan.window_start} != {expected_start}"
            )
        expected_visible_length = compute_visible_length(
            full_exact_end=plan.full_exact_end,
            window_start=plan.window_start,
        )
        if plan.window_visible_length != expected_visible_length:
            raise ValueError(
                "scalar plan visible length is inconsistent with its exact end: "
                f"{plan.window_visible_length} != {expected_visible_length}"
            )
        expected_capacity = compute_window_capacity_tokens(
            window_size=self.inventory.window_size,
            legacy_block_size=self.legacy_block_size,
        )
        if plan.window_capacity_tokens != expected_capacity:
            raise ValueError(
                "scalar plan capacity does not match provider geometry: "
                f"{plan.window_capacity_tokens} != {expected_capacity}"
            )
        span = self._group.block_span_tokens
        assert span is not None
        projection = project_direct_prefix(
            window_start=plan.window_start,
            window_visible_length=plan.window_visible_length,
            block_span_tokens=span,
        )
        if projection.first_token_offset != 0:
            raise NotImplementedError(
                "MRV2 direct-prefix provider cannot consume a nonzero first-token "
                f"offset: gid={self._group.global_group_id} "
                f"start={plan.window_start} span={span} "
                f"offset={projection.first_token_offset}"
            )
        if projection.required_columns > self._capacity_columns:
            raise RuntimeError(
                "draft-window projection exceeds provider capacity: "
                f"required={projection.required_columns} "
                f"capacity={self._capacity_columns}"
            )
        source_row_width = self.full_source.input_block_table_row_width(self._group.global_group_id)
        if valid_num_blocks > source_row_width:
            raise RuntimeError(
                "canonical valid block count exceeds source row width: "
                f"gid={self._group.global_group_id} valid={valid_num_blocks} "
                f"row_width={source_row_width}"
            )
        if projection.source_end_column > valid_num_blocks:
            raise RuntimeError(
                "draft-window source shortage: "
                f"method={self.inventory.method} "
                f"model={self.inventory.target_model_type} "
                f"gid={self._group.global_group_id} "
                f"start={projection.source_start_column} "
                f"required={projection.required_columns} "
                f"end={projection.source_end_column} valid={valid_num_blocks}"
            )
        return DraftWindowReadDescriptor(
            global_group_id=self._group.global_group_id,
            source_start_column=projection.source_start_column,
            first_token_offset=projection.first_token_offset,
            required_columns=projection.required_columns,
            safe_extent_tokens=projection.safe_extent_tokens,
            valid_num_blocks=valid_num_blocks,
            source_row_width=source_row_width,
        )


@dataclass(frozen=True, slots=True)
class DraftWindowSourceCapability:
    """P2 state attached to an eligible speculator after ``set_attn``."""

    inventory: DraftWindowInventory
    full_source: FullDraftKVSource
    provider: SingleGroupDirectPrefixProvider


def _normalize_method(method: str) -> str:
    return method.strip().lower()


def _normalize_model_type(model_type: str) -> str:
    return model_type.strip().lower()


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
    """Return source-column capacity for the default zero-offset geometry.

    Every legacy-aligned anchor must also be block-span aligned.  Requiring
    ``block_span_tokens`` to divide ``legacy_block_size`` makes that invariant
    hold for every possible anchor rather than only for a particular request.
    Offset-aware layouts must use a separate capacity contract.
    """
    if block_span_tokens <= 0:
        raise ValueError("block span must be greater than 0")
    capacity_tokens = compute_window_capacity_tokens(
        window_size=window_size,
        legacy_block_size=legacy_block_size,
    )
    if legacy_block_size % block_span_tokens != 0:
        raise ValueError("zero-offset direct-prefix capacity requires block span to divide legacy block size")
    return (capacity_tokens + block_span_tokens - 1) // block_span_tokens


def _is_subclass(
    candidate: type[Any],
    parent: type[Any] | tuple[type[Any], ...],
) -> bool:
    try:
        return issubclass(candidate, parent)
    except TypeError:
        return False


def _classify_layout(
    backend_class: type[Any],
    builder_class: type[Any],
    kv_cache_spec: Any,
) -> DraftWindowLayoutCategory:
    if _is_subclass(backend_class, AscendDSABackend):
        return DraftWindowLayoutCategory.DSA
    if _is_subclass(backend_class, AscendMLABackend) or isinstance(
        kv_cache_spec,
        MLAAttentionSpec,
    ):
        return DraftWindowLayoutCategory.MLA
    if _is_subclass(backend_class, (AscendSFABackend, AscendSFAIndexerBackend)):
        return DraftWindowLayoutCategory.SFA
    if isinstance(
        kv_cache_spec,
        (ChunkedLocalAttentionSpec, CircularBufferSpec, RSWASpec, SlidingWindowSpec),
    ):
        return DraftWindowLayoutCategory.NATIVE_OR_RECYCLING
    if isinstance(kv_cache_spec, FullAttentionSpec) and (
        kv_cache_spec.sliding_window is not None or kv_cache_spec.attention_chunk_size is not None
    ):
        return DraftWindowLayoutCategory.NATIVE_OR_RECYCLING
    if _is_subclass(builder_class, AscendAttentionMetadataBuilder) and isinstance(
        kv_cache_spec,
        FullAttentionSpec,
    ):
        return DraftWindowLayoutCategory.DENSE_GQA_DIRECT_PREFIX
    return DraftWindowLayoutCategory.UNKNOWN


def _runtime_capability_failures(
    runtime_context: DraftWindowRuntimeContext,
) -> list[str]:
    failures: list[str] = []
    if runtime_context.graph_mode != "NONE":
        failures.append(
            f"gid=none backend=none layout=runtime reason=Graph mode {runtime_context.graph_mode} is not verified in P2"
        )
    if runtime_context.async_scheduling:
        failures.append("gid=none backend=none layout=runtime reason=Async scheduling is not verified in P2")
    topology = (
        ("TP", runtime_context.tensor_parallel_size),
        ("DP", runtime_context.data_parallel_size),
        ("PP", runtime_context.pipeline_parallel_size),
        ("PCP", runtime_context.prefill_context_parallel_size),
        ("DCP", runtime_context.decode_context_parallel_size),
    )
    failures.extend(
        f"gid=none backend=none layout=runtime reason={name}={size} is not verified in P2"
        for name, size in topology
        if size != 1
    )
    if runtime_context.sequence_parallel:
        failures.append("gid=none backend=none layout=runtime reason=sequence parallel is not verified in P2")
    if runtime_context.dsa_context_parallel:
        failures.append("gid=none backend=none layout=runtime reason=DSA-CP is not verified in P2")
    if runtime_context.flashcomm:
        failures.append("gid=none backend=none layout=runtime reason=FlashComm is not verified in P2")
    return failures


def build_draft_window_inventory(
    *,
    policy: DraftWindowPolicyDecision,
    runtime_route: str,
    attn_groups: Sequence[Sequence[AttentionGroup]],
    block_tables: Any,
    causal: bool | Mapping[int, bool],
    runtime_context: DraftWindowRuntimeContext,
) -> DraftWindowInventory:
    """Inventory every executable draft consumer in global-gid order."""
    if policy.state is not DraftWindowPolicyState.ELIGIBLE_ON:
        raise ValueError("P2 inventory requires an ELIGIBLE_ON policy decision")
    if policy.effective_window_size is None:
        raise AssertionError("eligible draft-window policy has no window size")

    runtime_route = _normalize_method(runtime_route)
    num_global_groups = len(attn_groups)
    block_sizes = tuple(block_tables.block_sizes)
    kernel_block_sizes = tuple(block_tables.kernel_block_sizes)
    failures = _runtime_capability_failures(runtime_context)
    if len(block_sizes) != num_global_groups:
        failures.append(
            "gid=none backend=none layout=source reason="
            f"logical block-size count {len(block_sizes)} does not match "
            f"attention group count {num_global_groups}"
        )
    if len(kernel_block_sizes) != num_global_groups:
        failures.append(
            "gid=none backend=none layout=source reason="
            f"kernel block-size count {len(kernel_block_sizes)} does not match "
            f"attention group count {num_global_groups}"
        )

    groups: list[DraftWindowGroupInventory] = []
    empty_global_group_ids: list[int] = []
    for outer_global_group_id, consumers in enumerate(attn_groups):
        if not consumers:
            empty_global_group_ids.append(outer_global_group_id)
            continue
        if outer_global_group_id >= len(block_sizes) or outer_global_group_id >= len(kernel_block_sizes):
            failures.append(
                "gid={gid} backend=unknown layout=source reason=missing block geometry".format(
                    gid=outer_global_group_id
                )
            )
            continue
        logical_block_size = int(block_sizes[outer_global_group_id])
        kernel_block_size = int(kernel_block_sizes[outer_global_group_id])
        for local_consumer_index, consumer in enumerate(consumers):
            backend_class = consumer.backend
            builder_class = type(consumer.get_metadata_builder(0))
            actual_global_group_id = int(consumer.kv_cache_group_id)
            layout_category = _classify_layout(
                backend_class,
                builder_class,
                consumer.kv_cache_spec,
            )
            group_failures: list[str] = []
            if actual_global_group_id != outer_global_group_id:
                group_failures.append(
                    "AttentionGroup global gid does not match its outer source "
                    f"index ({actual_global_group_id} != {outer_global_group_id})"
                )
            spec_block_size = int(consumer.kv_cache_spec.block_size)
            if spec_block_size != logical_block_size:
                group_failures.append(
                    f"KV spec logical block size does not match BlockTables ({spec_block_size} != {logical_block_size})"
                )
            if logical_block_size <= 0 or kernel_block_size <= 0:
                group_failures.append("block sizes must be greater than zero")
            elif logical_block_size % kernel_block_size != 0:
                group_failures.append(
                    f"kernel block size {kernel_block_size} does not divide logical block size {logical_block_size}"
                )
            if layout_category is not DraftWindowLayoutCategory.DENSE_GQA_DIRECT_PREFIX:
                group_failures.append(f"{layout_category.value} has no verified P2 direct-prefix provider")
            if not _is_subclass(backend_class, AscendAttentionBackend):
                group_failures.append("only the AscendAttentionBackend FIA route is a P2 direct-prefix candidate")
            if not _is_subclass(builder_class, AscendAttentionMetadataBuilder):
                group_failures.append("runtime metadata builder is not an Ascend dense-GQA builder")
            if type(consumer.kv_cache_spec) is not FullAttentionSpec:
                group_failures.append("KV spec is not the exact prefix-resident FullAttentionSpec")
            block_span_tokens = (
                kernel_block_size if not group_failures and runtime_context.decode_context_parallel_size == 1 else None
            )
            group_causal = causal if isinstance(causal, bool) else bool(causal.get(outer_global_group_id, True))
            groups.append(
                DraftWindowGroupInventory(
                    global_group_id=actual_global_group_id,
                    local_consumer_index=local_consumer_index,
                    layer_names=tuple(consumer.layer_names),
                    backend_class=backend_class,
                    builder_class=builder_class,
                    logical_block_size=logical_block_size,
                    kernel_block_size=kernel_block_size,
                    block_span_tokens=block_span_tokens,
                    causal=group_causal,
                    layout_category=layout_category,
                    capability_failures=tuple(group_failures),
                )
            )

    consumer_gids = tuple(dict.fromkeys(group.global_group_id for group in groups))
    if not groups:
        failures.append("gid=none backend=none layout=unknown reason=no executable draft attention consumer was found")
    if len(consumer_gids) > 1:
        failures.append(
            "gid={gids} backend=mixed layout=multi_group reason=homogeneous or "
            "heterogeneous multi-group runtime Window is not verified in P2".format(
                gids=",".join(str(gid) for gid in consumer_gids)
            )
        )
    for global_group_id in consumer_gids:
        count = sum(group.global_group_id == global_group_id for group in groups)
        if count > 1:
            failures.append(
                f"gid={global_group_id} backend=mixed layout=mixed reason="
                f"multiple executable consumers ({count}) share one global gid"
            )
    expected_policy_method = "dflash" if runtime_route == "dflash2" else runtime_route
    if expected_policy_method != policy.normalized_method:
        failures.append(
            "gid=none backend=none layout=runtime reason="
            f"runtime route {runtime_route!r} does not match policy method "
            f"{policy.normalized_method!r}"
        )
    if runtime_route == "dflash2":
        failures.append(
            "gid=none backend=none layout=runtime reason=DFlash2 runtime Window remains gated after P2 inventory"
        )
    elif runtime_route not in {"eagle3", "dflash", "dspark"}:
        failures.append(f"gid=none backend=none layout=runtime reason=runtime route {runtime_route!r} is not eligible")

    return DraftWindowInventory(
        method=policy.normalized_method,
        target_model_type=policy.target_model_type,
        runtime_route=runtime_route,
        window_size=policy.effective_window_size,
        groups=tuple(groups),
        empty_global_group_ids=tuple(empty_global_group_ids),
        capability_failures=tuple(failures),
    )


def build_full_draft_kv_source(
    *,
    inventory: DraftWindowInventory,
    block_tables: Any,
    idx_mapping: torch.Tensor | None = None,
) -> FullDraftKVSource:
    """Bind, but never copy or mutate, canonical full BlockTables tensors."""
    group = inventory.require_single_group_direct_prefix()
    context = _group_diagnostic_context(inventory, group)
    input_block_tables = tuple(block_tables.input_block_tables)
    logical_block_sizes = tuple(int(size) for size in block_tables.block_sizes)
    kernel_block_sizes = tuple(int(size) for size in block_tables.kernel_block_sizes)
    persistent_num_blocks = block_tables.num_blocks.gpu
    num_groups = len(input_block_tables)
    if persistent_num_blocks.ndim != 2:
        raise RuntimeError(f"{context}; reason=persistent num_blocks.gpu must be two-dimensional")
    if not (len(logical_block_sizes) == len(kernel_block_sizes) == persistent_num_blocks.shape[0] == num_groups):
        raise RuntimeError(
            f"{context}; reason=canonical BlockTables group dimensions disagree: "
            f"tables={num_groups}, logical={len(logical_block_sizes)}, "
            f"kernel={len(kernel_block_sizes)}, "
            f"num_blocks={persistent_num_blocks.shape[0]}"
        )
    _validate_source_geometry(
        inventory=inventory,
        group=group,
        logical_block_sizes=logical_block_sizes,
        kernel_block_sizes=kernel_block_sizes,
        num_groups=num_groups,
    )
    if idx_mapping is not None:
        _validate_idx_mapping(idx_mapping)
    for global_group_id, table in enumerate(input_block_tables):
        if table.ndim != 2:
            raise RuntimeError(f"{context}; reason=input_block_tables[{global_group_id}] must be two-dimensional")
        if table.shape[0] != persistent_num_blocks.shape[1]:
            raise RuntimeError(
                f"{context}; reason=input table {global_group_id} request capacity "
                f"{table.shape[0]} does not match persistent num_blocks "
                f"capacity {persistent_num_blocks.shape[1]}"
            )
    return FullDraftKVSource(
        logical_block_sizes=logical_block_sizes,
        kernel_block_sizes=kernel_block_sizes,
        source_identity=DraftWindowSourceIdentity(
            block_tables_owner_id=id(block_tables),
            input_block_table_ptrs=tuple(table.data_ptr() for table in input_block_tables),
            persistent_num_blocks_ptr=persistent_num_blocks.data_ptr(),
            idx_mapping_ptr=(None if idx_mapping is None else idx_mapping.data_ptr()),
        ),
        _block_tables_owner=block_tables,
        _input_block_tables=input_block_tables,
        _persistent_num_blocks=persistent_num_blocks,
        _idx_mapping=idx_mapping,
    )


def gather_valid_num_blocks(
    full_source: FullDraftKVSource,
    *,
    global_group_id: int,
    num_reqs: int,
    num_reqs_padded: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather persistent valid counts with the canonical batch mapping."""
    full_source.validate_owner_bindings()
    full_source._validate_global_group_id(global_group_id)
    if num_reqs < 0:
        raise ValueError("num_reqs must be greater than or equal to 0")
    if num_reqs_padded < num_reqs:
        raise ValueError("num_reqs_padded must be at least num_reqs")
    if full_source._idx_mapping is None:
        raise RuntimeError("FullSource must bind the current batch idx_mapping before gathering valid counts")
    if num_reqs > full_source._idx_mapping.shape[0]:
        raise ValueError("num_reqs exceeds idx_mapping capacity")
    persistent_counts = full_source._persistent_num_blocks[global_group_id]
    idx_mapping = full_source._idx_mapping[:num_reqs]
    if idx_mapping.device != persistent_counts.device:
        raise ValueError("idx_mapping and persistent num_blocks must share a device")
    if out is None:
        out = persistent_counts.new_zeros(num_reqs_padded)
    else:
        if out.ndim != 1 or out.shape[0] < num_reqs_padded:
            raise ValueError("valid-count output is too small for padded requests")
        if out.dtype != persistent_counts.dtype:
            raise TypeError("valid-count output dtype must match num_blocks.gpu")
        if out.device != persistent_counts.device:
            raise ValueError("valid-count output device must match num_blocks.gpu")
        borrowed_tensors = (
            *full_source._input_block_tables,
            full_source._persistent_num_blocks,
            full_source._idx_mapping,
        )
        if any(borrowed is not None and _shares_storage(out, borrowed) for borrowed in borrowed_tensors):
            raise ValueError("valid-count output must not alias borrowed FullSource storage")
        out.zero_()
    if num_reqs:
        torch.index_select(
            persistent_counts,
            0,
            idx_mapping,
            out=out[:num_reqs],
        )
    return out[:num_reqs_padded]


def _validate_idx_mapping(idx_mapping: torch.Tensor) -> None:
    if idx_mapping.ndim != 1:
        raise ValueError("idx_mapping must be one-dimensional")
    if idx_mapping.dtype not in (torch.int32, torch.int64):
        raise TypeError("idx_mapping must use int32 or int64 indices")


def _shares_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Conservatively reject output views into any borrowed source storage."""
    if left.device != right.device:
        return False
    try:
        return left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
    except (AttributeError, RuntimeError):
        return left.data_ptr() == right.data_ptr()


def _validate_source_geometry(
    *,
    inventory: DraftWindowInventory,
    group: DraftWindowGroupInventory,
    logical_block_sizes: tuple[int, ...],
    kernel_block_sizes: tuple[int, ...],
    num_groups: int,
) -> None:
    context = _group_diagnostic_context(inventory, group)
    global_group_id = group.global_group_id
    if not (
        0 <= global_group_id < num_groups
        and global_group_id < len(logical_block_sizes)
        and global_group_id < len(kernel_block_sizes)
    ):
        raise RuntimeError(f"{context}; reason=inventory gid has no canonical source geometry")
    source_logical = logical_block_sizes[global_group_id]
    source_kernel = kernel_block_sizes[global_group_id]
    if source_logical != group.logical_block_size or source_kernel != group.kernel_block_size:
        raise RuntimeError(
            f"{context}; reason=source geometry logical={source_logical} "
            f"kernel={source_kernel} does not match inventory "
            f"logical={group.logical_block_size} kernel={group.kernel_block_size}"
        )


def build_draft_window_runtime_context(
    vllm_config: Any,
    block_tables: Any,
) -> DraftWindowRuntimeContext:
    """Read the actual configured topology/mode gates without model guessing."""
    speculative_config = vllm_config.speculative_config
    compilation_config = vllm_config.compilation_config
    parallel_config = vllm_config.parallel_config
    scheduler_config = vllm_config.scheduler_config
    mode = compilation_config.cudagraph_mode
    mode_name = getattr(mode, "name", str(mode).rsplit(".", 1)[-1]).upper()
    if bool(getattr(speculative_config, "enforce_eager", False)):
        mode_name = "NONE"
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    sequence_parallel = bool(getattr(parallel_config, "use_sequence_parallel_moe", False))
    dsa_context_parallel = validate_additional_config_bool(
        additional_config.get("enable_dsa_cp", False),
        "additional_config.enable_dsa_cp",
    )
    flashcomm_explicitly_enabled = validate_additional_config_bool(
        additional_config.get("enable_flashcomm1", False),
        "additional_config.enable_flashcomm1",
    ) or os.getenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", "0").strip().lower() in (
        "1",
        "true",
    )
    flashcomm = flashcomm_explicitly_enabled or dsa_context_parallel
    return DraftWindowRuntimeContext(
        graph_mode=mode_name,
        async_scheduling=bool(getattr(scheduler_config, "async_scheduling", False)),
        tensor_parallel_size=int(parallel_config.tensor_parallel_size),
        data_parallel_size=int(parallel_config.data_parallel_size),
        pipeline_parallel_size=int(parallel_config.pipeline_parallel_size),
        prefill_context_parallel_size=int(parallel_config.prefill_context_parallel_size),
        decode_context_parallel_size=max(
            int(parallel_config.decode_context_parallel_size),
            int(getattr(block_tables, "cp_size", 1)),
        ),
        sequence_parallel=sequence_parallel,
        dsa_context_parallel=dsa_context_parallel,
        flashcomm=flashcomm,
    )


def build_draft_window_source_capability(
    *,
    policy: DraftWindowPolicyDecision,
    runtime_route: str,
    attn_groups: Sequence[Sequence[AttentionGroup]],
    block_tables: Any,
    causal: bool | Mapping[int, bool],
    vllm_config: Any,
) -> DraftWindowSourceCapability:
    """Build P2 state after ``set_attn`` and before any Window allocation."""
    inventory = build_draft_window_inventory(
        policy=policy,
        runtime_route=runtime_route,
        attn_groups=attn_groups,
        block_tables=block_tables,
        causal=causal,
        runtime_context=build_draft_window_runtime_context(
            vllm_config,
            block_tables,
        ),
    )
    group = inventory.require_single_group_direct_prefix()
    full_source = build_full_draft_kv_source(
        inventory=inventory,
        block_tables=block_tables,
    )
    provider = SingleGroupDirectPrefixProvider(
        inventory=inventory,
        full_source=full_source,
        # MRV1 aligned its anchor to the first executable draft kernel block.
        # With one verified consumer this is the only legacy anchor quantum.
        legacy_block_size=group.kernel_block_size,
        expected_future_offset=resolve_static_future_offset(
            normalized_method=policy.normalized_method,
            configured_k=int(vllm_config.speculative_config.num_speculative_tokens),
            uses_dynamic_k=bool(vllm_config.speculative_config.uses_dynamic_speculative_decoding()),
        ),
    )
    return DraftWindowSourceCapability(
        inventory=inventory,
        full_source=full_source,
        provider=provider,
    )
