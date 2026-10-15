# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Policy, source, and generation-scoped read-plane contracts for draft windows.

P1 decides whether a configured window is eligible and describes its scalar
interval. P2 inventories the actual draft attention consumers and records how
that interval projects onto the canonical full block tables. P3 owns the
materialized read state and its lifetime, but still does not connect it to
attention.
"""

import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
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
from vllm_ascend.device.hardware_profile import (
    AttentionBackendFamily,
    get_current_hardware_profile,
)

if TYPE_CHECKING:
    from vllm_ascend.worker.v2.spec_decode.draft_window_eager_capability import (
        P7EagerTopologyFingerprint,
    )


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

    attention_backend_family: AttentionBackendFamily
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
    prefix_caching: bool
    tensor_parallel_rank: int = 0


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
    # Geometry facts are kept separate from phase-specific capability gates.
    # P7-X may close a conservative P2 backend/runtime gate, but it must never
    # erase a source-owner or block-geometry failure.
    source_geometry_failures: tuple[str, ...] = ()
    # DSA drafting supports only the uncompressed/native SWA cache contract.
    # This value comes from the actual metadata-builder instance, not a model
    # name or backend-class guess.
    drafting_compression_ratio: int | None = None


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
    _persistent_num_blocks_cpu: torch.Tensor = field(repr=False, compare=False)
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

    def references_current_num_blocks_gpu_view(self, tensor: torch.Tensor) -> bool:
        """Return whether ``tensor`` is the accelerator view active right now.

        ``UvaBackedTensor`` may rotate this view, so it is intentionally not a
        persistent source identity. The stable owner check is the CPU method
        below.
        """
        return self._block_tables_owner.num_blocks.gpu is tensor

    def references_persistent_num_blocks_cpu(self, tensor: torch.Tensor) -> bool:
        return self._persistent_num_blocks_cpu is tensor

    def references_same_idx_mapping_tensor(self, tensor: torch.Tensor) -> bool:
        """Return whether ``tensor`` is the same object, not whether it is fresh."""
        return self._idx_mapping is tensor

    def shares_canonical_storage(self, tensor: torch.Tensor) -> bool:
        """Return whether a tensor aliases any canonical FullSource backing."""
        self.validate_owner_bindings()
        canonical_tensors = self._input_block_tables + (
            self._persistent_num_blocks_cpu,
            self._block_tables_owner.num_blocks.gpu,
        )
        return any(draft_window_tensors_share_storage(tensor, canonical) for canonical in canonical_tensors)

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
            _persistent_num_blocks_cpu=self._persistent_num_blocks_cpu,
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
        if self._block_tables_owner.num_blocks.cpu is not self._persistent_num_blocks_cpu:
            raise RuntimeError("canonical persistent num_blocks.cpu was replaced after FullSource binding")
        current_gpu = self._block_tables_owner.num_blocks.gpu
        if (
            current_gpu.ndim != self._persistent_num_blocks_cpu.ndim
            or current_gpu.shape != self._persistent_num_blocks_cpu.shape
            or current_gpu.dtype != self._persistent_num_blocks_cpu.dtype
        ):
            raise RuntimeError("current num_blocks.gpu view does not match its canonical CPU owner")

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
    # P2 uses the implicit failure-free dense group. P7-X can explicitly bind
    # one source-safe backend group (currently uncompressed DSA/DSA-CP) while
    # preserving the original inventory diagnostics and layout identity.
    explicit_source_group: DraftWindowGroupInventory | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    source_block_span_tokens: int | None = None
    _group: DraftWindowGroupInventory = field(init=False, repr=False)
    _capacity_columns: int = field(init=False, repr=False)
    _block_span_tokens: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.explicit_source_group is None:
            group = self.inventory.require_single_group_direct_prefix()
            span = group.block_span_tokens
            assert span is not None
            if self.source_block_span_tokens is not None and self.source_block_span_tokens != span:
                raise ValueError("explicit source block span disagrees with the P2 group")
        else:
            group = self.explicit_source_group
            if len(self.inventory.groups) != 1 or self.inventory.groups[0] != group:
                raise ValueError("explicit source group must be the sole inventory consumer")
            if group.source_geometry_failures:
                raise NotImplementedError(
                    "explicit backend provider has unresolved source geometry: "
                    + "; ".join(group.source_geometry_failures)
                )
            if group.layout_category not in (
                DraftWindowLayoutCategory.DENSE_GQA_DIRECT_PREFIX,
                DraftWindowLayoutCategory.DSA,
            ):
                raise NotImplementedError(
                    f"layout {group.layout_category.value} has no explicit direct-prefix source contract"
                )
            span = self.source_block_span_tokens
            if span is None:
                raise ValueError("explicit backend provider requires a source block span")
            if span != group.kernel_block_size:
                raise ValueError("explicit source block span must equal the canonical kernel block size")
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
        object.__setattr__(self, "_block_span_tokens", span)

    @property
    def global_group_id(self) -> int:
        """Return the sole executable global cache-group id."""
        return self._group.global_group_id

    @property
    def capacity_columns(self) -> int:
        """Return the fixed narrow-table capacity proven by P2 geometry."""
        return self._capacity_columns

    @property
    def block_span_tokens(self) -> int:
        """Return the source-token span represented by one table column."""
        return self._block_span_tokens

    @property
    def source_group(self) -> DraftWindowGroupInventory:
        """Return the exact inventory consumer bound by this provider."""
        return self._group

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
        return self._describe_interval(
            window_start=plan.window_start,
            window_visible_length=plan.window_visible_length,
            valid_num_blocks=valid_num_blocks,
        )

    def describe_frozen_step(
        self,
        proposal_plan: DraftWindowScalarPlan,
        *,
        full_exact_end: int,
        valid_num_blocks: int,
    ) -> DraftWindowReadDescriptor:
        """Project one Eagle3 step using the proposal-frozen anchor.

        ``describe`` validates the one-shot ``E + K`` contract used by P2 and
        parallel drafting. Eagle3 instead freezes ``S`` from proposal-entry
        ``L0 + K`` and advances only the real per-step end ``E_s``. This method
        keeps that distinction explicit and refuses to infer a new anchor from
        ``E_s``.
        """
        self.full_source.validate_owner_bindings()
        if not isinstance(valid_num_blocks, int) or isinstance(valid_num_blocks, bool):
            raise TypeError("valid_num_blocks must be a Python int")
        if valid_num_blocks < 0:
            raise ValueError("valid_num_blocks must be greater than or equal to 0")
        if self.inventory.method != "eagle3":
            raise ValueError("frozen per-step projection is defined only for Eagle3")
        proposal_offset = proposal_plan.full_horizon_end - proposal_plan.full_exact_end
        if proposal_offset != self.expected_future_offset:
            raise ValueError(
                "proposal future offset does not match configured static future offset: "
                f"offset={proposal_offset} expected={self.expected_future_offset}"
            )
        expected_start = compute_legacy_anchor(
            full_context_length=proposal_plan.full_exact_end,
            future_offset=proposal_offset,
            window_size=self.inventory.window_size,
            legacy_block_size=self.legacy_block_size,
        )
        if proposal_plan.window_start != expected_start:
            raise ValueError(
                "proposal plan does not use the canonical frozen anchor: "
                f"{proposal_plan.window_start} != {expected_start}"
            )
        if not proposal_plan.window_start <= full_exact_end <= proposal_plan.full_horizon_end:
            raise ValueError(
                "Eagle3 step end is outside the frozen proposal interval: "
                f"start={proposal_plan.window_start} end={full_exact_end} "
                f"horizon={proposal_plan.full_horizon_end}"
            )
        expected_capacity = compute_window_capacity_tokens(
            window_size=self.inventory.window_size,
            legacy_block_size=self.legacy_block_size,
        )
        if proposal_plan.window_capacity_tokens != expected_capacity:
            raise ValueError(
                "proposal plan capacity does not match provider geometry: "
                f"{proposal_plan.window_capacity_tokens} != {expected_capacity}"
            )
        return self._describe_interval(
            window_start=proposal_plan.window_start,
            window_visible_length=compute_visible_length(
                full_exact_end=full_exact_end,
                window_start=proposal_plan.window_start,
            ),
            valid_num_blocks=valid_num_blocks,
        )

    def _describe_interval(
        self,
        *,
        window_start: int,
        window_visible_length: int,
        valid_num_blocks: int,
    ) -> DraftWindowReadDescriptor:
        """Project a previously validated absolute interval onto FullSource."""
        span = self._block_span_tokens
        projection = project_direct_prefix(
            window_start=window_start,
            window_visible_length=window_visible_length,
            block_span_tokens=span,
        )
        if projection.first_token_offset != 0:
            raise NotImplementedError(
                "MRV2 direct-prefix provider cannot consume a nonzero first-token "
                f"offset: gid={self._group.global_group_id} "
                f"start={window_start} span={span} "
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
    eager_topology_fingerprint: "P7EagerTopologyFingerprint | None" = None
    # P7-X mixed/prefix rows can require a full-resident row in the same
    # stable backing that carries narrow decode rows.  Existing P3 fixtures
    # omit this field and retain the original narrow capacity.
    workspace_capacity_columns: int | None = None


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
    if runtime_context.prefix_caching:
        failures.append("gid=none backend=none layout=runtime reason=prefix cache is not verified for Window ON")
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
            metadata_builder = consumer.get_metadata_builder(0)
            builder_class = type(metadata_builder)
            actual_global_group_id = int(consumer.kv_cache_group_id)
            layout_category = _classify_layout(
                backend_class,
                builder_class,
                consumer.kv_cache_spec,
            )
            group_failures: list[str] = []
            source_geometry_failures: list[str] = []
            if actual_global_group_id != outer_global_group_id:
                source_geometry_failures.append(
                    "AttentionGroup global gid does not match its outer source "
                    f"index ({actual_global_group_id} != {outer_global_group_id})"
                )
            spec_block_size = int(consumer.kv_cache_spec.block_size)
            if spec_block_size != logical_block_size:
                source_geometry_failures.append(
                    f"KV spec logical block size does not match BlockTables ({spec_block_size} != {logical_block_size})"
                )
            if logical_block_size <= 0 or kernel_block_size <= 0:
                source_geometry_failures.append("block sizes must be greater than zero")
            elif logical_block_size % kernel_block_size != 0:
                source_geometry_failures.append(
                    f"kernel block size {kernel_block_size} does not divide logical block size {logical_block_size}"
                )
            group_failures.extend(source_geometry_failures)
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
            drafting_compression_ratio: int | None = None
            if layout_category is DraftWindowLayoutCategory.DSA:
                raw_ratio = getattr(metadata_builder, "compressor_ratio", None)
                if isinstance(raw_ratio, int) and not isinstance(raw_ratio, bool):
                    drafting_compression_ratio = raw_ratio
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
                    source_geometry_failures=tuple(source_geometry_failures),
                    drafting_compression_ratio=drafting_compression_ratio,
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
    source_group: DraftWindowGroupInventory | None = None,
) -> FullDraftKVSource:
    """Bind, but never copy or mutate, canonical full BlockTables tensors."""
    if source_group is None:
        group = inventory.require_single_group_direct_prefix()
    else:
        group = source_group
        if len(inventory.groups) != 1 or inventory.groups[0] != group:
            raise ValueError("explicit FullSource group must be the sole inventory consumer")
        if group.source_geometry_failures:
            raise NotImplementedError(
                "explicit FullSource group has unresolved source geometry: " + "; ".join(group.source_geometry_failures)
            )
    context = _group_diagnostic_context(inventory, group)
    input_block_tables = tuple(block_tables.input_block_tables)
    logical_block_sizes = tuple(int(size) for size in block_tables.block_sizes)
    kernel_block_sizes = tuple(int(size) for size in block_tables.kernel_block_sizes)
    persistent_num_blocks = block_tables.num_blocks.gpu
    persistent_num_blocks_cpu = block_tables.num_blocks.cpu
    num_groups = len(input_block_tables)
    if persistent_num_blocks.ndim != 2:
        raise RuntimeError(f"{context}; reason=persistent num_blocks.gpu must be two-dimensional")
    if persistent_num_blocks_cpu.ndim != 2:
        raise RuntimeError(f"{context}; reason=persistent num_blocks.cpu must be two-dimensional")
    if persistent_num_blocks_cpu.device.type != "cpu":
        raise RuntimeError(f"{context}; reason=persistent num_blocks.cpu must live on CPU")
    if persistent_num_blocks_cpu.dtype != persistent_num_blocks.dtype:
        raise RuntimeError(f"{context}; reason=CPU/GPU persistent num_blocks dtype mismatch")
    if persistent_num_blocks_cpu.shape != persistent_num_blocks.shape:
        raise RuntimeError(f"{context}; reason=CPU/GPU persistent num_blocks shape mismatch")
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
            # UvaBackedTensor rotates its accelerator view. The CPU source of
            # truth is the stable owner; its pointer belongs in source identity.
            persistent_num_blocks_ptr=persistent_num_blocks_cpu.data_ptr(),
            idx_mapping_ptr=(None if idx_mapping is None else idx_mapping.data_ptr()),
        ),
        _block_tables_owner=block_tables,
        _input_block_tables=input_block_tables,
        _persistent_num_blocks_cpu=persistent_num_blocks_cpu,
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
    persistent_counts = full_source._block_tables_owner.num_blocks.gpu[global_group_id]
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
            full_source._block_tables_owner.num_blocks.gpu,
            full_source._persistent_num_blocks_cpu,
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


def gather_valid_num_blocks_cpu(
    full_source: FullDraftKVSource,
    *,
    global_group_id: int,
    idx_mapping_np: np.ndarray,
    num_reqs: int,
    num_reqs_padded: int,
) -> torch.Tensor:
    """Gather valid counts from the canonical CPU owner in batch order.

    The accelerator view of ``UvaBackedTensor`` is a rotating transport and is
    not a stable freshness identity. P4 eager planning is host-side, so using
    the source-of-truth CPU owner avoids a device-to-host synchronization.
    """
    full_source.validate_owner_bindings()
    full_source._validate_global_group_id(global_group_id)
    if num_reqs < 0:
        raise ValueError("num_reqs must be greater than or equal to 0")
    if num_reqs_padded < num_reqs:
        raise ValueError("num_reqs_padded must be at least num_reqs")
    mapping = np.asarray(idx_mapping_np)
    if mapping.ndim != 1:
        raise ValueError("CPU idx_mapping must be one-dimensional")
    if not np.issubdtype(mapping.dtype, np.integer):
        raise TypeError("CPU idx_mapping must contain integer indices")
    if mapping.shape[0] < num_reqs:
        raise ValueError("num_reqs exceeds CPU idx_mapping capacity")
    persistent_counts = full_source._persistent_num_blocks_cpu[global_group_id]
    out = torch.zeros(num_reqs_padded, dtype=torch.int32, device="cpu")
    if num_reqs:
        indices = torch.from_numpy(mapping[:num_reqs].astype(np.int64, copy=False))
        torch.index_select(
            persistent_counts,
            0,
            indices,
            out=out[:num_reqs],
        )
    return out


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


def draft_window_tensors_share_storage(
    left: torch.Tensor,
    right: torch.Tensor,
) -> bool:
    """Expose the conservative storage-alias oracle to sibling read-plane owners."""
    return _shares_storage(left, right)


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
    dsa_context_parallel_requested = validate_additional_config_bool(
        additional_config.get("enable_dsa_cp", False),
        "additional_config.enable_dsa_cp",
    )
    dsa_context_parallel = dsa_context_parallel_requested and _effective_dsa_context_parallel()
    flashcomm_explicitly_enabled = validate_additional_config_bool(
        additional_config.get("enable_flashcomm1", False),
        "additional_config.enable_flashcomm1",
    ) or os.getenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", "0").strip().lower() in (
        "1",
        "true",
    )
    flashcomm = flashcomm_explicitly_enabled or dsa_context_parallel
    return DraftWindowRuntimeContext(
        attention_backend_family=get_current_hardware_profile().attention_backend_family,
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
        prefix_caching=bool(getattr(getattr(vllm_config, "cache_config", None), "enable_prefix_caching", False)),
        tensor_parallel_rank=_get_tensor_parallel_rank(parallel_config),
    )


def _effective_dsa_context_parallel() -> bool:
    """Read the same validated DSA-CP authority used by backend selection."""
    from vllm_ascend.utils import enable_dsa_cp

    return bool(enable_dsa_cp())


def _get_tensor_parallel_rank(parallel_config: Any) -> int:
    """Resolve process-local TP identity without inventing a rank.

    Unit/config-only callers may provide an explicit rank.  A live TP route
    must have an initialized vLLM TP group; failure to resolve it is a hard
    source/configuration error rather than a rank-zero fallback.
    """
    world_size = int(parallel_config.tensor_parallel_size)
    configured_rank = getattr(parallel_config, "tensor_parallel_rank", None)
    if configured_rank is not None:
        rank = int(configured_rank)
    elif world_size == 1:
        rank = 0
    else:
        try:
            from vllm.distributed import get_tp_group

            rank = int(get_tp_group().rank_in_group)
        except (AssertionError, AttributeError, RuntimeError) as error:
            raise RuntimeError("TP Draft Window route requires an initialized TP group rank") from error
    if rank < 0 or rank >= world_size:
        raise ValueError(f"TP rank {rank} is outside [0, {world_size})")
    return rank


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
    runtime_context = build_draft_window_runtime_context(
        vllm_config,
        block_tables,
    )
    inventory = build_draft_window_inventory(
        policy=policy,
        runtime_route=runtime_route,
        attn_groups=attn_groups,
        block_tables=block_tables,
        causal=causal,
        runtime_context=runtime_context,
    )
    uses_dynamic_k = bool(vllm_config.speculative_config.uses_dynamic_speculative_decoding())
    from vllm_ascend.worker.v2.spec_decode.draft_window_nongraph import (
        evaluate_p7x_nongraph_capability,
    )

    eager_decision = evaluate_p7x_nongraph_capability(
        policy=policy,
        inventory=inventory,
        runtime_context=runtime_context,
        uses_dynamic_k=uses_dynamic_k,
        provider_version=DraftWindowRuntimeState.PROVIDER_VERSION,
        execution_lane=0,
    )
    eager_topology_fingerprint = eager_decision.require_local_materialization()
    if len(inventory.groups) != 1:
        raise NotImplementedError("P7-X live source currently requires one executable group")
    group = inventory.groups[0]
    if (
        group.layout_category
        not in (
            DraftWindowLayoutCategory.DENSE_GQA_DIRECT_PREFIX,
            DraftWindowLayoutCategory.DSA,
        )
        or group.source_geometry_failures
    ):
        raise NotImplementedError("P7-X live source requires one source-safe dense or uncompressed DSA prefix provider")
    full_source = build_full_draft_kv_source(
        inventory=inventory,
        block_tables=block_tables,
        source_group=group,
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
            uses_dynamic_k=uses_dynamic_k,
        ),
        explicit_source_group=group,
        source_block_span_tokens=group.kernel_block_size,
    )
    return DraftWindowSourceCapability(
        inventory=inventory,
        full_source=full_source,
        provider=provider,
        eager_topology_fingerprint=eager_topology_fingerprint,
        # Mixed/chunked rows may carry a canonical full-resident read beside
        # windowed decode rows. Reserve one stable full-width backing once;
        # active rows still copy only their descriptor columns.
        workspace_capacity_columns=full_source.input_block_table_row_width(
            group.global_group_id,
        ),
    )


class DraftWindowExecutionPhase(Enum):
    """Typed identity for a future live/capture binding point."""

    PROPOSAL = "proposal"
    STEP = "step"


@dataclass(frozen=True, slots=True)
class DraftWindowGenerationIdentity:
    """Immutable freshness identity for one materialized Draft read state.

    ``mapping_epoch`` belongs to the caller.  Pointer and Python-object
    identity are deliberately only part of ``source_identity`` diagnostics;
    neither substitutes for the epoch or ordered stable request IDs.
    """

    generation_id: int
    mapping_epoch: int
    ordered_request_fingerprint: tuple[str, ...]
    actual_request_count: int
    padded_request_count: int
    normalized_method: str
    global_group_id: int
    read_descriptor_fingerprint: tuple[DraftWindowReadDescriptor, ...]
    provider_id: int
    provider_version: int
    source_identity: DraftWindowSourceIdentity
    execution_lane: int
    phase: DraftWindowExecutionPhase
    step: int
    topology_fingerprint: "P7EagerTopologyFingerprint | None" = None
    row_phase_fingerprint: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_nonnegative_python_int(self.generation_id, "generation id")
        _require_nonnegative_python_int(self.mapping_epoch, "mapping epoch")
        _require_nonnegative_python_int(self.actual_request_count, "actual request count")
        _require_nonnegative_python_int(self.padded_request_count, "padded request count")
        _require_nonnegative_python_int(self.global_group_id, "global group id")
        _require_nonnegative_python_int(self.execution_lane, "execution lane")
        _require_nonnegative_python_int(self.step, "step")
        if self.provider_id <= 0:
            raise ValueError("provider id must be greater than 0")
        if self.provider_version <= 0:
            raise ValueError("provider version must be greater than 0")
        if self.actual_request_count != len(self.ordered_request_fingerprint):
            raise ValueError("actual request count must match the ordered request fingerprint")
        if self.padded_request_count < self.actual_request_count:
            raise ValueError("padded request count must be at least the actual request count")
        if len(self.read_descriptor_fingerprint) != self.actual_request_count:
            raise ValueError("read descriptor fingerprint must match the actual request count")
        if any(descriptor.global_group_id != self.global_group_id for descriptor in self.read_descriptor_fingerprint):
            raise ValueError("read descriptor fingerprint must use the generation global gid")
        if any(not isinstance(request_id, str) or not request_id for request_id in self.ordered_request_fingerprint):
            raise ValueError("request fingerprint must contain non-empty stable request IDs")
        if len(set(self.ordered_request_fingerprint)) != len(self.ordered_request_fingerprint):
            raise ValueError("request fingerprint must contain unique stable request IDs")
        if not self.normalized_method or self.normalized_method != _normalize_method(self.normalized_method):
            raise ValueError("normalized method must be a non-empty canonical method name")
        if not isinstance(self.phase, DraftWindowExecutionPhase):
            raise TypeError("phase must be a DraftWindowExecutionPhase")
        if self.row_phase_fingerprint and len(self.row_phase_fingerprint) != self.actual_request_count:
            raise ValueError("row phase fingerprint must match the actual request count")


class DraftWindowCompletionToken(Protocol):
    """Minimal completion contract; concrete NPU events remain a later gate."""

    def is_complete(self) -> bool: ...


class DraftWindowLeaseState(Enum):
    """Observable states of the single serialized P3 execution lane."""

    ACQUIRED = "acquired"
    READER_PENDING = "reader_pending"
    RETIRED_PENDING = "retired_pending"
    RETIRED = "retired"


class DraftWindowWorkspace:
    """Fixed-capacity, stable-address backing for one serialized read lane."""

    def __init__(
        self,
        *,
        max_num_reqs: int,
        table_capacity_columns: int,
        table_template: torch.Tensor,
        valid_count_template: torch.Tensor,
    ) -> None:
        if max_num_reqs <= 0:
            raise ValueError("workspace request capacity must be greater than 0")
        if table_capacity_columns <= 0:
            raise ValueError("workspace table capacity must be greater than 0")
        self.max_num_reqs = max_num_reqs
        self.table_capacity_columns = table_capacity_columns
        self.window_block_table = table_template.new_zeros((max_num_reqs, table_capacity_columns))
        self.valid_num_blocks = valid_count_template.new_zeros(max_num_reqs)
        self.device_window_exact = torch.zeros(
            max_num_reqs,
            dtype=torch.int32,
            device=table_template.device,
        )
        self.cpu_window_exact = torch.zeros(max_num_reqs, dtype=torch.int32)
        self.cpu_window_upper = torch.zeros(max_num_reqs, dtype=torch.int32)
        self._bound_identity: DraftWindowGenerationIdentity | None = None
        self._active_table_columns: tuple[int, ...] = ()

    @property
    def storage_pointers(self) -> tuple[int, ...]:
        """Expose addresses for stability tests, never as freshness tokens."""
        return (
            self.window_block_table.data_ptr(),
            self.valid_num_blocks.data_ptr(),
            self.device_window_exact.data_ptr(),
            self.cpu_window_exact.data_ptr(),
            self.cpu_window_upper.data_ptr(),
        )

    @property
    def bound_identity(self) -> DraftWindowGenerationIdentity | None:
        return self._bound_identity

    def materialize(
        self,
        *,
        identity: DraftWindowGenerationIdentity,
        full_source: FullDraftKVSource,
        read_descriptors: Sequence[DraftWindowReadDescriptor],
        device_window_exact: torch.Tensor | None,
        cpu_window_exact: torch.Tensor,
        cpu_window_upper: torch.Tensor,
        cpu_valid_num_blocks: torch.Tensor | None = None,
    ) -> None:
        """Rewrite all active and inert state from the canonical full source."""
        if self._bound_identity is not None:
            raise RuntimeError("workspace is already bound to an active generation")
        actual = identity.actual_request_count
        padded = identity.padded_request_count
        if padded > self.max_num_reqs:
            raise RuntimeError(f"draft-window request capacity exceeded: padded={padded} capacity={self.max_num_reqs}")
        if len(read_descriptors) != actual:
            raise ValueError("read descriptor count must match the actual request count")
        if tuple(read_descriptors) != identity.read_descriptor_fingerprint:
            raise RuntimeError("read descriptors do not match the generation fingerprint")
        if device_window_exact is not None:
            _validate_window_length_input(
                device_window_exact,
                name="device Window exact",
                padded_request_count=padded,
                expected_device=self.device_window_exact.device,
            )
        _validate_window_length_input(
            cpu_window_exact,
            name="CPU Window exact",
            padded_request_count=padded,
            expected_device=torch.device("cpu"),
        )
        if cpu_valid_num_blocks is not None:
            _validate_window_length_input(
                cpu_valid_num_blocks,
                name="CPU valid block counts",
                padded_request_count=padded,
                expected_device=torch.device("cpu"),
            )
        _validate_window_length_input(
            cpu_window_upper,
            name="CPU Window upper",
            padded_request_count=padded,
            expected_device=torch.device("cpu"),
        )
        workspace_tensors = (
            self.window_block_table,
            self.valid_num_blocks,
            self.device_window_exact,
            self.cpu_window_exact,
            self.cpu_window_upper,
        )
        if any(
            _shares_storage(supplied, backing)
            for supplied in tuple(
                tensor
                for tensor in (
                    device_window_exact,
                    cpu_window_exact,
                    cpu_window_upper,
                    cpu_valid_num_blocks,
                )
                if tensor is not None
            )
            for backing in workspace_tensors
        ):
            raise ValueError("Window length inputs must not alias workspace backing")
        if (
            device_window_exact is not None
            and actual
            and not torch.equal(
                device_window_exact[:actual].cpu(),
                cpu_window_exact[:actual],
            )
        ):
            raise ValueError("device and CPU Window exact lengths must agree for active requests")
        if actual and bool(torch.any(cpu_window_exact[:actual] < 0)):
            raise ValueError("Window exact lengths must be greater than or equal to 0")
        if actual and bool(torch.any(cpu_window_upper[:actual] < cpu_window_exact[:actual])):
            raise ValueError("CPU Window upper must be greater than or equal to exact")

        full_source.validate_owner_bindings()
        full_source._validate_global_group_id(identity.global_group_id)
        if full_source.source_identity != identity.source_identity:
            raise RuntimeError("materialization source identity does not match generation identity")
        source_table = full_source._input_block_tables[identity.global_group_id]
        if source_table.device != self.window_block_table.device:
            raise ValueError("canonical source table and Window workspace must share a device")
        if source_table.dtype != self.window_block_table.dtype:
            raise TypeError("canonical source table and Window workspace must share a dtype")

        try:
            if cpu_valid_num_blocks is None:
                gather_valid_num_blocks(
                    full_source,
                    global_group_id=identity.global_group_id,
                    num_reqs=actual,
                    num_reqs_padded=padded,
                    out=self.valid_num_blocks,
                )
            elif actual:
                self.valid_num_blocks[:actual].copy_(cpu_valid_num_blocks[:actual])
            active_columns: list[int] = []
            for row, descriptor in enumerate(read_descriptors):
                if descriptor.global_group_id != identity.global_group_id:
                    raise ValueError("read descriptor global gid does not match generation identity")
                if (
                    descriptor.source_start_column < 0
                    or descriptor.required_columns < 0
                    or descriptor.safe_extent_tokens < 0
                    or descriptor.valid_num_blocks < 0
                ):
                    raise ValueError("read descriptor extents must be greater than or equal to 0")
                if descriptor.first_token_offset != 0:
                    raise NotImplementedError("P3 supports only direct-prefix descriptors with zero token offset")
                if descriptor.source_row_width != source_table.shape[1]:
                    raise RuntimeError("read descriptor source row width is stale")
                if descriptor.required_columns > self.table_capacity_columns:
                    raise RuntimeError(
                        "draft-window table capacity exceeded: "
                        f"required={descriptor.required_columns} "
                        f"capacity={self.table_capacity_columns}"
                    )
                gathered_valid = int(
                    (self.valid_num_blocks[row] if cpu_valid_num_blocks is None else cpu_valid_num_blocks[row]).item()
                )
                if gathered_valid > source_table.shape[1]:
                    raise RuntimeError("current valid block count exceeds the canonical source row width")
                if descriptor.valid_num_blocks != gathered_valid:
                    raise RuntimeError(
                        "read descriptor valid count is stale: "
                        f"row={row} descriptor={descriptor.valid_num_blocks} "
                        f"current={gathered_valid}"
                    )
                if descriptor.source_end_column > gathered_valid:
                    raise RuntimeError(
                        "draft-window source shortage during materialization: "
                        f"row={row} end={descriptor.source_end_column} "
                        f"valid={gathered_valid}"
                    )
                required = descriptor.required_columns
                if required:
                    self.window_block_table[row, :required].copy_(
                        source_table[
                            row,
                            descriptor.source_start_column : descriptor.source_end_column,
                        ]
                    )
                active_columns.append(required)
            if actual:
                device_exact_source = cpu_window_exact if device_window_exact is None else device_window_exact
                self.device_window_exact[:actual].copy_(device_exact_source[:actual])
                self.cpu_window_exact[:actual].copy_(cpu_window_exact[:actual])
                self.cpu_window_upper[:actual].copy_(cpu_window_upper[:actual])
            self._active_table_columns = tuple(active_columns)
            self._bound_identity = identity
        except BaseException:
            self._zero_all()
            raise

    def build_inputs(self, lease: "DraftWindowLease") -> "DraftMetadataInputs":
        identity = lease.identity
        self.assert_bound(identity)
        padded = identity.padded_request_count
        return DraftMetadataInputs(
            identity=identity,
            window_block_table=self.window_block_table[:padded],
            valid_num_blocks=self.valid_num_blocks[:padded],
            device_window_exact=self.device_window_exact[:padded],
            cpu_window_exact=self.cpu_window_exact[:padded],
            cpu_window_upper=self.cpu_window_upper[:padded],
            active_table_columns=self._active_table_columns,
            actual_request_count=identity.actual_request_count,
            padded_request_count=padded,
            table_capacity_columns=self.table_capacity_columns,
            _lease=lease,
        )

    def assert_bound(self, expected_identity: DraftWindowGenerationIdentity) -> None:
        if self._bound_identity != expected_identity:
            raise RuntimeError("workspace generation identity does not match the expected generation")

    def reset(self, expected_identity: DraftWindowGenerationIdentity) -> None:
        # Validation may fail before materialization binds the identity.  The
        # lease pool already proved that ``expected_identity`` owns this lane,
        # so an unbound workspace is safe to clear.  A different bound
        # generation remains a hard error.
        if self._bound_identity is not None:
            self.assert_bound(expected_identity)
        self._zero_all()

    def _zero_all(self) -> None:
        self.window_block_table.zero_()
        self.valid_num_blocks.zero_()
        self.device_window_exact.zero_()
        self.cpu_window_exact.zero_()
        self.cpu_window_upper.zero_()
        self._active_table_columns = ()
        self._bound_identity = None


class DraftWindowLease:
    """Generation handle whose retirement is coordinated by its runtime owner."""

    def __init__(
        self,
        *,
        owner: "DraftWindowRuntimeState",
        identity: DraftWindowGenerationIdentity,
        workspace: DraftWindowWorkspace,
    ) -> None:
        self._owner = owner
        self.identity = identity
        self.workspace = workspace
        self._state = DraftWindowLeaseState.ACQUIRED
        self._completion_token: DraftWindowCompletionToken | None = None

    @property
    def state(self) -> DraftWindowLeaseState:
        return self._state

    def assert_current(self, expected_identity: DraftWindowGenerationIdentity) -> None:
        if self.identity != expected_identity:
            raise RuntimeError("lease generation identity does not match the expected generation")
        if self._state in (
            DraftWindowLeaseState.RETIRED_PENDING,
            DraftWindowLeaseState.RETIRED,
        ):
            raise RuntimeError("cannot use a retired lease")
        self.workspace.assert_bound(expected_identity)

    def retire(self, expected_identity: DraftWindowGenerationIdentity) -> None:
        """Retire through the one runtime owner so store and backing agree."""
        self._owner._retire_lease(self, expected_identity)


class _DraftWindowLeasePool:
    """One stable lane; a P7 pool may own several independent instances."""

    def __init__(self, workspace: DraftWindowWorkspace, execution_lane: int) -> None:
        self._workspace = workspace
        self._execution_lane = execution_lane
        self._current: DraftWindowLease | None = None

    def acquire(
        self,
        *,
        owner: "DraftWindowRuntimeState",
        identity: DraftWindowGenerationIdentity,
    ) -> DraftWindowLease:
        if self._current is not None:
            raise RuntimeError(f"draft-window execution lane {self._execution_lane} is still in use")
        if identity.execution_lane != self._execution_lane:
            raise RuntimeError(
                "draft-window generation lane does not match its workspace: "
                f"generation={identity.execution_lane} workspace={self._execution_lane}"
            )
        lease = DraftWindowLease(
            owner=owner,
            identity=identity,
            workspace=self._workspace,
        )
        self._current = lease
        return lease

    def require_current(
        self,
        lease: DraftWindowLease,
        expected_identity: DraftWindowGenerationIdentity,
    ) -> None:
        if lease.identity != expected_identity:
            raise RuntimeError("lease generation identity does not match the expected generation")
        if self._current is not lease:
            raise RuntimeError("cannot use a retired lease")

    def mark_reader_pending(
        self,
        lease: DraftWindowLease,
        token: DraftWindowCompletionToken,
    ) -> None:
        self.require_current(lease, lease.identity)
        if lease._state is not DraftWindowLeaseState.ACQUIRED:
            raise RuntimeError("a completion token is already bound to this lease")
        if not callable(getattr(token, "is_complete", None)):
            raise TypeError("completion token must implement is_complete()")
        lease._completion_token = token
        lease._state = DraftWindowLeaseState.READER_PENDING

    def retire(
        self,
        lease: DraftWindowLease,
        expected_identity: DraftWindowGenerationIdentity,
    ) -> None:
        self.require_current(lease, expected_identity)
        if lease._state is DraftWindowLeaseState.ACQUIRED:
            self._finish_retirement(lease)
            return
        if lease._state is DraftWindowLeaseState.READER_PENDING:
            token = lease._completion_token
            assert token is not None
            if bool(token.is_complete()):
                self._finish_retirement(lease)
            else:
                lease._state = DraftWindowLeaseState.RETIRED_PENDING
            return
        raise RuntimeError("cannot retire an already retired lease")

    def reclaim_completed_reader(
        self,
        expected_identity: DraftWindowGenerationIdentity,
    ) -> DraftWindowGenerationIdentity | None:
        lease = self._current
        if lease is None:
            raise RuntimeError("draft-window lane has no pending reader")
        if lease.identity != expected_identity:
            raise RuntimeError("pending reader belongs to a different generation")
        if lease._state is not DraftWindowLeaseState.RETIRED_PENDING:
            raise RuntimeError("draft-window reader is not pending retirement")
        token = lease._completion_token
        assert token is not None
        if bool(token.is_complete()):
            self._finish_retirement(lease)
            return expected_identity
        return None

    def _finish_retirement(self, lease: DraftWindowLease) -> None:
        self._workspace.reset(lease.identity)
        lease._state = DraftWindowLeaseState.RETIRED
        self._current = None


@dataclass(frozen=True, slots=True, eq=False)
class DraftMetadataInputs:
    """Immutable typed transport for one workspace binding.

    The tensor references are workspace views protected by ``_lease``.  They
    are read inputs only: positions, query layout, slot mappings, Target state,
    and the canonical FullSource owner are intentionally absent.
    """

    identity: DraftWindowGenerationIdentity
    window_block_table: torch.Tensor
    valid_num_blocks: torch.Tensor
    device_window_exact: torch.Tensor
    cpu_window_exact: torch.Tensor
    cpu_window_upper: torch.Tensor
    active_table_columns: tuple[int, ...]
    actual_request_count: int
    padded_request_count: int
    table_capacity_columns: int
    _lease: DraftWindowLease = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.actual_request_count != self.identity.actual_request_count:
            raise ValueError("typed inputs actual request count disagrees with identity")
        if self.padded_request_count != self.identity.padded_request_count:
            raise ValueError("typed inputs padded request count disagrees with identity")
        if len(self.active_table_columns) != self.actual_request_count:
            raise ValueError("active table-column counts must match actual requests")
        if self.table_capacity_columns <= 0:
            raise ValueError("typed inputs table capacity must be greater than 0")
        if any(columns < 0 or columns > self.table_capacity_columns for columns in self.active_table_columns):
            raise ValueError("active table-column counts exceed typed input capacity")
        if self.window_block_table.shape != (
            self.padded_request_count,
            self.table_capacity_columns,
        ):
            raise ValueError("Window table shape does not match padded rows and capacity")
        for name, tensor in (
            ("valid counts", self.valid_num_blocks),
            ("device Window exact", self.device_window_exact),
            ("CPU Window exact", self.cpu_window_exact),
            ("CPU Window upper", self.cpu_window_upper),
        ):
            if tensor.ndim != 1 or tensor.shape[0] != self.padded_request_count:
                raise ValueError(f"{name} shape does not match padded request count")
        if self.valid_num_blocks.dtype != torch.int32:
            raise TypeError("valid counts must use int32")
        if self.window_block_table.dtype != torch.int32:
            raise TypeError("Window block table must use int32")
        if self.valid_num_blocks.device != self.window_block_table.device:
            raise ValueError("valid counts and Window block table must share a device")
        if self.device_window_exact.device != self.window_block_table.device:
            raise ValueError("device Window exact and Window block table must share a device")
        for name, tensor in (
            ("device Window exact", self.device_window_exact),
            ("CPU Window exact", self.cpu_window_exact),
            ("CPU Window upper", self.cpu_window_upper),
        ):
            if tensor.dtype != torch.int32:
                raise TypeError(f"{name} must use int32")
        if self.cpu_window_exact.device.type != "cpu" or self.cpu_window_upper.device.type != "cpu":
            raise ValueError("CPU Window lengths must live on CPU")
        if self.actual_request_count and bool(
            torch.any(
                self.cpu_window_upper[: self.actual_request_count] < self.cpu_window_exact[: self.actual_request_count]
            )
        ):
            raise ValueError("CPU Window upper must be greater than or equal to exact")
        if self.padded_request_count > self.actual_request_count:
            tail = slice(self.actual_request_count, self.padded_request_count)
            inert_tensors = [
                self.cpu_window_exact[tail],
                self.cpu_window_upper[tail],
            ]
            # CPU tests retain strict device-value coverage. Production NPU
            # inputs rely on workspace zero-before-write and must not insert a
            # device reduction/synchronization into every proposal.
            if self.window_block_table.device.type == "cpu":
                inert_tensors.extend(
                    (
                        self.window_block_table[tail],
                        self.valid_num_blocks[tail],
                        self.device_window_exact[tail],
                    )
                )
            if any(bool(tensor.count_nonzero()) for tensor in inert_tensors):
                raise ValueError("padded Draft metadata rows must be deterministically zero")

    @property
    def lease(self) -> DraftWindowLease:
        return self._lease

    def validate_for(self, expected_identity: DraftWindowGenerationIdentity) -> None:
        if self.identity != expected_identity:
            raise RuntimeError("typed inputs generation identity does not match the expected generation")
        self._lease.assert_current(expected_identity)


@dataclass(frozen=True, slots=True, eq=False)
class _DraftMetadataStoreEntry:
    inputs: DraftMetadataInputs
    metadata: Any


class DraftMetadataStore:
    """Speculator-owned Draft-only metadata sink for one active generation."""

    def __init__(self) -> None:
        self._entry: _DraftMetadataStoreEntry | None = None

    def publish(
        self,
        inputs: DraftMetadataInputs,
        metadata: Any,
        *,
        target_metadata: Any,
    ) -> None:
        inputs.validate_for(inputs.identity)
        if self._entry is not None:
            raise RuntimeError("cannot overwrite active draft metadata")
        validate_draft_metadata_isolation(
            metadata,
            target_metadata=target_metadata,
        )
        self._entry = _DraftMetadataStoreEntry(inputs=inputs, metadata=metadata)

    def read(self, expected_identity: DraftWindowGenerationIdentity) -> Any:
        if self._entry is None:
            raise RuntimeError("no active draft metadata")
        self._entry.inputs.validate_for(expected_identity)
        return self._entry.metadata

    def discard(self, expected_identity: DraftWindowGenerationIdentity) -> None:
        if self._entry is None:
            return
        self._entry.inputs.validate_for(expected_identity)
        self._entry = None

    def discard_for_abort(self, identity: DraftWindowGenerationIdentity) -> None:
        if self._entry is None:
            return
        if self._entry.inputs.identity != identity:
            raise RuntimeError("active draft metadata belongs to a different generation")
        self._entry = None


class DraftWindowRuntimeState:
    """The single speculator-owned P3 lifecycle and retirement owner."""

    PROVIDER_VERSION = 1

    def __init__(
        self,
        *,
        capability: DraftWindowSourceCapability,
        workspace: DraftWindowWorkspace,
        execution_lane: int = 0,
        topology_fingerprint: "P7EagerTopologyFingerprint | None" = None,
    ) -> None:
        if execution_lane < 0:
            raise ValueError("execution lane must be nonnegative")
        self.capability = capability
        self.workspace = workspace
        self.execution_lane = execution_lane
        self.topology_fingerprint = (
            capability.eager_topology_fingerprint if topology_fingerprint is None else topology_fingerprint
        )
        self.metadata_store = DraftMetadataStore()
        self._lease_pool = _DraftWindowLeasePool(workspace, execution_lane)
        self._active_inputs: DraftMetadataInputs | None = None
        self._last_mapping_epoch = -1
        self._next_generation_id = 0
        self._last_retired_identity: DraftWindowGenerationIdentity | None = None
        self._pending_retirement_identity: DraftWindowGenerationIdentity | None = None

    def materialize(
        self,
        *,
        mapping_epoch: int,
        ordered_request_ids: Sequence[str],
        idx_mapping: torch.Tensor,
        read_descriptors: Sequence[DraftWindowReadDescriptor],
        device_window_exact: torch.Tensor | None,
        cpu_window_exact: torch.Tensor,
        cpu_window_upper: torch.Tensor,
        cpu_valid_num_blocks: torch.Tensor | None = None,
        num_reqs_padded: int,
        phase: DraftWindowExecutionPhase,
        step: int,
        execution_lane: int,
        row_phase_fingerprint: Sequence[str] = (),
    ) -> DraftMetadataInputs:
        """Acquire, bind, and materialize one generation or cleanly fail.

        ``mapping_epoch`` is the caller's monotonic attestation for the exact
        ``idx_mapping`` contents supplied to this call.  A future live caller
        must advance it whenever that mapping generation changes; a pointer,
        object identity, or unrelated counter is not a valid substitute.
        """
        _require_nonnegative_python_int(mapping_epoch, "mapping epoch")
        if mapping_epoch <= self._last_mapping_epoch:
            raise ValueError(
                f"mapping epoch must strictly increase: current={mapping_epoch} previous={self._last_mapping_epoch}"
            )
        request_fingerprint = tuple(ordered_request_ids)
        descriptor_fingerprint = tuple(read_descriptors)
        actual_request_count = len(request_fingerprint)
        bound_source = self.capability.full_source.bind_batch(idx_mapping)
        identity = DraftWindowGenerationIdentity(
            generation_id=self._next_generation_id,
            mapping_epoch=mapping_epoch,
            ordered_request_fingerprint=request_fingerprint,
            actual_request_count=actual_request_count,
            padded_request_count=num_reqs_padded,
            normalized_method=self.capability.inventory.method,
            global_group_id=self.capability.provider.global_group_id,
            read_descriptor_fingerprint=descriptor_fingerprint,
            provider_id=id(self.capability.provider),
            provider_version=self.PROVIDER_VERSION,
            source_identity=bound_source.source_identity,
            execution_lane=execution_lane,
            phase=phase,
            step=step,
            topology_fingerprint=self.topology_fingerprint,
            row_phase_fingerprint=tuple(row_phase_fingerprint),
        )
        self._last_retired_identity = None
        return self._materialize_identity(
            identity=identity,
            bound_source=bound_source,
            read_descriptors=read_descriptors,
            device_window_exact=device_window_exact,
            cpu_window_exact=cpu_window_exact,
            cpu_window_upper=cpu_window_upper,
            cpu_valid_num_blocks=cpu_valid_num_blocks,
        )

    def materialize_serialized_step(
        self,
        *,
        previous_identity: DraftWindowGenerationIdentity,
        ordered_request_ids: Sequence[str],
        idx_mapping: torch.Tensor,
        read_descriptors: Sequence[DraftWindowReadDescriptor],
        device_window_exact: torch.Tensor | None,
        cpu_window_exact: torch.Tensor,
        cpu_window_upper: torch.Tensor,
        cpu_valid_num_blocks: torch.Tensor | None = None,
        num_reqs_padded: int,
        step: int,
        execution_lane: int,
    ) -> DraftMetadataInputs:
        """Materialize the next serialized AR step under one mapping epoch.

        The ordinary :meth:`materialize` contract remains strictly monotonic:
        only this continuation API may reuse a mapping epoch. It requires the
        immediately preceding generation to have retired successfully, keeps
        request/source/provider identity fixed, and advances the step by one.
        This models several Eagle3 forwards within one proposal without lying
        that the batch mapping changed between those forwards.
        """
        self._validate_topology_fingerprint(previous_identity)
        self._reclaim_completed_reader()
        if self._active_inputs is not None:
            raise RuntimeError("cannot refresh a Draft Window while the previous reader is active")
        if self._last_retired_identity != previous_identity:
            raise RuntimeError("serialized step does not continue the most recently retired generation")
        if previous_identity.mapping_epoch != self._last_mapping_epoch:
            raise RuntimeError("serialized step mapping epoch is not the active proposal epoch")
        if previous_identity.normalized_method != "eagle3":
            raise RuntimeError("serialized per-step continuation is supported only for Eagle3")
        if previous_identity.phase not in (
            DraftWindowExecutionPhase.PROPOSAL,
            DraftWindowExecutionPhase.STEP,
        ):
            raise RuntimeError("serialized step has an invalid predecessor phase")
        if step != previous_identity.step + 1:
            raise RuntimeError(
                "serialized Draft Window steps must strictly advance by one: "
                f"previous={previous_identity.step} current={step}"
            )
        request_fingerprint = tuple(ordered_request_ids)
        if request_fingerprint != previous_identity.ordered_request_fingerprint:
            raise RuntimeError("serialized step request fingerprint changed within one proposal")
        if num_reqs_padded != previous_identity.padded_request_count:
            raise RuntimeError("serialized step padded request count changed within one proposal")
        if execution_lane != previous_identity.execution_lane:
            raise RuntimeError("serialized step execution lane changed within one proposal")

        descriptor_fingerprint = tuple(read_descriptors)
        bound_source = self.capability.full_source.bind_batch(idx_mapping)
        if bound_source.source_identity != previous_identity.source_identity:
            raise RuntimeError("serialized step FullSource identity changed within one proposal")
        identity = DraftWindowGenerationIdentity(
            generation_id=self._next_generation_id,
            mapping_epoch=previous_identity.mapping_epoch,
            ordered_request_fingerprint=request_fingerprint,
            actual_request_count=len(request_fingerprint),
            padded_request_count=num_reqs_padded,
            normalized_method=previous_identity.normalized_method,
            global_group_id=previous_identity.global_group_id,
            read_descriptor_fingerprint=descriptor_fingerprint,
            provider_id=previous_identity.provider_id,
            provider_version=previous_identity.provider_version,
            source_identity=bound_source.source_identity,
            execution_lane=execution_lane,
            phase=DraftWindowExecutionPhase.STEP,
            step=step,
            topology_fingerprint=previous_identity.topology_fingerprint,
            row_phase_fingerprint=previous_identity.row_phase_fingerprint,
        )
        self._last_retired_identity = None
        return self._materialize_identity(
            identity=identity,
            bound_source=bound_source,
            read_descriptors=read_descriptors,
            device_window_exact=device_window_exact,
            cpu_window_exact=cpu_window_exact,
            cpu_window_upper=cpu_window_upper,
            cpu_valid_num_blocks=cpu_valid_num_blocks,
        )

    def _materialize_identity(
        self,
        *,
        identity: DraftWindowGenerationIdentity,
        bound_source: FullDraftKVSource,
        read_descriptors: Sequence[DraftWindowReadDescriptor],
        device_window_exact: torch.Tensor | None,
        cpu_window_exact: torch.Tensor,
        cpu_window_upper: torch.Tensor,
        cpu_valid_num_blocks: torch.Tensor | None,
    ) -> DraftMetadataInputs:
        """Acquire and populate the one serialized backing for an identity."""
        self._validate_topology_fingerprint(identity)
        self._reclaim_completed_reader()
        lease = self._lease_pool.acquire(owner=self, identity=identity)
        self._last_mapping_epoch = max(self._last_mapping_epoch, identity.mapping_epoch)
        self._next_generation_id += 1
        try:
            self.workspace.materialize(
                identity=identity,
                full_source=bound_source,
                read_descriptors=read_descriptors,
                device_window_exact=device_window_exact,
                cpu_window_exact=cpu_window_exact,
                cpu_window_upper=cpu_window_upper,
                cpu_valid_num_blocks=cpu_valid_num_blocks,
            )
            inputs = self.workspace.build_inputs(lease)
            self._active_inputs = inputs
            return inputs
        except BaseException:
            self._lease_pool.retire(lease, identity)
            self._active_inputs = None
            raise

    def publish_metadata(
        self,
        inputs: DraftMetadataInputs,
        metadata: Any,
        *,
        target_metadata: Any,
    ) -> None:
        try:
            self._require_active_inputs(inputs)
            self.metadata_store.publish(
                inputs,
                metadata,
                target_metadata=target_metadata,
            )
        except BaseException:
            if self._active_inputs is inputs:
                self._abort_active(inputs.identity)
            raise

    def read_metadata(self, expected_identity: DraftWindowGenerationIdentity) -> Any:
        return self.metadata_store.read(expected_identity)

    def mark_reader_pending(
        self,
        inputs: DraftMetadataInputs,
        token: DraftWindowCompletionToken,
    ) -> None:
        self._require_active_inputs(inputs)
        self._lease_pool.mark_reader_pending(inputs.lease, token)

    @contextmanager
    def generation_guard(
        self,
        inputs: DraftMetadataInputs,
    ) -> Iterator[DraftMetadataInputs]:
        """Abort this generation on an exception without retiring on success."""
        self._require_active_inputs(inputs)
        try:
            yield inputs
        except BaseException:
            if self._active_inputs is inputs:
                self._abort_active(inputs.identity)
            raise

    def retire_generation(self, expected_identity: DraftWindowGenerationIdentity) -> None:
        inputs = self._active_inputs
        if inputs is None:
            raise RuntimeError("cannot use a retired lease")
        self._retire_lease(inputs.lease, expected_identity)

    def poll_reader_completion(
        self,
        expected_identity: DraftWindowGenerationIdentity,
    ) -> bool:
        """Poll an explicitly retired reader without allowing early reuse."""
        if self._last_retired_identity == expected_identity:
            return True
        inputs = self._active_inputs
        if inputs is not None:
            inputs.validate_for(expected_identity)
            if inputs.lease.state is not DraftWindowLeaseState.READER_PENDING:
                raise RuntimeError("active Draft Window reader has not been submitted")
            self._retire_lease(inputs.lease, expected_identity)
            if self._last_retired_identity == expected_identity:
                return True
            if self._pending_retirement_identity == expected_identity:
                return False
        if self._pending_retirement_identity != expected_identity:
            raise RuntimeError("pending reader belongs to a different generation")
        retired = self._lease_pool.reclaim_completed_reader(expected_identity)
        if retired is None:
            return False
        self._pending_retirement_identity = None
        self._last_retired_identity = retired
        return True

    def abort_generation(self, expected_identity: DraftWindowGenerationIdentity) -> None:
        inputs = self._active_inputs
        if inputs is None:
            raise RuntimeError("cannot use a retired lease")
        inputs.validate_for(expected_identity)
        self._abort_active(expected_identity)

    def _retire_lease(
        self,
        lease: DraftWindowLease,
        expected_identity: DraftWindowGenerationIdentity,
    ) -> None:
        inputs = self._active_inputs
        if inputs is None or inputs.lease is not lease:
            raise RuntimeError("cannot use a retired lease")
        inputs.validate_for(expected_identity)
        self.metadata_store.discard(expected_identity)
        self._lease_pool.retire(lease, expected_identity)
        self._active_inputs = None
        if lease.state is DraftWindowLeaseState.RETIRED:
            self._last_retired_identity = expected_identity
            self._pending_retirement_identity = None
        else:
            self._last_retired_identity = None
            self._pending_retirement_identity = expected_identity

    def _abort_active(self, identity: DraftWindowGenerationIdentity) -> None:
        inputs = self._active_inputs
        if inputs is None:
            return
        inputs.validate_for(identity)
        self.metadata_store.discard_for_abort(identity)
        self._lease_pool.retire(inputs.lease, identity)
        self._active_inputs = None
        self._last_retired_identity = None
        self._pending_retirement_identity = (
            identity if inputs.lease.state is DraftWindowLeaseState.RETIRED_PENDING else None
        )

    def _reclaim_completed_reader(self) -> None:
        pending = self._pending_retirement_identity
        if pending is None:
            return
        retired = self._lease_pool.reclaim_completed_reader(pending)
        if retired is not None:
            self._pending_retirement_identity = None
            self._last_retired_identity = retired

    def _require_active_inputs(self, inputs: DraftMetadataInputs) -> None:
        if self._active_inputs is not inputs:
            raise RuntimeError("typed inputs are not the active runtime generation")
        self._validate_topology_fingerprint(inputs.identity)
        inputs.validate_for(inputs.identity)

    def _validate_topology_fingerprint(
        self,
        identity: DraftWindowGenerationIdentity,
    ) -> None:
        if identity.topology_fingerprint != self.topology_fingerprint:
            raise RuntimeError("generation topology fingerprint does not match the runtime topology fingerprint")


def build_draft_window_runtime_state(
    capability: DraftWindowSourceCapability,
    *,
    execution_lane: int = 0,
    topology_fingerprint: "P7EagerTopologyFingerprint | None" = None,
) -> DraftWindowRuntimeState:
    """Allocate P3 state only after P2 has closed all capability gates."""
    group = capability.provider.source_group
    if len(capability.inventory.groups) != 1 or capability.inventory.groups[0] != group:
        raise RuntimeError("provider source group does not match the capability inventory")
    if group.global_group_id != capability.provider.global_group_id:
        raise RuntimeError("provider gid does not match the inventory candidate")
    capability.full_source.validate_owner_bindings()
    source_table = capability.full_source._input_block_tables[group.global_group_id]
    persistent_counts = capability.full_source._block_tables_owner.num_blocks.gpu[group.global_group_id]
    if source_table.dtype != torch.int32 or persistent_counts.dtype != torch.int32:
        raise TypeError("P3 direct-prefix source tables and valid counts must use int32")
    if source_table.device != persistent_counts.device:
        raise ValueError("P3 source table and valid counts must share a device")
    table_capacity_columns = (
        capability.provider.capacity_columns
        if capability.workspace_capacity_columns is None
        else capability.workspace_capacity_columns
    )
    if table_capacity_columns < capability.provider.capacity_columns:
        raise ValueError("configured workspace capacity is smaller than the provider capacity")
    if table_capacity_columns > source_table.shape[1]:
        raise ValueError("configured workspace capacity exceeds the canonical source row width")
    workspace = DraftWindowWorkspace(
        max_num_reqs=source_table.shape[0],
        table_capacity_columns=table_capacity_columns,
        table_template=source_table,
        valid_count_template=persistent_counts,
    )
    return DraftWindowRuntimeState(
        capability=capability,
        workspace=workspace,
        execution_lane=execution_lane,
        topology_fingerprint=topology_fingerprint,
    )


def _validate_window_length_input(
    tensor: torch.Tensor,
    *,
    name: str,
    padded_request_count: int,
    expected_device: torch.device,
) -> None:
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if tensor.shape[0] < padded_request_count:
        raise ValueError(f"{name} is too small for the padded request count")
    if tensor.dtype != torch.int32:
        raise TypeError(f"{name} must use int32")
    if tensor.device != expected_device:
        raise ValueError(f"{name} must live on {expected_device}")


def _require_nonnegative_python_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be a Python int")
    if value < 0:
        raise ValueError(f"{name} must be greater than or equal to 0")


def validate_draft_metadata_isolation(
    metadata: Any,
    *,
    target_metadata: Any,
    subject: str = "Draft metadata",
) -> None:
    """Reject any mutable object or storage shared with Target metadata.

    P3's store and P6's graph carrier intentionally share this oracle.  The
    latter builds metadata before graph replay and therefore must enforce the
    same nested Tensor/NumPy/object isolation contract without publishing
    through a Target-owned store.
    """
    if target_metadata is None:
        raise ValueError("Target metadata is required for Draft isolation validation")
    shared = _mutable_reference_keys(metadata) & _mutable_reference_keys(target_metadata)
    if shared:
        raise ValueError(f"{subject} mutable storage aliases Target metadata")


def _mutable_reference_keys(value: Any) -> set[tuple[str, Any]]:
    """Collect mutable object/storage identities without retaining Target state."""
    keys: set[tuple[str, Any]] = set()
    seen: set[int] = set()

    def visit(current: Any) -> None:
        if current is None or isinstance(
            current,
            (
                bool,
                int,
                float,
                complex,
                str,
                bytes,
                type,
                Enum,
                torch.dtype,
                torch.device,
            ),
        ):
            return
        current_id = id(current)
        if current_id in seen:
            return
        seen.add(current_id)
        if isinstance(current, torch.Tensor):
            keys.add(("object", current_id))
            if current.untyped_storage().nbytes():
                keys.add(
                    (
                        "tensor_storage",
                        (str(current.device), current.untyped_storage().data_ptr()),
                    )
                )
            return
        if isinstance(current, np.ndarray):
            keys.add(("object", current_id))
            keys.add(("numpy_storage", _numpy_storage_owner_id(current)))
            return
        if isinstance(current, Mapping):
            keys.add(("object", current_id))
            for key, item in current.items():
                visit(key)
                visit(item)
            return
        if isinstance(current, (list, set, bytearray)):
            keys.add(("object", current_id))
            for item in current:
                visit(item)
            return
        if isinstance(current, tuple):
            for item in current:
                visit(item)
            return
        if is_dataclass(current) and not isinstance(current, type):
            keys.add(("object", current_id))
            for data_field in fields(current):
                visit(getattr(current, data_field.name))
            return
        values = getattr(current, "__dict__", None)
        slot_names: list[str] = []
        for owner_type in type(current).__mro__:
            declared_slots = owner_type.__dict__.get("__slots__", ())
            if isinstance(declared_slots, str):
                declared_slots = (declared_slots,)
            if isinstance(declared_slots, (tuple, list)):
                slot_names.extend(
                    name for name in declared_slots if isinstance(name, str) and name not in ("__dict__", "__weakref__")
                )
        if isinstance(values, dict) or slot_names:
            keys.add(("object", current_id))
            if isinstance(values, dict):
                for item in values.values():
                    visit(item)
            for name in slot_names:
                try:
                    visit(getattr(current, name))
                except AttributeError:
                    # Uninitialized slots carry no reachable mutable state.
                    continue

    visit(value)
    return keys


def _numpy_storage_owner_id(array: np.ndarray) -> int:
    """Return a stable in-process identity for NumPy views of one buffer."""
    owner: Any = array
    seen: set[int] = set()
    while id(owner) not in seen:
        seen.add(id(owner))
        if isinstance(owner, np.ndarray) and owner.base is not None:
            owner = owner.base
            continue
        if isinstance(owner, memoryview):
            owner = owner.obj
            continue
        break
    return id(owner)
