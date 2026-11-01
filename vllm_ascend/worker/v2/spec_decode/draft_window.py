# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Canonical Draft KV sources and generation-owned Graph read buffers.

Method policy and interval arithmetic are shared with the ordinary eager
runtime. This module owns only source geometry and persistent Graph backing;
Full write tables and Target state remain owned by the existing runner.
"""

import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import torch
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.ascend_config import (
    validate_additional_config_bool,
)
from vllm_ascend.device.hardware_profile import (
    AttentionBackendFamily,
    get_current_hardware_profile,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_backends import (
    DraftWindowGroupRoute,
    DraftWindowProviderStatus,
    DraftWindowTopology,
    inventory_draft_window_groups,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_plan import (
    DraftWindowPlan,
    compute_draft_window_plan,
    project_draft_window_interval,
    window_capacity_tokens,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_policy import (
    DraftWindowPolicyDecision,
    DraftWindowPolicyState,
)

if TYPE_CHECKING:
    from vllm_ascend.worker.v2.spec_decode.draft_window_eager_capability import (
        P7EagerTopologyFingerprint,
    )
    from vllm_ascend.worker.v2.spec_decode.draft_window_swa_graph import NativeSWAGraphPlan


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
class DraftWindowInventory:
    """Immutable inventory and fail-closed result for one eligible route."""

    method: str
    target_model_type: str
    runtime_route: str
    window_size: int
    groups: tuple[DraftWindowGroupRoute, ...]
    capability_failures: tuple[str, ...]
    dflash2_graph_source: bool = False

    @property
    def consumer_global_group_ids(self) -> tuple[int, ...]:
        """Unique global gids in first-consumer order."""
        return tuple(dict.fromkeys(group.gid for group in self.groups))

    def require_single_group_direct_prefix(self) -> DraftWindowGroupRoute:
        """Select the one already-inventoried consumer admitted by this Graph source."""
        failures = list(self.capability_failures)
        if len(self.groups) != 1:
            failures.append(f"expected one executable consumer, found {len(self.groups)}")
        for group in self.groups:
            native_dflash2 = (
                self.dflash2_graph_source and group.status is DraftWindowProviderStatus.NATIVE_SWA_CANDIDATE
            )
            dense = group.status is DraftWindowProviderStatus.DIRECT_DENSE and group.native_window is None
            if not dense and not native_dflash2:
                failures.append(
                    f"gid={group.gid} backend={group.backend} builder={group.builder} "
                    f"layout={group.status.value} reason={group.reason}"
                )
        if failures:
            raise NotImplementedError(
                "MRV2 draft-window Graph source rejected: "
                f"method={self.method} model={self.target_model_type} route={self.runtime_route}; "
                + "; ".join(failures)
            )
        return self.groups[0]


def _group_diagnostic_context(
    inventory: DraftWindowInventory,
    group: DraftWindowGroupRoute,
) -> str:
    """Return the common identity required on every P2 source failure."""
    return (
        f"method={inventory.method} model={inventory.target_model_type} "
        f"route={inventory.runtime_route} gid={group.gid} "
        f"backend={group.backend} layout={group.status.value}"
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
    # The backend inventory and method contract admit one actual reader;
    # this provider adds only the Graph bank's canonical-source coordinates.
    _group: DraftWindowGroupRoute = field(init=False, repr=False)
    _capacity_columns: int = field(init=False, repr=False)
    _block_span_tokens: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        group = self.inventory.require_single_group_direct_prefix()
        span = group.table_column_span
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
        if group.gid >= self.full_source.num_groups:
            raise RuntimeError(f"inventory gid {group.gid} has no canonical full source")
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
                f"gid={group.gid} span={span} "
                f"legacy_block_size={self.legacy_block_size}"
            ) from error
        object.__setattr__(self, "_group", group)
        object.__setattr__(self, "_capacity_columns", capacity_columns)
        object.__setattr__(self, "_block_span_tokens", span)

    @property
    def global_group_id(self) -> int:
        """Return the sole executable global cache-group id."""
        return self._group.gid

    @property
    def capacity_columns(self) -> int:
        """Return the fixed narrow-table capacity proven by P2 geometry."""
        return self._capacity_columns

    @property
    def block_span_tokens(self) -> int:
        """Return the source-token span represented by one table column."""
        return self._block_span_tokens

    @property
    def source_group(self) -> DraftWindowGroupRoute:
        """Return the exact inventory consumer bound by this provider."""
        return self._group

    def describe(
        self,
        plan: DraftWindowPlan,
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
        expected_start = compute_draft_window_plan(
            full_exact_end=plan.full_exact_end,
            future_offset=future_offset,
            window_size=self.inventory.window_size,
            block_span_tokens=self.legacy_block_size,
        ).window_start
        if plan.window_start != expected_start:
            raise ValueError(
                f"scalar plan does not use the canonical P1 anchor: {plan.window_start} != {expected_start}"
            )
        expected_visible_length = plan.full_exact_end - plan.window_start
        if plan.window_visible_length != expected_visible_length:
            raise ValueError(
                "scalar plan visible length is inconsistent with its exact end: "
                f"{plan.window_visible_length} != {expected_visible_length}"
            )
        expected_capacity = window_capacity_tokens(self.inventory.window_size, self.legacy_block_size)
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
        proposal_plan: DraftWindowPlan,
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
        expected_start = compute_draft_window_plan(
            full_exact_end=proposal_plan.full_exact_end,
            future_offset=proposal_offset,
            window_size=self.inventory.window_size,
            block_span_tokens=self.legacy_block_size,
        ).window_start
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
        expected_capacity = window_capacity_tokens(self.inventory.window_size, self.legacy_block_size)
        if proposal_plan.window_capacity_tokens != expected_capacity:
            raise ValueError(
                "proposal plan capacity does not match provider geometry: "
                f"{proposal_plan.window_capacity_tokens} != {expected_capacity}"
            )
        return self._describe_interval(
            window_start=proposal_plan.window_start,
            window_visible_length=full_exact_end - proposal_plan.window_start,
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
        projection = project_draft_window_interval(
            window_start=window_start,
            full_exact_end=window_start + window_visible_length,
            block_span_tokens=span,
        )
        if projection.first_token_offset != 0:
            raise NotImplementedError(
                "MRV2 direct-prefix provider cannot consume a nonzero first-token "
                f"offset: gid={self._group.gid} "
                f"start={window_start} span={span} "
                f"offset={projection.first_token_offset}"
            )
        if projection.required_columns > self._capacity_columns:
            raise RuntimeError(
                "draft-window projection exceeds provider capacity: "
                f"required={projection.required_columns} "
                f"capacity={self._capacity_columns}"
            )
        source_row_width = self.full_source.input_block_table_row_width(self._group.gid)
        if valid_num_blocks > source_row_width:
            raise RuntimeError(
                "canonical valid block count exceeds source row width: "
                f"gid={self._group.gid} valid={valid_num_blocks} "
                f"row_width={source_row_width}"
            )
        if projection.source_end_column > valid_num_blocks:
            raise RuntimeError(
                "draft-window source shortage: "
                f"method={self.inventory.method} "
                f"model={self.inventory.target_model_type} "
                f"gid={self._group.gid} "
                f"start={projection.source_start_column} "
                f"required={projection.required_columns} "
                f"end={projection.source_end_column} valid={valid_num_blocks}"
            )
        return DraftWindowReadDescriptor(
            global_group_id=self._group.gid,
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
    # Only a reviewed native-SWA/Full Graph route may retain its independent
    # native groups beside the single P3 extra-Window source.
    native_swa_graph: "NativeSWAGraphPlan | None" = None


def _normalize_method(method: str) -> str:
    return method.strip().lower()


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
    capacity_tokens = window_capacity_tokens(window_size, legacy_block_size)
    if legacy_block_size % block_span_tokens != 0:
        raise ValueError("zero-offset direct-prefix capacity requires block span to divide legacy block size")
    return (capacity_tokens + block_span_tokens - 1) // block_span_tokens


def _runtime_capability_failures(
    runtime_context: DraftWindowRuntimeContext,
    *,
    allow_replicated_tp_graph_source: bool = False,
    allow_parallel_prefix_graph_source: bool = False,
    allow_parallel_dp_graph_source: bool = False,
) -> list[str]:
    failures: list[str] = []
    if runtime_context.attention_backend_family is not AttentionBackendFamily.STANDARD:
        failures.append("Draft Window source requires the STANDARD attention backend family")
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
        and not (
            name == "TP"
            and type(size) is int
            and size in (2, 4, 16)
            and allow_replicated_tp_graph_source
            or name == "DP"
            and type(size) is int
            and size == 2
            and runtime_context.tensor_parallel_size == 4
            and allow_replicated_tp_graph_source
            and allow_parallel_dp_graph_source
        )
    )
    if allow_replicated_tp_graph_source and (
        type(runtime_context.tensor_parallel_rank) is not int
        or not 0 <= runtime_context.tensor_parallel_rank < runtime_context.tensor_parallel_size
    ):
        failures.append("gid=none backend=none layout=runtime reason=replicated TP source requires a real integer rank")
    if runtime_context.prefix_caching and not allow_parallel_prefix_graph_source:
        failures.append("gid=none backend=none layout=runtime reason=prefix cache is not verified for Window ON")
    return failures


def build_draft_window_inventory(
    *,
    policy: DraftWindowPolicyDecision,
    runtime_route: str,
    attn_groups: Sequence[Sequence[AttentionGroup]],
    block_tables: Any,
    kv_cache_config: Any,
    causal: bool | Mapping[int, bool],
    vllm_config: Any,
    runtime_context: DraftWindowRuntimeContext,
    allow_replicated_tp_graph_source: bool = False,
    allow_parallel_prefix_graph_source: bool = False,
    allow_parallel_dp_graph_source: bool = False,
    dflash2_graph_contract: Any = None,
) -> DraftWindowInventory:
    """Inventory every executable draft consumer in global-gid order."""
    if policy.state is not DraftWindowPolicyState.ELIGIBLE_ON:
        raise ValueError("P2 inventory requires an ELIGIBLE_ON policy decision")
    if policy.effective_window_size is None:
        raise AssertionError("eligible draft-window policy has no window size")

    runtime_route = _normalize_method(runtime_route)
    native_windows: dict[int, int | None] = {}
    if dflash2_graph_contract is not None:
        from vllm_ascend.worker.v2.spec_decode.draft_window_dflash2_graph import DFlash2GraphContract

        if runtime_route != "dflash2" or not isinstance(dflash2_graph_contract, DFlash2GraphContract):
            raise TypeError("DFlash2 Graph source requires its actual typed method contract")
        native_windows = dflash2_graph_contract.validate_source(
            attn_groups,
            block_tables,
            causal,
            policy.effective_window_size,
            dflash2_graph_contract.query_width,
        )
    if type(allow_parallel_dp_graph_source) is not bool:
        raise TypeError("parallel DP Graph source allowance must be an explicit boolean")
    if allow_parallel_dp_graph_source and (
        runtime_route not in ("dflash", "eagle3")
        or (runtime_route == "eagle3" and (causal is not True or policy.effective_window_size != 1024))
        or not allow_replicated_tp_graph_source
        or runtime_context.tensor_parallel_size != 4
        or runtime_context.data_parallel_size != 2
    ):
        raise NotImplementedError("parallel DP source requires the reviewed rank-local DFlash TP4/DP2 route")
    if type(allow_parallel_prefix_graph_source) is not bool:
        raise TypeError("parallel prefix source allowance must be an explicit boolean")
    eagle_tp4_prefix = (
        runtime_route == "eagle3"
        and allow_replicated_tp_graph_source
        and runtime_context.tensor_parallel_size == 4
        and (runtime_context.data_parallel_size == 1 or allow_parallel_dp_graph_source)
    )
    if (
        allow_parallel_prefix_graph_source
        and runtime_route not in {"dspark", "dflash", "dflash2"}
        and not eagle_tp4_prefix
    ):
        raise NotImplementedError("prefix Graph source requires parallel Draft or the reviewed Eagle3 TP4 owner")
    failures = _runtime_capability_failures(
        runtime_context,
        allow_replicated_tp_graph_source=allow_replicated_tp_graph_source,
        allow_parallel_prefix_graph_source=allow_parallel_prefix_graph_source,
        allow_parallel_dp_graph_source=allow_parallel_dp_graph_source,
    )
    topology = DraftWindowTopology.from_config(
        vllm_config, block_tables, dsa_cp_effective=runtime_context.dsa_context_parallel
    )
    groups = inventory_draft_window_groups(
        attn_groups=attn_groups,
        kv_cache_config=kv_cache_config,
        block_tables=block_tables,
        topology=topology,
        full_retention_dflash2=dflash2_graph_contract is not None,
    )
    if dflash2_graph_contract is not None:
        for group in groups:
            if group.gid not in native_windows or group.native_window != native_windows[group.gid]:
                failures.append(f"gid={group.gid} backend inventory and DFlash2 native reader contract disagree")
    expected_policy_method = "dflash" if runtime_route == "dflash2" else runtime_route
    if expected_policy_method != policy.normalized_method:
        failures.append(f"runtime route {runtime_route!r} does not match policy method {policy.normalized_method!r}")
    if runtime_route == "dflash2" and dflash2_graph_contract is None:
        failures.append("DFlash2 Graph source requires its typed native-reader contract")
    elif runtime_route not in {"eagle3", "dflash", "dflash2", "dspark"}:
        failures.append(f"runtime route {runtime_route!r} is not eligible")
    return DraftWindowInventory(
        method=policy.normalized_method,
        target_model_type=policy.target_model_type,
        runtime_route=runtime_route,
        window_size=policy.effective_window_size,
        groups=groups,
        capability_failures=tuple(failures),
        dflash2_graph_source=dflash2_graph_contract is not None,
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
    group: DraftWindowGroupRoute,
    logical_block_sizes: tuple[int, ...],
    kernel_block_sizes: tuple[int, ...],
    num_groups: int,
) -> None:
    context = _group_diagnostic_context(inventory, group)
    global_group_id = group.gid
    if not (
        0 <= global_group_id < num_groups
        and global_group_id < len(logical_block_sizes)
        and global_group_id < len(kernel_block_sizes)
    ):
        raise RuntimeError(f"{context}; reason=inventory gid has no canonical source geometry")
    source_logical = logical_block_sizes[global_group_id]
    source_kernel = kernel_block_sizes[global_group_id]
    if source_logical != group.scheduler_block_size or source_kernel != group.table_column_span:
        raise RuntimeError(
            f"{context}; reason=source geometry logical={source_logical} "
            f"kernel={source_kernel} does not match inventory "
            f"logical={group.scheduler_block_size} kernel={group.table_column_span}"
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
    # AscendConfig disables FlashComm when SP MoE is unavailable.  Retain the
    # effective executor identity here rather than treating a requested but
    # rejected option as active.  A DSA-CP request auto-enables FlashComm
    # before the model-specific DSA gate is resolved, so that request still
    # contributes when SP itself is effective.
    flashcomm = sequence_parallel and (flashcomm_explicitly_enabled or dsa_context_parallel_requested)
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


def _build_draft_window_source_capability_for_context(
    *,
    policy: DraftWindowPolicyDecision,
    runtime_route: str,
    attn_groups: Sequence[Sequence[AttentionGroup]],
    block_tables: Any,
    kv_cache_config: Any,
    causal: bool | Mapping[int, bool],
    vllm_config: Any,
    runtime_context: DraftWindowRuntimeContext,
    allow_replicated_tp_graph_source: bool = False,
    allow_parallel_prefix_graph_source: bool = False,
    allow_parallel_dp_graph_source: bool = False,
    dflash2_graph_contract: Any = None,
) -> DraftWindowSourceCapability:
    """Build source geometry after the caller validates its real Graph executor.

    Executor features do not turn a dense Draft cache into a sharded provider.
    Keep the actual context in the owner identity and check the source layout
    directly, without constructing an artificial eager capability decision.
    """
    inventory = build_draft_window_inventory(
        policy=policy,
        runtime_route=runtime_route,
        attn_groups=attn_groups,
        block_tables=block_tables,
        kv_cache_config=kv_cache_config,
        causal=causal,
        vllm_config=vllm_config,
        runtime_context=runtime_context,
        allow_replicated_tp_graph_source=allow_replicated_tp_graph_source,
        allow_parallel_prefix_graph_source=allow_parallel_prefix_graph_source,
        allow_parallel_dp_graph_source=allow_parallel_dp_graph_source,
        dflash2_graph_contract=dflash2_graph_contract,
    )
    uses_dynamic_k = bool(vllm_config.speculative_config.uses_dynamic_speculative_decoding())
    if uses_dynamic_k:
        raise NotImplementedError("Draft Window Graph source requires static speculative decoding")
    group = inventory.require_single_group_direct_prefix()
    from vllm_ascend.worker.v2.spec_decode.draft_window_eager_capability import _build_topology_fingerprint

    eager_topology_fingerprint = _build_topology_fingerprint(
        inventory=inventory,
        runtime_context=runtime_context,
        provider_version=DraftWindowRuntimeState.PROVIDER_VERSION,
        execution_lane=0,
    )
    full_source = build_full_draft_kv_source(
        inventory=inventory,
        block_tables=block_tables,
    )
    provider = SingleGroupDirectPrefixProvider(
        inventory=inventory,
        full_source=full_source,
        # MRV1 aligned its anchor to the first executable draft kernel block.
        # With one verified consumer this is the only legacy anchor quantum.
        legacy_block_size=group.table_column_span,
        expected_future_offset=resolve_static_future_offset(
            normalized_method=policy.normalized_method,
            configured_k=int(vllm_config.speculative_config.num_speculative_tokens),
            uses_dynamic_k=uses_dynamic_k,
        ),
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
            group.gid,
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
        self._needs_clear = False

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
        preserve_on_failure: bool = False,
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
            if self._needs_clear:
                self._zero_all()
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
            if not preserve_on_failure:
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

    def reset(self, expected_identity: DraftWindowGenerationIdentity, *, clear_buffers: bool = True) -> None:
        if self._bound_identity is not None:
            self.assert_bound(expected_identity)
        if clear_buffers:
            self._zero_all()
        else:
            self._bound_identity = None
            self._active_table_columns = ()
            self._needs_clear = True

    def _zero_all(self) -> None:
        self.window_block_table.zero_()
        self.valid_num_blocks.zero_()
        self.device_window_exact.zero_()
        self.cpu_window_exact.zero_()
        self.cpu_window_upper.zero_()
        self._active_table_columns = ()
        self._bound_identity = None
        self._needs_clear = False


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
        self._quarantine_on_failure = False

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


class DraftWindowRuntimeState:
    """The single speculator-owned P3 lifecycle and retirement owner."""

    PROVIDER_VERSION = 1

    def __init__(
        self,
        *,
        capability: DraftWindowSourceCapability,
        workspace: DraftWindowWorkspace | None = None,
        execution_lane: int = 0,
        topology_fingerprint: "P7EagerTopologyFingerprint | None" = None,
    ) -> None:
        if execution_lane < 0:
            raise ValueError("execution lane must be nonnegative")
        self.capability = capability
        self._workspace = workspace
        self.execution_lane = execution_lane
        self.topology_fingerprint = (
            capability.eager_topology_fingerprint if topology_fingerprint is None else topology_fingerprint
        )
        self._metadata: Any = None
        self._current_lease: DraftWindowLease | None = None
        self._active_inputs: DraftMetadataInputs | None = None
        self._last_mapping_epoch = -1
        self._last_retired_identity: DraftWindowGenerationIdentity | None = None
        self._pending_retirement_identity: DraftWindowGenerationIdentity | None = None
        self._quarantined_identity: DraftWindowGenerationIdentity | None = None

    @property
    def workspace(self) -> DraftWindowWorkspace:
        # Non-FULL consumers need a temporary read buffer; FULL Graph writes
        # directly to its capture bank and never allocates this scratch.
        if self._workspace is None:
            self._workspace = self.new_workspace()
        return self._workspace

    def new_workspace(self, num_reqs: int | None = None) -> DraftWindowWorkspace:
        capability = self.capability
        gid = capability.provider.global_group_id
        table = capability.full_source._input_block_tables[gid]
        return DraftWindowWorkspace(
            max_num_reqs=table.shape[0] if num_reqs is None else num_reqs,
            table_capacity_columns=capability.workspace_capacity_columns or capability.provider.capacity_columns,
            table_template=table,
            valid_count_template=capability.full_source._block_tables_owner.num_blocks.gpu[gid],
        )

    @property
    def quarantined_identity(self) -> DraftWindowGenerationIdentity | None:
        return self._quarantined_identity

    def quarantine_generation(self, expected_identity: DraftWindowGenerationIdentity) -> None:
        """Keep possibly submitted Draft backing intact until process isolation.

        Unknown native partial submission cannot justify another device write
        or a completion claim. The live lease intentionally blocks reuse.
        """
        inputs = self._active_inputs
        if inputs is None:
            raise RuntimeError("cannot quarantine a generation without live inputs")
        inputs.validate_for(expected_identity)
        self._metadata = None
        self._quarantined_identity = expected_identity

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
        quarantine_on_failure: bool = False,
        output_workspace: DraftWindowWorkspace | None = None,
    ) -> DraftMetadataInputs:
        """Acquire, bind, and materialize one generation or cleanly fail.

        ``mapping_epoch`` is the caller's monotonic attestation for the exact
        ``idx_mapping`` contents supplied to this call.  A future live caller
        must advance it whenever that mapping generation changes; a pointer,
        object identity, or unrelated counter is not a valid substitute.

        The opt-in Graph and async eager callers request
        ``quarantine_on_failure`` because a native exception cannot attest that
        no device work was submitted. Other callers retain the default cleanup
        contract until they establish an explicit native submission owner.
        """
        if self._quarantined_identity is not None:
            raise RuntimeError("draft-window backing is quarantined after uncertain Graph submission")
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
            quarantine_on_failure=quarantine_on_failure,
            output_workspace=output_workspace,
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
        quarantine_on_failure: bool = False,
        output_workspace: DraftWindowWorkspace | None = None,
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
        # Only logical requests/source/epoch are fixed across AR consumers.
        # Each consumer keeps its own padding in its generation identity.
        # The active-reader and last-retired guards above still govern scratch
        # reuse; Graph hands scratch to distinct descriptor/ordinal banks.
        if type(num_reqs_padded) is not int or num_reqs_padded < len(request_fingerprint):
            raise ValueError("serialized step padding must be an integer covering active requests")
        if num_reqs_padded > (self.workspace if output_workspace is None else output_workspace).max_num_reqs:
            raise RuntimeError("serialized step padded request capacity exceeded")
        if execution_lane != previous_identity.execution_lane:
            raise RuntimeError("serialized step execution lane changed within one proposal")

        descriptor_fingerprint = tuple(read_descriptors)
        bound_source = self.capability.full_source.bind_batch(idx_mapping)
        if bound_source.source_identity != previous_identity.source_identity:
            raise RuntimeError("serialized step FullSource identity changed within one proposal")
        identity = DraftWindowGenerationIdentity(
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
            quarantine_on_failure=quarantine_on_failure,
            output_workspace=output_workspace,
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
        quarantine_on_failure: bool = False,
        output_workspace: DraftWindowWorkspace | None = None,
    ) -> DraftMetadataInputs:
        """Acquire and populate the one serialized backing for an identity."""
        self._validate_topology_fingerprint(identity)
        self._reclaim_completed_reader()
        if self._current_lease is not None:
            raise RuntimeError(f"draft-window execution lane {self.execution_lane} is still in use")
        if identity.execution_lane != self.execution_lane:
            raise RuntimeError("draft-window generation lane does not match its owner")
        workspace = self.workspace if output_workspace is None else output_workspace
        lease = DraftWindowLease(owner=self, identity=identity, workspace=workspace)
        self._current_lease = lease
        lease._quarantine_on_failure = quarantine_on_failure
        self._last_mapping_epoch = max(self._last_mapping_epoch, identity.mapping_epoch)
        try:
            workspace.materialize(
                identity=identity,
                full_source=bound_source,
                read_descriptors=read_descriptors,
                device_window_exact=device_window_exact,
                cpu_window_exact=cpu_window_exact,
                cpu_window_upper=cpu_window_upper,
                cpu_valid_num_blocks=cpu_valid_num_blocks,
                preserve_on_failure=quarantine_on_failure,
            )
            inputs = workspace.build_inputs(lease)
            self._active_inputs = inputs
            return inputs
        except BaseException:
            if quarantine_on_failure:
                # A native write may have been submitted before raising, even
                # though no typed inputs exist yet. Keep the acquired lease and
                # backing; only Host state may be changed before child isolation.
                self._quarantined_identity = identity
                raise
            self._finish_lease(lease)
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
            if self._metadata is not None:
                raise RuntimeError("cannot overwrite active draft metadata")
            from vllm_ascend.worker.v2.spec_decode.draft_window_graph_metadata import validate_draft_metadata_isolation

            validate_draft_metadata_isolation(metadata, target_metadata=target_metadata)
            self._metadata = metadata
        except BaseException:
            if self._active_inputs is inputs:
                self._handle_active_failure(inputs)
            raise

    def read_metadata(self, expected_identity: DraftWindowGenerationIdentity) -> Any:
        if self._active_inputs is None or self._metadata is None:
            raise RuntimeError("no active draft metadata")
        self._active_inputs.validate_for(expected_identity)
        return self._metadata

    def mark_reader_pending(self, inputs: DraftMetadataInputs, token: DraftWindowCompletionToken) -> None:
        self._require_active_inputs(inputs)
        lease = inputs.lease
        if lease.state is not DraftWindowLeaseState.ACQUIRED:
            raise RuntimeError("a completion token is already bound to this lease")
        if not callable(getattr(token, "is_complete", None)):
            raise TypeError("completion token must implement is_complete()")
        lease._completion_token = token
        lease._state = DraftWindowLeaseState.READER_PENDING

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
                self._handle_active_failure(inputs)
            raise

    def _handle_active_failure(self, inputs: DraftMetadataInputs) -> None:
        if inputs.lease._quarantine_on_failure:
            self.quarantine_generation(inputs.identity)
        else:
            self._abort_active(inputs.identity)

    def retire_generation(self, expected_identity: DraftWindowGenerationIdentity) -> None:
        if self._quarantined_identity is not None:
            raise RuntimeError("cannot retire quarantined draft-window backing")
        inputs = self._active_inputs
        if inputs is None:
            raise RuntimeError("cannot use a retired lease")
        self._retire_lease(inputs.lease, expected_identity)

    def poll_reader_completion(
        self,
        expected_identity: DraftWindowGenerationIdentity,
    ) -> bool:
        """Poll an explicitly retired reader without allowing early reuse."""
        if self._quarantined_identity is not None:
            raise RuntimeError("cannot poll quarantined draft-window backing")
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
        retired = self._reclaim_lease(expected_identity)
        if retired is None:
            return False
        self._pending_retirement_identity = None
        self._last_retired_identity = retired
        return True

    def abort_generation(self, expected_identity: DraftWindowGenerationIdentity) -> None:
        if self._quarantined_identity is not None:
            raise RuntimeError("cannot abort quarantined draft-window backing")
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
        if self._quarantined_identity is not None:
            raise RuntimeError("cannot retire quarantined draft-window backing")
        inputs = self._active_inputs
        if inputs is None or inputs.lease is not lease:
            raise RuntimeError("cannot use a retired lease")
        inputs.validate_for(expected_identity)
        self._metadata = None
        self._release_lease(lease)
        self._active_inputs = None
        if lease.state is DraftWindowLeaseState.RETIRED:
            self._last_retired_identity = expected_identity
            self._pending_retirement_identity = None
        else:
            self._last_retired_identity = None
            self._pending_retirement_identity = expected_identity

    def _abort_active(self, identity: DraftWindowGenerationIdentity) -> None:
        if self._quarantined_identity is not None:
            raise RuntimeError("cannot clean up quarantined draft-window backing")
        inputs = self._active_inputs
        if inputs is None:
            return
        inputs.validate_for(identity)
        self._metadata = None
        self._release_lease(inputs.lease)
        self._active_inputs = None
        self._last_retired_identity = None
        self._pending_retirement_identity = (
            identity if inputs.lease.state is DraftWindowLeaseState.RETIRED_PENDING else None
        )

    def _reclaim_completed_reader(self) -> None:
        pending = self._pending_retirement_identity
        if pending is None:
            return
        retired = self._reclaim_lease(pending)
        if retired is not None:
            self._pending_retirement_identity = None
            self._last_retired_identity = retired

    def _finish_lease(self, lease: DraftWindowLease) -> None:
        # A Graph bank outlives this Host generation. Never enqueue a clear
        # at retirement; its next same-stream prepare clears it before writing.
        lease.workspace.reset(lease.identity, clear_buffers=lease.workspace is self._workspace)
        lease._state = DraftWindowLeaseState.RETIRED
        self._current_lease = None

    def _release_lease(self, lease: DraftWindowLease) -> None:
        if self._current_lease is not lease:
            raise RuntimeError("cannot use a retired lease")
        token = lease._completion_token
        if token is None or bool(token.is_complete()):
            self._finish_lease(lease)
        else:
            lease._state = DraftWindowLeaseState.RETIRED_PENDING

    def _reclaim_lease(self, identity: DraftWindowGenerationIdentity) -> DraftWindowGenerationIdentity | None:
        lease = self._current_lease
        if lease is None or lease.identity != identity or lease.state is not DraftWindowLeaseState.RETIRED_PENDING:
            raise RuntimeError("pending reader belongs to a different generation")
        assert lease._completion_token is not None
        if not bool(lease._completion_token.is_complete()):
            return None
        self._finish_lease(lease)
        return identity

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
    if group.gid != capability.provider.global_group_id:
        raise RuntimeError("provider gid does not match the inventory candidate")
    capability.full_source.validate_owner_bindings()
    source_table = capability.full_source._input_block_tables[group.gid]
    persistent_counts = capability.full_source._block_tables_owner.num_blocks.gpu[group.gid]
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
    return DraftWindowRuntimeState(
        capability=capability,
        workspace=None,
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
