# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Canonical Draft KV sources and generation-owned Graph read buffers.

Method policy and interval arithmetic are shared with the ordinary eager
runtime. This module owns only source geometry and persistent Graph backing;
Full write tables and Target state remain owned by the existing runner.
"""

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

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
                failures.append(f"gid={group.gid} layout={group.status.value} reason={group.reason}")
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
        f"layout={group.status.value}"
    )


@dataclass(frozen=True, slots=True)
class FullDraftKVSource:
    """Borrowed, read-only references to canonical full BlockTables state.

    Canonical tables are already gathered into batch order by BlockTables.
    The Window owner reads them without changing Full write state.
    """

    logical_block_sizes: tuple[int, ...]
    kernel_block_sizes: tuple[int, ...]
    _block_tables_owner: Any = field(repr=False, compare=False)
    _input_block_tables: tuple[torch.Tensor, ...] = field(
        repr=False,
        compare=False,
    )
    _persistent_num_blocks_cpu: torch.Tensor = field(repr=False, compare=False)

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
    """Canonical read projection; it never materializes a window table."""

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
    runtime_context: DraftWindowRuntimeContext | None = None
    # Mixed/prefix rows can require a full-resident row in the same stable
    # backing that carries narrow decode rows.
    workspace_capacity_columns: int | None = None


def _read_coordinate_semantics(groups: Sequence[DraftWindowGroupRoute]) -> str:
    """Read coordinates belong to the actual Draft consumer, not Target flags."""
    semantics = {
        "dsa_cp_derived"
        if group.topology.dsa_cp_effective
        else "dcp_interleaved"
        if group.topology.decode_context_parallel_size > 1
        else "replicated"
        for group in groups
    }
    if len(semantics) != 1:
        raise NotImplementedError("Draft Graph source requires one parallel read-coordinate contract")
    return semantics.pop()


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
    topology = DraftWindowTopology.from_config(vllm_config)
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
        _block_tables_owner=block_tables,
        _input_block_tables=input_block_tables,
        _persistent_num_blocks_cpu=persistent_num_blocks_cpu,
    )


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
        runtime_context=runtime_context,
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


@dataclass(frozen=True, slots=True, eq=False)
class DraftWindowGenerationIdentity:
    """Immutable freshness identity for one materialized Draft read state.

    ``mapping_epoch`` belongs to the caller.  Pointer and Python-object
    identity are allocation guards only; neither substitutes for the epoch
    or ordered stable request IDs. Equal fields from another runtime do not
    identify this generation; its owner retains the actual immutable handle.
    """

    mapping_epoch: int
    ordered_request_fingerprint: tuple[str, ...]
    actual_request_count: int
    padded_request_count: int
    read_descriptor_fingerprint: tuple[DraftWindowReadDescriptor, ...]
    idx_mapping_ptr: int
    phase: DraftWindowExecutionPhase
    step: int
    row_phase_fingerprint: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_nonnegative_python_int(self.mapping_epoch, "mapping epoch")
        _require_nonnegative_python_int(self.actual_request_count, "actual request count")
        _require_nonnegative_python_int(self.padded_request_count, "padded request count")
        _require_nonnegative_python_int(self.step, "step")
        if self.actual_request_count != len(self.ordered_request_fingerprint):
            raise ValueError("actual request count must match the ordered request fingerprint")
        if self.padded_request_count < self.actual_request_count:
            raise ValueError("padded request count must be at least the actual request count")
        if len(self.read_descriptor_fingerprint) != self.actual_request_count:
            raise ValueError("read descriptor fingerprint must match the actual request count")
        if any(not isinstance(request_id, str) or not request_id for request_id in self.ordered_request_fingerprint):
            raise ValueError("request fingerprint must contain non-empty stable request IDs")
        if len(set(self.ordered_request_fingerprint)) != len(self.ordered_request_fingerprint):
            raise ValueError("request fingerprint must contain unique stable request IDs")
        if not isinstance(self.phase, DraftWindowExecutionPhase):
            raise TypeError("phase must be a DraftWindowExecutionPhase")
        if self.row_phase_fingerprint and len(self.row_phase_fingerprint) != self.actual_request_count:
            raise ValueError("row phase fingerprint must match the actual request count")


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
        if table_template.ndim != 2 or valid_count_template.ndim != 1:
            raise ValueError("Draft workspace templates require a table and one-dimensional valid counts")
        if table_template.dtype != torch.int32 or valid_count_template.dtype != torch.int32:
            raise TypeError("Draft workspace tables and valid counts must use int32")
        if table_template.device != valid_count_template.device:
            raise ValueError("Draft workspace table and valid-count templates must share a device")
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
        global_group_id: int,
        read_descriptors: Sequence[DraftWindowReadDescriptor],
        cpu_window_exact: torch.Tensor,
        cpu_window_upper: torch.Tensor,
        cpu_valid_num_blocks: torch.Tensor,
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
        _validate_window_length_input(
            cpu_window_exact,
            name="CPU Window exact",
            padded_request_count=padded,
            expected_device=torch.device("cpu"),
        )
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
                    cpu_window_exact,
                    cpu_window_upper,
                    cpu_valid_num_blocks,
                )
                if tensor is not None
            )
            for backing in workspace_tensors
        ):
            raise ValueError("Window length inputs must not alias workspace backing")
        if actual and bool(torch.any(cpu_window_exact[:actual] < 0)):
            raise ValueError("Window exact lengths must be greater than or equal to 0")
        if actual and bool(torch.any(cpu_window_upper[:actual] < cpu_window_exact[:actual])):
            raise ValueError("CPU Window upper must be greater than or equal to exact")

        full_source.validate_owner_bindings()
        full_source._validate_global_group_id(global_group_id)
        source_table = full_source._input_block_tables[global_group_id]
        if source_table.device != self.window_block_table.device:
            raise ValueError("canonical source table and Window workspace must share a device")
        if source_table.dtype != self.window_block_table.dtype:
            raise TypeError("canonical source table and Window workspace must share a dtype")

        try:
            if self._needs_clear:
                self._zero_all()
            if actual:
                self.valid_num_blocks[:actual].copy_(cpu_valid_num_blocks[:actual])
            active_columns: list[int] = []
            for row, descriptor in enumerate(read_descriptors):
                if descriptor.global_group_id != global_group_id:
                    raise ValueError("read descriptor global gid does not match the runtime source")
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
                gathered_valid = int(cpu_valid_num_blocks[row].item())
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
                self.device_window_exact[:actual].copy_(cpu_window_exact[:actual])
                self.cpu_window_exact[:actual].copy_(cpu_window_exact[:actual])
                self.cpu_window_upper[:actual].copy_(cpu_window_upper[:actual])
            self._active_table_columns = tuple(active_columns)
            self._bound_identity = identity
        except BaseException:
            if not preserve_on_failure:
                self._zero_all()
            raise

    def build_inputs(
        self, owner: "DraftWindowRuntimeState", identity: DraftWindowGenerationIdentity
    ) -> "DraftMetadataInputs":
        return DraftMetadataInputs(identity=identity, workspace=self, _owner=owner)

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


@dataclass(frozen=True, slots=True, eq=False)
class DraftMetadataInputs:
    """One generation handle retaining its actual read bank and runtime owner.

    The bank owns all read tensors. A new handle on every materialization
    rejects stale references even when the next generation reuses that bank.
    Positions, query layout, write slots and Target metadata remain separate.
    """

    identity: DraftWindowGenerationIdentity
    workspace: DraftWindowWorkspace = field(repr=False, compare=False)
    _owner: "DraftWindowRuntimeState" = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        self.workspace.assert_bound(self.identity)

    @property
    def retired(self) -> bool:
        return self._owner._active_inputs is not self

    def validate_for(self, expected_identity: DraftWindowGenerationIdentity) -> None:
        if self.identity != expected_identity:
            raise RuntimeError("typed inputs generation identity does not match the expected generation")
        if self.retired:
            raise RuntimeError("cannot use retired Draft inputs")
        self.workspace.assert_bound(expected_identity)


class DraftWindowRuntimeState:
    """The single speculator-owned P3 lifecycle and retirement owner."""

    def __init__(
        self,
        *,
        capability: DraftWindowSourceCapability,
        workspace: DraftWindowWorkspace | None = None,
    ) -> None:
        self.capability = capability
        self._workspace = workspace
        self._source = capability.full_source
        self._metadata: Any = None
        self._current_workspace: DraftWindowWorkspace | None = None
        self._quarantine_on_failure = False
        self._active_inputs: DraftMetadataInputs | None = None
        self._last_mapping_epoch = -1
        self._last_retired_identity: DraftWindowGenerationIdentity | None = None
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
        or a completion claim. The live input handle intentionally blocks reuse.
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
        cpu_window_exact: torch.Tensor,
        cpu_window_upper: torch.Tensor,
        cpu_valid_num_blocks: torch.Tensor,
        num_reqs_padded: int,
        phase: DraftWindowExecutionPhase,
        step: int,
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
        no device work was submitted. The existing manager owns normal Graph
        completion and orders the next bank write on the same stream.
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
        _validate_idx_mapping(idx_mapping)
        bound_source = self.capability.full_source
        identity = DraftWindowGenerationIdentity(
            mapping_epoch=mapping_epoch,
            ordered_request_fingerprint=request_fingerprint,
            actual_request_count=actual_request_count,
            padded_request_count=num_reqs_padded,
            read_descriptor_fingerprint=descriptor_fingerprint,
            idx_mapping_ptr=idx_mapping.data_ptr(),
            phase=phase,
            step=step,
            row_phase_fingerprint=tuple(row_phase_fingerprint),
        )
        self._last_retired_identity = None
        return self._materialize_identity(
            identity=identity,
            bound_source=bound_source,
            read_descriptors=read_descriptors,
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
        cpu_window_exact: torch.Tensor,
        cpu_window_upper: torch.Tensor,
        cpu_valid_num_blocks: torch.Tensor,
        num_reqs_padded: int,
        step: int,
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
        if self._active_inputs is not None:
            raise RuntimeError("cannot refresh a Draft Window while the previous reader is active")
        if self._last_retired_identity != previous_identity:
            raise RuntimeError("serialized step does not continue the most recently retired generation")
        if previous_identity.mapping_epoch != self._last_mapping_epoch:
            raise RuntimeError("serialized step mapping epoch is not the active proposal epoch")
        if self.capability.inventory.method != "eagle3":
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
        # reuse; Graph uses distinct descriptor/ordinal banks.
        if type(num_reqs_padded) is not int or num_reqs_padded < len(request_fingerprint):
            raise ValueError("serialized step padding must be an integer covering active requests")
        if num_reqs_padded > (self.workspace if output_workspace is None else output_workspace).max_num_reqs:
            raise RuntimeError("serialized step padded request capacity exceeded")

        descriptor_fingerprint = tuple(read_descriptors)
        _validate_idx_mapping(idx_mapping)
        bound_source = self.capability.full_source
        if bound_source is not self._source or idx_mapping.data_ptr() != previous_identity.idx_mapping_ptr:
            raise RuntimeError("serialized step FullSource or request mapping changed within one proposal")
        identity = DraftWindowGenerationIdentity(
            mapping_epoch=previous_identity.mapping_epoch,
            ordered_request_fingerprint=request_fingerprint,
            actual_request_count=len(request_fingerprint),
            padded_request_count=num_reqs_padded,
            read_descriptor_fingerprint=descriptor_fingerprint,
            idx_mapping_ptr=idx_mapping.data_ptr(),
            phase=DraftWindowExecutionPhase.STEP,
            step=step,
            row_phase_fingerprint=previous_identity.row_phase_fingerprint,
        )
        self._last_retired_identity = None
        return self._materialize_identity(
            identity=identity,
            bound_source=bound_source,
            read_descriptors=read_descriptors,
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
        cpu_window_exact: torch.Tensor,
        cpu_window_upper: torch.Tensor,
        cpu_valid_num_blocks: torch.Tensor,
        quarantine_on_failure: bool = False,
        output_workspace: DraftWindowWorkspace | None = None,
    ) -> DraftMetadataInputs:
        """Acquire and populate the one serialized backing for an identity."""
        if self._current_workspace is not None:
            raise RuntimeError("draft-window runtime still owns an active read bank")
        if bound_source is not self._source:
            raise RuntimeError("draft-window canonical source changed after runtime initialization")
        workspace = self.workspace if output_workspace is None else output_workspace
        self._current_workspace = workspace
        self._quarantine_on_failure = quarantine_on_failure
        self._last_mapping_epoch = max(self._last_mapping_epoch, identity.mapping_epoch)
        try:
            workspace.materialize(
                identity=identity,
                full_source=bound_source,
                global_group_id=self.capability.provider.global_group_id,
                read_descriptors=read_descriptors,
                cpu_window_exact=cpu_window_exact,
                cpu_window_upper=cpu_window_upper,
                cpu_valid_num_blocks=cpu_valid_num_blocks,
                preserve_on_failure=quarantine_on_failure,
            )
            inputs = workspace.build_inputs(self, identity)
            self._active_inputs = inputs
            return inputs
        except BaseException:
            if quarantine_on_failure:
                # A native write may have been submitted before raising, even
                # though no typed inputs exist yet. Keep the workspace and
                # backing; only Host state may be changed before child isolation.
                self._quarantined_identity = identity
                raise
            self._finish_generation(identity)
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

    def _handle_active_failure(self, inputs: DraftMetadataInputs) -> None:
        if self._quarantine_on_failure:
            self.quarantine_generation(inputs.identity)
        else:
            self._abort_active(inputs.identity)

    def retire_generation(self, expected_identity: DraftWindowGenerationIdentity) -> None:
        if self._quarantined_identity is not None:
            raise RuntimeError("cannot retire quarantined draft-window backing")
        inputs = self._active_inputs
        if inputs is None:
            raise RuntimeError("cannot use retired Draft inputs")
        inputs.validate_for(expected_identity)
        self._finish_generation(expected_identity)
        self._last_retired_identity = expected_identity

    def abort_generation(self, expected_identity: DraftWindowGenerationIdentity) -> None:
        if self._quarantined_identity is not None:
            raise RuntimeError("cannot abort quarantined draft-window backing")
        inputs = self._active_inputs
        if inputs is None:
            raise RuntimeError("cannot use retired Draft inputs")
        inputs.validate_for(expected_identity)
        self._abort_active(expected_identity)

    def _abort_active(self, identity: DraftWindowGenerationIdentity) -> None:
        if self._quarantined_identity is not None:
            raise RuntimeError("cannot clean up quarantined draft-window backing")
        inputs = self._active_inputs
        if inputs is None:
            return
        inputs.validate_for(identity)
        self._finish_generation(identity)
        self._last_retired_identity = None

    def _finish_generation(self, identity: DraftWindowGenerationIdentity) -> None:
        # Graph bank lifetime belongs to capture, not this Host generation.
        # Clear only at the next same-stream prepare, after the existing tail
        # dependency. Non-FULL scratch follows its serialized current stream.
        workspace = self._current_workspace
        assert workspace is not None
        workspace.reset(identity, clear_buffers=workspace is self._workspace)
        self._metadata = None
        self._active_inputs = None
        self._current_workspace = None

    def _require_active_inputs(self, inputs: DraftMetadataInputs) -> None:
        if self._active_inputs is not inputs:
            raise RuntimeError("typed inputs are not the active runtime generation")
        inputs.validate_for(inputs.identity)


def build_draft_window_runtime_state(
    capability: DraftWindowSourceCapability,
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
