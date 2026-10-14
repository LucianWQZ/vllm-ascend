# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""P7-E capability decisions for MRV2 Draft Window eager routes.

This module does not enable a backend or topology.  It turns the exact P2
inventory and runtime context into one auditable state.  Only
``READY_FOR_NPU`` is allowed to enter the existing live eager data plane;
``STATIC_READY`` means that a host contract exists but the production route
must remain fail closed until its device/lifetime evidence is available.
"""

from dataclasses import dataclass, replace
from enum import Enum

from vllm_ascend.worker.v2.spec_decode.draft_window import (
    DraftWindowInventory,
    DraftWindowLayoutCategory,
    DraftWindowPolicyDecision,
    DraftWindowPolicyState,
    DraftWindowRuntimeContext,
)


class P7EagerCapabilityStatus(Enum):
    """The only states exposed by the P7-E capability inventory."""

    CONFIG_OFF = "CONFIG_OFF"
    FORCED_OFF = "FORCED_OFF"
    OWNER_BLOCKED = "OWNER_BLOCKED"
    FAIL_CLOSED = "FAIL_CLOSED"
    STATIC_READY = "STATIC_READY"
    READY_FOR_NPU = "READY_FOR_NPU"


@dataclass(frozen=True, slots=True)
class P7EagerTopologyFingerprint:
    """Immutable route/topology identity for an eager Draft read carrier."""

    normalized_method: str
    runtime_route: str
    backend_impls: tuple[str, ...]
    builder_impls: tuple[str, ...]
    global_group_ids: tuple[int, ...]
    layouts: tuple[str, ...]
    block_geometry: tuple[tuple[int, int, int], ...]
    tensor_parallel_size: int
    data_parallel_size: int
    pipeline_parallel_size: int
    prefill_context_parallel_size: int
    decode_context_parallel_size: int
    sequence_parallel: bool
    dsa_context_parallel: bool
    flashcomm: bool
    graph_mode: str
    async_scheduling: bool
    prefix_caching: bool
    provider_version: int
    execution_lane: int

    def __post_init__(self) -> None:
        if not self.normalized_method or self.normalized_method != self.normalized_method.strip().lower():
            raise ValueError("topology method must be a non-empty normalized name")
        if not self.runtime_route or self.runtime_route != self.runtime_route.strip().lower():
            raise ValueError("topology runtime route must be a non-empty normalized name")
        if self.provider_version <= 0:
            raise ValueError("topology provider version must be greater than zero")
        if self.execution_lane < 0:
            raise ValueError("topology execution lane must be nonnegative")
        topology_sizes = (
            self.tensor_parallel_size,
            self.data_parallel_size,
            self.pipeline_parallel_size,
            self.prefill_context_parallel_size,
            self.decode_context_parallel_size,
        )
        if any(size <= 0 for size in topology_sizes):
            raise ValueError("topology world sizes must be greater than zero")
        group_count = len(self.global_group_ids)
        if not (
            len(self.backend_impls)
            == len(self.builder_impls)
            == len(self.layouts)
            == len(self.block_geometry)
            == group_count
        ):
            raise ValueError("topology group fingerprints must have equal lengths")
        if any(gid < 0 for gid in self.global_group_ids):
            raise ValueError("topology global gids must be nonnegative")

    def for_execution_lane(
        self,
        execution_lane: int,
        *,
        async_scheduling: bool | None = None,
    ) -> "P7EagerTopologyFingerprint":
        """Return an explicit lane variant; never mutate the route identity."""
        if execution_lane < 0:
            raise ValueError("topology execution lane must be nonnegative")
        if async_scheduling is None:
            return replace(self, execution_lane=execution_lane)
        return replace(
            self,
            execution_lane=execution_lane,
            async_scheduling=async_scheduling,
        )


@dataclass(frozen=True, slots=True)
class P7EagerCapabilityDecision:
    """One final local capability classification for a configured route."""

    status: P7EagerCapabilityStatus
    reason: str
    topology_fingerprint: P7EagerTopologyFingerprint | None

    def require_live_materialization(self) -> P7EagerTopologyFingerprint:
        """Allow only an exact eager candidate to allocate P2/P3 state."""
        if self.status is not P7EagerCapabilityStatus.READY_FOR_NPU:
            raise NotImplementedError(
                "MRV2 Draft Window P7-E route rejected before materialization: "
                f"status={self.status.value}; {self.reason}"
            )
        if self.topology_fingerprint is None:
            raise RuntimeError("READY_FOR_NPU decision has no topology fingerprint")
        return self.topology_fingerprint


@dataclass(frozen=True, slots=True)
class DCPWindowIntervalProjection:
    """Rank-local projection of one absolute global Draft read interval."""

    global_start: int
    global_end: int
    local_start: int
    local_end: int
    local_visible_length: int
    dcp_world_size: int
    dcp_rank: int
    interleave_size: int


def _qualified_type_name(value: type[object]) -> str:
    return f"{value.__module__}.{value.__qualname__}"


def _build_topology_fingerprint(
    *,
    inventory: DraftWindowInventory,
    runtime_context: DraftWindowRuntimeContext,
    provider_version: int,
    execution_lane: int,
) -> P7EagerTopologyFingerprint:
    groups = inventory.groups
    return P7EagerTopologyFingerprint(
        normalized_method=inventory.method,
        runtime_route=inventory.runtime_route,
        backend_impls=tuple(_qualified_type_name(group.backend_class) for group in groups),
        builder_impls=tuple(_qualified_type_name(group.builder_class) for group in groups),
        global_group_ids=tuple(group.global_group_id for group in groups),
        layouts=tuple(group.layout_category.value for group in groups),
        block_geometry=tuple(
            (
                group.global_group_id,
                group.logical_block_size,
                group.kernel_block_size,
            )
            for group in groups
        ),
        tensor_parallel_size=runtime_context.tensor_parallel_size,
        data_parallel_size=runtime_context.data_parallel_size,
        pipeline_parallel_size=runtime_context.pipeline_parallel_size,
        prefill_context_parallel_size=runtime_context.prefill_context_parallel_size,
        decode_context_parallel_size=runtime_context.decode_context_parallel_size,
        sequence_parallel=runtime_context.sequence_parallel,
        dsa_context_parallel=runtime_context.dsa_context_parallel,
        flashcomm=runtime_context.flashcomm,
        graph_mode=runtime_context.graph_mode,
        async_scheduling=runtime_context.async_scheduling,
        prefix_caching=runtime_context.prefix_caching,
        provider_version=provider_version,
        execution_lane=execution_lane,
    )


def _decision(
    status: P7EagerCapabilityStatus,
    reason: str,
    fingerprint: P7EagerTopologyFingerprint | None,
) -> P7EagerCapabilityDecision:
    return P7EagerCapabilityDecision(
        status=status,
        reason=reason,
        topology_fingerprint=fingerprint,
    )


def _is_heterogeneous(inventory: DraftWindowInventory) -> bool:
    signatures = {
        (
            group.backend_class,
            group.builder_class,
            group.layout_category,
            group.logical_block_size,
            group.kernel_block_size,
        )
        for group in inventory.groups
    }
    return len(signatures) > 1


def evaluate_draft_window_eager_capability(
    *,
    policy: DraftWindowPolicyDecision,
    inventory: DraftWindowInventory | None,
    runtime_context: DraftWindowRuntimeContext | None,
    uses_dynamic_k: bool,
    provider_version: int,
    execution_lane: int,
) -> P7EagerCapabilityDecision:
    """Classify a route without allocating FullSource, workspace, or a lease."""
    if policy.state is DraftWindowPolicyState.CONFIG_OFF:
        return _decision(P7EagerCapabilityStatus.CONFIG_OFF, "draft_window_size is not configured", None)
    if policy.state is DraftWindowPolicyState.FORCED_OFF:
        return _decision(
            P7EagerCapabilityStatus.FORCED_OFF,
            f"method/model policy forced Window OFF: {policy.reason}",
            None,
        )
    if inventory is None or runtime_context is None:
        raise ValueError("eligible P7-E evaluation requires inventory and runtime context")

    fingerprint = _build_topology_fingerprint(
        inventory=inventory,
        runtime_context=runtime_context,
        provider_version=provider_version,
        execution_lane=execution_lane,
    )
    if uses_dynamic_k:
        return _decision(
            P7EagerCapabilityStatus.OWNER_BLOCKED,
            "Eagle3 dynamic-K has no owner-approved Window contract",
            fingerprint,
        )
    if inventory.runtime_route == "dflash2":
        return _decision(
            P7EagerCapabilityStatus.FAIL_CLOSED,
            "DFlash2 has no verified P7-E metadata consumer contract",
            fingerprint,
        )
    if runtime_context.graph_mode != "NONE":
        return _decision(
            P7EagerCapabilityStatus.FAIL_CLOSED,
            f"Graph mode {runtime_context.graph_mode} belongs to the frozen Graph Track",
            fingerprint,
        )
    if runtime_context.prefix_caching:
        return _decision(
            P7EagerCapabilityStatus.FAIL_CLOSED,
            "prefix cache has no Draft Window read/write isolation contract",
            fingerprint,
        )
    if not inventory.groups:
        return _decision(
            P7EagerCapabilityStatus.FAIL_CLOSED,
            "no executable Draft attention consumer exists",
            fingerprint,
        )

    global_gids = tuple(dict.fromkeys(group.global_group_id for group in inventory.groups))
    if len(global_gids) > 1:
        if _is_heterogeneous(inventory):
            return _decision(
                P7EagerCapabilityStatus.OWNER_BLOCKED,
                "heterogeneous multi-group Window semantics require an owner decision",
                fingerprint,
            )
        return _decision(
            P7EagerCapabilityStatus.FAIL_CLOSED,
            "homogeneous multi-group lacks verified same-generation per-gid projection and consumers",
            fingerprint,
        )

    complex_layouts = {
        DraftWindowLayoutCategory.DSA,
        DraftWindowLayoutCategory.SFA,
        DraftWindowLayoutCategory.MLA,
        DraftWindowLayoutCategory.NATIVE_OR_RECYCLING,
    }
    if any(group.layout_category in complex_layouts for group in inventory.groups):
        layouts = ",".join(group.layout_category.value for group in inventory.groups)
        return _decision(
            P7EagerCapabilityStatus.FAIL_CLOSED,
            f"backend layout {layouts} has no typed visibility/residency provider for "
            "V_final = V_draft intersection V_native intersection V_resident",
            fingerprint,
        )

    parallel_reasons: list[str] = []
    for name, size in (
        ("TP", runtime_context.tensor_parallel_size),
        ("DP", runtime_context.data_parallel_size),
        ("PP", runtime_context.pipeline_parallel_size),
        ("PCP", runtime_context.prefill_context_parallel_size),
        ("DCP", runtime_context.decode_context_parallel_size),
    ):
        if size != 1:
            parallel_reasons.append(f"{name}={size}")
    if runtime_context.sequence_parallel:
        parallel_reasons.append("SP")
    if runtime_context.dsa_context_parallel:
        parallel_reasons.append("DSA-CP")
    if runtime_context.flashcomm:
        parallel_reasons.append("FlashComm")
    if parallel_reasons:
        return _decision(
            P7EagerCapabilityStatus.FAIL_CLOSED,
            "parallel eager route lacks a verified rank-local read carrier: " + ", ".join(parallel_reasons),
            fingerprint,
        )

    async_failure = "Async scheduling is not verified in P2"
    inventory_failures = tuple(failure for failure in inventory.capability_failures if async_failure not in failure)
    group_failures = tuple(failure for group in inventory.groups for failure in group.capability_failures)
    if inventory_failures or group_failures:
        return _decision(
            P7EagerCapabilityStatus.FAIL_CLOSED,
            "inventory/source contract is not closed: " + "; ".join(inventory_failures + group_failures),
            fingerprint,
        )
    if len(inventory.groups) != 1:
        return _decision(
            P7EagerCapabilityStatus.FAIL_CLOSED,
            f"expected one executable Draft consumer, found {len(inventory.groups)}",
            fingerprint,
        )
    group = inventory.groups[0]
    if group.layout_category is not DraftWindowLayoutCategory.DENSE_GQA_DIRECT_PREFIX:
        return _decision(
            P7EagerCapabilityStatus.FAIL_CLOSED,
            f"layout {group.layout_category.value} is not the dense direct-prefix reference route",
            fingerprint,
        )
    if runtime_context.async_scheduling:
        return _decision(
            P7EagerCapabilityStatus.STATIC_READY,
            "bounded host lifecycle exists, but a real NPU completion token/event and live consumer binding are absent",
            fingerprint,
        )
    return _decision(
        P7EagerCapabilityStatus.READY_FOR_NPU,
        "single-group dense-GQA serialized eager route is ready for exact NPU validation",
        fingerprint,
    )


def _dcp_local_prefix_length(
    length: int,
    *,
    dcp_world_size: int,
    dcp_rank: int,
    interleave_size: int,
) -> int:
    if length < 0:
        raise ValueError("DCP prefix length must be nonnegative")
    base = length // interleave_size // dcp_world_size * interleave_size
    remainder = length - base * dcp_world_size
    rank_remainder = min(max(remainder - dcp_rank * interleave_size, 0), interleave_size)
    return base + rank_remainder


def project_dcp_window_interval(
    *,
    window_start: int,
    window_end: int,
    dcp_world_size: int,
    dcp_rank: int,
    interleave_size: int,
) -> DCPWindowIntervalProjection:
    """Project ``[S, E)`` as ``local(E) - local(S)``, never ``V / DCP``."""
    if window_start < 0 or window_end < window_start:
        raise ValueError("DCP Window interval must satisfy 0 <= start <= end")
    if dcp_world_size <= 0:
        raise ValueError("DCP world size must be greater than zero")
    if dcp_rank < 0 or dcp_rank >= dcp_world_size:
        raise ValueError("DCP rank must be within the DCP world")
    if interleave_size <= 0:
        raise ValueError("DCP interleave size must be greater than zero")
    local_start = _dcp_local_prefix_length(
        window_start,
        dcp_world_size=dcp_world_size,
        dcp_rank=dcp_rank,
        interleave_size=interleave_size,
    )
    local_end = _dcp_local_prefix_length(
        window_end,
        dcp_world_size=dcp_world_size,
        dcp_rank=dcp_rank,
        interleave_size=interleave_size,
    )
    return DCPWindowIntervalProjection(
        global_start=window_start,
        global_end=window_end,
        local_start=local_start,
        local_end=local_end,
        local_visible_length=local_end - local_start,
        dcp_world_size=dcp_world_size,
        dcp_rank=dcp_rank,
        interleave_size=interleave_size,
    )
