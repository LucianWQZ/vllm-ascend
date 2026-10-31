# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""P7-X typed contracts for mandatory non-Graph Draft Window routes.

The module deliberately contains no graph transport and no scheduler mutation.
It converts source-owned batch/topology facts into immutable contracts consumed
by the existing P1-P4 read plane.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch

from vllm_ascend.attention.context_parallel.dsa_cp import (
    AscendDSACPMetadataBuilder,
)
from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder
from vllm_ascend.device.hardware_profile import AttentionBackendFamily
from vllm_ascend.worker.v2.spec_decode.draft_window import (
    DraftWindowInventory,
    DraftWindowLayoutCategory,
    DraftWindowPolicyDecision,
    DraftWindowPolicyState,
    DraftWindowReadDescriptor,
    DraftWindowRuntimeContext,
    FullDraftKVSource,
    SingleGroupDirectPrefixProvider,
    compute_draft_window_scalar_plan,
    gather_valid_num_blocks_cpu,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_eager_capability import (
    P7EagerTopologyFingerprint,
    _build_topology_fingerprint,
)


class P7XCapabilityStatus(Enum):
    """Only statuses permitted by the P7-X capability matrix."""

    CONFIG_OFF = "CONFIG_OFF"
    FORCED_OFF = "FORCED_OFF"
    LOCAL_STATIC_READY = "LOCAL_STATIC_READY"
    REPORT1_RUNTIME_DEPENDENT = "REPORT1_RUNTIME_DEPENDENT"
    SOURCE_BLOCKED = "SOURCE_BLOCKED"
    GRAPH_FROZEN = "GRAPH_FROZEN"


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


class DraftWindowBackendProviderKind(Enum):
    """How a cache group derives its final Draft read visibility."""

    DENSE_DIRECT_PREFIX = "dense_direct_prefix"
    DSA_NATIVE_SWA = "dsa_native_swa"
    DSA_CP_NATIVE_SWA = "dsa_cp_native_swa"
    UNRESOLVED = "unresolved"


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


@dataclass(frozen=True, slots=True)
class P7XCapabilityDecision:
    status: P7XCapabilityStatus
    reason: str
    topology_fingerprint: P7EagerTopologyFingerprint | None
    backend_provider_kinds: tuple[DraftWindowBackendProviderKind, ...] = ()

    def require_local_materialization(self) -> P7EagerTopologyFingerprint:
        if self.status is not P7XCapabilityStatus.LOCAL_STATIC_READY:
            raise NotImplementedError(
                "MRV2 Draft Window P7-X route rejected before materialization: "
                f"status={self.status.value}; {self.reason}"
            )
        if self.topology_fingerprint is None:
            raise RuntimeError("LOCAL_STATIC_READY decision has no topology fingerprint")
        return self.topology_fingerprint


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

    phases: list[DraftWindowRowPhase] = []
    for row in range(num_reqs):
        if not bool(actual_prefill[row]):
            phases.append(DraftWindowRowPhase.PURE_DECODE)
        elif int(computed[row]) + int(scheduled[row]) < int(prefill_len[row]):
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
            scalar = compute_draft_window_scalar_plan(
                full_exact_end=full_exact,
                future_offset=0,
                window_size=provider.inventory.window_size,
                legacy_block_size=provider.legacy_block_size,
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


def _is_subclass(candidate: type[Any], parent: type[Any]) -> bool:
    try:
        return issubclass(candidate, parent)
    except TypeError:
        return False


def _provider_kind(
    group: Any,
    *,
    dsa_context_parallel: bool,
) -> DraftWindowBackendProviderKind:
    layout = group.layout_category
    if layout is DraftWindowLayoutCategory.DENSE_GQA_DIRECT_PREFIX:
        return DraftWindowBackendProviderKind.DENSE_DIRECT_PREFIX
    if layout is DraftWindowLayoutCategory.DSA:
        if dsa_context_parallel and _is_subclass(group.builder_class, AscendDSACPMetadataBuilder):
            return DraftWindowBackendProviderKind.DSA_CP_NATIVE_SWA
        if not dsa_context_parallel and _is_subclass(group.builder_class, AscendDSAMetadataBuilder):
            return DraftWindowBackendProviderKind.DSA_NATIVE_SWA
    return DraftWindowBackendProviderKind.UNRESOLVED


def evaluate_p7x_nongraph_capability(
    *,
    policy: DraftWindowPolicyDecision,
    inventory: DraftWindowInventory | None,
    runtime_context: DraftWindowRuntimeContext | None,
    uses_dynamic_k: bool,
    provider_version: int,
    execution_lane: int,
    allow_parallel_dp_graph_source: bool = False,
) -> P7XCapabilityDecision:
    """Classify the mandatory non-Graph scope without allocating read state."""
    if policy.state is DraftWindowPolicyState.CONFIG_OFF:
        return P7XCapabilityDecision(P7XCapabilityStatus.CONFIG_OFF, "draft_window_size is not configured", None)
    if policy.state is DraftWindowPolicyState.FORCED_OFF:
        return P7XCapabilityDecision(
            P7XCapabilityStatus.FORCED_OFF,
            f"method/model policy forced Window OFF: {policy.reason}",
            None,
        )
    if inventory is None or runtime_context is None:
        raise ValueError("eligible P7-X evaluation requires inventory and runtime context")
    if type(allow_parallel_dp_graph_source) is not bool:
        raise TypeError("parallel DP source allowance must be an explicit boolean")
    if allow_parallel_dp_graph_source and (
        inventory.runtime_route != "dflash"
        or runtime_context.tensor_parallel_size != 4
        or runtime_context.data_parallel_size != 2
    ):
        raise ValueError("parallel DP source allowance is limited to reviewed rank-local DFlash TP4/DP2")
    fingerprint = _build_topology_fingerprint(
        inventory=inventory,
        runtime_context=runtime_context,
        provider_version=provider_version,
        execution_lane=execution_lane,
    )
    kinds = tuple(
        _provider_kind(
            group,
            dsa_context_parallel=runtime_context.dsa_context_parallel,
        )
        for group in inventory.groups
    )
    if runtime_context.graph_mode != "NONE":
        return P7XCapabilityDecision(
            P7XCapabilityStatus.GRAPH_FROZEN,
            f"Graph mode {runtime_context.graph_mode} belongs to the frozen Graph Track",
            fingerprint,
            kinds,
        )
    if runtime_context.attention_backend_family is not AttentionBackendFamily.STANDARD:
        return P7XCapabilityDecision(
            P7XCapabilityStatus.SOURCE_BLOCKED,
            f"hardware family {runtime_context.attention_backend_family.name} has no P7-X carrier",
            fingerprint,
            kinds,
        )
    if uses_dynamic_k:
        return P7XCapabilityDecision(
            P7XCapabilityStatus.SOURCE_BLOCKED,
            "dynamic-K Eagle3 is outside the mandatory static-K contract",
            fingerprint,
            kinds,
        )
    if inventory.runtime_route == "dflash2" and not inventory.dflash2_graph_source:
        return P7XCapabilityDecision(
            P7XCapabilityStatus.SOURCE_BLOCKED,
            "DFlash2 has no verified explicit read consumer",
            fingerprint,
            kinds,
        )
    if runtime_context.async_scheduling and inventory.method == "eagle3":
        return P7XCapabilityDecision(
            P7XCapabilityStatus.REPORT1_RUNTIME_DEPENDENT,
            "Eagle3 multi-step async needs a proposal/step live binding before it may allocate a Window source",
            fingerprint,
            kinds,
        )
    if not inventory.groups:
        return P7XCapabilityDecision(
            P7XCapabilityStatus.SOURCE_BLOCKED,
            "no executable Draft attention consumer exists",
            fingerprint,
            kinds,
        )
    if any(kind is DraftWindowBackendProviderKind.UNRESOLVED for kind in kinds):
        return P7XCapabilityDecision(
            P7XCapabilityStatus.REPORT1_RUNTIME_DEPENDENT,
            "one or more backend layouts need an actual runtime consumer/provider identity",
            fingerprint,
            kinds,
        )
    if any(group.source_geometry_failures for group in inventory.groups):
        failures = tuple(failure for group in inventory.groups for failure in group.source_geometry_failures)
        return P7XCapabilityDecision(
            P7XCapabilityStatus.SOURCE_BLOCKED,
            "source geometry is not safe: " + "; ".join(failures),
            fingerprint,
            kinds,
        )
    dsa_groups = tuple(group for group in inventory.groups if group.layout_category is DraftWindowLayoutCategory.DSA)
    if any(group.drafting_compression_ratio is None for group in dsa_groups):
        return P7XCapabilityDecision(
            P7XCapabilityStatus.REPORT1_RUNTIME_DEPENDENT,
            "DSA builder did not expose an actual drafting compression ratio",
            fingerprint,
            kinds,
        )
    if any((group.drafting_compression_ratio or 0) > 1 for group in dsa_groups):
        return P7XCapabilityDecision(
            P7XCapabilityStatus.SOURCE_BLOCKED,
            "compressed DSA/QLI drafting has no safe Window source contract",
            fingerprint,
            kinds,
        )
    if runtime_context.async_scheduling and dsa_groups:
        return P7XCapabilityDecision(
            P7XCapabilityStatus.REPORT1_RUNTIME_DEPENDENT,
            "DSA async requires lane-owned SAS/local/index buffers before live materialization",
            fingerprint,
            kinds,
        )
    if any(
        group.capability_failures and group.layout_category is DraftWindowLayoutCategory.DENSE_GQA_DIRECT_PREFIX
        for group in inventory.groups
    ):
        return P7XCapabilityDecision(
            P7XCapabilityStatus.SOURCE_BLOCKED,
            "dense provider inventory contains unresolved source/geometry failures",
            fingerprint,
            kinds,
        )
    if len({group.global_group_id for group in inventory.groups}) != len(inventory.groups):
        return P7XCapabilityDecision(
            P7XCapabilityStatus.SOURCE_BLOCKED,
            "multiple executable consumers share a global gid",
            fingerprint,
            kinds,
        )
    if len(inventory.groups) > 1:
        signatures = {
            (group.layout_category, group.logical_block_size, group.kernel_block_size) for group in inventory.groups
        }
        status = P7XCapabilityStatus.REPORT1_RUNTIME_DEPENDENT
        return P7XCapabilityDecision(
            status,
            "homogeneous groups have a source-safe canonical-anchor contract "
            "but the live P3/P4 carrier is not yet bundled"
            if len(signatures) == 1
            else "heterogeneous per-layout consumers require Report1 runtime identity",
            fingerprint,
            kinds,
        )
    unsupported_parallel: list[str] = []
    for name, size in (
        ("DP", runtime_context.data_parallel_size),
        ("PP", runtime_context.pipeline_parallel_size),
        ("PCP", runtime_context.prefill_context_parallel_size),
        ("DCP", runtime_context.decode_context_parallel_size),
    ):
        if size != 1 and not (name == "DP" and allow_parallel_dp_graph_source):
            unsupported_parallel.append(f"{name}={size}")
    if runtime_context.sequence_parallel:
        unsupported_parallel.append("SP")
    if runtime_context.flashcomm and not runtime_context.dsa_context_parallel:
        unsupported_parallel.append("FlashComm")
    if runtime_context.dsa_context_parallel and kinds != (DraftWindowBackendProviderKind.DSA_CP_NATIVE_SWA,):
        unsupported_parallel.append("DSA-CP-without-DSA-provider")
    if unsupported_parallel:
        return P7XCapabilityDecision(
            P7XCapabilityStatus.REPORT1_RUNTIME_DEPENDENT,
            "parallel route requires a runtime-specific carrier: " + ", ".join(unsupported_parallel),
            fingerprint,
            kinds,
        )
    supported_single_kinds = {
        DraftWindowBackendProviderKind.DENSE_DIRECT_PREFIX,
        DraftWindowBackendProviderKind.DSA_NATIVE_SWA,
        DraftWindowBackendProviderKind.DSA_CP_NATIVE_SWA,
    }
    if len(kinds) != 1 or kinds[0] not in supported_single_kinds:
        return P7XCapabilityDecision(
            P7XCapabilityStatus.REPORT1_RUNTIME_DEPENDENT,
            "backend-derived provider is source-specified but not installed in the live P3/P4 carrier",
            fingerprint,
            kinds,
        )
    return P7XCapabilityDecision(
        P7XCapabilityStatus.LOCAL_STATIC_READY,
        "non-Graph route has explicit topology, residency, and backend-provider contracts",
        fingerprint,
        kinds,
    )
