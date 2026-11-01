# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Persistent Draft Graph read banks and runtime binding validation.

Startup capture fixes addresses. Each proposal updates only its private read
banks and metadata; canonical Full write tables, slots, and absolute positions
remain unchanged. The bank owner retains current generation and request
identity through the actual typed Graph UPDATE entrance.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vllm_ascend.worker.v2.spec_decode.draft_window_swa_graph import NativeSWAFullSnapshot

from vllm_ascend.worker.v2.spec_decode.draft_window import (
    DraftMetadataInputs,
    DraftWindowExecutionPhase,
    DraftWindowGenerationIdentity,
    DraftWindowRuntimeState,
)


class DraftWindowGraphRoute(Enum):
    """Method-specific graph source route; never a generic support boolean."""

    EAGLE3_FIRST_PASS = "eagle3_first_pass"
    EAGLE3_AR_STEP = "eagle3_ar_step"
    DFLASH = "dflash"
    DSPARK = "dspark"


class DraftWindowGraphCarrierState(Enum):
    """Lifecycle of one persistent per-consumer carrier backing."""

    EMPTY = "empty"
    BOUND = "bound"
    SUBMITTED = "submitted"
    UPDATED = "updated"


@dataclass(frozen=True, slots=True)
class DraftWindowGraphTensorIdentity:
    """Address/geometry fingerprint; deliberately not a freshness token."""

    storage_data_ptr: int
    data_ptr: int
    storage_offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    device: str


@dataclass(frozen=True, slots=True)
class DraftWindowGraphBackingIdentity:
    """Stable P3 workspace owner and all graph-visible backing addresses."""

    owner_id: int
    window_block_table: DraftWindowGraphTensorIdentity
    valid_num_blocks: DraftWindowGraphTensorIdentity
    device_window_exact: DraftWindowGraphTensorIdentity
    cpu_window_exact: DraftWindowGraphTensorIdentity
    cpu_window_upper: DraftWindowGraphTensorIdentity


@dataclass(frozen=True, slots=True)
class DraftWindowGraphRouteFingerprint:
    """Complete capture/runtime compatibility key for one graph consumer.

    This value deliberately contains no request generation.  It describes the
    graph program, topology, cache geometry, and user-visible feature route
    that must be identical when a later runtime generation binds to a captured
    operand.
    """

    method: str
    route: DraftWindowGraphRoute
    graph_mode: str
    backend_impl: str
    builder_impl: str
    global_group_id: int
    layout: str
    capture_descriptor: str
    capture_bucket: tuple[int, int]
    query_width: int
    padded_request_count: int
    padded_token_count: int
    static_k: int
    window_size: int
    logical_block_size: int
    kernel_block_size: int
    block_span_tokens: int
    tensor_parallel_size: int
    data_parallel_size: int
    pipeline_parallel_size: int
    prefill_context_parallel_size: int
    decode_context_parallel_size: int
    sequence_parallel: bool
    dsa_context_parallel: bool
    flashcomm: bool
    async_scheduling: bool
    prefix_caching: bool
    tensor_parallel_rank: int = 0
    parallel_read_semantics: str = "replicated"

    def __post_init__(self) -> None:
        if not isinstance(self.route, DraftWindowGraphRoute):
            raise TypeError("graph route fingerprint requires a DraftWindowGraphRoute")
        expected_method = _ROUTE_CONTRACTS[self.route][0]
        if self.method != expected_method:
            raise ValueError("graph route fingerprint method does not match its route")
        if self.graph_mode not in {"FULL", "FULL_DECODE_ONLY", "PIECEWISE"}:
            raise ValueError("graph route fingerprint requires an explicit graph mode")
        integer_fields = (
            ("global gid", self.global_group_id, 0),
            ("capture request bucket", self.capture_bucket[0], 1),
            ("capture token bucket", self.capture_bucket[1], 1),
            ("query width", self.query_width, 1),
            ("padded request count", self.padded_request_count, 1),
            ("padded token count", self.padded_token_count, 1),
            ("static K", self.static_k, 1),
            ("window size", self.window_size, 1),
            ("logical block size", self.logical_block_size, 1),
            ("kernel block size", self.kernel_block_size, 1),
            ("block span", self.block_span_tokens, 1),
            ("TP", self.tensor_parallel_size, 1),
            ("TP rank", self.tensor_parallel_rank, 0),
            ("DP", self.data_parallel_size, 1),
            ("PP", self.pipeline_parallel_size, 1),
            ("PCP", self.prefill_context_parallel_size, 1),
            ("DCP", self.decode_context_parallel_size, 1),
        )
        for name, value, minimum in integer_fields:
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"graph route fingerprint {name} is invalid")
        if self.capture_bucket != (
            self.padded_request_count,
            self.padded_token_count,
        ):
            raise ValueError("graph route fingerprint capture bucket does not match padded geometry")
        if not self.capture_descriptor:
            raise ValueError("graph route fingerprint capture descriptor must be non-empty")
        if not self.backend_impl or not self.builder_impl:
            raise ValueError("graph route fingerprint backend and builder must be explicit")
        if self.route in (
            DraftWindowGraphRoute.EAGLE3_FIRST_PASS,
            DraftWindowGraphRoute.DFLASH,
            DraftWindowGraphRoute.DSPARK,
        ):
            expected_tokens = self.padded_request_count * self.query_width
        else:
            if self.query_width != 1:
                raise ValueError("Eagle3 AR graph consumers require query width 1")
            expected_tokens = self.padded_request_count
        if self.padded_token_count != expected_tokens:
            raise ValueError("graph route fingerprint query width does not match padded tokens")
        if self.tensor_parallel_rank >= self.tensor_parallel_size:
            raise ValueError("graph route fingerprint TP rank is outside its world")
        if self.parallel_read_semantics != "replicated":
            raise ValueError("graph carrier currently requires replicated logical read semantics")
        if any(
            size != 1
            for size in (
                self.pipeline_parallel_size,
                self.prefill_context_parallel_size,
                self.decode_context_parallel_size,
            )
        ):
            raise ValueError("Draft Graph read banks require PP=PCP=DCP=1")


@dataclass(frozen=True, slots=True)
class DraftWindowGraphCaptureIdentity:
    """Startup capture identity; never a request-generation surrogate."""

    fingerprint: DraftWindowGraphRouteFingerprint
    execution_lane: int
    consumer_ordinal: int
    provider_schema_version: int
    backing_identity: DraftWindowGraphBackingIdentity


@dataclass(frozen=True, slots=True)
class DraftWindowGraphReadInputs:
    """Only the persistent Draft read carrier presented to a metadata builder."""

    window_block_table: torch.Tensor
    valid_num_blocks: torch.Tensor
    device_window_exact: torch.Tensor
    cpu_window_exact: torch.Tensor
    cpu_window_upper: torch.Tensor
    actual_request_count: int
    padded_request_count: int


@dataclass(frozen=True, slots=True)
class DraftWindowGraphWriteSideReference:
    """Full-context write/query state that Window metadata must preserve.

    This reference is supplied explicitly by the speculator from its stable
    full-state owners. It is separate from Target metadata because Draft and
    Target query geometry can legitimately differ. P6 never derives these
    fields from the Window read carrier.
    """

    slot_mapping: torch.Tensor
    positions: torch.Tensor
    query_start_loc: torch.Tensor
    actual_seq_lengths_q: tuple[int, ...]
    attn_state: Any
    num_actual_tokens: int
    causal: bool

    def __post_init__(self) -> None:
        for name, tensor in (
            ("slot mapping", self.slot_mapping),
            ("positions", self.positions),
            ("query start", self.query_start_loc),
        ):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"graph write-side {name} must be a Tensor")
            if tensor.ndim != 1:
                raise ValueError(f"graph write-side {name} must be one-dimensional")
        if not isinstance(self.num_actual_tokens, int) or isinstance(self.num_actual_tokens, bool):
            raise TypeError("graph write-side actual token count must be a Python int")
        if self.num_actual_tokens < 0:
            raise ValueError("graph write-side actual token count must be non-negative")
        if self.slot_mapping.shape[0] != self.num_actual_tokens:
            raise ValueError("graph write-side slot mapping must match the actual token count")
        if self.positions.shape[0] < self.num_actual_tokens:
            raise ValueError("graph write-side positions do not cover all actual tokens")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in self.actual_seq_lengths_q
        ):
            raise ValueError("graph write-side query lengths must be non-negative Python ints")
        if not isinstance(self.causal, bool):
            raise TypeError("graph write-side causal flag must be bool")


@dataclass(frozen=True, slots=True)
class _DraftWindowGraphWriteSideSnapshot:
    """Clone-free mutation guard for full write/query owners."""

    slot_mapping: DraftWindowGraphTensorIdentity
    positions: DraftWindowGraphTensorIdentity
    query_start_loc: DraftWindowGraphTensorIdentity
    mutation_versions: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class DraftWindowGraphBindingIdentity:
    """Complete generation and route identity for one graph binding."""

    method: str
    route: DraftWindowGraphRoute
    global_group_id: int
    proposal_id: int
    generation_id: int
    mapping_epoch: int
    ordered_request_fingerprint: tuple[str, ...]
    actual_request_count: int
    padded_request_count: int
    provider_id: int
    provider_version: int
    source_owner_id: int
    execution_lane: int
    phase: DraftWindowExecutionPhase
    step: int
    tensor_parallel_size: int
    tensor_parallel_rank: int
    parallel_read_semantics: str

    def __post_init__(self) -> None:
        world, rank = self.tensor_parallel_size, self.tensor_parallel_rank
        if type(world) is not int or world < 1 or type(rank) is not int or not 0 <= rank < world:
            raise ValueError("graph binding TP world/rank must be explicit integers within range")
        if self.parallel_read_semantics not in {"replicated", "dcp_interleaved", "dsa_cp_derived"}:
            raise ValueError("graph binding parallel read semantics are not recognized")


@dataclass(slots=True)
class _DraftWindowGraphCarrierBacking:
    """Persistent, provider-owned buffers for one captured consumer operand."""

    window_block_table: torch.Tensor
    valid_num_blocks: torch.Tensor
    device_window_exact: torch.Tensor
    cpu_window_exact: torch.Tensor
    cpu_window_upper: torch.Tensor


@dataclass(slots=True)
class DraftWindowGraphReadCarrier:
    """One persistent capture operand rebound to explicit runtime metadata."""

    capture_identity: DraftWindowGraphCaptureIdentity
    read_inputs: DraftWindowGraphReadInputs
    runtime_identity: DraftWindowGraphBindingIdentity | None = None
    metadata: Mapping[str, Any] | None = None
    write_side_reference: DraftWindowGraphWriteSideReference | None = None
    write_side_snapshot: _DraftWindowGraphWriteSideSnapshot | None = None
    state: DraftWindowGraphCarrierState = DraftWindowGraphCarrierState.EMPTY
    native_full_snapshot: NativeSWAFullSnapshot | None = None


@dataclass(frozen=True, slots=True)
class DraftWindowGraphMetadataBuild:
    """Final metadata plus the pre-restore Draft-private validation view.

    Graph metadata legitimately reuses canonical full write/query owners.
    ``private_isolation_view`` is retained separately so Target isolation and
    canonical write identity remain two independent, non-contradictory
    invariants.  Ordinary P6 callers can still return a Mapping and receive the
    original all-fields isolation behavior.
    """

    metadata: Mapping[str, Any]
    private_isolation_view: Mapping[str, Any]
    native_full_snapshot: NativeSWAFullSnapshot | None = None

    def __post_init__(self) -> None:
        if not self.metadata or not self.private_isolation_view:
            raise ValueError("graph metadata build requires non-empty metadata views")
        if self.metadata is self.private_isolation_view:
            raise ValueError("graph final metadata and private isolation view must be distinct mappings")
        if tuple(self.metadata) != tuple(self.private_isolation_view):
            raise ValueError("graph metadata views must contain the same ordered layer keys")


@dataclass(frozen=True, slots=True)
class DraftWindowGraphUpdateBinding:
    """Exact submitted carrier set passed to the real backend update seam."""

    provider: DraftWindowGraphBindingProvider
    route: DraftWindowGraphRoute
    carriers: tuple[DraftWindowGraphReadCarrier, ...]
    metadata_object_ids: tuple[int, ...]


_ROUTE_CONTRACTS: dict[
    DraftWindowGraphRoute,
    tuple[str, DraftWindowExecutionPhase, bool],
] = {
    DraftWindowGraphRoute.EAGLE3_FIRST_PASS: (
        "eagle3",
        DraftWindowExecutionPhase.PROPOSAL,
        True,
    ),
    DraftWindowGraphRoute.EAGLE3_AR_STEP: (
        "eagle3",
        DraftWindowExecutionPhase.STEP,
        False,
    ),
    DraftWindowGraphRoute.DFLASH: (
        "dflash",
        DraftWindowExecutionPhase.PROPOSAL,
        True,
    ),
    DraftWindowGraphRoute.DSPARK: (
        "dspark",
        DraftWindowExecutionPhase.PROPOSAL,
        True,
    ),
}


def _tensor_identity(tensor: torch.Tensor) -> DraftWindowGraphTensorIdentity:
    try:
        storage_data_ptr = tensor.untyped_storage().data_ptr()
    except (AttributeError, RuntimeError):
        storage_data_ptr = tensor.data_ptr()
    return DraftWindowGraphTensorIdentity(
        storage_data_ptr=storage_data_ptr,
        data_ptr=tensor.data_ptr(),
        storage_offset=tensor.storage_offset(),
        shape=tuple(tensor.shape),
        stride=tuple(tensor.stride()),
        dtype=str(tensor.dtype),
        device=str(tensor.device),
    )


def _write_side_snapshot(
    reference: DraftWindowGraphWriteSideReference,
) -> _DraftWindowGraphWriteSideSnapshot:
    return _DraftWindowGraphWriteSideSnapshot(
        slot_mapping=_tensor_identity(reference.slot_mapping),
        positions=_tensor_identity(reference.positions),
        query_start_loc=_tensor_identity(reference.query_start_loc),
        mutation_versions=(
            int(reference.slot_mapping._version),
            int(reference.positions._version),
            int(reference.query_start_loc._version),
        ),
    )


def _carrier_backing_identity(
    backing: _DraftWindowGraphCarrierBacking,
) -> DraftWindowGraphBackingIdentity:
    return DraftWindowGraphBackingIdentity(
        owner_id=id(backing),
        window_block_table=_tensor_identity(backing.window_block_table),
        valid_num_blocks=_tensor_identity(backing.valid_num_blocks),
        device_window_exact=_tensor_identity(backing.device_window_exact),
        cpu_window_exact=_tensor_identity(backing.cpu_window_exact),
        cpu_window_upper=_tensor_identity(backing.cpu_window_upper),
    )


def _class_path(value: type[Any] | Any) -> str:
    value_type = value if isinstance(value, type) else type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _validate_route(
    route: DraftWindowGraphRoute,
    generation: DraftWindowGenerationIdentity,
) -> None:
    if not isinstance(route, DraftWindowGraphRoute):
        raise TypeError("graph route must be a DraftWindowGraphRoute")
    method, phase, zero_step = _ROUTE_CONTRACTS[route]
    if generation.normalized_method != method:
        raise ValueError(f"graph route method mismatch: route={route.value} method={generation.normalized_method}")
    if generation.phase is not phase:
        raise ValueError(f"graph route phase mismatch: route={route.value} phase={generation.phase.value}")
    if zero_step and generation.step != 0:
        raise ValueError(f"graph route {route.value} requires proposal step 0")
    if not zero_step and generation.step <= 0:
        raise ValueError(f"graph route {route.value} requires an autoregressive step")


class DraftWindowGraphBindingProvider:
    """Own stable capture banks and bind each current runtime generation."""

    def __init__(
        self,
        *,
        runtime_state: DraftWindowRuntimeState,
    ) -> None:
        if not isinstance(runtime_state, DraftWindowRuntimeState):
            raise TypeError("graph binding provider requires a P3 runtime state")
        self._runtime_state = runtime_state
        self._native_swa_capture_plan = getattr(runtime_state.capability, "native_swa_graph", None)
        self._carriers: dict[DraftWindowGraphCaptureIdentity, DraftWindowGraphReadCarrier] = {}
        self._submitted_carriers: tuple[DraftWindowGraphReadCarrier, ...] = ()

    @property
    def submitted_carriers(self) -> tuple[DraftWindowGraphReadCarrier, ...]:
        return self._submitted_carriers

    def registered_capture_identities(
        self,
        *,
        route: DraftWindowGraphRoute,
    ) -> tuple[DraftWindowGraphCaptureIdentity, ...]:
        """Return immutable startup identities for one explicitly typed route."""
        if not isinstance(route, DraftWindowGraphRoute):
            raise TypeError("capture identity lookup requires a graph route")
        return tuple(
            sorted(
                (
                    carrier.capture_identity
                    for carrier in self._carriers.values()
                    if carrier.capture_identity.fingerprint.route is route
                ),
                key=lambda identity: (
                    identity.consumer_ordinal,
                    identity.fingerprint.capture_bucket,
                ),
            )
        )

    def startup_capture_read_inputs(
        self,
        capture_identity: DraftWindowGraphCaptureIdentity,
    ) -> DraftWindowGraphReadInputs:
        """Resolve stable startup backing without manufacturing a generation."""
        if not isinstance(capture_identity, DraftWindowGraphCaptureIdentity):
            raise TypeError("startup capture requires a capture identity")
        carrier = self._require_registered_capture(capture_identity)
        if (
            carrier.state is not DraftWindowGraphCarrierState.EMPTY
            or carrier.runtime_identity is not None
            or carrier.metadata is not None
        ):
            raise RuntimeError("startup capture carrier is already bound to a runtime generation")
        read_inputs = carrier.read_inputs
        actual_backing = capture_identity.backing_identity
        tensor_identities = (
            (_tensor_identity(read_inputs.window_block_table), actual_backing.window_block_table),
            (_tensor_identity(read_inputs.valid_num_blocks), actual_backing.valid_num_blocks),
            (_tensor_identity(read_inputs.device_window_exact), actual_backing.device_window_exact),
            (_tensor_identity(read_inputs.cpu_window_exact), actual_backing.cpu_window_exact),
            (_tensor_identity(read_inputs.cpu_window_upper), actual_backing.cpu_window_upper),
        )
        if any(actual != expected for actual, expected in tensor_identities):
            raise RuntimeError("startup capture carrier backing identity changed")
        if read_inputs.padded_request_count != capture_identity.fingerprint.padded_request_count:
            raise RuntimeError("startup capture carrier request geometry changed")
        if read_inputs.actual_request_count == read_inputs.padded_request_count:
            return read_inputs
        capture_view = DraftWindowGraphReadInputs(
            window_block_table=read_inputs.window_block_table,
            valid_num_blocks=read_inputs.valid_num_blocks,
            device_window_exact=read_inputs.device_window_exact,
            cpu_window_exact=read_inputs.cpu_window_exact,
            cpu_window_upper=read_inputs.cpu_window_upper,
            actual_request_count=read_inputs.padded_request_count,
            padded_request_count=read_inputs.padded_request_count,
        )
        carrier.read_inputs = capture_view
        return capture_view

    @property
    def inventory_method(self) -> str:
        """Return the P2-normalized method owned by this provider."""
        return self._runtime_state.capability.inventory.method

    @property
    def native_swa_graph_plan(self):
        plan = getattr(self._runtime_state.capability, "native_swa_graph", None)
        if plan is not self._native_swa_capture_plan:
            raise RuntimeError("native SWA program/source contract changed after provider installation")
        return plan

    def _window_metadata(self, metadata):
        plan = self.native_swa_graph_plan
        return metadata if plan is None else plan.window_metadata(metadata)

    def _validate_native_metadata(self, metadata, snapshot, identity, write_side):
        plan = self.native_swa_graph_plan
        if plan is None:
            if snapshot is not None:
                raise RuntimeError("unexpected native SWA snapshot on a single-group Graph route")
            return
        from vllm_ascend.worker.v2.spec_decode.draft_window_swa_graph import NativeSWAFullSnapshot

        if not isinstance(snapshot, NativeSWAFullSnapshot):
            raise RuntimeError("native SWA Graph carrier has no generation-owned full snapshot")
        if identity is None:
            raise RuntimeError("native SWA Graph update has no active generation")
        plan.validate_native_metadata(
            metadata,
            source=self._runtime_state.capability.full_source._block_tables_owner,
            snapshot=snapshot,
            expected=identity,
            write_side=write_side,
        )

    def register_capture_identity(
        self,
        *,
        fingerprint: DraftWindowGraphRouteFingerprint,
        execution_lane: int,
        consumer_ordinal: int,
    ) -> DraftWindowGraphCaptureIdentity:
        """Allocate one stable per-consumer backing without a fake request.

        Registration defines the address contract consumed by the optional
        P7-X startup adapter. It does *not* attest that an ACL Graph or NPU
        operator captured this address; that remains a device-side evidence
        gate.
        """
        if not isinstance(fingerprint, DraftWindowGraphRouteFingerprint):
            raise TypeError("capture identity requires a graph route fingerprint")
        for name, value in (
            ("execution lane", execution_lane),
            ("consumer ordinal", consumer_ordinal),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"capture identity {name} must be a non-negative integer")
        if any(
            registered.capture_identity.fingerprint == fingerprint
            and registered.capture_identity.execution_lane == execution_lane
            and registered.capture_identity.consumer_ordinal == consumer_ordinal
            for registered in self._carriers.values()
        ):
            raise RuntimeError("capture consumer identity is already registered")
        self._validate_capture_topology(fingerprint)
        group = self._runtime_state.capability.inventory.require_single_group_direct_prefix()
        expected = {
            "method": self._runtime_state.capability.inventory.method,
            "global gid": group.global_group_id,
            "layout": group.layout_category.value,
            "builder": _class_path(group.builder_class),
            "window size": self._runtime_state.capability.inventory.window_size,
            "logical block size": group.logical_block_size,
            "kernel block size": group.kernel_block_size,
            "block span": group.block_span_tokens,
        }
        actual = {
            "method": fingerprint.method,
            "global gid": fingerprint.global_group_id,
            "layout": fingerprint.layout,
            "builder": fingerprint.builder_impl,
            "window size": fingerprint.window_size,
            "logical block size": fingerprint.logical_block_size,
            "kernel block size": fingerprint.kernel_block_size,
            "block span": fingerprint.block_span_tokens,
        }
        for diagnostic, expected_value in expected.items():
            if actual[diagnostic] != expected_value:
                raise RuntimeError(f"capture identity {diagnostic} mismatch")

        workspace = self._runtime_state.workspace
        padded_reqs = fingerprint.padded_request_count
        if padded_reqs > workspace.window_block_table.shape[0]:
            raise RuntimeError("capture request bucket exceeds graph carrier workspace capacity")
        backing = _DraftWindowGraphCarrierBacking(
            window_block_table=torch.zeros_like(workspace.window_block_table[:padded_reqs]),
            valid_num_blocks=torch.zeros_like(workspace.valid_num_blocks[:padded_reqs]),
            device_window_exact=torch.zeros_like(workspace.device_window_exact[:padded_reqs]),
            cpu_window_exact=torch.zeros_like(workspace.cpu_window_exact[:padded_reqs]),
            cpu_window_upper=torch.zeros_like(workspace.cpu_window_upper[:padded_reqs]),
        )
        backing_identity = _carrier_backing_identity(backing)
        capture_identity = DraftWindowGraphCaptureIdentity(
            fingerprint=fingerprint,
            execution_lane=execution_lane,
            consumer_ordinal=consumer_ordinal,
            provider_schema_version=3,
            backing_identity=backing_identity,
        )
        if capture_identity in self._carriers:
            raise RuntimeError("capture identity is already registered")
        read_inputs = DraftWindowGraphReadInputs(
            window_block_table=backing.window_block_table,
            valid_num_blocks=backing.valid_num_blocks,
            device_window_exact=backing.device_window_exact,
            cpu_window_exact=backing.cpu_window_exact,
            cpu_window_upper=backing.cpu_window_upper,
            actual_request_count=0,
            padded_request_count=fingerprint.padded_request_count,
        )
        self._carriers[capture_identity] = DraftWindowGraphReadCarrier(
            capture_identity=capture_identity,
            read_inputs=read_inputs,
        )
        return capture_identity

    def bind_read_carrier(
        self,
        *,
        capture_identity: DraftWindowGraphCaptureIdentity,
        inputs: DraftMetadataInputs,
        proposal_id: int,
        metadata_builder: Callable[
            [DraftWindowGraphReadInputs],
            Mapping[str, Any] | DraftWindowGraphMetadataBuild,
        ],
        target_metadata: Any,
        write_side_reference: DraftWindowGraphWriteSideReference | None = None,
    ) -> DraftWindowGraphReadCarrier:
        """Copy one P3 generation into persistent graph backing and build metadata."""
        if not isinstance(capture_identity, DraftWindowGraphCaptureIdentity):
            raise TypeError("graph read carrier requires a capture identity")
        carrier = self._require_registered_capture(capture_identity)
        if carrier.state in (
            DraftWindowGraphCarrierState.BOUND,
            DraftWindowGraphCarrierState.SUBMITTED,
        ):
            raise RuntimeError("cannot overwrite an active Draft Window graph read carrier")
        if not callable(metadata_builder):
            raise TypeError("graph metadata builder must be callable")
        if not isinstance(proposal_id, int) or isinstance(proposal_id, bool) or proposal_id < 0:
            raise ValueError("graph proposal id must be a non-negative integer")

        generation = inputs.identity
        inputs.validate_for(generation)
        if inputs.lease.workspace is not self._runtime_state.workspace:
            raise RuntimeError("graph carrier inputs do not belong to the provider P3 workspace")
        _validate_route(capture_identity.fingerprint.route, generation)
        self._validate_runtime_capture_compatibility(capture_identity, generation)
        self._runtime_state.capability.full_source.validate_owner_bindings()
        if write_side_reference is None:
            raise RuntimeError("Draft Window graph carrier requires an explicit full write-side reference")
        write_side_snapshot = self._validate_write_side_reference(
            capture_identity,
            write_side_reference,
        )

        read_inputs = carrier.read_inputs
        sources_and_destinations = (
            (inputs.window_block_table, read_inputs.window_block_table),
            (inputs.valid_num_blocks, read_inputs.valid_num_blocks),
            (inputs.device_window_exact, read_inputs.device_window_exact),
            (inputs.cpu_window_exact, read_inputs.cpu_window_exact),
            (inputs.cpu_window_upper, read_inputs.cpu_window_upper),
        )
        for source, destination in sources_and_destinations:
            if (
                source.shape != destination.shape
                or source.dtype != destination.dtype
                or source.device != destination.device
            ):
                raise RuntimeError("runtime generation does not fit capture backing geometry")
            destination.copy_(source)
        carrier.read_inputs = DraftWindowGraphReadInputs(
            window_block_table=read_inputs.window_block_table,
            valid_num_blocks=read_inputs.valid_num_blocks,
            device_window_exact=read_inputs.device_window_exact,
            cpu_window_exact=read_inputs.cpu_window_exact,
            cpu_window_upper=read_inputs.cpu_window_upper,
            actual_request_count=generation.actual_request_count,
            padded_request_count=generation.padded_request_count,
        )
        runtime_identity = self._make_binding_identity(
            generation,
            proposal_id=proposal_id,
            route=capture_identity.fingerprint.route,
        )
        build_result = metadata_builder(carrier.read_inputs)
        if isinstance(build_result, DraftWindowGraphMetadataBuild):
            metadata = build_result.metadata
            isolation_view = build_result.private_isolation_view
            native_snapshot = build_result.native_full_snapshot
        else:
            metadata = build_result
            isolation_view = metadata
            native_snapshot = None
        if not isinstance(metadata, Mapping) or not metadata:
            raise RuntimeError("explicit Draft Window graph carrier produced no Draft metadata")
        if not isinstance(isolation_view, Mapping) or not isolation_view:
            raise RuntimeError("explicit Draft Window graph carrier produced no private isolation view")
        if tuple(metadata) != tuple(isolation_view):
            raise RuntimeError("Draft graph metadata and private isolation view layer keys differ")
        # The publication boundary below checks this same private view and
        # Target owner once, after all read/write/native carrier validation.
        self._validate_metadata_uses_read_carrier(
            self._window_metadata(isolation_view),
            carrier.read_inputs,
        )
        self._validate_metadata_uses_carrier(
            self._window_metadata(metadata),
            carrier.read_inputs,
            write_side_reference,
            write_side_snapshot,
        )
        self._validate_native_metadata(metadata, native_snapshot, generation, write_side_reference)
        self._runtime_state.publish_metadata(
            inputs,
            metadata,
            target_metadata=target_metadata,
            isolation_view=isolation_view,
        )
        metadata = self._runtime_state.read_metadata(generation)

        carrier.runtime_identity = runtime_identity
        carrier.metadata = metadata
        carrier.native_full_snapshot = native_snapshot
        carrier.write_side_reference = write_side_reference
        carrier.write_side_snapshot = write_side_snapshot
        carrier.state = DraftWindowGraphCarrierState.BOUND
        self._submitted_carriers = ()
        return carrier

    def consume_metadata(
        self,
        *,
        route: DraftWindowGraphRoute,
        expected_consumer_count: int,
    ) -> tuple[Mapping[str, Any], ...]:
        """Submit exactly one explicit carrier set; baseline metadata is forbidden."""
        if not isinstance(route, DraftWindowGraphRoute):
            raise TypeError("graph metadata consumption requires a DraftWindowGraphRoute")
        if not isinstance(expected_consumer_count, int) or expected_consumer_count <= 0:
            raise ValueError("expected graph consumer count must be positive")
        carriers = tuple(
            sorted(
                (
                    carrier
                    for carrier in self._carriers.values()
                    if carrier.capture_identity.fingerprint.route is route
                    and carrier.state is DraftWindowGraphCarrierState.BOUND
                ),
                key=lambda carrier: carrier.capture_identity.consumer_ordinal,
            )
        )
        if len(carriers) != expected_consumer_count:
            raise RuntimeError(
                "graph Window provider requires an explicit Draft Window graph read carrier "
                f"for every consumer: route={route.value} expected={expected_consumer_count} actual={len(carriers)}"
            )
        ordinals = tuple(carrier.capture_identity.consumer_ordinal for carrier in carriers)
        if ordinals != tuple(range(expected_consumer_count)):
            raise RuntimeError("graph read carrier consumer ordinals are not contiguous")
        identities = tuple(carrier.runtime_identity for carrier in carriers)
        if any(identity is None for identity in identities):
            raise RuntimeError("graph read carrier has no runtime generation")
        typed_identities = tuple(identity for identity in identities if identity is not None)
        first = typed_identities[0]
        for identity in typed_identities[1:]:
            if identity.proposal_id != first.proposal_id:
                raise RuntimeError("graph read carriers belong to different proposals")
            if identity.mapping_epoch != first.mapping_epoch:
                raise RuntimeError("graph read carriers have different mapping_epoch values")
            if identity.ordered_request_fingerprint != first.ordered_request_fingerprint:
                raise RuntimeError("graph read carriers have different request fingerprints")
            if identity.global_group_id != first.global_group_id:
                raise RuntimeError("graph read carriers have different global gids")
        metadata = tuple(carrier.metadata for carrier in carriers)
        if any(entry is None for entry in metadata):
            raise RuntimeError("graph read carrier has no Draft-private metadata")
        for carrier in carriers:
            carrier.state = DraftWindowGraphCarrierState.SUBMITTED
        self._submitted_carriers = carriers
        return tuple(entry for entry in metadata if entry is not None)

    def assert_metadata_batch(
        self,
        *,
        route: DraftWindowGraphRoute,
        draft_attn_metadatas: Sequence[Mapping[str, Any]],
        num_reqs: int | None = None,
        num_tokens: int | None = None,
        graph_mode: str | None = None,
    ) -> None:
        carriers = self._require_submitted_route(route)
        actual_ids = tuple(id(metadata) for metadata in draft_attn_metadatas)
        expected_ids = tuple(id(carrier.metadata) for carrier in carriers)
        if actual_ids != expected_ids:
            raise RuntimeError("baseline or Target-derived metadata cannot replace the explicit graph read carrier")
        for carrier in carriers:
            fingerprint = carrier.capture_identity.fingerprint
            if num_reqs is not None and fingerprint.padded_request_count != num_reqs:
                raise RuntimeError("graph replay capture bucket request count mismatch")
            if num_tokens is not None and fingerprint.padded_token_count != num_tokens:
                raise RuntimeError("graph replay capture bucket token count mismatch")
            if graph_mode is not None and fingerprint.graph_mode != graph_mode:
                raise RuntimeError("graph replay mode mismatch")

    def prepare_update_binding(
        self,
        *,
        route: DraftWindowGraphRoute,
        draft_attn_metadatas: Sequence[Mapping[str, Any]],
    ) -> DraftWindowGraphUpdateBinding:
        self.assert_metadata_batch(
            route=route,
            draft_attn_metadatas=draft_attn_metadatas,
        )
        carriers = self._require_submitted_route(route)
        return DraftWindowGraphUpdateBinding(
            provider=self,
            route=route,
            carriers=carriers,
            metadata_object_ids=tuple(id(metadata) for metadata in draft_attn_metadatas),
        )

    def validate_backend_update(
        self,
        binding: DraftWindowGraphUpdateBinding,
        *,
        attn_backend: Any,
        impl_cls: type[Any],
        builder_cls: type[Any],
        draft_attn_metadatas: Sequence[Mapping[str, Any]],
        update_function_path: str,
    ) -> None:
        if binding.provider is not self:
            raise RuntimeError("graph update binding belongs to another provider")
        carriers = self._require_submitted_route(binding.route)
        if carriers != binding.carriers:
            raise RuntimeError("graph update carrier set changed before backend attribution")
        if tuple(id(metadata) for metadata in draft_attn_metadatas) != binding.metadata_object_ids:
            raise RuntimeError("graph update metadata differs from the submitted carrier batch")

        # Only this exact implementation has source-verified per-task rebinding
        # of Draft block_tables and read lengths in P6. Subclasses/other
        # backends must be inventoried independently instead of inheriting the
        # classification by name.
        from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl

        if impl_cls is not AscendAttentionBackendImpl:
            raise RuntimeError("graph backend consumer classification is unresolved")
        expected_update_path = "vllm_ascend.compilation.updatable_graph.UpdatableGraph.update"
        if update_function_path != expected_update_path:
            raise RuntimeError("graph update function identity mismatch")

        first_fingerprint = carriers[0].capture_identity.fingerprint
        actual_backend_impl = _class_path(impl_cls)
        actual_builder_impl = _class_path(builder_cls)
        if actual_backend_impl != first_fingerprint.backend_impl:
            raise RuntimeError("graph route fingerprint backend mismatch")
        if actual_builder_impl != first_fingerprint.builder_impl:
            raise RuntimeError("graph route fingerprint builder mismatch")
        if any(
            carrier.capture_identity.fingerprint.backend_impl != actual_backend_impl
            or carrier.capture_identity.fingerprint.builder_impl != actual_builder_impl
            for carrier in carriers
        ):
            raise RuntimeError("per-consumer backend or builder identity mismatch")

        group = self._runtime_state.capability.inventory.require_single_group_direct_prefix()
        expected_layers = set(group.layer_names)
        for metadata in draft_attn_metadatas:
            metadata_layers = set(self._window_metadata(metadata))
            if not metadata_layers or not metadata_layers.issubset(expected_layers):
                raise RuntimeError("Draft graph metadata layer group does not match backend inventory")
        for carrier in carriers:
            assert carrier.metadata is not None
            if carrier.write_side_reference is None or carrier.write_side_snapshot is None:
                raise RuntimeError("Draft graph carrier lost its full write-side reference")
            self._validate_metadata_uses_carrier(
                self._window_metadata(carrier.metadata),
                carrier.read_inputs,
                carrier.write_side_reference,
                carrier.write_side_snapshot,
            )
            self._validate_native_metadata(
                carrier.metadata,
                carrier.native_full_snapshot,
                getattr(self._runtime_state._active_inputs, "identity", None),
                carrier.write_side_reference,
            )
        for carrier in carriers:
            carrier.state = DraftWindowGraphCarrierState.UPDATED

    def discard_bound_carriers(self) -> None:
        """Release submitted/bound identities without replacing stable buffers."""
        for carrier in self._carriers.values():
            if carrier.state in (
                DraftWindowGraphCarrierState.BOUND,
                DraftWindowGraphCarrierState.SUBMITTED,
            ):
                carrier.state = DraftWindowGraphCarrierState.UPDATED
                carrier.metadata = None
        self._submitted_carriers = ()

    def _require_registered_capture(
        self,
        capture_identity: DraftWindowGraphCaptureIdentity,
    ) -> DraftWindowGraphReadCarrier:
        carrier = self._carriers.get(capture_identity)
        if carrier is not None:
            return carrier
        candidates = tuple(
            registered
            for registered in self._carriers.values()
            if registered.capture_identity.backing_identity == capture_identity.backing_identity
            and registered.capture_identity.execution_lane == capture_identity.execution_lane
            and registered.capture_identity.consumer_ordinal == capture_identity.consumer_ordinal
        )
        if len(candidates) != 1:
            raise RuntimeError("capture identity is not registered by this provider")
        expected = candidates[0].capture_identity.fingerprint
        actual = capture_identity.fingerprint
        diagnostics = {
            "method": "method",
            "route": "route",
            "graph_mode": "graph mode",
            "backend_impl": "backend",
            "builder_impl": "builder",
            "global_group_id": "global gid",
            "layout": "layout",
            "capture_descriptor": "capture descriptor",
            "capture_bucket": "capture bucket",
            "query_width": "query width",
            "padded_request_count": "padded request count",
            "padded_token_count": "padded token count",
            "static_k": "static K",
            "window_size": "window size",
            "logical_block_size": "logical block size",
            "kernel_block_size": "kernel block size",
            "block_span_tokens": "block span",
            "tensor_parallel_size": "TP",
            "tensor_parallel_rank": "TP rank",
            "parallel_read_semantics": "parallel read semantics",
            "data_parallel_size": "DP",
            "pipeline_parallel_size": "PP",
            "prefill_context_parallel_size": "PCP",
            "decode_context_parallel_size": "DCP",
            "sequence_parallel": "SP",
            "dsa_context_parallel": "DSA-CP",
            "flashcomm": "FlashComm",
            "async_scheduling": "async",
            "prefix_caching": "prefix cache",
        }
        for field_name, diagnostic in diagnostics.items():
            if getattr(expected, field_name) != getattr(actual, field_name):
                raise RuntimeError(f"capture identity {diagnostic} mismatch")
        raise RuntimeError("capture identity is not registered by this provider")

    def _validate_runtime_capture_compatibility(
        self,
        capture_identity: DraftWindowGraphCaptureIdentity,
        generation: DraftWindowGenerationIdentity,
    ) -> None:
        fingerprint = capture_identity.fingerprint
        self._validate_capture_topology(fingerprint)
        comparisons = (
            (generation.normalized_method, fingerprint.method, "method"),
            (generation.global_group_id, fingerprint.global_group_id, "global gid"),
            (generation.padded_request_count, fingerprint.padded_request_count, "padded request count"),
            (generation.execution_lane, capture_identity.execution_lane, "lane"),
        )
        for actual, expected, diagnostic in comparisons:
            if actual != expected:
                raise RuntimeError(f"capture/runtime {diagnostic} mismatch")

    def _capture_topology(self) -> tuple[int, int, str]:
        topology = self._runtime_state.capability.eager_topology_fingerprint
        if topology is None:
            # Compatibility for explicit single-card P3 fixtures only; never
            # infer a multi-rank identity from a missing P2 topology.
            if self._runtime_state.topology_fingerprint is not None:
                raise RuntimeError("Graph explicit runtime topology requires its P2 topology")
            return 1, 0, "replicated"
        world, rank = topology.tensor_parallel_size, topology.tensor_parallel_rank
        if type(world) is not int or world < 1 or type(rank) is not int or not 0 <= rank < world:
            raise RuntimeError("Graph P2 TP world/rank must be explicit integers within range")
        expected = world, rank, topology.parallel_read_semantics
        runtime = self._runtime_state.topology_fingerprint
        if (
            runtime is None
            or type(runtime.tensor_parallel_size) is not int
            or type(runtime.tensor_parallel_rank) is not int
            or expected != (runtime.tensor_parallel_size, runtime.tensor_parallel_rank, runtime.parallel_read_semantics)
        ):
            raise RuntimeError("Graph P2/runtime TP identity drift")
        return expected

    def _validate_capture_topology(self, fingerprint: DraftWindowGraphRouteFingerprint) -> None:
        actual = (
            fingerprint.tensor_parallel_size,
            fingerprint.tensor_parallel_rank,
            fingerprint.parallel_read_semantics,
        )
        if actual != self._capture_topology():
            raise RuntimeError("capture/P2 TP world, rank or parallel read semantics mismatch")
        topology = self._runtime_state.capability.eager_topology_fingerprint
        source_prefix = False if topology is None else topology.prefix_caching
        runtime = self._runtime_state.topology_fingerprint
        source_dp = 1 if topology is None else topology.data_parallel_size
        if fingerprint.data_parallel_size != source_dp or (
            runtime is not None and runtime.data_parallel_size != source_dp
        ):
            raise RuntimeError("capture/P2/runtime DP identity mismatch")
        if fingerprint.prefix_caching is not source_prefix or (
            runtime is not None and runtime.prefix_caching is not source_prefix
        ):
            raise RuntimeError("capture/P2/runtime prefix caching identity mismatch")

    def _make_binding_identity(
        self,
        generation: DraftWindowGenerationIdentity,
        *,
        proposal_id: int,
        route: DraftWindowGraphRoute,
    ) -> DraftWindowGraphBindingIdentity:
        world, rank, semantics = self._capture_topology()
        return DraftWindowGraphBindingIdentity(
            method=generation.normalized_method,
            route=route,
            global_group_id=generation.global_group_id,
            proposal_id=proposal_id,
            generation_id=generation.generation_id,
            mapping_epoch=generation.mapping_epoch,
            ordered_request_fingerprint=generation.ordered_request_fingerprint,
            actual_request_count=generation.actual_request_count,
            padded_request_count=generation.padded_request_count,
            provider_id=generation.provider_id,
            provider_version=generation.provider_version,
            source_owner_id=generation.source_identity.block_tables_owner_id,
            execution_lane=generation.execution_lane,
            phase=generation.phase,
            step=generation.step,
            tensor_parallel_size=world,
            tensor_parallel_rank=rank,
            parallel_read_semantics=semantics,
        )

    def _validate_write_side_reference(
        self,
        capture_identity: DraftWindowGraphCaptureIdentity,
        reference: DraftWindowGraphWriteSideReference,
    ) -> _DraftWindowGraphWriteSideSnapshot:
        if not isinstance(reference, DraftWindowGraphWriteSideReference):
            raise TypeError("graph carrier requires a DraftWindowGraphWriteSideReference")
        fingerprint = capture_identity.fingerprint
        if len(reference.actual_seq_lengths_q) != fingerprint.padded_request_count:
            raise RuntimeError("full query layout does not match the capture request bucket")
        if reference.query_start_loc.shape[0] != fingerprint.padded_request_count + 1:
            raise RuntimeError("full query-start carrier does not match the capture request bucket")
        if tuple(sorted(reference.actual_seq_lengths_q)) != reference.actual_seq_lengths_q:
            raise RuntimeError("full query layout must be monotonically non-decreasing")
        if reference.actual_seq_lengths_q and reference.actual_seq_lengths_q[-1] != reference.num_actual_tokens:
            raise RuntimeError("full query layout does not end at the actual token count")
        if reference.num_actual_tokens > fingerprint.padded_token_count:
            raise RuntimeError("full write-side token count exceeds the capture token bucket")

        full_source = self._runtime_state.capability.full_source
        slot_mappings = getattr(full_source._block_tables_owner, "slot_mappings", None)
        if not isinstance(slot_mappings, torch.Tensor) or slot_mappings.ndim != 2:
            raise RuntimeError("canonical BlockTables has no full slot-mapping owner")
        gid = fingerprint.global_group_id
        if not 0 <= gid < slot_mappings.shape[0]:
            raise RuntimeError("full slot-mapping owner has no row for the graph gid")
        canonical_row = slot_mappings[gid]
        reference_slot = _tensor_identity(reference.slot_mapping)
        canonical_slot = _tensor_identity(canonical_row)
        if (
            reference_slot.storage_data_ptr != canonical_slot.storage_data_ptr
            or reference_slot.data_ptr != canonical_slot.data_ptr
            or reference.slot_mapping.shape[0] > canonical_row.shape[0]
        ):
            raise RuntimeError("graph write-side reference must use the canonical full slot mapping")
        return _write_side_snapshot(reference)

    @staticmethod
    def _validate_write_side_unchanged(
        reference: DraftWindowGraphWriteSideReference,
        expected: _DraftWindowGraphWriteSideSnapshot,
    ) -> None:
        actual = _write_side_snapshot(reference)
        fields = (
            ("slot mapping", expected.slot_mapping, actual.slot_mapping),
            ("positions", expected.positions, actual.positions),
            ("query layout", expected.query_start_loc, actual.query_start_loc),
        )
        for diagnostic, expected_identity, actual_identity in fields:
            if actual_identity != expected_identity:
                raise RuntimeError(f"full {diagnostic} owner changed during graph metadata build")
        if actual.mutation_versions != expected.mutation_versions:
            raise RuntimeError("full write/query state changed during graph metadata build")

    @classmethod
    def _validate_metadata_uses_carrier(
        cls,
        metadata: Mapping[str, Any],
        read_inputs: DraftWindowGraphReadInputs,
        write_side_reference: DraftWindowGraphWriteSideReference,
        write_side_snapshot: _DraftWindowGraphWriteSideSnapshot,
    ) -> None:
        identity = _tensor_identity
        cls._validate_metadata_uses_read_carrier(metadata, read_inputs, _identity=identity)
        cls._validate_write_side_unchanged(
            write_side_reference,
            write_side_snapshot,
        )
        expected_slot = identity(write_side_reference.slot_mapping)
        expected_query = identity(write_side_reference.query_start_loc)
        expected_positions = identity(write_side_reference.positions)
        for layer_name, layer_metadata in metadata.items():
            if layer_metadata is None:
                continue
            slot_mapping = getattr(layer_metadata, "slot_mapping", None)
            if not isinstance(slot_mapping, torch.Tensor) or identity(slot_mapping) != expected_slot:
                raise RuntimeError(f"Draft graph metadata {layer_name!r} does not preserve the full slot mapping")
            query_start_loc = getattr(layer_metadata, "query_start_loc", None)
            if not isinstance(query_start_loc, torch.Tensor) or identity(query_start_loc) != expected_query:
                raise RuntimeError(f"Draft graph metadata {layer_name!r} does not preserve the full query layout")
            actual_seq_lengths_q = getattr(layer_metadata, "actual_seq_lengths_q", None)
            if actual_seq_lengths_q != list(write_side_reference.actual_seq_lengths_q):
                raise RuntimeError(f"Draft graph metadata {layer_name!r} has a mismatched full query layout")
            if getattr(layer_metadata, "attn_state", None) is not write_side_reference.attn_state:
                raise RuntimeError(f"Draft graph metadata {layer_name!r} has a mismatched attention state")
            if getattr(layer_metadata, "num_actual_tokens", None) != write_side_reference.num_actual_tokens:
                raise RuntimeError(f"Draft graph metadata {layer_name!r} has a mismatched actual token count")
            if getattr(layer_metadata, "causal", None) is not write_side_reference.causal:
                raise RuntimeError(f"Draft graph metadata {layer_name!r} has a mismatched causal contract")
            metadata_positions = getattr(layer_metadata, "positions", None)
            if metadata_positions is not None and (
                not isinstance(metadata_positions, torch.Tensor) or identity(metadata_positions) != expected_positions
            ):
                raise RuntimeError(f"Draft graph metadata {layer_name!r} does not preserve full positions")

    @staticmethod
    def _validate_metadata_uses_read_carrier(
        metadata: Mapping[str, Any],
        read_inputs: DraftWindowGraphReadInputs,
        *,
        _identity=None,
    ) -> None:
        """Validate only Draft-private read carriers, never full write state."""
        identity = _tensor_identity if _identity is None else _identity
        expected_table_storage = identity(read_inputs.window_block_table).storage_data_ptr
        expected_device_storage = identity(read_inputs.device_window_exact).storage_data_ptr
        expected_cpu_storage = identity(read_inputs.cpu_window_exact).storage_data_ptr
        expected_lengths = read_inputs.cpu_window_exact.tolist()
        for layer_name, layer_metadata in metadata.items():
            if layer_metadata is None:
                continue
            block_tables = getattr(layer_metadata, "block_tables", None)
            seq_lens = getattr(layer_metadata, "seq_lens", None)
            seq_lens_cpu = getattr(layer_metadata, "seq_lens_cpu", None)
            seq_lens_list = getattr(layer_metadata, "seq_lens_list", None)
            if not isinstance(block_tables, torch.Tensor) or (
                identity(block_tables).storage_data_ptr != expected_table_storage
            ):
                raise RuntimeError(f"Draft graph metadata {layer_name!r} does not use carrier block tables")
            if not isinstance(seq_lens, torch.Tensor) or (
                identity(seq_lens).storage_data_ptr != expected_device_storage
            ):
                raise RuntimeError(f"Draft graph metadata {layer_name!r} does not use carrier device lengths")
            if not isinstance(seq_lens_cpu, torch.Tensor) or (
                identity(seq_lens_cpu).storage_data_ptr != expected_cpu_storage
            ):
                raise RuntimeError(f"Draft graph metadata {layer_name!r} does not use carrier CPU lengths")
            if not isinstance(seq_lens_list, list) or seq_lens_list[: len(expected_lengths)] != expected_lengths:
                raise RuntimeError(f"Draft graph metadata {layer_name!r} does not use carrier exact length values")

    def _require_submitted_route(
        self,
        route: DraftWindowGraphRoute,
    ) -> tuple[DraftWindowGraphReadCarrier, ...]:
        if not self._submitted_carriers:
            raise RuntimeError("no explicit Draft Window graph read carrier batch was submitted")
        if any(carrier.capture_identity.fingerprint.route is not route for carrier in self._submitted_carriers):
            raise RuntimeError("submitted graph read carrier route mismatch")
        return self._submitted_carriers


def install_draft_window_graph_binding_provider(
    owner: Any,
) -> DraftWindowGraphBindingProvider:
    """Install one owner for stable per-consumer Graph read banks."""
    attribute = "_draft_window_graph_binding_provider"
    if hasattr(owner, attribute):
        raise RuntimeError("Draft Window graph binding provider is already installed")
    runtime_state = getattr(owner, "_draft_window_runtime_state", None)
    if not isinstance(runtime_state, DraftWindowRuntimeState):
        raise RuntimeError("Draft Window graph binding requires an initialized P3 runtime")
    provider = DraftWindowGraphBindingProvider(runtime_state=runtime_state)
    setattr(owner, attribute, provider)
    return provider


def resolve_parallel_draft_window_graph_route(
    owner: Any,
) -> DraftWindowGraphRoute | None:
    """Resolve DFlash/DSpark from the provider's canonical P2 inventory.

    The class-name/debug label on a speculator is not a policy or capability
    source.  When no provider is installed this remains an exact OFF-path
    no-op.  A provider for any non-parallel method is a hard route mismatch.
    """
    provider = getattr(owner, "_draft_window_graph_binding_provider", None)
    if provider is None:
        return None
    if not isinstance(provider, DraftWindowGraphBindingProvider):
        raise TypeError("invalid instance-owned Draft Window graph binding provider")
    method = provider.inventory_method
    if method == "dflash":
        return DraftWindowGraphRoute.DFLASH
    if method == "dspark":
        return DraftWindowGraphRoute.DSPARK
    raise RuntimeError(f"parallel Draft Window graph manager cannot consume method {method!r}")


def validate_draft_window_graph_capture(
    owner: Any,
    *,
    capture_context: Any = None,
) -> None:
    """Require the startup context belonging to this instance and provider."""
    provider = getattr(owner, "_draft_window_graph_binding_provider", None)
    if provider is None:
        return
    if not isinstance(provider, DraftWindowGraphBindingProvider):
        raise TypeError("invalid instance-owned Draft Window graph binding provider")
    if capture_context is None:
        raise RuntimeError("Draft Window Graph capture has no owner-bound persistent read context")
    validate = getattr(capture_context, "validate_for_provider", None)
    if not callable(validate):
        raise TypeError("invalid Draft Window startup capture context")
    validate(provider)


def consume_draft_window_graph_metadata(
    owner: Any,
    *,
    route: DraftWindowGraphRoute,
    expected_consumer_count: int,
) -> tuple[Mapping[str, Any], ...] | None:
    """Return explicit carrier metadata, or preserve exact OFF behavior."""
    provider = getattr(owner, "_draft_window_graph_binding_provider", None)
    if provider is None:
        return None
    if not isinstance(provider, DraftWindowGraphBindingProvider):
        raise TypeError("invalid instance-owned Draft Window graph binding provider")
    return provider.consume_metadata(
        route=route,
        expected_consumer_count=expected_consumer_count,
    )


def assert_draft_window_graph_metadata_batch(
    owner: Any,
    *,
    route: DraftWindowGraphRoute,
    draft_attn_metadatas: Sequence[Mapping[str, Any]],
    num_reqs: int | None = None,
    num_tokens: int | None = None,
    graph_mode: str | None = None,
) -> None:
    """Defense-in-depth check immediately before replay/forward."""
    provider = getattr(owner, "_draft_window_graph_binding_provider", None)
    if provider is None:
        return
    if not isinstance(provider, DraftWindowGraphBindingProvider):
        raise TypeError("invalid instance-owned Draft Window graph binding provider")
    provider.assert_metadata_batch(
        route=route,
        draft_attn_metadatas=draft_attn_metadatas,
        num_reqs=num_reqs,
        num_tokens=num_tokens,
        graph_mode=graph_mode,
    )


def build_draft_window_graph_update_binding(
    owner: Any,
    *,
    route: DraftWindowGraphRoute,
    draft_attn_metadatas: Sequence[Mapping[str, Any]],
) -> DraftWindowGraphUpdateBinding | None:
    """Build the typed binding passed through the actual UPDATE entrance."""
    provider = getattr(owner, "_draft_window_graph_binding_provider", None)
    if provider is None:
        return None
    if not isinstance(provider, DraftWindowGraphBindingProvider):
        raise TypeError("invalid instance-owned Draft Window graph binding provider")
    return provider.prepare_update_binding(
        route=route,
        draft_attn_metadatas=draft_attn_metadatas,
    )


def validate_draft_window_graph_backend_update(
    binding: DraftWindowGraphUpdateBinding | None,
    *,
    attn_backend: Any,
    impl_cls: type[Any],
    builder_cls: type[Any],
    draft_attn_metadatas: Sequence[Mapping[str, Any]] | None,
    update_function_path: str,
) -> None:
    """Derive consumer identity where the real backend implementation is known."""
    if binding is None:
        return None
    if not isinstance(binding, DraftWindowGraphUpdateBinding):
        raise TypeError("invalid Draft Window graph update binding")
    if draft_attn_metadatas is None:
        raise RuntimeError("Draft Window graph update requires Draft-private metadata")
    return binding.provider.validate_backend_update(
        binding,
        attn_backend=attn_backend,
        impl_cls=impl_cls,
        builder_cls=builder_cls,
        draft_attn_metadatas=draft_attn_metadatas,
        update_function_path=update_function_path,
    )


def validate_draft_window_update_tasks(graph, tasks, binding, metadata_batch, backend, num_tokens):
    """Validate resolved Draft operands immediately before ordinary Graph UPDATE."""
    import torch_npu

    from vllm_ascend.attention.attention_v1 import FIAParamProvider

    if backend is None or num_tokens is None or metadata_batch is None:
        raise RuntimeError("Draft Window UPDATE lost its backend, token count or metadata owner")
    if len(tasks) != len(graph.tasks) or not tasks:
        raise RuntimeError("Draft Window UPDATE task inventory changed")
    for captured, task in zip(graph.tasks, tasks):
        if (
            task.handle is not captured.handle
            or task.event is not captured.event
            or task.provider != captured.provider
            or task.provider_index != captured.provider_index
        ):
            raise RuntimeError("Draft Window UPDATE task does not belong to this captured graph")
        if not isinstance(task.provider, FIAParamProvider) or not task.provider.is_draft_model:
            raise RuntimeError("Draft Window typed UPDATE requires its inventoried Draft FIA consumer")
        if task.operation != torch_npu.npu_fused_infer_attention_score.out:
            raise RuntimeError("Draft Window typed UPDATE operation differs from its inventoried FIA path")
        index = task.provider_index
        if index >= len(metadata_batch):
            raise RuntimeError("Draft Window UPDATE consumer index exceeds its metadata batch")
        layer_metadata = metadata_batch[index].get(task.provider.layer_name)
        if layer_metadata is None:
            raise RuntimeError("Draft Window UPDATE lost its actual layer metadata")
        if (
            task.kwargs.get("block_table") is not layer_metadata.block_tables
            or task.kwargs.get("actual_seq_lengths_kv") is not layer_metadata.seq_lens_list
            or task.kwargs.get("actual_seq_lengths") is not layer_metadata.actual_seq_lengths_q
        ):
            raise RuntimeError("Draft Window UPDATE operands are not the submitted metadata operands")
    validate_draft_window_graph_backend_update(
        binding,
        attn_backend=backend,
        impl_cls=backend.get_impl_cls(),
        builder_cls=backend.get_builder_cls(),
        draft_attn_metadatas=metadata_batch,
        update_function_path="vllm_ascend.compilation.updatable_graph.UpdatableGraph.update",
    )
