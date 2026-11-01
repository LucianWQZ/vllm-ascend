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
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

if TYPE_CHECKING:
    from vllm_ascend.worker.v2.spec_decode.draft_window_swa_graph import NativeSWAFullSnapshot

from vllm_ascend.worker.v2.spec_decode.draft_window import (
    DraftMetadataInputs,
    DraftWindowExecutionPhase,
    DraftWindowGenerationIdentity,
    DraftWindowRuntimeState,
    DraftWindowWorkspace,
    _read_coordinate_semantics,
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
class DraftWindowGraphCaptureIdentity:
    """One real manager descriptor and its method-specific consumer."""

    route: DraftWindowGraphRoute
    descriptor: BatchExecutionDescriptor
    # PIECEWISE descriptors omit request padding. Their owning manager resolves
    # the count once; FULL descriptors must carry this same number directly.
    num_reqs: int
    consumer_ordinal: int


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


@dataclass(slots=True)
class DraftWindowGraphReadCarrier:
    """One persistent capture operand rebound to explicit runtime metadata."""

    capture_identity: DraftWindowGraphCaptureIdentity
    read_inputs: DraftWindowGraphReadInputs
    workspace: DraftWindowWorkspace
    backing_snapshot: tuple[DraftWindowGraphTensorIdentity, ...]
    runtime_identity: DraftWindowGenerationIdentity | None = None
    metadata: Mapping[str, Any] | None = None
    write_side_reference: DraftWindowGraphWriteSideReference | None = None
    write_side_snapshot: _DraftWindowGraphWriteSideSnapshot | None = None
    state: DraftWindowGraphCarrierState = DraftWindowGraphCarrierState.EMPTY
    native_full_snapshot: NativeSWAFullSnapshot | None = None


@dataclass(frozen=True, slots=True)
class DraftWindowGraphMetadataBuild:
    metadata: Mapping[str, Any]
    native_full_snapshot: NativeSWAFullSnapshot | None = None


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


def _bank_snapshot(backing: Any) -> tuple[DraftWindowGraphTensorIdentity, ...]:
    """Check only the five stable Graph operands, including offset/stride."""
    return tuple(
        _tensor_identity(getattr(backing, name))
        for name in (
            "window_block_table",
            "valid_num_blocks",
            "device_window_exact",
            "cpu_window_exact",
            "cpu_window_upper",
        )
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
        self._capability = runtime_state.capability
        self.group = self._capability.inventory.require_single_group_direct_prefix()
        self.window_size = self._capability.inventory.window_size
        self._topology = self._capability.runtime_context
        if self._topology is None:
            raise RuntimeError("Draft Graph provider requires the actual source topology")
        self._capture_topology()
        if _read_coordinate_semantics(self._capability.inventory.groups) != "replicated" or any(
            value != 1
            for value in (
                self._topology.pipeline_parallel_size,
                self._topology.prefill_context_parallel_size,
                self._topology.decode_context_parallel_size,
            )
        ):
            raise ValueError("Draft Graph read banks require replicated reads and PP=PCP=DCP=1")
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
                    if carrier.capture_identity.route is route
                ),
                key=lambda identity: (
                    identity.consumer_ordinal,
                    (identity.num_reqs, identity.descriptor.num_tokens),
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
        if read_inputs.padded_request_count != capture_identity.num_reqs:
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
        route: DraftWindowGraphRoute,
        descriptor: BatchExecutionDescriptor,
        num_reqs: int,
        consumer_ordinal: int = 0,
    ) -> DraftWindowGraphCaptureIdentity:
        """Bind the manager's exact frozen descriptor to one persistent bank."""
        if not isinstance(route, DraftWindowGraphRoute) or _ROUTE_CONTRACTS[route][0] != self.inventory_method:
            raise ValueError("capture route does not match the actual Draft method")
        if type(descriptor) is not BatchExecutionDescriptor:
            raise TypeError("capture requires the actual BatchExecutionDescriptor")
        if descriptor.cg_mode.name not in ("FULL", "PIECEWISE"):
            raise ValueError("Draft Window capture requires FULL or context-only PIECEWISE")
        if descriptor.num_active_loras or descriptor.num_ubatches != 1:
            raise NotImplementedError("Draft Window capture does not support LoRA or microbatch variants")
        if type(num_reqs) is not int or num_reqs < 1 or type(consumer_ordinal) is not int or consumer_ordinal < 0:
            raise ValueError("capture request count and consumer ordinal must be valid integers")
        if descriptor.num_reqs is not None and descriptor.num_reqs != num_reqs:
            raise ValueError("capture request count differs from the actual descriptor")
        if type(descriptor.num_tokens) is not int or descriptor.num_tokens < 1 or descriptor.num_tokens % num_reqs:
            raise ValueError("capture token count must be positive and divisible by request count")
        query_width = descriptor.num_tokens // num_reqs
        if descriptor.uniform_token_count is not None and descriptor.uniform_token_count != query_width:
            raise ValueError("capture query width differs from the actual descriptor")
        if route is DraftWindowGraphRoute.EAGLE3_AR_STEP and query_width != 1:
            raise ValueError("Eagle3 AR capture requires query width 1")
        self._validate_source_contract()
        identity = DraftWindowGraphCaptureIdentity(route, descriptor, num_reqs, consumer_ordinal)
        if identity in self._carriers:
            raise RuntimeError("capture consumer identity is already registered")
        source = self._capability.full_source
        if num_reqs > source._input_block_tables[self.group.gid].shape[0]:
            raise RuntimeError("capture request bucket exceeds source capacity")
        backing = self._runtime_state.new_workspace(num_reqs)
        backing._needs_clear = True
        read_inputs = DraftWindowGraphReadInputs(
            window_block_table=backing.window_block_table,
            valid_num_blocks=backing.valid_num_blocks,
            device_window_exact=backing.device_window_exact,
            cpu_window_exact=backing.cpu_window_exact,
            cpu_window_upper=backing.cpu_window_upper,
            actual_request_count=0,
            padded_request_count=num_reqs,
        )
        self._carriers[identity] = DraftWindowGraphReadCarrier(
            capture_identity=identity,
            read_inputs=read_inputs,
            workspace=backing,
            backing_snapshot=_bank_snapshot(backing),
        )
        return identity

    def prepare_workspace(self, capture_identity: DraftWindowGraphCaptureIdentity) -> DraftWindowWorkspace:
        """Select the final capture bank before any canonical read is written.

        The existing Graph tail-notify/current-stream dependency orders reuse.
        UPDATED is only Host submission attribution, never device completion.
        """
        carrier = self._require_registered_capture(capture_identity)
        if carrier.state in (DraftWindowGraphCarrierState.BOUND, DraftWindowGraphCarrierState.SUBMITTED):
            raise RuntimeError("cannot overwrite an active Draft Window graph read carrier")
        if self._runtime_state.quarantined_identity is not None:
            raise RuntimeError("Draft Window bank is quarantined after uncertain submission")
        if carrier.workspace.window_block_table.device.type != "cpu":
            stream = int(torch.npu.current_stream().npu_stream)
            previous = getattr(self, "_owner_stream", stream)
            if stream != previous:
                raise RuntimeError("Draft Window bank reuse changed stream without a completion dependency")
            self._owner_stream = stream
        return carrier.workspace

    def bind_read_carrier(
        self,
        *,
        capture_identity: DraftWindowGraphCaptureIdentity,
        inputs: DraftMetadataInputs,
        metadata_builder: Callable[
            [DraftWindowGraphReadInputs],
            Mapping[str, Any] | DraftWindowGraphMetadataBuild,
        ],
        target_metadata: Any,
        write_side_reference: DraftWindowGraphWriteSideReference | None = None,
    ) -> DraftWindowGraphReadCarrier:
        """Publish this generation against its selected persistent Graph bank."""
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
        generation = inputs.identity
        inputs.validate_for(generation)
        if inputs._owner is not self._runtime_state:
            raise RuntimeError("graph carrier inputs do not belong to the provider runtime")
        _validate_route(capture_identity.route, generation)
        self._validate_runtime_capture_compatibility(capture_identity, generation)
        self._runtime_state.capability.full_source.validate_owner_bindings()
        if write_side_reference is None:
            raise RuntimeError("Draft Window graph carrier requires an explicit full write-side reference")
        write_side_snapshot = self._validate_write_side_reference(
            capture_identity,
            write_side_reference,
        )

        read_inputs = carrier.read_inputs
        if inputs.workspace is not carrier.workspace:
            raise RuntimeError("Graph generation was not materialized into its exact capture bank")
        carrier.read_inputs = DraftWindowGraphReadInputs(
            window_block_table=read_inputs.window_block_table,
            valid_num_blocks=read_inputs.valid_num_blocks,
            device_window_exact=read_inputs.device_window_exact,
            cpu_window_exact=read_inputs.cpu_window_exact,
            cpu_window_upper=read_inputs.cpu_window_upper,
            actual_request_count=generation.actual_request_count,
            padded_request_count=generation.padded_request_count,
        )
        build_result = metadata_builder(carrier.read_inputs)
        if isinstance(build_result, DraftWindowGraphMetadataBuild):
            metadata, native_snapshot = build_result.metadata, build_result.native_full_snapshot
        else:
            metadata, native_snapshot = build_result, None
        if not isinstance(metadata, Mapping) or not metadata:
            raise RuntimeError("explicit Draft Window graph carrier produced no Draft metadata")
        self._validate_metadata_uses_carrier(
            self._window_metadata(metadata), carrier.read_inputs, write_side_reference, write_side_snapshot
        )
        self._validate_native_metadata(metadata, native_snapshot, generation, write_side_reference)
        self._runtime_state.publish_metadata(inputs, metadata, target_metadata=target_metadata)

        carrier.runtime_identity = generation
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
                    if carrier.capture_identity.route is route and carrier.state is DraftWindowGraphCarrierState.BOUND
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
            self._require_registered_capture(carrier.capture_identity)
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
            identity = carrier.capture_identity
            if num_reqs is not None and identity.num_reqs != num_reqs:
                raise RuntimeError("graph replay capture bucket request count mismatch")
            if num_tokens is not None and identity.descriptor.num_tokens != num_tokens:
                raise RuntimeError("graph replay capture bucket token count mismatch")
            if graph_mode is not None and identity.descriptor.cg_mode.name != graph_mode:
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

        if _class_path(builder_cls) != self.group.builder:
            raise RuntimeError("Graph metadata builder differs from the actual source route")
        self._validate_source_contract()
        group = self._runtime_state.capability.inventory.require_single_group_direct_prefix()
        expected_layers = set(group.layers)
        for metadata in draft_attn_metadatas:
            metadata_layers = set(self._window_metadata(metadata))
            if not metadata_layers or not metadata_layers.issubset(expected_layers):
                raise RuntimeError("Draft graph metadata layer group does not match backend inventory")
        for carrier in carriers:
            self._require_registered_capture(carrier.capture_identity)
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
        if carrier is None or carrier.capture_identity is not capture_identity:
            raise RuntimeError("capture identity/bucket is not registered by this Draft provider")
        if (
            _bank_snapshot(carrier.workspace) != carrier.backing_snapshot
            or _bank_snapshot(carrier.read_inputs) != carrier.backing_snapshot
        ):
            raise RuntimeError("Graph capture bank address or tensor geometry changed")
        return carrier

    def _validate_runtime_capture_compatibility(
        self,
        capture_identity: DraftWindowGraphCaptureIdentity,
        generation: DraftWindowGenerationIdentity,
    ) -> None:
        self._validate_source_contract()
        comparisons = (
            (generation.normalized_method, self.inventory_method, "method"),
            (generation.global_group_id, self.group.gid, "global gid"),
            (generation.padded_request_count, capture_identity.num_reqs, "padded request count"),
            (generation.execution_lane, self._runtime_state.execution_lane, "lane"),
        )
        for actual, expected, diagnostic in comparisons:
            if actual != expected:
                raise RuntimeError(f"capture/runtime {diagnostic} mismatch")

    def _capture_topology(self) -> tuple[int, int]:
        topology = self._runtime_state.capability.runtime_context
        if topology is None:
            raise RuntimeError("Graph read banks require their actual source topology")
        world, rank = topology.tensor_parallel_size, topology.tensor_parallel_rank
        if type(world) is not int or world < 1 or type(rank) is not int or not 0 <= rank < world:
            raise RuntimeError("Graph TP world/rank must be explicit integers within range")
        return world, rank

    def _validate_source_contract(self) -> None:
        # Both are frozen construction facts, not per-proposal identities.
        if self._runtime_state.capability is not self._capability:
            raise RuntimeError("Draft Graph source capability changed after provider installation")
        if self._capability.runtime_context is not self._topology:
            raise RuntimeError("Draft Graph runtime topology changed after capture")

    def _validate_write_side_reference(
        self,
        capture_identity: DraftWindowGraphCaptureIdentity,
        reference: DraftWindowGraphWriteSideReference,
    ) -> _DraftWindowGraphWriteSideSnapshot:
        if not isinstance(reference, DraftWindowGraphWriteSideReference):
            raise TypeError("graph carrier requires a DraftWindowGraphWriteSideReference")
        if len(reference.actual_seq_lengths_q) != capture_identity.num_reqs:
            raise RuntimeError("full query layout does not match the capture request bucket")
        if reference.query_start_loc.shape[0] != capture_identity.num_reqs + 1:
            raise RuntimeError("full query-start carrier does not match the capture request bucket")
        if tuple(sorted(reference.actual_seq_lengths_q)) != reference.actual_seq_lengths_q:
            raise RuntimeError("full query layout must be monotonically non-decreasing")
        if reference.actual_seq_lengths_q and reference.actual_seq_lengths_q[-1] != reference.num_actual_tokens:
            raise RuntimeError("full query layout does not end at the actual token count")
        if reference.num_actual_tokens > capture_identity.descriptor.num_tokens:
            raise RuntimeError("full write-side token count exceeds the capture token bucket")

        full_source = self._runtime_state.capability.full_source
        slot_mappings = getattr(full_source._block_tables_owner, "slot_mappings", None)
        if not isinstance(slot_mappings, torch.Tensor) or slot_mappings.ndim != 2:
            raise RuntimeError("canonical BlockTables has no full slot-mapping owner")
        gid = self.group.gid
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
        if any(carrier.capture_identity.route is not route for carrier in self._submitted_carriers):
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
