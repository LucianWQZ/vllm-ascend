# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Ascend-only graph binding contracts for the Draft Window read plane.

P5 deliberately does not enable Draft Window graph execution.  It defines the
instance-owned descriptor and observation seam that P6 can use to test the
expected P3 backing at live, capture, replay, graph-task update, and consumer
boundaries.  These Python observations are not evidence that captured graph
metadata or an NPU attention operator consumed that backing; P6 must establish
the explicit read-carrier path and verify it on NPU before enabling Graph.

The provider never owns tensors.  P3's :class:`DraftWindowWorkspace` remains
the sole mutable owner: addresses are stable, contents are generation-scoped,
and every observation revalidates the active P3 lease and full identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

import torch

from vllm_ascend.worker.v2.spec_decode.draft_window import (
    DraftMetadataInputs,
    DraftWindowExecutionPhase,
    DraftWindowGenerationIdentity,
    DraftWindowLeaseState,
    DraftWindowRuntimeState,
)


class DraftWindowGraphRoute(Enum):
    """Method-specific graph source route; never a generic support boolean."""

    EAGLE3_FIRST_PASS = "eagle3_first_pass"
    EAGLE3_AR_STEP = "eagle3_ar_step"
    DFLASH = "dflash"
    DSPARK = "dspark"


class DraftWindowGraphStage(Enum):
    """Ordered source-observation points in the graph lifecycle."""

    LIVE = "live"
    CAPTURE = "capture"
    REPLAY = "replay"
    UPDATE = "update"
    CONSUMER = "consumer"


class DraftWindowGraphConsumerKind(Enum):
    """Expected future carrier path, not backend-derived consumption proof."""

    DIRECT_CAPTURED = "direct_captured"
    TASK_REBOUND = "task_rebound"
    BACKEND_DERIVED = "backend_derived"


class DraftWindowGraphMetadataOwner(Enum):
    """Expected metadata owner; P5 does not attest the runtime backend owner."""

    DRAFT_METADATA_STORE = "draft_metadata_store"
    ATTENTION_BACKEND = "attention_backend"
    GRAPH_TASK = "graph_task"


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


@dataclass(frozen=True, slots=True)
class DraftWindowGraphBindingExpectation:
    """What one capture/replay/update/consumer site expects to consume."""

    identity: DraftWindowGraphBindingIdentity
    consumer_kind: DraftWindowGraphConsumerKind
    metadata_owner: DraftWindowGraphMetadataOwner
    backing_identity: DraftWindowGraphBackingIdentity


@dataclass(frozen=True, slots=True)
class DraftWindowGraphBindingDescriptor:
    """Typed graph read carrier backed only by the P3 workspace owner."""

    identity: DraftWindowGraphBindingIdentity
    consumer_kind: DraftWindowGraphConsumerKind
    metadata_owner: DraftWindowGraphMetadataOwner
    backing_identity: DraftWindowGraphBackingIdentity
    expectation: DraftWindowGraphBindingExpectation
    window_block_table: torch.Tensor
    valid_num_blocks: torch.Tensor
    device_window_exact: torch.Tensor
    cpu_window_exact: torch.Tensor
    cpu_window_upper: torch.Tensor


@dataclass(frozen=True, slots=True)
class DraftWindowGraphSourceEvent:
    """One expected-binding observation, not an operator-consumption attestation."""

    sequence: int
    stage: DraftWindowGraphStage
    identity: DraftWindowGraphBindingIdentity
    consumer_kind: DraftWindowGraphConsumerKind
    metadata_owner: DraftWindowGraphMetadataOwner
    backing_identity: DraftWindowGraphBackingIdentity
    table_fingerprint: DraftWindowGraphTensorIdentity
    length_fingerprint: tuple[
        DraftWindowGraphTensorIdentity,
        DraftWindowGraphTensorIdentity,
        DraftWindowGraphTensorIdentity,
    ]


class DraftWindowGraphSourceObserver(Protocol):
    """Test/diagnostic sink supplied by one speculator instance."""

    def record(self, event: DraftWindowGraphSourceEvent) -> None: ...


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


def _workspace_backing_identity(
    runtime_state: DraftWindowRuntimeState,
) -> DraftWindowGraphBackingIdentity:
    workspace = runtime_state.workspace
    return DraftWindowGraphBackingIdentity(
        owner_id=id(workspace),
        window_block_table=_tensor_identity(workspace.window_block_table),
        valid_num_blocks=_tensor_identity(workspace.valid_num_blocks),
        device_window_exact=_tensor_identity(workspace.device_window_exact),
        cpu_window_exact=_tensor_identity(workspace.cpu_window_exact),
        cpu_window_upper=_tensor_identity(workspace.cpu_window_upper),
    )


def _shares_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.device != right.device:
        return False
    return _tensor_identity(left).storage_data_ptr == _tensor_identity(right).storage_data_ptr


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
    """Instance-owned P5 binding/validation seam over one P3 runtime."""

    def __init__(
        self,
        *,
        runtime_state: DraftWindowRuntimeState,
        observer: DraftWindowGraphSourceObserver | None = None,
    ) -> None:
        if not isinstance(runtime_state, DraftWindowRuntimeState):
            raise TypeError("graph binding provider requires a P3 runtime state")
        self._runtime_state = runtime_state
        self._observer = observer
        self._current_descriptor: DraftWindowGraphBindingDescriptor | None = None
        self._current_inputs: DraftMetadataInputs | None = None
        self._sequence = 0

    @property
    def observer(self) -> DraftWindowGraphSourceObserver | None:
        return self._observer

    @property
    def current_descriptor(self) -> DraftWindowGraphBindingDescriptor:
        descriptor = self._current_descriptor
        if descriptor is None:
            raise RuntimeError("no active Draft Window graph binding")
        return descriptor

    def bind(
        self,
        inputs: DraftMetadataInputs,
        *,
        proposal_id: int,
        route: DraftWindowGraphRoute,
        consumer_kind: DraftWindowGraphConsumerKind,
        metadata_owner: DraftWindowGraphMetadataOwner,
    ) -> DraftWindowGraphBindingDescriptor:
        """Bind one active P3 generation without allocating or cloning tensors."""
        if self._current_inputs is not None and self._current_inputs.lease.state is not DraftWindowLeaseState.RETIRED:
            raise RuntimeError("cannot replace an active graph binding")
        if not isinstance(proposal_id, int) or isinstance(proposal_id, bool) or proposal_id < 0:
            raise ValueError("graph proposal id must be a non-negative integer")
        if not isinstance(consumer_kind, DraftWindowGraphConsumerKind):
            raise TypeError("graph consumer must be a DraftWindowGraphConsumerKind")
        if not isinstance(metadata_owner, DraftWindowGraphMetadataOwner):
            raise TypeError("graph metadata owner must be a DraftWindowGraphMetadataOwner")

        generation = inputs.identity
        inputs.validate_for(generation)
        if inputs.lease.workspace is not self._runtime_state.workspace:
            raise RuntimeError("graph binding inputs do not belong to the provider P3 workspace")
        _validate_route(route, generation)

        workspace = self._runtime_state.workspace
        if workspace.bound_identity != generation:
            raise RuntimeError("graph binding workspace generation is stale")
        full_source = self._runtime_state.capability.full_source
        full_source.validate_owner_bindings()
        canonical_tensors = (
            *full_source._input_block_tables,
            full_source._block_tables_owner.num_blocks.gpu,
            full_source._persistent_num_blocks_cpu,
        )
        graph_tensors = (
            workspace.window_block_table,
            workspace.valid_num_blocks,
            workspace.device_window_exact,
            workspace.cpu_window_exact,
            workspace.cpu_window_upper,
        )
        if any(
            _shares_storage(graph_tensor, canonical)
            for graph_tensor in graph_tensors
            for canonical in canonical_tensors
        ):
            raise RuntimeError("graph backing must not alias canonical FullSource storage")

        backing_identity = _workspace_backing_identity(self._runtime_state)
        identity = DraftWindowGraphBindingIdentity(
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
        )
        expectation = DraftWindowGraphBindingExpectation(
            identity=identity,
            consumer_kind=consumer_kind,
            metadata_owner=metadata_owner,
            backing_identity=backing_identity,
        )
        descriptor = DraftWindowGraphBindingDescriptor(
            identity=identity,
            consumer_kind=consumer_kind,
            metadata_owner=metadata_owner,
            backing_identity=backing_identity,
            expectation=expectation,
            window_block_table=workspace.window_block_table,
            valid_num_blocks=workspace.valid_num_blocks,
            device_window_exact=workspace.device_window_exact,
            cpu_window_exact=workspace.cpu_window_exact,
            cpu_window_upper=workspace.cpu_window_upper,
        )
        self._current_inputs = inputs
        self._current_descriptor = descriptor
        return descriptor

    def observe(
        self,
        stage: DraftWindowGraphStage,
        *,
        expected: DraftWindowGraphBindingExpectation,
    ) -> DraftWindowGraphSourceEvent:
        """Validate the live generation/backing and optionally emit one event."""
        if not isinstance(stage, DraftWindowGraphStage):
            raise TypeError("graph stage must be a DraftWindowGraphStage")
        descriptor = self.current_descriptor
        inputs = self._current_inputs
        assert inputs is not None
        self._validate_expectation(expected, descriptor.expectation)
        inputs.validate_for(inputs.identity)
        if descriptor.identity.generation_id != inputs.identity.generation_id:
            raise RuntimeError("graph binding generation no longer matches P3 inputs")
        if descriptor.identity.mapping_epoch != inputs.identity.mapping_epoch:
            raise RuntimeError("graph binding mapping_epoch no longer matches P3 inputs")
        self._runtime_state.capability.full_source.validate_owner_bindings()
        current_backing = _workspace_backing_identity(self._runtime_state)
        if current_backing != descriptor.backing_identity:
            raise RuntimeError("graph backing address or geometry changed after binding")

        event = DraftWindowGraphSourceEvent(
            sequence=self._sequence,
            stage=stage,
            identity=descriptor.identity,
            consumer_kind=descriptor.consumer_kind,
            metadata_owner=descriptor.metadata_owner,
            backing_identity=descriptor.backing_identity,
            table_fingerprint=descriptor.backing_identity.window_block_table,
            length_fingerprint=(
                descriptor.backing_identity.device_window_exact,
                descriptor.backing_identity.cpu_window_exact,
                descriptor.backing_identity.cpu_window_upper,
            ),
        )
        self._sequence += 1
        if self._observer is not None:
            self._observer.record(event)
        return event

    @staticmethod
    def _validate_expectation(
        expected: DraftWindowGraphBindingExpectation,
        actual: DraftWindowGraphBindingExpectation,
    ) -> None:
        if not isinstance(expected, DraftWindowGraphBindingExpectation):
            raise TypeError("expected binding must be a DraftWindowGraphBindingExpectation")
        expected_identity = expected.identity
        actual_identity = actual.identity
        identity_fields = (
            ("method", "method"),
            ("route", "route"),
            ("global_group_id", "global gid"),
            ("proposal_id", "proposal"),
            ("generation_id", "generation"),
            ("mapping_epoch", "mapping_epoch"),
            ("ordered_request_fingerprint", "request fingerprint"),
            ("actual_request_count", "actual request count"),
            ("padded_request_count", "padded request count"),
            ("provider_id", "provider"),
            ("provider_version", "provider version"),
            ("source_owner_id", "source owner"),
            ("execution_lane", "lane"),
            ("phase", "phase"),
            ("step", "step"),
        )
        for field_name, diagnostic in identity_fields:
            if getattr(expected_identity, field_name) != getattr(actual_identity, field_name):
                raise RuntimeError(f"graph binding expectation {diagnostic} mismatch")
        if expected.consumer_kind is not actual.consumer_kind:
            raise RuntimeError("graph binding expectation consumer mismatch")
        if expected.metadata_owner is not actual.metadata_owner:
            raise RuntimeError("graph binding expectation metadata owner mismatch")
        if expected.backing_identity != actual.backing_identity:
            raise RuntimeError("graph binding expectation backing mismatch")


def install_draft_window_graph_binding_provider(
    owner: Any,
    *,
    observer: DraftWindowGraphSourceObserver | None = None,
) -> DraftWindowGraphBindingProvider:
    """Explicitly install one provider on an eligible speculator instance.

    P5 has no automatic caller: CONFIG_OFF, FORCED_OFF, and the existing Graph
    fail-closed gate therefore allocate no descriptor/provider/backing. P6 may
    call this only after it has closed its method-specific runtime gates.
    """
    attribute = "_draft_window_graph_binding_provider"
    if hasattr(owner, attribute):
        raise RuntimeError("Draft Window graph binding provider is already installed")
    runtime_state = getattr(owner, "_draft_window_runtime_state", None)
    if not isinstance(runtime_state, DraftWindowRuntimeState):
        raise RuntimeError("Draft Window graph binding requires an initialized P3 runtime")
    provider = DraftWindowGraphBindingProvider(
        runtime_state=runtime_state,
        observer=observer,
    )
    setattr(owner, attribute, provider)
    return provider


def observe_draft_window_graph_source(
    owner: Any,
    stage: DraftWindowGraphStage,
) -> DraftWindowGraphSourceEvent | None:
    """Observe one source point, or remain an exact no-op when P5 is absent."""
    provider = getattr(owner, "_draft_window_graph_binding_provider", None)
    if provider is None:
        return None
    if not isinstance(provider, DraftWindowGraphBindingProvider):
        raise TypeError("invalid instance-owned Draft Window graph binding provider")
    descriptor = provider.current_descriptor
    return provider.observe(stage, expected=descriptor.expectation)
