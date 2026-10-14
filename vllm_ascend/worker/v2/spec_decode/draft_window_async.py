# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Bounded host lifecycle foundation for future eager async Draft Windows.

The pool deliberately has no production installation point.  It proves lane,
backing, generation, abort, and completion-token ownership on the host while
the existing async capability gate remains closed pending a real NPU event and
live consumer integration.
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

from vllm_ascend.worker.v2.spec_decode.draft_window import (
    DraftMetadataInputs,
    DraftWindowCompletionToken,
    DraftWindowExecutionPhase,
    DraftWindowReadDescriptor,
    DraftWindowRuntimeState,
    DraftWindowSourceCapability,
    build_draft_window_runtime_state,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_eager_capability import (
    P7EagerTopologyFingerprint,
)


class DraftWindowEagerAsyncState(Enum):
    """Host lifecycle states; they are not a claim of NPU async support."""

    FREE = "free"
    ACQUIRED = "acquired"
    MATERIALIZED = "materialized"
    BOUND = "bound"
    SUBMITTED = "submitted"
    CONSUMER_COMPLETE = "consumer_complete"
    RETIRED = "retired"


@dataclass(slots=True, eq=False)
class DraftWindowEagerAsyncHandle:
    """One non-reusable operation handle owned by a bounded pool."""

    operation_id: int
    mapping_epoch: int
    ordered_request_fingerprint: tuple[str, ...]
    phase: DraftWindowExecutionPhase
    step: int
    execution_lane: int
    topology_fingerprint: P7EagerTopologyFingerprint
    state: DraftWindowEagerAsyncState = DraftWindowEagerAsyncState.ACQUIRED
    inputs: DraftMetadataInputs | None = field(default=None, repr=False)
    completion_token: DraftWindowCompletionToken | None = field(default=None, repr=False)
    backend_derived_buffers: tuple[torch.Tensor, ...] = field(default=(), repr=False)
    abort_requested: bool = False


class DraftWindowEagerAsyncPool:
    """Fixed-capacity, independent P3 lane pool for host contract testing."""

    def __init__(
        self,
        *,
        capability: DraftWindowSourceCapability,
        topology_template: P7EagerTopologyFingerprint,
        capacity: int,
    ) -> None:
        if capacity <= 0:
            raise ValueError("async lane capacity must be greater than zero")
        if topology_template.graph_mode != "NONE":
            raise ValueError("async eager pool requires Graph mode NONE")
        if not topology_template.async_scheduling:
            raise ValueError("async eager pool requires an async topology fingerprint")
        if topology_template.prefix_caching:
            raise ValueError("async eager pool does not support prefix caching")
        if topology_template.normalized_method != capability.inventory.method:
            raise ValueError("async topology method does not match the source capability")
        if topology_template.runtime_route != capability.inventory.runtime_route:
            raise ValueError("async topology route does not match the source capability")

        self.capacity = capacity
        self._topology_template = topology_template
        self._runtimes: tuple[DraftWindowRuntimeState, ...] = tuple(
            build_draft_window_runtime_state(
                capability,
                execution_lane=lane,
                topology_fingerprint=topology_template.for_execution_lane(
                    lane,
                    async_scheduling=True,
                ),
            )
            for lane in range(capacity)
        )
        self._active: list[DraftWindowEagerAsyncHandle | None] = [None] * capacity
        self._last_mapping_epoch = -1
        self._next_operation_id = 0

    def acquire(
        self,
        *,
        mapping_epoch: int,
        ordered_request_ids: Sequence[str],
        phase: DraftWindowExecutionPhase,
        step: int,
    ) -> DraftWindowEagerAsyncHandle:
        if not isinstance(mapping_epoch, int) or isinstance(mapping_epoch, bool) or mapping_epoch < 0:
            raise ValueError("mapping epoch must be a nonnegative integer")
        if mapping_epoch <= self._last_mapping_epoch:
            raise ValueError(
                "mapping epoch must strictly increase across async operations: "
                f"current={mapping_epoch} previous={self._last_mapping_epoch}"
            )
        fingerprint = tuple(ordered_request_ids)
        if not fingerprint or any(not isinstance(req_id, str) or not req_id for req_id in fingerprint):
            raise ValueError("async request fingerprint must contain non-empty stable request IDs")
        if len(set(fingerprint)) != len(fingerprint):
            raise ValueError("async request fingerprint must contain unique stable request IDs")
        if not isinstance(phase, DraftWindowExecutionPhase):
            raise TypeError("async phase must be a DraftWindowExecutionPhase")
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError("async step must be a nonnegative integer")
        try:
            lane = self._active.index(None)
        except ValueError as error:
            raise RuntimeError(f"draft-window async pool pressure: all {self.capacity} lanes are owned") from error

        topology = self._topology_template.for_execution_lane(
            lane,
            async_scheduling=True,
        )
        handle = DraftWindowEagerAsyncHandle(
            operation_id=self._next_operation_id,
            mapping_epoch=mapping_epoch,
            ordered_request_fingerprint=fingerprint,
            phase=phase,
            step=step,
            execution_lane=lane,
            topology_fingerprint=topology,
        )
        self._next_operation_id += 1
        self._last_mapping_epoch = mapping_epoch
        self._active[lane] = handle
        return handle

    def materialize(
        self,
        handle: DraftWindowEagerAsyncHandle,
        *,
        idx_mapping: torch.Tensor,
        read_descriptors: Sequence[DraftWindowReadDescriptor],
        device_window_exact: torch.Tensor | None,
        cpu_window_exact: torch.Tensor,
        cpu_window_upper: torch.Tensor,
        num_reqs_padded: int,
        cpu_valid_num_blocks: torch.Tensor | None = None,
    ) -> DraftMetadataInputs:
        self._require_state(handle, DraftWindowEagerAsyncState.ACQUIRED)
        runtime = self._runtimes[handle.execution_lane]
        try:
            inputs = runtime.materialize(
                mapping_epoch=handle.mapping_epoch,
                ordered_request_ids=handle.ordered_request_fingerprint,
                idx_mapping=idx_mapping,
                read_descriptors=read_descriptors,
                device_window_exact=device_window_exact,
                cpu_window_exact=cpu_window_exact,
                cpu_window_upper=cpu_window_upper,
                cpu_valid_num_blocks=cpu_valid_num_blocks,
                num_reqs_padded=num_reqs_padded,
                phase=handle.phase,
                step=handle.step,
                execution_lane=handle.execution_lane,
            )
            if inputs.identity.topology_fingerprint != handle.topology_fingerprint:
                runtime.abort_generation(inputs.identity)
                raise RuntimeError("materialized generation topology fingerprint does not match its async handle")
            handle.inputs = inputs
            handle.state = DraftWindowEagerAsyncState.MATERIALIZED
            return inputs
        except BaseException:
            handle.state = DraftWindowEagerAsyncState.RETIRED
            handle.inputs = None
            raise

    def bind(
        self,
        handle: DraftWindowEagerAsyncHandle,
        *,
        metadata: Any,
        target_metadata: Any,
        backend_derived_buffers: Sequence[torch.Tensor] = (),
    ) -> None:
        self._require_state(handle, DraftWindowEagerAsyncState.MATERIALIZED)
        inputs = handle.inputs
        assert inputs is not None
        buffers = tuple(backend_derived_buffers)
        if any(not isinstance(buffer, torch.Tensor) for buffer in buffers):
            self._runtimes[handle.execution_lane].abort_generation(inputs.identity)
            handle.inputs = None
            handle.state = DraftWindowEagerAsyncState.RETIRED
            raise TypeError("backend derived buffers must be tensors")
        try:
            self._runtimes[handle.execution_lane].publish_metadata(
                inputs,
                metadata,
                target_metadata=target_metadata,
            )
        except BaseException:
            handle.inputs = None
            handle.state = DraftWindowEagerAsyncState.RETIRED
            raise
        handle.backend_derived_buffers = buffers
        handle.state = DraftWindowEagerAsyncState.BOUND

    def submit(
        self,
        handle: DraftWindowEagerAsyncHandle,
        completion_token: DraftWindowCompletionToken,
    ) -> None:
        self._require_state(handle, DraftWindowEagerAsyncState.BOUND)
        inputs = handle.inputs
        assert inputs is not None
        runtime = self._runtimes[handle.execution_lane]
        try:
            runtime.mark_reader_pending(inputs, completion_token)
        except BaseException:
            runtime.abort_generation(inputs.identity)
            handle.inputs = None
            handle.backend_derived_buffers = ()
            handle.state = DraftWindowEagerAsyncState.RETIRED
            raise
        handle.completion_token = completion_token
        try:
            runtime.retire_generation(inputs.identity)
        except BaseException:
            # A device-event query failure does not prove completion. Keep the
            # lane and all derived buffers owned; poll() can retry retirement.
            handle.state = DraftWindowEagerAsyncState.SUBMITTED
            raise
        handle.state = DraftWindowEagerAsyncState.SUBMITTED

    def poll(self, handle: DraftWindowEagerAsyncHandle) -> bool:
        self._require_state(handle, DraftWindowEagerAsyncState.SUBMITTED)
        inputs = handle.inputs
        assert inputs is not None
        complete = self._runtimes[handle.execution_lane].poll_reader_completion(inputs.identity)
        if complete:
            handle.state = DraftWindowEagerAsyncState.CONSUMER_COMPLETE
        return complete

    def retire(self, handle: DraftWindowEagerAsyncHandle) -> None:
        self._require_active(handle)
        if handle.state is DraftWindowEagerAsyncState.SUBMITTED:
            raise RuntimeError("cannot retire before consumer completion")
        if handle.state is not DraftWindowEagerAsyncState.CONSUMER_COMPLETE:
            raise RuntimeError(f"cannot retire an async operation from state {handle.state.value}")
        handle.state = DraftWindowEagerAsyncState.RETIRED

    def release(self, handle: DraftWindowEagerAsyncHandle) -> None:
        self._require_state(handle, DraftWindowEagerAsyncState.RETIRED)
        lane = handle.execution_lane
        handle.inputs = None
        handle.completion_token = None
        handle.backend_derived_buffers = ()
        handle.state = DraftWindowEagerAsyncState.FREE
        self._active[lane] = None

    def abort(self, handle: DraftWindowEagerAsyncHandle) -> None:
        self._require_active(handle)
        if handle.state is DraftWindowEagerAsyncState.SUBMITTED:
            handle.abort_requested = True
            return
        if handle.state is DraftWindowEagerAsyncState.CONSUMER_COMPLETE:
            handle.abort_requested = True
            handle.state = DraftWindowEagerAsyncState.RETIRED
            return
        if handle.state in (
            DraftWindowEagerAsyncState.MATERIALIZED,
            DraftWindowEagerAsyncState.BOUND,
        ):
            inputs = handle.inputs
            assert inputs is not None
            self._runtimes[handle.execution_lane].abort_generation(inputs.identity)
        elif handle.state is not DraftWindowEagerAsyncState.ACQUIRED:
            raise RuntimeError(f"cannot abort an async operation from state {handle.state.value}")
        handle.inputs = None
        handle.backend_derived_buffers = ()
        handle.abort_requested = True
        handle.state = DraftWindowEagerAsyncState.RETIRED

    @contextmanager
    def operation_guard(
        self,
        handle: DraftWindowEagerAsyncHandle,
    ) -> Iterator[DraftWindowEagerAsyncHandle]:
        self._require_active(handle)
        try:
            yield handle
        except BaseException:
            self.abort(handle)
            if handle.state is DraftWindowEagerAsyncState.RETIRED:
                self.release(handle)
            raise

    def validate_topology(
        self,
        handle: DraftWindowEagerAsyncHandle,
        expected: P7EagerTopologyFingerprint,
    ) -> None:
        self._require_active(handle)
        if handle.topology_fingerprint != expected:
            raise RuntimeError("async topology fingerprint does not match the active operation")

    def _require_active(self, handle: DraftWindowEagerAsyncHandle) -> None:
        lane = handle.execution_lane
        if lane < 0 or lane >= self.capacity or self._active[lane] is not handle:
            raise RuntimeError("async handle is not the active operation for its lane")

    def _require_state(
        self,
        handle: DraftWindowEagerAsyncHandle,
        state: DraftWindowEagerAsyncState,
    ) -> None:
        self._require_active(handle)
        if handle.state is not state:
            raise RuntimeError(f"async operation state mismatch: expected={state.value} actual={handle.state.value}")
