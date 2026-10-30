# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Bounded lifecycle owner for eager asynchronous Draft Windows.

P7-X installs the pool only for an eligible non-Graph DFlash/DSpark route.
The pool owns one independent P3 runtime per lane and keeps Draft-private
metadata alive until a completion token proves the device consumer is done.
Host tests prove the state machine; real stream ordering remains an NPU gate.
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
    draft_window_tensors_share_storage,
    validate_draft_metadata_isolation,
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
    QUARANTINED = "quarantined"
    CONSUMER_COMPLETE = "consumer_complete"
    RETIRED = "retired"


@dataclass(slots=True)
class DraftWindowNPUEventCompletionToken:
    """Minimal real-device event adapter used by the live eager owner.

    Construction is dependency-injected in host tests. Production passes a
    ``torch.npu.Event`` and records it on the Draft consumer stream exactly
    once; querying never synchronizes the host.
    """

    event: Any
    _recorded: bool = False

    def record(self, stream: Any) -> None:
        if self._recorded:
            raise RuntimeError("Draft Window completion event is already recorded")
        self.event.record(stream)
        self._recorded = True

    def is_complete(self) -> bool:
        if not self._recorded:
            raise RuntimeError("Draft Window completion event was not recorded")
        return bool(self.event.query())

    def wait(self) -> None:
        """Block for this event only when bounded-pool pressure requires it."""
        if not self._recorded:
            raise RuntimeError("Draft Window completion event was not recorded")
        self.event.synchronize()

    @classmethod
    def record_current_stream(cls) -> "DraftWindowNPUEventCompletionToken":
        event = torch.npu.Event()  # type: ignore[attr-defined]
        token = cls(event=event)
        token.record(torch.npu.current_stream())  # type: ignore[attr-defined]
        return token


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
    # Retain the complete Draft-private metadata object until the device
    # completion token proves that every late backend reader has finished.
    # A returned builder result is retained before publication can fail too.
    # P3's store is generation-scoped and is intentionally discarded when the
    # lease enters pending retirement, so it cannot be the async lifetime
    # owner after submit.
    bound_metadata: Any | None = field(default=None, repr=False)
    completion_token: DraftWindowCompletionToken | None = field(default=None, repr=False)
    backend_derived_buffers: tuple[torch.Tensor, ...] = field(default=(), repr=False)
    abort_requested: bool = False
    preparation_started: bool = False
    consumer_started: bool = False


class DraftWindowEagerAsyncPool:
    """Fixed-capacity pool of independent P3 lanes for eager execution."""

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
        self._last_epoch_fingerprint: tuple[str, ...] | None = None
        self._last_epoch_step = -1
        self._next_operation_id = 0
        self._quarantined_reader: DraftWindowEagerAsyncHandle | None = None

    @property
    def has_uncertain_submission(self) -> bool:
        return self._quarantined_reader is not None

    def acquire(
        self,
        *,
        mapping_epoch: int,
        ordered_request_ids: Sequence[str],
        phase: DraftWindowExecutionPhase,
        step: int,
        allow_same_mapping_epoch: bool = False,
    ) -> DraftWindowEagerAsyncHandle:
        if self.has_uncertain_submission:
            raise RuntimeError("draft-window async pool is quarantined after uncertain reader submission")
        if not isinstance(mapping_epoch, int) or isinstance(mapping_epoch, bool) or mapping_epoch < 0:
            raise ValueError("mapping epoch must be a nonnegative integer")
        fingerprint = tuple(ordered_request_ids)
        if mapping_epoch < self._last_mapping_epoch or (
            mapping_epoch == self._last_mapping_epoch and not allow_same_mapping_epoch
        ):
            raise ValueError(
                "mapping epoch must strictly increase across async operations: "
                f"current={mapping_epoch} previous={self._last_mapping_epoch}"
            )
        if not fingerprint or any(not isinstance(req_id, str) or not req_id for req_id in fingerprint):
            raise ValueError("async request fingerprint must contain non-empty stable request IDs")
        if len(set(fingerprint)) != len(fingerprint):
            raise ValueError("async request fingerprint must contain unique stable request IDs")
        if not isinstance(phase, DraftWindowExecutionPhase):
            raise TypeError("async phase must be a DraftWindowExecutionPhase")
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError("async step must be a nonnegative integer")
        if mapping_epoch == self._last_mapping_epoch:
            if fingerprint != self._last_epoch_fingerprint or step != self._last_epoch_step + 1:
                raise ValueError("same-epoch continuation must preserve request identity and advance one step")
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
        self._last_epoch_fingerprint = fingerprint
        self._last_epoch_step = step
        self._active[lane] = handle
        return handle

    def acquire_with_backpressure(
        self,
        *,
        mapping_epoch: int,
        ordered_request_ids: Sequence[str],
        phase: DraftWindowExecutionPhase,
        step: int,
        allow_same_mapping_epoch: bool = False,
    ) -> DraftWindowEagerAsyncHandle:
        """Acquire safely, waiting only when every bounded lane is owned.

        Normal execution remains nonblocking: a free lane is acquired without
        querying unrelated readers.  Only actual pool pressure polls completed
        readers; if all lanes are still submitted, pressure is applied to the
        oldest operation by synchronizing its real completion event.  No backing
        is overwritten and no polling loop or global stream sync is introduced.
        """
        try:
            return self.acquire(
                mapping_epoch=mapping_epoch,
                ordered_request_ids=ordered_request_ids,
                phase=phase,
                step=step,
                allow_same_mapping_epoch=allow_same_mapping_epoch,
            )
        except RuntimeError:
            self.reclaim_completed()
            try:
                return self.acquire(
                    mapping_epoch=mapping_epoch,
                    ordered_request_ids=ordered_request_ids,
                    phase=phase,
                    step=step,
                    allow_same_mapping_epoch=allow_same_mapping_epoch,
                )
            except RuntimeError as retry_error:
                pressure_error = retry_error
            if any(handle is None for handle in self._active):
                raise
            submitted = tuple(
                handle
                for handle in self._active
                if handle is not None and handle.state is DraftWindowEagerAsyncState.SUBMITTED
            )
            if len(submitted) != self.capacity:
                raise RuntimeError(
                    "draft-window async pool pressure cannot wait because an owned lane has not reached SUBMITTED"
                ) from pressure_error
            oldest = min(submitted, key=lambda handle: handle.operation_id)
            completion = oldest.completion_token
            wait = getattr(completion, "wait", None)
            if completion is None or not callable(wait):
                raise RuntimeError(
                    "draft-window async pool pressure requires a waitable completion token"
                ) from pressure_error
            wait()
            if not self.poll(oldest):
                raise RuntimeError("draft-window completion wait returned before the oldest reader completed")
            self.retire(oldest)
            self.release(oldest)
            return self.acquire(
                mapping_epoch=mapping_epoch,
                ordered_request_ids=ordered_request_ids,
                phase=phase,
                step=step,
                allow_same_mapping_epoch=allow_same_mapping_epoch,
            )

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
        row_phase_fingerprint: Sequence[str] = (),
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
                row_phase_fingerprint=row_phase_fingerprint,
                quarantine_on_failure=True,
            )
            handle.inputs = inputs
            handle.state = DraftWindowEagerAsyncState.MATERIALIZED
            if inputs.identity.topology_fingerprint != handle.topology_fingerprint:
                runtime.quarantine_generation(inputs.identity)
                raise RuntimeError("materialized generation topology fingerprint does not match its async handle")
            return inputs
        except BaseException:
            if handle.preparation_started or runtime.quarantined_identity is not None:
                # Preparation can fail before typed inputs or a lease exist.
                # An acquired runtime lease/backing stays retained, and the
                # pool latch blocks every lane without clearing/querying.
                self._quarantined_reader = handle
                handle.state = DraftWindowEagerAsyncState.QUARANTINED
            else:
                handle.state = DraftWindowEagerAsyncState.RETIRED
                handle.inputs = None
                handle.bound_metadata = None
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
        runtime = self._runtimes[handle.execution_lane]
        if handle.preparation_started:
            handle.bound_metadata = metadata
            handle.backend_derived_buffers = buffers
        try:
            if any(not isinstance(buffer, torch.Tensor) for buffer in buffers):
                raise TypeError("backend derived buffers must be tensors")
            workspace_tensors = (
                inputs.window_block_table,
                inputs.valid_num_blocks,
                inputs.device_window_exact,
                inputs.cpu_window_exact,
                inputs.cpu_window_upper,
            )
            if any(
                draft_window_tensors_share_storage(buffer, backing)
                for buffer in buffers
                for backing in workspace_tensors
            ):
                raise ValueError("backend derived buffers must not alias Draft Window backing")
            if any(runtime.capability.full_source.shares_canonical_storage(buffer) for buffer in buffers):
                raise ValueError("backend derived buffers must not alias canonical FullSource")
            validate_draft_metadata_isolation(
                buffers,
                target_metadata=target_metadata,
                subject="backend derived buffers",
            )
        except BaseException:
            if handle.preparation_started:
                self.quarantine_reader(handle)
            else:
                runtime.abort_generation(inputs.identity)
                handle.inputs = None
                handle.bound_metadata = None
                handle.state = DraftWindowEagerAsyncState.RETIRED
            raise
        handle.bound_metadata = metadata
        handle.backend_derived_buffers = buffers
        try:
            runtime.publish_metadata(
                inputs,
                metadata,
                target_metadata=target_metadata,
            )
        except BaseException:
            if handle.preparation_started or runtime.quarantined_identity is not None:
                self.quarantine_reader(handle)
            else:
                handle.inputs = None
                handle.bound_metadata = None
                handle.backend_derived_buffers = ()
                handle.state = DraftWindowEagerAsyncState.RETIRED
            raise
        handle.state = DraftWindowEagerAsyncState.BOUND

    def mark_preparation_started(self, handle: DraftWindowEagerAsyncHandle) -> None:
        """Host boundary before the proposal body can enqueue native work.

        This includes input preparation before Window materialization, the
        length snapshot and metadata construction, as well as the forward.
        A failure does not attest that the body submitted no work.
        """
        self._require_state(handle, DraftWindowEagerAsyncState.ACQUIRED)
        handle.preparation_started = True

    def mark_consumer_started(self, handle: DraftWindowEagerAsyncHandle) -> None:
        """Mark the Host boundary before an eager forward can enqueue a reader."""
        self._require_state(handle, DraftWindowEagerAsyncState.BOUND)
        handle.consumer_started = True

    def quarantine_reader(self, handle: DraftWindowEagerAsyncHandle) -> None:
        """Retain all possible reader backing; no device operation or recovery."""
        self._require_active(handle)
        if handle.state not in (DraftWindowEagerAsyncState.ACQUIRED,
                                DraftWindowEagerAsyncState.MATERIALIZED, DraftWindowEagerAsyncState.BOUND):
            raise RuntimeError("possible native preparation or reader requires an owned operation")
        inputs = handle.inputs
        self._quarantined_reader = handle
        handle.state = DraftWindowEagerAsyncState.QUARANTINED
        if inputs is not None:
            self._runtimes[handle.execution_lane].quarantine_generation(inputs.identity)

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
            if handle.consumer_started or handle.preparation_started:
                self.quarantine_reader(handle)
                raise
            runtime.abort_generation(inputs.identity)
            handle.inputs = None
            handle.bound_metadata = None
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

    def reclaim_completed(self) -> tuple[int, ...]:
        """Poll, retire, and release completed lanes without blocking."""
        reclaimed: list[int] = []
        for handle in tuple(self._active):
            if handle is None or handle.state is not DraftWindowEagerAsyncState.SUBMITTED:
                continue
            if self.poll(handle):
                lane = handle.execution_lane
                self.retire(handle)
                self.release(handle)
                reclaimed.append(lane)
        return tuple(reclaimed)

    def read_bound_metadata(self, handle: DraftWindowEagerAsyncHandle) -> Any:
        self._require_state(handle, DraftWindowEagerAsyncState.BOUND)
        if handle.bound_metadata is None:
            raise RuntimeError("async handle has no bound Draft metadata")
        return handle.bound_metadata

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
        handle.bound_metadata = None
        handle.completion_token = None
        handle.backend_derived_buffers = ()
        handle.state = DraftWindowEagerAsyncState.FREE
        self._active[lane] = None

    def abort(self, handle: DraftWindowEagerAsyncHandle) -> None:
        if self._quarantined_reader is not None and handle is self._quarantined_reader:
            return
        self._require_active(handle)
        if handle.state is DraftWindowEagerAsyncState.SUBMITTED:
            handle.abort_requested = True
            return
        if handle.state is DraftWindowEagerAsyncState.CONSUMER_COMPLETE:
            handle.abort_requested = True
            handle.state = DraftWindowEagerAsyncState.RETIRED
            return
        if handle.state in (DraftWindowEagerAsyncState.ACQUIRED,
                            DraftWindowEagerAsyncState.MATERIALIZED, DraftWindowEagerAsyncState.BOUND) and (
            handle.consumer_started or handle.preparation_started
        ):
            handle.abort_requested = True
            self.quarantine_reader(handle)
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
        handle.bound_metadata = None
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
        if self.has_uncertain_submission:
            raise RuntimeError("draft-window async pool is quarantined after uncertain reader submission")
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
