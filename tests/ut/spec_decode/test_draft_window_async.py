# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU-only P7-E bounded async lifecycle contracts."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionMetadataBuilder,
)
from vllm_ascend.device.hardware_profile import AttentionBackendFamily
from vllm_ascend.worker.v2.spec_decode.draft_window import (
    DraftWindowExecutionPhase,
    DraftWindowGroupInventory,
    DraftWindowInventory,
    DraftWindowLayoutCategory,
    DraftWindowReadDescriptor,
    DraftWindowSourceCapability,
    SingleGroupDirectPrefixProvider,
    build_full_draft_kv_source,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_async import (
    DraftWindowEagerAsyncPool,
    DraftWindowEagerAsyncState,
    DraftWindowNPUEventCompletionToken,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_eager_capability import (
    P7EagerTopologyFingerprint,
)


class _FakeCompletionToken:
    def __init__(
        self,
        *,
        complete: bool = False,
        wait_error: BaseException | None = None,
    ) -> None:
        self.complete = complete
        self.wait_error = wait_error
        self.wait_calls = 0
        self.on_wait: Callable[[], None] | None = None

    def is_complete(self) -> bool:
        return self.complete

    def wait(self) -> None:
        self.wait_calls += 1
        if self.on_wait is not None:
            self.on_wait()
        if self.wait_error is not None:
            raise self.wait_error
        self.complete = True


class _RaisingCompletionToken(_FakeCompletionToken):
    def __init__(self) -> None:
        super().__init__()
        self.raise_on_query = True

    def is_complete(self) -> bool:
        if self.raise_on_query:
            raise RuntimeError("completion query failed")
        return super().is_complete()


def _capability() -> tuple[DraftWindowSourceCapability, Any]:
    row_width = 8
    max_num_reqs = 16
    canonical_counts = torch.full((1, max_num_reqs), row_width, dtype=torch.int32)
    block_tables = SimpleNamespace(
        block_sizes=[64],
        kernel_block_sizes=[64],
        input_block_tables=[
            torch.arange(max_num_reqs * row_width, dtype=torch.int32).reshape(
                max_num_reqs,
                row_width,
            )
        ],
        num_blocks=SimpleNamespace(gpu=canonical_counts.clone(), cpu=canonical_counts),
    )
    group = DraftWindowGroupInventory(
        global_group_id=0,
        local_consumer_index=0,
        layer_names=("draft.layers.0.self_attn",),
        backend_class=AscendAttentionBackend,
        builder_class=AscendAttentionMetadataBuilder,
        logical_block_size=64,
        kernel_block_size=64,
        block_span_tokens=64,
        causal=True,
        layout_category=DraftWindowLayoutCategory.DENSE_GQA_DIRECT_PREFIX,
        capability_failures=(),
    )
    inventory = DraftWindowInventory(
        method="eagle3",
        target_model_type="qwen3",
        runtime_route="eagle3",
        window_size=128,
        groups=(group,),
        empty_global_group_ids=(),
        capability_failures=(),
    )
    source = build_full_draft_kv_source(inventory=inventory, block_tables=block_tables)
    capability = DraftWindowSourceCapability(
        inventory=inventory,
        full_source=source,
        provider=SingleGroupDirectPrefixProvider(
            inventory=inventory,
            full_source=source,
            legacy_block_size=64,
            expected_future_offset=3,
        ),
    )
    return capability, block_tables


def _topology() -> P7EagerTopologyFingerprint:
    return P7EagerTopologyFingerprint(
        normalized_method="eagle3",
        runtime_route="eagle3",
        attention_backend_family=AttentionBackendFamily.STANDARD,
        backend_impls=("vllm_ascend.attention.attention_v1.AscendAttentionBackend",),
        builder_impls=("vllm_ascend.attention.attention_v1.AscendAttentionMetadataBuilder",),
        global_group_ids=(0,),
        layouts=("dense_gqa_direct_prefix",),
        block_geometry=((0, 64, 64),),
        tensor_parallel_size=1,
        data_parallel_size=1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        sequence_parallel=False,
        dsa_context_parallel=False,
        flashcomm=False,
        graph_mode="NONE",
        async_scheduling=True,
        prefix_caching=False,
        provider_version=1,
        execution_lane=0,
    )


def _pool(capacity: int = 2) -> tuple[DraftWindowEagerAsyncPool, Any]:
    capability, block_tables = _capability()
    return (
        DraftWindowEagerAsyncPool(
            capability=capability,
            topology_template=_topology(),
            capacity=capacity,
        ),
        block_tables,
    )


def _descriptor(start: int = 0) -> DraftWindowReadDescriptor:
    return DraftWindowReadDescriptor(
        global_group_id=0,
        source_start_column=start,
        first_token_offset=0,
        required_columns=2,
        safe_extent_tokens=128,
        valid_num_blocks=8,
        source_row_width=8,
    )


def _acquire(pool: DraftWindowEagerAsyncPool, epoch: int, request_ids: tuple[str, ...]):
    return pool.acquire(
        mapping_epoch=epoch,
        ordered_request_ids=request_ids,
        phase=DraftWindowExecutionPhase.PROPOSAL,
        step=0,
    )


def test_eagle_steps_can_share_one_mapping_epoch_but_not_change_requests() -> None:
    pool, _ = _pool(capacity=3)
    first = pool.acquire(
        mapping_epoch=10,
        ordered_request_ids=("A", "B"),
        phase=DraftWindowExecutionPhase.PROPOSAL,
        step=0,
    )
    second = pool.acquire(
        mapping_epoch=10,
        ordered_request_ids=("A", "B"),
        phase=DraftWindowExecutionPhase.STEP,
        step=1,
        allow_same_mapping_epoch=True,
    )
    assert first.execution_lane != second.execution_lane
    with pytest.raises(ValueError, match="same-epoch continuation"):
        pool.acquire(
            mapping_epoch=10,
            ordered_request_ids=("B", "A"),
            phase=DraftWindowExecutionPhase.STEP,
            step=2,
            allow_same_mapping_epoch=True,
        )


def test_event_completion_token_records_once_and_queries_real_event_contract() -> None:
    class FakeEvent:
        def __init__(self) -> None:
            self.recorded_with: object | None = None
            self.complete = False
            self.synchronize_calls = 0

        def record(self, stream: object) -> None:
            self.recorded_with = stream

        def query(self) -> bool:
            return self.complete

        def synchronize(self) -> None:
            self.synchronize_calls += 1
            self.complete = True

    event = FakeEvent()
    stream = object()
    token = DraftWindowNPUEventCompletionToken(event=event)
    token.record(stream)
    assert event.recorded_with is stream
    assert token.is_complete() is False
    token.wait()
    assert event.synchronize_calls == 1
    assert token.is_complete() is True
    event.complete = True
    assert token.is_complete() is True
    with pytest.raises(RuntimeError, match="already recorded"):
        token.record(stream)


def _materialize(
    pool: DraftWindowEagerAsyncPool,
    handle: Any,
    *,
    idx_mapping: torch.Tensor | None = None,
) -> Any:
    if idx_mapping is None:
        idx_mapping = torch.arange(len(handle.ordered_request_fingerprint), dtype=torch.int32)
    count = len(handle.ordered_request_fingerprint)
    padded = max(2, count)
    exact = torch.tensor([96] * count + [0] * (padded - count), dtype=torch.int32)
    upper = torch.tensor([128] * count + [0] * (padded - count), dtype=torch.int32)
    return pool.materialize(
        handle,
        idx_mapping=idx_mapping,
        read_descriptors=tuple(_descriptor() for _ in range(count)),
        device_window_exact=exact,
        cpu_window_exact=exact,
        cpu_window_upper=upper,
        num_reqs_padded=padded,
    )


def _bind_and_submit(
    pool: DraftWindowEagerAsyncPool,
    handle: Any,
    token: _FakeCompletionToken,
) -> None:
    inputs = _materialize(pool, handle)
    assert handle.state is DraftWindowEagerAsyncState.MATERIALIZED
    derived = torch.zeros(4, dtype=torch.int32)
    pool.bind(
        handle,
        metadata={"generation": inputs.identity.generation_id, "derived": derived},
        target_metadata={"target": torch.ones(1)},
        backend_derived_buffers=(derived,),
    )
    assert handle.state is DraftWindowEagerAsyncState.BOUND
    pool.submit(handle, token)
    assert handle.state is DraftWindowEagerAsyncState.SUBMITTED


def test_two_inflight_generations_complete_in_reverse_without_early_reuse() -> None:
    pool, _block_tables = _pool()
    first = _acquire(pool, 10, ("A",))
    second = _acquire(pool, 11, ("B",))
    event_a = _FakeCompletionToken()
    event_b = _FakeCompletionToken()
    _bind_and_submit(pool, first, event_a)
    _bind_and_submit(pool, second, event_b)

    assert first.execution_lane != second.execution_lane
    with pytest.raises(RuntimeError, match="pool pressure"):
        _acquire(pool, 12, ("C",))

    event_b.complete = True
    assert pool.poll(second) is True
    assert second.state is DraftWindowEagerAsyncState.CONSUMER_COMPLETE
    pool.retire(second)
    assert second.state is DraftWindowEagerAsyncState.RETIRED
    pool.release(second)
    assert second.state is DraftWindowEagerAsyncState.FREE

    third = _acquire(pool, 12, ("C",))
    assert third.execution_lane == second.execution_lane
    assert pool.poll(first) is False
    with pytest.raises(RuntimeError, match="not the active operation"):
        pool.poll(second)


def test_pressure_waits_oldest_submitted_lane_without_overwrite() -> None:
    pool, _block_tables = _pool(capacity=2)
    first = _acquire(pool, 10, ("A",))
    second = _acquire(pool, 11, ("B",))
    event_a = _FakeCompletionToken()
    event_b = _FakeCompletionToken()
    _bind_and_submit(pool, first, event_a)
    _bind_and_submit(pool, second, event_b)
    retained_b = second.bound_metadata

    def assert_oldest_is_still_owned() -> None:
        assert first.bound_metadata is not None
        assert second.bound_metadata is retained_b
        assert pool._active == [first, second]

    event_a.on_wait = assert_oldest_is_still_owned
    third = pool.acquire_with_backpressure(
        mapping_epoch=12,
        ordered_request_ids=("C",),
        phase=DraftWindowExecutionPhase.PROPOSAL,
        step=0,
    )

    assert event_a.wait_calls == 1
    assert event_b.wait_calls == 0
    assert third.execution_lane == first.execution_lane
    assert pool._active[second.execution_lane] is second
    assert second.bound_metadata is retained_b


def test_pressure_reclaims_reverse_completed_lane_without_wait() -> None:
    pool, _block_tables = _pool(capacity=2)
    first = _acquire(pool, 10, ("A",))
    second = _acquire(pool, 11, ("B",))
    event_a = _FakeCompletionToken()
    event_b = _FakeCompletionToken(complete=True)
    _bind_and_submit(pool, first, event_a)
    _bind_and_submit(pool, second, event_b)

    third = pool.acquire_with_backpressure(
        mapping_epoch=12,
        ordered_request_ids=("C",),
        phase=DraftWindowExecutionPhase.PROPOSAL,
        step=0,
    )

    assert event_a.wait_calls == 0
    assert event_b.wait_calls == 0
    assert third.execution_lane == second.execution_lane
    assert pool._active[first.execution_lane] is first


def test_pressure_wait_failure_keeps_all_lanes_owned() -> None:
    pool, _block_tables = _pool(capacity=1)
    handle = _acquire(pool, 1, ("A",))
    event = _FakeCompletionToken(wait_error=RuntimeError("event wait failed"))
    _bind_and_submit(pool, handle, event)

    with pytest.raises(RuntimeError, match="event wait failed"):
        pool.acquire_with_backpressure(
            mapping_epoch=2,
            ordered_request_ids=("B",),
            phase=DraftWindowExecutionPhase.PROPOSAL,
            step=0,
        )

    assert event.wait_calls == 1
    assert handle.state is DraftWindowEagerAsyncState.SUBMITTED
    assert pool._active == [handle]
    assert handle.bound_metadata is not None


def test_pressure_after_abort_submit_waits_before_reusing_lane() -> None:
    pool, _block_tables = _pool(capacity=1)
    handle = _acquire(pool, 1, ("A",))
    event = _FakeCompletionToken()
    _bind_and_submit(pool, handle, event)
    retained = handle.bound_metadata

    pool.abort(handle)
    assert handle.abort_requested is True
    assert handle.state is DraftWindowEagerAsyncState.SUBMITTED

    replacement = pool.acquire_with_backpressure(
        mapping_epoch=2,
        ordered_request_ids=("B",),
        phase=DraftWindowExecutionPhase.PROPOSAL,
        step=0,
    )

    assert event.wait_calls == 1
    assert handle.state is DraftWindowEagerAsyncState.FREE
    assert handle.bound_metadata is None
    assert retained is not None
    assert replacement.execution_lane == handle.execution_lane
    assert replacement.operation_id > handle.operation_id


def test_pressure_completion_query_error_keeps_lane_and_metadata_owned() -> None:
    pool, _block_tables = _pool(capacity=1)
    handle = _acquire(pool, 1, ("A",))
    _materialize(pool, handle)
    pool.bind(handle, metadata={"draft": 1}, target_metadata={"target": 1})
    event = _RaisingCompletionToken()

    with pytest.raises(RuntimeError, match="completion query failed"):
        pool.submit(handle, event)
    retained = handle.bound_metadata

    with pytest.raises(RuntimeError, match="completion query failed"):
        pool.acquire_with_backpressure(
            mapping_epoch=2,
            ordered_request_ids=("B",),
            phase=DraftWindowExecutionPhase.PROPOSAL,
            step=0,
        )

    assert handle.state is DraftWindowEagerAsyncState.SUBMITTED
    assert handle.bound_metadata is retained
    assert pool._active == [handle]
    assert event.wait_calls == 0


def test_pressure_preserves_prefix_and_tp_topology_identity() -> None:
    capability, _block_tables = _capability()
    topology = replace_topology(
        _topology(),
        prefix_caching=True,
        tensor_parallel_size=16,
        tensor_parallel_rank=7,
        parallel_read_semantics="replicated",
    )
    pool = DraftWindowEagerAsyncPool(
        capability=capability,
        topology_template=topology,
        capacity=1,
    )
    first = _acquire(pool, 1, ("A",))
    event = _FakeCompletionToken()
    _bind_and_submit(pool, first, event)

    second = pool.acquire_with_backpressure(
        mapping_epoch=2,
        ordered_request_ids=("B",),
        phase=DraftWindowExecutionPhase.PROPOSAL,
        step=0,
    )

    assert second.topology_fingerprint.prefix_caching is True
    assert second.topology_fingerprint.tensor_parallel_size == 16
    assert second.topology_fingerprint.tensor_parallel_rank == 7
    assert second.topology_fingerprint.parallel_read_semantics == "replicated"


def test_stable_lane_backing_survives_a_b_a_without_stale_generation() -> None:
    pool, block_tables = _pool(capacity=1)
    canonical_before = block_tables.input_block_tables[0].clone()
    idx_mapping = torch.tensor([4], dtype=torch.int32)

    first = _acquire(pool, 1, ("A",))
    first_inputs = _materialize(pool, first, idx_mapping=idx_mapping)
    table_ptr = first_inputs.window_block_table.data_ptr()
    pool.bind(first, metadata={"id": "A"}, target_metadata={"target": [1]})
    event = _FakeCompletionToken(complete=True)
    pool.submit(first, event)
    assert pool.poll(first)
    pool.retire(first)
    pool.release(first)

    second = _acquire(pool, 2, ("B",))
    second_inputs = _materialize(pool, second, idx_mapping=idx_mapping)
    assert second_inputs.window_block_table.data_ptr() == table_ptr
    assert second_inputs.identity != first_inputs.identity
    pool.abort(second)
    pool.release(second)

    third = _acquire(pool, 3, ("A",))
    third_inputs = _materialize(pool, third, idx_mapping=idx_mapping)
    assert third_inputs.window_block_table.data_ptr() == table_ptr
    assert third_inputs.identity != first_inputs.identity
    with pytest.raises(RuntimeError, match="not the active operation"):
        pool.bind(first, metadata={}, target_metadata={})
    torch.testing.assert_close(block_tables.input_block_tables[0], canonical_before)


def test_reorder_finish_add_and_same_row_replacement_require_new_epoch() -> None:
    pool, _block_tables = _pool(capacity=1)
    same_mapping = torch.tensor([2, 3], dtype=torch.int32)

    for epoch, requests in (
        (1, ("A", "B")),
        (2, ("B", "A")),
        (3, ("A",)),
        (4, ("A", "C")),
        (5, ("D", "C")),
    ):
        handle = _acquire(pool, epoch, requests)
        mapping = same_mapping[: len(requests)]
        _materialize(pool, handle, idx_mapping=mapping)
        pool.abort(handle)
        pool.release(handle)

    with pytest.raises(ValueError, match="strictly increase"):
        _acquire(pool, 5, ("D", "C"))


def test_abort_before_submit_releases_materialized_state_and_derived_buffers() -> None:
    pool, _block_tables = _pool(capacity=1)
    handle = _acquire(pool, 1, ("A",))
    _materialize(pool, handle)
    pool.bind(
        handle,
        metadata={"draft": torch.zeros(1)},
        target_metadata={"target": torch.ones(1)},
        backend_derived_buffers=(torch.zeros(1),),
    )

    assert handle.bound_metadata is not None

    pool.abort(handle)

    assert handle.state is DraftWindowEagerAsyncState.RETIRED
    assert handle.bound_metadata is None
    assert handle.backend_derived_buffers == ()
    pool.release(handle)
    assert _acquire(pool, 2, ("B",)).execution_lane == 0


@pytest.mark.parametrize(
    "alias_owner",
    ["window", "canonical", "target"],
)
def test_backend_derived_buffers_reject_read_plane_and_target_aliases(
    alias_owner: str,
) -> None:
    pool, block_tables = _pool(capacity=1)
    handle = _acquire(pool, 1, ("A",))
    inputs = _materialize(pool, handle)
    target_tensor = torch.ones(4, dtype=torch.int32)
    aliases = {
        "window": inputs.window_block_table[:, :1],
        "canonical": block_tables.input_block_tables[0][:1, :1],
        "target": target_tensor[:1],
    }

    with pytest.raises(ValueError, match="backend derived buffers.*alias"):
        pool.bind(
            handle,
            metadata={"draft": torch.zeros(1)},
            target_metadata={"target": target_tensor},
            backend_derived_buffers=(aliases[alias_owner],),
        )

    assert handle.state is DraftWindowEagerAsyncState.RETIRED
    assert handle.inputs is None
    pool.release(handle)


def test_abort_after_submit_waits_for_consumer_completion() -> None:
    pool, _block_tables = _pool(capacity=1)
    handle = _acquire(pool, 1, ("A",))
    event = _FakeCompletionToken()
    _bind_and_submit(pool, handle, event)

    # ``submit`` retires the P3 metadata store immediately, but the async
    # operation must keep the complete Draft-private metadata object (and all
    # nested backend-derived tensors) alive until the device reader completes.
    retained = handle.bound_metadata
    assert retained is not None
    assert retained["derived"] is handle.backend_derived_buffers[0]

    pool.abort(handle)

    assert handle.abort_requested is True
    assert handle.state is DraftWindowEagerAsyncState.SUBMITTED
    with pytest.raises(RuntimeError, match="consumer completion"):
        pool.retire(handle)
    with pytest.raises(RuntimeError, match="pool pressure"):
        _acquire(pool, 2, ("B",))

    event.complete = True
    assert pool.poll(handle)
    assert handle.bound_metadata is retained
    pool.retire(handle)
    pool.release(handle)
    assert handle.bound_metadata is None
    assert _acquire(pool, 2, ("B",)).execution_lane == 0


def test_completion_query_exception_keeps_lane_owned_and_poll_can_retry() -> None:
    pool, _block_tables = _pool(capacity=1)
    handle = _acquire(pool, 1, ("A",))
    _materialize(pool, handle)
    pool.bind(handle, metadata={"draft": 1}, target_metadata={"target": 1})
    event = _RaisingCompletionToken()

    with pytest.raises(RuntimeError, match="completion query failed"):
        pool.submit(handle, event)

    assert handle.state is DraftWindowEagerAsyncState.SUBMITTED
    with pytest.raises(RuntimeError, match="pool pressure"):
        _acquire(pool, 2, ("B",))

    event.raise_on_query = False
    event.complete = True
    assert pool.poll(handle)
    pool.retire(handle)
    pool.release(handle)
    assert _acquire(pool, 2, ("B",)).execution_lane == 0


def test_exception_guard_cleans_pre_submit_but_never_reuses_pending_reader() -> None:
    pool, _block_tables = _pool(capacity=1)
    handle = _acquire(pool, 1, ("A",))
    _materialize(pool, handle)

    with pytest.raises(RuntimeError, match="builder failed"), pool.operation_guard(handle):
        raise RuntimeError("builder failed")

    assert handle.state is DraftWindowEagerAsyncState.FREE
    replacement = _acquire(pool, 2, ("B",))
    event = _FakeCompletionToken()
    _bind_and_submit(pool, replacement, event)

    with pytest.raises(RuntimeError, match="forward failed"), pool.operation_guard(replacement):
        raise RuntimeError("forward failed")

    assert replacement.state is DraftWindowEagerAsyncState.SUBMITTED
    assert replacement.abort_requested is True
    with pytest.raises(RuntimeError, match="pool pressure"):
        _acquire(pool, 3, ("C",))


def test_bind_and_pre_submit_validation_exceptions_release_runtime_ownership() -> None:
    pool, _block_tables = _pool(capacity=1)
    invalid_buffer = _acquire(pool, 1, ("A",))
    _materialize(pool, invalid_buffer)

    with pytest.raises(TypeError, match="derived buffers"):
        pool.bind(
            invalid_buffer,
            metadata={"draft": 1},
            target_metadata={"target": 1},
            backend_derived_buffers=(object(),),  # type: ignore[arg-type]
        )

    assert invalid_buffer.state is DraftWindowEagerAsyncState.RETIRED
    pool.release(invalid_buffer)

    invalid_token = _acquire(pool, 2, ("B",))
    _materialize(pool, invalid_token)
    pool.bind(
        invalid_token,
        metadata={"draft": 2},
        target_metadata={"target": 2},
    )
    with pytest.raises(TypeError, match="completion token"):
        pool.submit(invalid_token, object())  # type: ignore[arg-type]

    assert invalid_token.state is DraftWindowEagerAsyncState.RETIRED
    pool.release(invalid_token)
    assert _acquire(pool, 3, ("C",)).execution_lane == 0


def test_wrong_topology_generation_and_lane_are_rejected() -> None:
    pool, _block_tables = _pool(capacity=1)
    handle = _acquire(pool, 1, ("A",))
    inputs = _materialize(pool, handle)

    assert inputs.identity.execution_lane == handle.execution_lane
    assert inputs.identity.topology_fingerprint == handle.topology_fingerprint
    forged = handle.topology_fingerprint.for_execution_lane(
        handle.execution_lane + 1,
        async_scheduling=True,
    )
    with pytest.raises(RuntimeError, match="topology fingerprint"):
        pool.validate_topology(handle, forged)


def test_serialized_runtime_rejects_topology_drift_before_workspace_write() -> None:
    pool, _block_tables = _pool(capacity=1)
    handle = _acquire(pool, 1, ("A",))
    idx_mapping = torch.tensor([0], dtype=torch.int32)
    first = _materialize(pool, handle, idx_mapping=idx_mapping)
    runtime = pool._runtimes[handle.execution_lane]
    runtime.retire_generation(first.identity)
    runtime.topology_fingerprint = replace_topology(
        handle.topology_fingerprint,
        runtime_route="dflash",
    )
    exact = torch.tensor([97, 0], dtype=torch.int32)
    upper = torch.tensor([128, 0], dtype=torch.int32)

    with pytest.raises(RuntimeError, match="runtime topology fingerprint"):
        runtime.materialize_serialized_step(
            previous_identity=first.identity,
            ordered_request_ids=("A",),
            idx_mapping=idx_mapping,
            read_descriptors=(_descriptor(),),
            device_window_exact=exact,
            cpu_window_exact=exact,
            cpu_window_upper=upper,
            num_reqs_padded=2,
            step=1,
            execution_lane=0,
        )

    assert runtime.workspace.bound_identity is None


def test_preempt_resume_uses_new_generation_and_preserves_backing_address() -> None:
    pool, _block_tables = _pool(capacity=1)
    preempted = _acquire(pool, 7, ("A",))
    before = _materialize(pool, preempted)
    ptr = before.window_block_table.data_ptr()
    pool.abort(preempted)
    pool.release(preempted)

    resumed = _acquire(pool, 8, ("A",))
    after = _materialize(pool, resumed)

    assert after.window_block_table.data_ptr() == ptr
    assert after.identity.mapping_epoch == 8
    assert after.identity.generation_id > before.identity.generation_id


@pytest.mark.parametrize("capacity", [0, -1])
def test_async_pool_capacity_is_bounded_and_positive(capacity: int) -> None:
    capability, _block_tables = _capability()
    with pytest.raises(ValueError, match="capacity"):
        DraftWindowEagerAsyncPool(
            capability=capability,
            topology_template=_topology(),
            capacity=capacity,
        )


def test_async_pool_rejects_graph_and_non_async_fingerprints() -> None:
    capability, _block_tables = _capability()
    topology = _topology()
    invalid = (
        replace_topology(topology, graph_mode="FULL"),
        replace_topology(topology, async_scheduling=False),
    )

    for value in invalid:
        with pytest.raises(ValueError):
            DraftWindowEagerAsyncPool(
                capability=capability,
                topology_template=value,
                capacity=2,
            )


def test_async_pool_accepts_prefix_residency_with_independent_lane_backing() -> None:
    capability, _block_tables = _capability()
    pool = DraftWindowEagerAsyncPool(
        capability=capability,
        topology_template=replace_topology(_topology(), prefix_caching=True),
        capacity=2,
    )

    assert pool.capacity == 2
    assert (
        pool._runtimes[0].workspace.window_block_table.data_ptr()
        != pool._runtimes[1].workspace.window_block_table.data_ptr()
    )


def replace_topology(
    value: P7EagerTopologyFingerprint,
    **changes: object,
) -> P7EagerTopologyFingerprint:
    from dataclasses import replace

    return replace(value, **changes)
