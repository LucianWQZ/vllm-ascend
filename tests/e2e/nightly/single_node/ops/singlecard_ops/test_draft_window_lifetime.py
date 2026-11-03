# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Ownership failures after a real native enqueue, without engine mocking.

These tests prove the owner protocol at a device boundary. They do not prove
that an engine/backend joins all of its secondary-stream readers.
"""

from contextlib import contextmanager
from typing import Any

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from vllm_ascend.worker.v2.spec_decode.draft_window_lifetime import (  # noqa: E402
    DraftWindowOwnerPool,
)

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(), reason="requires an Ascend NPU"
)


@pytest.mark.parametrize("operation", ["gather"])
def test_private_kernel_outputs_survive_exception_after_real_enqueue(monkeypatch, operation):
    from vllm_ascend.ops.triton.v2.spec_decode import draft_window
    from vllm_ascend.worker.v2.spec_decode.draft_window_dense import DenseDraftRead, gather_dense_draft_kv

    pool = DraftWindowOwnerPool(1)
    stream = torch.npu.Stream()
    held_outputs: list[torch.Tensor] = []
    diagnostic_events: list[Any] = []
    kernel_name = {
        "gather": "_gather_draft_kv_fp32",
    }[operation]
    original = getattr(draft_window, kernel_name)

    class RaiseAfterEnqueue:
        def __getitem__(self, grid):
            launcher = original[grid]

            def launch(*args):
                outputs = args[3:5]
                assert all(any(value is output for value in owner.retained) for output in outputs)
                held_outputs.extend(outputs)
                launcher(*args)
                # This real event is a diagnostic cleanup token only. The
                # production operation deliberately fails without receiving
                # a token, so its owner must remain quarantined.
                event = torch.npu.Event()
                event.record(stream)
                diagnostic_events.append(event)
                raise RuntimeError("injected host exception after real kernel enqueue")

            return launch

    monkeypatch.setattr(draft_window, kernel_name, RaiseAfterEnqueue())
    with (
        pytest.raises(RuntimeError, match="after real kernel enqueue"),
        torch.npu.stream(stream),
        pool.operation() as owner,
    ):
        retain = lambda *objects: owner.retained.extend(objects)
        if operation == "gather":
            slots = torch.tensor([[1, 2, 3]], dtype=torch.int64, device="npu")
            read = DenseDraftRead(slots, torch.tensor([3]), torch.empty(0), 3, 1, retain)
            gather_dense_draft_kv(
                read,
                torch.arange(64, dtype=torch.bfloat16, device="npu").reshape(2, 4, 1, 8),
                torch.ones((2, 4, 1, 8), dtype=torch.bfloat16, device="npu"),
            )

    assert diagnostic_events and held_outputs
    assert pool.poisoned
    assert all(any(value is output for value in owner.retained) for output in held_outputs)
    with pytest.raises(RuntimeError, match="quarantined"):
        pool.acquire()
    diagnostic_events[0].synchronize()
    if operation == "gather":
        torch.testing.assert_close(held_outputs[0].cpu().flatten(), torch.arange(8, 32, dtype=torch.float32))
        assert held_outputs[1].eq(1).all()
    # Even an external diagnostic wait does not make production's unknown
    # completion contract safe to reuse.
    with pytest.raises(RuntimeError, match="quarantined"):
        pool.acquire()


@pytest.mark.parametrize("failure", ["host_after_enqueue", "event_create", "event_record"])
def test_native_enqueue_failure_quarantines_backing(monkeypatch, failure):
    pool = DraftWindowOwnerPool(1)
    stream = torch.npu.Stream()
    payload = torch.arange(1024, dtype=torch.float32, device="npu")
    stream.wait_stream(torch.npu.current_stream())

    native_event = torch.npu.Event

    class FaultEvent(native_event):
        def __new__(cls):
            if failure == "event_create":
                raise RuntimeError("injected event creation failure")
            return super().__new__(cls)

        def record(self, stream=None):
            super().record(stream)
            if failure == "event_record":
                raise RuntimeError("injected event recording failure")

    monkeypatch.setattr(torch.npu, "Event", FaultEvent)

    with pytest.raises((RuntimeError, TypeError)), torch.npu.stream(stream), pool.operation() as owner:
        owner.retained.append(payload)
        observed = payload.square()
        owner.retained.append(observed)
        if failure == "host_after_enqueue":
            raise RuntimeError("injected host failure after native enqueue")
    assert pool.poisoned
    assert owner.retained[0] is payload
    with pytest.raises(RuntimeError, match="quarantined"):
        pool.acquire()
    # Only test cleanup waits for the real stream. Production failure does
    # not infer completion or release a quarantined context.
    stream.synchronize()
    torch.testing.assert_close(observed.cpu(), torch.arange(1024, dtype=torch.float32).square())
    with pytest.raises(RuntimeError, match="quarantined"):
        pool.acquire()


def test_native_completion_covers_an_explicit_late_reader_join():
    pool = DraftWindowOwnerPool(1)
    producer, reader = torch.npu.Stream(), torch.npu.Stream()
    with torch.npu.stream(producer), pool.operation() as owner:
        payload = torch.arange(1024, dtype=torch.float32, device="npu")
        owner.retained.append(payload)
        reader.wait_stream(producer)
        with torch.npu.stream(reader):
            result = payload * 3
        owner.retained.append(result)
        # The actual executor must provide an equivalent join for every
        # secondary reader before recording its proposal completion.
        producer.wait_stream(reader)
    replacement = pool.acquire()
    assert replacement is not owner
    assert not owner.retained and owner.completion is None
    torch.testing.assert_close(result.cpu(), torch.arange(1024, dtype=torch.float32) * 3)


class ObservedNativeEvent(torch.npu.Event):
    """Record native observations without substituting a completion result."""

    def __init__(self):
        super().__init__()
        self.queries = []
        self.waits = 0
        self.failure = None

    def query(self):
        complete = super().query()
        self.queries.append(complete)
        if self.failure is not None and complete:
            pytest.skip("reader completed before native pending-completion fault boundary")
        if self.failure == "query":
            raise RuntimeError("injected host error after native pending query")
        return complete

    def synchronize(self):
        self.waits += 1
        if self.failure == "wait":
            if super().query():
                pytest.skip("reader completed before native pending-wait fault boundary")
            raise RuntimeError("injected host error before native completion wait")
        super().synchronize()


@contextmanager
def pending_native_reader(pool):
    """Queue bounded device work, then a real reader of the retained payload.

    Matrix multiplication supplies work rather than a host sleep or an event
    whose value was manufactured by a mock. On a device that drains the queue
    before observation this fixture is inconclusive. Keep every allocation
    alive and synchronize only this test's stream during cleanup.
    """
    completion: ObservedNativeEvent
    matrix_size = 4096
    delay_operations = 128
    stream = torch.npu.Stream()
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(torch.npu, "Event", ObservedNativeEvent)
            with torch.npu.stream(stream), pool.operation() as owner:
                payload = torch.arange(1024, dtype=torch.float32, device="npu")
                observed = torch.empty_like(payload)
                left = torch.full((matrix_size, matrix_size), 1 / matrix_size, dtype=torch.bfloat16, device="npu")
                right = torch.eye(matrix_size, dtype=torch.bfloat16, device="npu")
                scratch = torch.empty_like(left)
                owner.retained.extend((payload, observed, left, right, scratch))
                for _ in range(delay_operations):
                    torch.mm(left, right, out=scratch)
                    left, scratch = scratch, left
                observed.copy_(payload)
        completion = owner.completion
        if super(ObservedNativeEvent, completion).query():
            pytest.skip("bounded native queue completed before an in-flight reader was observed")
        yield owner, completion, payload, observed
    finally:
        stream.synchronize()


def test_native_pool_pressure_waits_before_overwriting_read_backing():
    pool = DraftWindowOwnerPool(1)
    with pending_native_reader(pool) as (owner, completion, payload, observed):
        replacement = pool.acquire()
        if completion.waits == 0:
            pytest.skip("reader completed before pool pressure reached the native wait")
        assert completion.queries[0] is False
        # A successful native wait already establishes completion. Observe
        # the real event independently; no extra production poll is required.
        assert super(ObservedNativeEvent, completion).query() is True
        assert completion.waits == 1
        assert not owner.retained and owner.completion is None
        assert replacement is not owner
        # The replacement may now overwrite the exact backing read by the
        # delayed native copy. That copy must have consumed the old values.
        payload.fill_(-1)
        torch.testing.assert_close(observed.cpu(), torch.arange(1024, dtype=torch.float32))


def test_native_reverse_completion_reclaims_only_the_completed_owner():
    pool = DraftWindowOwnerPool(2)
    with pending_native_reader(pool) as (slow_owner, slow, payload, observed):
        fast_stream = torch.npu.Stream()
        try:
            with torch.npu.stream(fast_stream), pool.operation() as fast_owner:
                fast_payload = torch.ones(1024, device="npu")
                fast_result = fast_payload + 1
                fast_owner.retained.extend((fast_payload, fast_result))
            fast_owner.completion.synchronize()
            if super(ObservedNativeEvent, slow).query():
                pytest.skip("device did not exhibit reverse completion of the two native readers")
            replacement = pool.acquire()
            if not slow_owner.retained and slow_owner.completion is None:
                pytest.skip("slow reader completed before pool reclamation observed it")
            assert slow.queries[-1] is False
            assert slow.waits == 0
            assert slow_owner.completion is not None
            assert any(value is payload for value in slow_owner.retained)
            assert replacement is not fast_owner and replacement is not slow_owner
            assert not fast_owner.retained and fast_owner.completion is None
            slow.synchronize()
            torch.testing.assert_close(observed.cpu(), torch.arange(1024, dtype=torch.float32))
        finally:
            fast_stream.synchronize()


@pytest.mark.parametrize("failure", ["query", "wait"])
def test_native_pending_completion_fault_quarantines_the_execution(failure):
    pool = DraftWindowOwnerPool(1)
    with pending_native_reader(pool) as (owner, completion, payload, observed):
        completion.failure = failure
        with pytest.raises(RuntimeError, match="injected host error"):
            pool.acquire()
        assert completion.queries[0] is False
        assert completion.waits == (1 if failure == "wait" else 0)
        assert pool.poisoned
        assert any(value is payload for value in owner.retained)
        assert owner.completion is completion
        with pytest.raises(RuntimeError, match="quarantined"):
            pool.acquire()
        # A later successful native completion cannot silently unpoison an
        # executor after its bookkeeping failed. No replacement can write.
        super(ObservedNativeEvent, completion).synchronize()
        torch.testing.assert_close(observed.cpu(), torch.arange(1024, dtype=torch.float32))
        with pytest.raises(RuntimeError, match="quarantined"):
            pool.acquire()
