# SPDX-License-Identifier: Apache-2.0
"""Host ownership tests. Controlled completions are not NPU evidence."""

import gc
import weakref
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.worker.v2.spec_decode.draft_window_lifetime import (
    DraftWindowOwnerPool,
)


class Completion:
    def __init__(self, *, complete=False, query_error=False, wait_error=False):
        self.complete = complete
        self.query_error = query_error
        self.wait_error = wait_error
        self.queries = 0
        self.waits = 0
        self.on_wait = None

    def record(self, stream):
        self.stream = stream

    def query(self):
        self.queries += 1
        if self.query_error:
            raise RuntimeError("event query failed")
        return self.complete

    def synchronize(self):
        self.waits += 1
        if self.on_wait is not None:
            self.on_wait()
        if self.wait_error:
            raise RuntimeError("event wait failed")
        self.complete = True


@pytest.fixture(autouse=True)
def event_boundary(monkeypatch):
    # These controlled Event methods prove Host ownership, not device progress.
    monkeypatch.setattr(torch.npu, "Event", Completion)
    monkeypatch.setattr(torch.npu, "current_stream", lambda: None)


class Payload:
    pass


def submit(pool, completion, *payloads):
    with patch.object(torch.npu, "Event", return_value=completion), pool.operation() as owner:
        owner.retained.extend(payloads)
    return owner


def test_entered_proposal_failure_never_infers_that_no_work_was_submitted():
    pool = DraftWindowOwnerPool(1)
    with pytest.raises(ValueError, match="invalid input"), pool.operation() as owner:
        owner.retained.append(Payload())
        raise ValueError("invalid input")
    assert owner.retained and owner.completion is None and pool.poisoned
    with pytest.raises(RuntimeError, match="quarantined"):
        pool.acquire()


def test_empty_entered_proposal_still_records_a_completion():
    pool, completion = DraftWindowOwnerPool(1), Completion()
    with patch.object(torch.npu, "Event", return_value=completion), pool.operation() as owner:
        pass
    assert owner.completion is completion and not pool.poisoned
    assert pool.reclaim_completed() == 0
    completion.complete = True
    assert pool.reclaim_completed() == 1


def test_partial_native_submission_then_host_error_retains_and_blocks_reuse():
    pool = DraftWindowOwnerPool(2)
    payload = Payload()
    reader_view = weakref.ref(payload)
    with (
        pytest.raises(RuntimeError, match="host failed after enqueue"),
        pool.operation() as owner,
    ):
        owner.retained.append(payload)
        del payload
        # This models entry into a possibly submitted region, not an NPU
        # operation. Host failure does not supply a completion boundary.
        raise RuntimeError("host failed after enqueue")
    gc.collect()
    assert reader_view() is not None
    assert pool.poisoned
    assert pool.poisoned
    # Even a free lane must not resume a context whose completion is unknown.
    with pytest.raises(RuntimeError, match="quarantined"):
        pool.acquire()
    assert reader_view() is not None


@pytest.mark.parametrize("failure", ["event construction", "event recording"])
def test_event_construction_or_partial_record_failure_keeps_all_ownership(monkeypatch, failure):
    pool = DraftWindowOwnerPool(1)
    payload = Payload()

    class Event(Completion):
        def record(self, stream):
            assert owner.completion is self
            assert owner.retained == [payload]
            raise RuntimeError("event recording")

    event = Event()

    def make_event():
        if failure == "event construction":
            raise RuntimeError(failure)
        return event

    monkeypatch.setattr(torch.npu, "Event", make_event)
    with pytest.raises(RuntimeError, match=failure), pool.operation() as owner:
        owner.retained.append(payload)
    assert pool.poisoned and owner.retained == [payload]
    assert owner.completion is (event if failure == "event recording" else None)
    with pytest.raises(RuntimeError, match="quarantined"):
        pool.acquire()


def test_free_capacity_does_not_query_or_wait_for_unrelated_reader():
    pool = DraftWindowOwnerPool(2)
    completion = Completion(query_error=True)
    first = submit(pool, completion, Payload())
    second = pool.acquire()
    assert second is not first
    assert completion.queries == completion.waits == 0


def test_pressure_reclaims_reverse_completed_reader_without_wait():
    pool = DraftWindowOwnerPool(2)
    slow, fast = Completion(), Completion(complete=True)
    first = submit(pool, slow, Payload())
    second = submit(pool, fast, Payload())
    replacement = pool.acquire()
    assert replacement is not second and replacement is not first
    assert first.retained
    assert not second.retained
    assert slow.waits == fast.waits == 0


def test_pressure_waits_for_oldest_and_retains_until_that_wait_completes():
    pool = DraftWindowOwnerPool(2)
    slow, later = Completion(), Completion()
    first = submit(pool, slow, Payload())
    second = submit(pool, later, Payload())
    before_wait = []
    slow.on_wait = lambda: before_wait.append(bool(first.retained))
    replacement = pool.acquire()
    assert before_wait == [True]
    assert slow.waits == 1
    assert later.waits == 0
    assert replacement is not first and not first.retained
    assert second.retained


@pytest.mark.parametrize("failure", ["query", "wait"])
def test_completion_failure_never_releases_the_backing(failure):
    pool = DraftWindowOwnerPool(1)
    completion = Completion(query_error=failure == "query", wait_error=failure == "wait")
    payload = Payload()
    owner = submit(pool, completion, payload)
    with pytest.raises(RuntimeError):
        pool.acquire()
    assert owner.retained == [payload]
    assert owner.completion is completion
    assert pool.poisoned
    assert pool.poisoned
    completion.query_error = completion.wait_error = False
    completion.complete = True
    with pytest.raises(RuntimeError, match="quarantined"):
        pool.acquire()


def test_all_eagle_steps_and_nested_derived_state_live_until_proposal_completion():
    pool = DraftWindowOwnerPool(1)
    completion = Completion()
    read_views, derived_views = [], []
    with patch.object(torch.npu, "Event", return_value=completion), pool.operation() as owner:
        for _step in range(4):
            reads, derived = Payload(), Payload()
            read_views.append(weakref.ref(reads))
            derived_views.append(weakref.ref(derived))
            owner.retained.extend((reads, {"builder": {"local_and_index": derived}}))
            del reads, derived
    assert len(owner.retained) == 8
    gc.collect()
    assert all(ref() is not None for ref in read_views + derived_views)
    assert pool.reclaim_completed() == 0
    assert all(ref() is not None for ref in read_views + derived_views)
    completion.complete = True
    assert pool.reclaim_completed() == 1
    gc.collect()
    assert all(ref() is None for ref in read_views + derived_views)


def test_recorded_reader_keeps_completion_authoritative_until_it_finishes():
    pool = DraftWindowOwnerPool(1)
    completion = Completion()
    owner = submit(pool, completion, Payload())
    assert owner.retained
    assert owner.completion is completion
    assert not pool.poisoned
    assert pool.reclaim_completed() == 0
    completion.complete = True
    assert pool.reclaim_completed() == 1
    assert not owner.retained and owner.completion is None


def test_late_reader_requires_all_reader_completion_not_primary_only():
    class AllReaders(Completion):
        def __init__(self, readers):
            self.readers = readers

        def query(self):
            return all(reader.query() for reader in self.readers)

        def synchronize(self):
            for reader in self.readers:
                reader.synchronize()

    primary, communication = Completion(complete=True), Completion()
    pool = DraftWindowOwnerPool(1)
    owner = submit(pool, AllReaders([primary, communication]), Payload())
    assert pool.reclaim_completed() == 0
    assert owner.retained
    communication.complete = True
    assert pool.reclaim_completed() == 1


def test_unsubmitted_pool_pressure_never_waits_or_discards():
    pool = DraftWindowOwnerPool(1)
    owner = pool.acquire()
    owner.retained.append(Payload())
    with pytest.raises(RuntimeError, match="unsubmitted proposal"):
        pool.acquire()
    assert owner.retained
    assert owner.completion is None


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5])
def test_invalid_capacity(capacity):
    with pytest.raises(ValueError, match="positive integer"):
        DraftWindowOwnerPool(capacity)
