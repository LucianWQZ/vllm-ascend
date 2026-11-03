# SPDX-License-Identifier: Apache-2.0
"""Runtime retention follows the existing completed exact-copy boundary."""

import pytest
import torch

from tests.ut.worker.v2 import test_draft_window_runtime as runtime_tests

setup = runtime_tests.setup


def build(runtime):
    with runtime.proposal_context(runtime_tests.batch()):
        runtime.metadata_callback(3, 2)(0, runtime_tests.common())


def test_previous_operands_survive_host_retirement_until_completed_exact_copy(setup, monkeypatch):
    runtime, _, _, completions = setup
    payload = object()
    with runtime.proposal_context(runtime_tests.batch()):
        runtime.metadata_callback(3, 2)(0, runtime_tests.common())
        runtime.retain_metadata(payload)
        retired = runtime._owner
    assert runtime._owner is None
    assert any(value is payload for value in runtime._pending)
    original = torch.Tensor.to
    copies = []

    def observed_copy(value, *args, **kwargs):
        if kwargs.get("device") == "cpu" and kwargs.get("copy") is True:
            assert any(item is payload for item in runtime._pending)
            copies.append(value)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", observed_copy)
    with runtime.proposal_context(runtime_tests.batch()):
        assert any(value is payload for value in runtime._pending)
        runtime.metadata_callback(3, 2)(0, runtime_tests.common())
        assert runtime._pending == [] and retired == []
    assert len(copies) == 1 and not completions
    assert not any(value is payload for value in runtime._pending)


def test_failed_exact_copy_keeps_previous_and_current_then_quarantines(setup, monkeypatch):
    runtime, _, _, _ = setup
    build(runtime)
    previous = list(runtime._pending)
    metadata = runtime_tests.common()
    original = torch.Tensor.to

    def failed_copy(value, *args, **kwargs):
        if kwargs.get("device") == "cpu" and kwargs.get("copy") is True:
            original(value, *args, **kwargs)
            raise RuntimeError("partial exact copy failure")
        return original(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", failed_copy)
    with pytest.raises(RuntimeError, match="partial exact"), runtime.proposal_context(runtime_tests.batch()):
        runtime.metadata_callback(3, 2)(0, metadata)
    assert all(any(value is old for value in runtime._pending) for old in previous)
    assert any(value is metadata for value in runtime._pending)
    assert runtime._poisoned
    with pytest.raises(RuntimeError, match="quarantined"), runtime.proposal_context(runtime_tests.batch()):
        pytest.fail("unknown completion must stay isolated")


def test_context_only_and_dummy_do_not_release_or_accumulate_window_operands(setup):
    runtime, _, _, completions = setup
    build(runtime)
    previous = [id(value) for value in runtime._pending]
    for _ in range(8):
        with runtime.proposal_context(runtime_tests.batch(), context_kv_only=True):
            assert runtime._owner == []
        with runtime.proposal_context(runtime_tests.batch(), dummy_run=True):
            assert runtime.metadata_callback(3, 2) is None
        assert [id(value) for value in runtime._pending] == previous
    assert not completions


def test_already_completed_mla_exact_copy_retires_without_second_transfer(setup, monkeypatch):
    runtime, _, _, _ = setup
    build(runtime)
    metadata = runtime_tests.common()
    completed = metadata.seq_lens[:2].to(device="cpu", dtype=torch.int32, copy=True)
    original = torch.Tensor.to

    def forbid_duplicate_copy(value, *args, **kwargs):
        assert not (kwargs.get("device") == "cpu" and kwargs.get("copy") is True)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", forbid_duplicate_copy)
    with runtime.proposal_context(runtime_tests.batch()):
        assert runtime._pending
        runtime.metadata_callback(3, 2, prepared_exact_cpu=completed)(0, metadata)
        assert runtime._pending == []


def test_scope_failure_after_completed_copy_keeps_current_operands(setup):
    runtime, _, _, _ = setup
    build(runtime)
    previous = list(runtime._pending)
    payload = object()
    with pytest.raises(RuntimeError, match="after reader enqueue"), runtime.proposal_context(runtime_tests.batch()):
        runtime.metadata_callback(3, 2)(0, runtime_tests.common())
        assert runtime._pending == []
        runtime.retain_metadata(payload)
        raise RuntimeError("after reader enqueue")
    assert any(value is payload for value in runtime._pending)
    assert not any(any(value is old for value in runtime._pending) for old in previous)
    assert runtime._poisoned


def test_stale_callback_cannot_retire_or_change_current_scope(setup):
    runtime, _, _, _ = setup
    with runtime.proposal_context(runtime_tests.batch()):
        stale = runtime.metadata_callback(3, 2)
        stale(0, runtime_tests.common())
    pending = [id(value) for value in runtime._pending]
    with runtime.proposal_context(runtime_tests.batch()):
        with pytest.raises(RuntimeError, match="stale proposal"):
            stale(0, runtime_tests.common())
        assert [id(value) for value in runtime._pending] == pending
        assert runtime._owner == []
        runtime.metadata_callback(3, 2)(0, runtime_tests.common())


def test_context_only_retention_cannot_replace_previous_unfinished_reader(setup):
    runtime, _, _, _ = setup
    build(runtime)
    previous = list(runtime._pending)
    context = object()
    with runtime.proposal_context(runtime_tests.batch(), context_kv_only=True):
        runtime.retain_metadata(context)
    assert all(any(value is old for value in runtime._pending) for old in previous)
    assert any(value is context for value in runtime._pending)


def test_real_dense_retention_cycles_end_at_the_next_completed_copy(setup):
    import gc
    import weakref

    runtime, _, _, _ = setup
    runtime._dense_reads = True

    class Payload:
        pass

    previous = None
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(4):
            payload = Payload()
            current = weakref.ref(payload)
            metadata = runtime_tests.common()
            metadata.query_start_loc_cpu = torch.tensor([0, 3, 6, 6], dtype=torch.int32)
            metadata.causal = False
            with runtime.proposal_context(runtime_tests.batch()):
                runtime.metadata_callback(3, 2)(0, metadata)
                runtime.retain_metadata(payload)
            del payload
            assert current() is not None
            if previous is not None:
                assert previous() is None, "completion must break actual read/retain/owner cycles without GC"
            previous = current
    finally:
        if was_enabled:
            gc.enable()
