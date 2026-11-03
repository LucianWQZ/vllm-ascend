# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Proposal ownership must precede a possibly partially submitted copy.

CPU dispatch substitutes expose the ownership boundary only. Actual NPU
enqueue, completion and mixed engine regression need separate device tests.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.ut.worker.v2 import test_draft_window_runtime as runtime_tests
from vllm_ascend.worker.v2.spec_decode.draft_window_dense import DenseDraftRead, prepare_dense_draft_read

setup = runtime_tests.setup


@pytest.mark.parametrize("stage", ["phase", "cohort"])
def test_copy_failure_keeps_actual_input_and_output_owned(monkeypatch, stage, setup):
    table = torch.arange(32, dtype=torch.int32).reshape(2, 16)
    source = (table[list((1, 0))], (16, 16), 128)
    exact = torch.tensor([1151, 1408], dtype=torch.int32)
    runtime, _, _, _ = setup
    submitted = []
    original_copy = torch.Tensor.copy_

    def selected(tensor):
        return tensor.dtype == (torch.bool if stage == "phase" else torch.int64)

    def after_copy(input_tensor, output_tensor):
        submitted.append((input_tensor, output_tensor))
        assert any(item is input_tensor for item in owner), "copy source not owned before submission"
        assert any(item is output_tensor for item in owner), "copy destination not owned before submission"
        raise RuntimeError("deliberate error after copy submission")

    def copy(destination, tensor, *args, **kwargs):
        output = original_copy(destination, tensor, *args, **kwargs)
        if selected(tensor):
            after_copy(tensor, destination)
        return output

    monkeypatch.setattr(torch.Tensor, "copy_", copy)
    full_rows = torch.tensor([True, True] if stage == "phase" else [False, True])
    with pytest.raises(RuntimeError, match="deliberate error"), runtime.proposal_context(runtime_tests.batch()):
        owner = runtime._owner
        assert owner is not None
        prepare_dense_draft_read(
            *source,
            exact_cpu=exact,
            query_ends_cpu=exact,
            frozen_starts=torch.where(
                full_rows, 0, torch.div(torch.clamp(exact - 1024, min=0), 128, rounding_mode="floor") * 128
            ),
            upper_cpu=exact + 3,
            query_width=4,
            causal=False,
            full_rows_cpu=full_rows,
            retain=lambda *items: owner.extend(items),
        )
    assert len(submitted) == 1
    assert runtime._poisoned
    assert all(any(item is tensor for item in owner) for tensor in submitted[0])
    with pytest.raises(RuntimeError, match="quarantined"), runtime.proposal_context(runtime_tests.batch()):
        pytest.fail("uncertain copy must not be reused")


def test_planning_copy_failure_keeps_prepared_input_owned(monkeypatch, setup):
    source = (torch.arange(16, dtype=torch.int32)[None][list((0,))], (16,), 128)
    exact = torch.tensor([1151], dtype=torch.int32)
    runtime, _, _, _ = setup
    original_copy = torch.Tensor.copy_

    def failed_copy(destination, value, *args, **kwargs):
        assert any(item is exact for item in owner)
        assert any(item is value for item in owner)
        assert any(item is destination for item in owner)
        original_copy(destination, value, *args, **kwargs)
        raise RuntimeError("deliberate error during planning copy")

    monkeypatch.setattr(torch.Tensor, "copy_", failed_copy)
    with pytest.raises(RuntimeError, match="planning copy"), runtime.proposal_context(runtime_tests.batch()):
        owner = runtime._owner
        assert owner is not None
        prepare_dense_draft_read(
            *source,
            exact_cpu=exact,
            query_ends_cpu=exact,
            frozen_starts=torch.div(torch.clamp(exact - 1024, min=0), 128, rounding_mode="floor") * 128,
            full_rows_cpu=torch.zeros_like(exact, dtype=torch.bool, device="cpu"),
            upper_cpu=exact + 1,
            query_width=4,
            causal=False,
            retain=lambda *items: owner.extend(items),
        )
    assert runtime._poisoned and any(item is exact for item in owner)


def test_metadata_length_copy_is_owned_before_partial_fill(monkeypatch, setup):
    runtime, _, _, _ = setup
    original = torch.Tensor.__setitem__
    read = DenseDraftRead(
        torch.zeros(2, 4, dtype=torch.int64),
        torch.tensor([3, 4], dtype=torch.int32),
        torch.zeros(2, 1, 1, 4, dtype=torch.bool),
        4,
        1,
        lambda *items: owner.extend(items),
    )
    submitted = []

    def filled(destination, index, values):
        result = original(destination, index, values)
        if values is read.seq_lens:
            submitted.append(destination)
            assert any(item is destination for item in owner), "metadata copy destination is not owned"
            raise RuntimeError("deliberate error after metadata fill")
        return result

    monkeypatch.setattr(torch.Tensor, "__setitem__", filled)
    with pytest.raises(RuntimeError, match="after metadata fill"), runtime.proposal_context(runtime_tests.batch()):
        owner = runtime._owner
        assert owner is not None
        owner.append(read)
        read.apply(SimpleNamespace(num_reqs=3))
    assert len(submitted) == 1
    assert runtime._poisoned
    assert any(item is submitted[0] for item in owner)
