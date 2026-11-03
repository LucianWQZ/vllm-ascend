# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Real kernel operands survive an ordinary Runtime preparation failure."""

from typing import Any

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from tests.e2e.nightly.single_node.ops.singlecard_ops import (  # noqa: E402
    test_draft_window_completion_boundary as runtime_tests,
)

prepared = runtime_tests.prepared
pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires an Ascend NPU")


@pytest.mark.parametrize("operation", ["gather"])
def test_private_kernel_outputs_survive_exception_after_real_enqueue(monkeypatch, operation, prepared):
    from vllm_ascend.ops.triton.v2.spec_decode import draft_window
    from vllm_ascend.worker.v2.spec_decode.draft_window_dense import DenseDraftRead, gather_dense_draft_kv

    runtime, batch, _ = prepared
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
                assert all(any(value is output for value in owner) for output in outputs)
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
        runtime.proposal_context(batch),
    ):
        owner = runtime._owner
        assert owner is not None
        retain = lambda *objects: owner.extend(objects)
        if operation == "gather":
            slots = torch.tensor([[1, 2, 3]], dtype=torch.int64, device="npu")
            read = DenseDraftRead(slots, torch.tensor([3]), torch.empty(0), 3, 1, retain)
            gather_dense_draft_kv(
                read,
                torch.arange(64, dtype=torch.bfloat16, device="npu").reshape(2, 4, 1, 8),
                torch.ones((2, 4, 1, 8), dtype=torch.bfloat16, device="npu"),
            )

    assert diagnostic_events and held_outputs
    assert runtime._poisoned
    assert all(any(value is output for value in owner) for output in held_outputs)
    with pytest.raises(RuntimeError, match="quarantined"), runtime.proposal_context(batch):
        pytest.fail("uncertain kernel submission cannot be reused")
    diagnostic_events[0].synchronize()
    if operation == "gather":
        torch.testing.assert_close(held_outputs[0].cpu().flatten(), torch.arange(8, 32, dtype=torch.float32))
        assert held_outputs[1].eq(1).all()
    # Even an external diagnostic wait does not make production's unknown
    # completion contract safe to reuse.
    with pytest.raises(RuntimeError, match="quarantined"), runtime.proposal_context(batch):
        pytest.fail("uncertain kernel submission cannot be reused")
