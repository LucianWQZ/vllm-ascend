# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quarantine actual private copies that return before a Host exception."""

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from tests.e2e.nightly.single_node.ops.singlecard_ops import (  # noqa: E402
    test_draft_window_completion_boundary as runtime_tests,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_dense import prepare_dense_draft_read  # noqa: E402

prepared = runtime_tests.prepared

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires an Ascend NPU")


@pytest.mark.parametrize("stage", ["phase", "cohort"])
def test_real_private_copy_failure_retains_its_actual_operands(monkeypatch, stage, prepared):
    table = torch.arange(32, dtype=torch.int32, device="npu").reshape(2, 16)
    before = table.clone()
    source = (table[list((1, 0))], (16, 16), 128)
    runtime, batch, _ = prepared
    events, submitted = [], []
    original_copy = torch.Tensor.copy_

    def selected(tensor):
        dtype = torch.bool if stage == "phase" else torch.int64
        return tensor.device.type == "cpu" and tensor.dtype == dtype

    def after_copy(input_tensor, output_tensor):
        assert output_tensor.is_npu
        event = torch.npu.Event()
        event.record(torch.npu.current_stream())
        events.append(event)
        submitted.append((input_tensor, output_tensor))
        assert any(item is input_tensor for item in owner), "copy source missing from active owner"
        assert any(item is output_tensor for item in owner), "copy destination missing from active owner"
        raise RuntimeError("deliberate Host failure after real NPU copy")

    def copy(destination, tensor, *args, **kwargs):
        output = original_copy(destination, tensor, *args, **kwargs)
        if selected(tensor):
            after_copy(tensor, destination)
        return output

    monkeypatch.setattr(torch.Tensor, "copy_", copy)
    full_rows = torch.tensor([True, True] if stage == "phase" else [False, True])
    try:
        with pytest.raises(RuntimeError, match="after real NPU copy"), runtime.proposal_context(batch):
            owner = runtime._owner
            assert owner is not None
            prepare_dense_draft_read(
                *source,
                exact_cpu=torch.tensor([1151, 1408], dtype=torch.int32),
                query_ends_cpu=torch.tensor([1151, 1408], dtype=torch.int32),
                frozen_starts=torch.where(full_rows, 0, torch.tensor([0, 384], dtype=torch.int32)),
                upper_cpu=torch.tensor([1154, 1411], dtype=torch.int32),
                query_width=4,
                causal=False,
                full_rows_cpu=full_rows,
                retain=lambda *items: owner.extend(items),
            )
        assert len(submitted) == 1
        assert runtime._poisoned
        assert all(any(item is tensor for item in owner) for tensor in submitted[0])
        with pytest.raises(RuntimeError, match="quarantined"), runtime.proposal_context(batch):
            pytest.fail("uncertain submission must not be reused")
    finally:
        # Diagnostic cleanup only. The product never receives these events
        # and must keep its unknown-completion owner quarantined.
        for event in events:
            event.synchronize()
    torch.testing.assert_close(table, before, atol=0, rtol=0)
