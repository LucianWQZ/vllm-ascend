# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quarantine actual private copies that return before a Host exception."""

from types import SimpleNamespace

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from vllm_ascend.worker.v2.spec_decode.draft_window_dense import DenseDraftRead, prepare_dense_draft_read  # noqa: E402
from vllm_ascend.worker.v2.spec_decode.draft_window_lifetime import DraftWindowOwnerPool  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires an Ascend NPU")


@pytest.mark.parametrize("stage", ["phase", "cohort"])
def test_real_private_copy_failure_retains_its_actual_operands(monkeypatch, stage):
    table = torch.arange(32, dtype=torch.int32, device="npu").reshape(2, 16)
    before = table.clone()
    source = (table[list((1, 0))], (16, 16), 128)
    pool = DraftWindowOwnerPool(1)
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
        assert any(item is input_tensor for item in owner.retained), "copy source missing from active owner"
        assert any(item is output_tensor for item in owner.retained), "copy destination missing from active owner"
        raise RuntimeError("deliberate Host failure after real NPU copy")

    def copy(destination, tensor, *args, **kwargs):
        output = original_copy(destination, tensor, *args, **kwargs)
        if selected(tensor):
            after_copy(tensor, destination)
        return output

    monkeypatch.setattr(torch.Tensor, "copy_", copy)
    full_rows = torch.tensor([True, True] if stage == "phase" else [False, True])
    try:
        with pytest.raises(RuntimeError, match="after real NPU copy"), pool.operation() as owner:
            prepare_dense_draft_read(
                *source,
                exact_cpu=torch.tensor([1151, 1408], dtype=torch.int32),
                query_ends_cpu=torch.tensor([1151, 1408], dtype=torch.int32),
                frozen_starts=torch.where(full_rows, 0, torch.tensor([0, 384], dtype=torch.int32)),
                upper_cpu=torch.tensor([1154, 1411], dtype=torch.int32),
                query_width=4,
                causal=False,
                full_rows_cpu=full_rows,
                retain=lambda *items: owner.retained.extend(items),
            )
        assert len(submitted) == 1
        assert pool.poisoned
        assert all(any(item is tensor for item in owner.retained) for tensor in submitted[0])
        with pytest.raises(RuntimeError, match="quarantined"):
            pool.acquire()
    finally:
        # Diagnostic cleanup only. The product never receives these events
        # and must keep its unknown-completion owner quarantined.
        for event in events:
            event.synchronize()
    torch.testing.assert_close(table, before, atol=0, rtol=0)


def test_cpu_metadata_fill_failure_keeps_device_operands_quarantined(monkeypatch):
    pool = DraftWindowOwnerPool(1)
    read = DenseDraftRead(
        torch.zeros(2, 4, dtype=torch.int64, device="npu"),
        torch.tensor([3, 4], dtype=torch.int32, device="cpu"),
        torch.zeros(2, 1, 1, 4, dtype=torch.bool, device="npu"),
        4,
        1,
        lambda *items: owner.retained.extend(items),
    )
    original = torch.Tensor.__setitem__
    events, submitted = [], []

    def filled(destination, index, values):
        result = original(destination, index, values)
        if values is read.seq_lens:
            event = torch.npu.Event()
            event.record(torch.npu.current_stream())
            events.append(event)
            submitted.append(destination)
            assert any(item is destination for item in owner.retained), "metadata destination is not owned"
            raise RuntimeError("deliberate Host failure after real metadata fill")
        return result

    monkeypatch.setattr(torch.Tensor, "__setitem__", filled)
    try:
        with pytest.raises(RuntimeError, match="after real metadata fill"), pool.operation() as owner:
            owner.retained.append(read)
            read.apply(SimpleNamespace(num_reqs=3))
        assert len(submitted) == 1
        assert pool.poisoned
        assert any(item is submitted[0] for item in owner.retained)
        with pytest.raises(RuntimeError, match="quarantined"):
            pool.acquire()
    finally:
        for event in events:
            event.synchronize()
