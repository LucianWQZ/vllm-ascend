# SPDX-License-Identifier: Apache-2.0
"""Real ordinary Runtime copies retire earlier readers, without a new wait."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from vllm_ascend.worker.v2.spec_decode.draft_window_runtime import DraftWindowRuntime  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires an Ascend NPU")


@pytest.fixture
def prepared(monkeypatch):
    # This leaf test supplies one already-qualified direct group. The real
    # Runtime, metadata copies, reader kernels and completion are not mocked.
    monkeypatch.setattr(DraftWindowRuntime, "_validate_route", staticmethod(lambda owner: (0,)))
    table = torch.arange(8, dtype=torch.int32, device="npu").reshape(1, 8)
    spec = SimpleNamespace(
        method="dflash",
        draft_model_config=SimpleNamespace(enforce_eager=True),
        speculative_config=SimpleNamespace(enforce_eager=None),
        vllm_config=SimpleNamespace(max_concurrent_batches=2),
        num_query_per_req=3,
        num_speculative_steps=3,
        supports_decode_metadata_update=False,
        block_tables=SimpleNamespace(num_blocks=SimpleNamespace(np=np.array([[8]])), kernel_block_sizes=[4]),
    )
    runtime = DraftWindowRuntime(spec, 8)
    batch = SimpleNamespace(
        num_reqs=1,
        req_ids=["one"],
        idx_mapping_np=np.array([0]),
        num_computed_prefill_tokens_np=np.array([24]),
        prefill_len_np=np.array([16]),
        num_scheduled_tokens=np.array([3]),
        num_draft_tokens_per_req=np.array([2]),
        is_prefilling_np=np.array([False]),
    )

    def metadata():
        offsets = torch.tensor([0, 3], dtype=torch.int32)
        return SimpleNamespace(
            num_reqs=1,
            seq_lens=torch.tensor([12], dtype=torch.int32, device="npu"),
            seq_lens_cpu_upper_bound=torch.tensor([14], dtype=torch.int32),
            block_table_tensor=table,
            query_start_loc=offsets.npu(),
            query_start_loc_cpu=offsets,
            causal=False,
        )

    spec.input_batch = batch
    return runtime, batch, metadata


def test_existing_exact_copy_completes_previous_actual_window_reader(prepared):
    runtime, batch, metadata = prepared
    first, second = metadata(), metadata()  # No blocking allocation between the two proposals.
    observed = torch.empty((1, 2), dtype=torch.int32, device="npu")
    left = torch.ones((2048, 2048), dtype=torch.bfloat16, device="npu")
    scratch = torch.empty_like(left)
    completion = torch.npu.Event()
    with runtime.proposal_context(batch):
        runtime.metadata_callback(3, 1)(0, first)
        runtime.retain_metadata((left, scratch, observed))
        for _ in range(8):
            torch.mm(left, left, out=scratch)
        observed.copy_(first.block_table_tensor)
        completion.record(torch.npu.current_stream())
    previous = list(runtime._pending)
    with runtime.proposal_context(batch):
        assert [id(value) for value in runtime._pending] == [id(value) for value in previous]
        runtime.metadata_callback(3, 1)(0, second)
        # query never waits: the production exact copy has already completed
        # the previous stream's real table reader and pressure operations.
        assert completion.query()
        assert not runtime._pending
    torch.testing.assert_close(observed.cpu(), torch.tensor([[1, 2]], dtype=torch.int32), atol=0, rtol=0)
    torch.npu.current_stream().synchronize()  # Diagnostic cleanup of the final proposal only.


def test_partial_real_pinned_length_copy_keeps_operands_quarantined(prepared, monkeypatch):
    runtime, batch, metadata = prepared
    common = metadata()
    original = torch.Tensor.copy_
    submitted, cleanup = [], []

    def copy(destination, source, *args, **kwargs):
        result = original(destination, source, *args, **kwargs)
        if source.device.type == "cpu" and source.is_pinned() and destination.is_npu:
            assert all(any(value is operand for value in runtime._owner) for operand in (source, destination))
            submitted.extend((source, destination))
            event = torch.npu.Event()
            event.record(torch.npu.current_stream())
            cleanup.append(event)
            raise RuntimeError("after actual pinned copy")
        return result

    monkeypatch.setattr(torch.Tensor, "copy_", copy)
    try:
        with pytest.raises(RuntimeError, match="actual pinned"), runtime.proposal_context(batch):
            runtime.metadata_callback(3, 1)(0, common)
        assert len(submitted) == 2
        assert all(any(value is operand for value in runtime._pending) for operand in submitted)
        with pytest.raises(RuntimeError, match="quarantined"), runtime.proposal_context(batch):
            pytest.fail("uncertain submission cannot be reused")
    finally:
        for event in cleanup:
            event.synchronize()


def test_real_fp32_kernel_failure_keeps_submitted_outputs(prepared, monkeypatch):
    from vllm_ascend.ops.triton.v2.spec_decode import draft_window
    from vllm_ascend.worker.v2.spec_decode.draft_window_dense import gather_dense_draft_kv

    runtime, batch, metadata = prepared
    runtime._dense_reads = True
    common = metadata()
    key = torch.arange(256, dtype=torch.bfloat16, device="npu").reshape(8, 4, 1, 8)
    value = torch.ones_like(key)
    original = draft_window._gather_draft_kv_fp32
    outputs, cleanup = [], []

    class FailedKernel:
        def __getitem__(self, grid):
            launch = original[grid]

            def call(*args):
                assert all(any(value is operand for value in runtime._owner) for operand in args[3:5])
                outputs.extend(args[3:5])
                launch(*args)
                event = torch.npu.Event()
                event.record(torch.npu.current_stream())
                cleanup.append(event)
                raise RuntimeError("after actual FP32 gather")

            return call

    monkeypatch.setattr(draft_window, "_gather_draft_kv_fp32", FailedKernel())
    try:
        with pytest.raises(RuntimeError, match="actual FP32"), runtime.proposal_context(batch):
            runtime.metadata_callback(3, 1)(0, common)
            gather_dense_draft_kv(common._draft_window_device_read, key, value)
        assert len(outputs) == 2
        assert all(any(value is output for value in runtime._pending) for output in outputs)
        with pytest.raises(RuntimeError, match="quarantined"), runtime.proposal_context(batch):
            pytest.fail("failed reader cannot be reused")
    finally:
        for event in cleanup:
            event.synchronize()
    torch.testing.assert_close(outputs[0][0, :8].cpu().flatten(), torch.arange(32, 96, dtype=torch.float32))
    assert outputs[1].eq(1).all()
