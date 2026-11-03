# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Native band projection at the ordinary metadata extension point."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_ascend.attention.utils import _select_seq_lens
from vllm_ascend.worker.v2.spec_decode import draft_window_runtime as runtime_module
from vllm_ascend.worker.v2.spec_decode.draft_window_policy import resolve_draft_window_policy


@pytest.fixture
def prepared(monkeypatch):
    def make(span=128, continuing=(False, False), causal=True, groups=(0,), window=1024):
        # Null prefix columns are valid native-SWA storage, not Full residency.
        canonical = torch.arange(80, dtype=torch.int32).reshape(4, 20)
        canonical[:, :2] = 0
        rows = np.array([3, 1])
        seq_lens = torch.tensor([1151, 1152], dtype=torch.int32)
        common = SimpleNamespace(
            num_reqs=2,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens + 7,
            _seq_lens_cpu=seq_lens + 7,
            seq_lens_cpu_upper_bound=torch.tensor([1152, 1159], dtype=torch.int32),
            block_table_tensor=canonical[rows].clone(),
            query_start_loc_cpu=torch.tensor([0, 8, 16], dtype=torch.int32),
            max_seq_len=1159,
            slot_mapping=torch.arange(16),
            positions=torch.arange(16) + 1136,
            causal=causal,
        )
        batch = SimpleNamespace(
            num_reqs=2,
            req_ids=["new", "other"],
            idx_mapping_np=rows,
            idx_mapping=torch.from_numpy(rows),
            num_computed_prefill_tokens_np=np.zeros(2, dtype=np.int32),
            prefill_len_np=np.where(continuing, 100, 8),
            num_scheduled_tokens=np.full(2, 8),
            num_draft_tokens_per_req=None,
            is_prefilling_np=np.asarray(continuing),
        )
        speculator = SimpleNamespace(
            method="dflash",
            num_query_per_req=8,
            num_speculative_steps=7,
            block_tables=SimpleNamespace(
                block_tables=[SimpleNamespace(gpu=canonical) for _ in range(max(groups) + 1)],
                kernel_block_sizes=[span] * (max(groups) + 1),
                num_blocks=SimpleNamespace(np=np.full((max(groups) + 1, 4), 20)),
            ),
            draft_model_config=SimpleNamespace(enforce_eager=True),
            speculative_config=SimpleNamespace(enforce_eager=True, parallel_drafting=True, use_dspark=lambda: False),
            vllm_config=SimpleNamespace(max_concurrent_batches=2),
        )

        def validate(runtime, spec):
            runtime.native_swa_causal = dict.fromkeys(groups, causal)
            return groups

        completions = []

        class Completion:
            complete = False

            def __init__(self):
                completions.append(self)

            def record(self, stream):
                self.stream = stream

            def query(self):
                return self.complete

            def synchronize(self):
                raise AssertionError("No wait is needed with a free proposal lane")

        monkeypatch.setattr(runtime_module.DraftWindowRuntime, "_validate_route", validate)
        monkeypatch.setattr(torch.npu, "Event", Completion)
        monkeypatch.setattr(torch.npu, "current_stream", lambda: None)
        runtime = runtime_module.DraftWindowRuntime(
            speculator, resolve_draft_window_policy(requested_window_size=window, method="dflash")
        )
        return runtime, speculator, batch, common, canonical, completions

    return make


def build_native(common, speculator):
    # The existing builder selects its read certificate before deriving its
    # tensor/list fields. It still owns native mask and query/write fields.
    seq_lens = _select_seq_lens(common, None, speculator.speculative_config, SimpleNamespace(model_config=None))
    return SimpleNamespace(
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens,
        seq_lens_gpu=common.seq_lens,
        seq_lens_list=seq_lens.tolist(),
        block_tables=common.block_table_tensor,
        slot_mapping=common.slot_mapping,
        positions=common.positions,
        causal=common.causal,
        native_window=4096,
    )


@pytest.mark.parametrize("span,starts", [(128, [0, 128]), (64, [64, 128])])
def test_current_exact_rejection_and_group_geometry(prepared, span, starts):
    runtime, spec, batch, common, canonical, _ = prepared(span)
    target = dict(vars(common))
    before = canonical.clone()
    with runtime.proposal_context(batch):
        owner = runtime._owner
        runtime.metadata_callback(8, 2)(0, common)
        result = build_native(common, spec)
        runtime.retain_metadata(result)
        assert result.seq_lens_list == [1151 - starts[0], 1152 - starts[1]]
        assert result.slot_mapping is target["slot_mapping"]
        assert result.positions is target["positions"]
        assert result.native_window == 4096
        for i, row in enumerate([3, 1]):
            end_column = (int(target["seq_lens"][i]) + span - 1) // span
            expected = before[row, starts[i] // span : end_column]
            torch.testing.assert_close(result.block_tables[i, : len(expected)], expected, rtol=0, atol=0)
        assert any(item is result for item in owner)
        assert target["seq_lens"].tolist() == [1151, 1152]
        torch.testing.assert_close(target["block_table_tensor"], before[batch.idx_mapping_np], rtol=0, atol=0)
    torch.testing.assert_close(canonical, before, rtol=0, atol=0)


def test_native_band_read_set_intersection_uses_actual_query_positions():
    for end in [8, 1151, 1152, 4095, 4096, 33023]:
        for span in [64, 128]:
            for window in [8, 1000, 1024, 4096, 4097]:
                start = max(end - window, 0) // span * span
                for q in range(end - 8, end):
                    native_absolute = set(range(max(q - 4096, 0), q + 1))
                    local_q = q - start
                    actual = {start + k for k in range(max(local_q - 4096, 0), local_q + 1)}
                    assert actual == {k for k in native_absolute if k >= start}


@pytest.mark.parametrize("continuing", [(True, True), (True, False)])
def test_continuing_prefill_preserves_native_full_read(prepared, continuing):
    runtime, spec, batch, common, _, _ = prepared(continuing=continuing)
    old_table, old_lengths = common.block_table_tensor, common.seq_lens
    with runtime.proposal_context(batch):
        runtime.metadata_callback(8, 2)(0, common)
        result = build_native(common, spec)
        assert result.seq_lens_list[0] == 1151
        if all(continuing):
            assert result.block_tables is old_table
            assert result.seq_lens_gpu is old_lengths
        else:
            assert result.seq_lens_list[1] == 1024


@pytest.mark.parametrize("continuing", [False, True])
@pytest.mark.parametrize("causal", [False, True])
def test_native_metadata_uses_one_actual_snapshot_for_all_groups(prepared, monkeypatch, continuing, causal):
    runtime, spec, batch, common, _, _ = prepared(continuing=(continuing, continuing), causal=causal, groups=(0, 1))
    original = torch.Tensor.to
    copies = []

    def counted(tensor, *args, **kwargs):
        if kwargs.get("device") == "cpu" and kwargs.get("copy") is True:
            copies.append(tuple(tensor.shape))
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", counted)
    with runtime.proposal_context(batch):
        callback = runtime.metadata_callback(8, 2)
        second = SimpleNamespace(**vars(common))
        for gid, item in enumerate((common, second)):
            callback(gid, item)
            result = build_native(item, spec)
            assert result.causal is causal
            runtime.retain_metadata(result)
        assert runtime._build_count == 1
        assert copies == [(2,)]


@pytest.mark.parametrize("changed", ["causal", "query_width", "padded", "dtype", "extent"])
def test_native_projection_rejects_wrong_actual_preparation(prepared, changed):
    runtime, spec, batch, common, _, _ = prepared()
    if changed == "causal":
        common.causal = False
    elif changed == "query_width":
        common.query_start_loc_cpu[1] = 7
    elif changed == "padded":
        common.num_reqs = 3
    elif changed == "dtype":
        common.seq_lens = common.seq_lens.long()
    else:
        common.seq_lens = torch.cat((common.seq_lens, torch.zeros(1, dtype=torch.int32)))
    with pytest.raises(ValueError, match="Native SWA"), runtime.proposal_context(batch):
        runtime.metadata_callback(8, 2)(0, common)
    assert runtime._poisoned


def test_native_projection_must_cover_every_query(prepared):
    runtime, spec, batch, common, _, _ = prepared(window=1)
    common.seq_lens = torch.tensor([1025, 1025], dtype=torch.int32)
    common.seq_lens_cpu_upper_bound = common.seq_lens + 1
    with pytest.raises(ValueError, match="every native query"), runtime.proposal_context(batch):
        runtime.metadata_callback(8, 2)(0, common)


def test_next_native_proposal_owns_independent_read_buffers(prepared):
    runtime, spec, batch, common, canonical, completions = prepared()
    with runtime.proposal_context(batch):
        runtime.metadata_callback(8, 2)(0, common)
        first_owner = runtime._owner
    first = common.seq_lens.clone(), common.block_table_tensor.clone()
    second = SimpleNamespace(**vars(common))
    second.seq_lens = torch.tensor([1159, 1160], dtype=torch.int32)
    second.block_table_tensor = canonical[batch.idx_mapping_np.tolist()].clone()
    second.seq_lens_cpu_upper_bound = second.seq_lens + 3
    del second._draft_window_exact_cpu
    with runtime.proposal_context(batch):
        runtime.metadata_callback(8, 2)(0, second)
    torch.testing.assert_close(common.seq_lens, first[0], rtol=0, atol=0)
    torch.testing.assert_close(common.block_table_tensor, first[1], rtol=0, atol=0)
    assert not completions and first_owner == []


def test_native_callback_cannot_cross_proposal_ownership(prepared):
    runtime, spec, batch, common, _, _ = prepared()
    with runtime.proposal_context(batch):
        callback = runtime.metadata_callback(8, 2)
        callback(0, common)
    with pytest.raises(RuntimeError, match="stale proposal"), runtime.proposal_context(batch):
        callback(0, common)


def test_builder_failure_keeps_projected_read_operands_owned(prepared):
    runtime, spec, batch, common, _, _ = prepared()
    compact = common.block_table_tensor
    with pytest.raises(RuntimeError, match="builder failed"), runtime.proposal_context(batch):
        owner = runtime._owner
        runtime.metadata_callback(8, 2)(0, common)
        assert any(item is common for item in owner)
        assert any(item is compact for item in owner)
        raise RuntimeError("builder failed")
    assert runtime._poisoned
