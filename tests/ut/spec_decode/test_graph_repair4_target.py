# SPDX-License-Identifier: Apache-2.0
"""CPU actual-forward tests; no result here establishes NPU or post4 readiness."""

import importlib.util
import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("repair4_target_tested", ROOT / "tools/p7x_graph_repair4_target.py")
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


def inputs():
    return {
        "input_ids": torch.tensor([1, 2]),
        "positions": torch.tensor([2, 3]),
        "query_start_loc": torch.tensor([0, 2]),
        "seq_lens": torch.tensor([4]),
        "history": (7, 8),
    }


def execute(*, tweak=0.0, outside=False, block_ids=None):
    caches = {"layer/K": torch.arange(16.0).reshape(4, 2, 1, 2), "layer/V": torch.ones(4, 2, 1, 2)}
    recorder = tool.FullForwardCheckpointer(fixture_id="actual-toy-transformer:fixed-history")
    calls = []
    data = inputs()

    def forward():
        calls.append("forward")
        # Exercise the backbone computation and KV read/write, not an LM head.
        q = data["input_ids"].float().unsqueeze(1).repeat(1, 2)
        k = caches["layer/K"].reshape(8, 2)
        v = caches["layer/V"].reshape(8, 2)
        read_k, read_v = k, v
        if block_ids is not None:
            read_k = caches["layer/K"].index_select(0, block_ids["layer/K"]).reshape(-1, 2)
            read_v = caches["layer/V"].index_select(0, block_ids["layer/V"]).reshape(-1, 2)
        hidden = torch.softmax(q @ read_k.T, dim=-1) @ read_v
        k[2:4].copy_(q)
        v[2:4].copy_(hidden)
        if outside:
            k[0].add_(1)
        return hidden + tweak

    result = recorder.run(
        forward,
        inputs=data,
        kv_access=lambda: caches,
        write_slots={name: torch.tensor([2, 3]) for name in caches},
        synchronize=lambda: calls.append("sync"),
        block_ids=block_ids,
        declared_reads=(
            {name: {"table": ids.unsqueeze(0), "lengths": [len(ids) * 2]} for name, ids in block_ids.items()}
            if block_ids is not None
            else None
        ),
    )
    return recorder, result, caches, calls


def compare(a, b, **kwargs):
    return tool.compare_records(a, b, tolerance_reason="CPU deterministic fixture", **kwargs)


def test_full_forward_runs_once_and_snapshots_are_independent():
    recorder, output, caches, calls = execute()
    record = recorder.records[0]
    assert calls == ["sync", "forward", "sync"]
    assert record["status"] == "CAPTURED"
    assert record["output"]["tensors"][""]["data"].shape == (2, 2)
    assert record["kv_before"]["tensors"]["/layer~1K"]["data"].flatten()[4] == 4
    saved_output = record["output"]["tensors"][""]["data"].clone()
    output.fill_(99)
    caches["layer/K"].zero_()
    assert torch.equal(record["output"]["tensors"][""]["data"], saved_output)
    assert record["kv_after"]["tensors"]["/layer~1K"]["data"].sum() > 0
    other, *_ = execute()
    report = compare(record, other.records[0])
    assert report["observed_forward_pair"] == "PASS"
    assert report["live_full_target"] == "OPEN"
    assert report["update_effective_generation"] == "UNKNOWN"


@pytest.mark.parametrize("field", ["inputs", "kv_before"])
def test_unaligned_real_inputs_or_pre_kv_remain_open(field):
    left, *_ = execute()
    right, *_ = execute()
    entry = next(iter(right.records[0][field]["tensors"].values()))
    entry["data"].flatten()[0] += 1
    result = compare(left.records[0], right.records[0])
    assert result["observed_forward_pair"] == "OPEN"
    assert "unaligned" in result["errors"][0]


@pytest.mark.parametrize("field", ["kv_before", "kv_after", "write_slots"])
def test_missing_kv_or_write_coverage_never_passes(field):
    left, *_ = execute()
    right, *_ = execute()
    right.records[0][field]["tensors"].pop("/layer~1V")
    assert compare(left.records[0], right.records[0])["observed_forward_pair"] == "INVALID_EVIDENCE"


def test_output_difference_is_detected_and_explicit_tolerance_is_used():
    left, *_ = execute()
    right, *_ = execute(tweak=0.01)
    assert compare(left.records[0], right.records[0])["observed_forward_pair"] == "FAIL"
    assert compare(left.records[0], right.records[0], atol=0.02)["observed_forward_pair"] == "PASS"
    assert compare(left.records[0], right.records[0], atol=float("nan"))["observed_forward_pair"] == "INVALID_EVIDENCE"


def test_wrong_post_kv_and_out_of_write_slot_corruption_are_detected():
    left, *_ = execute()
    right, *_ = execute()
    right.records[0]["kv_after"]["tensors"]["/layer~1V"]["data"][1, 0] += 1
    assert compare(left.records[0], right.records[0])["observed_forward_pair"] == "FAIL"
    # Even identical corruption in both arms is not equivalence success.
    a, *_ = execute(outside=True)
    b, *_ = execute(outside=True)
    report = compare(a.records[0], b.records[0])
    assert report["observed_forward_pair"] == "FAIL"
    assert any("outside" in error for error in report["errors"])


@pytest.mark.parametrize("field", ["output", "kv_before", "kv_after"])
def test_nonfinite_evidence_is_rejected(field):
    left, *_ = execute()
    right, *_ = execute()
    next(iter(right.records[0][field]["tensors"].values()))["data"].flatten()[0] = float("nan")
    assert compare(left.records[0], right.records[0])["observed_forward_pair"] == "INVALID_EVIDENCE"


def test_mapped_block_selection_includes_full_read_prefix_padding_and_write_blocks():
    # Includes a padding row reading block zero, a non-contiguous long history,
    # and a write to a block not read by the current forward.
    ids = tool.select_mapped_blocks(
        torch.tensor([[7, 3, 5, 99], [0, 99, 99, 99]]),
        [5, 1],
        torch.tensor([16, -1]),
        num_blocks=12,
        block_size=2,
    )
    assert ids.tolist() == [0, 3, 5, 7, 8]


@pytest.mark.parametrize("kind", ["missing_read", "bad_read", "bad_write", "wrong_dtype"])
def test_invalid_mapped_coverage_fails_closed(kind):
    table, lengths, slots = torch.tensor([[1, 2]]), [4], torch.tensor([0])
    if kind == "missing_read":
        lengths = [5]
    elif kind == "bad_read":
        table[0, 1] = 4
    elif kind == "bad_write":
        slots[0] = 8
    else:
        slots = slots.float()
    with pytest.raises(tool.EvidenceError):
        tool.select_mapped_blocks(table, lengths, slots, num_blocks=4, block_size=2)


def test_selected_blocks_copy_complete_blocks_and_keep_physical_write_mapping():
    ids = {"layer/K": torch.tensor([1, 3]), "layer/V": torch.tensor([1, 3])}
    a, *_ = execute(block_ids=ids)
    b, *_ = execute(block_ids=ids)
    record = a.records[0]
    assert record["kv_before"]["tensors"]["/layer~1K"]["data"].shape == (2, 2, 1, 2)
    assert record["write_coverage"]["/layer~1K"]["changed_outside_write_slots"] == 0
    report = compare(record, b.records[0])
    assert report["observed_forward_pair"] == "PASS"
    assert report["outside_selected_blocks"] == "UNOBSERVED"
    assert report["kv_scope"] == "FULL_DECLARED_READ_WRITE_BLOCKS"
    assert report["alignment"] == "MATCHED_COMPLETE_DECLARED_PRE_KV"
    assert report["live_readiness"] == "NOT_ESTABLISHED"


def test_checker_recomputes_read_coverage_and_rejects_missing_read_block():
    ids = {"layer/K": torch.tensor([1, 3]), "layer/V": torch.tensor([1, 3])}
    a, *_ = execute(block_ids=ids)
    b, *_ = execute(block_ids=ids)
    b.records[0]["declared_reads"]["tensors"]["/layer~1K/table"]["data"][0, 0] = 2
    result = compare(a.records[0], b.records[0])
    assert result["observed_forward_pair"] == "INVALID_EVIDENCE"
    assert "coverage" in result["errors"][0]


def test_budget_preflight_does_not_execute_or_mutate_the_model():
    recorder = tool.FullForwardCheckpointer(fixture_id="x", max_bytes=10)
    cache = torch.ones(4, 2, 1, 2)
    calls = []
    with pytest.raises(tool.EvidenceError, match="budget"):
        recorder.run(
            lambda: calls.append("forward"),
            inputs=inputs(),
            kv_access=lambda: {"K": cache},
            write_slots={"K": torch.tensor([0])},
            synchronize=lambda: None,
        )
    assert not calls and torch.equal(cache, torch.ones_like(cache))


def test_forward_exception_is_preserved_without_post_failure_synchronization():
    recorder = tool.FullForwardCheckpointer(fixture_id="x")
    calls = []
    error = RuntimeError("real graph update failed")

    def fail():
        calls.append("forward")
        raise error

    with pytest.raises(RuntimeError) as caught:
        recorder.run(
            fail,
            inputs=inputs(),
            kv_access=lambda: {"K": torch.ones(4, 2, 1, 2)},
            write_slots={"K": torch.tensor([0])},
            synchronize=lambda: calls.append("sync"),
        )
    assert caught.value is error
    assert calls == ["sync", "forward"]
    assert recorder.records[0]["status"] == "FORWARD_FAILED"


@pytest.mark.parametrize("failure_at", [1, 2])
def test_explicit_runner_propagates_native_sync_failures(failure_at):
    recorder = tool.FullForwardCheckpointer(fixture_id="x")
    calls = {"sync": 0, "forward": 0}
    error = RuntimeError("injected native synchronization failure")
    cache = torch.ones(4, 2, 1, 2)

    def sync():
        calls["sync"] += 1
        if calls["sync"] == failure_at:
            raise error

    def forward():
        calls["forward"] += 1
        return torch.ones(2, 2)

    with pytest.raises(RuntimeError) as caught:
        recorder.run(
            forward,
            inputs=inputs(),
            kv_access=lambda: {"K": cache},
            write_slots={"K": torch.tensor([0])},
            synchronize=sync,
        )
    assert caught.value is error
    assert calls["forward"] == failure_at - 1
    if failure_at == 2:
        assert recorder.records[0]["status"] == "CORE_RUNTIME_FAILED"


@pytest.mark.parametrize("failure_at", [1, 5])
def test_explicit_runner_propagates_native_snapshot_failures(monkeypatch, failure_at):
    original_snapshot = tool._snapshot
    calls = {"snapshot": 0}
    error = RuntimeError("injected D2H failure")

    def snapshot(*args, **kwargs):
        calls["snapshot"] += 1
        if calls["snapshot"] == failure_at:
            raise error
        return original_snapshot(*args, **kwargs)

    monkeypatch.setattr(tool, "_snapshot", snapshot)
    with pytest.raises(RuntimeError) as caught:
        execute()
    assert caught.value is error


def test_alias_topology_and_backing_replacement_are_not_silently_accepted():
    a, *_ = execute()
    b, *_ = execute()
    b.records[0]["kv_before"]["tensors"]["/layer~1V"]["alias"] = 0
    assert compare(a.records[0], b.records[0])["observed_forward_pair"] == "OPEN"
    b, *_ = execute()
    b.records[0]["kv_after"]["tensors"]["/layer~1V"]["address"] += 1
    assert compare(a.records[0], b.records[0])["observed_forward_pair"] == "INVALID_EVIDENCE"


@dataclass(frozen=True)
class Descriptor:
    num_tokens: int = 2
    num_reqs: int = 1
    uniform_token_count: int = 2
    max_query_len: int = 2
    num_active_loras: int = 0
    num_ubatches: int = 1

    @property
    def cg_mode(self):
        return SimpleNamespace(name="FULL")


class FakeManager:
    def run_fullgraph(self, desc):
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        self.layer.kv_cache[0][0].fill_(3)
        return self.output


class FakeRunner:
    def prepare_inputs(self):
        return self.batch


def live_fixture(monkeypatch):
    runner = FakeRunner()
    data = inputs()
    runner.input_buffers = SimpleNamespace(
        input_ids=data["input_ids"], positions=data["positions"], is_padding=torch.tensor([False, False])
    )
    runner.batch = SimpleNamespace(
        **vars(runner.input_buffers),
        is_dummy=False,
        num_tokens_after_padding=2,
        num_reqs_after_padding=1,
        num_tokens=2,
        num_reqs=1,
        query_start_loc=data["query_start_loc"],
        seq_lens=data["seq_lens"],
        idx_mapping=torch.tensor([0]),
        req_ids=["r"],
        num_computed_tokens_np=[2],
    )
    k, v = torch.ones(8, 2, 1, 2), torch.ones(8, 2, 1, 2)
    from vllm_ascend.attention import attention_v1 as backend

    impl = object.__new__(backend.AscendAttentionBackendImpl)
    impl.use_bnsd_kv_cache = impl.enable_c8_quant = False
    impl.sliding_window = impl.sinks = impl.kv_sharing_target_layer_name = None
    impl.key_cache, impl.value_cache = k, v
    impl.num_heads = impl.num_kv_heads = 1
    impl.head_size, impl.scale = 2, 1.0
    impl._use_max_workspace_for_fia_graph = False
    impl._use_layer_aware_fia_graph_replay = False
    layer = SimpleNamespace(kv_cache=(k, v), impl=impl)
    runner.model = SimpleNamespace(modules=lambda: iter([layer]))
    runner.block_tables = SimpleNamespace(slot_mappings=torch.tensor([[0, 1]]))
    metadata = SimpleNamespace(
        causal=True,
        num_actual_tokens=2,
        slot_mapping=runner.block_tables.slot_mappings[0],
        block_tables=torch.tensor([[0, 2]]),
        seq_lens=data["seq_lens"],
        seq_lens_list=[4],
        actual_seq_lengths_q=[2],
        query_start_loc=data["query_start_loc"],
        attn_mask=None,
        attn_state=backend.AscendAttentionState.DecodeOnly,
    )
    runner.model_state = SimpleNamespace(attn_metadata={"target": metadata})
    runner.attn_groups = [[SimpleNamespace(layer_names=["target"])]]
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=["target"])])
    runner.speculator = SimpleNamespace(draft_attn_layer_names=[])
    runner.compilation_config = SimpleNamespace(static_forward_context={"target": layer})
    runner.cudagraph_manager = FakeManager()
    manager = runner.cudagraph_manager
    manager.graphs, manager.calls, manager.failure = {Descriptor(): object()}, 0, None
    manager.layer, manager.output = layer, torch.arange(4.0).reshape(2, 2)
    graph_params = SimpleNamespace(attn_params={2: []}, handles={2: []}, events={2: []}, workspaces={2: object()})
    monkeypatch.setattr(backend, "get_graph_params", lambda: graph_params)
    monkeypatch.setattr(backend, "_EXTRA_CTX", SimpleNamespace(is_draft_model=False))
    monkeypatch.setattr(backend, "weak_ref_tensors", lambda tensor: tensor)
    # Only native operations are substituted. The real _get_fia_params and
    # full_graph_fia construct the captured tuple consumed by the target probe.
    monkeypatch.setattr(backend.torch_npu.npu, "current_stream", lambda: object())
    monkeypatch.setattr(torch.npu, "ExternalEvent", MagicMock())
    monkeypatch.setattr(torch.npu, "graph_task_group_begin", MagicMock())
    monkeypatch.setattr(torch.npu, "graph_task_group_end", MagicMock())
    monkeypatch.setattr(backend.torch_npu.npu_fused_infer_attention_score, "out", MagicMock())
    q = torch.ones(2, 1, 2)
    impl.full_graph_fia(q, k, v, metadata, q.clone())
    original_import = tool.importlib.import_module
    monkeypatch.setattr(
        tool.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(get_graph_params=lambda: graph_params)
            if name == "vllm_ascend.compilation.acl_graph"
            else original_import(name)
        ),
    )
    # Only the live-device/type gate and actual stream wait are substituted.
    # The production inventory, input-view, captured-KV and block checks run.
    monkeypatch.setattr(tool.LiveTargetAdapter, "_validate_runner", lambda self: None)
    monkeypatch.setattr(tool.LiveTargetAdapter, "_synchronize", lambda self: None)
    return runner, graph_params


def test_production_fia_tuple_is_consumed_by_target_probe(monkeypatch):
    runner, params = live_fixture(monkeypatch)
    p = params.attn_params[2][0]
    assert len(p) == 22
    k, v = runner.cudagraph_manager.layer.kv_cache
    assert tool._view(p[1]) == tool._view(k.view(8, 2, 2))
    assert tool._view(p[2]) == tool._view(v.view(8, 2, 2))
    assert p[17:21] == (None,) * 4 and p[21] is None
    tool.LiveTargetAdapter(runner, fixture_id="host")._boundary(Descriptor(), runner.batch)


@pytest.mark.parametrize("fault", ["length", "swap_kv", "block_size", "heads", "quant", "layer", "lengths"])
def test_production_fia_schema_drift_is_rejected(monkeypatch, fault):
    runner, params = live_fixture(monkeypatch)
    p = list(params.attn_params[2][0])
    if fault == "length":
        p.append(None)
    elif fault == "swap_kv":
        p[1], p[2] = p[2], p[1]
    else:
        index, value = {
            "block_size": (5, 128),
            "heads": (8, 99),
            "quant": (17, torch.ones(1)),
            "layer": (21, "draft"),
            "lengths": (7, [99]),
        }[fault]
        p[index] = value
    params.attn_params[2][0] = tuple(p)
    with pytest.raises(tool.EvidenceError):
        tool.LiveTargetAdapter(runner, fixture_id="host")._boundary(Descriptor(), runner.batch)


@pytest.mark.parametrize("statuses", [("PASS", "OPEN"), ("FAIL", "OPEN"), ("FAIL", "INVALID_EVIDENCE")])
def test_cli_summary_retains_confirmed_failures(tmp_path, monkeypatch, statuses):
    a, *_ = execute()
    b, *_ = execute()
    left, right = tmp_path / "left.pt", tmp_path / "right.pt"
    a.records *= 2
    b.records *= 2
    if statuses[0] == "FAIL":
        b.records[0] = deepcopy(b.records[0])
        b.records[0]["output"]["tensors"][""]["data"] += 1
    b.records[1] = deepcopy(b.records[1])
    if statuses[1] == "OPEN":
        b.records[1]["inputs"]["tensors"]["/input_ids"]["data"][0] += 1
    else:
        del b.records[1]["output"]
    a.save(left)
    b.save(right)
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        sys, "argv", ["target", str(left), str(right), "--tolerance-reason", "exact", "--output", str(output)]
    )
    assert tool.main() == 2
    report = json.loads(output.read_text())
    assert report["observed_forward_pair"] == ("FAIL" if "FAIL" in statuses else "OPEN")
    assert report["confirmed_failure_count"] == statuses.count("FAIL")
    assert report["evidence_complete"] is False


def test_live_instance_seam_observes_real_return_and_restores_methods(monkeypatch):
    runner, _ = live_fixture(monkeypatch)
    before = runner.cudagraph_manager.run_fullgraph.__func__
    with tool.LiveTargetAdapter(runner, fixture_id="fake-device") as probe:
        assert runner.prepare_inputs() is runner.batch
        assert runner.cudagraph_manager.run_fullgraph(Descriptor()) is runner.cudagraph_manager.output
    assert runner.cudagraph_manager.calls == 1
    assert runner.cudagraph_manager.run_fullgraph.__func__ is before
    assert "run_fullgraph" not in vars(runner.cudagraph_manager) and "prepare_inputs" not in vars(runner)
    record = probe.records[0]
    assert record["status"] == "CAPTURED"
    assert record["kv_before"]["tensors"]["/target~1K"]["block_ids"] == (0, 2)
    assert compare(record, deepcopy(record))["live_full_target"] == "OPEN"


@pytest.mark.parametrize("fault", ["missing_layer", "wrong_captured_kv", "missing_task", "wrong_slots", "stale_batch"])
def test_live_seam_blocks_incomplete_coverage_but_executes_original_once(monkeypatch, fault):
    runner, params = live_fixture(monkeypatch)
    if fault == "missing_layer":
        runner.kv_cache_config.kv_cache_groups[0].layer_names.append("missing")
    elif fault == "wrong_captured_kv":
        p = list(params.attn_params[2][0])
        p[1] = p[1].clone()
        params.attn_params[2] = [tuple(p)]
    elif fault == "missing_task":
        params.handles[2] = []
    elif fault == "wrong_slots":
        runner.model_state.attn_metadata["target"].slot_mapping = torch.tensor([0, 1])
    with tool.LiveTargetAdapter(runner, fixture_id="fake-device") as probe:
        if fault != "stale_batch":
            runner.prepare_inputs()
        result = runner.cudagraph_manager.run_fullgraph(Descriptor())
    assert result is runner.cudagraph_manager.output
    assert runner.cudagraph_manager.calls == 1
    assert probe.records[0]["status"] == "BLOCKED"


def test_live_seam_preserves_forward_error_and_record_limit(monkeypatch):
    runner, _ = live_fixture(monkeypatch)
    error = RuntimeError("original error")
    with tool.LiveTargetAdapter(runner, fixture_id="fake-device", max_records=1) as probe:
        runner.prepare_inputs()
        runner.cudagraph_manager.run_fullgraph(Descriptor())
        runner.cudagraph_manager.failure = error
        runner.prepare_inputs()
        with pytest.raises(RuntimeError) as caught:
            runner.cudagraph_manager.run_fullgraph(Descriptor())
        assert caught.value is error
    assert runner.cudagraph_manager.calls == 2 and probe.skipped_for_limit == 1


@pytest.mark.parametrize("failure_at", [1, 2, 3])
def test_live_seam_does_not_swallow_native_sync_or_continue_forward(monkeypatch, failure_at):
    runner, _ = live_fixture(monkeypatch)
    error = RuntimeError("injected native synchronization failure")
    calls = []

    def sync(_self):
        calls.append("sync")
        if len(calls) == failure_at:
            raise error

    monkeypatch.setattr(tool.LiveTargetAdapter, "_synchronize", sync)
    with pytest.raises(RuntimeError) as caught, tool.LiveTargetAdapter(runner, fixture_id="fake-device") as probe:
        runner.prepare_inputs()
        runner.cudagraph_manager.run_fullgraph(Descriptor())
    assert caught.value is error
    assert runner.cudagraph_manager.calls == (1 if failure_at == 3 else 0)
    assert "run_fullgraph" not in vars(runner.cudagraph_manager)
    if failure_at == 3:
        assert probe.records[0]["status"] == "CORE_RUNTIME_FAILED"


@pytest.mark.parametrize("failure_at", range(1, 7))
def test_live_seam_propagates_native_snapshot_errors(monkeypatch, failure_at):
    runner, _ = live_fixture(monkeypatch)
    original_snapshot = tool._snapshot
    calls = []
    error = RuntimeError("injected native D2H failure")

    def snapshot(*args, **kwargs):
        calls.append("snapshot")
        if len(calls) == failure_at:
            raise error
        return original_snapshot(*args, **kwargs)

    monkeypatch.setattr(tool, "_snapshot", snapshot)
    with pytest.raises(RuntimeError) as caught, tool.LiveTargetAdapter(runner, fixture_id="fake-device") as probe:
        runner.prepare_inputs()
        runner.cudagraph_manager.run_fullgraph(Descriptor())
    assert caught.value is error
    assert runner.cudagraph_manager.calls == (1 if failure_at >= 5 else 0)
    assert "run_fullgraph" not in vars(runner.cudagraph_manager)
    assert "prepare_inputs" not in vars(runner)
    if failure_at >= 5:
        assert probe.records[0]["status"] == "CORE_RUNTIME_FAILED"


def test_live_budget_evidence_failure_still_calls_original_once(monkeypatch):
    runner, _ = live_fixture(monkeypatch)
    with tool.LiveTargetAdapter(runner, fixture_id="fake-device", max_bytes=1) as probe:
        runner.prepare_inputs()
        result = runner.cudagraph_manager.run_fullgraph(Descriptor())
    assert result is runner.cudagraph_manager.output
    assert runner.cudagraph_manager.calls == 1
    assert probe.records[0]["status"] == "BLOCKED"
    assert "budget" in probe.records[0]["error"]


def test_caller_origin_and_npu_labels_cannot_promote_observation_to_full_target():
    a, *_ = execute()
    b, *_ = execute()
    for record in (a.records[0], b.records[0]):
        record["origin"] = "LIVE_ASCEND_TARGET_FULL_GRAPH"
        record["case_kind"] = "NATURAL_OBSERVATION"
        for field in ("kv_before", "kv_after"):
            for entry in record[field]["tensors"].values():
                entry["device_type"] = "npu"
    result = compare(a.records[0], b.records[0])
    assert result["observed_forward_pair"] == "PASS"
    assert result["live_full_target"] == "OPEN"
    assert result["controlled_case"] == "NOT_ESTABLISHED"
    assert result["live_readiness"] == "NOT_ESTABLISHED"


def test_observer_refuses_predeclared_controlled_label():
    a, *_ = execute()
    b, *_ = execute()
    a.records[0]["case_kind"] = b.records[0]["case_kind"] = "PREDECLARED_CONTROLLED"
    assert compare(a.records[0], b.records[0])["observed_forward_pair"] == "INVALID_EVIDENCE"


def test_cli_roundtrip_and_incomplete_collection(tmp_path):
    a, *_ = execute()
    b, *_ = execute()
    left, right = tmp_path / "a.pt", tmp_path / "b.pt"
    a.save(left)
    b.save(right)
    command = [
        sys.executable,
        str(ROOT / "tools/p7x_graph_repair4_target.py"),
        str(left),
        str(right),
        "--tolerance-reason",
        "CPU exact test",
        "--output",
        str(tmp_path / "comparison.json"),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["live_full_target"] == "OPEN"
    assert report == json.loads((tmp_path / "comparison.json").read_text())
    assert report["pairs"][0]["atol"] == report["pairs"][0]["rtol"] == 0
    b.skipped_for_limit = 1
    b.save(right)
    assert subprocess.run(command, capture_output=True).returncode == 2


def test_driver_exposes_fixed_runner_probe_flags():
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/p7x_graph_repair2.py"), "run", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--full-target-probe" in result.stdout
    assert "--full-target-max-bytes" in result.stdout
