# SPDX-License-Identifier: Apache-2.0
"""Host contracts only: synthetic evidence is never an NPU execution result."""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / "tools/p7x_graph_repair4_fixture.py"
SPEC = importlib.util.spec_from_file_location("repair4_fixture_under_test", TOOL)
fixture = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fixture
SPEC.loader.exec_module(fixture)


def manifest(abort_at="none"):
    return fixture.make_manifest("host-test-not-device", "1", "npu:0", abort_at)


def test_tolerance_is_frozen_in_manifest_and_outside_only_control_not_claimed():
    scope = manifest()["scope"]
    assert scope["numerical_acceptance"]["atol"] == fixture.ABS_TOL
    assert scope["numerical_acceptance"]["rtol"] == 0
    assert scope["outside_window_only_perturbation"] == "IMPLEMENTED_PRIVATE_FIA_NUMERIC_ONLY"


def test_fp16_input_and_output_rounding_fit_predeclared_tolerance():
    import torch

    for case in fixture.cases():
        oracle = fixture.full_table_oracle(case)
        points = [(case["full_table"][pos // 128], pos % 128) for pos in range(oracle["start"], case["exact"])]
        values = torch.tensor(
            [[fixture.case_value(case, p, o, c) for c in range(64)] for p, o in points], dtype=torch.float64
        )
        quantized = values.half().double()
        # This quantization bound does not prove native FIA accumulation accuracy.
        rounded_output = quantized.mean(dim=0).half().double()
        reference = torch.tensor(oracle["output"][0][0], dtype=torch.float64)
        assert (rounded_output - reference).abs().max().item() < fixture.ABS_TOL


def synthetic_document(expected):
    """Manufacture checker inputs in tests; never write them as device evidence."""
    shapes = {
        "q": [7, 2, 64],
        "k": [32, 128, 1, 64],
        "v": [32, 128, 1, 64],
        "output": [7, 2, 64],
        "table": [1, 9],
        "full_table": [1, 12],
        "length": [1],
        "slots": [7],
        "positions": [7],
        "query_locs": [2],
        "workspace": [1024],
    }
    buffers = {}
    for index, (name, shape) in enumerate(shapes.items()):
        size, stride = 1, []
        for dim in reversed(shape):
            stride.insert(0, size)
            size *= dim
        dtype = (
            "torch.float16"
            if name in ("q", "k", "v", "output")
            else ("torch.int64" if name in ("slots", "positions") else "torch.int32")
        )
        buffers[name] = {
            "address": 1000 * (index + 1),
            "shape": shape,
            "stride": stride,
            "offset": 0,
            "device": "npu:0",
            "dtype": dtype,
        }
    source = {
        "source": "synthetic source, host checker test only",
        "path": "/synthetic/api.py",
        "line": 1,
        "file_sha256": "a" * 64,
        "signature": "()",
    }
    names = (
        "graph_task_group_begin",
        "graph_task_group_end",
        "graph_task_update_begin",
        "graph_task_update_end",
        "NPUGraph.replay",
        "Event.record",
        "Event.synchronize",
        "Event.query",
        "ExternalEvent.wait",
        "ExternalEvent.reset",
        "ExternalEvent.record",
        "get_cann_version",
    )
    document = {
        "schema": fixture.SCHEMA,
        "identity": expected,
        "manifest_digest": fixture.digest(expected),
        "status": "COMPLETE",
        "process_id": 123,
        "ready": False,
        "blockers": dict(fixture.BLOCKERS),
        "runtime": {
            "torch_npu": "2.10.0.post4",
            "torch": "2.10.0",
            "cann": "9.1.0",
            "api_sources": {name: copy.deepcopy(source) for name in names},
            "production_sources": {name: copy.deepcopy(source) for name in ("capture", "backend_update", "helper")},
            "native_extension_sha256": "b" * 64,
            "fia_schema": "actual_seq_lengths_kv",
        },
        "capture": {
            "count": 1,
            "backend": "AscendAttentionBackendImpl.full_graph_fia",
            "buffers": buffers,
            "provider_id": 456,
            "provider_schema": 2,
        },
        "records": [],
    }
    for name, filename in (
        ("capture", "vllm_ascend/attention/attention_v1.py"),
        ("backend_update", "vllm_ascend/attention/attention_v1.py"),
        ("helper", "vllm_ascend/compilation/acl_graph.py"),
    ):
        document["runtime"]["production_sources"][name]["file_sha256"] = expected["source"]["files"][filename]
    for case in expected["cases"]:
        oracle = fixture.full_table_oracle(case)
        document["records"].append(
            {
                "label": case["label"],
                "generation": case["generation"],
                "epoch": case["epoch"],
                "process_id": 123,
                "manifest_digest": fixture.digest(expected),
                "nonce": expected["nonce"],
                "buffers": copy.deepcopy(buffers),
                "full_table": list(case["full_table"]),
                "table": oracle["table"],
                "length": oracle["length"],
                "host_lengths": [oracle["length"]],
                "actual_seq_lengths_q": [7],
                "output": oracle["output"],
                "binding": {
                    "provider_id": 456,
                    "provider_schema": 2,
                    "p3_generation": case["generation"] - 1,
                    "epoch": case["epoch"],
                    "unsafe_reuse_rejected": True,
                    "retired_after_consumer_completion": True,
                    "update_function": "vllm_ascend.compilation.acl_graph.update_full_graph_params",
                },
                "completion": {
                    "verified": True,
                    "method": "ordinary_events_after_update_and_last_consumer",
                    "consumer_event": 22,
                    "update_event": 23,
                    "consumer_stream": 24,
                    "update_stream": 25,
                },
                "controls": {
                    "slots": list(range(7)),
                    "positions": list(range(7)),
                    "query_locs": [0, 7],
                    "q": "a" * 64,
                    "k": "b" * 64,
                    "v": fixture.expected_value_hash(case),
                },
            }
        )
    return document


def test_primary_base7_geometry_and_independent_full_table_oracle():
    assert (fixture.QUERY_WIDTH, fixture.WINDOW_SIZE, fixture.BLOCK_SIZE) == (7, 1024, 128)
    assert (fixture.CAPACITY_TOKENS, fixture.TABLE_WIDTH) == (1151, 9)
    a, b, again, short, nonaligned, outside, inside, restored = [fixture.full_table_oracle(c) for c in fixture.cases()]
    assert outside["output"] == a["output"] == restored["output"]
    assert abs(inside["output"][0][0][0] - a["output"][0][0][0]) > 10 * fixture.ABS_TOL
    assert (a["start"], a["length"], b["start"], b["length"]) == (128, 1024, 256, 1024)
    assert a["output"] == again["output"] != b["output"]
    assert a["output"][0][0][0] == 5.5 + 127 / 512
    assert b["output"][0][0][0] == 20.5 + 127 / 512
    assert (short["start"], short["length"]) == (0, 64)
    assert short["table"] == [1] + [0] * 8
    assert (nonaligned["start"], nonaligned["length"]) == (128, 1025)
    assert nonaligned["table"][-1] == 31
    assert abs(nonaligned["output"][0][0][0] - a["output"][0][0][0]) > 10 * fixture.ABS_TOL
    for case in fixture.cases():
        oracle = fixture.full_table_oracle(case)
        assert (case["table"], case["length"]) == (oracle["table"], oracle["length"])


@pytest.mark.parametrize(
    "key,value",
    [("exact", True), ("window", 0), ("exact", 2000), ("horizon", 0), ("full_table", [33]), ("full_table", [])],
)
def test_oracle_rejects_bad_inputs(key, value):
    case = fixture.cases()[0]
    case[key] = value
    with pytest.raises(ValueError):
        fixture.full_table_oracle(case)


def test_checker_pass_means_microfixture_only_never_target_ready():
    expected = manifest()
    result = fixture.check_artifact(synthetic_document(expected), expected)
    assert result["fixture"] == "PASS", result
    assert result["ready"] is False
    assert result["production_integration"] == "PARTIAL"
    assert result["G4F"].startswith("BLOCKED")


@pytest.mark.parametrize(
    "path,value",
    [
        (("schema",), "stale-v0"),
        (("manifest_digest",), "stale"),
        (("ready",), True),
        (("status",), "ABORTED"),
        (("records", 1, "nonce"), "old-run"),
        (("records", 2, "generation"), 1),
        (("records", 0, "generation"), True),
        (("records", 1, "epoch"), 1),
        (("records", 1, "process_id"), 124),
        (("records", 1, "table", 0), 2),
        (("records", 1, "full_table", 0), 31),
        (("records", 1, "length"), 64),
        (("records", 1, "host_lengths"), [64]),
        (("records", 1, "actual_seq_lengths_q"), [2]),
        (("records", 1, "buffers", "table", "address"), 999),
        (("capture", "buffers", "table", "shape"), [1, 3]),
        (("capture", "buffers", "table", "dtype"), "torch.float16"),
        (("capture", "buffers", "table", "device"), "cpu"),
        (("capture", "buffers", "table", "stride"), [1, 9]),
        (("capture", "buffers", "output", "address"), 1000),
        (("records", 1, "completion", "verified"), False),
        (("records", 1, "completion", "method"), "update_only"),
        (("records", 1, "completion", "consumer_event"), 23),
        (("records", 1, "binding", "provider_id"), 457),
        (("records", 1, "binding", "p3_generation"), 0),
        (("records", 1, "binding", "unsafe_reuse_rejected"), False),
        (("records", 1, "binding", "retired_after_consumer_completion"), False),
        (("records", 1, "output", 0, 0, 0), float("nan")),
        (("records", 1, "output", 0, 0), [0]),
        (("records", 1, "controls", "positions"), [0]),
        (("records", 1, "controls", "v"), "modified-KV"),
        (("runtime", "torch_npu"), "2.10.0"),
        (("runtime", "cann"), "9.0.0"),
        (("runtime", "api_sources"), {}),
        (("runtime", "production_sources"), {}),
    ],
)
def test_checker_negative_evidence(path, value):
    expected = manifest()
    document = synthetic_document(expected)
    node = document
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    result = fixture.check_artifact(document, expected)
    assert result["fixture"] == "FAIL" and result["ready"] is False, result
    assert result["errors"]


def test_stale_b_output_and_missing_replay_are_not_passes():
    expected = manifest()
    document = synthetic_document(expected)
    document["records"][1]["output"] = document["records"][0]["output"]
    assert "measured output" in fixture.check_artifact(document, expected)["errors"][0]
    document = synthetic_document(expected)
    document["records"].pop()
    assert fixture.check_artifact(document, expected)["fixture"] == "FAIL"


def test_checker_requires_independent_challenge_and_live_source():
    expected = manifest()
    document = synthetic_document(expected)
    assert fixture.check_artifact(document, manifest())["fixture"] == "FAIL"
    expected["source"]["files"][fixture.SOURCE_FILES[0]] = "0" * 64
    document["manifest_digest"] = fixture.digest(expected)
    assert "source identity" in fixture.check_artifact(document, expected)["errors"][0]


def test_guard_rejects_overwrite_through_update_until_consumer_completion():
    guard = fixture.CompletionGuard()
    guard.stage(1)
    for transition in (lambda: None, guard.submitted, guard.updated):
        transition()
        with pytest.raises(ValueError, match="unsafe reuse"):
            guard.stage(2)
    guard.completed()
    with pytest.raises(ValueError, match="stale generation"):
        guard.stage(1)
    guard.stage(2)


class OrderingAdapter:
    def __init__(self, *, fail_update=False, fail_completion=False):
        self.calls = []
        self.fail_update, self.fail_completion = fail_update, fail_completion

    def stage(self, case):
        self.calls.append("stage")

    def replay(self):
        self.calls.append("replay_then_consumer_event")

    def update(self):
        self.calls.append("update_then_update_event")
        if self.fail_update:
            raise RuntimeError("native update failed")

    def complete(self):
        self.calls.append("complete_both")
        if self.fail_completion:
            raise RuntimeError("last consumer not complete")
        return {"verified": True}

    def snapshot(self, case, completion):
        self.calls.append("snapshot")
        return {"label": case["label"], "completion": completion}


@pytest.mark.parametrize(
    "abort_at,verified,completed",
    [
        ("before_replay", False, False),
        ("after_replay", False, False),
        ("after_update", True, True),
        ("after_completion", True, True),
    ],
)
def test_injected_abort_records_and_permanent_reuse_rejection(abort_at, verified, completed):
    adapter, document = OrderingAdapter(), {"records": []}
    fixture.execute_cases(adapter, manifest(abort_at), document, lambda *_: None)
    assert document["status"] == "ABORTED"
    assert document["abort"]["completion_verified"] is verified
    assert document["abort"]["unsafe_reuse_rejected"] is True
    assert document["abort"]["reused"] is False
    assert ("complete_both" in adapter.calls) is completed
    assert "snapshot" not in adapter.calls


@pytest.mark.parametrize("fail_update,fail_completion", [(True, False), (False, True)])
def test_native_failure_does_not_fake_completion_or_release(fail_update, fail_completion):
    adapter = OrderingAdapter(fail_update=fail_update, fail_completion=fail_completion)
    document = {"records": []}
    fixture.execute_cases(adapter, manifest(), document, lambda *_: None)
    assert document["abort"]["completion_verified"] is False
    assert document["abort"]["unsafe_reuse_rejected"] is True
    if fail_update:
        assert "complete_both" not in adapter.calls  # replay can be waiting on an unrecorded ExternalEvent
    else:
        assert adapter.calls.count("complete_both") == 1  # no retry after a failed completion boundary


def test_replay_update_last_consumer_completion_order():
    adapter, document = OrderingAdapter(), {"records": []}
    fixture.execute_cases(adapter, manifest(), document, lambda *_: None)
    assert document["status"] == "COMPLETE"
    assert (
        adapter.calls
        == ["stage", "replay_then_consumer_event", "update_then_update_event", "complete_both", "snapshot"] * 8
    )


@pytest.mark.parametrize("consumer_complete", [True, False])
def test_actual_completion_method_requires_last_consumer(consumer_complete):
    calls = []

    class Event:
        def __init__(self, name, complete):
            self.name, self.complete = name, complete

        def synchronize(self):
            calls.append((self.name, "synchronize"))

        def query(self):
            calls.append((self.name, "query"))
            return self.complete

    owner = SimpleNamespace(
        update_done=Event("update", True),
        consumer_done=Event("consumer", consumer_complete),
        update_stream=SimpleNamespace(npu_stream=1),
        replay_stream=SimpleNamespace(npu_stream=2),
    )
    if consumer_complete:
        assert fixture.NPUFixture.complete(owner)["verified"] is True
    else:
        with pytest.raises(ValueError, match="both streams"):
            fixture.NPUFixture.complete(owner)
    assert calls == [("update", "synchronize"), ("consumer", "synchronize"), ("update", "query"), ("consumer", "query")]


def test_diagnostic_uses_ordinary_last_consumer_event_and_real_helper():
    tree = ast.parse(TOOL.read_text())
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    methods = {node.name: node for node in classes["NPUFixture"].body if isinstance(node, ast.FunctionDef)}
    replay = ast.unparse(methods["replay"])
    assert replay.index("self.graph.replay()") < replay.index("self.consumer_done.record(self.replay_stream)")
    complete = ast.unparse(methods["complete"])
    assert "self.consumer_done.synchronize()" in complete and "self.update_done.synchronize()" in complete
    assert "self.consumer_done.query()" in complete and "self.update_done.query()" in complete
    update = ast.unparse(methods["update"])
    assert "update_full_graph_params" in update and "draft_window_graph_update_binding" in update
    # ExternalEvent query is never called, including inherited/native fallbacks.
    assert "ExternalEvent.query" not in TOOL.read_text()


def test_cli_help_and_dry_run_are_standalone_host_only():
    for argv in (["--help"], ["run", "--help"], ["run", "--dry-run"]):
        proc = subprocess.run([sys.executable, str(TOOL), *argv], text=True, capture_output=True, timeout=30)
        assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "DRY_RUN_NOT_DEVICE_EVIDENCE"
    assert payload["scope"]["geometry"]["q"] == [7, 2, 64]
    assert payload["scope"]["ready"] is False
    assert len(payload["cases"]) == 8


def test_npu_run_rejects_cpu_before_launch(tmp_path):
    args = SimpleNamespace(run_id="bad", attempt="1", device="cpu", abort_at="none", output=tmp_path / "result")
    with pytest.raises(ValueError, match="device must be explicit npu"):
        fixture.launch(args)
    assert not args.output.exists()


def test_existing_output_is_not_overwritten(tmp_path):
    args = SimpleNamespace(run_id="bad", attempt="1", device="npu:0", abort_at="none", output=tmp_path)
    with pytest.raises(FileExistsError):
        fixture.launch(args)


def test_real_p3_p6_binding_on_host_owned_buffers():
    """CPU contract test of real production classes; not a device execution mode."""
    import torch

    from vllm_ascend.attention.attention_v1 import (
        AscendAttentionBackend,
        AscendAttentionBackendImpl,
        AscendAttentionMetadataBuilder,
        AscendAttentionState,
        AscendMetadata,
    )

    owner = SimpleNamespace(
        torch=torch,
        device="cpu",
        full_table=torch.zeros((1, 12), dtype=torch.int32),
        slots=torch.arange(7),
        positions=torch.arange(7),
        query_locs=torch.tensor([0, 7], dtype=torch.int32),
        metadata=AscendMetadata(attn_state=AscendAttentionState.SpecDecoding),
    )
    binding = fixture.ProductionBinding(owner)
    addresses = (owner.table.data_ptr(), owner.length.data_ptr())
    for case in fixture.cases():
        owner.full_table.copy_(torch.tensor([case["full_table"]]))
        binding.bind(case)
        assert binding.reuse_rejected is True
        assert (owner.table.data_ptr(), owner.length.data_ptr()) == addresses
        assert owner.table.tolist() == [fixture.full_table_oracle(case)["table"]]
        assert owner.length.tolist() == [case["length"]]
        binding.provider.validate_backend_update(
            binding.update_binding,
            attn_backend=AscendAttentionBackend,
            impl_cls=AscendAttentionBackendImpl,
            builder_cls=AscendAttentionMetadataBuilder,
            draft_attn_metadatas=binding.batch,
            update_function_path="vllm_ascend.compilation.acl_graph.update_full_graph_params",
        )
        # Host test explicitly supplies the completion boundary; no NPU evidence.
        record = binding.retire_after_completion()
        assert record["p3_generation"] == case["generation"] - 1
