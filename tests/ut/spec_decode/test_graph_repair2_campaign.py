# SPDX-License-Identifier: Apache-2.0
"""No empty/vacuous evidence PASS; test-only records never become NPU proof."""

import importlib.util
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _tool():
    path = Path(__file__).resolve().parents[3] / "tools/p7x_graph_repair2.py"
    spec = importlib.util.spec_from_file_location("repair2_campaign", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _empty():
    return {
        "schema": "p7x-repair2-campaign-v1",
        "identity": {
            "run_id": "run",
            "case_id": "B",
            "attempt_id": "1",
            "source_sha": "a" * 40,
            "source_tree": "b" * 40,
            "vllm_sha": "84030bbe3d74d99bad477a3d2e37a973ccd8865c",
            "engine_id": 123,
            "process_id": 1,
        },
        "startup_complete": False,
        "status": "EXCEPTION",
        "snapshot": {
            "schema": "mrv2-draft-window-graph-characterization-v2",
            "process_id": 1,
            "provider_identity": 9,
            "recorder_id": "r",
            "artifact_errors": [],
            "events": [],
            "operand_events": [],
            "g4_records": [],
            "lifecycle_events": [],
        },
    }


def test_repair3_boundary_report_requires_runtime_not_prompt():
    result = _tool().analyze_campaign(_empty())
    assert result["boundaries"]["S_gt_0"] == "NOT_REACHED"
    assert result["aba_scan"]["status"] == "OPEN"


def test_repair3_probe_set_missing_never_equals_missing(tmp_path):
    result = _tool().compare_probe_sets({"records": []}, {"records": []}, tmp_path, tmp_path)
    assert result["write_semantics"] == "OPEN"
    assert result["head_only_numeric"] == "OPEN"
    assert result["full_target_numeric"] == "OPEN"


def test_repair3_launcher_preserves_failure_and_identity(tmp_path):
    args = SimpleNamespace(
        output=tmp_path / "launch",
        run_id="r",
        case_id="B0",
        attempt="2",
        mode="B",
        timeout=10,
        argv=[sys.executable, "-c", "raise SystemExit(3)"],
    )
    assert _tool().launch_case(args) == 3
    record = json.loads((args.output / "launcher.json").read_text())
    assert record["returncode"] == 3
    assert record["case_id"] == "B0" and record["driver_mode"] == "B"
    assert not (args.output / "campaign.json").exists()


def test_repair3_launcher_records_signal_without_fabricating_campaign(tmp_path, monkeypatch):
    tool = _tool()
    monkeypatch.setattr(tool, "_git", lambda *a: "a" * 40)
    monkeypatch.setattr(tool.subprocess, "Popen", lambda *a, **kw: SimpleNamespace(pid=123, wait=lambda **kw: -11))
    args = SimpleNamespace(
        output=tmp_path / "launch",
        run_id="r",
        case_id="B0",
        attempt="1",
        mode="B",
        timeout=10,
        argv=["stub-native-crash"],
    )
    assert tool.launch_case(args) == 139
    record = json.loads((args.output / "launcher.json").read_text())
    assert record["returncode"] == -11 and record["signal"] == 11
    assert not (args.output / "campaign.json").exists()


@pytest.mark.parametrize(
    "mutation", ["none", "missing_layer", "count", "nan", "shape", "dtype", "duplicate_hash", "missing_raw"]
)
def test_repair3_probe_collection_strictness(tmp_path, mutation):
    import hashlib

    tensor = {"shape": [1], "dtype": "torch.float32", "values": [1]}
    state = dict(
        canonical_tables=[tensor],
        slots=tensor,
        positions=tensor,
        query=tensor,
        target_layers={"l": {"table": tensor, "exact": tensor}},
    )
    x = torch.tensor([1.0])
    digest = hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()
    left, right = tmp_path / "a", tmp_path / "b"
    left.mkdir()
    right.mkdir()
    a = {
        "records": [
            {"kind": "full_write_target", "before": state, "after": state},
            {
                "kind": "target_logits_head",
                "path": "head.pt",
                "input_sha256": digest,
                "input_shape": [1],
                "output_shape": [1],
                "dtype": "torch.float32",
            },
        ]
    }
    b = deepcopy(a)
    torch.save({"hidden_states": x, "logits": x}, left / "head.pt")
    torch.save({"hidden_states": x, "logits": x}, right / "head.pt")
    if mutation == "missing_layer":
        b["records"][0]["after"]["target_layers"] = {}
    elif mutation == "count":
        b["records"].append(deepcopy(b["records"][0]))
    elif mutation == "nan":
        torch.save({"hidden_states": x, "logits": torch.tensor([float("nan")])}, right / "head.pt")
    elif mutation == "shape":
        b["records"][1]["input_shape"] = [2]
    elif mutation == "dtype":
        b["records"][1]["dtype"] = "torch.bfloat16"
    elif mutation == "duplicate_hash":
        # Legitimate repeats are distinct samples, NOT dictionary overwritten.
        a["records"].append(deepcopy(a["records"][1]))
        b["records"].append(deepcopy(b["records"][1]))
    elif mutation == "missing_raw":
        b["records"][1]["path"] = "missing.pt"
    result = _tool().compare_probe_sets(a, b, left, right)
    if mutation in ("none", "duplicate_hash"):
        assert result["head_only_numeric"] == result["write_semantics"] == "PASS"
        assert len(result["pairs"]) == (2 if mutation == "duplicate_hash" else 1)
    else:
        assert "PASS" not in (result["write_semantics"], result["head_only_numeric"]) or not (
            result["write_semantics"] == result["head_only_numeric"] == "PASS"
        )
    assert result["full_target_numeric"] == "OPEN"


@pytest.mark.parametrize("mutation", ["missing", "wrong", "duplicate"])
def test_repair3_per_layer_association_rejected(mutation):
    doc = _complete_host_fixture()
    records = doc["snapshot"]["operand_events"]
    if mutation == "missing":
        records[:] = [r for r in records if r["layer_name"] != "layer.1"]
    elif mutation == "wrong":
        records[0]["layer_name"] = "not.a.draft.layer"
    else:
        records[0]["handle_identity"] = 123
        records.append(dict(deepcopy(records[0]), sequence=999))
    result = _tool().audit_campaign(doc)
    assert result["HOST_CONTRACTS"] != "PASS"


def test_repair3_full_sequence_scan_finds_earlier_aba(monkeypatch):
    tool = _tool()
    bound = [
        dict(
            sequence=n,
            generation_id=n,
            mapping_epoch=n,
            capture_descriptor="d",
            backing_table_address=100,
            table_values=[[v]],
            exact_lengths=[1],
            upper_lengths=[1],
            stage="bound",
            record_scope={"recorder_id": "r", "process_id": 1, "provider_identity": 9},
        )
        for n, v in enumerate((1, 2, 1, 3, 4))
    ]
    raw = dict(source_identity={"table": 50}, full_exact=[1], full_upper=[1], valid_counts=[1], mapping=[0])
    doc = {"snapshot": {"events": bound, "g4_records": [dict(r, runtime_oracle_inputs=raw) for r in bound]}}
    # This isolates the scan algorithm; the unmodified audit runs first in CLI.
    monkeypatch.setattr(tool, "audit_campaign", lambda _: {"EVIDENCE_INTEGRITY": "PASS"})
    assert tool.check_aba(bound, doc["snapshot"]["g4_records"]) == "OPEN"
    assert tool.analyze_campaign(doc)["aba_scan"]["candidates"][0]["sequences"] == [0, 1, 2]


def test_empty_artifact_never_passes_device_or_host_contracts():
    result = _tool().audit_campaign(_empty())
    assert result["EVIDENCE_INTEGRITY"] == "PASS"
    assert result["HOST_CONTRACTS"] == "OPEN"
    assert result["DEVICE_FUNCTIONAL_GATES"]["fixed_inventory_replay"] == "NOT_RUN"
    assert all(value != "PASS" for value in result["DEVICE_EVIDENCE_LEVELS"].values())


@pytest.mark.parametrize("mutation", ["missing_identity", "process_mix", "provider_mix", "duplicate", "io_error"])
def test_invalid_evidence_is_not_a_gate_pass(mutation):
    record = _empty()
    snap = record["snapshot"]
    if mutation == "missing_identity":
        del record["identity"]["run_id"]
    elif mutation == "process_mix":
        snap["process_id"] = 2
    elif mutation == "provider_mix":
        snap["operand_events"] = [{"sequence": 1, "provider_identity": 8}]
    elif mutation == "duplicate":
        snap["events"] = [{"sequence": 1}, {"sequence": 1}]
    else:
        snap["artifact_errors"] = ["disk full"]
    assert _tool().audit_campaign(record)["EVIDENCE_INTEGRITY"] == "INVALID_EVIDENCE"


def test_numeric_comparison_missing_is_open_not_equal():
    tool = _tool()
    assert tool.compare_target_numeric({}, {})["status"] == "OPEN"
    left = {
        "aligned_input_state_sha256": "same",
        "dtype": "float32",
        "values": [1.0, 2.0],
        "atol": 0.0,
        "rtol": 0.0,
        "tolerance_reason": "controlled exact CPU fixture",
    }
    assert tool.compare_target_numeric(left, dict(left))["status"] == "PASS"
    assert tool.compare_target_numeric(left, dict(left, aligned_input_state_sha256="different"))["status"] == "OPEN"
    assert tool.compare_target_numeric(left, dict(left, values=[1.0, 3.0]))["status"] == "FAIL"


@pytest.mark.parametrize(
    "h,e,u,s,v,upper,columns",
    [
        (32, 32, 39, 0, 32, 39, 1),
        (1024, 1024, 1031, 0, 1024, 1024, 8),
        (1025, 1025, 1032, 0, 1025, 1032, 9),
        (1151, 1151, 1158, 0, 1151, 1152, 9),
        (1152, 1152, 1159, 128, 1024, 1024, 8),
        (1153, 1153, 1160, 128, 1025, 1032, 9),
        (2049, 2049, 2056, 1024, 1025, 1032, 9),
        (1152, 1149, 1155, 128, 1021, 1024, 8),
    ],
)
def test_oracle_meaningful_128_token_boundaries(h, e, u, s, v, upper, columns):
    result = _tool().oracle(
        full_table=list(range(100, 164)),
        full_horizon_end=h,
        full_exact_end=e,
        full_upper_end=u,
        window_size=1024,
        legacy_block_size=128,
        block_span_tokens=128,
        query_width=7,
        valid_block_count=64,
    )
    assert (result.window_start, result.visible_length, result.window_upper, result.required_columns) == (
        s,
        v,
        upper,
        columns,
    )
    assert result.expected_block_ids == tuple(range(100 + s // 128, 100 + s // 128 + columns))


def _complete_host_fixture():
    doc = _empty()
    doc.update(
        status="OK",
        startup_complete=True,
        draft_layer_names=["layer.0", "layer.1"],
        draft_graph_keys=[[1, 7], [2, 14], [4, 28]],
    )
    snap = doc["snapshot"]
    sequence = 0
    for n in (1, 2, 4):
        desc = f"mode=FULL;reqs={n};tokens={n * 7};uniform=7"
        snap["events"].append(
            dict(
                sequence=sequence,
                stage="capture",
                route="dflash",
                capture_descriptor=desc,
                generation_id=None,
                mapping_epoch=None,
                backing_table_address=200,
                exact_lengths=[1] * n,
            )
        )
        sequence += 1
        for layer in doc["draft_layer_names"]:
            snap["operand_events"].append(
                dict(
                    sequence=sequence,
                    stage="capture_operator",
                    route="dflash",
                    capture_descriptor=desc,
                    generation_id=None,
                    mapping_epoch=None,
                    provider_identity=9,
                    backend_impl="FIA",
                    consumer="FIA.out",
                    layer_name=layer,
                    actual_table_address=200,
                    expected_table_address=200,
                    captured_table_address=200,
                    table_shape=[n, 9],
                    table_dtype="int32",
                    cpu_exact=[1] * n,
                    evidence_level="HOST_OPERATOR_BOUNDARY_FACT",
                )
            )
            sequence += 1
        bound_sequence = sequence
        for stage in ("bound", "replay", "consumer", "update"):
            snap["events"].append(
                dict(
                    sequence=sequence,
                    stage=stage,
                    route="dflash",
                    capture_descriptor=desc,
                    generation_id=n,
                    mapping_epoch=n,
                    backing_table_address=200,
                    table_values=[[101, 102, 103, 104, 105, 106, 107, 108, 0]] * n,
                    exact_lengths=[1024] * n,
                    upper_lengths=[1024] * n,
                    values_sampled_at_sequence=None if stage == "bound" else bound_sequence,
                )
            )
            sequence += 1
        for layer in doc["draft_layer_names"]:
            snap["operand_events"].append(
                dict(
                    sequence=sequence,
                    stage="update_operator",
                    route="dflash",
                    capture_descriptor=desc,
                    generation_id=n,
                    mapping_epoch=n,
                    provider_identity=9,
                    backend_impl="FIA",
                    consumer="FIA.out",
                    layer_name=layer,
                    actual_table_address=200,
                    expected_table_address=200,
                    captured_table_address=200,
                    table_shape=[n, 9],
                    table_dtype="int32",
                    cpu_exact=[1024] * n,
                    evidence_level="HOST_OPERATOR_BOUNDARY_FACT",
                )
            )
            sequence += 1
        snap["g4_records"].append(
            dict(
                sequence=sequence,
                route="dflash",
                generation_id=n,
                mapping_epoch=n,
                capture_descriptor=desc,
                canonical_table_address=100,
                canonical_table_values=[list(range(100, 120))] * n,
                window_table_address=200,
                window_table_values=[[101, 102, 103, 104, 105, 106, 107, 108, 0]] * n,
                runtime_oracle_inputs=dict(
                    full_exact=[1152] * n,
                    full_upper=[1159] * n,
                    valid_counts=[20] * n,
                    mapping=list(range(n)),
                    request_ids=[str(i) for i in range(n)],
                    source_identity={"input_block_table_ptrs": [100]},
                    window_size=1024,
                    legacy_block_size=128,
                    block_span_tokens=128,
                    query_width=7,
                ),
            )
        )
        sequence += 1
    for section in ("events", "operand_events", "g4_records", "lifecycle_events"):
        for record in snap[section]:
            record["record_scope"] = {"recorder_id": "r", "process_id": 1, "provider_identity": 9}
    return doc


def test_host_fixture_does_not_prove_task_physical_or_completion():
    result = _tool().audit_campaign(_complete_host_fixture())
    assert result["HOST_CONTRACTS"] == "PASS"
    assert result["DEVICE_FUNCTIONAL_GATES"]["fixed_inventory_replay"] == "PASS"
    assert result["DEVICE_EVIDENCE_LEVELS"]["task_internal"] == "OPEN"
    assert result["DEVICE_EVIDENCE_LEVELS"]["physical_read"] == "OPEN"
    assert result["DEVICE_EVIDENCE_LEVELS"]["completion_and_effective_generation"] == "OPEN"


@pytest.mark.parametrize("mutation", ["lag", "table", "length", "layer", "bucket", "source"])
def test_checker_rejects_lag_and_incomplete_actual_coverage(mutation):
    doc = _complete_host_fixture()
    if mutation == "lag":
        next(e for e in doc["snapshot"]["operand_events"] if e["stage"] == "update_operator")["generation_id"] = 99
    elif mutation == "table":
        doc["snapshot"]["g4_records"][0]["window_table_values"][0][0] = 100
    elif mutation == "length":
        next(e for e in doc["snapshot"]["events"] if e["stage"] == "bound")["exact_lengths"] = [1023]
    elif mutation == "layer":
        doc["draft_layer_names"].append("layer.2")
    elif mutation == "source":
        doc["snapshot"]["g4_records"][0]["runtime_oracle_inputs"]["source_identity"] = {"input_block_table_ptrs": [123]}
    else:
        doc["draft_graph_keys"] = [[1, 7]]
    result = _tool().audit_campaign(doc)
    assert result["DEVICE_FUNCTIONAL_GATES"]["fixed_inventory_replay"] != "PASS"


@pytest.mark.parametrize("field", ["recorder_id", "process_id", "provider_identity"])
def test_checker_rejects_record_spliced_from_another_engine_or_run(field):
    doc = _complete_host_fixture()
    # Source/engine namespace must accompany each record, not just its file.
    for section in ("events", "operand_events", "g4_records", "lifecycle_events"):
        for record in doc["snapshot"][section]:
            record["record_scope"] = {"recorder_id": "r", "process_id": 1, "provider_identity": 9}
    doc["snapshot"]["g4_records"][0]["record_scope"][field] = "other"
    assert _tool().audit_campaign(doc)["EVIDENCE_INTEGRITY"] == "INVALID_EVIDENCE"


def test_cross_process_write_pointers_not_compared_and_missing_values_not_equal():
    left = dict(
        alignment_sha256="controlled",
        canonical_table_values=[[1]],
        slot_mapping_values=[1],
        position_values=[1],
        query_layout_values=[0, 1],
        target_length_values=[1],
        pointer=100,
    )
    right = dict(left, pointer=200)
    assert _tool().compare_full_write(left, right)["status"] == "PASS"
    right["target_length_values"] = None
    assert _tool().compare_full_write(left, right)["status"] == "OPEN"


def test_target_probe_is_instance_scoped_and_restores_on_exception(tmp_path):
    class Model:
        def compute_logits(self, hidden):
            return hidden + 1

    class Spec:
        def propose(self):
            raise RuntimeError("business failure")

    runner = SimpleNamespace(model=Model(), speculator=Spec())
    probe = _tool().TargetBoundaryProbe(runner, tmp_path)
    probe.install()
    hidden = torch.ones(2, 4)
    assert torch.equal(runner.model.compute_logits(hidden), hidden + 1)
    with pytest.raises(RuntimeError, match="business failure"):
        runner.speculator.propose()
    # Missing fake runner metadata is an evidence error, not a replacement
    # business exception. Production methods were neither globally patched nor
    # assigned to another owner.
    assert probe.errors
    probe.close()
    assert "compute_logits" not in vars(runner.model)
    assert "propose" not in vars(runner.speculator)
    assert len(list(tmp_path.glob("target-logits-*.pt"))) == 1


def test_aba_requires_distinct_epoch_and_semantic_source_lineage():
    tool = _tool()
    bound = [
        dict(
            sequence=n * 10,
            generation_id=n,
            mapping_epoch=n,
            capture_descriptor="same",
            backing_table_address=100,
            table_values=[[v]],
            exact_lengths=[1],
            upper_lengths=[1],
            record_scope={"recorder_id": "r", "process_id": 1, "provider_identity": 9},
        )
        for n, v in [(10, 1), (11, 2), (12, 1)]
    ]
    source = dict(source_identity={"table": 50}, full_exact=[1], full_upper=[1], valid_counts=[1], mapping=[0])
    records = [dict(r, runtime_oracle_inputs=source) for r in bound]
    assert tool.check_aba(bound, records) == "PASS"
    bound[2]["mapping_epoch"] = 10
    records[2]["mapping_epoch"] = 10
    assert tool.check_aba(bound, records) == "FAIL"


@pytest.mark.parametrize("stage", ["capture_operator", "update_operator"])
@pytest.mark.parametrize("also_forge_captured_pointer", [False, True])
def test_review_f1_matching_wrong_operand_addresses_cannot_authorize_backing(stage, also_forge_captured_pointer):
    doc = _complete_host_fixture()
    for event in doc["snapshot"]["operand_events"]:
        if event["stage"] == stage:
            event["actual_table_address"] = event["expected_table_address"] = 999
            if also_forge_captured_pointer:
                event["captured_table_address"] = 999
    result = _tool().audit_campaign(doc)
    assert result["EVIDENCE_INTEGRITY"] == "INVALID_EVIDENCE"


def test_review_f2_bound_table_must_match_g4_and_independent_oracle():
    doc = _complete_host_fixture()
    bound = next(e for e in doc["snapshot"]["events"] if e["stage"] == "bound")
    bound["table_values"] = deepcopy(bound["table_values"])
    bound["table_values"][0][0] = 999
    assert _tool().audit_campaign(doc)["EVIDENCE_INTEGRITY"] == "INVALID_EVIDENCE"


def test_review_f3_duplicate_g4_does_not_cover_missing_generations():
    doc = _complete_host_fixture()
    records = doc["snapshot"]["g4_records"]
    doc["snapshot"]["g4_records"] = [dict(deepcopy(records[0]), sequence=r["sequence"]) for r in records]
    assert _tool().audit_campaign(doc)["EVIDENCE_INTEGRITY"] == "INVALID_EVIDENCE"


def test_review_f4_aba_epochs_must_increase_not_merely_differ():
    bound = [
        dict(
            sequence=i * 10,
            generation_id=i + 10,
            mapping_epoch=epoch,
            capture_descriptor="same",
            backing_table_address=100,
            table_values=[[value]],
            exact_lengths=[1],
            upper_lengths=[1],
            record_scope={"recorder_id": "r", "process_id": 1, "provider_identity": 9},
        )
        for i, (epoch, value) in enumerate([(30, 1), (20, 2), (10, 1)])
    ]
    raw = dict(source_identity={"table": 50}, full_exact=[1], full_upper=[1], valid_counts=[1], mapping=[0])
    records = [dict(r, sequence=r["sequence"] + 1, runtime_oracle_inputs=raw) for r in bound]
    assert _tool().check_aba(bound, records) == "FAIL"


@pytest.mark.parametrize("field", ["window_table_address", "route"])
def test_review_g4_must_resolve_bound_backing_and_route(field):
    doc = _complete_host_fixture()
    doc["snapshot"]["g4_records"][0][field] = 999
    assert _tool().audit_campaign(doc)["EVIDENCE_INTEGRITY"] == "INVALID_EVIDENCE"


@pytest.mark.parametrize("mutation", ["active", "tail", "padded_row", "padded_length", "missing_row"])
def test_bound_oracle_validates_active_content_and_padding_even_if_cached_events_agree(mutation):
    doc = _complete_host_fixture()
    events = [e for e in doc["snapshot"]["events"] if e.get("generation_id") == 1]
    for event in events:
        event["table_values"] = deepcopy(event["table_values"])
        if mutation == "active":
            event["table_values"][0][0] = 999
        elif mutation == "tail":
            event["table_values"][0][-1] = 999
        elif mutation == "padded_row":
            event["table_values"].append([999] * 9)
        elif mutation == "padded_length":
            event["upper_lengths"] = [1024, 99]
        else:
            event["table_values"] = []
    result = _tool().audit_campaign(doc)
    assert result["EVIDENCE_INTEGRITY"] == "INVALID_EVIDENCE"
    assert any("oracle mismatch" in error for error in result["errors"])


def test_zero_extra_capacity_and_padded_rows_need_not_have_identical_shapes():
    doc = _complete_host_fixture()
    for record in doc["snapshot"]["g4_records"]:
        record["window_table_values"] = [row + [0, 0] for row in record["window_table_values"]]
        record["window_table_values"].append([0] * 11)
    assert _tool().audit_campaign(doc)["HOST_CONTRACTS"] == "PASS"


def test_single_boundary_bucket_is_valid_host_evidence_but_not_complete_smoke():
    doc = _complete_host_fixture()
    descriptor = "mode=FULL;reqs=1;tokens=7;uniform=7"
    for section in ("events", "operand_events", "g4_records"):
        doc["snapshot"][section] = [r for r in doc["snapshot"][section] if r["capture_descriptor"] == descriptor]
    doc["draft_graph_keys"] = [[1, 7]]
    result = _tool().audit_campaign(doc)
    assert result["HOST_CONTRACTS"] == "PASS"
    assert result["DEVICE_FUNCTIONAL_GATES"]["fixed_inventory_replay"] == "OPEN"
    assert result["missing_replay_buckets"] == result["missing_inventory_buckets"] == [(2, 14), (4, 28)]


@pytest.mark.parametrize("kind", ["duplicate_bound", "missing_g4", "orphan_g4", "cached_sample", "no_capture"])
def test_generation_associations_never_pass_incomplete_evidence(kind):
    doc = _complete_host_fixture()
    snap = doc["snapshot"]
    if kind == "duplicate_bound":
        snap["events"].append(dict(next(e for e in snap["events"] if e["stage"] == "bound"), sequence=1000))
    elif kind == "missing_g4":
        snap["g4_records"].pop()
    elif kind == "orphan_g4":
        snap["g4_records"][0]["generation_id"] = 999
    elif kind == "cached_sample":
        next(e for e in snap["events"] if e["stage"] == "replay")["values_sampled_at_sequence"] = 999
    else:
        snap["events"] = [e for e in snap["events"] if e["stage"] != "capture"]
    result = _tool().audit_campaign(doc)
    assert result["HOST_CONTRACTS"] != "PASS"
    assert result["DEVICE_FUNCTIONAL_GATES"]["fixed_inventory_replay"] != "PASS"


def test_repeated_capture_helper_uses_preceding_sample_not_last_dictionary_entry():
    doc = _complete_host_fixture()
    # A later legal capture of the first descriptor must not authorize an
    # earlier operand. Its startup lengths may have changed in between.
    sample = next(e for e in doc["snapshot"]["events"] if e["stage"] == "capture")
    doc["snapshot"]["events"].append(dict(sample, sequence=1000, exact_lengths=[2]))
    assert _tool().audit_campaign(doc)["HOST_CONTRACTS"] == "PASS"
    doc["snapshot"]["events"][-1]["backing_table_address"] = 999
    assert _tool().audit_campaign(doc)["EVIDENCE_INTEGRITY"] == "INVALID_EVIDENCE"


@pytest.mark.parametrize("field", ["generation_id", "mapping_epoch"])
def test_runtime_order_is_strict_and_uses_sequence_not_json_array_order(field):
    doc = _complete_host_fixture()
    for section in ("events", "operand_events", "g4_records"):
        for record in doc["snapshot"][section]:
            if record.get("generation_id") is not None:
                record[field] = 10 - record[field]
    assert _tool().audit_campaign(doc)["EVIDENCE_INTEGRITY"] == "INVALID_EVIDENCE"
    valid = _complete_host_fixture()
    valid["snapshot"]["events"].reverse()
    assert _tool().audit_campaign(valid)["HOST_CONTRACTS"] == "PASS"


@pytest.mark.parametrize("field", ["record_scope", "capture_descriptor", "backing_table_address"])
def test_aba_cannot_splice_different_owners_or_backings(field):
    bound = [
        dict(
            sequence=n * 10,
            generation_id=n,
            mapping_epoch=n + 100,
            capture_descriptor="same",
            backing_table_address=100,
            table_values=[[v]],
            exact_lengths=[1],
            upper_lengths=[1],
            record_scope={"recorder_id": "r", "process_id": 1, "provider_identity": 9},
        )
        for n, v in ((1, 1), (3, 2), (8, 1))
    ]
    raw = dict(source_identity={"table": 50}, full_exact=[1], full_upper=[1], valid_counts=[1], mapping=[0])
    records = [dict(r, sequence=r["sequence"] + 1, runtime_oracle_inputs=raw) for r in bound]
    assert _tool().check_aba(bound, records) == "PASS"  # monotonic gaps permitted
    bound[1][field] = {"provider_identity": 99} if field == "record_scope" else "other"
    records[1][field] = bound[1][field]
    assert _tool().check_aba(bound, records) != "PASS"


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("kind", ["valid", "wrong_address", "wrong_table", "truncated_json", "bad_shape", "null_root"])
def test_check_cli_writes_explicit_report_even_under_optimization(tmp_path, optimized, kind):
    doc = _complete_host_fixture()
    if kind == "wrong_address":
        doc["snapshot"]["operand_events"][0].update(actual_table_address=999, expected_table_address=999)
    elif kind == "wrong_table":
        doc["snapshot"]["g4_records"][0]["window_table_values"][0][0] = 999
    elif kind == "bad_shape":
        doc["snapshot"]["events"][0]["sequence"] = []
    payload = "{" if kind == "truncated_json" else json.dumps(None if kind == "null_root" else doc)
    artifact, output = tmp_path / "input.json", tmp_path / "result.json"
    artifact.write_text(payload)
    command = [
        sys.executable,
        *(["-O"] if optimized else []),
        str(Path(_tool().__file__)),
        "check",
        str(artifact),
        "--output",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == (0 if kind == "valid" else 2), result.stderr
    report = json.loads(output.read_text())
    assert report["EVIDENCE_INTEGRITY"] == ("PASS" if kind == "valid" else "INVALID_EVIDENCE")
    assert report["DEVICE_EVIDENCE_LEVELS"]["physical_read"] == "OPEN"


def test_real_recorder_snapshot_to_checker_preserves_generation_and_padding_contract():
    import numpy as np

    from tests.ut.spec_decode.test_draft_window_graph_carrier import _materialize
    from tests.ut.spec_decode.test_draft_window_graph_characterization import (
        _characterization_owner,
        _install_metadata_builder,
        _publish_eager_generation,
        _runtime_context,
    )
    from vllm_ascend.attention.attention_v1 import (
        AscendAttentionBackend,
        AscendAttentionBackendImpl,
        AscendAttentionMetadataBuilder,
    )
    from vllm_ascend.worker.v2.spec_decode.draft_window_graph import (
        DraftWindowGraphRoute,
        DraftWindowGraphStage,
        build_draft_window_graph_update_binding,
        consume_draft_window_graph_metadata,
        observe_draft_window_graph_source,
        validate_draft_window_graph_backend_update,
    )
    from vllm_ascend.worker.v2.spec_decode.draft_window_graph_characterization import (
        install_parallel_draft_window_graph_characterization,
        prepare_parallel_draft_window_graph_characterization,
        record_parallel_graph_update_operator_operand,
    )

    owner, descriptor = _characterization_owner()
    owner.input_batch.idx_mapping_np = np.array([1, 3])
    owner.input_batch.seq_lens_cpu_upper_bound = np.array([96, 96])
    owner.input_buffers.seq_lens[:2] = 96
    owner.max_model_len = 512
    session = install_parallel_draft_window_graph_characterization(owner, runtime_context=_runtime_context())
    identity = session.provider.registered_capture_identities(route=DraftWindowGraphRoute.DFLASH)[0]
    read = session.provider.startup_capture_read_inputs(identity)
    session.recorder.record_capture(
        identity=identity,
        read_inputs=read,
        slot_mapping=owner.block_tables.slot_mappings[0, :8],
        positions=owner.input_buffers.positions[:8],
        query_start_loc=owner.input_buffers.query_start_loc[:5],
    )
    layer_name = "draft.layers.0.self_attn"
    session.recorder.record_capture_operand(
        identity=identity,
        block_table=read.window_block_table,
        seq_lengths=read.cpu_window_exact.tolist(),
        layer_name=layer_name,
        num_tokens=8,
        handle=None,
        event=None,
        stream=None,
        backend_impl=AscendAttentionBackendImpl,
    )
    inputs = _materialize(
        owner._draft_window_runtime_state, mapping_epoch=10, exact=(96, 96, 0, 0), upper=(98, 98, 0, 0)
    )
    _publish_eager_generation(owner, inputs)
    _install_metadata_builder(owner)
    carrier = prepare_parallel_draft_window_graph_characterization(owner, descriptor)
    batch = consume_draft_window_graph_metadata(owner, route=DraftWindowGraphRoute.DFLASH, expected_consumer_count=1)
    observe_draft_window_graph_source(owner, DraftWindowGraphStage.REPLAY)
    observe_draft_window_graph_source(owner, DraftWindowGraphStage.CONSUMER)
    binding = build_draft_window_graph_update_binding(
        owner, route=DraftWindowGraphRoute.DFLASH, draft_attn_metadatas=list(batch)
    )
    validate_draft_window_graph_backend_update(
        binding,
        attn_backend=AscendAttentionBackend,
        impl_cls=AscendAttentionBackendImpl,
        builder_cls=AscendAttentionMetadataBuilder,
        draft_attn_metadatas=list(batch),
        update_function_path="vllm_ascend.compilation.acl_graph.update_full_graph_params",
    )
    layer = carrier.metadata[layer_name]
    record_parallel_graph_update_operator_operand(
        binding,
        draft_metadata=carrier.metadata,
        layer_name=layer_name,
        captured_block_table=read.window_block_table,
        actual_block_table=layer.block_tables,
        seq_lengths=layer.seq_lens_list,
        num_tokens=8,
        handle=None,
        event=None,
        stream=None,
        backend_impl=AscendAttentionBackendImpl,
    )
    doc = _empty()
    doc["snapshot"] = session.recorder.snapshot()
    doc["identity"]["process_id"] = doc["snapshot"]["process_id"]
    doc.update(startup_complete=True, status="OK", draft_layer_names=[layer_name], draft_graph_keys=[[4, 8]])
    result = _tool().audit_campaign(doc)
    assert result["HOST_CONTRACTS"] == "PASS", json.dumps(result, indent=2)
    # This real recorder CPU fixture is not the fixed server smoke inventory.
    assert result["DEVICE_FUNCTIONAL_GATES"]["fixed_inventory_replay"] == "OPEN"
    assert result["missing_replay_buckets"] == [(1, 7), (2, 14), (4, 28)]
    assert result["DEVICE_EVIDENCE_LEVELS"]["completion_and_effective_generation"] == "OPEN"
    update = next(e for e in doc["snapshot"]["events"] if e["stage"] == "update")
    bound = next(e for e in doc["snapshot"]["events"] if e["stage"] == "bound")
    assert update["values_sampled_at_sequence"] == bound["sequence"]
    assert len(bound["table_values"]) == 4  # two active + two actual zero-padded rows
    broken = deepcopy(doc)
    broken["snapshot"]["operand_events"][-1].update(actual_table_address=999, expected_table_address=999)
    assert _tool().audit_campaign(broken)["EVIDENCE_INTEGRITY"] == "INVALID_EVIDENCE"
