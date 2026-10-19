# SPDX-License-Identifier: Apache-2.0
"""No empty/vacuous evidence PASS; test-only records never become NPU proof."""

import importlib.util
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
        for stage in ("bound", "replay", "consumer", "update"):
            snap["events"].append(
                dict(
                    sequence=sequence,
                    stage=stage,
                    capture_descriptor=desc,
                    generation_id=n,
                    mapping_epoch=n,
                    backing_table_address=200,
                    table_values=[[101, 102, 103, 104, 105, 106, 107, 108, 0]] * n,
                    exact_lengths=[1024] * n,
                    upper_lengths=[1024] * n,
                )
            )
            sequence += 1
        for layer in doc["draft_layer_names"]:
            snap["operand_events"].append(
                dict(
                    sequence=sequence,
                    stage="update_operator",
                    capture_descriptor=desc,
                    generation_id=n,
                    mapping_epoch=n,
                    provider_identity=9,
                    backend_impl="FIA",
                    consumer="FIA.out",
                    layer_name=layer,
                    actual_table_address=200,
                    expected_table_address=200,
                    table_shape=[n, 9],
                    table_dtype="int32",
                    cpu_exact=[1024] * n,
                    evidence_level="HOST_OPERATOR_BOUNDARY_FACT",
                )
            )
            sequence += 1
            snap["operand_events"].append(
                dict(
                    snap["operand_events"][-1],
                    sequence=sequence,
                    stage="capture_operator",
                    generation_id=None,
                    mapping_epoch=None,
                )
            )
            sequence += 1
        snap["g4_records"].append(
            dict(
                sequence=sequence,
                generation_id=n,
                mapping_epoch=n,
                capture_descriptor=desc,
                canonical_table_address=100,
                canonical_table_values=[list(range(100, 120))] * n,
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
        doc["snapshot"]["operand_events"][0]["generation_id"] = 99
    elif mutation == "table":
        doc["snapshot"]["g4_records"][0]["window_table_values"][0][0] = 100
    elif mutation == "length":
        doc["snapshot"]["events"][0]["exact_lengths"] = [1023]
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
            generation_id=n,
            mapping_epoch=n,
            capture_descriptor="same",
            backing_table_address=100,
            table_values=[[v]],
            exact_lengths=[1],
            upper_lengths=[1],
        )
        for n, v in [(10, 1), (11, 2), (12, 1)]
    ]
    source = dict(source_identity={"table": 50}, full_exact=[1], full_upper=[1], valid_counts=[1], mapping=[0])
    records = [dict(r, runtime_oracle_inputs=source) for r in bound]
    assert tool.check_aba(bound, records) == "PASS"
    bound[2]["mapping_epoch"] = 10
    records[2]["mapping_epoch"] = 10
    assert tool.check_aba(bound, records) == "FAIL"
