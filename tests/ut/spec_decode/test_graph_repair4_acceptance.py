# SPDX-License-Identifier: Apache-2.0
"""Controlled CPU fixtures exercise the checker, never stand in for a producer."""

import hashlib
import importlib.util
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / "tools/p7x_graph_repair4_acceptance.py"


def _tool():
    spec = importlib.util.spec_from_file_location("repair4_acceptance", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _campaign():
    identity = dict(
        run_id="run",
        case_id="B0",
        attempt_id="1",
        source_sha="a" * 40,
        source_tree="b" * 40,
        vllm_sha="84030bbe3d74d99bad477a3d2e37a973ccd8865c",
        engine_id=123,
        process_id=1,
    )
    scope = dict(recorder_id="rec", process_id=1, provider_identity=9)
    snap = dict(
        schema="mrv2-draft-window-graph-characterization-v2",
        **scope,
        artifact_errors=[],
        events=[],
        operand_events=[],
        g4_records=[],
        lifecycle_events=[],
    )
    doc = dict(
        schema="p7x-repair2-campaign-v1",
        identity=identity,
        snapshot=snap,
        status="OK",
        startup_complete=True,
        draft_layer_names=["layer.0", "layer.1"],
        draft_graph_keys=[[1, 7], [2, 14], [4, 28]],
        outputs=[],
    )
    sequence = 0

    def add(section, **fields):
        nonlocal sequence
        record = dict(sequence=sequence, record_scope=deepcopy(scope), **fields)
        snap[section].append(record)
        sequence += 1
        return record

    for n in (1, 2, 4):
        desc = f"mode=FULL;reqs={n};tokens={n * 7};uniform=7"
        base = dict(route="dflash", capture_descriptor=desc, generation_id=None, mapping_epoch=None)
        add("events", **base, stage="capture", backing_table_address=200, exact_lengths=[1] * n)
        op = dict(
            provider_identity=9,
            backend_impl="FIA",
            consumer="FIA.out",
            actual_table_address=200,
            expected_table_address=200,
            captured_table_address=200,
            table_shape=[n, 9],
            table_dtype="int32",
            evidence_level="HOST_OPERATOR_BOUNDARY_FACT",
        )
        for layer in doc["draft_layer_names"]:
            add("operand_events", **base, **op, stage="capture_operator", layer_name=layer, cpu_exact=[1] * n)
        base.update(generation_id=n, mapping_epoch=n)
        values = dict(
            backing_table_address=200,
            table_values=[[101, 102, 103, 104, 105, 106, 107, 108, 0]] * n,
            exact_lengths=[1024] * n,
            upper_lengths=[1024] * n,
        )
        bound = add("events", **base, **values, stage="bound", values_sampled_at_sequence=None)
        for stage in ("replay", "consumer"):
            add("events", **base, **values, stage=stage, values_sampled_at_sequence=bound["sequence"])
        life = dict(generation_id=n, mapping_epoch=n)
        add("lifecycle_events", **life, stage="replay_submitted", stream_identity=4)
        for index, layer in enumerate(doc["draft_layer_names"]):
            event_id = 1000 + n * 10 + index
            add(
                "operand_events",
                **base,
                **op,
                stage="update_operator",
                layer_name=layer,
                cpu_exact=[1024] * n,
                handle_identity=index + 1,
                event_identity=event_id,
                stream_identity=4,
            )
            add("lifecycle_events", **life, stage="update_event_recorded", event_identity=event_id, stream_identity=4)
        add("events", **base, **values, stage="update", values_sampled_at_sequence=bound["sequence"])
        add(
            "g4_records",
            **base,
            canonical_table_address=100,
            canonical_table_values=[list(range(100, 120))] * n,
            window_table_address=200,
            window_table_values=values["table_values"],
            draft_output_hash=None,
            target_output_hash=None,
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
        add("lifecycle_events", **life, stage="proposal_retired", lease_state="retired")
        doc["outputs"].append(
            dict(
                label=f"smoke-{n}",
                host_start_ns=n * 100,
                host_end_ns=n * 100 + 1,
                prompt_token_ids=[[1]] * n,
                token_ids=[[2, 3]] * n,
            )
        )
    return doc


def joint_inputs(case="B0"):
    document = _campaign()
    document["identity"].update(case_id=case, driver_mode="A1" if case == "A1" else "B")
    requests = [{"label": f"smoke-{n}", "prompts": ["Hello"] * n, "max_tokens": 16} for n in (1, 2, 4)]
    payload = json.dumps(requests).encode()
    fingerprint = {"sha256": hashlib.sha256(payload).hexdigest(), "entries": requests}
    document["request_manifest"] = fingerprint
    for entry in document["outputs"]:
        entry["token_ids"] = [[3] * 16 for _ in entry["token_ids"]]
        entry["prompts"] = ["Hello"] * len(entry["token_ids"])
    argv = ["python", "tools/p7x_graph_repair2.py", "run"]
    expected = dict(document["identity"], requests_sha256=fingerprint["sha256"], argv=argv)
    launcher = dict(
        document["identity"],
        requests_sha256=fingerprint["sha256"],
        argv=argv,
        pid=1,
        status="EXITED",
        returncode=0,
        signal=None,
        timeout=False,
        host_start_ns=1,
        host_end_ns=1000,
    )
    return document, launcher, payload, expected


def test_joint_acceptance_requires_tokens_and_external_success():
    tool = _tool()
    doc, launch, payload, expected = joint_inputs()
    assert tool.audit_joint(doc, launch, payload, expected)["status"] == "PASS"
    doc["outputs"][0]["token_ids"][0].pop()
    assert tool.audit_campaign(doc)["DEVICE_FUNCTIONAL_GATES"]["B0"] == "PASS"
    assert tool.audit_joint(doc, launch, payload, expected)["status"] == "FAIL"


@pytest.mark.parametrize(
    "fault",
    [
        "timeout",
        "signal",
        "returncode",
        "run",
        "case",
        "attempt",
        "pid",
        "sha",
        "requests",
        "argv",
        "missing_output",
        "extra_output",
        "max_tokens",
    ],
)
def test_joint_acceptance_rejects_cross_artifact_mismatch(fault):
    doc, launch, payload, expected = joint_inputs()
    if fault in ("timeout", "signal", "returncode"):
        launch[fault] = {"timeout": True, "signal": 11, "returncode": 2}[fault]
    elif fault in ("run", "case", "attempt"):
        launch[fault + "_id"] = "other"
    elif fault == "pid":
        launch["pid"] = 999
    elif fault == "sha":
        launch["source_sha"] = "c" * 40
    elif fault == "requests":
        payload += b" "
    elif fault == "argv":
        launch["argv"] = launch["argv"] + ["--other"]
    elif fault == "missing_output":
        doc["outputs"].pop()
    elif fault == "extra_output":
        doc["outputs"].append(deepcopy(doc["outputs"][0]))
    else:
        doc["request_manifest"]["entries"][0]["max_tokens"] = 15
    assert _tool().audit_joint(doc, launch, payload, expected)["status"] != "PASS"


def test_joint_expected_public_gate_is_not_normal_or_resource_failure():
    doc, launch, payload, expected = joint_inputs("A1")
    doc.update(
        status="EXCEPTION",
        startup_complete=False,
        outputs=[],
        exception_type="NotImplementedError",
        exception_message="GRAPH_FROZEN",
    )
    result = _tool().audit_joint(doc, launch, payload, expected)
    assert result["status"] == "PASS" and result["acceptance_kind"] == "EXPECTED_PUBLIC_GATE"
    doc["exception_message"] = "out of memory"
    assert _tool().audit_joint(doc, launch, payload, expected)["status"] != "PASS"


def test_joint_cli_joins_real_json_files_and_keeps_raw_records(tmp_path):
    doc, launch, payload, expected = joint_inputs()
    paths = {name: tmp_path / (name + ".json") for name in ("campaign", "launcher", "plan", "requests")}
    plan = {
        "run_id": "run",
        "commit": expected["source_sha"],
        "tree": expected["source_tree"],
        "cases": {
            "B0": {
                "acceptance_identity": expected,
                "argv": expected["argv"],
                "requests_sha256": expected["requests_sha256"],
            }
        },
    }
    for name, data in (("campaign", doc), ("launcher", launch), ("plan", plan)):
        paths[name].write_text(json.dumps(data))
    paths["requests"].write_bytes(payload)
    before = {k: p.read_bytes() for k, p in paths.items()}
    command = [sys.executable, str(TOOL), "check-joint", str(paths["campaign"]), "--case-id", "B0"]
    for name in ("launcher", "plan", "requests"):
        command += ["--" + name, str(paths[name])]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["original_b0_checker"]["DEVICE_FUNCTIONAL_GATES"]["B0"] == "PASS"
    assert before == {k: p.read_bytes() for k, p in paths.items()}


def test_real_external_launcher_fingerprints_request_bytes(tmp_path):
    requests = tmp_path / "requests.json"
    requests.write_text('[{"label":"request","prompts":["hi"],"max_tokens":16}]')
    output = tmp_path / "launcher"
    args = SimpleNamespace(
        output=output,
        run_id="host",
        case_id="A0",
        mode="A0",
        attempt="1",
        timeout=10,
        argv=[sys.executable, "-c", "print('host child only')", "--requests", str(requests)],
    )
    assert _tool()._repair2().launch_case(args) == 0
    record = json.loads((output / "launcher.json").read_text())
    assert record["requests_sha256"] == hashlib.sha256(requests.read_bytes()).hexdigest()
    assert record["returncode"] == 0 and record["signal"] is None and record["timeout"] is False


@pytest.mark.parametrize("fault", [None, "signal", "timeout", "unrelated_error", "cross_run", "reuse"])
def test_expected_abort_nonzero_is_separate_from_normal_acceptance(fault):
    manifest = {"abort_at": "after_replay", "run_id": "run", "attempt": "abort"}
    doc = {
        "identity": deepcopy(manifest),
        "manifest_digest": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        "status": "ABORTED",
        "error": "RuntimeError: injected abort after replay; no completion claim",
        "process_exit": {"returncode": 2, "timeout": False},
        "abort": {"unsafe_reuse_rejected": True, "reused": False},
    }
    if fault == "signal":
        doc["process_exit"]["returncode"] = -11
    elif fault == "timeout":
        doc["process_exit"]["timeout"] = True
    elif fault == "unrelated_error":
        doc["error"] = "RuntimeError: out of memory"
    elif fault == "cross_run":
        doc["identity"]["run_id"] = "other"
    elif fault == "reuse":
        doc["abort"]["reused"] = True
    result = _tool().audit_abort(doc, manifest)
    assert (result["status"] == "EXPECTED_ABORT_OBSERVED") == (fault is None)
    assert result["completion"] == "NOT_ASSESSED"


def _tensor(values, dtype="int64"):
    return dict(shape=[len(values)], dtype=dtype, values=values)


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _target_pair():
    block = dict(
        owner="r0", version=1, reuse_epoch=0, key=_tensor([1.0, 2.0], "float32"), value=_tensor([3.0, 4.0], "float32")
    )
    state = dict(
        token_ids=_tensor([4]),
        positions=_tensor([1]),
        query_start_loc=_tensor([0, 1]),
        request_ids=["r0"],
        sequence_lengths=_tensor([2]),
        slot_mapping=_tensor([1]),
        target_layers={"l0": dict(block_table=dict(shape=[1, 1], dtype="int64", values=[[0]]), kv={"0": block})},
    )
    record = dict(
        sample_id="s0",
        entrypoint="model.forward",
        executed=True,
        call_id="call-1",
        before_phase="target_forward_pre",
        after_phase="target_forward_post",
        snapshot_call_id="call-1",
        snapshot_consistent=True,
        before=state,
        after=deepcopy(state),
        forward_output=_tensor([0.2, 0.8], "float32"),
    )
    record["aligned_input_state_sha256"] = _digest(state)
    doc = dict(
        schema="p7x-repair4-full-target-v1",
        identity=dict(run_id="run", case_id="G4-OFF", attempt_id="1"),
        fixture_id="controlled",
        model_sha256="a" * 64,
        target_layer_names=["l0"],
        selected_blocks={"l0": [0]},
        block_span_tokens=2,
        tolerance=dict(atol=0.0, rtol=0.0, reason="exact controlled CPU test"),
        records=[record],
    )
    other = deepcopy(doc)
    other["identity"]["case_id"] = "G4-ON"
    return doc, other


def test_b0_requires_more_than_legacy_presence_pass():
    tool = _tool()
    doc = _campaign()
    assert tool.audit_campaign(doc)["DEVICE_FUNCTIONAL_GATES"]["B0"] == "PASS"
    doc["outputs"] = None
    doc["audit"] = {"G4_WRITE_TARGET_ORACLE": "PASS", "ready_for_device_campaign": True}
    result = tool.audit_campaign(doc)
    assert result["DEVICE_FUNCTIONAL_GATES"]["B0"] == "FAIL"
    assert result["NUMERIC_GATES"]["draft_output"] == "OPEN"
    assert result["PRODUCER_STATUS"]["G4F"] == "BLOCKED"


@pytest.mark.parametrize(
    "mutation",
    [
        "empty_outputs",
        "null_tokens",
        "empty_tokens",
        "bool_token",
        "failed",
        "capture_only",
        "missing_bucket",
        "missing_layer",
        "no_retirement",
        "one_retirement",
        "early_retirement",
        "orphan_retirement",
        "duplicate_retirement",
        "event_mismatch",
        "duplicate_layer",
        "identity",
    ],
)
def test_b0_fail_closed(mutation):
    doc = _campaign()
    snap = doc["snapshot"]
    if mutation == "empty_outputs":
        doc["outputs"] = []
    elif mutation in ("null_tokens", "empty_tokens", "bool_token"):
        doc["outputs"][0]["token_ids"] = {"null_tokens": None, "empty_tokens": [[]], "bool_token": [[True]]}[mutation]
    elif mutation == "failed":
        doc["status"] = "EXCEPTION"
    elif mutation == "capture_only":
        snap["events"] = [e for e in snap["events"] if e["stage"] == "capture"]
    elif mutation == "missing_bucket":
        for key in ("events", "operand_events", "g4_records", "lifecycle_events"):
            snap[key] = [e for e in snap[key] if e.get("generation_id") != 4]
    elif mutation == "missing_layer":
        snap["operand_events"] = [e for e in snap["operand_events"] if e["layer_name"] != "layer.1"]
    elif mutation in ("no_retirement", "one_retirement"):
        snap["lifecycle_events"] = [
            e
            for e in snap["lifecycle_events"]
            if e["stage"] != "proposal_retired" or (mutation == "one_retirement" and e["generation_id"] == 1)
        ]
    elif mutation in ("early_retirement", "orphan_retirement", "duplicate_retirement"):
        event = next(e for e in snap["lifecycle_events"] if e["stage"] == "proposal_retired")
        if mutation == "early_retirement":
            event["sequence"] = -1
        elif mutation == "orphan_retirement":
            event["generation_id"] = 99
        else:
            snap["lifecycle_events"].append(dict(event, sequence=1000))
    elif mutation == "event_mismatch":
        next(e for e in snap["lifecycle_events"] if e["stage"] == "update_event_recorded")["event_identity"] = 9000
    elif mutation == "duplicate_layer":
        event = next(e for e in snap["operand_events"] if e["stage"] == "update_operator")
        snap["operand_events"].append(dict(event, sequence=1000, handle_identity=999))
    else:
        doc["identity"]["attempt_id"] = None
    assert _tool().audit_campaign(doc)["DEVICE_FUNCTIONAL_GATES"]["B0"] != "PASS"


@pytest.mark.parametrize("document", [None, [], {}, {"identity": []}, {"snapshot": None}])
def test_malformed_campaign_is_machine_readable(document):
    result = _tool().audit_campaign(document)
    assert result["DEVICE_FUNCTIONAL_GATES"]["B0"] != "PASS"
    assert result["EVIDENCE_INTEGRITY"] == "INVALID_EVIDENCE"


def test_hashes_do_not_silently_become_numeric_comparison():
    doc = _campaign()
    for record in doc["snapshot"]["g4_records"]:
        record.update(draft_output_hash="a" * 64, target_output_hash="b" * 64)
    result = _tool().audit_campaign(doc)
    assert result["OUTPUT_EVIDENCE"]["draft_hashes"] == "PRESENT"
    assert result["NUMERIC_GATES"]["draft_output"] == result["NUMERIC_GATES"]["target_output"] == "OPEN"
    assert result["DEVICE_EVIDENCE_LEVELS"]["completion_and_effective_generation"] == "OPEN"


def test_expected_identity_and_trace_cases_are_not_interchangeable():
    result = _tool().audit_campaign(_campaign(), expected_identity=dict(run_id="other", case_id="B0", attempt_id="1"))
    assert result["EVIDENCE_INTEGRITY"] == "INVALID_EVIDENCE"
    doc = _campaign()
    doc["identity"]["case_id"] = "T0"
    assert _tool().audit_campaign(doc)["DEVICE_FUNCTIONAL_GATES"]["B0"] == "NOT_APPLICABLE"


@pytest.mark.parametrize("case_id", ["B1", "B2", "T0", "T1"])
def test_non_b0_cases_do_not_require_smoke_output_cardinalities(case_id):
    doc = _campaign()
    doc["identity"]["case_id"] = case_id
    doc["outputs"] = doc["outputs"][:1]
    result = _tool().audit_campaign(doc)
    assert result["DEVICE_FUNCTIONAL_GATES"]["B0"] == "NOT_APPLICABLE"
    assert "normal_outputs" not in result["HOST_OBSERVATIONS"]


@pytest.mark.parametrize("mutation", ["wrong_mode", "substring", "duplicate_key", "wrong_width"])
def test_b0_parses_actual_replay_descriptors(mutation):
    doc = _campaign()
    for section in ("events", "operand_events", "g4_records"):
        for record in doc["snapshot"][section]:
            descriptor = record["capture_descriptor"]
            if mutation == "wrong_mode":
                descriptor = descriptor.replace("mode=FULL", "mode=PIECEWISE")
            elif mutation == "substring":
                descriptor = descriptor.replace("reqs=", "unrelated_reqs=")
            elif mutation == "duplicate_key":
                descriptor += ";tokens=999"
            else:
                descriptor = descriptor.replace("uniform=7", "uniform=8")
            record["capture_descriptor"] = descriptor
    result = _tool().audit_campaign(doc)
    assert result["DEVICE_FUNCTIONAL_GATES"]["B0"] == "FAIL"
    assert result["HOST_OBSERVATIONS"]["runtime_replays"]["status"] == "FAIL"


def test_trace_inventory_reads_headers_rows_and_never_infers_operands(tmp_path):
    path = tmp_path / "op_summary.csv"
    path.write_text(
        'Op Name,Task ID,Input Shapes,Input Data Types\nFusedInferAttentionScore,7,"1,128",INT32\nOther,8,,\n'
    )
    result = _tool().trace_fields([path], dict(run_id="r", case_id="T0", attempt_id="1"))
    file = result["files"][0]
    assert file["headers"] == ["Op Name", "Task ID", "Input Shapes", "Input Data Types"]
    assert file["row_count"] == 2 and file["nonempty_counts"]["Input Shapes"] == 1
    assert file["attention_rows"] == 1
    assert result["identity"]["case_id"] == "T0"
    assert result["DEVICE_EVIDENCE_LEVELS"]["task_internal"] == "OPEN"
    assert result["DEVICE_EVIDENCE_LEVELS"]["physical_read"] == "OPEN"


@pytest.mark.parametrize("content", ["", "A,A\n1,2\n", "A,B\n1\n", "A\n1,2\n"])
def test_trace_malformed_csv_fails_closed(tmp_path, content):
    path = tmp_path / "bad.csv"
    path.write_text(content)
    assert _tool().trace_fields([path], dict(run_id="r", case_id="T1", attempt_id="1"))["status"] != "PASS"


def test_g4_fixture_checker_does_not_verify_actual_producer_pair():
    tool = _tool()
    result = tool.compare_full_target(*_target_pair())
    assert result["status"] == "PASS"
    assert result["G4F"] == "BLOCKED"
    assert result["checker"] == "SUPPLIED_EVIDENCE_ONLY"
    assert result["producer"] == "tools/p7x_graph_repair4_target.py"
    assert result["live_aligned_pair"] == "NOT_VERIFIED"


@pytest.mark.parametrize(
    "mutation",
    [
        "head_only",
        "not_executed",
        "phase",
        "call",
        "snapshot_race",
        "missing_kv",
        "missing_layer",
        "missing_selected",
        "uncovered_block",
        "after_kv",
        "tokens",
        "positions",
        "query",
        "request",
        "version",
        "nan",
        "infinite_tolerance",
        "negative_tolerance",
        "shape",
        "dtype",
        "extra_sample",
        "duplicate_sample",
        "output",
        "identity",
        "hash",
    ],
)
def test_full_target_rejects_incomplete_or_misaligned_fixture(mutation):
    left, right = _target_pair()
    record = right["records"][0]
    if mutation == "head_only":
        record["entrypoint"] = "model.compute_logits"
    elif mutation == "not_executed":
        record["executed"] = False
    elif mutation == "phase":
        record["before_phase"] = "before_propose"
    elif mutation == "call":
        record["snapshot_call_id"] = "other"
    elif mutation == "snapshot_race":
        record["snapshot_consistent"] = False
    elif mutation == "missing_kv":
        record["before"]["target_layers"]["l0"]["kv"] = {}
    elif mutation == "missing_layer":
        record["after"]["target_layers"] = {}
    elif mutation == "missing_selected":
        right["selected_blocks"] = {}
    elif mutation == "uncovered_block":
        record["after"]["target_layers"]["l0"]["block_table"]["values"] = [[1]]
    elif mutation == "after_kv":
        record["after"]["target_layers"]["l0"]["kv"]["0"]["value"]["values"][0] = 99.0
    elif mutation in ("tokens", "positions", "query", "request"):
        key = dict(tokens="token_ids", positions="positions", query="query_start_loc", request="request_ids")[mutation]
        record["before"][key] = None
    elif mutation == "version":
        del record["before"]["target_layers"]["l0"]["kv"]["0"]["version"]
    elif mutation == "nan":
        record["forward_output"]["values"][0] = float("nan")
    elif mutation in ("infinite_tolerance", "negative_tolerance"):
        right["tolerance"]["atol"] = float("inf") if mutation == "infinite_tolerance" else -1
    elif mutation == "shape":
        record["forward_output"]["shape"] = [3]
    elif mutation == "dtype":
        record["forward_output"]["dtype"] = "float16"
    elif mutation in ("extra_sample", "duplicate_sample"):
        right["records"].append(deepcopy(record))
        if mutation == "extra_sample":
            right["records"][-1]["sample_id"] = "s1"
    elif mutation == "output":
        record["forward_output"]["values"][0] = 55.0
    elif mutation == "identity":
        right["identity"]["attempt_id"] = None
    else:
        record["aligned_input_state_sha256"] = "b" * 64
    assert _tool().compare_full_target(left, right)["status"] != "PASS"


@pytest.mark.parametrize("value", [None, ""])
def test_missing_hash_stays_open(value):
    left, right = _target_pair()
    left["records"][0]["aligned_input_state_sha256"] = value
    right["records"][0]["aligned_input_state_sha256"] = value
    assert _tool().compare_full_target(left, right)["status"] == "OPEN"


def test_cli_check_trace_and_full_target(tmp_path):
    def run(*args):
        return subprocess.run([sys.executable, str(TOOL), *map(str, args)], capture_output=True, text=True)

    assert run("--help").returncode == 0
    campaign = tmp_path / "campaign.json"
    campaign.write_text(json.dumps(_campaign()))
    result = run("check", campaign, "--run-id", "run", "--case-id", "B0", "--attempt-id", "1")
    assert result.returncode == 0
    assert json.loads(result.stdout)["DEVICE_FUNCTIONAL_GATES"]["B0"] == "PASS"
    campaign.write_text('{"outputs": null}')
    assert run("check", campaign).returncode == 2
    campaign.write_text("{broken")
    assert run("check", campaign).returncode == 2
    left, right = _target_pair()
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(left))
    b.write_text(json.dumps(right))
    result = run("check-target", a, b)
    assert result.returncode == 0 and json.loads(result.stdout)["G4F"] == "BLOCKED"
    trace = tmp_path / "trace.csv"
    trace.write_text("Task ID,Input Shapes\n1,128\n")
    result = run("trace-fields", trace, "--run-id", "r", "--case-id", "T1", "--attempt-id", "1")
    assert result.returncode == 0 and json.loads(result.stdout)["identity"]["case_id"] == "T1"


def test_null_graph_output_status_does_not_fail_normal_generation():
    doc = _campaign()
    for record in doc["snapshot"]["g4_records"]:
        record.update(
            draft_output_status="OUTPUT_NOT_CAPTURED",
            draft_output_reason="graph returned None",
            target_output_status="OUTPUT_NOT_CAPTURED",
            target_output_reason="TARGET_OUTPUT_NOT_SUPPLIED",
        )
    result = _tool().audit_campaign(doc)
    assert result["DEVICE_FUNCTIONAL_GATES"]["B0"] == "PASS"
    assert result["NUMERIC_GATES"]["draft_output"] == "OPEN"
    assert result["OUTPUT_EVIDENCE"]["records"][0]["draft_output_reason"] == "graph returned None"


def test_full_target_partial_block_content_is_not_complete_kv():
    left, right = _target_pair()
    for doc in (left, right):
        for phase in ("before", "after"):
            block = doc["records"][0][phase]["target_layers"]["l0"]["kv"]["0"]
            block.update(key=_tensor([1.0], "float32"), value=_tensor([2.0], "float32"))
        doc["records"][0]["aligned_input_state_sha256"] = _digest(doc["records"][0]["before"])
    assert _tool().compare_full_target(left, right)["status"] != "PASS"


def test_full_target_before_and_after_cannot_substitute_request_layout():
    left, right = _target_pair()
    for doc in (left, right):
        doc["records"][0]["after"]["token_ids"]["values"] = [9]
    assert _tool().compare_full_target(left, right)["status"] != "PASS"


def test_full_target_allows_declared_finite_tolerance():
    left, right = _target_pair()
    for doc in (left, right):
        doc["tolerance"] = dict(atol=0.01, rtol=0.0, reason="controlled diagnostic tolerance")
    right["records"][0]["forward_output"]["values"][0] += 0.001
    assert _tool().compare_full_target(left, right)["status"] == "PASS"


@pytest.mark.parametrize("value", [None, [], {}, {"records": None}])
def test_malformed_target_is_machine_readable(value):
    assert _tool().compare_full_target(value, value)["status"] == "INVALID_EVIDENCE"


def test_header_presence_alone_and_t0_do_not_supply_t1(tmp_path):
    path = tmp_path / "t0.csv"
    path.write_text("actual_task_table_address,actual_memory_read_set\n,\n")
    result = _tool().trace_fields([path], dict(run_id="r", case_id="T0", attempt_id="1"))
    assert result["files"][0]["nonempty_counts"] == dict(actual_task_table_address=0, actual_memory_read_set=0)
    assert result["DEVICE_EVIDENCE_LEVELS"]["task_internal"] == "OPEN"
    assert result["DEVICE_EVIDENCE_LEVELS"]["physical_read"] == "OPEN"
    missing = _tool().trace_fields([], dict(run_id="r", case_id="T1", attempt_id="1"))
    assert missing["status"] != "PASS" and missing["files"] == []


def test_optimized_python_preserves_repair2_reader_checks(tmp_path):
    for mutation in ("backing", "bound_table", "duplicate_g4", "epoch"):
        doc = _campaign()
        snap = doc["snapshot"]
        if mutation == "backing":
            for event in snap["operand_events"]:
                event.update(actual_table_address=999, expected_table_address=999, captured_table_address=999)
        elif mutation == "bound_table":
            next(e for e in snap["events"] if e["stage"] == "bound")["table_values"] = [[999] * 9]
        elif mutation == "duplicate_g4":
            snap["g4_records"] = [
                dict(deepcopy(snap["g4_records"][0]), sequence=r["sequence"]) for r in snap["g4_records"]
            ]
        else:
            for section in ("events", "operand_events", "g4_records", "lifecycle_events"):
                for event in snap[section]:
                    if event.get("generation_id") == 4:
                        event["mapping_epoch"] = 1
        path = tmp_path / f"{mutation}.json"
        path.write_text(json.dumps(doc))
        result = subprocess.run([sys.executable, "-O", str(TOOL), "check", str(path)], capture_output=True, text=True)
        assert result.returncode == 2
        assert json.loads(result.stdout)["DEVICE_FUNCTIONAL_GATES"]["B0"] == "FAIL"


def test_optimized_target_and_json_reader_fail_closed(tmp_path):
    left, right = _target_pair()
    right["records"][0]["entrypoint"] = "model.compute_logits"
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(left))
    b.write_text(json.dumps(right))
    result = subprocess.run(
        [sys.executable, "-O", str(TOOL), "check-target", str(a), str(b)], capture_output=True, text=True
    )
    assert result.returncode == 2
    assert json.loads(result.stdout)["status"] == "INVALID_EVIDENCE"
    for content in ('{"identity": {}, "identity": {}}', '{"outputs": NaN}'):
        a.write_text(content)
        result = subprocess.run([sys.executable, "-O", str(TOOL), "check", str(a)], capture_output=True, text=True)
        assert result.returncode == 2
        assert json.loads(result.stdout)["status"] == "INVALID_EVIDENCE"
