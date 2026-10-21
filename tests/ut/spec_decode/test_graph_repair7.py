# SPDX-License-Identifier: Apache-2.0
"""Repair7 real configuration boundary and execution acceptance regressions."""

from types import SimpleNamespace

import pytest

from vllm_ascend.ascend_config import AscendConfig


@pytest.mark.parametrize("raw, expected", [(True, True), (False, False), ("false", False), ("true", True)])
def test_numeric_configuration_schema(raw, expected):
    config = AscendConfig(sparse_kv_offload_config=None, draft_window_graph_numeric_diagnostic=raw)
    assert config.draft_window_graph_numeric_diagnostic is expected


@pytest.mark.parametrize(
    "raw, method, expected", [("false", None, False), (True, "dspark", True), (True, "dflash", None)]
)
def test_numeric_configuration_factory(monkeypatch, raw, method, expected):
    import vllm_ascend.ascend_config as module

    # Hardware/cross-model derivation only is replaced, NOT factory/schema/kwargs.
    monkeypatch.setattr(module.SparseKVOffloadConfig, "from_additional_config", lambda *a: None)
    monkeypatch.setattr(module.KVPPConfig, "from_vllm_config", lambda *a: module.KVPPConfig())
    monkeypatch.setattr(module.AscendConfig, "derive_and_validate", lambda *a: None)
    monkeypatch.setattr(module.FinegrainedTPConfig, "_validate_preconditions", lambda *a: None)
    monkeypatch.setattr(module.XliteGraphConfig, "_validate_preconditions", lambda *a: None)
    config = SimpleNamespace(
        additional_config={"draft_window_graph_numeric_diagnostic": raw, "refresh": True, "draft_window_size": 1024},
        speculative_config=SimpleNamespace(method=method),
    )
    if expected is None:
        with pytest.raises(ValueError, match="requires DSpark"):
            module.init_ascend_config(config)
        return
    result = module.init_ascend_config(config)
    assert result.draft_window_graph_numeric_diagnostic is expected


def test_unknown_configuration_is_still_rejected():
    with pytest.raises(ValueError):
        AscendConfig(sparse_kv_offload_config=None, draft_window_graph_typo=True)


def test_all_plan_configuration_contracts(monkeypatch, tmp_path):
    import hashlib
    import sys

    import tools.p7x_graph_repair5_campaign as old
    from tests.ut.spec_decode.test_graph_repair5_contracts import ROOT, token_joint
    from tools.p7x_graph_repair7 import config_matrix
    from tools.p7x_graph_repair7_campaign import build_plan

    fixture = b'{"files":[]}'
    (tmp_path / "fixture_manifest.json").write_bytes(fixture)
    monkeypatch.setattr(old, "FIXTURE_SHA", hashlib.sha256(fixture).hexdigest())
    (tmp_path / "requests").mkdir()
    for name in ("smoke", "combined", "mixed", "g4-long-group1", "g4-long-group2"):
        (tmp_path / "requests" / f"{name}.json").write_bytes(token_joint()[2])
    plan = build_plan(ROOT, tmp_path, tmp_path / "out", tmp_path / "vllm", sys.executable, "a" * 40, "b" * 40, "r")
    matrix = config_matrix(plan)
    assert matrix["status"] == "CONFIGS_VALID", matrix
    assert len(matrix["rows"]) == 39
    assert plan["cases"]["NUMERIC_TARGET4"]["dependencies"] == ["CONFIG", "NUMERIC_SENTINEL"]
    for case in plan["cases"].values():
        if "launch_argv" in case:
            assert case["launch_argv"][case["launch_argv"].index("--") + 1 :] == case["argv"]
    from tools.p7x_graph_repair6_runner import run_case

    with pytest.raises(ValueError, match="PRE_PUBLICATION"):
        run_case(plan, "B0", tmp_path / "runner")
    plan["publication_status"] = "VERIFIED"
    assert run_case(plan, "B0", tmp_path / "runner")["status"] == "BLOCKED_DEVICE_MAPPING"


def test_producer_first_failure_not_hidden_by_missing_artifact(tmp_path):
    import sys

    from tools.p7x_graph_repair6_runner import run_case

    plan = dict(
        publication_status="VERIFIED",
        cases={
            "NUMERIC": dict(
                argv=[sys.executable, "-c", "raise ValueError('configuration failure')"],
                timeout_seconds=5,
                validation_commands=[
                    dict(argv=[sys.executable, "-c", "raise SystemExit(4)"], timeout=5, kind="read_only_acceptance")
                ],
            )
        },
    )
    result = run_case(plan, "NUMERIC", tmp_path)
    assert result["status"] == "EXECUTION_FAILED"
    assert result["primary_failure"]["command_index"] == 0
    assert result["acceptance_status"] == "INVALID_EVIDENCE"


def test_string_false_capture_does_not_install_sink(monkeypatch):
    from tests.ut.spec_decode.test_graph_repair6_capture_chain import capture_owner

    owner, _, _, _, _ = capture_owner(monkeypatch, "false")
    assert not hasattr(owner, "_draft_graph_output_references")


def test_real_numeric_sentinel_is_not_full_matrix(monkeypatch, tmp_path):
    from tests.ut.spec_decode.test_graph_repair6_numeric_pipeline import joined, produce
    from tools.p7x_graph_repair6 import joint

    path, contract, snapshot = produce(monkeypatch, tmp_path, sample_limit=3)
    campaign, launcher, expected = joined(path, contract, snapshot)
    expected["numeric_samples"] = 3
    expected["argv"] += ["--repair6-target-prefix", "3"]
    launcher["argv"] = expected["argv"]
    expected["launch_argv"] = ["launch", "--", *expected["argv"]]
    result = joint(path, campaign, launcher, expected, "numeric")
    assert result["status"] == "PASS_NUMERIC_SENTINEL_ONLY", result
    expected["numeric_samples"] = 12
    assert joint(path, campaign, launcher, expected, "numeric")["status"] == "INVALID_EVIDENCE"


@pytest.mark.parametrize("damage", [None, "unauthorized", "expired", "namespace", "node", "probe", "evidence"])
def test_admission_requires_physical_provenance(tmp_path, damage):
    import copy

    from tools.p7x_graph_repair6_storage import digest
    from tools.p7x_graph_repair7_device import validate_receipt

    raw = tmp_path / "owner-record.txt"
    raw.write_text("HOST_TEST_ONLY: assigned node/PID record")
    evidence = dict(path=str(raw), sha256=digest(raw))
    state = dict(namespace="mnt:test", boot_id="host", visibility={}, nodes={"/dev/davinci0": dict(major=1, minor=2)})
    receipt = dict(
        schema="repair7-device-receipt-v1",
        run_id="r",
        allocation_owner="host-test",
        issued_at=1,
        expires_at=100,
        host_phy_id=3,
        tp=1,
        allocated_nodes=state["nodes"],
        namespace=state["namespace"],
        boot_id=state["boot_id"],
        visibility={},
        mapping_evidence=[evidence],
        resource_evidence=[evidence],
        probe=dict(
            host_pid=12,
            container_pid=2,
            command=["probe"],
            start_ns=1,
            end_ns=2,
            host_phy_id=3,
            status="EXITED",
            returncode=0,
            host_process_evidence=evidence,
            pid_namespace_evidence=evidence,
        ),
    )
    receipt = copy.deepcopy(receipt)
    if damage == "unauthorized":
        receipt["host_phy_id"] = 8
    if damage == "expired":
        receipt["expires_at"] = 2
    if damage == "namespace":
        receipt["namespace"] = "other"
    if damage == "node":
        receipt["allocated_nodes"]["/dev/davinci0"]["minor"] = 3
    if damage == "probe":
        receipt["probe"]["host_phy_id"] = 8
    if damage == "evidence":
        raw.write_text("changed")
    if damage:
        with pytest.raises(ValueError):
            validate_receipt(receipt, state, "r", now=10)
    else:
        assert validate_receipt(receipt, state, "r", now=10)["physical_id"] == 3


@pytest.mark.parametrize(
    "code,status", [(1, "EXECUTION_FAILED"), (3, "NUMERIC_MISMATCH"), (74, "SUBMISSION_UNCERTAIN_QUARANTINED")]
)
def test_runner_failure_classes(tmp_path, code, status):
    import sys

    from tools.p7x_graph_repair6_runner import run_case

    plan = dict(
        publication_status="VERIFIED",
        cases={"X": dict(argv=[sys.executable, "-c", f"raise SystemExit({code})"], timeout_seconds=2)},
    )
    assert run_case(plan, "X", tmp_path)["status"] == status


def test_producer_timeout_is_not_numeric_failure(monkeypatch, tmp_path):
    from tools import p7x_graph_repair6_runner as runner

    # Boundary under test is aggregation; subprocess watchdog has separate tests.
    def expired(argv, directory, timeout):
        directory.mkdir(parents=True)
        return dict(argv=argv, returncode=-15, timeout=True, status="COMPARISON_TIMEOUT")

    monkeypatch.setattr(runner, "execute", expired)
    plan = dict(publication_status="VERIFIED", cases={"X": dict(argv=["child"], timeout_seconds=1)})
    result = runner.run_case(plan, "X", tmp_path)
    assert result["status"] == "EXECUTION_TIMEOUT"
    assert not result["dependency_satisfied"]


@pytest.mark.parametrize("damage", [None, "source", "arm", "hash", "timeout"])
def test_target_final_join_real_controlled_producer(monkeypatch, tmp_path, damage):
    import json

    from tests.ut.spec_decode.test_graph_repair4_target import tool as base
    from tests.ut.spec_decode.test_graph_repair5_contracts import load
    from tests.ut.spec_decode.test_graph_repair5_producers import controlled_runner
    from tests.ut.spec_decode.test_graph_repair6_review import execution
    from tools.p7x_graph_repair6_storage import CollectionReader, atomic_json
    from tools.p7x_graph_repair7 import target_join

    tool = load("p7x_graph_repair5_target")
    monkeypatch.setattr(tool, "_base", lambda: base)
    plan = dict(run_id="r", commit="a" * 40, cases={})
    for arm, mode in [("OFF", "A0"), ("ON", "B")]:
        name = "G4F_" + arm
        identity = dict(
            run_id="r",
            case_id=name,
            attempt_id="1",
            source_sha="a" * 40,
            source_tree="b" * 40,
            process_id=7,
            engine_id=8,
            driver_mode=mode,
        )
        runner = controlled_runner(monkeypatch)
        runner.model.config = SimpleNamespace(hidden_size=2, vocab_size=4)
        runner.model_state.attn_metadata = {}
        fixture = tool.ControlledTarget(
            runner,
            fixture_id="host",
            arm=arm,
            scheduler=SimpleNamespace(requests={}, running=[], waiting=[]),
            identity=identity,
            bounded_store=True,
            sample_limit=1,
        )
        artifact = tmp_path / name / "controlled-target.json"
        artifact.parent.mkdir()
        fixture.execute(artifact)
        campaign, launcher, expected = execution(artifact, CollectionReader(artifact).header["identity"])
        plan["cases"][name] = expected
        atomic_json(tmp_path / name / "campaign.json", campaign)
        atomic_json(tmp_path / "launchers" / name / "launcher.json", launcher)
    if damage == "source":
        plan["commit"] = "c" * 40
    if damage == "arm":
        plan["cases"]["G4F_ON"]["acceptance_identity"]["driver_mode"] = "A0"
    if damage == "hash":
        path = tmp_path / "G4F_ON" / "controlled-target.json"
        path.write_text(path.read_text() + "\n")
    if damage == "timeout":
        path = tmp_path / "launchers/G4F_ON/launcher.json"
        data = json.loads(path.read_text())
        data["timeout"] = True
        atomic_json(path, data)
    result = target_join(tmp_path, plan, "G4F", dict(segment="HOST_NATIVE_SUBSTITUTE"))
    assert result["status"] != "SCOPED_TARGET_EXECUTION_AND_NUMERIC_VALID"
    if damage in ("source", "arm", "hash"):
        assert result["status"] == "INVALID_EVIDENCE", result
    else:
        assert result["comparison"]["complete_pairs"] == 1, result
