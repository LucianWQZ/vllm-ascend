# SPDX-License-Identifier: Apache-2.0
"""Executable CLI, plan paths, actual tensor artifact and negative acceptance."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.ut.spec_decode.test_graph_repair5_contracts import ROOT, load, token_joint


@pytest.mark.parametrize(
    "name,arguments",
    [
        ("p7x_graph_repair6", []),
        ("p7x_graph_repair6", ["check-joint"]),
        ("p7x_graph_repair6_campaign", []),
        ("p7x_graph_repair6_runner", []),
        ("p7x_graph_repair6_trace", []),
        ("p7x_graph_repair6_natural", []),
        ("p7x_graph_repair2", ["run"]),
    ],
)
def test_cli_help(name, arguments):
    r = subprocess.run(
        [sys.executable, str(ROOT / f"tools/{name}.py"), *arguments, "--help"],
        capture_output=True,
        text=True,
        timeout=40,
    )
    assert r.returncode == 0, r.stderr


def test_plan_all_argv_output_associations_and_dependencies(monkeypatch, tmp_path):
    import tools.p7x_graph_repair5_campaign as old
    from tools.p7x_graph_repair6_campaign import build_plan

    fixture = b'{"files":[]}'
    (tmp_path / "fixture_manifest.json").write_bytes(fixture)
    monkeypatch.setattr(old, "FIXTURE_SHA", hashlib.sha256(fixture).hexdigest())
    (tmp_path / "requests").mkdir()
    for name in ("smoke", "combined", "mixed", "g4-long-group1", "g4-long-group2"):
        (tmp_path / "requests" / f"{name}.json").write_bytes(token_joint()[2])
    plan = build_plan(ROOT, tmp_path, tmp_path / "out", tmp_path / "vllm", sys.executable, "a" * 40, "b" * 40, "r")
    seen = set()
    for name, case in plan["cases"].items():
        assert set(case["dependencies"]) <= seen
        seen.add(name)
        if "launch_argv" not in case:
            continue
        launch, child = case["launch_argv"], case["argv"]
        assert launch[launch.index("--") + 1 :] == child
        assert child[child.index("--output") + 1] == case["outputs"]
        assert case["acceptance_identity"]["case_id"] == name
        for command in case["validation_commands"]:
            assert command["timeout"] == 3300
            if "--artifact" in command["argv"]:
                path = command["argv"][command["argv"].index("--artifact") + 1]
                assert Path(path).parent == Path(case["outputs"])
    assert "--repair6-numeric-reference" in plan["cases"]["NUMERIC_TARGET4"]["argv"]
    assert "--repair6-target-store" in plan["cases"]["PREFIX_ON"]["argv"]
    assert "--repair5-lifecycle" not in plan["cases"]["NORMAL_ORDER"]["argv"]
    assert plan["publication_status"] == "NOT_PUBLISHED"


@pytest.mark.parametrize("damage", [None, "source", "pid", "timeout", "hash", "missing", "nan"])
def test_real_target_prefix_producer_external_join(monkeypatch, tmp_path, damage):
    from tests.ut.spec_decode.test_graph_repair4_target import tool as base
    from tests.ut.spec_decode.test_graph_repair5_producers import controlled_runner
    from tools.p7x_graph_repair6 import joint

    t = load("p7x_graph_repair5_target")
    monkeypatch.setattr(t, "_base", lambda: base)
    identity = dict(
        run_id="r",
        case_id="PREFIX_OFF",
        attempt_id="1",
        source_sha="a" * 40,
        source_tree="b" * 40,
        process_id=7,
        engine_id=8,
        driver_mode="A0",
    )
    runner = controlled_runner(monkeypatch)
    runner.model.config = SimpleNamespace(hidden_size=2, vocab_size=4)
    runner.model_state.attn_metadata = {}
    if damage == "nan":
        original = runner.model.compute_logits
        runner.model.compute_logits = lambda h: original(h).fill_(float("nan"))
    fixture = t.ControlledTarget(
        runner,
        fixture_id="host",
        arm="OFF",
        scheduler=SimpleNamespace(requests={}, running=[], waiting=[]),
        identity=identity,
        bounded_store=True,
        sample_limit=1,
    )
    artifact = tmp_path / "controlled-target.json"
    fixture.execute(artifact)
    sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    argv = ["python", "driver"]
    expected = dict(
        acceptance_identity=identity,
        argv=argv,
        launch_argv=["launch", "--", *argv],
        outputs=str(tmp_path),
        requests_sha256="requests",
        prefix_samples=1,
    )
    launcher = dict(
        identity,
        argv=argv,
        pid=7,
        status="EXITED",
        returncode=0,
        signal=None,
        timeout=False,
        requests_sha256="requests",
        host_start_ns=1,
        host_end_ns=100,
    )
    campaign = dict(
        identity=dict(identity),
        startup_complete=True,
        status="OK",
        request_manifest={"sha256": "requests"},
        diagnostic_artifacts={artifact.name: dict(sha256=sha, bytes=artifact.stat().st_size)},
    )
    if damage == "source":
        campaign["identity"]["source_sha"] = "c" * 40
    if damage == "pid":
        launcher["pid"] = 8
    if damage == "timeout":
        launcher["timeout"] = True
    if damage == "hash":
        campaign["diagnostic_artifacts"][artifact.name]["sha256"] = "bad"
    if damage == "missing":
        next(tmp_path.glob("controlled-target-sample-*.json")).unlink()
    result = joint(artifact, campaign, launcher, expected, "target-prefix")
    if damage is None:
        assert result["status"] == "PREFIX_STRUCTURE_VALID_NOT_CAMPAIGN_PASS", result
    else:
        assert result["status"] not in ("PREFIX_STRUCTURE_VALID_NOT_CAMPAIGN_PASS", "SINGLE_ARM_STRUCTURE_VALID"), (
            result
        )


def test_optimized_python_rejects_empty_collection(tmp_path):
    output = tmp_path / "result.json"
    result = subprocess.run(
        [
            sys.executable,
            "-O",
            str(ROOT / "tools/p7x_graph_repair6.py"),
            "compare-target",
            "--off",
            str(tmp_path / "missing"),
            "--on",
            str(tmp_path / "missing"),
            "--output",
            str(output),
        ],
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 4
    assert json.loads(output.read_text())["status"] == "INVALID_EVIDENCE"


@pytest.mark.parametrize("exit_code,checker,accepted", [(73, 0, True), (74, 0, False), (73, 2, False)])
def test_abort_runner_keeps_nonzero_exit_and_requires_specific_checker(
    monkeypatch, tmp_path, exit_code, checker, accepted
):
    import tools.p7x_graph_repair6_runner as runner

    def execute(argv, directory, timeout):
        directory.mkdir(parents=True)
        return dict(argv=argv, returncode=exit_code if argv == ["child"] else checker, timeout=False)

    monkeypatch.setattr(runner, "execute", execute)
    plan = dict(
        publication_status="VERIFIED",
        cases={
            "ABORT_after_update": dict(
                argv=["child"],
                timeout_seconds=1,
                validation_commands=[dict(argv=["check-abort"], timeout=1, kind="acceptance")],
            )
        },
    )
    result = runner.run_case(plan, "ABORT_after_update", tmp_path)
    assert result["dependency_satisfied"] is accepted
    assert result["records"][0]["returncode"] == exit_code


def test_natural_identity_pairing_preserves_duplicate_missing_and_numeric_failure(monkeypatch):
    import torch

    from tools import p7x_graph_repair6_natural as natural

    def row(ordinal, group, value):
        return dict(
            ordinal=ordinal,
            record={"logical_identity": {"request_group": group}},
            path=str(ordinal),
            sha256=str(ordinal),
            values={"hidden_states": torch.ones(1), "logits": torch.tensor([value])},
        )

    off = [row(0, "unique", 1.0), row(1, "duplicate", 1.0), row(2, "duplicate", 1.0), row(3, "missing", 1.0)]
    on = [row(0, "unique", 2.0), row(1, "duplicate", 1.0)]
    monkeypatch.setattr(natural, "records", lambda root: (off if root == "OFF" else on, []))
    result = natural.compare("OFF", "ON")
    assert result["status"] == "NUMERIC_MISMATCH"
    assert [r["status"] for r in result["pairs"]] == [
        "UNPAIRED_OR_AMBIGUOUS",
        "UNPAIRED_OR_AMBIGUOUS",
        "ALIGNED_HEAD_NUMERIC_FAIL",
    ]
    assert len(result["samples"]) == 6
