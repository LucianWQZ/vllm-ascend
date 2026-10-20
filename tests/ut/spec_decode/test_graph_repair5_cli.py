# SPDX-License-Identifier: Apache-2.0
"""Actual CLI/schema contracts; no models or NPU invocation."""

import hashlib
import json
import subprocess
import sys

import pytest

from tests.ut.spec_decode.test_graph_repair5_contracts import ROOT, load, token_joint


@pytest.mark.parametrize(
    "script,sub",
    [
        ("p7x_graph_repair5", []),
        ("p7x_graph_repair5", ["capabilities"]),
        ("p7x_graph_repair5", ["compare-target"]),
        ("p7x_graph_repair5", ["check-abort"]),
        ("p7x_graph_repair5", ["check-lifecycle"]),
        ("p7x_graph_repair5", ["check-mrv2-aba"]),
        ("p7x_graph_repair5", ["check-diagnostic"]),
        ("p7x_graph_repair5_profile", ["preflight"]),
        ("p7x_graph_repair5_profile", ["analyze"]),
        ("p7x_graph_repair5_pack", ["pack"]),
        ("p7x_graph_repair5_pack", ["verify"]),
        ("p7x_graph_repair5_campaign", []),
        ("p7x_graph_repair2", ["run"]),
    ],
)
def test_cli_help_is_real(script, sub):
    result = subprocess.run(
        [sys.executable, str(ROOT / f"tools/{script}.py"), *sub, "--help"], capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stderr


def test_plan_to_official_joint_expected_schema(monkeypatch, tmp_path):
    tool = load("p7x_graph_repair5_campaign")
    fixture = b'{"files":[]}'
    (tmp_path / "fixture_manifest.json").write_bytes(fixture)
    monkeypatch.setattr(tool, "FIXTURE_SHA", hashlib.sha256(fixture).hexdigest())
    (tmp_path / "requests").mkdir()
    document, launcher, raw, expected = token_joint()
    for name in ("smoke", "combined", "mixed", "g4-long-group1", "g4-long-group2"):
        (tmp_path / "requests" / f"{name}.json").write_bytes(raw)
    plan = tool.build_plan(
        ROOT, tmp_path, tmp_path / "out", tmp_path / "vllm", sys.executable, "a" * 40, "b" * 40, "run"
    )
    for name, case in plan["cases"].items():
        if "acceptance_identity" not in case:
            continue
        assert case["acceptance_identity"]["case_id"] == name
        assert case["acceptance_identity"]["source_tree"] == plan["tree"]
        argv = case["argv"]
        launch = case["launch_argv"]
        assert launch[launch.index("--") + 1 :] == argv
        # Launcher must not create the engine's exclusive output directory.
        outer = launch[launch.index("--output") + 1]
        child = argv[argv.index("--output") + 1]
        assert child == case["outputs"]
        assert not outer.startswith(child + "/")
        for check in [case.get("check_argv", [])] + case.get("additional_checks", []):
            if "--input" in check:
                assert check[check.index("--input") + 1].startswith(case["outputs"] + "/")
    assert "--repair5-controlled-mrv2" in plan["cases"]["G4F_ON"]["argv"]
    seen = set()
    for name, case in plan["cases"].items():
        assert set(case["dependencies"]) <= seen
        seen.add(name)
    assert "--target-probe" in plan["cases"]["G4_ON_long2"]["argv"]
    assert "--repair5-controlled-target" not in plan["cases"]["G4_ON_long2"]["argv"]
    assert "NORMAL_SCHEDULING" in plan["trace"]
    assert "G4F_OFF" not in plan["cases"]["TARGET4_OFF"]["dependencies"]
    assert "G4F_ON" not in plan["cases"]["TARGET4_ON"]["dependencies"]
    assert plan["limits"]["max_records_controlled"] == {"base": 11, "target4": 12}


def test_optimized_python_rejects_corrupt_lifecycle(tmp_path):
    path, output = tmp_path / "bad.json", tmp_path / "result.json"
    path.write_text(json.dumps({"schema": "repair5-real-mrv2-lifecycle-v1", "records": []}))
    result = subprocess.run(
        [
            sys.executable,
            "-O",
            str(ROOT / "tools/p7x_graph_repair5.py"),
            "check-lifecycle",
            str(path),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2, result.stderr
    assert json.loads(output.read_text())["status"] == "INVALID_EVIDENCE"
