# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[3] / "tools/p7x_graph_repair4_campaign.py"
spec = importlib.util.spec_from_file_location("repair4_campaign", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_plan_frozen_requests_and_real_parser(tmp_path, monkeypatch):
    fixture = tmp_path / "fixture_manifest.json"
    fixture.write_text("{}")
    requests = tmp_path / "requests"
    requests.mkdir()
    for name in ("smoke", "combined", "mixed", "g4-long-group1", "g4-long-group2"):
        (requests / f"{name}.json").write_text("{}")
    handoff = {
        "commit": "f" * 40,
        "tree": "a" * 40,
        "publication_status": "NOT_PUBLISHED",
        "fixture_manifest_sha256": module.digest(fixture),
    }
    plan = module.make_plan(handoff, tmp_path, tmp_path / "out", tmp_path / "vllm", "run")
    assert plan["execution_authorized"] is False
    assert plan["cases"]["B0"]["depends_on"] == ["A0", "A1"]
    driver_spec = importlib.util.spec_from_file_location("repair2_parser", PATH.parent / "p7x_graph_repair2.py")
    driver = importlib.util.module_from_spec(driver_spec)
    driver_spec.loader.exec_module(driver)
    parsed = []
    monkeypatch.setattr(driver, "launch_case", lambda args: parsed.append(args) or 0)
    monkeypatch.setattr(driver, "run_case", lambda args: parsed.append(args) or 0)
    for entry in plan["cases"].values():
        if "launcher_argv" not in entry:
            continue
        # --help exercises the actual parser without running a child/model.
        argv = entry["launcher_argv"]
        result = subprocess.run(
            [*argv[:3], "--help"], cwd=PATH.parent.parent, capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 0
        assert "--case-id" in result.stdout and "--mode" in result.stdout
        monkeypatch.setattr(sys, "argv", [argv[1], *argv[2:]])
        assert driver.main() == 0
        assert parsed[-1].case_id in plan["cases"]
        actual = entry["argv"]
        monkeypatch.setattr(sys, "argv", [actual[1], *actual[2:]])
        assert driver.main() == 0
    fixture.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="mismatch"):
        module.make_plan(handoff, tmp_path, tmp_path, tmp_path, "run")


def test_package_verifies_members_not_device_success(tmp_path):
    root = tmp_path / "input"
    root.mkdir()
    (root / "result.json").write_text(json.dumps({"device": "NOT_RUN"}))
    output = tmp_path / "package.zip"
    module.pack(root, output)
    assert module.verify(output)["device_correctness"] == "NOT_ASSESSED"
    with pytest.raises(ValueError, match="new and outside"):
        module.pack(root, output)
    with zipfile.ZipFile(output, "a") as z:
        z.writestr("unindexed.json", "{}")
    with pytest.raises(ValueError, match="unindexed"):
        module.verify(output)


def test_package_rejects_traversal_and_symlink(tmp_path):
    output = tmp_path / "bad.zip"
    with zipfile.ZipFile(output, "w") as z:
        z.writestr("../escape", "x")
    with pytest.raises(ValueError, match="unsafe"):
        module.verify(output)
    root = tmp_path / "input"
    root.mkdir()
    (root / "secret-link").symlink_to(output)
    with pytest.raises(ValueError, match="symlink"):
        module.pack(root, tmp_path / "new.zip")
