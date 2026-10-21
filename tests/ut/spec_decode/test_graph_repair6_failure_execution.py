# SPDX-License-Identifier: Apache-2.0
"""Real durable prefixes, actual subprocess exits and owned descendant cleanup."""

import os
import sys
from types import SimpleNamespace

import pytest
import torch

from tests.ut.spec_decode.test_graph_repair6_review import target_prefix
from tools.p7x_graph_repair6 import compare
from tools.p7x_graph_repair6_runner import execute, run_case
from tools.p7x_graph_repair6_storage import (
    CollectionReader,
    CollectionWriter,
    TensorStore,
    require_space,
    resource_plan,
)


@pytest.mark.parametrize("mismatch", [False, True])
@pytest.mark.parametrize("tail", ["preparing", "corrupt"])
def test_prefix_failure_priority_real_writer(monkeypatch, tmp_path, mismatch, tail):
    target_prefix(monkeypatch, tmp_path, count=2)
    for arm in ("OFF", "ON"):
        source = CollectionReader(tmp_path / (arm + ".json"))
        writer = CollectionWriter(tmp_path / "changed" / (arm + ".json"), source.header)
        for i in range(2):
            row = source.record(i)
            if mismatch and arm == "ON" and i == 0:
                row["logits"] = row["logits"].clone() + 1
            writer.start(row["identity"])
            if i == 1 and tail == "preparing":
                break
            writer.finish(row)
        writer.close("EXECUTION_FAILED")
        if tail == "corrupt":
            payload = writer.path.parent / writer.manifest["samples"][-1]["path"]
            payload.write_text("{}")
    result = compare(tmp_path / "changed/OFF.json", tmp_path / "changed/ON.json")
    assert result["status"] == (
        "NUMERIC_MISMATCH" if mismatch else "INVALID_EVIDENCE" if tail == "corrupt" else "PARTIAL_PREFIX_VALID"
    ), result
    assert result["pairs"][1]["status"] == ("INVALID_EVIDENCE" if tail == "corrupt" else "INCOMPLETE_COVERAGE")


def test_prefix_disk_gate_full_blocked_and_mid_chunk_preserved(monkeypatch, tmp_path):
    import tools.p7x_graph_repair6_storage as storage
    from tools.p7x_graph_repair5_target import case_plan

    geometry = dict(
        layers=36, kv_heads=8, head_dim=128, element_bytes=2, hidden_size=4096, vocab_size=151936, mask_bytes=2048**2
    )
    one = resource_plan(case_plan(), sample_limit=1, **geometry)
    full = resource_plan(case_plan(), **geometry)
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda _: SimpleNamespace(free=one["required_free_bytes"]))
    require_space(tmp_path, one["required_free_bytes"])
    with pytest.raises(ValueError, match="BLOCKED_RESOURCE"):
        require_space(tmp_path, full["required_free_bytes"])
    writer = CollectionWriter(tmp_path / "partial.json", {}, chunk_bytes=16)
    writer.start({"sample": 0})
    calls = []

    def disk(_):
        calls.append(1)
        return SimpleNamespace(free=1 << 30 if len(calls) <= 2 else 0)

    monkeypatch.setattr(storage.shutil, "disk_usage", disk)
    with pytest.raises(ValueError, match="BLOCKED_RESOURCE"):
        writer.finish(dict(identity={"sample": 0}, status="COMPLETE", tensor=torch.ones(32)))
    writer.close("EXECUTION_FAILED")
    assert CollectionReader(writer.path).incomplete(0)
    assert next(writer.store.root.glob("*.partial")).stat().st_size == 16
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda _: SimpleNamespace(free=1 << 30))
    independent = TensorStore(tmp_path / "independent")
    assert torch.equal(independent.put(torch.ones(1)), torch.ones(1))


@pytest.mark.parametrize(
    "code,status",
    [
        (0, "SCOPED_COMMANDS_ACCEPTED"),
        (2, "INCOMPLETE_COVERAGE"),
        (3, "NUMERIC_MISMATCH"),
        (4, "INVALID_EVIDENCE"),
        (5, "BLOCKED_RESOURCE"),
    ],
)
def test_actual_runner_read_only_exit_contract(tmp_path, code, status):
    plan = dict(
        publication_status="VERIFIED",
        cases={
            "COMPARE": dict(
                argv=[sys.executable, "-c", f"raise SystemExit({code})"],
                timeout_seconds=10,
                selection="READ_ONLY_ACCEPTANCE",
                dependencies=[],
            )
        },
    )
    result = run_case(plan, "COMPARE", tmp_path)
    assert result["status"] == status, result
    assert result["dependency_satisfied"] == (code == 0)


def test_timeout_terminates_only_owned_nested_session(tmp_path):
    # Child opens a new session; killpg(root) alone would leak it. No sleep is
    # used as a device race oracle: this is an OS watchdog timeout test.
    child_pid = tmp_path / "child-pid"
    code = (
        "import subprocess,sys,time; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True); "
        "open(sys.argv[1],'w').write(str(p.pid)); time.sleep(60)"
    )
    result = execute([sys.executable, "-c", code, str(child_pid)], tmp_path / "process", 1)
    pid = int(child_pid.read_text())
    assert result["timeout"] and pid in result["owned_processes_terminated"]
    assert os.getpid() not in result["owned_processes_terminated"]
    # A reparented zombie is terminated, even before init has reaped its PID.
    import subprocess

    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    assert not state or state.startswith("Z"), state


def test_profile_missing_installation_not_silent():
    from tools.p7x_graph_repair5_profile import RequestProfiler

    probe = RequestProfiler.__new__(RequestProfiler)
    probe.window_active = True
    labels = ("Draft_materialize", "Draft_metadata_bind", "Draft_UPDATE_CALL", "Draft_retire_host")
    probe.installations = {k: dict(installed=True) for k in labels}
    probe.calls = [dict(label=k, active=True, generation=3) for k in (*labels, "Draft_REPLAY_PLUS_UPDATE")]
    assert all(r["status"] == "OBSERVED" for r in probe.subrange_coverage().values())
    probe.installations["Draft_retire_host"]["installed"] = False
    assert probe.subrange_coverage()["Draft_retire_host"]["status"] == "MISSING"
    probe.calls[0]["generation"] = 2
    assert probe.subrange_coverage()["Draft_materialize"]["missing_generations"] == [3]


def test_runner_resource_failure_does_not_block_independent_case(monkeypatch, tmp_path):
    import tools.p7x_graph_repair6_storage as storage

    monkeypatch.setattr(storage.shutil, "disk_usage", lambda _: SimpleNamespace(free=100))
    simple = dict(argv=[sys.executable, "-c", "pass"], timeout_seconds=5, dependencies=[])
    plan = dict(
        publication_status="VERIFIED",
        cases={
            "FULL": dict(simple, resource_policy=dict(minimum_launch_free_bytes=200)),
            "API": dict(simple, resource_policy=dict(minimum_launch_free_bytes=10)),
        },
    )
    assert run_case(plan, "FULL", tmp_path)["status"] == "BLOCKED_RESOURCE"
    assert run_case(plan, "API", tmp_path)["dependency_satisfied"]


def test_packer_preflights_uncompressed_bound(monkeypatch, tmp_path):
    from tools import p7x_graph_repair5_pack as packer

    source = tmp_path / "source"
    source.mkdir()
    (source / "payload").write_bytes(b"important failed evidence")
    monkeypatch.setattr(packer.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    with pytest.raises(ValueError, match="BLOCKED_RESOURCE"):
        packer.pack(source, tmp_path / "raw.zip")
    assert not (tmp_path / "raw.zip").exists()
    assert (source / "payload").read_bytes() == b"important failed evidence"
