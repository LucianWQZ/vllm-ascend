# SPDX-License-Identifier: Apache-2.0
"""Review counterexamples use real CPU tensors, durable storage and execution join."""

import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from tests.ut.spec_decode.test_graph_repair5_contracts import ROOT, load
from tools.p7x_graph_repair6 import joint
from tools.p7x_graph_repair6_storage import CollectionWriter, digest


def execution(path, identity):
    argv = ["python", "driver"]
    expected = dict(
        acceptance_identity=identity,
        argv=argv,
        launch_argv=["launch", "--", *argv],
        outputs=str(path.parent),
        requests_sha256="requests",
        capture_profile="target4",
    )
    launcher = dict(
        identity,
        argv=argv,
        pid=identity["process_id"],
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
        diagnostic_artifacts={path.name: dict(sha256=digest(path), bytes=path.stat().st_size)},
    )
    return campaign, launcher, expected


IDENTITY = dict(
    run_id="r",
    case_id="NUMERIC_TARGET4",
    attempt_id="1",
    source_sha="a" * 40,
    source_tree="b" * 40,
    process_id=7,
    engine_id=8,
    driver_mode="B",
)


@pytest.mark.parametrize(
    "damage", ["missing_canonical", "negative_exact", "payload_fail", "execution_fail", "string_kv"]
)
def test_f1_real_tensor_store_external_join_rejects_invalid_numeric(tmp_path, damage):
    target = load("p7x_graph_repair5_target")
    p = tmp_path / "numeric-reference.json"
    w = CollectionWriter(p, dict(identity=IDENTITY))
    for c in target.case_plan("target4"):
        n = len(c["ends"])
        padded = 1 if n == 1 else 2 if n == 2 else 4
        r = dict(
            identity=dict(IDENTITY, generation=c["step"], epoch=c["step"]),
            generation=c["step"],
            epoch=c["step"],
            case_identity=target.sample_identity(c, "host"),
            descriptor=[padded, padded * 7],
            active=n,
            canonical_table=[[1] * 11] * n,
            exact=[e + 7 for e in c["ends"]],
            valid=[11] * n,
            oracle=[],
            selected_blocks=[0, 1],
            pre_kv={"layer/0": torch.ones(2, 128, 1, 1)},
            graph_post_kv={"layer/0": torch.ones(2, 128, 1, 1)},
            inputs={"input_ids": torch.ones(padded * 7, dtype=torch.int64)},
            graph=torch.ones(padded * 7, 2, dtype=torch.bfloat16),
            reference=torch.ones(padded * 7, 2, dtype=torch.bfloat16),
            status="PASS_SCOPED_CONTINUOUS",
        )
        if damage == "missing_canonical":
            for key in ("canonical_table", "exact", "valid", "oracle", "selected_blocks"):
                del r[key]
        elif damage == "negative_exact":
            r.update(exact=[-1] * n, canonical_table=[], oracle=[])
        elif damage == "payload_fail":
            r["status"] = "NUMERIC_MISMATCH"
        elif damage == "string_kv":
            r.update(pre_kv={"layer/0": "not tensor"}, graph_post_kv={"layer/0": "not tensor"})
        w.start(r["identity"])
        w.finish(r)
    w.close("EXECUTION_FAILED" if damage == "execution_fail" else "COMPLETE")
    result = joint(p, *execution(p, IDENTITY), "numeric")
    assert not result["status"].startswith("PASS"), result


def test_f4_natural_actual_cli_exit_does_not_hide_mismatch(tmp_path):
    for arm, number in (("off", 1.0), ("on", 2.0)):
        d = tmp_path / arm
        d.mkdir()
        hidden = torch.ones(1)
        torch.save(dict(hidden_states=hidden, logits=torch.tensor([number])), d / "sample.pt")
        (d / "target-boundary.json").write_text(
            json.dumps(
                dict(
                    errors=[],
                    records=[
                        dict(
                            kind="target_logits_head",
                            path="sample.pt",
                            logical_identity={"request_group": "same"},
                            input_sha256=hashlib.sha256(hidden.numpy().tobytes()).hexdigest(),
                        )
                    ],
                )
            )
        )
    output = tmp_path / "result.json"
    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/p7x_graph_repair6_natural.py"),
            "--off",
            str(tmp_path / "off"),
            "--on",
            str(tmp_path / "on"),
            "--output",
            str(output),
        ],
        capture_output=True,
        timeout=30,
    )
    assert json.loads(output.read_text())["status"] == "NUMERIC_MISMATCH"
    assert r.returncode != 0


def target_prefix(monkeypatch, tmp_path, count=2):
    from tests.ut.spec_decode.test_graph_repair4_target import tool as base
    from tests.ut.spec_decode.test_graph_repair5_producers import controlled_runner

    t = load("p7x_graph_repair5_target")
    monkeypatch.setattr(t, "_base", lambda: base)
    for arm in ("OFF", "ON"):
        r = controlled_runner(monkeypatch)
        r.model.config = SimpleNamespace(hidden_size=2, vocab_size=4)
        r.model_state.attn_metadata = {}
        fixture = t.ControlledTarget(
            r,
            fixture_id="host",
            arm=arm,
            scheduler=SimpleNamespace(requests={}, running=[], waiting=[]),
            identity=dict(IDENTITY, case_id=arm),
            bounded_store=True,
            sample_limit=count,
        )
        fixture.execute(tmp_path / (arm + ".json"))
    return t


def test_f5_complete_prefix_preparing_tail(monkeypatch, tmp_path):
    t = target_prefix(monkeypatch, tmp_path)
    for arm in ("OFF", "ON"):
        p = tmp_path / (arm + ".json")
        m = json.loads(p.read_text())
        m["samples"].append(dict(identity=t.sample_identity(t.case_plan()[2], "host"), status="PREPARING", path=None))
        m["execution"] = "EXECUTION_FAILED"
        p.write_text(json.dumps(m))
    result = load("p7x_graph_repair6").compare(tmp_path / "OFF.json", tmp_path / "ON.json")
    assert result["status"] == "PARTIAL_PREFIX_VALID", result
    assert result["complete_pairs"] == 2
    assert len(result["pairs"]) == len(t.case_plan())


def test_f6_prefix_runtime_budget_is_one_sample(monkeypatch, tmp_path):
    target_prefix(monkeypatch, tmp_path, count=1)
    budget = json.loads((tmp_path / "target-resource-plan.json").read_text())
    assert len(budget["samples"]) == 1, budget
