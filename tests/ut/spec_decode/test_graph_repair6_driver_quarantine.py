# SPDX-License-Identifier: Apache-2.0
"""Driver finally, not just a probe.__exit__ test. No NPU execution."""

import hashlib
import json
import os
from types import SimpleNamespace

import pytest
import torch

from tests.ut.spec_decode.test_graph_repair5_contracts import load
from tests.ut.spec_decode.test_graph_repair6_capture_chain import capture_owner


def driver_native_failure(monkeypatch, tmp_path, driver):
    import vllm

    for key in (
        "VLLM_USE_V2_MODEL_RUNNER",
        "VLLM_ENABLE_V1_MULTIPROCESSING",
        "VLLM_ASCEND_EXPERIMENTAL_DRAFT_WINDOW_GRAPH",
        "VLLM_ASCEND_EXPERIMENTAL_DRAFT_WINDOW_GRAPH_OBSERVE",
        "VLLM_ASCEND_EXPERIMENTAL_DRAFT_WINDOW_GRAPH_ARTIFACT",
    ):
        monkeypatch.setenv(key, os.environ.get(key, ""))
    monkeypatch.setenv("VLLM_ASCEND_EXPERIMENTAL_DRAFT_WINDOW_GRAPH_OBSERVE", "1")
    owner, session, manager, descs, _ = capture_owner(monkeypatch)
    manager.run_fullgraph = lambda _: (_ for _ in ()).throw(RuntimeError("native submission uncertain"))
    runner_type = type("NPUModelRunner", (), {"__module__": "vllm_ascend.worker.v2.model_runner"})
    runner = runner_type()
    runner.speculator, runner.cudagraph_manager = owner, manager
    shutdown = []
    core = SimpleNamespace(
        model_executor=SimpleNamespace(driver_worker=SimpleNamespace(worker=SimpleNamespace(model_runner=runner)))
    )
    llm = SimpleNamespace(
        llm_engine=SimpleNamespace(
            engine_core=SimpleNamespace(engine_core=core, shutdown=lambda: shutdown.append(True))
        )
    )
    llm.generate = lambda *a, **kw: manager.run_fullgraph(descs[1])
    monkeypatch.setattr(vllm, "LLM", lambda **kw: llm)
    monkeypatch.setattr(vllm, "SamplingParams", lambda **kw: kw)
    monkeypatch.setattr(torch.npu, "Event", lambda: pytest.fail("uncertain submit must not drain"))

    def git(path, *args):
        if args[0] == "status":
            return ""
        return (
            driver.PINNED_VLLM_TREE
            if str(path) == "pin" and args[-1] == "HEAD^{tree}"
            else driver.PINNED_VLLM
            if str(path) == "pin"
            else "a" * 40
        )

    monkeypatch.setattr(driver, "_git", git)
    p = tmp_path / "input"
    p.write_bytes(b"fixture")
    manifest = tmp_path / "fixture.json"
    manifest.write_text(json.dumps(dict(files=[dict(path=str(p), sha256=hashlib.sha256(b"fixture").hexdigest())] * 9)))
    args = SimpleNamespace(
        output=tmp_path / "output",
        run_id="r",
        case_id="NORMAL_ORDER",
        case="B-observer-off",
        attempt="1",
        vllm_repo="pin",
        expected_sha="a" * 40,
        requests=None,
        fixture_manifest=manifest,
        target_model="fixture",
        draft_model="fixture",
        target_probe=False,
        full_target_probe=False,
        repair5_controlled_target=False,
        repair5_controlled_mrv2=False,
        repair5_lifecycle=False,
        repair6_normal_lifecycle=True,
    )
    exited = []

    def stop(code):
        exited.append(code)
        raise SystemExit(code)

    monkeypatch.setattr(driver.os, "_exit", stop)
    try:
        result = driver.run_case(args)
    except SystemExit as error:
        result = error.code
    return result, shutdown, exited, json.loads((args.output / "campaign.json").read_text())


def test_actual_driver_finally_quarantines_normal_probe(monkeypatch, tmp_path):
    rc, shutdown, exited, campaign = driver_native_failure(monkeypatch, tmp_path, load("p7x_graph_repair2"))
    assert rc == 74 and exited == [74] and not shutdown, campaign
    assert campaign["exception_message"] == "native submission uncertain"
    assert campaign["diagnostic_quarantine"] == "PROCESS_EXIT_74_NO_REUSE"
