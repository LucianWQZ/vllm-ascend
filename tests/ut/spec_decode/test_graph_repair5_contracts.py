# SPDX-License-Identifier: Apache-2.0
"""Repair4 counterexamples: real FIA tuple construction, actual token input schema."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.ut.spec_decode import test_graph_repair4_acceptance as acceptance
from tests.ut.spec_decode import test_graph_repair4_target as target

ROOT = Path(__file__).resolve().parents[3]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"tools/{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_post4_version_api_has_no_npu_get_cann_version(monkeypatch):
    fixture = load("p7x_graph_repair4_fixture")
    native = lambda: None
    npu = SimpleNamespace(
        is_available=lambda: True,
        NPUGraph=SimpleNamespace(replay=native),
        Event=SimpleNamespace(record=native, synchronize=native, query=native),
        ExternalEvent=SimpleNamespace(wait=native, reset=native, record=native),
        **{
            name: native
            for name in (
                "graph_task_group_begin",
                "graph_task_group_end",
                "graph_task_update_begin",
                "graph_task_update_end",
            )
        },
    )
    torch = SimpleNamespace(
        __version__="2.10.0+cpu",
        npu=npu,
        ops=SimpleNamespace(
            npu=SimpleNamespace(
                npu_fused_infer_attention_score=SimpleNamespace(
                    out=SimpleNamespace(_schema="block_table actual_seq_lengths_kv workspace")
                )
            )
        ),
    )
    post4 = SimpleNamespace(
        __version__="2.10.0.post4",
        npu=SimpleNamespace(get_soc_version=native),
        utils=SimpleNamespace(get_cann_version=lambda: "9.1.0"),
        _C=SimpleNamespace(__file__=__file__),
    )
    monkeypatch.setattr(fixture, "api_source", lambda fn: {"status": "host-test"})
    assert fixture.runtime_evidence(torch, post4)["cann"] == "9.1.0"


def test_shared_registry_inventory_uses_actual_model_objects(monkeypatch):
    runner, params = target.live_fixture(monkeypatch)
    old_layer = runner.compilation_config.static_forward_context["target"]
    actual_layer = torch.nn.Module()
    actual_layer.impl, actual_layer.kv_cache = old_layer.impl, old_layer.kv_cache
    draft_layer = torch.nn.Module()
    runner.model = torch.nn.Module()
    runner.model.add_module("actual_target_attention", actual_layer)
    runner.compilation_config.static_forward_context["target"] = actual_layer
    runner.compilation_config.static_forward_context["draft"] = draft_layer
    runner.attn_groups[0][0].layer_names.append("draft")
    runner.kv_cache_config.kv_cache_groups[0].layer_names.append("draft")
    runner.speculator.draft_attn_layer_names = ["draft"]
    inputs, *_ = target.tool.LiveTargetAdapter(runner, fixture_id="shared")._boundary(target.Descriptor(), runner.batch)
    assert inputs["target_layer_names"] == ["target"]
    assert len(params.attn_params[2]) == 1


def token_joint():
    document, launcher, raw, expected = acceptance.joint_inputs()
    requests = json.loads(raw)
    for request, output in zip(requests, document["outputs"]):
        request["prompts"] = [{"prompt_token_ids": list(p)} for p in output["prompt_token_ids"]]
        output["prompts"] = request["prompts"]
    raw = json.dumps(requests).encode()
    checksum = hashlib.sha256(raw).hexdigest()
    document["request_manifest"] = {"entries": requests, "sha256": checksum}
    launcher["requests_sha256"] = expected["requests_sha256"] = checksum
    return document, launcher, raw, expected


def test_official_joint_accepts_exact_token_id_inputs():
    assert acceptance._tool().audit_joint(*token_joint())["status"] == "PASS"


@pytest.mark.parametrize("tokens", [[], [True], [-1], ["1"], [1.0]])
def test_joint_rejects_invalid_token_inputs(tokens):
    document, launcher, raw, expected = token_joint()
    requests = json.loads(raw)
    requests[0]["prompts"][0]["prompt_token_ids"] = tokens
    raw = json.dumps(requests).encode()
    checksum = hashlib.sha256(raw).hexdigest()
    document["request_manifest"] = {"entries": requests, "sha256": checksum}
    launcher["requests_sha256"] = expected["requests_sha256"] = checksum
    assert acceptance._tool().audit_joint(document, launcher, raw, expected)["status"] != "PASS"
