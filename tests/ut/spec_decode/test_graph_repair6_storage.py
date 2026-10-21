# SPDX-License-Identifier: Apache-2.0
"""CPU tensor contracts, not native Graph or device completion evidence."""

import json

import pytest
import torch

from tests.ut.spec_decode.test_graph_repair5_contracts import load


def test_snapshot_shared_group_mask_is_copied_once_without_losing_fields():
    tool = load("p7x_graph_repair4_target")
    mask = torch.ones(16, 16, dtype=torch.bool)
    # Real builder shares the same group metadata/mask across layer names.
    result = tool._snapshot({"layer0": mask, "layer1": mask}, [mask.numel()])
    a, b = result["tensors"]["/layer0"], result["tensors"]["/layer1"]
    assert a["data"].data_ptr() == b["data"].data_ptr()
    assert a["alias"] == b["alias"]
    assert torch.equal(a["data"], mask)


def test_chunk_writer_reader_and_integrity(tmp_path):
    tool = load("p7x_graph_repair6_storage")
    path = tmp_path / "collection.json"
    writer = tool.CollectionWriter(path, {"run": "test"}, chunk_bytes=64)
    identity = {"step": 0}
    writer.start(identity)
    value = torch.arange(200, dtype=torch.float32).reshape(10, 20)
    record = dict(identity=identity, status="CAPTURED_COMPLETE", values=value, generation=0)
    writer.finish(record)
    writer.close("COMPLETE")
    reader = tool.CollectionReader(path)
    assert torch.equal(reader.record(0)["values"], value)
    assert max(c["bytes"] for v in writer.store.refs.values() for c in v[1]["chunks"]) <= 64
    blob = next(writer.store.root.glob("*.bin"))
    with blob.open("r+b") as f:
        f.write(b"bad!")
    with pytest.raises(ValueError, match="hash"):
        reader.record(0)


def test_incomplete_duplicate_and_budget_rejected(tmp_path):
    tool = load("p7x_graph_repair6_storage")
    w = tool.CollectionWriter(tmp_path / "c.json", {}, max_bytes=32)
    w.start({"step": 0})
    with pytest.raises(ValueError, match="duplicate"):
        w.start({"step": 0})
    with pytest.raises(ValueError, match="incomplete"):
        tool.CollectionReader(w.path).record(0)
    with pytest.raises(ValueError, match="BLOCKED_RESOURCE"):
        w.store.put(torch.ones(64))


def test_real_geometry_budget_is_not_collection_bytes():
    t = load("p7x_graph_repair5_target")
    s = load("p7x_graph_repair6_storage")
    plan = s.resource_plan(
        t.case_plan("target4"),
        layers=36,
        kv_heads=8,
        head_dim=128,
        element_bytes=2,
        hidden_size=4096,
        vocab_size=151936,
        mask_bytes=2048**2,
    )
    assert len(plan["samples"]) == 12
    assert plan["samples"][0]["kv_one_copy"] == 169869312
    assert plan["one_arm_bytes"] > max(r["checkpoint"] for r in plan["samples"])
    assert plan["samples"][7]["checkpoint"] < 512 << 20  # shared mask is only one copy


def test_real_builder_to_chunk_store_to_comparator(monkeypatch, tmp_path):
    from types import SimpleNamespace

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
            identity=dict(
                run_id="host", case_id=arm, attempt_id="1", source_sha="host", source_tree="host", process_id=1
            ),
            bounded_store=True,
        )
        fixture.execute(tmp_path / (arm + ".json"))
        assert not fixture.probe.records
        assert all("checkpoint" not in r for r in fixture.records)
    result = load("p7x_graph_repair6").compare(tmp_path / "OFF.json", tmp_path / "ON.json")
    assert result["complete_pairs"] == 10, result
    assert result["status"] == "PARTIAL_PREFIX_VALID"  # real base manager has no Target (4,32)
    assert result["campaign"] == "INCOMPLETE_COVERAGE"
    doc = json.loads((tmp_path / "ON.json").read_text())
    doc["samples"].append(doc["samples"][0])
    (tmp_path / "ON.json").write_text(json.dumps(doc))
    assert (
        load("p7x_graph_repair6").compare(tmp_path / "OFF.json", tmp_path / "ON.json")["status"] == "INVALID_EVIDENCE"
    )
