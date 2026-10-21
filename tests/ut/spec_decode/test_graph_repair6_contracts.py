# SPDX-License-Identifier: Apache-2.0
"""Real Python model/capture/retirement boundaries, native NPU substitutes only."""

import inspect
import json
import sys
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode

from tests.ut.spec_decode.test_graph_repair4_integration import _batch, _lifetime_fixture, _publish_in_scope
from tests.ut.spec_decode.test_graph_repair5_contracts import load
from vllm_ascend.worker.v2.spec_decode.draft_window_graph_diagnostics import DraftGraphOutputReferences
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator


@contextmanager
def context(*args, **kwargs):
    yield


def test_real_pinned_backbone_capture_retains_actual_output(monkeypatch):
    import vllm.v1.worker.gpu.spec_decode.dflash.speculator as upstream

    monkeypatch.setattr(upstream, "set_forward_context", context)
    owner = AscendDSparkSpeculator.__new__(AscendDSparkSpeculator)
    owner.vllm_config = object()
    owner.input_buffers = SimpleNamespace(input_ids=torch.arange(28), positions=torch.arange(28))

    class Backbone(torch.nn.Module):
        def forward(self, input_ids, positions, inputs_embeds):
            return torch.stack((input_ids, positions), dim=1).to(torch.bfloat16)

    owner.model = Backbone()
    assert "set_forward_context" in inspect.getsource(upstream.DFlashSpeculator._run_model)
    # OFF does not create any sink or change the real upstream model return.
    output = owner._run_model(7, {}, {}, None)
    assert output.shape == (7, 2) and not hasattr(owner, "_draft_graph_output_references")
    sink = owner._draft_graph_output_references = DraftGraphOutputReferences(owner)
    outputs = {}
    for n in (28, 14, 7):
        owner._run_model(n, {}, {}, None, CUDAGraphMode.NONE)  # warmup
        assert n not in sink.outputs
        outputs[n] = owner._run_model(n, {}, {}, None, CUDAGraphMode.FULL)
    sink.freeze([7, 14, 28])
    old = outputs[7].clone()
    for n in (7, 14, 28, 7):
        # Native replay substitute writes the retained graph-output allocation.
        with torch.inference_mode():
            outputs[n].add_(1)
        assert sink.get(owner, n) is outputs[n]
    assert torch.equal(old + 2, outputs[7])
    with pytest.raises(RuntimeError, match="owner/scope"):
        sink.retain(owner, 7, outputs[7])
    with pytest.raises(RuntimeError, match="foreign"):
        sink.get(object(), 7)


def test_normal_order_real_binding_retirement_no_per_generation_query(monkeypatch, tmp_path):
    module = load("p7x_graph_repair6_lifecycle")
    owner, _session, manager, descriptors, _calls = _lifetime_fixture(monkeypatch)
    owner.query_cudagraph_manager = manager
    manager.max_num_reqs = 8
    manager._capture_descs = {CUDAGraphMode.FULL: list(descriptors.values())}
    owner.draft_tokens = torch.zeros((8, 7), dtype=torch.int64)
    events = []
    current = SimpleNamespace(npu_stream=1)
    manager.update_stream.npu_stream = 2

    class Event:
        def record(self, stream):
            events.append(("record", stream.npu_stream))

        def synchronize(self):
            events.append(("sync", None))

        def query(self):
            pytest.fail("no query is allowed")

    monkeypatch.setattr(torch.npu, "current_stream", lambda: current)
    monkeypatch.setattr(torch.npu, "Event", Event)
    path = tmp_path / "normal.json"
    with module.NormalOrderProbe(owner, path, {"run_id": "host"}, max_generations=6):
        for i, (active, padded) in enumerate(((1, 1), (2, 2), (3, 4), (1, 1), (2, 2), (4, 4))):
            _batch(owner, active)
            with owner._draft_window_proposal_scope(owner.input_batch, dummy_run=False, is_profile=False):
                _publish_in_scope(owner, descriptors[padded])
                owner.draft_tokens.fill_(i)
                assert manager.run_fullgraph(descriptors[padded]) is None
            assert not events
    document = json.loads(path.read_text())
    assert len(document["records"]) == 6
    assert [r["tokens"][0][0] for r in document["records"]] == list(range(6))
    assert all(r["retired"] and r["live_inputs_cleared"] for r in document["records"])
    assert len(events) == 4 and document["normal_unobserved_reuse"] == "UNPROVEN"
    assert "run_fullgraph" not in vars(manager)


def test_normal_order_failure_is_quarantined_without_device_cleanup(monkeypatch, tmp_path):
    module = load("p7x_graph_repair6_lifecycle")
    owner, _, manager, _, _ = _lifetime_fixture(monkeypatch)
    owner.query_cudagraph_manager = manager
    monkeypatch.setattr(torch.npu, "Event", lambda: pytest.fail("uncertain submit must not pretend drain"))
    path = tmp_path / "failed.json"
    with pytest.raises(RuntimeError, match="native failure"), module.NormalOrderProbe(owner, path, {}):
        raise RuntimeError("native failure")
    assert json.loads(path.read_text())["status"] == "QUARANTINED"


def test_independent_canonical_window_and_invalid_source():
    tool = load("p7x_graph_repair6_numeric")
    full = torch.arange(30, dtype=torch.int32).reshape(2, 15)
    table, lengths, rows = tool.independent_window(full, [1159, 39], [10, 1], 2, 4, 9)
    assert rows[0] == dict(S=128, E=1159, V=1031, columns=[1, 10])
    assert lengths == [1031, 39, 0, 0]
    assert table[0].tolist() == list(range(1, 10))
    assert torch.count_nonzero(table[2:]) == 0
    with pytest.raises(ValueError, match="shortage"):
        tool.independent_window(full, [1159], [9], 1, 1, 9)


def test_continuous_reference_actual_pinned_forward_and_real_generation(monkeypatch, tmp_path):
    import vllm.v1.worker.gpu.spec_decode.dflash.speculator as upstream

    from tests.ut.spec_decode import test_graph_repair4_integration as integration
    from tools.p7x_graph_repair6_numeric import ContinuousReference
    from tools.p7x_graph_repair6_storage import CollectionReader

    owner, _, manager, descriptors, _ = _lifetime_fixture(monkeypatch)
    # The existing ownership fixture uses toy 64-token blocks. This short case
    # occupies column zero in either geometry; no large-window NPU claim.
    descriptor = integration._read_descriptor
    monkeypatch.setattr(integration, "_read_descriptor", lambda **kw: replace(descriptor(**kw), required_columns=1))
    owner.__class__ = AscendDSparkSpeculator
    owner.query_cudagraph_manager = manager
    manager.max_num_reqs = 8
    manager._capture_descs = {CUDAGraphMode.FULL: list(descriptors.values())}
    manager.update_stream.synchronize = lambda: None
    monkeypatch.setattr(torch.npu, "current_stream", lambda: SimpleNamespace(synchronize=lambda: None))
    monkeypatch.setattr(upstream, "set_forward_context", context)
    owner.sample_from_anchor = True
    owner.vllm_config = object()
    owner.device = torch.device("cpu")
    owner.input_buffers.input_ids = torch.arange(64)
    owner.draft_attn_layer_names = {integration.LAYER}
    cache = torch.zeros((65, 128, 1, 1), dtype=torch.bfloat16)

    class Backbone(torch.nn.Module):
        def forward(self, input_ids, positions, inputs_embeds):
            value = torch.stack((input_ids, positions), 1).to(torch.bfloat16) + cache[1, 0, 0, 0]
            cache[1].add_(1)
            return value

    owner.model = Backbone()
    sink = owner._draft_graph_output_references = DraftGraphOutputReferences(owner)
    with torch.inference_mode():
        for n in (7, 14, 28):
            owner._run_model(n, {}, {}, None, CUDAGraphMode.FULL)
    sink.freeze([7, 14, 28])
    for desc in descriptors.values():

        def replay(d=desc):
            with torch.inference_mode():
                sink.get(owner, d.num_tokens).copy_(
                    owner.model(
                        owner.input_buffers.input_ids[: d.num_tokens],
                        owner.input_buffers.positions[: d.num_tokens],
                        None,
                    )
                )

        manager.graphs[desc].replay = replay
    runner = SimpleNamespace(
        speculator=owner,
        compilation_config=SimpleNamespace(
            static_forward_context={integration.LAYER: SimpleNamespace(kv_cache=[cache])}
        ),
    )
    path = tmp_path / "numeric.json"
    with ContinuousReference(
        runner, SimpleNamespace(requests={}, running=[], waiting=[]), path, {"run_id": "host"}
    ) as probe:
        for step, (active, padded) in enumerate(((1, 1), (2, 2), (3, 4), (1, 1))):
            _batch(owner, active)
            with owner._draft_window_proposal_scope(owner.input_batch, dummy_run=False, is_profile=False):
                _publish_in_scope(owner, descriptors[padded])
                owner.input_buffers.seq_lens[:active].fill_(64)
                probe.current_case = dict(active=active, step=step, scenario=str(step))
                with torch.inference_mode():
                    cache.zero_()
                    assert manager.run_fullgraph(descriptors[padded]) is None
                assert cache[1, 0, 0, 0] == 1  # reference post-state restored to actual Graph post-state
            assert owner._draft_window_live_inputs is None
    reader = CollectionReader(path)
    assert len(reader.samples) == 4
    assert all(reader.record(i)["status"] == "PASS_SCOPED_CONTINUOUS" for i in range(4))
    assert reader.record(0)["graph"][0, 0] == 0  # immutable durable snapshot, not last generation's live buffer


def test_complete_flow_context_no_adjacent_time_guess():
    tool = load("p7x_graph_repair6_trace")
    events = [
        dict(ph="X", name="repair5/Draft_REPLAY_PLUS_UPDATE", pid=1, tid=1, ts=0, dur=20),
        dict(ph="X", name="repair5/Draft_REPLAY_SUBMIT", pid=1, tid=1, ts=2, dur=5),
        dict(ph="s", name="torch_to_npu", cat="async_npu", id=9, pid=1, tid=1, ts=3),
        dict(ph="f", name="torch_to_npu", cat="async_npu", id=9, pid=2, tid=3, ts=22),
        dict(ph="X", name="device-task", pid=2, tid=3, ts=21, dur=5),
    ]
    result = tool.analyze(events)
    assert result["coverage"] == dict(total_replays=1, direct_matched=1)
    assert {r["original_event_index"] for r in result["fragment"]} == set(range(5))
    # Missing/duplicate endpoints must not be paired by proximity.
    assert tool.analyze(events[:3])["coverage"]["direct_matched"] == 0
    assert tool.analyze(events + [events[3]])["flow_counts"] == {"AMBIGUOUS": 1}
    local = [dict(e, id2={"local": 9}) if e["ph"] in ("s", "f") else e for e in events]
    assert tool.analyze(local)["flow_counts"] == {"UNMATCHED": 2}


def test_runner_timeout_and_prepublication_are_not_numeric_failure(tmp_path):
    module = load("p7x_graph_repair6_runner")
    result = module.execute([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path / "cmd", 0.05)
    assert result["timeout"] and result["status"] == "COMPARISON_TIMEOUT"
    with pytest.raises(ValueError, match="PRE_PUBLICATION"):
        module.run_case({"publication_status": "NOT_PUBLISHED", "cases": {"A": {}}}, "A", tmp_path)


def test_readonly_reader_missing_store_does_not_create_it(tmp_path):
    module = load("p7x_graph_repair6_storage")
    path = tmp_path / "missing"
    with pytest.raises(ValueError, match="missing tensor store"):
        module.TensorStore(path, readonly=True)
    assert not path.exists()
