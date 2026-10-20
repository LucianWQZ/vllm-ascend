# SPDX-License-Identifier: Apache-2.0
"""Actual Python builders/managers/checkpoints; native NPU execution substituted.

The Target native replay below is a deterministic host stand-in, NOT Qwen or
device validation. Its purpose is checking the entire producer/data/consumer
contract, including actual batch assembly and actual production FIA tuple.
"""

import json
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

from tests.ut.spec_decode import test_graph_repair4_integration as integration
from tests.ut.spec_decode import test_graph_repair4_target as old
from tests.ut.spec_decode.test_draft_window_capture_observer import _real_builder
from tests.ut.spec_decode.test_graph_repair5_contracts import load


@contextmanager
def context(*args, **kwargs):
    yield


def controlled_runner(monkeypatch):
    import vllm_ascend.attention.attention_v1 as backend
    import vllm_ascend.ops.rotary_embedding as rotary
    import vllm_ascend.worker.v2.aclgraph_utils as graph
    import vllm_ascend.worker.v2.input_batch as batch
    from vllm_ascend.worker.v2.model_states.default import AscendModelState

    runner, params = old.live_fixture(monkeypatch)
    monkeypatch.setattr(batch, "update_cos_sin", lambda _: None)
    monkeypatch.setattr(rotary, "update_cos_sin", lambda _: None)
    runner.device = torch.device("cpu")
    runner.input_buffers = batch.AscendInputBuffers(4, 32, runner.device)
    layer = runner.compilation_config.static_forward_context["target"]
    layer.kv_cache = (torch.zeros(128, 128, 1, 2), torch.zeros(128, 128, 1, 2))
    layer.impl.key_cache, layer.impl.value_cache = layer.kv_cache
    builder = _real_builder(monkeypatch)
    runner.attn_groups[0][0].get_metadata_builder = lambda _: builder
    runner.block_tables = SimpleNamespace(
        input_block_tables=(torch.zeros(4, 32, dtype=torch.int32),),
        slot_mappings=torch.full((1, 32), -1, dtype=torch.int64),
        num_blocks=SimpleNamespace(cpu=torch.zeros(1, 4, dtype=torch.int32), gpu=torch.zeros(1, 4, dtype=torch.int32)),
    )
    runner.model_state = AscendModelState.__new__(AscendModelState)
    runner.model_state.max_model_len = 4096
    manager = graph.ModelAclGraphManager.__new__(graph.ModelAclGraphManager)
    runner.cudagraph_manager = manager
    manager.model_runner = runner
    manager.vllm_config = SimpleNamespace()
    runner.dp_size = 1
    manager.is_last_pp_rank = True
    manager.use_aux_hidden_state_outputs = False
    manager.hidden_states = torch.zeros(32, 2)
    manager.update_stream = SimpleNamespace(wait_stream=lambda _: None)
    runner.model.compute_logits = lambda hidden: hidden.repeat(1, 2)
    monkeypatch.setattr(graph, "set_current_vllm_config", context)
    monkeypatch.setattr(graph, "set_forward_context", context)
    monkeypatch.setattr(graph, "get_forward_context", lambda: object())
    monkeypatch.setattr(graph, "_get_graph_update_backend", lambda _: object())
    monkeypatch.setattr(graph, "update_full_graph_params", lambda *a, **kw: None)
    runner.speculative_config = None
    manager.graphs = {}
    for n in (1, 2):
        desc = BatchExecutionDescriptor(CUDAGraphMode.FULL, num_tokens=n * 8, num_reqs=n, uniform_token_count=8)

        def replay(n=n):
            # Native replay substitute modifies declared KV slots and persistent
            # hidden buffer, keeping actual upstream/Ascend run_fullgraph intact.
            slots = runner.block_tables.slot_mappings[0, : n * 8]
            valid = slots >= 0
            for kv in layer.kv_cache:
                flat = kv.view(-1, 2)
                flat[slots[valid]] = 0.25
            manager.hidden_states[: n * 8] = runner.input_buffers.input_ids[: n * 8, None].float()

        manager.graphs[desc] = SimpleNamespace(replay=replay)
        params.attn_params[n * 8], params.handles[n * 8], params.events[n * 8] = [], [], []
        params.workspaces[n * 8] = object()
        md = SimpleNamespace(
            block_tables=runner.block_tables.input_block_tables[0][:n],
            seq_lens_list=[8] * n,
            actual_seq_lengths_q=[8 * (i + 1) for i in range(n)],
            attn_mask=None,
            attn_state=backend.AscendAttentionState.DecodeOnly,
            causal=True,
        )
        q = torch.zeros(n * 8, 1, 2)
        # Real production constructor, not a hand-assembled 22-item tuple.
        layer.impl.full_graph_fia(q, *layer.kv_cache, md, q.clone())
        assert len(params.attn_params[n * 8][0]) == 22
    return runner


def produce(monkeypatch, tmp_path, arm):
    runner = controlled_runner(monkeypatch)
    tool = load("p7x_graph_repair5_target")
    monkeypatch.setattr(tool, "_base", lambda: old.tool)
    fixture = tool.ControlledTarget(
        runner,
        fixture_id="test-fixture",
        arm=arm,
        scheduler=SimpleNamespace(requests={}, running=[], waiting=[]),
        identity=dict(run_id="host", case_id=arm, attempt_id="1", source_sha="host", source_tree="host", process_id=1),
    )
    result = fixture.execute(tmp_path / f"{arm}.pt")
    return tool, result


def test_controlled_plan_real_builder_tuple_manager_checkpoint_checker(monkeypatch, tmp_path):
    tool, left = produce(monkeypatch, tmp_path, "OFF")
    _, right = produce(monkeypatch, tmp_path, "ON")
    result = tool.compare_collections(left, right)
    assert result["status"] == "OPEN", result
    assert sum(p["status"] == "PASS" for p in result["pairs"]) == 10
    assert sum(p["status"] == "PRODUCER_BLOCKED" for p in result["pairs"]) == 1
    assert left["records"][-1]["error"].startswith("CAPTURE_BUCKET_ABSENT")
    assert left["records"][0]["block_owners"] != right["records"][0]["block_owners"]
    restored = torch.load(tmp_path / "ON.pt", weights_only=True)
    assert tool.compare_collections(left, restored) == result
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("damage", ["missing", "duplicate", "generation", "nan", "slot", "query", "post_kv", "output"])
def test_controlled_checker_rejects_incomplete_corrupt_and_numeric_failure(monkeypatch, tmp_path, damage):
    tool, left = produce(monkeypatch, tmp_path, "OFF")
    _, right = produce(monkeypatch, tmp_path, "ON")
    record = right["records"][0]
    if damage == "missing":
        right["records"].pop(0)
    elif damage == "duplicate":
        right["records"].append(deepcopy(record))
    elif damage == "generation":
        record["generation"] += 1
    elif damage in ("nan", "output"):
        record["logits"][0, 0] = float("nan") if damage == "nan" else 99
    elif damage in ("slot", "query"):
        entries = record["checkpoint"]["inputs"]["tensors"]
        key = next(k for k in entries if k.endswith("slot_mapping" if damage == "slot" else "query_start_loc"))
        entries[key]["data"][0] += 1
    else:
        blocks = next(iter(record["logical_post_kv"].values()))
        next(iter(blocks.values())).flatten()[0] += 1
    result = tool.compare_collections(left, right)
    assert result["status"] != "PASS"
    if damage == "output":
        assert result["status"] == "FAIL", result  # BLOCKED bucket cannot hide confirmed failure.


class OrdinaryEvent:
    def __init__(self):
        self.complete = False

    def record(self, stream):
        self.stream = stream

    def synchronize(self):
        self.complete = True

    def query(self):
        return self.complete


def install_native_update_observation(monkeypatch, session, manager, calls):
    def native_update(*_args, **kwargs):
        calls.append("native_update")
        binding = kwargs["draft_window_graph_update_binding"]
        carrier = binding.carriers[0]
        session.recorder.record_update_operand(
            binding=binding,
            draft_metadata=carrier.metadata,
            layer_name=integration.LAYER,
            captured_block_table=carrier.read_inputs.window_block_table,
            actual_block_table=carrier.read_inputs.window_block_table,
            seq_lengths=carrier.read_inputs.device_window_exact,
            num_tokens=carrier.capture_identity.fingerprint.query_width * carrier.read_inputs.padded_request_count,
            handle=object(),
            event=object(),
            stream=manager.update_stream,
            backend_impl=integration.AscendAttentionBackendImpl,
        )

    monkeypatch.setattr(integration.AscendAttentionBackendImpl, "update_graph_params", staticmethod(native_update))


def test_actual_proposal_manager_retire_next_generation_lifecycle(monkeypatch, tmp_path):
    tool = load("p7x_graph_repair5_lifecycle")
    owner, session, manager, descriptors, calls = integration._lifetime_fixture(monkeypatch)
    owner.query_cudagraph_manager = manager
    manager.max_num_reqs = 8
    owner.draft_tokens = torch.zeros(8, 7, dtype=torch.int64)
    manager.update_stream.npu_stream = 2
    monkeypatch.setattr(torch.npu, "current_stream", lambda: SimpleNamespace(npu_stream=1))
    monkeypatch.setattr(torch.npu, "Event", OrdinaryEvent)
    path = tmp_path / "life.json"
    inputs = []
    identity = dict(
        run_id="r",
        case_id="c",
        attempt_id="a",
        source_sha="a" * 40,
        source_tree="b" * 40,
        process_id=__import__("os").getpid(),
        engine_id=123,
        driver_mode="B",
    )
    owner.draft_attn_layer_names = [integration.LAYER]
    install_native_update_observation(monkeypatch, session, manager, calls)
    with tool.MRV2LifecycleProbe(owner, path, identity):
        for active, padded in ((1, 1), (2, 2), (4, 4), (1, 1), (3, 4)):
            integration._batch(owner, active)
            with owner._draft_window_proposal_scope(owner.input_batch, dummy_run=False, is_profile=False):
                inputs.append(integration._publish_in_scope(owner, descriptors[padded]))
                assert manager.run_fullgraph(descriptors[padded]) is None
    doc = json.loads(path.read_text())
    assert tool.check_lifecycle(doc)["status"] == "STRUCTURE_VALID_NOT_EXECUTION_ACCEPTED"
    assert len(doc["records"]) == 5
    evidence = load("p7x_graph_repair5_evidence")
    assert evidence.correlate_lifecycle(doc, session.recorder.snapshot())["status"] == "RECORDER_ASSOCIATED"
    for damage in ("empty", "duplicate", "pid", "layer", "address", "epoch"):
        snap = deepcopy(session.recorder.snapshot())
        if damage == "empty":
            snap["operand_events"] = []
        elif damage == "duplicate":
            snap["lifecycle_events"].append(snap["lifecycle_events"][-1])
        elif damage == "pid":
            snap["process_id"] += 1
        else:
            e = next(x for x in snap["operand_events"] if x["stage"] == "update_operator")
            e[{"layer": "layer_name", "address": "actual_table_address", "epoch": "mapping_epoch"}[damage]] = -1
        with pytest.raises(ValueError):
            evidence.correlate_lifecycle(doc, snap)
    assert calls == ["native_replay", "native_update"] * 5
    # Provider retains last submission for diagnostics; P3 lease, not this
    # Python reference, authorizes reuse/access.
    assert session.provider.submitted_carriers
    for published in inputs:
        assert published.lease.state.value == "retired"
        with pytest.raises(RuntimeError):
            published.validate_for(published.identity)
    for damage in ("duplicate", "provider", "completion", "retired", "epoch", "event", "backing"):
        corrupt = deepcopy(doc)
        if damage == "duplicate":
            corrupt["records"].append(corrupt["records"][0])
        elif damage == "provider":
            corrupt["records"][1]["provider"] += 1
        elif damage == "epoch":
            corrupt["records"][1]["epoch"] = 0
        elif damage == "event":
            del corrupt["records"][1]["consumer_event"]
        elif damage == "backing":
            corrupt["records"][1]["backing"]["window_block_table"]["data_ptr"] = 0
        else:
            corrupt["records"][1][damage] = False
        assert tool.check_lifecycle(corrupt)["status"] == "INVALID_EVIDENCE"


@pytest.mark.parametrize("damage", ["missing", "duplicate", "draft", "task", "tuple"])
def test_inventory_model_objects_and_complete_capture_chain(monkeypatch, damage):
    runner, params = old.live_fixture(monkeypatch)
    if damage == "missing":
        runner.model.modules = lambda: iter([])
    elif damage == "duplicate":
        runner.kv_cache_config.kv_cache_groups[0].layer_names.append("target")
    elif damage == "draft":
        runner.speculator.draft_attn_layer_names = ["target"]
    elif damage == "task":
        params.handles[2].clear()
    else:
        params.attn_params[2][0] += (None,)
    with pytest.raises(old.tool.EvidenceError):
        old.tool.LiveTargetAdapter(runner, fixture_id="bad")._boundary(old.Descriptor(), runner.batch)


@pytest.mark.parametrize("point", ["before_replay", "after_replay", "after_update", "after_completion"])
def test_abort_actual_scope_is_quarantined_and_launcher_checked(monkeypatch, tmp_path, point):
    tool = load("p7x_graph_repair5_lifecycle")
    owner, session, manager, descriptors, calls = integration._lifetime_fixture(monkeypatch)
    manager.max_num_reqs = 8
    owner.query_cudagraph_manager = manager
    owner.draft_attn_layer_names = [integration.LAYER]
    install_native_update_observation(monkeypatch, session, manager, calls)
    owner.draft_tokens = torch.ones(8, 7, dtype=torch.int64)
    manager.update_stream.npu_stream = 2
    monkeypatch.setattr(torch.npu, "current_stream", lambda: SimpleNamespace(npu_stream=1))
    monkeypatch.setattr(torch.npu, "Event", OrdinaryEvent)
    identity = dict(
        run_id="r",
        case_id="ABORT",
        attempt_id="1",
        source_sha="s",
        source_tree="t",
        driver_mode="B",
        process_id=owner._draft_window_graph_characterization.recorder.snapshot()["process_id"],
    )
    argv = ["python", "run", "--repair5-abort-at", point]
    launcher = dict(identity, argv=argv, pid=identity["process_id"], returncode=73, signal=None, timeout=False)
    expected = dict(acceptance_identity=identity, argv=argv)
    monkeypatch.setattr(tool.os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))
    path = tmp_path / "abort.json"
    with pytest.raises(SystemExit, match="73"), tool.MRV2LifecycleProbe(owner, path, identity, abort_at=point):
        integration._batch(owner, 1)
        with owner._draft_window_proposal_scope(owner.input_batch, dummy_run=False, is_profile=False):
            integration._publish_in_scope(owner, descriptors[1])
            manager.run_fullgraph(descriptors[1])
    doc = json.loads(path.read_text())
    assert tool.check_abort(doc, launcher, expected)["status"] == "EXPECTED_QUARANTINE_OBSERVED"
    evidence = load("p7x_graph_repair5_evidence")
    assert evidence.correlate_abort(doc, doc["recorder"])["status"] == "PARTIAL_RECORDER_ASSOCIATED"
    bad = deepcopy(doc)
    bad["records"][-1]["epoch"] += 1
    with pytest.raises(ValueError, match="generation"):
        evidence.correlate_abort(bad, bad["recorder"])
    assert tool.check_lifecycle(doc)["status"] == "QUARANTINED_NOT_NORMAL_PASS"
    for field, value in (("timeout", True), ("signal", 11), ("returncode", 0), ("pid", 456)):
        invalid = dict(launcher, **{field: value})
        assert tool.check_abort(doc, invalid, expected)["status"] == "INVALID_EVIDENCE"


def test_partial_instrument_install_rolls_back(monkeypatch, tmp_path):
    tool = load("p7x_graph_repair5_lifecycle")
    owner, _, manager, _, _ = integration._lifetime_fixture(monkeypatch)
    owner.query_cudagraph_manager = manager
    previous = [graph.replay for graph in manager.graphs.values()]
    probe = tool.MRV2LifecycleProbe(owner, tmp_path / "x", {})
    install = probe._install
    calls = []

    def fail(obj, name, fn):
        calls.append(name)
        if len(calls) == 2:
            raise ValueError("cannot instrument native object")
        install(obj, name, fn)

    monkeypatch.setattr(probe, "_install", fail)
    with pytest.raises(ValueError, match="native object"):
        probe.__enter__()
    assert [graph.replay for graph in manager.graphs.values()] == previous
    assert probe.saved == []


def test_full_target_and_later_draft_slot_phases_are_not_conflated(monkeypatch, tmp_path):
    """Propose is a HOST substitute here; real manager lifecycle is tested above."""
    documents = []
    tool = load("p7x_graph_repair5_target")
    monkeypatch.setattr(tool, "_base", lambda: old.tool)
    for arm in ("OFF", "ON"):
        runner = controlled_runner(monkeypatch)
        monkeypatch.setattr(torch.npu, "current_stream", lambda: SimpleNamespace(synchronize=lambda: None))
        spec = runner.speculator
        spec.sample_from_anchor, spec.num_query_per_req, spec.max_num_reqs = True, 7, 4
        spec.block_tables = runner.block_tables
        spec.input_buffers = SimpleNamespace(
            positions=torch.zeros(32, dtype=torch.int64), query_start_loc=torch.zeros(5, dtype=torch.int32)
        )
        spec.update_stream = SimpleNamespace(synchronize=lambda: None)

        def propose(batch, *args, spec=spec, runner=runner):
            for row in range(batch.num_reqs):
                end = int(batch.seq_lens[row])
                for q, pos in enumerate(range(end, end + 7)):
                    spec.block_tables.slot_mappings[0, row * 7 + q] = (
                        runner.block_tables.input_block_tables[0][row, pos // 128] * 128 + pos % 128
                    )
                    spec.input_buffers.positions[row * 7 + q] = pos
            spec.input_buffers.query_start_loc[: batch.num_reqs + 1] = torch.arange(batch.num_reqs + 1) * 7
            return torch.ones(batch.num_reqs, 7, dtype=torch.int64)

        spec.propose = propose
        fixture = tool.ControlledTarget(
            runner,
            fixture_id="host",
            arm=arm,
            scheduler=SimpleNamespace(requests={}, running=[], waiting=[]),
            identity=dict(run_id="r", case_id=arm, attempt_id="1", source_sha="s", source_tree="t", process_id=1),
        )
        documents.append(fixture.execute(tmp_path / (arm + ".pt"), include_draft=True))
    result = tool.compare_collections(*documents)
    assert result["status"] == "OPEN", result
    assert sum(p["status"] == "PASS" for p in result["pairs"]) == 10
    assert result["window_interference_coverage"] == "COLLECTED"
    documents[1]["records"][0]["target_after_draft"]["draft_slots"][0] += 1
    assert tool.compare_collections(*documents)["status"] == "INVALID_EVIDENCE"


def test_real_mrv2_aba_checker_schema_and_corruptions(monkeypatch, tmp_path):
    """Target records from actual host producer; Draft evidence is schema-only.

    Actual Draft builder/manager/lifecycle runs in the separate integration test.
    This test does not claim executing prepare_dflash_inputs Triton on CPU.
    """
    tool, doc = produce(monkeypatch, tmp_path, "ON")
    for rec in doc["records"]:
        if rec["status"] == "BLOCKED":
            continue
        ident = rec["identity"]
        data = old.tool._snapshot_value(rec["checkpoint"]["inputs"])
        table = next(iter(data["layers"].values()))["block_tables"].tolist()
        ends = [p[-1] + 8 for p in ident["positions"]]
        lengths, window = [], []
        for row, end in enumerate(ends):
            start, stop = max(end - 1024, 0) // 128, (end + 127) // 128
            selected = table[row][start:stop]
            lengths.append(end - start * 128)
            window.append(selected + [0] * (9 - len(selected)))
        from tests.ut.spec_decode.test_graph_repair5_review import actual_candidates

        desc = actual_candidates(7, [7, 14, 28]).dispatch(ident["active"], ident["active"] * 7, 7, 0)
        lengths += [0] * (desc.num_reqs - ident["active"])
        window += [[0] * 9 for _ in range(desc.num_reqs - ident["active"])]
        g4 = dict(
            generation_id=ident["step"],
            mapping_epoch=ident["step"],
            window_table_address=123,
            canonical_table_values=table,
            window_table_values=window,
            runtime_oracle_inputs=dict(
                request_ids=ident["logical_requests"],
                full_exact=ends,
                full_upper=ends,
                valid_counts=[32] * ident["active"],
                source_identity={"host": True},
            ),
        )
        rec["draft_snapshot"] = dict(
            provider_identity=456,
            g4_records=[g4],
            events=[
                dict(
                    stage="bound",
                    generation_id=ident["step"],
                    mapping_epoch=ident["step"],
                    exact_lengths=lengths,
                    upper_lengths=lengths,
                )
            ],
        )
        rec["draft_geometry"] = dict(
            descriptor=[desc.num_reqs, desc.num_tokens],
            active=ident["active"],
            padded=desc.num_reqs,
            query_layout=[i * 7 for i in range(desc.num_reqs + 1)],
            generation=ident["step"],
            epoch=ident["step"],
            source_identity={"host": True},
        )
        rec["draft_tokens"] = torch.full((ident["active"], 7), 2 if ident["scenario"] == "B" else 1)
    assert tool.check_mrv2_aba(doc)["status"] == "PASS_SCOPED_NUMERIC"
    for damage in ("generation", "provider", "table", "length", "sample_order", "output"):
        bad = deepcopy(doc)
        r = bad["records"][1]
        snapshot = r["draft_snapshot"]
        if damage == "generation":
            snapshot["g4_records"][0]["generation_id"] = 0
        elif damage == "provider":
            snapshot["provider_identity"] += 1
        elif damage == "table":
            snapshot["g4_records"][0]["window_table_values"][0][0] += 1
        elif damage == "length":
            snapshot["events"][0]["exact_lengths"][0] += 1
        elif damage == "sample_order":
            bad["records"][0], bad["records"][1] = bad["records"][1], bad["records"][0]
        else:
            bad["records"][2]["draft_tokens"].fill_(99)
        assert tool.check_mrv2_aba(bad)["status"] == "INVALID_EVIDENCE"
    doc["records"][1]["draft_tokens"].fill_(1)
    assert tool.check_mrv2_aba(doc)["status"] == "OPEN_NONDISCRIMINATING_DRAFT_TOKENS"
