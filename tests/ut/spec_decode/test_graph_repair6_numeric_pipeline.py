# SPDX-License-Identifier: Apache-2.0
"""Production capture/builder/P3/P6 + CPU native replay/FIA surrogate.

Real geometry W1024/B128/query7. This is Host framework-contract evidence,
not an NPU numerical or event-semantic verification.
"""

import copy
import json
import os
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.ut.spec_decode.test_graph_repair4_integration import LAYER, _batch, _publish_eager_generation
from tests.ut.spec_decode.test_graph_repair6_capture_chain import capture_owner
from tests.ut.spec_decode.test_graph_repair6_review import IDENTITY, execution
from tools.p7x_graph_repair5_target import case_plan, sample_identity
from tools.p7x_graph_repair6 import joint
from tools.p7x_graph_repair6_numeric import ContinuousReference
from tools.p7x_graph_repair6_storage import CollectionReader, CollectionWriter, atomic_json, digest
from vllm_ascend.worker.v2.spec_decode.draft_window import DraftWindowExecutionPhase, DraftWindowReadDescriptor


def produce(monkeypatch, directory, fault=None):
    import vllm.v1.worker.gpu.spec_decode.dflash.speculator as upstream

    owner, session, manager, descs, cache = capture_owner(monkeypatch)
    owner.kv_cache_config.kv_cache_groups[0].layer_names = [LAYER]
    manager.update_stream.synchronize = lambda: None
    monkeypatch.setattr(torch.npu, "current_stream", lambda: SimpleNamespace(synchronize=lambda: None))
    context = SimpleNamespace(metadata=None)

    @contextmanager
    def forward_context(metadata, *args, **kwargs):
        old = context.metadata
        context.metadata = metadata
        try:
            yield
        finally:
            context.metadata = old

    monkeypatch.setattr(upstream, "set_forward_context", forward_context)
    value_cache = cache.clone()
    caches = [cache, value_cache]
    forward_calls = []

    class Model(torch.nn.Module):
        def forward(self, input_ids, positions, inputs_embeds):
            forward_calls.append(len(input_ids))
            metadata = context.metadata[LAYER]
            out = torch.stack((input_ids.float(), positions.float() / 1024), dim=1)
            for row, e in enumerate(metadata.seq_lens_list):
                ids = metadata.block_tables[row, : (e + 127) // 128].long()
                visible = value_cache.index_select(0, ids).flatten()[:e]
                out[row * 7 : (row + 1) * 7] += visible.mean() if e else 0
            if fault == "numeric" and len(forward_calls) == 4:
                out += 1  # B's reference, after actual graph consumption.
            return out.to(torch.bfloat16)

    owner.model = Model()
    sink = owner._draft_graph_output_references
    for desc in descs.values():

        def replay(d=desc):
            carrier = session.provider.submitted_carriers[0]
            with forward_context(carrier.metadata), torch.inference_mode():
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
        compilation_config=SimpleNamespace(static_forward_context={LAYER: SimpleNamespace(kv_cache=caches)}),
    )
    path = directory / "numeric-reference.json"
    provenance = dict(profile="target4", fixture_id="host", config_sha256="config")
    with ContinuousReference(
        runner,
        SimpleNamespace(requests={}, running=[], waiting=[]),
        path,
        dict(IDENTITY, process_id=os.getpid()),
        provenance=provenance,
    ) as probe:
        for case in case_plan("target4"):
            active = len(case["ends"])
            padded = 1 if active == 1 else 2 if active == 2 else 4
            desc = descs[padded]
            _batch(owner, active)
            owner.input_batch.idx_mapping_np = np.arange(active, dtype=np.int32)
            owner.input_batch.req_ids = tuple(f"row-{i}" for i in range(active))
            exact = [e + 7 for e in case["ends"]]
            owner.input_batch.seq_lens_cpu_upper_bound = torch.tensor(exact, dtype=torch.int32)
            table = owner.block_tables.input_block_tables[0]
            table.zero_()
            owner.block_tables.slot_mappings.fill_(-1)
            owner.input_buffers.input_ids.zero_()
            owner.input_buffers.positions.zero_()
            owner.input_buffers.seq_lens.zero_()
            reads = []
            for row, e in enumerate(exact):
                count = (e + 127) // 128
                full = torch.arange(1 + row * 32, 1 + row * 32 + count, dtype=torch.int32)
                table[row, :count] = full
                owner.block_tables.num_blocks.cpu[0, row] = count
                owner.block_tables.num_blocks.gpu[0, row] = count
                start = max(e - 1024, 0) // 128
                reads.append(DraftWindowReadDescriptor(0, start, 0, count - start, (count - start) * 128, count, 32))
                pos = torch.arange(e - 7, e)
                owner.input_buffers.positions[row * 7 : (row + 1) * 7] = pos
                owner.input_buffers.input_ids[row * 7 : (row + 1) * 7] = 17 + row
                owner.input_buffers.seq_lens[row] = e
                owner.block_tables.slot_mappings[0, row * 7 : (row + 1) * 7] = full[pos // 128] * 128 + pos % 128
            for component, c in enumerate(caches):
                c.copy_(torch.arange(c.shape[0], dtype=torch.bfloat16).view(-1, 1, 1, 1).expand_as(c) / 128 + component)
            owner.input_buffers.query_start_loc[: padded + 1] = torch.arange(padded + 1) * 7
            with owner._draft_window_proposal_scope(owner.input_batch, dummy_run=False, is_profile=False):
                scope = owner._draft_window_proposal_context
                lens = torch.tensor(
                    [e - r.source_start_column * 128 for e, r in zip(exact, reads)] + [0] * (padded - active),
                    dtype=torch.int32,
                )
                inputs = owner._draft_window_runtime_state.materialize(
                    mapping_epoch=scope.mapping_epoch,
                    ordered_request_ids=scope.ordered_request_ids,
                    idx_mapping=torch.from_numpy(owner.input_batch.idx_mapping_np),
                    read_descriptors=tuple(reads),
                    device_window_exact=lens.clone(),
                    cpu_window_exact=lens,
                    cpu_window_upper=lens.clone(),
                    num_reqs_padded=padded,
                    phase=DraftWindowExecutionPhase.PROPOSAL,
                    step=0,
                    execution_lane=0,
                )
                _publish_eager_generation(owner, inputs)
                owner._draft_window_proposal_context = scope
                probe.current_case = sample_identity(case, "host")
                with torch.inference_mode():
                    assert manager.run_fullgraph(desc) is None
            assert owner._draft_window_live_inputs is None and inputs.lease.state.value == "retired"
    return path, probe.provenance, session.recorder.snapshot()


def joined(path, contract, snapshot):
    campaign, launcher, plan = execution(path, CollectionReader(path).header["identity"])
    campaign.update(numeric_contract=contract, capture_profile="target4", config_sha256="config", snapshot=snapshot)
    p = path.with_name("numeric-summary.json")
    campaign["diagnostic_artifacts"][p.name] = dict(sha256=digest(p), bytes=p.stat().st_size)
    plan["fixture_sha256"] = "host"
    return campaign, launcher, plan


def test_full_geometry_actual_capture_to_multigeneration_store_checker(monkeypatch, tmp_path):
    path, contract, snapshot = produce(monkeypatch, tmp_path)
    result = joint(path, *joined(path, contract, snapshot), "numeric")
    assert result["status"] == "PASS_DIAGNOSTIC_CONTINUOUS_REFERENCE", result
    assert result["aba"] == "PASS_SCOPED", result
    reader = CollectionReader(path)
    padded = reader.record(8)
    assert padded["target_descriptor"] == [2, 16] and padded["descriptor"] == [1, 7]
    assert len(reader.samples) == 12 and reader.record(10)["descriptor"] == [4, 28]
    assert not torch.equal(reader.record(0)["graph"], reader.record(1)["graph"])


@pytest.mark.parametrize(
    "damage",
    [
        "exact",
        "actual_table",
        "reference_table",
        "selected",
        "missing_kv",
        "string_kv",
        "dtype",
        "hidden_width",
        "positions",
        "query",
        "epoch",
        "padding",
        "scenario",
        "nan",
        "prestate",
        "payload_fail",
        "execution",
        "truncated",
        "source",
        "recorder",
    ],
)
def test_real_producer_artifact_mutations_never_pass(monkeypatch, tmp_path, damage):
    source = tmp_path / "original"
    source.mkdir()
    path, contract, snapshot = produce(monkeypatch, source)
    reader = CollectionReader(path)
    target = tmp_path / "mutated" / path.name
    writer = CollectionWriter(target, copy.deepcopy(reader.header))
    for i in range(len(reader.samples) - (1 if damage == "truncated" else 0)):
        r = reader.record(i)
        if i == (10 if damage == "padding" else 1):
            if damage == "exact":
                r["exact"][0] = -1
            if damage in ("actual_table", "reference_table"):
                r["actual_metadata" if damage == "actual_table" else "reference_metadata"][LAYER]["table"][0, 0] += 1
            if damage == "selected":
                r["selected_blocks"].pop()
            if damage == "missing_kv":
                r["graph_post_kv"].pop(LAYER + "/1")
            if damage == "string_kv":
                r["pre_kv"] = {k: "placeholder" for k in r["pre_kv"]}
            if damage == "dtype":
                r["pre_kv"][LAYER + "/0"] = r["pre_kv"][LAYER + "/0"].float()
            if damage == "hidden_width":
                r["graph"] = r["graph"][:, :1].contiguous()
            if damage == "positions":
                r["inputs"]["positions"][0] += 1
            if damage == "query":
                r["inputs"]["query_start_loc"][-1] -= 1
            if damage == "epoch":
                r["epoch"] = 0
            if damage == "padding":
                r["actual_metadata"][LAYER]["table"][-1, 0] = 7
            if damage == "scenario":
                r["case_identity"]["scenario"] = "wrong"
            if damage == "nan":
                r["graph"][0, 0] = float("nan")
            if damage == "prestate":
                r["reference_pre_kv"][LAYER + "/0"][0, 0, 0, 0] += 1
            if damage == "payload_fail":
                r["status"] = "NUMERIC_MISMATCH"
            if damage == "source":
                r["identity"]["source_sha"] = "c" * 40
        writer.start(r["identity"])
        writer.finish(r)
    writer.close("EXECUTION_FAILED" if damage == "execution" else "COMPLETE")
    summary = json.loads(path.with_name("numeric-summary.json").read_text())
    summary.update(
        execution=writer.manifest["execution"],
        samples=len(writer.manifest["samples"]),
        statuses=[s["status"] for s in writer.manifest["samples"]],
    )
    atomic_json(target.with_name("numeric-summary.json"), summary)
    campaign, launcher, plan = joined(target, contract, snapshot)
    if damage == "recorder":
        campaign["snapshot"]["events"] = [e for e in snapshot["events"] if e["stage"] != "update"]
    result = joint(target, campaign, launcher, plan, "numeric")
    assert not result["status"].startswith("PASS"), result
    if damage == "payload_fail":
        assert result["status"] == "NUMERIC_MISMATCH"


def test_producer_store_joint_cli_runner_end_to_end(monkeypatch, tmp_path):
    from tools.p7x_graph_repair6_runner import run_case

    path, contract, snapshot = produce(monkeypatch, tmp_path)
    campaign, launcher, plan = joined(path, contract, snapshot)
    for name, value in (("campaign", campaign), ("launcher", launcher), ("plan", dict(cases={"NUMERIC": plan}))):
        atomic_json(tmp_path / (name + ".json"), value)
    cli = [
        sys.executable,
        "-O",
        str(__import__("pathlib").Path(__file__).resolve().parents[3] / "tools/p7x_graph_repair6.py"),
        "check-joint",
        "--kind",
        "numeric",
        "--artifact",
        str(path),
        "--case-id",
        "NUMERIC",
    ]
    for name in ("campaign", "launcher", "plan", "output"):
        cli += ["--" + name, str(tmp_path / (name + ".json"))]
    runner_plan = dict(publication_status="VERIFIED", cases={"HOST_ACCEPTANCE": dict(argv=cli, timeout_seconds=30)})
    result = run_case(runner_plan, "HOST_ACCEPTANCE", tmp_path / "runner")
    assert result["dependency_satisfied"], result
    assert json.loads((tmp_path / "output.json").read_text())["status"] == "PASS_DIAGNOSTIC_CONTINUOUS_REFERENCE"


def test_real_numeric_mismatch_stops_next_generation_and_runner(monkeypatch, tmp_path):
    from tools.p7x_graph_repair6_runner import run_case
    from tools.p7x_graph_repair6_submission import NumericMismatch

    probes = []
    original = ContinuousReference.__enter__

    def enter(self):
        probes.append(self)
        return original(self)

    monkeypatch.setattr(ContinuousReference, "__enter__", enter)
    with pytest.raises(NumericMismatch):
        produce(monkeypatch, tmp_path, fault="numeric")
    probe = probes[0]
    assert probe.failed and not probe.poisoned
    reader = CollectionReader(probe.output)
    assert len(reader.samples) == 2 and reader.execution == "EXECUTION_FAILED"
    assert reader.record(1)["status"] == "NUMERIC_MISMATCH"
    snapshot = probe.owner._draft_window_graph_characterization.recorder.snapshot()
    campaign, launcher, plan = joined(probe.output, probe.provenance, snapshot)
    campaign["status"] = "EXCEPTION"
    launcher["returncode"] = 3
    for name, value in (("campaign", campaign), ("launcher", launcher), ("plan", dict(cases={"NUMERIC": plan}))):
        atomic_json(tmp_path / (name + ".json"), value)
    from pathlib import Path

    cli = [
        sys.executable,
        "-O",
        str(Path(__file__).resolve().parents[3] / "tools/p7x_graph_repair6.py"),
        "check-joint",
        "--kind",
        "numeric",
        "--artifact",
        str(probe.output),
        "--case-id",
        "NUMERIC",
    ]
    for name in ("campaign", "launcher", "plan", "output"):
        cli += ["--" + name, str(tmp_path / (name + ".json"))]
    runner = dict(
        publication_status="VERIFIED",
        cases={"CHECK": dict(argv=cli, timeout_seconds=30, selection="READ_ONLY_ACCEPTANCE")},
    )
    result = run_case(runner, "CHECK", tmp_path / "runner")
    assert result["status"] == "NUMERIC_MISMATCH" and not result["dependency_satisfied"], result
    report = json.loads((tmp_path / "output.json").read_text())
    assert report["status"] == "NUMERIC_MISMATCH" and report["execution"]["status"] != "EXECUTION_ASSOCIATED"
