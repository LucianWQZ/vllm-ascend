# SPDX-License-Identifier: Apache-2.0
"""Review counterexamples; native device work is not claimed."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager

from tests.ut.spec_decode.test_graph_repair5_contracts import load


def minimal_life():
    record = dict(
        provider=1,
        generation=0,
        epoch=99,
        backing={"fake": True},
        completion="DIAGNOSTIC_COMPLETE",
        retired=True,
        live_inputs_cleared=True,
        draft_tokens=[[1] * 7],
        before_replay_ns=1,
        replay_return_ns=2,
        update_return_ns=3,
        complete_ns=4,
        retirement_return_ns=5,
    )
    return dict(
        schema="repair5-real-mrv2-lifecycle-v1",
        quarantined=False,
        abort_at="none",
        identity=dict(run_id="r", case_id="LIFE", attempt_id="1"),
        records=[record],
    )


@pytest.mark.parametrize("rollback", [False, True])
def test_f1_incomplete_or_epoch_rollback_never_passes(rollback):
    document = minimal_life()
    if rollback:
        second = deepcopy(document["records"][0])
        second.update(
            generation=1,
            epoch=0,
            before_replay_ns=6,
            replay_return_ns=7,
            update_return_ns=8,
            complete_ns=9,
            retirement_return_ns=10,
        )
        document["records"].append(second)
    assert load("p7x_graph_repair5_lifecycle").check_lifecycle(document)["status"] == "INVALID_EVIDENCE"


def actual_candidates(width, sizes):
    # Execute pinned framework candidate construction and dispatch, not a
    # hand-written graphs dictionary. Native capture is deliberately absent.
    manager = CudaGraphManager.__new__(CudaGraphManager)
    manager.max_num_reqs = 4
    manager.decode_query_len = width
    manager.varlen_decode = False
    manager.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
    manager.compilation_config = SimpleNamespace(cudagraph_capture_sizes=sizes, max_cudagraph_capture_size=max(sizes))
    manager.vllm_config = SimpleNamespace(
        speculative_config=None, cache_config=SimpleNamespace(use_kda_recoverssm=False)
    )
    manager.lora_capture_cases = [0]
    manager._lora_dispatch_map = {}
    manager._candidates, manager._capture_descs = {}, {}
    manager._init_candidates()
    manager._graphs_captured = True  # permits dispatch; does not claim capture
    return manager


def test_real_managers_base_and_target4_inventory():
    for sizes, target in (([7, 14, 28], {(1, 8), (2, 16)}), ([7, 14, 28, 32], {(1, 8), (2, 16), (4, 32)})):
        draft = actual_candidates(7, sizes)
        target_manager = actual_candidates(8, sizes)
        inventory = lambda m: {(d.num_reqs, d.num_tokens) for d in m._capture_descs[CUDAGraphMode.FULL]}
        assert inventory(draft) == {(1, 7), (2, 14), (4, 28)}
        assert inventory(target_manager) == target
        desc = draft.dispatch(1, 7, 7, 0)
        assert (desc.num_reqs, desc.num_tokens) == (1, 7)
        desc = draft.dispatch(3, 21, 7, 0)
        assert (desc.num_reqs, desc.num_tokens) == (4, 28)


def test_f4_structural_pass_requires_external_joint(tmp_path):
    tool = load("p7x_graph_repair5")
    assert hasattr(tool, "load_tool")
    joint = load("p7x_graph_repair5_evidence")
    result = joint.audit_execution({}, {}, {}, {}, tmp_path / "missing")
    assert result["status"] == "INVALID_EVIDENCE"


def test_target_padding_does_not_authorize_draft_padding_real_builder(monkeypatch):
    from tests.ut.spec_decode import test_graph_repair4_integration as integration

    tool = load("p7x_graph_repair5_target")
    owner, session, manager, descriptors, _ = integration._lifetime_fixture(monkeypatch)
    integration._batch(owner, 1)
    draft_desc = actual_candidates(7, [7, 14, 28]).dispatch(1, 7, 7, 0)
    # A Target may deliberately execute its larger captured bucket. Draft dispatch
    # is independent, and production builder/recorder must use the smaller one.
    with owner._draft_window_proposal_scope(owner.input_batch, dummy_run=False, is_profile=False):
        integration._publish_in_scope(owner, descriptors[draft_desc.num_reqs])
        manager.run_fullgraph(descriptors[1])
        geometry = tool.draft_geometry(owner, session.recorder.snapshot(), 1)
        assert geometry["descriptor"] == [1, 7]
        assert geometry["padded"] == 1  # independent of Target padded=2
        assert session.recorder.snapshot()["events"][-1]["exact_lengths"] == [64]


def test_digest_chunking_preserves_every_unselected_byte():
    import hashlib

    import torch

    tool = load("p7x_graph_repair5_target")
    tensor = torch.arange(200 * 128, dtype=torch.float32).reshape(200, 128)
    selected = torch.tensor([0, 8, 9, 198])
    expected = hashlib.sha256()
    for i in range(200):
        if i not in selected.tolist():
            expected.update(tensor[i].view(torch.uint8).numpy().tobytes())
    assert tool.outside_digest({"kv": tensor}, {"kv": selected})["kv"] == {
        "sha256": expected.hexdigest(),
        "blocks": 196,
    }


@pytest.mark.parametrize("damage", ["pid", "sha", "argv", "hash", "timeout", "signal", "returncode"])
def test_external_association_rejects_forged_or_failed_execution(tmp_path, damage):
    import hashlib

    tool = load("p7x_graph_repair5_evidence")
    artifact = tmp_path / "controlled-target.pt"
    artifact.write_bytes(b"not loaded by execution-only checker")
    identity = dict(
        run_id="r",
        case_id="G4F_ON",
        attempt_id="1",
        source_sha="a" * 40,
        source_tree="b" * 40,
        driver_mode="B",
        process_id=1,
        engine_id=2,
    )
    argv = ["python", "driver"]
    expected = dict(
        acceptance_identity=identity,
        argv=argv,
        launch_argv=["launch", "--"] + argv,
        outputs=str(tmp_path),
        requests_sha256="requests",
    )
    launcher = dict(
        identity,
        argv=argv,
        pid=1,
        status="EXITED",
        returncode=0,
        signal=None,
        timeout=False,
        requests_sha256="requests",
        host_start_ns=1,
        host_end_ns=100,
    )
    campaign = dict(
        identity=identity,
        startup_complete=True,
        status="OK",
        request_manifest={"sha256": "requests"},
        diagnostic_artifacts={
            artifact.name: dict(sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(), bytes=artifact.stat().st_size)
        },
    )
    document = dict(identity=identity)
    assert tool.audit_execution(document, campaign, launcher, expected, artifact)["status"] == "EXECUTION_ASSOCIATED"
    if damage == "hash":
        artifact.write_bytes(b"changed")
    elif damage == "sha":
        launcher["source_sha"] = "c" * 40
    else:
        launcher[damage] = {"pid": 3, "argv": [], "timeout": True, "signal": 9, "returncode": 1}[damage]
    assert tool.audit_execution(document, campaign, launcher, expected, artifact)["status"] != "EXECUTION_ASSOCIATED"
