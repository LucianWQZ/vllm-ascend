# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU Graph metadata and source-owner contracts; no NPU execution.

These isolated CPU protocol tests establish exact-source/owner rejection;
model, numerical, capture and lifetime evidence remains separate.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

SOURCE_ROOT = Path(__file__).resolve().parents[3]
SOURCE = SOURCE_ROOT / "vllm_ascend/worker/v2/spec_decode/draft_window_eager.py"


@pytest.fixture(params=["dspark", "dflash", "dflash2", "eagle3"])
def graph_source_initialization(request, monkeypatch):
    """Exercise actual owner initialization without loading a device model."""
    method = request.param
    from vllm_ascend.worker.v2.spec_decode import draft_window_graph_runtime
    from vllm_ascend.worker.v2.spec_decode.draft_window_eager import DraftWindowGraphInputMixin

    source, provider = object(), object()
    build_source, build_provider = Mock(return_value=source), Mock(return_value=provider)
    monkeypatch.setattr(draft_window_graph_runtime, "build_parallel_graph_source", build_source)
    if method == "eagle3":
        from vllm_ascend.worker.v2.spec_decode import draft_window_eagle3_adapter, draft_window_eagle3_capture

        initialize = draft_window_eagle3_adapter.Eagle3GraphIntegrationMixin._initialize_draft_window_source_capability
        monkeypatch.setattr(draft_window_eagle3_capture, "build_eagle3_graph_source", build_source)
        monkeypatch.setattr(draft_window_eagle3_adapter, "DraftWindowGraphBindingProvider", build_provider)
    else:
        from vllm_ascend.worker.v2.spec_decode import draft_window_eager

        initialize = DraftWindowGraphInputMixin._initialize_draft_window_source_capability
        monkeypatch.setattr(draft_window_eager, "DraftWindowGraphBindingProvider", build_provider)
    owner = SimpleNamespace(
        _draft_window_policy=object(),
        _speculator_name=method,
        attn_groups=object(),
        block_tables=object(),
        kv_cache_config=object(),
        _group_causal=False,
        vllm_config=object(),
        _initialize_draft_window_graph_inputs=Mock(),
        _initialize_eagle3_draft_window_state=Mock(),
    )
    return SimpleNamespace(
        owner=owner,
        initialize=initialize,
        build_source=build_source,
        build_provider=build_provider,
        source=source,
        provider=provider,
    )


def test_graph_owner_installs_only_its_admitted_source(graph_source_initialization):
    c = graph_source_initialization
    c.initialize(c.owner)
    c.build_source.assert_called_once()
    c.build_provider.assert_called_once_with(source=c.source)
    assert c.owner._draft_window_graph_binding_provider is c.provider
    assert not hasattr(c.owner, "_draft_window_source_capability")
    assert not hasattr(c.owner, "_draft_window_runtime_state")
    assert not hasattr(c.owner, "_draft_window_runtime")


def test_graph_source_rejection_has_no_eager_fallback_or_partial_owner(graph_source_initialization):
    c = graph_source_initialization
    failure = NotImplementedError("unsupported actual Graph source")
    c.build_source.side_effect = failure
    with pytest.raises(NotImplementedError) as seen:
        c.initialize(c.owner)
    assert seen.value is failure
    c.build_provider.assert_not_called()
    assert not hasattr(c.owner, "_draft_window_graph_binding_provider")


def test_graph_owner_reinitialization_rejects_before_source_build(graph_source_initialization):
    c = graph_source_initialization
    c.owner._draft_window_graph_binding_provider = c.provider
    with pytest.raises(RuntimeError, match="already initialized"):
        c.initialize(c.owner)
    c.build_source.assert_not_called()


def test_window_off_without_graph_policy_does_not_construct_source(graph_source_initialization):
    c = graph_source_initialization
    del c.owner._draft_window_policy
    c.initialize(c.owner)
    c.build_source.assert_not_called()
    c.build_provider.assert_not_called()
    assert not hasattr(c.owner, "_draft_window_graph_binding_provider")


@pytest.fixture
def source_contract():
    from vllm_ascend.worker.v2.spec_decode.draft_window_graph import DraftWindowGraphBindingProvider

    parallel = SimpleNamespace(tensor_parallel_size=16)
    capability = SimpleNamespace(parallel_config=parallel, tensor_parallel_rank=7)
    owner: Any = object.__new__(DraftWindowGraphBindingProvider)
    owner._source = capability
    owner._tp_identity = (16, 7)
    return owner, DraftWindowGraphBindingProvider


@pytest.mark.parametrize(
    "field,value",
    [
        (None, None),
        ("parallel_config", None),
        ("tensor_parallel_size", 0),
        ("tensor_parallel_size", True),
        ("tensor_parallel_rank", None),
        ("tensor_parallel_rank", True),
        ("tensor_parallel_rank", -1),
        ("tensor_parallel_rank", 16),
    ],
)
def test_graph_capture_requires_real_explicit_world_and_rank(source_contract, field, value):
    owner, methods = source_contract
    if field == "parallel_config":
        owner._source.parallel_config = value
    elif field == "tensor_parallel_size":
        owner._source.parallel_config.tensor_parallel_size = value
    elif field is not None:
        owner._source.tensor_parallel_rank = value
    if field is None:
        assert methods._capture_topology(owner) == (16, 7)
    else:
        with pytest.raises(RuntimeError, match="parallel configuration|world/rank"):
            methods._capture_topology(owner)


@pytest.mark.parametrize("replacement", [None, "rank-drift"])
def test_graph_source_guard_keeps_actual_capability_parallel_object_and_rank(source_contract, replacement):
    owner, methods = source_contract
    if replacement == "rank-drift":
        owner._source.tensor_parallel_rank = 8
    if replacement is None:
        methods._validate_source_contract(owner)
    else:
        with pytest.raises(RuntimeError, match="parallel configuration or TP rank changed"):
            methods._validate_source_contract(owner)


@pytest.fixture
def bridge():
    from vllm_ascend.worker.v2.spec_decode.draft_window_eager import AscendDraftAttentionReadInputs

    device = torch.tensor([17, 1025, 0, 0], dtype=torch.int32)
    exact = device.clone()
    upper = torch.tensor([24, 1040, 0, 0], dtype=torch.int32)
    tables = (torch.arange(8).reshape(4, 2), torch.arange(8, 16).reshape(4, 2))
    read = AscendDraftAttentionReadInputs(tables, device, exact, upper, 1040)
    common = SimpleNamespace(
        num_reqs=4,
        max_seq_len=1040,
        seq_lens=device.view(-1),
        seq_lens_cpu=exact,
        seq_lens_cpu_upper_bound=upper,
        block_table_tensor=read.block_tables[0],
        slot_mapping=object(),
        positions=object(),
    )
    return SimpleNamespace(read=read, common=common)


def test_current_exact_pair_is_certified_without_reprojection_or_write_mutation(bridge):
    slots, positions = bridge.common.slot_mapping, bridge.common.positions
    bridge.read.certify_common_metadata(0, bridge.common)
    certificate = bridge.common._draft_window_read
    assert certificate.exact_cpu_for(bridge.common) is bridge.read.seq_lens_cpu_exact
    assert certificate.exact_cpu_for(bridge.common) is not bridge.read.seq_lens_cpu_upper_bound
    assert bridge.common.seq_lens is bridge.read.seq_lens
    assert bridge.common.slot_mapping is slots and bridge.common.positions is positions


@pytest.mark.parametrize("field", ["seq_lens", "seq_lens_cpu", "seq_lens_cpu_upper_bound", "block_table_tensor"])
def test_same_values_from_another_owner_are_rejected_at_actual_handoff(bridge, field):
    setattr(bridge.common, field, getattr(bridge.common, field).clone())
    with pytest.raises(ValueError, match="lost its current read owners"):
        bridge.read.certify_common_metadata(0, bridge.common)


def test_upper_cannot_be_substituted_for_exact_at_actual_handoff(bridge):
    bridge.common.seq_lens_cpu = bridge.read.seq_lens_cpu_upper_bound
    with pytest.raises(ValueError, match="lost its current read owners"):
        bridge.read.certify_common_metadata(0, bridge.common)


@pytest.mark.parametrize("field,value", [("num_reqs", 3), ("max_seq_len", 1025)])
def test_wrong_actual_handoff_geometry_is_rejected(bridge, field, value):
    setattr(bridge.common, field, value)
    with pytest.raises(ValueError, match="lost its current read owners"):
        bridge.read.certify_common_metadata(0, bridge.common)


@pytest.mark.parametrize(
    "field", ["seq_lens", "seq_lens_cpu", "_seq_lens_cpu", "seq_lens_cpu_upper_bound", "block_table_tensor", "num_reqs"]
)
def test_replaced_read_operand_invalidates_certificate(bridge, field):
    bridge.read.certify_common_metadata(0, bridge.common)
    certificate = bridge.common._draft_window_read
    setattr(bridge.common, field, object())
    with pytest.raises(ValueError, match="replaced after projection"):
        certificate.exact_cpu_for(bridge.common)


def test_existing_projection_owner_is_not_overwritten(bridge):
    bridge.common._draft_window_device_read = object()
    with pytest.raises(ValueError, match="already has a read projection owner"):
        bridge.read.certify_common_metadata(0, bridge.common)


@pytest.mark.parametrize("actual,padded", [(1, 2), (3, 4)])
def test_private_full_uses_one_fresh_builder_and_only_binds_its_device_lengths(monkeypatch, actual, padded):
    from vllm_ascend.worker.v2.spec_decode import draft_window_graph_runtime as runtime
    from vllm_ascend.worker.v2.spec_decode.draft_window_graph import DraftWindowGraphRoute

    exact = torch.tensor([17 + i * 1024 for i in range(actual)] + [0] * (padded - actual), dtype=torch.int32)
    read: Any = SimpleNamespace(
        max_num_reqs=padded,
        window_block_table=torch.arange(padded * 2, dtype=torch.int32).reshape(padded, 2),
        cpu_window_exact=exact,
        device_window_exact=exact.clone(),
    )
    group = SimpleNamespace(gid=0, layers=("draft.layer.a", "draft.layer.b"))
    owner = SimpleNamespace(
        _draft_window_graph_binding_provider=SimpleNamespace(source=SimpleNamespace(group=group)),
        block_tables=SimpleNamespace(input_block_tables=[torch.full_like(read.window_block_table, 99)]),
        _group_causal=False,
    )
    slots, query, state = object(), object(), object()
    fresh = SimpleNamespace(
        block_tables=read.window_block_table,
        seq_lens=read.cpu_window_exact,
        seq_lens_cpu=read.cpu_window_exact,
        seq_lens_gpu=read.device_window_exact,
        seq_lens_list=read.cpu_window_exact.tolist(),
        actual_seq_lengths_q=[(i + 1) * 8 for i in range(padded)],
        slot_mapping=slots,
        query_start_loc=query,
        attn_state=state,
        num_actual_tokens=padded * 8,
        causal=False,
    )
    calls = []

    def build(actual_owner, **kwargs):
        assert actual_owner is owner
        calls.append(kwargs)
        return {name: fresh for name in group.layers}

    monkeypatch.setattr(runtime, "build_draft_window_attn_metadata", build)
    capture: Any = SimpleNamespace(
        route=DraftWindowGraphRoute.DFLASH,
        descriptor=SimpleNamespace(num_reqs=padded, num_tokens=padded * 8),
    )
    result = runtime._build_private_graph_metadata(
        owner,
        workspace=read,
        actual_request_count=actual,
        capture_identity=capture,
    )
    assert len(calls) == 1
    assert calls[0]["num_reqs"] == calls[0]["num_reqs_padded"] == padded
    assert calls[0]["num_tokens_padded"] == padded * 8
    assert result["draft.layer.a"] is result["draft.layer.b"] is fresh
    assert fresh.seq_lens is read.device_window_exact
    assert fresh.seq_lens_cpu is read.cpu_window_exact
    assert fresh.slot_mapping is slots and fresh.query_start_loc is query
    assert fresh.attn_state is state and fresh.num_actual_tokens == padded * 8 and fresh.causal is False
    assert fresh.actual_seq_lengths_q == [(i + 1) * 8 for i in range(padded)]
    runtime.DraftWindowGraphBindingProvider._validate_metadata_uses_read_carrier(result, read)


@pytest.mark.parametrize(
    "relative,method_name",
    [
        ("dflash/aclgraph.py", "_updatable_graph_replay"),
        ("target_dependent_ar/aclgraph.py", "_run_eagle_window_fullgraph"),
    ],
)
def test_static_capture_check_precedes_replay_and_successful_update_marks_banks(relative, method_name):
    tree = ast.parse((SOURCE.parent / relative).read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == method_name)
    calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)]
    validate = next(
        node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "validate_graph"
    )
    replay = next(node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "run_fullgraph")
    update = next(node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "update")
    finish = next(node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "finish_update")
    assert validate.lineno < replay.lineno < update.lineno < finish.lineno
    assert all(not (isinstance(node.func, ast.Attribute) and node.func.attr == "validate_update") for node in calls)


def test_window_off_update_keeps_the_original_two_argument_call():
    for relative in ["dflash/aclgraph.py", "target_dependent_ar/aclgraph.py"]:
        tree = ast.parse((SOURCE.parent / relative).read_text())
        method = next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_updatable_graph_replay"
        )
        if relative.startswith("dflash"):
            branch = next(
                n for n in method.body if isinstance(n, ast.If) and ast.unparse(n.test) == "graph_route is None"
            )
            actual = branch.body
        else:
            branch = next(n for n in method.body if isinstance(n, ast.If) and ast.unparse(n.test) == "dp_dummy")
            # The actual idle-DP branch needs the pinned manager config while
            # preserving the same UPDATE/event API. Business OFF remains the
            # original two-argument call without a Window binding.
            actual = branch.orelse
            dummy_update = branch.body[0]
            assert isinstance(dummy_update, ast.With)
            assert ast.unparse(dummy_update.items[0].context_expr) == "set_current_vllm_config(self.vllm_config)"
            assert ast.dump(ast.Module(body=dummy_update.body, type_ignores=[])) == ast.dump(
                ast.Module(body=ast.parse("graph.update(self.update_stream, resolved_tasks)").body, type_ignores=[])
            )
        expected = ast.parse("graph.update(self.update_stream, resolved_tasks)").body
        assert ast.dump(ast.Module(body=actual, type_ignores=[])) == ast.dump(
            ast.Module(body=expected, type_ignores=[])
        )


@pytest.fixture
def full_metadata_dispatch():
    """Exercise the actual FULL-selection functions and provider type guard."""
    from vllm_ascend.worker.v2.spec_decode.draft_window_graph import DraftWindowGraphBindingProvider
    from vllm_ascend.worker.v2.spec_decode.draft_window_graph_metadata import (
        defer_parallel_graph_metadata,
        select_parallel_graph_metadata,
        validate_deferred_parallel_graph_metadata,
    )

    proposal = SimpleNamespace(mapping_epoch=7, ordered_request_ids=("req-a", "req-b"))
    identity = SimpleNamespace(proposal=proposal, padded_request_count=2)
    inputs = SimpleNamespace(identity=identity, metadata=None)
    inputs.validate_for = lambda expected: None if expected is identity else (_ for _ in ()).throw(ValueError("stale"))
    provider: Any = object.__new__(DraftWindowGraphBindingProvider)
    provider._active_inputs = inputs
    provider._source = SimpleNamespace(policy=SimpleNamespace(normalized_method="dflash"))
    owner = SimpleNamespace(
        _draft_window_proposal_context=proposal,
        _draft_window_graph_binding_provider=provider,
        num_query_per_req=8,
        _draft_window_live_inputs=inputs,
        _draft_window_deferred_metadata=None,
    )
    descriptor = SimpleNamespace(cg_mode=SimpleNamespace(name="FULL"), num_reqs=2, num_tokens=16)
    return SimpleNamespace(
        owner=owner,
        inputs=inputs,
        descriptor=descriptor,
        select_parallel_graph_metadata=select_parallel_graph_metadata,
        defer_parallel_graph_metadata=defer_parallel_graph_metadata,
        validate_deferred_parallel_graph_metadata=validate_deferred_parallel_graph_metadata,
    )


def test_full_metadata_dispatch_defers_once_for_current_generation(full_metadata_dispatch):
    c = full_metadata_dispatch
    c.select_parallel_graph_metadata(c.owner, c.descriptor)
    selected = c.owner._draft_window_selected_metadata
    assert selected.descriptor is c.descriptor
    assert selected.proposal is c.inputs.identity.proposal
    assert selected.descriptor.num_reqs == 2 and selected.descriptor.num_tokens == 16
    assert c.defer_parallel_graph_metadata(c.owner, c.inputs, padded_requests=2, padded_tokens=16)
    c.validate_deferred_parallel_graph_metadata(c.owner, c.inputs, c.descriptor)
    assert c.owner._draft_window_deferred_metadata is c.inputs.identity
    with pytest.raises(RuntimeError, match="existing publication"):
        c.defer_parallel_graph_metadata(c.owner, c.inputs, padded_requests=2, padded_tokens=16)


@pytest.mark.parametrize(
    "field,value",
    [
        ("proposal", SimpleNamespace(mapping_epoch=8, ordered_request_ids=("req-a", "req-b"))),
        ("proposal", SimpleNamespace(mapping_epoch=7, ordered_request_ids=("req-b", "req-a"))),
        ("proposal", SimpleNamespace(mapping_epoch=7, ordered_request_ids=("req-a", "req-b"))),
        ("padded_request_count", 4),
    ],
)
def test_full_metadata_dispatch_rejects_stale_generation_or_rows(full_metadata_dispatch, field, value):
    c = full_metadata_dispatch
    c.select_parallel_graph_metadata(c.owner, c.descriptor)
    setattr(c.inputs.identity, field, value)
    with pytest.raises(RuntimeError, match="stale or wrong-bucket"):
        c.defer_parallel_graph_metadata(c.owner, c.inputs, padded_requests=2, padded_tokens=16)
    assert c.owner._draft_window_deferred_metadata is None


@pytest.mark.parametrize("tokens,requests", [(8, 2), (16, 4)])
def test_full_metadata_dispatch_rejects_wrong_actual_bucket(full_metadata_dispatch, tokens, requests):
    c = full_metadata_dispatch
    c.select_parallel_graph_metadata(c.owner, c.descriptor)
    with pytest.raises(RuntimeError, match="stale or wrong-bucket"):
        c.defer_parallel_graph_metadata(c.owner, c.inputs, padded_requests=requests, padded_tokens=tokens)


def test_eager_dispatch_keeps_ordinary_metadata_build(full_metadata_dispatch):
    c = full_metadata_dispatch
    c.descriptor.cg_mode.name = "NONE"
    c.select_parallel_graph_metadata(c.owner, c.descriptor)
    assert c.owner._draft_window_selected_metadata is None
    assert not c.defer_parallel_graph_metadata(c.owner, c.inputs, padded_requests=2, padded_tokens=16)


def test_full_metadata_dispatch_does_not_replace_published_metadata(full_metadata_dispatch):
    c = full_metadata_dispatch
    c.select_parallel_graph_metadata(c.owner, c.descriptor)
    c.inputs.metadata = object()
    with pytest.raises(RuntimeError, match="existing publication"):
        c.defer_parallel_graph_metadata(c.owner, c.inputs, padded_requests=2, padded_tokens=16)
