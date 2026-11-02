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


@pytest.mark.parametrize(
    "field",
    [None, "pipeline_parallel_size", "prefill_context_parallel_size", "decode_context_parallel_size", "cp_size"],
)
def test_graph_initialization_validates_actual_unsharded_context_once(monkeypatch, field):
    from vllm_ascend.worker.v2.spec_decode import draft_window_graph_runtime as runtime

    monkeypatch.setattr(runtime, "draft_window_graph_enabled", lambda: True)
    monkeypatch.setattr(
        runtime,
        "get_current_hardware_profile",
        lambda: SimpleNamespace(attention_backend_family=runtime.AttentionBackendFamily.STANDARD),
    )
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(name="FULL_DECODE_ONLY")),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=16,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
    )
    tables = SimpleNamespace(cp_size=1)
    if field == "cp_size":
        tables.cp_size = 2
    elif field is not None:
        setattr(config.parallel_config, field, 2)
    if field is None:
        runtime.validate_graph_context(config, tables)
    else:
        with pytest.raises(RuntimeError, match="PP=PCP=DCP=1"):
            runtime.validate_graph_context(config, tables)


@pytest.fixture
def bridge():
    from vllm_ascend.worker.v2.spec_decode.draft_window_read import DraftReadView

    device = torch.tensor([17, 1025, 0, 0], dtype=torch.int32)
    exact = device.clone()
    table = torch.arange(8).reshape(4, 2)
    read = DraftReadView(table, device, exact, exact, None, 1025, num_reqs=4)
    common = SimpleNamespace(num_reqs=4, slot_mapping=object(), positions=object())
    read.apply(common)
    return SimpleNamespace(read=read, common=common)


def test_current_exact_uses_the_common_backend_read_interface_without_a_second_certificate(bridge):
    slots, positions = bridge.common.slot_mapping, bridge.common.positions
    assert bridge.common._draft_window_read is bridge.read
    assert bridge.read.exact_cpu_for(bridge.common) is bridge.read.seq_lens_cpu
    assert bridge.common.seq_lens_cpu_upper_bound is bridge.read.seq_lens_cpu
    assert bridge.common.seq_lens is bridge.read.seq_lens
    assert bridge.common.slot_mapping is slots and bridge.common.positions is positions


@pytest.mark.parametrize(
    "field", ["seq_lens", "seq_lens_cpu", "_seq_lens_cpu", "seq_lens_cpu_upper_bound", "block_table_tensor", "num_reqs"]
)
def test_replaced_read_operand_is_rejected_at_the_actual_backend_read_boundary(bridge, field):
    value = getattr(bridge.common, field)
    setattr(bridge.common, field, value.clone() if isinstance(value, torch.Tensor) else 3)
    with pytest.raises(ValueError, match="replaced after projection"):
        bridge.read.exact_cpu_for(bridge.common)


def test_a_different_host_upper_cannot_be_used_as_the_actual_exact_length(bridge):
    bridge.common.seq_lens_cpu = torch.tensor([24, 1040, 0, 0], dtype=torch.int32)
    with pytest.raises(ValueError, match="replaced after projection"):
        bridge.read.exact_cpu_for(bridge.common)


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


@pytest.mark.parametrize("active", [False, True])
def test_actual_dispatch_keeps_only_the_framework_descriptor(monkeypatch, active):
    from vllm_ascend.worker.v2.spec_decode.dflash.aclgraph import DFlashAclGraphManager, DFlashCudaGraphManager

    descriptor = SimpleNamespace()
    monkeypatch.setattr(DFlashCudaGraphManager, "dispatch", lambda *args: descriptor)
    manager = object.__new__(DFlashAclGraphManager)
    manager.speculator = SimpleNamespace(
        _draft_window_epoch_source=SimpleNamespace(_active=object() if active else None)
    )
    assert manager.dispatch(1, 8, 8, 0) is descriptor
    assert manager.speculator._draft_window_selected_metadata is (descriptor if active else None)
    manager.speculator = SimpleNamespace()
    assert manager.dispatch(1, 8, 8, 0) is descriptor
    assert not hasattr(manager.speculator, "_draft_window_selected_metadata")


@pytest.fixture
def full_metadata_dispatch():
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

    from vllm_ascend.worker.v2.spec_decode.draft_window import DraftMetadataInputs
    from vllm_ascend.worker.v2.spec_decode.draft_window_graph import DraftWindowGraphBindingProvider
    from vllm_ascend.worker.v2.spec_decode.draft_window_graph_runtime import prepare_parallel_graph

    proposal = SimpleNamespace(mapping_epoch=7, ordered_request_ids=("req-a", "req-b"))
    identity: Any = SimpleNamespace(proposal=proposal, padded_request_count=2)
    provider: Any = object.__new__(DraftWindowGraphBindingProvider)
    provider._quarantined_identity = None
    provider._source = SimpleNamespace(policy=SimpleNamespace(normalized_method="dflash"))
    bank: Any = SimpleNamespace(assert_bound=lambda expected: expected is identity or pytest.fail("stale bank"))
    inputs = DraftMetadataInputs(identity, bank, provider)
    provider._active_inputs = inputs
    descriptor = BatchExecutionDescriptor(CUDAGraphMode.FULL, num_tokens=16, num_reqs=2, uniform_token_count=8)
    capture = SimpleNamespace(descriptor=descriptor)
    provider.capture_identity_for = Mock(return_value=capture)
    carrier = object()
    provider.bind_read_carrier = Mock(return_value=carrier)
    owner = SimpleNamespace(
        _draft_window_epoch_source=SimpleNamespace(_active=proposal),
        _draft_window_graph_binding_provider=provider,
        _draft_window_graph_session=provider,
        _draft_window_selected_metadata=descriptor,
        model_state=SimpleNamespace(attn_metadata={"target.layer": object()}),
    )
    return SimpleNamespace(
        owner=owner,
        inputs=inputs,
        descriptor=descriptor,
        provider=provider,
        carrier=carrier,
        prepare=prepare_parallel_graph,
    )


def test_actual_replay_uses_the_current_provider_bank_without_a_second_generation_ledger(full_metadata_dispatch):
    c = full_metadata_dispatch
    assert c.prepare(c.owner, c.descriptor) is c.carrier
    c.provider.bind_read_carrier.assert_called_once()
    assert c.provider.bind_read_carrier.call_args.kwargs["inputs"] is c.inputs
    assert not hasattr(c.owner, "_draft_window_live_inputs")
    assert not hasattr(c.owner, "_draft_window_deferred_metadata")


@pytest.mark.parametrize("damage", ["proposal", "padded_rows", "descriptor", "published", "retired"])
def test_actual_replay_rejects_stale_rows_bucket_or_already_published_generation(full_metadata_dispatch, damage):
    c = full_metadata_dispatch
    if damage == "proposal":
        c.owner._draft_window_epoch_source._active = object()
    elif damage == "padded_rows":
        c.inputs.identity.padded_request_count = 4
    elif damage == "descriptor":
        c.owner._draft_window_selected_metadata = object()
    elif damage == "published":
        object.__setattr__(c.inputs, "metadata", {"old": object()})
    else:
        c.provider._active_inputs = None
    with pytest.raises(RuntimeError, match="proposal|bucket|unconsumed|generation"):
        c.prepare(c.owner, c.descriptor)
    c.provider.bind_read_carrier.assert_not_called()
