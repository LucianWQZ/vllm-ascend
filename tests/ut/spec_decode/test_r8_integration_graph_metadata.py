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

    capability, context, state = object(), object(), object()
    build_source = Mock(return_value=(capability, context))
    build_state = Mock(return_value=state)
    monkeypatch.setattr(draft_window_graph_runtime, "build_parallel_graph_source", build_source)
    if method == "eagle3":
        from vllm_ascend.worker.v2.spec_decode import draft_window_eagle3_adapter, draft_window_eagle3_capture

        initialize = draft_window_eagle3_adapter.Eagle3GraphIntegrationMixin._initialize_draft_window_source_capability
        monkeypatch.setattr(draft_window_eagle3_capture, "build_eagle3_graph_source", build_source)
        monkeypatch.setattr(draft_window_eagle3_adapter, "build_draft_window_runtime_state", build_state)
    else:
        from vllm_ascend.worker.v2.spec_decode import draft_window_eager

        initialize = DraftWindowGraphInputMixin._initialize_draft_window_source_capability
        monkeypatch.setattr(draft_window_eager, "build_draft_window_runtime_state", build_state)
    owner = SimpleNamespace(
        _draft_window_policy=object(),
        _speculator_name=method,
        attn_groups=object(),
        block_tables=object(),
        kv_cache_config=object(),
        _group_causal=False,
        vllm_config=object(),
        input_buffers=object(),
        _initialize_draft_window_graph_inputs=Mock(),
        _initialize_eagle3_draft_window_state=Mock(),
    )
    return SimpleNamespace(
        owner=owner,
        initialize=initialize,
        build_source=build_source,
        build_state=build_state,
        capability=capability,
        context=context,
        state=state,
        context_attr="_eagle3_graph_runtime_context" if method == "eagle3" else "_draft_window_graph_context",
    )


def test_graph_owner_installs_only_its_admitted_source(graph_source_initialization):
    c = graph_source_initialization
    c.initialize(c.owner)
    c.build_source.assert_called_once()
    c.build_state.assert_called_once_with(c.capability)
    assert c.owner._draft_window_source_capability is c.capability
    assert c.owner._draft_window_runtime_state is c.state
    assert getattr(c.owner, c.context_attr) is c.context
    assert not hasattr(c.owner, "_draft_window_runtime")


def test_graph_source_rejection_has_no_eager_fallback_or_partial_owner(graph_source_initialization):
    c = graph_source_initialization
    failure = NotImplementedError("unsupported actual Graph source")
    c.build_source.side_effect = failure
    with pytest.raises(NotImplementedError) as seen:
        c.initialize(c.owner)
    assert seen.value is failure
    c.build_state.assert_not_called()
    assert not hasattr(c.owner, "_draft_window_source_capability")
    assert not hasattr(c.owner, "_draft_window_runtime_state")


def test_graph_owner_reinitialization_rejects_before_source_build(graph_source_initialization):
    c = graph_source_initialization
    c.owner._draft_window_runtime_state = c.state
    with pytest.raises(RuntimeError, match="already initialized"):
        c.initialize(c.owner)
    c.build_source.assert_not_called()


def test_window_off_without_graph_policy_does_not_construct_source(graph_source_initialization):
    c = graph_source_initialization
    del c.owner._draft_window_policy
    c.initialize(c.owner)
    c.build_source.assert_not_called()
    assert not hasattr(c.owner, "_draft_window_runtime_state")


@pytest.fixture
def source_contract():
    from vllm_ascend.device.hardware_profile import AttentionBackendFamily
    from vllm_ascend.worker.v2.spec_decode.draft_window import DraftWindowRuntimeContext
    from vllm_ascend.worker.v2.spec_decode.draft_window_graph import DraftWindowGraphBindingProvider

    context = DraftWindowRuntimeContext(
        AttentionBackendFamily.STANDARD,
        "FULL",
        True,
        16,
        1,
        1,
        1,
        1,
        True,
        True,
        True,
        True,
        tensor_parallel_rank=7,
    )
    capability = SimpleNamespace(runtime_context=context)
    owner = SimpleNamespace(
        _runtime_state=SimpleNamespace(capability=capability), _capability=capability, _topology=context
    )
    return owner, DraftWindowGraphBindingProvider


@pytest.mark.parametrize(
    "field,value",
    [
        (None, None),
        ("runtime_context", None),
        ("tensor_parallel_size", 0),
        ("tensor_parallel_size", True),
        ("tensor_parallel_rank", None),
        ("tensor_parallel_rank", True),
        ("tensor_parallel_rank", -1),
        ("tensor_parallel_rank", 16),
    ],
)
def test_graph_capture_requires_real_explicit_world_and_rank(source_contract, field, value):
    from dataclasses import replace

    owner, methods = source_contract
    if field == "runtime_context":
        owner._capability.runtime_context = value
    elif field is not None:
        owner._capability.runtime_context = replace(owner._topology, **{field: value})
    if field is None:
        assert methods._capture_topology(owner) == (16, 7)
    else:
        with pytest.raises(RuntimeError, match="source topology|world/rank"):
            methods._capture_topology(owner)


@pytest.mark.parametrize("replacement", [None, "capability", "context", "rank-drift"])
def test_graph_source_guard_keeps_actual_capability_and_context_identity(source_contract, replacement):
    from dataclasses import replace

    owner, methods = source_contract
    if replacement == "capability":
        owner._runtime_state.capability = SimpleNamespace(**vars(owner._capability))
    elif replacement == "context":
        owner._capability.runtime_context = replace(owner._topology)
    elif replacement == "rank-drift":
        owner._capability.runtime_context = replace(owner._topology, tensor_parallel_rank=8)
    if replacement is None:
        methods._validate_source_contract(owner)
    else:
        with pytest.raises(RuntimeError, match="capability changed|topology changed"):
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
        seq_lens=device.view(-1),
        seq_lens_cpu=exact,
        seq_lens_cpu_upper_bound=upper,
        block_table_tensor=read.block_tables[0],
        slot_mapping=object(),
        positions=object(),
    )
    kwargs = dict(
        num_reqs=4,
        block_tables=list(read.block_tables),
        seq_lens=device,
        seq_lens_cpu=exact,
        seq_lens_cpu_upper_bound=upper,
        max_seq_len=1040,
    )
    return SimpleNamespace(read=read, common=common, kwargs=kwargs)


def test_current_exact_pair_is_certified_without_reprojection_or_write_mutation(bridge):
    bridge.read.validate_builder_sources(**bridge.kwargs)
    slots, positions = bridge.common.slot_mapping, bridge.common.positions
    bridge.read.certify_common_metadata(0, bridge.common)
    certificate = bridge.common._draft_window_read
    assert certificate.exact_cpu_for(bridge.common) is bridge.read.seq_lens_cpu_exact
    assert certificate.exact_cpu_for(bridge.common) is not bridge.read.seq_lens_cpu_upper_bound
    assert bridge.common.seq_lens is bridge.read.seq_lens
    assert bridge.common.slot_mapping is slots and bridge.common.positions is positions


@pytest.mark.parametrize("field", ["seq_lens", "seq_lens_cpu", "seq_lens_cpu_upper_bound"])
def test_same_values_from_another_owner_are_rejected(bridge, field):
    original = bridge.kwargs[field]
    bridge.kwargs[field] = original.clone()
    with pytest.raises(ValueError, match="owners differ"):
        bridge.read.validate_builder_sources(**bridge.kwargs)


def test_upper_cannot_be_substituted_for_exact(bridge):
    bridge.kwargs["seq_lens_cpu"] = bridge.read.seq_lens_cpu_upper_bound
    with pytest.raises(ValueError, match="owners differ"):
        bridge.read.validate_builder_sources(**bridge.kwargs)


def test_wrong_group_table_and_geometry_are_rejected(bridge):
    bridge.kwargs["block_tables"] = list(reversed(bridge.read.block_tables))
    with pytest.raises(ValueError, match="tables differ"):
        bridge.read.validate_builder_sources(**bridge.kwargs)
    bridge.kwargs["block_tables"] = bridge.read.block_tables
    bridge.kwargs["num_reqs"] = 3
    with pytest.raises(ValueError, match="geometry differs"):
        bridge.read.validate_builder_sources(**bridge.kwargs)


@pytest.mark.parametrize(
    "field", ["seq_lens", "seq_lens_cpu", "_seq_lens_cpu", "seq_lens_cpu_upper_bound", "block_table_tensor"]
)
def test_replaced_read_operand_invalidates_certificate(bridge, field):
    bridge.read.certify_common_metadata(0, bridge.common)
    certificate = bridge.common._draft_window_read
    setattr(bridge.common, field, object())
    with pytest.raises(ValueError, match="replaced after exact certification"):
        certificate.exact_cpu_for(bridge.common)


def test_existing_projection_owner_is_not_overwritten(bridge):
    bridge.common._draft_window_device_read = object()
    with pytest.raises(ValueError, match="already has a read projection owner"):
        bridge.read.certify_common_metadata(0, bridge.common)


@pytest.fixture
def graph_device_binding():
    from vllm_ascend.worker.v2.spec_decode.draft_window_graph import DraftWindowGraphBindingProvider
    from vllm_ascend.worker.v2.spec_decode.draft_window_graph_runtime import _bind_parallel_graph_device_lengths

    def make(actual, padded):
        assert (actual, padded) in ((1, 2), (3, 4))
        values = [17 + i * 1024 for i in range(actual)] + [0] * (padded - actual)
        table = torch.arange(padded * 2, dtype=torch.int32).reshape(padded, 2)
        exact = torch.tensor(values, dtype=torch.int32)
        device = exact.clone()
        read = SimpleNamespace(
            max_num_reqs=padded,
            window_block_table=table,
            cpu_window_exact=exact,
            device_window_exact=device,
        )
        slots, positions = object(), object()
        metadata = SimpleNamespace(
            block_tables=table,
            seq_lens=exact,
            seq_lens_cpu=exact,
            seq_lens_gpu=device,
            seq_lens_list=exact.tolist(),
            actual_seq_lengths_q=[(i + 1) * 8 for i in range(padded)],
            slot_mapping=slots,
            positions=positions,
        )
        return read, metadata

    return SimpleNamespace(
        make=make,
        bind=_bind_parallel_graph_device_lengths,
        guard=DraftWindowGraphBindingProvider._validate_metadata_uses_read_carrier,
    )


@pytest.mark.parametrize("actual,padded", [(1, 2), (3, 4)])
def test_private_full_binds_device_after_old_cpu_operand_is_rejected(
    graph_device_binding,
    actual,
    padded,
):
    read, metadata = graph_device_binding.make(actual, padded)
    slots, positions = metadata.slot_mapping, metadata.positions
    with pytest.raises(RuntimeError, match="carrier device lengths"):
        graph_device_binding.guard({"draft.layer": metadata}, read)
    graph_device_binding.bind(metadata, read, 8)
    # Use the actual unchanged provider guard, not a successful helper self-report.
    graph_device_binding.guard({"draft.layer": metadata}, read)
    assert metadata.seq_lens is read.device_window_exact
    assert metadata.seq_lens_gpu is read.device_window_exact
    assert metadata.seq_lens_cpu is read.cpu_window_exact
    assert metadata.seq_lens_list == read.cpu_window_exact.tolist()
    assert metadata.slot_mapping is slots and metadata.positions is positions
    assert read.max_num_reqs == padded


@pytest.mark.parametrize("field", ["seq_lens", "seq_lens_cpu", "seq_lens_gpu", "block_tables"])
def test_private_full_rejects_same_values_with_wrong_builder_owner(graph_device_binding, field):
    read, metadata = graph_device_binding.make(3, 4)
    original = getattr(metadata, field)
    setattr(metadata, field, original.clone())
    with pytest.raises(RuntimeError, match="lost its current " + field + " owner"):
        graph_device_binding.bind(metadata, read, 8)
    assert metadata.seq_lens is not read.device_window_exact


@pytest.mark.parametrize("query", [[8, 16, 24, 24, 32], [8, 16, 24, 31]])
def test_private_full_rejects_clamped_or_wrong_padded_query_layout(graph_device_binding, query):
    read, metadata = graph_device_binding.make(3, 4)
    metadata.actual_seq_lengths_q = query
    with pytest.raises(RuntimeError, match="captured padded query layout"):
        graph_device_binding.bind(metadata, read, 8)


def test_private_full_rejects_stale_exact_list_despite_correct_header_owners(graph_device_binding):
    read, metadata = graph_device_binding.make(3, 4)
    metadata.seq_lens_list = [17, 1041, 2065, 1]
    with pytest.raises(RuntimeError, match="current exact length values"):
        graph_device_binding.bind(metadata, read, 8)


def test_private_full_uses_padded_builder_then_isolates_before_binding():
    runtime = ast.parse(SOURCE.with_name("draft_window_graph_runtime.py").read_text())
    private = next(
        n for n in runtime.body if isinstance(n, ast.FunctionDef) and n.name == "_build_private_graph_metadata"
    )
    calls = [
        n
        for n in ast.walk(private)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "build_draft_window_attn_metadata"
    ]
    assert len(calls) == 1
    passed = {kw.arg: ast.unparse(kw.value) for kw in calls[0].keywords}
    assert passed["num_reqs"] == passed["num_reqs_padded"] == "workspace.max_num_reqs"
    private_statements = [ast.unparse(n) for n in private.body]
    isolation = next(
        i for i, text in enumerate(private_statements) if "copy_draft_attention_metadata(metadata)" in text
    )
    binding = next(i for i, text in enumerate(private_statements) if "_bind_parallel_graph_device_lengths(" in text)
    assert isolation < binding


def _run_actual_private_binding_loop(graph_device_binding, read, metadata_by_layer):
    """Execute only the actual caller loop, using the existing strict binder."""
    source = SOURCE.with_name("draft_window_graph_runtime.py")
    tree = ast.parse(source.read_text())
    private = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_build_private_graph_metadata")
    loop = next(
        n for n in private.body if isinstance(n, ast.For) and "_bind_parallel_graph_device_lengths(" in ast.unparse(n)
    )
    initializers = [
        n
        for n in private.body
        if isinstance(n, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(t, ast.Name) and t.id == "bound_metadata_ids"
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target])
        )
    ]
    calls = []

    def strict_bind(metadata, current_read, query_width):
        calls.append(metadata)
        return graph_device_binding.bind(metadata, current_read, query_width)

    namespace = dict(
        window_layers=tuple(metadata_by_layer),
        metadata=metadata_by_layer,
        workspace=read,
        capture_identity=SimpleNamespace(
            descriptor=SimpleNamespace(num_tokens=read.max_num_reqs * 8), num_reqs=read.max_num_reqs
        ),
        _bind_parallel_graph_device_lengths=strict_bind,
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[*initializers, loop], type_ignores=[])), str(source), "exec"
        ),
        namespace,
    )
    return calls


def test_private_full_shared_group_metadata_binds_once(graph_device_binding):
    read, metadata = graph_device_binding.make(3, 4)
    slots, positions = metadata.slot_mapping, metadata.positions
    # Ascend's real group builder maps one metadata object to every layer;
    # Target isolation memoization preserves that legitimate shared object.
    by_layer = {f"draft.layer.{i}": metadata for i in range(5)}
    calls = _run_actual_private_binding_loop(graph_device_binding, read, by_layer)
    assert calls == [metadata]
    graph_device_binding.guard(by_layer, read)
    assert metadata.seq_lens is read.device_window_exact
    assert metadata.seq_lens_cpu is read.cpu_window_exact
    assert metadata.slot_mapping is slots and metadata.positions is positions


def test_private_full_distinct_bad_owner_after_shared_group_is_rejected(graph_device_binding):
    read, metadata = graph_device_binding.make(3, 4)
    bad = SimpleNamespace(**vars(metadata))
    bad.seq_lens = read.cpu_window_exact.clone()
    by_layer = {f"draft.layer.{i}": metadata for i in range(5)}
    by_layer["other.distinct.layer"] = bad
    with pytest.raises(RuntimeError, match="lost its current seq_lens owner"):
        _run_actual_private_binding_loop(graph_device_binding, read, by_layer)
    assert metadata.seq_lens is read.device_window_exact
    assert bad.seq_lens is not read.device_window_exact


@pytest.mark.parametrize(
    "relative,method_name",
    [
        ("dflash/aclgraph.py", "_updatable_graph_replay"),
        ("target_dependent_ar/aclgraph.py", "_run_eagle_window_fullgraph"),
    ],
)
@pytest.mark.parametrize("failure", [None, "validation", "update"])
def test_typed_update_uses_manager_config_and_restores_context(relative, method_name, failure):
    """Strict Window validation and original UPDATE share the actual owner config."""
    from contextlib import contextmanager
    from contextvars import ContextVar

    source = SOURCE.parent / relative
    tree = ast.parse(source.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == method_name)
    statements = [
        n
        for n in ast.walk(method)
        if isinstance(n, ast.With)
        and any(ast.unparse(item.context_expr) == "set_current_vllm_config(self.vllm_config)" for item in n.items)
    ]
    assert len(statements) == 1
    statement = statements[0]
    assert len(statement.body) == 2
    validation, update_call = statement.body
    assert isinstance(validation, ast.Expr) and isinstance(validation.value, ast.Call)
    assert isinstance(update_call, ast.Expr) and isinstance(update_call.value, ast.Call)
    assert isinstance(validation.value.func, ast.Attribute) and validation.value.func.attr == "validate_update"
    assert ast.unparse(update_call.value.func) == "graph.update"
    assert len(update_call.value.args) == 2 and not update_call.value.keywords
    current: ContextVar[object | None] = ContextVar("current_vllm_config", default=None)
    manager_config, ambient_config = object(), object()
    context_entries: list[object] = []
    observations: list[str] = []
    backend, route = object(), object()
    resolved = object()

    @contextmanager
    def set_current_vllm_config(config):
        context_entries.append(config)
        token = current.set(config)
        try:
            yield
        finally:
            current.reset(token)

    def validate(graph_owner, tasks, *, route):
        assert current.get() is manager_config, "Current vLLM config is not set"
        assert graph_owner is graph and tasks is resolved
        assert route is namespace["route"]
        observations.append("validated")
        if failure == "validation":
            raise RuntimeError("original validation failure")

    def update(stream, tasks):
        assert current.get() is manager_config
        assert stream is self.update_stream and tasks is resolved
        assert observations == ["validated"]
        observations.append("updated")
        if failure == "update":
            raise RuntimeError("original UPDATE failure")

    self = SimpleNamespace(
        vllm_config=manager_config,
        update_stream=object(),
        speculator=SimpleNamespace(
            attn_backends={"draft": backend},
            _draft_window_graph_binding_provider=SimpleNamespace(validate_update=validate),
        ),
    )
    graph = SimpleNamespace(update=update)
    namespace = dict(
        self=self,
        graph=graph,
        desc=SimpleNamespace(num_tokens=16),
        resolved=resolved,
        resolved_tasks=resolved,
        route=route,
        graph_route=route,
        session=SimpleNamespace(provider=SimpleNamespace(validate_update=validate)),
        owner=SimpleNamespace(attn_backend=backend),
        set_current_vllm_config=set_current_vllm_config,
    )

    def execute(nodes):
        module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
        exec(compile(module, str(source), "exec"), namespace)

    with pytest.raises(AssertionError, match="Current vLLM config is not set"):
        execute([validation])
    assert observations == [] and context_entries == []
    token = current.set(ambient_config)
    try:
        if failure:
            with pytest.raises(RuntimeError, match="original"):
                execute([statement])
        else:
            execute([statement])
        assert current.get() is ambient_config
    finally:
        current.reset(token)
    assert current.get() is None
    assert context_entries == [manager_config]
    assert observations == (["validated"] if failure == "validation" else ["validated", "updated"])


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
    inputs = SimpleNamespace(identity=identity)
    inputs.validate_for = lambda expected: None if expected is identity else (_ for _ in ()).throw(ValueError("stale"))
    runtime: Any = SimpleNamespace(
        _active_inputs=inputs,
        _metadata=None,
        capability=SimpleNamespace(policy=SimpleNamespace(normalized_method="dflash")),
    )
    provider = object.__new__(DraftWindowGraphBindingProvider)
    provider._runtime_state = runtime
    owner = SimpleNamespace(
        _draft_window_proposal_context=proposal,
        _draft_window_graph_binding_provider=provider,
        num_query_per_req=8,
        _draft_window_runtime_state=runtime,
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
    c.owner._draft_window_runtime_state._metadata = object()
    with pytest.raises(RuntimeError, match="existing publication"):
        c.defer_parallel_graph_metadata(c.owner, c.inputs, padded_requests=2, padded_tokens=16)


@pytest.fixture
def full_metadata_finalization():
    """Finalize one private mapping without inventing a second reader group."""
    source = SOURCE.with_name("draft_window_graph_runtime.py")
    tree = ast.parse(source.read_text())
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_build_private_graph_metadata"
    )
    loop = next(n for n in function.body if isinstance(n, ast.For) and ast.unparse(n.iter) == "metadata.values()")
    writes = SimpleNamespace(
        slot_mapping=object(),
        positions=object(),
        query_start_loc=object(),
        actual_seq_lengths_q=(8,),
        attn_state=object(),
        num_actual_tokens=8,
        causal=False,
    )

    def finalize(metadata):
        exec(
            compile(ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[])), str(source), "exec"),
            {"metadata": metadata, "write_side": writes},
        )
        return metadata

    return SimpleNamespace(finalize=finalize, writes=writes)


def test_full_graph_final_metadata_preserves_group_aliases_and_canonical_write_state(full_metadata_finalization):
    case = full_metadata_finalization
    layer = SimpleNamespace(attn_state=object())
    metadata = {"draft.a": layer, "draft.b": layer, "unused": None}
    result = case.finalize(metadata)
    assert result is metadata and result["draft.a"] is result["draft.b"] is layer
    assert layer.causal is False and layer.slot_mapping is case.writes.slot_mapping
    assert layer.positions is case.writes.positions and layer.query_start_loc is case.writes.query_start_loc
    assert layer.actual_seq_lengths_q == [8] and layer.num_actual_tokens == 8
    assert layer.attn_state is case.writes.attn_state and result["unused"] is None
