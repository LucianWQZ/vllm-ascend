# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Actual Graph metadata owner controls, with no framework or NPU imports.

These isolated CPU protocol tests establish exact-source/owner rejection;
model, numerical, capture and lifetime evidence remains separate.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[3]
SOURCE = SOURCE_ROOT / "vllm_ascend/worker/v2/spec_decode/draft_window_eager.py"


@pytest.fixture(params=["dspark", "dflash", "dflash2", "eagle3"])
def graph_source_initialization(request, monkeypatch):
    """Exercise actual owner initialization without loading a device model."""
    method = request.param
    paths = {
        "dspark": ("dspark/speculator.py", "AscendDSparkSpeculator"),
        "dflash": ("dflash/speculator.py", "AscendDFlashSpeculator"),
        "dflash2": ("dflash/speculator.py", "AscendDFlashSpeculator"),
        "eagle3": ("draft_window_eagle3_adapter.py", "Eagle3GraphIntegrationMixin"),
    }
    relative, class_name = paths[method]
    path = SOURCE.parent / relative
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == class_name)
    initialize = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_initialize_draft_window_source_capability"
    )
    capability, context, state = object(), object(), object()
    build_source = Mock(return_value=(capability, context))
    build_state = Mock(return_value=state)
    capture_name = "vllm_ascend.worker.v2.spec_decode.draft_window_eagle3_capture"
    capture = ModuleType(capture_name)
    capture.__dict__["build_eagle3_graph_source"] = build_source
    monkeypatch.setitem(sys.modules, capture_name, capture)
    namespace = {"build_parallel_graph_source": build_source, "build_draft_window_runtime_state": build_state}
    exec(compile(ast.Module(body=[initialize], type_ignores=[]), str(path), "exec"), namespace)
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
        initialize=namespace[initialize.name],
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


@pytest.mark.parametrize("case", ["valid", "missing-source", "missing-runtime", "rank-drift", "invalid-world"])
def test_graph_capture_requires_explicit_matching_source_topology(case):
    path = SOURCE.with_name("draft_window_graph.py")
    provider = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "DraftWindowGraphBindingProvider"
    )
    method = next(n for n in provider.body if isinstance(n, ast.FunctionDef) and n.name == "_capture_topology")
    namespace: dict[str, Any] = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    source: SimpleNamespace | None = SimpleNamespace(
        tensor_parallel_size=16, tensor_parallel_rank=7, parallel_read_semantics="replicated"
    )
    runtime: SimpleNamespace | None = SimpleNamespace(**vars(source))
    if case == "missing-source":
        source = None
    elif case == "missing-runtime":
        runtime = None
    elif case == "rank-drift":
        assert runtime is not None
        runtime.tensor_parallel_rank = 8
    elif case == "invalid-world":
        assert source is not None
        source.tensor_parallel_size = 0
    owner = SimpleNamespace(
        _runtime_state=SimpleNamespace(
            capability=SimpleNamespace(eager_topology_fingerprint=source), topology_fingerprint=runtime
        )
    )
    if case == "valid":
        assert namespace[method.name](owner) == (16, 7, "replicated")
    else:
        with pytest.raises(RuntimeError, match="source topology|world/rank|identity drift"):
            namespace[method.name](owner)


class TensorProtocol:
    def __init__(self, values, device="cpu", pointer=None, dtype="int32"):
        self.values = tuple(values)
        self.shape: tuple[int, ...] = (len(values),)
        self.ndim = 1
        self.dtype = dtype
        self.device = SimpleNamespace(type=device)
        self.pointer = id(self) if pointer is None else pointer

    def data_ptr(self):
        return self.pointer

    def stride(self):
        return (1,)


@pytest.fixture
def bridge():
    tree = ast.parse(SOURCE.read_text())
    names = {"AscendDraftAttentionReadInputs", "_AscendDraftAttentionReadCertificate"}
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
    mixin = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DraftWindowGraphInputMixin")
    call = next(
        n for n in mixin.body if isinstance(n, ast.FunctionDef) and n.name == "_call_ascend_draft_attn_metadata_builder"
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes, call],
        type_ignores=[],
    )
    namespace: dict[str, Any] = {
        "dataclass": dataclass,
        "torch": SimpleNamespace(int32="int32"),
        "upstream_speculator_module": SimpleNamespace(build_attn_metadata=lambda **kw: kw),
    }
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    cls = namespace["AscendDraftAttentionReadInputs"]
    device = TensorProtocol([17, 1025, 0, 0], "npu")
    exact = TensorProtocol([17, 1025, 0, 0])
    upper = TensorProtocol([24, 1040, 0, 0])
    read = cls((object(), object()), device, exact, upper, 1040)
    common = SimpleNamespace(
        num_reqs=4,
        seq_lens=TensorProtocol(device.values, "npu", device.data_ptr()),
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
    return SimpleNamespace(
        read=read, common=common, kwargs=kwargs, call=namespace["_call_ascend_draft_attn_metadata_builder"]
    )


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
    bridge.kwargs[field] = TensorProtocol(original.values, original.device.type)
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


def test_shared_mixin_forwards_same_exact_read_owner_without_mutating_kwargs(bridge):
    result = bridge.call(None, read_inputs=bridge.read, builder_kwargs=bridge.kwargs)
    assert result["draft_window_read_inputs"] is bridge.read
    assert result["seq_lens_cpu"] is bridge.read.seq_lens_cpu_exact
    assert "draft_window_read_inputs" not in bridge.kwargs
    with pytest.raises(ValueError, match="already have a metadata build owner"):
        bridge.call(None, read_inputs=bridge.read, builder_kwargs=result)


class GraphTensorProtocol(TensorProtocol):
    """Header/value protocol for the actual Graph identity reader and guard."""

    def __init__(self, values, device="cpu", pointer=None, shape=None, dtype="int32"):
        super().__init__(values, device=device, pointer=pointer, dtype=dtype)
        if shape is not None:
            self.shape = tuple(shape)
            self.ndim = len(self.shape)

    def untyped_storage(self):
        return self

    def storage_offset(self):
        return 0

    def stride(self):
        return (self.shape[1], 1) if self.ndim == 2 else (1,)

    def tolist(self):
        return list(self.values)


@pytest.fixture
def graph_device_binding():
    runtime_source = SOURCE.with_name("draft_window_graph_runtime.py")
    graph_source = SOURCE.with_name("draft_window_graph.py")
    graph_tree = ast.parse(graph_source.read_text())
    identity_class = next(
        n for n in graph_tree.body if isinstance(n, ast.ClassDef) and n.name == "DraftWindowGraphTensorIdentity"
    )
    identity_function = next(
        n for n in graph_tree.body if isinstance(n, ast.FunctionDef) and n.name == "_tensor_identity"
    )
    provider = next(
        n for n in graph_tree.body if isinstance(n, ast.ClassDef) and n.name == "DraftWindowGraphBindingProvider"
    )
    guard = next(
        n for n in provider.body if isinstance(n, ast.FunctionDef) and n.name == "_validate_metadata_uses_read_carrier"
    )
    # Execute the actual method body as a standalone function, without importing
    # the provider/framework or replacing its owner and value checks.
    guard.decorator_list = []
    helper = next(
        n
        for n in ast.parse(runtime_source.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_bind_parallel_graph_device_lengths"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            identity_class,
            identity_function,
            guard,
            helper,
        ],
        type_ignores=[],
    )
    namespace = {"dataclass": dataclass, "torch": SimpleNamespace(Tensor=GraphTensorProtocol)}
    exec(compile(ast.fix_missing_locations(module), str(runtime_source), "exec"), namespace)

    def make(actual, padded):
        assert (actual, padded) in ((1, 2), (3, 4))
        values = [17 + i * 1024 for i in range(actual)] + [0] * (padded - actual)
        table = GraphTensorProtocol(list(range(padded * 2)), "npu", shape=(padded, 2))
        exact = GraphTensorProtocol(values)
        device = GraphTensorProtocol(values, "npu")
        read = SimpleNamespace(
            actual_request_count=actual,
            padded_request_count=padded,
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
        bind=namespace["_bind_parallel_graph_device_lengths"],
        guard=namespace["_validate_metadata_uses_read_carrier"],
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
    assert read.actual_request_count == actual and read.padded_request_count == padded


@pytest.mark.parametrize("field", ["seq_lens", "seq_lens_cpu", "seq_lens_gpu", "block_tables"])
def test_private_full_rejects_same_values_with_wrong_builder_owner(graph_device_binding, field):
    read, metadata = graph_device_binding.make(3, 4)
    original = getattr(metadata, field)
    setattr(metadata, field, GraphTensorProtocol(original.values, original.device.type, shape=original.shape))
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


def test_private_full_padded_build_is_separate_from_unchanged_true_eager_clamp():
    runtime = ast.parse(SOURCE.with_name("draft_window_graph_runtime.py").read_text())
    private = next(
        n for n in runtime.body if isinstance(n, ast.FunctionDef) and n.name == "_build_private_graph_metadata"
    )
    calls = [
        n
        for n in ast.walk(private)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "_build_ascend_draft_attn_metadata"
    ]
    assert len(calls) == 1
    passed = {kw.arg: ast.unparse(kw.value) for kw in calls[0].keywords}
    assert passed["num_reqs"] == passed["num_reqs_padded"] == "read_inputs.padded_request_count"
    private_statements = [ast.unparse(n) for n in private.body]
    isolation = next(
        i
        for i, text in enumerate(private_statements)
        if "private_isolation_view = _isolate_draft_metadata_from_target" in text
    )
    binding = next(i for i, text in enumerate(private_statements) if "_bind_parallel_graph_device_lengths(" in text)
    assert isolation < binding

    eager_tree = ast.parse(SOURCE.read_text())
    mixin = next(n for n in eager_tree.body if isinstance(n, ast.ClassDef) and n.name == "DraftWindowGraphInputMixin")
    eager = next(
        n for n in mixin.body if isinstance(n, ast.FunctionDef) and n.name == "_build_ascend_draft_attn_metadata"
    )
    # True eager retains active-clamped query boundaries; FULL uses its own padded call.
    assert "np.minimum(self.arange_np[:num_reqs_padded + 1], num_reqs)" in ast.unparse(eager)


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
        private_isolation_view=metadata_by_layer,
        read_inputs=read,
        fingerprint=SimpleNamespace(query_width=8),
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
    bad.seq_lens = GraphTensorProtocol(read.cpu_window_exact.values)
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
    assert ast.unparse(validation.value.func) == "validate_draft_window_update_tasks"
    assert ast.unparse(update_call.value.func) == "graph.update"
    assert len(update_call.value.args) == 2 and not update_call.value.keywords
    current: ContextVar[object | None] = ContextVar("current_vllm_config", default=None)
    manager_config, ambient_config = object(), object()
    context_entries: list[object] = []
    observations: list[str] = []
    backend, binding = object(), object()
    metadata, resolved = [object()], object()

    @contextmanager
    def set_current_vllm_config(config):
        context_entries.append(config)
        token = current.set(config)
        try:
            yield
        finally:
            current.reset(token)

    def validate(graph_owner, tasks, actual_binding, actual_metadata, actual_backend, tokens):
        assert current.get() is manager_config, "Current vLLM config is not set"
        assert graph_owner is graph and tasks is resolved and actual_binding is binding
        assert actual_metadata is metadata and actual_backend is backend and tokens == 16
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
        speculator=SimpleNamespace(attn_backends={"draft": backend}),
    )
    graph = SimpleNamespace(update=update)
    namespace = dict(
        self=self,
        graph=graph,
        desc=SimpleNamespace(num_tokens=16),
        resolved=resolved,
        resolved_tasks=resolved,
        binding=binding,
        draft_attn_metadatas=metadata,
        metadata_batch=metadata,
        owner=SimpleNamespace(attn_backend=backend),
        set_current_vllm_config=set_current_vllm_config,
        validate_draft_window_update_tasks=validate,
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
def typed_update_guard():
    """Run the actual UPDATE guard with opaque captured handles and operands."""
    source = SOURCE.with_name("draft_window_graph.py")
    tree = ast.parse(source.read_text())
    guard = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "validate_draft_window_update_tasks"
    )
    guard.body = [n for n in guard.body if not isinstance(n, (ast.ImportFrom, ast.Import))]

    class FIAParamProvider:
        is_draft_model = True
        layer_name = "draft.layer.0"

    operation = object()
    checks = []
    namespace = {
        "FIAParamProvider": FIAParamProvider,
        "torch_npu": SimpleNamespace(npu_fused_infer_attention_score=SimpleNamespace(out=operation)),
        "validate_draft_window_graph_backend_update": lambda *args, **kwargs: checks.append((args, kwargs)),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[guard], type_ignores=[])), str(source), "exec"), namespace)
    provider = FIAParamProvider()
    metadata = SimpleNamespace(block_tables=object(), seq_lens_list=[17], actual_seq_lengths_q=[8])
    captured = SimpleNamespace(handle=object(), event=object(), provider=provider, provider_index=0)
    task = SimpleNamespace(
        handle=captured.handle,
        event=captured.event,
        provider=provider,
        provider_index=0,
        operation=operation,
        kwargs={
            "block_table": metadata.block_tables,
            "actual_seq_lengths_kv": metadata.seq_lens_list,
            "actual_seq_lengths": metadata.actual_seq_lengths_q,
        },
    )
    backend = SimpleNamespace(get_impl_cls=lambda: object, get_builder_cls=lambda: object)
    return SimpleNamespace(
        call=namespace["validate_draft_window_update_tasks"],
        graph=SimpleNamespace(tasks=[captured]),
        task=task,
        metadata=[{"draft.layer.0": metadata}],
        backend=backend,
        checks=checks,
    )


def test_typed_update_keeps_actual_captured_handles_and_current_metadata(typed_update_guard):
    case = typed_update_guard
    binding = object()
    case.call(case.graph, [case.task], binding, case.metadata, case.backend, 8)
    assert len(case.checks) == 1
    assert case.checks[0][0] == (binding,)
    assert case.checks[0][1]["draft_attn_metadatas"] is case.metadata


@pytest.mark.parametrize(
    "field", ["handle", "event", "provider_index", "operation", "block_table", "actual_seq_lengths_kv"]
)
def test_typed_update_rejects_stale_task_or_replaced_read_operands(typed_update_guard, field):
    case = typed_update_guard
    if field == "provider_index":
        case.task.provider_index = 1
    elif field in case.task.kwargs:
        case.task.kwargs[field] = object()
    else:
        setattr(case.task, field, object())
    with pytest.raises(RuntimeError):
        case.call(case.graph, [case.task], object(), case.metadata, case.backend, 8)
    assert not case.checks


@pytest.fixture
def full_metadata_dispatch():
    """Execute the normal FULL-selection boundary with actual epoch/row keys."""
    source = SOURCE.with_name("draft_window_graph_metadata.py")
    tree = ast.parse(source.read_text())
    names = {
        "ParallelGraphMetadataSelection",
        "select_parallel_graph_metadata",
        "defer_parallel_graph_metadata",
        "validate_deferred_parallel_graph_metadata",
    }
    nodes = [n for n in tree.body if getattr(n, "name", None) in names]

    class Provider:
        inventory_method = "dflash"

    namespace = {"dataclass": dataclass, "DraftWindowGraphBindingProvider": Provider}
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    identity = SimpleNamespace(mapping_epoch=7, ordered_request_fingerprint=("req-a", "req-b"), padded_request_count=2)
    inputs = SimpleNamespace(identity=identity)
    inputs.validate_for = lambda expected: None if expected is identity else (_ for _ in ()).throw(ValueError("stale"))
    owner = SimpleNamespace(
        _draft_window_proposal_context=SimpleNamespace(mapping_epoch=7, ordered_request_ids=("req-a", "req-b")),
        _draft_window_graph_binding_provider=Provider(),
        num_query_per_req=8,
        _draft_window_runtime_state=SimpleNamespace(_active_inputs=inputs, metadata_store=SimpleNamespace(_entry=None)),
        _draft_window_live_inputs=inputs,
        _draft_window_deferred_metadata=None,
    )
    descriptor = SimpleNamespace(cg_mode=SimpleNamespace(name="FULL"), num_reqs=2, num_tokens=16)
    return SimpleNamespace(owner=owner, inputs=inputs, descriptor=descriptor, **namespace)


def test_full_metadata_dispatch_defers_once_for_current_generation(full_metadata_dispatch):
    c = full_metadata_dispatch
    c.select_parallel_graph_metadata(c.owner, c.descriptor)
    assert c.defer_parallel_graph_metadata(c.owner, c.inputs, padded_requests=2, padded_tokens=16)
    c.validate_deferred_parallel_graph_metadata(c.owner, c.inputs, c.descriptor)
    assert c.owner._draft_window_deferred_metadata is c.inputs.identity
    with pytest.raises(RuntimeError, match="existing publication"):
        c.defer_parallel_graph_metadata(c.owner, c.inputs, padded_requests=2, padded_tokens=16)


@pytest.mark.parametrize(
    "field,value",
    [("mapping_epoch", 8), ("ordered_request_fingerprint", ("req-b", "req-a")), ("padded_request_count", 4)],
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
    c.owner._draft_window_runtime_state.metadata_store._entry = object()
    with pytest.raises(RuntimeError, match="existing publication"):
        c.defer_parallel_graph_metadata(c.owner, c.inputs, padded_requests=2, padded_tokens=16)


def test_full_graph_final_metadata_reuses_groups_without_mixing_native_reads():
    """The actual finalization loop shares per-group objects, not reader semantics."""
    import copy

    source = SOURCE.with_name("draft_window_graph_runtime.py")
    tree = ast.parse(source.read_text())
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_build_private_graph_metadata"
    )
    loop = next(
        n for n in function.body if isinstance(n, ast.For) and ast.unparse(n.iter) == "private_isolation_view.items()"
    )
    shared = SimpleNamespace(attn_state=object())
    writes = SimpleNamespace(
        slot_mapping=object(),
        positions=object(),
        query_start_loc=object(),
        actual_seq_lengths_q=(8,),
        attn_state=object(),
        num_actual_tokens=8,
        causal=False,
    )
    validates = []
    snapshot = SimpleNamespace(
        validate=lambda *args: validates.append(args),
        device_exact=object(),
        cpu_exact=SimpleNamespace(tolist=lambda: [17]),
    )

    class Slots:
        def __getitem__(self, key):
            return key[0], "canonical-full-slots"

    namespace: dict[str, Any] = dict(
        copy=copy,
        private_isolation_view={"full.a": shared, "full.b": shared, "swa.a": shared, "swa.b": shared},
        final_metadata={},
        finalized_groups={},
        native_gids={"swa.a": 0, "swa.b": 1},
        native_snapshot=snapshot,
        native_swa=object(),
        write_side=writes,
        read_inputs=SimpleNamespace(padded_request_count=1),
        fingerprint=SimpleNamespace(padded_token_count=8),
        owner=SimpleNamespace(
            _draft_window_live_inputs=SimpleNamespace(identity=object()),
            block_tables=SimpleNamespace(input_block_tables=[[10], [20]], slot_mappings=Slots()),
        ),
    )
    exec(compile(ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[])), str(source), "exec"), namespace)
    result = namespace["final_metadata"]
    assert result["full.a"] is result["full.b"]
    assert result["full.a"] is not shared and result["full.a"].causal is False
    assert result["swa.a"] is not result["swa.b"] and result["swa.a"] is not result["full.a"]
    assert result["swa.a"].block_tables == [10] and result["swa.b"].block_tables == [20]
    assert result["swa.a"].causal is result["swa.b"].causal is True
    assert len(validates) == 2
    assert not hasattr(shared, "slot_mapping")
