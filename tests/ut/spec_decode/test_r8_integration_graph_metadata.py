# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Actual Graph metadata owner controls, with no framework or NPU imports.

These isolated CPU protocol tests establish exact-source/owner rejection;
model, numerical, capture and lifetime evidence remains separate.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[3]
SOURCE = SOURCE_ROOT / "vllm_ascend/worker/v2/spec_decode/draft_window_eager.py"


class TensorProtocol:
    def __init__(self, values, device="cpu", pointer=None, dtype="int32"):
        self.values = tuple(values)
        self.shape = (len(values),)
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
    mixin = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DraftWindowEagerMixin")
    call = next(
        n for n in mixin.body if isinstance(n, ast.FunctionDef) and n.name == "_call_ascend_draft_attn_metadata_builder"
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes, call],
        type_ignores=[],
    )
    namespace = {
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
    characterization_source = SOURCE.with_name("draft_window_graph_characterization.py")
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
        for n in ast.parse(characterization_source.read_text()).body
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
    exec(compile(ast.fix_missing_locations(module), str(characterization_source), "exec"), namespace)

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
    characterization = ast.parse(SOURCE.with_name("draft_window_graph_characterization.py").read_text())
    private = next(
        n for n in characterization.body if isinstance(n, ast.FunctionDef) and n.name == "_build_private_graph_metadata"
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
    mixin = next(n for n in eager_tree.body if isinstance(n, ast.ClassDef) and n.name == "DraftWindowEagerMixin")
    eager = next(
        n for n in mixin.body if isinstance(n, ast.FunctionDef) and n.name == "_build_ascend_draft_attn_metadata"
    )
    # True eager retains active-clamped query boundaries; FULL uses its own padded call.
    assert "np.minimum(self.arange_np[:num_reqs_padded + 1], num_reqs)" in ast.unparse(eager)


def _run_actual_private_binding_loop(graph_device_binding, read, metadata_by_layer):
    """Execute only the actual caller loop, using the existing strict binder."""
    source = SOURCE.with_name("draft_window_graph_characterization.py")
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
