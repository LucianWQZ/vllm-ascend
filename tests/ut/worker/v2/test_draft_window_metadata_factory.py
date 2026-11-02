# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Factory patch contracts; unchanged source bodies, no NPU execution claim."""

import ast
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.worker.v2.spec_decode.draft_window_lifetime import (
    DraftWindowOwnerPool,
)

CALLBACK_NAMES = ("common_metadata_callback", "metadata_builder_callback")


@pytest.fixture
def metadata_isolation():
    from vllm_ascend.worker.v2.spec_decode.draft_window_graph_metadata import (
        copy_draft_attention_metadata,
        validate_draft_metadata_isolation,
    )

    return copy_draft_attention_metadata, validate_draft_metadata_isolation


def _attention_metadata():
    return SimpleNamespace(
        block_tables=torch.arange(12, dtype=torch.int32).view(3, 4),
        seq_lens=torch.tensor([3, 4], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([3, 4], dtype=torch.int32),
        seq_lens_list=[3, 4],
        actual_seq_lengths_q=[1, 2],
        slot_mapping=torch.tensor([9, 13], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        attn_mask=torch.zeros(2, 2, dtype=torch.int8),
        attn_state="target",
        qfa_metadata_cache={"lengths": (torch.tensor([0, 1, 2]), torch.tensor([3, 4]))},
    )


def test_metadata_copy_owns_mutable_fields_and_borrows_read_only_operands(metadata_isolation):
    copy_metadata, validate = metadata_isolation
    layer = _attention_metadata()
    target = {"target": layer}
    private = copy_metadata({"draft": layer})
    copied = private["draft"]
    assert copied is not layer
    for field in ("block_tables", "seq_lens", "seq_lens_cpu", "slot_mapping", "query_start_loc", "attn_mask"):
        assert getattr(copied, field) is getattr(layer, field)
    # Cache payloads are read-only operator outputs; only the mutable cache
    # dictionary belongs to the per-build metadata object.
    assert copied.qfa_metadata_cache is not layer.qfa_metadata_cache
    assert copied.qfa_metadata_cache["lengths"] is layer.qfa_metadata_cache["lengths"]
    validate(private, target_metadata=target)
    copied.attn_state = "draft"
    copied.seq_lens_list[0] = 99
    copied.actual_seq_lengths_q.append(3)
    copied.qfa_metadata_cache.clear()
    assert layer.attn_state == "target"
    assert layer.seq_lens_list == [3, 4]
    assert layer.actual_seq_lengths_q == [1, 2]
    assert "lengths" in layer.qfa_metadata_cache


def test_real_ascend_metadata_preserves_canonical_full_operands(metadata_isolation):
    from vllm_ascend.attention.attention_v1 import AscendAttentionState, AscendMetadata

    copy_metadata, validate = metadata_isolation
    layer = AscendMetadata(**vars(_attention_metadata()))
    private = copy_metadata({"draft": layer})
    copied = private["draft"]
    copied.attn_state = AscendAttentionState.DecodeOnly
    copied.seq_lens_list[0] = 9
    assert layer.attn_state == "target" and layer.seq_lens_list == [3, 4]
    assert copied.slot_mapping is layer.slot_mapping
    assert copied.query_start_loc is layer.query_start_loc
    validate(private, target_metadata={"target": layer})


def test_metadata_copy_preserves_group_aliases_and_missing_layers(metadata_isolation):
    copy_metadata, validate = metadata_isolation
    layer = _attention_metadata()
    private = copy_metadata({"a": layer, "b": layer, "unused": None})
    assert private["a"] is private["b"] and private["a"] is not layer
    assert private["unused"] is None
    validate(private, target_metadata={"target": layer})


@pytest.mark.parametrize("field", [None, "seq_lens_list", "actual_seq_lengths_q", "qfa_metadata_cache"])
def test_publication_guard_rejects_actual_target_mutation_aliases(metadata_isolation, field):
    copy_metadata, validate = metadata_isolation
    target_layer = _attention_metadata()
    target = {"target": target_layer}
    draft = copy_metadata({"draft": target_layer})
    if field is None:
        draft["draft"] = target_layer
    else:
        setattr(draft["draft"], field, getattr(target_layer, field))
    with pytest.raises(ValueError, match="alias Target"):
        validate(draft, target_metadata=target)
    with pytest.raises(ValueError, match="alias Target"):
        validate(target, target_metadata=target)


def test_frozen_slotted_metadata_is_copied_before_field_replacement(metadata_isolation):
    copy_metadata, validate = metadata_isolation

    @dataclass(frozen=True, slots=True)
    class Metadata:
        seq_lens_list: list[int]
        actual_seq_lengths_q: list[int]
        qfa_metadata_cache: dict
        block_tables: torch.Tensor

    table = torch.arange(4, dtype=torch.int32).view(1, 4)
    layer = Metadata([3], [1], {}, table)
    private = copy_metadata({"draft": layer})
    private["draft"].seq_lens_list[0] = 9
    assert layer.seq_lens_list == [3]
    assert private["draft"].block_tables is table
    validate(private, target_metadata={"target": layer})


def test_metadata_copy_rejects_invalid_known_host_fields(metadata_isolation):
    copy_metadata, validate = metadata_isolation
    for field, value in (
        ("seq_lens_list", (3, 4)),
        ("actual_seq_lengths_q", torch.tensor([1, 2])),
        ("qfa_metadata_cache", []),
    ):
        layer = _attention_metadata()
        setattr(layer, field, value)
        with pytest.raises(TypeError, match=field):
            copy_metadata({"draft": layer})
    with pytest.raises(ValueError, match="non-empty layer mapping"):
        copy_metadata({})
    with pytest.raises(ValueError, match="layer mappings"):
        validate({}, target_metadata=None)


@pytest.fixture
def factory_contract():
    """Isolate the patched module and backend, retaining both real context managers."""
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/v2/attn_utils.py"
    names = {"build_attn_metadata_factory", "build_attn_metadata_wrapper"}
    nodes = [node for node in ast.parse(path.read_text()).body if getattr(node, "name", None) in names]
    assert len(nodes) == len(names)
    original = Mock(side_effect=lambda **kwargs: kwargs)
    ascend = Mock(side_effect=lambda **kwargs: kwargs)
    module = SimpleNamespace(build_attn_metadata=original)
    scope = {
        "contextmanager": contextmanager,
        "_BUILD_ATTN_METADATA_MODULE": module,
        "build_attn_metadata": ascend,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return SimpleNamespace(
        module=module,
        original=original,
        ascend=ascend,
        factory=scope["build_attn_metadata_factory"],
        wrapper=scope["build_attn_metadata_wrapper"],
    )


@pytest.mark.parametrize("callback_names", [(), CALLBACK_NAMES[:1], CALLBACK_NAMES[1:], CALLBACK_NAMES])
def test_factory_forwards_callbacks_and_baseline_fields_once(factory_contract, callback_names):
    contract = factory_contract
    callbacks = {name: Mock() for name in callback_names}
    flags, parallel, state = object(), object(), object()
    cpu_lengths = SimpleNamespace(numpy=Mock(return_value=object()))
    with contract.wrapper():
        wrapped = contract.module.build_attn_metadata
        with contract.factory(
            [0, 1, 2],
            2,
            flags,
            cpu_lengths,
            attn_state=state,
            parallel_config=parallel,
            offload_kwargs={"copy_sfa_draft_index": 2},
            **callbacks,
        ):
            result = contract.module.build_attn_metadata(causal=False)
        assert contract.module.build_attn_metadata is wrapped
    assert contract.module.build_attn_metadata is contract.original
    contract.original.assert_not_called()
    contract.ascend.assert_called_once()
    cpu_lengths.numpy.assert_called_once_with()
    assert result["positions"] == [0, 1]
    assert result["is_prefilling"] is flags
    assert result["attn_state"] is state
    assert result["parallel_config"] is parallel
    assert result["seq_lens_np"] is cpu_lengths.numpy.return_value
    assert result["copy_sfa_draft_index"] == 2
    assert result["causal"] is False
    for name, callback in callbacks.items():
        assert result[name] is callback
        callback.assert_not_called()  # The factory forwards; the actual builder invokes.
    assert all(name not in result for name in CALLBACK_NAMES if name not in callbacks)


@pytest.mark.parametrize("name", CALLBACK_NAMES)
def test_factory_without_callback_preserves_an_explicit_caller_callback(factory_contract, name):
    contract = factory_contract
    callback = Mock()
    with contract.factory([0], 1, False, **{name: None}):
        result = contract.module.build_attn_metadata(**{name: callback})
    assert result[name] is callback
    assert contract.module.build_attn_metadata is contract.original


@pytest.mark.parametrize("fail_inner", [False, True])
def test_factory_nested_scope_restores_the_exact_previous_function(factory_contract, fail_inner):
    contract = factory_contract
    callback = Mock()
    with contract.factory([0, 1], 2, False):
        outer = contract.module.build_attn_metadata
        try:
            with contract.factory([0, 1], 2, False, common_metadata_callback=callback):
                inner = contract.module.build_attn_metadata
                assert inner is not outer
                assert inner()["common_metadata_callback"] is callback
                if fail_inner:
                    raise RuntimeError("inner body failed")
        except RuntimeError as error:
            assert fail_inner and str(error) == "inner body failed"
        assert contract.module.build_attn_metadata is outer
        assert "common_metadata_callback" not in outer()
    assert contract.module.build_attn_metadata is contract.original


@pytest.mark.parametrize("name", CALLBACK_NAMES)
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("same_callback", [False, True])
def test_factory_conflicting_callbacks_fail_before_backend_and_restore(factory_contract, name, nested, same_callback):
    contract = factory_contract
    first = Mock()
    second = first if same_callback else Mock()
    with pytest.raises(ValueError, match="Multiple draft"), contract.factory([0], 1, False, **{name: first}):
        if nested:
            with contract.factory([0], 1, False, **{name: second}):
                contract.module.build_attn_metadata()
        else:
            contract.module.build_attn_metadata(**{name: second})
    assert contract.module.build_attn_metadata is contract.original
    contract.original.assert_not_called()
    first.assert_not_called()
    second.assert_not_called()


@pytest.mark.parametrize("stage", ["backend", *CALLBACK_NAMES])
def test_factory_restores_patch_when_callback_or_backend_raises(factory_contract, stage):
    contract = factory_contract

    def fail(*args, **kwargs):
        raise RuntimeError("injected build failure")

    # Only the final builder boundary is isolated. Its invocation of a callback
    # is a failure injection, not a substitute for backend interface coverage.
    def backend(**kwargs):
        if stage == "backend":
            fail()
        if stage == "common_metadata_callback":
            kwargs[stage](0, object())
        else:
            kwargs[stage](0, object(), ["draft.layer"])

    contract.ascend.side_effect = backend
    callbacks = {} if stage == "backend" else {stage: fail}
    with (
        pytest.raises(RuntimeError, match="injected build failure"),
        contract.wrapper(),
        contract.factory([0], 1, False, **callbacks),
    ):
        contract.module.build_attn_metadata()
    assert contract.module.build_attn_metadata is contract.original
    contract.ascend.assert_called_once()


@pytest.mark.parametrize("fail_inner", [False, True])
def test_factory_wrapper_nesting_restores_outer_factory(factory_contract, fail_inner):
    contract = factory_contract
    with contract.wrapper(), contract.factory([0], 1, False):
        outer = contract.module.build_attn_metadata
        try:
            with contract.wrapper():
                assert contract.module.build_attn_metadata is contract.ascend
                if fail_inner:
                    raise RuntimeError("nested wrapper failed")
        except RuntimeError as error:
            assert fail_inner and str(error) == "nested wrapper failed"
        assert contract.module.build_attn_metadata is outer
    assert contract.module.build_attn_metadata is contract.original


@pytest.mark.parametrize("stage", ["backend", *CALLBACK_NAMES])
def test_factory_error_restores_module_but_quarantines_active_owner(factory_contract, stage):
    contract = factory_contract
    pool = DraftWindowOwnerPool(1)
    payload = object()
    completion = Mock(side_effect=AssertionError("failed proposal cannot record a success completion"))

    def fail(*args, **kwargs):
        raise RuntimeError("injected partial-submit host error")

    def backend(**kwargs):
        if stage == "backend":
            fail()
        if stage == "common_metadata_callback":
            kwargs[stage](0, object())
        else:
            kwargs[stage](0, object(), ["draft.layer"])

    contract.ascend.side_effect = backend
    callbacks = {} if stage == "backend" else {stage: fail}
    with pytest.raises(RuntimeError, match="partial-submit host error"), pool.operation(completion) as owner:
        pool.retain(owner, payload)
        # Entered proposal lifetime; no native operation is claimed here.
        with contract.wrapper(), contract.factory([0], 1, False, **callbacks):
            contract.module.build_attn_metadata()
    assert contract.module.build_attn_metadata is contract.original
    assert pool.poisoned
    assert owner.retained == [payload]
    completion.assert_not_called()
    with pytest.raises(RuntimeError, match="quarantined"):
        pool.acquire()
