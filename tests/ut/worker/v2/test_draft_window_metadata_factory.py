# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Factory patch contracts; unchanged source bodies, no NPU execution claim."""

import ast
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm_ascend.worker.v2.spec_decode.draft_window_lifetime import (
    DraftWindowOwnerPool,
    DraftWindowOwnerState,
)

CALLBACK_NAMES = ("common_metadata_callback", "metadata_builder_callback")


@pytest.fixture
def metadata_isolation():
    from vllm_ascend.worker.v2.spec_decode.draft_window import (
        DraftMetadataStore,
        validate_draft_metadata_isolation,
    )
    from vllm_ascend.worker.v2.spec_decode.draft_window_eager import _isolate_draft_metadata_from_target

    return _isolate_draft_metadata_from_target, validate_draft_metadata_isolation, DraftMetadataStore


def test_metadata_copy_is_selective_and_isolates_shared_tensor_and_numpy_views(metadata_isolation):
    isolate, validate, _ = metadata_isolation
    tensor = torch.arange(12)
    array = np.arange(12)
    private_table = torch.arange(20).view(4, 5)
    target = {"tensor": tensor, "array": array}
    draft = {"read_table": private_table, "tensor": tensor[2:6], "array": array[3:8]}
    with pytest.raises(ValueError, match="aliases Target"):
        validate(draft, target_metadata=target)
    private = isolate(draft, target)
    assert private["read_table"] is private_table  # No unrelated Full table copy.
    validate(private, target_metadata=target)
    private["tensor"].fill_(-1)
    private["array"][:] = -1
    assert tensor.tolist() == list(range(12))
    assert array.tolist() == list(range(12))


def test_empty_tensors_do_not_alias_only_because_both_have_zero_address(metadata_isolation):
    isolate, validate, _ = metadata_isolation
    target, independent = torch.empty(0), torch.empty(0)
    assert target.data_ptr() == independent.data_ptr() == 0
    validate({"read": independent}, target_metadata={"read": target})
    assert isolate({"read": independent}, {"read": target})["read"] is independent
    with pytest.raises(ValueError, match="aliases Target"):
        validate({"read": target}, target_metadata={"read": target})


def test_frozen_slotted_metadata_keeps_layer_aliases_without_mutating_target(metadata_isolation):
    isolate, validate, _ = metadata_isolation

    @dataclass(frozen=True, slots=True)
    class Metadata:
        lengths: torch.Tensor
        nested: tuple

    target_layer = Metadata(torch.tensor([3]), ([np.arange(3)],))
    target = {"layer": target_layer}
    private = isolate({"a": target_layer, "b": target_layer}, target)
    assert private["a"] is private["b"] and private["a"] is not target_layer
    validate(private, target_metadata=target)
    private["a"].lengths.fill_(9)
    private["a"].nested[0][0][:] = 9
    assert target_layer.lengths.item() == 3
    assert target_layer.nested[0][0].tolist() == [0, 1, 2]


def test_mutable_metadata_cycles_bytearrays_and_inherited_slots_remain_isolated(metadata_isolation):
    isolate, validate, _ = metadata_isolation

    class Base:
        __slots__ = ("lengths",)

    class Layer(Base):
        __slots__ = ("unused",)

    layer = Layer()
    layer.lengths = torch.tensor([4])
    cycle = []
    cycle.append(cycle)
    target = {"cycle": cycle, "state": bytearray(b"ab"), "tags": {1}, "layer": layer}
    private = isolate(target, target)
    assert private["cycle"][0] is private["cycle"]
    assert not hasattr(private["layer"], "unused")
    validate(private, target_metadata=target)
    private["layer"].lengths.fill_(7)
    private["state"][0] = ord("z")
    private["tags"].add(2)
    assert layer.lengths.item() == 4 and target["state"] == b"ab" and target["tags"] == {1}


def test_numpy_memoryview_backing_and_publication_reject_remaining_aliases(metadata_isolation):
    isolate, validate, store_type = metadata_isolation
    backing = bytearray(range(12))
    target = {"array": np.frombuffer(backing, dtype=np.uint8)}
    draft = {"array": np.frombuffer(memoryview(backing)[2:], dtype=np.uint8)}
    with pytest.raises(ValueError, match="aliases Target"):
        validate(draft, target_metadata=target)
    private = isolate(draft, target)
    validate(private, target_metadata=target)
    identity = object()
    inputs = SimpleNamespace(identity=identity, validate_for=Mock())
    store = store_type()
    with pytest.raises(ValueError, match="aliases Target"):
        store.publish(inputs, {"array": target["array"][2:]}, target_metadata=target)
    store.publish(inputs, private, target_metadata=target)
    assert store.read(identity) is private


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
        # Real owner state transition, but no native operation is claimed.
        pool.begin_native(owner)
        with contract.wrapper(), contract.factory([0], 1, False, **callbacks):
            contract.module.build_attn_metadata()
    assert contract.module.build_attn_metadata is contract.original
    assert owner.state is DraftWindowOwnerState.QUARANTINED
    assert owner.retained == [payload]
    assert pool.poisoned
    completion.assert_not_called()
    with pytest.raises(RuntimeError, match="quarantined"):
        pool.acquire()
