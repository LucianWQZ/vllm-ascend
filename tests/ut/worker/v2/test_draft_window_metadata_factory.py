# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Normal metadata ownership and existing factory contracts; CPU only."""

import ast
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
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
    )

    return copy_draft_attention_metadata


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
    copy_metadata = metadata_isolation
    layer = _attention_metadata()
    private = copy_metadata({"draft": layer})
    copied = private["draft"]
    assert copied is not layer
    for field in ("block_tables", "seq_lens", "seq_lens_cpu", "slot_mapping", "query_start_loc", "attn_mask"):
        assert getattr(copied, field) is getattr(layer, field)
    # Cache payloads are read-only operator outputs; only the mutable cache
    # dictionary belongs to the per-build metadata object.
    assert copied.qfa_metadata_cache is not layer.qfa_metadata_cache
    assert copied.qfa_metadata_cache["lengths"] is layer.qfa_metadata_cache["lengths"]
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

    copy_metadata = metadata_isolation
    layer = AscendMetadata(**vars(_attention_metadata()))
    private = copy_metadata({"draft": layer})
    copied = private["draft"]
    copied.attn_state = AscendAttentionState.DecodeOnly
    copied.seq_lens_list[0] = 9
    assert layer.attn_state == "target" and layer.seq_lens_list == [3, 4]
    assert copied.slot_mapping is layer.slot_mapping
    assert copied.query_start_loc is layer.query_start_loc


def test_metadata_copy_preserves_group_aliases_and_missing_layers(metadata_isolation):
    copy_metadata = metadata_isolation
    layer = _attention_metadata()
    private = copy_metadata({"a": layer, "b": layer, "unused": None})
    assert private["a"] is private["b"] and private["a"] is not layer
    assert private["unused"] is None


def test_frozen_slotted_metadata_is_copied_before_field_replacement(metadata_isolation):
    copy_metadata = metadata_isolation

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


def test_metadata_copy_rejects_invalid_known_host_fields(metadata_isolation):
    copy_metadata = metadata_isolation
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


def test_real_builder_constructs_private_window_metadata_without_touching_target(monkeypatch):
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    from vllm_ascend.attention.attention_v1 import (
        AscendAttentionMetadataBuilder,
        AscendAttentionState,
        AscendMetadata,
    )
    from vllm_ascend.worker.v2.spec_decode.draft_window_eager import (
        build_draft_window_attn_metadata,
        draft_window_read_inputs,
    )

    # Only pinning and backend phase/mask dependencies are isolated. The real
    # builder creates both metadata objects and all mutable Host containers.
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda tensor: tensor)
    builder = SimpleNamespace(
        use_pcp=False,
        device=torch.device("cpu"),
        kv_cache_spec=object(),
        speculative_config=None,
        model_config=SimpleNamespace(runner_type="generate"),
        vllm_config=SimpleNamespace(use_v2_model_runner=True, model_config=object()),
        attn_mask_builder=SimpleNamespace(get_attention_mask=lambda *args: None),
        _split_decodes_and_prefills=lambda common: (2, 0, 2, 0),
        _build_backend_metadata=lambda *args, **kwargs: {},
        metadata_cls=AscendMetadata,
    )
    exact = torch.tensor([11, 14], dtype=torch.int32)
    query = torch.tensor([0, 1, 2], dtype=torch.int32)
    slots = torch.tensor([40, 80], dtype=torch.int64)
    table = torch.tensor([[11, 12, 13, 14], [21, 22, 23, 24]], dtype=torch.int32)
    common = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=2,
        num_input_tokens=2,
        query_start_loc=query,
        query_start_loc_cpu=query,
        max_query_len=1,
        attn_state=AscendAttentionState.DecodeOnly,
        is_prefilling=torch.tensor([False, False]),
        causal=True,
        block_table_tensor=table,
        slot_mapping=slots,
        seq_lens=exact,
        seq_lens_cpu=exact,
        _seq_lens_cpu=exact,
        seq_lens_cpu_upper_bound=exact,
        max_seq_len=14,
    )
    target = AscendAttentionMetadataBuilder.build(builder, 0, common)
    bank = SimpleNamespace(
        window_block_table=torch.tensor([[12, 13], [23, 24]], dtype=torch.int32),
        device_window_exact=torch.tensor([7, 6], dtype=torch.int32),
        cpu_window_exact=torch.tensor([7, 6], dtype=torch.int32),
    )
    builder.build = MethodType(AscendAttentionMetadataBuilder.build, builder)
    group = SimpleNamespace(layer_names=["draft"], get_metadata_builder=lambda _: builder)
    owner = SimpleNamespace(
        num_query_per_req=1,
        arange_np=np.arange(3, dtype=np.int32),
        dcp_size=1,
        max_model_len=16,
        draft_max_seq_len=16,
        draft_is_prefilling=torch.zeros(2, dtype=torch.bool),
        attn_groups=[[group]],
        input_buffers=SimpleNamespace(query_start_loc=query, seq_lens=exact, positions=torch.arange(2)),
        input_batch=SimpleNamespace(
            is_prefilling_np=np.array([False, False]), attn_state=AscendAttentionState.DecodeOnly
        ),
        block_tables=SimpleNamespace(input_block_tables=[table], slot_mappings=slots.unsqueeze(0)),
        attn_vllm_config=SimpleNamespace(parallel_config=None),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    kv_cache_spec=FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=16, dtype=torch.float16)
                )
            ]
        ),
        _draft_window_graph_binding_provider=SimpleNamespace(
            source=SimpleNamespace(group=SimpleNamespace(gid=0, layers=("draft",)))
        ),
    )
    draft = build_draft_window_attn_metadata(
        owner,
        num_reqs=2,
        num_reqs_padded=2,
        num_tokens_padded=2,
        read_inputs=draft_window_read_inputs(bank, actual_request_count=2, padded_request_count=2),
        causal=True,
        query_start_loc_np=None,
    )["draft"]
    assert draft is not target
    assert draft.block_tables.data_ptr() == bank.window_block_table.data_ptr()
    assert draft.block_tables.shape == bank.window_block_table.shape
    assert draft.seq_lens_cpu.data_ptr() == bank.cpu_window_exact.data_ptr()
    assert draft.seq_lens_list == [7, 6] and draft.seq_lens_list is not target.seq_lens_list
    assert draft.actual_seq_lengths_q is not target.actual_seq_lengths_q
    assert draft.slot_mapping.data_ptr() == target.slot_mapping.data_ptr() == slots.data_ptr()
    assert draft.query_start_loc_gpu.data_ptr() == target.query_start_loc_gpu.data_ptr() == query.data_ptr()
    draft.seq_lens_list[0] = 99
    draft.actual_seq_lengths_q.append(3)
    assert target.seq_lens_list == [11, 14] and target.actual_seq_lengths_q == [1, 2]
    assert target.block_tables is common.block_table_tensor is table
    assert target.seq_lens_cpu.data_ptr() == exact.data_ptr()
    assert common.max_seq_len == 14 and common.seq_lens.tolist() == [11, 14]


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
