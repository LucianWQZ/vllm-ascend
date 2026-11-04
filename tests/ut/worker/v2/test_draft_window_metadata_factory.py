# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Normal metadata ownership and existing factory contracts; CPU only."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.ut.worker.v2 import test_draft_window_runtime as runtime_tests

setup = runtime_tests.setup

CALLBACK_NAMES = ("common_metadata_callback",)


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


def test_metadata_copy_owns_header_and_borrows_read_only_operands(metadata_isolation):
    copy_metadata = metadata_isolation
    layer = _attention_metadata()
    private = copy_metadata({"draft": layer}, ("draft",))
    copied = private["draft"]
    assert copied is not layer
    for field in (
        "block_tables",
        "seq_lens",
        "seq_lens_cpu",
        "seq_lens_list",
        "actual_seq_lengths_q",
        "slot_mapping",
        "query_start_loc",
        "attn_mask",
        "qfa_metadata_cache",
    ):
        assert getattr(copied, field) is getattr(layer, field)
    copied.attn_state = "draft"
    copied.seq_lens_list = [99]
    assert layer.attn_state == "target"
    assert layer.seq_lens_list == [3, 4]
    assert layer.actual_seq_lengths_q == [1, 2]
    assert "lengths" in layer.qfa_metadata_cache


def test_real_ascend_metadata_preserves_canonical_full_operands(metadata_isolation):
    from vllm_ascend.attention.attention_v1 import AscendAttentionState, AscendMetadata

    copy_metadata = metadata_isolation
    layer = AscendMetadata(**vars(_attention_metadata()))
    private = copy_metadata({"draft": layer}, ("draft",))
    copied = private["draft"]
    copied.attn_state = AscendAttentionState.DecodeOnly
    copied.seq_lens_list = [9, 4]
    assert layer.attn_state == "target" and layer.seq_lens_list == [3, 4]
    assert copied.slot_mapping is layer.slot_mapping
    assert copied.query_start_loc is layer.query_start_loc


def test_metadata_copy_uses_the_actual_group_and_preserves_its_aliases(metadata_isolation):
    copy_metadata = metadata_isolation
    layer = _attention_metadata()
    private = copy_metadata({"a": layer, "b": layer, "target": _attention_metadata()}, ("a", "b"))
    assert set(private) == {"a", "b"}
    assert private["a"] is private["b"] and private["a"] is not layer


def test_real_builder_constructs_private_window_metadata_without_touching_target(monkeypatch):
    from vllm_ascend.attention.attention_v1 import (
        AscendAttentionMetadataBuilder,
        AscendAttentionState,
        AscendMetadata,
    )
    from vllm_ascend.worker.v2.spec_decode.draft_window_graph_metadata import (
        bind_draft_window_read,
        copy_draft_attention_metadata,
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
        cpu_window_exact=torch.tensor([7, 6], dtype=torch.int32),
    )
    # Native-prefill retains the real builder's query/write/state metadata;
    # only its private Draft copy receives the owned window read operands.
    draft = copy_draft_attention_metadata({"draft": target}, ("draft",))["draft"]
    bind_draft_window_read(draft, bank, 7)
    assert draft is not target
    assert draft.block_tables.data_ptr() == bank.window_block_table.data_ptr()
    assert draft.block_tables.shape == bank.window_block_table.shape
    assert draft.seq_lens_cpu.data_ptr() == bank.cpu_window_exact.data_ptr()
    assert draft.seq_lens is target.seq_lens and draft.seq_lens_gpu is target.seq_lens_gpu
    # The real FIA1 parameter consumer reads exact Host KV lengths, not these Full shape carriers.
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl

    impl = object.__new__(AscendAttentionBackendImpl)
    impl.key_cache, impl.value_cache = torch.empty(1), torch.empty(1)
    impl._get_kv_cache_view = lambda key, value: (key, value, 128)
    *_, actual_kv_lengths = impl._get_fia_params(torch.empty(1), torch.empty(1), draft)
    assert actual_kv_lengths == [7, 6]
    assert draft.seq_lens_list == [7, 6] and draft.seq_lens_list is not target.seq_lens_list
    assert draft.actual_seq_lengths_q is target.actual_seq_lengths_q
    assert draft.slot_mapping.data_ptr() == target.slot_mapping.data_ptr() == slots.data_ptr()
    assert draft.query_start_loc_gpu.data_ptr() == target.query_start_loc_gpu.data_ptr() == query.data_ptr()
    draft.seq_lens_list[0] = 99
    draft.actual_seq_lengths_q = [1, 2, 3]
    assert target.seq_lens_list == [11, 14] and target.actual_seq_lengths_q == [1, 2]
    assert target.block_tables is common.block_table_tensor is table
    assert target.seq_lens_cpu.data_ptr() == exact.data_ptr()
    assert common.max_seq_len == 14 and common.seq_lens.tolist() == [11, 14]


@pytest.fixture
def factory_contract(monkeypatch):
    """Use the real managers, isolating only their module and backend seam."""
    from vllm_ascend.worker.v2 import attn_utils

    original = Mock(side_effect=lambda **kwargs: kwargs)
    ascend = Mock(side_effect=lambda **kwargs: kwargs)
    module = SimpleNamespace(build_attn_metadata=original)
    monkeypatch.setattr(attn_utils, "_BUILD_ATTN_METADATA_MODULE", module)
    monkeypatch.setattr(attn_utils, "build_attn_metadata", ascend)
    return SimpleNamespace(
        module=module,
        original=original,
        ascend=ascend,
        factory=attn_utils.build_attn_metadata_factory,
        wrapper=attn_utils.build_attn_metadata_wrapper,
    )


@pytest.mark.parametrize("callback_names", [(), CALLBACK_NAMES])
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
        kwargs[stage](0, object())

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
def test_factory_error_restores_module_but_quarantines_active_owner(factory_contract, stage, monkeypatch, setup):
    contract = factory_contract
    runtime, _, _, _ = setup
    payload = object()
    completion = Mock(side_effect=AssertionError("failed proposal cannot record a success completion"))
    monkeypatch.setattr(torch.npu, "Event", completion)

    def fail(*args, **kwargs):
        raise RuntimeError("injected partial-submit host error")

    def backend(**kwargs):
        if stage == "backend":
            fail()
        kwargs[stage](0, object())

    contract.ascend.side_effect = backend
    callbacks = {} if stage == "backend" else {stage: fail}
    with (
        pytest.raises(RuntimeError, match="partial-submit host error"),
        runtime_tests.proposal_scope(runtime, runtime_tests.batch()),
    ):
        owner = runtime._owner
        assert owner is not None
        owner.append(payload)
        # Entered proposal lifetime; no native operation is claimed here.
        with contract.wrapper(), contract.factory([0], 1, False, **callbacks):
            contract.module.build_attn_metadata()
    assert contract.module.build_attn_metadata is contract.original
    assert runtime._poisoned
    assert owner == [payload]
    completion.assert_not_called()
    with pytest.raises(RuntimeError, match="quarantined"), runtime_tests.proposal_scope(runtime, runtime_tests.batch()):
        pytest.fail("uncertain builder must not be reused")
