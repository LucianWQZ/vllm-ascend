# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Actual capture banks and proposal boundaries, without retired DTO protocols."""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBuffers

from vllm_ascend.worker.v2.spec_decode.draft_window_backends import DraftWindowGroupRoute, DraftWindowProviderStatus
from vllm_ascend.worker.v2.spec_decode.draft_window_eager import DraftWindowGraphInputMixin
from vllm_ascend.worker.v2.spec_decode.draft_window_graph import DraftWindowGraphBindingProvider
from vllm_ascend.worker.v2.spec_decode.draft_window_plan import compute_draft_window_plan
from vllm_ascend.worker.v2.spec_decode.draft_window_policy import resolve_draft_window_policy


def _proposal(n=2):
    return (False,) * n


@pytest.fixture
def graph_buffers():
    table = torch.tensor([[11, 12, 13, 14], [21, 22, 23, 24]], dtype=torch.int32)
    counts = torch.full((1, 2), 4, dtype=torch.int32)
    owner = SimpleNamespace(
        input_block_tables=[table],
        num_blocks=SimpleNamespace(cpu=counts.clone(), gpu=counts),
        slot_mappings=torch.tensor([[40, 80]]),
        block_sizes=(4,),
        kernel_block_sizes=(4,),
    )
    group = DraftWindowGroupRoute(0, ("draft.layer",), 4, DraftWindowProviderStatus.DIRECT_DENSE)
    provider = DraftWindowGraphBindingProvider(
        policy=resolve_draft_window_policy(requested_window_size=4, method="dflash"), group=group, block_tables=owner
    )
    desc = BatchExecutionDescriptor(CUDAGraphMode.FULL, 2, 2, uniform_token_count=1)
    capture = provider.register_capture_identity(descriptor=desc)
    buffers = object.__new__(InputBuffers)
    buffers.positions = torch.arange(8)
    return SimpleNamespace(
        provider=provider,
        capture=capture,
        owner=owner,
        input_buffers=buffers,
        mapping=torch.tensor([0, 1]),
        target_metadata={"target.layer": SimpleNamespace(seq_lens_list=[91, 92], actual_seq_lengths_q=[1, 2])},
    )


@pytest.mark.parametrize("damage", [None, "owner", "route"])
def test_parallel_install_qualifies_the_real_owner_before_any_capture_bank(graph_buffers, damage):
    from vllm_ascend.worker.v2.spec_decode.draft_window_graph_runtime import install_parallel_graph

    c = graph_buffers
    provider = DraftWindowGraphBindingProvider(
        policy=resolve_draft_window_policy(requested_window_size=4, method="eagle3" if damage == "route" else "dflash"),
        group=c.provider.group,
        block_tables=c.owner,
    )
    owner = SimpleNamespace(
        _draft_window_graph_binding_provider=object() if damage == "owner" else provider,
        query_cudagraph_manager=SimpleNamespace(_capture_descs={CUDAGraphMode.FULL: [c.capture.descriptor]}),
        speculative_config=SimpleNamespace(num_speculative_tokens=1),
    )
    if damage:
        with pytest.raises(RuntimeError, match="initialized source owner|parallel Draft route"):
            install_parallel_graph(owner)
        assert not provider._carriers
    else:
        assert install_parallel_graph(owner) is provider
        bank = provider.capture_identity_for(descriptor=c.capture.descriptor)
        assert bank.owner is provider and bank.descriptor is c.capture.descriptor


def _prepare(case, ends=(11, 14), *, capture=None, step=0, plans=None, valid=None):
    bank = case.capture if capture is None else capture
    if valid is not None:
        case.owner.num_blocks.cpu[0, : len(valid)] = torch.tensor(valid)
    return case.provider.prepare(
        bank,
        cpu_full_exact=torch.tensor(ends, dtype=torch.int32),
        idx_mapping_np=np.arange(len(ends), dtype=np.int64),
        step=step,
        frozen_row_plans=plans,
    )


def _tensors(bank):
    return bank.window_block_table, bank.device_window_exact, bank.cpu_window_exact


def _resolve(provider, count=1):
    from vllm_ascend.attention.attention_v1 import FIAParamProvider
    from vllm_ascend.compilation.updatable_graph import GraphUpdateTask, UpdatableGraph

    p = FIAParamProvider(provider.group.layers[0], None, is_draft_model=True)
    tasks = [GraphUpdateTask(object(), {}, p, i, object(), object()) for i in range(count)]
    graph = SimpleNamespace(provider_sizes={p: count}, tasks=tasks)
    return graph, UpdatableGraph.resolve_tasks(graph, provider)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_parallel_exact_is_an_independent_completed_cpu_snapshot(graph_buffers, monkeypatch, dtype):
    c = graph_buffers
    owner = DraftWindowGraphInputMixin()
    owner._draft_window_graph_binding_provider = c.provider
    owner.input_buffers = SimpleNamespace(seq_lens=torch.tensor([11, 14], dtype=dtype))
    owner.input_batch = SimpleNamespace(num_reqs=2, idx_mapping_np=np.array([0, 1]))
    owner._draft_window_bypass_active = False
    owner.draft_attn_layer_names, owner.num_query_per_req = {"draft.layer"}, 1
    owner.block_tables = c.owner
    prepared = []
    monkeypatch.setattr(c.provider, "prepare", lambda capture, **kwargs: prepared.append(kwargs["cpu_full_exact"]))
    with c.provider.proposal_scope(_proposal()):
        owner._build_attn_metadata(2, c.capture.descriptor, np.arange(3, dtype=np.int32), torch.tensor([11, 14]), 1)
    exact = prepared[0]
    assert exact.device.type == "cpu" and exact.dtype == torch.int32
    owner.input_buffers.seq_lens.zero_()
    assert exact.tolist() == [11, 14]


def test_count_gather_uses_request_slots_and_leaves_padding_inert(graph_buffers):
    c = graph_buffers
    c.owner.num_blocks.cpu[0] = torch.tensor([1, 4])
    with c.provider.proposal_scope(_proposal(1)):
        bank = c.provider.prepare(c.capture, cpu_full_exact=torch.tensor([14]), idx_mapping_np=np.array([1]), step=0)
        assert bank.cpu_window_exact.tolist() == [6, 0]
        assert bank.window_block_table.tolist() == [[13, 14, 0, 0], [0, 0, 0, 0]]
        c.owner.num_blocks.cpu.zero_()
        assert bank.cpu_window_exact.tolist() == [6, 0]
        c.provider.bind_current(bank.descriptor)
        c.provider.finish_update()


@pytest.mark.parametrize("mapping", [np.array([-1]), np.array([2]), np.array([], dtype=np.int64)])
def test_bad_live_request_mapping_is_rejected_before_any_bank_write(graph_buffers, mapping):
    c = graph_buffers
    before = [tensor.clone() for tensor in _tensors(c.capture)]
    with pytest.raises((IndexError, ValueError)), c.provider.proposal_scope(_proposal(1)):
        c.provider.prepare(c.capture, cpu_full_exact=torch.tensor([14]), idx_mapping_np=mapping, step=0)
    assert c.provider.quarantined
    for original, actual in zip(before, _tensors(c.capture)):
        torch.testing.assert_close(actual, original)


def test_real_scope_reuses_addresses_clears_tail_and_preserves_full_and_target(graph_buffers):
    c = graph_buffers
    addresses = [t.data_ptr() for t in _tensors(c.capture)]
    full, slots = c.provider.table.clone(), c.owner.slot_mappings.clone()
    previous = None
    for ends in ((11, 14), (3,)):
        context = _proposal(len(ends))
        with c.provider.proposal_scope(context):
            bank = _prepare(c, ends)
            assert bank is c.capture and c.provider.proposal is context and bank is c.provider._active_bank
            assert bank.params is None
            c.provider.bind_current(bank.descriptor)
            assert bank.params["block_table"] is bank.window_block_table
            assert bank.params["actual_seq_lengths_kv"] == bank.cpu_window_exact.tolist()
            if previous is not None:
                assert bank.params is not previous
                assert not bank.window_block_table[1].count_nonzero()
                assert bank.cpu_window_exact.tolist() == [3, 0]
            previous = bank.params
            _resolve(c.provider)
            c.provider.finish_update()
            assert c.provider._active_bank is bank  # Host UPDATE is not reader completion.
        assert c.provider.proposal is None and c.provider._active_bank is None
    assert [t.data_ptr() for t in _tensors(c.capture)] == addresses
    assert torch.equal(c.provider.table, full) and torch.equal(c.owner.slot_mappings, slots)
    assert c.target_metadata["target.layer"].seq_lens_list == [91, 92]


@pytest.mark.parametrize("failure", ["unbound", "pending", "after_update", "prepare"])
def test_actual_scope_quarantines_partial_or_uncertain_submission(graph_buffers, failure):
    c = graph_buffers
    with pytest.raises(RuntimeError), c.provider.proposal_scope(_proposal()):
        bank = _prepare(c)
        if failure != "unbound":
            c.provider.bind_current(bank.descriptor)
        if failure in ("after_update", "prepare"):
            c.provider.finish_update()
        if failure == "prepare":
            _prepare(c, valid=[1, 1])
        if failure in ("after_update", "prepare"):
            raise RuntimeError("after native work")
    assert c.provider.quarantined and c.provider._active_bank is c.capture
    with pytest.raises(RuntimeError, match="quarantined"), c.provider.proposal_scope(_proposal()):
        pass


def test_prepare_and_publish_reject_unbound_pending_wrong_descriptor_and_foreign_bank(graph_buffers):
    c = graph_buffers
    other = DraftWindowGraphBindingProvider(
        policy=resolve_draft_window_policy(requested_window_size=c.provider.window_size, method=c.provider.method),
        group=c.provider.group,
        block_tables=c.owner,
    )
    foreign = other.register_capture_identity(descriptor=c.capture.descriptor)
    with c.provider.proposal_scope(_proposal()):
        before = c.capture.window_block_table.clone()
        with pytest.raises(RuntimeError, match="owning proposal"):
            c.provider.prepare(foreign, cpu_full_exact=torch.tensor([11, 14]), idx_mapping_np=np.array([0, 1]), step=0)
        assert torch.equal(c.capture.window_block_table, before)
        bank = _prepare(c)
        with pytest.raises(RuntimeError, match="still active"):
            _prepare(c)
        with pytest.raises(RuntimeError, match="unconsumed preparation"):
            c.provider.bind_current(replace(bank.descriptor, max_query_len=9))
        c.provider.bind_current(bank.descriptor)
        with pytest.raises(RuntimeError, match="unconsumed preparation"):
            c.provider.bind_current(bank.descriptor)
        with pytest.raises(RuntimeError, match="pending"):
            _prepare(c)
        c.provider.finish_update()
    with pytest.raises(RuntimeError, match="unconsumed preparation"):
        c.provider.bind_current(c.capture.descriptor)


def test_actual_scope_rejects_nesting_and_stream_changes_before_bank_writes(graph_buffers, monkeypatch):
    c = graph_buffers
    c.owner.input_block_tables[0] = SimpleNamespace(device=SimpleNamespace(type="npu"))
    current = SimpleNamespace(npu_stream=17)
    monkeypatch.setattr(torch.npu, "current_stream", lambda: current)
    with c.provider.proposal_scope(_proposal()):
        with pytest.raises(RuntimeError, match="already active"), c.provider.proposal_scope(_proposal()):
            pytest.fail("nested proposal reached preparation")
        current.npu_stream = 19
        before = tuple(t.clone() for t in _tensors(c.capture))
        with pytest.raises(RuntimeError, match="changed stream"):
            _prepare(c)
        assert c.provider._active_bank is None
        assert all(torch.equal(a, b) for a, b in zip(before, _tensors(c.capture)))


@pytest.mark.parametrize("exact,valid", [([-1, 2], [4, 4]), ([11, 14], [1, 1]), ([11, 14], [5, 4])])
def test_current_geometry_fails_before_first_bank_write(graph_buffers, exact, valid):
    c = graph_buffers
    c.capture.window_block_table.fill_(87)
    c.capture.cpu_window_exact.fill_(59)
    with pytest.raises((ValueError, RuntimeError)), c.provider.proposal_scope(_proposal()):
        _prepare(c, exact, valid=valid)
    assert bool((c.capture.window_block_table == 87).all())
    assert c.capture.cpu_window_exact.tolist() == [59, 59]
    assert c.provider.quarantined


@pytest.mark.parametrize(
    "field,value", [("max_query_len", 5), ("uniform_token_count", None), ("num_active_loras", 1), ("num_ubatches", 2)]
)
def test_real_dispatch_requires_the_complete_registered_descriptor(graph_buffers, field, value):
    c = graph_buffers
    with pytest.raises(RuntimeError):
        c.provider.capture_identity_for(descriptor=replace(c.capture.descriptor, **{field: value}))


def test_capture_geometry_and_ordinal_are_unique_and_bounded(graph_buffers):
    c = graph_buffers
    assert c.provider.capture_banks_for(num_reqs=2, num_tokens=2) == (c.capture,)
    with pytest.raises(RuntimeError, match="ambiguous"):
        c.provider.register_capture_identity(
            descriptor=replace(c.capture.descriptor, max_query_len=9), consumer_ordinal=1
        )
    with pytest.raises(RuntimeError, match="serial ordinal"):
        c.provider.register_capture_identity(descriptor=c.capture.descriptor, consumer_ordinal=2)
    with pytest.raises((ValueError, NotImplementedError)):
        c.provider.register_capture_identity(descriptor=replace(c.capture.descriptor, num_active_loras=1))


def test_eagle_keeps_distinct_ordinals_and_one_frozen_anchor(graph_buffers):
    c = graph_buffers
    provider = DraftWindowGraphBindingProvider(
        policy=resolve_draft_window_policy(requested_window_size=4, method="eagle3"),
        group=c.provider.group,
        block_tables=c.owner,
    )
    desc = BatchExecutionDescriptor(CUDAGraphMode.FULL, 2, 2, uniform_token_count=1)
    banks = [provider.register_capture_identity(descriptor=desc, consumer_ordinal=i) for i in range(2)]
    plans = tuple(
        compute_draft_window_plan(full_exact_end=n, future_offset=3, window_size=4, block_span_tokens=4)
        for n in (11, 14)
    )
    c.provider = provider
    with provider.proposal_scope(_proposal()):
        first = _prepare(c, capture=banks[0], step=1, plans=plans)
        provider.bind_current(desc)
        first_values = [t.clone() for t in _tensors(first)]
        second = _prepare(c, (12, 15), capture=banks[1], step=2, plans=plans)
        provider.bind_current(desc)
        assert all(torch.equal(a, b) for a, b in zip(first_values, _tensors(first)))
        assert all(a.data_ptr() != b.data_ptr() for a, b in zip(_tensors(first), _tensors(second)))
        _, actual = _resolve(provider, 2)
        assert [task.kwargs["actual_seq_lengths_kv"] for task in actual] == [
            first.cpu_window_exact.tolist(),
            second.cpu_window_exact.tolist(),
        ]
        with pytest.raises(AssertionError):
            _resolve(provider, 1)
        provider.finish_update()


@pytest.mark.parametrize("span,boundary", [(4, 8), (128, 1024), (256, 2048)])
def test_bank_projects_only_initialized_columns_across_boundaries(span, boundary):
    table = torch.arange(64, dtype=torch.int32).reshape(1, 64)
    tables = SimpleNamespace(
        input_block_tables=[table],
        num_blocks=SimpleNamespace(cpu=torch.tensor([[64]])),
        block_sizes=(span,),
        kernel_block_sizes=(span,),
    )
    group = DraftWindowGroupRoute(0, ("draft",), span, DraftWindowProviderStatus.DIRECT_DENSE)
    provider = DraftWindowGraphBindingProvider(
        policy=resolve_draft_window_policy(requested_window_size=boundary, method="dflash"),
        group=group,
        block_tables=tables,
    )
    desc = BatchExecutionDescriptor(CUDAGraphMode.FULL, 1, 1, uniform_token_count=1)
    bank = provider.register_capture_identity(descriptor=desc)
    for exact in (boundary - 1, boundary, boundary + 1, 2 * boundary + 1):
        plan = compute_draft_window_plan(full_exact_end=exact, window_size=boundary, block_span_tokens=span)
        with provider.proposal_scope(_proposal(1)):
            provider.prepare(bank, cpu_full_exact=torch.tensor([exact]), idx_mapping_np=np.array([0]), step=0)
            start = plan.window_start // span
            count = (plan.window_visible_length + span - 1) // span
            assert torch.equal(bank.window_block_table[0, :count], table[0, start : start + count])
            assert not bank.window_block_table[0, count:].count_nonzero()
            assert bank.cpu_window_exact.tolist() == [plan.window_visible_length]
            provider.bind_current(desc)
            provider.finish_update()


def test_parallel_capture_uses_registered_geometry_and_preserves_full_tables(graph_buffers):
    from vllm_ascend.worker.v2.spec_decode.dflash.capture import (
        _seed_capture_workspace,
        draft_window_dflash_capture_scope,
    )

    case = graph_buffers
    buffers = object.__new__(InputBuffers)
    owner = SimpleNamespace(
        input_buffers=buffers,
        _draft_window_graph_binding_provider=case.provider,
        model_state=SimpleNamespace(attn_metadata=case.target_metadata),
    )
    with draft_window_dflash_capture_scope(owner, buffers) as context:
        assert context is not None
        identity = context.capture_banks_for(num_reqs=2, num_tokens=2)[0]
        read = identity
        assert identity is case.capture
        full = case.provider.table
        before = full.clone()
        maximum = _seed_capture_workspace(read, max_model_len=16, provider=case.provider)
        assert maximum == 7 and read.cpu_window_exact.tolist() == [7, 7]
        assert read.device_window_exact.tolist() == [7, 7]
        assert torch.equal(full, before)
        assert torch.equal(read.window_block_table[:, :2], full[:, :2])
        with pytest.raises(RuntimeError, match="no captured consumer for this geometry"):
            context.capture_banks_for(num_reqs=1, num_tokens=2)


@pytest.mark.parametrize("capture_fails", [False, True])
def test_parallel_capture_scope_owns_its_attachment_until_normal_or_error_exit(graph_buffers, capture_fails):
    from vllm_ascend.worker.v2.spec_decode.dflash import capture

    case = graph_buffers
    buffers = case.input_buffers
    owner = SimpleNamespace(input_buffers=buffers, _draft_window_graph_binding_provider=case.provider)
    full_before = case.owner.input_block_tables[0].clone()
    slots_before = case.owner.slot_mappings.clone()
    try:
        with capture.draft_window_dflash_capture_scope(owner, buffers) as context:
            assert getattr(buffers, capture._CAPTURE_CONTEXT_ATTRIBUTE) is context
            assert context is not None
            bucket = context.capture_banks_for(num_reqs=2, num_tokens=2)[0]
            bank = bucket
            assert bucket is case.capture and bank is case.capture
            with (
                pytest.raises(RuntimeError, match="already attached"),
                capture.draft_window_dflash_capture_scope(owner, buffers),
            ):
                pytest.fail("nested capture must not start")
            if capture_fails:
                raise RuntimeError("capture failed")
    except RuntimeError as error:
        assert capture_fails and str(error) == "capture failed"
    assert not hasattr(buffers, capture._CAPTURE_CONTEXT_ATTRIBUTE)
    assert torch.equal(case.owner.input_block_tables[0], full_before)
    assert torch.equal(case.owner.slot_mappings, slots_before)


def test_window_capture_delegates_original_dummy_preparation_then_only_rebinds_reads(graph_buffers, monkeypatch):
    from vllm.v1.worker.gpu.cudagraph_utils import AttentionState

    from vllm_ascend.worker.v2.spec_decode.dflash import capture

    case = graph_buffers
    buffers = object.__new__(InputBuffers)
    buffers.query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
    owner = SimpleNamespace(input_buffers=buffers, _draft_window_graph_binding_provider=case.provider)
    slots, state_marker = object(), object()
    layer = SimpleNamespace(slot_mapping=slots, actual_seq_lengths_q=[1, 2], attn_state=state_marker, causal=False)
    state = AttentionState({"draft.layer": layer}, {"draft.layer": slots})
    calls = []

    def upstream(*args):
        calls.append(args)
        return state

    monkeypatch.setattr(capture, "_ORIGINAL_PREPARE_DFLASH_INPUTS_TO_CAPTURE", upstream)
    full = case.owner.input_block_tables[0].clone()
    with capture.draft_window_dflash_capture_scope(owner, buffers):
        result = capture.ascend_prepare_dflash_inputs_to_capture(
            2, 2, buffers, case.owner, [], object(), 16, False, False
        )
    assert result is state and len(calls) == 1
    bank = case.capture
    assert layer.block_tables is bank.window_block_table
    assert layer.seq_lens is layer.seq_lens_gpu is bank.device_window_exact
    assert layer.seq_lens_cpu is bank.cpu_window_exact
    assert layer.seq_lens_list == [7, 7] and layer.max_seq_len == 7
    assert layer.query_start_loc.data_ptr() == buffers.query_start_loc.data_ptr()
    assert layer.slot_mapping is slots and layer.actual_seq_lengths_q == [1, 2]
    assert layer.attn_state is state_marker and layer.causal is False
    assert result.slot_mappings["draft.layer"] is slots
    assert torch.equal(case.owner.input_block_tables[0], full)


def test_context_only_capture_preserves_upstream_preparation_without_a_read_bank(graph_buffers, monkeypatch):
    from vllm_ascend.worker.v2.spec_decode.dflash import capture

    case = graph_buffers
    buffers = SimpleNamespace(_ascend_draft_window_capture_context=object())
    arguments: tuple[Any, ...] = (2, 2, buffers, object(), [], object(), 16, True, False)
    expected = object()
    received = []

    def upstream(*args):
        received.append(args)
        return expected

    monkeypatch.setattr(capture, "_ORIGINAL_PREPARE_DFLASH_INPUTS_TO_CAPTURE", upstream)
    banks = tuple(case.provider._carriers)
    assert capture.ascend_prepare_dflash_inputs_to_capture(*arguments) is expected
    assert received == [arguments] and tuple(case.provider._carriers) == banks
    assert case.provider._active_bank is None


@pytest.mark.parametrize(
    "dummy,profile,dp,captures,prefill,expected",
    [
        (True, False, 2, True, False, True),
        (True, True, 2, True, False, False),
        (True, False, 1, True, False, False),
        (True, False, 2, False, False, False),
        (False, False, 2, True, True, False),
    ],
)
def test_parallel_idle_marker_is_owned_by_real_scope_not_prefill_or_profile(
    graph_buffers, dummy, profile, dp, captures, prefill, expected
):
    from vllm_ascend.worker.v2.spec_decode.draft_window_eager import DraftWindowGraphInputMixin

    case = graph_buffers
    owner = object.__new__(DraftWindowGraphInputMixin)
    owner._draft_window_graph_binding_provider = case.provider
    owner._draft_window_bypass_active = False
    owner._draft_window_prefill_bypass_active = False
    owner._draft_window_dp_dummy_active = False
    owner.dp_size = dp
    if not captures:
        case.provider._carriers.clear()
    batch = SimpleNamespace(
        num_reqs=2,
        num_computed_prefill_tokens_np=np.array([0, 0]),
        prefill_len_np=np.array([100, 100]),
        num_scheduled_tokens=np.array([1, 1]),
        num_draft_tokens_per_req=None,
        is_prefilling_np=np.array([True, True]),
        has_prefill=True,
    )
    with owner._draft_window_proposal_scope(batch, dummy_run=dummy, is_profile=profile):
        assert owner._draft_window_dp_dummy_active is expected
        assert owner._draft_window_prefill_bypass_active is prefill
        assert case.provider.proposal is None and case.provider._active_bank is None
    assert owner._draft_window_dp_dummy_active is False
    assert owner._draft_window_prefill_bypass_active is False
    assert not case.provider.quarantined


def test_parallel_idle_failure_quarantines_before_next_real_scope(graph_buffers):
    from vllm_ascend.worker.v2.spec_decode.draft_window_eager import DraftWindowGraphInputMixin

    owner = object.__new__(DraftWindowGraphInputMixin)
    owner._draft_window_graph_binding_provider = graph_buffers.provider
    owner._draft_window_bypass_active = False
    owner._draft_window_prefill_bypass_active = False
    owner._draft_window_dp_dummy_active = False
    owner.dp_size = 2
    with (
        pytest.raises(RuntimeError, match="idle UPDATE failed"),
        owner._draft_window_proposal_scope(object(), dummy_run=True, is_profile=False),
    ):
        assert owner._draft_window_dp_dummy_active is True
        raise RuntimeError("idle UPDATE failed")
    assert graph_buffers.provider.quarantined and not owner._draft_window_dp_dummy_active
    with (
        pytest.raises(RuntimeError, match="quarantined"),
        owner._draft_window_proposal_scope(object(), dummy_run=True, is_profile=False),
    ):
        pytest.fail("an uncertain idle Graph must not be submitted again")


@pytest.mark.parametrize("field", ["provider", "operation", "is_non_pa"])
def test_capture_contract_rejects_an_unsupported_consumer(graph_buffers, field):
    import torch_npu

    from vllm_ascend.attention.attention_v1 import FIAParamProvider
    from vllm_ascend.compilation.updatable_graph import GraphUpdateTask

    c = graph_buffers
    p = FIAParamProvider("draft.layer", None, is_draft_model=True)
    operation = torch_npu.npu_fused_infer_attention_score.out
    if field == "provider":
        p = object()
    elif field == "operation":
        operation = object()
    else:
        p = replace(p, is_non_pa=True)
    graph = SimpleNamespace(tasks=[GraphUpdateTask(operation, {}, p, 0, object(), object())])
    with pytest.raises(RuntimeError, match="supported Draft FIA"):
        c.provider.validate_captured_graphs({c.capture.descriptor: graph})


@pytest.mark.parametrize("window,idle", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("fails", [False, True])
def test_actual_manager_keeps_resolve_wait_replay_update_order(graph_buffers, monkeypatch, window, idle, fails):
    from contextlib import nullcontext

    from vllm_ascend.attention.attention_v1 import FIAParamProvider
    from vllm_ascend.compilation.updatable_graph import GraphUpdateTask, UpdatableGraph
    from vllm_ascend.worker.v2.spec_decode.dflash import aclgraph

    c = graph_buffers
    owner = DraftWindowGraphInputMixin()
    owner.attn_backends = {"draft.layer": object()}
    owner.input_batch = SimpleNamespace(seq_lens_cpu_upper_bound=torch.tensor([16, 16]))
    owner.build_draft_attn_metadatas = Mock(
        return_value=[
            {
                "draft.layer": SimpleNamespace(
                    actual_seq_lengths_q=[1, 2], seq_lens_list=[11, 14], block_tables=c.provider.table
                )
            }
        ]
    )
    scope = nullcontext()
    if window:
        owner._draft_window_graph_binding_provider = c.provider
        owner._draft_window_bypass_active = False
        owner._draft_window_prefill_bypass_active = False
        owner._draft_window_dp_dummy_active = False
        if idle:
            owner.dp_size = 2
            scope = owner._draft_window_proposal_scope(object(), dummy_run=True, is_profile=False)
        else:
            scope = c.provider.proposal_scope(_proposal())
    task_provider = FIAParamProvider("draft.layer", None, is_draft_model=True)
    task = GraphUpdateTask(object(), {}, task_provider, 0, object(), object())
    events, resolved = [], []

    class Graph:
        tasks = [task]
        provider_sizes = {task_provider: 1}

        def resolve_tasks(self, source):
            events.append("resolve")
            resolved.extend(UpdatableGraph.resolve_tasks(self, source))
            return resolved

        def update(self, stream, tasks):
            assert tasks[0] is resolved[0]
            events.append("update")
            if fails:
                raise RuntimeError("UPDATE failed")

    manager = object.__new__(aclgraph.DFlashAclGraphManager)
    manager.speculator, manager.graphs = owner, {c.capture.descriptor: Graph()}
    manager.update_stream = SimpleNamespace(wait_stream=lambda _: events.append("wait"))
    monkeypatch.setattr(aclgraph, "UpdatableGraph", Graph)
    monkeypatch.setattr(aclgraph, "use_updatable_graph", lambda _: True)
    monkeypatch.setattr(torch.npu, "current_stream", lambda: object())
    output = object()
    monkeypatch.setattr(aclgraph.DFlashCudaGraphManager, "run_fullgraph", lambda *_: events.append("replay") or output)
    try:
        with scope:
            if window and not idle:
                _prepare(c)
            assert manager.run_fullgraph(c.capture.descriptor) is output
    except RuntimeError:
        assert fails
    else:
        assert not fails
    assert events == ["resolve", "wait", "replay", "update"]
    assert resolved[0].handle is task.handle and resolved[0].event is task.event
    assert resolved[0].kwargs["actual_seq_lengths_kv"] == ([7, 6] if window and not idle else [11, 14])
    if window:
        assert c.provider.quarantined is fails
        assert bool(c.provider._pending) is (fails and not idle)
    assert c.provider.table.tolist() == [[11, 12, 13, 14], [21, 22, 23, 24]]
    assert c.owner.slot_mappings.tolist() == [[40, 80]]
