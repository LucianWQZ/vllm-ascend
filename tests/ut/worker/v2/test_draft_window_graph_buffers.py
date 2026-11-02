# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU buffer ownership contracts; these do not assert NPU completion."""

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBuffers

from vllm_ascend.worker.v2.spec_decode.draft_window import (
    DraftWindowGraphSource,
    DraftWindowWorkspace,
    gather_valid_num_blocks_cpu,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_backends import (
    DraftWindowGroupRoute,
    DraftWindowProviderStatus,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_eager import DraftWindowProposalContext
from vllm_ascend.worker.v2.spec_decode.draft_window_graph import (
    DraftWindowGraphBindingProvider,
    DraftWindowGraphRoute,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_plan import compute_draft_window_plan
from vllm_ascend.worker.v2.spec_decode.draft_window_policy import resolve_draft_window_policy


def _proposal(epoch=0, requests=("request-a",)):
    return DraftWindowProposalContext(object(), (False,) * len(requests))


@pytest.fixture
def graph_buffers():
    input_buffers = object.__new__(InputBuffers)
    input_buffers.positions = torch.arange(8, dtype=torch.int64)
    table = torch.tensor([[11, 12, 13, 14], [21, 22, 23, 24]], dtype=torch.int32)
    counts = torch.full((1, 2), 4, dtype=torch.int32)
    owner = SimpleNamespace(
        input_block_tables=[table],
        num_blocks=SimpleNamespace(cpu=counts.clone(), gpu=counts),
        slot_mappings=torch.tensor([[40, 80]], dtype=torch.int64),
        block_sizes=(4,),
        kernel_block_sizes=(4,),
    )
    group = DraftWindowGroupRoute(
        gid=0,
        layers=("draft.layer",),
        scheduler_block_size=4,
        table_column_span=4,
        status=DraftWindowProviderStatus.DIRECT_DENSE,
    )
    projection = DraftWindowGraphSource(
        policy=resolve_draft_window_policy(requested_window_size=4, method="dflash", target_model_type="qwen3"),
        group=group,
        block_tables=owner,
    )
    provider = DraftWindowGraphBindingProvider(source=projection)
    descriptor = BatchExecutionDescriptor(CUDAGraphMode.FULL, num_tokens=2, num_reqs=2, uniform_token_count=1)
    capture = provider.register_capture_identity(route=DraftWindowGraphRoute.DFLASH, descriptor=descriptor)
    return SimpleNamespace(
        provider=provider,
        capture=capture,
        projection=projection,
        input_buffers=input_buffers,
        mapping=torch.tensor([0, 1], dtype=torch.int64),
        source=projection,
        owner=owner,
        target_metadata={"target.layer": SimpleNamespace(seq_lens_list=[91, 92], actual_seq_lengths_q=[1, 2])},
    )


def _materialize(case, *, epoch=0, ends=(11, 14)):
    exact = torch.tensor(list(ends) + [0] * (2 - len(ends)), dtype=torch.int32)
    return case.provider.materialize(
        proposal=_proposal(epoch, ("request-a", "request-b")[: len(ends)]),
        cpu_full_exact=exact,
        cpu_valid_num_blocks=torch.tensor([4] * len(ends) + [0] * (2 - len(ends)), dtype=torch.int32),
        num_reqs_padded=2,
        step=0,
        output_workspace=case.provider.prepare_workspace(case.capture),
    )


def _bank_tensors(bank):
    return (
        bank.window_block_table,
        bank.device_window_exact,
        bank.cpu_window_exact,
    )


def _bind(case, inputs):
    return case.provider.bind_read_carrier(capture_identity=case.capture, inputs=inputs)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_parallel_exact_snapshot_is_an_independent_cpu_int32_allocation(dtype):
    from vllm_ascend.worker.v2.spec_decode.draft_window_eager import DraftWindowGraphInputMixin

    owner = object.__new__(DraftWindowGraphInputMixin)
    source = torch.tensor([11, 14, 0, 0, 99], dtype=dtype)
    owner.input_buffers = SimpleNamespace(seq_lens=source)
    snapshot = owner._snapshot_prepared_draft_seq_lens(4)
    assert snapshot.device.type == "cpu" and snapshot.dtype == torch.int32
    assert snapshot.shape == (4,) and snapshot.tolist() == [11, 14, 0, 0]
    snapshot.fill_(7)
    assert source.tolist() == [11, 14, 0, 0, 99]


def test_canonical_count_gather_owns_cpu_int32_storage_and_inert_padding(graph_buffers):
    case = graph_buffers
    case.owner.num_blocks.cpu[0].copy_(torch.tensor([3, 4], dtype=torch.int32))
    counts = gather_valid_num_blocks_cpu(case.source, idx_mapping_np=np.array([1, 0]), num_reqs=2, num_reqs_padded=4)
    assert counts.device.type == "cpu" and counts.dtype == torch.int32
    assert counts.shape == (4,) and counts.tolist() == [4, 3, 0, 0]
    case.owner.num_blocks.cpu.fill_(9)
    assert counts.tolist() == [4, 3, 0, 0]
    counts.zero_()
    assert case.owner.num_blocks.cpu.tolist() == [[9, 9]]


def test_direct_capture_bank_never_allocates_scratch_or_mutates_full_source(graph_buffers):
    case = graph_buffers
    positions = case.input_buffers.positions
    positions_before = positions.clone()
    full_before = case.owner.input_block_tables[0].clone()
    inputs = _materialize(case)
    carrier = _bind(case, inputs)
    assert inputs.workspace.window_block_table.data_ptr() == carrier.workspace.window_block_table.data_ptr()
    assert inputs.workspace is carrier.workspace
    assert torch.equal(
        inputs.workspace.window_block_table, torch.tensor([[12, 13, 0, 0], [23, 24, 0, 0]], dtype=torch.int32)
    )
    assert torch.equal(case.owner.input_block_tables[0], full_before)
    assert case.input_buffers.positions is positions
    torch.testing.assert_close(positions, positions_before, rtol=0, atol=0)
    assert case.target_metadata["target.layer"].seq_lens_list == [91, 92]


def test_first_real_generation_clears_startup_capture_seed_and_padding(graph_buffers):
    case = graph_buffers
    bank = case.provider.prepare_workspace(case.capture)
    for tensor in _bank_tensors(bank):
        tensor.fill_(99)
    inputs = _materialize(case, ends=(4,))
    assert inputs.workspace.window_block_table.tolist() == [[11, 0, 0, 0], [0, 0, 0, 0]]
    for tensor in _bank_tensors(bank)[1:]:
        assert tensor[1].item() == 0
    assert inputs.workspace.device_window_exact.tolist() == [4, 0]


def test_retirement_preserves_bank_until_next_generation_then_clears_tail(graph_buffers):
    case = graph_buffers
    first = _materialize(case)
    bank = first.workspace
    snapshots = [tensor.clone() for tensor in _bank_tensors(bank)]
    pointers = [tensor.data_ptr() for tensor in _bank_tensors(bank)]
    case.provider.retire_generation(first)
    with pytest.raises(RuntimeError, match="retired"):
        first.validate()
    assert all(torch.equal(tensor, old) for tensor, old in zip(_bank_tensors(bank), snapshots))
    second = _materialize(case, epoch=1, ends=(4,))
    assert [tensor.data_ptr() for tensor in _bank_tensors(bank)] == pointers
    assert second.workspace.window_block_table.tolist() == [[11, 0, 0, 0], [0, 0, 0, 0]]
    for tensor in _bank_tensors(bank)[1:]:
        assert tensor.tolist() == [4, 0]
    with pytest.raises(RuntimeError, match="retired Draft inputs"):
        first.validate()


def test_active_generation_blocks_reuse_and_preserves_all_backing(graph_buffers):
    case = graph_buffers
    first = _materialize(case)
    snapshots = [tensor.clone() for tensor in _bank_tensors(first.workspace)]
    assert case.provider._active_inputs is first
    with pytest.raises(RuntimeError, match="still owns an active read bank"):
        _materialize(case, epoch=1, ends=(4,))
    assert all(torch.equal(tensor, old) for tensor, old in zip(_bank_tensors(first.workspace), snapshots))
    first.validate()
    case.provider.retire_generation(first)
    assert case.provider._active_inputs is None
    second = _materialize(case, epoch=1, ends=(4,))
    assert second.proposal is not first.proposal


def test_old_input_handle_cannot_retire_or_publish_over_new_bank(graph_buffers):
    case = graph_buffers
    first = _materialize(case)
    case.provider.retire_generation(first)
    current = _materialize(case)
    for stale in (first, replace(current)):
        with pytest.raises(RuntimeError, match="retired"):
            stale.validate()
        with pytest.raises(RuntimeError, match="retired"):
            case.provider.retire_generation(stale)
        with pytest.raises(RuntimeError, match="retired"):
            case.provider.quarantine_generation(stale)
    current.validate()


@pytest.mark.parametrize("submitted", [False, True])
def test_provider_rejects_overwrite_before_bound_consumer_has_finished(graph_buffers, submitted):
    case = graph_buffers
    inputs = _materialize(case)
    _bind(case, inputs)
    if submitted:
        case.provider.consume_metadata(route=DraftWindowGraphRoute.DFLASH, expected_consumer_count=1)
    with pytest.raises(RuntimeError, match="active Draft Window graph read carrier"):
        case.provider.prepare_workspace(case.capture)


def test_repeated_binding_preserves_the_actual_published_metadata_and_pending_bank(graph_buffers):
    case = graph_buffers
    inputs = _materialize(case)
    before = [tensor.clone() for tensor in _bank_tensors(inputs.workspace)]
    bucket = _bind(case, inputs)
    metadata = inputs.params
    with pytest.raises(RuntimeError, match="overwrite active draft metadata"):
        _bind(case, inputs)
    assert inputs.params is metadata and bucket.inputs is inputs
    assert case.provider._active_inputs is inputs
    assert all(torch.equal(tensor, old) for tensor, old in zip(_bank_tensors(inputs.workspace), before))
    with pytest.raises(RuntimeError, match="active Draft Window"):
        case.provider.prepare_workspace(case.capture)


def test_cpu_valid_counts_are_required_before_any_bank_materialization(graph_buffers):
    case = graph_buffers
    bank = case.provider.prepare_workspace(case.capture)
    before = [tensor.clone() for tensor in _bank_tensors(bank)]
    with pytest.raises(TypeError, match="cpu_valid_num_blocks"):
        case.provider.materialize(
            proposal=_proposal(),
            cpu_full_exact=torch.zeros(2, dtype=torch.int32),
            num_reqs_padded=2,
            step=0,
            output_workspace=bank,
        )
    assert all(torch.equal(tensor, old) for tensor, old in zip(_bank_tensors(bank), before))
    # A missing argument must leave the actual bank available for a real bind.
    inputs = _materialize(case)
    inputs.validate()


def test_eagle_serialized_steps_own_distinct_banks_and_preserve_previous_consumers(graph_buffers):
    case = graph_buffers
    capability = case.provider.source
    projection = replace(
        capability,
        policy=replace(capability.policy, normalized_method="eagle3"),
    )
    proposal = _proposal(7)
    provider = DraftWindowGraphBindingProvider(source=projection)
    case.owner.slot_mappings = torch.arange(8, dtype=torch.int64).reshape(1, 8)
    proposal_plan = compute_draft_window_plan(full_exact_end=11, future_offset=3, window_size=4, block_span_tokens=4)
    retained: list[tuple[DraftWindowWorkspace, list[torch.Tensor]]] = []
    for step, (padded, query_width) in enumerate(((2, 4), (2, 1), (2, 1))):
        route = DraftWindowGraphRoute.EAGLE3_FIRST_PASS if step == 0 else DraftWindowGraphRoute.EAGLE3_AR_STEP
        descriptor = BatchExecutionDescriptor(
            CUDAGraphMode.FULL, num_tokens=padded * query_width, num_reqs=padded, uniform_token_count=query_width
        )
        capture = provider.register_capture_identity(
            route=route,
            descriptor=descriptor,
            consumer_ordinal=max(step - 1, 0),
        )
        bank = provider.prepare_workspace(capture)
        exact = torch.tensor([11 + step] + [0] * (padded - 1), dtype=torch.int32)
        kwargs = dict(
            proposal=proposal,
            frozen_row_plans=(proposal_plan,),
            cpu_full_exact=exact,
            cpu_valid_num_blocks=torch.tensor([4] + [0] * (padded - 1), dtype=torch.int32),
            num_reqs_padded=padded,
            step=step,
            output_workspace=bank,
        )
        inputs = provider.materialize(**kwargs)
        current = SimpleNamespace(**{**vars(case), "provider": provider, "capture": capture})
        carrier = _bind(current, inputs)
        assert carrier.inputs is inputs
        assert inputs.proposal is proposal and inputs.step == step
        for old_bank, old_values in retained:
            for tensor, old_tensor, snapshot in zip(_bank_tensors(bank), _bank_tensors(old_bank), old_values):
                assert tensor.untyped_storage().data_ptr() != old_tensor.untyped_storage().data_ptr()
                assert torch.equal(old_tensor, snapshot)
        snapshots = [tensor.clone() for tensor in _bank_tensors(bank)]
        provider.retire_generation(inputs)
        assert all(torch.equal(tensor, snapshot) for tensor, snapshot in zip(_bank_tensors(bank), snapshots))
        retained.append((bank, snapshots))
        if step == 0:
            provider.consume_metadata(route=route, expected_consumer_count=1)
            provider.finish_update(route=route)
    provider.consume_metadata(route=DraftWindowGraphRoute.EAGLE3_AR_STEP, expected_consumer_count=2)
    provider.finish_update(route=DraftWindowGraphRoute.EAGLE3_AR_STEP)
    assert len(retained) == 3


@pytest.mark.parametrize(
    "field,value",
    [
        ("cg_mode", CUDAGraphMode.PIECEWISE),
        ("num_tokens", 4),
        ("num_reqs", 1),
        ("uniform_token_count", 2),
        ("max_query_len", 2),
        ("num_active_loras", 1),
        ("num_ubatches", 2),
    ],
)
def test_runtime_capture_resolves_all_actual_descriptor_fields(graph_buffers, field, value):
    provider = graph_buffers.provider
    descriptor = graph_buffers.capture.descriptor
    route = DraftWindowGraphRoute.DFLASH
    assert hash(descriptor) == hash(replace(descriptor))
    assert provider.capture_identity_for(route=route, descriptor=replace(descriptor)) is graph_buffers.capture
    with pytest.raises(RuntimeError, match="descriptor does not resolve|no captured consumer"):
        provider.capture_identity_for(route=route, descriptor=replace(descriptor, **{field: value}))


def test_capture_geometry_cannot_hide_a_different_full_descriptor(graph_buffers):
    case = graph_buffers
    with pytest.raises(RuntimeError, match="geometry is ambiguous"):
        case.provider.register_capture_identity(
            route=case.capture.route, descriptor=replace(case.capture.descriptor, max_query_len=1)
        )


def test_actual_binding_rejects_changed_padded_rows_before_publication(graph_buffers):
    case = graph_buffers
    inputs = _materialize(case)
    object.__setattr__(inputs, "padded_request_count", 4)
    with pytest.raises(RuntimeError, match="padded request count mismatch"):
        _bind(case, inputs)
    assert inputs.params is None and case.capture.inputs is None


@pytest.mark.parametrize("field,value", [("num_active_loras", 1), ("num_ubatches", 2)])
def test_unqualified_descriptor_variants_are_not_opened(graph_buffers, field, value):
    with pytest.raises(NotImplementedError, match="LoRA or microbatch"):
        graph_buffers.provider.register_capture_identity(
            route=DraftWindowGraphRoute.DFLASH,
            descriptor=replace(graph_buffers.capture.descriptor, **{field: value}),
        )


def test_capture_token_cannot_be_forged_or_cross_provider(graph_buffers):
    case = graph_buffers
    with pytest.raises(RuntimeError, match="not registered"):
        case.provider.prepare_workspace(replace(case.capture))
    other = DraftWindowGraphBindingProvider(source=case.projection)
    foreign = other.register_capture_identity(route=case.capture.route, descriptor=case.capture.descriptor)
    with pytest.raises(RuntimeError, match="not registered"):
        case.provider.prepare_workspace(foreign)


@pytest.mark.parametrize("field", ["window_block_table", "device_window_exact", "cpu_window_exact"])
def test_capture_bank_has_no_operand_replacement_api(graph_buffers, field):
    carrier = graph_buffers.capture
    with pytest.raises(FrozenInstanceError):
        setattr(carrier.workspace, field, getattr(carrier.workspace, field).clone())
    with pytest.raises(FrozenInstanceError):
        carrier.workspace = carrier.workspace


def test_startup_capture_exposes_the_real_bank_without_fake_generation(graph_buffers):
    case = graph_buffers
    carrier = case.capture
    bank = case.provider.startup_capture_workspace(case.capture)
    assert bank is carrier.workspace
    assert bank.max_num_reqs == case.capture.descriptor.num_reqs
    # A startup placeholder must not bind an invented business generation.
    inputs = _materialize(case)
    inputs.validate()
    with pytest.raises(FrozenInstanceError):
        bank.max_num_reqs += 1


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
        identity = context.capture_banks_for(route=case.capture.route, num_reqs=2, num_tokens=2)[0]
        read = context.startup_capture_workspace(identity)
        assert identity is case.capture
        full = case.source.table
        before = full.clone()
        maximum = _seed_capture_workspace(read, max_model_len=16, provider=case.provider)
        assert maximum == 7 and read.cpu_window_exact.tolist() == [7, 7]
        assert read.device_window_exact.tolist() == [7, 7]
        assert torch.equal(full, before)
        assert torch.equal(read.window_block_table[:, :2], full[:, :2])
        with pytest.raises(RuntimeError, match="no captured consumer for this geometry"):
            context.capture_banks_for(route=case.capture.route, num_reqs=1, num_tokens=2)


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
            bucket = context.capture_banks_for(route=case.capture.route, num_reqs=2, num_tokens=2)[0]
            bank = context.startup_capture_workspace(bucket)
            assert bucket is case.capture and bank is case.capture.workspace
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
    bank = case.provider.startup_capture_workspace(case.capture)
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
    assert case.provider._active_inputs is None


def test_piecewise_context_only_descriptor_does_not_register_an_attention_bank(graph_buffers):
    case = graph_buffers
    with pytest.raises(ValueError, match="FULL attention descriptor"):
        case.provider.register_capture_identity(
            route=case.capture.route,
            descriptor=replace(case.capture.descriptor, cg_mode=CUDAGraphMode.PIECEWISE, num_reqs=None),
        )
    assert tuple(case.provider._carriers.values()) == ((case.capture,),)


def test_materialization_reads_current_canonical_table_without_a_stale_source_mirror(graph_buffers):
    case = graph_buffers
    replacement = case.owner.input_block_tables[0].clone() + 100
    case.owner.input_block_tables[0] = replacement
    inputs = _materialize(case)
    assert inputs.workspace.window_block_table.tolist() == [[112, 113, 0, 0], [123, 124, 0, 0]]
    assert case.source.table is replacement


def test_workspace_rejects_invalid_native_table_dtype_at_its_creation_boundary():
    with pytest.raises(TypeError, match="must use int32"):
        DraftWindowWorkspace(
            max_num_reqs=2, table_capacity_columns=2, table_template=torch.zeros((2, 4), dtype=torch.int64)
        )


def test_captured_payload_uses_exact_current_lengths_and_inert_padding(graph_buffers):
    case = graph_buffers
    inputs = _materialize(case, ends=(11,))
    bucket = _bind(case, inputs)
    assert inputs.params["actual_seq_lengths_kv"] == [7, 0]
    assert inputs.params["actual_seq_lengths"] == [1, 2]
    assert inputs.params["block_table"] is bucket.workspace.window_block_table
    assert inputs.params["block_table"][1].tolist() == [0, 0, 0, 0]
    case.provider.retire_generation(inputs)
    with pytest.raises(RuntimeError, match="retired Draft inputs"):
        inputs.validate()


def test_graph_source_has_one_readonly_owner_and_no_second_runtime(graph_buffers):
    case = graph_buffers
    with pytest.raises(AttributeError):
        case.provider.source = replace(case.projection)
    assert case.provider.source is case.projection
    assert not hasattr(case.provider, "_runtime_state")
    assert not hasattr(case.provider, "_metadata")
    assert case.provider._active_inputs is None


@pytest.mark.parametrize(
    "exact,valid,capacity,error",
    [
        (20, 4, 2, RuntimeError),
        (11, 4, 1, RuntimeError),
        (4, -1, 2, RuntimeError),
        (4, 5, 2, RuntimeError),
    ],
)
def test_actual_bank_boundary_rejects_invalid_capacity_and_residency(graph_buffers, exact, valid, capacity, error):
    case = graph_buffers
    bank = case.provider.prepare_workspace(case.capture)
    # Capacity is fixed at construction, rather than a mutable runtime field.
    bank = DraftWindowWorkspace(max_num_reqs=2, table_capacity_columns=capacity, table_template=case.source.table)
    before = [tensor.clone() for tensor in _bank_tensors(bank)]
    with pytest.raises(error):
        case.provider.materialize(
            proposal=_proposal(),
            cpu_full_exact=torch.tensor([exact], dtype=torch.int32),
            cpu_valid_num_blocks=torch.tensor([valid], dtype=torch.int32),
            num_reqs_padded=1,
            step=0,
            output_workspace=bank,
        )
    assert case.provider.quarantined
    assert all(torch.equal(tensor, old) for tensor, old in zip(_bank_tensors(bank), before))
    with pytest.raises(RuntimeError, match="quarantined"):
        case.provider.prepare_workspace(case.capture)


def test_actual_eagle_bank_rejects_an_intra_block_frozen_anchor(graph_buffers):
    from vllm_ascend.worker.v2.spec_decode.draft_window_plan import DraftWindowPlan

    case = graph_buffers
    source = replace(
        case.projection,
        policy=replace(case.projection.policy, normalized_method="eagle3"),
    )
    provider = DraftWindowGraphBindingProvider(source=source)
    capture = provider.register_capture_identity(
        route=DraftWindowGraphRoute.EAGLE3_FIRST_PASS, descriptor=case.capture.descriptor
    )
    bank = provider.prepare_workspace(capture)
    with pytest.raises(NotImplementedError, match="intra-block"):
        provider.materialize(
            proposal=_proposal(),
            cpu_full_exact=torch.tensor([4], dtype=torch.int32),
            cpu_valid_num_blocks=torch.tensor([4], dtype=torch.int32),
            frozen_row_plans=(DraftWindowPlan(4, 7, 1),),
            num_reqs_padded=1,
            step=0,
            output_workspace=bank,
        )
    assert provider.quarantined


@pytest.fixture
def typed_update_guard(graph_buffers):
    import torch_npu

    from vllm_ascend.attention.attention_v1 import FIAParamProvider
    from vllm_ascend.compilation.updatable_graph import GraphUpdateTask

    case = graph_buffers
    carrier = _bind(case, _materialize(case))
    batch = case.provider.consume_metadata(route=case.capture.route, expected_consumer_count=1)
    provider = FIAParamProvider(layer_name="draft.layer", sliding_window=None, is_draft_model=True)
    task = GraphUpdateTask(torch_npu.npu_fused_infer_attention_score.out, {}, provider, 0, object(), object())

    class Graph:
        tasks = [task]
        provider_sizes = {provider: 1}

    return SimpleNamespace(case=case, carrier=carrier, graph=Graph(), source=batch, task=task)


def test_original_task_resolver_binds_current_metadata_without_replacing_capture_state(typed_update_guard):
    from vllm_ascend.compilation.updatable_graph import UpdatableGraph

    c = typed_update_guard
    c.case.provider.validate_captured_graphs({c.case.capture.descriptor: c.graph})
    resolved = UpdatableGraph.resolve_tasks(c.graph, c.source)
    assert len(resolved) == 1
    actual = resolved[0]
    assert actual.operation is c.task.operation
    assert actual.handle is c.task.handle and actual.event is c.task.event
    assert actual.provider is c.task.provider and actual.provider_index == 0
    assert actual.kwargs["block_table"] is c.carrier.inputs.params["block_table"]
    assert actual.kwargs["actual_seq_lengths_kv"] is c.carrier.inputs.params["actual_seq_lengths_kv"]
    assert actual.kwargs["actual_seq_lengths"] is c.carrier.inputs.params["actual_seq_lengths"]
    with pytest.raises(RuntimeError, match="active Draft Window"):
        c.case.provider.prepare_workspace(c.case.capture)
    c.case.provider.finish_update(route=c.case.capture.route)
    assert c.case.provider.prepare_workspace(c.case.capture) is c.carrier.workspace


@pytest.mark.parametrize("field", ["provider", "operation"])
def test_capture_contract_rejects_an_unsupported_native_consumer(typed_update_guard, field):
    c = typed_update_guard
    setattr(c.task, field, object())
    with pytest.raises(RuntimeError, match="supported Draft FIA consumer"):
        c.case.provider.validate_captured_graphs({c.case.capture.descriptor: c.graph})


@pytest.mark.parametrize("window", [False, True])
@pytest.mark.parametrize("update_fails", [False, True])
def test_actual_parallel_manager_updates_after_replay_and_only_then_retires_banks(
    graph_buffers, monkeypatch, window, update_fails
):
    from vllm_ascend.attention.attention_v1 import FIAParamProvider
    from vllm_ascend.compilation.updatable_graph import GraphUpdateTask, UpdatableGraph
    from vllm_ascend.worker.v2.spec_decode.dflash import aclgraph

    case = graph_buffers
    inputs = _materialize(case) if window else None
    owner = SimpleNamespace(
        attn_backends={"draft.layer": object()},
        input_batch=SimpleNamespace(seq_lens_cpu_upper_bound=torch.tensor([16, 16])),
        build_draft_attn_metadatas=Mock(
            return_value=[
                {
                    "draft.layer": SimpleNamespace(
                        actual_seq_lengths_q=[1, 2], seq_lens_list=[11, 14], block_tables=case.source.table
                    )
                }
            ]
        ),
    )
    if window:
        owner._draft_window_graph_binding_provider = case.provider
        owner._draft_window_proposal = inputs.proposal
        owner._draft_window_selected_metadata = case.capture
    provider = FIAParamProvider("draft.layer", None, is_draft_model=True)
    task = GraphUpdateTask(object(), {}, provider, 0, object(), object())
    events, resolved = [], []

    class Graph:
        provider_sizes = {provider: 1}
        tasks = [task]

        def resolve_tasks(self, source):
            events.append("resolve")
            result = UpdatableGraph.resolve_tasks(self, source)
            resolved.extend(result)
            return result

        def update(self, stream, tasks):
            assert tasks[0] is resolved[0]
            events.append("update")
            if update_fails:
                raise RuntimeError("native UPDATE failed")

    manager = object.__new__(aclgraph.DFlashAclGraphManager)
    manager.speculator = owner
    manager.graphs = {case.capture.descriptor: Graph()}
    manager.update_stream = SimpleNamespace(wait_stream=lambda stream: events.append("wait"))
    monkeypatch.setattr(aclgraph, "UpdatableGraph", Graph)
    monkeypatch.setattr(aclgraph, "use_updatable_graph", lambda backend: True)
    monkeypatch.setattr(torch.npu, "current_stream", lambda: object())
    output = object()

    def replay(self, descriptor):
        events.append("replay")
        return output

    monkeypatch.setattr(aclgraph.DFlashCudaGraphManager, "run_fullgraph", replay)
    if update_fails:
        with pytest.raises(RuntimeError, match="native UPDATE failed"):
            manager.run_fullgraph(case.capture.descriptor)
    else:
        assert manager.run_fullgraph(case.capture.descriptor) is output
    assert events == ["resolve", "wait", "replay", "update"]
    actual = resolved[0]
    assert actual.handle is task.handle and actual.event is task.event and actual.provider is provider
    assert actual.kwargs["actual_seq_lengths"] == [1, 2]
    if window:
        owner.build_draft_attn_metadatas.assert_not_called()
        assert actual.kwargs["actual_seq_lengths_kv"] == [7, 6]
        assert actual.kwargs["block_table"] is case.capture.workspace.window_block_table
        assert case.provider._updating is update_fails
        assert bool(case.provider._pending) is update_fails
        assert case.capture.inputs is inputs and inputs.params is not None
    else:
        owner.build_draft_attn_metadatas.assert_called_once()
        assert actual.kwargs["actual_seq_lengths_kv"] == [11, 14]
        assert actual.kwargs["block_table"] is case.source.table
    assert case.source.table.tolist() == [[11, 12, 13, 14], [21, 22, 23, 24]]
    assert case.owner.slot_mappings.tolist() == [[40, 80]]
    assert case.target_metadata["target.layer"].seq_lens_list == [91, 92]


def test_parallel_capture_checks_tasks_after_the_original_manager_stores_its_graphs(typed_update_guard, monkeypatch):
    from contextlib import nullcontext

    from vllm_ascend.worker.v2.spec_decode.dflash import aclgraph, capture

    c = typed_update_guard
    manager = object.__new__(aclgraph.DFlashAclGraphManager)
    manager.speculator = SimpleNamespace(
        input_buffers=c.case.input_buffers,
        _draft_window_graph_binding_provider=c.case.provider,
    )
    manager._capture_descs = {CUDAGraphMode.FULL: [c.case.capture.descriptor]}
    manager.graphs = {}
    checked: list[dict[Any, Any]] = []
    validate = c.case.provider.validate_captured_graphs

    def upstream(self, *_args, **_kwargs):
        assert not checked
        self.graphs[c.case.capture.descriptor] = c.graph

    def after_capture(graphs):
        assert graphs[c.case.capture.descriptor] is c.graph
        checked.append(graphs)
        validate(graphs)

    monkeypatch.setattr(aclgraph.DFlashCudaGraphManager, "capture", upstream)
    monkeypatch.setattr(aclgraph, "communicator_switch", nullcontext)
    monkeypatch.setattr(aclgraph, "model_capture_wrapper", lambda *_args: nullcontext())
    monkeypatch.setattr(c.case.provider, "validate_captured_graphs", after_capture)
    manager.capture(lambda: None, c.case.input_buffers, c.case.owner, [], object(), 16)
    assert checked == [manager.graphs]
    assert not hasattr(c.case.input_buffers, capture._CAPTURE_CONTEXT_ATTRIBUTE)


def test_publication_retains_only_fia_update_operands_and_capture_bank(graph_buffers):
    case = graph_buffers
    inputs = _materialize(case)
    carrier = _bind(case, inputs)
    assert set(inputs.params) == {"block_table", "actual_seq_lengths", "actual_seq_lengths_kv"}
    assert inputs.params["block_table"] is carrier.workspace.window_block_table
    assert inputs.params["actual_seq_lengths_kv"] == carrier.workspace.cpu_window_exact.tolist()
    with pytest.raises(RuntimeError, match="no submitted consumer batch"):
        case.provider.get(object())


def test_eagle_first_bank_retains_full_table_capacity_for_a_mixed_prefill(graph_buffers):
    case = graph_buffers
    # The shared production factory already retains the canonical Full table
    # width, including rows that continue prefill in an otherwise Window batch.
    bounded = replace(
        case.projection,
        policy=resolve_draft_window_policy(requested_window_size=4, method="eagle3", target_model_type="qwen3"),
    )
    provider = DraftWindowGraphBindingProvider(source=bounded)
    capture = provider.register_capture_identity(
        route=DraftWindowGraphRoute.EAGLE3_FIRST_PASS,
        descriptor=BatchExecutionDescriptor(CUDAGraphMode.FULL, num_tokens=8, num_reqs=2, uniform_token_count=4),
    )
    bank = provider.prepare_workspace(capture)
    inputs = provider.materialize(
        proposal=replace(_proposal(0, ("request-a", "request-b")), full_read_rows=(True, False)),
        cpu_full_exact=torch.tensor([11, 14], dtype=torch.int32),
        cpu_valid_num_blocks=torch.tensor([4, 4], dtype=torch.int32),
        num_reqs_padded=2,
        step=0,
        output_workspace=bank,
        frozen_row_plans=tuple(
            compute_draft_window_plan(full_exact_end=end, future_offset=3, window_size=4, block_span_tokens=4)
            for end in (11, 14)
        ),
    )
    assert inputs.workspace is bank
    assert bank.cpu_window_exact.tolist() == [11, 2]
    torch.testing.assert_close(
        bank.window_block_table, torch.tensor([[11, 12, 13, 14], [24, 0, 0, 0]], dtype=torch.int32)
    )
    torch.testing.assert_close(
        case.owner.input_block_tables[0], torch.tensor([[11, 12, 13, 14], [21, 22, 23, 24]], dtype=torch.int32)
    )
    assert case.target_metadata["target.layer"].seq_lens_list == [91, 92]


@pytest.mark.parametrize("failure", [None, "update", "republish", "prepare"])
def test_actual_proposal_scope_owns_fresh_inputs_and_preserves_failed_submission(graph_buffers, failure, monkeypatch):
    from contextlib import nullcontext

    from vllm_ascend.worker.v2.spec_decode.draft_window_eager import DraftWindowGraphInputMixin

    case = graph_buffers
    owner = object.__new__(DraftWindowGraphInputMixin)
    owner._draft_window_graph_binding_provider = case.provider
    owner._initialize_draft_window_graph_inputs()
    batch = SimpleNamespace(
        num_reqs=2,
        req_ids=["request-a", "request-b"],
        idx_mapping_np=np.array([0, 1]),
        num_computed_prefill_tokens_np=np.array([16, 16]),
        prefill_len_np=np.array([0, 0]),
        num_scheduled_tokens=np.array([1, 1]),
        num_draft_tokens_per_req=None,
        is_prefilling_np=np.array([False, False]),
        has_prefill=False,
    )
    owner.input_batch = batch
    errors = {
        "update": (RuntimeError, "native UPDATE failed"),
        "republish": (RuntimeError, "overwrite active draft metadata"),
        "prepare": (RuntimeError, "native Window copy failed"),
    }
    expected = pytest.raises(errors[failure][0], match=errors[failure][1]) if failure else nullcontext()
    preparation_error = None
    if failure == "prepare":
        original_copy = torch.Tensor.copy_
        bank_storage = case.capture.workspace.window_block_table.untyped_storage().data_ptr()

        def failed_copy(destination, *args, **kwargs):
            if destination.untyped_storage().data_ptr() == bank_storage:
                raise RuntimeError("native Window copy failed")
            return original_copy(destination, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, "copy_", failed_copy)
    with expected as caught, owner._draft_window_proposal_scope(batch, dummy_run=False, is_profile=False):
        context = owner._draft_window_proposal
        assert context is not None and context.input_batch is batch
        context.validate(batch)
        with (
            pytest.raises(RuntimeError, match="already active"),
            owner._draft_window_proposal_scope(batch, dummy_run=False, is_profile=False),
        ):
            pytest.fail("a real proposal cannot nest or replace its current context")
        try:
            inputs = case.provider.materialize(
                proposal=context,
                cpu_full_exact=torch.tensor([11, 14], dtype=torch.int32),
                cpu_valid_num_blocks=torch.tensor([4, 4], dtype=torch.int32),
                num_reqs_padded=2,
                step=0,
                output_workspace=case.provider.prepare_workspace(case.capture),
            )
        except RuntimeError as error:
            preparation_error = error
            raise
        _bind(case, inputs)
        if failure == "republish":
            _bind(case, inputs)
        case.provider.consume_metadata(route=case.capture.route, expected_consumer_count=1)
        if failure == "update":
            raise RuntimeError("native UPDATE failed")
        case.provider.finish_update(route=case.capture.route)
    assert owner._draft_window_proposal is None
    if failure:
        active = case.provider._active_inputs
        assert case.provider.quarantined and active is not None and active.workspace is case.capture.workspace
        case.provider.quarantine_generation(active)
        assert case.provider._active_inputs is active
        if failure == "prepare":
            assert caught.value is preparation_error
        with pytest.raises(RuntimeError, match="quarantined"):
            case.provider.prepare_workspace(case.capture)
    else:
        assert case.provider._active_inputs is None
        with pytest.raises(RuntimeError, match="retired"):
            inputs.validate()


@pytest.mark.parametrize("failure", ["missing", "wrong_route", "before_consume", "twice", "wrong_finish"])
def test_incomplete_or_wrong_update_cannot_release_the_real_pending_bank(graph_buffers, failure):
    case = graph_buffers
    inputs = _materialize(case)
    bucket = _bind(case, inputs)
    route = case.capture.route
    if failure in ("twice", "wrong_finish"):
        case.provider.consume_metadata(route=route, expected_consumer_count=1)
    with pytest.raises(RuntimeError):
        if failure == "missing":
            case.provider.consume_metadata(route=route, expected_consumer_count=2)
        elif failure == "wrong_route":
            case.provider.consume_metadata(route=DraftWindowGraphRoute.EAGLE3_AR_STEP, expected_consumer_count=1)
        elif failure in ("before_consume", "wrong_finish"):
            case.provider.finish_update(
                route=route if failure == "before_consume" else DraftWindowGraphRoute.EAGLE3_AR_STEP
            )
        else:
            case.provider.consume_metadata(route=route, expected_consumer_count=1)
    with pytest.raises(RuntimeError, match="active Draft Window"):
        case.provider.prepare_workspace(case.capture)
    if failure not in ("twice", "wrong_finish"):
        case.provider.consume_metadata(route=route, expected_consumer_count=1)
    case.provider.finish_update(route=route)
    assert case.provider.prepare_workspace(case.capture) is bucket.workspace
    # Host UPDATE release does not drop the real CPU metadata/bank owners.
    assert bucket.inputs is inputs and bucket.inputs.params is not None


def test_pending_eagle_consumers_reject_a_different_proposal_without_releasing_old_bank(graph_buffers):
    case = graph_buffers
    source = replace(case.projection, policy=replace(case.projection.policy, normalized_method="eagle3"))
    provider = DraftWindowGraphBindingProvider(source=source)
    descriptor = BatchExecutionDescriptor(CUDAGraphMode.FULL, num_tokens=1, num_reqs=1, uniform_token_count=1)
    first = provider.register_capture_identity(route=DraftWindowGraphRoute.EAGLE3_AR_STEP, descriptor=descriptor)
    second = provider.register_capture_identity(
        route=DraftWindowGraphRoute.EAGLE3_AR_STEP, descriptor=descriptor, consumer_ordinal=1
    )
    for ordinal, capture in enumerate((first, second)):
        inputs = provider.materialize(
            proposal=_proposal(ordinal),
            cpu_full_exact=torch.tensor([11], dtype=torch.int32),
            cpu_valid_num_blocks=torch.tensor([4], dtype=torch.int32),
            num_reqs_padded=1,
            step=ordinal + 1,
            output_workspace=provider.prepare_workspace(capture),
        )
        current = SimpleNamespace(**{**vars(case), "provider": provider, "capture": capture})
        if ordinal == 0:
            _bind(current, inputs)
            provider.retire_generation(inputs)
        else:
            with pytest.raises(RuntimeError, match="same actual proposal/route"):
                _bind(current, inputs)
            provider.quarantine_generation(inputs)
    with pytest.raises(RuntimeError, match="quarantined"):
        provider.prepare_workspace(first)


def test_captured_consumer_cannot_skip_the_actual_serial_ordinal(graph_buffers):
    case = graph_buffers
    skipped = case.provider.register_capture_identity(
        route=case.capture.route, descriptor=case.capture.descriptor, consumer_ordinal=1
    )
    current = SimpleNamespace(**{**vars(case), "capture": skipped})
    inputs = _materialize(current)
    with pytest.raises(RuntimeError, match="serial ordinal order"):
        _bind(current, inputs)
    case.provider.quarantine_generation(inputs)
    with pytest.raises(RuntimeError, match="quarantined"):
        case.provider.prepare_workspace(skipped)


@pytest.mark.parametrize("span", [1, 4, 128])
@pytest.mark.parametrize("boundary", [-1, 0, 1])
def test_actual_aligned_bank_reads_only_the_plan_interval_across_block_boundaries(span, boundary):
    window, end = 7, span + boundary
    columns = max((end + span - 1) // span, (window + 2 * span - 2) // span)
    table = torch.arange(columns, dtype=torch.int32).view(1, columns)
    counts = torch.tensor([[(end + span - 1) // span]], dtype=torch.int32)
    full = SimpleNamespace(
        input_block_tables=[table],
        num_blocks=SimpleNamespace(cpu=counts, gpu=counts),
        block_sizes=[span],
        kernel_block_sizes=[span],
    )
    group = DraftWindowGroupRoute(0, ("draft.layer",), span, span, DraftWindowProviderStatus.DIRECT_DENSE)
    source = DraftWindowGraphSource(
        resolve_draft_window_policy(requested_window_size=window, method="dflash"), group, full
    )
    provider = DraftWindowGraphBindingProvider(source=source)
    descriptor = BatchExecutionDescriptor(CUDAGraphMode.FULL, num_tokens=1, num_reqs=1, uniform_token_count=1)
    bucket = provider.register_capture_identity(route=DraftWindowGraphRoute.DFLASH, descriptor=descriptor)
    bank = provider.prepare_workspace(bucket)
    provider.materialize(
        proposal=_proposal(),
        cpu_full_exact=torch.tensor([end], dtype=torch.int32),
        cpu_valid_num_blocks=counts[0],
        num_reqs_padded=1,
        step=0,
        output_workspace=bank,
    )
    # Literal aligned-token interval, independent of the removed projection DTO.
    start = max(range(0, max(end - window, 0) + 1, span))
    expected = table[0, start // span : (end + span - 1) // span]
    torch.testing.assert_close(bank.window_block_table[0, : expected.numel()], expected, atol=0, rtol=0)
    assert not bank.window_block_table[0, expected.numel() :].any()
    assert bank.cpu_window_exact.tolist() == bank.device_window_exact.tolist() == [len(range(start, end))]
    assert table.tolist() == [list(range(columns))]
