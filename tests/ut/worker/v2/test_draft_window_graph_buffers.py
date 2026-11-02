# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU buffer ownership contracts; these do not assert NPU completion."""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBuffers

from vllm_ascend.worker.v2.spec_decode.draft_window import (
    DraftWindowSourceCapability,
    DraftWindowWorkspace,
    FullDraftKVSource,
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
from vllm_ascend.worker.v2.spec_decode.draft_window_nongraph import DraftWindowBatchPhasePlan
from vllm_ascend.worker.v2.spec_decode.draft_window_plan import compute_draft_window_plan
from vllm_ascend.worker.v2.spec_decode.draft_window_policy import resolve_draft_window_policy


def _proposal(epoch=0, requests=("request-a",)):
    return DraftWindowProposalContext(
        epoch,
        0,
        requests,
        tuple(range(len(requests))),
        DraftWindowBatchPhasePlan((False,) * len(requests)),
    )


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
    )
    source = FullDraftKVSource(
        logical_block_sizes=(4,),
        kernel_block_sizes=(4,),
        _block_tables_owner=owner,
        _input_block_tables=(table,),
        _persistent_num_blocks_cpu=owner.num_blocks.cpu,
    )
    group = DraftWindowGroupRoute(
        gid=0,
        layers=("draft.layer",),
        scheduler_block_size=4,
        table_column_span=4,
        read_semantics="replicated",
        status=DraftWindowProviderStatus.DIRECT_DENSE,
    )
    parallel = SimpleNamespace(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
    )
    projection = DraftWindowSourceCapability(
        policy=resolve_draft_window_policy(requested_window_size=4, method="dflash", target_model_type="qwen3"),
        runtime_route="dflash",
        group=group,
        full_source=source,
        expected_future_offset=0,
        parallel_config=parallel,
        tensor_parallel_rank=0,
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
        source=source,
        owner=owner,
        target_metadata={"target.layer": SimpleNamespace(seq_lens_list=[91, 92], actual_seq_lengths_q=[1, 2])},
    )


def _materialize(case, *, epoch=0, ends=(11, 14), quarantine=False):
    plans = [
        compute_draft_window_plan(full_exact_end=end, future_offset=0, window_size=4, block_span_tokens=4)
        for end in ends
    ]
    exact = torch.tensor([plan.window_visible_length for plan in plans] + [0] * (2 - len(plans)), dtype=torch.int32)
    return case.provider.materialize(
        proposal=_proposal(epoch, ("request-a", "request-b")[: len(plans)]),
        idx_mapping=case.mapping,
        read_descriptors=[case.projection.describe(plan, valid_num_blocks=4) for plan in plans],
        cpu_window_exact=exact.clone(),
        cpu_window_upper=exact.clone(),
        cpu_valid_num_blocks=torch.tensor([4] * len(plans) + [0] * (2 - len(plans)), dtype=torch.int32),
        num_reqs_padded=2,
        step=0,
        quarantine_on_failure=quarantine,
        output_workspace=case.provider.prepare_workspace(case.capture),
    )


def _bank_tensors(bank):
    return (
        bank.window_block_table,
        bank.device_window_exact,
        bank.cpu_window_exact,
        bank.cpu_window_upper,
    )


def _bind(case, inputs):
    actual = len(inputs.identity.proposal.ordered_request_ids)
    padded = inputs.identity.padded_request_count
    query_width = case.capture.descriptor.num_tokens // case.capture.descriptor.num_reqs
    tokens = actual * query_width
    query_lengths = tuple(row * query_width for row in range(1, actual + 1)) + (tokens,) * (padded - actual)
    reference = SimpleNamespace(
        slot_mapping=case.owner.slot_mappings[0, :tokens],
        query_start_loc=torch.tensor((0,) + query_lengths, dtype=torch.int32),
        actual_seq_lengths_q=query_lengths,
        attn_state=object(),
        num_actual_tokens=tokens,
        causal=True,
    )

    def metadata_builder(read):
        return {
            "draft.layer": SimpleNamespace(
                block_tables=read.window_block_table,
                seq_lens=read.device_window_exact,
                seq_lens_cpu=read.cpu_window_exact,
                seq_lens_list=read.cpu_window_exact.tolist(),
                slot_mapping=reference.slot_mapping,
                query_start_loc=reference.query_start_loc,
                actual_seq_lengths_q=list(reference.actual_seq_lengths_q),
                attn_state=reference.attn_state,
                num_actual_tokens=reference.num_actual_tokens,
                causal=reference.causal,
            )
        }

    return case.provider.bind_read_carrier(
        capture_identity=case.capture,
        inputs=inputs,
        metadata_builder=metadata_builder,
        target_metadata=case.target_metadata,
    )


def test_direct_capture_bank_never_allocates_scratch_or_mutates_full_source(graph_buffers):
    case = graph_buffers
    positions = case.input_buffers.positions
    positions_before = positions.clone()
    full_before = case.owner.input_block_tables[0].clone()
    inputs = _materialize(case)
    carrier = _bind(case, inputs)
    assert case.provider._workspace is None
    assert inputs.workspace.window_block_table.data_ptr() == carrier.workspace.window_block_table.data_ptr()
    assert inputs.workspace is carrier.workspace
    assert torch.equal(inputs.workspace.window_block_table, torch.tensor([[12, 13], [23, 24]], dtype=torch.int32))
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
    assert inputs.workspace.window_block_table.tolist() == [[11, 0], [0, 0]]
    for tensor in _bank_tensors(bank)[1:]:
        assert tensor[1].item() == 0
    assert inputs.workspace.device_window_exact.tolist() == [4, 0]
    assert case.provider._workspace is None


def test_retirement_preserves_bank_until_next_generation_then_clears_tail(graph_buffers):
    case = graph_buffers
    first = _materialize(case)
    bank = first.workspace
    snapshots = [tensor.clone() for tensor in _bank_tensors(bank)]
    pointers = [tensor.data_ptr() for tensor in _bank_tensors(bank)]
    case.provider.retire_generation(first.identity)
    assert bank.bound_identity is None
    assert all(torch.equal(tensor, old) for tensor, old in zip(_bank_tensors(bank), snapshots))
    second = _materialize(case, epoch=1, ends=(4,))
    assert [tensor.data_ptr() for tensor in _bank_tensors(bank)] == pointers
    assert second.workspace.window_block_table.tolist() == [[11, 0], [0, 0]]
    for tensor in _bank_tensors(bank)[1:]:
        assert tensor.tolist() == [4, 0]
    with pytest.raises(RuntimeError, match="retired Draft inputs"):
        first.validate_for(first.identity)


def test_active_generation_blocks_reuse_and_preserves_all_backing(graph_buffers):
    case = graph_buffers
    first = _materialize(case)
    snapshots = [tensor.clone() for tensor in _bank_tensors(first.workspace)]
    assert not first.retired
    assert case.provider._active_inputs is first
    with pytest.raises(RuntimeError, match="still owns an active read bank"):
        _materialize(case, epoch=1, ends=(4,))
    assert all(torch.equal(tensor, old) for tensor, old in zip(_bank_tensors(first.workspace), snapshots))
    first.validate_for(first.identity)
    case.provider.retire_generation(first.identity)
    assert first.retired
    assert case.provider._active_inputs is None
    second = _materialize(case, epoch=1, ends=(4,))
    assert second.identity.proposal.mapping_epoch == first.identity.proposal.mapping_epoch + 1


def test_same_mapping_pointer_does_not_allow_epoch_or_generation_aba(graph_buffers):
    case = graph_buffers
    first = _materialize(case)
    case.provider.retire_generation(first.identity)
    case.mapping.copy_(torch.tensor([1, 0]))
    with pytest.raises(ValueError, match="mapping epoch must strictly increase"):
        _materialize(case)
    current = _materialize(case, epoch=1)
    assert current.identity.idx_mapping_ptr == first.identity.idx_mapping_ptr
    with pytest.raises(RuntimeError, match="generation identity"):
        current.validate_for(first.identity)
    with pytest.raises(RuntimeError, match="generation identity"):
        case.provider.retire_generation(
            replace(current.identity, proposal=replace(current.identity.proposal, mapping_epoch=0))
        )
    current.validate_for(current.identity)


@pytest.mark.parametrize("submitted", [False, True])
def test_provider_rejects_overwrite_before_bound_consumer_has_finished(graph_buffers, submitted):
    case = graph_buffers
    inputs = _materialize(case)
    _bind(case, inputs)
    if submitted:
        case.provider.consume_metadata(route=DraftWindowGraphRoute.DFLASH, expected_consumer_count=1)
    with pytest.raises(RuntimeError, match="active Draft Window graph read carrier"):
        case.provider.prepare_workspace(case.capture)


def test_failed_metadata_isolation_quarantines_bank_without_clearing(graph_buffers):
    case = graph_buffers
    inputs = _materialize(case, quarantine=True)
    before = [tensor.clone() for tensor in _bank_tensors(inputs.workspace)]
    with pytest.raises(ValueError, match="alias Target"):
        inputs.publish(case.target_metadata, target_metadata=case.target_metadata)
    assert case.provider.quarantined_identity == inputs.identity
    assert case.provider._active_inputs.workspace is inputs.workspace
    assert case.provider._active_inputs is inputs
    assert inputs.workspace.bound_identity == inputs.identity
    assert all(torch.equal(tensor, old) for tensor, old in zip(_bank_tensors(inputs.workspace), before))
    with pytest.raises(RuntimeError, match="quarantined"):
        case.provider.prepare_workspace(case.capture)
    with pytest.raises(RuntimeError, match="quarantined"):
        case.provider.retire_generation(inputs.identity)


def test_cpu_valid_counts_are_required_before_any_bank_materialization(graph_buffers):
    case = graph_buffers
    bank = case.provider.prepare_workspace(case.capture)
    before = [tensor.clone() for tensor in _bank_tensors(bank)]
    with pytest.raises(TypeError, match="cpu_valid_num_blocks"):
        case.provider.materialize(
            proposal=_proposal(),
            idx_mapping=case.mapping,
            read_descriptors=(),
            cpu_window_exact=torch.zeros(2, dtype=torch.int32),
            cpu_window_upper=torch.zeros(2, dtype=torch.int32),
            num_reqs_padded=2,
            step=0,
            output_workspace=bank,
        )
    assert bank.bound_identity is None
    assert all(torch.equal(tensor, old) for tensor, old in zip(_bank_tensors(bank), before))
    assert case.provider._workspace is None


def test_eagle_serialized_steps_own_distinct_banks_and_preserve_previous_consumers(graph_buffers):
    case = graph_buffers
    capability = case.provider.source
    projection = replace(
        capability,
        policy=replace(capability.policy, normalized_method="eagle3"),
        runtime_route="eagle3",
        expected_future_offset=3,
    )
    proposal = _proposal(7)
    provider = DraftWindowGraphBindingProvider(source=projection)
    case.owner.slot_mappings = torch.arange(8, dtype=torch.int64).reshape(1, 8)
    proposal_plan = compute_draft_window_plan(full_exact_end=11, future_offset=3, window_size=4, block_span_tokens=4)
    previous = None
    retained: list[tuple[DraftWindowWorkspace, list[torch.Tensor]]] = []
    for step, (padded, query_width) in enumerate(((2, 4), (1, 1), (2, 1))):
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
        exact = torch.tensor([11 + step - proposal_plan.window_start] + [0] * (padded - 1), dtype=torch.int32)
        kwargs = dict(
            proposal=proposal,
            idx_mapping=case.mapping,
            read_descriptors=[
                projection.describe_frozen_step(proposal_plan, full_exact_end=11 + step, valid_num_blocks=4)
            ],
            cpu_window_exact=exact.clone(),
            cpu_window_upper=exact.clone(),
            cpu_valid_num_blocks=torch.tensor([4] + [0] * (padded - 1), dtype=torch.int32),
            num_reqs_padded=padded,
            step=step,
            output_workspace=bank,
        )
        if previous is None:
            inputs = provider.materialize(**kwargs)
        else:
            if step == 1:
                replaced_mapping = case.mapping.clone()
                assert replaced_mapping.data_ptr() != previous.idx_mapping_ptr
                with pytest.raises(RuntimeError, match="request mapping changed"):
                    provider.materialize(previous_identity=previous, **{**kwargs, "idx_mapping": replaced_mapping})
                with pytest.raises(RuntimeError, match="proposal or request mapping changed"):
                    provider.materialize(previous_identity=previous, **{**kwargs, "proposal": replace(proposal)})
                with pytest.raises(RuntimeError, match="request capacity exceeded"):
                    provider.materialize(previous_identity=previous, **{**kwargs, "num_reqs_padded": padded + 1})
                assert provider._last_retired_identity is previous
                assert bank.bound_identity is None
            inputs = provider.materialize(previous_identity=previous, **kwargs)
        current = SimpleNamespace(**{**vars(case), "provider": provider, "capture": capture})
        carrier = _bind(current, inputs)
        assert carrier.inputs is inputs
        assert provider._workspace is None
        assert inputs.identity.proposal.mapping_epoch == 7 and inputs.identity.step == step
        for old_bank, old_values in retained:
            for tensor, old_tensor, snapshot in zip(_bank_tensors(bank), _bank_tensors(old_bank), old_values):
                assert tensor.untyped_storage().data_ptr() != old_tensor.untyped_storage().data_ptr()
                assert torch.equal(old_tensor, snapshot)
        snapshots = [tensor.clone() for tensor in _bank_tensors(bank)]
        provider.retire_generation(inputs.identity)
        assert all(torch.equal(tensor, snapshot) for tensor, snapshot in zip(_bank_tensors(bank), snapshots))
        retained.append((bank, snapshots))
        previous = inputs.identity
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
    with pytest.raises(RuntimeError, match="descriptor does not resolve"):
        provider.capture_identity_for(route=route, descriptor=replace(descriptor, **{field: value}))


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


@pytest.mark.parametrize("field", ["window_block_table", "device_window_exact", "cpu_window_exact", "cpu_window_upper"])
def test_capture_bank_rejects_replaced_operand_before_materialization(graph_buffers, field):
    case = graph_buffers
    carrier = case.provider._carriers[case.capture]
    original = getattr(carrier.workspace, field)
    setattr(carrier.workspace, field, original.clone())
    with pytest.raises(RuntimeError, match="address or tensor geometry changed"):
        case.provider.prepare_workspace(case.capture)


def test_capture_bank_rejects_same_storage_with_changed_stride(graph_buffers):
    carrier = graph_buffers.provider._carriers[graph_buffers.capture]
    original = carrier.workspace.window_block_table
    changed = original.t()
    assert changed.data_ptr() == original.data_ptr() and changed.shape == original.shape
    assert changed.stride() != original.stride()
    carrier.workspace.window_block_table = changed
    with pytest.raises(RuntimeError, match="address or tensor geometry changed"):
        graph_buffers.provider.prepare_workspace(graph_buffers.capture)


def test_startup_capture_exposes_the_real_bank_without_fake_generation(graph_buffers):
    case = graph_buffers
    carrier = case.provider._carriers[case.capture]
    bank = case.provider.startup_capture_workspace(case.capture)
    assert bank is carrier.workspace and bank.bound_identity is None
    assert bank.max_num_reqs == case.capture.descriptor.num_reqs
    bank.max_num_reqs += 1
    with pytest.raises(RuntimeError, match="request geometry changed"):
        case.provider.startup_capture_workspace(case.capture)


def test_parallel_capture_uses_exact_manager_descriptor_and_preserves_full_tables(graph_buffers):
    from vllm_ascend.worker.v2.spec_decode.dflash.capture import _build_capture_context, _seed_capture_workspace

    case = graph_buffers
    buffers = object.__new__(InputBuffers)
    owner = SimpleNamespace(
        input_buffers=buffers,
        _draft_window_graph_binding_provider=case.provider,
        model_state=SimpleNamespace(attn_metadata=case.target_metadata),
    )
    desc = case.capture.descriptor
    ctx = _build_capture_context(owner, buffers, {CUDAGraphMode.FULL: [desc]})
    assert ctx is not None
    identity, read = ctx.resolve(2, 2)
    assert identity is case.capture
    full = case.source._input_block_tables[0]
    before = full.clone()
    maximum = _seed_capture_workspace(identity, read, (full,), max_model_len=16, provider=case.provider)
    assert maximum == 7 and read.cpu_window_exact.tolist() == read.cpu_window_upper.tolist() == [7, 7]
    assert read.device_window_exact.tolist() == [7, 7]
    assert torch.equal(full, before)
    assert torch.equal(read.window_block_table, full[:, :2])
    with pytest.raises(RuntimeError, match="descriptor does not resolve"):
        _build_capture_context(owner, buffers, {CUDAGraphMode.FULL: [replace(desc, max_query_len=1)]})


def test_window_capture_delegates_original_dummy_preparation_then_only_rebinds_reads(graph_buffers, monkeypatch):
    from vllm.v1.worker.gpu.cudagraph_utils import AttentionState

    from vllm_ascend.worker.v2.spec_decode.dflash import capture

    case = graph_buffers
    buffers = object.__new__(InputBuffers)
    buffers.query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
    owner = SimpleNamespace(input_buffers=buffers, _draft_window_graph_binding_provider=case.provider)
    ctx = capture._build_capture_context(owner, buffers, {CUDAGraphMode.FULL: [case.capture.descriptor]})
    setattr(buffers, capture._CAPTURE_CONTEXT_ATTRIBUTE, ctx)
    slots, state_marker = object(), object()
    layer = SimpleNamespace(slot_mapping=slots, actual_seq_lengths_q=[1, 2], attn_state=state_marker, causal=False)
    state = AttentionState({"draft.layer": layer}, {"draft.layer": slots})
    calls = []

    def upstream(*args):
        calls.append(args)
        return state

    monkeypatch.setattr(capture, "_ORIGINAL_PREPARE_DFLASH_INPUTS_TO_CAPTURE", upstream)
    full = case.owner.input_block_tables[0].clone()
    result = capture.ascend_prepare_dflash_inputs_to_capture(2, 2, buffers, case.owner, [], object(), 16, False, False)
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
    assert tuple(case.provider._carriers) == (case.capture,)


def test_materialization_rejects_actual_canonical_source_owner_replacement(graph_buffers):
    case = graph_buffers
    case.owner.input_block_tables[0] = case.owner.input_block_tables[0].clone()
    with pytest.raises(RuntimeError, match="replaced after FullSource binding"):
        _materialize(case)


def test_workspace_rejects_invalid_native_table_dtype_at_its_creation_boundary():
    with pytest.raises(TypeError, match="must use int32"):
        DraftWindowWorkspace(
            max_num_reqs=2, table_capacity_columns=2, table_template=torch.zeros((2, 4), dtype=torch.int64)
        )


def test_nonfull_scratch_narrows_the_actual_bank_only_at_the_ascend_read_seam(graph_buffers):
    from vllm_ascend.worker.v2.spec_decode.draft_window_eager import draft_window_read_inputs

    case = graph_buffers
    plan = compute_draft_window_plan(full_exact_end=11, future_offset=0, window_size=4, block_span_tokens=4)
    inputs = case.provider.materialize(
        proposal=_proposal(),
        idx_mapping=case.mapping,
        read_descriptors=[case.projection.describe(plan, valid_num_blocks=4)],
        cpu_window_exact=torch.tensor([plan.window_visible_length], dtype=torch.int32),
        cpu_window_upper=torch.tensor([plan.window_visible_length + 1], dtype=torch.int32),
        cpu_valid_num_blocks=torch.tensor([4], dtype=torch.int32),
        num_reqs_padded=1,
        step=0,
    )
    assert inputs.workspace.window_block_table.shape[0] == 2
    inputs.validate_for(inputs.identity)
    read = draft_window_read_inputs(
        inputs.workspace,
        actual_request_count=1,
        padded_request_count=1,
        block_tables=[case.source._input_block_tables[0]],
        global_group_id=case.projection.group.gid,
    )
    assert read.block_tables[0].shape == (1, 2)
    assert read.seq_lens.shape == read.seq_lens_cpu_exact.shape == read.seq_lens_cpu_upper_bound.shape == (1,)
    assert read.block_tables[0].data_ptr() == inputs.workspace.window_block_table.data_ptr()
    assert read.seq_lens.data_ptr() == inputs.workspace.device_window_exact.data_ptr()
    assert read.seq_lens_cpu_upper_bound.item() == read.seq_lens_cpu_exact.item() + 1
    case.provider.retire_generation(inputs.identity)
    with pytest.raises(RuntimeError, match="retired Draft inputs"):
        inputs.validate_for(inputs.identity)


def test_equal_dynamic_fields_cannot_forge_an_actual_generation(graph_buffers):
    case = graph_buffers
    current = _materialize(case)
    forged = replace(current.identity)
    assert current.identity != forged
    with pytest.raises(RuntimeError, match="generation identity"):
        current.validate_for(forged)
    with pytest.raises(RuntimeError, match="generation identity"):
        case.provider.retire_generation(forged)
    current.validate_for(current.identity)


def test_graph_source_has_one_readonly_owner_and_no_second_runtime(graph_buffers):
    case = graph_buffers
    with pytest.raises(AttributeError):
        case.provider.source = replace(case.projection, full_source=replace(case.source))
    assert case.provider.source is case.projection
    assert not hasattr(case.provider, "_runtime_state")
    assert not hasattr(case.provider, "_metadata")
    assert case.provider._active_inputs is None


def test_runtime_group_rejects_a_descriptor_for_another_cache_group(graph_buffers):
    case = graph_buffers
    plan = compute_draft_window_plan(full_exact_end=11, future_offset=0, window_size=4, block_span_tokens=4)
    descriptor = replace(case.projection.describe(plan, valid_num_blocks=4), global_group_id=1)
    with pytest.raises(ValueError, match="global gid does not match the runtime source"):
        case.provider.materialize(
            proposal=_proposal(),
            idx_mapping=case.mapping,
            read_descriptors=[descriptor],
            cpu_window_exact=torch.tensor([plan.window_visible_length], dtype=torch.int32),
            cpu_window_upper=torch.tensor([plan.window_visible_length], dtype=torch.int32),
            cpu_valid_num_blocks=torch.tensor([4], dtype=torch.int32),
            num_reqs_padded=1,
            step=0,
        )


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

    return SimpleNamespace(case=case, carrier=carrier, graph=Graph(), metadata=batch[0], task=task)


def test_original_task_resolver_binds_current_metadata_without_replacing_capture_state(typed_update_guard):
    from vllm_ascend.compilation.updatable_graph import ContextSource, UpdatableGraph
    from vllm_ascend.worker.v2.spec_decode.draft_window_graph import DraftWindowGraphCarrierState

    c = typed_update_guard
    c.case.provider.validate_graph(c.graph, route=c.case.capture.route)
    resolved = UpdatableGraph.resolve_tasks(c.graph, ContextSource(c.metadata))
    assert len(resolved) == 1
    actual = resolved[0]
    assert actual.operation is c.task.operation
    assert actual.handle is c.task.handle and actual.event is c.task.event
    assert actual.provider is c.task.provider and actual.provider_index == 0
    assert actual.kwargs["block_table"] is c.metadata["draft.layer"].block_tables
    assert actual.kwargs["actual_seq_lengths_kv"] is c.metadata["draft.layer"].seq_lens_list
    assert actual.kwargs["actual_seq_lengths"] is c.metadata["draft.layer"].actual_seq_lengths_q
    assert c.carrier.state is DraftWindowGraphCarrierState.SUBMITTED
    c.case.provider.finish_update(route=c.case.capture.route)
    assert c.carrier.state is DraftWindowGraphCarrierState.UPDATED


@pytest.mark.parametrize("field", ["provider", "operation"])
def test_capture_contract_rejects_an_unsupported_native_consumer(typed_update_guard, field):
    c = typed_update_guard
    setattr(c.task, field, object())
    with pytest.raises(RuntimeError, match="supported Draft FIA consumer"):
        c.case.provider.validate_graph(c.graph, route=c.case.capture.route)


def test_static_graph_consumer_contract_is_not_rechecked_each_generation(typed_update_guard):
    c = typed_update_guard
    c.case.provider.validate_graph(c.graph, route=c.case.capture.route)
    assert c.case.provider._validated_graphs[c.graph] is c.case.capture.route
    # Captured tasks belong to the immutable graph. Replays use its resolver,
    # not a second hand-built task inventory or handle authentication pass.
    c.case.provider.validate_graph(c.graph, route=c.case.capture.route)
    with pytest.raises(RuntimeError, match="route changed"):
        c.case.provider.validate_graph(c.graph, route=DraftWindowGraphRoute.EAGLE3_FIRST_PASS)


@pytest.mark.parametrize("field", ["block_tables", "seq_lens", "seq_lens_cpu", "seq_lens_list"])
def test_publication_rejects_a_builder_that_lost_the_actual_read_bank(graph_buffers, monkeypatch, field):
    case = graph_buffers
    inputs = _materialize(case)
    original = case.provider._validate_metadata_uses_read_carrier

    def corrupt(metadata, workspace):
        layer = metadata["draft.layer"]
        value = getattr(layer, field)
        setattr(layer, field, value.clone() if isinstance(value, torch.Tensor) else [999, 999])
        original(metadata, workspace)

    monkeypatch.setattr(case.provider, "_validate_metadata_uses_read_carrier", corrupt)
    with pytest.raises(RuntimeError, match="carrier"):
        _bind(case, inputs)
