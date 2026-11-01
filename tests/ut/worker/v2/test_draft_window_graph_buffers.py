# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU buffer ownership contracts; these do not assert NPU completion."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.device.hardware_profile import AttentionBackendFamily
from vllm_ascend.worker.v2.spec_decode.draft_window import (
    DraftWindowExecutionPhase,
    DraftWindowInventory,
    DraftWindowRuntimeContext,
    DraftWindowSourceCapability,
    DraftWindowSourceIdentity,
    DraftWindowWorkspace,
    FullDraftKVSource,
    SingleGroupDirectPrefixProvider,
    build_draft_window_runtime_state,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_backends import (
    DraftWindowGroupRoute,
    DraftWindowProviderStatus,
    DraftWindowTopology,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_eager_capability import _build_topology_fingerprint
from vllm_ascend.worker.v2.spec_decode.draft_window_graph import (
    DraftWindowGraphBindingProvider,
    DraftWindowGraphRoute,
    DraftWindowGraphRouteFingerprint,
    DraftWindowGraphWriteSideReference,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_plan import compute_draft_window_plan


@pytest.fixture
def graph_buffers():
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
        source_identity=DraftWindowSourceIdentity(id(owner), (table.data_ptr(),), owner.num_blocks.cpu.data_ptr()),
        _block_tables_owner=owner,
        _input_block_tables=(table,),
        _persistent_num_blocks_cpu=owner.num_blocks.cpu,
    )
    group = DraftWindowGroupRoute(
        gid=0,
        layers=("draft.layer",),
        backend="dense_backend",
        builder="dense_builder",
        family="GQA",
        kv_specs=("Full",),
        scheduler_block_size=4,
        table_column_span=4,
        physical_block_size=4,
        compression_ratio=1,
        topology=DraftWindowTopology(1, 1, 1, 1, 1, 0, 1, False, False, False, False, False),
        status=DraftWindowProviderStatus.DIRECT_DENSE,
        reason="direct CPU fixture",
    )
    inventory = DraftWindowInventory("dflash", "qwen3", "graph", 4, (group,), ())
    projection = SingleGroupDirectPrefixProvider(inventory, source, legacy_block_size=4, expected_future_offset=0)
    context = DraftWindowRuntimeContext(
        AttentionBackendFamily.STANDARD, "FULL", True, 1, 1, 1, 1, 1, False, False, False, False
    )
    topology = _build_topology_fingerprint(
        inventory=inventory, runtime_context=context, provider_version=1, execution_lane=0
    )
    runtime = build_draft_window_runtime_state(DraftWindowSourceCapability(inventory, source, projection, topology))
    provider = DraftWindowGraphBindingProvider(runtime_state=runtime)
    fingerprint = DraftWindowGraphRouteFingerprint(
        method="dflash",
        route=DraftWindowGraphRoute.DFLASH,
        graph_mode="FULL",
        backend_impl=group.backend,
        builder_impl=group.builder,
        global_group_id=0,
        layout=group.status.value,
        capture_descriptor="two-request",
        capture_bucket=(2, 2),
        query_width=1,
        padded_request_count=2,
        padded_token_count=2,
        static_k=1,
        window_size=4,
        logical_block_size=4,
        kernel_block_size=4,
        block_span_tokens=4,
        tensor_parallel_size=1,
        data_parallel_size=1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        sequence_parallel=False,
        dsa_context_parallel=False,
        flashcomm=False,
        async_scheduling=True,
        prefix_caching=False,
    )
    capture = provider.register_capture_identity(fingerprint=fingerprint, execution_lane=0, consumer_ordinal=0)
    return SimpleNamespace(
        runtime=runtime,
        provider=provider,
        capture=capture,
        projection=projection,
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
    return case.runtime.materialize(
        mapping_epoch=epoch,
        ordered_request_ids=("request-a", "request-b")[: len(plans)],
        idx_mapping=case.mapping,
        read_descriptors=[case.projection.describe(plan, valid_num_blocks=4) for plan in plans],
        cpu_window_exact=exact.clone(),
        cpu_window_upper=exact.clone(),
        cpu_valid_num_blocks=torch.tensor([4] * len(plans) + [0] * (2 - len(plans)), dtype=torch.int32),
        num_reqs_padded=2,
        phase=DraftWindowExecutionPhase.PROPOSAL,
        step=0,
        execution_lane=0,
        quarantine_on_failure=quarantine,
        output_workspace=case.provider.prepare_workspace(case.capture),
    )


def _bank_tensors(bank):
    return (
        bank.window_block_table,
        bank.valid_num_blocks,
        bank.device_window_exact,
        bank.cpu_window_exact,
        bank.cpu_window_upper,
    )


def _bind(case, inputs):
    actual = inputs.actual_request_count
    padded = inputs.padded_request_count
    query_width = case.capture.fingerprint.query_width
    tokens = actual * query_width
    query_lengths = tuple(row * query_width for row in range(1, actual + 1)) + (tokens,) * (padded - actual)
    reference = DraftWindowGraphWriteSideReference(
        slot_mapping=case.owner.slot_mappings[0, :tokens],
        positions=torch.arange(tokens),
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
        write_side_reference=reference,
    )


def test_direct_capture_bank_never_allocates_scratch_or_mutates_full_source(graph_buffers):
    case = graph_buffers
    full_before = case.owner.input_block_tables[0].clone()
    inputs = _materialize(case)
    carrier = _bind(case, inputs)
    assert case.runtime._workspace is None
    assert inputs.window_block_table.data_ptr() == carrier.read_inputs.window_block_table.data_ptr()
    assert inputs.lease.workspace is carrier.workspace
    assert torch.equal(inputs.window_block_table, torch.tensor([[12, 13], [23, 24]], dtype=torch.int32))
    assert torch.equal(case.owner.input_block_tables[0], full_before)
    assert case.target_metadata["target.layer"].seq_lens_list == [91, 92]


def test_first_real_generation_clears_startup_capture_seed_and_padding(graph_buffers):
    case = graph_buffers
    bank = case.provider.prepare_workspace(case.capture)
    for tensor in _bank_tensors(bank):
        tensor.fill_(99)
    inputs = _materialize(case, ends=(4,))
    assert inputs.window_block_table.tolist() == [[11, 0], [0, 0]]
    for tensor in _bank_tensors(bank)[1:]:
        assert tensor[1].item() == 0
    assert inputs.device_window_exact.tolist() == [4, 0]
    assert case.runtime._workspace is None


def test_retirement_preserves_bank_until_next_generation_then_clears_tail(graph_buffers):
    case = graph_buffers
    first = _materialize(case)
    bank = first.lease.workspace
    snapshots = [tensor.clone() for tensor in _bank_tensors(bank)]
    pointers = [tensor.data_ptr() for tensor in _bank_tensors(bank)]
    case.runtime.retire_generation(first.identity)
    assert bank.bound_identity is None
    assert all(torch.equal(tensor, old) for tensor, old in zip(_bank_tensors(bank), snapshots))
    second = _materialize(case, epoch=1, ends=(4,))
    assert [tensor.data_ptr() for tensor in _bank_tensors(bank)] == pointers
    assert second.window_block_table.tolist() == [[11, 0], [0, 0]]
    assert second.valid_num_blocks.tolist() == [4, 0]
    for tensor in _bank_tensors(bank)[2:]:
        assert tensor.tolist() == [4, 0]
    with pytest.raises(RuntimeError, match="retired lease"):
        first.validate_for(first.identity)


def test_pending_reader_blocks_reuse_and_preserves_all_backing(graph_buffers):
    case = graph_buffers
    token = SimpleNamespace(complete=False)
    token.is_complete = lambda: token.complete
    first = _materialize(case)
    snapshots = [tensor.clone() for tensor in _bank_tensors(first.lease.workspace)]
    case.runtime.mark_reader_pending(first, token)
    case.runtime.retire_generation(first.identity)
    assert not case.runtime.poll_reader_completion(first.identity)
    with pytest.raises(RuntimeError, match="still in use"):
        _materialize(case, epoch=1, ends=(4,))
    assert all(torch.equal(tensor, old) for tensor, old in zip(_bank_tensors(first.lease.workspace), snapshots))
    token.complete = True
    assert case.runtime.poll_reader_completion(first.identity)
    second = _materialize(case, epoch=1, ends=(4,))
    assert second.identity.mapping_epoch == first.identity.mapping_epoch + 1


def test_same_mapping_pointer_does_not_allow_epoch_or_generation_aba(graph_buffers):
    case = graph_buffers
    first = _materialize(case)
    case.runtime.retire_generation(first.identity)
    case.mapping.copy_(torch.tensor([1, 0]))
    with pytest.raises(ValueError, match="mapping epoch must strictly increase"):
        _materialize(case)
    current = _materialize(case, epoch=1)
    assert current.identity.idx_mapping_ptr == first.identity.idx_mapping_ptr
    with pytest.raises(RuntimeError, match="generation identity"):
        current.validate_for(first.identity)
    with pytest.raises(RuntimeError, match="generation identity"):
        case.runtime.retire_generation(replace(current.identity, mapping_epoch=0))
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


def test_exception_after_possible_submission_quarantines_bank_without_clearing(graph_buffers):
    case = graph_buffers
    inputs = _materialize(case, quarantine=True)
    before = [tensor.clone() for tensor in _bank_tensors(inputs.lease.workspace)]
    with pytest.raises(ValueError, match="native submission failed"), case.runtime.generation_guard(inputs):
        raise ValueError("native submission failed")
    assert case.runtime.quarantined_identity == inputs.identity
    assert inputs.lease.workspace.bound_identity == inputs.identity
    assert all(torch.equal(tensor, old) for tensor, old in zip(_bank_tensors(inputs.lease.workspace), before))
    with pytest.raises(RuntimeError, match="quarantined"):
        case.provider.prepare_workspace(case.capture)
    with pytest.raises(RuntimeError, match="quarantined"):
        case.runtime.retire_generation(inputs.identity)


def test_cpu_valid_counts_are_required_before_any_bank_materialization(graph_buffers):
    case = graph_buffers
    bank = case.provider.prepare_workspace(case.capture)
    before = [tensor.clone() for tensor in _bank_tensors(bank)]
    with pytest.raises(TypeError, match="cpu_valid_num_blocks"):
        case.runtime.materialize(
            mapping_epoch=0,
            ordered_request_ids=(),
            idx_mapping=case.mapping,
            read_descriptors=(),
            cpu_window_exact=torch.zeros(2, dtype=torch.int32),
            cpu_window_upper=torch.zeros(2, dtype=torch.int32),
            num_reqs_padded=2,
            phase=DraftWindowExecutionPhase.PROPOSAL,
            step=0,
            execution_lane=0,
            output_workspace=bank,
        )
    assert bank.bound_identity is None
    assert all(torch.equal(tensor, old) for tensor, old in zip(_bank_tensors(bank), before))
    assert case.runtime._workspace is None


def test_eagle_serialized_steps_own_distinct_banks_and_preserve_previous_consumers(graph_buffers):
    case = graph_buffers
    capability = case.runtime.capability
    inventory = replace(capability.inventory, method="eagle3")
    projection = SingleGroupDirectPrefixProvider(inventory, case.source, legacy_block_size=4, expected_future_offset=3)
    runtime = build_draft_window_runtime_state(
        replace(
            capability,
            inventory=inventory,
            provider=projection,
            eager_topology_fingerprint=replace(capability.eager_topology_fingerprint, normalized_method="eagle3"),
        )
    )
    provider = DraftWindowGraphBindingProvider(runtime_state=runtime)
    case.owner.slot_mappings = torch.arange(8, dtype=torch.int64).reshape(1, 8)
    proposal_plan = compute_draft_window_plan(full_exact_end=11, future_offset=3, window_size=4, block_span_tokens=4)
    previous = None
    retained: list[tuple[DraftWindowWorkspace, list[torch.Tensor]]] = []
    for step, (padded, query_width) in enumerate(((2, 4), (1, 1), (2, 1))):
        route = DraftWindowGraphRoute.EAGLE3_FIRST_PASS if step == 0 else DraftWindowGraphRoute.EAGLE3_AR_STEP
        capture = provider.register_capture_identity(
            fingerprint=replace(
                case.capture.fingerprint,
                method="eagle3",
                route=route,
                capture_descriptor=f"eagle-step-{step}",
                capture_bucket=(padded, padded * query_width),
                query_width=query_width,
                padded_request_count=padded,
                padded_token_count=padded * query_width,
                static_k=3,
            ),
            execution_lane=0,
            consumer_ordinal=max(step - 1, 0),
        )
        bank = provider.prepare_workspace(capture)
        exact = torch.tensor([11 + step - proposal_plan.window_start] + [0] * (padded - 1), dtype=torch.int32)
        kwargs = dict(
            ordered_request_ids=("request-a",),
            idx_mapping=case.mapping,
            read_descriptors=[
                projection.describe_frozen_step(proposal_plan, full_exact_end=11 + step, valid_num_blocks=4)
            ],
            cpu_window_exact=exact.clone(),
            cpu_window_upper=exact.clone(),
            cpu_valid_num_blocks=torch.tensor([4] + [0] * (padded - 1), dtype=torch.int32),
            num_reqs_padded=padded,
            step=step,
            execution_lane=0,
            output_workspace=bank,
        )
        if previous is None:
            inputs = runtime.materialize(mapping_epoch=7, phase=DraftWindowExecutionPhase.PROPOSAL, **kwargs)
        else:
            if step == 1:
                replaced_mapping = case.mapping.clone()
                assert replaced_mapping.data_ptr() != previous.idx_mapping_ptr
                with pytest.raises(RuntimeError, match="request mapping changed"):
                    runtime.materialize_serialized_step(
                        previous_identity=previous, **{**kwargs, "idx_mapping": replaced_mapping}
                    )
                assert bank.bound_identity is None
            inputs = runtime.materialize_serialized_step(previous_identity=previous, **kwargs)
        current = SimpleNamespace(**{**vars(case), "runtime": runtime, "provider": provider, "capture": capture})
        carrier = _bind(current, inputs)
        assert carrier.runtime_identity is inputs.identity
        assert runtime._workspace is None
        assert inputs.identity.mapping_epoch == 7 and inputs.identity.step == step
        for old_bank, old_values in retained:
            for tensor, old_tensor, snapshot in zip(_bank_tensors(bank), _bank_tensors(old_bank), old_values):
                assert tensor.untyped_storage().data_ptr() != old_tensor.untyped_storage().data_ptr()
                assert torch.equal(old_tensor, snapshot)
        snapshots = [tensor.clone() for tensor in _bank_tensors(bank)]
        runtime.retire_generation(inputs.identity)
        assert all(torch.equal(tensor, snapshot) for tensor, snapshot in zip(_bank_tensors(bank), snapshots))
        retained.append((bank, snapshots))
        previous = inputs.identity
    assert len(retained) == 3
