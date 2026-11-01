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

from vllm_ascend.device.hardware_profile import AttentionBackendFamily
from vllm_ascend.worker.v2.spec_decode.draft_window import (
    DraftWindowExecutionPhase,
    DraftWindowRuntimeContext,
    DraftWindowSourceCapability,
    DraftWindowWorkspace,
    FullDraftKVSource,
    build_draft_window_runtime_state,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_backends import (
    DraftWindowGroupRoute,
    DraftWindowProviderStatus,
    DraftWindowTopology,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_eager import DraftWindowProposalContext
from vllm_ascend.worker.v2.spec_decode.draft_window_graph import (
    DraftWindowGraphBindingProvider,
    DraftWindowGraphRoute,
    DraftWindowGraphWriteSideReference,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_nongraph import DraftWindowBatchPhasePlan, DraftWindowRowPhase
from vllm_ascend.worker.v2.spec_decode.draft_window_plan import compute_draft_window_plan
from vllm_ascend.worker.v2.spec_decode.draft_window_policy import resolve_draft_window_policy


def _proposal(epoch=0, requests=("request-a",)):
    return DraftWindowProposalContext(
        epoch,
        0,
        requests,
        tuple(range(len(requests))),
        DraftWindowBatchPhasePlan((DraftWindowRowPhase.PURE_DECODE,) * len(requests), requests),
    )


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
        _block_tables_owner=owner,
        _input_block_tables=(table,),
        _persistent_num_blocks_cpu=owner.num_blocks.cpu,
    )
    group = DraftWindowGroupRoute(
        gid=0,
        layers=("draft.layer",),
        scheduler_block_size=4,
        table_column_span=4,
        topology=DraftWindowTopology(1, 1),
        status=DraftWindowProviderStatus.DIRECT_DENSE,
        reason="direct CPU fixture",
    )
    context = DraftWindowRuntimeContext(
        AttentionBackendFamily.STANDARD, "FULL", True, 1, 1, 1, 1, 1, False, False, False, False
    )
    projection = DraftWindowSourceCapability(
        policy=resolve_draft_window_policy(requested_window_size=4, method="dflash", target_model_type="qwen3"),
        runtime_route="dflash",
        group=group,
        full_source=source,
        expected_future_offset=0,
        runtime_context=context,
    )
    runtime = build_draft_window_runtime_state(projection)
    provider = DraftWindowGraphBindingProvider(runtime_state=runtime)
    descriptor = BatchExecutionDescriptor(CUDAGraphMode.FULL, num_tokens=2, num_reqs=2, uniform_token_count=1)
    capture = provider.register_capture_identity(route=DraftWindowGraphRoute.DFLASH, descriptor=descriptor, num_reqs=2)
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
        proposal=_proposal(epoch, ("request-a", "request-b")[: len(plans)]),
        idx_mapping=case.mapping,
        read_descriptors=[case.projection.describe(plan, valid_num_blocks=4) for plan in plans],
        cpu_window_exact=exact.clone(),
        cpu_window_upper=exact.clone(),
        cpu_valid_num_blocks=torch.tensor([4] * len(plans) + [0] * (2 - len(plans)), dtype=torch.int32),
        num_reqs_padded=2,
        phase=DraftWindowExecutionPhase.PROPOSAL,
        step=0,
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
    actual = len(inputs.identity.proposal.ordered_request_ids)
    padded = inputs.identity.padded_request_count
    query_width = case.capture.descriptor.num_tokens // case.capture.num_reqs
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
    assert inputs.workspace.window_block_table.data_ptr() == carrier.workspace.window_block_table.data_ptr()
    assert inputs.workspace is carrier.workspace
    assert torch.equal(inputs.workspace.window_block_table, torch.tensor([[12, 13], [23, 24]], dtype=torch.int32))
    assert torch.equal(case.owner.input_block_tables[0], full_before)
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
    assert case.runtime._workspace is None


def test_retirement_preserves_bank_until_next_generation_then_clears_tail(graph_buffers):
    case = graph_buffers
    first = _materialize(case)
    bank = first.workspace
    snapshots = [tensor.clone() for tensor in _bank_tensors(bank)]
    pointers = [tensor.data_ptr() for tensor in _bank_tensors(bank)]
    case.runtime.retire_generation(first.identity)
    assert bank.bound_identity is None
    assert all(torch.equal(tensor, old) for tensor, old in zip(_bank_tensors(bank), snapshots))
    second = _materialize(case, epoch=1, ends=(4,))
    assert [tensor.data_ptr() for tensor in _bank_tensors(bank)] == pointers
    assert second.workspace.window_block_table.tolist() == [[11, 0], [0, 0]]
    assert second.workspace.valid_num_blocks.tolist() == [4, 0]
    for tensor in _bank_tensors(bank)[2:]:
        assert tensor.tolist() == [4, 0]
    with pytest.raises(RuntimeError, match="retired Draft inputs"):
        first.validate_for(first.identity)


def test_active_generation_blocks_reuse_and_preserves_all_backing(graph_buffers):
    case = graph_buffers
    first = _materialize(case)
    snapshots = [tensor.clone() for tensor in _bank_tensors(first.workspace)]
    assert not first.retired
    assert case.runtime._current_workspace is first.workspace
    with pytest.raises(RuntimeError, match="still owns an active read bank"):
        _materialize(case, epoch=1, ends=(4,))
    assert all(torch.equal(tensor, old) for tensor, old in zip(_bank_tensors(first.workspace), snapshots))
    first.validate_for(first.identity)
    case.runtime.retire_generation(first.identity)
    assert first.retired
    assert case.runtime._current_workspace is None
    second = _materialize(case, epoch=1, ends=(4,))
    assert second.identity.proposal.mapping_epoch == first.identity.proposal.mapping_epoch + 1


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
        case.runtime.retire_generation(
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
        case.runtime.publish_metadata(inputs, case.target_metadata, target_metadata=case.target_metadata)
    assert case.runtime.quarantined_identity == inputs.identity
    assert case.runtime._current_workspace is inputs.workspace
    assert case.runtime._active_inputs is inputs
    assert inputs.workspace.bound_identity == inputs.identity
    assert all(torch.equal(tensor, old) for tensor, old in zip(_bank_tensors(inputs.workspace), before))
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
            proposal=_proposal(0, ()),
            idx_mapping=case.mapping,
            read_descriptors=(),
            cpu_window_exact=torch.zeros(2, dtype=torch.int32),
            cpu_window_upper=torch.zeros(2, dtype=torch.int32),
            num_reqs_padded=2,
            phase=DraftWindowExecutionPhase.PROPOSAL,
            step=0,
            output_workspace=bank,
        )
    assert bank.bound_identity is None
    assert all(torch.equal(tensor, old) for tensor, old in zip(_bank_tensors(bank), before))
    assert case.runtime._workspace is None


def test_eagle_serialized_steps_own_distinct_banks_and_preserve_previous_consumers(graph_buffers):
    case = graph_buffers
    capability = case.runtime.capability
    projection = replace(
        capability,
        policy=replace(capability.policy, normalized_method="eagle3"),
        runtime_route="eagle3",
        expected_future_offset=3,
    )
    runtime = build_draft_window_runtime_state(projection)
    proposal = _proposal(7)
    provider = DraftWindowGraphBindingProvider(runtime_state=runtime)
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
            num_reqs=padded,
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
            inputs = runtime.materialize(phase=DraftWindowExecutionPhase.PROPOSAL, **kwargs)
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
        assert inputs.identity.proposal.mapping_epoch == 7 and inputs.identity.step == step
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
    import ast
    from pathlib import Path

    source = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/v2/spec_decode/draft_window_graph_runtime.py"
    cls = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "DraftWindowGraphSession"
    )
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "resolve_capture_identity")
    ns: dict[str, Any] = {}
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(
                    body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
                    type_ignores=[],
                )
            ),
            str(source),
            "exec",
        ),
        ns,
    )
    session = SimpleNamespace(provider=graph_buffers.provider, route=DraftWindowGraphRoute.DFLASH)
    descriptor = graph_buffers.capture.descriptor
    assert hash(descriptor) == hash(replace(descriptor))
    assert ns[method.name](session, replace(descriptor)) is graph_buffers.capture
    with pytest.raises(RuntimeError, match="descriptor does not resolve"):
        ns[method.name](session, replace(descriptor, **{field: value}))


@pytest.mark.parametrize("field,value", [("num_active_loras", 1), ("num_ubatches", 2)])
def test_unqualified_descriptor_variants_are_not_opened(graph_buffers, field, value):
    with pytest.raises(NotImplementedError, match="LoRA or microbatch"):
        graph_buffers.provider.register_capture_identity(
            route=DraftWindowGraphRoute.DFLASH,
            descriptor=replace(graph_buffers.capture.descriptor, **{field: value}),
            num_reqs=2,
        )


def test_capture_token_cannot_be_forged_or_cross_provider(graph_buffers):
    case = graph_buffers
    with pytest.raises(RuntimeError, match="not registered"):
        case.provider.prepare_workspace(replace(case.capture))
    other = DraftWindowGraphBindingProvider(runtime_state=case.runtime)
    foreign = other.register_capture_identity(route=case.capture.route, descriptor=case.capture.descriptor, num_reqs=2)
    with pytest.raises(RuntimeError, match="not registered"):
        case.provider.prepare_workspace(foreign)


@pytest.mark.parametrize(
    "field", ["window_block_table", "valid_num_blocks", "device_window_exact", "cpu_window_exact", "cpu_window_upper"]
)
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
    assert bank.max_num_reqs == case.capture.num_reqs
    bank.max_num_reqs += 1
    with pytest.raises(RuntimeError, match="request geometry changed"):
        case.provider.startup_capture_workspace(case.capture)


def test_parallel_capture_uses_exact_manager_descriptor_and_preserves_full_tables(graph_buffers):
    from vllm.v1.worker.gpu.input_batch import InputBuffers

    from vllm_ascend.worker.v2.spec_decode.dflash.capture import _build_capture_context, _seed_capture_workspace

    case = graph_buffers
    buffers = object.__new__(InputBuffers)
    owner = SimpleNamespace(
        input_buffers=buffers,
        _draft_window_graph_binding_provider=case.provider,
        model_state=SimpleNamespace(attn_metadata=case.target_metadata),
    )
    desc = case.capture.descriptor
    ctx = _build_capture_context(owner, buffers, {CUDAGraphMode.FULL: [desc]}, max_num_reqs=2)
    assert ctx is not None
    identity, read = ctx.resolve(2, 2, False)
    assert identity is case.capture
    full = case.source._input_block_tables[0]
    before = full.clone()
    maximum = _seed_capture_workspace(identity, read, (full,), max_model_len=16, provider=case.provider)
    assert maximum == 7 and read.cpu_window_exact.tolist() == read.cpu_window_upper.tolist() == [7, 7]
    assert read.device_window_exact.tolist() == [7, 7] and read.valid_num_blocks.tolist() == [2, 2]
    assert torch.equal(full, before)
    assert torch.equal(read.window_block_table, full[:, :2])
    with pytest.raises(RuntimeError, match="does not resolve exactly one"):
        _build_capture_context(owner, buffers, {CUDAGraphMode.FULL: [replace(desc, max_query_len=1)]}, max_num_reqs=2)


def test_materialization_rejects_actual_canonical_source_owner_replacement(graph_buffers):
    case = graph_buffers
    case.owner.input_block_tables[0] = case.owner.input_block_tables[0].clone()
    with pytest.raises(RuntimeError, match="replaced after FullSource binding"):
        _materialize(case)


@pytest.mark.parametrize("field", ["table", "counts"])
def test_workspace_rejects_invalid_native_dtype_at_its_creation_boundary(field):
    table = torch.zeros((2, 4), dtype=torch.int64 if field == "table" else torch.int32)
    counts = torch.zeros(2, dtype=torch.int64 if field == "counts" else torch.int32)
    with pytest.raises(TypeError, match="must use int32"):
        DraftWindowWorkspace(
            max_num_reqs=2, table_capacity_columns=2, table_template=table, valid_count_template=counts
        )


def test_nonfull_scratch_narrows_the_actual_bank_only_at_the_ascend_read_seam(graph_buffers):
    from vllm_ascend.worker.v2.spec_decode.draft_window_eager import DraftWindowGraphInputMixin

    case = graph_buffers
    plan = compute_draft_window_plan(full_exact_end=11, future_offset=0, window_size=4, block_span_tokens=4)
    inputs = case.runtime.materialize(
        proposal=_proposal(),
        idx_mapping=case.mapping,
        read_descriptors=[case.projection.describe(plan, valid_num_blocks=4)],
        cpu_window_exact=torch.tensor([plan.window_visible_length], dtype=torch.int32),
        cpu_window_upper=torch.tensor([plan.window_visible_length + 1], dtype=torch.int32),
        cpu_valid_num_blocks=torch.tensor([4], dtype=torch.int32),
        num_reqs_padded=1,
        phase=DraftWindowExecutionPhase.PROPOSAL,
        step=0,
    )
    assert inputs.workspace.window_block_table.shape[0] == 2
    owner: Any = SimpleNamespace(block_tables=SimpleNamespace(input_block_tables=[case.source._input_block_tables[0]]))
    read = DraftWindowGraphInputMixin._validated_draft_attention_read_inputs(owner, inputs, inputs.identity)
    assert read.block_tables[0].shape == (1, 2)
    assert read.seq_lens.shape == read.seq_lens_cpu_exact.shape == read.seq_lens_cpu_upper_bound.shape == (1,)
    assert read.block_tables[0].data_ptr() == inputs.workspace.window_block_table.data_ptr()
    assert read.seq_lens.data_ptr() == inputs.workspace.device_window_exact.data_ptr()
    assert read.seq_lens_cpu_upper_bound.item() == read.seq_lens_cpu_exact.item() + 1
    case.runtime.retire_generation(inputs.identity)
    with pytest.raises(RuntimeError, match="retired Draft inputs"):
        DraftWindowGraphInputMixin._validated_draft_attention_read_inputs(owner, inputs, inputs.identity)


def test_equal_dynamic_fields_cannot_forge_an_actual_generation(graph_buffers):
    case = graph_buffers
    current = _materialize(case)
    forged = replace(current.identity)
    assert current.identity != forged
    with pytest.raises(RuntimeError, match="generation identity"):
        current.validate_for(forged)
    with pytest.raises(RuntimeError, match="generation identity"):
        case.runtime.retire_generation(forged)
    current.validate_for(current.identity)


def test_runtime_rejects_replacing_its_fixed_canonical_source(graph_buffers):
    case = graph_buffers
    case.runtime.capability = replace(case.runtime.capability, full_source=replace(case.source))
    with pytest.raises(RuntimeError, match="canonical source changed"):
        _materialize(case)
    assert case.runtime._active_inputs is None
    assert case.runtime._current_workspace is None


def test_runtime_group_rejects_a_descriptor_for_another_cache_group(graph_buffers):
    case = graph_buffers
    plan = compute_draft_window_plan(full_exact_end=11, future_offset=0, window_size=4, block_span_tokens=4)
    descriptor = replace(case.projection.describe(plan, valid_num_blocks=4), global_group_id=1)
    with pytest.raises(ValueError, match="global gid does not match the runtime source"):
        case.runtime.materialize(
            proposal=_proposal(),
            idx_mapping=case.mapping,
            read_descriptors=[descriptor],
            cpu_window_exact=torch.tensor([plan.window_visible_length], dtype=torch.int32),
            cpu_window_upper=torch.tensor([plan.window_visible_length], dtype=torch.int32),
            cpu_valid_num_blocks=torch.tensor([4], dtype=torch.int32),
            num_reqs_padded=1,
            phase=DraftWindowExecutionPhase.PROPOSAL,
            step=0,
        )


@pytest.fixture
def typed_update_guard(graph_buffers):
    import torch_npu

    from vllm_ascend.attention.attention_v1 import FIAParamProvider

    case = graph_buffers
    inputs = _materialize(case)
    carrier = _bind(case, inputs)
    metadata = case.provider.consume_metadata(route=case.capture.route, expected_consumer_count=1)[0]["draft.layer"]
    provider = FIAParamProvider(layer_name="draft.layer", sliding_window=None, is_draft_model=True)
    captured = SimpleNamespace(handle=object(), event=object(), provider=provider, provider_index=0)
    task = SimpleNamespace(
        handle=captured.handle,
        event=captured.event,
        provider=provider,
        provider_index=0,
        operation=torch_npu.npu_fused_infer_attention_score.out,
        kwargs={
            "block_table": metadata.block_tables,
            "actual_seq_lengths_kv": metadata.seq_lens_list,
            "actual_seq_lengths": metadata.actual_seq_lengths_q,
        },
    )
    return SimpleNamespace(case=case, carrier=carrier, graph=SimpleNamespace(tasks=[captured]), task=task)


def test_update_uses_actual_submitted_metadata_and_marks_only_that_bank_updated(typed_update_guard):
    from vllm_ascend.worker.v2.spec_decode.draft_window_graph import DraftWindowGraphCarrierState

    c = typed_update_guard
    c.case.provider.validate_update(c.graph, [c.task], route=c.case.capture.route)
    assert c.carrier.state is DraftWindowGraphCarrierState.UPDATED


@pytest.mark.parametrize(
    "field",
    ["handle", "event", "provider_index", "operation", "block_table", "actual_seq_lengths_kv", "actual_seq_lengths"],
)
def test_update_rejects_stale_handles_or_replaced_operands(typed_update_guard, field):
    from vllm_ascend.worker.v2.spec_decode.draft_window_graph import DraftWindowGraphCarrierState

    c = typed_update_guard
    if field == "provider_index":
        c.task.provider_index = 1
    elif field in c.task.kwargs:
        c.task.kwargs[field] = object()
    else:
        setattr(c.task, field, object())
    with pytest.raises(RuntimeError):
        c.case.provider.validate_update(c.graph, [c.task], route=c.case.capture.route)
    assert c.carrier.state is DraftWindowGraphCarrierState.SUBMITTED
