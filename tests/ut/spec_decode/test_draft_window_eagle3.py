# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU/static contracts for P4b Eagle3 static-K eager Draft Windows."""

import ast
import hashlib
import inspect
import textwrap
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
    _prepare_decode_inputs_kernel,
    _prepare_prefill_inputs_kernel,
    _update_draft_inputs_kernel,
    prepare_decode_inputs,
    prepare_prefill_inputs,
    update_draft_inputs,
)
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionMetadataBuilder,
    AscendAttentionState,
)
from vllm_ascend.worker.v2.spec_decode.autoregressive.speculator import (
    AscendAutoRegressiveSpeculator,
)
from vllm_ascend.worker.v2.spec_decode.draft_window import (
    DraftMetadataInputs,
    DraftWindowExecutionPhase,
    DraftWindowGroupInventory,
    DraftWindowInventory,
    DraftWindowLayoutCategory,
    DraftWindowSourceCapability,
    SingleGroupDirectPrefixProvider,
    build_draft_window_runtime_state,
    build_full_draft_kv_source,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_eagle3 import (
    Eagle3DraftWindowMixin,
    build_eagle3_frozen_proposal_plan,
    build_eagle3_step_materialization_plan,
)


def _eagle3_runtime_fixture(
    *,
    block_size: int = 16,
    window_size: int = 64,
    static_k: int = 3,
    max_num_reqs: int = 4,
    row_width: int = 8,
) -> tuple[Any, Any, SimpleNamespace]:
    canonical_counts = torch.full(
        (1, max_num_reqs),
        row_width,
        dtype=torch.int32,
    )
    canonical_table = torch.arange(
        max_num_reqs * row_width,
        dtype=torch.int32,
    ).reshape(max_num_reqs, row_width)
    block_tables = SimpleNamespace(
        block_sizes=[block_size],
        kernel_block_sizes=[block_size],
        input_block_tables=[canonical_table],
        num_blocks=SimpleNamespace(
            gpu=canonical_counts.clone(),
            cpu=canonical_counts,
        ),
    )
    group = DraftWindowGroupInventory(
        global_group_id=0,
        local_consumer_index=0,
        layer_names=("draft.layers.0.self_attn",),
        backend_class=AscendAttentionBackend,
        builder_class=AscendAttentionMetadataBuilder,
        logical_block_size=block_size,
        kernel_block_size=block_size,
        block_span_tokens=block_size,
        causal=True,
        layout_category=DraftWindowLayoutCategory.DENSE_GQA_DIRECT_PREFIX,
        capability_failures=(),
    )
    inventory = DraftWindowInventory(
        method="eagle3",
        target_model_type="qwen3",
        runtime_route="eagle3",
        window_size=window_size,
        groups=(group,),
        empty_global_group_ids=(),
        capability_failures=(),
    )
    full_source = build_full_draft_kv_source(
        inventory=inventory,
        block_tables=block_tables,
    )
    capability = DraftWindowSourceCapability(
        inventory=inventory,
        full_source=full_source,
        provider=SingleGroupDirectPrefixProvider(
            inventory=inventory,
            full_source=full_source,
            legacy_block_size=block_size,
            expected_future_offset=static_k,
        ),
    )
    return build_draft_window_runtime_state(capability), capability, block_tables


def _cpu(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32)


def test_proposal_anchor_is_frozen_while_real_step_end_changes() -> None:
    _runtime, capability, _block_tables = _eagle3_runtime_fixture()
    proposal = build_eagle3_frozen_proposal_plan(
        provider=capability.provider,
        cpu_proposal_l0=_cpu([100]),
        actual_request_count=1,
        padded_request_count=1,
    )

    assert proposal.static_k == 3
    assert proposal.row_plans[0].full_exact_end == 100
    assert proposal.row_plans[0].full_horizon_end == 103
    assert proposal.row_plans[0].window_start == 32

    starts = []
    visible = []
    for step, exact in enumerate((100, 98, 99)):
        materialization = build_eagle3_step_materialization_plan(
            provider=capability.provider,
            proposal=proposal,
            cpu_full_exact=_cpu([exact]),
            cpu_apply_time_full_upper=_cpu([103]),
            cpu_valid_num_blocks=_cpu([8]),
            step=step,
        )
        starts.append(materialization.window_starts[0])
        visible.append(int(materialization.cpu_window_exact[0]))

    assert starts == [32, 32, 32]
    assert visible == [68, 66, 67]


@pytest.mark.parametrize(
    ("proposal_l0", "expected_start"),
    [
        (60, 0),  # H=63 <= W: short context.
        (93, 32),  # H-W=32: exact legacy-block boundary.
        (94, 32),  # H-W=33: floor-align to the same boundary.
    ],
)
def test_proposal_anchor_covers_short_exact_and_nonboundary_cases(
    proposal_l0: int,
    expected_start: int,
) -> None:
    _runtime, capability, _block_tables = _eagle3_runtime_fixture()

    proposal = build_eagle3_frozen_proposal_plan(
        provider=capability.provider,
        cpu_proposal_l0=_cpu([proposal_l0]),
        actual_request_count=1,
        padded_request_count=1,
    )

    assert proposal.row_plans[0].full_horizon_end == proposal_l0 + 3
    assert proposal.row_plans[0].window_start == expected_start


def test_rejection_that_crosses_frozen_anchor_fails_closed() -> None:
    _runtime, capability, _block_tables = _eagle3_runtime_fixture(
        block_size=64,
        window_size=4,
        static_k=4,
    )
    proposal = build_eagle3_frozen_proposal_plan(
        provider=capability.provider,
        cpu_proposal_l0=_cpu([64]),
        actual_request_count=1,
        padded_request_count=1,
    )

    assert proposal.row_plans[0].window_start == 64
    with pytest.raises(ValueError, match="outside the frozen proposal interval"):
        build_eagle3_step_materialization_plan(
            provider=capability.provider,
            proposal=proposal,
            cpu_full_exact=_cpu([63]),
            cpu_apply_time_full_upper=_cpu([68]),
            cpu_valid_num_blocks=_cpu([8]),
            step=1,
        )


def test_rejection_crosses_a_block_boundary_without_recomputing_anchor() -> None:
    _runtime, capability, _block_tables = _eagle3_runtime_fixture(
        block_size=16,
        window_size=32,
        static_k=3,
    )
    proposal = build_eagle3_frozen_proposal_plan(
        provider=capability.provider,
        cpu_proposal_l0=_cpu([65]),
        actual_request_count=1,
        padded_request_count=1,
    )
    assert proposal.row_plans[0].window_start == 32

    # prepare_decode_inputs can rewind across 64 after rejection, but the
    # proposal anchor remains the one frozen from H=65+3.
    step1 = build_eagle3_step_materialization_plan(
        provider=capability.provider,
        proposal=proposal,
        cpu_full_exact=_cpu([63]),
        cpu_apply_time_full_upper=_cpu([66]),
        cpu_valid_num_blocks=_cpu([8]),
        step=1,
    )
    step2 = build_eagle3_step_materialization_plan(
        provider=capability.provider,
        proposal=proposal,
        cpu_full_exact=_cpu([64]),
        cpu_apply_time_full_upper=_cpu([67]),
        cpu_valid_num_blocks=_cpu([8]),
        step=2,
    )

    assert step1.window_starts == step2.window_starts == (32,)
    assert step1.cpu_window_exact.tolist() == [31]
    assert step2.cpu_window_exact.tolist() == [32]


def test_next_proposal_recomputes_anchor_instead_of_reusing_previous_one() -> None:
    _runtime, capability, _block_tables = _eagle3_runtime_fixture(
        block_size=16,
        window_size=32,
        static_k=3,
    )
    first = build_eagle3_frozen_proposal_plan(
        provider=capability.provider,
        cpu_proposal_l0=_cpu([64]),
        actual_request_count=1,
        padded_request_count=1,
    )
    second = build_eagle3_frozen_proposal_plan(
        provider=capability.provider,
        cpu_proposal_l0=_cpu([50]),
        actual_request_count=1,
        padded_request_count=1,
    )

    assert first.row_plans[0].window_start == 32
    assert second.row_plans[0].window_start == 16


def test_future_block_column_is_refreshed_from_canonical_full_source() -> None:
    runtime, capability, block_tables = _eagle3_runtime_fixture(
        block_size=16,
        window_size=64,
        static_k=3,
        max_num_reqs=1,
    )
    proposal = build_eagle3_frozen_proposal_plan(
        provider=capability.provider,
        cpu_proposal_l0=_cpu([64]),
        actual_request_count=1,
        padded_request_count=1,
    )
    step0 = build_eagle3_step_materialization_plan(
        provider=capability.provider,
        proposal=proposal,
        cpu_full_exact=_cpu([64]),
        cpu_apply_time_full_upper=_cpu([67]),
        cpu_valid_num_blocks=_cpu([8]),
        step=0,
    )
    idx_mapping = torch.tensor([0], dtype=torch.int32)
    first = runtime.materialize(
        mapping_epoch=10,
        ordered_request_ids=("A",),
        idx_mapping=idx_mapping,
        read_descriptors=step0.read_descriptors,
        device_window_exact=None,
        cpu_window_exact=step0.cpu_window_exact,
        cpu_window_upper=step0.cpu_window_upper,
        cpu_valid_num_blocks=step0.cpu_valid_num_blocks,
        num_reqs_padded=1,
        phase=DraftWindowExecutionPhase.PROPOSAL,
        step=0,
        execution_lane=0,
    )
    assert first.active_table_columns == (4,)
    runtime.retire_generation(first.identity)

    # A stale Window must never become the source for the next AR step.
    runtime.workspace.window_block_table.fill_(-99)
    step1 = build_eagle3_step_materialization_plan(
        provider=capability.provider,
        proposal=proposal,
        cpu_full_exact=_cpu([65]),
        cpu_apply_time_full_upper=_cpu([67]),
        cpu_valid_num_blocks=_cpu([8]),
        step=1,
    )
    second = runtime.materialize_serialized_step(
        previous_identity=first.identity,
        ordered_request_ids=("A",),
        idx_mapping=idx_mapping,
        read_descriptors=step1.read_descriptors,
        device_window_exact=None,
        cpu_window_exact=step1.cpu_window_exact,
        cpu_window_upper=step1.cpu_window_upper,
        cpu_valid_num_blocks=step1.cpu_valid_num_blocks,
        num_reqs_padded=1,
        step=1,
        execution_lane=0,
    )

    assert second.identity.mapping_epoch == first.identity.mapping_epoch
    assert second.identity.generation_id == first.identity.generation_id + 1
    assert second.identity.phase is DraftWindowExecutionPhase.STEP
    assert second.identity.step == 1
    assert second.active_table_columns == (5,)
    torch.testing.assert_close(
        second.window_block_table[0, :5],
        block_tables.input_block_tables[0][0, :5],
    )
    with pytest.raises(RuntimeError, match="retired lease"):
        first.validate_for(first.identity)


@pytest.mark.parametrize(
    ("request_ids", "step", "error"),
    [
        (("B",), 1, "request fingerprint"),
        (("A",), 2, "strictly advance"),
    ],
)
def test_serialized_step_continuation_rejects_wrong_proposal_identity(
    request_ids: tuple[str, ...],
    step: int,
    error: str,
) -> None:
    runtime, capability, _block_tables = _eagle3_runtime_fixture(max_num_reqs=1)
    proposal = build_eagle3_frozen_proposal_plan(
        provider=capability.provider,
        cpu_proposal_l0=_cpu([64]),
        actual_request_count=1,
        padded_request_count=1,
    )
    plan = build_eagle3_step_materialization_plan(
        provider=capability.provider,
        proposal=proposal,
        cpu_full_exact=_cpu([64]),
        cpu_apply_time_full_upper=_cpu([67]),
        cpu_valid_num_blocks=_cpu([8]),
        step=0,
    )
    idx_mapping = torch.tensor([0], dtype=torch.int32)
    first = runtime.materialize(
        mapping_epoch=3,
        ordered_request_ids=("A",),
        idx_mapping=idx_mapping,
        read_descriptors=plan.read_descriptors,
        device_window_exact=None,
        cpu_window_exact=plan.cpu_window_exact,
        cpu_window_upper=plan.cpu_window_upper,
        cpu_valid_num_blocks=plan.cpu_valid_num_blocks,
        num_reqs_padded=1,
        phase=DraftWindowExecutionPhase.PROPOSAL,
        step=0,
        execution_lane=0,
    )
    runtime.retire_generation(first.identity)

    with pytest.raises(RuntimeError, match=error):
        runtime.materialize_serialized_step(
            previous_identity=first.identity,
            ordered_request_ids=request_ids,
            idx_mapping=idx_mapping,
            read_descriptors=plan.read_descriptors,
            device_window_exact=None,
            cpu_window_exact=plan.cpu_window_exact,
            cpu_window_upper=plan.cpu_window_upper,
            cpu_valid_num_blocks=plan.cpu_valid_num_blocks,
            num_reqs_padded=1,
            step=step,
            execution_lane=0,
        )


def test_serialized_step_rejects_aba_stale_predecessor() -> None:
    runtime, capability, _block_tables = _eagle3_runtime_fixture(max_num_reqs=1)
    proposal = build_eagle3_frozen_proposal_plan(
        provider=capability.provider,
        cpu_proposal_l0=_cpu([64]),
        actual_request_count=1,
        padded_request_count=1,
    )
    plan = build_eagle3_step_materialization_plan(
        provider=capability.provider,
        proposal=proposal,
        cpu_full_exact=_cpu([64]),
        cpu_apply_time_full_upper=_cpu([67]),
        cpu_valid_num_blocks=_cpu([8]),
        step=0,
    )
    idx_mapping = torch.tensor([0], dtype=torch.int32)
    first = runtime.materialize(
        mapping_epoch=3,
        ordered_request_ids=("A",),
        idx_mapping=idx_mapping,
        read_descriptors=plan.read_descriptors,
        device_window_exact=None,
        cpu_window_exact=plan.cpu_window_exact,
        cpu_window_upper=plan.cpu_window_upper,
        cpu_valid_num_blocks=plan.cpu_valid_num_blocks,
        num_reqs_padded=1,
        phase=DraftWindowExecutionPhase.PROPOSAL,
        step=0,
        execution_lane=0,
    )
    runtime.retire_generation(first.identity)
    second = runtime.materialize_serialized_step(
        previous_identity=first.identity,
        ordered_request_ids=("A",),
        idx_mapping=idx_mapping,
        read_descriptors=plan.read_descriptors,
        device_window_exact=None,
        cpu_window_exact=plan.cpu_window_exact,
        cpu_window_upper=plan.cpu_window_upper,
        cpu_valid_num_blocks=plan.cpu_valid_num_blocks,
        num_reqs_padded=1,
        step=1,
        execution_lane=0,
    )
    runtime.retire_generation(second.identity)

    with pytest.raises(RuntimeError, match="most recently retired"):
        runtime.materialize_serialized_step(
            previous_identity=first.identity,
            ordered_request_ids=("A",),
            idx_mapping=idx_mapping,
            read_descriptors=plan.read_descriptors,
            device_window_exact=None,
            cpu_window_exact=plan.cpu_window_exact,
            cpu_window_upper=plan.cpu_window_upper,
            cpu_valid_num_blocks=plan.cpu_valid_num_blocks,
            num_reqs_padded=1,
            step=1,
            execution_lane=0,
        )


def test_regular_materialize_still_rejects_reusing_mapping_epoch() -> None:
    runtime, capability, _block_tables = _eagle3_runtime_fixture(max_num_reqs=1)
    proposal = build_eagle3_frozen_proposal_plan(
        provider=capability.provider,
        cpu_proposal_l0=_cpu([64]),
        actual_request_count=1,
        padded_request_count=1,
    )
    plan = build_eagle3_step_materialization_plan(
        provider=capability.provider,
        proposal=proposal,
        cpu_full_exact=_cpu([64]),
        cpu_apply_time_full_upper=_cpu([67]),
        cpu_valid_num_blocks=_cpu([8]),
        step=0,
    )
    kwargs = dict(
        mapping_epoch=3,
        ordered_request_ids=("A",),
        idx_mapping=torch.tensor([0], dtype=torch.int32),
        read_descriptors=plan.read_descriptors,
        device_window_exact=None,
        cpu_window_exact=plan.cpu_window_exact,
        cpu_window_upper=plan.cpu_window_upper,
        cpu_valid_num_blocks=plan.cpu_valid_num_blocks,
        num_reqs_padded=1,
        phase=DraftWindowExecutionPhase.PROPOSAL,
        step=0,
        execution_lane=0,
    )
    first = runtime.materialize(**kwargs)
    runtime.retire_generation(first.identity)

    with pytest.raises(ValueError, match="strictly increase"):
        runtime.materialize(**kwargs)


def _eagle3_input_batch(
    *,
    req_ids: tuple[str, ...] = ("A",),
    mapping: tuple[int, ...] = (0,),
    seq_lens: tuple[int, ...] = (64,),
    upper: tuple[int, ...] = (64,),
    has_prefill: bool = False,
    is_dummy: bool = False,
) -> SimpleNamespace:
    num_reqs = len(req_ids)
    mapping_np = np.asarray(mapping, dtype=np.intp)
    if has_prefill:
        prefill_len = np.full(num_reqs, 4, dtype=np.int32)
        computed = np.zeros(num_reqs, dtype=np.int32)
        scheduled = np.full(num_reqs, 4, dtype=np.int32)
    else:
        prefill_len = np.full(num_reqs, 4, dtype=np.int32)
        computed = prefill_len.copy()
        scheduled = np.ones(num_reqs, dtype=np.int32)
    return SimpleNamespace(
        req_ids=list(req_ids),
        num_reqs=num_reqs,
        idx_mapping_np=mapping_np,
        idx_mapping=torch.tensor(mapping, dtype=torch.int32),
        seq_lens_np=np.asarray(seq_lens, dtype=np.int32),
        seq_lens_cpu_upper_bound=torch.tensor(upper, dtype=torch.int32),
        has_prefill=has_prefill,
        is_prefilling_np=np.full(num_reqs, has_prefill, dtype=np.bool_),
        prefill_len_np=prefill_len,
        num_computed_prefill_tokens_np=computed,
        num_scheduled_tokens=scheduled,
        is_dummy=is_dummy,
    )


class _Eagle3LifecycleHarness(Eagle3DraftWindowMixin):
    pass


def _lifecycle_harness(
    *,
    static_k: int = 3,
    window_size: int = 64,
) -> tuple[_Eagle3LifecycleHarness, SimpleNamespace]:
    runtime, capability, block_tables = _eagle3_runtime_fixture(
        static_k=static_k,
        window_size=window_size,
    )
    harness = _Eagle3LifecycleHarness()
    harness._draft_window_runtime_state = runtime
    harness._draft_window_source_capability = capability
    harness.block_tables = block_tables
    harness.num_speculative_steps = static_k
    harness.max_model_len = 4096
    harness.method = "eagle3"
    harness.speculative_config = SimpleNamespace(
        uses_dynamic_speculative_decoding=lambda: False,
    )
    harness.use_fused_multi_step_decode = True
    harness.input_buffers = SimpleNamespace(
        seq_lens=torch.tensor([64, 0, 0, 0], dtype=torch.int32),
    )
    harness._initialize_eagle3_draft_window_state()
    batch = _eagle3_input_batch()
    harness.input_batch = batch
    return harness, batch


def test_eagle3_initialization_requires_static_k_and_disables_fused_decode() -> None:
    harness, _batch = _lifecycle_harness(static_k=3)

    assert harness.use_fused_multi_step_decode is False
    assert harness._eagle3_window_static_k == 3

    dynamic, _batch = _lifecycle_harness(static_k=3)
    dynamic._eagle3_window_epoch_source = None
    dynamic.speculative_config = SimpleNamespace(
        uses_dynamic_speculative_decoding=lambda: True,
    )
    with pytest.raises(NotImplementedError, match="dynamic"):
        dynamic._initialize_eagle3_draft_window_state()


@pytest.mark.parametrize(("dummy", "profile"), [(True, False), (False, True)])
def test_profile_and_dummy_scope_create_no_production_generation(
    dummy: bool,
    profile: bool,
) -> None:
    harness, batch = _lifecycle_harness()

    with harness._eagle3_draft_window_proposal_scope(
        batch,
        dummy_run=dummy,
        is_profile=profile,
    ):
        assert harness._eagle3_window_bypass_active
        assert harness._eagle3_window_proposal_context is None

    assert harness._eagle3_window_epoch_source._next_epoch == 0
    assert harness._draft_window_runtime_state.workspace.bound_identity is None


def test_scheduler_prefill_classification_does_not_bypass_decode_first_pass() -> None:
    harness, batch = _lifecycle_harness()
    prefill_batch = _eagle3_input_batch(has_prefill=True)

    with harness._eagle3_draft_window_proposal_scope(
        prefill_batch,
        dummy_run=False,
        is_profile=False,
    ):
        assert harness._eagle3_window_bypass_active
    assert harness._eagle3_window_epoch_source._next_epoch == 0

    with (
        pytest.raises(RuntimeError, match="without consuming all Eagle3 steps"),
        harness._eagle3_draft_window_proposal_scope(
            batch,
            dummy_run=False,
            is_profile=False,
        ),
    ):
        assert not harness._eagle3_window_bypass_active
        assert harness._eagle3_window_proposal_context is not None
    assert harness._eagle3_window_epoch_source._next_epoch == 1


def test_continuing_pure_chunked_prefill_uses_baseline_without_window_state() -> None:
    harness, _batch = _lifecycle_harness()
    batch = _eagle3_input_batch(has_prefill=True)
    batch.prefill_len_np[:] = 8
    batch.num_computed_prefill_tokens_np[:] = 0
    batch.num_scheduled_tokens[:] = 4

    with harness._eagle3_draft_window_proposal_scope(
        batch,
        dummy_run=False,
        is_profile=False,
    ):
        assert harness._eagle3_window_bypass_active
        assert harness._eagle3_window_proposal_context is None

    assert harness._eagle3_window_epoch_source._next_epoch == 0
    assert harness._eagle3_window_live_inputs is None
    assert harness._draft_window_runtime_state.workspace.bound_identity is None
    assert harness._draft_window_runtime_state.metadata_store._entry is None


def test_first_pass_freezes_l0_without_device_to_host_snapshot() -> None:
    harness, batch = _lifecycle_harness()
    harness._snapshot_eagle3_decode_step_one = MagicMock()

    with (
        pytest.raises(RuntimeError, match="without consuming all Eagle3 steps"),
        harness._eagle3_draft_window_proposal_scope(
            batch,
            dummy_run=False,
            is_profile=False,
        ),
    ):
        inputs = harness._materialize_eagle3_first_pass(
            num_reqs=1,
            num_reqs_padded=1,
        )
        assert harness._eagle3_window_proposal_plan.row_plans[0].full_exact_end == 64
        assert inputs.identity.step == 0
        assert inputs.cpu_window_exact.tolist() == [64]
        harness._finish_eagle3_window_step(inputs, succeeded=True)

    harness._snapshot_eagle3_decode_step_one.assert_not_called()


def test_one_decode_d2h_snapshot_drives_all_remaining_real_step_ends() -> None:
    harness, batch = _lifecycle_harness(static_k=3)

    with harness._eagle3_draft_window_proposal_scope(
        batch,
        dummy_run=False,
        is_profile=False,
    ):
        first = harness._materialize_eagle3_first_pass(
            num_reqs=1,
            num_reqs_padded=1,
        )
        identities = [first.identity]
        harness._finish_eagle3_window_step(first, succeeded=True)

        # prepare_decode_inputs has applied a two-token rejection: E1=63.
        harness.input_buffers.seq_lens[0] = 63
        harness._snapshot_eagle3_decode_step_one(
            num_reqs=1,
            num_reqs_padded=1,
        )
        harness.input_buffers.seq_lens[0] = 999  # later device changes are not re-snapshotted

        step1 = harness._materialize_eagle3_decode_step(
            step=1,
            num_reqs=1,
            num_reqs_padded=1,
            seq_lens_cpu_upper_bound=batch.seq_lens_cpu_upper_bound,
        )
        assert step1.cpu_window_exact.tolist() == [63]
        identities.append(step1.identity)
        harness._finish_eagle3_window_step(step1, succeeded=True)

        step2 = harness._materialize_eagle3_decode_step(
            step=2,
            num_reqs=1,
            num_reqs_padded=1,
            seq_lens_cpu_upper_bound=batch.seq_lens_cpu_upper_bound,
        )
        assert step2.cpu_window_exact.tolist() == [64]
        identities.append(step2.identity)
        harness._finish_eagle3_window_step(step2, succeeded=True)

    assert harness._eagle3_window_epoch_source._next_epoch == 1
    assert [identity.step for identity in identities] == [0, 1, 2]
    assert len({identity.mapping_epoch for identity in identities}) == 1
    assert [identity.generation_id for identity in identities] == [0, 1, 2]


def test_decode_step_revalidates_request_fingerprint_before_workspace_write() -> None:
    harness, batch = _lifecycle_harness(static_k=3)

    with (
        pytest.raises(RuntimeError, match="request ordering changed"),
        harness._eagle3_draft_window_proposal_scope(
            batch,
            dummy_run=False,
            is_profile=False,
        ),
    ):
        first = harness._materialize_eagle3_first_pass(
            num_reqs=1,
            num_reqs_padded=1,
        )
        harness._finish_eagle3_window_step(first, succeeded=True)
        harness.input_buffers.seq_lens[0] = 65
        harness._snapshot_eagle3_decode_step_one(
            num_reqs=1,
            num_reqs_padded=1,
        )
        batch.req_ids[0] = "replacement-A"
        harness._materialize_eagle3_decode_step(
            step=1,
            num_reqs=1,
            num_reqs_padded=1,
            seq_lens_cpu_upper_bound=batch.seq_lens_cpu_upper_bound,
        )

    assert harness._draft_window_runtime_state.workspace.bound_identity is None


def test_decode_step_detects_same_pointer_in_place_reorder() -> None:
    harness, _batch = _lifecycle_harness(static_k=3)
    batch = _eagle3_input_batch(
        req_ids=("A", "B"),
        mapping=(0, 1),
        seq_lens=(64, 65),
        upper=(64, 65),
    )
    harness.input_batch = batch
    harness.input_buffers.seq_lens[:2].copy_(torch.tensor([64, 65], dtype=torch.int32))
    mapping_pointer = batch.idx_mapping.data_ptr()

    with (
        pytest.raises(RuntimeError, match="request ordering changed"),
        harness._eagle3_draft_window_proposal_scope(
            batch,
            dummy_run=False,
            is_profile=False,
        ),
    ):
        first = harness._materialize_eagle3_first_pass(
            num_reqs=2,
            num_reqs_padded=2,
        )
        harness._finish_eagle3_window_step(first, succeeded=True)
        harness.input_buffers.seq_lens[:2].copy_(torch.tensor([65, 66], dtype=torch.int32))
        harness._snapshot_eagle3_decode_step_one(
            num_reqs=2,
            num_reqs_padded=2,
        )

        batch.req_ids[:] = ["B", "A"]
        batch.idx_mapping.copy_(torch.tensor([1, 0], dtype=torch.int32))
        batch.idx_mapping_np[:] = np.asarray([1, 0], dtype=np.intp)
        assert batch.idx_mapping.data_ptr() == mapping_pointer
        harness._materialize_eagle3_decode_step(
            step=1,
            num_reqs=2,
            num_reqs_padded=2,
            seq_lens_cpu_upper_bound=batch.seq_lens_cpu_upper_bound,
        )

    assert harness._draft_window_runtime_state.workspace.bound_identity is None


def _integration_speculator() -> tuple[AscendAutoRegressiveSpeculator, SimpleNamespace]:
    runtime, capability, block_tables = _eagle3_runtime_fixture(
        static_k=3,
        max_num_reqs=4,
    )
    block_tables.slot_mappings = torch.tensor([[31, 32, 33, 34]], dtype=torch.int64)
    block_tables.cp_size = 1
    block_tables.cp_rank = 0
    block_tables.cp_interleave = 1
    speculator = object.__new__(AscendAutoRegressiveSpeculator)
    speculator._draft_window_runtime_state = runtime
    speculator._draft_window_source_capability = capability
    speculator.block_tables = block_tables
    speculator.num_speculative_steps = 3
    speculator.max_model_len = 4096
    speculator.method = "eagle3"
    speculator.speculative_config = SimpleNamespace(
        uses_dynamic_speculative_decoding=lambda: False,
    )
    speculator.use_fused_multi_step_decode = True
    speculator.input_buffers = SimpleNamespace(
        seq_lens=torch.tensor([64, 0, 0, 0], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 4, 4, 4, 4], dtype=torch.int32),
        positions=torch.tensor([60, 61, 62, 63], dtype=torch.int64),
        dcp_local_seq_lens=None,
    )
    batch = _eagle3_input_batch()
    batch.query_start_loc_np = np.asarray([0, 4], dtype=np.int32)
    speculator.input_batch = batch
    target_tensor = torch.tensor([701, 702], dtype=torch.int32)
    target_numpy = np.asarray([801, 802], dtype=np.int32)
    speculator.model_state = SimpleNamespace(
        attn_metadata={
            "draft.layer": SimpleNamespace(
                attn_state="target-first-pass",
                nested=[target_tensor, target_numpy],
            )
        }
    )
    speculator.draft_attn_layer_names = {"draft.layer"}
    speculator.attn_groups = [[]]
    speculator.kv_cache_config = object()
    speculator.arange = torch.arange(8, dtype=torch.int32)
    speculator.attn_architecture = "GQA"
    speculator.replicated_pcp = False
    speculator._initialize_eagle3_draft_window_state()
    return speculator, batch


def test_eagle3_first_pass_builds_private_metadata_and_preserves_full_write_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    speculator, batch = _integration_speculator()
    target = speculator.model_state.attn_metadata
    target_identity = id(target)
    target_nested = target["draft.layer"].nested[0]
    target_nested_numpy = target["draft.layer"].nested[1]
    full_table = speculator.block_tables.input_block_tables[0].clone()
    positions_before = speculator.input_buffers.positions.clone()
    query_start_before = speculator.input_buffers.query_start_loc.clone()
    full_slots = {"draft.layer": torch.tensor([31, 32, 33, 34], dtype=torch.int64)}
    events: list[str] = []
    captured: dict[str, Any] = {}
    original_validate = DraftMetadataInputs.validate_for

    def validate(self: DraftMetadataInputs, expected: Any) -> None:
        events.append("validate")
        original_validate(self, expected)

    def builder(**kwargs: Any) -> dict[str, Any]:
        events.append("builder")
        captured["builder_kwargs"] = kwargs
        return {
            "draft.layer": SimpleNamespace(
                attn_state="builder-default",
                block_tables=kwargs["block_tables"][0],
                seq_lens=kwargs["seq_lens"],
                seq_lens_cpu=kwargs["seq_lens_cpu"],
                slot_mapping=kwargs["slot_mappings"],
                positions=speculator.input_buffers.positions,
                nested=[target_nested, target_nested_numpy],
            )
        }

    def baseline_prefill(
        _self: Any,
        _num_reqs: int,
        _num_tokens: int,
        metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        *_args: Any,
        **_kwargs: Any,
    ) -> None:
        events.append("forward")
        captured["metadata"] = metadata
        captured["slot_mappings"] = slot_mappings

    monkeypatch.setattr(DraftMetadataInputs, "validate_for", validate)
    monkeypatch.setattr(speculator, "_call_eagle3_draft_attn_metadata_builder", builder)
    monkeypatch.setattr(AutoRegressiveSpeculator, "_prefill", baseline_prefill)

    with speculator._eagle3_draft_window_proposal_scope(
        batch,
        dummy_run=False,
        is_profile=False,
    ):
        speculator._prefill(
            num_reqs=1,
            num_tokens=4,
            attn_metadata=target,
            slot_mappings=full_slots,
            num_tokens_across_dp=None,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
        )
        # Complete the remaining synthetic steps without invoking a model.
        speculator.input_buffers.seq_lens[0] = 65
        speculator._snapshot_eagle3_decode_step_one(num_reqs=1, num_reqs_padded=1)
        for step in (1, 2):
            inputs = speculator._materialize_eagle3_decode_step(
                step=step,
                num_reqs=1,
                num_reqs_padded=1,
                seq_lens_cpu_upper_bound=batch.seq_lens_cpu_upper_bound,
            )
            speculator._finish_eagle3_window_step(inputs, succeeded=True)

    builder_index = events.index("builder")
    assert events[:builder_index] == ["validate"]
    assert builder_index < events.index("forward")
    draft = captured["metadata"]
    assert draft is not target
    assert draft["draft.layer"].attn_state == "target-first-pass"
    assert draft["draft.layer"].nested[0].data_ptr() != target_nested.data_ptr()
    assert not np.shares_memory(draft["draft.layer"].nested[1], target_nested_numpy)
    assert captured["slot_mappings"] is full_slots
    assert captured["builder_kwargs"]["slot_mappings"].data_ptr() == speculator.block_tables.slot_mappings.data_ptr()
    assert captured["builder_kwargs"]["query_start_loc_gpu"].data_ptr() == (
        speculator.input_buffers.query_start_loc.data_ptr()
    )
    assert captured["builder_kwargs"]["query_start_loc_cpu"].tolist() == [0, 4]
    assert captured["builder_kwargs"]["max_query_len"] == 4
    assert draft["draft.layer"].positions is speculator.input_buffers.positions
    assert id(speculator.model_state.attn_metadata) == target_identity
    assert speculator.model_state.attn_metadata["draft.layer"].nested[0] is target_nested
    assert speculator.model_state.attn_metadata["draft.layer"].nested[1] is target_nested_numpy
    torch.testing.assert_close(speculator.block_tables.input_block_tables[0], full_table)
    torch.testing.assert_close(speculator.input_buffers.positions, positions_before)
    torch.testing.assert_close(speculator.input_buffers.query_start_loc, query_start_before)
    assert speculator._draft_window_runtime_state.workspace.bound_identity is None


def test_eagle3_decode_build_and_forward_retire_each_step_without_baseline_metadata_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    speculator, batch = _integration_speculator()
    forwarded: list[dict[str, Any]] = []
    builder_inputs: list[dict[str, Any]] = []
    forward_carriers: list[tuple[list[int], list[int], int]] = []
    batch.seq_lens_np[0] = 100
    batch.seq_lens_cpu_upper_bound[0] = 105
    speculator.input_buffers.seq_lens[0] = 100

    def builder(**kwargs: Any) -> dict[str, Any]:
        builder_inputs.append(
            {
                "seq_lens_cpu": kwargs["seq_lens_cpu"].clone(),
                "seq_lens_cpu_upper_bound": kwargs["seq_lens_cpu_upper_bound"].clone(),
            }
        )
        return {
            "draft.layer": SimpleNamespace(
                attn_state="unset",
                block_tables=kwargs["block_tables"][0],
                seq_lens_cpu=kwargs["seq_lens_cpu"],
                seq_lens_list=kwargs["seq_lens_cpu"].tolist(),
                actual_seq_lengths_q=kwargs["query_start_loc_cpu"][1:].tolist(),
                slot_mapping=kwargs["slot_mappings"],
            )
        }

    def generate(
        _self: Any,
        _num_reqs: int,
        _num_tokens_padded: int,
        metadata: dict[str, Any],
        _slot_mappings: Any,
        *_args: Any,
        **_kwargs: Any,
    ) -> None:
        forwarded.append(metadata)
        layer = metadata["draft.layer"]
        forward_carriers.append(
            (
                list(layer.seq_lens_list),
                list(layer.actual_seq_lengths_q),
                int(layer.seq_lens_cpu[0]),
            )
        )

    monkeypatch.setattr(speculator, "_call_eagle3_draft_attn_metadata_builder", builder)
    monkeypatch.setattr(AutoRegressiveSpeculator, "_generate_draft", generate)
    speculator._update_decode_attn_metadata = MagicMock()

    with speculator._eagle3_draft_window_proposal_scope(
        batch,
        dummy_run=False,
        is_profile=False,
    ):
        first = speculator._materialize_eagle3_first_pass(num_reqs=1, num_reqs_padded=1)
        speculator._finish_eagle3_window_step(first, succeeded=True)
        speculator.input_buffers.seq_lens[0] = 101
        speculator._snapshot_eagle3_decode_step_one(num_reqs=1, num_reqs_padded=1)

        for step in (1, 2):
            metadata = speculator._build_draft_attn_metadata(
                num_reqs=1,
                num_reqs_padded=1,
                num_tokens_padded=1,
                seq_lens_cpu_upper_bound=batch.seq_lens_cpu_upper_bound,
                step=step,
            )
            assert metadata["draft.layer"].attn_state.name == "DecodeOnly"
            speculator._generate_draft(
                1,
                1,
                metadata,
                {"draft.layer": torch.tensor([40 + step])},
                None,
                CUDAGraphMode.NONE,
            )
            assert speculator._draft_window_runtime_state.workspace.bound_identity is None

    assert len(forwarded) == 2
    assert [kwargs["seq_lens_cpu"].tolist() for kwargs in builder_inputs] == [[69], [70]]
    assert [kwargs["seq_lens_cpu_upper_bound"].tolist() for kwargs in builder_inputs] == [[74], [75]]
    assert [carrier[0] for carrier in forward_carriers] == [[69], [70]]
    assert [carrier[1] for carrier in forward_carriers] == [[1], [1]]
    # Baseline full lengths for the same two forwards are 101/102.  The only
    # read-length delta is the proposal-frozen S=32 Window cap; the optimistic
    # planning upper is not used as FIA's physical exact carrier.
    full_exact = [101, 102]
    assert [full - carrier[2] for full, carrier in zip(full_exact, forward_carriers, strict=True)] == [32, 32]
    speculator._update_decode_attn_metadata.assert_not_called()


def test_decode_state_assignment_cannot_mutate_a_builder_returned_target_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    speculator, batch = _integration_speculator()
    target = speculator.model_state.attn_metadata
    target_layer = target["draft.layer"]
    target_state = target_layer.attn_state
    monkeypatch.setattr(
        speculator,
        "_call_eagle3_draft_attn_metadata_builder",
        lambda **_kwargs: target,
    )

    with speculator._eagle3_draft_window_proposal_scope(
        batch,
        dummy_run=False,
        is_profile=False,
    ):
        first = speculator._materialize_eagle3_first_pass(num_reqs=1, num_reqs_padded=1)
        speculator._finish_eagle3_window_step(first, succeeded=True)
        speculator.input_buffers.seq_lens[0] = 65
        speculator._snapshot_eagle3_decode_step_one(num_reqs=1, num_reqs_padded=1)
        metadata = speculator._build_draft_attn_metadata(
            num_reqs=1,
            num_reqs_padded=1,
            num_tokens_padded=1,
            seq_lens_cpu_upper_bound=batch.seq_lens_cpu_upper_bound,
            step=1,
        )

        assert metadata is not target
        assert metadata["draft.layer"] is not target_layer
        assert metadata["draft.layer"].attn_state is AscendAttentionState.DecodeOnly
        assert target_layer.attn_state == target_state
        assert speculator._eagle3_window_live_inputs is not None
        speculator._finish_eagle3_window_step(
            speculator._eagle3_window_live_inputs,
            succeeded=True,
        )
        final = speculator._materialize_eagle3_decode_step(
            step=2,
            num_reqs=1,
            num_reqs_padded=1,
            seq_lens_cpu_upper_bound=batch.seq_lens_cpu_upper_bound,
        )
        speculator._finish_eagle3_window_step(final, succeeded=True)

    assert speculator.model_state.attn_metadata is target
    assert target["draft.layer"] is target_layer


def test_eagle3_window_rejects_missing_or_nonowner_target_metadata_before_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    speculator, batch = _integration_speculator()
    builder = MagicMock()
    monkeypatch.setattr(speculator, "_call_eagle3_draft_attn_metadata_builder", builder)
    speculator.model_state.attn_metadata = {}

    with (
        pytest.raises(ValueError, match="non-empty Target metadata"),
        speculator._eagle3_draft_window_proposal_scope(
            batch,
            dummy_run=False,
            is_profile=False,
        ),
    ):
        speculator._prefill(
            1,
            4,
            {},
            {"draft.layer": torch.arange(4)},
            None,
            CUDAGraphMode.NONE,
        )

    builder.assert_not_called()


def test_eagle3_graph_first_pass_rejects_before_workspace_materialization() -> None:
    speculator, batch = _integration_speculator()

    with (
        pytest.raises(NotImplementedError, match="first-pass Window supports eager"),
        speculator._eagle3_draft_window_proposal_scope(
            batch,
            dummy_run=False,
            is_profile=False,
        ),
    ):
        speculator._prefill(
            1,
            4,
            speculator.model_state.attn_metadata,
            {"draft.layer": torch.arange(4)},
            None,
            CUDAGraphMode.FULL,
        )

    assert speculator._draft_window_runtime_state.workspace.bound_identity is None
    assert speculator._eagle3_window_epoch_source._next_epoch == 1


@pytest.mark.parametrize(("dummy", "profile"), [(True, False), (False, True)])
def test_eligible_profile_and_dummy_execute_real_baseline_prefill_path(
    dummy: bool,
    profile: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    speculator, batch = _integration_speculator()
    baseline_prefill = MagicMock()
    builder = MagicMock()
    monkeypatch.setattr(AutoRegressiveSpeculator, "_prefill", baseline_prefill)
    monkeypatch.setattr(speculator, "_call_eagle3_draft_attn_metadata_builder", builder)

    with speculator._eagle3_draft_window_proposal_scope(
        batch,
        dummy_run=dummy,
        is_profile=profile,
    ):
        speculator._prefill(
            1,
            4,
            speculator.model_state.attn_metadata,
            {"draft.layer": torch.arange(4)},
            None,
            CUDAGraphMode.NONE,
        )

    baseline_prefill.assert_called_once()
    builder.assert_not_called()
    assert speculator._eagle3_window_epoch_source._next_epoch == 0
    assert speculator._draft_window_runtime_state.workspace.bound_identity is None


@pytest.mark.parametrize("failure_site", ["builder", "forward"])
def test_eagle3_first_pass_exception_releases_store_and_lease(
    failure_site: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    speculator, batch = _integration_speculator()

    def builder(**kwargs: Any) -> dict[str, Any]:
        if failure_site == "builder":
            raise RuntimeError("synthetic builder failure")
        return {
            "draft.layer": SimpleNamespace(
                attn_state="unset",
                seq_lens_cpu=kwargs["seq_lens_cpu"],
            )
        }

    def prefill(*_args: Any, **_kwargs: Any) -> None:
        if failure_site == "forward":
            raise RuntimeError("synthetic forward failure")

    monkeypatch.setattr(speculator, "_call_eagle3_draft_attn_metadata_builder", builder)
    monkeypatch.setattr(AutoRegressiveSpeculator, "_prefill", prefill)

    with (
        pytest.raises(RuntimeError, match=f"synthetic {failure_site} failure"),
        speculator._eagle3_draft_window_proposal_scope(
            batch,
            dummy_run=False,
            is_profile=False,
        ),
    ):
        speculator._prefill(
            1,
            4,
            speculator.model_state.attn_metadata,
            {"draft.layer": torch.arange(4)},
            None,
            CUDAGraphMode.NONE,
        )

    assert speculator._draft_window_runtime_state.workspace.bound_identity is None
    assert speculator._draft_window_runtime_state.metadata_store._entry is None
    assert speculator._eagle3_window_live_inputs is None


def _upstream_method_ast_sha256(function: Any) -> str:
    source = textwrap.dedent(inspect.getsource(function))
    normalized = ast.dump(ast.parse(source), include_attributes=False)
    return hashlib.sha256(normalized.encode()).hexdigest()


def test_eagle3_upstream_drift_guard_pins_proposal_and_step_contracts() -> None:
    expected = {
        DraftModelSpeculator.__init__: "845967bd921af0170e727c232a4744ed6402a153230cb3fb0195ef0453db0c20",
        AutoRegressiveSpeculator.propose: "7be4f9d03c982996c6b44ae4520192add871944f0e00e96bf747f4b7af20a958",
        AutoRegressiveSpeculator._prefill: "a61a22cafb2ec5cdd26c6e27e4fe0654c8bdc86a7acb070d0fb6fa18ae175b33",
        AutoRegressiveSpeculator._multi_step_decode: "786624dd70d1dc319131d71c2ba8c41797c6b45c8245c870c446bad818ab9f1e",
        AutoRegressiveSpeculator._generate_draft: "6b7743a5f1dee78e09ab8be5e53fd28f9d7456d47290c784b7f10ac131c08c06",
        prepare_prefill_inputs: "196900ab1deabb6b75edd084866dc0654657bc9198f7d4825a94515990c0f43f",
        prepare_decode_inputs: "b2340a7c82ffc539865b5c668b9aedfa3a43d85a4b727af1a6310dd1dcd6b681",
        update_draft_inputs: "8571c138376ba65a6d52c021a57ba5026d83e86ec941266902da5116d51cfd28",
        _prepare_prefill_inputs_kernel: "fc094fbcf36c318670ece79a9cbbde4013966270c30bbba5c435c5778b8e57cc",
        _prepare_decode_inputs_kernel: "ebd9bbd1817b0d7da3def7442f64a0c82935b8040a528dd5f8554211572c150e",
        _update_draft_inputs_kernel: "6a01da77eb14e736abc6504c9c1cc53383e4cf92f6bec820bb7b1b42728fcdf1",
    }

    assert {function: _upstream_method_ast_sha256(function) for function in expected} == expected
    assert tuple(inspect.signature(AutoRegressiveSpeculator.propose).parameters) == (
        "self",
        "input_batch",
        "attn_metadata",
        "slot_mappings",
        "last_hidden_states",
        "aux_hidden_states",
        "num_sampled",
        "num_rejected",
        "last_sampled",
        "next_prefill_tokens",
        "temperature",
        "seeds",
        "dp_sync",
        "dummy_run",
        "skip_attn_for_dummy_run",
        "mm_inputs",
        "is_profile",
    )
