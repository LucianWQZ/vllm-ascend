# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""P7-X non-Graph mandatory delivery contracts."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionMetadataBuilder,
)
from vllm_ascend.attention.context_parallel.dsa_cp import (
    AscendDSACPMetadataBuilder,
)
from vllm_ascend.attention.dsa_v1 import (
    AscendDSABackend,
    AscendDSAMetadataBuilder,
)
from vllm_ascend.device.hardware_profile import AttentionBackendFamily
from vllm_ascend.worker.v2.spec_decode.draft_window import (
    DraftWindowGroupInventory,
    DraftWindowInventory,
    DraftWindowLayoutCategory,
    DraftWindowPolicyState,
    DraftWindowRuntimeContext,
    DraftWindowSourceCapability,
    SingleGroupDirectPrefixProvider,
    build_full_draft_kv_source,
    compute_draft_window_scalar_plan,
    resolve_draft_window_policy,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_nongraph import (
    DraftWindowBackendProviderKind,
    DraftWindowBatchPhase,
    DraftWindowRowPhase,
    P7XCapabilityStatus,
    build_phase_aware_materialization_plan,
    build_prefix_resident_snapshot,
    classify_draft_window_batch_phase,
    evaluate_p7x_nongraph_capability,
)


def _batch(
    *,
    req_ids: tuple[str, ...],
    computed: tuple[int, ...],
    scheduled: tuple[int, ...],
    prompt: tuple[int, ...],
    mapping: tuple[int, ...] | None = None,
) -> SimpleNamespace:
    count = len(req_ids)
    if mapping is None:
        mapping = tuple(range(count))
    computed_np = np.asarray(computed, dtype=np.int32)
    prompt_np = np.asarray(prompt, dtype=np.int32)
    is_prefilling = computed_np < prompt_np
    mapping_np = np.asarray(mapping, dtype=np.intp)
    return SimpleNamespace(
        num_reqs=count,
        req_ids=list(req_ids),
        idx_mapping_np=mapping_np,
        idx_mapping=torch.tensor(mapping, dtype=torch.int32),
        num_computed_prefill_tokens_np=computed_np,
        num_scheduled_tokens=np.asarray(scheduled, dtype=np.int32),
        prefill_len_np=prompt_np,
        is_prefilling_np=is_prefilling,
        has_prefill=bool(np.any(is_prefilling)),
    )


def _capability(*, prefix: bool = False, tp: int = 1) -> tuple[DraftWindowSourceCapability, SimpleNamespace]:
    rows = 8
    width = 16
    counts = torch.full((1, rows), width, dtype=torch.int32)
    block_tables = SimpleNamespace(
        block_sizes=[64],
        kernel_block_sizes=[64],
        input_block_tables=[torch.arange(rows * width, dtype=torch.int32).reshape(rows, width)],
        num_blocks=SimpleNamespace(cpu=counts, gpu=counts.clone()),
    )
    group = DraftWindowGroupInventory(
        global_group_id=0,
        local_consumer_index=0,
        layer_names=("draft.layers.0.self_attn",),
        backend_class=AscendAttentionBackend,
        builder_class=AscendAttentionMetadataBuilder,
        logical_block_size=64,
        kernel_block_size=64,
        block_span_tokens=64,
        causal=True,
        layout_category=DraftWindowLayoutCategory.DENSE_GQA_DIRECT_PREFIX,
        capability_failures=(),
    )
    inventory = DraftWindowInventory(
        method="dflash",
        target_model_type="qwen3",
        runtime_route="dflash",
        window_size=128,
        groups=(group,),
        empty_global_group_ids=(),
        capability_failures=(),
    )
    source = build_full_draft_kv_source(inventory=inventory, block_tables=block_tables)
    topology = evaluate_p7x_nongraph_capability(
        policy=resolve_draft_window_policy(
            requested_window_size=128,
            method="dflash",
            target_model_type="qwen3",
        ),
        inventory=inventory,
        runtime_context=DraftWindowRuntimeContext(
            attention_backend_family=AttentionBackendFamily.STANDARD,
            graph_mode="NONE",
            async_scheduling=False,
            tensor_parallel_size=tp,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            sequence_parallel=False,
            dsa_context_parallel=False,
            flashcomm=False,
            prefix_caching=prefix,
            tensor_parallel_rank=min(1, tp - 1),
        ),
        uses_dynamic_k=False,
        provider_version=2,
        execution_lane=0,
    ).require_local_materialization()
    capability = DraftWindowSourceCapability(
        inventory=inventory,
        full_source=source,
        provider=SingleGroupDirectPrefixProvider(
            inventory=inventory,
            full_source=source,
            legacy_block_size=64,
            expected_future_offset=0,
        ),
        eager_topology_fingerprint=topology,
        workspace_capacity_columns=width,
    )
    return capability, block_tables


def test_row_phase_classifier_distinguishes_continuing_terminal_decode_and_mixed() -> None:
    plan = classify_draft_window_batch_phase(
        _batch(
            req_ids=("continue", "terminal", "decode"),
            computed=(32, 60, 80),
            scheduled=(16, 4, 1),
            prompt=(80, 64, 80),
        )
    )

    assert plan is not None
    assert plan.batch_phase is DraftWindowBatchPhase.MIXED
    assert plan.row_phases == (
        DraftWindowRowPhase.PREFILL_CONTINUING,
        DraftWindowRowPhase.PREFILL_TERMINAL,
        DraftWindowRowPhase.PURE_DECODE,
    )
    assert plan.baseline_only_rows == (0,)
    assert plan.window_rows == (1, 2)
    assert plan.row_phase_fingerprint == (
        "prefill_continuing",
        "prefill_terminal",
        "pure_decode",
    )


def test_phase_aware_plan_keeps_continuing_row_full_and_windows_decode_rows() -> None:
    capability, _ = _capability()
    phases = classify_draft_window_batch_phase(
        _batch(
            req_ids=("P", "D"),
            computed=(64, 320),
            scheduled=(32, 1),
            prompt=(256, 256),
        )
    )

    plan = build_phase_aware_materialization_plan(
        provider=capability.provider,
        phase_plan=phases,
        cpu_full_exact=torch.tensor([96, 321], dtype=torch.int32),
        cpu_apply_time_full_upper=torch.tensor([104, 322], dtype=torch.int32),
        cpu_valid_num_blocks=torch.tensor([8, 8], dtype=torch.int32),
        actual_request_count=2,
        padded_request_count=2,
    )

    prefill, decode = plan.read_descriptors
    assert prefill.source_start_column == 0
    assert prefill.required_columns == 8
    assert plan.cpu_window_exact.tolist()[0] == 96
    assert decode.source_start_column == 3
    assert decode.required_columns == 3
    assert plan.cpu_window_exact.tolist()[1] == 129
    assert plan.row_phase_fingerprint == phases.row_phase_fingerprint


def test_prefix_resident_snapshot_tracks_mapping_and_detects_eviction() -> None:
    capability, block_tables = _capability(prefix=True)
    batch = _batch(
        req_ids=("A", "B"),
        computed=(256, 256),
        scheduled=(1, 1),
        prompt=(256, 256),
        mapping=(3, 5),
    )
    block_tables.num_blocks.cpu[0, 3] = 7
    block_tables.num_blocks.cpu[0, 5] = 6

    first = build_prefix_resident_snapshot(
        full_source=capability.full_source,
        global_group_id=0,
        idx_mapping_np=batch.idx_mapping_np,
        ordered_request_ids=batch.req_ids,
        num_reqs=2,
        num_reqs_padded=2,
    )
    assert first.valid_num_blocks.tolist() == [7, 6]
    assert first.mapping_fingerprint == (3, 5)

    block_tables.num_blocks.cpu[0, 3] = 4
    second = build_prefix_resident_snapshot(
        full_source=capability.full_source,
        global_group_id=0,
        idx_mapping_np=batch.idx_mapping_np,
        ordered_request_ids=batch.req_ids,
        num_reqs=2,
        num_reqs_padded=2,
    )
    assert second.valid_num_blocks.tolist() == [4, 6]
    assert second.source_identity == first.source_identity
    assert second.mapping_fingerprint == first.mapping_fingerprint
    assert not torch.equal(second.valid_num_blocks, first.valid_num_blocks)


@pytest.mark.parametrize("tp_size", [2, 4, 16])
def test_dense_tp_is_replicated_and_locally_materializable(tp_size: int) -> None:
    capability, _ = _capability(tp=tp_size)
    topology = capability.eager_topology_fingerprint
    assert topology is not None
    assert topology.tensor_parallel_size == tp_size
    assert topology.tensor_parallel_rank == min(1, tp_size - 1)
    assert topology.parallel_read_semantics == "replicated"


def test_graph_and_compatibility_family_remain_closed_in_p7x() -> None:
    capability, _ = _capability()
    inventory = capability.inventory
    policy = resolve_draft_window_policy(
        requested_window_size=128,
        method="dflash",
        target_model_type="qwen3",
    )
    base = DraftWindowRuntimeContext(
        attention_backend_family=AttentionBackendFamily.STANDARD,
        graph_mode="NONE",
        async_scheduling=False,
        tensor_parallel_size=1,
        data_parallel_size=1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        sequence_parallel=False,
        dsa_context_parallel=False,
        flashcomm=False,
        prefix_caching=False,
    )
    graph = evaluate_p7x_nongraph_capability(
        policy=policy,
        inventory=inventory,
        runtime_context=replace(base, graph_mode="PIECEWISE"),
        uses_dynamic_k=False,
        provider_version=2,
        execution_lane=0,
    )
    compatibility = evaluate_p7x_nongraph_capability(
        policy=policy,
        inventory=inventory,
        runtime_context=replace(
            base,
            attention_backend_family=AttentionBackendFamily.COMPATIBILITY,
        ),
        uses_dynamic_k=False,
        provider_version=2,
        execution_lane=0,
    )

    assert graph.status is P7XCapabilityStatus.GRAPH_FROZEN
    assert compatibility.status is P7XCapabilityStatus.SOURCE_BLOCKED
    with pytest.raises(NotImplementedError):
        graph.require_local_materialization()


def test_backend_provider_kind_is_explicit_not_inferred_from_model_name() -> None:
    capability, _ = _capability()
    decision = evaluate_p7x_nongraph_capability(
        policy=resolve_draft_window_policy(
            requested_window_size=128,
            method="dflash",
            target_model_type="not-a-real-model-name",
        ),
        inventory=replace(capability.inventory, target_model_type="not-a-real-model-name"),
        runtime_context=DraftWindowRuntimeContext(
            attention_backend_family=AttentionBackendFamily.STANDARD,
            graph_mode="NONE",
            async_scheduling=False,
            tensor_parallel_size=1,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            sequence_parallel=False,
            dsa_context_parallel=False,
            flashcomm=False,
            prefix_caching=False,
        ),
        uses_dynamic_k=False,
        provider_version=2,
        execution_lane=0,
    )
    assert decision.status is P7XCapabilityStatus.LOCAL_STATIC_READY
    assert decision.backend_provider_kinds == (DraftWindowBackendProviderKind.DENSE_DIRECT_PREFIX,)


@pytest.mark.parametrize(
    ("dsa_cp", "builder_class", "expected_kind"),
    [
        (False, AscendDSAMetadataBuilder, DraftWindowBackendProviderKind.DSA_NATIVE_SWA),
        (True, AscendDSACPMetadataBuilder, DraftWindowBackendProviderKind.DSA_CP_NATIVE_SWA),
    ],
)
def test_uncompressed_single_group_dsa_provider_is_source_safe(
    dsa_cp: bool,
    builder_class: type[object],
    expected_kind: DraftWindowBackendProviderKind,
) -> None:
    dense, block_tables = _capability(tp=2 if dsa_cp else 1)
    source_group = replace(
        dense.inventory.groups[0],
        backend_class=AscendDSABackend,
        builder_class=builder_class,
        layout_category=DraftWindowLayoutCategory.DSA,
        block_span_tokens=None,
        capability_failures=("dsa has no verified P2 direct-prefix provider",),
        source_geometry_failures=(),
        drafting_compression_ratio=1,
    )
    inventory = replace(
        dense.inventory,
        groups=(source_group,),
        capability_failures=(("DSA-CP is not verified in P2",) if dsa_cp else ()),
    )
    runtime = DraftWindowRuntimeContext(
        attention_backend_family=AttentionBackendFamily.STANDARD,
        graph_mode="NONE",
        async_scheduling=False,
        tensor_parallel_size=2 if dsa_cp else 1,
        data_parallel_size=1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        sequence_parallel=False,
        dsa_context_parallel=dsa_cp,
        flashcomm=dsa_cp,
        prefix_caching=False,
        tensor_parallel_rank=1 if dsa_cp else 0,
    )

    decision = evaluate_p7x_nongraph_capability(
        policy=resolve_draft_window_policy(
            requested_window_size=128,
            method="dflash",
            target_model_type="runtime-owned",
        ),
        inventory=inventory,
        runtime_context=runtime,
        uses_dynamic_k=False,
        provider_version=2,
        execution_lane=0,
    )

    assert decision.status is P7XCapabilityStatus.LOCAL_STATIC_READY
    assert decision.backend_provider_kinds == (expected_kind,)
    source = build_full_draft_kv_source(
        inventory=inventory,
        block_tables=block_tables,
        source_group=source_group,
    )
    provider = SingleGroupDirectPrefixProvider(
        inventory=inventory,
        full_source=source,
        legacy_block_size=64,
        expected_future_offset=0,
        explicit_source_group=source_group,
        source_block_span_tokens=64,
    )
    descriptor = provider.describe(
        compute_draft_window_scalar_plan(
            full_exact_end=320,
            future_offset=0,
            window_size=128,
            legacy_block_size=64,
        ),
        valid_num_blocks=16,
    )
    assert descriptor.source_start_column == 3
    assert descriptor.required_columns == 2


def test_compressed_dsa_and_dsa_async_remain_gated_before_source() -> None:
    dense, _ = _capability()
    group = replace(
        dense.inventory.groups[0],
        backend_class=AscendDSABackend,
        builder_class=AscendDSAMetadataBuilder,
        layout_category=DraftWindowLayoutCategory.DSA,
        block_span_tokens=None,
        capability_failures=("dsa has no verified P2 direct-prefix provider",),
        source_geometry_failures=(),
        drafting_compression_ratio=4,
    )
    inventory = replace(dense.inventory, groups=(group,))
    base = DraftWindowRuntimeContext(
        attention_backend_family=AttentionBackendFamily.STANDARD,
        graph_mode="NONE",
        async_scheduling=False,
        tensor_parallel_size=1,
        data_parallel_size=1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        sequence_parallel=False,
        dsa_context_parallel=False,
        flashcomm=False,
        prefix_caching=False,
    )
    policy = resolve_draft_window_policy(
        requested_window_size=128,
        method="dflash",
        target_model_type="runtime-owned",
    )

    compressed = evaluate_p7x_nongraph_capability(
        policy=policy,
        inventory=inventory,
        runtime_context=base,
        uses_dynamic_k=False,
        provider_version=2,
        execution_lane=0,
    )
    async_dsa = evaluate_p7x_nongraph_capability(
        policy=policy,
        inventory=replace(
            inventory,
            groups=(replace(group, drafting_compression_ratio=1),),
        ),
        runtime_context=replace(base, async_scheduling=True),
        uses_dynamic_k=False,
        provider_version=2,
        execution_lane=0,
    )

    assert compressed.status is P7XCapabilityStatus.SOURCE_BLOCKED
    assert "compressed" in compressed.reason
    assert async_dsa.status is P7XCapabilityStatus.REPORT1_RUNTIME_DEPENDENT
    assert "lane-owned" in async_dsa.reason


def test_forced_off_policy_cannot_reach_p7x_inventory() -> None:
    forced = resolve_draft_window_policy(
        requested_window_size=-1,
        method="mtp",
        target_model_type="qwen3",
    )
    assert forced.state is DraftWindowPolicyState.FORCED_OFF
    decision = evaluate_p7x_nongraph_capability(
        policy=forced,
        inventory=None,
        runtime_context=None,
        uses_dynamic_k=False,
        provider_version=2,
        execution_lane=0,
    )
    assert decision.status is P7XCapabilityStatus.FORCED_OFF
    assert decision.topology_fingerprint is None
