# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""P7-E eager capability and topology contracts."""

from dataclasses import replace

import pytest
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens

from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionMetadataBuilder,
)
from vllm_ascend.device.hardware_profile import AttentionBackendFamily
from vllm_ascend.worker.v2.spec_decode.draft_window import (
    DraftWindowGroupInventory,
    DraftWindowInventory,
    DraftWindowLayoutCategory,
    DraftWindowPolicyDecision,
    DraftWindowPolicyState,
    DraftWindowRuntimeContext,
    resolve_draft_window_policy,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_eager_capability import (
    P7EagerCapabilityStatus,
    P7EagerTopologyFingerprint,
    evaluate_draft_window_eager_capability,
    project_dcp_window_interval,
)


def _policy(method: str = "eagle3") -> DraftWindowPolicyDecision:
    return resolve_draft_window_policy(
        requested_window_size=128,
        method=method,
        target_model_type="qwen3",
    )


def _context(**changes: object) -> DraftWindowRuntimeContext:
    value = DraftWindowRuntimeContext(
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
    return replace(value, **changes)


def _group(
    *,
    gid: int = 0,
    layout: DraftWindowLayoutCategory = DraftWindowLayoutCategory.DENSE_GQA_DIRECT_PREFIX,
    backend: type[object] = AscendAttentionBackend,
    builder: type[object] = AscendAttentionMetadataBuilder,
    logical_block_size: int = 64,
    kernel_block_size: int = 64,
    failures: tuple[str, ...] = (),
) -> DraftWindowGroupInventory:
    return DraftWindowGroupInventory(
        global_group_id=gid,
        local_consumer_index=0,
        layer_names=(f"draft.layers.{gid}.self_attn",),
        backend_class=backend,
        builder_class=builder,
        logical_block_size=logical_block_size,
        kernel_block_size=kernel_block_size,
        block_span_tokens=(kernel_block_size if layout is DraftWindowLayoutCategory.DENSE_GQA_DIRECT_PREFIX else None),
        causal=True,
        layout_category=layout,
        capability_failures=failures,
    )


def _inventory(
    *,
    method: str = "eagle3",
    route: str | None = None,
    groups: tuple[DraftWindowGroupInventory, ...] | None = None,
    failures: tuple[str, ...] = (),
) -> DraftWindowInventory:
    return DraftWindowInventory(
        method=method,
        target_model_type="qwen3",
        runtime_route=route or method,
        window_size=128,
        groups=groups or (_group(),),
        empty_global_group_ids=(),
        capability_failures=failures,
    )


def _evaluate(
    *,
    policy: DraftWindowPolicyDecision | None = None,
    inventory: DraftWindowInventory | None = None,
    context: DraftWindowRuntimeContext | None = None,
    uses_dynamic_k: bool = False,
):
    return evaluate_draft_window_eager_capability(
        policy=policy or _policy(),
        inventory=inventory,
        runtime_context=context,
        uses_dynamic_k=uses_dynamic_k,
        provider_version=1,
        execution_lane=0,
    )


def test_config_off_and_forced_off_have_zero_topology_state() -> None:
    config_off = DraftWindowPolicyDecision(
        state=DraftWindowPolicyState.CONFIG_OFF,
        requested_window_size=None,
        effective_window_size=None,
        normalized_method="eagle3",
        target_model_type="qwen3",
        reason=None,
    )
    forced_off = resolve_draft_window_policy(
        requested_window_size=-1,
        method="mtp",
        target_model_type="qwen3",
    )

    off_decision = _evaluate(policy=config_off)
    forced_decision = _evaluate(policy=forced_off)

    assert off_decision.status is P7EagerCapabilityStatus.CONFIG_OFF
    assert forced_decision.status is P7EagerCapabilityStatus.FORCED_OFF
    assert off_decision.topology_fingerprint is None
    assert forced_decision.topology_fingerprint is None


def test_dense_single_group_eager_route_is_ready_for_npu_not_supported() -> None:
    decision = _evaluate(inventory=_inventory(), context=_context())

    assert decision.status is P7EagerCapabilityStatus.READY_FOR_NPU
    assert decision.topology_fingerprint == P7EagerTopologyFingerprint(
        normalized_method="eagle3",
        runtime_route="eagle3",
        attention_backend_family=AttentionBackendFamily.STANDARD,
        backend_impls=("vllm_ascend.attention.attention_v1.AscendAttentionBackend",),
        builder_impls=("vllm_ascend.attention.attention_v1.AscendAttentionMetadataBuilder",),
        global_group_ids=(0,),
        layouts=("dense_gqa_direct_prefix",),
        block_geometry=((0, 64, 64),),
        tensor_parallel_size=1,
        data_parallel_size=1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        sequence_parallel=False,
        dsa_context_parallel=False,
        flashcomm=False,
        graph_mode="NONE",
        async_scheduling=False,
        prefix_caching=False,
        provider_version=1,
        execution_lane=0,
    )
    decision.require_live_materialization()


def test_compatibility_attention_backend_family_fails_closed() -> None:
    """A subclass-compatible backend must not erase its hardware contract."""

    class _CompatibilityBackend(AscendAttentionBackend):
        pass

    class _CompatibilityBuilder(AscendAttentionMetadataBuilder):
        pass

    decision = _evaluate(
        inventory=_inventory(
            groups=(
                _group(
                    backend=_CompatibilityBackend,
                    builder=_CompatibilityBuilder,
                ),
            ),
        ),
        context=_context(
            attention_backend_family=AttentionBackendFamily.COMPATIBILITY,
        ),
    )

    assert decision.status is P7EagerCapabilityStatus.FAIL_CLOSED
    assert "COMPATIBILITY" in decision.reason
    with pytest.raises(NotImplementedError, match="FAIL_CLOSED.*COMPATIBILITY"):
        decision.require_live_materialization()


@pytest.mark.parametrize("graph_mode", ["FULL", "FULL_DECODE_ONLY", "PIECEWISE"])
def test_graph_routes_remain_fail_closed(graph_mode: str) -> None:
    decision = _evaluate(inventory=_inventory(), context=_context(graph_mode=graph_mode))

    assert decision.status is P7EagerCapabilityStatus.FAIL_CLOSED
    assert "Graph" in decision.reason
    with pytest.raises(NotImplementedError, match="FAIL_CLOSED.*Graph"):
        decision.require_live_materialization()


def test_async_has_only_a_static_host_contract_and_cannot_materialize_live() -> None:
    decision = _evaluate(
        inventory=_inventory(
            failures=("gid=none backend=none layout=runtime reason=Async scheduling is not verified in P2",)
        ),
        context=_context(async_scheduling=True),
    )

    assert decision.status is P7EagerCapabilityStatus.STATIC_READY
    assert decision.topology_fingerprint is not None
    assert decision.topology_fingerprint.async_scheduling is True
    with pytest.raises(NotImplementedError, match="STATIC_READY.*NPU completion"):
        decision.require_live_materialization()


@pytest.mark.parametrize(
    "change",
    [
        {"tensor_parallel_size": 2},
        {"data_parallel_size": 2},
        {"pipeline_parallel_size": 2},
        {"prefill_context_parallel_size": 2},
        {"decode_context_parallel_size": 2},
        {"sequence_parallel": True},
        {"dsa_context_parallel": True},
        {"flashcomm": True},
        {"prefix_caching": True},
    ],
)
def test_unverified_parallel_prefix_and_flashcomm_routes_fail_closed(
    change: dict[str, object],
) -> None:
    decision = _evaluate(inventory=_inventory(), context=_context(**change))

    assert decision.status is P7EagerCapabilityStatus.FAIL_CLOSED
    with pytest.raises(NotImplementedError):
        decision.require_live_materialization()


@pytest.mark.parametrize(
    "layout",
    [
        DraftWindowLayoutCategory.DSA,
        DraftWindowLayoutCategory.SFA,
        DraftWindowLayoutCategory.MLA,
        DraftWindowLayoutCategory.NATIVE_OR_RECYCLING,
    ],
)
def test_complex_backend_without_typed_visibility_provider_fails_closed(
    layout: DraftWindowLayoutCategory,
) -> None:
    decision = _evaluate(
        inventory=_inventory(groups=(_group(layout=layout),)),
        context=_context(),
    )

    assert decision.status is P7EagerCapabilityStatus.FAIL_CLOSED
    assert "typed visibility/residency provider" in decision.reason


def test_homogeneous_multi_group_is_fail_closed_but_heterogeneous_is_owner_blocked() -> None:
    homogeneous = _evaluate(
        inventory=_inventory(groups=(_group(gid=0), _group(gid=1))),
        context=_context(),
    )
    heterogeneous = _evaluate(
        inventory=_inventory(
            groups=(
                _group(gid=0),
                _group(gid=1, layout=DraftWindowLayoutCategory.DSA),
            )
        ),
        context=_context(),
    )

    assert homogeneous.status is P7EagerCapabilityStatus.FAIL_CLOSED
    assert "per-gid projection" in homogeneous.reason
    assert heterogeneous.status is P7EagerCapabilityStatus.OWNER_BLOCKED
    assert "heterogeneous" in heterogeneous.reason


def test_plain_eagle_dynamic_k_and_dflash2_have_unique_fail_closed_states() -> None:
    with pytest.raises(NotImplementedError, match="plain Eagle"):
        resolve_draft_window_policy(
            requested_window_size=128,
            method="eagle",
            target_model_type="qwen3",
        )

    dynamic = _evaluate(
        inventory=_inventory(),
        context=_context(),
        uses_dynamic_k=True,
    )
    dflash2 = _evaluate(
        policy=_policy("dflash"),
        inventory=_inventory(method="dflash", route="dflash2"),
        context=_context(),
    )

    assert dynamic.status is P7EagerCapabilityStatus.OWNER_BLOCKED
    assert dflash2.status is P7EagerCapabilityStatus.FAIL_CLOSED


def test_topology_fingerprint_lane_changes_are_explicit_and_immutable() -> None:
    decision = _evaluate(inventory=_inventory(), context=_context())
    fingerprint = decision.topology_fingerprint
    assert fingerprint is not None

    lane_one = fingerprint.for_execution_lane(1, async_scheduling=True)

    assert fingerprint.execution_lane == 0
    assert fingerprint.async_scheduling is False
    assert lane_one.execution_lane == 1
    assert lane_one.async_scheduling is True
    assert lane_one != fingerprint


@pytest.mark.parametrize(
    ("start", "end", "world_size", "rank", "interleave"),
    [
        (0, 9, 2, 0, 1),
        (1, 9, 2, 1, 1),
        (3, 19, 4, 2, 2),
        (16, 39, 3, 1, 4),
    ],
)
def test_dcp_interval_projection_is_local_end_minus_local_start(
    start: int,
    end: int,
    world_size: int,
    rank: int,
    interleave: int,
) -> None:
    projected = project_dcp_window_interval(
        window_start=start,
        window_end=end,
        dcp_world_size=world_size,
        dcp_rank=rank,
        interleave_size=interleave,
    )
    expected = get_dcp_local_seq_lens(
        __import__("torch").tensor([start, end]),
        dcp_size=world_size,
        dcp_rank=rank,
        cp_kv_cache_interleave_size=interleave,
    )

    assert projected.local_start == int(expected[0])
    assert projected.local_end == int(expected[1])
    assert projected.local_visible_length == int(expected[1] - expected[0])


def test_dcp_projection_is_not_naive_visible_length_division() -> None:
    projected = project_dcp_window_interval(
        window_start=0,
        window_end=1,
        dcp_world_size=2,
        dcp_rank=0,
        interleave_size=2,
    )

    assert projected.local_visible_length == 1
    assert projected.local_visible_length != (1 - 0) // 2
