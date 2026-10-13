# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU-only contracts for the P5 Draft Window graph binding foundation."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import (
    SpeculatorCudaGraphManager,
)
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager

import vllm_ascend.worker.v2.spec_decode.autoregressive.aclgraph as ar_aclgraph_module
import vllm_ascend.worker.v2.spec_decode.dflash.aclgraph as dflash_aclgraph_module
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionMetadataBuilder,
)
from vllm_ascend.worker.v2.spec_decode.autoregressive.aclgraph import (
    AutoRegressiveAclGraphManager,
)
from vllm_ascend.worker.v2.spec_decode.dflash.aclgraph import (
    DFlashAclGraphManager,
)
from vllm_ascend.worker.v2.spec_decode.draft_window import (
    DraftMetadataInputs,
    DraftWindowExecutionPhase,
    DraftWindowGroupInventory,
    DraftWindowInventory,
    DraftWindowLayoutCategory,
    DraftWindowReadDescriptor,
    DraftWindowSourceCapability,
    SingleGroupDirectPrefixProvider,
    build_draft_window_runtime_state,
    build_full_draft_kv_source,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_eager import (
    DraftWindowEagerMixin,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_eagle3 import (
    Eagle3DraftWindowMixin,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_graph import (
    DraftWindowGraphBindingProvider,
    DraftWindowGraphConsumerKind,
    DraftWindowGraphMetadataOwner,
    DraftWindowGraphRoute,
    DraftWindowGraphSourceEvent,
    DraftWindowGraphStage,
    install_draft_window_graph_binding_provider,
)


class _SourceSpy:
    def __init__(self) -> None:
        self.events: list[DraftWindowGraphSourceEvent] = []

    def record(self, event: DraftWindowGraphSourceEvent) -> None:
        self.events.append(event)


def _runtime_fixture(
    *,
    method: str = "eagle3",
    max_num_reqs: int = 8,
    row_width: int = 8,
) -> tuple[Any, Any, SimpleNamespace]:
    canonical_counts = torch.full((1, max_num_reqs), row_width, dtype=torch.int32)
    canonical_table = torch.arange(
        1,
        max_num_reqs * row_width + 1,
        dtype=torch.int32,
    ).reshape(max_num_reqs, row_width)
    block_tables = SimpleNamespace(
        block_sizes=[64],
        kernel_block_sizes=[64],
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
        logical_block_size=64,
        kernel_block_size=64,
        block_span_tokens=64,
        causal=True,
        layout_category=DraftWindowLayoutCategory.DENSE_GQA_DIRECT_PREFIX,
        capability_failures=(),
    )
    inventory = DraftWindowInventory(
        method=method,
        target_model_type="qwen3",
        runtime_route=method,
        window_size=128,
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
            legacy_block_size=64,
            expected_future_offset=3 if method == "eagle3" else 0,
        ),
    )
    return build_draft_window_runtime_state(capability), capability, block_tables


def _descriptor(
    *,
    start: int = 0,
    required: int = 2,
    valid: int = 8,
) -> DraftWindowReadDescriptor:
    return DraftWindowReadDescriptor(
        global_group_id=0,
        source_start_column=start,
        first_token_offset=0,
        required_columns=required,
        safe_extent_tokens=required * 64,
        valid_num_blocks=valid,
        source_row_width=8,
    )


def _materialize(
    runtime: Any,
    *,
    mapping_epoch: int = 10,
    request_ids: tuple[str, ...] = ("A", "B"),
    idx_mapping: torch.Tensor | None = None,
    phase: DraftWindowExecutionPhase = DraftWindowExecutionPhase.PROPOSAL,
    step: int = 0,
    start: int = 0,
    exact: tuple[int, ...] = (96, 64, 0, 0),
    upper: tuple[int, ...] = (128, 64, 0, 0),
) -> DraftMetadataInputs:
    if idx_mapping is None:
        idx_mapping = torch.tensor([1, 3], dtype=torch.int32)
    descriptors = tuple(_descriptor(start=start) for _ in request_ids)
    return runtime.materialize(
        mapping_epoch=mapping_epoch,
        ordered_request_ids=request_ids,
        idx_mapping=idx_mapping,
        read_descriptors=descriptors,
        device_window_exact=torch.tensor(exact, dtype=torch.int32),
        cpu_window_exact=torch.tensor(exact, dtype=torch.int32),
        cpu_window_upper=torch.tensor(upper, dtype=torch.int32),
        num_reqs_padded=4,
        phase=phase,
        step=step,
        execution_lane=0,
    )


def _bind(
    runtime: Any,
    inputs: DraftMetadataInputs,
    *,
    route: DraftWindowGraphRoute,
    spy: _SourceSpy | None = None,
) -> tuple[DraftWindowGraphBindingProvider, Any]:
    provider = DraftWindowGraphBindingProvider(runtime_state=runtime, observer=spy)
    descriptor = provider.bind(
        inputs,
        proposal_id=inputs.identity.mapping_epoch,
        route=route,
        consumer_kind=DraftWindowGraphConsumerKind.TASK_REBOUND,
        metadata_owner=DraftWindowGraphMetadataOwner.DRAFT_METADATA_STORE,
    )
    return provider, descriptor


@pytest.mark.parametrize(
    ("method", "phase", "step", "route"),
    [
        ("eagle3", DraftWindowExecutionPhase.PROPOSAL, 0, DraftWindowGraphRoute.EAGLE3_FIRST_PASS),
        ("eagle3", DraftWindowExecutionPhase.STEP, 1, DraftWindowGraphRoute.EAGLE3_AR_STEP),
        ("dflash", DraftWindowExecutionPhase.PROPOSAL, 0, DraftWindowGraphRoute.DFLASH),
        ("dspark", DraftWindowExecutionPhase.PROPOSAL, 0, DraftWindowGraphRoute.DSPARK),
    ],
)
def test_binding_routes_are_explicit_and_use_runtime_owned_backing(
    method: str,
    phase: DraftWindowExecutionPhase,
    step: int,
    route: DraftWindowGraphRoute,
) -> None:
    runtime, capability, block_tables = _runtime_fixture(method=method)
    inputs = _materialize(runtime, phase=phase, step=step)

    provider, descriptor = _bind(runtime, inputs, route=route)

    assert descriptor.identity.method == method
    assert descriptor.identity.route is route
    assert descriptor.identity.global_group_id == 0
    assert descriptor.identity.proposal_id == inputs.identity.mapping_epoch
    assert descriptor.identity.generation_id == inputs.identity.generation_id
    assert descriptor.identity.phase is phase
    assert descriptor.identity.step == step
    assert descriptor.identity.execution_lane == 0
    assert descriptor.window_block_table is runtime.workspace.window_block_table
    assert descriptor.device_window_exact is runtime.workspace.device_window_exact
    assert descriptor.cpu_window_exact is runtime.workspace.cpu_window_exact
    assert descriptor.cpu_window_upper is runtime.workspace.cpu_window_upper
    assert descriptor.valid_num_blocks is runtime.workspace.valid_num_blocks
    assert descriptor.backing_identity == provider.current_descriptor.backing_identity
    assert descriptor.backing_identity.owner_id == id(runtime.workspace)
    assert descriptor.window_block_table.untyped_storage().data_ptr() != (
        capability.full_source._input_block_tables[0].untyped_storage().data_ptr()
    )
    assert descriptor.valid_num_blocks.untyped_storage().data_ptr() != (
        block_tables.num_blocks.gpu.untyped_storage().data_ptr()
    )


def test_binding_is_immutable_and_has_no_target_or_write_side_fields() -> None:
    runtime, _capability, _block_tables = _runtime_fixture()
    inputs = _materialize(runtime)
    _provider, descriptor = _bind(
        runtime,
        inputs,
        route=DraftWindowGraphRoute.EAGLE3_FIRST_PASS,
    )

    with pytest.raises(FrozenInstanceError):
        descriptor.identity.step = 1  # type: ignore[misc]
    field_names = set(descriptor.__dataclass_fields__)
    assert "positions" not in field_names
    assert "slot_mapping" not in field_names
    assert "target_metadata" not in field_names
    assert "canonical_full_source" not in field_names


@pytest.mark.parametrize(
    ("mutate", "diagnostic"),
    [
        (lambda e: replace(e, generation_id=e.generation_id + 1), "generation"),
        (lambda e: replace(e, mapping_epoch=e.mapping_epoch + 1), "mapping_epoch"),
        (
            lambda e: replace(e, ordered_request_fingerprint=tuple(reversed(e.ordered_request_fingerprint))),
            "request fingerprint",
        ),
        (lambda e: replace(e, global_group_id=e.global_group_id + 1), "global gid"),
        (lambda e: replace(e, route=DraftWindowGraphRoute.EAGLE3_AR_STEP), "route"),
        (lambda e: replace(e, phase=DraftWindowExecutionPhase.STEP), "phase"),
        (lambda e: replace(e, step=e.step + 1), "step"),
        (lambda e: replace(e, execution_lane=e.execution_lane + 1), "lane"),
    ],
)
def test_expected_identity_mismatch_fails_closed(mutate: Any, diagnostic: str) -> None:
    runtime, _capability, _block_tables = _runtime_fixture()
    inputs = _materialize(runtime)
    provider, descriptor = _bind(
        runtime,
        inputs,
        route=DraftWindowGraphRoute.EAGLE3_FIRST_PASS,
    )
    bad = replace(
        descriptor.expectation,
        identity=mutate(descriptor.expectation.identity),
    )

    with pytest.raises(RuntimeError, match=diagnostic):
        provider.observe(DraftWindowGraphStage.CONSUMER, expected=bad)


@pytest.mark.parametrize(
    ("field", "value", "diagnostic"),
    [
        (
            "consumer_kind",
            DraftWindowGraphConsumerKind.DIRECT_CAPTURED,
            "consumer",
        ),
        (
            "metadata_owner",
            DraftWindowGraphMetadataOwner.ATTENTION_BACKEND,
            "metadata owner",
        ),
    ],
)
def test_expected_transport_mismatch_fails_closed(field: str, value: Any, diagnostic: str) -> None:
    runtime, _capability, _block_tables = _runtime_fixture()
    inputs = _materialize(runtime)
    provider, descriptor = _bind(
        runtime,
        inputs,
        route=DraftWindowGraphRoute.EAGLE3_FIRST_PASS,
    )
    bad = replace(descriptor.expectation, **{field: value})

    with pytest.raises(RuntimeError, match=diagnostic):
        provider.observe(DraftWindowGraphStage.UPDATE, expected=bad)


def test_spy_distinguishes_live_capture_replay_update_and_consumer() -> None:
    runtime, _capability, _block_tables = _runtime_fixture()
    inputs = _materialize(runtime)
    spy = _SourceSpy()
    provider, descriptor = _bind(
        runtime,
        inputs,
        route=DraftWindowGraphRoute.EAGLE3_FIRST_PASS,
        spy=spy,
    )

    for stage in DraftWindowGraphStage:
        provider.observe(stage, expected=descriptor.expectation)

    assert [event.stage for event in spy.events] == list(DraftWindowGraphStage)
    assert [event.sequence for event in spy.events] == list(range(len(DraftWindowGraphStage)))
    assert all(event.identity == descriptor.identity for event in spy.events)
    assert all(event.backing_identity == descriptor.backing_identity for event in spy.events)
    assert all(event.table_fingerprint == descriptor.backing_identity.window_block_table for event in spy.events)
    assert all(
        event.length_fingerprint
        == (
            descriptor.backing_identity.device_window_exact,
            descriptor.backing_identity.cpu_window_exact,
            descriptor.backing_identity.cpu_window_upper,
        )
        for event in spy.events
    )


def test_default_provider_records_no_hidden_global_trace() -> None:
    runtime, _capability, _block_tables = _runtime_fixture()
    inputs = _materialize(runtime)
    provider, descriptor = _bind(
        runtime,
        inputs,
        route=DraftWindowGraphRoute.EAGLE3_FIRST_PASS,
    )

    event = provider.observe(
        DraftWindowGraphStage.LIVE,
        expected=descriptor.expectation,
    )

    assert event.stage is DraftWindowGraphStage.LIVE
    assert provider.observer is None


@pytest.mark.parametrize(
    ("method", "route", "mixin"),
    [
        ("dflash", DraftWindowGraphRoute.DFLASH, DraftWindowEagerMixin),
        (
            "eagle3",
            DraftWindowGraphRoute.EAGLE3_FIRST_PASS,
            Eagle3DraftWindowMixin,
        ),
    ],
)
def test_live_validation_boundary_emits_live_source_event(
    method: str,
    route: DraftWindowGraphRoute,
    mixin: type[Any],
) -> None:
    runtime, _capability, block_tables = _runtime_fixture(method=method)
    inputs = _materialize(runtime)
    spy = _SourceSpy()
    owner = SimpleNamespace(
        _draft_window_runtime_state=runtime,
        block_tables=block_tables,
    )
    provider = install_draft_window_graph_binding_provider(owner, observer=spy)
    descriptor = provider.bind(
        inputs,
        proposal_id=10,
        route=route,
        consumer_kind=DraftWindowGraphConsumerKind.TASK_REBOUND,
        metadata_owner=DraftWindowGraphMetadataOwner.DRAFT_METADATA_STORE,
    )

    if mixin is DraftWindowEagerMixin:
        read_inputs = mixin._validated_draft_attention_read_inputs(
            owner,
            inputs,
            inputs.identity,
        )
    else:
        read_inputs = mixin._validated_eagle3_read_inputs(
            owner,
            inputs,
            inputs.identity,
        )

    assert read_inputs.block_tables[0].untyped_storage().data_ptr() == (
        runtime.workspace.window_block_table.untyped_storage().data_ptr()
    )
    assert [event.stage for event in spy.events] == [DraftWindowGraphStage.LIVE]
    assert spy.events[0].identity == descriptor.identity


def test_backing_address_is_stable_while_contents_and_generation_change() -> None:
    runtime, capability, block_tables = _runtime_fixture()
    canonical_before = block_tables.input_block_tables[0].clone()
    inputs_a = _materialize(runtime, mapping_epoch=10, start=0)
    provider, descriptor_a = _bind(
        runtime,
        inputs_a,
        route=DraftWindowGraphRoute.EAGLE3_FIRST_PASS,
    )
    table_a = descriptor_a.window_block_table.clone()
    provider.observe(DraftWindowGraphStage.CAPTURE, expected=descriptor_a.expectation)
    runtime.retire_generation(inputs_a.identity)

    inputs_b = _materialize(
        runtime,
        mapping_epoch=11,
        request_ids=("C", "A"),
        idx_mapping=torch.tensor([4, 1], dtype=torch.int32),
        start=1,
        exact=(32, 48, 0, 0),
        upper=(64, 64, 0, 0),
    )
    descriptor_b = provider.bind(
        inputs_b,
        proposal_id=11,
        route=DraftWindowGraphRoute.EAGLE3_FIRST_PASS,
        consumer_kind=DraftWindowGraphConsumerKind.TASK_REBOUND,
        metadata_owner=DraftWindowGraphMetadataOwner.DRAFT_METADATA_STORE,
    )

    assert descriptor_b.backing_identity == descriptor_a.backing_identity
    assert descriptor_b.identity.generation_id != descriptor_a.identity.generation_id
    assert descriptor_b.identity.mapping_epoch == 11
    assert not torch.equal(descriptor_b.window_block_table, table_a)
    torch.testing.assert_close(block_tables.input_block_tables[0], canonical_before)
    assert descriptor_b.backing_identity.window_block_table.storage_data_ptr != (
        capability.full_source._input_block_tables[0].untyped_storage().data_ptr()
    )
    with pytest.raises(RuntimeError, match="binding expectation"):
        provider.observe(DraftWindowGraphStage.REPLAY, expected=descriptor_a.expectation)
    provider.observe(DraftWindowGraphStage.REPLAY, expected=descriptor_b.expectation)


def test_observe_rejects_canonical_full_source_table_rebinding() -> None:
    runtime, _capability, block_tables = _runtime_fixture()
    inputs = _materialize(runtime)
    provider, descriptor = _bind(
        runtime,
        inputs,
        route=DraftWindowGraphRoute.EAGLE3_FIRST_PASS,
    )
    block_tables.input_block_tables[0] = torch.zeros_like(block_tables.input_block_tables[0])

    with pytest.raises(
        RuntimeError,
        match="canonical input block table 0 was replaced",
    ):
        provider.observe(
            DraftWindowGraphStage.REPLAY,
            expected=descriptor.expectation,
        )


def test_bind_rejects_canonical_full_source_table_rebinding() -> None:
    runtime, _capability, block_tables = _runtime_fixture()
    inputs = _materialize(runtime)
    block_tables.input_block_tables[0] = torch.zeros_like(block_tables.input_block_tables[0])
    provider = DraftWindowGraphBindingProvider(runtime_state=runtime)

    with pytest.raises(
        RuntimeError,
        match="canonical input block table 0 was replaced",
    ):
        provider.bind(
            inputs,
            proposal_id=inputs.identity.mapping_epoch,
            route=DraftWindowGraphRoute.EAGLE3_FIRST_PASS,
            consumer_kind=DraftWindowGraphConsumerKind.TASK_REBOUND,
            metadata_owner=DraftWindowGraphMetadataOwner.DRAFT_METADATA_STORE,
        )


def test_active_generation_cannot_be_rebound_or_replaced() -> None:
    runtime, _capability, _block_tables = _runtime_fixture()
    inputs = _materialize(runtime)
    provider, _descriptor = _bind(
        runtime,
        inputs,
        route=DraftWindowGraphRoute.EAGLE3_FIRST_PASS,
    )

    with pytest.raises(RuntimeError, match="active graph binding"):
        provider.bind(
            inputs,
            proposal_id=10,
            route=DraftWindowGraphRoute.EAGLE3_FIRST_PASS,
            consumer_kind=DraftWindowGraphConsumerKind.TASK_REBOUND,
            metadata_owner=DraftWindowGraphMetadataOwner.DRAFT_METADATA_STORE,
        )


def test_retired_or_stale_generation_cannot_be_observed() -> None:
    runtime, _capability, _block_tables = _runtime_fixture()
    inputs = _materialize(runtime)
    provider, descriptor = _bind(
        runtime,
        inputs,
        route=DraftWindowGraphRoute.EAGLE3_FIRST_PASS,
    )
    runtime.retire_generation(inputs.identity)

    with pytest.raises(RuntimeError, match="retired"):
        provider.observe(DraftWindowGraphStage.CONSUMER, expected=descriptor.expectation)


@pytest.mark.parametrize(
    ("method", "phase", "step", "route"),
    [
        ("dflash", DraftWindowExecutionPhase.PROPOSAL, 0, DraftWindowGraphRoute.DSPARK),
        ("dspark", DraftWindowExecutionPhase.PROPOSAL, 0, DraftWindowGraphRoute.DFLASH),
        ("eagle3", DraftWindowExecutionPhase.PROPOSAL, 0, DraftWindowGraphRoute.EAGLE3_AR_STEP),
        ("eagle3", DraftWindowExecutionPhase.STEP, 1, DraftWindowGraphRoute.EAGLE3_FIRST_PASS),
    ],
)
def test_route_phase_method_mismatch_is_rejected_at_bind(
    method: str,
    phase: DraftWindowExecutionPhase,
    step: int,
    route: DraftWindowGraphRoute,
) -> None:
    runtime, _capability, _block_tables = _runtime_fixture(method=method)
    inputs = _materialize(runtime, phase=phase, step=step)
    provider = DraftWindowGraphBindingProvider(runtime_state=runtime)

    with pytest.raises((ValueError, RuntimeError), match="route"):
        provider.bind(
            inputs,
            proposal_id=inputs.identity.mapping_epoch,
            route=route,
            consumer_kind=DraftWindowGraphConsumerKind.TASK_REBOUND,
            metadata_owner=DraftWindowGraphMetadataOwner.DRAFT_METADATA_STORE,
        )


def test_install_is_explicit_instance_owned_and_off_owner_allocates_nothing() -> None:
    owner = SimpleNamespace()
    with pytest.raises(RuntimeError, match="P3 runtime"):
        install_draft_window_graph_binding_provider(owner)
    assert not hasattr(owner, "_draft_window_graph_binding_provider")

    runtime, _capability, _block_tables = _runtime_fixture(method="dflash")
    eligible_owner = SimpleNamespace(_draft_window_runtime_state=runtime)
    provider = install_draft_window_graph_binding_provider(eligible_owner)
    assert eligible_owner._draft_window_graph_binding_provider is provider
    with pytest.raises(RuntimeError, match="already installed"):
        install_draft_window_graph_binding_provider(eligible_owner)


def test_dflash_manager_capture_rejects_runtime_generation_as_startup_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _capability, _block_tables = _runtime_fixture(method="dflash")
    inputs = _materialize(runtime)
    spy = _SourceSpy()
    speculator = SimpleNamespace(_draft_window_runtime_state=runtime)
    provider = install_draft_window_graph_binding_provider(speculator, observer=spy)
    provider.bind(
        inputs,
        proposal_id=10,
        route=DraftWindowGraphRoute.DFLASH,
        consumer_kind=DraftWindowGraphConsumerKind.TASK_REBOUND,
        metadata_owner=DraftWindowGraphMetadataOwner.DRAFT_METADATA_STORE,
    )
    manager = DFlashAclGraphManager.__new__(DFlashAclGraphManager)
    manager.speculator = speculator

    @contextmanager
    def _no_op_context(*_args: Any, **_kwargs: Any):
        yield

    monkeypatch.setattr(dflash_aclgraph_module, "communicator_switch", _no_op_context)
    monkeypatch.setattr(dflash_aclgraph_module, "model_capture_wrapper", _no_op_context)
    monkeypatch.setattr(DFlashCudaGraphManager, "capture", lambda self, *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="safe Ascend-only capture carrier"):
        manager.capture(None, None, None, None, None, 1)
    assert spy.events == []


@pytest.mark.parametrize(
    ("is_first_pass", "phase", "step", "route"),
    [
        (
            True,
            DraftWindowExecutionPhase.PROPOSAL,
            0,
            DraftWindowGraphRoute.EAGLE3_FIRST_PASS,
        ),
        (
            False,
            DraftWindowExecutionPhase.STEP,
            1,
            DraftWindowGraphRoute.EAGLE3_AR_STEP,
        ),
    ],
)
def test_eagle3_manager_capture_rejects_runtime_identity_for_both_routes(
    monkeypatch: pytest.MonkeyPatch,
    is_first_pass: bool,
    phase: DraftWindowExecutionPhase,
    step: int,
    route: DraftWindowGraphRoute,
) -> None:
    runtime, _capability, _block_tables = _runtime_fixture(method="eagle3")
    inputs = _materialize(runtime, phase=phase, step=step)
    spy = _SourceSpy()
    speculator = SimpleNamespace(_draft_window_runtime_state=runtime)
    provider = install_draft_window_graph_binding_provider(speculator, observer=spy)
    provider.bind(
        inputs,
        proposal_id=10,
        route=route,
        consumer_kind=DraftWindowGraphConsumerKind.TASK_REBOUND,
        metadata_owner=DraftWindowGraphMetadataOwner.DRAFT_METADATA_STORE,
    )
    manager = AutoRegressiveAclGraphManager.__new__(AutoRegressiveAclGraphManager)
    manager.speculator = speculator
    manager.is_draft_model_prefill = is_first_pass

    @contextmanager
    def _no_op_context(*_args: Any, **_kwargs: Any):
        yield

    monkeypatch.setattr(ar_aclgraph_module, "communicator_switch", _no_op_context)
    monkeypatch.setattr(ar_aclgraph_module, "model_capture_wrapper", _no_op_context)
    monkeypatch.setattr(SpeculatorCudaGraphManager, "capture", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(CudaGraphManager, "capture", lambda self, *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="safe Ascend-only capture carrier"):
        manager.capture(None, None, None, None, None, None)
    assert spy.events == []


def test_graph_manager_without_provider_is_an_exact_observation_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    speculator = SimpleNamespace()
    manager = DFlashAclGraphManager.__new__(DFlashAclGraphManager)
    manager.speculator = speculator

    @contextmanager
    def _no_op_context(*_args: Any, **_kwargs: Any):
        yield

    monkeypatch.setattr(dflash_aclgraph_module, "communicator_switch", _no_op_context)
    monkeypatch.setattr(dflash_aclgraph_module, "model_capture_wrapper", _no_op_context)
    monkeypatch.setattr(DFlashCudaGraphManager, "capture", lambda self, *args, **kwargs: None)

    manager.capture(None, None, None, None, None, 1)

    assert not hasattr(speculator, "_draft_window_graph_binding_provider")


def test_dflash_manager_rejects_p5_observation_without_p6_carrier_before_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _capability, _block_tables = _runtime_fixture(method="dspark")
    inputs = _materialize(runtime)
    spy = _SourceSpy()
    speculator = SimpleNamespace(
        _speculator_name="DSpark",
        _draft_window_runtime_state=runtime,
        input_batch=SimpleNamespace(seq_lens_cpu_upper_bound=object()),
        build_draft_attn_metadatas=lambda *_args: [{"draft.0": object()}],
        attn_backends={"draft.0": object()},
        dp_size=1,
        model_state=SimpleNamespace(attn_metadata={}),
        speculative_config=object(),
    )
    provider = install_draft_window_graph_binding_provider(speculator, observer=spy)
    provider.bind(
        inputs,
        proposal_id=10,
        route=DraftWindowGraphRoute.DSPARK,
        consumer_kind=DraftWindowGraphConsumerKind.TASK_REBOUND,
        metadata_owner=DraftWindowGraphMetadataOwner.DRAFT_METADATA_STORE,
    )
    manager = DFlashAclGraphManager.__new__(DFlashAclGraphManager)
    manager.speculator = speculator
    manager.update_stream = SimpleNamespace(wait_stream=lambda _stream: None)
    manager.device = torch.device("cpu")
    manager.vllm_config = object()

    @contextmanager
    def _forward_context(*_args: Any, **_kwargs: Any):
        yield

    monkeypatch.setattr(dflash_aclgraph_module, "set_forward_context", _forward_context)
    monkeypatch.setattr(dflash_aclgraph_module, "get_forward_context", lambda: object())
    monkeypatch.setattr(dflash_aclgraph_module, "update_full_graph_params", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(torch.npu, "current_stream", lambda: object())
    replayed: list[Any] = []
    monkeypatch.setattr(
        DFlashCudaGraphManager,
        "run_fullgraph",
        lambda self, desc: replayed.append(desc),
    )
    monkeypatch.setattr(
        dflash_aclgraph_module,
        "_EXTRA_CTX",
        SimpleNamespace(is_draft_model=False, is_draft_model_prefill=False),
    )

    desc = SimpleNamespace(num_tokens=8, num_reqs=1, cg_mode=CUDAGraphMode.FULL)
    with pytest.raises(RuntimeError, match="explicit Draft Window graph read carrier"):
        manager.run_fullgraph(desc)
    assert replayed == []
    assert spy.events == []


@pytest.mark.parametrize(
    ("is_first_pass", "phase", "step", "route"),
    [
        (
            True,
            DraftWindowExecutionPhase.PROPOSAL,
            0,
            DraftWindowGraphRoute.EAGLE3_FIRST_PASS,
        ),
        (
            False,
            DraftWindowExecutionPhase.STEP,
            1,
            DraftWindowGraphRoute.EAGLE3_AR_STEP,
        ),
    ],
)
def test_eagle3_manager_rejects_p5_observation_without_p6_carrier_before_replay(
    monkeypatch: pytest.MonkeyPatch,
    is_first_pass: bool,
    phase: DraftWindowExecutionPhase,
    step: int,
    route: DraftWindowGraphRoute,
) -> None:
    runtime, _capability, _block_tables = _runtime_fixture(method="eagle3")
    inputs = _materialize(runtime, phase=phase, step=step)
    spy = _SourceSpy()
    speculator = SimpleNamespace(
        _draft_window_runtime_state=runtime,
        build_draft_attn_metadatas=lambda *_args: [{"draft.0": object()}],
        dp_size=1,
        model_state=SimpleNamespace(attn_metadata={}),
        draft_vllm_config=object(),
        attn_backend=object(),
        speculative_config=object(),
    )
    provider = install_draft_window_graph_binding_provider(speculator, observer=spy)
    provider.bind(
        inputs,
        proposal_id=10,
        route=route,
        consumer_kind=DraftWindowGraphConsumerKind.TASK_REBOUND,
        metadata_owner=DraftWindowGraphMetadataOwner.DRAFT_METADATA_STORE,
    )
    manager = AutoRegressiveAclGraphManager.__new__(AutoRegressiveAclGraphManager)
    manager.speculator = speculator
    manager.is_draft_model_prefill = is_first_pass
    manager.update_stream = SimpleNamespace(wait_stream=lambda _stream: None)

    @contextmanager
    def _no_op_context(*_args: Any, **_kwargs: Any):
        yield

    monkeypatch.setattr(ar_aclgraph_module, "set_current_vllm_config", _no_op_context)
    monkeypatch.setattr(ar_aclgraph_module, "set_forward_context", _no_op_context)
    monkeypatch.setattr(ar_aclgraph_module, "get_forward_context", lambda: object())
    monkeypatch.setattr(ar_aclgraph_module, "update_full_graph_params", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(torch.npu, "current_stream", lambda: object())
    replayed: list[Any] = []
    monkeypatch.setattr(
        SpeculatorCudaGraphManager,
        "run_fullgraph",
        lambda self, desc: replayed.append(desc),
    )
    monkeypatch.setattr(
        ar_aclgraph_module,
        "_EXTRA_CTX",
        SimpleNamespace(is_draft_model=False, is_draft_model_prefill=False),
    )

    desc = SimpleNamespace(num_tokens=8, num_reqs=1, cg_mode=CUDAGraphMode.FULL)
    with pytest.raises(RuntimeError, match="explicit Draft Window graph read carrier"):
        manager.run_fullgraph(desc)
    assert replayed == []
    assert spy.events == []


def test_graph_window_user_gate_remains_fail_closed() -> None:
    source = __import__(
        "vllm_ascend.worker.v2.spec_decode.draft_window",
        fromlist=["unused"],
    )._runtime_capability_failures(
        SimpleNamespace(
            graph_mode="FULL",
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
    )
    assert source == ["gid=none backend=none layout=runtime reason=Graph mode FULL is not verified in P2"]
