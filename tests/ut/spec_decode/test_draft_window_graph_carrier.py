# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU contracts for the P6 per-consumer Draft Window graph carrier."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import (
    SpeculatorCudaGraphManager,
)
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import (
    DFlashCudaGraphManager,
)

import vllm_ascend.worker.v2.spec_decode.autoregressive.aclgraph as ar_aclgraph_module
import vllm_ascend.worker.v2.spec_decode.dflash.aclgraph as dflash_aclgraph_module
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
)
from vllm_ascend.compilation.acl_graph import update_full_graph_params
from vllm_ascend.worker.v2.spec_decode.autoregressive.aclgraph import (
    AutoRegressiveAclGraphManager,
)
from vllm_ascend.worker.v2.spec_decode.autoregressive.speculator import (
    AscendAutoRegressiveSpeculator,
)
from vllm_ascend.worker.v2.spec_decode.dflash.aclgraph import (
    DFlashAclGraphManager,
)
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import (
    AscendDFlashSpeculator,
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
from vllm_ascend.worker.v2.spec_decode.draft_window_graph import (
    DraftWindowGraphBackendAttribution,
    DraftWindowGraphCaptureIdentity,
    DraftWindowGraphConsumerKind,
    DraftWindowGraphRoute,
    DraftWindowGraphRouteFingerprint,
    DraftWindowGraphStage,
    DraftWindowGraphTimelineStatus,
    assert_draft_window_graph_metadata_batch,
    build_draft_window_graph_update_binding,
    consume_draft_window_graph_metadata,
    install_draft_window_graph_binding_provider,
    observe_draft_window_graph_capture,
    observe_draft_window_graph_source,
    validate_draft_window_graph_backend_update,
)
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import (
    AscendDSparkSpeculator,
)


@dataclass
class _Metadata:
    block_tables: torch.Tensor
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    seq_lens_list: list[int]
    actual_seq_lengths_q: list[int]
    slot_mapping: torch.Tensor
    positions: torch.Tensor
    query_start_loc: torch.Tensor
    nested: dict[str, Any]


class _SourceSpy:
    def __init__(self) -> None:
        self.events: list[Any] = []

    def record(self, event: Any) -> None:
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


def _descriptor(*, start: int = 0, valid: int = 8) -> DraftWindowReadDescriptor:
    return DraftWindowReadDescriptor(
        global_group_id=0,
        source_start_column=start,
        first_token_offset=0,
        required_columns=2,
        safe_extent_tokens=128,
        valid_num_blocks=valid,
        source_row_width=8,
    )


def _materialize(
    runtime: Any,
    *,
    mapping_epoch: int,
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
    return runtime.materialize(
        mapping_epoch=mapping_epoch,
        ordered_request_ids=request_ids,
        idx_mapping=idx_mapping,
        read_descriptors=tuple(_descriptor(start=start) for _ in request_ids),
        device_window_exact=torch.tensor(exact, dtype=torch.int32),
        cpu_window_exact=torch.tensor(exact, dtype=torch.int32),
        cpu_window_upper=torch.tensor(upper, dtype=torch.int32),
        num_reqs_padded=4,
        phase=phase,
        step=step,
        execution_lane=0,
    )


def _fingerprint(
    *,
    method: str = "eagle3",
    route: DraftWindowGraphRoute = DraftWindowGraphRoute.EAGLE3_FIRST_PASS,
    query_width: int = 4,
    padded_tokens: int | None = None,
) -> DraftWindowGraphRouteFingerprint:
    if padded_tokens is None:
        padded_tokens = 4 * query_width
    return DraftWindowGraphRouteFingerprint(
        method=method,
        route=route,
        graph_mode="FULL",
        backend_impl=("vllm_ascend.attention.attention_v1.AscendAttentionBackendImpl"),
        builder_impl=("vllm_ascend.attention.attention_v1.AscendAttentionMetadataBuilder"),
        global_group_id=0,
        layout=DraftWindowLayoutCategory.DENSE_GQA_DIRECT_PREFIX.value,
        capture_descriptor="tokens=8,reqs=4,mode=FULL",
        capture_bucket=(4, padded_tokens),
        query_width=query_width,
        padded_request_count=4,
        padded_token_count=padded_tokens,
        static_k=3 if method == "eagle3" else query_width - 1,
        window_size=128,
        logical_block_size=64,
        kernel_block_size=64,
        block_span_tokens=64,
        tensor_parallel_size=1,
        data_parallel_size=1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        sequence_parallel=False,
        dsa_context_parallel=False,
        flashcomm=False,
        async_scheduling=False,
        prefix_caching=False,
    )


def _target_metadata() -> dict[str, _Metadata]:
    return {
        "draft.layers.0.self_attn": _Metadata(
            block_tables=torch.full((4, 2), 99, dtype=torch.int32),
            seq_lens=torch.full((4,), 99, dtype=torch.int32),
            seq_lens_cpu=torch.full((4,), 99, dtype=torch.int32),
            seq_lens_list=[99, 99, 99, 99],
            actual_seq_lengths_q=[1, 2, 3, 4],
            slot_mapping=torch.arange(8, dtype=torch.int32),
            positions=torch.arange(8, dtype=torch.int64),
            query_start_loc=torch.arange(5, dtype=torch.int32),
            nested={
                "tensor": torch.tensor([7], dtype=torch.int32),
                "array": np.asarray([7], dtype=np.int32),
            },
        )
    }


def _metadata_builder(
    target: dict[str, _Metadata],
    *,
    alias_target: bool = False,
):
    target_layer = target["draft.layers.0.self_attn"]

    def build(read_inputs: Any) -> dict[str, _Metadata]:
        if alias_target:
            return target
        return {
            "draft.layers.0.self_attn": _Metadata(
                block_tables=read_inputs.window_block_table,
                seq_lens=read_inputs.device_window_exact,
                seq_lens_cpu=read_inputs.cpu_window_exact,
                seq_lens_list=read_inputs.cpu_window_exact.tolist(),
                actual_seq_lengths_q=[1, 2, 3, 4],
                slot_mapping=target_layer.slot_mapping.clone(),
                positions=target_layer.positions.clone(),
                query_start_loc=target_layer.query_start_loc.clone(),
                nested={
                    "tensor": target_layer.nested["tensor"].clone(),
                    "array": target_layer.nested["array"].copy(),
                },
            )
        }

    return build


def _register(
    provider: Any,
    *,
    fingerprint: DraftWindowGraphRouteFingerprint | None = None,
    consumer_ordinal: int = 0,
) -> DraftWindowGraphCaptureIdentity:
    return provider.register_capture_identity(
        fingerprint=fingerprint or _fingerprint(),
        execution_lane=0,
        consumer_ordinal=consumer_ordinal,
    )


def test_capture_identity_is_not_a_runtime_generation_namespace() -> None:
    runtime, _capability, _block_tables = _runtime_fixture()
    owner = SimpleNamespace(_draft_window_runtime_state=runtime)
    provider = install_draft_window_graph_binding_provider(owner)

    capture = _register(provider)

    assert isinstance(capture, DraftWindowGraphCaptureIdentity)
    capture_fields = {field.name for field in fields(capture)}
    assert "generation_id" not in capture_fields
    assert "mapping_epoch" not in capture_fields
    assert "ordered_request_fingerprint" not in capture_fields
    with pytest.raises(TypeError, match="capture identity"):
        provider.bind_read_carrier(  # type: ignore[arg-type]
            capture_identity=SimpleNamespace(generation_id=0, mapping_epoch=0),
            inputs=_materialize(runtime, mapping_epoch=10),
            proposal_id=10,
            metadata_builder=_metadata_builder(_target_metadata()),
            target_metadata=_target_metadata(),
        )


@pytest.mark.parametrize(
    ("field", "value", "diagnostic"),
    [
        ("capture_descriptor", "other-bucket", "capture descriptor"),
        ("capture_bucket", (8, 32), "capture bucket"),
        ("global_group_id", 1, "global gid"),
        ("backend_impl", "fake.Backend", "backend"),
        ("builder_impl", "fake.Builder", "builder"),
    ],
)
def test_capture_route_fingerprint_mismatch_fails_closed(
    field: str,
    value: Any,
    diagnostic: str,
) -> None:
    runtime, _capability, _block_tables = _runtime_fixture()
    owner = SimpleNamespace(_draft_window_runtime_state=runtime)
    provider = install_draft_window_graph_binding_provider(owner)
    capture = _register(provider)
    inputs = _materialize(runtime, mapping_epoch=10)
    target = _target_metadata()

    changes = {field: value}
    if field == "capture_bucket":
        changes.update(
            padded_request_count=value[0],
            padded_token_count=value[1],
        )
    bad = replace(capture, fingerprint=replace(capture.fingerprint, **changes))
    with pytest.raises(RuntimeError, match=diagnostic):
        provider.bind_read_carrier(
            capture_identity=bad,
            inputs=inputs,
            proposal_id=10,
            metadata_builder=_metadata_builder(target),
            target_metadata=target,
        )


def test_capture_query_width_must_match_padded_token_geometry() -> None:
    with pytest.raises(ValueError, match="query width"):
        _fingerprint(query_width=1, padded_tokens=16)


def test_provider_exists_without_explicit_carrier_fails_before_baseline_metadata_can_win() -> None:
    runtime, _capability, _block_tables = _runtime_fixture(method="dflash")
    owner = SimpleNamespace(_draft_window_runtime_state=runtime)
    install_draft_window_graph_binding_provider(owner)
    baseline_target_metadata = [_target_metadata()]

    with pytest.raises(RuntimeError, match="explicit Draft Window graph read carrier"):
        consume_draft_window_graph_metadata(
            owner,
            route=DraftWindowGraphRoute.DFLASH,
            expected_consumer_count=1,
        )
    with pytest.raises(RuntimeError, match="explicit Draft Window graph read carrier"):
        assert_draft_window_graph_metadata_batch(
            owner,
            route=DraftWindowGraphRoute.DFLASH,
            draft_attn_metadatas=baseline_target_metadata,
        )


@pytest.mark.parametrize(
    ("method", "route", "speculator_cls"),
    [
        ("dflash", DraftWindowGraphRoute.DFLASH, AscendDFlashSpeculator),
        ("dspark", DraftWindowGraphRoute.DSPARK, AscendDSparkSpeculator),
    ],
)
def test_parallel_speculator_build_entry_consumes_only_explicit_carrier_metadata(
    method: str,
    route: DraftWindowGraphRoute,
    speculator_cls: type[Any],
) -> None:
    runtime, _capability, _block_tables = _runtime_fixture(method=method)
    owner = SimpleNamespace(
        _draft_window_runtime_state=runtime,
        _speculator_name=method,
    )
    provider = install_draft_window_graph_binding_provider(owner)
    capture = _register(
        provider,
        fingerprint=_fingerprint(method=method, route=route),
    )
    target = _target_metadata()
    inputs = _materialize(runtime, mapping_epoch=10)
    carrier = provider.bind_read_carrier(
        capture_identity=capture,
        inputs=inputs,
        proposal_id=10,
        metadata_builder=_metadata_builder(target),
        target_metadata=target,
    )

    result = speculator_cls.build_draft_attn_metadatas(owner, 4, torch.zeros(4))

    assert result == [carrier.metadata]


def test_explicit_carrier_builds_draft_private_metadata_and_preserves_target() -> None:
    runtime, capability, block_tables = _runtime_fixture()
    canonical_before = block_tables.input_block_tables[0].clone()
    target = _target_metadata()
    target_root = target
    target_layer = target["draft.layers.0.self_attn"]
    target_snapshot = {
        "table": target_layer.block_tables.clone(),
        "seq": target_layer.seq_lens.clone(),
        "slot": target_layer.slot_mapping.clone(),
        "positions": target_layer.positions.clone(),
        "query": target_layer.query_start_loc.clone(),
        "nested_tensor": target_layer.nested["tensor"].clone(),
        "nested_array": target_layer.nested["array"].copy(),
    }
    owner = SimpleNamespace(_draft_window_runtime_state=runtime)
    provider = install_draft_window_graph_binding_provider(owner)
    capture = _register(provider)
    inputs = _materialize(runtime, mapping_epoch=10)

    carrier = provider.bind_read_carrier(
        capture_identity=capture,
        inputs=inputs,
        proposal_id=10,
        metadata_builder=_metadata_builder(target),
        target_metadata=target,
    )
    batch = consume_draft_window_graph_metadata(
        owner,
        route=DraftWindowGraphRoute.EAGLE3_FIRST_PASS,
        expected_consumer_count=1,
    )

    assert batch is not None and len(batch) == 1
    metadata = batch[0]["draft.layers.0.self_attn"]
    assert metadata.block_tables is carrier.read_inputs.window_block_table
    assert metadata.seq_lens is carrier.read_inputs.device_window_exact
    assert metadata.seq_lens_cpu is carrier.read_inputs.cpu_window_exact
    assert metadata.seq_lens_list == [96, 64, 0, 0]
    assert carrier.capture_identity.backing_identity.owner_id != id(runtime.workspace)
    assert carrier.capture_identity.backing_identity.window_block_table.storage_data_ptr != (
        capability.full_source._input_block_tables[0].untyped_storage().data_ptr()
    )
    assert target is target_root
    assert target["draft.layers.0.self_attn"] is target_layer
    torch.testing.assert_close(target_layer.block_tables, target_snapshot["table"])
    torch.testing.assert_close(target_layer.seq_lens, target_snapshot["seq"])
    torch.testing.assert_close(target_layer.slot_mapping, target_snapshot["slot"])
    torch.testing.assert_close(target_layer.positions, target_snapshot["positions"])
    torch.testing.assert_close(target_layer.query_start_loc, target_snapshot["query"])
    torch.testing.assert_close(target_layer.nested["tensor"], target_snapshot["nested_tensor"])
    np.testing.assert_array_equal(target_layer.nested["array"], target_snapshot["nested_array"])
    torch.testing.assert_close(block_tables.input_block_tables[0], canonical_before)


def test_carrier_rejects_nested_target_aliases() -> None:
    runtime, _capability, _block_tables = _runtime_fixture()
    owner = SimpleNamespace(_draft_window_runtime_state=runtime)
    provider = install_draft_window_graph_binding_provider(owner)
    capture = _register(provider)
    inputs = _materialize(runtime, mapping_epoch=10)
    target = _target_metadata()

    with pytest.raises(ValueError, match="aliases Target metadata"):
        provider.bind_read_carrier(
            capture_identity=capture,
            inputs=inputs,
            proposal_id=10,
            metadata_builder=_metadata_builder(target, alias_target=True),
            target_metadata=target,
        )


def test_backend_attribution_is_derived_from_actual_update_path() -> None:
    runtime, _capability, _block_tables = _runtime_fixture(method="dflash")
    owner = SimpleNamespace(_draft_window_runtime_state=runtime)
    provider = install_draft_window_graph_binding_provider(owner)
    capture = _register(
        provider,
        fingerprint=_fingerprint(
            method="dflash",
            route=DraftWindowGraphRoute.DFLASH,
            query_width=4,
        ),
    )
    target = _target_metadata()
    inputs = _materialize(runtime, mapping_epoch=10)
    provider.bind_read_carrier(
        capture_identity=capture,
        inputs=inputs,
        proposal_id=10,
        metadata_builder=_metadata_builder(target),
        target_metadata=target,
    )
    metadata = consume_draft_window_graph_metadata(
        owner,
        route=DraftWindowGraphRoute.DFLASH,
        expected_consumer_count=1,
    )
    assert metadata is not None
    binding = build_draft_window_graph_update_binding(
        owner,
        route=DraftWindowGraphRoute.DFLASH,
        draft_attn_metadatas=list(metadata),
    )

    attribution = validate_draft_window_graph_backend_update(
        binding,
        attn_backend=AscendAttentionBackend,
        impl_cls=AscendAttentionBackendImpl,
        builder_cls=AscendAttentionMetadataBuilder,
        draft_attn_metadatas=list(metadata),
        update_function_path=("vllm_ascend.compilation.acl_graph.update_full_graph_params"),
    )

    assert isinstance(attribution, DraftWindowGraphBackendAttribution)
    assert attribution.consumer_kind is DraftWindowGraphConsumerKind.TASK_REBOUND
    assert attribution.backend_impl.endswith("AscendAttentionBackendImpl")
    assert attribution.builder_impl.endswith("AscendAttentionMetadataBuilder")
    assert all(carrier.backend_attribution == attribution for carrier in binding.carriers)


def test_actual_update_entry_resolves_backend_before_attribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _capability, _block_tables = _runtime_fixture(method="dflash")
    owner = SimpleNamespace(_draft_window_runtime_state=runtime)
    provider = install_draft_window_graph_binding_provider(owner)
    capture = _register(
        provider,
        fingerprint=_fingerprint(
            method="dflash",
            route=DraftWindowGraphRoute.DFLASH,
        ),
    )
    target = _target_metadata()
    inputs = _materialize(runtime, mapping_epoch=10)
    provider.bind_read_carrier(
        capture_identity=capture,
        inputs=inputs,
        proposal_id=10,
        metadata_builder=_metadata_builder(target),
        target_metadata=target,
    )
    metadata = consume_draft_window_graph_metadata(
        owner,
        route=DraftWindowGraphRoute.DFLASH,
        expected_consumer_count=1,
    )
    assert metadata is not None
    binding = build_draft_window_graph_update_binding(
        owner,
        route=DraftWindowGraphRoute.DFLASH,
        draft_attn_metadatas=list(metadata),
    )
    backend_calls: list[Any] = []

    monkeypatch.setattr(
        AscendAttentionBackendImpl,
        "update_graph_params",
        staticmethod(lambda *args, **kwargs: backend_calls.append((args, kwargs))),
    )

    update_full_graph_params(
        AscendAttentionBackend,
        object(),
        object(),
        8,
        SimpleNamespace(
            parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        ),
        draft_attn_metadatas=list(metadata),
        draft_window_graph_update_binding=binding,
    )

    assert len(backend_calls) == 1
    assert binding.carriers[0].backend_attribution is not None
    assert binding.carriers[0].backend_attribution.consumer_kind is DraftWindowGraphConsumerKind.TASK_REBOUND


def test_fake_caller_consumer_declaration_cannot_override_backend_fact() -> None:
    consumer_fields = {field.name for field in fields(DraftWindowGraphCaptureIdentity)}
    assert "consumer_kind" not in consumer_fields
    with pytest.raises(TypeError):
        DraftWindowGraphBackendAttribution(  # type: ignore[call-arg]
            consumer_kind=DraftWindowGraphConsumerKind.DIRECT_CAPTURED,
        )


def test_replay_update_timeline_refuses_to_claim_same_or_next_generation() -> None:
    runtime, _capability, _block_tables = _runtime_fixture(method="dflash")
    owner = SimpleNamespace(_draft_window_runtime_state=runtime)
    provider = install_draft_window_graph_binding_provider(owner)
    capture = _register(
        provider,
        fingerprint=_fingerprint(
            method="dflash",
            route=DraftWindowGraphRoute.DFLASH,
        ),
    )
    target = _target_metadata()
    inputs = _materialize(runtime, mapping_epoch=10)
    provider.bind_read_carrier(
        capture_identity=capture,
        inputs=inputs,
        proposal_id=10,
        metadata_builder=_metadata_builder(target),
        target_metadata=target,
    )
    metadata = consume_draft_window_graph_metadata(
        owner,
        route=DraftWindowGraphRoute.DFLASH,
        expected_consumer_count=1,
    )
    assert metadata is not None

    observe_draft_window_graph_source(owner, DraftWindowGraphStage.REPLAY)
    observe_draft_window_graph_source(owner, DraftWindowGraphStage.CONSUMER)
    binding = build_draft_window_graph_update_binding(
        owner,
        route=DraftWindowGraphRoute.DFLASH,
        draft_attn_metadatas=list(metadata),
    )
    validate_draft_window_graph_backend_update(
        binding,
        attn_backend=AscendAttentionBackend,
        impl_cls=AscendAttentionBackendImpl,
        builder_cls=AscendAttentionMetadataBuilder,
        draft_attn_metadatas=list(metadata),
        update_function_path=("vllm_ascend.compilation.acl_graph.update_full_graph_params"),
    )

    timeline = provider.timeline
    assert timeline.status is DraftWindowGraphTimelineStatus.NEEDS_NPU_TIMELINE
    assert timeline.materialized_generation_ids == (inputs.identity.generation_id,)
    assert timeline.submitted_replay_generation_ids == (inputs.identity.generation_id,)
    assert timeline.consumer_generation_ids is None
    assert timeline.update_source_generation_ids == (inputs.identity.generation_id,)
    assert timeline.update_effective_generation_ids is None
    with pytest.raises(RuntimeError, match="NPU timeline"):
        provider.assert_update_effective_for(inputs.identity.generation_id)


def test_startup_capture_with_runtime_provider_fails_instead_of_faking_generation() -> None:
    runtime, _capability, _block_tables = _runtime_fixture()
    owner = SimpleNamespace(_draft_window_runtime_state=runtime)
    install_draft_window_graph_binding_provider(owner)

    with pytest.raises(RuntimeError, match="safe Ascend-only capture carrier"):
        observe_draft_window_graph_capture(owner)


def test_host_a_b_a_has_stable_backing_distinct_generations_and_no_stale_reuse() -> None:
    runtime, _capability, block_tables = _runtime_fixture()
    canonical_before = block_tables.input_block_tables[0].clone()
    owner = SimpleNamespace(_draft_window_runtime_state=runtime)
    provider = install_draft_window_graph_binding_provider(owner)
    capture = _register(provider)
    target = _target_metadata()

    inputs_a = _materialize(runtime, mapping_epoch=10, start=0)
    carrier_a = provider.bind_read_carrier(
        capture_identity=capture,
        inputs=inputs_a,
        proposal_id=10,
        metadata_builder=_metadata_builder(target),
        target_metadata=target,
    )
    address = carrier_a.capture_identity.backing_identity
    table_a = carrier_a.read_inputs.window_block_table.clone()
    provider.discard_bound_carriers()
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
    carrier_b = provider.bind_read_carrier(
        capture_identity=capture,
        inputs=inputs_b,
        proposal_id=11,
        metadata_builder=_metadata_builder(target),
        target_metadata=target,
    )
    table_b = carrier_b.read_inputs.window_block_table.clone()
    provider.discard_bound_carriers()
    runtime.retire_generation(inputs_b.identity)

    inputs_a2 = _materialize(runtime, mapping_epoch=12, start=0)
    carrier_a2 = provider.bind_read_carrier(
        capture_identity=capture,
        inputs=inputs_a2,
        proposal_id=12,
        metadata_builder=_metadata_builder(target),
        target_metadata=target,
    )

    assert carrier_b.capture_identity.backing_identity == address
    assert carrier_a2.capture_identity.backing_identity == address
    assert not torch.equal(table_a, table_b)
    torch.testing.assert_close(carrier_a2.read_inputs.window_block_table, table_a)
    assert inputs_a.identity.generation_id != inputs_a2.identity.generation_id
    with pytest.raises(RuntimeError, match="stale runtime binding"):
        provider.require_runtime_identity(inputs_a.identity)
    torch.testing.assert_close(block_tables.input_block_tables[0], canonical_before)


def test_eagle3_k3_carriers_keep_distinct_step_sources_and_frozen_proposal() -> None:
    runtime, _capability, _block_tables = _runtime_fixture()
    owner = SimpleNamespace(_draft_window_runtime_state=runtime)
    provider = install_draft_window_graph_binding_provider(owner)
    target = _target_metadata()
    captures = [
        _register(
            provider,
            fingerprint=_fingerprint(
                route=DraftWindowGraphRoute.EAGLE3_AR_STEP,
                query_width=1,
                padded_tokens=4,
            ),
            consumer_ordinal=step - 1,
        )
        for step in (1, 2, 3)
    ]

    previous = None
    idx_mapping = torch.tensor([1, 3], dtype=torch.int32)
    source_starts = (0, 1, 2)
    exact_values = (65, 66, 67)
    for step, capture, start, exact in zip(
        (1, 2, 3),
        captures,
        source_starts,
        exact_values,
    ):
        if previous is None:
            inputs = _materialize(
                runtime,
                mapping_epoch=10,
                idx_mapping=idx_mapping,
                phase=DraftWindowExecutionPhase.STEP,
                step=step,
                start=start,
                exact=(exact, exact, 0, 0),
                upper=(67, 67, 0, 0),
            )
        else:
            inputs = runtime.materialize_serialized_step(
                previous_identity=previous,
                ordered_request_ids=("A", "B"),
                idx_mapping=idx_mapping,
                read_descriptors=(_descriptor(start=start), _descriptor(start=start)),
                device_window_exact=torch.tensor([exact, exact, 0, 0], dtype=torch.int32),
                cpu_window_exact=torch.tensor([exact, exact, 0, 0], dtype=torch.int32),
                cpu_window_upper=torch.tensor([67, 67, 0, 0], dtype=torch.int32),
                num_reqs_padded=4,
                step=step,
                execution_lane=0,
            )
        provider.bind_read_carrier(
            capture_identity=capture,
            inputs=inputs,
            proposal_id=10,
            metadata_builder=_metadata_builder(target),
            target_metadata=target,
        )
        previous = inputs.identity
        runtime.retire_generation(inputs.identity)

    owner.input_buffers = SimpleNamespace(
        draft_seq_lens_cpus=[object(), object(), object()],
    )
    batch = AscendAutoRegressiveSpeculator.build_draft_attn_metadatas(
        owner,
        4,
        4,
        False,
    )
    assert [entry["draft.layers.0.self_attn"].seq_lens_list[0] for entry in batch] == [65, 66, 67]
    assert [carrier.runtime_identity.step for carrier in provider.submitted_carriers] == [1, 2, 3]
    assert len({carrier.runtime_identity.mapping_epoch for carrier in provider.submitted_carriers}) == 1
    assert (
        len(
            {
                carrier.capture_identity.backing_identity.window_block_table.storage_data_ptr
                for carrier in provider.submitted_carriers
            }
        )
        == 3
    )


@pytest.mark.parametrize(
    ("method", "route"),
    [
        ("dflash", DraftWindowGraphRoute.DFLASH),
        ("dspark", DraftWindowGraphRoute.DSPARK),
    ],
)
def test_parallel_graph_manager_carries_explicit_metadata_to_actual_update_seam(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    route: DraftWindowGraphRoute,
) -> None:
    runtime, _capability, _block_tables = _runtime_fixture(method=method)
    target = _target_metadata()
    spy = _SourceSpy()
    speculator = SimpleNamespace(
        _speculator_name=method,
        _draft_window_runtime_state=runtime,
        input_batch=SimpleNamespace(seq_lens_cpu_upper_bound=object()),
        attn_backends={"draft.layers.0.self_attn": AscendAttentionBackend},
        dp_size=1,
        model_state=SimpleNamespace(attn_metadata=target),
        speculative_config=object(),
    )
    provider = install_draft_window_graph_binding_provider(speculator, observer=spy)
    capture = _register(
        provider,
        fingerprint=_fingerprint(
            method=method,
            route=route,
            query_width=2,
            padded_tokens=8,
        ),
    )
    inputs = _materialize(runtime, mapping_epoch=10)
    provider.bind_read_carrier(
        capture_identity=capture,
        inputs=inputs,
        proposal_id=10,
        metadata_builder=_metadata_builder(target),
        target_metadata=target,
    )

    def _build(*_args: Any) -> list[Mapping[str, Any]]:
        metadata = consume_draft_window_graph_metadata(
            speculator,
            route=route,
            expected_consumer_count=1,
        )
        assert metadata is not None
        return list(metadata)

    speculator.build_draft_attn_metadatas = _build
    manager = DFlashAclGraphManager.__new__(DFlashAclGraphManager)
    manager.speculator = speculator
    manager.update_stream = SimpleNamespace(wait_stream=lambda _stream: None)
    manager.device = torch.device("cpu")
    manager.vllm_config = object()

    @contextmanager
    def _context(*_args: Any, **_kwargs: Any):
        yield

    def _update(
        attn_backend: Any,
        *_args: Any,
        draft_attn_metadatas: Any,
        draft_window_graph_update_binding: Any,
        **_kwargs: Any,
    ) -> None:
        validate_draft_window_graph_backend_update(
            draft_window_graph_update_binding,
            attn_backend=attn_backend,
            impl_cls=AscendAttentionBackendImpl,
            builder_cls=AscendAttentionMetadataBuilder,
            draft_attn_metadatas=draft_attn_metadatas,
            update_function_path=("vllm_ascend.compilation.acl_graph.update_full_graph_params"),
        )

    monkeypatch.setattr(dflash_aclgraph_module, "set_forward_context", _context)
    monkeypatch.setattr(dflash_aclgraph_module, "get_forward_context", lambda: object())
    monkeypatch.setattr(dflash_aclgraph_module, "update_full_graph_params", _update)
    monkeypatch.setattr(torch.npu, "current_stream", lambda: object())
    monkeypatch.setattr(DFlashCudaGraphManager, "run_fullgraph", lambda self, desc: "replayed")
    monkeypatch.setattr(
        dflash_aclgraph_module,
        "_EXTRA_CTX",
        SimpleNamespace(is_draft_model=False, is_draft_model_prefill=False),
    )

    desc = SimpleNamespace(num_tokens=8, num_reqs=4, cg_mode=CUDAGraphMode.FULL)
    assert manager.run_fullgraph(desc) == "replayed"
    assert [event.stage for event in spy.events] == [
        DraftWindowGraphStage.LIVE,
        DraftWindowGraphStage.REPLAY,
        DraftWindowGraphStage.CONSUMER,
        DraftWindowGraphStage.UPDATE,
    ]
    assert [event.consumer_kind for event in spy.events[:-1]] == [
        DraftWindowGraphConsumerKind.UNRESOLVED,
        DraftWindowGraphConsumerKind.UNRESOLVED,
        DraftWindowGraphConsumerKind.UNRESOLVED,
    ]
    assert spy.events[-1].consumer_kind is DraftWindowGraphConsumerKind.TASK_REBOUND
    assert provider.timeline.status is DraftWindowGraphTimelineStatus.NEEDS_NPU_TIMELINE


@pytest.mark.parametrize(
    ("is_first_pass", "phase", "step", "route", "num_tokens"),
    [
        (
            True,
            DraftWindowExecutionPhase.PROPOSAL,
            0,
            DraftWindowGraphRoute.EAGLE3_FIRST_PASS,
            8,
        ),
        (
            False,
            DraftWindowExecutionPhase.STEP,
            1,
            DraftWindowGraphRoute.EAGLE3_AR_STEP,
            4,
        ),
    ],
)
def test_eagle3_graph_manager_carries_first_pass_and_ar_metadata_to_update(
    monkeypatch: pytest.MonkeyPatch,
    is_first_pass: bool,
    phase: DraftWindowExecutionPhase,
    step: int,
    route: DraftWindowGraphRoute,
    num_tokens: int,
) -> None:
    runtime, _capability, _block_tables = _runtime_fixture()
    target = _target_metadata()
    spy = _SourceSpy()
    speculator = SimpleNamespace(
        _draft_window_runtime_state=runtime,
        dp_size=1,
        model_state=SimpleNamespace(attn_metadata=target),
        draft_vllm_config=object(),
        attn_backend=AscendAttentionBackend,
        speculative_config=object(),
    )
    provider = install_draft_window_graph_binding_provider(speculator, observer=spy)
    capture = _register(
        provider,
        fingerprint=_fingerprint(
            route=route,
            query_width=2 if is_first_pass else 1,
            padded_tokens=num_tokens,
        ),
    )
    inputs = _materialize(
        runtime,
        mapping_epoch=10,
        phase=phase,
        step=step,
    )
    provider.bind_read_carrier(
        capture_identity=capture,
        inputs=inputs,
        proposal_id=10,
        metadata_builder=_metadata_builder(target),
        target_metadata=target,
    )

    def _build(*_args: Any) -> list[Mapping[str, Any]]:
        metadata = consume_draft_window_graph_metadata(
            speculator,
            route=route,
            expected_consumer_count=1,
        )
        assert metadata is not None
        return list(metadata)

    speculator.build_draft_attn_metadatas = _build
    manager = AutoRegressiveAclGraphManager.__new__(AutoRegressiveAclGraphManager)
    manager.speculator = speculator
    manager.is_draft_model_prefill = is_first_pass
    manager.update_stream = SimpleNamespace(wait_stream=lambda _stream: None)

    @contextmanager
    def _context(*_args: Any, **_kwargs: Any):
        yield

    def _update(
        attn_backend: Any,
        *_args: Any,
        draft_attn_metadatas: Any,
        draft_window_graph_update_binding: Any,
        **_kwargs: Any,
    ) -> None:
        validate_draft_window_graph_backend_update(
            draft_window_graph_update_binding,
            attn_backend=attn_backend,
            impl_cls=AscendAttentionBackendImpl,
            builder_cls=AscendAttentionMetadataBuilder,
            draft_attn_metadatas=draft_attn_metadatas,
            update_function_path=("vllm_ascend.compilation.acl_graph.update_full_graph_params"),
        )

    monkeypatch.setattr(ar_aclgraph_module, "set_current_vllm_config", _context)
    monkeypatch.setattr(ar_aclgraph_module, "set_forward_context", _context)
    monkeypatch.setattr(ar_aclgraph_module, "get_forward_context", lambda: object())
    monkeypatch.setattr(ar_aclgraph_module, "update_full_graph_params", _update)
    monkeypatch.setattr(torch.npu, "current_stream", lambda: object())
    monkeypatch.setattr(SpeculatorCudaGraphManager, "run_fullgraph", lambda self, desc: "replayed")
    monkeypatch.setattr(
        ar_aclgraph_module,
        "_EXTRA_CTX",
        SimpleNamespace(is_draft_model=False, is_draft_model_prefill=False),
    )

    desc = SimpleNamespace(num_tokens=num_tokens, num_reqs=4, cg_mode=CUDAGraphMode.FULL)
    assert manager.run_fullgraph(desc) == "replayed"
    assert [event.stage for event in spy.events] == [
        DraftWindowGraphStage.LIVE,
        DraftWindowGraphStage.REPLAY,
        DraftWindowGraphStage.CONSUMER,
        DraftWindowGraphStage.UPDATE,
    ]
