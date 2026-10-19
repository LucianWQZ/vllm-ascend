# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU contracts for the internal Draft Window graph characterization route."""

from __future__ import annotations

import copy
import json
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import (
    DFlashCudaGraphManager,
)

import vllm_ascend.attention.attention_v1 as attention_v1_module
import vllm_ascend.worker.v2.spec_decode.dflash.aclgraph as dflash_aclgraph_module
import vllm_ascend.worker.v2.spec_decode.dflash.capture as dflash_capture_module
import vllm_ascend.worker.v2.spec_decode.dflash.speculator as dflash_speculator_module
import vllm_ascend.worker.v2.spec_decode.draft_window as draft_window_module
from tests.ut.spec_decode.test_draft_window_graph_carrier import (
    _call_capture_helper,
    _materialize,
    _Metadata,
    _runtime_fixture,
    _target_metadata,
)
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
)
from vllm_ascend.device.hardware_profile import AttentionBackendFamily
from vllm_ascend.worker.v2.spec_decode.dflash.aclgraph import (
    DFlashAclGraphManager,
)
from vllm_ascend.worker.v2.spec_decode.dflash.capture import (
    draft_window_dflash_capture_scope,
    format_dflash_capture_descriptor,
)
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import (
    AscendDFlashSpeculator,
)
from vllm_ascend.worker.v2.spec_decode.draft_window import (
    DraftWindowRuntimeContext,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_graph import (
    DraftWindowGraphRoute,
    build_draft_window_graph_update_binding,
    consume_draft_window_graph_metadata,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_graph_characterization import (
    DraftWindowGraphCaptureOperandContext,
    DraftWindowGraphCharacterizationStage,
    DraftWindowGraphLifecycleStage,
    DraftWindowGraphOperandStage,
    audit_characterization_snapshot,
    experimental_draft_window_graph_characterization_enabled,
    install_parallel_draft_window_graph_characterization,
    maybe_install_parallel_draft_window_graph_characterization,
    prepare_parallel_draft_window_graph_characterization,
    record_parallel_graph_capture_operator_operand,
    record_parallel_graph_output,
    record_parallel_graph_replay_submission,
    record_parallel_graph_update_operator_operand,
    validate_base_graph_characterization_context,
)


class _FakeExternalEvent:
    def __init__(self) -> None:
        self.recorded_stream: Any | None = None

    def wait(self, _stream: Any) -> None:
        return None

    def reset(self, _stream: Any) -> None:
        return None

    def record(self, stream: Any) -> None:
        self.recorded_stream = stream

    def query(self) -> bool:
        return False


def _runtime_context(**overrides: Any) -> DraftWindowRuntimeContext:
    values: dict[str, Any] = {
        "attention_backend_family": AttentionBackendFamily.STANDARD,
        "graph_mode": "FULL",
        "async_scheduling": False,
        "tensor_parallel_size": 1,
        "data_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "sequence_parallel": False,
        "dsa_context_parallel": False,
        "flashcomm": False,
        "prefix_caching": False,
        "tensor_parallel_rank": 0,
    }
    values.update(overrides)
    return DraftWindowRuntimeContext(**values)


def _descriptor() -> BatchExecutionDescriptor:
    return BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=8,
        num_reqs=4,
        uniform_token_count=2,
    )


def _characterization_owner(*, method: str = "dflash") -> tuple[Any, Any]:
    runtime, capability, block_tables = _runtime_fixture(method=method)
    descriptor = _descriptor()
    input_buffers = InputBuffers(8, 64, torch.device("cpu"))
    input_buffers.query_start_loc[:5].copy_(torch.tensor([0, 2, 4, 6, 8], dtype=torch.int32))
    input_buffers.positions[:8].copy_(torch.arange(8, dtype=torch.int64))
    manager = SimpleNamespace(
        _capture_descs={CUDAGraphMode.FULL: [descriptor]},
        max_num_reqs=8,
    )
    owner = SimpleNamespace(
        _draft_window_runtime_state=runtime,
        _draft_window_source_capability=capability,
        query_cudagraph_manager=manager,
        num_query_per_req=2,
        input_buffers=input_buffers,
        block_tables=block_tables,
        input_batch=SimpleNamespace(num_reqs=2),
        model_state=SimpleNamespace(attn_metadata=_target_metadata()),
        attn_backends={"draft.layers.0.self_attn": AscendAttentionBackend},
        speculative_config=SimpleNamespace(num_speculative_tokens=1),
        _group_causal=True,
    )
    return owner, descriptor


def _install_metadata_builder(owner: Any) -> None:
    target = owner.model_state.attn_metadata

    def build_metadata(**kwargs: Any) -> dict[str, _Metadata]:
        read_inputs = kwargs["read_inputs"]
        target_layer = target["draft.layers.0.self_attn"]
        return {
            "draft.layers.0.self_attn": _Metadata(
                block_tables=read_inputs.block_tables[0],
                seq_lens=read_inputs.seq_lens,
                seq_lens_cpu=read_inputs.seq_lens_cpu_exact,
                seq_lens_list=read_inputs.seq_lens_cpu_exact.tolist(),
                actual_seq_lengths_q=[2, 4, 6, 8],
                slot_mapping=owner.block_tables.slot_mappings[0, :8],
                positions=owner.input_buffers.positions[:8],
                query_start_loc=owner.input_buffers.query_start_loc[:5],
                nested={
                    "tensor": target_layer.nested["tensor"].clone(),
                    "array": target_layer.nested["array"].copy(),
                },
                attn_state=target_layer.attn_state,
                num_actual_tokens=8,
                causal=True,
            )
        }

    owner._build_ascend_draft_attn_metadata = build_metadata


def _publish_eager_generation(owner: Any, inputs: Any) -> None:
    target = owner.model_state.attn_metadata
    owner._draft_window_runtime_state.publish_metadata(
        inputs,
        copy.deepcopy(target),
        target_metadata=target,
    )
    owner._draft_window_live_inputs = inputs
    owner._draft_window_proposal_context = SimpleNamespace(
        mapping_epoch=inputs.identity.mapping_epoch,
    )


def test_characterization_flag_is_strict_and_default_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VLLM_ASCEND_EXPERIMENTAL_DRAFT_WINDOW_GRAPH", raising=False)
    assert not experimental_draft_window_graph_characterization_enabled()
    monkeypatch.setenv("VLLM_ASCEND_EXPERIMENTAL_DRAFT_WINDOW_GRAPH", "1")
    assert experimental_draft_window_graph_characterization_enabled()
    monkeypatch.setenv("VLLM_ASCEND_EXPERIMENTAL_DRAFT_WINDOW_GRAPH", "true")
    with pytest.raises(ValueError, match="must be either '0' or '1'"):
        experimental_draft_window_graph_characterization_enabled()


@pytest.mark.parametrize(
    ("overrides", "diagnostic"),
    [
        ({"graph_mode": "NONE"}, "Graph mode"),
        ({"graph_mode": "PIECEWISE"}, "FULL"),
        ({"async_scheduling": True}, "async"),
        ({"tensor_parallel_size": 2}, "TP=1"),
        ({"prefix_caching": True}, "prefix"),
        (
            {"attention_backend_family": AttentionBackendFamily.COMPATIBILITY},
            "STANDARD",
        ),
    ],
)
def test_base_characterization_context_rejects_feature_expansions(
    overrides: dict[str, Any],
    diagnostic: str,
) -> None:
    with pytest.raises(NotImplementedError, match=diagnostic):
        validate_base_graph_characterization_context(_runtime_context(**overrides))


def test_flag_off_installs_no_provider_or_characterization_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_ASCEND_EXPERIMENTAL_DRAFT_WINDOW_GRAPH", "0")
    owner, _descriptor_value = _characterization_owner()

    assert (
        maybe_install_parallel_draft_window_graph_characterization(
            owner,
            runtime_context=_runtime_context(),
        )
        is None
    )
    assert not hasattr(owner, "_draft_window_graph_binding_provider")
    assert not hasattr(owner, "_draft_window_graph_characterization")


@pytest.mark.parametrize("owner_type", [AscendDFlashSpeculator])
def test_policy_off_owner_creates_zero_characterization_state(
    owner_type: type[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_ASCEND_EXPERIMENTAL_DRAFT_WINDOW_GRAPH", "1")
    owner = object.__new__(owner_type)

    owner._initialize_draft_window_source_capability()

    assert not hasattr(owner, "_draft_window_source_capability")
    assert not hasattr(owner, "_draft_window_runtime_state")
    assert not hasattr(owner, "_draft_window_graph_characterization_context")
    assert not hasattr(owner, "_draft_window_graph_characterization")
    assert not hasattr(owner, "_draft_window_graph_binding_provider")


def test_flag_on_eager_route_preserves_baseline_source_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_ASCEND_EXPERIMENTAL_DRAFT_WINDOW_GRAPH", "1")
    owner = object.__new__(AscendDFlashSpeculator)
    owner._draft_window_policy = object()
    owner._speculator_name = "DFlash"
    owner._group_causal = True
    owner.attn_groups = object()
    owner.block_tables = object()
    owner.vllm_config = object()
    capability = object()
    runtime = object()
    baseline_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(
        draft_window_module,
        "build_draft_window_runtime_context",
        lambda *_args: _runtime_context(graph_mode="NONE"),
    )
    monkeypatch.setattr(
        dflash_speculator_module,
        "build_draft_window_source_capability",
        lambda **kwargs: baseline_calls.append(kwargs) or capability,
    )
    monkeypatch.setattr(
        dflash_speculator_module,
        "build_draft_window_runtime_state",
        lambda actual: runtime if actual is capability else None,
    )
    monkeypatch.setattr(owner, "_initialize_draft_window_eager_state", lambda: None)

    owner._initialize_draft_window_source_capability()

    assert len(baseline_calls) == 1
    assert owner._draft_window_source_capability is capability
    assert owner._draft_window_runtime_state is runtime
    assert not hasattr(owner, "_draft_window_graph_characterization_context")
    assert not hasattr(owner, "_draft_window_graph_characterization")
    assert not hasattr(owner, "_draft_window_graph_binding_provider")


@pytest.mark.parametrize(
    ("method", "route"),
    [
        ("dflash", DraftWindowGraphRoute.DFLASH),
        ("dspark", DraftWindowGraphRoute.DSPARK),
    ],
)
def test_enabled_candidate_installs_instance_provider_and_registers_real_bucket(
    method: str,
    route: DraftWindowGraphRoute,
) -> None:
    owner, descriptor = _characterization_owner(method=method)

    session = install_parallel_draft_window_graph_characterization(
        owner,
        runtime_context=_runtime_context(),
    )

    assert owner._draft_window_graph_characterization is session
    identities = session.provider.registered_capture_identities(route=route)
    assert len(identities) == 1
    identity = identities[0]
    assert identity.fingerprint.capture_descriptor == format_dflash_capture_descriptor(
        descriptor,
        max_num_reqs=8,
    )
    assert identity.fingerprint.capture_bucket == (4, 8)
    assert identity.fingerprint.query_width == 2
    assert identity.fingerprint.graph_mode == "FULL"


def test_runtime_candidate_reuses_real_p3_generation_and_builds_private_metadata() -> None:
    owner, descriptor = _characterization_owner()
    session = install_parallel_draft_window_graph_characterization(
        owner,
        runtime_context=_runtime_context(),
    )
    runtime = owner._draft_window_runtime_state
    inputs = _materialize(runtime, mapping_epoch=10)
    target = owner.model_state.attn_metadata
    target_layer = target["draft.layers.0.self_attn"]
    target_table = target_layer.block_tables.clone()
    target_nested_tensor = target_layer.nested["tensor"].clone()
    target_nested_array = target_layer.nested["array"].copy()
    # Mirror the real upstream call immediately before run_fullgraph: P4 has
    # already published an eager metadata object for this exact generation.
    _publish_eager_generation(owner, inputs)
    _install_metadata_builder(owner)
    carrier = prepare_parallel_draft_window_graph_characterization(owner, descriptor)
    assert carrier is not None
    assert carrier.runtime_identity is not None
    assert carrier.runtime_identity.mapping_epoch == 10
    assert carrier.runtime_identity.generation_id == inputs.identity.generation_id
    assert carrier.read_inputs.window_block_table.data_ptr() != inputs.window_block_table.data_ptr()
    assert torch.equal(carrier.read_inputs.window_block_table, inputs.window_block_table)
    assert torch.equal(owner.block_tables.input_block_tables[0], runtime.capability.full_source._input_block_tables[0])
    batch = consume_draft_window_graph_metadata(
        owner,
        route=DraftWindowGraphRoute.DFLASH,
        expected_consumer_count=1,
    )
    assert batch is not None
    assert batch[0] is carrier.metadata
    assert owner.model_state.attn_metadata is target
    assert target["draft.layers.0.self_attn"] is target_layer
    torch.testing.assert_close(target_layer.block_tables, target_table)
    torch.testing.assert_close(target_layer.nested["tensor"], target_nested_tensor)
    assert (target_layer.nested["array"] == target_nested_array).all()
    graph_layer = carrier.metadata["draft.layers.0.self_attn"]
    assert graph_layer is not target_layer
    assert graph_layer.nested["tensor"].data_ptr() != target_layer.nested["tensor"].data_ptr()
    assert graph_layer.slot_mapping.data_ptr() == owner.block_tables.slot_mappings[0, :8].data_ptr()
    stages = [event.stage for event in session.recorder.events]
    assert DraftWindowGraphCharacterizationStage.MATERIALIZED in stages
    assert DraftWindowGraphCharacterizationStage.BOUND in stages


def test_real_target_metadata_sharing_is_allowed_for_full_write_carriers() -> None:
    """Mirror the real engine's canonical Target/write owner relationship."""
    owner, descriptor = _characterization_owner()
    session = install_parallel_draft_window_graph_characterization(
        owner,
        runtime_context=_runtime_context(),
    )
    runtime = owner._draft_window_runtime_state
    inputs = _materialize(runtime, mapping_epoch=10)

    target = owner.model_state.attn_metadata
    target_layer = target["draft.layers.0.self_attn"]
    canonical_slot = owner.block_tables.slot_mappings[0, :8]
    canonical_positions = owner.input_buffers.positions[:8]
    canonical_query_start = owner.input_buffers.query_start_loc[:5]
    target_layer.slot_mapping = canonical_slot
    target_layer.positions = canonical_positions
    target_layer.query_start_loc = canonical_query_start

    _publish_eager_generation(owner, inputs)
    _install_metadata_builder(owner)
    carrier = prepare_parallel_draft_window_graph_characterization(owner, descriptor)

    assert carrier is not None and carrier.metadata is not None
    graph_layer = carrier.metadata["draft.layers.0.self_attn"]
    assert graph_layer.slot_mapping.untyped_storage().data_ptr() == canonical_slot.untyped_storage().data_ptr()
    assert graph_layer.positions.untyped_storage().data_ptr() == canonical_positions.untyped_storage().data_ptr()
    assert (
        graph_layer.query_start_loc.untyped_storage().data_ptr() == canonical_query_start.untyped_storage().data_ptr()
    )
    assert graph_layer.block_tables is carrier.read_inputs.window_block_table
    assert graph_layer.seq_lens is carrier.read_inputs.device_window_exact
    assert (
        graph_layer.block_tables.untyped_storage().data_ptr() != target_layer.block_tables.untyped_storage().data_ptr()
    )
    assert graph_layer.seq_lens.untyped_storage().data_ptr() != target_layer.seq_lens.untyped_storage().data_ptr()
    assert session.recorder.events[-1].stage is DraftWindowGraphCharacterizationStage.BOUND


def test_capture_helper_records_real_persistent_operand_and_full_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, descriptor = _characterization_owner()
    owner.target_input_buffers = InputBuffers(8, 64, torch.device("cpu"))
    block_tables = owner.block_tables
    block_tables.cp_size = 1
    block_tables.cp_rank = 0
    block_tables.cp_interleave = 1
    block_tables.get_dummy_block_tables = lambda count: tuple(
        table[:count] for table in block_tables.input_block_tables
    )
    block_tables.get_dummy_slot_mappings = lambda count: block_tables.slot_mappings[:, :count].fill_(-1)
    session = install_parallel_draft_window_graph_characterization(
        owner,
        runtime_context=_runtime_context(),
    )
    identity = session.provider.registered_capture_identities(
        route=DraftWindowGraphRoute.DFLASH,
    )[0]

    monkeypatch.setattr(
        dflash_capture_module,
        "build_slot_mappings_by_layer",
        lambda slots, _config: {"draft": slots},
    )
    monkeypatch.setattr(
        dflash_capture_module,
        "maybe_prepare_dcp_local_seq_lens",
        lambda _out, seq_lens, *_args, **_kwargs: seq_lens,
    )
    monkeypatch.setattr(
        dflash_capture_module,
        "build_attn_metadata",
        lambda **_kwargs: {"draft": SimpleNamespace()},
    )

    with draft_window_dflash_capture_scope(
        owner,
        owner.input_buffers,
        {CUDAGraphMode.FULL: [descriptor]},
        max_num_reqs=8,
    ):
        state = _call_capture_helper(owner.input_buffers, block_tables)

    event = session.recorder.events[-1]
    assert event.stage is DraftWindowGraphCharacterizationStage.CAPTURE
    assert event.generation_id is None
    assert event.mapping_epoch is None
    assert event.backing_table_address == identity.backing_identity.window_block_table.data_ptr
    assert event.slot_mapping_address == state.slot_mappings["draft"].data_ptr()
    assert event.slot_mapping_address == block_tables.slot_mappings[:, :8].data_ptr()
    assert event.slot_mapping_values == (-1,) * 8
    assert event.position_values == (0,) * 8
    assert event.query_layout_values == (0, 2, 4, 6, 8)

    capture_metadata = state.attn_metadata["draft"]
    read_inputs = session.provider.startup_capture_read_inputs(identity)
    fake_event = SimpleNamespace(query=lambda: False)
    fake_handle = object()
    fake_stream = object()
    record_parallel_graph_capture_operator_operand(
        capture_metadata,
        block_table=read_inputs.window_block_table,
        seq_lengths=read_inputs.cpu_window_exact.tolist(),
        layer_name="draft.layers.0.self_attn",
        num_tokens=8,
        handle=fake_handle,
        event=fake_event,
        stream=fake_stream,
        backend_impl=type("Backend", (), {}),
    )
    operand = session.recorder.operand_events[-1]
    assert operand.stage is DraftWindowGraphOperandStage.CAPTURE_OPERATOR
    assert operand.evidence_level == "HOST_OPERATOR_BOUNDARY_FACT"
    assert operand.actual_table_address == identity.backing_identity.window_block_table.data_ptr
    assert operand.length_values == tuple(read_inputs.cpu_window_exact.tolist())


def test_runtime_update_operand_and_g4_records_are_machine_readable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    artifact = tmp_path / "g1-g5.json"
    monkeypatch.setenv("VLLM_ASCEND_EXPERIMENTAL_DRAFT_WINDOW_GRAPH_ARTIFACT", str(artifact))
    owner, descriptor = _characterization_owner()
    session = install_parallel_draft_window_graph_characterization(
        owner,
        runtime_context=_runtime_context(),
    )
    runtime = owner._draft_window_runtime_state
    inputs = _materialize(runtime, mapping_epoch=10)
    target = owner.model_state.attn_metadata
    _publish_eager_generation(owner, inputs)
    _install_metadata_builder(owner)
    carrier = prepare_parallel_draft_window_graph_characterization(owner, descriptor)
    assert carrier is not None and carrier.metadata is not None
    batch = consume_draft_window_graph_metadata(
        owner,
        route=DraftWindowGraphRoute.DFLASH,
        expected_consumer_count=1,
    )
    assert batch is not None
    binding = build_draft_window_graph_update_binding(
        owner,
        route=DraftWindowGraphRoute.DFLASH,
        draft_attn_metadatas=list(batch),
    )
    assert binding is not None
    record_parallel_graph_output(owner, torch.tensor([1.0, 2.0]))
    layer_name = "draft.layers.0.self_attn"
    layer = carrier.metadata[layer_name]
    fake_event = SimpleNamespace(query=lambda: False)
    record_parallel_graph_update_operator_operand(
        binding,
        draft_metadata=carrier.metadata,
        layer_name=layer_name,
        captured_block_table=carrier.read_inputs.window_block_table,
        actual_block_table=layer.block_tables,
        seq_lengths=layer.seq_lens_list,
        num_tokens=8,
        handle=object(),
        event=fake_event,
        stream=object(),
        backend_impl=type("Backend", (), {}),
    )

    assert session.recorder.g4_records[-1].target_metadata_identity == id(target)
    assert session.recorder.g4_records[-1].draft_output_hash is not None
    assert session.recorder.g4_records[-1].target_output_hash is None
    assert session.recorder.operand_events[-1].stage is DraftWindowGraphOperandStage.UPDATE_OPERATOR
    assert session.recorder.lifecycle_events[-1].stage is DraftWindowGraphLifecycleStage.UPDATE_EVENT_RECORDED
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["schema"] == "mrv2-draft-window-graph-characterization-v2"
    assert payload["evidence_boundary"]["physical_memory_transaction"] == ("NOT_PROVEN_REQUIRES_DEVICE_TOOL")
    assert payload["operand_events"][-1]["actual_table_address"] == (carrier.read_inputs.window_block_table.data_ptr())


def test_real_fia_capture_call_records_actual_operator_operand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, _descriptor_value = _characterization_owner()
    session = install_parallel_draft_window_graph_characterization(
        owner,
        runtime_context=_runtime_context(),
    )
    identity = session.provider.registered_capture_identities(
        route=DraftWindowGraphRoute.DFLASH,
    )[0]
    read_inputs = session.provider.startup_capture_read_inputs(identity)
    read_inputs.window_block_table.copy_(
        owner.block_tables.input_block_tables[0][
            : read_inputs.padded_request_count,
            : read_inputs.window_block_table.shape[1],
        ]
    )
    read_inputs.cpu_window_exact.fill_(12)
    read_inputs.device_window_exact.fill_(12)
    metadata = SimpleNamespace(
        actual_seq_lengths_q=[8],
        attn_mask=None,
        causal=True,
    )
    metadata._draft_window_graph_capture_operand_context = DraftWindowGraphCaptureOperandContext(
        recorder=session.recorder,
        identity=identity,
    )

    impl = AscendAttentionBackendImpl.__new__(AscendAttentionBackendImpl)
    impl.num_heads = 1
    impl.head_size = 4
    impl.num_kv_heads = 1
    impl.scale = 1.0
    impl.sliding_window = None
    impl.enable_c8_quant = False
    impl._use_max_workspace_for_fia_graph = False
    impl._use_layer_aware_fia_graph_replay = False
    impl.kv_sharing_target_layer_name = None
    impl._layer_name = None
    impl._get_fia_params = lambda key, value, _metadata: (
        key,
        value,
        16,
        read_inputs.window_block_table,
        read_inputs.cpu_window_exact.tolist(),
    )

    graph_params = SimpleNamespace(
        workspaces={8: torch.zeros(1)},
        events={8: []},
        handles={8: []},
        attn_params={8: []},
    )
    fake_stream = object()
    monkeypatch.setattr(attention_v1_module, "get_draft_graph_params", lambda: graph_params)
    monkeypatch.setattr(attention_v1_module, "weak_ref_tensors", lambda value: value)
    monkeypatch.setattr(attention_v1_module.torch_npu.npu, "current_stream", lambda: fake_stream)
    monkeypatch.setattr(torch.npu, "ExternalEvent", _FakeExternalEvent)
    monkeypatch.setattr(torch.npu, "graph_task_group_begin", lambda _stream: None)
    monkeypatch.setattr(torch.npu, "graph_task_group_end", lambda _stream: object())
    monkeypatch.setattr(
        attention_v1_module.torch_npu,
        "npu_fused_infer_attention_score",
        SimpleNamespace(out=lambda **_kwargs: None),
    )
    monkeypatch.setattr(
        attention_v1_module,
        "_EXTRA_CTX",
        SimpleNamespace(is_draft_model=True, is_draft_model_prefill=False),
    )

    query = torch.zeros((8, 1, 4))
    key = torch.zeros((8, 1, 4))
    value = torch.zeros((8, 1, 4))
    output = torch.zeros((8, 1, 4))
    result, num_tokens = impl.full_graph_fia(
        query,
        key,
        value,
        metadata,
        output,
    )

    assert result.shape == (8, 1, 4)
    assert num_tokens == 8
    operand = session.recorder.operand_events[-1]
    assert operand.stage is DraftWindowGraphOperandStage.CAPTURE_OPERATOR
    assert operand.actual_table_address == read_inputs.window_block_table.data_ptr()
    assert operand.length_values == (12, 12, 12, 12)


def test_real_fia_update_call_records_rebound_operator_operand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, descriptor = _characterization_owner()
    session = install_parallel_draft_window_graph_characterization(
        owner,
        runtime_context=_runtime_context(),
    )
    runtime = owner._draft_window_runtime_state
    inputs = _materialize(runtime, mapping_epoch=10)
    _publish_eager_generation(owner, inputs)
    _install_metadata_builder(owner)
    carrier = prepare_parallel_draft_window_graph_characterization(owner, descriptor)
    assert carrier is not None and carrier.metadata is not None
    batch = consume_draft_window_graph_metadata(
        owner,
        route=DraftWindowGraphRoute.DFLASH,
        expected_consumer_count=1,
    )
    assert batch is not None
    binding = build_draft_window_graph_update_binding(
        owner,
        route=DraftWindowGraphRoute.DFLASH,
        draft_attn_metadatas=list(batch),
    )
    assert binding is not None

    layer_name = "draft.layers.0.self_attn"
    query = torch.zeros((8, 1, 4))
    key = torch.zeros((8, 1, 4))
    value = torch.zeros((8, 1, 4))
    output = torch.zeros((8, 1, 4))
    softmax_lse = torch.zeros(1)
    captured_table = carrier.read_inputs.window_block_table
    event = _FakeExternalEvent()
    handle = object()
    graph_params = SimpleNamespace(
        workspaces={8: torch.zeros(1)},
        events={8: [event]},
        handles={8: [handle]},
        attn_params={
            8: [
                (
                    query,
                    key,
                    value,
                    captured_table,
                    None,
                    16,
                    [999],
                    None,
                    1,
                    1,
                    1.0,
                    output,
                    softmax_lse,
                    3,
                    2**31 - 1,
                    2**31 - 1,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            ]
        },
    )
    operator_calls: list[dict[str, Any]] = []

    @contextmanager
    def stream_context(_stream: Any):
        yield

    monkeypatch.setattr(attention_v1_module, "needs_layer_aware_fia_graph_replay", lambda: False)
    monkeypatch.setattr(attention_v1_module, "using_paged_attention", lambda *_args: False)
    monkeypatch.setattr(attention_v1_module, "get_draft_graph_params", lambda: graph_params)
    monkeypatch.setattr(torch.npu, "stream", stream_context)
    monkeypatch.setattr(torch.npu, "graph_task_update_begin", lambda *_args: None)
    monkeypatch.setattr(torch.npu, "graph_task_update_end", lambda *_args: None)
    monkeypatch.setattr(
        attention_v1_module.torch_npu,
        "npu_fused_infer_attention_score",
        SimpleNamespace(out=lambda **kwargs: operator_calls.append(kwargs)),
    )
    monkeypatch.setattr(
        attention_v1_module,
        "_EXTRA_CTX",
        SimpleNamespace(
            is_draft_model=True,
            is_draft_model_prefill=False,
            sinks=False,
        ),
    )

    update_stream = object()
    AscendAttentionBackendImpl.update_graph_params(
        update_stream,
        SimpleNamespace(attn_metadata={}),
        8,
        object(),
        object(),
        draft_attn_metadatas=[carrier.metadata],
        draft_window_graph_update_binding=binding,
    )

    assert len(operator_calls) == 1
    assert operator_calls[0]["block_table"] is captured_table
    assert operator_calls[0]["actual_seq_lengths_kv"] == (carrier.metadata[layer_name].seq_lens_list)
    operand = session.recorder.operand_events[-1]
    assert operand.stage is DraftWindowGraphOperandStage.UPDATE_OPERATOR
    assert operand.generation_id == inputs.identity.generation_id
    assert operand.actual_table_address == captured_table.data_ptr()
    assert event.recorded_stream is update_stream


def test_replay_submission_records_real_event_without_synchronizing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, descriptor = _characterization_owner()
    session = install_parallel_draft_window_graph_characterization(
        owner,
        runtime_context=_runtime_context(),
    )
    runtime = owner._draft_window_runtime_state
    inputs = _materialize(runtime, mapping_epoch=10)
    _publish_eager_generation(owner, inputs)
    _install_metadata_builder(owner)
    prepare_parallel_draft_window_graph_characterization(owner, descriptor)
    consume_draft_window_graph_metadata(
        owner,
        route=DraftWindowGraphRoute.DFLASH,
        expected_consumer_count=1,
    )

    stream = object()
    from vllm_ascend.worker.v2.spec_decode import draft_window_async

    monkeypatch.setattr(
        draft_window_async.DraftWindowNPUEventCompletionToken,
        "record_current_stream",
        classmethod(lambda _cls: pytest.fail("passive recorder must not create a device event")),
    )
    monkeypatch.setattr(torch.npu, "current_stream", lambda: stream)
    record_parallel_graph_replay_submission(owner)

    lifecycle = session.recorder.lifecycle_events[-1]
    assert lifecycle.stage is DraftWindowGraphLifecycleStage.REPLAY_SUBMITTED
    assert lifecycle.stream_identity == id(stream)
    assert lifecycle.event_identity is None
    assert lifecycle.event_complete is None  # Passive observation is not completion polling.


def test_campaign_auditor_requires_all_evidence_layers_and_detects_host_a_b_a() -> None:
    stages = (
        "capture",
        "materialized",
        "bound",
        "live",
        "replay",
        "consumer",
        "update",
    )
    events = [{"stage": stage} for stage in stages]
    events.extend(
        [
            {
                "stage": "bound",
                "backing_table_address": 123,
                "table_values": [[1, 2]],
                "generation_id": 10,
            },
            {
                "stage": "bound",
                "backing_table_address": 123,
                "table_values": [[9, 8]],
                "generation_id": 11,
            },
            {
                "stage": "bound",
                "backing_table_address": 123,
                "table_values": [[1, 2]],
                "generation_id": 12,
            },
        ]
    )
    result = audit_characterization_snapshot(
        {
            "schema": "mrv2-draft-window-graph-characterization-v2",
            "events": events,
            "operand_events": [
                {"stage": "capture_operator"},
                {"stage": "update_operator"},
            ],
            "g4_records": [{}],
            "lifecycle_events": [
                {"stage": "replay_submitted"},
                {"stage": "update_event_recorded"},
                {"stage": "proposal_retired"},
            ],
        }
    )
    assert result["G1_HOST_OPERATOR_BOUNDARY"] == "PASS"
    assert result["G2_HOST_A_B_A"] == "PASS"
    assert result["G3_PHYSICAL_MEMORY_TRANSACTION"] == "DEVICE_TOOL_REQUIRED"
    assert result["G4_WRITE_TARGET_ORACLE"] == "PASS"
    assert result["G5_UPDATE_EFFECTIVE_GENERATION"] == "NPU_TASK_TRACE_REQUIRED"
    assert result["ready_for_device_campaign"] is True


def test_runtime_candidate_rejects_wrong_mapping_epoch_before_binding() -> None:
    owner, descriptor = _characterization_owner()
    session = install_parallel_draft_window_graph_characterization(
        owner,
        runtime_context=_runtime_context(),
    )
    runtime = owner._draft_window_runtime_state
    inputs = _materialize(runtime, mapping_epoch=10)
    _publish_eager_generation(owner, inputs)
    owner._draft_window_proposal_context = SimpleNamespace(mapping_epoch=11)
    _install_metadata_builder(owner)

    with pytest.raises(RuntimeError, match="proposal identity is stale"):
        prepare_parallel_draft_window_graph_characterization(owner, descriptor)
    assert session.provider.submitted_carriers == ()


def test_runtime_candidate_rejects_early_overwrite_and_cleans_bound_state() -> None:
    owner, descriptor = _characterization_owner()
    session = install_parallel_draft_window_graph_characterization(
        owner,
        runtime_context=_runtime_context(),
    )
    runtime = owner._draft_window_runtime_state
    inputs = _materialize(runtime, mapping_epoch=10)
    _publish_eager_generation(owner, inputs)
    _install_metadata_builder(owner)
    carrier = prepare_parallel_draft_window_graph_characterization(owner, descriptor)
    assert carrier is not None and carrier.metadata is not None

    with pytest.raises(RuntimeError, match="cannot overwrite an active"):
        prepare_parallel_draft_window_graph_characterization(owner, descriptor)
    assert session.provider.submitted_carriers == ()
    assert carrier.metadata is None
    runtime.abort_generation(inputs.identity)


def test_runtime_candidate_recorder_exception_releases_bound_carrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, descriptor = _characterization_owner()
    session = install_parallel_draft_window_graph_characterization(
        owner,
        runtime_context=_runtime_context(),
    )
    runtime = owner._draft_window_runtime_state
    inputs = _materialize(runtime, mapping_epoch=10)
    _publish_eager_generation(owner, inputs)
    _install_metadata_builder(owner)

    original_record_bound = session.recorder.record_bound
    attempts = 0

    def fail_once(carrier: Any) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("injected characterization recorder failure")
        original_record_bound(carrier)

    monkeypatch.setattr(session.recorder, "record_bound", fail_once)
    with pytest.raises(RuntimeError, match="injected characterization recorder failure"):
        prepare_parallel_draft_window_graph_characterization(owner, descriptor)

    # The proposal scope remains the sole P3 retirement owner on an exception.
    runtime.abort_generation(inputs.identity)
    inputs_next = _materialize(runtime, mapping_epoch=11)
    _publish_eager_generation(owner, inputs_next)
    carrier = prepare_parallel_draft_window_graph_characterization(owner, descriptor)
    assert carrier is not None
    assert carrier.runtime_identity is not None
    assert carrier.runtime_identity.mapping_epoch == 11


def test_runtime_candidate_host_a_b_a_refreshes_content_at_stable_address() -> None:
    owner, descriptor = _characterization_owner()
    session = install_parallel_draft_window_graph_characterization(
        owner,
        runtime_context=_runtime_context(),
    )
    runtime = owner._draft_window_runtime_state
    _install_metadata_builder(owner)

    inputs_a = _materialize(runtime, mapping_epoch=10, start=0)
    _publish_eager_generation(owner, inputs_a)
    carrier_a = prepare_parallel_draft_window_graph_characterization(owner, descriptor)
    assert carrier_a is not None
    address = carrier_a.read_inputs.window_block_table.data_ptr()
    table_a = carrier_a.read_inputs.window_block_table.clone()
    session.provider.discard_bound_carriers()
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
    _publish_eager_generation(owner, inputs_b)
    carrier_b = prepare_parallel_draft_window_graph_characterization(owner, descriptor)
    assert carrier_b is not None
    table_b = carrier_b.read_inputs.window_block_table.clone()
    assert carrier_b.read_inputs.window_block_table.data_ptr() == address
    assert not torch.equal(table_a, table_b)
    session.provider.discard_bound_carriers()
    runtime.retire_generation(inputs_b.identity)

    inputs_a2 = _materialize(runtime, mapping_epoch=12, start=0)
    _publish_eager_generation(owner, inputs_a2)
    carrier_a2 = prepare_parallel_draft_window_graph_characterization(owner, descriptor)
    assert carrier_a2 is not None
    assert carrier_a2.read_inputs.window_block_table.data_ptr() == address
    torch.testing.assert_close(carrier_a2.read_inputs.window_block_table, table_a)
    assert inputs_a2.identity.generation_id != inputs_a.identity.generation_id
    with pytest.raises(RuntimeError, match="stale runtime binding"):
        session.provider.require_runtime_identity(inputs_a.identity)


def test_manager_prepares_candidate_before_consuming_graph_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    target = _target_metadata()
    speculator = SimpleNamespace(
        input_batch=SimpleNamespace(seq_lens_cpu_upper_bound=object()),
        attn_backends={"draft.layers.0.self_attn": AscendAttentionBackend},
        dp_size=1,
        model_state=SimpleNamespace(attn_metadata=target),
        speculative_config=object(),
    )

    def prepare(owner: Any, desc: Any) -> None:
        assert owner is speculator
        assert desc.cg_mode is CUDAGraphMode.FULL
        calls.append("prepare")

    def build(*_args: Any) -> list[Any]:
        calls.append("build")
        return [target]

    speculator.build_draft_attn_metadatas = build
    manager = DFlashAclGraphManager.__new__(DFlashAclGraphManager)
    manager.speculator = speculator
    manager.update_stream = SimpleNamespace(wait_stream=lambda _stream: None)
    manager.device = torch.device("cpu")
    manager.vllm_config = object()

    @contextmanager
    def no_op_context(*_args: Any, **_kwargs: Any):
        yield

    monkeypatch.setattr(
        dflash_aclgraph_module,
        "prepare_parallel_draft_window_graph_characterization",
        prepare,
    )
    monkeypatch.setattr(dflash_aclgraph_module, "set_forward_context", no_op_context)
    monkeypatch.setattr(dflash_aclgraph_module, "get_forward_context", lambda: object())
    monkeypatch.setattr(
        dflash_aclgraph_module,
        "record_parallel_graph_replay_submission",
        lambda _owner: calls.append("replay_event"),
    )
    monkeypatch.setattr(
        dflash_aclgraph_module,
        "record_parallel_graph_output",
        lambda _owner, _output: calls.append("output_hash"),
    )
    monkeypatch.setattr(
        dflash_aclgraph_module,
        "update_full_graph_params",
        lambda *_args, **_kwargs: calls.append("update"),
    )
    monkeypatch.setattr(torch.npu, "current_stream", lambda: object())
    monkeypatch.setattr(
        DFlashCudaGraphManager,
        "run_fullgraph",
        lambda self, desc: calls.append("replay") or "replayed",
    )
    monkeypatch.setattr(
        dflash_aclgraph_module,
        "_EXTRA_CTX",
        SimpleNamespace(is_draft_model=False, is_draft_model_prefill=False),
    )

    assert manager.run_fullgraph(_descriptor()) == "replayed"
    assert calls == [
        "prepare",
        "build",
        "replay",
        "replay_event",
        "update",
        # D2H cannot wait on a graph whose ExternalEvent update is unsubmitted.
        "output_hash",
    ]
