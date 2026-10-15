# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU-only contracts for the MRV2 draft-window P3 read plane."""

from dataclasses import FrozenInstanceError, replace
from enum import Enum
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionMetadataBuilder,
)
from vllm_ascend.worker.v2.spec_decode.draft_window import (
    DraftMetadataInputs,
    DraftWindowExecutionPhase,
    DraftWindowGenerationIdentity,
    DraftWindowGroupInventory,
    DraftWindowInventory,
    DraftWindowLayoutCategory,
    DraftWindowLeaseState,
    DraftWindowReadDescriptor,
    DraftWindowSourceCapability,
    SingleGroupDirectPrefixProvider,
    build_draft_window_runtime_state,
    build_full_draft_kv_source,
)


class _FakeCompletionToken:
    def __init__(self, *, complete: bool = False) -> None:
        self.complete = complete

    def is_complete(self) -> bool:
        return self.complete


class _RaisingCompletionToken(_FakeCompletionToken):
    def __init__(self) -> None:
        super().__init__()
        self.raise_on_query = True

    def is_complete(self) -> bool:
        if self.raise_on_query:
            raise RuntimeError("completion query failed")
        return super().is_complete()


class _SlottedTensorHolder:
    __slots__ = ("tensor",)

    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor


class _ImmutablePhase(Enum):
    DECODE = "decode"


def _runtime_fixture(
    *,
    max_num_reqs: int = 128,
    row_width: int = 8,
) -> tuple[Any, Any, SimpleNamespace]:
    canonical_counts = torch.full(
        (1, max_num_reqs),
        row_width,
        dtype=torch.int32,
    )
    block_tables = SimpleNamespace(
        block_sizes=[64],
        kernel_block_sizes=[64],
        input_block_tables=[
            torch.arange(max_num_reqs * row_width, dtype=torch.int32).reshape(
                max_num_reqs,
                row_width,
            )
        ],
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
        method="eagle3",
        target_model_type="qwen3",
        runtime_route="eagle3",
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
            expected_future_offset=64,
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
    mapping_epoch: int,
    idx_mapping: torch.Tensor | None = None,
    request_ids: tuple[str, ...] = ("A", "B"),
    descriptors: tuple[DraftWindowReadDescriptor, ...] | None = None,
    num_reqs_padded: int = 4,
    exact: tuple[int, ...] = (96, 64, 999, 999),
    upper: tuple[int, ...] = (128, 64, 999, 999),
    step: int = 0,
) -> DraftMetadataInputs:
    if idx_mapping is None:
        idx_mapping = torch.tensor([17, 93], dtype=torch.int32)
    if descriptors is None:
        descriptors = tuple(_descriptor() for _ in request_ids)
    return runtime.materialize(
        mapping_epoch=mapping_epoch,
        ordered_request_ids=request_ids,
        idx_mapping=idx_mapping,
        read_descriptors=descriptors,
        device_window_exact=torch.tensor(exact, dtype=torch.int32),
        cpu_window_exact=torch.tensor(exact, dtype=torch.int32),
        cpu_window_upper=torch.tensor(upper, dtype=torch.int32),
        num_reqs_padded=num_reqs_padded,
        phase=DraftWindowExecutionPhase.PROPOSAL,
        step=step,
        execution_lane=0,
    )


def test_workspace_materializes_once_without_aliasing_canonical_source() -> None:
    runtime, _capability, block_tables = _runtime_fixture()
    canonical_table_before = block_tables.input_block_tables[0].clone()
    canonical_counts_before = block_tables.num_blocks.gpu.clone()
    workspace = runtime.workspace
    pointers_before = workspace.storage_pointers

    inputs = _materialize(runtime, mapping_epoch=10)

    assert runtime.workspace is workspace
    assert workspace.storage_pointers == pointers_before
    assert inputs.window_block_table.shape == (4, 3)
    assert inputs.window_block_table[:2].tolist() == [
        block_tables.input_block_tables[0][0, :2].tolist() + [0],
        block_tables.input_block_tables[0][1, :2].tolist() + [0],
    ]
    assert inputs.window_block_table[2:].count_nonzero().item() == 0
    assert inputs.valid_num_blocks.tolist() == [8, 8, 0, 0]
    assert inputs.device_window_exact.tolist() == [96, 64, 0, 0]
    assert inputs.cpu_window_exact.tolist() == [96, 64, 0, 0]
    assert inputs.cpu_window_upper.tolist() == [128, 64, 0, 0]
    assert inputs.active_table_columns == (2, 2)
    assert not inputs.window_block_table.is_set_to(block_tables.input_block_tables[0])
    assert not inputs.valid_num_blocks.is_set_to(block_tables.num_blocks.gpu)
    torch.testing.assert_close(block_tables.input_block_tables[0], canonical_table_before)
    torch.testing.assert_close(block_tables.num_blocks.gpu, canonical_counts_before)


def test_workspace_same_source_mode_avoids_device_to_host_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _capability, block_tables = _runtime_fixture()
    descriptors = (_descriptor(), _descriptor())
    cpu_exact = torch.tensor([96, 64, 0, 0], dtype=torch.int32)
    cpu_upper = torch.tensor([128, 64, 0, 0], dtype=torch.int32)
    cpu_valid = torch.tensor([8, 8, 0, 0], dtype=torch.int32)

    def forbidden_cpu(_tensor: torch.Tensor, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("same-source production materialization must not call Tensor.cpu()")

    monkeypatch.setattr(torch.Tensor, "cpu", forbidden_cpu)
    inputs = runtime.materialize(
        mapping_epoch=1,
        ordered_request_ids=("A", "B"),
        idx_mapping=torch.tensor([17, 93], dtype=torch.int32),
        read_descriptors=descriptors,
        device_window_exact=None,
        cpu_window_exact=cpu_exact,
        cpu_window_upper=cpu_upper,
        cpu_valid_num_blocks=cpu_valid,
        num_reqs_padded=4,
        phase=DraftWindowExecutionPhase.PROPOSAL,
        step=0,
        execution_lane=0,
    )

    assert inputs.device_window_exact.tolist() == [96, 64, 0, 0]
    assert inputs.valid_num_blocks.tolist() == [8, 8, 0, 0]
    assert not inputs.valid_num_blocks.is_set_to(block_tables.num_blocks.gpu)


def test_workspace_reuse_clears_shrunk_rows_and_tails() -> None:
    runtime, _capability, _block_tables = _runtime_fixture()
    first = _materialize(runtime, mapping_epoch=1)
    table_pointer = first.window_block_table.data_ptr()
    runtime.retire_generation(first.identity)

    second = _materialize(
        runtime,
        mapping_epoch=2,
        idx_mapping=torch.tensor([4], dtype=torch.int32),
        request_ids=("C",),
        descriptors=(_descriptor(start=1, required=1),),
        num_reqs_padded=2,
        exact=(32, 999),
        upper=(64, 999),
    )

    assert second.window_block_table.data_ptr() == table_pointer
    assert second.window_block_table[0, 1:].count_nonzero().item() == 0
    assert second.window_block_table[1].count_nonzero().item() == 0
    assert second.valid_num_blocks.tolist() == [8, 0]
    assert second.cpu_window_exact.tolist() == [32, 0]


def test_workspace_success_path_avoids_redundant_preclear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _capability, _block_tables = _runtime_fixture(
        max_num_reqs=16,
        row_width=32,
    )
    workspace = runtime.workspace
    full_clear_calls = 0
    original_zero_all = workspace._zero_all

    def counted_zero_all() -> None:
        nonlocal full_clear_calls
        full_clear_calls += 1
        original_zero_all()

    monkeypatch.setattr(workspace, "_zero_all", counted_zero_all)
    inputs = _materialize(
        runtime,
        mapping_epoch=1,
        idx_mapping=torch.tensor([3, 5], dtype=torch.int32),
        descriptors=(
            replace(_descriptor(required=2, valid=32), source_row_width=32),
            replace(_descriptor(required=1, valid=32), source_row_width=32),
        ),
    )
    runtime.retire_generation(inputs.identity)

    assert full_clear_calls == 1
    assert workspace.bound_identity is None
    assert workspace.window_block_table.count_nonzero().item() == 0
    assert workspace.valid_num_blocks.count_nonzero().item() == 0
    assert workspace.device_window_exact.count_nonzero().item() == 0
    assert workspace.cpu_window_exact.count_nonzero().item() == 0
    assert workspace.cpu_window_upper.count_nonzero().item() == 0


def test_workspace_materializes_an_empty_batch_without_stale_state() -> None:
    runtime, _capability, _block_tables = _runtime_fixture()

    inputs = _materialize(
        runtime,
        mapping_epoch=1,
        idx_mapping=torch.empty(0, dtype=torch.int32),
        request_ids=(),
        descriptors=(),
        num_reqs_padded=0,
        exact=(),
        upper=(),
    )

    assert inputs.actual_request_count == 0
    assert inputs.padded_request_count == 0
    assert inputs.window_block_table.shape == (0, 3)
    assert inputs.active_table_columns == ()
    runtime.retire_generation(inputs.identity)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"num_reqs_padded": 129}, "request capacity"),
        (
            {
                "descriptors": (_descriptor(required=4), _descriptor()),
            },
            "table capacity",
        ),
    ],
)
def test_workspace_fixed_capacity_fails_deterministically(
    change: dict[str, Any],
    message: str,
) -> None:
    runtime, _capability, _block_tables = _runtime_fixture()

    with pytest.raises(RuntimeError, match=message):
        _materialize(runtime, mapping_epoch=1, **change)

    recovered = _materialize(runtime, mapping_epoch=2)
    assert recovered.identity.mapping_epoch == 2


def test_same_mapping_tensor_rewrite_cannot_reuse_generation_state() -> None:
    runtime, _capability, _block_tables = _runtime_fixture()
    idx_mapping = torch.tensor([17, 93], dtype=torch.int32)
    pointer = idx_mapping.data_ptr()
    generation_a = _materialize(
        runtime,
        mapping_epoch=10,
        idx_mapping=idx_mapping,
        request_ids=("B", "A"),
    )
    runtime.publish_metadata(
        generation_a,
        {"generation": "A"},
        target_metadata={},
    )

    idx_mapping.copy_(torch.tensor([4, 17], dtype=torch.int32))
    assert idx_mapping.data_ptr() == pointer
    expected_b = replace(
        generation_a.identity,
        mapping_epoch=11,
        ordered_request_fingerprint=("C", "A"),
    )

    with pytest.raises(RuntimeError, match="generation identity"):
        generation_a.validate_for(expected_b)
    with pytest.raises(RuntimeError, match="generation identity"):
        runtime.read_metadata(expected_b)

    runtime.retire_generation(generation_a.identity)
    generation_b = _materialize(
        runtime,
        mapping_epoch=11,
        idx_mapping=idx_mapping,
        request_ids=("C", "A"),
    )
    assert generation_b.identity.source_identity.idx_mapping_ptr == pointer
    with pytest.raises(RuntimeError, match="retired lease"):
        generation_a.validate_for(generation_a.identity)
    with pytest.raises(RuntimeError, match="no active draft metadata"):
        runtime.read_metadata(generation_a.identity)


@pytest.mark.parametrize(
    "fingerprint",
    [
        ("B", "A"),
        ("A",),
        ("C", "A"),
        ("D", "B"),
    ],
)
def test_request_reorder_finish_add_and_same_row_replacement_are_stale(
    fingerprint: tuple[str, ...],
) -> None:
    runtime, _capability, _block_tables = _runtime_fixture()
    inputs = _materialize(runtime, mapping_epoch=3, request_ids=("A", "B"))
    stale = replace(
        inputs.identity,
        ordered_request_fingerprint=fingerprint,
        actual_request_count=len(fingerprint),
        read_descriptor_fingerprint=inputs.identity.read_descriptor_fingerprint[: len(fingerprint)],
    )
    payload = {"generation": "original"}
    runtime.publish_metadata(inputs, payload, target_metadata={})

    with pytest.raises(RuntimeError, match="generation identity"):
        inputs.validate_for(stale)
    with pytest.raises(RuntimeError, match="generation identity"):
        runtime.read_metadata(stale)
    assert runtime.read_metadata(inputs.identity) is payload
    runtime.retire_generation(inputs.identity)


def test_descriptor_and_valid_count_snapshot_are_generation_bound() -> None:
    runtime, _capability, block_tables = _runtime_fixture()
    inputs = _materialize(runtime, mapping_epoch=1)
    stale_descriptor = replace(
        inputs.identity.read_descriptor_fingerprint[0],
        valid_num_blocks=7,
    )
    stale_identity = replace(
        inputs.identity,
        read_descriptor_fingerprint=(
            stale_descriptor,
            inputs.identity.read_descriptor_fingerprint[1],
        ),
    )

    with pytest.raises(RuntimeError, match="generation identity"):
        inputs.validate_for(stale_identity)

    runtime.retire_generation(inputs.identity)
    block_tables.num_blocks.gpu[0, 17] = 7
    with pytest.raises(RuntimeError, match="valid count is stale"):
        _materialize(runtime, mapping_epoch=2)

    block_tables.num_blocks.gpu[0, 17] = 8
    assert _materialize(runtime, mapping_epoch=3).identity.mapping_epoch == 3


def test_mapping_epoch_is_strictly_monotonic_and_aba_generation_is_rejected() -> None:
    runtime, _capability, _block_tables = _runtime_fixture()
    first = _materialize(runtime, mapping_epoch=7)
    first_generation_id = first.identity.generation_id
    runtime.retire_generation(first.identity)

    with pytest.raises(ValueError, match="strictly increase"):
        _materialize(runtime, mapping_epoch=7)

    second = _materialize(runtime, mapping_epoch=8)
    assert second.identity.generation_id > first_generation_id
    aba = replace(second.identity, generation_id=first_generation_id)
    with pytest.raises(RuntimeError, match="generation identity"):
        second.validate_for(aba)


def test_incomplete_reader_blocks_reuse_until_completion() -> None:
    runtime, _capability, _block_tables = _runtime_fixture()
    inputs = _materialize(runtime, mapping_epoch=1)
    event = _FakeCompletionToken()
    runtime.mark_reader_pending(inputs, event)
    table_before_pressure = inputs.window_block_table.clone()
    assert inputs.lease.state is DraftWindowLeaseState.READER_PENDING
    runtime.retire_generation(inputs.identity)
    assert inputs.lease.state is DraftWindowLeaseState.RETIRED_PENDING

    with pytest.raises(RuntimeError, match="execution lane 0 is still in use"):
        _materialize(runtime, mapping_epoch=2)
    torch.testing.assert_close(inputs.window_block_table, table_before_pressure)

    event.complete = True
    replacement = _materialize(runtime, mapping_epoch=2)
    assert replacement.lease.state is DraftWindowLeaseState.ACQUIRED
    assert inputs.lease.state is DraftWindowLeaseState.RETIRED


def test_completion_query_failure_keeps_lane_owned_and_can_be_retried() -> None:
    runtime, _capability, _block_tables = _runtime_fixture()
    inputs = _materialize(runtime, mapping_epoch=1)
    event = _RaisingCompletionToken()
    runtime.publish_metadata(
        inputs,
        {"generation": "A"},
        target_metadata={},
    )
    runtime.mark_reader_pending(inputs, event)

    with pytest.raises(RuntimeError, match="completion query failed"):
        runtime.retire_generation(inputs.identity)

    assert inputs.lease.state is DraftWindowLeaseState.READER_PENDING
    with pytest.raises(RuntimeError, match="no active draft metadata"):
        runtime.read_metadata(inputs.identity)
    with pytest.raises(RuntimeError, match="execution lane 0 is still in use"):
        _materialize(runtime, mapping_epoch=2)

    event.raise_on_query = False
    event.complete = True
    runtime.retire_generation(inputs.identity)
    assert inputs.lease.state is DraftWindowLeaseState.RETIRED
    assert _materialize(runtime, mapping_epoch=2).identity.mapping_epoch == 2


def test_lease_rejects_wrong_generation_double_acquire_and_double_retire() -> None:
    runtime, _capability, _block_tables = _runtime_fixture()
    inputs = _materialize(runtime, mapping_epoch=1)
    wrong = replace(inputs.identity, mapping_epoch=2)

    with pytest.raises(RuntimeError, match="execution lane 0 is still in use"):
        _materialize(runtime, mapping_epoch=2)
    with pytest.raises(RuntimeError, match="generation identity"):
        runtime.retire_generation(wrong)

    runtime.retire_generation(inputs.identity)
    with pytest.raises(RuntimeError, match="retired lease"):
        inputs.lease.retire(inputs.identity)


def test_materialize_validation_failure_releases_lease_but_consumes_epoch() -> None:
    runtime, _capability, _block_tables = _runtime_fixture()

    with pytest.raises(ValueError, match="upper.*exact"):
        _materialize(
            runtime,
            mapping_epoch=20,
            upper=(95, 64, 0, 0),
        )

    recovered = _materialize(runtime, mapping_epoch=21)
    assert recovered.identity.mapping_epoch == 21


def test_negative_exact_length_fails_closed_and_releases_lease() -> None:
    runtime, _capability, _block_tables = _runtime_fixture()

    with pytest.raises(ValueError, match="exact lengths.*greater than or equal to 0"):
        _materialize(
            runtime,
            mapping_epoch=1,
            exact=(-1, 64, 0, 0),
            upper=(0, 64, 0, 0),
        )

    assert _materialize(runtime, mapping_epoch=2).identity.mapping_epoch == 2


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("cpu_window_exact", torch.tensor([96, 64, 0, 0], dtype=torch.int64), "int32"),
        ("cpu_window_upper", torch.tensor([[128, 64, 0, 0]], dtype=torch.int32), "one-dimensional"),
        ("device_window_exact", torch.tensor([96], dtype=torch.int32), "padded request count"),
    ],
)
def test_typed_length_inputs_validate_dtype_and_shape(
    field: str,
    value: torch.Tensor,
    error: str,
) -> None:
    runtime, _capability, _block_tables = _runtime_fixture()
    kwargs = {
        "device_window_exact": torch.tensor([96, 64, 0, 0], dtype=torch.int32),
        "cpu_window_exact": torch.tensor([96, 64, 0, 0], dtype=torch.int32),
        "cpu_window_upper": torch.tensor([128, 64, 0, 0], dtype=torch.int32),
    }
    kwargs[field] = value

    with pytest.raises((TypeError, ValueError), match=error):
        runtime.materialize(
            mapping_epoch=1,
            ordered_request_ids=("A", "B"),
            idx_mapping=torch.tensor([17, 93], dtype=torch.int32),
            read_descriptors=(_descriptor(), _descriptor()),
            num_reqs_padded=4,
            phase=DraftWindowExecutionPhase.PROPOSAL,
            step=0,
            execution_lane=0,
            **kwargs,
        )


def test_typed_inputs_are_immutable_and_do_not_transport_write_state() -> None:
    runtime, _capability, _block_tables = _runtime_fixture()
    inputs = _materialize(runtime, mapping_epoch=1)

    with pytest.raises(FrozenInstanceError):
        inputs.actual_request_count = 9  # type: ignore[misc]
    assert not hasattr(inputs, "positions")
    assert not hasattr(inputs, "query_start_loc")
    assert not hasattr(inputs, "slot_mapping")
    assert not hasattr(inputs, "target_metadata")
    assert not hasattr(inputs, "canonical_full_block_table")


def test_materialize_rejects_length_input_aliasing_workspace_backing() -> None:
    runtime, _capability, _block_tables = _runtime_fixture()

    with pytest.raises(ValueError, match="must not alias workspace backing"):
        runtime.materialize(
            mapping_epoch=1,
            ordered_request_ids=("A", "B"),
            idx_mapping=torch.tensor([17, 93], dtype=torch.int32),
            read_descriptors=(_descriptor(), _descriptor()),
            device_window_exact=torch.tensor([96, 64, 0, 0], dtype=torch.int32),
            cpu_window_exact=runtime.workspace.cpu_window_exact[:4],
            cpu_window_upper=torch.tensor([128, 64, 0, 0], dtype=torch.int32),
            num_reqs_padded=4,
            phase=DraftWindowExecutionPhase.PROPOSAL,
            step=0,
            execution_lane=0,
        )

    assert _materialize(runtime, mapping_epoch=2).identity.mapping_epoch == 2


def test_store_rejects_target_nested_alias_and_cleans_generation() -> None:
    runtime, _capability, _block_tables = _runtime_fixture()
    inputs = _materialize(runtime, mapping_epoch=1)
    target_tensor = torch.tensor([11], dtype=torch.int32)
    target = {"nested": [{"tensor": target_tensor}]}
    draft = {"nested": [{"tensor": target_tensor}]}

    with pytest.raises(ValueError, match="mutable storage aliases Target metadata"):
        runtime.publish_metadata(inputs, draft, target_metadata=target)

    assert target["nested"][0]["tensor"] is target_tensor
    recovered = _materialize(runtime, mapping_epoch=2)
    assert recovered.identity.mapping_epoch == 2


def test_store_rejects_numpy_view_aliasing_target_storage() -> None:
    runtime, _capability, _block_tables = _runtime_fixture()
    inputs = _materialize(runtime, mapping_epoch=1)
    target_buffer = np.arange(8, dtype=np.int32)
    target = {"nested": [{"buffer": target_buffer}]}
    draft = {"nested": [{"buffer": target_buffer[2:6]}]}

    with pytest.raises(ValueError, match="mutable storage aliases Target metadata"):
        runtime.publish_metadata(inputs, draft, target_metadata=target)

    np.testing.assert_array_equal(target_buffer, np.arange(8, dtype=np.int32))
    assert _materialize(runtime, mapping_epoch=2).identity.mapping_epoch == 2


def test_store_requires_target_metadata_and_rejects_none() -> None:
    runtime, _capability, _block_tables = _runtime_fixture()
    inputs = _materialize(runtime, mapping_epoch=1)

    with pytest.raises(TypeError, match="target_metadata"):
        runtime.publish_metadata(inputs, {"draft": []})

    with pytest.raises(ValueError, match="Target metadata is required"):
        runtime.publish_metadata(
            inputs,
            {"draft": []},
            target_metadata=None,
        )

    assert _materialize(runtime, mapping_epoch=2).identity.mapping_epoch == 2


def test_store_rejects_target_alias_reachable_through_slotted_holder() -> None:
    runtime, _capability, _block_tables = _runtime_fixture()
    inputs = _materialize(runtime, mapping_epoch=1)
    shared = torch.tensor([11], dtype=torch.int32)
    target = {"holder": _SlottedTensorHolder(shared)}
    draft = {"holder": _SlottedTensorHolder(shared)}

    with pytest.raises(ValueError, match="mutable storage aliases Target metadata"):
        runtime.publish_metadata(inputs, draft, target_metadata=target)

    assert target["holder"].tensor is shared
    assert _materialize(runtime, mapping_epoch=2).identity.mapping_epoch == 2


def test_store_isolates_target_identity_and_content_then_invalidates_on_retire() -> None:
    runtime, _capability, _block_tables = _runtime_fixture()
    inputs = _materialize(runtime, mapping_epoch=1)
    target_tensor = torch.tensor([11], dtype=torch.int32)
    target_nested = [{"tensor": target_tensor}]
    target = {"nested": target_nested}
    draft_tensor = target_tensor.clone()
    draft = {"nested": [{"tensor": draft_tensor}]}
    target_identity = id(target)
    nested_identity = id(target_nested)

    runtime.publish_metadata(inputs, draft, target_metadata=target)
    stored = runtime.read_metadata(inputs.identity)
    stored["nested"][0]["tensor"].fill_(99)

    assert id(target) == target_identity
    assert id(target["nested"]) == nested_identity
    assert target["nested"][0]["tensor"].tolist() == [11]
    with pytest.raises(RuntimeError, match="active draft metadata"):
        runtime.publish_metadata(inputs, {"other": []}, target_metadata=target)
    with pytest.raises(RuntimeError, match="no active draft metadata"):
        runtime.read_metadata(inputs.identity)


def test_store_does_not_treat_shared_immutable_enum_as_mutable_alias() -> None:
    runtime, _capability, _block_tables = _runtime_fixture()
    inputs = _materialize(runtime, mapping_epoch=1)

    runtime.publish_metadata(
        inputs,
        {"phase": _ImmutablePhase.DECODE, "draft": torch.tensor([1])},
        target_metadata={"phase": _ImmutablePhase.DECODE, "target": torch.tensor([2])},
    )

    assert runtime.read_metadata(inputs.identity)["phase"] is _ImmutablePhase.DECODE


def test_store_stale_read_and_wrong_retire_do_not_destroy_active_entry() -> None:
    runtime, _capability, _block_tables = _runtime_fixture()
    inputs = _materialize(runtime, mapping_epoch=1)
    payload = {"value": [1]}
    runtime.publish_metadata(inputs, payload, target_metadata={})
    wrong = replace(inputs.identity, mapping_epoch=2)

    with pytest.raises(RuntimeError, match="generation identity"):
        runtime.read_metadata(wrong)
    with pytest.raises(RuntimeError, match="generation identity"):
        runtime.retire_generation(wrong)

    assert runtime.read_metadata(inputs.identity) is payload
    runtime.retire_generation(inputs.identity)
    with pytest.raises(RuntimeError, match="no active draft metadata"):
        runtime.read_metadata(inputs.identity)


def test_exception_guard_cleanup_preserves_pending_reader_safety() -> None:
    runtime, _capability, _block_tables = _runtime_fixture()
    inputs = _materialize(runtime, mapping_epoch=1)
    event = _FakeCompletionToken()
    runtime.mark_reader_pending(inputs, event)

    with pytest.raises(RuntimeError, match="reader failed"), runtime.generation_guard(inputs):
        raise RuntimeError("reader failed")
    with pytest.raises(RuntimeError, match="still in use"):
        _materialize(runtime, mapping_epoch=2)

    event.complete = True
    assert _materialize(runtime, mapping_epoch=2).identity.mapping_epoch == 2


def test_generation_identity_rejects_invalid_request_contracts() -> None:
    source_identity = _runtime_fixture()[1].full_source.source_identity
    common = dict(
        generation_id=1,
        mapping_epoch=1,
        ordered_request_fingerprint=("A", "B"),
        actual_request_count=2,
        padded_request_count=2,
        normalized_method="eagle3",
        global_group_id=0,
        read_descriptor_fingerprint=(_descriptor(), _descriptor()),
        provider_id=1,
        provider_version=1,
        source_identity=source_identity,
        execution_lane=0,
        phase=DraftWindowExecutionPhase.PROPOSAL,
        step=0,
    )

    with pytest.raises(ValueError, match="actual request count"):
        DraftWindowGenerationIdentity(**(common | {"actual_request_count": 1}))
    with pytest.raises(ValueError, match="unique stable request IDs"):
        DraftWindowGenerationIdentity(**(common | {"ordered_request_fingerprint": ("A", "A")}))
    with pytest.raises(ValueError, match="padded request count"):
        DraftWindowGenerationIdentity(**(common | {"padded_request_count": 1}))
    with pytest.raises(ValueError, match="descriptor fingerprint"):
        DraftWindowGenerationIdentity(**(common | {"read_descriptor_fingerprint": (_descriptor(),)}))
