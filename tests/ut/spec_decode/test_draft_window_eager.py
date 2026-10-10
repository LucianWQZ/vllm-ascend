# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU/static contracts for the Ascend-only MRV2 eager Window consumer."""

import ast
import hashlib
import inspect
import textwrap
from dataclasses import dataclass, fields, replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import vllm.v1.worker.gpu.spec_decode.speculator as upstream_speculator_module
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionMetadataBuilder,
)
from vllm_ascend.worker.v2 import attn_utils as ascend_attn_utils
from vllm_ascend.worker.v2.attn_utils import (
    build_attn_metadata_wrapper,
    build_draft_attn_metadata_factory,
)
from vllm_ascend.worker.v2.spec_decode.draft_window import (
    DraftMetadataInputs,
    DraftWindowGroupInventory,
    DraftWindowInventory,
    DraftWindowLayoutCategory,
    DraftWindowSourceCapability,
    SingleGroupDirectPrefixProvider,
    build_draft_window_runtime_state,
    build_full_draft_kv_source,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_eager import (
    AscendDraftAttentionReadInputs,
    DraftWindowEagerMixin,
    DraftWindowProposalEpochSource,
    _isolate_draft_metadata_from_target,
    build_eager_window_materialization_plan,
)


def _runtime_fixture(
    *,
    window_size: int = 128,
    max_num_reqs: int = 8,
    row_width: int = 16,
) -> tuple[Any, DraftWindowSourceCapability, SimpleNamespace]:
    canonical_counts = torch.full(
        (1, max_num_reqs),
        row_width,
        dtype=torch.int32,
    )
    block_tables = SimpleNamespace(
        block_sizes=[64],
        kernel_block_sizes=[64],
        input_block_tables=[
            torch.arange(
                max_num_reqs * row_width,
                dtype=torch.int32,
            ).reshape(max_num_reqs, row_width)
        ],
        num_blocks=SimpleNamespace(
            cpu=canonical_counts,
            gpu=canonical_counts.clone(),
        ),
        cp_size=1,
        cp_rank=0,
        cp_interleave=1,
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
        causal=False,
        layout_category=DraftWindowLayoutCategory.DENSE_GQA_DIRECT_PREFIX,
        capability_failures=(),
    )
    inventory = DraftWindowInventory(
        method="dflash",
        target_model_type="qwen3",
        runtime_route="dflash",
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
            legacy_block_size=64,
            expected_future_offset=0,
        ),
    )
    return build_draft_window_runtime_state(capability), capability, block_tables


def _input_batch(
    *,
    req_ids: tuple[str, ...] = ("A", "B"),
    mapping: tuple[int, ...] = (1, 3),
    upper: tuple[int, ...] = (190, 130),
    has_prefill: bool = False,
) -> SimpleNamespace:
    num_reqs = len(req_ids)
    idx_mapping_np = np.array(mapping, dtype=np.intp)
    if has_prefill:
        prefill_len_np = np.full(num_reqs, 4, dtype=np.int32)
        num_computed_prefill_tokens_np = np.zeros(num_reqs, dtype=np.int32)
        num_scheduled_tokens = np.full(num_reqs, 4, dtype=np.int32)
    else:
        prefill_len_np = np.full(num_reqs, 4, dtype=np.int32)
        num_computed_prefill_tokens_np = prefill_len_np.copy()
        num_scheduled_tokens = np.ones(num_reqs, dtype=np.int32)
    return SimpleNamespace(
        req_ids=list(req_ids),
        num_reqs=num_reqs,
        idx_mapping_np=idx_mapping_np,
        idx_mapping=torch.from_numpy(idx_mapping_np.astype(np.int32, copy=True)),
        seq_lens_cpu_upper_bound=torch.tensor(upper, dtype=torch.int32),
        has_prefill=has_prefill,
        is_prefilling_np=np.full(num_reqs, has_prefill, dtype=np.bool_),
        prefill_len_np=prefill_len_np,
        num_computed_prefill_tokens_np=num_computed_prefill_tokens_np,
        num_scheduled_tokens=num_scheduled_tokens,
    )


def test_production_epoch_is_one_per_real_proposal_not_one_per_read() -> None:
    source = DraftWindowProposalEpochSource()
    batch = _input_batch()
    context_a = source.begin(batch)

    source.validate(context_a, batch)
    source.validate(context_a, batch)
    assert context_a.mapping_epoch == 0

    source.finish(context_a)
    context_b = source.begin(batch)
    assert context_b.mapping_epoch == 1
    source.finish(context_b)


@pytest.mark.parametrize(
    ("req_ids", "mapping"),
    [
        (("B", "A"), (3, 1)),
        (("A",), (1,)),
        (("C", "A"), (4, 1)),
        (("D", "B"), (1, 3)),
    ],
)
def test_epoch_detects_reorder_finish_add_and_same_row_replacement(
    req_ids: tuple[str, ...],
    mapping: tuple[int, ...],
) -> None:
    source = DraftWindowProposalEpochSource()
    batch = _input_batch()
    first = source.begin(batch)
    source.finish(first)

    batch.req_ids[:] = req_ids
    batch.num_reqs = len(req_ids)
    batch.idx_mapping_np = np.array(mapping, dtype=np.intp)
    batch.idx_mapping = torch.tensor(mapping, dtype=torch.int32)
    second = source.begin(batch)

    assert second.mapping_epoch == first.mapping_epoch + 1
    assert second.ordered_request_ids == req_ids
    assert second.idx_mapping_fingerprint == mapping
    with pytest.raises(RuntimeError, match="stale proposal epoch"):
        source.validate(first, batch)
    source.finish(second)


def test_epoch_detects_same_mapping_tensor_in_place_rewrite() -> None:
    source = DraftWindowProposalEpochSource()
    batch = _input_batch(req_ids=("B", "A"), mapping=(17, 93))
    pointer = batch.idx_mapping.data_ptr()
    generation_a = source.begin(batch)
    source.finish(generation_a)

    batch.idx_mapping.copy_(torch.tensor([4, 17], dtype=torch.int32))
    batch.idx_mapping_np[:] = np.array([4, 17], dtype=np.intp)
    batch.req_ids[:] = ["C", "A"]
    generation_b = source.begin(batch)

    assert batch.idx_mapping.data_ptr() == pointer
    assert generation_b.mapping_epoch == generation_a.mapping_epoch + 1
    assert generation_b.idx_mapping_fingerprint == (4, 17)
    source.finish(generation_b)


@pytest.mark.parametrize(
    (
        "full_exact",
        "full_upper",
        "expected_start",
        "expected_exact",
        "expected_columns",
    ),
    [
        (63, 64, 0, 63, 1),
        (192, 192, 64, 128, 2),
        (193, 200, 64, 129, 3),
        (300, 320, 128, 172, 3),
    ],
)
def test_eager_plan_covers_short_boundary_nonboundary_and_long_context(
    full_exact: int,
    full_upper: int,
    expected_start: int,
    expected_exact: int,
    expected_columns: int,
) -> None:
    _runtime, capability, _block_tables = _runtime_fixture()

    plan = build_eager_window_materialization_plan(
        provider=capability.provider,
        cpu_full_exact=torch.tensor([full_exact], dtype=torch.int32),
        cpu_apply_time_full_upper=torch.tensor([full_upper], dtype=torch.int32),
        cpu_valid_num_blocks=torch.tensor([16], dtype=torch.int32),
        actual_request_count=1,
        padded_request_count=1,
    )

    assert plan.read_descriptors[0].source_start_column == expected_start // 64
    assert plan.read_descriptors[0].required_columns == expected_columns
    assert plan.cpu_window_exact.tolist() == [expected_exact]
    assert plan.cpu_window_upper[0] >= plan.cpu_window_exact[0]


def test_eager_plan_fails_on_source_shortage() -> None:
    _runtime, capability, _block_tables = _runtime_fixture()

    with pytest.raises(RuntimeError, match="source shortage"):
        build_eager_window_materialization_plan(
            provider=capability.provider,
            cpu_full_exact=torch.tensor([300], dtype=torch.int32),
            cpu_apply_time_full_upper=torch.tensor([320], dtype=torch.int32),
            cpu_valid_num_blocks=torch.tensor([3], dtype=torch.int32),
            actual_request_count=1,
            padded_request_count=1,
        )


class _BaseDraftBuilder:
    def _build_draft_attn_metadata(self, *_args: Any, **kwargs: Any) -> dict[str, Any]:
        self.baseline_builder_calls += 1
        self.baseline_builder_kwargs = kwargs
        return {"baseline": True}

    def _call_ascend_draft_attn_metadata_builder(
        self,
        *,
        read_inputs: AscendDraftAttentionReadInputs,
        builder_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        self.builder_events.append("builder")
        if self.raise_in_builder:
            raise RuntimeError("synthetic builder failure")
        self.captured_read_inputs = read_inputs
        self.captured_builder_kwargs = builder_kwargs
        if self.replace_target_in_builder:
            self.model_state.attn_metadata = {"replacement.target": object()}
        return {
            "draft.layer": SimpleNamespace(
                block_tables=builder_kwargs["block_tables"][0],
                seq_lens=builder_kwargs["seq_lens"],
                seq_lens_cpu=builder_kwargs["seq_lens_cpu"],
                slot_mapping=builder_kwargs["slot_mappings"],
                positions=self.input_buffers.positions,
            )
        }


class _EagerHarness(DraftWindowEagerMixin, _BaseDraftBuilder):
    def _call_ascend_draft_attn_metadata_builder(
        self,
        *,
        read_inputs: AscendDraftAttentionReadInputs,
        builder_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        return _BaseDraftBuilder._call_ascend_draft_attn_metadata_builder(
            self,
            read_inputs=read_inputs,
            builder_kwargs=builder_kwargs,
        )


def test_scope_without_eligible_policy_state_is_a_true_noop() -> None:
    harness = _EagerHarness()
    with harness._draft_window_proposal_scope(
        _input_batch(has_prefill=True),
        dummy_run=True,
        is_profile=True,
    ):
        pass

    assert not hasattr(harness, "_draft_window_epoch_source")
    assert not hasattr(harness, "_draft_window_live_inputs")


def _harness(*, query_width: int = 4) -> tuple[_EagerHarness, SimpleNamespace]:
    runtime, capability, block_tables = _runtime_fixture()
    batch = _input_batch()
    block_tables.slot_mappings = torch.tensor(
        [[101, 102, 103, 104, 105, 106, 107, 108]],
        dtype=torch.int64,
    )
    harness = _EagerHarness()
    harness._draft_window_runtime_state = runtime
    harness._draft_window_source_capability = capability
    harness._initialize_draft_window_eager_state()
    harness.draft_attn_layer_names = {"draft.layer"}
    harness.block_tables = block_tables
    harness.input_buffers = SimpleNamespace(
        query_start_loc=torch.arange(5, dtype=torch.int32),
        seq_lens=torch.tensor([193, 129, 0, 0], dtype=torch.int32),
        positions=torch.tensor([190, 191, 192, 126, 127, 128, 0, 0]),
        dcp_local_seq_lens=None,
    )
    harness.num_query_per_req = query_width
    harness.arange = torch.arange(9, dtype=torch.int32)
    harness.max_model_len = 4096
    harness.attn_groups = [[]]
    harness.kv_cache_config = object()
    harness.model_state = SimpleNamespace(
        attn_metadata={"target.layer": SimpleNamespace(slot_mapping=torch.tensor([7, 8]))}
    )
    harness.input_batch = batch
    harness.builder_events = []
    harness.raise_in_builder = False
    harness.replace_target_in_builder = False
    harness.baseline_builder_calls = 0
    harness.baseline_builder_kwargs = None
    harness.captured_read_inputs = None
    harness.captured_builder_kwargs = None
    return harness, batch


def test_single_validation_boundary_runs_before_real_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness, batch = _harness()
    original_validate = DraftMetadataInputs.validate_for

    def record_validate(self, expected_identity):
        harness.builder_events.append("validate")
        return original_validate(self, expected_identity)

    monkeypatch.setattr(DraftMetadataInputs, "validate_for", record_validate)
    with harness._draft_window_proposal_scope(
        batch,
        dummy_run=False,
        is_profile=False,
    ):
        metadata = harness._build_draft_attn_metadata(
            num_reqs=2,
            num_reqs_padded=4,
            num_tokens_padded=8,
            seq_lens_cpu_upper_bound=batch.seq_lens_cpu_upper_bound,
            step=4,
            causal=False,
        )
        assert metadata is harness._draft_window_runtime_state.read_metadata(harness._draft_window_live_inputs.identity)

    builder_index = harness.builder_events.index("builder")
    assert harness.builder_events[:builder_index] == ["validate"]
    assert isinstance(harness.captured_read_inputs, AscendDraftAttentionReadInputs)
    assert not hasattr(harness.captured_read_inputs, "slot_mapping")
    assert harness._draft_window_runtime_state.workspace.bound_identity is None


def test_stale_proposal_cannot_reach_builder() -> None:
    harness, batch = _harness()

    with (
        pytest.raises(RuntimeError, match="proposal.*changed"),
        harness._draft_window_proposal_scope(
            batch,
            dummy_run=False,
            is_profile=False,
        ),
    ):
        batch.idx_mapping_np[:] = np.array([3, 1], dtype=np.intp)
        harness._build_draft_attn_metadata(
            num_reqs=2,
            num_reqs_padded=4,
            num_tokens_padded=8,
            seq_lens_cpu_upper_bound=batch.seq_lens_cpu_upper_bound,
            step=4,
            causal=False,
        )

    assert "builder" not in harness.builder_events


def test_builder_exception_aborts_generation_and_allows_next_proposal() -> None:
    harness, batch = _harness()
    harness.raise_in_builder = True

    with (
        pytest.raises(RuntimeError, match="synthetic builder failure"),
        harness._draft_window_proposal_scope(
            batch,
            dummy_run=False,
            is_profile=False,
        ),
    ):
        harness._build_draft_attn_metadata(
            num_reqs=2,
            num_reqs_padded=4,
            num_tokens_padded=8,
            seq_lens_cpu_upper_bound=batch.seq_lens_cpu_upper_bound,
            step=4,
            causal=False,
        )

    assert harness._draft_window_runtime_state.workspace.bound_identity is None
    assert harness._draft_window_live_inputs is None

    harness.raise_in_builder = False
    with harness._draft_window_proposal_scope(
        batch,
        dummy_run=False,
        is_profile=False,
    ):
        assert harness._build_draft_attn_metadata(
            num_reqs=2,
            num_reqs_padded=4,
            num_tokens_padded=8,
            seq_lens_cpu_upper_bound=batch.seq_lens_cpu_upper_bound,
            step=4,
            causal=False,
        )


def test_materialization_takes_one_controlled_full_exact_snapshot() -> None:
    harness, batch = _harness()
    harness._snapshot_prepared_draft_seq_lens = MagicMock(
        return_value=torch.tensor([193, 129, 0, 0], dtype=torch.int32)
    )

    with harness._draft_window_proposal_scope(
        batch,
        dummy_run=False,
        is_profile=False,
    ):
        harness._build_draft_attn_metadata(
            num_reqs=2,
            num_reqs_padded=4,
            num_tokens_padded=8,
            seq_lens_cpu_upper_bound=batch.seq_lens_cpu_upper_bound,
            step=4,
            causal=False,
        )

    harness._snapshot_prepared_draft_seq_lens.assert_called_once_with(4)


def test_production_path_changes_only_draft_read_visibility_and_preserves_target() -> None:
    harness, batch = _harness()
    full_table_before = harness.block_tables.input_block_tables[0].clone()
    slot_mapping = harness.block_tables.slot_mappings
    positions = harness.input_buffers.positions
    target_nested = {"slot_mapping": slot_mapping, "values": [torch.tensor([7, 8])]}
    harness.model_state.attn_metadata = target_nested
    target_identity = id(harness.model_state.attn_metadata)
    target_values = target_nested["values"][0].clone()

    with harness._draft_window_proposal_scope(
        batch,
        dummy_run=False,
        is_profile=False,
    ):
        metadata = harness._build_draft_attn_metadata(
            num_reqs=2,
            num_reqs_padded=4,
            num_tokens_padded=8,
            seq_lens_cpu_upper_bound=batch.seq_lens_cpu_upper_bound,
            step=4,
            causal=False,
        )
        draft = metadata["draft.layer"]
        assert draft.block_tables.data_ptr() != harness.block_tables.input_block_tables[0].data_ptr()
        torch.testing.assert_close(draft.slot_mapping, slot_mapping)
        assert draft.slot_mapping.data_ptr() != slot_mapping.data_ptr()
        assert draft.positions is positions

        assert id(harness.model_state.attn_metadata) == target_identity
        assert harness.model_state.attn_metadata["slot_mapping"] is slot_mapping
        torch.testing.assert_close(harness.model_state.attn_metadata["values"][0], target_values)
        torch.testing.assert_close(harness.block_tables.input_block_tables[0], full_table_before)


@pytest.mark.parametrize(
    ("_route", "query_width", "expected_upper"),
    [
        pytest.param("dspark-default-anchor", 3, [129, 133, 0, 0], id="dspark-default-K"),
        pytest.param("dspark-explicit-no-anchor", 4, [130, 134, 0, 0], id="dspark-false-K-plus-1"),
        pytest.param("dflash", 4, [130, 134, 0, 0], id="dflash-K-plus-1"),
    ],
)
def test_apply_time_upper_uses_actual_dflash_or_dspark_query_width(
    _route: str,
    query_width: int,
    expected_upper: list[int],
) -> None:
    harness, batch = _harness(query_width=query_width)

    with harness._draft_window_proposal_scope(
        batch,
        dummy_run=False,
        is_profile=False,
    ):
        harness._build_draft_attn_metadata(
            num_reqs=2,
            num_reqs_padded=4,
            num_tokens_padded=8,
            seq_lens_cpu_upper_bound=batch.seq_lens_cpu_upper_bound,
            step=query_width,
            causal=False,
        )
        assert harness.captured_read_inputs.seq_lens_cpu_upper_bound.tolist() == expected_upper


def test_terminal_pure_prefill_uses_baseline_then_decode_uses_window() -> None:
    harness, batch = _harness()
    batch = _input_batch(has_prefill=True)
    harness.input_batch = batch

    with harness._draft_window_proposal_scope(
        batch,
        dummy_run=False,
        is_profile=False,
    ):
        result = harness._build_draft_attn_metadata(
            num_reqs=2,
            num_reqs_padded=4,
            num_tokens_padded=8,
            seq_lens_cpu_upper_bound=batch.seq_lens_cpu_upper_bound,
            step=4,
            causal=False,
        )

    assert result == {"baseline": True}
    assert harness.baseline_builder_calls == 1
    assert harness.builder_events == []
    assert harness._draft_window_epoch_source._next_epoch == 0
    assert harness._draft_window_proposal_context is None
    assert harness._draft_window_live_inputs is None
    assert harness._draft_window_runtime_state.workspace.bound_identity is None

    batch.has_prefill = False
    batch.is_prefilling_np.fill(False)
    batch.num_computed_prefill_tokens_np[:] = batch.prefill_len_np
    with harness._draft_window_proposal_scope(
        batch,
        dummy_run=False,
        is_profile=False,
    ):
        metadata = harness._build_draft_attn_metadata(
            num_reqs=2,
            num_reqs_padded=4,
            num_tokens_padded=8,
            seq_lens_cpu_upper_bound=batch.seq_lens_cpu_upper_bound,
            step=4,
            causal=False,
        )

    assert "draft.layer" in metadata
    assert harness.builder_events == ["builder"]
    assert harness._draft_window_epoch_source._next_epoch == 1
    assert harness._draft_window_runtime_state.workspace.bound_identity is None


def test_mixed_prefill_decode_fails_before_materialization() -> None:
    harness, batch = _harness()
    batch.has_prefill = True
    batch.is_prefilling_np[:] = np.array([True, False])
    batch.num_computed_prefill_tokens_np[:] = np.array([0, 4])

    with (
        pytest.raises(NotImplementedError, match="mixed prefill/decode"),
        harness._draft_window_proposal_scope(
            batch,
            dummy_run=False,
            is_profile=False,
        ),
    ):
        pass

    assert harness._draft_window_runtime_state.workspace.bound_identity is None


def test_chunked_prefill_fails_before_materialization() -> None:
    harness, batch = _harness()
    batch.has_prefill = True
    batch.is_prefilling_np.fill(True)
    batch.prefill_len_np[:] = np.array([8, 4])
    batch.num_computed_prefill_tokens_np.fill(0)
    batch.num_scheduled_tokens[:] = np.array([4, 4])

    with (
        pytest.raises(NotImplementedError, match="chunked prefill"),
        harness._draft_window_proposal_scope(
            batch,
            dummy_run=False,
            is_profile=False,
        ),
    ):
        pass

    assert harness._draft_window_epoch_source._next_epoch == 0
    assert harness._draft_window_runtime_state.workspace.bound_identity is None


def test_metadata_isolation_clones_nested_target_aliases_without_mutating_target() -> None:
    shared = torch.tensor([11, 12], dtype=torch.int32)
    target_nested = [{"tensor": shared}]
    target = {"layer": SimpleNamespace(slot_mapping=shared, nested=target_nested)}
    draft = {
        "layer": SimpleNamespace(
            slot_mapping=shared[:],
            nested=[{"tensor": shared}],
            private=torch.tensor([9], dtype=torch.int32),
        )
    }

    isolated = _isolate_draft_metadata_from_target(draft, target)

    assert isolated is draft
    assert draft["layer"].slot_mapping.data_ptr() != shared.data_ptr()
    assert draft["layer"].nested[0]["tensor"].data_ptr() != shared.data_ptr()
    assert target["layer"].slot_mapping is shared
    assert target["layer"].nested is target_nested
    assert target["layer"].nested[0]["tensor"].tolist() == [11, 12]


def test_metadata_isolation_clones_numpy_views_that_alias_target_storage() -> None:
    target_buffer = np.arange(8, dtype=np.int32)
    target = {"nested": [{"buffer": target_buffer}]}
    draft = {"nested": [{"buffer": target_buffer[2:6]}]}

    isolated = _isolate_draft_metadata_from_target(draft, target)

    assert not np.shares_memory(isolated["nested"][0]["buffer"], target_buffer)
    isolated["nested"][0]["buffer"].fill(99)
    np.testing.assert_array_equal(target_buffer, np.arange(8, dtype=np.int32))


@dataclass
class _MutableMetadata:
    tensor: torch.Tensor
    nested: list[torch.Tensor]


def test_metadata_isolation_does_not_mutate_shared_mutable_dataclass() -> None:
    target_tensor = torch.tensor([21, 22], dtype=torch.int32)
    shared_metadata = _MutableMetadata(
        tensor=target_tensor,
        nested=[target_tensor],
    )
    target = {"layer": shared_metadata}
    draft = {"layer": shared_metadata}

    isolated = _isolate_draft_metadata_from_target(draft, target)

    assert isolated["layer"] is not shared_metadata
    assert isolated["layer"].tensor.data_ptr() != target_tensor.data_ptr()
    assert isolated["layer"].nested[0].data_ptr() != target_tensor.data_ptr()
    assert target["layer"] is shared_metadata
    assert target["layer"].tensor is target_tensor
    assert target["layer"].nested[0] is target_tensor


def _upstream_method_ast_sha256(function: Any) -> str:
    source = textwrap.dedent(inspect.getsource(function))
    normalized = ast.dump(ast.parse(source), include_attributes=False)
    return hashlib.sha256(normalized.encode()).hexdigest()


def test_unmodified_upstream_baseline_has_no_prototype_api_and_matches_drift_guard() -> None:
    assert not hasattr(upstream_speculator_module, "DraftAttentionReadInputs")
    assert _upstream_method_ast_sha256(DraftModelSpeculator._build_draft_attn_metadata) == (
        "91099e775ec44ef5520ca8231fae89ff9a75d2f7386b32d3104774b2ecab8654"
    )
    assert _upstream_method_ast_sha256(DFlashSpeculator._build_draft_attn_metadata) == (
        "0e2df9e2c57ce3d28ea1a311d4956e3a974eec76b3b518cfe3c99292f1a9ad4b"
    )
    assert tuple(inspect.signature(DraftModelSpeculator._build_draft_attn_metadata).parameters) == (
        "self",
        "num_reqs",
        "num_reqs_padded",
        "num_tokens_padded",
        "seq_lens_cpu_upper_bound",
        "step",
        "num_query_per_req",
        "causal",
        "query_start_loc_np",
        "dcp_local_seq_lens",
    )


def test_ascend_read_transport_cannot_override_write_or_query_state() -> None:
    assert {field.name for field in fields(AscendDraftAttentionReadInputs)} == {
        "block_tables",
        "seq_lens",
        "seq_lens_cpu_exact",
        "seq_lens_cpu_upper_bound",
        "max_seq_len",
        "dcp_local_seq_lens",
    }


class _OffParityHarness(DraftWindowEagerMixin, DFlashSpeculator):
    pass


def _off_parity_harness() -> _OffParityHarness:
    harness = object.__new__(_OffParityHarness)
    harness.draft_attn_layer_names = {"draft.layer"}
    harness.num_query_per_req = 4
    harness.arange = torch.arange(9, dtype=torch.int32)
    harness.max_model_len = 4096
    harness.draft_max_seq_len = 4096
    harness.attn_groups = [[]]
    harness.kv_cache_config = object()
    harness.block_tables = SimpleNamespace(
        input_block_tables=(torch.arange(32, dtype=torch.int32).reshape(4, 8),),
        slot_mappings=torch.arange(8, dtype=torch.int64).reshape(1, 8),
        cp_size=1,
        cp_rank=0,
        cp_interleave=1,
    )
    harness.input_buffers = SimpleNamespace(
        query_start_loc=torch.arange(5, dtype=torch.int32),
        seq_lens=torch.tensor([193, 129, 0, 0], dtype=torch.int32),
        dcp_local_seq_lens=None,
    )
    return harness


def test_window_on_adapter_uses_existing_ascend_wrapper_with_explicit_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _off_parity_harness()
    captured: list[dict[str, Any]] = []

    def capture(**kwargs: Any) -> dict[str, Any]:
        captured.append(kwargs)
        return kwargs

    monkeypatch.setattr(ascend_attn_utils, "build_attn_metadata", capture)
    positions = torch.arange(8, dtype=torch.int64)
    is_prefilling = torch.tensor([False, False])
    cpu_exact = torch.tensor([96, 64, 0, 0], dtype=torch.int32)
    cpu_upper = torch.tensor([112, 64, 0, 0], dtype=torch.int32)
    read_inputs = AscendDraftAttentionReadInputs(
        block_tables=(torch.zeros((4, 2), dtype=torch.int32),),
        seq_lens=cpu_exact.clone(),
        seq_lens_cpu_exact=cpu_exact,
        seq_lens_cpu_upper_bound=cpu_upper,
        max_seq_len=112,
    )
    builder_kwargs = {
        "seq_lens": read_inputs.seq_lens,
        "seq_lens_cpu": cpu_exact,
        "seq_lens_cpu_upper_bound": cpu_upper,
        "block_tables": list(read_inputs.block_tables),
        "slot_mappings": harness.block_tables.slot_mappings,
    }

    with (
        build_attn_metadata_wrapper(),
        build_draft_attn_metadata_factory(positions, 8, is_prefilling),
    ):
        result = harness._call_ascend_draft_attn_metadata_builder(
            read_inputs=read_inputs,
            builder_kwargs=builder_kwargs,
        )

    assert result is captured[0]
    assert captured[0]["seq_lens_cpu"] is cpu_exact
    assert captured[0]["seq_lens_cpu_upper_bound"] is cpu_upper
    assert captured[0]["positions"].data_ptr() == positions.data_ptr()
    assert captured[0]["is_prefilling"] is is_prefilling
    assert captured[0]["slot_mappings"] is harness.block_tables.slot_mappings


def _capture_off_builder_kwargs(
    monkeypatch: pytest.MonkeyPatch,
    harness: _OffParityHarness,
    *,
    through_ascend_override: bool,
    positions: torch.Tensor,
    is_prefilling: torch.Tensor,
) -> dict[str, Any]:
    captured: list[dict[str, Any]] = []

    def capture(**kwargs: Any) -> dict[str, Any]:
        captured.append(kwargs)
        return kwargs

    monkeypatch.setattr(ascend_attn_utils, "build_attn_metadata", capture)
    call = (
        harness._build_draft_attn_metadata
        if through_ascend_override
        else lambda **kwargs: DFlashSpeculator._build_draft_attn_metadata(harness, **kwargs)
    )
    with (
        build_attn_metadata_wrapper(),
        build_draft_attn_metadata_factory(positions, 8, is_prefilling),
    ):
        call(
            num_reqs=2,
            num_reqs_padded=4,
            num_tokens_padded=8,
            seq_lens_cpu_upper_bound=torch.tensor([190, 130], dtype=torch.int32),
            step=4,
            causal=False,
        )
    assert len(captured) == 1
    return captured[0]


def test_window_off_uses_exact_upstream_super_path_with_full_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _off_parity_harness()
    positions = torch.arange(8, dtype=torch.int64)
    is_prefilling = torch.tensor([False, False])
    baseline = _capture_off_builder_kwargs(
        monkeypatch,
        harness,
        through_ascend_override=False,
        positions=positions,
        is_prefilling=is_prefilling,
    )
    ascend_off = _capture_off_builder_kwargs(
        monkeypatch,
        harness,
        through_ascend_override=True,
        positions=positions,
        is_prefilling=is_prefilling,
    )

    assert baseline.keys() == ascend_off.keys()
    for name in ("max_query_len", "max_seq_len", "causal", "kv_cache_config"):
        assert baseline[name] == ascend_off[name]
    for name in (
        "query_start_loc_gpu",
        "seq_lens",
        "slot_mappings",
        "positions",
        "is_prefilling",
    ):
        assert baseline[name].data_ptr() == ascend_off[name].data_ptr()
        torch.testing.assert_close(baseline[name], ascend_off[name])
    assert baseline["block_tables"][0].data_ptr() == ascend_off["block_tables"][0].data_ptr()
    torch.testing.assert_close(
        baseline["block_tables"][0],
        ascend_off["block_tables"][0],
    )
    torch.testing.assert_close(
        baseline["query_start_loc_cpu"],
        ascend_off["query_start_loc_cpu"],
    )
    torch.testing.assert_close(
        baseline["seq_lens_cpu_upper_bound"],
        ascend_off["seq_lens_cpu_upper_bound"],
    )


def test_window_on_vendor_assembly_matches_upstream_for_full_equivalent_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _off_parity_harness()
    positions = torch.arange(8, dtype=torch.int64)
    is_prefilling = torch.tensor([False, False])
    baseline = _capture_off_builder_kwargs(
        monkeypatch,
        harness,
        through_ascend_override=False,
        positions=positions,
        is_prefilling=is_prefilling,
    )
    captured: list[dict[str, Any]] = []

    def capture(**kwargs: Any) -> dict[str, Any]:
        captured.append(kwargs)
        return kwargs

    monkeypatch.setattr(ascend_attn_utils, "build_attn_metadata", capture)
    cpu_exact = harness.input_buffers.seq_lens.clone()
    read_inputs = AscendDraftAttentionReadInputs(
        block_tables=tuple(table[:4] for table in harness.block_tables.input_block_tables),
        seq_lens=harness.input_buffers.seq_lens,
        seq_lens_cpu_exact=cpu_exact,
        seq_lens_cpu_upper_bound=torch.tensor([194, 134, 0, 0], dtype=torch.int32),
        max_seq_len=harness.draft_max_seq_len,
    )
    with (
        build_attn_metadata_wrapper(),
        build_draft_attn_metadata_factory(positions, 8, is_prefilling),
    ):
        harness._build_ascend_draft_attn_metadata(
            num_reqs=2,
            num_reqs_padded=4,
            num_tokens_padded=8,
            read_inputs=read_inputs,
            causal=False,
            query_start_loc_np=None,
            dcp_local_seq_lens=None,
        )

    assert len(captured) == 1
    vendor = captured[0]
    assert set(vendor) == set(baseline) | {"seq_lens_cpu"}
    for name in ("attn_groups", "num_reqs", "num_tokens", "max_query_len", "max_seq_len", "causal"):
        assert vendor[name] == baseline[name]
    assert vendor["kv_cache_config"] is baseline["kv_cache_config"]
    for name in (
        "query_start_loc_gpu",
        "seq_lens",
        "slot_mappings",
        "positions",
        "is_prefilling",
    ):
        assert vendor[name].data_ptr() == baseline[name].data_ptr()
        torch.testing.assert_close(vendor[name], baseline[name])
    torch.testing.assert_close(vendor["query_start_loc_cpu"], baseline["query_start_loc_cpu"])
    torch.testing.assert_close(vendor["seq_lens_cpu_upper_bound"], baseline["seq_lens_cpu_upper_bound"])
    assert vendor["block_tables"][0].data_ptr() == baseline["block_tables"][0].data_ptr()
    torch.testing.assert_close(vendor["block_tables"][0], baseline["block_tables"][0])
    assert vendor["seq_lens_cpu"] is cpu_exact


def test_window_on_vendor_assembly_rejects_nonuniform_query_layout() -> None:
    harness = _off_parity_harness()
    read_inputs = AscendDraftAttentionReadInputs(
        block_tables=tuple(table[:4] for table in harness.block_tables.input_block_tables),
        seq_lens=harness.input_buffers.seq_lens,
        seq_lens_cpu_exact=harness.input_buffers.seq_lens.clone(),
        seq_lens_cpu_upper_bound=torch.tensor([194, 134, 0, 0], dtype=torch.int32),
        max_seq_len=harness.draft_max_seq_len,
    )
    harness._call_ascend_draft_attn_metadata_builder = MagicMock(return_value={})

    with pytest.raises(NotImplementedError, match="uniform DFlash/DSpark query layout"):
        harness._build_ascend_draft_attn_metadata(
            num_reqs=2,
            num_reqs_padded=4,
            num_tokens_padded=8,
            read_inputs=read_inputs,
            causal=False,
            query_start_loc_np=np.array([0, 1, 3], dtype=np.int32),
            dcp_local_seq_lens=None,
        )

    harness._call_ascend_draft_attn_metadata_builder.assert_not_called()


def test_eager_module_has_no_hidden_global_or_contextvar_transport() -> None:
    import vllm_ascend.worker.v2.spec_decode.draft_window_eager as eager_module

    source = inspect.getsource(eager_module)
    tree = ast.parse(source)
    assert not any(isinstance(node, ast.Global) for node in ast.walk(tree))
    assert not any(isinstance(node, ast.Name) and node.id == "ContextVar" for node in ast.walk(tree))


@pytest.mark.parametrize(
    ("dummy_run", "is_profile"),
    [(True, False), (False, True), (True, True)],
)
def test_profile_and_dummy_use_baseline_without_production_window_state(
    dummy_run: bool,
    is_profile: bool,
) -> None:
    harness, batch = _harness()

    with harness._draft_window_proposal_scope(
        batch,
        dummy_run=dummy_run,
        is_profile=is_profile,
    ):
        result = harness._build_draft_attn_metadata(
            num_reqs=2,
            num_reqs_padded=4,
            num_tokens_padded=8,
            seq_lens_cpu_upper_bound=batch.seq_lens_cpu_upper_bound,
            step=4,
            causal=False,
        )

    assert result == {"baseline": True}
    assert harness.baseline_builder_calls == 1
    assert harness.builder_events == []
    assert harness._draft_window_epoch_source._next_epoch == 0
    assert harness._draft_window_proposal_context is None
    assert harness._draft_window_live_inputs is None
    assert harness._draft_window_runtime_state.workspace.bound_identity is None
    with pytest.raises(RuntimeError, match="no active draft metadata"):
        harness._draft_window_runtime_state.read_metadata(SimpleNamespace())


@pytest.mark.parametrize("invalid_target", [None, {}])
def test_window_on_requires_nonempty_real_target_before_builder(
    invalid_target: Any,
) -> None:
    harness, batch = _harness()
    harness.model_state.attn_metadata = invalid_target

    with (
        pytest.raises(ValueError, match="non-empty Target metadata"),
        harness._draft_window_proposal_scope(
            batch,
            dummy_run=False,
            is_profile=False,
        ),
    ):
        harness._build_draft_attn_metadata(
            num_reqs=2,
            num_reqs_padded=4,
            num_tokens_padded=8,
            seq_lens_cpu_upper_bound=batch.seq_lens_cpu_upper_bound,
            step=4,
            causal=False,
        )

    assert harness.builder_events == []
    assert harness._draft_window_runtime_state.workspace.bound_identity is None
    assert harness._draft_window_live_inputs is None


def test_window_on_rejects_target_owner_replacement_during_builder() -> None:
    harness, batch = _harness()
    harness.replace_target_in_builder = True

    with (
        pytest.raises(RuntimeError, match="Target metadata owner changed"),
        harness._draft_window_proposal_scope(
            batch,
            dummy_run=False,
            is_profile=False,
        ),
    ):
        harness._build_draft_attn_metadata(
            num_reqs=2,
            num_reqs_padded=4,
            num_tokens_padded=8,
            seq_lens_cpu_upper_bound=batch.seq_lens_cpu_upper_bound,
            step=4,
            causal=False,
        )

    assert harness._draft_window_runtime_state.workspace.bound_identity is None
    assert harness._draft_window_live_inputs is None


def test_adapter_rejects_wrong_generation_before_constructing_upstream_inputs() -> None:
    harness, batch = _harness()
    with harness._draft_window_proposal_scope(
        batch,
        dummy_run=False,
        is_profile=False,
    ):
        inputs = harness._materialize_draft_window_read_inputs(
            num_reqs=2,
            num_reqs_padded=4,
        )
        harness._draft_window_live_inputs = inputs
        wrong = replace(inputs.identity, mapping_epoch=inputs.identity.mapping_epoch + 1)

        with pytest.raises(RuntimeError, match="generation identity"):
            harness._validated_draft_attention_read_inputs(inputs, wrong)
