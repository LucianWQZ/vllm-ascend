# SPDX-License-Identifier: Apache-2.0
"""Real builder Host counterexamples; no device kernels or Graph emulation."""

import gc
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFADSACPMetadataBuilder


def builder():
    owner = AscendSFADSACPMetadataBuilder.__new__(AscendSFADSACPMetadataBuilder)
    owner.dsa_cp_max_num_tokens = 64
    owner.dsa_cp_metadata_buffers = {}
    owner.dsa_cp_actual_seq_lengths_query = torch.zeros(5, dtype=torch.int32)
    owner.dsa_cp_actual_seq_lengths_key = torch.zeros(5, dtype=torch.int32)
    owner.dsa_cp_spec_actual_seq_lengths_query = [torch.zeros(5, dtype=torch.int32)]
    owner.dsa_cp_spec_actual_seq_lengths_key = [torch.zeros(5, dtype=torch.int32)]
    return owner


def prepare(owner, *, rows=4, value=0, draft_index=None):
    n = rows * 9
    starts = torch.arange(rows + 1, dtype=torch.int32) * 9
    common = SimpleNamespace(num_reqs=rows, num_input_tokens=n, num_actual_tokens=n, query_start_loc=starts)
    rope = torch.full((n, 1, 1, 64), float(value))
    slots = torch.arange(n, dtype=torch.int64) + value * 100
    with patch(
        "vllm_ascend.attention.context_parallel.sfa_cp.get_tp_group",
        return_value=SimpleNamespace(world_size=16, rank_in_group=2),
    ):
        result = owner._prepare_parallel_metadata(
            common, rope, rope + 1, slots, starts[1:], torch.full((rows,), 1024, dtype=torch.int32), draft_index
        )
    return result


def test_captured_padded_inputs_keep_owner_after_capture_metadata_is_discarded():
    owner = builder()
    result = prepare(owner)
    storage = weakref.ref(result[2] if result[2]._base is None else result[2]._base)
    del result
    gc.collect()
    assert storage() is not None, "builder discarded the storage captured by native KV/RoPE kernels"


@pytest.mark.parametrize("component", [0, 1, 2])
def test_live_metadata_updates_the_same_address_as_each_captured_input(component):
    owner = builder()
    captured = prepare(owner)[component]
    current = prepare(owner, value=7)[component]
    assert captured.data_ptr() == current.data_ptr(), "capture/runtime padded input addresses differ"
    torch.testing.assert_close(captured, current)
    assert captured.flatten()[0] == (700 if component == 2 else 7 + component)


def test_bucket_switch_retains_large_capture_storage_and_clears_padding():
    owner = builder()
    large = prepare(owner)[2]
    small = prepare(owner, rows=2, value=5)[2]
    assert large.untyped_storage().data_ptr() == small.untyped_storage().data_ptr()
    assert small[:18].tolist() == list(range(500, 518))
    assert small[18:].tolist() == [-1] * 14
    large_again = prepare(owner, value=3)[2]
    assert large.data_ptr() == large_again.data_ptr()
    assert large[:36].tolist() == list(range(300, 336))
    assert large[36:].tolist() == [-1] * 12


def test_draft_iteration_has_separate_owned_bank_and_capacity_does_not_grow():
    owner = builder()
    main = prepare(owner, value=1)[2]
    step = prepare(owner, value=2, draft_index=1)[2]
    assert main.untyped_storage().data_ptr() != step.untyped_storage().data_ptr()
    assert main[:36].tolist() == list(range(100, 136))
    owner.dsa_cp_max_num_tokens = 32
    with pytest.raises(RuntimeError, match="capacity"):
        prepare(owner)


def test_nope_slots_keep_owned_address_without_inventing_rope_inputs():
    owner = builder()
    first = owner._padded_metadata_inputs(None, None, torch.arange(36, dtype=torch.int32), 48, None)
    second = owner._padded_metadata_inputs(None, None, torch.arange(18, dtype=torch.int32) + 100, 32, None)
    assert first[0] is None and first[1] is None
    assert second[0] is None and second[1] is None
    assert first[2].data_ptr() == second[2].data_ptr()
    assert second[2][:18].tolist() == list(range(100, 118))
    assert second[2][18:].tolist() == [-1] * 14


def test_changed_bank_layout_is_rejected_without_replacing_captured_storage():
    owner = builder()
    captured = prepare(owner)[2]
    original = captured.untyped_storage().data_ptr()
    with pytest.raises(RuntimeError, match="layout changed"):
        owner._padded_metadata_inputs(
            torch.zeros(36, 1, 1, 32), torch.zeros(36, 1, 1, 32), torch.arange(36, dtype=torch.int64), 48, None
        )
    assert captured.untyped_storage().data_ptr() == original


def test_outer_cp_update_preserves_captured_local_slot_address():
    owner = builder()
    _, _, _, extra = prepare(owner)
    context = extra["dsa_cp_context"]
    captured = context.slot_mapping_cp
    pointer = captured.data_ptr()
    metadata = SimpleNamespace(dsa_cp_context=context)
    owner._update_parallel_slot_mapping(metadata, torch.arange(36, dtype=torch.int64) + 100, 36)
    assert context.slot_mapping_cp is captured
    assert captured.data_ptr() == pointer
    assert captured.tolist() == [106, 107, 108]
