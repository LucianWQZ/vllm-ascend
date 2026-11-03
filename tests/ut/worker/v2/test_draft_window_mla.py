# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ordinary MLA builders feed the current forward before the next proposal.

These CPU checks exercise actual metadata and forward-context interfaces. They
do not simulate asynchronous device completion or qualify a native kernel.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm import forward_context
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

from vllm_ascend.attention.mla_v1 import AscendMLAMetadata, AscendMLAMetadataBuilder
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.ops import rotary_embedding
from vllm_ascend.worker.v2.spec_decode.draft_window_read import materialize_draft_read


@pytest.mark.parametrize("groups", [1, 5])
@pytest.mark.parametrize("query_width", [1, 8])
@pytest.mark.parametrize("interleaved", [False, True])
def test_actual_builders_forward_then_next_proposal(monkeypatch, groups, query_width, interleaved):
    # Both official cache registration layouts include offsets and strided
    # views. No RoPE producer or metadata-builder method is replaced.
    cache = torch.arange(256 * 4, dtype=torch.float32).reshape(256, 4)
    flat = torch.stack((cache, -cache), dim=1).reshape(-1, 4)
    cos_cache, sin_cache = (flat[::2], flat[1::2]) if interleaved else (cache, -cache)
    cos_buffer = torch.empty((18, 1, 1, 4))[1:17]
    sin_buffer = torch.empty_like(cos_buffer)
    for name, value in {
        "_cos_cache": cos_cache,
        "_sin_cache": sin_cache,
        "_cos_sin_flat": flat if interleaved else None,
        "_cos_mla": cos_buffer,
        "_sin_mla": sin_buffer,
    }.items():
        monkeypatch.setattr(rotary_embedding, name, value)

    builders = []
    for _ in range(groups):
        builder = object.__new__(AscendMLAMetadataBuilder)
        builder.flashmla_state, builder.use_pcp, builder.use_dcp = None, False, False
        builder.decode_threshold = 8
        builder.metadata_cls = AscendMLAMetadata
        builder.model_config = SimpleNamespace(get_head_size=lambda: 128)
        builder.attn_mask_builder = SimpleNamespace(get_splitfuse_attn_mask=lambda: None)
        builder.use_mla_rope, builder.nope_zero_rope_cache = True, None
        builders.append(builder)

    # Only platform communication extras are omitted at this CPU boundary.
    # The fixed _run_model still installs/restores ForwardContext and invokes
    # its model normally.
    monkeypatch.setattr(forward_context.current_platform, "set_additional_forward_context", lambda **kwargs: {})
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=1, use_sequence_parallel_moe=False),
        compilation_config=SimpleNamespace(fast_moe_cold_start=False, static_forward_context={}),
    )
    full_table = torch.arange(80, dtype=torch.int32).reshape(1, 80)
    full_snapshot = full_table.clone()
    target_lengths, target_table = torch.tensor([201]), full_table.clone() + 100
    target_snapshots = target_lengths.clone(), target_table.clone()
    observations: list[tuple[torch.Tensor, torch.Tensor]] = []
    owner = SimpleNamespace(vllm_config=config)

    def model(*, input_ids, positions, inputs_embeds):
        context = forward_context.get_forward_context()
        assert inputs_embeds is None
        assert context.attn_metadata is metadata
        assert context.is_padding.tolist() == [False] * query_width
        for value in metadata.values():
            decode = value.decode
            torch.testing.assert_close(decode.cos[:, 0, 0], cos_cache[positions], rtol=0, atol=0)
            torch.testing.assert_close(decode.sin[:, 0, 0], sin_cache[positions], rtol=0, atol=0)
            assert decode.cos.untyped_storage() is cos_buffer.untyped_storage()
            assert value.slot_mapping.untyped_storage() is slots.untyped_storage()
            assert value.query_start_loc is common.query_start_loc
            assert decode.input_positions.tolist() == positions.tolist()
            assert value.seq_lens.tolist() == [int(exact[0]) - start]
            assert context.slot_mapping["draft"] is slots
        observations.append((positions.clone(), metadata["draft.0"].decode.cos.clone()))
        return input_ids[:, None].float()

    owner.model = model
    for first_position in (128, 139, 128):
        # The actual five layers receive identical positions in one forward,
        # including when they have separate cache-group builders.
        position_storage = torch.empty(query_width * 2 + 1, dtype=torch.long)
        positions = position_storage[1::2]
        positions.copy_(torch.arange(first_position, first_position + query_width))
        exact = torch.tensor([first_position + query_width], dtype=torch.int32)
        start = (first_position - 8) // 4 * 4
        slots = torch.arange(query_width, dtype=torch.long) + 1000 + first_position
        common = AscendCommonAttentionMetadata(
            num_reqs=1,
            seq_lens=exact,
            seq_lens_cpu_upper_bound=exact + 7,
            block_table_tensor=full_table,
            max_seq_len=int(exact[0]) + 7,
            slot_mapping=slots,
            positions=positions,
            query_start_loc_cpu=torch.tensor([0, query_width]),
            query_start_loc=torch.tensor([0, query_width]),
            pcp_context=None,
            pcp_cache_group_idx=None,
            context_parallel_metadata=None,
            num_actual_tokens=query_width,
            num_input_tokens=query_width,
            max_query_len=query_width,
            graph_pad_size=-1,
            attn_state=object(),
            causal=False,
        )
        retained: list[object] = []
        materialize_draft_read(
            common,
            (40,),
            4,
            exact_cpu=exact,
            starts_cpu=torch.tensor([start]),
            retain=lambda *items, retained=retained: retained.extend(items),
        )
        results = [builder.build(0, common) for builder in builders]
        metadata = {f"draft.{index}": results[index % groups] for index in range(5)}
        owner.input_buffers = SimpleNamespace(
            input_ids=torch.arange(query_width),
            positions=positions,
            is_padding=torch.zeros(query_width, dtype=torch.bool),
        )
        output = DFlashSpeculator._run_model(owner, query_width, metadata, {"draft": slots}, None)
        torch.testing.assert_close(output[:, 0], torch.arange(query_width).float(), rtol=0, atol=0)
        assert not forward_context.is_forward_context_available()
        torch.testing.assert_close(full_table, full_snapshot, rtol=0, atol=0)
        torch.testing.assert_close(target_lengths, target_snapshots[0], rtol=0, atol=0)
        torch.testing.assert_close(target_table, target_snapshots[1], rtol=0, atol=0)
    # Later builds overwrite shared storage. Only this completed CPU observer's
    # results are snapshots; there is no production snapshot layer.
    assert not torch.equal(observations[0][1], observations[1][1])
    torch.testing.assert_close(observations[0][1], observations[2][1], rtol=0, atol=0)
