# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Device read projection oracle. This is not a model/serving coverage test."""

import math

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from vllm_ascend.attention.utils import AscendCommonAttentionMetadata  # noqa: E402
from vllm_ascend.worker.v2.spec_decode.draft_window_plan import compute_draft_window_start  # noqa: E402
from vllm_ascend.worker.v2.spec_decode.draft_window_read import (  # noqa: E402
    materialize_draft_read,
)

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(), reason="requires an Ascend NPU"
)


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("query_width", [1, 8, 9])
@pytest.mark.parametrize("exact_ends", [(127, 128), (129, 257), (513, 641)])
def test_same_window_attention_from_canonical_rows(causal, query_width, exact_ends):
    generator = torch.Generator().manual_seed(21021)
    block_size, heads, kv_heads, head_size = 128, 8, 2, 64
    window_size = 129
    scale = 1 / math.sqrt(head_size)
    dtype = torch.bfloat16
    columns = math.ceil(max(exact_ends) / block_size)
    # Allocate both rows independently, then share the first physical page.
    # The oracle follows logical tokens, never assumes physical ID order.
    order = torch.randperm(2 * columns, generator=generator)
    canonical_cpu = order.reshape(2, columns).to(torch.int32)
    canonical_cpu[1, 0] = canonical_cpu[0, 0]
    key_cpu = torch.randn(2 * columns, block_size, kv_heads, head_size, generator=generator).to(dtype)
    value_cpu = torch.randn(key_cpu.shape, generator=generator).to(dtype)
    key, value = key_cpu.npu(), value_cpu.npu()
    canonical = canonical_cpu.npu()
    source_before = canonical.cpu().clone()
    cache_before = (key.cpu().clone(), value.cpu().clone())
    # Request reorder is resolved at materialization, not from a prior view.
    request_rows = (1, 0)
    source = (canonical[list(request_rows)], (columns, columns), block_size)
    ends = torch.tensor(exact_ends, dtype=torch.int32)
    start_values = [
        compute_draft_window_start(full_exact_end=end, window_size=window_size, block_span_tokens=block_size)
        for end in exact_ends
    ]
    starts = torch.tensor(start_values, dtype=torch.int32)
    retained: list[torch.Tensor] = []
    query_offsets = torch.tensor([0, query_width, 2 * query_width, 2 * query_width], dtype=torch.int32)
    read = AscendCommonAttentionMetadata(
        query_start_loc=query_offsets.npu(),
        query_start_loc_cpu=query_offsets,
        seq_lens=ends.npu(),
        num_reqs=3,
        num_actual_tokens=2 * query_width,
        max_query_len=query_width,
        max_seq_len=int((ends + 3).max()),
        block_table_tensor=source[0],
        slot_mapping=torch.arange(2 * query_width, dtype=torch.int64, device="npu"),
        seq_lens_cpu_upper_bound=ends + 3,
        causal=causal,
    )
    materialize_draft_read(
        read,
        source[1],
        source[2],
        exact_cpu=ends,
        starts_cpu=starts,
        retain=lambda *objects: retained.extend(objects),
    )
    torch.testing.assert_close(read.seq_lens.cpu(), read.seq_lens_cpu, rtol=0, atol=0)
    # Query values correspond to the same absolute positions for both paths.
    positions = torch.cat([torch.arange(end - query_width, end) for end in exact_ends])
    query_cpu = torch.randn(2 * query_width, heads, head_size, generator=generator).to(dtype)
    query = query_cpu.npu()
    query_before = query.cpu().clone()
    mask = torch.ones((2048, 2048), dtype=torch.bool).triu(1).npu() if causal else None
    output, _ = torch_npu.npu_fused_infer_attention_score(
        query=query,
        # Match AscendAttentionBackendImpl._get_kv_cache_view for TND.
        key=key.view(2 * columns, block_size, -1),
        value=value.view(2 * columns, block_size, -1),
        block_table=read.block_table_tensor[:2],
        block_size=block_size,
        input_layout="TND",
        actual_seq_lengths=[query_width, 2 * query_width],
        actual_seq_lengths_kv=read.seq_lens_cpu[:2].tolist(),
        num_key_value_heads=kv_heads,
        num_heads=heads,
        scale=scale,
        sparse_mode=3 if causal else 0,
        atten_mask=mask,
    )
    expected = []
    for row, (source_row, end, start) in enumerate(zip(request_rows, exact_ends, start_values)):
        ids = canonical_cpu[source_row].long()
        logical_key = key_cpu[ids].flatten(0, 1)[start:end].float()
        logical_value = value_cpu[ids].flatten(0, 1)[start:end].float()
        logical_key = logical_key.repeat_interleave(heads // kv_heads, dim=1).transpose(0, 1)
        logical_value = logical_value.repeat_interleave(heads // kv_heads, dim=1).transpose(0, 1)
        q = query_cpu[row * query_width : (row + 1) * query_width].float().transpose(0, 1)
        scores = q @ logical_key.transpose(1, 2) * scale
        if causal:
            absolute_keys = torch.arange(start, end)
            absolute_queries = positions[row * query_width : (row + 1) * query_width]
            scores.masked_fill_(absolute_keys[None, :] > absolute_queries[:, None], -torch.inf)
        expected.append((scores.softmax(-1) @ logical_value).transpose(0, 1))
    torch.testing.assert_close(output.cpu().float(), torch.cat(expected), rtol=0.02, atol=0.02)
    torch.testing.assert_close(canonical.cpu(), source_before, rtol=0, atol=0)
    torch.testing.assert_close(key.cpu(), cache_before[0], rtol=0, atol=0)
    torch.testing.assert_close(value.cpu(), cache_before[1], rtol=0, atol=0)
    torch.testing.assert_close(query.cpu(), query_before, rtol=0, atol=0)
    assert read.seq_lens_cpu[-1] == 0
    assert read.block_table_tensor.data_ptr() != canonical.data_ptr()


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("query_width", [1, 8, 9])
def test_same_window_mla_latent_and_rope(query_width, causal):
    """Use the ordinary MLA decode ABI, including its head-major output."""
    generator = torch.Generator().manual_seed(21022)
    block_size, heads, latent_size, rope_size = 128, 8, 512, 64
    exact_ends, window_size, columns = (513, 641), 129, 6
    dtype = torch.bfloat16
    scale = 1 / math.sqrt(latent_size + rope_size)

    def random_tensor(*shape):
        return torch.randn(*shape, generator=generator).to(dtype)

    canonical_cpu = torch.randperm(2 * columns, generator=generator).reshape(2, columns).to(torch.int32)
    canonical_cpu[1, 0] = canonical_cpu[0, 0]
    latent_cpu = random_tensor(2 * columns, 1, block_size, latent_size)
    rope_cpu = random_tensor(2 * columns, 1, block_size, rope_size)
    query_cpu = random_tensor(2 * query_width, heads, latent_size)
    query_rope_cpu = random_tensor(2 * query_width, heads, rope_size)
    latent, rope, query, query_rope = (x.npu() for x in (latent_cpu, rope_cpu, query_cpu, query_rope_cpu))
    canonical = canonical_cpu.npu()
    request_rows = (1, 0)
    source = (canonical[list(request_rows)], (columns, columns), block_size)
    ends = torch.tensor(exact_ends, dtype=torch.int32)
    starts = torch.tensor(
        [
            compute_draft_window_start(full_exact_end=end, window_size=window_size, block_span_tokens=block_size)
            for end in exact_ends
        ],
        dtype=torch.int32,
    )
    retained: list[torch.Tensor] = []
    query_offsets = torch.tensor([0, query_width, 2 * query_width], dtype=torch.int32)
    read = AscendCommonAttentionMetadata(
        query_start_loc=query_offsets.npu(),
        query_start_loc_cpu=query_offsets,
        seq_lens=ends.npu(),
        num_reqs=2,
        num_actual_tokens=2 * query_width,
        max_query_len=query_width,
        max_seq_len=int((ends + 3).max()),
        block_table_tensor=source[0],
        slot_mapping=torch.arange(2 * query_width, dtype=torch.int64, device="npu"),
        seq_lens_cpu_upper_bound=ends + 3,
        causal=causal,
    )
    materialize_draft_read(
        read,
        source[1],
        source[2],
        exact_cpu=ends,
        starts_cpu=starts,
        retain=lambda *objects: retained.extend(objects),
    )
    mask = torch.ones((2048, 2048), dtype=torch.bool).triu(1).npu() if causal else None
    output, _ = torch_npu.npu_fused_infer_attention_score_v2(
        query,
        latent,
        latent,
        query_rope=query_rope,
        key_rope=rope,
        num_query_heads=heads,
        num_key_value_heads=1,
        input_layout="TND_NTD",
        atten_mask=mask,
        sparse_mode=3 if causal else 0,
        softmax_scale=scale,
        block_table=read.block_table_tensor,
        block_size=block_size,
        actual_seq_qlen=[query_width, 2 * query_width],
        actual_seq_kvlen=read.seq_lens_cpu.tolist(),
    )
    expected = []
    for row, (source_row, end, start) in enumerate(zip(request_rows, exact_ends, starts.tolist())):
        ids = canonical_cpu[source_row].long()
        keys = latent_cpu[ids].reshape(-1, latent_size)[start:end].float()
        ropes = rope_cpu[ids].reshape(-1, rope_size)[start:end].float()
        q = query_cpu[row * query_width : (row + 1) * query_width].float().transpose(0, 1)
        qr = query_rope_cpu[row * query_width : (row + 1) * query_width].float().transpose(0, 1)
        scores = (q @ keys.T + qr @ ropes.T) * scale
        if causal:
            scores.masked_fill_(
                torch.arange(start, end)[None, :] > torch.arange(end - query_width, end)[:, None], -torch.inf
            )
        expected.append(scores.softmax(-1) @ keys)
    torch.testing.assert_close(output.cpu().float(), torch.cat(expected, dim=1), rtol=0.02, atol=0.02)
    torch.testing.assert_close(read.seq_lens.cpu(), read.seq_lens_cpu, rtol=0, atol=0)
    for device, original in zip(
        (latent, rope, query, query_rope, canonical), (latent_cpu, rope_cpu, query_cpu, query_rope_cpu, canonical_cpu)
    ):
        torch.testing.assert_close(device.cpu(), original, rtol=0, atol=0)


@pytest.mark.parametrize("causal", [False, True])
def test_full_continuing_native_keeps_unused_table_columns_and_partial_pages_unread(causal):
    """Check the unchanged native Full read before removing its S=0 table copy."""
    generator = torch.Generator().manual_seed(93122)
    block, heads, kv_heads, dim, width = 128, 32, 4, 128, 8
    ends = (1151, 1281)
    scale, dtype = 1 / math.sqrt(dim), torch.bfloat16
    table_cpu = (torch.randperm(64, generator=generator) + 1).reshape(2, 32).to(torch.int32)
    keys_cpu = torch.full((65, block, kv_heads, dim), torch.nan, dtype=dtype)
    values_cpu = torch.full_like(keys_cpu, torch.nan)
    # Both the last partial page and every later logical column remain NaN.
    # The oracle below selects logical [0,E), independently of either read table.
    for row, end in enumerate(ends):
        for column in range(math.ceil(end / block)):
            valid = min(block, end - column * block)
            physical = int(table_cpu[row, column])
            for cache in (keys_cpu, values_cpu):
                cache[physical, :valid] = (torch.randn(valid, kv_heads, dim, generator=generator) * 0.2).to(dtype)
    query_cpu = (torch.randn(2 * width, heads, dim, generator=generator) * 0.2).to(dtype)
    reference = []
    for row, end in enumerate(ends):
        ids = table_cpu[row, : math.ceil(end / block)].long()
        key = keys_cpu[ids].flatten(0, 1)[:end].double().repeat_interleave(heads // kv_heads, dim=1)
        value = values_cpu[ids].flatten(0, 1)[:end].double().repeat_interleave(heads // kv_heads, dim=1)
        queries = query_cpu[row * width : (row + 1) * width].double()
        scores = torch.einsum("qhd,khd->hqk", queries, key) * scale
        if causal:
            scores.masked_fill_(torch.arange(end)[None, :] > torch.arange(end - width, end)[:, None], -torch.inf)
        reference.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), value))
    expected = torch.cat(reference)
    assert torch.isfinite(expected).all()
    cropped_cpu = torch.zeros((2, math.ceil(max(ends) / block)), dtype=torch.int32)
    for row, end in enumerate(ends):
        columns = math.ceil(end / block)
        cropped_cpu[row, :columns] = table_cpu[row, :columns]
    keys, values, query = keys_cpu.npu(), values_cpu.npu(), query_cpu.npu()
    mask = torch.ones((2048, 2048), dtype=torch.bool).triu(1).npu() if causal else None
    for table in (cropped_cpu.npu(), table_cpu.npu()):
        output, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=keys.view(65, block, -1),
            value=values.view(65, block, -1),
            block_table=table,
            block_size=block,
            input_layout="TND",
            actual_seq_lengths=[width, 2 * width],
            actual_seq_lengths_kv=list(ends),
            num_key_value_heads=kv_heads,
            num_heads=heads,
            scale=scale,
            sparse_mode=3 if causal else 0,
            atten_mask=mask,
        )
        actual = output.cpu().double()
        assert torch.isfinite(actual).all()
        # Preserve the pre-existing BF16 contract, rather than establishing a
        # new bitwise-equivalence requirement between two operator launches.
        torch.testing.assert_close(actual, expected, atol=2**-7, rtol=2**-6)
    for device, original in ((keys, keys_cpu), (values, values_cpu), (query, query_cpu)):
        torch.testing.assert_close(device.cpu(), original, atol=0, rtol=0, equal_nan=True)
