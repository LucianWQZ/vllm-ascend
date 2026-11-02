# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Validate logical launch-grid extents before unguarded model calls."""

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from vllm.triton_utils import triton  # noqa: E402

from vllm_ascend.ops.triton.v2.spec_decode.draft_window import (  # noqa: E402
    _gather_draft_kv_fp32,
    gather_draft_kv_fp32,
)

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires an Ascend NPU")


@pytest.mark.parametrize("tokens", [1, 7, 8, 9, 1151, 1152])
def test_gather_grid_extent_preserves_arbitrary_tail(tokens):
    source = torch.arange(12 * 128 * 2 * 32, dtype=torch.float32).reshape(12, 128, 2, 32)
    keys, values = source.to("npu"), (-source).to("npu")
    slots_cpu = torch.arange(tokens - 1, -1, -1, dtype=torch.int64)[None]
    slots = slots_cpu.to("npu")
    actual_keys, actual_values = gather_draft_kv_fp32(slots, keys, values)
    torch.npu.synchronize()
    expected = source.flatten(0, 1)[slots_cpu]
    torch.testing.assert_close(actual_keys.cpu(), expected, atol=0, rtol=0)
    torch.testing.assert_close(actual_values.cpu(), -expected, atol=0, rtol=0)


def test_aligned_capacity_changes_reuse_gather_binary():
    driver = triton.runtime.driver

    keys = torch.zeros((135, 128, 1, 32), device="npu", dtype=torch.bfloat16)
    values = torch.ones_like(keys)
    torch.npu.synchronize()
    cache = _gather_draft_kv_fp32.device_caches[driver.active.get_current_device()][0]
    # Clear only this test process's drained in-memory kernel cache. Disk
    # cache remains valid; distinct specializations still enter the compiler
    # and return distinct hashes even when their files already exist.
    cache.clear()
    original = _gather_draft_kv_fp32.compile
    hashes = []

    def compiled(source, *args, **kwargs):
        kernel = original(source, *args, **kwargs)
        hashes.append(kernel.hash)
        return kernel

    _gather_draft_kv_fp32.compile = compiled
    try:
        for capacity in (17216, 17248, 17280):
            slots = torch.arange(capacity, device="npu", dtype=torch.int64)[None]
            output = gather_draft_kv_fp32(slots, keys, values)
            torch.npu.synchronize()
            torch.testing.assert_close(output[1].cpu(), torch.ones_like(output[1].cpu()), atol=0, rtol=0)
        assert len(hashes) == 1, "Full capacity still creates separate gather binaries"
    finally:
        _gather_draft_kv_fp32.compile = original
