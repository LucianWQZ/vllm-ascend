# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Parallel Draft Graph installation and persistent read-metadata binding.

One provider owns the captured read banks. Runtime preparation binds the
current proposal to those banks while preserving canonical Full write slots,
absolute query positions, and independent Target metadata. No recorder,
output snapshot, artifact writer, or oracle participates in this path.
"""

from __future__ import annotations

from typing import Any

from vllm.config.compilation import CUDAGraphMode

import vllm_ascend.envs as envs
from vllm_ascend.device.hardware_profile import AttentionBackendFamily, get_current_hardware_profile
from vllm_ascend.worker.v2.spec_decode.draft_window_graph import (
    DraftWindowGraphBindingProvider,
)


def draft_window_graph_enabled() -> bool:
    return bool(envs.VLLM_ASCEND_EXPERIMENTAL_DRAFT_WINDOW_GRAPH)


def validate_graph_context(vllm_config: Any, block_tables: Any) -> None:
    """Check the actual executor and source, without copying feature state."""
    if not draft_window_graph_enabled():
        raise NotImplementedError("Draft Window Graph requires its explicit supported Graph configuration")
    if get_current_hardware_profile().attention_backend_family is not AttentionBackendFamily.STANDARD:
        raise NotImplementedError("Draft Window Graph requires the STANDARD hardware family")
    if vllm_config.compilation_config.cudagraph_mode.name != "FULL_DECODE_ONLY":
        raise ValueError("Draft Window Graph requires FULL_DECODE_ONLY")
    parallel = vllm_config.parallel_config
    if (
        any(
            getattr(parallel, name) != 1
            for name in ("pipeline_parallel_size", "prefill_context_parallel_size", "decode_context_parallel_size")
        )
        or getattr(block_tables, "cp_size", 1) != 1
    ):
        raise RuntimeError("Draft Window Graph requires PP=PCP=DCP=1 and unsharded Draft tables")


def install_parallel_graph(owner: Any) -> DraftWindowGraphBindingProvider:
    """Install one provider and register every real startup capture bucket."""
    provider = owner._draft_window_graph_binding_provider
    if provider._carriers:
        raise RuntimeError("Draft Window Graph capture buckets are already installed")
    capture_descs = owner.query_cudagraph_manager._capture_descs
    descriptors = tuple(
        descriptor
        for mode_descriptors in capture_descs.values()
        for descriptor in mode_descriptors
        if descriptor.cg_mode is CUDAGraphMode.FULL
    )
    if not descriptors:
        raise RuntimeError("Draft Window Graph manager has no capture descriptors")
    for descriptor in descriptors:
        provider.register_capture_identity(
            descriptor=descriptor,
        )
    return provider
