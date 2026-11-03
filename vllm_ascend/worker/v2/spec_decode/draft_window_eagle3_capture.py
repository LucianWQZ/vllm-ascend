# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Instance-owned Eagle3 startup banks and normal submission adapter.

This module does not enable a gate or copy the upstream AR proposal loop. Each
real first-pass/merged-AR invocation supplies its own descriptor and ordinal.
Capture changes only private Draft read operands; full writes/query layout
remain the actual Ascend owners. No capture-time I/O, D2H or synchronization.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import torch

from vllm_ascend.worker.v2.spec_decode.draft_window import (
    _build_draft_window_source_capability_for_context,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_graph import (
    DraftWindowGraphBucket,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_graph_metadata import (
    bind_draft_window_read,
    copy_draft_attention_metadata,
)


def build_eagle3_graph_source(owner: Any):
    """Bind the actual consumer Source after normal Graph configuration."""
    if owner.method != "eagle3":
        raise NotImplementedError("Eagle3 Graph requires the normalized Eagle3 method")
    return _build_draft_window_source_capability_for_context(
        policy=owner._draft_window_policy,
        runtime_route="eagle3",
        attn_groups=owner.attn_groups,
        block_tables=owner.block_tables,
        kv_cache_config=owner.kv_cache_config,
        causal=True,
        vllm_config=owner.vllm_config,
    )


class Eagle3GraphSession:
    """Separate persistent bank per descriptor x real AR consumer ordinal."""

    def __init__(self, owner: Any) -> None:
        self.owner = owner
        self.provider = owner._draft_window_graph_binding_provider
        self._capture_call: Iterator[DraftWindowGraphBucket] | None = None
        for first, manager in ((True, owner.prefill_cudagraph_manager), (False, owner.decode_cudagraph_manager)):
            if owner.update_stream is None or manager.update_stream is not owner.update_stream:
                raise RuntimeError("Eagle3 capture managers must share their owner UPDATE stream")
            q = owner.num_speculative_steps + 1 if first else 1
            group = self.provider.source.group
            for descriptors in manager._capture_descs.values():
                for desc in descriptors:
                    if desc.cg_mode.name != "FULL":
                        raise NotImplementedError("Eagle3 Window capture requires an actual FULL descriptor")
                    n = desc.num_reqs
                    if n is None:
                        raise RuntimeError("Eagle3 FULL descriptor has no request count")
                    if desc.num_tokens != n * q:
                        raise RuntimeError("Eagle3 startup descriptor does not match its method-specific query width")
                    captures = tuple(
                        self.provider.register_capture_identity(descriptor=desc, consumer_ordinal=ordinal)
                        for ordinal in range(1 if first else owner.num_speculative_steps - 1)
                    )
                    for capture in captures:
                        # Seed dummy contents before any GLOBAL capture.
                        # Synchronous H2D is forbidden inside the callback.
                        read = capture
                        capacity = min(
                            owner.max_model_len, self.provider.source.window_size + group.table_column_span - 1
                        )
                        read.window_block_table.zero_()
                        read.cpu_window_exact.fill_(capacity)
                        read.device_window_exact.copy_(read.cpu_window_exact)
        if not self.provider._carriers:
            raise RuntimeError("Eagle3 Window requires a nonempty capture bucket inventory")

    def resolve(self, *, n: int, tokens: int, ordinal: int) -> DraftWindowGraphBucket:
        captures = self.provider.capture_banks_for(num_reqs=n, num_tokens=tokens)
        if not 0 <= ordinal < len(captures):
            raise RuntimeError("Eagle3 Window has no exact capture bank; no eager fallback")
        return captures[ordinal]

    @contextmanager
    def capture_call(self, *, n: int, tokens: int):
        if self._capture_call is not None:
            raise RuntimeError("nested Eagle3 startup capture invocation")
        self._capture_call = iter(self.provider.capture_banks_for(num_reqs=n, num_tokens=tokens))
        try:
            yield
            if next(self._capture_call, None) is not None:
                raise RuntimeError("Eagle3 capture did not visit every actual first-pass/AR model consumer")
        finally:
            self._capture_call = None

    def capture_metadata(self, metadata: Mapping[str, Any]) -> Any:
        """Called solely at the existing Ascend _run_model seam during startup."""
        if self._capture_call is None:
            raise RuntimeError("Eagle3 capture metadata has no active capture invocation")
        identity = next(self._capture_call, None)
        if identity is None:
            raise RuntimeError("Eagle3 capture visited more consumers than its real descriptor")
        n = identity.descriptor.num_reqs
        read = identity
        # Persistent dummy banks were initialized before GLOBAL capture.
        # No CPU->device copy, fill or reset is allowed in this callback.
        capacity = min(
            self.owner.max_model_len,
            self.provider.source.window_size + self.provider.source.group.table_column_span - 1,
        )
        private = copy_draft_attention_metadata(metadata)
        query = self.owner.input_buffers.query_start_loc
        if query.dtype != torch.int32 or query.ndim != 1 or query.numel() < n + 1:
            raise RuntimeError("Eagle3 capture requires its owned int32 Full input query")
        for layer in private.values():
            if layer is None:
                continue
            bind_draft_window_read(layer, read, capacity)
            layer.causal = True
            # The shallow metadata copy retains canonical Full write operands.
            # Match runtime write_reference: bind the owned Full input query,
            # not a builder's transient auxiliary GPU copy. FIA's padded Host
            # cumulative lengths remain a distinct field and are preserved.
            layer.query_start_loc = query[: n + 1]
        return private

    def stage(self, capture: DraftWindowGraphBucket, step: int) -> None:
        self.owner._materialize_eagle3_step(capture, step)
        self.provider.bind_current(capture.descriptor)
        self.owner._eagle3_window_next_step = step + 1
