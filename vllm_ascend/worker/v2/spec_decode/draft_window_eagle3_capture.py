# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Instance-owned Eagle3 startup banks and normal submission adapter.

This module does not enable a gate or copy the upstream AR proposal loop. Each
real first-pass/merged-AR invocation supplies its own descriptor and ordinal.
Capture changes only private Draft read operands; full writes/query layout
remain the actual Ascend owners. No capture-time I/O, D2H or synchronization.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any

import torch

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.worker.v2.spec_decode.draft_window import (
    _build_draft_window_source_capability_for_context,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_eager import (
    build_draft_window_attn_metadata,
    draft_window_read_inputs,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_graph import (
    DraftWindowGraphBucket,
    DraftWindowGraphRoute,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_graph_metadata import copy_draft_attention_metadata


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
        self._captures: dict[tuple[DraftWindowGraphRoute, int, int, int], DraftWindowGraphBucket] = {}
        self._capture_call: tuple[DraftWindowGraphRoute, int, int] | None = None
        self._capture_ordinal = 0
        for first, manager in ((True, owner.prefill_cudagraph_manager), (False, owner.decode_cudagraph_manager)):
            if owner.update_stream is None or manager.update_stream is not owner.update_stream:
                raise RuntimeError("Eagle3 capture managers must share their owner UPDATE stream")
            route = DraftWindowGraphRoute.EAGLE3_FIRST_PASS if first else DraftWindowGraphRoute.EAGLE3_AR_STEP
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
                    for ordinal in range(1 if first else owner.num_speculative_steps - 1):
                        key = route, n, desc.num_tokens, ordinal
                        if key in self._captures:
                            raise RuntimeError("ambiguous Eagle3 capture bank (including unsupported LoRA variants)")
                        self._captures[key] = self.provider.register_capture_identity(
                            route=route, descriptor=desc, consumer_ordinal=ordinal
                        )
                        # Seed dummy contents before any GLOBAL capture.
                        # Synchronous H2D is forbidden inside the callback.
                        read = self.provider.startup_capture_workspace(self._captures[key])
                        capacity = min(
                            owner.max_model_len, self.provider.source.window_size + group.table_column_span - 1
                        )
                        read.window_block_table.zero_()
                        read.cpu_window_exact.fill_(capacity)
                        read.device_window_exact.copy_(read.cpu_window_exact)
        if not self._captures:
            raise RuntimeError("Eagle3 Window requires a nonempty capture bucket inventory")

    def require_reusable(self) -> None:
        if getattr(self.owner, "_draft_window_graph_submission_uncertain", False):
            raise RuntimeError("Eagle3 Graph backing is quarantined; process isolation required")
        self.provider.require_reusable()

    def resolve(self, *, first: bool, n: int, tokens: int, ordinal: int) -> DraftWindowGraphBucket:
        route = DraftWindowGraphRoute.EAGLE3_FIRST_PASS if first else DraftWindowGraphRoute.EAGLE3_AR_STEP
        try:
            return self._captures[route, n, tokens, ordinal]
        except KeyError:
            raise RuntimeError("Eagle3 Window has no exact capture bank; no eager fallback") from None

    @contextmanager
    def capture_call(self, *, first: bool, n: int, tokens: int):
        if self._capture_call is not None:
            raise RuntimeError("nested Eagle3 startup capture invocation")
        route = DraftWindowGraphRoute.EAGLE3_FIRST_PASS if first else DraftWindowGraphRoute.EAGLE3_AR_STEP
        self._capture_call, self._capture_ordinal = (route, n, tokens), 0
        try:
            yield
            expected = 1 if first else self.owner.num_speculative_steps - 1
            if self._capture_ordinal != expected:
                raise RuntimeError("Eagle3 capture did not visit every actual first-pass/AR model consumer")
        finally:
            self._capture_call = None

    def capture_metadata(self, metadata: Mapping[str, Any]) -> tuple[Any, Any]:
        """Called solely at the existing Ascend _run_model seam during startup."""
        if self._capture_call is None:
            return metadata, None
        route, n, tokens = self._capture_call
        first = route is DraftWindowGraphRoute.EAGLE3_FIRST_PASS
        identity = self.resolve(first=first, n=n, tokens=tokens, ordinal=self._capture_ordinal)
        self._capture_ordinal += 1
        read = self.provider.startup_capture_workspace(identity)
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
            layer.block_tables = read.window_block_table
            layer.seq_lens = read.device_window_exact
            layer.seq_lens_cpu = read.cpu_window_exact
            layer.seq_lens_list = read.cpu_window_exact.tolist()
            layer.max_seq_len = capacity
            layer.causal = True
            # The shallow metadata copy retains canonical Full write operands.
            # Match runtime write_reference: bind the owned Full input query,
            # not a builder's transient auxiliary GPU copy. FIA's padded Host
            # cumulative lengths remain a distinct field and are preserved.
            layer.query_start_loc = query[: n + 1]
        return private, identity

    def stage(self, inputs: Any, *, first: bool, n: int, tokens: int, ordinal: int) -> None:
        from vllm_ascend.worker.v2.spec_decode.draft_graph_review_limits import (
            validate_eagle_single_request_graph_stage,
        )

        validate_eagle_single_request_graph_stage(
            self.owner,
            inputs,
            first=first,
            n=n,
            tokens=tokens,
            ordinal=ordinal,
        )
        identity = self.resolve(first=first, n=n, tokens=tokens, ordinal=ordinal)
        owner = self.owner

        try:
            self.require_reusable()
            context = owner._require_eagle3_window_context()
            if (
                inputs.proposal is not context
                or inputs.step != owner._eagle3_window_next_step
                or inputs.step >= owner._eagle3_window_expected_steps
                or first is not (inputs.step == 0)
                or ordinal != (0 if first else inputs.step - 1)
            ):
                raise RuntimeError("Eagle3 Graph consumer does not match the active proposal/step")
            read = draft_window_read_inputs(
                identity.workspace, actual_request_count=owner.input_batch.num_reqs, padded_request_count=n
            )
            metadata = build_draft_window_attn_metadata(
                owner,
                num_reqs=owner.input_batch.num_reqs,
                num_reqs_padded=n,
                num_tokens_padded=tokens,
                read_inputs=read,
                query_width=1,
                causal=True,
                query_start_loc_np=owner.input_batch.query_start_loc_np if first else None,
                attn_state=owner.input_batch.attn_state if first else AscendAttentionState.DecodeOnly,
                parallel_config=owner.draft_vllm_config.parallel_config,
            )
            self.provider.bind_read_carrier(capture_identity=identity, inputs=inputs, metadata=metadata)
            self.provider.retire_generation(inputs)
            owner._eagle3_window_next_step = inputs.step + 1
        except BaseException:
            owner._quarantine_eagle3_window()
            raise
