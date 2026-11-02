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

from vllm_ascend.worker.v2.spec_decode.draft_window import (
    DraftWindowWorkspace,
    _build_draft_window_source_capability_for_context,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_graph import (
    DraftWindowGraphBindingProvider,
    DraftWindowGraphCaptureIdentity,
    DraftWindowGraphCarrierState,
    DraftWindowGraphReadCarrier,
    DraftWindowGraphRoute,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_graph_metadata import copy_draft_attention_metadata
from vllm_ascend.worker.v2.spec_decode.draft_window_graph_runtime import (
    draft_window_graph_enabled,
    validate_graph_context,
)


def build_eagle3_graph_source(owner: Any):
    """Bind the actual consumer Source after normal Graph configuration."""
    if not draft_window_graph_enabled():
        raise RuntimeError("Eagle3 Graph Window requires the default-off experimental gate")
    validate_graph_context(owner.vllm_config, owner.block_tables)
    if owner.method != "eagle3" or owner.speculative_config.uses_dynamic_speculative_decoding():
        raise NotImplementedError("Eagle3 Graph requires normalized Eagle3 and static K")
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
        validate_graph_context(owner.vllm_config, owner.block_tables)
        self.owner = owner
        self.provider = owner._draft_window_graph_binding_provider
        if not isinstance(self.provider, DraftWindowGraphBindingProvider):
            raise TypeError("Eagle3 Graph requires its initialized source owner")
        self._carriers: list[DraftWindowGraphReadCarrier] = []
        self._captures: dict[tuple[DraftWindowGraphRoute, int, int, int], DraftWindowGraphCaptureIdentity] = {}
        self._capture_call: tuple[DraftWindowGraphRoute, int, int] | None = None
        self._capture_ordinal = 0
        for first, manager in ((True, owner.prefill_cudagraph_manager), (False, owner.decode_cudagraph_manager)):
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
                        read.cpu_window_upper.fill_(capacity)
                        read.device_window_exact.copy_(read.cpu_window_exact)

    def require_reusable(self) -> None:
        if getattr(self.owner, "_draft_window_graph_submission_uncertain", False):
            raise RuntimeError("Eagle3 Graph backing is quarantined; process isolation required")
        self.provider.require_reusable()

    def finish(self) -> None:
        """Check real UPDATE attribution; Host retirement is not completion."""
        self.require_reusable()
        if any(carrier.state is not DraftWindowGraphCarrierState.UPDATED for carrier in self._carriers):
            raise RuntimeError("Eagle3 Graph banks lack actual backend UPDATE attribution")
        self._carriers.clear()

    def validate_for_provider(self, provider: Any) -> None:
        if provider is not self.provider or not self._captures:
            raise RuntimeError("Eagle3 startup carrier belongs to another provider or has no inventory")

    def resolve(self, *, first: bool, n: int, tokens: int, ordinal: int) -> DraftWindowGraphCaptureIdentity:
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
        result: dict[str, Any] = {}
        for name, layer in private.items():
            if layer is None:
                result[name] = None
                continue
            source = metadata[name]
            layer.block_tables = read.window_block_table
            layer.seq_lens = read.device_window_exact
            layer.seq_lens_cpu = read.cpu_window_exact
            layer.seq_lens_list = read.cpu_window_exact.tolist()
            layer.max_seq_len = capacity
            layer.causal = True
            # Full write/query fields are preserved, never derived from V/S.
            for attr in ("slot_mapping", "positions"):
                if hasattr(source, attr):
                    setattr(layer, attr, getattr(source, attr))
            # Match runtime write_reference: bind the owned Full input query,
            # not a builder's transient auxiliary GPU copy. FIA's padded Host
            # cumulative lengths remain a distinct field and are preserved.
            query = self.owner.input_buffers.query_start_loc
            if query.dtype != torch.int32 or query.ndim != 1 or query.numel() < n + 1:
                raise RuntimeError("Eagle3 capture requires its owned int32 Full input query")
            layer.query_start_loc = query[: n + 1]
            result[name] = layer
        return result, identity

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
        target = owner._require_eagle3_target_metadata()

        def build(workspace: DraftWindowWorkspace):
            private = owner._build_eagle3_graph_metadata(
                inputs=inputs,
                workspace=workspace,
                num_reqs=len(inputs.identity.proposal.ordered_request_ids),
                num_reqs_padded=n,
                num_tokens_padded=tokens,
                query_start_loc_np=owner.input_batch.query_start_loc_np if first else None,
                target_metadata=target,
                first_pass=first,
            )
            return private

        try:
            self.require_reusable()
            current = inputs.identity
            inputs.validate_for(current)
            context = owner._require_eagle3_window_context()
            if (
                current.proposal is not context
                or current.step != owner._eagle3_window_next_step
                or current.step >= owner._eagle3_window_expected_steps
                or first is not (current.step == 0)
                or ordinal != (0 if first else current.step - 1)
            ):
                raise RuntimeError("Eagle3 Graph consumer does not match the active proposal/step")
            carrier = self.provider.bind_read_carrier(
                capture_identity=identity,
                inputs=inputs,
                metadata_builder=build,
                target_metadata=target,
            )
            if carrier.state is not DraftWindowGraphCarrierState.BOUND:
                raise RuntimeError("Eagle3 consumer has no bound persistent destination")
            self.require_reusable()
            self.provider.retire_generation(current)
            self._carriers.append(carrier)
            owner._eagle3_window_next_step = current.step + 1
        except BaseException:
            owner._quarantine_eagle3_window()
            raise
