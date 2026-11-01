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
from copy import copy
from typing import Any

import torch

from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl, AscendAttentionState
from vllm_ascend.worker.v2.spec_decode.draft_window import (
    _build_draft_window_source_capability_for_context,
    build_draft_window_runtime_context,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_eager import _isolate_draft_metadata_from_target
from vllm_ascend.worker.v2.spec_decode.draft_window_eagle3_graph import Eagle3GraphStagingOwner
from vllm_ascend.worker.v2.spec_decode.draft_window_graph import (
    DraftWindowGraphCaptureIdentity,
    DraftWindowGraphMetadataBuild,
    DraftWindowGraphReadInputs,
    DraftWindowGraphRoute,
    DraftWindowGraphRouteFingerprint,
    DraftWindowGraphWriteSideReference,
    install_draft_window_graph_binding_provider,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_graph_runtime import (
    draft_window_graph_enabled,
    validate_graph_context,
)


def build_eagle3_graph_source(owner: Any) -> tuple[Any, Any]:
    """Account for Graph explicitly before using the existing P2 inventory."""
    if not draft_window_graph_enabled():
        raise RuntimeError("Eagle3 Graph Window requires the default-off experimental gate")
    from vllm_ascend.worker.v2.spec_decode.draft_graph_review_limits import eagle_dp2_graph_review_enabled

    context = build_draft_window_runtime_context(owner.vllm_config, owner.block_tables)
    rank_local_dp = eagle_dp2_graph_review_enabled(owner.vllm_config)
    replicated_review = validate_graph_context(context, owner.vllm_config)
    if context.graph_mode != "FULL_DECODE_ONLY":
        raise NotImplementedError("Eagle3 Graph first implementation requires FULL_DECODE_ONLY")
    if owner.method != "eagle3" or owner.speculative_config.uses_dynamic_speculative_decoding():
        raise NotImplementedError("Eagle3 Graph requires normalized Eagle3 and static K")
    # Executor features and the rank-local Draft consumer are validated
    # separately, while the source retains the actual runtime identity.
    if context.dsa_context_parallel:
        raise NotImplementedError("Eagle3 Draft Graph does not support DSA-CP")
    capability = _build_draft_window_source_capability_for_context(
        policy=owner._draft_window_policy,
        runtime_route="eagle3",
        attn_groups=owner.attn_groups,
        block_tables=owner.block_tables,
        kv_cache_config=owner.kv_cache_config,
        causal=True,
        vllm_config=owner.vllm_config,
        runtime_context=context,
        allow_replicated_tp_graph_source=replicated_review,
        allow_parallel_prefix_graph_source=context.prefix_caching,
        allow_parallel_dp_graph_source=rank_local_dp,
    )
    topology = capability.eager_topology_fingerprint
    if topology is None:
        raise RuntimeError("Eagle3 Graph has no installed-runtime topology identity")
    if context.prefix_caching:
        capability.full_source.validate_owner_bindings()
        if topology.prefix_caching is not True:
            raise RuntimeError("Eagle3 prefix Graph source lost the actual caching identity")
    return capability, context


def _class_path(cls: type) -> str:
    return cls.__module__ + "." + cls.__qualname__


class Eagle3GraphSession:
    """Separate persistent bank per descriptor x real AR consumer ordinal."""

    def __init__(self, owner: Any, context: Any) -> None:
        validate_graph_context(context, getattr(owner, "vllm_config", None))
        if context.graph_mode != "FULL_DECODE_ONLY":
            raise NotImplementedError("Eagle3 Graph session requires FULL_DECODE_ONLY")
        self.owner = owner
        self.context = context
        self.provider = install_draft_window_graph_binding_provider(owner)
        self.controller = Eagle3GraphStagingOwner(
            owner._draft_window_runtime_state, self.provider, static_k=owner.num_speculative_steps
        )
        self._captures: dict[tuple[DraftWindowGraphRoute, int, int, int], DraftWindowGraphCaptureIdentity] = {}
        self._capture_call: tuple[DraftWindowGraphRoute, int, int] | None = None
        self._capture_ordinal = 0
        for first, manager in ((True, owner.prefill_cudagraph_manager), (False, owner.decode_cudagraph_manager)):
            route = DraftWindowGraphRoute.EAGLE3_FIRST_PASS if first else DraftWindowGraphRoute.EAGLE3_AR_STEP
            q = owner.num_speculative_steps + 1 if first else 1
            group = owner._draft_window_runtime_state.capability.inventory.require_single_group_direct_prefix()
            for descriptors in manager._capture_descs.values():
                for desc in descriptors:
                    if desc.cg_mode.name != "FULL":
                        raise NotImplementedError("Eagle3 Window capture requires an actual FULL descriptor")
                    n = desc.num_reqs or min(desc.num_tokens, manager.max_num_reqs)
                    if desc.num_tokens != n * q:
                        raise RuntimeError("Eagle3 startup descriptor does not match its method-specific query width")
                    fingerprint = DraftWindowGraphRouteFingerprint(
                        method="eagle3",
                        route=route,
                        graph_mode="FULL",
                        backend_impl=_class_path(AscendAttentionBackendImpl),
                        builder_impl=group.builder,
                        global_group_id=group.gid,
                        layout=group.status.value,
                        capture_descriptor=f"eagle3:{route.value}:{n}:{desc.num_tokens}",
                        capture_bucket=(n, desc.num_tokens),
                        query_width=q,
                        padded_request_count=n,
                        padded_token_count=desc.num_tokens,
                        static_k=owner.num_speculative_steps,
                        window_size=owner._draft_window_runtime_state.capability.inventory.window_size,
                        logical_block_size=group.scheduler_block_size,
                        kernel_block_size=group.table_column_span,
                        block_span_tokens=group.table_column_span,
                        tensor_parallel_size=context.tensor_parallel_size,
                        tensor_parallel_rank=context.tensor_parallel_rank,
                        parallel_read_semantics="replicated",
                        data_parallel_size=context.data_parallel_size,
                        pipeline_parallel_size=context.pipeline_parallel_size,
                        prefill_context_parallel_size=context.prefill_context_parallel_size,
                        decode_context_parallel_size=context.decode_context_parallel_size,
                        sequence_parallel=context.sequence_parallel,
                        dsa_context_parallel=context.dsa_context_parallel,
                        flashcomm=context.flashcomm,
                        async_scheduling=context.async_scheduling,
                        prefix_caching=context.prefix_caching,
                    )
                    for ordinal in range(1 if first else owner.num_speculative_steps - 1):
                        key = route, n, desc.num_tokens, ordinal
                        if key in self._captures:
                            raise RuntimeError("ambiguous Eagle3 capture bank (including unsupported LoRA variants)")
                        self._captures[key] = self.provider.register_capture_identity(
                            fingerprint=fingerprint, execution_lane=0, consumer_ordinal=ordinal
                        )
                        # Seed dummy contents before any GLOBAL capture.
                        # Synchronous H2D is forbidden inside the callback.
                        read = self.provider.startup_capture_read_inputs(self._captures[key])
                        capacity = min(owner.max_model_len, fingerprint.window_size + group.table_column_span - 1)
                        read.window_block_table.zero_()
                        read.valid_num_blocks.fill_(read.window_block_table.shape[1])
                        read.cpu_window_exact.fill_(capacity)
                        read.cpu_window_upper.fill_(capacity)
                        read.device_window_exact.copy_(read.cpu_window_exact)

    def validate_for_provider(self, provider: Any) -> None:
        if provider is not self.provider or not self._captures:
            raise RuntimeError("Eagle3 startup carrier belongs to another provider or has no inventory")

    def resolve(self, *, first: bool, n: int, tokens: int, ordinal: int) -> DraftWindowGraphCaptureIdentity:
        route = DraftWindowGraphRoute.EAGLE3_FIRST_PASS if first else DraftWindowGraphRoute.EAGLE3_AR_STEP
        try:
            return self._captures[route, n, tokens, ordinal]
        except KeyError as error:
            raise RuntimeError("Eagle3 Window has no exact capture bank; no eager fallback") from error

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
        read = self.provider.startup_capture_read_inputs(identity)
        # Persistent dummy banks were initialized before GLOBAL capture.
        # No CPU->device copy, fill or reset is allowed in this callback.
        capacity = min(
            self.owner.max_model_len, identity.fingerprint.window_size + identity.fingerprint.block_span_tokens - 1
        )
        private = _isolate_draft_metadata_from_target(metadata, metadata)
        result: dict[str, Any] = {}
        for name, layer in private.items():
            if layer is None:
                result[name] = None
                continue
            source = metadata[name]
            layer = copy(layer)
            layer.block_tables = read.window_block_table
            layer.seq_lens = read.device_window_exact
            layer.seq_lens_cpu = read.cpu_window_exact
            layer.seq_lens_list = read.cpu_window_exact.tolist()
            layer.max_seq_len = capacity
            layer.causal = True
            # Full write/query fields are preserved, never derived from V/S.
            for attr in ("slot_mapping", "actual_seq_lengths_q", "positions"):
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

    def write_reference(self, identity: DraftWindowGraphCaptureIdentity, target: Mapping[str, Any]):
        fp = identity.fingerprint
        state = next(v for v in target.values() if v is not None).attn_state
        if fp.route is DraftWindowGraphRoute.EAGLE3_AR_STEP:
            state = AscendAttentionState.DecodeOnly
        return DraftWindowGraphWriteSideReference(
            slot_mapping=self.owner.block_tables.slot_mappings[fp.global_group_id, : fp.padded_token_count],
            positions=self.owner.input_buffers.positions[: fp.padded_token_count],
            query_start_loc=self.owner.input_buffers.query_start_loc[: fp.padded_request_count + 1],
            actual_seq_lengths_q=tuple((i + 1) * fp.query_width for i in range(fp.padded_request_count)),
            attn_state=state,
            num_actual_tokens=fp.padded_token_count,
            causal=True,
        )

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
        write = self.write_reference(identity, target)

        def build(read: DraftWindowGraphReadInputs):
            private = owner._build_eagle3_graph_metadata(
                inputs=inputs,
                read=read,
                num_reqs=inputs.actual_request_count,
                num_reqs_padded=n,
                num_tokens_padded=tokens,
                query_start_loc_np=owner.input_batch.query_start_loc_np if first else None,
                target_metadata=target,
                first_pass=first,
            )
            final: dict[str, Any] = {}
            for name, layer in private.items():
                if layer is None:
                    final[name] = None
                    continue
                layer = copy(layer)
                layer.slot_mapping, layer.query_start_loc = write.slot_mapping, write.query_start_loc
                layer.actual_seq_lengths_q = list(write.actual_seq_lengths_q)
                layer.attn_state, layer.num_actual_tokens, layer.causal = (
                    write.attn_state,
                    write.num_actual_tokens,
                    True,
                )
                if hasattr(layer, "positions"):
                    layer.positions = write.positions
                final[name] = layer
            return DraftWindowGraphMetadataBuild(metadata=final, private_isolation_view=private)

        self.controller.bind_and_release_scratch(
            capture_identity=identity,
            inputs=inputs,
            metadata_builder=build,
            target_metadata=target,
            write_side_reference=write,
        )
        owner._eagle3_window_last_identity = inputs.identity
        owner._eagle3_window_next_step = inputs.identity.step + 1
        owner._eagle3_window_live_inputs = None
