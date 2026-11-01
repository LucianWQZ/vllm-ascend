# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Parallel Draft Graph installation and persistent read-metadata binding.

One session owns the captured read banks. Runtime preparation binds the
current proposal to those banks while preserving canonical Full write slots,
absolute query positions, and independent Target metadata. No recorder,
output snapshot, artifact writer, or oracle participates in this path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, replace
from typing import Any

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

import vllm_ascend.envs as envs
from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
from vllm_ascend.device.hardware_profile import AttentionBackendFamily
from vllm_ascend.worker.v2.attn_utils import build_attn_metadata_factory, build_attn_metadata_wrapper
from vllm_ascend.worker.v2.spec_decode.dflash.capture import format_dflash_capture_descriptor
from vllm_ascend.worker.v2.spec_decode.draft_window import (
    DraftMetadataInputs,
    DraftWindowPolicyDecision,
    DraftWindowRuntimeContext,
    DraftWindowRuntimeState,
    DraftWindowSourceCapability,
    _build_draft_window_source_capability_for_context,
    build_draft_window_runtime_context,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_eager import AscendDraftAttentionReadInputs
from vllm_ascend.worker.v2.spec_decode.draft_window_graph import (
    DraftWindowGraphBindingProvider,
    DraftWindowGraphCaptureIdentity,
    DraftWindowGraphMetadataBuild,
    DraftWindowGraphReadCarrier,
    DraftWindowGraphReadInputs,
    DraftWindowGraphRoute,
    DraftWindowGraphRouteFingerprint,
    DraftWindowGraphWriteSideReference,
    _tensor_identity,
    install_draft_window_graph_binding_provider,
    resolve_parallel_draft_window_graph_route,
)


@dataclass(slots=True)
class DraftWindowGraphSession:
    owner: Any
    provider: DraftWindowGraphBindingProvider
    runtime_context: DraftWindowRuntimeContext
    route: DraftWindowGraphRoute

    def resolve_capture_identity(
        self,
        descriptor: BatchExecutionDescriptor,
    ) -> DraftWindowGraphCaptureIdentity:
        manager = self.owner.query_cudagraph_manager
        descriptor_text = format_dflash_capture_descriptor(
            descriptor,
            max_num_reqs=int(manager.max_num_reqs),
        )
        num_reqs = _descriptor_request_count(
            descriptor,
            max_num_reqs=int(manager.max_num_reqs),
        )
        query_width = descriptor.uniform_token_count or descriptor.num_tokens // num_reqs
        matches = tuple(
            identity
            for identity in self.provider.registered_capture_identities(route=self.route)
            if identity.fingerprint.capture_descriptor == descriptor_text
            and identity.fingerprint.capture_bucket == (num_reqs, descriptor.num_tokens)
            and identity.fingerprint.query_width == query_width
            and identity.fingerprint.graph_mode == descriptor.cg_mode.name
        )
        if len(matches) != 1:
            raise RuntimeError(
                "runtime graph descriptor does not resolve one persistent carrier: "
                f"descriptor={descriptor_text!r} matches={len(matches)}"
            )
        return matches[0]


def draft_window_graph_enabled() -> bool:
    return bool(envs.VLLM_ASCEND_EXPERIMENTAL_DRAFT_WINDOW_GRAPH)


def validate_graph_context(runtime_context: DraftWindowRuntimeContext, vllm_config: Any) -> bool:
    """Match the actual executor to its once-resolved supported Graph route."""
    from vllm_ascend.ascend_config import validate_draft_window_graph_config

    if not validate_draft_window_graph_config(vllm_config):
        raise NotImplementedError("Draft Window Graph requires its explicit supported Graph configuration")
    if runtime_context.attention_backend_family is not AttentionBackendFamily.STANDARD:
        raise NotImplementedError("Draft Window Graph requires the STANDARD hardware family")
    if runtime_context.graph_mode != "FULL_DECODE_ONLY":
        raise ValueError("Draft Window Graph runtime mode differs from FULL_DECODE_ONLY")
    parallel = vllm_config.parallel_config
    actual = (
        runtime_context.tensor_parallel_size,
        runtime_context.data_parallel_size,
        runtime_context.pipeline_parallel_size,
        runtime_context.prefill_context_parallel_size,
        runtime_context.decode_context_parallel_size,
    )
    expected = (parallel.tensor_parallel_size, parallel.data_parallel_size, 1, 1, 1)
    if actual != expected:
        raise RuntimeError("Draft Window Graph source and configured executor topology differ")
    rank = runtime_context.tensor_parallel_rank
    if type(rank) is not int or not 0 <= rank < runtime_context.tensor_parallel_size:
        raise ValueError("Draft Window Graph requires its actual TP rank")
    return True


def build_parallel_graph_source(
    *,
    policy: DraftWindowPolicyDecision,
    runtime_route: str,
    attn_groups: Sequence[Sequence[Any]],
    block_tables: Any,
    kv_cache_config: Any,
    causal: bool | Mapping[int, bool],
    vllm_config: Any,
    input_buffers: Any = None,
    dflash2_graph_contract: Any = None,
) -> tuple[DraftWindowSourceCapability, DraftWindowRuntimeContext]:
    """Build P2/P3 source state only after the real Graph route is gated."""
    if not draft_window_graph_enabled():
        raise RuntimeError("internal Draft Window Graph flag is not enabled")
    runtime_context = build_draft_window_runtime_context(vllm_config, block_tables)
    replicated_review = validate_graph_context(runtime_context, vllm_config)
    if runtime_route not in {"dflash", "dflash2", "dspark"}:
        raise NotImplementedError("Draft Window Graph supports only DFlash and eligible DSpark")
    speculative_config = vllm_config.speculative_config
    if bool(speculative_config.uses_dynamic_speculative_decoding()):
        raise NotImplementedError("Draft Window Graph requires static K")

    # The source builder validates the real executor and rank-local Draft
    # consumer separately. Never hide Graph/Async/Target SP in a fake eager
    # context, or reconstruct a second topology after source construction.
    from vllm_ascend.worker.v2.spec_decode.draft_window_swa_graph import partition_source

    if runtime_route == "dflash2":
        from vllm_ascend.worker.v2.spec_decode.draft_window_dflash2_graph import DFlash2GraphContract

        if not isinstance(dflash2_graph_contract, DFlash2GraphContract):
            raise NotImplementedError("DFlash2 Graph has no admitted model/selector/native reader contract")
        # Native-only noncausal DFlash2 projects the same complete canonical
        # source through its own resident-interval contract. It does not use
        # the older causal Full+SWA passthrough partition or change native P.
        source_groups, native_swa = attn_groups, None
    else:
        if dflash2_graph_contract is not None:
            raise RuntimeError("DFlash2 source contract belongs to a different method")
        source_groups, native_swa = partition_source(
            attn_groups=attn_groups,
            block_tables=block_tables,
            input_buffers=input_buffers,
            causal=causal,
            window_size=policy.effective_window_size,
            query_width=int(speculative_config.num_speculative_tokens) + 1,
        )
    if native_swa is not None and (
        runtime_route != "dflash"
        or not replicated_review
        or runtime_context.tensor_parallel_size != 4
        or runtime_context.async_scheduling
        or runtime_context.prefix_caching
        or runtime_context.sequence_parallel
        or runtime_context.dsa_context_parallel
        or runtime_context.flashcomm
    ):
        raise NotImplementedError("native SWA Graph first requires the actual uncombined DFlash TP4 review route")
    capability = _build_draft_window_source_capability_for_context(
        policy=policy,
        runtime_route=runtime_route,
        attn_groups=source_groups,
        block_tables=block_tables,
        kv_cache_config=kv_cache_config,
        causal=causal,
        vllm_config=vllm_config,
        runtime_context=runtime_context,
        allow_replicated_tp_graph_source=replicated_review,
        allow_parallel_prefix_graph_source=runtime_context.prefix_caching,
        allow_parallel_dp_graph_source=replicated_review and runtime_context.data_parallel_size == 2,
        dflash2_graph_contract=dflash2_graph_contract,
    )
    topology = capability.eager_topology_fingerprint
    if topology is None:
        raise RuntimeError("Draft Graph source has no actual runtime topology")
    if runtime_context.prefix_caching:
        # Reuse P7's current-request resident snapshot, owned exact lengths and
        # canonical Full source. Prefix is kept in the real topology identity;
        # it is not hidden by replacing it with an OFF context. A prefix hit
        # never permits changing canonical Full slots or source ownership.
        capability.full_source.validate_owner_bindings()
        if topology.prefix_caching is not True:
            raise RuntimeError("parallel prefix Graph source lost the actual caching identity")
    capability = replace(capability, native_swa_graph=native_swa)
    return capability, runtime_context


def install_parallel_graph(
    owner: Any,
    *,
    runtime_context: DraftWindowRuntimeContext,
) -> DraftWindowGraphSession:
    """Install one provider and register every real startup capture bucket."""
    validate_graph_context(runtime_context, getattr(owner, "vllm_config", None))
    if hasattr(owner, "_draft_window_graph_session"):
        raise RuntimeError("Draft Window Graph is already installed on this speculator")
    runtime_state = getattr(owner, "_draft_window_runtime_state", None)
    if not isinstance(runtime_state, DraftWindowRuntimeState):
        raise RuntimeError("Draft Window Graph requires initialized P3 runtime state")
    manager = getattr(owner, "query_cudagraph_manager", None)
    capture_descs = getattr(manager, "_capture_descs", None)
    max_num_reqs = getattr(manager, "max_num_reqs", None)
    if not isinstance(capture_descs, Mapping) or not capture_descs:
        raise RuntimeError("Draft Window Graph requires real manager capture descriptors")
    if not isinstance(max_num_reqs, int) or max_num_reqs <= 0:
        raise RuntimeError("Draft Window Graph manager has no valid request capacity")

    provider = install_draft_window_graph_binding_provider(owner)
    route = resolve_parallel_draft_window_graph_route(owner)
    if route not in (DraftWindowGraphRoute.DFLASH, DraftWindowGraphRoute.DSPARK):
        raise RuntimeError("Draft Window Graph has no parallel Draft route")
    session = DraftWindowGraphSession(
        owner=owner,
        provider=provider,
        runtime_context=runtime_context,
        route=route,
    )
    descriptors = tuple(descriptor for mode_descriptors in capture_descs.values() for descriptor in mode_descriptors)
    if not descriptors:
        raise RuntimeError("Draft Window Graph manager has no capture descriptors")
    descriptor_keys = tuple(
        (
            descriptor.cg_mode.name,
            _descriptor_request_count(descriptor, max_num_reqs=max_num_reqs),
            descriptor.num_tokens,
        )
        for descriptor in descriptors
    )
    if len(set(descriptor_keys)) != len(descriptor_keys):
        raise NotImplementedError("Draft Window Graph does not support ambiguous LoRA capture buckets")

    group = runtime_state.capability.inventory.require_single_group_direct_prefix()
    block_span_tokens = group.table_column_span
    if not isinstance(block_span_tokens, int) or isinstance(block_span_tokens, bool) or block_span_tokens <= 0:
        raise RuntimeError("Draft Window Graph group has no positive block-span contract")
    static_k = int(owner.speculative_config.num_speculative_tokens)
    if static_k <= 0:
        raise ValueError("Draft Window Graph static K must be positive")
    for descriptor in descriptors:
        num_reqs = _descriptor_request_count(descriptor, max_num_reqs=max_num_reqs)
        if descriptor.num_tokens % num_reqs:
            raise RuntimeError("Graph capture token bucket is not divisible by request count")
        query_width = descriptor.uniform_token_count or descriptor.num_tokens // num_reqs
        fingerprint = DraftWindowGraphRouteFingerprint(
            method=runtime_state.capability.inventory.method,
            route=route,
            graph_mode=descriptor.cg_mode.name,
            backend_impl=_class_path(AscendAttentionBackendImpl),
            builder_impl=group.builder,
            global_group_id=group.gid,
            layout=group.status.value,
            capture_descriptor=format_dflash_capture_descriptor(
                descriptor,
                max_num_reqs=max_num_reqs,
            ),
            capture_bucket=(num_reqs, descriptor.num_tokens),
            query_width=query_width,
            padded_request_count=num_reqs,
            padded_token_count=descriptor.num_tokens,
            static_k=static_k,
            window_size=runtime_state.capability.inventory.window_size,
            logical_block_size=group.scheduler_block_size,
            kernel_block_size=group.table_column_span,
            block_span_tokens=block_span_tokens,
            tensor_parallel_size=runtime_context.tensor_parallel_size,
            tensor_parallel_rank=runtime_context.tensor_parallel_rank,
            parallel_read_semantics="replicated",
            data_parallel_size=runtime_context.data_parallel_size,
            pipeline_parallel_size=runtime_context.pipeline_parallel_size,
            prefill_context_parallel_size=runtime_context.prefill_context_parallel_size,
            decode_context_parallel_size=runtime_context.decode_context_parallel_size,
            sequence_parallel=runtime_context.sequence_parallel,
            dsa_context_parallel=runtime_context.dsa_context_parallel,
            flashcomm=runtime_context.flashcomm,
            async_scheduling=runtime_context.async_scheduling,
            prefix_caching=runtime_context.prefix_caching,
        )
        provider.register_capture_identity(
            fingerprint=fingerprint,
            execution_lane=0,
            consumer_ordinal=0,
        )
    owner._draft_window_graph_session = session
    return session


def prepare_parallel_graph(
    owner: Any,
    descriptor: BatchExecutionDescriptor,
) -> DraftWindowGraphReadCarrier | None:
    """Bind the real P3 proposal generation before Graph metadata consumption."""
    session = getattr(owner, "_draft_window_graph_session", None)
    if session is None:
        return None
    if not isinstance(session, DraftWindowGraphSession) or session.owner is not owner:
        raise TypeError("invalid instance-owned Draft Window Graph session")
    if descriptor.cg_mode is not CUDAGraphMode.FULL:
        raise RuntimeError("runtime Draft Window Graph requires a FULL replay descriptor")
    inputs = getattr(owner, "_draft_window_live_inputs", None)
    if not isinstance(inputs, DraftMetadataInputs):
        raise RuntimeError("Draft Window Graph replay has no real P3 proposal generation")
    inputs.validate_for(inputs.identity)
    proposal_context = getattr(owner, "_draft_window_proposal_context", None)
    if proposal_context is None or proposal_context.mapping_epoch != inputs.identity.mapping_epoch:
        raise RuntimeError("Draft Window Graph proposal identity is stale")
    capture_identity = session.resolve_capture_identity(descriptor)
    fingerprint = capture_identity.fingerprint
    if inputs.padded_request_count != fingerprint.padded_request_count:
        raise RuntimeError("Draft Window Graph generation does not match the capture request bucket")
    target_metadata = owner.model_state.attn_metadata
    if not isinstance(target_metadata, Mapping) or not target_metadata:
        raise RuntimeError("Draft Window Graph requires real non-empty Target metadata")
    write_side = _build_write_side_reference(owner, capture_identity, target_metadata)

    from vllm_ascend.worker.v2.spec_decode.draft_window_graph_metadata import (
        validate_deferred_parallel_graph_metadata,
    )

    validate_deferred_parallel_graph_metadata(owner, inputs, descriptor)

    # FULL dispatch deferred its unused temporary metadata. Build and publish
    # exactly this generation once, backed by the persistent Graph read bank.
    carrier = session.provider.bind_read_carrier(
        capture_identity=capture_identity,
        inputs=inputs,
        metadata_builder=lambda read_inputs: _build_private_graph_metadata(
            owner,
            read_inputs=read_inputs,
            write_side=write_side,
            target_metadata=target_metadata,
            fingerprint=fingerprint,
        ),
        target_metadata=target_metadata,
        write_side_reference=write_side,
    )
    return carrier


def _bind_parallel_graph_device_lengths(
    metadata: Any,
    read_inputs: DraftWindowGraphReadInputs,
    query_width: int,
) -> None:
    """Bind the private FULL consumer after checking the fresh builder owners.

    The a982 builder uses CPU exact lengths while assembling its metadata and
    separately carries the device view as seq_lens_gpu. FULL FIA consumes
    seq_lens, so that field must use the persistent Graph device bank. This
    adapter runs only after Target isolation; no copy, sync, or projection is
    added and the eager/baseline builder selection is left unchanged.
    """
    expected = (
        ("block_tables", read_inputs.window_block_table),
        ("seq_lens", read_inputs.cpu_window_exact),
        ("seq_lens_cpu", read_inputs.cpu_window_exact),
        ("seq_lens_gpu", read_inputs.device_window_exact),
    )
    for name, source in expected:
        actual = getattr(metadata, name, None)
        actual_identity = _tensor_identity(actual) if isinstance(actual, torch.Tensor) else None
        expected_identity = _tensor_identity(source)
        if actual_identity != expected_identity:
            raise RuntimeError(
                f"private Graph builder lost its current {name} owner: "
                f"actual={actual_identity}, expected={expected_identity}"
            )
    query_lengths = [(i + 1) * query_width for i in range(read_inputs.padded_request_count)]
    if metadata.actual_seq_lengths_q != query_lengths:
        raise RuntimeError("private Graph builder changed its captured padded query layout")
    if metadata.seq_lens_list != read_inputs.cpu_window_exact.tolist():
        raise RuntimeError("private Graph builder changed its current exact length values")
    metadata.seq_lens = read_inputs.device_window_exact


def _build_private_graph_metadata(
    owner: Any,
    *,
    read_inputs: DraftWindowGraphReadInputs,
    write_side: DraftWindowGraphWriteSideReference,
    target_metadata: Mapping[Any, Any],
    fingerprint: DraftWindowGraphRouteFingerprint,
) -> DraftWindowGraphMetadataBuild:
    active_upper = read_inputs.cpu_window_upper[: read_inputs.actual_request_count]
    max_seq_len = int(active_upper.max().item()) if active_upper.numel() else 0
    group_id = fingerprint.global_group_id
    block_tables = tuple(
        read_inputs.window_block_table if gid == group_id else full_table[: read_inputs.padded_request_count]
        for gid, full_table in enumerate(owner.block_tables.input_block_tables)
    )
    typed_read_inputs = AscendDraftAttentionReadInputs(
        block_tables=block_tables,
        seq_lens=read_inputs.device_window_exact,
        seq_lens_cpu_exact=read_inputs.cpu_window_exact,
        seq_lens_cpu_upper_bound=read_inputs.cpu_window_upper,
        max_seq_len=max_seq_len,
    )
    with ExitStack() as stack:
        stack.enter_context(build_attn_metadata_wrapper())
        if fingerprint.route is DraftWindowGraphRoute.DSPARK:
            stack.enter_context(
                build_attn_metadata_factory(
                    owner.input_buffers.positions,
                    fingerprint.padded_token_count,
                    torch.from_numpy(owner.input_batch.is_prefilling_np),
                )
            )
        metadata = owner._build_ascend_draft_attn_metadata(
            # FULL attention consumes the captured padded Q, including PAD
            # slots. Keep that uniform layout in the builder itself, avoiding
            # a second synthetic request. Generation identity stays active-only.
            num_reqs=read_inputs.padded_request_count,
            num_reqs_padded=read_inputs.padded_request_count,
            num_tokens_padded=fingerprint.padded_token_count,
            read_inputs=typed_read_inputs,
            causal=write_side.causal,
            query_start_loc_np=None,
            dcp_local_seq_lens=None,
        )

    from vllm_ascend.worker.v2.spec_decode.draft_window_graph_metadata import copy_draft_attention_metadata

    metadata = copy_draft_attention_metadata(metadata)
    window_layers = owner._draft_window_runtime_state.capability.inventory.require_single_group_direct_prefix().layers
    bound_metadata_ids: set[int] = set()
    for layer_name in window_layers:
        layer_metadata = metadata.get(layer_name)
        if layer_metadata is not None and id(layer_metadata) not in bound_metadata_ids:
            _bind_parallel_graph_device_lengths(layer_metadata, read_inputs, fingerprint.query_width)
            bound_metadata_ids.add(id(layer_metadata))
    finalized_groups: dict[int, int | None] = {}
    native_swa = getattr(owner._draft_window_runtime_state.capability, "native_swa_graph", None)
    native_snapshot = getattr(owner, "_draft_window_native_full_snapshot", None) if native_swa is not None else None
    native_gids = (
        {} if native_swa is None else {name: group.gid for group in native_swa.native_groups for name in group.layers}
    )
    for layer_name, layer_metadata in metadata.items():
        if layer_metadata is None:
            continue
        gid = native_gids.get(layer_name)
        key = id(layer_metadata)
        if key in finalized_groups:
            if finalized_groups[key] != gid:
                raise RuntimeError("one Draft metadata object spans incompatible native groups")
            continue
        finalized_groups[key] = gid
        final_layer_metadata = layer_metadata
        final_layer_metadata.slot_mapping = write_side.slot_mapping
        final_layer_metadata.positions = write_side.positions
        final_layer_metadata.query_start_loc = write_side.query_start_loc
        final_layer_metadata.actual_seq_lengths_q = list(write_side.actual_seq_lengths_q)
        final_layer_metadata.attn_state = write_side.attn_state
        final_layer_metadata.num_actual_tokens = write_side.num_actual_tokens
        final_layer_metadata.causal = write_side.causal
        if layer_name in native_gids:
            gid = native_gids[layer_name]
            if native_snapshot is None:
                raise RuntimeError("native SWA private Graph metadata has no current full-length snapshot")
            native_snapshot.validate(owner._draft_window_live_inputs.identity, native_swa)
            final_layer_metadata.block_tables = owner.block_tables.input_block_tables[gid][
                : read_inputs.padded_request_count
            ]
            final_layer_metadata.seq_lens = native_snapshot.device_exact
            final_layer_metadata.seq_lens_cpu = native_snapshot.cpu_exact
            final_layer_metadata.seq_lens_list = native_snapshot.cpu_exact.tolist()
            final_layer_metadata.slot_mapping = owner.block_tables.slot_mappings[gid, : fingerprint.padded_token_count]
            final_layer_metadata.causal = True
            final_layer_metadata.attn_state = layer_metadata.attn_state
    return DraftWindowGraphMetadataBuild(metadata=metadata, native_full_snapshot=native_snapshot)


def _build_write_side_reference(
    owner: Any,
    capture_identity: DraftWindowGraphCaptureIdentity,
    target_metadata: Mapping[Any, Any],
) -> DraftWindowGraphWriteSideReference:
    fingerprint = capture_identity.fingerprint
    gid = fingerprint.global_group_id
    token_count = fingerprint.padded_token_count
    request_count = fingerprint.padded_request_count
    query_width = fingerprint.query_width
    target_layer = next((value for value in target_metadata.values() if value is not None), None)
    if target_layer is None:
        raise RuntimeError("Draft Window Graph Target metadata contains no layer state")
    causal = owner._group_causal
    if isinstance(causal, Mapping):
        causal = causal[gid]
    return DraftWindowGraphWriteSideReference(
        slot_mapping=owner.block_tables.slot_mappings[gid, :token_count],
        positions=owner.input_buffers.positions[:token_count],
        query_start_loc=owner.input_buffers.query_start_loc[: request_count + 1],
        actual_seq_lengths_q=tuple((row + 1) * query_width for row in range(request_count)),
        attn_state=getattr(target_layer, "attn_state", None),
        num_actual_tokens=token_count,
        causal=bool(causal),
    )


def _descriptor_request_count(
    descriptor: BatchExecutionDescriptor,
    *,
    max_num_reqs: int,
) -> int:
    return descriptor.num_reqs or min(descriptor.num_tokens, max_num_reqs)


def _class_path(value: type[Any]) -> str:
    return f"{value.__module__}.{value.__qualname__}"
