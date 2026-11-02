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
from typing import Any

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

import vllm_ascend.envs as envs
from vllm_ascend.device.hardware_profile import AttentionBackendFamily, get_current_hardware_profile
from vllm_ascend.worker.v2.spec_decode.draft_window import (
    DraftMetadataInputs,
    DraftWindowPolicyDecision,
    DraftWindowSourceCapability,
    DraftWindowWorkspace,
    _build_draft_window_source_capability_for_context,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_eager import (
    build_draft_window_attn_metadata,
    draft_window_read_inputs,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_graph import (
    DraftWindowGraphBindingProvider,
    DraftWindowGraphCaptureIdentity,
    DraftWindowGraphReadCarrier,
    DraftWindowGraphRoute,
    resolve_parallel_draft_window_graph_route,
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


def build_parallel_graph_source(
    *,
    policy: DraftWindowPolicyDecision,
    runtime_route: str,
    attn_groups: Sequence[Sequence[Any]],
    block_tables: Any,
    kv_cache_config: Any,
    causal: bool | Mapping[int, bool],
    vllm_config: Any,
    dflash2_graph_contract: Any = None,
) -> DraftWindowSourceCapability:
    """Build P2/P3 source state only after the real Graph route is gated."""
    if not draft_window_graph_enabled():
        raise RuntimeError("internal Draft Window Graph flag is not enabled")
    validate_graph_context(vllm_config, block_tables)
    if runtime_route not in {"dflash", "dflash2", "dspark"}:
        raise NotImplementedError("Draft Window Graph supports only DFlash and eligible DSpark")
    if runtime_route == "dflash2":
        from vllm_ascend.worker.v2.spec_decode.draft_window_dflash2_graph import DFlash2GraphContract

        if not isinstance(dflash2_graph_contract, DFlash2GraphContract):
            raise NotImplementedError("DFlash2 Graph has no admitted model/selector/native reader contract")
    elif dflash2_graph_contract is not None:
        raise RuntimeError("DFlash2 source contract belongs to a different method")
    capability = _build_draft_window_source_capability_for_context(
        policy=policy,
        runtime_route=runtime_route,
        attn_groups=attn_groups,
        block_tables=block_tables,
        kv_cache_config=kv_cache_config,
        causal=causal,
        vllm_config=vllm_config,
        dflash2_graph_contract=dflash2_graph_contract,
    )
    return capability


def install_parallel_graph(owner: Any) -> DraftWindowGraphBindingProvider:
    """Install one provider and register every real startup capture bucket."""
    if hasattr(owner, "_draft_window_graph_session"):
        raise RuntimeError("Draft Window Graph is already installed on this speculator")
    provider = getattr(owner, "_draft_window_graph_binding_provider", None)
    if not isinstance(provider, DraftWindowGraphBindingProvider):
        raise RuntimeError("Draft Window Graph requires its initialized source owner")
    manager = getattr(owner, "query_cudagraph_manager", None)
    capture_descs = getattr(manager, "_capture_descs", None)
    if not isinstance(capture_descs, Mapping) or not capture_descs:
        raise RuntimeError("Draft Window Graph requires real manager capture descriptors")

    route = resolve_parallel_draft_window_graph_route(owner)
    if route not in (DraftWindowGraphRoute.DFLASH, DraftWindowGraphRoute.DSPARK):
        raise RuntimeError("Draft Window Graph has no parallel Draft route")
    descriptors = tuple(
        descriptor
        for mode_descriptors in capture_descs.values()
        for descriptor in mode_descriptors
        if descriptor.cg_mode is CUDAGraphMode.FULL
    )
    if not descriptors:
        raise RuntimeError("Draft Window Graph manager has no capture descriptors")
    descriptor_keys = tuple(
        (
            descriptor.cg_mode.name,
            descriptor.num_reqs,
            descriptor.num_tokens,
        )
        for descriptor in descriptors
    )
    if len(set(descriptor_keys)) != len(descriptor_keys):
        raise NotImplementedError("Draft Window Graph does not support ambiguous LoRA capture buckets")

    if int(owner.speculative_config.num_speculative_tokens) <= 0:
        raise ValueError("Draft Window Graph static K must be positive")
    for descriptor in descriptors:
        provider.register_capture_identity(
            route=route,
            descriptor=descriptor,
        )
    owner._draft_window_graph_session = provider
    return provider


def prepare_parallel_graph(
    owner: Any,
    descriptor: BatchExecutionDescriptor,
) -> DraftWindowGraphReadCarrier | None:
    """Bind the real P3 proposal generation before Graph metadata consumption."""
    provider = getattr(owner, "_draft_window_graph_session", None)
    if provider is None:
        return None
    if provider is not owner._draft_window_graph_binding_provider:
        raise TypeError("invalid instance-owned Draft Window Graph provider")
    if descriptor.cg_mode is not CUDAGraphMode.FULL:
        raise RuntimeError("runtime Draft Window Graph requires a FULL replay descriptor")
    inputs = provider._active_inputs
    if not isinstance(inputs, DraftMetadataInputs):
        raise RuntimeError("Draft Window Graph replay has no real P3 proposal generation")
    inputs.validate_for(inputs.identity)
    proposal_context = owner._draft_window_epoch_source._active
    if proposal_context is not inputs.identity.proposal:
        raise RuntimeError("Draft Window Graph proposal identity is stale")
    route = resolve_parallel_draft_window_graph_route(owner)
    assert route is not None
    capture_identity = provider.capture_identity_for(route=route, descriptor=descriptor)
    if inputs.identity.padded_request_count != capture_identity.descriptor.num_reqs:
        raise RuntimeError("Draft Window Graph generation does not match the capture request bucket")
    target_metadata = owner.model_state.attn_metadata
    if not isinstance(target_metadata, Mapping) or not target_metadata:
        raise RuntimeError("Draft Window Graph requires real non-empty Target metadata")

    if owner._draft_window_selected_metadata != descriptor or inputs.metadata is not None:
        raise RuntimeError("Draft Window replay does not own the selected unconsumed FULL dispatch")

    # FULL dispatch deferred its unused temporary metadata. Build and publish
    # exactly this generation once, backed by the persistent Graph read bank.
    carrier = provider.bind_read_carrier(
        capture_identity=capture_identity,
        inputs=inputs,
        metadata_builder=lambda workspace: _build_private_graph_metadata(
            owner,
            workspace=workspace,
            actual_request_count=len(inputs.identity.proposal.ordered_request_ids),
            capture_identity=capture_identity,
        ),
        target_metadata=target_metadata,
    )
    return carrier


def _build_private_graph_metadata(
    owner: Any,
    *,
    workspace: DraftWindowWorkspace,
    actual_request_count: int,
    capture_identity: DraftWindowGraphCaptureIdentity,
) -> Mapping[str, Any]:
    read_inputs = draft_window_read_inputs(
        workspace,
        actual_request_count=actual_request_count,
        padded_request_count=workspace.max_num_reqs,
    )
    metadata = build_draft_window_attn_metadata(
        owner,
        # FULL FIA includes the real capture padding in its uniform Q layout.
        num_reqs=workspace.max_num_reqs,
        num_reqs_padded=workspace.max_num_reqs,
        num_tokens_padded=capture_identity.descriptor.num_tokens,
        read_inputs=read_inputs,
        causal=owner._group_causal,
        query_start_loc_np=None,
    )
    for layer_name in owner._draft_window_graph_binding_provider.source.group.layers:
        # The normal builder uses CPU exact while preparing tiling. FULL FIA
        # capture/update additionally reads the persistent device length bank.
        metadata[layer_name].seq_lens = workspace.device_window_exact
    return metadata
