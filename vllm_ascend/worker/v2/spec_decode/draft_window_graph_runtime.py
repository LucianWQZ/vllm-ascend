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
from dataclasses import dataclass
from typing import Any

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

import vllm_ascend.envs as envs
from vllm_ascend.device.hardware_profile import AttentionBackendFamily, get_current_hardware_profile
from vllm_ascend.worker.v2.attn_utils import build_attn_metadata_factory, build_attn_metadata_wrapper
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
    DraftWindowGraphWriteSideReference,
    _tensor_identity,
    resolve_parallel_draft_window_graph_route,
)


@dataclass(slots=True)
class DraftWindowGraphSession:
    owner: Any
    provider: DraftWindowGraphBindingProvider
    route: DraftWindowGraphRoute

    def resolve_capture_identity(
        self,
        descriptor: BatchExecutionDescriptor,
    ) -> DraftWindowGraphCaptureIdentity:
        matches = tuple(
            identity
            for identity in self.provider.registered_capture_identities(route=self.route)
            if identity.descriptor == descriptor
        )
        if len(matches) != 1:
            raise RuntimeError(f"runtime descriptor does not resolve one captured consumer: {descriptor!r}")
        return matches[0]


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


def install_parallel_graph(owner: Any) -> DraftWindowGraphSession:
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
    session = DraftWindowGraphSession(
        owner=owner,
        provider=provider,
        route=route,
    )
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
    if proposal_context is not inputs.identity.proposal:
        raise RuntimeError("Draft Window Graph proposal identity is stale")
    capture_identity = session.resolve_capture_identity(descriptor)
    if inputs.identity.padded_request_count != capture_identity.descriptor.num_reqs:
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
        metadata_builder=lambda workspace: _build_private_graph_metadata(
            owner,
            workspace=workspace,
            actual_request_count=len(inputs.identity.proposal.ordered_request_ids),
            write_side=write_side,
            capture_identity=capture_identity,
        ),
        target_metadata=target_metadata,
        write_side_reference=write_side,
    )
    return carrier


def _bind_parallel_graph_device_lengths(
    metadata: Any,
    workspace: DraftWindowWorkspace,
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
        ("block_tables", workspace.window_block_table),
        ("seq_lens", workspace.cpu_window_exact),
        ("seq_lens_cpu", workspace.cpu_window_exact),
        ("seq_lens_gpu", workspace.device_window_exact),
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
    query_lengths = [(i + 1) * query_width for i in range(workspace.max_num_reqs)]
    if metadata.actual_seq_lengths_q != query_lengths:
        raise RuntimeError("private Graph builder changed its captured padded query layout")
    if metadata.seq_lens_list != workspace.cpu_window_exact.tolist():
        raise RuntimeError("private Graph builder changed its current exact length values")
    metadata.seq_lens = workspace.device_window_exact


def _build_private_graph_metadata(
    owner: Any,
    *,
    workspace: DraftWindowWorkspace,
    actual_request_count: int,
    write_side: DraftWindowGraphWriteSideReference,
    capture_identity: DraftWindowGraphCaptureIdentity,
) -> Mapping[str, Any]:
    typed_read_inputs = draft_window_read_inputs(
        workspace,
        actual_request_count=actual_request_count,
        padded_request_count=workspace.max_num_reqs,
        block_tables=owner.block_tables.input_block_tables,
        global_group_id=owner._draft_window_graph_binding_provider.source.group.gid,
    )
    with ExitStack() as stack:
        stack.enter_context(build_attn_metadata_wrapper())
        if capture_identity.route is DraftWindowGraphRoute.DSPARK:
            stack.enter_context(
                build_attn_metadata_factory(
                    owner.input_buffers.positions,
                    capture_identity.descriptor.num_tokens,
                    torch.from_numpy(owner.input_batch.is_prefilling_np),
                )
            )
        metadata = build_draft_window_attn_metadata(
            owner,
            # FULL attention consumes the captured padded Q, including PAD
            # slots. Keep that uniform layout in the builder itself, avoiding
            # a second synthetic request. Generation identity stays active-only.
            num_reqs=workspace.max_num_reqs,
            num_reqs_padded=workspace.max_num_reqs,
            num_tokens_padded=capture_identity.descriptor.num_tokens,
            read_inputs=typed_read_inputs,
            causal=write_side.causal,
            query_start_loc_np=None,
            dcp_local_seq_lens=None,
        )

    from vllm_ascend.worker.v2.spec_decode.draft_window_graph_metadata import copy_draft_attention_metadata

    metadata = copy_draft_attention_metadata(metadata)
    window_layers = owner._draft_window_graph_binding_provider.source.group.layers
    bound_metadata_ids: set[int] = set()
    for layer_name in window_layers:
        layer_metadata = metadata.get(layer_name)
        if layer_metadata is not None and id(layer_metadata) not in bound_metadata_ids:
            _bind_parallel_graph_device_lengths(
                layer_metadata,
                workspace,
                (capture_identity.descriptor.num_tokens // capture_identity.descriptor.num_reqs),
            )
            bound_metadata_ids.add(id(layer_metadata))
    for layer_metadata in metadata.values():
        if layer_metadata is None:
            continue
        layer_metadata.slot_mapping = write_side.slot_mapping
        layer_metadata.positions = write_side.positions
        layer_metadata.query_start_loc = write_side.query_start_loc
        layer_metadata.actual_seq_lengths_q = list(write_side.actual_seq_lengths_q)
        layer_metadata.attn_state = write_side.attn_state
        layer_metadata.num_actual_tokens = write_side.num_actual_tokens
        layer_metadata.causal = write_side.causal
    return metadata


def _build_write_side_reference(
    owner: Any,
    capture_identity: DraftWindowGraphCaptureIdentity,
    target_metadata: Mapping[Any, Any],
) -> DraftWindowGraphWriteSideReference:
    gid = owner._draft_window_graph_binding_provider.source.group.gid
    token_count = capture_identity.descriptor.num_tokens
    request_count = capture_identity.descriptor.num_reqs
    query_width = capture_identity.descriptor.num_tokens // capture_identity.descriptor.num_reqs
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
