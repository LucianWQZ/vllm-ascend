# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Default-off MRV2 Draft Window ACL Graph characterization support.

This is not a product capability switch.  It wires the already source-proven
P6 carrier into the DFlash/eligible-DSpark FULL graph path so a controlled NPU
campaign can characterize captured addresses, read contents, write-side
invariants, and replay/update ordering.  Public Graph Window capability gates
remain unchanged when the strict experimental environment switch is absent.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

import vllm_ascend.envs as envs
from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
from vllm_ascend.device.hardware_profile import AttentionBackendFamily
from vllm_ascend.worker.v2.attn_utils import (
    build_attn_metadata_wrapper,
    build_draft_attn_metadata_factory,
)
from vllm_ascend.worker.v2.spec_decode.dflash.capture import (
    format_dflash_capture_descriptor,
)
from vllm_ascend.worker.v2.spec_decode.draft_window import (
    DraftMetadataInputs,
    DraftWindowLayoutCategory,
    DraftWindowPolicyDecision,
    DraftWindowRuntimeContext,
    DraftWindowRuntimeState,
    DraftWindowSourceCapability,
    _build_draft_window_source_capability_for_context,
    build_draft_window_runtime_context,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_eager import (
    AscendDraftAttentionReadInputs,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_graph import (
    DraftWindowGraphBindingProvider,
    DraftWindowGraphCaptureIdentity,
    DraftWindowGraphReadCarrier,
    DraftWindowGraphReadInputs,
    DraftWindowGraphRoute,
    DraftWindowGraphRouteFingerprint,
    DraftWindowGraphSourceEvent,
    DraftWindowGraphWriteSideReference,
    install_draft_window_graph_binding_provider,
    resolve_parallel_draft_window_graph_route,
)


class DraftWindowGraphCharacterizationStage(Enum):
    CAPTURE = "capture"
    MATERIALIZED = "materialized"
    BOUND = "bound"
    LIVE = "live"
    REPLAY = "replay"
    CONSUMER = "consumer"
    UPDATE = "update"


@dataclass(frozen=True, slots=True)
class DraftWindowGraphCharacterizationEvent:
    """Bounded in-memory evidence record for one explicitly enabled run."""

    sequence: int
    stage: DraftWindowGraphCharacterizationStage
    route: DraftWindowGraphRoute
    capture_descriptor: str | None
    generation_id: int | None
    mapping_epoch: int | None
    backing_table_address: int
    table_values: tuple[tuple[int, ...], ...]
    exact_lengths: tuple[int, ...]
    upper_lengths: tuple[int, ...]
    slot_mapping_address: int | None
    slot_mapping_values: tuple[int, ...]
    positions_address: int | None
    position_values: tuple[int, ...]
    query_layout_address: int | None
    query_layout_values: tuple[int, ...]


class DraftWindowGraphCharacterizationRecorder:
    """Experimental-only G1-G5 source observer; silent unless inspected."""

    def __init__(self, *, max_events: int = 512) -> None:
        if max_events <= 0:
            raise ValueError("characterization recorder capacity must be positive")
        self._max_events = max_events
        self._events: list[DraftWindowGraphCharacterizationEvent] = []
        self._provider: DraftWindowGraphBindingProvider | None = None

    @property
    def events(self) -> tuple[DraftWindowGraphCharacterizationEvent, ...]:
        return tuple(self._events)

    def attach_provider(self, provider: DraftWindowGraphBindingProvider) -> None:
        if self._provider is not None:
            raise RuntimeError("characterization recorder already owns a provider")
        self._provider = provider

    def record(self, event: DraftWindowGraphSourceEvent) -> None:
        """Implement the P6 observer protocol and snapshot the submitted carrier."""
        provider = self._require_provider()
        carriers = tuple(
            carrier
            for carrier in provider.submitted_carriers
            if carrier.runtime_identity is not None
            and carrier.runtime_identity.generation_id == event.identity.generation_id
        )
        if len(carriers) != 1:
            raise RuntimeError("characterization source event does not resolve one submitted carrier")
        carrier = carriers[0]
        reference = carrier.write_side_reference
        self._append(
            self._event_from_read_inputs(
                stage=DraftWindowGraphCharacterizationStage(event.stage.value),
                identity=carrier.capture_identity,
                read_inputs=carrier.read_inputs,
                generation_id=event.identity.generation_id,
                mapping_epoch=event.identity.mapping_epoch,
                write_side_reference=reference,
            )
        )

    def record_capture(
        self,
        *,
        identity: DraftWindowGraphCaptureIdentity,
        read_inputs: DraftWindowGraphReadInputs,
        slot_mapping: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
    ) -> None:
        """Record G1/G3/G4 after the actual patched capture helper builds metadata."""
        self._append(
            self._event_from_read_inputs(
                stage=DraftWindowGraphCharacterizationStage.CAPTURE,
                identity=identity,
                read_inputs=read_inputs,
                generation_id=None,
                mapping_epoch=None,
                write_side_reference=None,
                slot_mapping=slot_mapping,
                positions=positions,
                query_start_loc=query_start_loc,
            )
        )

    def record_materialized(
        self,
        *,
        route: DraftWindowGraphRoute,
        inputs: DraftMetadataInputs,
    ) -> None:
        """Record the real P3 generation before it is copied to capture backing."""
        if any(
            event.stage is DraftWindowGraphCharacterizationStage.MATERIALIZED
            and event.generation_id == inputs.identity.generation_id
            and event.mapping_epoch == inputs.identity.mapping_epoch
            for event in self._events
        ):
            return
        self._append(
            DraftWindowGraphCharacterizationEvent(
                sequence=len(self._events),
                stage=DraftWindowGraphCharacterizationStage.MATERIALIZED,
                route=route,
                capture_descriptor=None,
                generation_id=inputs.identity.generation_id,
                mapping_epoch=inputs.identity.mapping_epoch,
                backing_table_address=inputs.window_block_table.data_ptr(),
                table_values=_tensor_rows(inputs.window_block_table),
                exact_lengths=_tensor_values(inputs.cpu_window_exact),
                upper_lengths=_tensor_values(inputs.cpu_window_upper),
                slot_mapping_address=None,
                slot_mapping_values=(),
                positions_address=None,
                position_values=(),
                query_layout_address=None,
                query_layout_values=(),
            )
        )

    def record_bound(self, carrier: DraftWindowGraphReadCarrier) -> None:
        runtime_identity = carrier.runtime_identity
        if runtime_identity is None:
            raise RuntimeError("bound characterization carrier has no runtime identity")
        self._append(
            self._event_from_read_inputs(
                stage=DraftWindowGraphCharacterizationStage.BOUND,
                identity=carrier.capture_identity,
                read_inputs=carrier.read_inputs,
                generation_id=runtime_identity.generation_id,
                mapping_epoch=runtime_identity.mapping_epoch,
                write_side_reference=carrier.write_side_reference,
            )
        )

    def _event_from_read_inputs(
        self,
        *,
        stage: DraftWindowGraphCharacterizationStage,
        identity: DraftWindowGraphCaptureIdentity,
        read_inputs: DraftWindowGraphReadInputs,
        generation_id: int | None,
        mapping_epoch: int | None,
        write_side_reference: DraftWindowGraphWriteSideReference | None,
        slot_mapping: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
    ) -> DraftWindowGraphCharacterizationEvent:
        if write_side_reference is not None:
            slot_mapping = write_side_reference.slot_mapping
            query_start_loc = write_side_reference.query_start_loc
            positions = write_side_reference.positions
        return DraftWindowGraphCharacterizationEvent(
            sequence=len(self._events),
            stage=stage,
            route=identity.fingerprint.route,
            capture_descriptor=identity.fingerprint.capture_descriptor,
            generation_id=generation_id,
            mapping_epoch=mapping_epoch,
            backing_table_address=read_inputs.window_block_table.data_ptr(),
            table_values=_tensor_rows(read_inputs.window_block_table),
            exact_lengths=_tensor_values(read_inputs.cpu_window_exact),
            upper_lengths=_tensor_values(read_inputs.cpu_window_upper),
            slot_mapping_address=None if slot_mapping is None else slot_mapping.data_ptr(),
            slot_mapping_values=() if slot_mapping is None else _tensor_values(slot_mapping),
            positions_address=None if positions is None else positions.data_ptr(),
            position_values=() if positions is None else _tensor_values(positions),
            query_layout_address=None if query_start_loc is None else query_start_loc.data_ptr(),
            query_layout_values=() if query_start_loc is None else _tensor_values(query_start_loc),
        )

    def _append(self, event: DraftWindowGraphCharacterizationEvent) -> None:
        if len(self._events) >= self._max_events:
            raise RuntimeError("characterization recorder capacity is exhausted")
        self._events.append(event)

    def _require_provider(self) -> DraftWindowGraphBindingProvider:
        if self._provider is None:
            raise RuntimeError("characterization recorder has no provider")
        return self._provider


@dataclass(slots=True)
class DraftWindowGraphCharacterizationSession:
    owner: Any
    provider: DraftWindowGraphBindingProvider
    recorder: DraftWindowGraphCharacterizationRecorder
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
                "runtime graph descriptor does not resolve one characterization carrier: "
                f"descriptor={descriptor_text!r} matches={len(matches)}"
            )
        return matches[0]

    def observe_materialized(self, inputs: DraftMetadataInputs) -> None:
        self.recorder.record_materialized(route=self.route, inputs=inputs)


def experimental_draft_window_graph_characterization_enabled() -> bool:
    return bool(envs.VLLM_ASCEND_EXPERIMENTAL_DRAFT_WINDOW_GRAPH)


def validate_base_graph_characterization_context(
    runtime_context: DraftWindowRuntimeContext,
) -> None:
    """Accept only the single-card dense base route selected for G1-G5."""
    if runtime_context.attention_backend_family is not AttentionBackendFamily.STANDARD:
        raise NotImplementedError("base Graph characterization requires the STANDARD hardware family")
    if runtime_context.graph_mode == "NONE":
        raise NotImplementedError("base Graph characterization requires an active Graph mode")
    if runtime_context.graph_mode not in {"FULL", "FULL_DECODE_ONLY"}:
        raise NotImplementedError("base Graph characterization is limited to FULL/FULL_DECODE_ONLY")
    if runtime_context.async_scheduling:
        raise NotImplementedError("base Graph characterization does not include async scheduling")
    topology = (
        runtime_context.tensor_parallel_size,
        runtime_context.data_parallel_size,
        runtime_context.pipeline_parallel_size,
        runtime_context.prefill_context_parallel_size,
        runtime_context.decode_context_parallel_size,
    )
    if topology != (1, 1, 1, 1, 1):
        raise NotImplementedError("base Graph characterization requires TP=1, DP=1, PP=1, PCP=1, DCP=1")
    if runtime_context.sequence_parallel:
        raise NotImplementedError("base Graph characterization does not include sequence parallel")
    if runtime_context.dsa_context_parallel:
        raise NotImplementedError("base Graph characterization does not include DSA-CP")
    if runtime_context.flashcomm:
        raise NotImplementedError("base Graph characterization does not include FlashComm")
    if runtime_context.prefix_caching:
        raise NotImplementedError("base Graph characterization does not include prefix caching")


def build_parallel_draft_window_graph_characterization_source(
    *,
    policy: DraftWindowPolicyDecision,
    runtime_route: str,
    attn_groups: Sequence[Sequence[Any]],
    block_tables: Any,
    causal: bool | Mapping[int, bool],
    vllm_config: Any,
) -> tuple[DraftWindowSourceCapability, DraftWindowRuntimeContext]:
    """Build P2/P3 source state only after the real Graph route is gated."""
    if not experimental_draft_window_graph_characterization_enabled():
        raise RuntimeError("internal Graph characterization flag is not enabled")
    runtime_context = build_draft_window_runtime_context(vllm_config, block_tables)
    validate_base_graph_characterization_context(runtime_context)
    if runtime_route not in {"dflash", "dspark"}:
        raise NotImplementedError("base Graph characterization supports only DFlash and eligible DSpark")
    speculative_config = vllm_config.speculative_config
    if bool(speculative_config.uses_dynamic_speculative_decoding()):
        raise NotImplementedError("base Graph characterization requires static K")

    # The exact Graph context was validated above.  Reuse the frozen P2/P7-X
    # source inventory by masking only the already-accounted graph dimension.
    source_context = replace(runtime_context, graph_mode="NONE")
    capability = _build_draft_window_source_capability_for_context(
        policy=policy,
        runtime_route=runtime_route,
        attn_groups=attn_groups,
        block_tables=block_tables,
        causal=causal,
        vllm_config=vllm_config,
        runtime_context=source_context,
    )
    group = capability.inventory.require_single_group_direct_prefix()
    if group.layout_category is not DraftWindowLayoutCategory.DENSE_GQA_DIRECT_PREFIX:
        raise NotImplementedError("base Graph characterization requires dense GQA/FIA direct-prefix metadata")
    topology = capability.eager_topology_fingerprint
    if topology is None:
        raise RuntimeError("base Graph characterization source has no topology fingerprint")
    capability = replace(
        capability,
        eager_topology_fingerprint=replace(
            topology,
            graph_mode=runtime_context.graph_mode,
        ),
    )
    return capability, runtime_context


def maybe_install_parallel_draft_window_graph_characterization(
    owner: Any,
    *,
    runtime_context: DraftWindowRuntimeContext,
) -> DraftWindowGraphCharacterizationSession | None:
    if not experimental_draft_window_graph_characterization_enabled():
        return None
    return install_parallel_draft_window_graph_characterization(
        owner,
        runtime_context=runtime_context,
    )


def install_parallel_draft_window_graph_characterization(
    owner: Any,
    *,
    runtime_context: DraftWindowRuntimeContext,
) -> DraftWindowGraphCharacterizationSession:
    """Install one provider and register every real startup capture bucket."""
    validate_base_graph_characterization_context(runtime_context)
    if hasattr(owner, "_draft_window_graph_characterization"):
        raise RuntimeError("Graph characterization is already installed on this speculator")
    runtime_state = getattr(owner, "_draft_window_runtime_state", None)
    if not isinstance(runtime_state, DraftWindowRuntimeState):
        raise RuntimeError("Graph characterization requires initialized P3 runtime state")
    manager = getattr(owner, "query_cudagraph_manager", None)
    capture_descs = getattr(manager, "_capture_descs", None)
    max_num_reqs = getattr(manager, "max_num_reqs", None)
    if not isinstance(capture_descs, Mapping) or not capture_descs:
        raise RuntimeError("Graph characterization requires real manager capture descriptors")
    if not isinstance(max_num_reqs, int) or max_num_reqs <= 0:
        raise RuntimeError("Graph characterization manager has no valid request capacity")

    recorder = DraftWindowGraphCharacterizationRecorder()
    provider = install_draft_window_graph_binding_provider(owner, observer=recorder)
    recorder.attach_provider(provider)
    route = resolve_parallel_draft_window_graph_route(owner)
    if route not in (DraftWindowGraphRoute.DFLASH, DraftWindowGraphRoute.DSPARK):
        raise RuntimeError("Graph characterization has no parallel Draft route")
    session = DraftWindowGraphCharacterizationSession(
        owner=owner,
        provider=provider,
        recorder=recorder,
        runtime_context=runtime_context,
        route=route,
    )
    descriptors = tuple(descriptor for mode_descriptors in capture_descs.values() for descriptor in mode_descriptors)
    if not descriptors:
        raise RuntimeError("Graph characterization manager has no capture descriptors")
    descriptor_keys = tuple(
        (
            descriptor.cg_mode.name,
            _descriptor_request_count(descriptor, max_num_reqs=max_num_reqs),
            descriptor.num_tokens,
        )
        for descriptor in descriptors
    )
    if len(set(descriptor_keys)) != len(descriptor_keys):
        raise NotImplementedError("base Graph characterization does not support ambiguous LoRA capture buckets")

    group = runtime_state.capability.inventory.require_single_group_direct_prefix()
    block_span_tokens = group.block_span_tokens
    if not isinstance(block_span_tokens, int) or isinstance(block_span_tokens, bool) or block_span_tokens <= 0:
        raise RuntimeError("Graph characterization group has no positive block-span contract")
    static_k = int(owner.speculative_config.num_speculative_tokens)
    if static_k <= 0:
        raise ValueError("Graph characterization static K must be positive")
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
            builder_impl=_class_path(group.builder_class),
            global_group_id=group.global_group_id,
            layout=group.layout_category.value,
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
            logical_block_size=group.logical_block_size,
            kernel_block_size=group.kernel_block_size,
            block_span_tokens=block_span_tokens,
            tensor_parallel_size=runtime_context.tensor_parallel_size,
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
    owner._draft_window_graph_characterization = session
    return session


def prepare_parallel_draft_window_graph_characterization(
    owner: Any,
    descriptor: BatchExecutionDescriptor,
) -> DraftWindowGraphReadCarrier | None:
    """Bind the real P3 proposal generation before Graph metadata consumption."""
    session = getattr(owner, "_draft_window_graph_characterization", None)
    if session is None:
        return None
    if not isinstance(session, DraftWindowGraphCharacterizationSession) or session.owner is not owner:
        raise TypeError("invalid instance-owned Graph characterization session")
    if descriptor.cg_mode is not CUDAGraphMode.FULL:
        raise RuntimeError("runtime Graph characterization requires a FULL replay descriptor")
    inputs = getattr(owner, "_draft_window_live_inputs", None)
    if not isinstance(inputs, DraftMetadataInputs):
        raise RuntimeError("Graph characterization replay has no real P3 proposal generation")
    inputs.validate_for(inputs.identity)
    proposal_context = getattr(owner, "_draft_window_proposal_context", None)
    if proposal_context is None or proposal_context.mapping_epoch != inputs.identity.mapping_epoch:
        raise RuntimeError("Graph characterization proposal identity is stale")
    capture_identity = session.resolve_capture_identity(descriptor)
    fingerprint = capture_identity.fingerprint
    if inputs.padded_request_count != fingerprint.padded_request_count:
        raise RuntimeError("Graph characterization generation does not match the capture request bucket")
    session.observe_materialized(inputs)
    target_metadata = owner.model_state.attn_metadata
    if not isinstance(target_metadata, Mapping) or not target_metadata:
        raise RuntimeError("Graph characterization requires real non-empty Target metadata")
    write_side = _build_write_side_reference(owner, capture_identity, target_metadata)

    # The normal DFlash path rebuilt and published eager metadata immediately
    # before run_fullgraph.  It is unused by FULL replay.  Replace that exact
    # generation entry with Draft-private metadata backed by the persistent
    # graph carrier; never materialize a second generation or advance epoch.
    owner._draft_window_runtime_state.metadata_store.discard(inputs.identity)
    try:
        carrier = session.provider.bind_read_carrier(
            capture_identity=capture_identity,
            inputs=inputs,
            proposal_id=inputs.identity.mapping_epoch,
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
        session.recorder.record_bound(carrier)
        return carrier
    except BaseException:
        session.provider.discard_bound_carriers()
        raise


def observe_parallel_graph_materialization(
    owner: Any,
    inputs: DraftMetadataInputs,
) -> bool:
    """Record candidate materialization and suppress premature P6 observation."""
    session = getattr(owner, "_draft_window_graph_characterization", None)
    if session is None:
        return False
    if not isinstance(session, DraftWindowGraphCharacterizationSession):
        raise TypeError("invalid Graph characterization session")
    session.observe_materialized(inputs)
    return True


def record_parallel_graph_capture(
    provider: DraftWindowGraphBindingProvider,
    *,
    identity: DraftWindowGraphCaptureIdentity,
    read_inputs: DraftWindowGraphReadInputs,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> None:
    recorder = provider.observer
    if isinstance(recorder, DraftWindowGraphCharacterizationRecorder):
        recorder.record_capture(
            identity=identity,
            read_inputs=read_inputs,
            slot_mapping=slot_mapping,
            positions=positions,
            query_start_loc=query_start_loc,
        )


def _build_private_graph_metadata(
    owner: Any,
    *,
    read_inputs: DraftWindowGraphReadInputs,
    write_side: DraftWindowGraphWriteSideReference,
    target_metadata: Mapping[Any, Any],
    fingerprint: DraftWindowGraphRouteFingerprint,
) -> Mapping[str, Any]:
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
                build_draft_attn_metadata_factory(
                    owner.input_buffers.positions,
                    fingerprint.padded_token_count,
                    torch.from_numpy(owner.input_batch.is_prefilling_np),
                )
            )
        metadata = owner._build_ascend_draft_attn_metadata(
            num_reqs=read_inputs.actual_request_count,
            num_reqs_padded=read_inputs.padded_request_count,
            num_tokens_padded=fingerprint.padded_token_count,
            read_inputs=typed_read_inputs,
            causal=write_side.causal,
            query_start_loc_np=None,
            dcp_local_seq_lens=None,
        )

    # Use the same deep isolation routine as eager P4, then explicitly restore
    # only the immutable full write/query carriers frozen by P6.
    from vllm_ascend.worker.v2.spec_decode.draft_window_eager import (
        _isolate_draft_metadata_from_target,
    )

    metadata = _isolate_draft_metadata_from_target(metadata, target_metadata)
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
    fingerprint = capture_identity.fingerprint
    gid = fingerprint.global_group_id
    token_count = fingerprint.padded_token_count
    request_count = fingerprint.padded_request_count
    query_width = fingerprint.query_width
    target_layer = next((value for value in target_metadata.values() if value is not None), None)
    if target_layer is None:
        raise RuntimeError("Graph characterization Target metadata contains no layer state")
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


def _tensor_values(tensor: torch.Tensor) -> tuple[int, ...]:
    # Characterization is explicitly opt-in and profiler-accounted.  This
    # host snapshot intentionally synchronizes device tensors to establish a
    # CPU oracle; it is never present in the default product path.
    return tuple(int(value) for value in tensor.detach().to(device="cpu").reshape(-1).tolist())


def _tensor_rows(tensor: torch.Tensor) -> tuple[tuple[int, ...], ...]:
    host = tensor.detach().to(device="cpu")
    if host.ndim != 2:
        raise ValueError("characterization table snapshot must be two-dimensional")
    return tuple(tuple(int(value) for value in row.tolist()) for row in host)


def _class_path(value: type[Any]) -> str:
    return f"{value.__module__}.{value.__qualname__}"
