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

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass, is_dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, cast

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
    DraftWindowGraphMetadataBuild,
    DraftWindowGraphReadCarrier,
    DraftWindowGraphReadInputs,
    DraftWindowGraphRoute,
    DraftWindowGraphRouteFingerprint,
    DraftWindowGraphSourceEvent,
    DraftWindowGraphWriteSideReference,
    install_draft_window_graph_binding_provider,
    resolve_parallel_draft_window_graph_route,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_graph_oracle import (
    audit_characterization_snapshot as _audit_characterization_snapshot,
)


class DraftWindowGraphCharacterizationStage(Enum):
    CAPTURE = "capture"
    MATERIALIZED = "materialized"
    BOUND = "bound"
    LIVE = "live"
    REPLAY = "replay"
    CONSUMER = "consumer"
    UPDATE = "update"


class DraftWindowGraphOperandStage(Enum):
    CAPTURE_OPERATOR = "capture_operator"
    UPDATE_OPERATOR = "update_operator"


class DraftWindowGraphLifecycleStage(Enum):
    REPLAY_SUBMITTED = "replay_submitted"
    UPDATE_EVENT_RECORDED = "update_event_recorded"
    PROPOSAL_RETIRED = "proposal_retired"


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


@dataclass(frozen=True, slots=True)
class DraftWindowGraphOperandEvent:
    """Actual Python/operator-call-boundary operand evidence.

    This is intentionally not named a physical memory transaction.  The
    recorded values are the exact objects supplied to the Ascend FIA capture
    or task-update call; CANN/msprof evidence is still required for G3's
    device-level read-set claim.
    """

    sequence: int
    stage: DraftWindowGraphOperandStage
    evidence_level: str
    route: DraftWindowGraphRoute
    capture_descriptor: str
    generation_id: int | None
    mapping_epoch: int | None
    backend_impl: str
    consumer: str
    layer_name: str | None
    num_tokens: int
    expected_table_address: int
    captured_table_address: int
    actual_table_address: int
    length_kind: str
    length_address: int | None
    length_values: tuple[int, ...]
    handle_identity: int
    event_identity: int
    stream_identity: int


@dataclass(frozen=True, slots=True)
class DraftWindowGraphG4OracleRecord:
    sequence: int
    route: DraftWindowGraphRoute
    capture_descriptor: str
    generation_id: int
    mapping_epoch: int
    canonical_table_address: int
    canonical_table_values: tuple[tuple[int, ...], ...]
    window_table_address: int
    window_table_values: tuple[tuple[int, ...], ...]
    slot_mapping_address: int
    slot_mapping_values: tuple[int, ...]
    positions_address: int
    position_values: tuple[int, ...]
    query_layout_address: int
    query_layout_values: tuple[int, ...]
    target_metadata_identity: int
    target_layer_identities: tuple[tuple[str, int], ...]
    target_table_addresses: tuple[tuple[str, int | None], ...]
    target_length_values: tuple[tuple[str, tuple[int, ...]], ...]
    draft_output_hash: str | None
    target_output_hash: str | None


@dataclass(frozen=True, slots=True)
class DraftWindowGraphLifecycleEvent:
    sequence: int
    stage: DraftWindowGraphLifecycleStage
    generation_id: int
    mapping_epoch: int
    stream_identity: int | None
    event_identity: int | None
    event_complete: bool | None
    lease_state: str | None
    note: str


@dataclass(frozen=True, slots=True)
class DraftWindowGraphCaptureOperandContext:
    recorder: DraftWindowGraphCharacterizationRecorder
    identity: DraftWindowGraphCaptureIdentity


_CAPTURE_OPERAND_CONTEXT_ATTRIBUTE = "_draft_window_graph_capture_operand_context"


class DraftWindowGraphCharacterizationRecorder:
    """Experimental-only G1-G5 source observer; silent unless inspected."""

    def __init__(
        self,
        *,
        max_events: int = 512,
        artifact_path: str | None = None,
    ) -> None:
        if max_events <= 0:
            raise ValueError("characterization recorder capacity must be positive")
        self._max_events = max_events
        self._events: list[DraftWindowGraphCharacterizationEvent] = []
        self._operand_events: list[DraftWindowGraphOperandEvent] = []
        self._g4_records: list[DraftWindowGraphG4OracleRecord] = []
        self._lifecycle_events: list[DraftWindowGraphLifecycleEvent] = []
        self._completion_tokens: dict[int, Any] = {}
        self._provider: DraftWindowGraphBindingProvider | None = None
        self._sequence = 0
        self._artifact_path = None if artifact_path is None else Path(artifact_path).expanduser().resolve()
        if self._artifact_path is not None and not self._artifact_path.parent.is_dir():
            raise ValueError("characterization artifact parent directory does not exist")

    @property
    def events(self) -> tuple[DraftWindowGraphCharacterizationEvent, ...]:
        return tuple(self._events)

    @property
    def operand_events(self) -> tuple[DraftWindowGraphOperandEvent, ...]:
        return tuple(self._operand_events)

    @property
    def g4_records(self) -> tuple[DraftWindowGraphG4OracleRecord, ...]:
        return tuple(self._g4_records)

    @property
    def lifecycle_events(self) -> tuple[DraftWindowGraphLifecycleEvent, ...]:
        return tuple(self._lifecycle_events)

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
                sequence=self._next_sequence(),
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

    def record_capture_operand(
        self,
        *,
        identity: DraftWindowGraphCaptureIdentity,
        block_table: torch.Tensor,
        seq_lengths: Any,
        layer_name: str | None,
        num_tokens: int,
        handle: Any,
        event: Any,
        stream: Any,
        backend_impl: type[Any],
    ) -> None:
        expected_address = identity.backing_identity.window_block_table.data_ptr
        actual_address = _tensor_data_address(block_table)
        if actual_address != expected_address:
            raise RuntimeError("captured FIA block-table operand does not use the registered Window backing")
        length_kind, length_address, length_values = _length_operand_snapshot(seq_lengths)
        expected_lengths = tuple(
            self._require_provider().startup_capture_read_inputs(identity).cpu_window_exact.tolist()
        )
        if length_values and length_values[: len(expected_lengths)] != expected_lengths:
            raise RuntimeError("captured FIA length operand does not use the registered Window lengths")
        self._append_operand(
            DraftWindowGraphOperandEvent(
                sequence=self._next_sequence(),
                stage=DraftWindowGraphOperandStage.CAPTURE_OPERATOR,
                evidence_level="HOST_OPERATOR_BOUNDARY_FACT",
                route=identity.fingerprint.route,
                capture_descriptor=identity.fingerprint.capture_descriptor,
                generation_id=None,
                mapping_epoch=None,
                backend_impl=_class_path(backend_impl),
                consumer="npu_fused_infer_attention_score.out",
                layer_name=layer_name,
                num_tokens=num_tokens,
                expected_table_address=expected_address,
                captured_table_address=actual_address,
                actual_table_address=actual_address,
                length_kind=length_kind,
                length_address=length_address,
                length_values=length_values,
                handle_identity=id(handle),
                event_identity=id(event),
                stream_identity=id(stream),
            )
        )

    def record_update_operand(
        self,
        *,
        binding: Any,
        draft_metadata: Mapping[str, Any],
        layer_name: str,
        captured_block_table: torch.Tensor,
        actual_block_table: torch.Tensor,
        seq_lengths: Any,
        num_tokens: int,
        handle: Any,
        event: Any,
        stream: Any,
        backend_impl: type[Any],
    ) -> None:
        carriers = tuple(carrier for carrier in binding.carriers if carrier.metadata is draft_metadata)
        if len(carriers) != 1:
            raise RuntimeError("operator update metadata does not resolve one submitted Window carrier")
        carrier = carriers[0]
        runtime_identity = carrier.runtime_identity
        if runtime_identity is None:
            raise RuntimeError("operator update carrier has no runtime identity")
        expected_address = carrier.capture_identity.backing_identity.window_block_table.data_ptr
        captured_address = _tensor_data_address(captured_block_table)
        actual_address = _tensor_data_address(actual_block_table)
        if captured_address != expected_address or actual_address != expected_address:
            raise RuntimeError("updated FIA block-table operand does not use persistent Window backing")
        length_kind, length_address, length_values = _length_operand_snapshot(seq_lengths)
        expected_lengths = tuple(carrier.read_inputs.cpu_window_exact.tolist())
        if length_values and length_values[: len(expected_lengths)] != expected_lengths:
            raise RuntimeError("updated FIA length operand does not use runtime Window lengths")
        self._append_operand(
            DraftWindowGraphOperandEvent(
                sequence=self._next_sequence(),
                stage=DraftWindowGraphOperandStage.UPDATE_OPERATOR,
                evidence_level="HOST_OPERATOR_BOUNDARY_FACT",
                route=carrier.capture_identity.fingerprint.route,
                capture_descriptor=carrier.capture_identity.fingerprint.capture_descriptor,
                generation_id=runtime_identity.generation_id,
                mapping_epoch=runtime_identity.mapping_epoch,
                backend_impl=_class_path(backend_impl),
                consumer="npu_fused_infer_attention_score.out",
                layer_name=layer_name,
                num_tokens=num_tokens,
                expected_table_address=expected_address,
                captured_table_address=captured_address,
                actual_table_address=actual_address,
                length_kind=length_kind,
                length_address=length_address,
                length_values=length_values,
                handle_identity=id(handle),
                event_identity=id(event),
                stream_identity=id(stream),
            )
        )
        self.record_lifecycle(
            stage=DraftWindowGraphLifecycleStage.UPDATE_EVENT_RECORDED,
            generation_id=runtime_identity.generation_id,
            mapping_epoch=runtime_identity.mapping_epoch,
            stream=stream,
            event=event,
            lease_state=None,
            note="existing backend ExternalEvent recorded after graph task update",
        )

    def record_g4(
        self,
        *,
        owner: Any,
        carrier: DraftWindowGraphReadCarrier,
        target_metadata: Mapping[Any, Any],
    ) -> None:
        runtime_identity = carrier.runtime_identity
        reference = carrier.write_side_reference
        if runtime_identity is None or reference is None:
            raise RuntimeError("G4 record requires a bound carrier and full write reference")
        gid = carrier.capture_identity.fingerprint.global_group_id
        canonical_table = owner.block_tables.input_block_tables[gid]
        target_layer_ids: list[tuple[str, int]] = []
        target_table_addresses: list[tuple[str, int | None]] = []
        target_length_values: list[tuple[str, tuple[int, ...]]] = []
        for key, metadata in target_metadata.items():
            if metadata is None:
                continue
            layer_name = str(key)
            target_layer_ids.append((layer_name, id(metadata)))
            target_table = getattr(metadata, "block_tables", None)
            target_table_addresses.append(
                (
                    layer_name,
                    _tensor_data_address(target_table) if isinstance(target_table, torch.Tensor) else None,
                )
            )
            seq_lens = getattr(metadata, "seq_lens_list", None)
            if seq_lens is None:
                seq_lens = getattr(metadata, "seq_lens", None)
            _kind, _address, values = _length_operand_snapshot(seq_lens)
            target_length_values.append((layer_name, values))
        self._append_g4(
            DraftWindowGraphG4OracleRecord(
                sequence=self._next_sequence(),
                route=carrier.capture_identity.fingerprint.route,
                capture_descriptor=carrier.capture_identity.fingerprint.capture_descriptor,
                generation_id=runtime_identity.generation_id,
                mapping_epoch=runtime_identity.mapping_epoch,
                canonical_table_address=canonical_table.data_ptr(),
                canonical_table_values=_tensor_rows(canonical_table),
                window_table_address=carrier.read_inputs.window_block_table.data_ptr(),
                window_table_values=_tensor_rows(carrier.read_inputs.window_block_table),
                slot_mapping_address=reference.slot_mapping.data_ptr(),
                slot_mapping_values=_tensor_values(reference.slot_mapping),
                positions_address=reference.positions.data_ptr(),
                position_values=_tensor_values(reference.positions),
                query_layout_address=reference.query_start_loc.data_ptr(),
                query_layout_values=_tensor_values(reference.query_start_loc),
                target_metadata_identity=id(target_metadata),
                target_layer_identities=tuple(target_layer_ids),
                target_table_addresses=tuple(target_table_addresses),
                target_length_values=tuple(target_length_values),
                draft_output_hash=None,
                target_output_hash=None,
            )
        )

    def record_output_hashes(
        self,
        *,
        generation_id: int,
        draft_output: Any,
        target_output: Any | None = None,
    ) -> None:
        matches = [index for index, record in enumerate(self._g4_records) if record.generation_id == generation_id]
        if len(matches) != 1:
            raise RuntimeError("Graph output does not resolve one G4 generation record")
        index = matches[0]
        self._g4_records[index] = replace(
            self._g4_records[index],
            draft_output_hash=_hash_tensor_tree(draft_output),
            target_output_hash=(None if target_output is None else _hash_tensor_tree(target_output)),
        )
        self.write_artifact()

    def record_lifecycle(
        self,
        *,
        stage: DraftWindowGraphLifecycleStage,
        generation_id: int,
        mapping_epoch: int,
        stream: Any | None,
        event: Any | None,
        lease_state: str | None,
        note: str,
    ) -> None:
        event_identity = None if event is None else id(event)
        if event_identity is not None:
            self._completion_tokens[event_identity] = event
        self._append_lifecycle(
            DraftWindowGraphLifecycleEvent(
                sequence=self._next_sequence(),
                stage=stage,
                generation_id=generation_id,
                mapping_epoch=mapping_epoch,
                stream_identity=None if stream is None else id(stream),
                event_identity=event_identity,
                event_complete=_query_event_without_sync(event),
                lease_state=lease_state,
                note=note,
            )
        )

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe, evidence-level-labelled campaign artifact."""
        return {
            "schema": "mrv2-draft-window-graph-characterization-v2",
            "evidence_boundary": {
                "operand_events": "HOST_OPERATOR_BOUNDARY_FACT",
                "physical_memory_transaction": "NOT_PROVEN_REQUIRES_DEVICE_TOOL",
                "update_effective_generation": "NOT_PROVEN_REQUIRES_NPU_TIMELINE",
            },
            "events": [_json_safe(event) for event in self._events],
            "operand_events": [_json_safe(event) for event in self._operand_events],
            "g4_records": [_json_safe(record) for record in self._g4_records],
            "lifecycle_events": [self._lifecycle_snapshot(event) for event in self._lifecycle_events],
        }

    def write_artifact(self) -> None:
        if self._artifact_path is None:
            return
        self._artifact_path.write_text(
            json.dumps(self.snapshot(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _lifecycle_snapshot(self, event: DraftWindowGraphLifecycleEvent) -> dict[str, Any]:
        result = _json_safe(event)
        event_identity = event.event_identity
        token = None if event_identity is None else self._completion_tokens.get(event_identity)
        result["event_complete"] = _query_event_without_sync(token)
        return result

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
            sequence=self._next_sequence(),
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
        self._require_capacity()
        self._events.append(event)
        self.write_artifact()

    def _append_operand(self, event: DraftWindowGraphOperandEvent) -> None:
        self._require_capacity()
        self._operand_events.append(event)
        self.write_artifact()

    def _append_g4(self, record: DraftWindowGraphG4OracleRecord) -> None:
        self._require_capacity()
        self._g4_records.append(record)
        self.write_artifact()

    def _append_lifecycle(self, event: DraftWindowGraphLifecycleEvent) -> None:
        self._require_capacity()
        self._lifecycle_events.append(event)
        self.write_artifact()

    def _require_capacity(self) -> None:
        total = len(self._events) + len(self._operand_events) + len(self._g4_records) + len(self._lifecycle_events)
        if total >= self._max_events:
            raise RuntimeError("characterization recorder capacity is exhausted")

    def _next_sequence(self) -> int:
        sequence = self._sequence
        self._sequence += 1
        return sequence

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

    recorder = DraftWindowGraphCharacterizationRecorder(
        artifact_path=envs.VLLM_ASCEND_EXPERIMENTAL_DRAFT_WINDOW_GRAPH_ARTIFACT,
    )
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
        session.recorder.record_g4(
            owner=owner,
            carrier=carrier,
            target_metadata=target_metadata,
        )
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


def attach_parallel_graph_capture_operand_observer(
    provider: DraftWindowGraphBindingProvider,
    *,
    identity: DraftWindowGraphCaptureIdentity,
    metadata: Mapping[str, Any],
) -> None:
    """Attach a capture-scoped observer only to explicit Draft metadata."""
    recorder = provider.observer
    if not isinstance(recorder, DraftWindowGraphCharacterizationRecorder):
        return
    context = DraftWindowGraphCaptureOperandContext(
        recorder=recorder,
        identity=identity,
    )
    attached = 0
    for layer_metadata in metadata.values():
        if layer_metadata is None:
            continue
        if hasattr(layer_metadata, _CAPTURE_OPERAND_CONTEXT_ATTRIBUTE):
            raise RuntimeError("capture operand observer is already attached")
        setattr(layer_metadata, _CAPTURE_OPERAND_CONTEXT_ATTRIBUTE, context)
        attached += 1
    if attached == 0:
        raise RuntimeError("capture operand observer found no Draft attention metadata")


def record_parallel_graph_capture_operator_operand(
    metadata: Any,
    *,
    block_table: torch.Tensor,
    seq_lengths: Any,
    layer_name: str | None,
    num_tokens: int,
    handle: Any,
    event: Any,
    stream: Any,
    backend_impl: type[Any],
) -> None:
    context = getattr(metadata, _CAPTURE_OPERAND_CONTEXT_ATTRIBUTE, None)
    if context is None:
        return
    if not isinstance(context, DraftWindowGraphCaptureOperandContext):
        raise TypeError("invalid Draft Window capture operand observer context")
    context.recorder.record_capture_operand(
        identity=context.identity,
        block_table=block_table,
        seq_lengths=seq_lengths,
        layer_name=layer_name,
        num_tokens=num_tokens,
        handle=handle,
        event=event,
        stream=stream,
        backend_impl=backend_impl,
    )


def record_parallel_graph_update_operator_operand(
    binding: Any,
    *,
    draft_metadata: Mapping[str, Any],
    layer_name: str,
    captured_block_table: torch.Tensor,
    actual_block_table: torch.Tensor,
    seq_lengths: Any,
    num_tokens: int,
    handle: Any,
    event: Any,
    stream: Any,
    backend_impl: type[Any],
) -> None:
    if binding is None:
        return
    recorder = getattr(binding.provider, "observer", None)
    if not isinstance(recorder, DraftWindowGraphCharacterizationRecorder):
        return
    recorder.record_update_operand(
        binding=binding,
        draft_metadata=draft_metadata,
        layer_name=layer_name,
        captured_block_table=captured_block_table,
        actual_block_table=actual_block_table,
        seq_lengths=seq_lengths,
        num_tokens=num_tokens,
        handle=handle,
        event=event,
        stream=stream,
        backend_impl=backend_impl,
    )


def record_parallel_graph_replay_submission(owner: Any) -> None:
    """Record a real stream event after replay without synchronizing it."""
    session = getattr(owner, "_draft_window_graph_characterization", None)
    if session is None:
        return
    if not isinstance(session, DraftWindowGraphCharacterizationSession):
        raise TypeError("invalid Graph characterization session")
    carriers = session.provider.submitted_carriers
    if not carriers:
        raise RuntimeError("Graph replay submission has no submitted Window carrier")
    from vllm_ascend.worker.v2.spec_decode.draft_window_async import (
        DraftWindowNPUEventCompletionToken,
    )

    token = DraftWindowNPUEventCompletionToken.record_current_stream()
    stream = torch.npu.current_stream()
    for carrier in carriers:
        identity = carrier.runtime_identity
        if identity is None:
            raise RuntimeError("Graph replay submission carrier has no generation")
        session.recorder.record_lifecycle(
            stage=DraftWindowGraphLifecycleStage.REPLAY_SUBMITTED,
            generation_id=identity.generation_id,
            mapping_epoch=identity.mapping_epoch,
            stream=stream,
            event=token.event,
            lease_state=None,
            note="torch.npu.Event recorded after ACL Graph replay; no host synchronization",
        )


def record_parallel_graph_output(
    owner: Any,
    output: Any,
    *,
    target_output: Any | None = None,
) -> None:
    """Attach reproducible output hashes to the current G4 machine record.

    The host copy is characterization-only and therefore not performance
    representative.  The Draft graph manager can provide its output here;
    Target output remains ``None`` unless an external campaign driver supplies
    the real Target result explicitly.
    """
    session = getattr(owner, "_draft_window_graph_characterization", None)
    if session is None:
        return
    if not isinstance(session, DraftWindowGraphCharacterizationSession):
        raise TypeError("invalid Graph characterization session")
    carriers = session.provider.submitted_carriers
    identities = tuple(carrier.runtime_identity for carrier in carriers if carrier.runtime_identity is not None)
    if len(identities) != 1:
        raise RuntimeError("Graph output does not resolve one submitted generation")
    session.recorder.record_output_hashes(
        generation_id=identities[0].generation_id,
        draft_output=output,
        target_output=target_output,
    )


def record_parallel_graph_retirement(
    owner: Any,
    inputs: DraftMetadataInputs,
) -> None:
    """Record the existing proposal-owner retirement without redefining it."""
    session = getattr(owner, "_draft_window_graph_characterization", None)
    if session is None:
        return
    if not isinstance(session, DraftWindowGraphCharacterizationSession):
        raise TypeError("invalid Graph characterization session")
    session.recorder.record_lifecycle(
        stage=DraftWindowGraphLifecycleStage.PROPOSAL_RETIRED,
        generation_id=inputs.identity.generation_id,
        mapping_epoch=inputs.identity.mapping_epoch,
        stream=None,
        event=None,
        lease_state=inputs.lease.state.value,
        note=("host proposal owner retired the P3 lease; this is not device completion proof"),
    )


def audit_characterization_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Compatibility export for the standalone characterization audit."""
    return _audit_characterization_snapshot(snapshot)


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

    private_isolation_view = _isolate_draft_metadata_from_target(metadata, target_metadata)
    # Preserve the isolated view unchanged for the private-state oracle.  The
    # final metadata uses shallow copies of its layer objects so restoring the
    # canonical full write/query owners cannot retroactively weaken that view.
    final_metadata: dict[str, Any] = {}
    for layer_name, layer_metadata in private_isolation_view.items():
        if layer_metadata is None:
            final_metadata[layer_name] = None
            continue
        final_layer_metadata = copy.copy(layer_metadata)
        final_layer_metadata.slot_mapping = write_side.slot_mapping
        final_layer_metadata.positions = write_side.positions
        final_layer_metadata.query_start_loc = write_side.query_start_loc
        final_layer_metadata.actual_seq_lengths_q = list(write_side.actual_seq_lengths_q)
        final_layer_metadata.attn_state = write_side.attn_state
        final_layer_metadata.num_actual_tokens = write_side.num_actual_tokens
        final_layer_metadata.causal = write_side.causal
        final_metadata[layer_name] = final_layer_metadata
    return DraftWindowGraphMetadataBuild(
        metadata=final_metadata,
        private_isolation_view=private_isolation_view,
    )


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


def _tensor_data_address(tensor: torch.Tensor) -> int:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("graph operand must be a Tensor")
    return int(tensor.data_ptr())


def _length_operand_snapshot(value: Any) -> tuple[str, int | None, tuple[int, ...]]:
    if isinstance(value, torch.Tensor):
        # Do not insert a hot D2H synchronization at the operator boundary.
        # Tensor values are already captured from the typed carrier by the
        # MATERIALIZED/BOUND records and are correlated by storage address.
        return ("tensor", value.data_ptr(), ())
    if isinstance(value, (list, tuple)):
        if any(not isinstance(item, int) or isinstance(item, bool) for item in value):
            raise TypeError("host length operand must contain integers")
        return ("host_sequence", None, tuple(int(item) for item in value))
    raise TypeError(f"unsupported graph length operand type: {type(value).__name__}")


def _query_event_without_sync(event: Any | None) -> bool | None:
    if event is None:
        return None
    query = getattr(event, "query", None)
    if not callable(query):
        return None
    try:
        return bool(query())
    except RuntimeError:
        # A backend ExternalEvent may not be queryable on every CANN release.
        # The event identity is still useful, but completion remains an NPU
        # trace question rather than a fabricated False result.
        return None


def _hash_tensor_tree(value: Any) -> str:
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            host = item.detach().to(device="cpu").contiguous()
            digest.update(str(host.dtype).encode())
            digest.update(str(tuple(host.shape)).encode())
            digest.update(host.view(torch.uint8).numpy().tobytes())
            return
        if isinstance(item, (tuple, list)):
            digest.update(f"{type(item).__name__}:{len(item)}".encode())
            for child in item:
                update(child)
            return
        raise TypeError(f"unsupported graph output hash type: {type(item).__name__}")

    update(value)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _json_safe(asdict(cast(Any, value)))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"value is not JSON-safe: {type(value).__name__}")


def _class_path(value: type[Any]) -> str:
    return f"{value.__module__}.{value.__qualname__}"
