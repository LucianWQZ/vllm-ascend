# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Ownership transfer for the frozen Eagle3 Graph per-consumer bank design.

P3 is staging scratch. P6 owns separate persistent first-pass/AR banks. The
scratch's final reader is the ordered copy into a bank, not the later FIA.
Releasing scratch after that copy has been enqueued relies on same-stream FIFO;
it neither releases the destination bank nor reports device completion. The
normal Graph replay/UPDATE/event contract separately orders bank reuse.

This owner is not installed by default and does not open any Graph gate.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch

from vllm_ascend.worker.v2.spec_decode.draft_window import DraftMetadataInputs, DraftWindowRuntimeState
from vllm_ascend.worker.v2.spec_decode.draft_window_graph import (
    DraftWindowGraphBindingProvider,
    DraftWindowGraphCaptureIdentity,
    DraftWindowGraphCarrierState,
    DraftWindowGraphReadCarrier,
    DraftWindowGraphRoute,
    DraftWindowGraphWriteSideReference,
)


def _current_stream_identity() -> int:
    # Host stream identity only: no event, query, synchronize or tensor read.
    return int(torch.npu.current_stream().npu_stream)


class Eagle3GraphStagingOwner:
    """One proposal's scratch-to-bank ownership and conservative Host guard.

    A destination remains BOUND until the real manager consumes the full
    route, then SUBMITTED until the actual backend UPDATE validates it. This
    class never fabricates either transition or a native completion event.
    """

    def __init__(
        self,
        runtime: DraftWindowRuntimeState,
        provider: DraftWindowGraphBindingProvider,
        *,
        static_k: int,
        current_stream_identity: Callable[[], int] = _current_stream_identity,
    ) -> None:
        if type(static_k) is not int or static_k < 1:
            raise ValueError("Eagle3 Graph requires a positive static K")
        if runtime.capability.inventory.method != "eagle3" or provider._runtime_state is not runtime:
            raise ValueError("Eagle3 Graph staging requires its own P3/P6 Eagle3 owners")
        if runtime.capability.provider.expected_future_offset != static_k:
            raise ValueError("Eagle3 static K differs from the canonical source contract")
        self.runtime, self.provider, self.static_k = runtime, provider, static_k
        self._stream_identity = current_stream_identity
        self._proposal_epoch: int | None = None
        self._stream: int | None = None
        self._owner_stream: int | None = None
        self._last_epoch = -1
        self._request_fingerprint: tuple[str, ...] | None = None
        self._next_step = 0
        self._expected_steps = static_k
        self._first_pass_is_graph = True
        self._carriers: list[DraftWindowGraphReadCarrier] = []
        self._quarantined = False
        self._quarantine_reason: str | None = None

    @property
    def quarantined(self) -> bool:
        return self._quarantined

    def begin(
        self,
        mapping_epoch: int,
        ordered_request_ids: tuple[str, ...],
        *,
        first_pass_is_graph: bool = True,
        num_speculative_tokens: int | None = None,
    ) -> None:
        from vllm_ascend.worker.v2.spec_decode.draft_window_eagle3 import _eagle3_graph_scheduled_consumers

        expected_steps = _eagle3_graph_scheduled_consumers(self.static_k, num_speculative_tokens)
        self.require_reusable()
        if self._proposal_epoch is not None:
            raise RuntimeError("Eagle3 Graph proposal already owns its banks")
        if type(mapping_epoch) is not int or mapping_epoch <= self._last_epoch:
            raise ValueError("Eagle3 proposal mapping epoch must strictly increase")
        if type(first_pass_is_graph) is not bool:
            raise TypeError("Eagle3 first-pass dispatch must be an explicit boolean")
        if (
            not isinstance(ordered_request_ids, tuple)
            or not ordered_request_ids
            or any(not isinstance(r, str) or not r for r in ordered_request_ids)
            or len(set(ordered_request_ids)) != len(ordered_request_ids)
        ):
            raise ValueError("Eagle3 Graph needs unique ordered request identities")
        stream = self._stream_identity()
        if type(stream) is not int or stream < 0:
            raise ValueError("Eagle3 Graph requires a concrete current stream identity")
        if self._owner_stream is not None and stream != self._owner_stream:
            self.quarantine("current stream changed across proposals without an explicit consumer dependency")
            raise RuntimeError("Eagle3 Graph backing is quarantined after cross-proposal stream change")
        self._proposal_epoch, self._request_fingerprint = mapping_epoch, ordered_request_ids
        self._owner_stream = self._stream = stream
        self._last_epoch = mapping_epoch
        self._next_step = 0
        self._expected_steps = expected_steps
        self._first_pass_is_graph = first_pass_is_graph
        self._carriers = []

    def adopt_eager_first_pass(self, identity: Any) -> None:
        """Account for the real nonuniform-prefill eager first pass.

        FULL_DECODE_ONLY legitimately dispatches this first pass eagerly while
        retaining a merged AR Graph. This attests only the existing serialized
        P3 submission order, not Graph capture, an eager fallback for a missing
        bucket, or native completion. The real dispatch caller must select it.
        """
        self.require_reusable()
        try:
            self._require_stream()
            if (
                self._first_pass_is_graph
                or self._next_step != 0
                or self.runtime._last_retired_identity != identity
                or identity.step != 0
                or identity.normalized_method != "eagle3"
                or identity.mapping_epoch != self._proposal_epoch
                or identity.ordered_request_fingerprint != self._request_fingerprint
            ):
                raise RuntimeError("Eagle3 eager first pass does not match the actual retired P3 submission")
            self._next_step = 1
        except BaseException as error:
            self.quarantine(type(error).__name__ + ": " + str(error))
            raise

    def require_reusable(self) -> None:
        if self._quarantined or self.runtime.quarantined_identity is not None:
            raise RuntimeError("Eagle3 Graph backing is quarantined; process isolation required")

    def _require_stream(self) -> None:
        if self._stream_identity() != self._stream:
            raise RuntimeError("Eagle3 staging stream changed within one proposal")

    def bind_and_release_scratch(
        self,
        *,
        capture_identity: DraftWindowGraphCaptureIdentity,
        inputs: DraftMetadataInputs,
        metadata_builder: Callable[[Any], Any],
        target_metadata: Mapping[str, Any],
        write_side_reference: DraftWindowGraphWriteSideReference,
    ) -> None:
        """Copy all read state into its registered bank, retaining that bank.

        Every later scratch materialization must stay on the same current
        stream. CPU lengths have owning copies in P6; device copies precede
        subsequent scratch writes in that stream. No device completion is
        inferred from Python returning or P3's Host lease retirement.
        """
        self.require_reusable()
        try:
            self._require_stream()
            identity = inputs.identity
            inputs.validate_for(identity)
            if (
                self._proposal_epoch is None
                or identity.mapping_epoch != self._proposal_epoch
                or identity.ordered_request_fingerprint != self._request_fingerprint
                or identity.normalized_method != "eagle3"
                or identity.step != self._next_step
                or identity.step >= self._expected_steps
            ):
                raise RuntimeError("Eagle3 staged generation does not match this proposal/step")
            first = identity.step == 0
            if first and not self._first_pass_is_graph:
                raise RuntimeError("eager first-pass dispatch cannot manufacture a Graph bank")
            route = DraftWindowGraphRoute.EAGLE3_FIRST_PASS if first else DraftWindowGraphRoute.EAGLE3_AR_STEP
            ordinal = 0 if first else identity.step - 1
            fingerprint = capture_identity.fingerprint
            if (
                fingerprint.route is not route
                or capture_identity.consumer_ordinal != ordinal
                or fingerprint.static_k != self.static_k
                or fingerprint.query_width != (self.static_k + 1 if first else 1)
                or fingerprint.graph_mode != "FULL"
            ):
                raise RuntimeError("Eagle3 destination does not identify this exact consumer ordinal")
            carrier = self.provider.bind_read_carrier(
                capture_identity=capture_identity,
                inputs=inputs,
                proposal_id=self._proposal_epoch,
                metadata_builder=metadata_builder,
                target_metadata=target_metadata,
                write_side_reference=write_side_reference,
            )
            if carrier.state is not DraftWindowGraphCarrierState.BOUND:
                raise RuntimeError("Eagle3 scratch handoff has no bound persistent destination")
            # The provider made owning CPU and current-stream device copies.
            # Check all five buffers again at the transfer boundary. A single
            # alias would let later staging overwrite a prior Graph consumer.
            for name in (
                "window_block_table",
                "valid_num_blocks",
                "device_window_exact",
                "cpu_window_exact",
                "cpu_window_upper",
            ):
                source, destination = getattr(inputs, name), getattr(carrier.read_inputs, name)
                if source.untyped_storage().data_ptr() == destination.untyped_storage().data_ptr():
                    raise RuntimeError("Eagle3 persistent read bank aliases reusable staging scratch")
            self._require_stream()
            self.runtime.retire_generation(identity)
            self._carriers.append(carrier)
            self._next_step += 1
        except BaseException as error:
            self.quarantine(type(error).__name__ + ": " + str(error))
            raise

    def finish(self) -> None:
        """End Host submission bookkeeping, never announce device completion."""
        self.require_reusable()
        try:
            self._require_stream()
            if self._proposal_epoch is None or self._next_step != self._expected_steps:
                raise RuntimeError("Eagle3 Graph proposal omitted a real first-pass/AR consumer")
            if any(carrier.state is not DraftWindowGraphCarrierState.UPDATED for carrier in self._carriers):
                raise RuntimeError("Eagle3 Graph banks lack actual backend UPDATE attribution")
            self._proposal_epoch = self._request_fingerprint = self._stream = None
            self._carriers = []
            self._expected_steps = self.static_k
        except BaseException as error:
            self.quarantine(type(error).__name__ + ": " + str(error))
            raise

    def quarantine(self, reason: str) -> None:
        """Host-only isolation: retain scratch/banks, prohibit the next scope."""
        self._quarantined = True
        self._quarantine_reason = reason
        active = self.runtime._active_inputs
        if active is not None and self.runtime.quarantined_identity is None:
            self.runtime.quarantine_generation(active.identity)
