# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Persistent Draft Graph read banks and runtime binding validation.

Startup capture fixes addresses. Each proposal updates only its private read
banks and metadata; canonical Full write tables, slots, and absolute positions
remain unchanged. The bank owner retains current generation and request
identity through the actual typed Graph UPDATE entrance.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

from vllm_ascend.worker.v2.spec_decode.draft_window import (
    DraftMetadataInputs,
    DraftWindowGraphSource,
    DraftWindowWorkspace,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_plan import DraftWindowPlan

if TYPE_CHECKING:
    from vllm_ascend.worker.v2.spec_decode.draft_window_eager import DraftWindowProposalContext


class DraftWindowGraphRoute(Enum):
    """Method-specific graph source route; never a generic support boolean."""

    EAGLE3_FIRST_PASS = "eagle3_first_pass"
    EAGLE3_AR_STEP = "eagle3_ar_step"
    DFLASH = "dflash"
    DSPARK = "dspark"


@dataclass(frozen=True, slots=True, eq=False)
class DraftWindowGraphBucket:
    """One real descriptor/consumer and its immutable-address read bank."""

    route: DraftWindowGraphRoute
    descriptor: BatchExecutionDescriptor
    consumer_ordinal: int
    workspace: DraftWindowWorkspace
    inputs: DraftMetadataInputs | None = None


_ROUTE_METHODS = {
    DraftWindowGraphRoute.EAGLE3_FIRST_PASS: "eagle3",
    DraftWindowGraphRoute.EAGLE3_AR_STEP: "eagle3",
    DraftWindowGraphRoute.DFLASH: "dflash",
    DraftWindowGraphRoute.DSPARK: "dspark",
}


class DraftWindowGraphBindingProvider:
    """Own stable capture banks and bind each current runtime generation."""

    def __init__(self, *, source: DraftWindowGraphSource) -> None:
        self._source = source
        self._active_inputs: DraftMetadataInputs | None = None
        self._quarantined = False
        self._carriers: dict[tuple[DraftWindowGraphRoute, BatchExecutionDescriptor, int], DraftWindowGraphBucket] = {}
        self._pending: list[DraftWindowGraphBucket] = []
        self._updating = False

    @property
    def source(self) -> DraftWindowGraphSource:
        return self._source

    @property
    def quarantined(self) -> bool:
        return self._quarantined

    def quarantine_generation(self, inputs: DraftMetadataInputs) -> None:
        """Retain possibly submitted backing; uncertain native work is not reusable."""
        if self._active_inputs is not inputs:
            raise RuntimeError("cannot quarantine retired Draft inputs")
        self._quarantined = True

    def materialize(
        self,
        *,
        proposal: DraftWindowProposalContext,
        cpu_full_exact: torch.Tensor,
        cpu_valid_num_blocks: torch.Tensor,
        num_reqs_padded: int,
        step: int,
        output_workspace: DraftWindowWorkspace,
        frozen_row_plans: tuple[DraftWindowPlan, ...] | None = None,
    ) -> DraftMetadataInputs:
        """Populate a new proposal or the next retired Eagle consumer's bank."""
        if self.quarantined:
            raise RuntimeError("draft-window backing is quarantined after uncertain Graph submission")
        if self._active_inputs is not None:
            raise RuntimeError("Draft Graph provider still owns an active read bank")
        if (
            type(step) is not int
            or step < 0
            or type(num_reqs_padded) is not int
            or num_reqs_padded < len(proposal.ordered_request_ids)
        ):
            raise ValueError("Draft Window consumer step or active/padded geometry is invalid")
        if frozen_row_plans is not None and self.method != "eagle3":
            raise ValueError("proposal-frozen Window anchors belong only to Eagle3")
        inputs = DraftMetadataInputs(proposal, num_reqs_padded, step, output_workspace, self)
        self._active_inputs = inputs
        try:
            output_workspace.materialize(
                inputs=inputs,
                source=self.source,
                cpu_full_exact=cpu_full_exact,
                frozen_row_plans=frozen_row_plans,
                cpu_valid_num_blocks=cpu_valid_num_blocks,
            )
            return inputs
        except BaseException:
            # Preparation may have queued writes. Keep the current bank and
            # refuse reuse, without clearing or inspecting device work.
            self._quarantined = True
            raise

    def retire_generation(self, inputs: DraftMetadataInputs) -> None:
        inputs.validate()
        # Host retirement does not imply device completion. Capture owns the
        # banks; the established tail dependency orders their next overwrite.
        self._active_inputs = None

    def finish_proposal(self) -> None:
        """All staged consumers must have reached the real Graph UPDATE seam."""
        self.require_reusable()
        if self._active_inputs is not None or self._pending:
            raise RuntimeError("Draft Window proposal has unfinished Graph consumers")

    def capture_identity_for(
        self,
        *,
        route: DraftWindowGraphRoute,
        descriptor: BatchExecutionDescriptor,
        consumer_ordinal: int = 0,
    ) -> DraftWindowGraphBucket:
        """Select the bank using the actual manager's complete descriptor."""
        key = route, descriptor, consumer_ordinal
        try:
            return self._carriers[key]
        except KeyError as error:
            raise RuntimeError(f"runtime descriptor does not resolve a captured consumer: {descriptor!r}") from error

    def startup_capture_workspace(
        self,
        capture_identity: DraftWindowGraphBucket,
    ) -> DraftWindowWorkspace:
        """Return the real persistent bank before any business generation."""
        carrier = self._require_registered_capture(capture_identity)
        if carrier in self._pending:
            raise RuntimeError("startup capture carrier is already bound to a runtime generation")
        return carrier.workspace

    @property
    def method(self) -> str:
        """Return the policy-normalized method owned by this provider."""
        return self.source.policy.normalized_method

    def register_capture_identity(
        self,
        *,
        route: DraftWindowGraphRoute,
        descriptor: BatchExecutionDescriptor,
        consumer_ordinal: int = 0,
    ) -> DraftWindowGraphBucket:
        """Bind the manager's exact frozen descriptor to one persistent bank."""
        if not isinstance(route, DraftWindowGraphRoute) or _ROUTE_METHODS[route] != self.method:
            raise ValueError("capture route does not match the actual Draft method")
        if type(descriptor) is not BatchExecutionDescriptor:
            raise TypeError("capture requires the actual BatchExecutionDescriptor")
        if descriptor.cg_mode.name != "FULL":
            raise ValueError("Draft Window read capture requires a FULL attention descriptor")
        if descriptor.num_active_loras or descriptor.num_ubatches != 1:
            raise NotImplementedError("Draft Window capture does not support LoRA or microbatch variants")
        num_reqs = descriptor.num_reqs
        if type(num_reqs) is not int or num_reqs < 1 or type(consumer_ordinal) is not int or consumer_ordinal < 0:
            raise ValueError("capture request count and consumer ordinal must be valid integers")
        if type(descriptor.num_tokens) is not int or descriptor.num_tokens < 1 or descriptor.num_tokens % num_reqs:
            raise ValueError("capture token count must be positive and divisible by request count")
        query_width = descriptor.num_tokens // num_reqs
        if descriptor.uniform_token_count is not None and descriptor.uniform_token_count != query_width:
            raise ValueError("capture query width differs from the actual descriptor")
        if route is DraftWindowGraphRoute.EAGLE3_AR_STEP and query_width != 1:
            raise ValueError("Eagle3 AR capture requires query width 1")
        key = route, descriptor, consumer_ordinal
        if key in self._carriers:
            raise RuntimeError("capture consumer identity is already registered")
        source = self.source
        if num_reqs > source.table.shape[0]:
            raise RuntimeError("capture request bucket exceeds source capacity")
        backing = DraftWindowWorkspace(
            max_num_reqs=num_reqs,
            table_capacity_columns=source.table.shape[1],
            table_template=source.table,
        )
        bucket = DraftWindowGraphBucket(route, descriptor, consumer_ordinal, backing)
        self._carriers[key] = bucket
        return bucket

    def prepare_workspace(self, capture_identity: DraftWindowGraphBucket) -> DraftWindowWorkspace:
        """Select the final capture bank before any canonical read is written.

        The existing Graph tail-notify/current-stream dependency orders reuse.
        A successful Host UPDATE is not device completion.
        """
        carrier = self._require_registered_capture(capture_identity)
        self.require_reusable()
        if carrier in self._pending:
            raise RuntimeError("cannot overwrite an active Draft Window graph read carrier")
        return carrier.workspace

    def require_reusable(self) -> None:
        """Keep every proposal on its established completion stream."""
        if self.quarantined:
            raise RuntimeError("Draft Window bank is quarantined after uncertain submission")
        source = self.source.table
        if source.device.type != "cpu":
            stream = int(torch.npu.current_stream().npu_stream)
            previous = getattr(self, "_owner_stream", stream)
            if stream != previous:
                raise RuntimeError("Draft Window bank reuse changed stream without a completion dependency")
            self._owner_stream = stream

    def bind_read_carrier(
        self,
        *,
        capture_identity: DraftWindowGraphBucket,
        inputs: DraftMetadataInputs,
        metadata: Mapping[str, Any],
    ) -> DraftWindowGraphBucket:
        """Publish this generation against its selected persistent Graph bank."""
        carrier = self._require_registered_capture(capture_identity)
        inputs.validate()
        if inputs.metadata is not None:
            raise RuntimeError("cannot overwrite active draft metadata")
        if carrier in self._pending:
            raise RuntimeError("cannot overwrite an active Draft Window graph read carrier")
        if inputs._owner is not self:
            raise RuntimeError("graph carrier inputs do not belong to the provider runtime")
        route = capture_identity.route
        if (route is DraftWindowGraphRoute.EAGLE3_AR_STEP) != (inputs.step > 0):
            raise RuntimeError("Graph route does not match the actual consumer step")
        if inputs.padded_request_count != capture_identity.descriptor.num_reqs:
            raise RuntimeError("capture/runtime padded request count mismatch")
        if inputs.workspace is not carrier.workspace:
            raise RuntimeError("Graph generation was not materialized into its exact capture bank")
        ordinal = capture_identity.consumer_ordinal
        if self._updating or ordinal != len(self._pending):
            raise RuntimeError("Graph Window consumers must bind in their actual serial ordinal order")
        if self._pending:
            previous = self._pending[0]
            assert previous.inputs is not None
            if previous.route is not route or previous.inputs.proposal is not inputs.proposal:
                raise RuntimeError("graph read carriers do not belong to the same actual proposal/route")
        if not isinstance(metadata, Mapping) or not metadata:
            raise RuntimeError("explicit Draft Window graph carrier produced no Draft metadata")
        object.__setattr__(inputs, "metadata", metadata)
        object.__setattr__(carrier, "inputs", inputs)
        self._pending.append(carrier)
        return carrier

    def consume_metadata(
        self,
        *,
        route: DraftWindowGraphRoute,
        expected_consumer_count: int,
    ) -> tuple[Mapping[str, Any], ...]:
        """Submit exactly one explicit carrier set; baseline metadata is forbidden."""
        if (
            expected_consumer_count <= 0
            or len(self._pending) != expected_consumer_count
            or self._pending[0].route is not route
            or self._updating
        ):
            raise RuntimeError("Graph Window requires its complete unconsumed consumer batch")
        metadata = []
        for carrier in self._pending:
            assert carrier.inputs is not None and carrier.inputs.metadata is not None
            metadata.append(carrier.inputs.metadata)
        self._updating = True
        return tuple(metadata)

    def validate_captured_graphs(self, graphs: Mapping[BatchExecutionDescriptor, Any]) -> None:
        """Check actual FIA tasks once, after the manager has stored its graphs.

        The original manager owns these graphs by descriptor. Replays use its
        task resolver and the already registered bank/consumer, without a
        second graph-to-route inventory.
        """
        import torch_npu

        from vllm_ascend.attention.attention_v1 import FIAParamProvider

        for descriptor, graph in graphs.items():
            if descriptor.cg_mode.name != "FULL":
                continue
            if not graph.tasks:
                raise RuntimeError("Draft Window Graph captured no attention consumer")
            for task in graph.tasks:
                if (
                    not isinstance(task.provider, FIAParamProvider)
                    or not task.provider.is_draft_model
                    or task.provider.layer_name not in self.source.group.layers
                    or task.operation != torch_npu.npu_fused_infer_attention_score.out
                ):
                    raise RuntimeError("Draft Window Graph requires its supported Draft FIA consumer")

    def finish_update(self, *, route: DraftWindowGraphRoute) -> None:
        """Mark successful Host UPDATE; stream/event order still owns reuse."""
        if not self._updating or self._pending[0].route is not route:
            raise RuntimeError("no current Draft Window Graph UPDATE for this route")
        self._pending.clear()
        self._updating = False

    def _require_registered_capture(
        self,
        capture_identity: DraftWindowGraphBucket,
    ) -> DraftWindowGraphBucket:
        carrier = self._carriers.get(
            (capture_identity.route, capture_identity.descriptor, capture_identity.consumer_ordinal)
        )
        if carrier is not capture_identity:
            raise RuntimeError("capture identity/bucket is not registered by this Draft provider")
        return carrier


def resolve_parallel_draft_window_graph_route(
    owner: Any,
) -> DraftWindowGraphRoute | None:
    """Resolve DFlash/DSpark from the provider's actual method policy.

    The class-name/debug label on a speculator is not a policy or capability
    source.  When no provider is installed this remains an exact OFF-path
    no-op.  A provider for any non-parallel method is a hard route mismatch.
    """
    provider = getattr(owner, "_draft_window_graph_binding_provider", None)
    if provider is None:
        return None
    if not isinstance(provider, DraftWindowGraphBindingProvider):
        raise TypeError("invalid instance-owned Draft Window graph binding provider")
    method = provider.method
    if method == "dflash":
        return DraftWindowGraphRoute.DFLASH
    if method == "dspark":
        return DraftWindowGraphRoute.DSPARK
    raise RuntimeError(f"parallel Draft Window graph manager cannot consume method {method!r}")


def consume_draft_window_graph_metadata(
    owner: Any,
    *,
    route: DraftWindowGraphRoute,
    expected_consumer_count: int,
) -> tuple[Mapping[str, Any], ...] | None:
    """Return explicit carrier metadata, or preserve exact OFF behavior."""
    provider = getattr(owner, "_draft_window_graph_binding_provider", None)
    if provider is None:
        return None
    if not isinstance(provider, DraftWindowGraphBindingProvider):
        raise TypeError("invalid instance-owned Draft Window graph binding provider")
    return provider.consume_metadata(
        route=route,
        expected_consumer_count=expected_consumer_count,
    )
