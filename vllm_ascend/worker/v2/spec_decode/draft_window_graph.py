# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Persistent Draft Graph read banks and runtime binding validation.

Startup capture fixes addresses. Each proposal updates only its private read
banks and metadata; canonical Full write tables, slots, and absolute positions
remain unchanged. One proposal scope owns preparation and UPDATE, retaining
capture banks across failure and until the original stream can reuse them.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

from vllm_ascend.worker.v2.spec_decode.draft_window import (
    DraftWindowGraphSource,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_plan import DraftWindowPlan, compute_draft_window_plan

if TYPE_CHECKING:
    from vllm_ascend.worker.v2.spec_decode.draft_window_eager import DraftWindowProposalContext


class DraftWindowGraphRoute(Enum):
    """Method-specific graph source route; never a generic support boolean."""

    EAGLE3_FIRST_PASS = "eagle3_first_pass"
    EAGLE3_AR_STEP = "eagle3_ar_step"
    DFLASH = "dflash"
    DSPARK = "dspark"


@dataclass(slots=True, eq=False)
class DraftWindowGraphBucket:
    """Actual capture consumer, its read tensors and last prepared payload.

    Only its provider writes these private fields. Capture and replay borrow
    the same tensors; a proposal updates their contents, never their address.
    """

    route: DraftWindowGraphRoute
    descriptor: BatchExecutionDescriptor
    consumer_ordinal: int
    window_block_table: torch.Tensor
    device_window_exact: torch.Tensor
    cpu_window_exact: torch.Tensor
    owner: DraftWindowGraphBindingProvider = field(repr=False)
    params: dict[str, Any] | None = field(default=None, repr=False)

    def materialize(
        self,
        source: DraftWindowGraphSource,
        cpu_full_exact: torch.Tensor,
        cpu_valid_num_blocks: torch.Tensor,
        frozen_row_plans: tuple[DraftWindowPlan, ...] | None,
    ) -> None:
        assert self.owner.proposal is not None
        rows = self.owner.proposal.full_read_rows
        if any(
            t.device.type != "cpu" or t.ndim != 1 or len(t) < len(rows) or t.dtype not in (torch.int32, torch.int64)
            for t in (cpu_full_exact, cpu_valid_num_blocks)
        ):
            raise ValueError("current CPU exact lengths and initialized block counts are required")
        if frozen_row_plans is not None and len(frozen_row_plans) != len(rows):
            raise ValueError("frozen Window anchors must match active request rows")
        source_table, span = source.table, source.group.table_column_span
        intervals = []
        # Validate the whole current preparation before the first tensor write.
        for row, full_read in enumerate(rows):
            exact, valid = int(cpu_full_exact[row]), int(cpu_valid_num_blocks[row])
            if exact < 0 or not 0 <= valid <= source_table.shape[1]:
                raise RuntimeError("current exact length or initialized block count is invalid")
            if full_read:
                start, required, visible = 0, valid, exact
            else:
                plan = (
                    compute_draft_window_plan(
                        full_exact_end=exact, window_size=source.window_size, block_span_tokens=span
                    )
                    if frozen_row_plans is None
                    else frozen_row_plans[row].at_exact_end(exact)
                )
                if plan.window_start % span:
                    raise NotImplementedError("Draft Graph cannot read an intra-block Window start")
                start, visible = plan.window_start // span, plan.window_visible_length
                required = (visible + span - 1) // span
            if required > self.window_block_table.shape[1]:
                raise RuntimeError("draft-window table capacity exceeded")
            if start + required > valid or visible > required * span:
                raise RuntimeError("draft-window source does not cover the exact readable interval")
            intervals.append((start, required, visible))
        self.window_block_table.zero_()
        self.cpu_window_exact.zero_()
        for row, (start, required, visible) in enumerate(intervals):
            if required:
                self.window_block_table[row, :required].copy_(source_table[row, start : start + required])
            self.cpu_window_exact[row] = visible
        self.device_window_exact.copy_(self.cpu_window_exact)


_ROUTE_METHODS = {
    DraftWindowGraphRoute.EAGLE3_FIRST_PASS: "eagle3",
    DraftWindowGraphRoute.EAGLE3_AR_STEP: "eagle3",
    DraftWindowGraphRoute.DFLASH: "dflash",
    DraftWindowGraphRoute.DSPARK: "dspark",
}


class DraftWindowGraphBindingProvider:
    """Own capture banks and the single active proposal's bindings."""

    def __init__(self, *, source: DraftWindowGraphSource) -> None:
        self._source = source
        self.proposal: DraftWindowProposalContext | None = None
        self._active_bank: DraftWindowGraphBucket | None = None
        self._quarantined = False
        self._carriers: dict[tuple[DraftWindowGraphRoute, int, int], tuple[DraftWindowGraphBucket, ...]] = {}
        self._pending: list[DraftWindowGraphBucket] = []

    @property
    def source(self) -> DraftWindowGraphSource:
        return self._source

    @property
    def quarantined(self) -> bool:
        return self._quarantined

    def quarantine(self) -> None:
        """Retain all banks after uncertain work, even without active inputs."""
        self._quarantined = True

    @contextmanager
    def proposal_scope(self, proposal: DraftWindowProposalContext) -> Iterator[None]:
        self.require_reusable()
        if self.proposal is not None or self._active_bank is not None or self._pending:
            raise RuntimeError("a Draft Window proposal is already active")
        self.proposal = proposal
        try:
            yield
            if self._pending or (self._active_bank is not None and self._active_bank.params is None):
                raise RuntimeError("Draft Window proposal has unfinished Graph consumers")
        except BaseException:
            self.quarantine()
            raise
        else:
            self._active_bank = None
        finally:
            self.proposal = None

    def prepare(
        self,
        capture: DraftWindowGraphBucket,
        *,
        cpu_full_exact: torch.Tensor,
        cpu_valid_num_blocks: torch.Tensor,
        step: int,
        frozen_row_plans: tuple[DraftWindowPlan, ...] | None = None,
    ) -> DraftWindowGraphBucket:
        """Select and update the actual consumer before any preparation write."""
        self.require_reusable()
        proposal = self.proposal
        if proposal is None or capture.owner is not self:
            raise RuntimeError("Draft Window bank has no current owning proposal")
        if self._active_bank is not None and self._active_bank.params is None:
            raise RuntimeError("previous Draft Window consumer is still active")
        if capture in self._pending:
            raise RuntimeError("cannot overwrite a pending Draft Window consumer")
        if type(step) is not int or step < 0 or len(proposal.full_read_rows) > capture.descriptor.num_reqs:
            raise ValueError("Draft Window consumer step or active/padded geometry is invalid")
        if (capture.route is DraftWindowGraphRoute.EAGLE3_AR_STEP) != (step > 0):
            raise RuntimeError("Graph route does not match the actual consumer step")
        if frozen_row_plans is not None and self.method != "eagle3":
            raise ValueError("proposal-frozen Window anchors belong only to Eagle3")
        if capture.consumer_ordinal != len(self._pending):
            raise RuntimeError("Graph Window consumers must prepare in their actual serial ordinal order")
        if self._pending and self._pending[0].route is not capture.route:
            raise RuntimeError("Graph consumers belong to different routes")
        capture.params = None
        self._active_bank = capture
        try:
            capture.materialize(self.source, cpu_full_exact, cpu_valid_num_blocks, frozen_row_plans)
        except BaseException:
            self.quarantine()
            raise
        return capture

    def complete_native(self) -> None:
        """Release the Host preparation after the real native first forward.

        Tensor ownership stays with capture and reuse keeps the existing
        current-stream dependency; this is not a device-completion claim.
        """
        if self.quarantined or self._active_bank is None or self._active_bank.params is not None:
            raise RuntimeError("Draft Window has no current native first consumer")
        self._active_bank = None

    def capture_banks_for(
        self, *, route: DraftWindowGraphRoute, num_reqs: int, num_tokens: int
    ) -> tuple[DraftWindowGraphBucket, ...]:
        """The fixed capture callbacks expose geometry, so index it once here."""
        try:
            return self._carriers[route, num_reqs, num_tokens]
        except KeyError as error:
            raise RuntimeError("Draft Window has no captured consumer for this geometry") from error

    def capture_identity_for(
        self,
        *,
        route: DraftWindowGraphRoute,
        descriptor: BatchExecutionDescriptor,
        consumer_ordinal: int = 0,
    ) -> DraftWindowGraphBucket:
        banks = self.capture_banks_for(route=route, num_reqs=descriptor.num_reqs, num_tokens=descriptor.num_tokens)
        if banks[0].descriptor != descriptor or not 0 <= consumer_ordinal < len(banks):
            raise RuntimeError(f"runtime descriptor does not resolve a captured consumer: {descriptor!r}")
        return banks[consumer_ordinal]

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
        key = route, num_reqs, descriptor.num_tokens
        banks = self._carriers.get(key, ())
        if banks and banks[0].descriptor != descriptor:
            raise RuntimeError("Draft Window capture descriptor geometry is ambiguous")
        if consumer_ordinal != len(banks):
            raise RuntimeError("capture consumer identity is duplicate or outside serial ordinal order")
        source = self.source
        if num_reqs > source.table.shape[0]:
            raise RuntimeError("capture request bucket exceeds source capacity")
        table = source.table
        if table.ndim != 2 or table.dtype != torch.int32:
            raise ValueError("Draft capture requires a two-dimensional int32 table")
        bucket = DraftWindowGraphBucket(
            route,
            descriptor,
            consumer_ordinal,
            table.new_zeros((num_reqs, table.shape[1])),
            torch.zeros(num_reqs, dtype=torch.int32, device=table.device),
            torch.zeros(num_reqs, dtype=torch.int32),
            self,
        )
        self._carriers[key] = (*banks, bucket)
        return bucket

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

    def bind_current(self, descriptor: BatchExecutionDescriptor) -> None:
        """Publish the prepared bank directly to the original Graph resolver."""
        bank = self._active_bank
        if self.quarantined or bank is None or bank.params is not None or bank.descriptor != descriptor:
            raise RuntimeError("Graph replay does not own the current unconsumed preparation")
        query_width = descriptor.num_tokens // descriptor.num_reqs
        bank.params = {
            "actual_seq_lengths": [query_width * (row + 1) for row in range(descriptor.num_reqs)],
            "actual_seq_lengths_kv": bank.cpu_window_exact.tolist(),
            "block_table": bank.window_block_table,
        }
        self._pending.append(bank)

    def get(self, _provider: Any) -> tuple[dict[str, Any], ...]:
        """Resolve one payload per actual layer consumer, in capture order."""
        result = []
        for carrier in self._pending:
            assert carrier.params is not None
            result.append(carrier.params)
        return tuple(result)

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
                    or task.provider.is_non_pa
                    or task.provider.layer_name not in self.source.group.layers
                    or task.operation != torch_npu.npu_fused_infer_attention_score.out
                ):
                    raise RuntimeError("Draft Window Graph requires its supported Draft FIA consumer")

    def finish_update(self) -> None:
        """Drop staged Host bindings after UPDATE; actual streams still own reuse."""
        self._pending.clear()


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
