# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Freeze Eagle's proposal anchor and prepare each real Graph consumer's read bank."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch

from vllm_ascend.worker.v2.spec_decode.draft_window_graph import DraftWindowGraphBindingProvider
from vllm_ascend.worker.v2.spec_decode.draft_window_plan import compute_draft_window_start
from vllm_ascend.worker.v2.spec_decode.draft_window_runtime import continuing_prefill_rows


def _eagle3_graph_scheduled_consumers(static_k: int, scheduled: int | None) -> int:
    """Separate a zero-draft first reader from the fixed capture geometry."""
    if type(static_k) is not int or static_k < 1:
        raise ValueError("Eagle3 Graph requires a positive configured static K")
    if scheduled is None:
        return static_k
    if type(scheduled) is not int or scheduled not in (0, static_k):
        raise NotImplementedError(
            "Eagle3 Window Graph permits only zero-draft first-only or its configured static K; "
            "partial merged AR Graph proposals are not qualified"
        )
    return 1 if scheduled == 0 else static_k


class Eagle3DraftWindowMixin:
    """Proposal scope and frozen window geometry for static-K Eagle Graph."""

    _draft_window_graph_binding_provider: DraftWindowGraphBindingProvider
    _eagle3_window_intervals: tuple[tuple[int, int], ...] | None
    _eagle3_window_decode_step_one_exact: torch.Tensor | None
    _eagle3_window_bypass_active: bool
    _eagle3_window_next_step: int
    input_batch: Any
    input_buffers: Any
    block_tables: Any
    max_model_len: int
    method: str
    num_speculative_steps: int
    speculative_config: Any

    def _initialize_eagle3_draft_window_state(self) -> None:
        """Initialize proposal state after source/configuration validation."""
        self._eagle3_window_intervals = None
        self._eagle3_window_decode_step_one_exact = None
        self._eagle3_window_bypass_active = False
        self._eagle3_window_continuing_prefill = False
        self._eagle3_window_next_step = 0
        self._eagle3_window_expected_steps = self.num_speculative_steps

    @contextmanager
    def _eagle3_draft_window_proposal_scope(
        self,
        input_batch: Any,
        *,
        dummy_run: bool,
        is_profile: bool,
        num_speculative_tokens: int | None = None,
    ) -> Iterator[None]:
        """Own one real proposal, or explicitly run the baseline path."""
        if not hasattr(self, "_draft_window_graph_binding_provider"):
            yield
            return
        if self._eagle3_window_bypass_active or self._draft_window_graph_binding_provider.proposal is not None:
            raise RuntimeError("an Eagle3 Draft Window proposal scope is already active")

        expected_steps = _eagle3_graph_scheduled_consumers(self.num_speculative_steps, num_speculative_tokens)
        graph_session = getattr(self, "_eagle3_graph_session", None)
        provider = self._draft_window_graph_binding_provider
        if provider.quarantined:
            raise RuntimeError("Eagle3 Draft Window backing is quarantined")
        rows = None
        if not dummy_run and not is_profile and not bool(getattr(input_batch, "is_dummy", False)):
            rows = tuple(map(bool, continuing_prefill_rows(input_batch, input_batch.num_reqs)))
        bypass_for_prefill = rows is not None and bool(rows) and all(rows)
        if dummy_run or is_profile or bool(getattr(input_batch, "is_dummy", False)) or bypass_for_prefill:
            self._eagle3_window_bypass_active = True
            self._eagle3_window_continuing_prefill = bypass_for_prefill
            try:
                from vllm_ascend.worker.v2.spec_decode.draft_window_eagle3_dp import eagle3_dp_dummy_scope

                with eagle3_dp_dummy_scope(self, input_batch, dummy_run=dummy_run, is_profile=is_profile):
                    yield
            finally:
                self._eagle3_window_bypass_active = False
                self._eagle3_window_continuing_prefill = False
            return

        if graph_session is None:
            raise RuntimeError("Eagle3 Draft Window business proposals require their installed Graph manager")
        assert rows is not None
        self._eagle3_window_intervals = None
        self._eagle3_window_decode_step_one_exact = None
        self._eagle3_window_next_step = 0
        self._eagle3_window_expected_steps = expected_steps
        try:
            with provider.proposal_scope(rows):
                yield
                if self._eagle3_window_next_step != expected_steps:
                    raise RuntimeError(
                        "eligible proposal completed without consuming all Eagle3 steps: "
                        f"consumed={self._eagle3_window_next_step} expected={expected_steps} "
                        f"static_K={self.num_speculative_steps}"
                    )
        finally:
            self._eagle3_window_intervals = None
            self._eagle3_window_decode_step_one_exact = None
            self._eagle3_window_next_step = 0
            self._eagle3_window_expected_steps = self.num_speculative_steps

    def _eagle3_window_is_active(self) -> bool:
        return (
            hasattr(self, "_draft_window_graph_binding_provider")
            and self._draft_window_graph_binding_provider.proposal is not None
        )

    def _snapshot_eagle3_decode_step_one(
        self,
        *,
        num_reqs: int,
        num_reqs_padded: int,
    ) -> torch.Tensor:
        """Take the sole feature D2H exact snapshot after rejection correction."""
        if self._eagle3_window_next_step != 1:
            raise RuntimeError("decode exact snapshot requires a completed Eagle3 first pass")
        if self._eagle3_window_decode_step_one_exact is not None:
            raise RuntimeError("Eagle3 decode exact state was already snapshotted")
        self._validate_request_counts(num_reqs, num_reqs_padded)
        exact = self.input_buffers.seq_lens[:num_reqs].to(device="cpu", dtype=torch.int32, copy=True)
        self._eagle3_window_decode_step_one_exact = exact
        return exact

    def _materialize_eagle3_step(
        self,
        capture: Any,
        step: int,
    ) -> Any:
        """Freeze the first anchor, then project each AR end into its own bank."""
        if (
            type(step) is not int
            or not 0 <= step < self._eagle3_window_expected_steps
            or step != self._eagle3_window_next_step
        ):
            raise RuntimeError("Eagle3 Window step is stale or its previous reader remains active")
        provider = self._draft_window_graph_binding_provider
        rows = provider.proposal
        if rows is None:
            raise RuntimeError("Eagle3 Draft Window operation has no active proposal")
        batch = self.input_batch
        num_reqs, num_reqs_padded = batch.num_reqs, capture.descriptor.num_reqs
        if num_reqs != len(rows):
            raise RuntimeError("Eagle3 request geometry changed within one proposal")
        proposal = self._eagle3_window_intervals
        if step == 0:
            if proposal is not None:
                raise RuntimeError("Eagle3 first-pass Window state is already materialized")
            values = np.asarray(batch.seq_lens_np)
            if values.ndim != 1 or values.shape[0] < num_reqs or not np.issubdtype(values.dtype, np.integer):
                raise ValueError("proposal Full exact lengths must cover the active integer request rows")
            if bool(np.any(values[:num_reqs] < 0)):
                raise ValueError("proposal Full exact lengths must be non-negative")
            exact = torch.tensor(values[:num_reqs], dtype=torch.int32)
            proposal = tuple(
                (
                    compute_draft_window_start(
                        full_exact_end=int(value),
                        future_offset=self.num_speculative_steps,
                        window_size=provider.window_size,
                        block_span_tokens=provider.group.table_column_span,
                    ),
                    int(value) + self.num_speculative_steps,
                )
                for value in exact[:num_reqs]
            )
            if batch.has_prefill:
                widths = np.diff(batch.query_start_loc_np[: num_reqs + 1])
                if any(
                    not rows[row] and query > int(exact[row]) - interval[0]
                    for row, (query, interval) in enumerate(zip(widths, proposal))
                ):
                    raise NotImplementedError(
                        "Eagle3 Window terminal prefill query begins before its frozen read window; "
                        "use a smaller chunked-prefill budget"
                    )
        else:
            step_one_exact = self._eagle3_window_decode_step_one_exact
            if proposal is None or step_one_exact is None:
                raise RuntimeError("Eagle3 decode step is missing proposal-owned state")
            if len(proposal) != num_reqs:
                raise RuntimeError("Eagle3 request geometry changed within one proposal")
            exact = step_one_exact.add(step - 1).clamp_(max=self.max_model_len)
        from vllm_ascend.worker.v2.spec_decode.draft_graph_review_limits import (
            validate_eagle_single_request_review_exact,
        )

        validate_eagle_single_request_review_exact(
            self, batch, exact, num_reqs=num_reqs, padded=num_reqs_padded, step=step
        )
        bank = provider.prepare(
            capture,
            frozen_intervals=proposal,
            cpu_full_exact=exact,
            idx_mapping_np=batch.idx_mapping_np,
        )
        self._eagle3_window_intervals = proposal
        return bank

    def _validate_request_counts(self, num_reqs: int, num_reqs_padded: int) -> None:
        if num_reqs != self.input_batch.num_reqs:
            raise RuntimeError("Eagle3 metadata request count does not match the proposal")
        if num_reqs_padded < num_reqs:
            raise ValueError("padded request count must cover active Eagle3 requests")
