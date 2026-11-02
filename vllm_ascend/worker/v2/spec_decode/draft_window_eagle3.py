# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Freeze Eagle's proposal anchor and prepare each real Graph consumer's read bank."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch

from vllm_ascend.worker.v2.spec_decode.draft_window import (
    DraftMetadataInputs,
    gather_valid_num_blocks_cpu,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_eager import (
    DraftWindowProposalContext,
    DraftWindowProposalEpochSource,
    _requires_prefill_baseline_bypass,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_graph import DraftWindowGraphBindingProvider
from vllm_ascend.worker.v2.spec_decode.draft_window_nongraph import (
    build_phase_aware_materialization_plan,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_plan import DraftWindowPlan, compute_draft_window_plan


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
    """Proposal identity and frozen window geometry for static-K Eagle Graph."""

    _draft_window_graph_binding_provider: DraftWindowGraphBindingProvider
    _eagle3_window_epoch_source: DraftWindowProposalEpochSource
    _eagle3_window_proposal_context: DraftWindowProposalContext | None
    _eagle3_window_proposal_plan: tuple[DraftWindowPlan, ...] | None
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
    supports_decode_metadata_update: bool

    def _initialize_eagle3_draft_window_state(self) -> None:
        """Bind a static-K proposal controller after P2/P3 capability closure."""
        if self.speculative_config.uses_dynamic_speculative_decoding():
            raise NotImplementedError("MRV2 Eagle3 Draft Window does not support dynamic speculative decoding")
        if hasattr(self, "_eagle3_window_epoch_source"):
            raise RuntimeError("P4b Eagle3 Draft Window state is already initialized")
        capability = self._draft_window_graph_binding_provider.source
        if self.method != "eagle3" or capability.policy.normalized_method != "eagle3":
            raise RuntimeError("P4b Draft Window state is restricted to normalized Eagle3")
        static_k = self.num_speculative_steps
        if static_k <= 0:
            raise ValueError("Eagle3 static K must be greater than 0")
        if capability.expected_future_offset != static_k:
            raise RuntimeError("configured Eagle3 K does not match the P2 provider")

        self._eagle3_window_epoch_source = DraftWindowProposalEpochSource()
        self._eagle3_window_proposal_context = None
        self._eagle3_window_proposal_plan = None
        self._eagle3_window_decode_step_one_exact = None
        self._eagle3_window_bypass_active = False
        self._eagle3_window_continuing_prefill = False
        self._eagle3_window_next_step = 0
        self._eagle3_window_expected_steps = static_k
        # Fused decode builds metadata once and mutates it for later steps.
        # P4b must rematerialize every step from canonical FullSource.
        self.supports_decode_metadata_update = False

    @contextmanager
    def _eagle3_draft_window_proposal_scope(
        self,
        input_batch: Any,
        *,
        dummy_run: bool,
        is_profile: bool,
        num_speculative_tokens: int | None = None,
    ) -> Iterator[None]:
        """Own one real proposal epoch, or explicitly run the baseline path."""
        epoch_source: DraftWindowProposalEpochSource | None = getattr(
            self,
            "_eagle3_window_epoch_source",
            None,
        )
        if epoch_source is None:
            yield
            return

        expected_steps = _eagle3_graph_scheduled_consumers(self.num_speculative_steps, num_speculative_tokens)
        graph_session = getattr(self, "_eagle3_graph_session", None)
        if graph_session is not None:
            graph_session.require_reusable()
        bypass_for_prefill = False
        if not dummy_run and not is_profile:
            bypass_for_prefill = _requires_prefill_baseline_bypass(input_batch)
        if dummy_run or is_profile or bool(getattr(input_batch, "is_dummy", False)) or bypass_for_prefill:
            if self._eagle3_window_bypass_active or self._eagle3_window_proposal_context is not None:
                raise RuntimeError("an Eagle3 Draft Window proposal scope is already active")
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
        context = epoch_source.begin(input_batch)
        self._eagle3_window_proposal_context = context
        self._eagle3_window_proposal_plan = None
        self._eagle3_window_decode_step_one_exact = None
        self._eagle3_window_next_step = 0
        self._eagle3_window_expected_steps = expected_steps
        try:
            yield
            if self._draft_window_graph_binding_provider._active_inputs is not None:
                raise RuntimeError("Eagle3 proposal returned with an active Draft Window reader")
            if self._eagle3_window_next_step != expected_steps:
                raise RuntimeError(
                    "eligible proposal completed without consuming all Eagle3 steps: "
                    f"consumed={self._eagle3_window_next_step} expected={expected_steps} "
                    f"static_K={self.num_speculative_steps}"
                )
            graph_session.finish()
        except BaseException:
            self._quarantine_eagle3_window()
            raise
        finally:
            self._eagle3_window_proposal_context = None
            self._eagle3_window_proposal_plan = None
            self._eagle3_window_decode_step_one_exact = None
            self._eagle3_window_next_step = 0
            self._eagle3_window_expected_steps = self.num_speculative_steps
            epoch_source.finish(context)

    def _eagle3_window_is_active(self) -> bool:
        return (
            hasattr(self, "_draft_window_graph_binding_provider")
            and not getattr(self, "_eagle3_window_bypass_active", False)
            and getattr(self, "_eagle3_window_proposal_context", None) is not None
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
        context = self._require_eagle3_window_context()
        self._eagle3_window_epoch_source.validate(context, self.input_batch)
        self._validate_request_counts(num_reqs, num_reqs_padded)
        exact = torch.zeros(num_reqs_padded, dtype=torch.int32, device="cpu")
        if num_reqs:
            exact[:num_reqs].copy_(
                self.input_buffers.seq_lens[:num_reqs].to(
                    device="cpu",
                    dtype=torch.int32,
                    copy=True,
                )
            )
        self._eagle3_window_epoch_source.validate(context, self.input_batch)
        self._eagle3_window_decode_step_one_exact = exact
        return exact

    def _materialize_eagle3_step(
        self,
        *,
        step: int,
        num_reqs: int,
        num_reqs_padded: int,
        output_workspace: Any,
    ) -> DraftMetadataInputs:
        """Freeze the first anchor, then project each AR end into its own bank."""
        if (
            type(step) is not int
            or not 0 <= step < self._eagle3_window_expected_steps
            or step != self._eagle3_window_next_step
            or self._draft_window_graph_binding_provider._active_inputs is not None
        ):
            raise RuntimeError("Eagle3 Window step is stale or its previous reader remains active")
        context = self._require_eagle3_window_context()
        batch = self.input_batch
        self._validate_request_counts(num_reqs, num_reqs_padded)
        provider = self._draft_window_graph_binding_provider
        proposal = self._eagle3_window_proposal_plan
        previous = provider._last_retired_identity if step else None
        if step == 0:
            if proposal is not None:
                raise RuntimeError("Eagle3 first-pass Window state is already materialized")
            exact = _copy_padded_cpu_exact(
                batch.seq_lens_np,
                actual_request_count=num_reqs,
                padded_request_count=num_reqs_padded,
                name="proposal L0",
            )
            proposal = tuple(
                compute_draft_window_plan(
                    full_exact_end=int(value),
                    future_offset=self.num_speculative_steps,
                    window_size=provider.source.window_size,
                    block_span_tokens=provider.source.group.table_column_span,
                )
                for value in exact[:num_reqs]
            )
        else:
            step_one_exact = self._eagle3_window_decode_step_one_exact
            if proposal is None or previous is None or step_one_exact is None:
                raise RuntimeError("Eagle3 decode step is missing proposal-owned state")
            if len(proposal) != num_reqs:
                raise RuntimeError("Eagle3 request geometry changed within one proposal")
            exact = torch.zeros(num_reqs_padded, dtype=torch.int32, device="cpu")
            torch.add(step_one_exact[:num_reqs], step - 1, out=exact[:num_reqs])
            exact[:num_reqs].clamp_(max=self.max_model_len)
        from vllm_ascend.worker.v2.spec_decode.draft_graph_review_limits import (
            validate_eagle_single_request_review_exact,
        )

        validate_eagle_single_request_review_exact(
            self, batch, exact, num_reqs=num_reqs, padded=num_reqs_padded, step=step
        )
        valid = gather_valid_num_blocks_cpu(
            provider.source.full_source,
            global_group_id=provider.source.group.gid,
            idx_mapping_np=batch.idx_mapping_np,
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs_padded,
        )
        plan = build_phase_aware_materialization_plan(
            source=provider.source,
            frozen_row_plans=proposal,
            cpu_full_exact=exact,
            cpu_valid_num_blocks=valid,
            phase_plan=context.phase_plan,
            actual_request_count=num_reqs,
            padded_request_count=num_reqs_padded,
        )
        self._eagle3_window_epoch_source.validate(context, batch)
        inputs = provider.materialize(
            previous_identity=previous,
            proposal=context,
            idx_mapping=batch.idx_mapping,
            read_descriptors=plan.read_descriptors,
            cpu_window_exact=plan.cpu_window_exact,
            cpu_valid_num_blocks=plan.cpu_valid_num_blocks,
            num_reqs_padded=num_reqs_padded,
            step=step,
            output_workspace=output_workspace,
        )
        self._eagle3_window_proposal_plan = proposal
        return inputs

    def _quarantine_eagle3_window(self) -> None:
        """Keep actual banks alive after an uncertain partial submission."""
        self._draft_window_graph_submission_uncertain = True
        runtime = self._draft_window_graph_binding_provider
        if runtime._active_inputs is not None and runtime.quarantined_identity is None:
            runtime.quarantine_generation(runtime._active_inputs.identity)

    def _require_eagle3_window_context(self) -> DraftWindowProposalContext:
        context = self._eagle3_window_proposal_context
        if context is None:
            raise RuntimeError("Eagle3 Draft Window operation has no active proposal epoch")
        return context

    def _validate_request_counts(self, num_reqs: int, num_reqs_padded: int) -> None:
        if num_reqs != self.input_batch.num_reqs:
            raise RuntimeError("Eagle3 metadata request count does not match the proposal")
        if num_reqs_padded < num_reqs:
            raise ValueError("padded request count must cover active Eagle3 requests")


def _copy_padded_cpu_exact(
    values: np.ndarray,
    *,
    actual_request_count: int,
    padded_request_count: int,
    name: str,
) -> torch.Tensor:
    array = np.asarray(values)
    if array.ndim != 1 or array.shape[0] < actual_request_count:
        raise ValueError(f"{name} does not cover active requests")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must contain integers")
    active = array[:actual_request_count].astype(np.int64, copy=False)
    if bool(np.any(active < 0)):
        raise ValueError(f"{name} must be non-negative")
    result = torch.zeros(padded_request_count, dtype=torch.int32, device="cpu")
    if actual_request_count:
        result[:actual_request_count].copy_(torch.from_numpy(active.copy()).to(torch.int32))
    return result
