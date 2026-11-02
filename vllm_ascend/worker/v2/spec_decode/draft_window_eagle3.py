# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""P4b host plans for static-K Eagle3 serialized eager Draft Windows.

This module contains no model-forward code. It freezes the legacy proposal
anchor from ``L0 + K`` and projects each real autoregressive end ``E_s`` onto
the immutable P2 FullSource through the existing direct-prefix provider.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from vllm_ascend.worker.v2.spec_decode.draft_window import (
    DraftMetadataInputs,
    DraftWindowGenerationIdentity,
    DraftWindowSourceCapability,
    gather_valid_num_blocks_cpu,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_eager import (
    DraftWindowProposalContext,
    DraftWindowProposalEpochSource,
    _requires_prefill_baseline_bypass,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_graph import DraftWindowGraphBindingProvider
from vllm_ascend.worker.v2.spec_decode.draft_window_nongraph import (
    DraftWindowBatchPhasePlan,
    DraftWindowPhaseAwareMaterializationPlan,
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


@dataclass(frozen=True, slots=True)
class Eagle3FrozenProposalPlan:
    """Per-row ``L0/H/S`` values frozen for one static-K proposal."""

    static_k: int
    row_plans: tuple[DraftWindowPlan, ...]
    actual_request_count: int
    padded_request_count: int

    def __post_init__(self) -> None:
        if self.static_k <= 0:
            raise ValueError("Eagle3 static K must be greater than 0")
        if self.actual_request_count != len(self.row_plans):
            raise ValueError("Eagle3 proposal row plans must match active requests")
        if self.padded_request_count < self.actual_request_count:
            raise ValueError("Eagle3 padded request count must cover active requests")
        for row_plan in self.row_plans:
            if row_plan.full_horizon_end - row_plan.full_exact_end != self.static_k:
                raise ValueError("Eagle3 proposal rows must use the configured static K")


class Eagle3DraftWindowMixin:
    """P4b proposal/step ownership for static-K Eagle3 eager execution.

    This mixin deliberately stops before metadata assembly and model forward.
    The Ascend autoregressive speculator owns those method-specific operations;
    this class owns only proposal identity, the frozen anchor, per-step
    materialization, and serialized retirement.
    """

    _draft_window_graph_binding_provider: DraftWindowGraphBindingProvider
    _eagle3_window_epoch_source: DraftWindowProposalEpochSource
    _eagle3_window_proposal_context: DraftWindowProposalContext | None
    _eagle3_window_proposal_plan: Eagle3FrozenProposalPlan | None
    _eagle3_window_live_inputs: DraftMetadataInputs | None
    _eagle3_window_last_identity: DraftWindowGenerationIdentity | None
    _eagle3_window_decode_step_one_exact: torch.Tensor | None
    _eagle3_window_bypass_active: bool
    _eagle3_window_next_step: int
    _eagle3_window_static_k: int
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
        self._eagle3_window_live_inputs = None
        self._eagle3_window_last_identity = None
        self._eagle3_window_decode_step_one_exact = None
        self._eagle3_window_bypass_active = False
        self._eagle3_window_continuing_prefill = False
        self._eagle3_window_next_step = 0
        self._eagle3_window_static_k = static_k
        self._eagle3_window_scheduled_tokens: int | None = None
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

        expected_steps = _eagle3_graph_scheduled_consumers(self._eagle3_window_static_k, num_speculative_tokens)
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

        context = epoch_source.begin(input_batch)
        self._eagle3_window_proposal_context = context
        self._eagle3_window_proposal_plan = None
        self._eagle3_window_live_inputs = None
        self._eagle3_window_last_identity = None
        self._eagle3_window_decode_step_one_exact = None
        self._eagle3_window_next_step = 0
        self._eagle3_window_scheduled_tokens = num_speculative_tokens
        self._eagle3_window_expected_steps = expected_steps
        try:
            yield
            if self._eagle3_window_live_inputs is not None:
                raise RuntimeError("Eagle3 proposal returned with an active Draft Window reader")
            if self._eagle3_window_next_step != expected_steps:
                raise RuntimeError(
                    "eligible proposal completed without consuming all Eagle3 steps: "
                    f"consumed={self._eagle3_window_next_step} expected={expected_steps} "
                    f"static_K={self._eagle3_window_static_k}"
                )
            if graph_session is not None:
                graph_session.finish()
        except BaseException:
            if graph_session is not None:
                self._quarantine_eagle3_window()
            raise
        finally:
            live_inputs = self._eagle3_window_live_inputs
            try:
                if (
                    live_inputs is not None
                    and not live_inputs.retired
                    and not getattr(self, "_draft_window_graph_submission_uncertain", False)
                ):
                    self._draft_window_graph_binding_provider.abort_generation(live_inputs.identity)
            finally:
                self._eagle3_window_proposal_context = None
                self._eagle3_window_proposal_plan = None
                self._eagle3_window_live_inputs = None
                self._eagle3_window_last_identity = None
                self._eagle3_window_decode_step_one_exact = None
                self._eagle3_window_next_step = 0
                self._eagle3_window_scheduled_tokens = None
                self._eagle3_window_expected_steps = self._eagle3_window_static_k
                epoch_source.finish(context)

    def _eagle3_window_is_active(self) -> bool:
        return (
            hasattr(self, "_draft_window_graph_binding_provider")
            and not getattr(self, "_eagle3_window_bypass_active", False)
            and getattr(self, "_eagle3_window_proposal_context", None) is not None
        )

    def _materialize_eagle3_first_pass(
        self,
        *,
        num_reqs: int,
        num_reqs_padded: int,
        output_workspace: Any = None,
    ) -> DraftMetadataInputs:
        """Freeze proposal L0/H/S and materialize first-pass read state."""
        if self._eagle3_window_next_step != 0 or self._eagle3_window_proposal_plan is not None:
            raise RuntimeError("Eagle3 first-pass Window state is already materialized")
        context = self._require_eagle3_window_context()
        input_batch = self.input_batch
        self._eagle3_window_epoch_source.validate(context, input_batch)
        self._validate_request_counts(num_reqs, num_reqs_padded)

        cpu_l0 = _copy_padded_cpu_exact(
            input_batch.seq_lens_np,
            actual_request_count=num_reqs,
            padded_request_count=num_reqs_padded,
            name="proposal L0",
        )
        from vllm_ascend.worker.v2.spec_decode.draft_graph_review_limits import (
            validate_eagle_single_request_review_exact,
        )

        validate_eagle_single_request_review_exact(
            self,
            input_batch,
            cpu_l0,
            num_reqs=num_reqs,
            padded=num_reqs_padded,
            step=0,
        )
        cpu_upper = _copy_padded_cpu_upper(
            input_batch.seq_lens_cpu_upper_bound,
            actual_request_count=num_reqs,
            padded_request_count=num_reqs_padded,
            increment=0,
            max_model_len=self.max_model_len,
        )
        capability = self._draft_window_graph_binding_provider.source
        cpu_valid_num_blocks = self._gather_eagle3_valid_counts(
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs_padded,
        )
        proposal = build_eagle3_frozen_proposal_plan(
            source=capability,
            cpu_proposal_l0=cpu_l0,
            actual_request_count=num_reqs,
            padded_request_count=num_reqs_padded,
        )
        step_plan = build_eagle3_step_materialization_plan(
            source=capability,
            proposal=proposal,
            cpu_full_exact=cpu_l0,
            cpu_apply_time_full_upper=cpu_upper,
            cpu_valid_num_blocks=cpu_valid_num_blocks,
            step=0,
            phase_plan=context.phase_plan,
        )
        self._eagle3_window_epoch_source.validate(context, input_batch)
        inputs = self._draft_window_graph_binding_provider.materialize(
            proposal=context,
            idx_mapping=input_batch.idx_mapping,
            read_descriptors=step_plan.read_descriptors,
            cpu_window_exact=step_plan.cpu_window_exact,
            cpu_window_upper=step_plan.cpu_window_upper,
            cpu_valid_num_blocks=step_plan.cpu_valid_num_blocks,
            num_reqs_padded=num_reqs_padded,
            step=0,
            quarantine_on_failure=hasattr(self, "_eagle3_graph_session"),
            output_workspace=output_workspace,
        )
        self._eagle3_window_proposal_plan = proposal
        self._eagle3_window_live_inputs = inputs
        try:
            self._eagle3_window_epoch_source.validate(context, input_batch)
        except BaseException:
            if getattr(self, "_eagle3_graph_session", None) is not None:
                self._quarantine_eagle3_window()
            else:
                self._draft_window_graph_binding_provider.abort_generation(inputs.identity)
                self._eagle3_window_live_inputs = None
                self._eagle3_window_proposal_plan = None
            raise
        return inputs

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

    def _materialize_eagle3_decode_step(
        self,
        *,
        step: int,
        num_reqs: int,
        num_reqs_padded: int,
        seq_lens_cpu_upper_bound: torch.Tensor,
        output_workspace: Any = None,
    ) -> DraftMetadataInputs:
        """Refresh one AR step from FullSource under the proposal epoch."""
        if step != self._eagle3_window_next_step:
            raise RuntimeError(
                f"Eagle3 metadata step is stale or out of order: expected={self._eagle3_window_next_step} actual={step}"
            )
        if self._eagle3_window_live_inputs is not None:
            raise RuntimeError("previous Eagle3 Draft Window reader is still active")
        proposal = self._eagle3_window_proposal_plan
        previous_identity = self._eagle3_window_last_identity
        step_one_exact = self._eagle3_window_decode_step_one_exact
        if proposal is None or previous_identity is None or step_one_exact is None:
            raise RuntimeError("Eagle3 decode step is missing proposal-owned state")
        context = self._require_eagle3_window_context()
        input_batch = self.input_batch
        self._eagle3_window_epoch_source.validate(context, input_batch)
        self._validate_request_counts(num_reqs, num_reqs_padded)
        if proposal.actual_request_count != num_reqs:
            raise RuntimeError("Eagle3 request geometry changed within one proposal")

        cpu_exact = torch.zeros(num_reqs_padded, dtype=torch.int32, device="cpu")
        if num_reqs:
            torch.add(
                step_one_exact[:num_reqs],
                step - 1,
                out=cpu_exact[:num_reqs],
            )
            cpu_exact[:num_reqs].clamp_(max=self.max_model_len)
        from vllm_ascend.worker.v2.spec_decode.draft_graph_review_limits import (
            validate_eagle_single_request_review_exact,
        )

        validate_eagle_single_request_review_exact(
            self,
            input_batch,
            cpu_exact,
            num_reqs=num_reqs,
            padded=num_reqs_padded,
            step=step,
        )
        cpu_upper = _copy_padded_cpu_upper(
            seq_lens_cpu_upper_bound,
            actual_request_count=num_reqs,
            padded_request_count=num_reqs_padded,
            increment=step,
            max_model_len=self.max_model_len,
        )
        cpu_valid_num_blocks = self._gather_eagle3_valid_counts(
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs_padded,
        )
        step_plan = build_eagle3_step_materialization_plan(
            source=self._draft_window_graph_binding_provider.source,
            proposal=proposal,
            cpu_full_exact=cpu_exact,
            cpu_apply_time_full_upper=cpu_upper,
            cpu_valid_num_blocks=cpu_valid_num_blocks,
            step=step,
            phase_plan=context.phase_plan,
            padded_request_count=num_reqs_padded,
        )
        self._eagle3_window_epoch_source.validate(context, input_batch)
        inputs = self._draft_window_graph_binding_provider.materialize(
            previous_identity=previous_identity,
            proposal=context,
            idx_mapping=input_batch.idx_mapping,
            read_descriptors=step_plan.read_descriptors,
            cpu_window_exact=step_plan.cpu_window_exact,
            cpu_window_upper=step_plan.cpu_window_upper,
            cpu_valid_num_blocks=step_plan.cpu_valid_num_blocks,
            num_reqs_padded=num_reqs_padded,
            step=step,
            quarantine_on_failure=hasattr(self, "_eagle3_graph_session"),
            output_workspace=output_workspace,
        )
        self._eagle3_window_live_inputs = inputs
        try:
            self._eagle3_window_epoch_source.validate(context, input_batch)
        except BaseException:
            if getattr(self, "_eagle3_graph_session", None) is not None:
                self._quarantine_eagle3_window()
            else:
                self._draft_window_graph_binding_provider.abort_generation(inputs.identity)
                self._eagle3_window_live_inputs = None
            raise
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

    def _gather_eagle3_valid_counts(
        self,
        *,
        num_reqs: int,
        num_reqs_padded: int,
    ) -> torch.Tensor:
        capability = self._draft_window_graph_binding_provider.source
        return gather_valid_num_blocks_cpu(
            capability.full_source,
            global_group_id=capability.group.gid,
            idx_mapping_np=self.input_batch.idx_mapping_np,
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs_padded,
        )


def build_eagle3_frozen_proposal_plan(
    *,
    source: DraftWindowSourceCapability,
    cpu_proposal_l0: torch.Tensor,
    actual_request_count: int,
    padded_request_count: int,
) -> Eagle3FrozenProposalPlan:
    """Freeze ``H=L0+K`` and canonical ``S`` once for one proposal."""
    _validate_counts(actual_request_count, padded_request_count)
    _validate_cpu_int32_vector(
        cpu_proposal_l0,
        name="proposal L0",
        padded_request_count=padded_request_count,
    )
    if source.policy.normalized_method != "eagle3":
        raise ValueError("Eagle3 proposal planning requires an Eagle3 provider")
    static_k = source.expected_future_offset
    if static_k <= 0:
        raise ValueError("Eagle3 proposal planning requires a positive configured static K")

    row_plans: list[DraftWindowPlan] = []
    for row in range(actual_request_count):
        l0 = int(cpu_proposal_l0[row].item())
        row_plans.append(
            compute_draft_window_plan(
                full_exact_end=l0,
                future_offset=static_k,
                window_size=source.window_size,
                block_span_tokens=source.group.table_column_span,
            )
        )
    return Eagle3FrozenProposalPlan(
        static_k=static_k,
        row_plans=tuple(row_plans),
        actual_request_count=actual_request_count,
        padded_request_count=padded_request_count,
    )


def build_eagle3_step_materialization_plan(
    *,
    source: DraftWindowSourceCapability,
    proposal: Eagle3FrozenProposalPlan,
    cpu_full_exact: torch.Tensor,
    cpu_apply_time_full_upper: torch.Tensor,
    cpu_valid_num_blocks: torch.Tensor,
    step: int,
    phase_plan: DraftWindowBatchPhasePlan,
    padded_request_count: int | None = None,
) -> DraftWindowPhaseAwareMaterializationPlan:
    """Project one real ``E_s`` without recomputing the proposal anchor."""
    if not isinstance(step, int) or isinstance(step, bool):
        raise TypeError("Eagle3 proposal step must be a Python int")
    if not 0 <= step < proposal.static_k:
        raise ValueError(f"Eagle3 proposal step is outside configured static K: step={step} K={proposal.static_k}")
    if source.expected_future_offset != proposal.static_k:
        raise RuntimeError("Eagle3 proposal K no longer matches the provider")
    # L0/H/S and logical request ownership are proposal state. Padding belongs
    # to this consumer: a Q4 first pass may use pad2 while merged Q1 AR uses
    # pad1. The frozen first-pass plan itself remains immutable.
    padded = proposal.padded_request_count if padded_request_count is None else padded_request_count
    if type(padded) is not int:
        raise TypeError("Eagle3 consumer padding must be a Python int")
    _validate_counts(proposal.actual_request_count, padded)
    if step == 0 and padded != proposal.padded_request_count:
        raise ValueError("first-pass consumer padding must match its frozen proposal layout")
    result = build_phase_aware_materialization_plan(
        source=source,
        phase_plan=phase_plan,
        cpu_full_exact=cpu_full_exact,
        cpu_apply_time_full_upper=cpu_apply_time_full_upper,
        cpu_valid_num_blocks=cpu_valid_num_blocks,
        actual_request_count=proposal.actual_request_count,
        padded_request_count=padded,
        frozen_row_plans=proposal.row_plans,
    )
    if step == 0 and any(
        int(cpu_full_exact[row]) != plan.full_exact_end for row, plan in enumerate(proposal.row_plans)
    ):
        raise ValueError("Eagle3 first-pass exact end must equal proposal-entry L0")
    return result


def _validate_counts(actual_request_count: int, padded_request_count: int) -> None:
    if actual_request_count < 0:
        raise ValueError("actual request count must be greater than or equal to 0")
    if padded_request_count < actual_request_count:
        raise ValueError("padded request count must cover active requests")


def _validate_cpu_int32_vector(
    tensor: torch.Tensor,
    *,
    name: str,
    padded_request_count: int,
) -> None:
    if tensor.device.type != "cpu":
        raise ValueError(f"CPU {name} must live on CPU")
    if tensor.dtype != torch.int32:
        raise TypeError(f"CPU {name} must use int32")
    if tensor.ndim != 1 or tensor.shape[0] != padded_request_count:
        raise ValueError(f"CPU {name} must match the padded request count")


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


def _copy_padded_cpu_upper(
    values: torch.Tensor,
    *,
    actual_request_count: int,
    padded_request_count: int,
    increment: int,
    max_model_len: int,
) -> torch.Tensor:
    if values.device.type != "cpu":
        raise ValueError("Eagle3 CPU upper lengths must live on CPU")
    if values.dtype != torch.int32:
        raise TypeError("Eagle3 CPU upper lengths must use int32")
    if values.ndim != 1 or values.shape[0] < actual_request_count:
        raise ValueError("Eagle3 CPU upper lengths do not cover active requests")
    result = torch.zeros(padded_request_count, dtype=torch.int32, device="cpu")
    if actual_request_count:
        torch.add(
            values[:actual_request_count],
            increment,
            out=result[:actual_request_count],
        )
        result[:actual_request_count].clamp_(max=max_model_len)
    return result
