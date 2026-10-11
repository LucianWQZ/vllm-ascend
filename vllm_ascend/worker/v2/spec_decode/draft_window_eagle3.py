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
    DraftWindowExecutionPhase,
    DraftWindowGenerationIdentity,
    DraftWindowLeaseState,
    DraftWindowReadDescriptor,
    DraftWindowRuntimeState,
    DraftWindowScalarPlan,
    DraftWindowSourceCapability,
    SingleGroupDirectPrefixProvider,
    compute_draft_window_scalar_plan,
    gather_valid_num_blocks_cpu,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_eager import (
    AscendDraftAttentionReadInputs,
    DraftWindowProposalContext,
    DraftWindowProposalEpochSource,
    _requires_prefill_baseline_bypass,
)


@dataclass(frozen=True, slots=True)
class Eagle3FrozenProposalPlan:
    """Per-row ``L0/H/S`` values frozen for one static-K proposal."""

    static_k: int
    row_plans: tuple[DraftWindowScalarPlan, ...]
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


@dataclass(frozen=True, slots=True)
class Eagle3StepMaterializationPlan:
    """Host descriptor/length snapshot for one proposal step."""

    step: int
    read_descriptors: tuple[DraftWindowReadDescriptor, ...]
    cpu_window_exact: torch.Tensor
    cpu_window_upper: torch.Tensor
    cpu_valid_num_blocks: torch.Tensor
    window_starts: tuple[int, ...]


class Eagle3DraftWindowMixin:
    """P4b proposal/step ownership for static-K Eagle3 eager execution.

    This mixin deliberately stops before metadata assembly and model forward.
    The Ascend autoregressive speculator owns those method-specific operations;
    this class owns only proposal identity, the frozen anchor, per-step
    materialization, and serialized retirement.
    """

    _draft_window_runtime_state: DraftWindowRuntimeState
    _draft_window_source_capability: DraftWindowSourceCapability
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
    use_fused_multi_step_decode: bool

    def _initialize_eagle3_draft_window_state(self) -> None:
        """Bind a static-K proposal controller after P2/P3 capability closure."""
        if self.speculative_config.uses_dynamic_speculative_decoding():
            raise NotImplementedError("MRV2 Eagle3 Draft Window does not support dynamic speculative decoding")
        if hasattr(self, "_eagle3_window_epoch_source"):
            raise RuntimeError("P4b Eagle3 Draft Window state is already initialized")
        capability = self._draft_window_source_capability
        if self.method != "eagle3" or capability.inventory.method != "eagle3":
            raise RuntimeError("P4b Draft Window state is restricted to normalized Eagle3")
        static_k = self.num_speculative_steps
        if static_k <= 0:
            raise ValueError("Eagle3 static K must be greater than 0")
        if capability.provider.expected_future_offset != static_k:
            raise RuntimeError("configured Eagle3 K does not match the P2 provider")

        self._eagle3_window_epoch_source = DraftWindowProposalEpochSource()
        self._eagle3_window_proposal_context = None
        self._eagle3_window_proposal_plan = None
        self._eagle3_window_live_inputs = None
        self._eagle3_window_last_identity = None
        self._eagle3_window_decode_step_one_exact = None
        self._eagle3_window_bypass_active = False
        self._eagle3_window_next_step = 0
        self._eagle3_window_static_k = static_k
        # Fused decode builds metadata once and mutates it for later steps.
        # P4b must rematerialize every step from canonical FullSource.
        self.use_fused_multi_step_decode = False

    @contextmanager
    def _eagle3_draft_window_proposal_scope(
        self,
        input_batch: Any,
        *,
        dummy_run: bool,
        is_profile: bool,
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

        bypass_for_prefill = False
        if not dummy_run and not is_profile:
            bypass_for_prefill = _requires_prefill_baseline_bypass(input_batch)
        if dummy_run or is_profile or bool(getattr(input_batch, "is_dummy", False)) or bypass_for_prefill:
            if self._eagle3_window_bypass_active or self._eagle3_window_proposal_context is not None:
                raise RuntimeError("an Eagle3 Draft Window proposal scope is already active")
            self._eagle3_window_bypass_active = True
            try:
                yield
            finally:
                self._eagle3_window_bypass_active = False
            return

        context = epoch_source.begin(input_batch)
        self._eagle3_window_proposal_context = context
        self._eagle3_window_proposal_plan = None
        self._eagle3_window_live_inputs = None
        self._eagle3_window_last_identity = None
        self._eagle3_window_decode_step_one_exact = None
        self._eagle3_window_next_step = 0
        try:
            yield
            if self._eagle3_window_live_inputs is not None:
                raise RuntimeError("Eagle3 proposal returned with an active Draft Window reader")
            if self._eagle3_window_next_step != self._eagle3_window_static_k:
                raise RuntimeError(
                    "eligible proposal completed without consuming all Eagle3 steps: "
                    f"consumed={self._eagle3_window_next_step} K={self._eagle3_window_static_k}"
                )
        finally:
            live_inputs = self._eagle3_window_live_inputs
            try:
                if live_inputs is not None and live_inputs.lease.state is not DraftWindowLeaseState.RETIRED:
                    self._draft_window_runtime_state.abort_generation(live_inputs.identity)
            finally:
                self._eagle3_window_proposal_context = None
                self._eagle3_window_proposal_plan = None
                self._eagle3_window_live_inputs = None
                self._eagle3_window_last_identity = None
                self._eagle3_window_decode_step_one_exact = None
                self._eagle3_window_next_step = 0
                epoch_source.finish(context)

    def _eagle3_window_is_active(self) -> bool:
        return (
            hasattr(self, "_draft_window_runtime_state")
            and not getattr(self, "_eagle3_window_bypass_active", False)
            and getattr(self, "_eagle3_window_proposal_context", None) is not None
        )

    def _materialize_eagle3_first_pass(
        self,
        *,
        num_reqs: int,
        num_reqs_padded: int,
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
        cpu_upper = _copy_padded_cpu_upper(
            input_batch.seq_lens_cpu_upper_bound,
            actual_request_count=num_reqs,
            padded_request_count=num_reqs_padded,
            increment=0,
            max_model_len=self.max_model_len,
        )
        capability = self._draft_window_source_capability
        cpu_valid_num_blocks = self._gather_eagle3_valid_counts(
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs_padded,
        )
        proposal = build_eagle3_frozen_proposal_plan(
            provider=capability.provider,
            cpu_proposal_l0=cpu_l0,
            actual_request_count=num_reqs,
            padded_request_count=num_reqs_padded,
        )
        step_plan = build_eagle3_step_materialization_plan(
            provider=capability.provider,
            proposal=proposal,
            cpu_full_exact=cpu_l0,
            cpu_apply_time_full_upper=cpu_upper,
            cpu_valid_num_blocks=cpu_valid_num_blocks,
            step=0,
        )
        self._eagle3_window_epoch_source.validate(context, input_batch)
        inputs = self._draft_window_runtime_state.materialize(
            mapping_epoch=context.mapping_epoch,
            ordered_request_ids=context.ordered_request_ids,
            idx_mapping=input_batch.idx_mapping,
            read_descriptors=step_plan.read_descriptors,
            device_window_exact=None,
            cpu_window_exact=step_plan.cpu_window_exact,
            cpu_window_upper=step_plan.cpu_window_upper,
            cpu_valid_num_blocks=step_plan.cpu_valid_num_blocks,
            num_reqs_padded=num_reqs_padded,
            phase=DraftWindowExecutionPhase.PROPOSAL,
            step=0,
            execution_lane=0,
        )
        self._eagle3_window_proposal_plan = proposal
        self._eagle3_window_live_inputs = inputs
        try:
            self._eagle3_window_epoch_source.validate(context, input_batch)
        except BaseException:
            self._draft_window_runtime_state.abort_generation(inputs.identity)
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
        if proposal.actual_request_count != num_reqs or proposal.padded_request_count != num_reqs_padded:
            raise RuntimeError("Eagle3 request geometry changed within one proposal")

        cpu_exact = torch.zeros(num_reqs_padded, dtype=torch.int32, device="cpu")
        if num_reqs:
            torch.add(
                step_one_exact[:num_reqs],
                step - 1,
                out=cpu_exact[:num_reqs],
            )
            cpu_exact[:num_reqs].clamp_(max=self.max_model_len)
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
            provider=self._draft_window_source_capability.provider,
            proposal=proposal,
            cpu_full_exact=cpu_exact,
            cpu_apply_time_full_upper=cpu_upper,
            cpu_valid_num_blocks=cpu_valid_num_blocks,
            step=step,
        )
        self._eagle3_window_epoch_source.validate(context, input_batch)
        inputs = self._draft_window_runtime_state.materialize_serialized_step(
            previous_identity=previous_identity,
            ordered_request_ids=context.ordered_request_ids,
            idx_mapping=input_batch.idx_mapping,
            read_descriptors=step_plan.read_descriptors,
            device_window_exact=None,
            cpu_window_exact=step_plan.cpu_window_exact,
            cpu_window_upper=step_plan.cpu_window_upper,
            cpu_valid_num_blocks=step_plan.cpu_valid_num_blocks,
            num_reqs_padded=num_reqs_padded,
            step=step,
            execution_lane=0,
        )
        self._eagle3_window_live_inputs = inputs
        try:
            self._eagle3_window_epoch_source.validate(context, input_batch)
        except BaseException:
            self._draft_window_runtime_state.abort_generation(inputs.identity)
            self._eagle3_window_live_inputs = None
            raise
        return inputs

    def _validated_eagle3_read_inputs(
        self,
        inputs: DraftMetadataInputs,
        expected_identity: DraftWindowGenerationIdentity,
    ) -> AscendDraftAttentionReadInputs:
        """The one formal P4b validation/narrowing boundary before building."""
        inputs.validate_for(expected_identity)
        gid = expected_identity.global_group_id
        block_tables = tuple(
            inputs.window_block_table if group_id == gid else full_table[: inputs.padded_request_count]
            for group_id, full_table in enumerate(self.block_tables.input_block_tables)
        )
        active_upper = inputs.cpu_window_upper[: inputs.actual_request_count]
        max_seq_len = int(active_upper.max().item()) if active_upper.numel() else 0
        return AscendDraftAttentionReadInputs(
            block_tables=block_tables,
            seq_lens=inputs.device_window_exact,
            seq_lens_cpu_exact=inputs.cpu_window_exact,
            seq_lens_cpu_upper_bound=inputs.cpu_window_upper,
            max_seq_len=max_seq_len,
        )

    def _finish_eagle3_window_step(
        self,
        inputs: DraftMetadataInputs,
        *,
        succeeded: bool,
    ) -> None:
        """Retire only after the serialized forward has consumed this step."""
        if self._eagle3_window_live_inputs is not inputs:
            raise RuntimeError("Eagle3 step completion does not own the active generation")
        try:
            if succeeded:
                self._draft_window_runtime_state.retire_generation(inputs.identity)
                self._eagle3_window_last_identity = inputs.identity
                self._eagle3_window_next_step = inputs.identity.step + 1
            else:
                if inputs.lease.state is not DraftWindowLeaseState.RETIRED:
                    self._draft_window_runtime_state.abort_generation(inputs.identity)
                self._eagle3_window_last_identity = None
        except BaseException:
            if inputs.lease.state is not DraftWindowLeaseState.RETIRED:
                self._draft_window_runtime_state.abort_generation(inputs.identity)
            self._eagle3_window_last_identity = None
            self._eagle3_window_live_inputs = None
            raise
        self._eagle3_window_live_inputs = None

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
        capability = self._draft_window_source_capability
        return gather_valid_num_blocks_cpu(
            capability.full_source,
            global_group_id=capability.provider.global_group_id,
            idx_mapping_np=self.input_batch.idx_mapping_np,
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs_padded,
        )


def build_eagle3_frozen_proposal_plan(
    *,
    provider: SingleGroupDirectPrefixProvider,
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
    if provider.inventory.method != "eagle3":
        raise ValueError("Eagle3 proposal planning requires an Eagle3 provider")
    static_k = provider.expected_future_offset
    if static_k <= 0:
        raise ValueError("Eagle3 proposal planning requires a positive configured static K")

    row_plans: list[DraftWindowScalarPlan] = []
    for row in range(actual_request_count):
        l0 = int(cpu_proposal_l0[row].item())
        row_plans.append(
            compute_draft_window_scalar_plan(
                full_exact_end=l0,
                future_offset=static_k,
                window_size=provider.inventory.window_size,
                legacy_block_size=provider.legacy_block_size,
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
    provider: SingleGroupDirectPrefixProvider,
    proposal: Eagle3FrozenProposalPlan,
    cpu_full_exact: torch.Tensor,
    cpu_apply_time_full_upper: torch.Tensor,
    cpu_valid_num_blocks: torch.Tensor,
    step: int,
) -> Eagle3StepMaterializationPlan:
    """Project one real ``E_s`` without recomputing the proposal anchor."""
    if not isinstance(step, int) or isinstance(step, bool):
        raise TypeError("Eagle3 proposal step must be a Python int")
    if not 0 <= step < proposal.static_k:
        raise ValueError(f"Eagle3 proposal step is outside configured static K: step={step} K={proposal.static_k}")
    if provider.expected_future_offset != proposal.static_k:
        raise RuntimeError("Eagle3 proposal K no longer matches the provider")
    padded = proposal.padded_request_count
    for name, tensor in (
        ("full exact", cpu_full_exact),
        ("apply-time full upper", cpu_apply_time_full_upper),
        ("valid block counts", cpu_valid_num_blocks),
    ):
        _validate_cpu_int32_vector(
            tensor,
            name=name,
            padded_request_count=padded,
        )

    cpu_window_exact = torch.zeros(padded, dtype=torch.int32)
    cpu_window_upper = torch.zeros(padded, dtype=torch.int32)
    descriptors: list[DraftWindowReadDescriptor] = []
    window_starts: list[int] = []
    for row, proposal_row in enumerate(proposal.row_plans):
        full_exact = int(cpu_full_exact[row].item())
        full_upper = int(cpu_apply_time_full_upper[row].item())
        valid_num_blocks = int(cpu_valid_num_blocks[row].item())
        if step == 0 and full_exact != proposal_row.full_exact_end:
            raise ValueError("Eagle3 first-pass exact end must equal proposal-entry L0")
        if full_upper < full_exact:
            raise ValueError("apply-time full upper must be greater than or equal to exact")
        descriptor = provider.describe_frozen_step(
            proposal_row,
            full_exact_end=full_exact,
            valid_num_blocks=valid_num_blocks,
        )
        visible = full_exact - proposal_row.window_start
        raw_upper = max(full_upper - proposal_row.window_start, visible)
        window_upper = min(raw_upper, descriptor.safe_extent_tokens)
        if window_upper < visible:
            raise RuntimeError("Eagle3 Window upper is shorter than Window exact")
        descriptors.append(descriptor)
        window_starts.append(proposal_row.window_start)
        cpu_window_exact[row] = visible
        cpu_window_upper[row] = window_upper

    return Eagle3StepMaterializationPlan(
        step=step,
        read_descriptors=tuple(descriptors),
        cpu_window_exact=cpu_window_exact,
        cpu_window_upper=cpu_window_upper,
        cpu_valid_num_blocks=cpu_valid_num_blocks,
        window_starts=tuple(window_starts),
    )


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
