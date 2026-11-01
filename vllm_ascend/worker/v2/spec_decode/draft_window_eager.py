# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Graph-owned proposal inputs for the MRV2 Draft Window read plane.

This module owns the parallel Graph proposal read view. It
connects P1/P2/P3 read state through an Ascend-owned metadata-assembly seam,
while leaving positions, query layout, slot mappings, and Target metadata
untouched. The upstream vLLM checkout remains unmodified.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import vllm.v1.worker.gpu.spec_decode.speculator as upstream_speculator_module
from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens
from vllm.v1.worker.gpu.input_batch import InputBatch

from vllm_ascend.worker.v2.spec_decode.draft_window import (
    DraftMetadataInputs,
    DraftWindowExecutionPhase,
    DraftWindowGenerationIdentity,
    gather_valid_num_blocks_cpu,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_graph import DraftWindowGraphBindingProvider
from vllm_ascend.worker.v2.spec_decode.draft_window_graph_metadata import copy_draft_attention_metadata
from vllm_ascend.worker.v2.spec_decode.draft_window_nongraph import (
    DraftWindowBatchPhasePlan,
    build_phase_aware_materialization_plan,
    classify_draft_window_batch_phase,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_read import DraftReadView


@dataclass(frozen=True, slots=True)
class AscendDraftAttentionReadInputs:
    """Ascend-owned, read-only transport for one Draft metadata build.

    The repository-local seam cannot override positions, query layout, or slot
    mappings. CPU exact and upper lengths are already final for this build.
    """

    block_tables: tuple[torch.Tensor, ...]
    seq_lens: torch.Tensor
    seq_lens_cpu_exact: torch.Tensor
    seq_lens_cpu_upper_bound: torch.Tensor
    max_seq_len: int
    dcp_local_seq_lens: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.max_seq_len < 0:
            raise ValueError("Draft attention max sequence length must be non-negative")
        for name, tensor in (
            ("device sequence lengths", self.seq_lens),
            ("CPU exact sequence lengths", self.seq_lens_cpu_exact),
            ("CPU upper sequence lengths", self.seq_lens_cpu_upper_bound),
        ):
            if tensor.ndim != 1:
                raise ValueError(f"Draft attention {name} must be one-dimensional")
            if tensor.dtype != torch.int32:
                raise TypeError(f"Draft attention {name} must use int32")
        if self.seq_lens_cpu_exact.device.type != "cpu":
            raise ValueError("Draft attention CPU exact sequence lengths must live on CPU")
        if self.seq_lens_cpu_upper_bound.device.type != "cpu":
            raise ValueError("Draft attention CPU upper sequence lengths must live on CPU")
        if self.seq_lens.shape != self.seq_lens_cpu_exact.shape:
            raise ValueError("Draft attention device and CPU exact lengths must have the same shape")
        if self.seq_lens_cpu_exact.shape != self.seq_lens_cpu_upper_bound.shape:
            raise ValueError("Draft attention CPU exact and upper lengths must have the same shape")

    def certify_common_metadata(self, gid: int, common: Any) -> None:
        """Certify a fresh per-group object; never project or clone read data."""
        if not 0 <= gid < len(self.block_tables):
            raise ValueError("Graph metadata group is outside current read inputs")
        if hasattr(common, "_draft_window_read") or hasattr(common, "_draft_window_device_read"):
            raise ValueError("Graph metadata already has a read projection owner")
        if (
            common.num_reqs != self.seq_lens.shape[0]
            or common.max_seq_len != self.max_seq_len
            or common.block_table_tensor is not self.block_tables[gid]
            or common.seq_lens_cpu is not self.seq_lens_cpu_exact
            or common.seq_lens_cpu_upper_bound is not self.seq_lens_cpu_upper_bound
            or common.seq_lens.data_ptr() != self.seq_lens.data_ptr()
            or common.seq_lens.shape != self.seq_lens.shape
            or common.seq_lens.stride() != self.seq_lens.stride()
            or common.seq_lens.dtype != self.seq_lens.dtype
            or common.seq_lens.device != self.seq_lens.device
        ):
            raise ValueError("Fresh Graph common metadata lost its current read owners")
        # The generic builder slices device lengths. Reattach that same view's
        # explicit owner for the certificate; no data is copied or projected.
        common.seq_lens = self.seq_lens
        common._seq_lens_cpu = self.seq_lens_cpu_exact
        common._draft_window_read = DraftReadView(
            block_table=self.block_tables[gid],
            seq_lens=self.seq_lens,
            seq_lens_cpu=self.seq_lens_cpu_exact,
            seq_lens_cpu_upper_bound=self.seq_lens_cpu_upper_bound,
            starts_cpu=None,
            max_seq_len=self.max_seq_len,
            num_reqs=common.num_reqs,
        )


@dataclass(frozen=True, slots=True)
class DraftWindowProposalContext:
    """Caller-owned identity for one real outer ``propose`` invocation."""

    mapping_epoch: int
    input_batch_id: int
    ordered_request_ids: tuple[str, ...]
    idx_mapping_fingerprint: tuple[int, ...]
    phase_plan: DraftWindowBatchPhasePlan

    def __post_init__(self) -> None:
        if type(self.mapping_epoch) is not int or self.mapping_epoch < 0:
            raise ValueError("proposal mapping epoch must be a nonnegative Python int")
        requests = self.ordered_request_ids
        if any(not isinstance(request, str) or not request for request in requests) or len(set(requests)) != len(
            requests
        ):
            raise ValueError("proposal requires unique nonempty ordered request IDs")
        if len(self.idx_mapping_fingerprint) != len(requests):
            raise ValueError("proposal CPU mapping must match its ordered request count")
        if self.phase_plan.ordered_request_fingerprint != requests:
            raise ValueError("proposal row phases belong to another ordered request mapping")


class DraftWindowProposalEpochSource:
    """Issue exactly one monotonic mapping epoch per real proposal."""

    def __init__(self) -> None:
        self._next_epoch = 0
        self._active: DraftWindowProposalContext | None = None

    def begin(self, input_batch: InputBatch | Any) -> DraftWindowProposalContext:
        if self._active is not None:
            raise RuntimeError("a draft-window proposal epoch is already active")
        request_ids, mapping = _proposal_fingerprint(input_batch)
        phase_plan = classify_draft_window_batch_phase(input_batch)
        context = DraftWindowProposalContext(
            mapping_epoch=self._next_epoch,
            input_batch_id=id(input_batch),
            ordered_request_ids=request_ids,
            idx_mapping_fingerprint=mapping,
            phase_plan=phase_plan,
        )
        self._next_epoch += 1
        self._active = context
        return context

    def validate(
        self,
        expected: DraftWindowProposalContext,
        input_batch: InputBatch | Any,
    ) -> None:
        if self._active is not expected:
            raise RuntimeError("stale proposal epoch is not the active generation")
        if id(input_batch) != expected.input_batch_id:
            raise RuntimeError("proposal input batch object changed within one epoch")
        request_ids, mapping = _proposal_fingerprint(input_batch)
        if request_ids != expected.ordered_request_ids:
            raise RuntimeError("proposal request ordering changed within one epoch")
        if mapping != expected.idx_mapping_fingerprint:
            raise RuntimeError("proposal idx_mapping changed within one epoch")
        phase_plan = classify_draft_window_batch_phase(input_batch)
        if phase_plan != expected.phase_plan:
            raise RuntimeError("proposal row phases changed within one epoch")

    def finish(self, expected: DraftWindowProposalContext) -> None:
        if self._active is not expected:
            raise RuntimeError("stale proposal epoch cannot finish the active generation")
        self._active = None


def draft_window_read_inputs(
    workspace: Any,
    *,
    actual_request_count: int,
    padded_request_count: int,
    block_tables: Any,
    global_group_id: int,
) -> AscendDraftAttentionReadInputs:
    """Narrow one real scratch/capture bank to its builder's request rows."""
    active_upper = workspace.cpu_window_upper[:actual_request_count]
    return AscendDraftAttentionReadInputs(
        block_tables=tuple(
            (workspace.window_block_table if gid == global_group_id else table)[:padded_request_count]
            for gid, table in enumerate(block_tables)
        ),
        seq_lens=workspace.device_window_exact[:padded_request_count],
        seq_lens_cpu_exact=workspace.cpu_window_exact[:padded_request_count],
        seq_lens_cpu_upper_bound=workspace.cpu_window_upper[:padded_request_count],
        max_seq_len=int(active_upper.max().item()) if active_upper.numel() else 0,
    )


def build_draft_window_attn_metadata(
    owner: Any,
    *,
    num_reqs: int,
    num_reqs_padded: int,
    num_tokens_padded: int,
    read_inputs: AscendDraftAttentionReadInputs,
    query_width: int | None = None,
    causal: bool | Mapping[int, bool],
    query_start_loc_np: np.ndarray | None,
    dcp_local_seq_lens: torch.Tensor | None,
    attn_groups: Any = None,
) -> dict[str, Any]:
    """Assemble canonical query/write fields with one explicit Window read view.

    The fixed upstream assembler reads Full buffers from its owner. Keep this
    small Ascend seam shared by parallel drafting and Eagle's first/AR reads;
    proposal, position, write-slot preparation and sampling remain upstream.
    """
    if query_width is None:
        query_width = owner.num_query_per_req
    if query_start_loc_np is None:
        query_start_loc_cpu = torch.from_numpy(
            np.minimum(owner.arange_np[: num_reqs_padded + 1], num_reqs) * query_width
        )
        max_query_len = query_width
    else:
        query_start_loc_cpu = torch.empty(num_reqs_padded + 1, dtype=torch.int32)
        query_start_loc_cpu[: num_reqs + 1] = torch.from_numpy(query_start_loc_np[: num_reqs + 1])
        query_start_loc_cpu[num_reqs:] = query_start_loc_cpu[num_reqs]
        lengths = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        max_query_len = int(lengths.max().item()) if lengths.numel() else 0
    if len(read_inputs.block_tables) != len(owner.block_tables.input_block_tables):
        raise ValueError("Draft attention read block-table group count does not match the configured cache groups")
    if read_inputs.seq_lens.shape[0] != num_reqs_padded:
        raise ValueError("Draft attention read lengths do not match the padded request count")
    if dcp_local_seq_lens is None:
        dcp_local_seq_lens = read_inputs.dcp_local_seq_lens
    if dcp_local_seq_lens is None and owner.block_tables.cp_size > 1:
        dcp_local_seq_lens = prepare_dcp_local_seq_lens(
            owner.input_buffers.dcp_local_seq_lens,
            read_inputs.seq_lens,
            num_reqs,
            owner.block_tables.cp_size,
            owner.block_tables.cp_rank,
            owner.block_tables.cp_interleave,
        )
    return upstream_speculator_module.build_attn_metadata(
        draft_window_read_inputs=read_inputs,
        attn_groups=owner.attn_groups if attn_groups is None else attn_groups,
        num_reqs=num_reqs_padded,
        num_tokens=num_tokens_padded,
        query_start_loc_gpu=owner.input_buffers.query_start_loc[: num_reqs_padded + 1],
        query_start_loc_cpu=query_start_loc_cpu,
        max_query_len=max_query_len,
        seq_lens=read_inputs.seq_lens,
        seq_lens_cpu=read_inputs.seq_lens_cpu_exact,
        dcp_local_seq_lens=(None if dcp_local_seq_lens is None else dcp_local_seq_lens[:num_reqs_padded]),
        max_seq_len=read_inputs.max_seq_len,
        block_tables=list(read_inputs.block_tables),
        slot_mappings=owner.block_tables.slot_mappings[:, :num_tokens_padded],
        kv_cache_config=owner.kv_cache_config,
        causal=causal,
        seq_lens_cpu_upper_bound=read_inputs.seq_lens_cpu_upper_bound,
    )


class DraftWindowGraphInputMixin:
    """Proposal input ownership for DFlash, DFlash2 and eligible DSpark Graph."""

    _speculator_name: str
    _group_causal: bool | Mapping[int, bool]
    _draft_window_graph_binding_provider: DraftWindowGraphBindingProvider
    _draft_window_epoch_source: DraftWindowProposalEpochSource
    _draft_window_proposal_context: DraftWindowProposalContext | None
    _draft_window_live_inputs: DraftMetadataInputs | None
    _draft_window_bypass_active: bool
    _draft_window_prefill_bypass_active: bool
    _draft_window_dp_dummy_active: bool
    input_batch: InputBatch
    input_buffers: Any
    block_tables: Any
    model_state: Any
    vllm_config: Any
    draft_attn_layer_names: set[str] | None
    arange: torch.Tensor
    arange_np: np.ndarray
    _draft_window_graph_session: Any
    attn_groups: Any
    kv_cache_config: Any
    num_query_per_req: int
    max_model_len: int

    def _initialize_draft_window_source_capability(self) -> None:
        """Atomically bind eligible P2 source and P3 ownership state."""
        policy = getattr(self, "_draft_window_policy", None)
        if policy is None:
            return
        if hasattr(self, "_draft_window_graph_binding_provider"):
            raise RuntimeError("draft-window runtime ownership is already initialized")
        if hasattr(self, "_draft_window_runtime"):
            raise RuntimeError("Graph Window cannot replace the typed eager runtime")
        from vllm_ascend.worker.v2.spec_decode.draft_window_graph_runtime import build_parallel_graph_source

        capability = build_parallel_graph_source(
            policy=policy,
            runtime_route=self._speculator_name.lower(),
            attn_groups=self.attn_groups,
            block_tables=self.block_tables,
            kv_cache_config=self.kv_cache_config,
            causal=self._group_causal,
            vllm_config=self.vllm_config,
            dflash2_graph_contract=getattr(self, "_dflash2_graph_contract", None),
        )
        self._draft_window_graph_binding_provider = DraftWindowGraphBindingProvider(source=capability)
        self._initialize_draft_window_graph_inputs()

    def _initialize_draft_window_graph_inputs(self) -> None:
        if not hasattr(self, "_draft_window_graph_binding_provider"):
            raise RuntimeError("P4a eager state requires initialized P3 runtime state")
        if hasattr(self, "_draft_window_epoch_source"):
            raise RuntimeError("P4a eager state is already initialized")
        self._draft_window_epoch_source = DraftWindowProposalEpochSource()
        self._draft_window_proposal_context: DraftWindowProposalContext | None = None
        self._draft_window_live_inputs: DraftMetadataInputs | None = None
        self._draft_window_bypass_active = False
        self._draft_window_prefill_bypass_active = False
        self._draft_window_dp_dummy_active = False
        self._draft_window_selected_metadata = None
        self._draft_window_deferred_metadata = None

    @contextmanager
    def _draft_window_proposal_scope(
        self,
        input_batch: InputBatch,
        *,
        dummy_run: bool,
        is_profile: bool,
    ) -> Iterator[None]:
        epoch_source: DraftWindowProposalEpochSource | None = getattr(
            self,
            "_draft_window_epoch_source",
            None,
        )
        if epoch_source is None:
            yield
            return
        if self._draft_window_graph_binding_provider.quarantined_identity is not None or getattr(
            self, "_draft_window_graph_submission_uncertain", False
        ):
            raise RuntimeError("draft-window backing is quarantined after uncertain Graph submission")
        bypass_for_prefill = False
        if not dummy_run and not is_profile:
            bypass_for_prefill = _requires_prefill_baseline_bypass(input_batch)
        if dummy_run or is_profile or bypass_for_prefill:
            if self._draft_window_bypass_active or self._draft_window_proposal_context is not None:
                raise RuntimeError("a draft-window proposal scope is already active")
            self._draft_window_bypass_active = True
            # Only a real all-continuing prefill may select the established
            # full-read baseline. Dummy/profile bypasses are not this phase.
            self._draft_window_prefill_bypass_active = bypass_for_prefill
            # An idle DP replica still submits the official dummy Graph to
            # participate in communication. It has no business generation.
            # Keep this distinct from real prefill and startup profiling.
            dp_size = getattr(self, "dp_size", 1)
            self._draft_window_dp_dummy_active = (
                dummy_run
                and not is_profile
                and type(dp_size) is int
                and dp_size > 1
                and hasattr(self, "_draft_window_graph_session")
            )
            try:
                # Startup/profile and pure-prefill proposals exercise the
                # unmodified upstream metadata path without allocating a
                # production Window generation. Mixed prefill/decode remains
                # unsupported and is rejected by the classifier above.
                yield
            except BaseException:
                if self._draft_window_dp_dummy_active:
                    # Dummy preparation/UPDATE can also partially submit.
                    # Retain owners and refuse the next scope without any
                    # device inspection, reset, drain or attempted recovery.
                    self._draft_window_graph_submission_uncertain = True
                raise
            finally:
                self._draft_window_bypass_active = False
                self._draft_window_prefill_bypass_active = False
                self._draft_window_dp_dummy_active = False
            return
        context = epoch_source.begin(input_batch)
        self._draft_window_proposal_context = context
        self._draft_window_selected_metadata = None
        self._draft_window_deferred_metadata = None
        succeeded = False
        try:
            yield
            succeeded = True
        finally:
            if not succeeded:
                # Native work may have been submitted before a carrier exists.
                self._draft_window_graph_submission_uncertain = True
            live_inputs = self._draft_window_live_inputs
            try:
                if live_inputs is not None and not live_inputs.retired:
                    if succeeded:
                        try:
                            self._draft_window_graph_binding_provider.retire_generation(live_inputs.identity)
                        except BaseException:
                            self._draft_window_graph_submission_uncertain = True
                            succeeded = False
                            raise
                    else:
                        # Retain uncertain backing without device RPC on unwind.
                        self._draft_window_graph_binding_provider.quarantine_generation(live_inputs.identity)
                elif succeeded:
                    raise RuntimeError("Draft Graph proposal completed without consuming its current inputs")
            finally:
                self._draft_window_live_inputs = None
                self._draft_window_proposal_context = None
                self._draft_window_selected_metadata = None
                self._draft_window_deferred_metadata = None
                epoch_source.finish(context)

    def _snapshot_prepared_draft_seq_lens(
        self,
        num_reqs_padded: int,
    ) -> torch.Tensor:
        """Perform the one controlled D2H already present in the baseline builder."""
        return self.input_buffers.seq_lens[:num_reqs_padded].to(
            device="cpu",
            dtype=torch.int32,
            copy=True,
        )

    def _materialize_draft_window_read_inputs(
        self,
        *,
        num_reqs: int,
        num_reqs_padded: int,
    ) -> DraftMetadataInputs:
        context = self._require_current_draft_window_proposal()
        input_batch = self.input_batch
        self._draft_window_epoch_source.validate(context, input_batch)
        if num_reqs != input_batch.num_reqs:
            raise RuntimeError("metadata request count does not match proposal epoch")
        if num_reqs_padded < num_reqs:
            raise ValueError("padded request count must be at least actual requests")

        selected = getattr(self, "_draft_window_selected_metadata", None)
        output_workspace = None
        if selected is not None:
            session = self._draft_window_graph_session
            capture_identity = session.resolve_capture_identity(selected.descriptor)
            output_workspace = session.provider.prepare_workspace(capture_identity)

        cpu_full_exact = self._snapshot_prepared_draft_seq_lens(num_reqs_padded)
        cpu_apply_time_full_upper = torch.zeros(
            num_reqs_padded,
            dtype=torch.int32,
        )
        if num_reqs:
            torch.add(
                input_batch.seq_lens_cpu_upper_bound[:num_reqs],
                self.num_query_per_req,
                out=cpu_apply_time_full_upper[:num_reqs],
            )
            cpu_apply_time_full_upper[:num_reqs].clamp_(max=self.max_model_len)
        capability = self._draft_window_graph_binding_provider.source
        valid_num_blocks = gather_valid_num_blocks_cpu(
            capability.full_source,
            global_group_id=capability.group.gid,
            idx_mapping_np=input_batch.idx_mapping_np,
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs_padded,
        )
        plan = build_phase_aware_materialization_plan(
            source=capability,
            phase_plan=context.phase_plan,
            cpu_full_exact=cpu_full_exact,
            cpu_apply_time_full_upper=cpu_apply_time_full_upper,
            cpu_valid_num_blocks=valid_num_blocks,
            actual_request_count=num_reqs,
            padded_request_count=num_reqs_padded,
        )
        # Recheck freshness after the controlled length snapshot and host plan.
        self._draft_window_epoch_source.validate(context, input_batch)
        inputs = self._draft_window_graph_binding_provider.materialize(
            proposal=context,
            idx_mapping=input_batch.idx_mapping,
            read_descriptors=plan.read_descriptors,
            cpu_window_exact=plan.cpu_window_exact,
            cpu_window_upper=plan.cpu_window_upper,
            cpu_valid_num_blocks=plan.cpu_valid_num_blocks,
            num_reqs_padded=num_reqs_padded,
            phase=DraftWindowExecutionPhase.PROPOSAL,
            step=0,
            quarantine_on_failure=hasattr(self, "_draft_window_graph_session"),
            output_workspace=output_workspace,
        )
        return inputs

    def _validated_draft_attention_read_inputs(
        self,
        inputs: DraftMetadataInputs,
        expected_identity: DraftWindowGenerationIdentity,
    ) -> AscendDraftAttentionReadInputs:
        """The single formal P4a validation and transport-narrowing boundary."""
        inputs.validate_for(expected_identity)
        return draft_window_read_inputs(
            inputs.workspace,
            actual_request_count=len(expected_identity.proposal.ordered_request_ids),
            padded_request_count=expected_identity.padded_request_count,
            block_tables=self.block_tables.input_block_tables,
            global_group_id=inputs._owner.source.group.gid,
        )

    def _build_draft_attn_metadata(
        self,
        num_reqs: int,
        num_reqs_padded: int,
        num_tokens_padded: int,
        seq_lens_cpu_upper_bound: torch.Tensor,
        step: int,
        num_query_per_req: int | None = None,
        causal: bool | Mapping[int, bool] = False,
        query_start_loc_np: np.ndarray | None = None,
        dcp_local_seq_lens: torch.Tensor | None = None,
    ) -> dict[str, Any] | None:
        provider = getattr(self, "_draft_window_graph_binding_provider", None)
        if provider is None or getattr(self, "_draft_window_bypass_active", False):
            return super()._build_draft_attn_metadata(  # type: ignore[misc]
                num_reqs,
                num_reqs_padded,
                num_tokens_padded,
                seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                step=step,
                num_query_per_req=num_query_per_req,
                causal=causal,
                query_start_loc_np=query_start_loc_np,
                dcp_local_seq_lens=dcp_local_seq_lens,
            )
        if not self.draft_attn_layer_names:
            return None
        if num_query_per_req is not None:
            raise AssertionError("DFlash/DSpark query width must come from the speculator instance")
        if step != self.num_query_per_req:
            raise RuntimeError("P4a apply-time step does not match the actual query width")
        if query_start_loc_np is not None:
            raise NotImplementedError("MRV2 draft-window P4a requires the uniform DFlash/DSpark query layout")
        target_metadata = self._require_nonempty_target_metadata()
        inputs = self._materialize_draft_window_read_inputs(
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs_padded,
        )
        # Register ownership immediately. The outer proposal scope is the one
        # exception/retirement owner if validation, building, or publishing
        # fails after the workspace has been materialized.
        self._draft_window_live_inputs = inputs
        live_read_inputs = self._validated_draft_attention_read_inputs(
            inputs,
            inputs.identity,
        )
        from vllm_ascend.worker.v2.spec_decode.draft_window_graph_metadata import (
            defer_parallel_graph_metadata,
        )

        if defer_parallel_graph_metadata(
            self, inputs, padded_requests=num_reqs_padded, padded_tokens=num_tokens_padded
        ):
            # The pinned FULL branch does not consume this temporary value.
            # P6 builds/publishes the real bank metadata once before replay.
            return None
        metadata = build_draft_window_attn_metadata(
            self,
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs_padded,
            num_tokens_padded=num_tokens_padded,
            read_inputs=live_read_inputs,
            causal=causal,
            query_start_loc_np=query_start_loc_np,
            dcp_local_seq_lens=dcp_local_seq_lens,
        )
        if metadata is None:
            raise RuntimeError("eligible P4a route produced no Draft metadata")
        if self.model_state.attn_metadata is not target_metadata:
            raise RuntimeError("Target metadata owner changed during Draft metadata build")
        metadata = copy_draft_attention_metadata(metadata)
        inputs.publish(metadata, target_metadata=target_metadata)
        return metadata

    def _require_nonempty_target_metadata(self) -> Mapping[Any, Any]:
        target_metadata = self.model_state.attn_metadata
        if not isinstance(target_metadata, Mapping) or not target_metadata:
            raise ValueError("Window ON requires non-empty Target metadata from model_state.attn_metadata")
        return target_metadata

    def _require_current_draft_window_proposal(
        self,
    ) -> DraftWindowProposalContext:
        context = self._draft_window_proposal_context
        if context is None:
            raise RuntimeError("Draft Window metadata build has no active proposal epoch")
        return context


def _proposal_fingerprint(
    input_batch: InputBatch | Any,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    num_reqs = int(input_batch.num_reqs)
    request_ids = tuple(input_batch.req_ids[:num_reqs])
    if len(request_ids) != num_reqs:
        raise ValueError("proposal request IDs do not match num_reqs")
    mapping = np.asarray(input_batch.idx_mapping_np)
    if mapping.ndim != 1 or mapping.shape[0] < num_reqs:
        raise ValueError("proposal CPU idx_mapping does not cover active requests")
    if not np.issubdtype(mapping.dtype, np.integer):
        raise TypeError("proposal CPU idx_mapping must contain integers")
    return request_ids, tuple(int(value) for value in mapping[:num_reqs])


def _requires_prefill_baseline_bypass(
    input_batch: InputBatch | Any,
) -> bool:
    """Bypass only an all-continuing scheduler-prefill batch.

    Terminal rows transition into speculative decode and mixed batches carry
    an explicit row-wise full/window contract.  This helper remains for the
    existing proposal-scope call sites; the immutable phase plan is bound to
    the proposal epoch before any materialization.
    """
    return classify_draft_window_batch_phase(input_batch).bypass_entire_batch
