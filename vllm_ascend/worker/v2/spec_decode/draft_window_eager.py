# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Graph-owned proposal inputs for the MRV2 Draft Window read plane.

This module joins real proposal inputs to capture-owned Window buffers and
the normal Ascend metadata builder. Positions, query layout, slot mappings
and Target metadata remain owned by the existing runner.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from vllm.v1.worker.gpu.input_batch import InputBatch

from vllm_ascend.worker.v2.spec_decode.draft_window import (
    DraftMetadataInputs,
    gather_valid_num_blocks_cpu,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_graph import DraftWindowGraphBindingProvider
from vllm_ascend.worker.v2.spec_decode.draft_window_nongraph import (
    DraftWindowBatchPhasePlan,
    classify_draft_window_batch_phase,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_read import DraftReadView


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
        if len(self.phase_plan.full_read_rows) != len(requests):
            raise ValueError("proposal Full-read markers must match its ordered request count")


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
) -> DraftReadView:
    """Borrow the selected persistent bank through the common read interface."""
    exact = workspace.cpu_window_exact[:padded_request_count]
    active_exact = exact[:actual_request_count]
    return DraftReadView(
        block_table=workspace.window_block_table[:padded_request_count],
        seq_lens=workspace.device_window_exact[:padded_request_count],
        seq_lens_cpu=exact,
        seq_lens_cpu_upper_bound=exact,
        starts_cpu=None,
        max_seq_len=int(active_exact.max().item()) if active_exact.numel() else 0,
        num_reqs=padded_request_count,
    )


def build_draft_window_attn_metadata(
    owner: Any,
    *,
    num_reqs: int,
    num_reqs_padded: int,
    num_tokens_padded: int,
    read_inputs: DraftReadView,
    query_width: int | None = None,
    causal: bool | Mapping[int, bool],
    query_start_loc_np: np.ndarray | None,
    attn_groups: Any = None,
    attn_state: Any = None,
    parallel_config: Any = None,
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
    if read_inputs.seq_lens.shape[0] != num_reqs_padded:
        raise ValueError("Draft attention read lengths do not match the padded request count")
    # Use the fixed Ascend builder directly. Window assembly does not need to
    # replace the upstream module function or inherit an enclosing factory.
    from vllm_ascend.worker.v2.attn_utils import build_attn_metadata

    if attn_state is None:
        attn_state = next(layer.attn_state for layer in owner.model_state.attn_metadata.values() if layer is not None)
    if parallel_config is None:
        parallel_config = owner.attn_vllm_config.parallel_config
    window_gid = owner._draft_window_graph_binding_provider.source.group.gid
    tables = list(owner.block_tables.input_block_tables)
    tables[window_gid] = read_inputs.block_table

    def bind_window_read(gid, common):
        if gid == window_gid:
            read_inputs.apply(common)

    metadata = build_attn_metadata(
        common_metadata_callback=bind_window_read,
        attn_groups=owner.attn_groups if attn_groups is None else attn_groups,
        num_reqs=num_reqs_padded,
        num_tokens=num_tokens_padded,
        query_start_loc_gpu=owner.input_buffers.query_start_loc[: num_reqs_padded + 1],
        query_start_loc_cpu=query_start_loc_cpu,
        max_query_len=max_query_len,
        seq_lens=read_inputs.seq_lens,
        seq_lens_cpu=read_inputs.seq_lens_cpu,
        max_seq_len=read_inputs.max_seq_len,
        block_tables=tables,
        slot_mappings=owner.block_tables.slot_mappings[:, :num_tokens_padded],
        kv_cache_config=owner.kv_cache_config,
        causal=causal,
        seq_lens_cpu_upper_bound=read_inputs.seq_lens_cpu_upper_bound,
        positions=owner.input_buffers.positions[:num_tokens_padded],
        is_prefilling=torch.from_numpy(owner.input_batch.is_prefilling_np),
        parallel_config=parallel_config,
        attn_state=attn_state,
    )
    # The generic builder also makes a temporary GPU query-layout copy.
    # Capture/replay owns the original persistent input-buffer address.
    for layer in metadata.values():
        if layer is not None:
            layer.query_start_loc = owner.input_buffers.query_start_loc[: num_reqs_padded + 1]
    return metadata


class DraftWindowGraphInputMixin:
    """Proposal input ownership for DFlash, DFlash2 and eligible DSpark Graph."""

    _speculator_name: str
    _group_causal: bool | Mapping[int, bool]
    _draft_window_graph_binding_provider: DraftWindowGraphBindingProvider
    _draft_window_epoch_source: DraftWindowProposalEpochSource
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
        self._draft_window_bypass_active = False
        self._draft_window_prefill_bypass_active = False
        self._draft_window_dp_dummy_active = False
        self._draft_window_selected_metadata = None

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
            if self._draft_window_bypass_active or self._draft_window_epoch_source._active is not None:
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
        self._draft_window_selected_metadata = None
        succeeded = False
        try:
            yield
            succeeded = True
        finally:
            if not succeeded:
                # Native work may have been submitted before a carrier exists.
                self._draft_window_graph_submission_uncertain = True
            live_inputs = self._draft_window_graph_binding_provider._active_inputs
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
                self._draft_window_selected_metadata = None
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
        if num_reqs != input_batch.num_reqs:
            raise RuntimeError("metadata request count does not match proposal epoch")
        if num_reqs_padded < num_reqs:
            raise ValueError("padded request count must be at least actual requests")

        selected = self._draft_window_selected_metadata
        if selected is None:
            raise RuntimeError("Draft Window input preparation requires the actual FULL dispatch")
        from vllm_ascend.worker.v2.spec_decode.draft_window_graph import resolve_parallel_draft_window_graph_route

        provider = self._draft_window_graph_binding_provider
        route = resolve_parallel_draft_window_graph_route(self)
        assert route is not None
        capture_identity = provider.capture_identity_for(route=route, descriptor=selected)
        output_workspace = provider.prepare_workspace(capture_identity)

        cpu_full_exact = self._snapshot_prepared_draft_seq_lens(num_reqs_padded)
        capability = self._draft_window_graph_binding_provider.source
        valid_num_blocks = gather_valid_num_blocks_cpu(
            capability.full_source,
            global_group_id=capability.group.gid,
            idx_mapping_np=input_batch.idx_mapping_np,
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs_padded,
        )
        # Recheck request order after the controlled exact-length snapshot.
        self._draft_window_epoch_source.validate(context, input_batch)
        inputs = self._draft_window_graph_binding_provider.materialize(
            proposal=context,
            idx_mapping=input_batch.idx_mapping,
            cpu_full_exact=cpu_full_exact,
            cpu_valid_num_blocks=valid_num_blocks,
            num_reqs_padded=num_reqs_padded,
            step=0,
            output_workspace=output_workspace,
        )
        return inputs

    def _build_attn_metadata(
        self,
        num_reqs: int,
        batch_desc: Any,
        query_start_loc_np: np.ndarray,
        seq_lens_cpu_upper_bound: torch.Tensor,
        step: int,
        causal: bool | Mapping[int, bool] = True,
        dcp_local_seq_lens: torch.Tensor | None = None,
        slot_mappings: torch.Tensor | None = None,
    ) -> dict[str, Any] | None:
        if not hasattr(self, "_draft_window_graph_binding_provider") or self._draft_window_bypass_active:
            slot_kwargs = {} if slot_mappings is None else {"slot_mappings": slot_mappings}
            return super()._build_attn_metadata(  # type: ignore[misc]
                num_reqs,
                batch_desc,
                query_start_loc_np,
                seq_lens_cpu_upper_bound,
                step,
                causal,
                dcp_local_seq_lens,
                **slot_kwargs,
            )
        if not self.draft_attn_layer_names:
            return None
        padded = batch_desc.num_reqs or num_reqs
        tokens = batch_desc.num_tokens
        expected_query: np.ndarray = np.arange(num_reqs + 1, dtype=query_start_loc_np.dtype) * self.num_query_per_req
        if (
            step != self.num_query_per_req
            or batch_desc.cg_mode.name != "FULL"
            or tokens != padded * self.num_query_per_req
            or query_start_loc_np.shape != expected_query.shape
            or not np.array_equal(query_start_loc_np, expected_query)
            or self._draft_window_selected_metadata != batch_desc
        ):
            raise RuntimeError("Draft Window requires its actual uniform FULL dispatch; no eager fallback")
        from vllm_ascend.worker.v2.spec_decode.draft_graph_review_limits import (
            validate_full8_single_request_graph_stage,
        )

        validate_full8_single_request_graph_stage(self, actual=num_reqs, padded=padded, tokens=tokens)
        full_slots = self.block_tables.slot_mappings[:, :tokens]
        if slot_mappings is not None and (
            slot_mappings.shape != full_slots.shape
            or slot_mappings.dtype != full_slots.dtype
            or slot_mappings.device != full_slots.device
            or slot_mappings.stride() != full_slots.stride()
            or slot_mappings.data_ptr() != full_slots.data_ptr()
        ):
            raise RuntimeError("Draft Graph metadata must retain the canonical Full write slots")
        self._materialize_draft_window_read_inputs(num_reqs=num_reqs, num_reqs_padded=padded)
        # The fixed manager builds this bank's metadata once immediately before replay.
        return None

    def _require_current_draft_window_proposal(
        self,
    ) -> DraftWindowProposalContext:
        context = self._draft_window_epoch_source._active
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
