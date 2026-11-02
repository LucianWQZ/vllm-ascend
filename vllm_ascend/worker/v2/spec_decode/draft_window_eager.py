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
    _build_draft_window_source_capability_for_context,
    gather_valid_num_blocks_cpu,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_graph import DraftWindowGraphBindingProvider
from vllm_ascend.worker.v2.spec_decode.draft_window_read import DraftReadView
from vllm_ascend.worker.v2.spec_decode.draft_window_runtime import continuing_prefill_rows


@dataclass(frozen=True, slots=True, eq=False)
class DraftWindowProposalContext:
    """Caller-owned identity for one real outer ``propose`` invocation."""

    input_batch: InputBatch | Any
    ordered_request_ids: tuple[str, ...]
    idx_mapping_fingerprint: tuple[int, ...]
    full_read_rows: tuple[bool, ...]

    @classmethod
    def from_batch(cls, input_batch: InputBatch | Any) -> "DraftWindowProposalContext":
        requests, mapping = _proposal_fingerprint(input_batch)
        return cls(input_batch, requests, mapping, classify_draft_window_batch_phase(input_batch))

    def validate(self, input_batch: InputBatch | Any) -> None:
        if input_batch is not self.input_batch:
            raise RuntimeError("proposal input batch object changed")
        requests, mapping = _proposal_fingerprint(input_batch)
        if requests != self.ordered_request_ids or mapping != self.idx_mapping_fingerprint:
            raise RuntimeError("proposal request ordering or idx_mapping changed")
        if classify_draft_window_batch_phase(input_batch) != self.full_read_rows:
            raise RuntimeError("proposal row phases changed")


def classify_draft_window_batch_phase(input_batch: Any) -> tuple[bool, ...]:
    """Classify every active row from pinned ``InputBatch`` source fields."""
    num_reqs = int(input_batch.num_reqs)
    # The common eager/Graph helper checks each real prompt-progress counter,
    # including speculative placeholders, once before deriving these rows.
    continuing = continuing_prefill_rows(input_batch, num_reqs)
    computed = np.asarray(input_batch.num_computed_prefill_tokens_np)[:num_reqs]
    prefill_len = np.asarray(input_batch.prefill_len_np)[:num_reqs]
    declared = np.asarray(input_batch.is_prefilling_np)
    if declared.ndim != 1 or declared.shape[0] < num_reqs:
        raise ValueError("is_prefilling does not cover active request rows")
    actual_prefill = computed < prefill_len
    if not np.array_equal(actual_prefill, declared[:num_reqs].astype(np.bool_, copy=False)):
        raise RuntimeError("InputBatch is_prefilling rows disagree with computed/prefill lengths")
    if bool(getattr(input_batch, "has_prefill", False)) != bool(np.any(actual_prefill)):
        raise RuntimeError("InputBatch has_prefill disagrees with row phase state")
    return tuple(bool(value) for value in continuing)


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
        # Target metadata also contains indexer/cache entries without a state.
        # Use the same current-batch input as the normal Target builder.
        attn_state = owner.input_batch.attn_state
    if parallel_config is None:
        parallel_config = owner.attn_vllm_config.parallel_config
    window_group = owner._draft_window_graph_binding_provider.source.group
    window_gid = window_group.gid
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
    # One executable group is admitted at initialization. The original
    # builder shares its newly created metadata across that group's layers.
    # Bind the two Graph-specific device operands once on this actual object.
    layer = metadata[window_group.layers[0]]
    layer.query_start_loc = owner.input_buffers.query_start_loc[: num_reqs_padded + 1]
    layer.seq_lens = read_inputs.seq_lens
    return metadata


class DraftWindowGraphInputMixin:
    """Proposal input ownership for DFlash, DFlash2 and eligible DSpark Graph."""

    _speculator_name: str
    _group_causal: bool | Mapping[int, bool]
    _draft_window_graph_binding_provider: DraftWindowGraphBindingProvider
    _draft_window_proposal: DraftWindowProposalContext | None
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
    attn_groups: Any
    kv_cache_config: Any
    num_query_per_req: int
    max_model_len: int

    def _initialize_draft_window_source_capability(self) -> None:
        """Bind the eligible Source and its Graph input owner atomically."""
        policy = getattr(self, "_draft_window_policy", None)
        if policy is None:
            return
        if hasattr(self, "_draft_window_graph_binding_provider"):
            raise RuntimeError("draft-window runtime ownership is already initialized")
        if hasattr(self, "_draft_window_runtime"):
            raise RuntimeError("Graph Window cannot replace the typed eager runtime")
        capability = _build_draft_window_source_capability_for_context(
            policy=policy,
            runtime_route=self._speculator_name.lower(),
            attn_groups=self.attn_groups,
            block_tables=self.block_tables,
            kv_cache_config=self.kv_cache_config,
            causal=self._group_causal,
            vllm_config=self.vllm_config,
            dflash2_graph_owner=self if self._speculator_name.lower() == "dflash2" else None,
        )
        self._draft_window_graph_binding_provider = DraftWindowGraphBindingProvider(source=capability)
        self._initialize_draft_window_graph_inputs()

    def _initialize_draft_window_graph_inputs(self) -> None:
        if not hasattr(self, "_draft_window_graph_binding_provider"):
            raise RuntimeError("Draft Window Graph inputs require an initialized Source owner")
        if hasattr(self, "_draft_window_proposal"):
            raise RuntimeError("Draft Window Graph inputs are already initialized")
        self._draft_window_proposal = None
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
        if not hasattr(self, "_draft_window_graph_binding_provider"):
            yield
            return
        if self._draft_window_bypass_active or self._draft_window_proposal is not None:
            raise RuntimeError("a draft-window proposal scope is already active")
        if self._draft_window_graph_binding_provider.quarantined or getattr(
            self, "_draft_window_graph_submission_uncertain", False
        ):
            raise RuntimeError("draft-window backing is quarantined after uncertain Graph submission")
        bypass_for_prefill = False
        if not dummy_run and not is_profile:
            bypass_for_prefill = _requires_prefill_baseline_bypass(input_batch)
        if dummy_run or is_profile or bypass_for_prefill:
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
                and bool(self._draft_window_graph_binding_provider._carriers)
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
        context = DraftWindowProposalContext.from_batch(input_batch)
        self._draft_window_proposal = context
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
                if live_inputs is not None:
                    if succeeded:
                        try:
                            self._draft_window_graph_binding_provider.retire_generation(live_inputs)
                            self._draft_window_graph_binding_provider.finish_proposal()
                        except BaseException:
                            self._draft_window_graph_submission_uncertain = True
                            succeeded = False
                            raise
                    else:
                        # Retain uncertain backing without device RPC on unwind.
                        self._draft_window_graph_binding_provider.quarantine_generation(live_inputs)
                elif succeeded:
                    raise RuntimeError("Draft Graph proposal completed without consuming its current inputs")
            finally:
                self._draft_window_selected_metadata = None
                self._draft_window_proposal = None

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
            capability,
            idx_mapping_np=input_batch.idx_mapping_np,
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs_padded,
        )
        # Recheck request order after the controlled exact-length snapshot.
        context.validate(input_batch)
        inputs = self._draft_window_graph_binding_provider.materialize(
            proposal=context,
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
        context = self._draft_window_proposal
        if context is None:
            raise RuntimeError("Draft Window metadata build has no active proposal")
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
    if any(not isinstance(request, str) or not request for request in request_ids) or len(set(request_ids)) != num_reqs:
        raise ValueError("proposal requires unique nonempty request IDs")
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
    full_read_rows = classify_draft_window_batch_phase(input_batch)
    return bool(full_read_rows) and all(full_read_rows)
