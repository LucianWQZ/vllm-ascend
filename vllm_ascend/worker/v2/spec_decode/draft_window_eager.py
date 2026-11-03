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
    _build_draft_window_source_capability_for_context,
    gather_valid_num_blocks_cpu,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_graph import (
    DraftWindowGraphBindingProvider,
    resolve_parallel_draft_window_graph_route,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_runtime import continuing_prefill_rows


@dataclass(frozen=True, slots=True, eq=False)
class DraftWindowProposalContext:
    """One real propose scope and its fixed full/window row semantics.

    The fixed runner prepares request mapping and prompt progress before the
    synchronous propose call; only device lengths and consumer steps change
    inside it. Keep one phase snapshot, not repeated copies of runner IDs.
    """

    input_batch: InputBatch | Any
    full_read_rows: tuple[bool, ...]

    @classmethod
    def from_batch(cls, input_batch: InputBatch | Any) -> "DraftWindowProposalContext":
        return cls(input_batch, tuple(map(bool, continuing_prefill_rows(input_batch, input_batch.num_reqs))))

    def validate(self, input_batch: InputBatch | Any) -> None:
        if input_batch is not self.input_batch or input_batch.num_reqs != len(self.full_read_rows):
            raise RuntimeError("proposal input batch or request count changed")


class DraftWindowGraphInputMixin:
    """Proposal input ownership for DFlash, DFlash2 and eligible DSpark Graph."""

    _speculator_name: str
    _group_causal: bool | Mapping[int, bool]
    _draft_window_graph_binding_provider: DraftWindowGraphBindingProvider
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
        if hasattr(self, "_draft_window_bypass_active"):
            raise RuntimeError("Draft Window Graph inputs are already initialized")
        self._draft_window_bypass_active = False
        self._draft_window_prefill_bypass_active = False
        self._draft_window_dp_dummy_active = False

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
        if self._draft_window_bypass_active or self._draft_window_graph_binding_provider.proposal is not None:
            raise RuntimeError("a draft-window proposal scope is already active")
        if self._draft_window_graph_binding_provider.quarantined:
            raise RuntimeError("draft-window backing is quarantined after uncertain Graph submission")
        context = None if dummy_run or is_profile else DraftWindowProposalContext.from_batch(input_batch)
        bypass_for_prefill = context is not None and bool(context.full_read_rows) and all(context.full_read_rows)
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
                # production Window generation. Mixed rows retain the phase
                # snapshot for their actual per-row Window reads.
                yield
            except BaseException:
                if self._draft_window_dp_dummy_active:
                    # Dummy preparation/UPDATE can also partially submit.
                    # Retain owners and refuse the next scope without any
                    # device inspection, reset, drain or attempted recovery.
                    self._draft_window_graph_binding_provider.quarantine()
                raise
            finally:
                self._draft_window_bypass_active = False
                self._draft_window_prefill_bypass_active = False
                self._draft_window_dp_dummy_active = False
            return
        assert context is not None
        provider = self._draft_window_graph_binding_provider
        with provider.proposal_scope(context):
            yield
            if provider._active_bank is None:
                raise RuntimeError("Draft Graph proposal completed without consuming its current inputs")

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
        batch_desc: Any,
    ) -> None:
        context = self._require_current_draft_window_proposal()
        input_batch = self.input_batch
        num_reqs_padded = batch_desc.num_reqs
        if num_reqs != input_batch.num_reqs:
            raise RuntimeError("metadata request count does not match proposal epoch")
        if num_reqs_padded < num_reqs:
            raise ValueError("padded request count must be at least actual requests")

        provider = self._draft_window_graph_binding_provider
        route = resolve_parallel_draft_window_graph_route(self)
        if route is None:
            raise RuntimeError("Draft Window has no parallel Graph capture route")
        selected = provider.capture_identity_for(route=route, descriptor=batch_desc)
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
        provider.prepare(
            selected,
            cpu_full_exact=cpu_full_exact,
            cpu_valid_num_blocks=valid_num_blocks,
            step=0,
        )

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
            or (getattr(self, "_draft_graph_review_limit", None) is not None and padded != num_reqs)
            or query_start_loc_np.shape != expected_query.shape
            or not np.array_equal(query_start_loc_np, expected_query)
        ):
            raise RuntimeError("Draft Window requires its actual uniform FULL dispatch; no eager fallback")
        full_slots = self.block_tables.slot_mappings[:, :tokens]
        if slot_mappings is not None and (
            slot_mappings.shape != full_slots.shape
            or slot_mappings.dtype != full_slots.dtype
            or slot_mappings.device != full_slots.device
            or slot_mappings.stride() != full_slots.stride()
            or slot_mappings.data_ptr() != full_slots.data_ptr()
        ):
            raise RuntimeError("Draft Graph metadata must retain the canonical Full write slots")
        self._materialize_draft_window_read_inputs(num_reqs=num_reqs, batch_desc=batch_desc)
        # The fixed manager builds this bank's metadata once immediately before replay.
        return None

    def _require_current_draft_window_proposal(
        self,
    ) -> DraftWindowProposalContext:
        context = self._draft_window_graph_binding_provider.proposal
        if context is None:
            raise RuntimeError("Draft Window metadata build has no active proposal")
        return context
