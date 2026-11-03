# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Graph-owned proposal inputs for the MRV2 Draft Window read plane.

This module joins real proposal inputs to capture-owned Window buffers and
the normal Ascend metadata builder. Positions, query layout, slot mappings
and Target metadata remain owned by the existing runner.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
from vllm.v1.worker.gpu.input_batch import InputBatch

from vllm_ascend.worker.v2.spec_decode.draft_window import (
    create_draft_window_graph_provider,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_graph import (
    DraftWindowGraphBindingProvider,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_runtime import continuing_prefill_rows


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
        self._draft_window_graph_binding_provider = create_draft_window_graph_provider(self)
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
        rows = (
            None
            if dummy_run or is_profile
            else tuple(map(bool, continuing_prefill_rows(input_batch, input_batch.num_reqs)))
        )
        bypass_for_prefill = rows is not None and bool(rows) and all(rows)
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
        assert rows is not None
        provider = self._draft_window_graph_binding_provider
        with provider.proposal_scope(rows):
            yield
            if provider._active_bank is None:
                raise RuntimeError("Draft Graph proposal completed without consuming its current inputs")

    def _build_uniform_attn_metadata(
        self,
        batch_desc: Any,
        num_reqs: int,
        num_query_per_req: int,
        seq_lens_cpu_upper_bound: torch.Tensor,
        step: int,
        causal: bool | Mapping[int, bool] = True,
        dcp_local_seq_lens: torch.Tensor | None = None,
    ) -> dict[str, Any] | None:
        if not hasattr(self, "_draft_window_graph_binding_provider") or self._draft_window_bypass_active:
            return super()._build_uniform_attn_metadata(  # type: ignore[misc]
                batch_desc,
                num_reqs,
                num_query_per_req,
                seq_lens_cpu_upper_bound,
                step,
                causal,
                dcp_local_seq_lens,
            )
        if not self.draft_attn_layer_names:
            return None
        padded = batch_desc.num_reqs or num_reqs
        tokens = batch_desc.num_tokens
        if (
            num_query_per_req != self.num_query_per_req
            or step != self.num_query_per_req
            or batch_desc.cg_mode.name != "FULL"
            or tokens != padded * self.num_query_per_req
            or (getattr(self, "_draft_graph_review_limit", None) is not None and padded != num_reqs)
        ):
            raise RuntimeError("Draft Window requires its actual uniform FULL dispatch; no eager fallback")
        provider = self._draft_window_graph_binding_provider
        rows = provider.proposal
        input_batch = self.input_batch
        if rows is None:
            raise RuntimeError("Draft Window metadata build has no active proposal")
        if num_reqs != input_batch.num_reqs or num_reqs != len(rows):
            raise RuntimeError("metadata request count does not match proposal epoch")
        if padded < num_reqs:
            raise ValueError("padded request count must be at least actual requests")
        selected = provider.capture_identity_for(descriptor=batch_desc)
        # The private bank initializes padded rows; snapshot only actual requests.
        cpu_full_exact = self.input_buffers.seq_lens[:num_reqs].to(device="cpu", dtype=torch.int32, copy=True)
        provider.prepare(selected, cpu_full_exact=cpu_full_exact, idx_mapping_np=input_batch.idx_mapping_np)
        # The fixed manager builds this bank's metadata once immediately before replay.
        return None
