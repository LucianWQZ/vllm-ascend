# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Static Eagle3 Graph owner adapter for the fixed 544 MRV2 interfaces.

The ordinary PR eager owner and MTP base remain unchanged. This adapter reuses
existing proposal math and per-consumer banks; it does not own sampling, input
kernels, model weights, or the upstream autoregressive loop.
"""

from __future__ import annotations

from typing import Any

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.worker.v2.spec_decode.draft_window_eagle3 import (
    Eagle3DraftWindowMixin,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_graph import (
    DraftWindowGraphBindingProvider,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_graph_metadata import (
    bind_draft_window_read,
    copy_draft_attention_metadata,
)
from vllm_ascend.worker.v2.spec_decode.target_dependent_ar.speculator import AscendTargetDependentARSpeculator


class Eagle3GraphIntegrationMixin(Eagle3DraftWindowMixin, AscendTargetDependentARSpeculator, EagleSpeculator):
    decode_cudagraph_manager: Any
    _eagle3_window_next_step: int

    def _initialize_draft_window_source_capability(self) -> None:
        """Initialize the validated Graph source and its proposal state."""
        policy = getattr(self, "_draft_window_policy", None)
        if policy is None:
            return
        if hasattr(self, "_draft_window_graph_binding_provider"):
            raise RuntimeError("draft-window runtime ownership is already initialized")
        from vllm_ascend.worker.v2.spec_decode.draft_window_eagle3_capture import build_eagle3_graph_source

        capability = build_eagle3_graph_source(self)
        self._draft_window_graph_binding_provider = DraftWindowGraphBindingProvider(source=capability)
        self._initialize_eagle3_draft_window_state()

    def _configure_decode_metadata_update(self) -> None:
        super()._configure_decode_metadata_update()
        if getattr(self, "_draft_window_policy", None) is not None:
            # The fixed merged AR route stages a canonical view per consumer.
            self.supports_decode_metadata_update = False

    def set_attn(self, *args, **kwargs):
        result = super().set_attn(*args, **kwargs)
        self._initialize_draft_window_source_capability()
        return result

    def init_cudagraph_manager(self, cudagraph_mode):
        result = super().init_cudagraph_manager(cudagraph_mode)
        if hasattr(self, "_draft_window_graph_binding_provider"):
            from vllm_ascend.worker.v2.spec_decode.draft_window_eagle3_capture import Eagle3GraphSession

            self._eagle3_graph_session = Eagle3GraphSession(self)
        return result

    def _prefill(self, *args, **kwargs):
        # DP synchronization may select native first even when this replica
        # has no true prefill rows. Track the actual AR entry, not that flag.
        self._eagle3_window_native_prefill_active = True
        try:
            return super()._prefill(*args, **kwargs)
        finally:
            self._eagle3_window_native_prefill_active = False

    def _run_model(
        self,
        num_tokens,
        attn_metadata,
        slot_mappings,
        num_tokens_across_dp,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        mm_inputs=None,
    ):
        session = getattr(self, "_eagle3_graph_session", None)
        prefill_inputs = None
        if session is not None and session._capture_call is not None:
            attn_metadata = session.capture_metadata(attn_metadata)
        elif self._eagle3_window_is_active():
            if (
                session is None
                or self._eagle3_window_next_step != 0
                or not getattr(self, "_eagle3_window_native_prefill_active", False)
                or cudagraph_runtime_mode not in (CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE)
            ):
                raise RuntimeError("Eagle3 Window decode uses its FULL Graph managers; no eager fallback")
            # The fixed AR proposer executes its synchronized native first
            # outside FULL graphs. Reuse its prepared query/write metadata and
            # the same persistent first-reader bank, not a second eager owner.
            n = self.input_batch.num_reqs
            queries = self.input_batch.query_start_loc_np[1 : n + 1].tolist()
            if any(
                layer is not None
                and (
                    layer.actual_seq_lengths_q != queries
                    or layer.num_actual_tokens != queries[-1]
                    or queries[-1] > num_tokens
                    or layer.block_tables.shape[0] < n
                    or len(layer.seq_lens_list) < n
                )
                for layer in attn_metadata.values()
            ):
                raise NotImplementedError(
                    "Eagle3 Window native prefill requires canonical effective query rows; "
                    "additional padding requests are not qualified for this read bank"
                )
            capture = session.resolve(first=True, n=n, tokens=n * (self.num_speculative_steps + 1), ordinal=0)
            self.block_tables.gather_block_tables(self.input_batch.idx_mapping, num_reqs_padded=n)
            bank = prefill_inputs = self._materialize_eagle3_step(capture, 0)
            attn_metadata = copy_draft_attention_metadata(attn_metadata)
            max_seq_len = int(bank.cpu_window_exact.max().item())
            for layer in attn_metadata.values():
                if layer is not None:
                    bind_draft_window_read(layer, bank, max_seq_len)
                    layer.seq_lens_gpu = bank.device_window_exact
                    layer.attn_state = AscendAttentionState.ChunkedPrefill
        output = super()._run_model(
            num_tokens,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
            mm_inputs,
        )
        if prefill_inputs is not None:
            # Host preparation ends here, not the device reader. The original
            # proposal scope quarantines any later failure without losing banks.
            self._draft_window_graph_binding_provider.complete_native()
            self._eagle3_window_next_step = 1
        return output

    def _multi_step_decode(
        self,
        num_reqs,
        skip_attn,
        batch_desc,
        num_tokens_across_dp,
        seq_lens_cpu_upper_bound=None,
        num_speculative_steps=None,
    ):
        if self._eagle3_window_is_active():
            session = self._eagle3_graph_session
            if (
                self._eagle3_window_expected_steps == 1
                or batch_desc.cg_mode != CUDAGraphMode.FULL
                or skip_attn
                or num_speculative_steps not in (None, self.num_speculative_steps)
            ):
                raise RuntimeError("Eagle3 Window requires its actual static-K FULL AR descriptor; no eager fallback")
            padded = batch_desc.num_reqs or num_reqs
            self._snapshot_eagle3_decode_step_one(num_reqs=num_reqs, num_reqs_padded=padded)
            for step in range(1, self.num_speculative_steps):
                capture = session.resolve(first=False, n=padded, tokens=batch_desc.num_tokens, ordinal=step - 1)
                session.stage(capture, step)
        return super()._multi_step_decode(
            num_reqs,
            skip_attn,
            batch_desc,
            num_tokens_across_dp,
            seq_lens_cpu_upper_bound,
            num_speculative_steps,
        )
