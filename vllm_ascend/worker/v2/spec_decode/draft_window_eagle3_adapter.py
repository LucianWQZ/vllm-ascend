# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Static Eagle3 Graph owner adapter for the fixed 544 MRV2 interfaces.

The ordinary PR eager owner and MTP base remain unchanged. This adapter reuses
existing proposal math and per-consumer banks; it does not own sampling, input
kernels, model weights, or the upstream autoregressive loop.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.worker.v2.spec_decode.draft_window_eagle3 import (
    Eagle3DraftWindowMixin,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_graph import DraftWindowGraphBucket
from vllm_ascend.worker.v2.spec_decode.draft_window_graph_metadata import (
    bind_draft_window_read,
    copy_draft_attention_metadata,
)
from vllm_ascend.worker.v2.spec_decode.target_dependent_ar.speculator import AscendTargetDependentARSpeculator


class Eagle3GraphIntegrationMixin(Eagle3DraftWindowMixin, AscendTargetDependentARSpeculator, EagleSpeculator):
    decode_cudagraph_manager: Any
    _eagle3_window_next_step: int
    _eagle3_capture_call: Iterator[DraftWindowGraphBucket] | None

    def _configure_decode_metadata_update(self) -> None:
        super()._configure_decode_metadata_update()
        if getattr(self, "_draft_window_policy", None) is not None:
            # The fixed merged AR route stages a canonical view per consumer.
            self.supports_decode_metadata_update = False

    def init_cudagraph_manager(self, cudagraph_mode):
        result = super().init_cudagraph_manager(cudagraph_mode)
        provider = getattr(self, "_draft_window_graph_binding_provider", None)
        if provider is not None:
            self._eagle3_capture_call = None
            capacity = min(self.max_model_len, provider.window_size + provider.group.table_column_span - 1)
            for first, manager in ((True, self.prefill_cudagraph_manager), (False, self.decode_cudagraph_manager)):
                if self.update_stream is None or manager.update_stream is not self.update_stream:
                    raise RuntimeError("Eagle3 capture managers must share their owner UPDATE stream")
                q = self.num_speculative_steps + 1 if first else 1
                for descriptors in manager._capture_descs.values():
                    for desc in descriptors:
                        if desc.cg_mode.name != "FULL":
                            raise NotImplementedError("Eagle3 Window capture requires an actual FULL descriptor")
                        n = desc.num_reqs
                        if n is None or desc.num_tokens != n * q:
                            raise RuntimeError("Eagle3 startup descriptor does not match its request/query width")
                        for ordinal in range(1 if first else self.num_speculative_steps - 1):
                            bank = provider.register_capture_identity(descriptor=desc, consumer_ordinal=ordinal)
                            # Seed before GLOBAL capture; no H2D belongs in its callback.
                            bank.window_block_table.zero_()
                            bank.cpu_window_exact.fill_(capacity)
                            bank.device_window_exact.copy_(bank.cpu_window_exact)
            if not provider._carriers:
                raise RuntimeError("Eagle3 Window requires a nonempty capture bucket inventory")
        return result

    @contextmanager
    def _eagle3_capture_scope(self, *, n: int, tokens: int):
        if self._eagle3_capture_call is not None:
            raise RuntimeError("nested Eagle3 startup capture invocation")
        self._eagle3_capture_call = iter(
            self._draft_window_graph_binding_provider.capture_banks_for(num_reqs=n, num_tokens=tokens)
        )
        try:
            yield
            if next(self._eagle3_capture_call, None) is not None:
                raise RuntimeError("Eagle3 capture did not visit every actual first-pass/AR model consumer")
        finally:
            self._eagle3_capture_call = None

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
        prefill_inputs = None
        capture_call = getattr(self, "_eagle3_capture_call", None)
        if capture_call is not None:
            provider = self._draft_window_graph_binding_provider
            bank = next(capture_call, None)
            if bank is None:
                raise RuntimeError("Eagle3 capture visited more consumers than its real descriptor")
            n = bank.descriptor.num_reqs
            capacity = min(self.max_model_len, provider.window_size + provider.group.table_column_span - 1)
            attn_metadata = copy_draft_attention_metadata(attn_metadata)
            query = self.input_buffers.query_start_loc
            if query.dtype != torch.int32 or query.ndim != 1 or query.numel() < n + 1:
                raise RuntimeError("Eagle3 capture requires its owned int32 Full input query")
            for layer in attn_metadata.values():
                if layer is not None:
                    bind_draft_window_read(layer, bank, capacity)
                    layer.causal = True
                    # Use the owned Full query, not a builder's transient GPU copy.
                    layer.query_start_loc = query[: n + 1]
        elif self._eagle3_window_is_active():
            provider = self._draft_window_graph_binding_provider
            if (
                self._eagle3_window_next_step != 0
                or not getattr(self, "_eagle3_window_native_prefill_active", False)
                or cudagraph_runtime_mode not in (CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE)
            ):
                raise RuntimeError("Eagle3 Window decode uses its FULL Graph managers; no eager fallback")
            # The fixed AR proposer executes its synchronized native first
            # outside FULL graphs. Reuse its prepared query/write metadata and
            # the same persistent first-reader bank, not a second eager owner.
            n = self.input_batch.num_reqs
            queries = self.input_batch.query_start_loc_np[1 : n + 1].tolist()
            # The AR entry already filters Draft layers; the supported
            # single-group builder publishes one shared header for them.
            layer = attn_metadata[self._draft_window_graph_binding_provider.group.layers[0]]
            if (
                layer.actual_seq_lengths_q != queries
                or layer.num_actual_tokens != queries[-1]
                or queries[-1] > num_tokens
                or layer.block_tables.shape[0] < n
                or len(layer.seq_lens_list) < n
            ):
                raise NotImplementedError(
                    "Eagle3 Window native prefill requires canonical effective query rows; "
                    "additional padding requests are not qualified for this read bank"
                )
            capture = provider.capture_banks_for(num_reqs=n, num_tokens=n * (self.num_speculative_steps + 1))[0]
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
            provider = self._draft_window_graph_binding_provider
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
                capture = provider.capture_banks_for(num_reqs=padded, num_tokens=batch_desc.num_tokens)[step - 1]
                self._materialize_eagle3_step(capture, step)
                provider.bind_current(capture.descriptor)
                self._eagle3_window_next_step = step + 1
        return super()._multi_step_decode(
            num_reqs,
            skip_attn,
            batch_desc,
            num_tokens_across_dp,
            seq_lens_cpu_upper_bound,
            num_speculative_steps,
        )
