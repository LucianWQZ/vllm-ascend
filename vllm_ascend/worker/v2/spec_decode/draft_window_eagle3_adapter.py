# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Static Eagle3 Graph owner adapter for the fixed 544 MRV2 interfaces.

The ordinary PR eager owner and MTP base remain unchanged. This adapter reuses
existing proposal math and per-consumer banks; it does not own sampling, input
kernels, model weights, or the upstream autoregressive loop.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.worker.v2.spec_decode.draft_window import (
    DraftMetadataInputs,
    DraftWindowWorkspace,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_eager import (
    build_draft_window_attn_metadata,
    draft_window_read_inputs,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_eagle3 import (
    Eagle3DraftWindowMixin,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_graph import (
    DraftWindowGraphBindingProvider,
    DraftWindowGraphRoute,
    consume_draft_window_graph_metadata,
)
from vllm_ascend.worker.v2.spec_decode.target_dependent_ar.speculator import AscendTargetDependentARSpeculator


class Eagle3GraphIntegrationMixin(Eagle3DraftWindowMixin, AscendTargetDependentARSpeculator, EagleSpeculator):
    decode_cudagraph_manager: Any

    def _initialize_draft_window_source_capability(self) -> None:
        """Atomically bind eligible P2 source and P3 ownership state."""
        policy = getattr(self, "_draft_window_policy", None)
        if policy is None:
            return
        if hasattr(self, "_draft_window_graph_binding_provider"):
            raise RuntimeError("draft-window runtime ownership is already initialized")
        from vllm_ascend.worker.v2.spec_decode.draft_window_eagle3_capture import build_eagle3_graph_source

        capability = build_eagle3_graph_source(self)
        self._draft_window_graph_binding_provider = DraftWindowGraphBindingProvider(source=capability)
        try:
            self._initialize_eagle3_draft_window_state()
        except BaseException:
            del self._draft_window_graph_binding_provider
            raise

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
        if session is not None and session._capture_call is not None:
            attn_metadata, _ = session.capture_metadata(attn_metadata)
        elif self._eagle3_window_is_active():
            raise RuntimeError("Eagle3 Window uses its FULL Graph managers; ordinary eager uses its own runtime")
        return super()._run_model(
            num_tokens,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
            mm_inputs,
        )

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
                or seq_lens_cpu_upper_bound is None
                or num_speculative_steps not in (None, self.num_speculative_steps)
            ):
                raise RuntimeError("Eagle3 Window requires its actual static-K FULL AR descriptor; no eager fallback")
            try:
                padded = batch_desc.num_reqs or num_reqs
                self._snapshot_eagle3_decode_step_one(num_reqs=num_reqs, num_reqs_padded=padded)
                for step in range(1, self.num_speculative_steps):
                    bank = session.provider.prepare_workspace(
                        session.resolve(first=False, n=padded, tokens=batch_desc.num_tokens, ordinal=step - 1)
                    )
                    inputs = self._materialize_eagle3_step(
                        step=step,
                        num_reqs=num_reqs,
                        num_reqs_padded=padded,
                        seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                        output_workspace=bank,
                    )
                    session.stage(inputs, first=False, n=padded, tokens=batch_desc.num_tokens, ordinal=step - 1)
            except BaseException:
                self._quarantine_eagle3_window()
                raise
        return super()._multi_step_decode(
            num_reqs,
            skip_attn,
            batch_desc,
            num_tokens_across_dp,
            seq_lens_cpu_upper_bound,
            num_speculative_steps,
        )

    def build_draft_attn_metadatas(self, num_reqs_padded, num_tokens_padded, is_draft_model_prefill):
        if getattr(self, "_eagle3_graph_session", None) is None or not self._eagle3_window_is_active():
            return super().build_draft_attn_metadatas(num_reqs_padded, num_tokens_padded, is_draft_model_prefill)
        route = (
            DraftWindowGraphRoute.EAGLE3_FIRST_PASS if is_draft_model_prefill else DraftWindowGraphRoute.EAGLE3_AR_STEP
        )
        return consume_draft_window_graph_metadata(
            self,
            route=route,
            expected_consumer_count=1 if is_draft_model_prefill else self.num_speculative_steps - 1,
        )

    def _build_eagle3_graph_metadata(
        self,
        *,
        inputs: DraftMetadataInputs,
        workspace: DraftWindowWorkspace,
        num_reqs: int,
        num_reqs_padded: int,
        num_tokens_padded: int,
        query_start_loc_np,
        target_metadata: Mapping[Any, Any],
        first_pass: bool,
    ) -> dict[str, Any]:
        """Use the normal builder with this consumer's persistent read bank."""
        inputs.validate_for(inputs.identity)
        if self.block_tables.cp_size != 1:
            raise NotImplementedError("MRV2 Eagle3 Draft Window does not support DCP")
        if self.model_state.attn_metadata is not target_metadata:
            raise RuntimeError("Target metadata owner changed before Eagle3 Draft metadata build")
        view = draft_window_read_inputs(
            workspace,
            actual_request_count=num_reqs,
            padded_request_count=num_reqs_padded,
            block_tables=self.block_tables.input_block_tables,
            global_group_id=inputs._owner.source.group.gid,
        )
        state = next(layer.attn_state for layer in target_metadata.values() if layer is not None)
        metadata = build_draft_window_attn_metadata(
            self,
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs_padded,
            num_tokens_padded=num_tokens_padded,
            read_inputs=view,
            query_width=1,
            causal=True,
            query_start_loc_np=query_start_loc_np,
            dcp_local_seq_lens=None,
            attn_state=state if first_pass else AscendAttentionState.DecodeOnly,
            parallel_config=self.draft_vllm_config.parallel_config,
        )
        if not metadata:
            raise RuntimeError("eligible Eagle3 Window route produced no Draft metadata")
        if self.model_state.attn_metadata is not target_metadata:
            raise RuntimeError("Target metadata owner changed during Eagle3 Draft metadata build")
        for layer in metadata.values():
            if layer is None:
                continue
            layer.seq_lens = workspace.device_window_exact
            layer.seq_lens_cpu = workspace.cpu_window_exact
            layer.seq_lens_list = workspace.cpu_window_exact.tolist()
        return metadata

    def _require_eagle3_target_metadata(
        self,
        passed_metadata: dict[str, Any] | None = None,
    ) -> Mapping[Any, Any]:
        target_metadata = self.model_state.attn_metadata
        if not isinstance(target_metadata, Mapping) or not target_metadata:
            raise ValueError("Window ON requires non-empty Target metadata from model_state.attn_metadata")
        if passed_metadata is not None and passed_metadata is not target_metadata:
            raise RuntimeError("Eagle3 first pass did not receive the real Target metadata owner")
        return target_metadata
