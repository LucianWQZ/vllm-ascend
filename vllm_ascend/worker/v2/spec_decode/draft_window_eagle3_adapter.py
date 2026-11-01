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

import numpy as np
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.worker.v2.attn_utils import build_attn_metadata_factory
from vllm_ascend.worker.v2.spec_decode.draft_window import (
    DraftMetadataInputs,
    DraftWindowWorkspace,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_eager import (
    AscendDraftAttentionReadInputs,
    build_draft_window_attn_metadata,
    draft_window_read_inputs,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_eagle3 import (
    Eagle3DraftWindowMixin,
    _eagle3_graph_scheduled_consumers,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_graph import (
    DraftWindowGraphBindingProvider,
    DraftWindowGraphRoute,
    consume_draft_window_graph_metadata,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_graph_metadata import copy_draft_attention_metadata
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

    def propose(
        self,
        input_batch,
        attn_metadata,
        slot_mappings,
        last_hidden_states,
        aux_hidden_states,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        temperature,
        seeds,
        dp_sync_state=None,
        dummy_run=False,
        skip_attn_for_dummy_run=False,
        mm_inputs=None,
        is_profile=None,
        num_speculative_tokens=None,
    ):
        from vllm_ascend.worker.v2.spec_decode.draft_graph_review_limits import validate_draft_graph_review_requests

        validate_draft_graph_review_requests(self, input_batch, dummy_run=dummy_run, is_profile=bool(is_profile))
        if (
            hasattr(self, "_eagle3_window_epoch_source")
            and not dummy_run
            and not is_profile
            and not getattr(input_batch, "is_dummy", False)
        ):
            _eagle3_graph_scheduled_consumers(self.num_speculative_steps, num_speculative_tokens)
        self.input_batch = input_batch
        args = (
            input_batch,
            attn_metadata,
            slot_mappings,
            last_hidden_states,
            aux_hidden_states,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            temperature,
            seeds,
        )
        kwargs = dict(
            dp_sync_state=dp_sync_state,
            dummy_run=dummy_run,
            skip_attn_for_dummy_run=skip_attn_for_dummy_run,
            mm_inputs=mm_inputs,
            is_profile=is_profile,
            num_speculative_tokens=num_speculative_tokens,
        )
        if not hasattr(self, "_eagle3_window_epoch_source"):
            return super().propose(*args, **kwargs)
        with self._eagle3_draft_window_proposal_scope(
            input_batch,
            dummy_run=dummy_run,
            is_profile=bool(is_profile),
            num_speculative_tokens=None
            if dummy_run or is_profile or getattr(input_batch, "is_dummy", False)
            else num_speculative_tokens,
        ):
            return super().propose(*args, **kwargs)

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
        return super()._run_model(
            num_tokens,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
            mm_inputs,
        )

    def _generate_draft(
        self,
        num_reqs,
        num_tokens_padded,
        attn_metadata,
        slot_mappings,
        num_tokens_across_dp,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        num_speculative_steps=None,
    ):
        inputs = None
        if self._eagle3_window_is_active():
            if cudagraph_runtime_mode != CUDAGraphMode.NONE:
                raise RuntimeError("Eagle3 Window per-step eager reader received a Graph descriptor")
            inputs = self._eagle3_window_live_inputs
            if inputs is None:
                raise RuntimeError("Eagle3 Draft forward has no current Window inputs")
            inputs.validate_for(inputs.identity)
            if attn_metadata is not inputs.metadata:
                raise RuntimeError("Eagle3 Draft forward lost its current private Window metadata")
        try:
            result = super()._generate_draft(
                num_reqs,
                num_tokens_padded,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
                num_speculative_steps,
            )
        except BaseException:
            if inputs is not None:
                self._finish_eagle3_window_step(inputs, succeeded=False)
            raise
        if inputs is not None:
            self._finish_eagle3_window_step(inputs, succeeded=True)
        return result

    def _update_decode_attn_metadata(self, attn_metadata, step, num_reqs=None):
        # The Graph-owner eager control already published exact private lengths;
        # the baseline update would replace them with Full/upper-derived lengths.
        if self._eagle3_window_is_active():
            return
        return super()._update_decode_attn_metadata(attn_metadata, step, num_reqs)

    def _multi_step_decode(
        self,
        num_reqs,
        skip_attn,
        batch_desc,
        num_tokens_across_dp,
        seq_lens_cpu_upper_bound=None,
        num_speculative_steps=None,
    ):
        session = getattr(self, "_eagle3_graph_session", None)
        if self._eagle3_window_is_active() and self._eagle3_window_expected_steps == 1:
            raise RuntimeError("zero-draft Eagle3 first-only proposal cannot enter AR Graph")
        if session is not None and self._eagle3_window_is_active() and batch_desc.cg_mode == CUDAGraphMode.FULL:
            if skip_attn or seq_lens_cpu_upper_bound is None:
                raise RuntimeError("Eagle3 Graph requires real attention and current CPU upper lengths")
            if num_speculative_steps not in (None, self.num_speculative_steps):
                raise RuntimeError("Eagle3 merged Graph differs from its static K")
            try:
                padded = batch_desc.num_reqs or num_reqs
                self._snapshot_eagle3_decode_step_one(num_reqs=num_reqs, num_reqs_padded=padded)
                for step in range(1, self.num_speculative_steps):
                    bank = session.provider.prepare_workspace(
                        session.resolve(first=False, n=padded, tokens=batch_desc.num_tokens, ordinal=step - 1)
                    )
                    inputs = self._materialize_eagle3_decode_step(
                        step=step,
                        num_reqs=num_reqs,
                        num_reqs_padded=padded,
                        seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                        output_workspace=bank,
                    )
                    session.stage(inputs, first=False, n=padded, tokens=batch_desc.num_tokens, ordinal=step - 1)
                return self.decode_cudagraph_manager.run_fullgraph(batch_desc)
            except BaseException:
                self._quarantine_eagle3_window()
                raise
        if self._eagle3_window_is_active():
            if session is not None:
                self._quarantine_eagle3_window()
                raise RuntimeError("Eagle3 Window merged AR Graph has no exact FULL descriptor; no eager fallback")
            if batch_desc.cg_mode != CUDAGraphMode.NONE or skip_attn or seq_lens_cpu_upper_bound is None:
                raise RuntimeError("Eagle3 eager proposal requires real per-step metadata")
            self._snapshot_eagle3_decode_step_one(
                num_reqs=num_reqs,
                num_reqs_padded=batch_desc.num_reqs or num_reqs,
            )
        return super()._multi_step_decode(
            num_reqs,
            skip_attn,
            batch_desc,
            num_tokens_across_dp,
            seq_lens_cpu_upper_bound,
            num_speculative_steps,
        )

    def _prefill(
        self,
        num_reqs,
        num_tokens,
        attn_metadata,
        slot_mappings,
        num_tokens_across_dp,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        mm_inputs=None,
        skip_sample=False,
    ):
        if not self._eagle3_window_is_active():
            return super()._prefill(
                num_reqs,
                num_tokens,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
                mm_inputs,
                skip_sample=skip_sample,
            )
        if cudagraph_runtime_mode != CUDAGraphMode.NONE or self.replicated_pcp:
            raise NotImplementedError("Eagle3 Window first-pass eager route requires its unsharded context")
        if slot_mappings is None:
            raise ValueError("Eagle3 first pass requires canonical Full slots")
        session = getattr(self, "_eagle3_graph_session", None)
        if session is not None:
            session.require_reusable()
        target = self._require_eagle3_target_metadata(attn_metadata)
        inputs = self._materialize_eagle3_first_pass(num_reqs=num_reqs, num_reqs_padded=num_reqs)
        try:
            private = self._build_eagle3_private_attn_metadata(
                inputs=inputs,
                num_reqs=num_reqs,
                num_reqs_padded=num_reqs,
                num_tokens_padded=num_tokens,
                query_start_loc_np=self.input_batch.query_start_loc_np,
                target_metadata=target,
                first_pass=True,
            )
            super()._prefill(
                num_reqs,
                num_tokens,
                private,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
                mm_inputs,
                skip_sample=skip_sample,
            )
        except BaseException:
            self._finish_eagle3_window_step(inputs, succeeded=False)
            raise
        self._finish_eagle3_window_step(inputs, succeeded=True)

    def _build_attn_metadata(
        self,
        num_reqs,
        batch_desc,
        query_start_loc_np,
        seq_lens_cpu_upper_bound,
        step,
        causal=True,
        dcp_local_seq_lens=None,
        slot_mappings=None,
    ):
        if not self._eagle3_window_is_active():
            return super()._build_attn_metadata(
                num_reqs,
                batch_desc,
                query_start_loc_np,
                seq_lens_cpu_upper_bound,
                step,
                causal,
                dcp_local_seq_lens,
                slot_mappings,
            )
        expected_query = self.arange_np[: num_reqs + 1]
        if (
            step <= 0
            or causal is not True
            or dcp_local_seq_lens is not None
            or query_start_loc_np.shape != expected_query.shape
            or not np.array_equal(query_start_loc_np, expected_query)
        ):
            raise ValueError("Eagle3 AR Window requires its causal uniform one-query layout")
        padded = batch_desc.num_reqs or num_reqs
        tokens = batch_desc.num_tokens if batch_desc.cg_mode == CUDAGraphMode.FULL else int(query_start_loc_np[-1])
        full_slots = self.block_tables.slot_mappings[:, :tokens]
        if slot_mappings is not None and (
            slot_mappings.shape != full_slots.shape
            or slot_mappings.dtype != full_slots.dtype
            or slot_mappings.device != full_slots.device
            or slot_mappings.stride() != full_slots.stride()
            or slot_mappings.data_ptr() != full_slots.data_ptr()
        ):
            raise RuntimeError("Eagle3 AR metadata lost canonical Full write slots")
        target = self._require_eagle3_target_metadata()
        inputs = self._materialize_eagle3_decode_step(
            step=step,
            num_reqs=num_reqs,
            num_reqs_padded=padded,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
        )
        try:
            return self._build_eagle3_private_attn_metadata(
                inputs=inputs,
                num_reqs=num_reqs,
                num_reqs_padded=padded,
                num_tokens_padded=tokens,
                query_start_loc_np=None,
                target_metadata=target,
                first_pass=False,
            )
        except BaseException:
            if self._eagle3_window_live_inputs is inputs:
                self._finish_eagle3_window_step(inputs, succeeded=False)
            raise

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

    def _build_eagle3_private_attn_metadata(
        self,
        *,
        inputs: DraftMetadataInputs,
        num_reqs: int,
        num_reqs_padded: int,
        num_tokens_padded: int,
        query_start_loc_np: np.ndarray | None,
        target_metadata: Mapping[Any, Any],
        first_pass: bool,
        read_inputs_override: AscendDraftAttentionReadInputs | None = None,
        publish: bool = True,
    ) -> dict[str, Any]:
        """Build one Draft-only metadata object from validated read inputs.

        This is the narrow Ascend-owned assembly seam required by R-REPO-01.
        It duplicates only upstream query-layout/builder kwargs; proposal,
        positions, slot calculation, forward, and sampling stay upstream.
        """
        if read_inputs_override is None:
            read_inputs = self._validated_eagle3_read_inputs(inputs, inputs.identity)
        else:
            inputs.validate_for(inputs.identity)
            read_inputs = read_inputs_override
        if self.block_tables.cp_size != 1:
            raise NotImplementedError("MRV2 Eagle3 Draft Window does not support DCP")
        if self.model_state.attn_metadata is not target_metadata:
            raise RuntimeError("Target metadata owner changed before Eagle3 Draft metadata build")

        input_batch = self.input_batch
        assert input_batch is not None
        is_prefilling = torch.from_numpy(input_batch.is_prefilling_np)
        with build_attn_metadata_factory(
            self.input_buffers.positions,
            num_tokens_padded,
            is_prefilling,
            parallel_config=self.draft_vllm_config.parallel_config,
        ):
            metadata = build_draft_window_attn_metadata(
                self,
                num_reqs=num_reqs,
                num_reqs_padded=num_reqs_padded,
                num_tokens_padded=num_tokens_padded,
                read_inputs=read_inputs,
                query_width=1,
                causal=True,
                query_start_loc_np=query_start_loc_np,
                dcp_local_seq_lens=None,
            )
        if not isinstance(metadata, dict) or not metadata:
            raise RuntimeError("eligible Eagle3 Window route produced no Draft metadata")
        if self.model_state.attn_metadata is not target_metadata:
            raise RuntimeError("Target metadata owner changed during Eagle3 Draft metadata build")

        # Isolate before assigning per-layer state: a buggy/reusing builder
        # must not let the state write below mutate a Target-owned object.
        metadata = copy_draft_attention_metadata(metadata)
        for layer_name, layer_metadata in metadata.items():
            if layer_metadata is None:
                continue
            if first_pass:
                target_layer_metadata = target_metadata.get(layer_name)
                if target_layer_metadata is None or not hasattr(target_layer_metadata, "attn_state"):
                    raise RuntimeError(f"Target metadata has no first-pass state for Draft layer {layer_name!r}")
                layer_metadata.attn_state = target_layer_metadata.attn_state
            else:
                layer_metadata.attn_state = AscendAttentionState.DecodeOnly

        if not publish:
            return metadata
        inputs.publish(metadata, target_metadata=target_metadata)
        return metadata

    def _build_eagle3_graph_metadata(self, *, workspace: DraftWindowWorkspace, **kwargs: Any) -> dict[str, Any]:
        view = draft_window_read_inputs(
            workspace,
            actual_request_count=kwargs["num_reqs"],
            padded_request_count=kwargs["num_reqs_padded"],
            block_tables=self.block_tables.input_block_tables,
            global_group_id=kwargs["inputs"]._owner.source.group.gid,
        )
        metadata = self._build_eagle3_private_attn_metadata(**kwargs, read_inputs_override=view, publish=False)
        # The normal nonparallel FIA builder deliberately prefers its CPU
        # seq_lens. FULL Graph capture owns a separate device operand. Bind
        # both exact owners explicitly here, leaving eager assembly unchanged.
        for layer in metadata.values():
            if layer is not None:
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
