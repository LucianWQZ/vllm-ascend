# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Mapping
from contextlib import contextmanager, nullcontext
from typing import Any, cast

import numpy as np
import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash_speculator
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

from vllm_ascend.ops.triton.v2.spec_decode.prepare_dflash_inputs import prepare_dflash_inputs_triton
from vllm_ascend.worker.v2.attn_utils import build_attn_metadata_factory, build_attn_metadata_wrapper
from vllm_ascend.worker.v2.spec_decode.draft_window_eager import DraftWindowGraphInputMixin
from vllm_ascend.worker.v2.spec_decode.draft_window_graph import (
    DraftWindowGraphRoute,
    consume_draft_window_graph_metadata,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_graph_runtime import (
    draft_window_graph_enabled,
    install_parallel_graph,
)
from vllm_ascend.worker.v2.spec_decode.lmhead_tp_utils import LmheadTPDraftSamplingMixin
from vllm_ascend.worker.v2.spec_decode.pcp_utils import (
    disable_profiling_chunk_for_draft,
)


def prepare_dflash_inputs_factory(
    kv_cache_block_size: int,
    block_tables: Any | None = None,
) -> Callable[..., None]:
    # vLLM main (#57169) dropped ``kernel_block_sizes`` from ``BlockTables``
    # and the DFlash caller now forwards ``BlockTables.block_sizes`` (physical
    # KV blocks) as ``block_size``, while ``AscendBlockTables`` keeps the block
    # table kernel-block addressed (a physical block is re-expanded into
    # ``blocks_per_kv_block`` kernel rows) and the Ascend prep indexes it at
    # kernel granularity. DCP ownership still uses the physical size, so only
    # ``block_size`` is restored, per group.
    kernel_block_sizes = getattr(block_tables, "kernel_block_sizes", None)
    input_block_tables = getattr(block_tables, "input_block_tables", None)
    if block_tables is not None and (kernel_block_sizes is None or input_block_tables is None):
        raise RuntimeError(
            "DFlash on vLLM main needs AscendBlockTables.kernel_block_sizes and "
            "input_block_tables to restore the kernel block size"
        )
    # The upstream caller passes ``input_block_tables[gid]`` verbatim, so the
    # per-group kernel block size is looked up by tensor identity. Callers
    # that do not supply Ascend block tables keep the physical size forwarded
    # by the DFlash caller.
    kernel_block_size_by_table = (
        {id(table): int(size) for table, size in zip(input_block_tables, kernel_block_sizes)}
        if kernel_block_sizes is not None and input_block_tables is not None
        else {}
    )

    def prepare_with_block_size(
        input_buffers: InputBuffers,
        query_slot_mapping: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor,
        sample_indices: torch.Tensor,
        sample_pos: torch.Tensor,
        sample_idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        input_batch: InputBatch,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        input_temperature: torch.Tensor,
        input_seeds: torch.Tensor,
        block_table: torch.Tensor,
        block_size: int,
        cp_rank: int,
        cp_size: int,
        cp_interleave: int,
        parallel_drafting_token_id: int,
        num_query_per_req: int,
        num_speculative_steps: int,
        max_num_reqs: int,
        max_num_tokens: int,
        max_model_len: int,
        sample_from_anchor: bool = False,
        num_reqs_override: int | None = None,
    ) -> None:
        kernel_block_size = kernel_block_size_by_table.get(id(block_table))
        if kernel_block_size is None:
            if kernel_block_size_by_table:
                raise RuntimeError("DFlash received a block table unknown to AscendBlockTables")
            # No Ascend block tables were supplied; keep the caller's size.
            kernel_block_size = block_size
        prepare_dflash_inputs(
            input_buffers,
            query_slot_mapping,
            context_positions,
            context_slot_mapping,
            sample_indices,
            sample_pos,
            sample_idx_mapping,
            temperature,
            seeds,
            input_batch,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            input_temperature,
            input_seeds,
            block_table,
            kernel_block_size,
            cp_rank,
            cp_size,
            cp_interleave,
            parallel_drafting_token_id,
            num_query_per_req,
            num_speculative_steps,
            max_num_reqs,
            max_num_tokens,
            max_model_len,
            sample_from_anchor,
            num_reqs_override=num_reqs_override,
            kv_cache_block_size=kv_cache_block_size,
        )

    return prepare_with_block_size


@contextmanager
def _draft_window_context_kv_scope(owner, input_batch):
    """Protect the upstream zero-draft write without creating a Graph reader.

    The fixed core stores context K/V on the current stream and returns before
    query metadata or Graph dispatch. Reuse the ordinary Source completion
    contract for these writes; captured Window banks and generations stay idle.
    """
    runtime = owner._draft_window_graph_binding_provider
    if runtime.quarantined_identity is not None or getattr(owner, "_draft_window_graph_submission_uncertain", False):
        raise RuntimeError("Draft Window backing is quarantined before a context-only KV write")
    from vllm_ascend.worker.v2.spec_decode.draft_window_lifetime import DraftWindowOwnerPool
    from vllm_ascend.worker.v2.spec_decode.draft_window_runtime import _NPUCompletion

    if (
        getattr(owner, "_draft_window_proposal_context", None) is not None
        or getattr(owner, "_draft_window_bypass_active", False)
        or getattr(owner, "_draft_window_context_kv_active", False)
    ):
        raise RuntimeError("A Draft Window proposal scope is already active")
    pool = getattr(owner, "_draft_window_context_kv_pool", None)
    if pool is None:
        pool = DraftWindowOwnerPool(int(owner.vllm_config.max_concurrent_batches))
        owner._draft_window_context_kv_pool = pool
    owner._draft_window_context_kv_active = True
    try:
        with pool.operation(_NPUCompletion) as completion_owner:
            pool.retain(completion_owner, owner, input_batch)
            pool.begin_native(completion_owner)
            yield
    except BaseException:
        # A partial context write or failed completion record also makes the
        # canonical state unsafe for a later normal Graph proposal.
        owner._draft_window_graph_submission_uncertain = True
        raise
    finally:
        owner._draft_window_context_kv_active = False


def _draft_window_owner_scope(owner, input_batch, dummy_run=False, is_profile=False, *, num_speculative_tokens=None):
    """Choose exactly one existing proposal owner at the Ascend boundary."""
    dflash2_contract = getattr(owner, "_dflash2_graph_contract", None)
    if dflash2_contract is not None:
        # These captured sampler/selector addresses are separate from the
        # Window bank. Check their Host descriptors before replacing state;
        # no tensor value read or device synchronization is introduced here.
        dflash2_contract.validate_owner(owner)
    runtime = getattr(owner, "_draft_window_runtime", None)
    graph_runtime = getattr(owner, "_draft_window_graph_binding_provider", None)
    limit = getattr(owner, "_draft_graph_review_limit", None)
    context_only_full8 = (
        limit is not None
        and limit.method == "dflash"
        and type(num_speculative_tokens) is int
        and num_speculative_tokens == 0
    )
    if limit is not None:
        from vllm_ascend.worker.v2.spec_decode.draft_graph_review_limits import validate_draft_graph_review_requests

        # Original request limits apply before either owner is changed.
        # Core544's int0 branch writes only context K/V, so its Target padding
        # is not a Draft query shape. The real request identity is still checked.
        validate_draft_graph_review_requests(
            owner,
            input_batch,
            dummy_run=dummy_run,
            is_profile=is_profile,
            context_kv_only=context_only_full8,
        )
    if runtime is not None and graph_runtime is not None:
        raise RuntimeError("Draft Window eager and Graph owners cannot share one proposal")
    if graph_runtime is not None:
        if getattr(owner, "_draft_window_context_kv_active", False):
            raise RuntimeError("A context-only Draft KV write is already active")
        if num_speculative_tokens is not None and (
            type(num_speculative_tokens) is not int or num_speculative_tokens not in (0, owner.num_speculative_steps)
        ):
            raise RuntimeError("Draft Window Graph cannot change its captured static K")
        effective_dummy = dummy_run or getattr(input_batch, "is_dummy", False)
        if num_speculative_tokens == 0 and not (effective_dummy or is_profile):
            return _draft_window_context_kv_scope(owner, input_batch)
        return owner._draft_window_proposal_scope(input_batch, dummy_run=effective_dummy, is_profile=is_profile)
    if runtime is not None:
        return runtime.proposal_context(
            owner, input_batch, dummy_run, is_profile, context_kv_only=num_speculative_tokens == 0
        )
    return nullcontext()


class AscendDFlashSpeculator(DraftWindowGraphInputMixin, LmheadTPDraftSamplingMixin, DFlashSpeculator):
    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        with disable_profiling_chunk_for_draft(self.vllm_config):
            model = super().load_draft_model(target_model, target_attn_layer_names)
        # Upstream's MRV2 loader shares the target's vocab layers, but does
        # not align a DFlash head with a QuaRot residual stream. Reuse the
        # existing Ascend load-time alignment before any graph or KV setup.
        from vllm_ascend.models.qwen3_dflash import align_dflash_weights

        align_dflash_weights(model, self.vllm_config)
        return model

    def build_draft_attn_metadatas(self, num_reqs_padded, seq_lens_cpu_upper_bound):
        from vllm_ascend.worker.v2.spec_decode.draft_window_graph_metadata import is_parallel_graph_dp_dummy

        if hasattr(self, "_draft_window_graph_binding_provider") and not is_parallel_graph_dp_dummy(self):
            graph_metadata = consume_draft_window_graph_metadata(
                self, route=DraftWindowGraphRoute.DFLASH, expected_consumer_count=1
            )
            if graph_metadata is not None:
                return list(graph_metadata)
        num_tokens_padded = num_reqs_padded * self.num_query_per_req
        with build_attn_metadata_wrapper():
            # vLLM main (#56181) replaced _build_draft_attn_metadata with
            # _build_uniform_attn_metadata (BatchExecutionDescriptor).
            batch_desc = BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.FULL,
                num_tokens=num_tokens_padded,
                num_reqs=num_reqs_padded,
            )
            attn_metadata = self._build_uniform_attn_metadata(
                num_reqs=self.input_batch.num_reqs,
                batch_desc=batch_desc,
                num_query_per_req=self.num_query_per_req,
                seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                step=self.num_query_per_req,
                causal=self._group_causal,
            )
        self._update_draft_attn_metadata(attn_metadata, num_reqs_padded)
        return [attn_metadata]

    def _update_draft_attn_metadata(self, attn_metadata, num_reqs_padded):
        """Rebuild ``actual_seq_lengths_q`` from the padded request count,
        mirroring Eagle's ``_update_decode_attn_metadata``.

        Upstream ``Speculator._build_draft_attn_metadata`` clamps
        ``query_start_loc`` at the real ``num_reqs`` to keep the cumulative
        series non-decreasing, so when a batch is padded to a capture size
        (``num_reqs_padded > num_reqs``) the cumulative query lengths stop at
        ``num_reqs * num_query_per_req`` instead of ``num_tokens_padded``. The
        Ascend FIA operator requires, in TND layout, that the last element of
        ``actual_seq_lengths_q`` equals the query token count of the graph
        being replayed; otherwise tiling fails with
        ``queryT != last element of actualSequenceLengthQ``.
        """
        query_lens_list = [(i + 1) * self.num_query_per_req for i in range(num_reqs_padded)]
        for metadata in attn_metadata.values():
            metadata.actual_seq_lengths_q = query_lens_list

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        self._lmhead_tp_validate_draft_sampling()

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        if self.speculative_config.enforce_eager:
            cudagraph_mode = CUDAGraphMode.NONE
        super().init_cudagraph_manager(cudagraph_mode)
        # The Ascend graph manager is patched onto the upstream module and
        # created by super().init_cudagraph_manager without a speculator ref.
        # It needs this speculator to update full-graph params, so set it here.
        self.query_cudagraph_manager.speculator = self
        self.query_cudagraph_manager.update_stream = self.update_stream
        if hasattr(self, "_draft_window_graph_binding_provider"):
            if not draft_window_graph_enabled():
                raise RuntimeError("Draft Graph configuration changed during initialization")
            install_parallel_graph(self)

    def set_attn(
        self,
        model_state: Any,
        kv_cache_config: Any,
        block_tables: Any,
        target_input_buffers: Any,
        target_attn_groups: Any,
    ) -> None:
        super().set_attn(
            model_state,
            kv_cache_config,
            block_tables,
            target_input_buffers,
            target_attn_groups,
        )
        self._context_slot_mappings = torch.zeros(
            len(self.draft_kv_cache_group_ids),
            self.max_num_tokens,
            dtype=torch.int32,
            device=self.device,
        )
        # npu needs attn_backends to update full graph params in run_fullgraph.
        attn_backends: dict[str, type[AttentionBackend]] = {}
        active_layer_names = self.draft_attn_layer_names
        for kv_cache_group_spec in kv_cache_config.kv_cache_groups:
            layer_names = kv_cache_group_spec.layer_names
            if active_layer_names is not None:
                layer_names = list(active_layer_names.intersection(layer_names))

            layer_type = cast(type[Any], AttentionLayerBase)
            attn_layers = get_layers_from_vllm_config(self.vllm_config, layer_type, layer_names)

            for layer_name in layer_names:
                attn_backends[layer_name] = attn_layers[layer_name].get_attn_backend()

        self.attn_backends = attn_backends
        dflash_speculator.prepare_dflash_inputs = prepare_dflash_inputs_factory(  # type: ignore[call-arg]
            self.vllm_config.cache_config.block_size,
            block_tables,
        )

        from vllm_ascend.worker.v2.spec_decode.draft_window_native_attention import report_owner_draft_native_attention

        # Actual Draft layers/builders now exist. This startup-only inventory
        # creates no eager runtime and does not infer attention from KV storage.
        report_owner_draft_native_attention(self, logger)

        if hasattr(self, "_draft_window_policy"):
            if hasattr(self, "draft_window_policy"):
                raise RuntimeError("Draft Window cannot initialize both eager and Graph policies")
            self._initialize_draft_window_source_capability()
        elif hasattr(self, "draft_window_policy"):
            from vllm_ascend.worker.v2.spec_decode.draft_window_runtime import DraftWindowRuntime

            self._draft_window_runtime = DraftWindowRuntime(self, self.draft_window_policy)

    def _build_attn_metadata(
        self,
        num_reqs: int,
        batch_desc: BatchExecutionDescriptor,
        query_start_loc_np: np.ndarray,
        seq_lens_cpu_upper_bound: torch.Tensor,
        step: int,
        causal: bool | Mapping[int, bool] = True,
        dcp_local_seq_lens: torch.Tensor | None = None,
        slot_mappings: torch.Tensor | None = None,
    ) -> dict[str, Any] | None:
        runtime = getattr(self, "_draft_window_runtime", None)
        # Older vLLM releases do not accept this optional upstream argument.
        slot_kwargs = {} if slot_mappings is None else {"slot_mappings": slot_mappings}
        if runtime is None:
            return super()._build_attn_metadata(
                num_reqs,
                batch_desc,
                query_start_loc_np,
                seq_lens_cpu_upper_bound,
                step,
                causal,
                dcp_local_seq_lens,
                **slot_kwargs,
            )
        with build_attn_metadata_factory(
            self.input_buffers.positions,
            batch_desc.num_tokens,
            self.draft_is_prefilling,
            common_metadata_callback=runtime.metadata_callback(step, num_reqs),
            metadata_builder_callback=runtime.metadata_builder_callback(step),
        ):
            metadata = super()._build_attn_metadata(
                num_reqs,
                batch_desc,
                query_start_loc_np,
                seq_lens_cpu_upper_bound,
                step,
                causal,
                dcp_local_seq_lens,
                **slot_kwargs,
            )
        runtime.retain_metadata(metadata)
        return metadata

    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync_state: Any = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        num_speculative_tokens: int | None = None,
    ) -> torch.Tensor:
        # Check checkpoint admission/static K before replacing proposal state.
        owner_scope = _draft_window_owner_scope(
            self, input_batch, dummy_run, is_profile, num_speculative_tokens=num_speculative_tokens
        )
        self.input_batch = input_batch
        sync_state = dp_sync_state
        if dummy_run and skip_attn_for_dummy_run:
            # Profiling runs the draft with its own query token count, which
            # can differ from the target batch. Let forward_context coordinate
            # the actual draft counts instead of reusing the target DP state.
            # TODO: Remove this guard once main2main includes upstream vLLM
            # #54856 (facd9a74a1), which resets the profiling DP counts.
            sync_state = None
        with (
            owner_scope,
            build_attn_metadata_wrapper(),
        ):
            return super().propose(
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
                sync_state,
                dummy_run,
                skip_attn_for_dummy_run,
                mm_inputs,
                is_profile=is_profile,
                num_speculative_tokens=num_speculative_tokens,
            )


def prepare_dflash_inputs(
    input_buffers: InputBuffers,
    query_slot_mapping: torch.Tensor,
    context_positions: torch.Tensor,
    context_slot_mapping: torch.Tensor,
    sample_indices: torch.Tensor,
    sample_pos: torch.Tensor,
    sample_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    input_batch: InputBatch,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    last_sampled: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    input_temperature: torch.Tensor,
    input_seeds: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    cp_rank: int,
    cp_size: int,
    cp_interleave: int,
    parallel_drafting_token_id: int,
    num_query_per_req: int,
    num_speculative_steps: int,
    max_num_reqs: int,
    max_num_tokens: int,
    max_model_len: int,
    sample_from_anchor: bool = False,
    num_reqs_override: int | None = None,
    *,
    kv_cache_block_size: int,
) -> None:
    prepare_dflash_inputs_triton(
        input_buffers,
        query_slot_mapping,
        context_positions,
        context_slot_mapping,
        sample_indices,
        sample_pos,
        sample_idx_mapping,
        temperature,
        seeds,
        input_batch,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        input_temperature,
        input_seeds,
        block_table,
        block_size,
        cp_rank,
        cp_size,
        cp_interleave,
        parallel_drafting_token_id,
        num_query_per_req,
        num_speculative_steps,
        max_num_reqs,
        max_num_tokens,
        max_model_len,
        sample_from_anchor,
        num_reqs_override=num_reqs_override,
        kv_cache_block_size=kv_cache_block_size,
    )
