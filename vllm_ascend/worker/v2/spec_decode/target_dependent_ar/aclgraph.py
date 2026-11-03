# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
from collections.abc import Callable
from copy import copy
from typing import Any

import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.logger import logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
    prepare_inputs_to_capture,
)
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.cudagraph_utils import (
    SpeculatorCudaGraphManager,
)
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.compilation.acl_graph import (
    set_draft_graph_params,
    set_draft_graph_prefill_params,
    update_full_graph_params,
)
from vllm_ascend.compilation.updatable_graph import (
    SharedSource,
    UpdatableGraph,
)
from vllm_ascend.utils import use_updatable_graph
from vllm_ascend.worker.v2.aclgraph_utils import (
    collect_sorted_captured_token_sizes,
    model_capture_wrapper,
)
from vllm_ascend.worker.v2.utils import communicator_switch


def _eagle_ar_capture_config(vllm_config: VllmConfig, cudagraph_mode: CUDAGraphMode, decode_query_len: int):
    """Keep Target SP alignment out of the admitted dense Eagle AR buckets.

    Ascend filters the shared Target capture sizes to multiples of TP. K3
    first passes use Q4, so their valid request buckets survive that filter;
    the same sizes cannot represent Q1 AR for those requests. Only this
    explicitly enabled TP4 Eagle Graph route gets an independent descriptor.
    Target/first-pass configs, topology and computation remain unchanged.
    """
    speculative = vllm_config.speculative_config
    if (
        decode_query_len != 1
        or speculative is None
        or not isinstance(cudagraph_mode, CUDAGraphMode)
        or cudagraph_mode.decode_mode() != CUDAGraphMode.FULL
        or speculative.method != "eagle3"
        or speculative.enforce_eager
        or speculative.num_speculative_tokens != 3
        or speculative.uses_dynamic_speculative_decoding()
        or any(
            getattr(vllm_config.parallel_config, name) != 1
            for name in (
                "pipeline_parallel_size",
                "prefill_context_parallel_size",
                "decode_context_parallel_size",
            )
        )
    ):
        return vllm_config
    from vllm_ascend import envs
    from vllm_ascend.ascend_config import _eagle3_tp4_graph_review

    if not envs.VLLM_ASCEND_EXPERIMENTAL_DRAFT_WINDOW_GRAPH or not _eagle3_tp4_graph_review(vllm_config):
        return vllm_config
    # The qualified startup contract already checks every admitted first Q4
    # bucket, its capacity and the independent non-SP Draft. Reuse those same
    # request counts for Q1 AR; never renormalize them through Target SP.
    compilation = vllm_config.compilation_config
    requests = list(range(1, vllm_config.scheduler_config.max_num_seqs + 1))
    draft_config = copy(vllm_config)
    draft_config.compilation_config = copy(compilation)
    draft_config.compilation_config.cudagraph_capture_sizes = requests
    draft_config.compilation_config.max_cudagraph_capture_size = max(requests)
    return draft_config


class AutoRegressiveAclGraphManager(SpeculatorCudaGraphManager):
    """ACL graph manager for autoregressive speculative decoding."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        decode_query_len: int,
        lora_capture_cases: list[int] | None = None,
    ):
        vllm_config = _eagle_ar_capture_config(vllm_config, cudagraph_mode, decode_query_len)
        super().__init__(
            vllm_config,
            device,
            cudagraph_mode,
            decode_query_len,
            lora_capture_cases=lora_capture_cases,
        )

        # Upstream constructs graph managers without a speculator reference.
        # AscendTargetDependentARSpeculator attaches it after construction so replay
        # can rebuild draft metadata and update graph parameters.
        self.speculator: Any = None
        # The attention backend keys its per-size graph params by the actual
        # captured token counts (rounded up to decode_query_len when using
        # speculative decoding), so derive them from the capture descriptors
        # instead of the raw config sizes.
        self.capture_sizes = collect_sorted_captured_token_sizes(self._capture_descs)
        # Upstream uses num_speculative_steps + 1 as the draft-prefill query
        # length and 1 for draft decode.
        self.is_draft_model_prefill = decode_query_len > 1
        if super().needs_capture():
            if self.is_draft_model_prefill:
                set_draft_graph_prefill_params(self.capture_sizes)
            else:
                set_draft_graph_params(self.capture_sizes)

    def capture(
        self,
        forward_fn: Callable,
        model_state: ModelState,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        """Capture ACL graphs for autoregressive speculative decoding."""

        session = getattr(self.speculator, "_eagle3_graph_session", None)
        if session is not None:
            actual_forward = forward_fn

            def forward_fn(*args, **kwargs):
                if self.is_draft_model_prefill:
                    n, tokens = args[0], args[1]
                else:
                    desc = args[2]
                    n = desc.num_reqs or min(desc.num_tokens, self.max_num_reqs)
                    tokens = desc.num_tokens
                with session.capture_call(first=self.is_draft_model_prefill, n=n, tokens=tokens):
                    return actual_forward(*args, **kwargs)

        with communicator_switch(), model_capture_wrapper(self.speculator, self.is_draft_model_prefill):
            if self.is_draft_model_prefill:
                super().capture(
                    forward_fn,
                    model_state,
                    input_buffers,
                    block_tables,
                    attn_groups,
                    kv_cache_config,
                    progress_bar_desc=progress_bar_desc,
                )
                if session is not None:
                    session.provider.validate_captured_graphs(self.graphs)
                return

            def create_forward_fn(desc: BatchExecutionDescriptor, warmup: bool):
                num_tokens = desc.num_tokens
                num_reqs = desc.num_reqs or min(num_tokens, self.max_num_reqs)
                num_tokens_across_dp = (
                    torch.full((self.dp_size,), num_tokens, dtype=torch.int32, device="cpu")
                    if self.dp_size > 1
                    else None
                )
                prepare_inputs_to_capture(
                    num_reqs,
                    num_tokens,
                    model_state,
                    input_buffers,
                    block_tables,
                    attn_groups,
                    kv_cache_config,
                    full_cudagraph=(desc.cg_mode == CUDAGraphMode.FULL),
                    # vLLM #58275 makes mixed FULL graphs capture prefill kernels
                    # by bounding the dummy query length to the capture
                    # descriptor.
                    max_query_len=desc.max_query_len or desc.uniform_token_count,
                )
                seq_lens_cpu_upper_bound = input_buffers.seq_lens_cpu[:num_reqs]
                return lambda cg_mode: forward_fn(
                    num_reqs,
                    cg_mode == CUDAGraphMode.PIECEWISE,
                    BatchExecutionDescriptor(cg_mode=cg_mode, num_tokens=num_tokens, num_reqs=num_reqs),
                    num_tokens_across_dp,
                    seq_lens_cpu_upper_bound,
                    # vLLM #60335 added `num_speculative_steps` on main.
                    *(self.speculator.num_speculative_steps,),
                )

            CudaGraphManager.capture(self, create_forward_fn, progress_bar_desc=progress_bar_desc)
            if session is not None:
                session.provider.validate_captured_graphs(self.graphs)

    def run_fullgraph(self, desc: BatchExecutionDescriptor) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Replay the draft ACL graph and update its attention parameters."""
        session = getattr(self.speculator, "_eagle3_graph_session", None)
        if session is not None and self.speculator._eagle3_window_is_active():
            return self._run_eagle_window_fullgraph(desc, session)
        # Without a Window session the ordinary graph still needs the finite
        # admitted consumer shape. Target input padding alone does not
        # constrain the AR graph bucket selected by the fixed core dispatcher.
        # Source-created dummy/profile batches retain their original behavior.
        if getattr(getattr(self.speculator, "input_batch", None), "is_dummy", False) is not True:
            from vllm_ascend.worker.v2.spec_decode.draft_graph_review_limits import (
                validate_eagle_single_request_graph_descriptor,
            )

            validate_eagle_single_request_graph_descriptor(
                self.speculator, first=self.is_draft_model_prefill, n=desc.num_reqs, tokens=desc.num_tokens
            )
        num_tokens = desc.num_tokens
        logger.info_once(
            "ACL graph replay is active for the V2 draft model (%s, logged once per phase).",
            "prefill" if self.is_draft_model_prefill else "decode",
        )
        assert self.update_stream is not None

        attn_backend = self.speculator.attn_backend
        draft_vllm_config = self.speculator.draft_vllm_config
        draft_attn_metadatas = self.speculator.build_draft_attn_metadatas(
            desc.num_reqs,
            desc.num_tokens,
            self.is_draft_model_prefill,
        )
        if use_updatable_graph(attn_backend):
            return self._updatable_graph_replay(desc, draft_attn_metadatas)
        else:
            # This will be removed once the refactoring is fully complete.
            return self._graph_replay(desc, attn_backend, num_tokens, draft_vllm_config, draft_attn_metadatas)

    def _graph_replay(self, desc, attn_backend, num_tokens, draft_vllm_config, draft_attn_metadatas):
        self.update_stream.wait_stream(torch.npu.current_stream())
        ret = super().run_fullgraph(desc)
        # Mirror vLLM's DP graph-replay token-count metadata.
        num_tokens_across_dp = torch.full([self.speculator.dp_size], num_tokens)
        attn_metadata = self.speculator.model_state.attn_metadata
        # sfa_v1.py:AscendSFABackend.get_impl_cls reaches
        # sfa_cp.py:resolve_sfa_impl, whose SFA CP selector reads the current
        # ModelConfig. Publish the draft config because set_forward_context()
        # does not update it.
        # TODO: Remove this explicit current-config scope once ACL graph replay
        # passes VllmConfig directly through the graph-update interfaces.
        with (
            set_current_vllm_config(draft_vllm_config),
            set_forward_context(
                attn_metadata,
                draft_vllm_config,
                num_tokens=num_tokens,
                cudagraph_runtime_mode=desc.cg_mode,
                num_tokens_across_dp=num_tokens_across_dp,
                batch_descriptor=None,  # Full graph model don't need batch_descriptor
                slot_mapping=None,
            ),
        ):
            # Select the draft prefill/decode graph-parameter pool.
            _EXTRA_CTX.is_draft_model = True
            _EXTRA_CTX.is_draft_model_prefill = self.is_draft_model_prefill

            forward_context = get_forward_context()
            assert attn_backend is not None, "Speculator attention backend is not initialized"
            update_full_graph_params(
                # FIXME(Ronald1995): support hybrid attn backend
                attn_backend,
                self.update_stream,
                forward_context,
                num_tokens,
                draft_vllm_config,
                self.speculator.speculative_config,
                draft_attn_metadatas=draft_attn_metadatas,
            )
        return ret

    def _updatable_graph_replay(self, desc, draft_attn_metadatas, *, window_source=None):
        from vllm_ascend.worker.v2.spec_decode.draft_window_eagle3_dp import validate_eagle_dp_dummy_replay

        dp_dummy = validate_eagle_dp_dummy_replay(self.speculator, self, desc, draft_attn_metadatas)
        graph = self.graphs[desc]
        assert isinstance(graph, UpdatableGraph)
        if window_source is None:
            fia_params = self.speculator.build_fia_params(
                desc.num_reqs,
                draft_attn_metadatas[0],
                self.is_draft_model_prefill,
            )
            window_source = SharedSource(fia_params)
        resolved_tasks = graph.resolve_tasks(window_source)
        self.update_stream.wait_stream(torch.npu.current_stream())
        ret = super().run_fullgraph(desc)
        if dp_dummy:
            # Keep the fixed Full/PAD dummy operands and original UPDATE/event
            # order; no Window generation or business bank is staged here.
            with set_current_vllm_config(self.vllm_config):
                graph.update(self.update_stream, resolved_tasks)
        else:
            graph.update(self.update_stream, resolved_tasks)
        return ret

    def dispatch(self, num_reqs, num_tokens, uniform_token_count, num_active_loras, max_query_len=None, num_ubatches=1):
        owner = self.speculator
        if getattr(owner, "_eagle3_graph_session", None) is not None and getattr(
            owner, "_eagle3_window_continuing_prefill", False
        ):
            # Only the explicit existing nonuniform/continuing-prefill phase.
            # Missing buckets and UPDATE exceptions never enter this branch.
            return BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE,
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                num_active_loras=self._resolve_effective_loras(num_active_loras),
                num_ubatches=num_ubatches,
            )
        return super().dispatch(
            num_reqs, num_tokens, uniform_token_count, num_active_loras, max_query_len, num_ubatches
        )

    def _run_eagle_window_fullgraph(self, desc, session):
        if self.is_draft_model_prefill:
            capture = session.resolve(first=True, n=desc.num_reqs, tokens=desc.num_tokens, ordinal=0)
            session.stage(capture, 0)
        ret = self._updatable_graph_replay(desc, (), window_source=session.provider)
        session.provider.finish_update()
        return ret
