from collections.abc import Callable, Mapping
from typing import Any

import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import (  # type: ignore[import-not-found]
    BatchExecutionDescriptor,
)
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.fa_v1 import AscendFlashAttentionBackend
from vllm_ascend.compilation.acl_graph import (
    set_draft_graph_params,
    update_full_graph_params,
)
from vllm_ascend.compilation.updatable_graph import (
    ContextSource,
    UpdatableGraph,
)
from vllm_ascend.utils import use_updatable_graph
from vllm_ascend.worker.v2.aclgraph_utils import collect_sorted_captured_token_sizes, model_capture_wrapper
from vllm_ascend.worker.v2.spec_decode.dflash.capture import draft_window_dflash_capture_scope
from vllm_ascend.worker.v2.spec_decode.draft_window_graph import (
    resolve_parallel_draft_window_graph_route,
    validate_draft_window_graph_capture,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_graph_runtime import (
    prepare_parallel_graph,
)
from vllm_ascend.worker.v2.utils import communicator_switch


class DFlashAclGraphManager(DFlashCudaGraphManager):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        decode_query_len: int,
        speculator: Any = None,
    ):
        super().__init__(
            vllm_config,
            device,
            cudagraph_mode,
            decode_query_len,
        )

        # It is set by AscendDFlashSpeculator.init_cudagraph_manager after creation,
        # because upstream's init_cudagraph_manager creates the manager without it.
        self.speculator = speculator
        # The attention backend keys its per-size graph params by the actual
        # captured token counts (rounded up to decode_query_len when using
        # speculative decoding), so derive them from the capture descriptors
        # instead of the raw config sizes.
        self.capture_sizes = collect_sorted_captured_token_sizes(self._capture_descs)
        # DFlash's parallel drafting forward has its own dedicated draft graph
        # path, independent of Eagle's prefill/decode split, so it always uses
        # the default draft params bucket (is_draft_model_prefill stays False in
        # both capture and replay to keep them consistent).
        if super().needs_capture():
            set_draft_graph_params(self.capture_sizes)

    def dispatch(
        self,
        num_reqs: int,
        num_tokens: int,
        uniform_token_count: int | None,
        num_active_loras: int,
        max_query_len: int | None = None,
        num_ubatches: int = 1,
    ) -> BatchExecutionDescriptor:
        if getattr(self.speculator, "_draft_window_prefill_bypass_active", False):
            # Continuing prefill owns no Window generation. Captured Window
            # operands cannot stand in for its baseline full read. Select the
            # existing eager branch before upstream metadata/padding is built;
            # this is a declared phase rule, never a missing-bucket fallback.
            return BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE,
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                num_active_loras=self._resolve_effective_loras(num_active_loras),
                num_ubatches=num_ubatches,
            )
        descriptor = super().dispatch(
            num_reqs, num_tokens, uniform_token_count, num_active_loras, max_query_len, num_ubatches
        )
        from vllm_ascend.worker.v2.spec_decode.draft_window_graph_metadata import select_parallel_graph_metadata

        select_parallel_graph_metadata(self.speculator, descriptor)
        return descriptor

    def capture(
        self,
        forward_fn: Callable,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        causal: bool | Mapping[int, bool] = False,
        precompute_context_kv: Callable[[int], None] | None = None,
        prepare_forward: Callable[[int, dict[str, Any] | None], None] | None = None,
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        """Capture ACL graphs for DFlash.

        vLLM #60692 renamed ``precompute_context_kv(num_reqs)`` to
        ``prepare_forward(num_reqs, attn_metadata)``; accept both and forward the
        one the installed vLLM expects.
        """
        prepare_kwargs: dict[str, Any] = {"prepare_forward": prepare_forward}
        with draft_window_dflash_capture_scope(
            self.speculator,
            input_buffers,
            getattr(self, "_capture_descs", {}),
        ) as capture_context:
            validate_draft_window_graph_capture(self.speculator, capture_context=capture_context)
            # Keep core544's preparation callback and canonical context slots;
            # it receives both num_reqs and the actual built metadata.
            with communicator_switch(), model_capture_wrapper(self.speculator, False):
                super().capture(
                    forward_fn,
                    input_buffers,
                    block_tables,
                    attn_groups,
                    kv_cache_config,
                    max_model_len,
                    causal=causal,
                    progress_bar_desc=progress_bar_desc,
                    **prepare_kwargs,
                )

    def run_fullgraph(self, desc: BatchExecutionDescriptor) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Override run_fullgraph to update full graph params in run_fullgraph."""
        num_tokens = desc.num_tokens
        from vllm_ascend.worker.v2.spec_decode.draft_window_graph_metadata import is_parallel_graph_dp_dummy

        dp_dummy = is_parallel_graph_dp_dummy(self.speculator)
        graph_route = None if dp_dummy else resolve_parallel_draft_window_graph_route(self.speculator)
        attn_backend = list(self.speculator.attn_backends.values())[0]
        if graph_route is not None and not use_updatable_graph(attn_backend):
            raise NotImplementedError("Draft Window Graph requires the admitted updatable backend binding contract")
        if all(backend is AscendFlashAttentionBackend for backend in self.speculator.attn_backends.values()):
            # Length preparation and FlashAttn tiling run inside the graph,
            # reading the draft's persistent device buffers on every replay.
            return super().run_fullgraph(desc)
        if not dp_dummy:
            prepare_parallel_graph(self.speculator, desc)
        draft_attn_metadatas = self.speculator.build_draft_attn_metadatas(
            desc.num_reqs,
            self.speculator.input_batch.seq_lens_cpu_upper_bound,
        )
        if graph_route is not None:
            graph_mode_name = getattr(desc.cg_mode, "name", None)
            if not isinstance(graph_mode_name, str):
                raise RuntimeError("Draft Window Graph requires an explicit replay mode")
            self.speculator._draft_window_graph_binding_provider.assert_metadata_batch(
                route=graph_route,
                draft_attn_metadatas=draft_attn_metadatas,
                num_reqs=desc.num_reqs,
                num_tokens=num_tokens,
                graph_mode=graph_mode_name,
            )
        if use_updatable_graph(attn_backend):
            return self._updatable_graph_replay(desc, draft_attn_metadatas)
        else:
            # Ordinary OFF/eager Target policy keeps the original fallback.
            return self._graph_replay(desc, attn_backend, num_tokens, draft_attn_metadatas)

    def _graph_replay(self, desc, attn_backend, num_tokens, draft_attn_metadatas):
        self.update_stream.wait_stream(torch.npu.current_stream())
        ret = super().run_fullgraph(desc)
        # refer to vllm.v1.worker.gpu.dp_utils.sync_cudagraph_and_dp_padding to
        # calculate num_tokens_across_dp.
        num_tokens_across_dp = torch.full([self.speculator.dp_size], num_tokens)

        with set_forward_context(
            self.speculator.model_state.attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=desc.cg_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            batch_descriptor=None,  # Full graph model don't need batch_descriptor
            slot_mapping=None,
        ):
            _EXTRA_CTX.is_draft_model = True
            _EXTRA_CTX.is_draft_model_prefill = False
            forward_context = get_forward_context()
            update_full_graph_params(
                # FIXME(Ronald1995): support hybrid attn backend
                attn_backend,
                self.update_stream,
                forward_context,
                num_tokens,
                self.vllm_config,
                self.speculator.speculative_config,
                draft_attn_metadatas=draft_attn_metadatas,
            )
        return ret

    def _updatable_graph_replay(self, desc, draft_attn_metadatas):
        graph = self.graphs[desc]
        assert isinstance(graph, UpdatableGraph)
        resolved_tasks = graph.resolve_tasks(ContextSource(draft_attn_metadatas[0]))
        from vllm_ascend.worker.v2.spec_decode.draft_window_graph_metadata import is_parallel_graph_dp_dummy

        dp_dummy = is_parallel_graph_dp_dummy(self.speculator)
        graph_route = None if dp_dummy else resolve_parallel_draft_window_graph_route(self.speculator)
        self.update_stream.wait_stream(torch.npu.current_stream())
        ret = super().run_fullgraph(desc)
        if graph_route is None:
            # Exact existing OFF updatable call; no Window-only construction.
            graph.update(self.update_stream, resolved_tasks)
        else:
            # Backend dispatch resolves CP classes from the captured manager's
            # configuration, also when this path is entered outside a model forward.
            with set_current_vllm_config(self.vllm_config):
                self.speculator._draft_window_graph_binding_provider.validate_update(
                    graph, resolved_tasks, route=graph_route
                )
                graph.update(self.update_stream, resolved_tasks)
        return ret
