from collections.abc import Callable, Mapping
from typing import Any

import torch
from vllm.config import VllmConfig
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
from vllm_ascend.compilation.acl_graph import (
    set_draft_graph_params,
    update_full_graph_params,
)
from vllm_ascend.worker.v2.aclgraph_utils import collect_sorted_captured_token_sizes, model_capture_wrapper
from vllm_ascend.worker.v2.spec_decode.dflash.capture import (
    draft_window_dflash_capture_scope,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_graph import (
    DraftWindowGraphStage,
    assert_draft_window_graph_metadata_batch,
    build_draft_window_graph_update_binding,
    observe_draft_window_graph_capture,
    observe_draft_window_graph_source,
    resolve_parallel_draft_window_graph_route,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_graph_characterization import (
    prepare_parallel_draft_window_graph_characterization,
    record_parallel_graph_output,
    record_parallel_graph_replay_submission,
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

    def capture(
        self,
        forward_fn: Callable,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        causal: bool | Mapping[int, bool] = False,
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        """Capture ACL graphs for DFlash."""
        with draft_window_dflash_capture_scope(
            self.speculator,
            input_buffers,
            getattr(self, "_capture_descs", {}),
            max_num_reqs=getattr(self, "max_num_reqs", 0),
        ) as capture_context:
            observe_draft_window_graph_capture(
                self.speculator,
                capture_context=capture_context,
            )
            with communicator_switch(), model_capture_wrapper(self.speculator, False):
                super().capture(
                    forward_fn,
                    input_buffers,
                    block_tables,
                    attn_groups,
                    kv_cache_config,
                    max_model_len,
                    causal,
                    progress_bar_desc,
                )

    def run_fullgraph(self, desc: BatchExecutionDescriptor) -> None:
        """Submit replay and UPDATE; propose reads persistent draft_tokens later."""
        num_tokens = desc.num_tokens

        prepare_parallel_draft_window_graph_characterization(
            self.speculator,
            desc,
        )
        draft_attn_metadatas = self.speculator.build_draft_attn_metadatas(
            desc.num_reqs,
            self.speculator.input_batch.seq_lens_cpu_upper_bound,
        )
        graph_route = resolve_parallel_draft_window_graph_route(self.speculator)
        graph_mode_name = getattr(desc.cg_mode, "name", None)
        if graph_route is not None:
            if not isinstance(graph_mode_name, str):
                raise RuntimeError("Draft Window graph carrier requires an explicit graph mode")
            assert_draft_window_graph_metadata_batch(
                self.speculator,
                route=graph_route,
                draft_attn_metadatas=draft_attn_metadatas,
                num_reqs=desc.num_reqs,
                num_tokens=desc.num_tokens,
                graph_mode=graph_mode_name,
            )
        self.update_stream.wait_stream(torch.npu.current_stream())
        observe_draft_window_graph_source(
            self.speculator,
            DraftWindowGraphStage.REPLAY,
        )
        ret = super().run_fullgraph(desc)
        record_parallel_graph_replay_submission(self.speculator)
        observe_draft_window_graph_source(
            self.speculator,
            DraftWindowGraphStage.CONSUMER,
        )

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
            graph_update_binding = (
                None
                if graph_route is None
                else build_draft_window_graph_update_binding(
                    self.speculator,
                    route=graph_route,
                    draft_attn_metadatas=draft_attn_metadatas,
                )
            )
            self._update_full_graph_params(
                # FIXME(Ronald1995): support hybrid attn backend
                list(self.speculator.attn_backends.values())[0],
                self.update_stream,
                forward_context,
                num_tokens,
                self.vllm_config,
                self.speculator.speculative_config,
                draft_attn_metadatas=draft_attn_metadatas,
                draft_window_graph_update_binding=graph_update_binding,
            )
        # Evidence-only D2H must not block submission of the backend UPDATE
        # that records the ExternalEvent awaited by the captured FIA task.
        # This does not reorder replay/consumer/update or prove its device timing.
        record_parallel_graph_output(self.speculator, ret)
        return ret

    def _update_full_graph_params(self, *args, **kwargs):
        """Instance boundary for opt-in Host ranges; exact original UPDATE call."""
        return update_full_graph_params(*args, **kwargs)
