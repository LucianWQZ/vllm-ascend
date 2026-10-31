#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from vllm.config.compilation import CUDAGraphMode
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import tl_rand32
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator

from vllm_ascend.worker.v2.spec_decode.dflash.speculator import (
    AscendDFlashSpeculator,
)


@triton.jit
def _selector_walk_kernel_ascend(
    scores_ptr,
    candidate_ptr,
    sample_pos_ptr,
    req_state_ptr,
    temperature_ptr,
    seeds_ptr,
    tokens_ptr,
    realized_scores_ptr,
    num_steps: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SAMPLE_PROBABILISTIC: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    """Ascend variant of upstream ``_selector_walk_kernel``.

    triton-ascend cannot lower ``tldevice.log1p`` (AST parse fails even for
    the greedy path), so the Gumbel noise uses the algebraically equivalent
    ``log(1 - u)`` transform. The signature mirrors upstream exactly so the
    inherited ``DFlash2Speculator._sample_path`` can call it unmodified:
    ``SAMPLE_PROBABILISTIC`` gates greedy vs. probabilistic, and ``USE_FP64``
    is rejected via ``tl.static_assert`` (NPU Triton has no fp64
    philox/rand path), matching the other vllm_ascend gumbel kernels.
    """
    tl.static_assert(not USE_FP64, "fp64 gumbel is not supported on NPU")
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < top_k
    req_state = tl.load(req_state_ptr + row * num_steps)
    valid = req_state >= 0
    temperature = tl.load(temperature_ptr + req_state, mask=valid, other=0.0)
    seed = tl.load(seeds_ptr + req_state, mask=valid, other=0)
    # Match upstream: a non-probabilistic draft is always greedy, regardless
    # of the request temperature
    effective_temp = temperature if SAMPLE_PROBABILISTIC else 0.0
    previous = 0
    for step in range(num_steps):
        flat = row * num_steps + step
        score_base = (flat * top_k + previous) * top_k
        scores = tl.load(
            scores_ptr + score_base + offsets,
            mask=mask & valid,
            other=float("-inf"),
        ).to(tl.float32)
        candidate_base = flat * top_k
        candidates = tl.load(
            candidate_ptr + candidate_base + offsets,
            mask=mask & valid,
            other=0,
        )

        if effective_temp == 0.0:
            best = tl.max(scores, axis=0)
            index = tl.min(tl.where(scores == best, offsets, BLOCK_K), axis=0)
        else:
            position = tl.load(sample_pos_ptr + flat) - 1
            # triton-ascend's philox requires int32 seed/offset operands
            # (int64 counters lower to 64-bit multiplies whose runtime
            # helper, __multi3, is unavailable); seeds, token ids and
            # positions all fit int32.
            gumbel_seed = tl.randint(seed.to(tl.int32), position.to(tl.int32))
            uniform = tl_rand32(gumbel_seed, candidates.to(tl.int32), includes_zero=False)
            noise = -tl.log(-tl.log(1.0 - uniform))
            sampled_scores = tl.where(mask, scores / effective_temp + noise, -float("inf"))
            best = tl.max(sampled_scores, axis=0)
            index = tl.min(tl.where(sampled_scores == best, offsets, BLOCK_K), axis=0)

        tl.store(
            realized_scores_ptr + candidate_base + offsets,
            scores,
            mask=mask & valid,
        )
        token = tl.load(candidate_ptr + candidate_base + index, mask=valid, other=0)
        tl.store(tokens_ptr + flat, token, mask=valid)
        previous = index


class AscendDFlash2Speculator(DFlash2Speculator, AscendDFlashSpeculator):
    """DFlash2 speculative drafter on Ascend NPUs.

    Every Ascend-specific behavior is inherited from AscendDFlashSpeculator:
    the NPU attention-metadata building (build_draft_attn_metadatas), the
    aclgraph manager wiring (init_cudagraph_manager), KV-cache/slot bookkeeping
    (set_attn), and the propose wrapper. Placing DFlash2Speculator first in
    the MRO makes its ``__init__`` (selector buffers and the optional persistent
    full-vocab proposal logits) and its ``_generate_draft`` override win: the
    candidate top-k, selector scoring, sequential path walk, and draft-logits
    scatter run instead of plain DFlash drafting -- including inside the
    aclgraph capture, which captures ``self._generate_draft``.
    """

    # _generate_draft -> get_top_k_tokens samples on the vocab-sharded head
    # outside sample_draft; reject lmhead TP at construction (as DSpark).
    _lmhead_tp_sample_draft_supported = False

    def load_draft_model(self, target_model, target_attn_layer_names):
        model = super().load_draft_model(target_model, target_attn_layer_names)
        from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
        from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl

        for layer, causal in zip(model.model.layers, model.get_draft_attn_causal()):
            impl = layer.self_attn.attn.impl
            if causal or impl.sliding_window is None:
                continue
            if type(impl) is not AscendAttentionBackendImpl or impl.sinks is not None or impl.use_bnsd_kv_cache:
                raise NotImplementedError("DFlash2 reference needs ordinary noncausal native GQA/SWA")
            # A single cached compressed mask/device. It is interpreted with
            # symmetric pre/next distances, not as a causal default mask.
            impl._dflash2_native_swa_mask = AttentionMaskBuilder(self.device).get_splitfuse_attn_mask()
        self._dflash2_full_kv_reference = True
        return model

    def _initialize_draft_window_source_capability(self) -> None:
        from vllm_ascend.worker.v2.spec_decode.draft_window_dflash2_graph import validate_dflash2_graph_source

        # set_attn has already published the actual groups/model/native cache
        # spec. Install this contract before the shared Source inventory runs.
        self._dflash2_graph_contract = validate_dflash2_graph_source(self)
        super()._initialize_draft_window_source_capability()

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        if self.speculative_config.enforce_eager:
            # Preserve the existing typed eager path and its independent owner.
            super().init_cudagraph_manager(CUDAGraphMode.NONE)
            return
        from vllm_ascend import envs
        from vllm_ascend.worker.v2.spec_decode.draft_window_dflash2_graph import validate_dflash2_graph_config

        # Window OFF bypasses the Window factory. It must retain the same
        # explicit experimental gate before any DFlash2 Graph is captured.
        if not envs.VLLM_ASCEND_EXPERIMENTAL_DRAFT_WINDOW_GRAPH:
            raise ValueError("DFlash2 Graph requires the experimental Draft Graph gate, including Window OFF")
        if cudagraph_mode not in (CUDAGraphMode.FULL, CUDAGraphMode.FULL_DECODE_ONLY):
            raise NotImplementedError(
                "DFlash2 Graph first admits FULL/FULL_DECODE_ONLY only; no implicit eager fallback"
            )
        validate_dflash2_graph_config(self.vllm_config)
        if cudagraph_mode.decode_mode() is CUDAGraphMode.FULL:
            import vllm.v1.worker.gpu.spec_decode.dflash2.speculator as upstream_dflash2

            from vllm_ascend.worker.v2.spec_decode.draft_window_dflash2_graph import validate_dflash2_graph_source

            # Do not capture the unsupported CUDA selector accidentally. The
            # installed Ascend patch is already used by ordinary DFlash2 eager.
            if upstream_dflash2._selector_walk_kernel is not _selector_walk_kernel_ascend:
                raise RuntimeError("DFlash2 Graph requires the actual Ascend selector-walk kernel binding")
            contract = getattr(self, "_dflash2_graph_contract", None)
            if contract is None:
                self._dflash2_graph_contract = validate_dflash2_graph_source(self)
            else:
                contract.validate_owner(self)
        super().init_cudagraph_manager(cudagraph_mode)
        if (
            cudagraph_mode.decode_mode() is CUDAGraphMode.FULL
            and self.query_cudagraph_manager.cudagraph_mode.decode_mode() is not CUDAGraphMode.FULL
        ):
            raise NotImplementedError("DFlash2 Graph backend did not admit FULL capture; no eager fallback is selected")

    def capture(self) -> None:
        contract = getattr(self, "_dflash2_graph_contract", None)
        if contract is not None:
            contract.validate_owner(self)
        # The parent captures prepare_graph_forward and the virtual DFlash2
        # _generate_draft, including its NPU conv/TopK/selector/cache kernels.
        # Python request control, prepare inputs and metadata stay outside it.
        super().capture()
        if contract is not None:
            contract.validate_owner(self)
