# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/spec_decode/__init__.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""MRV2 speculative-decoder factory for Ascend.

The serving lifecycle initializes the process-wide ``AscendConfig`` before
constructing ``NPUModelRunner``.  Direct callers of :func:`init_speculator`
must establish the same precondition.
"""

import torch
from vllm.config import VllmConfig
from vllm.logger import logger

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.spec_decode.draft_window import (
    DraftWindowPolicyDecision,
    DraftWindowPolicyState,
    resolve_draft_window_policy,
    resolve_static_future_offset,
    validate_eligible_window_numbers,
)


def init_speculator(
    vllm_config: VllmConfig,
    device: torch.device,
):
    """Override GPU init_speculator for Ascend NPUs.
    Use AscendEagleSpeculator when eagle is used.
    """
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    requested_window_size = get_ascend_config().draft_window_size
    target_hf_config = getattr(getattr(vllm_config, "model_config", None), "hf_config", None)
    target_model_type = getattr(target_hf_config, "model_type", "") or ""
    decision = resolve_draft_window_policy(
        requested_window_size=requested_window_size,
        method=speculative_config.method,
        target_model_type=target_model_type,
    )

    if decision.state is DraftWindowPolicyState.CONFIG_OFF:
        return _init_speculator_baseline(vllm_config, device)

    if decision.state is DraftWindowPolicyState.FORCED_OFF:
        warning_message = _forced_off_warning_message(decision)
        speculator = _init_speculator_baseline(vllm_config, device)
        logger.warning(warning_message)
        return speculator

    assert decision.effective_window_size is not None
    future_offset = resolve_static_future_offset(
        normalized_method=decision.normalized_method,
        configured_k=speculative_config.num_speculative_tokens,
        uses_dynamic_k=speculative_config.uses_dynamic_speculative_decoding(),
    )
    validate_eligible_window_numbers(
        window_size=decision.effective_window_size,
        future_offset=future_offset,
    )
    speculator = _init_speculator_baseline(vllm_config, device)
    _configure_eligible_draft_window(speculator, decision)
    return speculator


def _forced_off_warning_message(decision: DraftWindowPolicyDecision) -> str:
    """Return the exhaustive diagnostic for a forced-OFF decision.

    ``init_speculator`` is the single construction point for an Ascend MRV2
    speculator.  It emits this message once after each successful construction,
    which gives the warning instance scope without mutable deduplication state.
    The vLLM ``warning_once`` helper instead uses a process-global cache.
    """
    if decision.reason == "mtp":
        message = "MRV2 draft_window_size is ignored for MTP because MTP reuses target layers; forcing the window OFF."
    elif decision.reason == "deepseek_v4_cotrained_dspark":
        message = "MRV2 draft_window_size is ignored for DeepSeek-V4 co-trained DSpark; forcing the window OFF."
    else:
        raise RuntimeError(f"unknown forced-OFF draft-window policy reason: {decision.reason!r}")
    return message


def _configure_eligible_draft_window(
    speculator,
    decision: DraftWindowPolicyDecision,
) -> None:
    speculator._configure_draft_window(decision)


def _init_speculator_baseline(
    vllm_config: VllmConfig,
    device: torch.device,
):
    """Run the pre-window factory dispatch without changing its branch order."""
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    if speculative_config.method == "extract_hidden_states":
        # vLLM #49811 adds this MRV2 speculator on main only. The release
        # configuration rejects this method for MRV2; keep direct calls explicit.
        if vllm_version_is("0.28.0"):
            raise NotImplementedError("extract_hidden_states is not supported by Model Runner V2 in vLLM 0.28.0.")
        # No Ascend-specific behavior beyond update_stream assignment in
        # NPUModelRunner; reuse upstream ExtractHiddenStatesSpeculator as-is.
        from vllm.v1.worker.gpu.spec_decode.extract_hidden_states import (
            ExtractHiddenStatesSpeculator,
        )

        return ExtractHiddenStatesSpeculator(vllm_config, device)
    if speculative_config.use_dspark():
        from vllm_ascend.worker.v2.spec_decode.dspark.speculator import (
            AscendDSparkSpeculator,
        )

        return AscendDSparkSpeculator(vllm_config, device)
    if speculative_config.use_dflash():
        if "DFlash2DraftModel" in speculative_config.draft_model_config.architectures:
            from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import (
                AscendDFlash2Speculator,
            )

            return AscendDFlash2Speculator(vllm_config, device)
        from vllm_ascend.worker.v2.spec_decode.dflash.speculator import (
            AscendDFlashSpeculator,
        )

        return AscendDFlashSpeculator(vllm_config, device)
    if (
        speculative_config.method == "mtp"
        and not speculative_config.use_gemma4_mtp()
        and not speculative_config.use_step3p5_mtp()
    ):
        from vllm_ascend.worker.v2.spec_decode.mtp.speculator import (
            AscendMTPSpeculator,
        )

        return AscendMTPSpeculator(vllm_config, device)
    if speculative_config.use_eagle():
        from vllm_ascend.worker.v2.spec_decode.eagle.speculator import AscendEagleSpeculator

        return AscendEagleSpeculator(vllm_config, device)
    raise NotImplementedError(f"{speculative_config.method} is not supported yet.")
