# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Resolve Draft Window policy before constructing any runtime resources."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DraftWindowPolicyDecision:
    effective_window_size: int | None
    normalized_method: str
    reason: str | None = None

    @property
    def enabled(self) -> bool:
        return self.effective_window_size is not None


def resolve_draft_window_policy(
    *,
    requested_window_size: int | None,
    method: str,
    target_model_type: str = "",
    builtin_v4_dspark: bool | None = None,
    dynamic_k: bool = False,
    is_dflash2: bool = False,
) -> DraftWindowPolicyDecision:
    """Return a policy decision without inspecting a backend or allocating state.

    ``None`` is CONFIG_OFF. MTP and an explicitly identified co-trained V4
    DSpark route are forced off before number or capability validation. The
    caller identifies the dedicated built-in category from normalized target
    and draft config. Paths and revisions do not define that category, and
    target model type alone does not establish it. ``False`` explicitly
    identifies another draft category; ``None`` leaves that boundary open.

    An enabled decision establishes method policy only. Runtime consumers must
    independently validate their actual backend, geometry and ownership.
    """
    method = method.strip().lower()
    model_type = target_model_type.strip().lower()

    if requested_window_size is None:
        return DraftWindowPolicyDecision(None, method)
    if method == "mtp":
        return DraftWindowPolicyDecision(None, method, "mtp")
    if method == "dspark" and builtin_v4_dspark is True:
        return DraftWindowPolicyDecision(None, method, "builtin_v4_dspark")

    if method not in {"eagle3", "dflash", "dspark"}:
        raise NotImplementedError(f"draft_window_size is not supported for speculative method {method!r}")
    if is_dflash2 and method != "dflash":
        raise NotImplementedError("DFlash2 draft_window_size requires speculative method 'dflash'")
    if dynamic_k:
        raise NotImplementedError("draft_window_size requires static speculative decoding")
    if method == "dspark" and model_type.startswith("deepseek_v4") and builtin_v4_dspark is not False:
        raise ValueError(
            "DSpark with a V4 target requires a known model-route identity before enabling draft_window_size"
        )
    if isinstance(requested_window_size, bool) or not isinstance(requested_window_size, int):
        raise TypeError("draft_window_size must be an integer or None")
    if requested_window_size <= 0:
        raise ValueError("draft_window_size must be greater than zero")
    return DraftWindowPolicyDecision(requested_window_size, method)
