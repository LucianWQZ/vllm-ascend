# SPDX-License-Identifier: Apache-2.0
"""Bind one metadata build to the parallel Draft FULL dispatch.

The fixed MRV2 proposal requests metadata before invoking its Graph manager.
That temporary result is not consumed by FULL replay: build once against the
selected persistent bank instead, after checking the proposal and bucket.
Non-FULL and prefill bypasses retain their original metadata path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vllm_ascend.worker.v2.spec_decode.draft_window_graph import DraftWindowGraphBindingProvider


@dataclass(frozen=True, slots=True)
class ParallelGraphMetadataSelection:
    mapping_epoch: int
    request_ids: tuple[str, ...]
    padded_requests: int
    padded_tokens: int


def is_parallel_graph_dp_dummy(owner: Any) -> bool:
    """Require the explicit no-generation scope before using baseline operands.

    This is an idle replica's real Graph submission, not an eager fallback,
    a capture carrier or a reuse of the preceding business generation.
    """
    active = getattr(owner, "_draft_window_dp_dummy_active", False)
    if active is False:
        return False
    if active is not True:
        raise RuntimeError("DP dummy Graph scope marker must be an explicit boolean")
    provider = getattr(owner, "_draft_window_graph_binding_provider", None)
    dp_size = getattr(owner, "dp_size", None)
    if (
        type(dp_size) is not int
        or dp_size <= 1
        or getattr(owner, "_draft_window_bypass_active", False) is not True
        or getattr(owner, "_draft_window_prefill_bypass_active", False) is not False
        or getattr(owner, "_draft_window_proposal_context", None) is not None
        or getattr(owner, "_draft_window_live_inputs", None) is not None
        or not isinstance(provider, DraftWindowGraphBindingProvider)
        or provider.inventory_method not in ("dflash", "dspark")
    ):
        raise RuntimeError("DP dummy Graph requires its explicit full-read scope without a live Window generation")
    return True


def select_parallel_graph_metadata(owner: Any, descriptor: Any) -> None:
    context = getattr(owner, "_draft_window_proposal_context", None)
    owner._draft_window_selected_metadata = None
    if context is None or getattr(descriptor.cg_mode, "name", None) != "FULL":
        return
    provider = getattr(owner, "_draft_window_graph_binding_provider", None)
    if not isinstance(provider, DraftWindowGraphBindingProvider) or provider.inventory_method not in (
        "dspark",
        "dflash",
    ):
        raise RuntimeError("FULL metadata selection requires the real parallel Graph provider")
    n, tokens = descriptor.num_reqs, descriptor.num_tokens
    if type(n) is not int or n < len(context.ordered_request_ids) or tokens != n * owner.num_query_per_req:
        raise RuntimeError("single-metadata selection does not match actual FULL dispatch geometry")
    owner._draft_window_selected_metadata = ParallelGraphMetadataSelection(
        context.mapping_epoch, context.ordered_request_ids, n, tokens
    )


def defer_parallel_graph_metadata(owner: Any, inputs: Any, *, padded_requests: int, padded_tokens: int) -> bool:
    selected = getattr(owner, "_draft_window_selected_metadata", None)
    if selected is None:
        return False  # Eager/phase bypass retains its actual metadata build.
    identity = inputs.identity
    inputs.validate_for(identity)
    if (
        selected.mapping_epoch != identity.mapping_epoch
        or selected.request_ids != identity.ordered_request_fingerprint
        or selected.padded_requests != padded_requests
        or selected.padded_requests != identity.padded_request_count
        or selected.padded_tokens != padded_tokens
    ):
        raise RuntimeError("stale or wrong-bucket deferred Graph metadata selection")
    runtime = owner._draft_window_runtime_state
    if runtime._active_inputs is not inputs or owner._draft_window_live_inputs is not inputs:
        raise RuntimeError("deferred Graph metadata does not own current P3 materialization")
    if runtime.metadata_store._entry is not None or getattr(owner, "_draft_window_deferred_metadata", None) is not None:
        raise RuntimeError("deferred Graph metadata cannot replace an existing publication")
    owner._draft_window_deferred_metadata = identity
    return True


def validate_deferred_parallel_graph_metadata(owner: Any, inputs: Any, descriptor: Any) -> None:
    selected = getattr(owner, "_draft_window_selected_metadata", None)
    if selected is None:
        raise RuntimeError("single-build FULL replay has no actual dispatch selection")
    if (
        getattr(descriptor.cg_mode, "name", None) != "FULL"
        or descriptor.num_reqs != selected.padded_requests
        or descriptor.num_tokens != selected.padded_tokens
        or getattr(owner, "_draft_window_deferred_metadata", None) != inputs.identity
        or owner._draft_window_runtime_state.metadata_store._entry is not None
    ):
        raise RuntimeError("deferred FULL replay does not match the current unconsumed generation")
