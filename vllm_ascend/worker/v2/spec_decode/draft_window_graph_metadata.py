# SPDX-License-Identifier: Apache-2.0
"""Isolate mutable Draft metadata and identify the real idle-DP scope."""

from __future__ import annotations

from collections.abc import Mapping
from copy import copy
from typing import Any


def copy_draft_attention_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the concrete metadata fields that Draft preparation may modify.

    Builders create one metadata object per attention group. Preserve those
    layer aliases, but give Draft its own object and Host length/cache
    containers. Tensor operands remain borrowed: window read operands are
    bound explicitly to the owned bank, while Full query/write fields and
    masks are intentionally read-only canonical references.
    """
    if not isinstance(metadata, Mapping) or not metadata:
        raise ValueError("Draft attention metadata must be a non-empty layer mapping")
    copies: dict[int, Any] = {}
    result: dict[str, Any] = {}
    for name, layer in metadata.items():
        if layer is None:
            result[name] = None
            continue
        if id(layer) not in copies:
            private = copy(layer)
            if private is layer:
                raise TypeError("Draft attention metadata layers must support a private shallow copy")
            for field in ("seq_lens_list", "actual_seq_lengths_q"):
                value = getattr(layer, field, None)
                if value is not None:
                    if not isinstance(value, list):
                        raise TypeError(f"Draft attention metadata {field} must be a list")
                    object.__setattr__(private, field, list(value))
            cache = getattr(layer, "qfa_metadata_cache", None)
            if cache is not None:
                if not isinstance(cache, dict):
                    raise TypeError("Draft attention metadata qfa_metadata_cache must be a dict")
                object.__setattr__(private, "qfa_metadata_cache", dict(cache))
            copies[id(layer)] = private
        result[name] = copies[id(layer)]
    return result


def validate_draft_metadata_isolation(metadata: Mapping[str, Any], *, target_metadata: Mapping[str, Any]) -> None:
    """Guard the actual metadata mutation boundary, not arbitrary object graphs.

    Read-only Tensor sharing is intentional. The Graph provider separately
    checks the precise window bank and canonical Full write/query operands.
    """
    if not isinstance(metadata, Mapping) or not isinstance(target_metadata, Mapping):
        raise ValueError("Draft and Target metadata must be layer mappings")
    mutable_fields = ("seq_lens_list", "actual_seq_lengths_q", "qfa_metadata_cache")
    target_objects = {id(layer) for layer in target_metadata.values() if layer is not None}
    target_containers = {
        id(value)
        for layer in target_metadata.values()
        if layer is not None
        for field in mutable_fields
        if isinstance(value := getattr(layer, field, None), (list, dict))
    }
    if metadata is target_metadata or any(
        layer is not None
        and (
            id(layer) in target_objects
            or any(
                isinstance(value := getattr(layer, field, None), (list, dict)) and id(value) in target_containers
                for field in mutable_fields
            )
        )
        for layer in metadata.values()
    ):
        raise ValueError("Draft metadata mutable fields alias Target metadata")


def is_parallel_graph_dp_dummy(owner: Any) -> bool:
    """Require the explicit no-generation scope before using baseline operands.

    This is an idle replica's real Graph submission, not an eager fallback,
    a capture carrier or a reuse of the preceding business generation.
    """
    from vllm_ascend.worker.v2.spec_decode.draft_window_graph import DraftWindowGraphBindingProvider

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
        or owner._draft_window_epoch_source._active is not None
        or not isinstance(provider, DraftWindowGraphBindingProvider)
        or provider._active_inputs is not None
        or provider.method not in ("dflash", "dspark")
    ):
        raise RuntimeError("DP dummy Graph requires its explicit full-read scope without a live Window generation")
    return True
