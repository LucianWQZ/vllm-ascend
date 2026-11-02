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


def bind_draft_window_read(layer: Any, bank: Any, max_seq_len: int) -> None:
    """Bind only common Draft read operands; query/write/state stay method-owned."""
    layer.block_tables = bank.window_block_table
    layer.seq_lens = bank.device_window_exact
    layer.seq_lens_cpu = bank.cpu_window_exact
    layer.seq_lens_list = bank.cpu_window_exact.tolist()
    layer.max_seq_len = max_seq_len


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
        or owner._draft_window_proposal is not None
        or not isinstance(provider, DraftWindowGraphBindingProvider)
        or provider._active_inputs is not None
        or provider.method not in ("dflash", "dspark")
    ):
        raise RuntimeError("DP dummy Graph requires its explicit full-read scope without a live Window generation")
    return True
