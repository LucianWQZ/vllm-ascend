# SPDX-License-Identifier: Apache-2.0
"""Private Draft metadata copies and the shared Window read-field binding."""

from __future__ import annotations

from collections.abc import Mapping
from copy import copy
from typing import Any


def copy_draft_attention_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Give Draft its own mutable header, preserving read-only Full operands.

    Builders create one metadata object per attention group. Preserve those
    layer aliases. Both callers immediately replace read fields with the
    bank; the supported dense consumer only reads inherited query/Full fields.
    Copy the header rather than reconstructing it so dynamic fields survive.
    """
    if not metadata:
        raise ValueError("Draft attention metadata must be a non-empty layer mapping")
    copies: dict[int, Any] = {}
    result: dict[str, Any] = {}
    for name, layer in metadata.items():
        if layer is None:
            result[name] = None
            continue
        if id(layer) not in copies:
            copies[id(layer)] = copy(layer)
        result[name] = copies[id(layer)]
    return result


def bind_draft_window_read(layer: Any, bank: Any, max_seq_len: int) -> None:
    """Bind only common Draft read operands; query/write/state stay method-owned."""
    layer.block_tables = bank.window_block_table
    layer.seq_lens = bank.device_window_exact
    layer.seq_lens_cpu = bank.cpu_window_exact
    layer.seq_lens_list = bank.cpu_window_exact.tolist()
    layer.max_seq_len = max_seq_len
