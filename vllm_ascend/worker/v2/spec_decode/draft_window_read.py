# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Own read projections from the framework's current compact Draft tables.

The ordinary builder receives the already mapped Full table in active request
order. CPU lengths are exact snapshots of this same preparation, never upper
bounds. The proposal retains every projection until its final reader completes.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class DraftReadView:
    """Owned read operands; deliberately carries no positions or write slots."""

    block_table: torch.Tensor
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    seq_lens_cpu_upper_bound: torch.Tensor
    max_seq_len: int
    num_reqs: int

    def apply(self, common_metadata: Any) -> None:
        """Apply only to a newly created draft common metadata object."""
        common_metadata.block_table_tensor = self.block_table
        common_metadata.seq_lens = self.seq_lens
        common_metadata.seq_lens_cpu = self.seq_lens_cpu
        common_metadata._seq_lens_cpu = self.seq_lens_cpu
        common_metadata.seq_lens_cpu_upper_bound = self.seq_lens_cpu_upper_bound
        common_metadata.max_seq_len = self.max_seq_len
        common_metadata._draft_window_read = self

    def exact_cpu_for(self, common_metadata: Any) -> torch.Tensor:
        """Certify the snapshot only while all of its read operands are intact."""
        if (
            common_metadata.num_reqs != self.num_reqs
            or common_metadata.seq_lens is not self.seq_lens
            or common_metadata.seq_lens_cpu is not self.seq_lens_cpu
            or common_metadata._seq_lens_cpu is not self.seq_lens_cpu
            or common_metadata.block_table_tensor is not self.block_table
            or common_metadata.seq_lens_cpu_upper_bound is not self.seq_lens_cpu_upper_bound
        ):
            raise ValueError("Draft Window read operands were replaced after projection")
        return self.seq_lens_cpu


def materialize_draft_read(
    block_table: torch.Tensor,
    valid_columns: Sequence[int],
    block_span: int,
    *,
    exact_cpu: torch.Tensor,
    exact_device: torch.Tensor,
    upper_cpu: torch.Tensor,
    starts_cpu: torch.Tensor,
    padded_rows: int,
    retain: Callable[..., None],
) -> DraftReadView:
    """Copy only initialized [S,E) columns from the current compact Full table.

    The framework has already gathered persistent request slots into batch
    order. Reusing it avoids another request mapping and device gather. Each
    copied row and the exact-length DMA source belong to this proposal, so a
    later batch or Eagle consumer cannot overwrite an outstanding reader.
    """
    n = len(valid_columns)
    if block_table.ndim != 2 or block_table.dtype != torch.int32 or block_span <= 0:
        raise ValueError("Invalid compact int32 table geometry")
    if block_table.shape[0] < n or any(count < 0 or count > block_table.shape[1] for count in valid_columns):
        raise ValueError("Invalid initialized compact column extent")
    cpu_values = (exact_cpu, upper_cpu, starts_cpu)
    if any(t.device.type != "cpu" or t.ndim != 1 or t.numel() != n for t in cpu_values):
        raise ValueError("Draft read planning requires one CPU value per active request")
    if any(t.dtype not in (torch.int32, torch.int64) for t in cpu_values):
        raise ValueError("Draft read lengths and starts must be integers")
    if padded_rows < n or exact_device.ndim != 1 or exact_device.numel() < n:
        raise ValueError("Missing device exact lengths or padded row extent")
    if exact_device.device != block_table.device:
        raise ValueError("Draft source and exact lengths must be on the same device")
    ends, upper, starts = (t.to(torch.int64) for t in cpu_values)
    if bool(((ends < 0) | (upper < ends) | (starts < 0) | (starts > ends)).any()):
        raise ValueError("Invalid exact, upper, or start ordering")
    if bool((starts % block_span != 0).any()):
        raise ValueError("This consumer cannot represent an intra-block read start")
    end_columns = (ends + block_span - 1) // block_span
    if any(count < int(end) for count, end in zip(valid_columns, end_columns)):
        raise ValueError("Compact source does not cover the exact readable extent")
    start_columns = starts // block_span
    counts = end_columns - start_columns
    width = max(1, int(counts.max()) if n else 0)
    table = block_table.new_zeros((padded_rows, width))
    retain(block_table, exact_device, table)
    for row, (start, count) in enumerate(zip(start_columns.tolist(), counts.tolist())):
        if count:
            destination, source = table[row, :count], block_table[row, start : start + count]
            retain(source, destination)
            destination.copy_(source, non_blocking=True)
    lengths_cpu = torch.zeros(
        padded_rows, dtype=exact_cpu.dtype, device="cpu", pin_memory=block_table.device.type == "npu"
    )
    lengths_cpu[:n] = ends - starts
    lengths = exact_device.new_empty(padded_rows)
    # Retain both actual operands before a copy can partially submit and fail.
    retain(lengths_cpu, lengths)
    lengths.copy_(lengths_cpu, non_blocking=True)
    # Once E is completed, its visible lengths are also the tight allocation
    # bounds. Query extents and positions remain owned by the original builder.
    return DraftReadView(
        table, lengths, lengths_cpu, lengths_cpu, int(lengths_cpu.max()) if padded_rows else 0, padded_rows
    )
