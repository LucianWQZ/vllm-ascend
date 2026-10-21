# SPDX-License-Identifier: Apache-2.0
"""Opt-in retained capture outputs; not a transport or completion mechanism.

Only strong Python references are retained. No extra operator, tensor read,
device query or I/O occurs during capture. Retention can affect allocator reuse
and is therefore diagnostic, never a normal-performance claim.
"""

from __future__ import annotations

import torch


class DraftGraphOutputReferences:
    def __init__(self, owner):
        self.owner = owner
        self.outputs: dict[int, torch.Tensor] = {}
        self.frozen = False

    def retain(self, owner, num_tokens, output):
        if owner is not self.owner or self.frozen:
            raise RuntimeError("numeric capture owner/scope mismatch")
        if not isinstance(output, torch.Tensor) or output.ndim != 2 or output.shape[0] != num_tokens:
            raise RuntimeError("unknown Draft model continuous output contract")
        # Warmup is not retained. Multiple FULL capture invocations for a bucket
        # replace the reference before freeze; the final capture owns the tensor.
        self.outputs[num_tokens] = output

    def freeze(self, captured_tokens):
        if set(self.outputs) != set(captured_tokens):
            raise RuntimeError("numeric output inventory differs from actual captured graphs")
        self.frozen = True

    def get(self, owner, num_tokens):
        if owner is not self.owner or not self.frozen or num_tokens not in self.outputs:
            raise RuntimeError("missing or foreign captured output")
        return self.outputs[num_tokens]
