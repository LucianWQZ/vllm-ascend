# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Retain eager Draft read state through the final proposal reader event.

Only the proposal scope may submit a reader. Tensor allocation and stream
ordering belong to its caller; a retained reference never prevents writes.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch


@dataclass(eq=False)
class DraftWindowOwner:
    retained: list[object] = field(default_factory=list, repr=False)
    completion: torch.npu.Event | None = field(default=None, repr=False)


class DraftWindowOwnerPool:
    """Bounded proposal references on the runner's single execution thread.

    Free capacity requires no polling. At capacity, reclaim completed readers
    or wait only for the oldest proposal. Any preparation, event, query or
    wait failure quarantines the pool and keeps all outstanding references.
    """

    def __init__(self, capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("Draft Window owner capacity must be a positive integer")
        self._capacity = capacity
        self._owners: list[DraftWindowOwner] = []
        self._poisoned = False

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    def acquire(self) -> DraftWindowOwner:
        if self.poisoned:
            raise RuntimeError("Draft Window execution is quarantined: device completion is unknown")
        if len(self._owners) == self._capacity:
            self.reclaim_completed()
        if len(self._owners) == self._capacity:
            if any(owner.completion is None for owner in self._owners):
                raise RuntimeError("Draft Window pool is full with an unsubmitted proposal")
            self._reclaim(self._owners[0], wait=True)
        owner = DraftWindowOwner()
        self._owners.append(owner)
        return owner

    def _reclaim(self, owner: DraftWindowOwner, *, wait: bool = False) -> bool:
        try:
            assert owner.completion is not None
            if wait:
                owner.completion.synchronize()
                complete = True
            else:
                complete = owner.completion.query()
        except BaseException:
            self._poisoned = True
            raise
        if complete:
            owner.retained.clear()
            owner.completion = None
            self._owners.remove(owner)
        return bool(complete)

    def reclaim_completed(self) -> int:
        if self.poisoned:
            raise RuntimeError("Draft Window execution is quarantined: device completion is unknown")
        return sum(self._reclaim(owner) for owner in tuple(self._owners) if owner.completion is not None)

    @contextmanager
    def operation(self) -> Iterator[DraftWindowOwner]:
        """Enter before preparation can enqueue; finish after all readers join.

        No separate caller-driven PREPARING/ACTIVE/submit state exists: the
        real runtime always entered native preparation before yielding. Dummy
        and profile bypasses stay outside this scope. A host exception is not
        proof that no native operation was submitted and must not release it.
        """
        owner = self.acquire()
        try:
            yield owner
            # Own the event before recording: a failed/partial record must not
            # release either the token or any possibly submitted read operands.
            owner.completion = torch.npu.Event()
            owner.completion.record(torch.npu.current_stream())
        except BaseException:
            self._poisoned = True
            raise
