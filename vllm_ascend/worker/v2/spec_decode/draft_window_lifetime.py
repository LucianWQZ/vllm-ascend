# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Retain eager Draft read state through the final proposal reader event.

Only the proposal scope may submit a reader. Tensor allocation and stream
ordering belong to its caller; a retained reference never prevents writes.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Protocol


class DraftWindowCompletion(Protocol):
    def is_complete(self) -> bool: ...

    def wait(self) -> None: ...


@dataclass(eq=False)
class DraftWindowOwner:
    sequence: int
    lane: int
    retained: list[object] = field(default_factory=list, repr=False)
    completion: DraftWindowCompletion | None = field(default=None, repr=False)


class DraftWindowOwnerPool:
    """Bounded proposal references on the runner's single execution thread.

    A free lane requires no polling. At capacity, reclaim completed readers
    or wait only for the oldest proposal. Any preparation, event, query or
    wait failure quarantines the pool and keeps all outstanding references.
    """

    def __init__(self, capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("Draft Window owner capacity must be a positive integer")
        self._owners: list[DraftWindowOwner | None] = [None] * capacity
        self._next_sequence = 0
        self._poisoned = False

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    def acquire(self) -> DraftWindowOwner:
        if self.poisoned:
            raise RuntimeError("Draft Window execution is quarantined: device completion is unknown")
        if None not in self._owners:
            self.reclaim_completed()
        if None not in self._owners:
            submitted = [owner for owner in self._owners if owner is not None]
            if any(owner.completion is None for owner in submitted):
                raise RuntimeError("Draft Window pool is full with an unsubmitted proposal")
            oldest = min(submitted, key=lambda owner: owner.sequence)
            self._reclaim(oldest, wait=True)
        lane = self._owners.index(None)
        owner = DraftWindowOwner(sequence=self._next_sequence, lane=lane)
        self._owners[lane] = owner
        self._next_sequence += 1
        return owner

    def retain(self, owner: DraftWindowOwner, *objects: object) -> None:
        if not 0 <= owner.lane < len(self._owners) or self._owners[owner.lane] is not owner:
            raise RuntimeError("Draft Window owner is stale or belongs to another pool")
        if owner.completion is not None or self.poisoned:
            raise RuntimeError("Draft Window read state is already submitted or quarantined")
        owner.retained.extend(objects)

    def _reclaim(self, owner: DraftWindowOwner, *, wait: bool = False) -> bool:
        try:
            assert owner.completion is not None
            if wait:
                owner.completion.wait()
            complete = owner.completion.is_complete()
            if wait and not complete:
                raise RuntimeError("Draft Window completion wait did not complete the reader")
        except BaseException:
            self._poisoned = True
            raise
        if complete:
            owner.retained.clear()
            owner.completion = None
            self._owners[owner.lane] = None
        return bool(complete)

    def reclaim_completed(self) -> int:
        if self.poisoned:
            raise RuntimeError("Draft Window execution is quarantined: device completion is unknown")
        return sum(self._reclaim(owner) for owner in self._owners if owner is not None and owner.completion is not None)

    @contextmanager
    def operation(self, record_completion: Callable[[], DraftWindowCompletion]) -> Iterator[DraftWindowOwner]:
        """Enter before preparation can enqueue; finish after all readers join.

        No separate caller-driven PREPARING/ACTIVE/submit state exists: the
        real runtime always entered native preparation before yielding. Dummy
        and profile bypasses stay outside this scope. A host exception is not
        proof that no native operation was submitted and must not release it.
        """
        owner = self.acquire()
        try:
            yield owner
            # Keep even an invalid/partially recorded token on bookkeeping error.
            owner.completion = record_completion()
            if not callable(getattr(owner.completion, "is_complete", None)) or not callable(
                getattr(owner.completion, "wait", None)
            ):
                raise TypeError("Draft Window completion must implement is_complete and wait")
        except BaseException:
            self._poisoned = True
            raise
