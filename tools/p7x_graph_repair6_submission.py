# SPDX-License-Identifier: Apache-2.0
"""Disposable diagnostic-child ownership, NOT a production completion protocol."""


class NumericMismatch(RuntimeError):
    """A durable, completed comparison failed; normal branch must stop."""


class CompletedDiagnosticContractError(ValueError):
    """Host contract failure after the diagnostic has drained its submissions."""


class SubmissionState:
    def __init__(self):
        self.phase = "NOT_SUBMITTED"
        self.pending = False
        self.failure = None
        self.transitions = [self.phase]

    def submit(self, phase):
        self.phase, self.pending = phase, True
        self.transitions.append(phase)

    def complete(self, phase):
        # Caller may invoke only AFTER its explicit diagnostic stream drains.
        self.phase, self.pending = phase, False
        self.transitions.append(phase)

    def failed(self, error):
        self.failure = dict(
            type=type(error).__name__, message=str(error), phase=self.phase, uncertain_submission=self.pending
        )

    def snapshot(self):
        return dict(
            phase=self.phase,
            pending=self.pending,
            failure=self.failure,
            transitions=list(self.transitions),
            scope="DIAGNOSTIC_ONLY_NOT_NORMAL_REUSE_PROOF",
        )


def needs_quarantine(*probes):
    return any(p is not None and getattr(p, "poisoned", False) for p in probes)
