# SPDX-License-Identifier: Apache-2.0
"""Bounded shutdown of only this launcher's descendants, including new sessions."""

import os
import signal
import subprocess
from contextlib import suppress


def stop_owned_tree(process, grace=5):
    try:
        listing = subprocess.check_output(["ps", "-axo", "pid=,ppid="], text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        # At least stop the known process group. Do not claim complete descendant
        # cleanup when the OS denies inventory; propagate that separate failure.
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=grace)
        raise
    parents = {int(p): int(parent) for p, parent in (line.split() for line in listing.splitlines())}
    owned = {process.pid}
    while True:
        descendants = {p for p, parent in parents.items() if parent in owned}
        if descendants <= owned:
            break
        owned |= descendants

    def send(sig):
        for pid in sorted(owned, reverse=True):
            with suppress(ProcessLookupError):
                os.kill(pid, sig)

    send(signal.SIGTERM)
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=grace)
    # A launcher can exit before a setsid child; terminate every captured
    # descendant, not merely the original process group. Never inspect/kill
    # unrelated Git/test/server processes.
    send(signal.SIGKILL)
    return process.wait(timeout=grace), sorted(owned)
