# SPDX-License-Identifier: Apache-2.0
"""Eagle's rank-local DP dummy ownership, separate from business K0 readers.

The fixed core executes a real dummy on an idle replica to preserve DP/EP
collectives. Its original Full/PAD operands must update the captured first/AR
consumers without creating a Window generation or borrowing a business bank.
"""

from contextlib import contextmanager


@contextmanager
def eagle3_dp_dummy_scope(owner, input_batch, *, dummy_run, is_profile):
    """Retain the fixed dummy computation and quarantine uncertain submissions.

    A real K0 proposal is not dummy and never enters this scope. Initial
    profile/capture keeps its existing path and is not runtime idle evidence.
    No device completion is inferred from leaving this Host scope.
    """
    session = getattr(owner, "_eagle3_graph_session", None)
    if (
        session is None
        or getattr(owner, "dp_size", None) != 2
        or is_profile
        or session._capture_call is not None
        or not (dummy_run or getattr(input_batch, "is_dummy", False))
    ):
        yield
        return
    if dummy_run is not True or getattr(input_batch, "is_dummy", False) is not True:
        raise RuntimeError("Eagle DP idle must be the fixed core's real dummy input, not a K0 business row")
    if getattr(owner, "_eagle3_graph_dp_dummy_active", False) is not False:
        raise RuntimeError("Eagle DP dummy Graph scope cannot nest")
    owner._eagle3_graph_dp_dummy_active = True
    try:
        provider = session.provider
        provider.require_reusable()
        if owner.input_batch.is_dummy is not True or provider.proposal != () or provider._active_bank is not None:
            raise RuntimeError("Eagle DP dummy requires its explicit local1 no-business-generation Full scope")
        yield
    except BaseException:
        session.provider.quarantine()
        raise
    finally:
        owner._eagle3_graph_dp_dummy_active = False


def validate_eagle_dp_dummy_replay(owner, manager, descriptor, metadata_batch):
    """Attribute both actual managers without changing their baseline operands."""
    if not getattr(owner, "_eagle3_graph_dp_dummy_active", False):
        return False
    session = owner._eagle3_graph_session
    session.provider.require_reusable()
    first = manager.is_draft_model_prefill
    if (
        getattr(descriptor.cg_mode, "name", None) != "FULL"
        or type(descriptor.num_reqs) is not int
        or descriptor.num_reqs != 1
        or type(descriptor.num_tokens) is not int
        or descriptor.num_tokens != (4 if first else 1)
        or len(metadata_batch) != (1 if first else 2)
    ):
        raise RuntimeError("Eagle DP dummy lost its real first Q4 or two AR Q1 consumer/update-stream identity")
    return True
