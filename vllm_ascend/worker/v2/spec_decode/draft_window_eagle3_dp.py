# SPDX-License-Identifier: Apache-2.0
"""Eagle's rank-local DP dummy ownership, separate from business K0 readers.

The fixed core executes a real dummy on an idle replica to preserve DP/EP
collectives. Its original Full/PAD operands must update the captured first/AR
consumers without creating a Window generation or borrowing a business bank.
"""

from contextlib import contextmanager


def _validate_eagle_dp_dummy(owner, session):
    session.require_reusable()
    provider = getattr(owner, "_draft_window_graph_binding_provider", None)
    support = getattr(owner, "_draft_graph_review_limit", None)
    if (
        getattr(owner, "method", None) != "eagle3"
        or getattr(owner, "dp_size", None) != 2
        or getattr(owner, "num_speculative_steps", None) != 3
        or getattr(owner, "max_num_reqs", None) != 1
        or support is None
        or support.method != "eagle3"
        or support.data_parallel_size != 2
        or support.max_requests != 1
        or owner.input_batch.is_dummy is not True
        or getattr(owner, "_eagle3_window_bypass_active", False) is not True
        or getattr(owner, "_eagle3_window_continuing_prefill", False) is not False
        or getattr(owner, "_eagle3_window_proposal_context", None) is not None
        or provider is None
        or getattr(provider, "_active_inputs", None) is not None
        or getattr(owner, "_draft_window_graph_submission_uncertain", False) is not False
        or session._capture_call is not None
        or provider is not session.provider
        or getattr(provider, "method", None) != "eagle3"
    ):
        raise RuntimeError("Eagle DP dummy requires its explicit local1 no-business-generation Full scope")


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
        _validate_eagle_dp_dummy(owner, session)
        yield
        _validate_eagle_dp_dummy(owner, session)
    except BaseException:
        owner._quarantine_eagle3_window()
        raise
    finally:
        owner._eagle3_graph_dp_dummy_active = False


def validate_eagle_dp_dummy_replay(owner, manager, descriptor, metadata_batch):
    """Attribute both actual managers without changing their baseline operands."""
    active = getattr(owner, "_eagle3_graph_dp_dummy_active", False)
    if active is False:
        return False
    if active is not True:
        raise RuntimeError("Eagle DP dummy marker must be an exact boolean")
    session = owner._eagle3_graph_session
    _validate_eagle_dp_dummy(owner, session)
    first = manager.is_draft_model_prefill
    if (
        type(first) is not bool
        or manager.speculator is not owner
        or manager.update_stream is not owner.update_stream
        or getattr(descriptor.cg_mode, "name", None) != "FULL"
        or type(descriptor.num_reqs) is not int
        or descriptor.num_reqs != 1
        or type(descriptor.num_tokens) is not int
        or descriptor.num_tokens != (4 if first else 1)
        or len(metadata_batch) != (1 if first else 2)
    ):
        raise RuntimeError("Eagle DP dummy lost its real first Q4 or two AR Q1 consumer/update-stream identity")
    return True
