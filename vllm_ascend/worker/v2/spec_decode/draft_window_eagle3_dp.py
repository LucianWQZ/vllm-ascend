# SPDX-License-Identifier: Apache-2.0
"""Eagle's rank-local DP dummy ownership, separate from business K0 readers.

The fixed core executes a real dummy on an idle replica to preserve DP/EP
collectives. Its original Full/PAD operands must update the captured first/AR
consumers without creating a Window generation or borrowing a business bank.
"""


def validate_eagle_dp_dummy_replay(owner, manager, descriptor, metadata_batch):
    """Attribute both actual managers without changing their baseline operands."""
    if not getattr(owner, "_eagle3_graph_dp_dummy_active", False):
        return False
    owner._draft_window_graph_binding_provider.require_reusable()
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
