# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Qualify the actual Draft consumer and create its capture-bank owner.

Method policy and interval arithmetic are shared with the ordinary eager
runtime. The provider owns source geometry and persistent Graph backing;
Full write tables and Target state remain owned by the existing runner.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.worker.v2.spec_decode.draft_window_backends import (
    DraftWindowProviderStatus,
    inventory_draft_window_groups,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_policy import (
    DraftWindowPolicyDecision,
)

if TYPE_CHECKING:
    from vllm_ascend.worker.v2.spec_decode.draft_window_graph import DraftWindowGraphBindingProvider


def create_draft_window_graph_provider(
    *,
    policy: DraftWindowPolicyDecision,
    runtime_route: str,
    attn_groups: Sequence[Sequence[AttentionGroup]],
    block_tables: Any,
    kv_cache_config: Any,
    vllm_config: Any,
    dflash2_graph_owner: Any = None,
) -> "DraftWindowGraphBindingProvider":
    """Build source geometry after the caller validates its real Graph executor.

    Executor features do not turn a dense Draft cache into a sharded provider.
    Check the actual source layout directly, without constructing an
    artificial eager capability decision.
    """
    from vllm_ascend.worker.v2.spec_decode.draft_window_graph_runtime import validate_graph_context

    validate_graph_context(vllm_config, block_tables)
    if not policy.enabled:
        raise ValueError("Graph source requires an eligible Window policy")
    runtime_route = runtime_route.strip().lower()
    expected_method = "dflash" if runtime_route == "dflash2" else runtime_route
    if runtime_route not in {"eagle3", "dflash", "dflash2", "dspark"} or expected_method != policy.normalized_method:
        raise ValueError("Graph source route differs from the actual method policy")
    if runtime_route != "dflash2" and dflash2_graph_owner is not None:
        raise ValueError("DFlash2 source owner belongs to a different method")
    groups = inventory_draft_window_groups(
        attn_groups=attn_groups,
        kv_cache_config=kv_cache_config,
        block_tables=block_tables,
        parallel_config=vllm_config.parallel_config,
        full_retention_dflash2=dflash2_graph_owner is not None,
    )
    if len(groups) != 1:
        raise NotImplementedError(f"Draft Graph requires one executable consumer group, found {len(groups)}")
    group = groups[0]
    if runtime_route != "dflash2" and (
        group.status is not DraftWindowProviderStatus.DIRECT_DENSE or group.native_window is not None
    ):
        raise NotImplementedError("Draft Graph requires its supported dense or DFlash2 native reader")
    if runtime_route == "dflash2":
        from vllm_ascend.worker.v2.spec_decode.draft_window_dflash2_graph import validate_dflash2_graph_read_source

        if dflash2_graph_owner is None:
            raise TypeError("DFlash2 Graph requires its actual model owner")
        validate_dflash2_graph_read_source(dflash2_graph_owner, group, window_size=policy.effective_window_size)
    if vllm_config.speculative_config.uses_dynamic_speculative_decoding():
        raise NotImplementedError("Draft Window Graph source requires static speculative decoding")
    from vllm_ascend.worker.v2.spec_decode.draft_window_graph import DraftWindowGraphBindingProvider

    return DraftWindowGraphBindingProvider(policy=policy, group=group, block_tables=block_tables)
