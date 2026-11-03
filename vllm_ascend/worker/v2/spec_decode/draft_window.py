# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Canonical Draft KV source geometry for capture-owned Graph read banks.

Method policy and interval arithmetic are shared with the ordinary eager
runtime. This module owns only source geometry and persistent Graph backing;
Full write tables and Target state remain owned by the existing runner.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.worker.v2.spec_decode.draft_window_backends import (
    DraftWindowGroupRoute,
    DraftWindowProviderStatus,
    inventory_draft_window_groups,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_plan import (
    window_capacity_tokens,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_policy import (
    DraftWindowPolicyDecision,
    DraftWindowPolicyState,
)


@dataclass(frozen=True, slots=True)
class DraftWindowGraphSource:
    """Actual canonical BlockTables owner and one executable Draft group.

    Only capture banks own read operands. Full tables are borrowed at each
    prepare from BlockTables, whose CPU counts are the source of truth even
    while its UVA accelerator transport rotates.
    """

    policy: DraftWindowPolicyDecision
    group: DraftWindowGroupRoute
    block_tables: Any = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        # Fixed BlockTables constructs matching int32 table/count geometry;
        # inventory already checks its scheduler/kernel spans. Only the new
        # Window capacity is an additional initialization constraint.
        if self.table.shape[1] * self.group.table_column_span < window_capacity_tokens(
            self.window_size, self.group.table_column_span
        ):
            raise ValueError("Draft Graph canonical table cannot cover the Window capacity")

    @property
    def table(self) -> torch.Tensor:
        return self.block_tables.input_block_tables[self.group.gid]

    @property
    def window_size(self) -> int:
        assert self.policy.effective_window_size is not None
        return self.policy.effective_window_size


def _build_draft_window_source_capability_for_context(
    *,
    policy: DraftWindowPolicyDecision,
    runtime_route: str,
    attn_groups: Sequence[Sequence[AttentionGroup]],
    block_tables: Any,
    kv_cache_config: Any,
    vllm_config: Any,
    dflash2_graph_owner: Any = None,
) -> DraftWindowGraphSource:
    """Build source geometry after the caller validates its real Graph executor.

    Executor features do not turn a dense Draft cache into a sharded provider.
    Check the actual source layout directly, without constructing an
    artificial eager capability decision.
    """
    from vllm_ascend.worker.v2.spec_decode.draft_window_graph_runtime import validate_graph_context

    validate_graph_context(vllm_config, block_tables)
    if policy.state is not DraftWindowPolicyState.ELIGIBLE_ON or policy.effective_window_size is None:
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
    return DraftWindowGraphSource(policy=policy, group=group, block_tables=block_tables)
