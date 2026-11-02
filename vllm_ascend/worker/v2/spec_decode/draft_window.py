# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Canonical Draft KV sources and generation-owned Graph read buffers.

Method policy and interval arithmetic are shared with the ordinary eager
runtime. This module owns only source geometry and persistent Graph backing;
Full write tables and Target state remain owned by the existing runner.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.worker.v2.spec_decode.draft_window_backends import (
    DraftWindowGroupRoute,
    DraftWindowProviderStatus,
    inventory_draft_window_groups,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_plan import (
    DraftWindowPlan,
    compute_draft_window_plan,
    window_capacity_tokens,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_policy import (
    DraftWindowPolicyDecision,
    DraftWindowPolicyState,
)

if TYPE_CHECKING:
    from vllm_ascend.worker.v2.spec_decode.draft_window_eager import DraftWindowProposalContext
    from vllm_ascend.worker.v2.spec_decode.draft_window_graph import DraftWindowGraphBindingProvider


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


def gather_valid_num_blocks_cpu(
    source: DraftWindowGraphSource,
    *,
    idx_mapping_np: np.ndarray,
    num_reqs: int,
    num_reqs_padded: int,
) -> torch.Tensor:
    """Gather canonical request-slot counts into the current batch order."""
    mapping = np.asarray(idx_mapping_np)
    if num_reqs < 0 or num_reqs_padded < num_reqs or mapping.ndim != 1 or mapping.shape[0] < num_reqs:
        raise ValueError("CPU mapping and active/padded request counts disagree")
    if not np.issubdtype(mapping.dtype, np.integer):
        raise TypeError("CPU idx_mapping must contain integer indices")
    counts = source.block_tables.num_blocks.cpu[source.group.gid]
    out = torch.zeros(num_reqs_padded, dtype=torch.int32, device="cpu")
    if num_reqs:
        torch.index_select(
            counts, 0, torch.from_numpy(mapping[:num_reqs].astype(np.int64, copy=False)), out=out[:num_reqs]
        )
    return out


def _shares_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Conservatively reject output views into any borrowed source storage."""
    if left.device != right.device:
        return False
    try:
        return left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
    except (AttributeError, RuntimeError):
        return left.data_ptr() == right.data_ptr()


def _build_draft_window_source_capability_for_context(
    *,
    policy: DraftWindowPolicyDecision,
    runtime_route: str,
    attn_groups: Sequence[Sequence[AttentionGroup]],
    block_tables: Any,
    kv_cache_config: Any,
    causal: bool | Mapping[int, bool],
    vllm_config: Any,
    dflash2_graph_owner: Any = None,
) -> DraftWindowGraphSource:
    """Build source geometry after the caller validates its real Graph executor.

    Executor features do not turn a dense Draft cache into a sharded provider.
    Keep the actual context in the owner identity and check the source layout
    directly, without constructing an artificial eager capability decision.
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
        validate_dflash2_graph_read_source(
            dflash2_graph_owner, group, causal=causal, window_size=policy.effective_window_size
        )
    if vllm_config.speculative_config.uses_dynamic_speculative_decoding():
        raise NotImplementedError("Draft Window Graph source requires static speculative decoding")
    return DraftWindowGraphSource(policy=policy, group=group, block_tables=block_tables)


@dataclass(frozen=True, slots=True, init=False)
class DraftWindowWorkspace:
    """Fixed-capacity, stable-address backing for one serialized read lane."""

    max_num_reqs: int
    table_capacity_columns: int
    window_block_table: torch.Tensor
    device_window_exact: torch.Tensor
    cpu_window_exact: torch.Tensor

    def __init__(
        self,
        *,
        max_num_reqs: int,
        table_capacity_columns: int,
        table_template: torch.Tensor,
    ) -> None:
        if max_num_reqs <= 0:
            raise ValueError("workspace request capacity must be greater than 0")
        if table_capacity_columns <= 0:
            raise ValueError("workspace table capacity must be greater than 0")
        if table_template.ndim != 2:
            raise ValueError("Draft workspace requires a two-dimensional source table")
        if table_template.dtype != torch.int32:
            raise TypeError("Draft workspace tables must use int32")
        object.__setattr__(self, "max_num_reqs", max_num_reqs)
        object.__setattr__(self, "table_capacity_columns", table_capacity_columns)
        object.__setattr__(self, "window_block_table", table_template.new_zeros((max_num_reqs, table_capacity_columns)))
        object.__setattr__(
            self, "device_window_exact", torch.zeros(max_num_reqs, dtype=torch.int32, device=table_template.device)
        )
        object.__setattr__(self, "cpu_window_exact", torch.zeros(max_num_reqs, dtype=torch.int32))

    def materialize(
        self,
        *,
        inputs: "DraftMetadataInputs",
        source: DraftWindowGraphSource,
        cpu_full_exact: torch.Tensor,
        cpu_valid_num_blocks: torch.Tensor,
        frozen_row_plans: tuple[DraftWindowPlan, ...] | None,
    ) -> None:
        """Rewrite all active and inert state from the canonical full source."""
        actual = len(inputs.proposal.ordered_request_ids)
        padded = inputs.padded_request_count
        if padded > self.max_num_reqs:
            raise RuntimeError(f"draft-window request capacity exceeded: padded={padded} capacity={self.max_num_reqs}")
        _validate_window_length_input(
            cpu_full_exact,
            name="CPU Full exact",
            padded_request_count=padded,
            expected_device=torch.device("cpu"),
        )
        _validate_window_length_input(
            cpu_valid_num_blocks,
            name="CPU valid block counts",
            padded_request_count=padded,
            expected_device=torch.device("cpu"),
        )
        workspace_tensors = (
            self.window_block_table,
            self.device_window_exact,
            self.cpu_window_exact,
        )
        if any(
            _shares_storage(supplied, backing)
            for supplied in (cpu_full_exact, cpu_valid_num_blocks)
            for backing in workspace_tensors
        ):
            raise ValueError("Window length inputs must not alias workspace backing")
        if actual and bool(torch.any(cpu_full_exact[:actual] < 0)):
            raise ValueError("Full exact lengths must be greater than or equal to 0")
        if frozen_row_plans is not None and len(frozen_row_plans) != actual:
            raise ValueError("frozen Window anchors must match active request rows")

        source_table = source.table
        if source_table.device != self.window_block_table.device or source_table.dtype != self.window_block_table.dtype:
            raise ValueError("canonical source table and Window bank must share a dtype and device")
        self.window_block_table.zero_()
        self.cpu_window_exact.zero_()
        span = source.group.table_column_span
        for row, full_read in enumerate(inputs.proposal.phase_plan.full_read_rows):
            exact = int(cpu_full_exact[row])
            valid = int(cpu_valid_num_blocks[row])
            if not 0 <= valid <= source_table.shape[1]:
                raise RuntimeError("current valid block count is outside the canonical source row")
            if full_read:
                start, required, visible = 0, valid, exact
            else:
                plan = (
                    compute_draft_window_plan(
                        full_exact_end=exact, window_size=source.window_size, block_span_tokens=span
                    )
                    if frozen_row_plans is None
                    else frozen_row_plans[row].at_exact_end(exact)
                )
                if plan.window_start % span:
                    raise NotImplementedError("Draft Graph cannot read an intra-block Window start")
                start = plan.window_start // span
                visible = plan.window_visible_length
                required = (visible + span - 1) // span
            if required > self.table_capacity_columns:
                raise RuntimeError("draft-window table capacity exceeded")
            if start + required > valid or visible > required * span:
                raise RuntimeError("draft-window source does not cover the exact readable interval")
            if required:
                self.window_block_table[row, :required].copy_(source_table[row, start : start + required])
            self.cpu_window_exact[row] = visible
        # The capture-owned exact CPU buffer is also the valid metadata bound.
        # No intermediate Window length vector or upper-vector copy is needed.
        self.device_window_exact.copy_(self.cpu_window_exact)


@dataclass(frozen=True, slots=True, eq=False)
class DraftMetadataInputs:
    """One generation handle retaining its actual read bank and runtime owner.

    The bank owns all read tensors. A new handle on every materialization
    rejects stale references even when the next generation reuses that bank.
    Positions, query layout, write slots and Target metadata remain separate.
    """

    proposal: "DraftWindowProposalContext"
    padded_request_count: int
    step: int
    workspace: DraftWindowWorkspace = field(repr=False, compare=False)
    _owner: "DraftWindowGraphBindingProvider" = field(repr=False, compare=False)
    metadata: Mapping[str, Any] | None = field(init=False, default=None, repr=False, compare=False)

    def validate(self) -> None:
        if self._owner._active_inputs is not self:
            raise RuntimeError("cannot use retired Draft inputs")
        if self._owner.quarantined:
            raise RuntimeError("cannot use quarantined Draft inputs")

    def publish(self, metadata: Mapping[str, Any]) -> None:
        """Publish once on this actual generation, without a second owner store."""
        try:
            self.validate()
            if self.metadata is not None:
                raise RuntimeError("cannot overwrite active draft metadata")
            object.__setattr__(self, "metadata", metadata)
        except BaseException:
            if self._owner._active_inputs is self and not self._owner.quarantined:
                self._owner.quarantine_generation(self)
            raise


def _validate_window_length_input(
    tensor: torch.Tensor,
    *,
    name: str,
    padded_request_count: int,
    expected_device: torch.device,
) -> None:
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if tensor.shape[0] < padded_request_count:
        raise ValueError(f"{name} is too small for the padded request count")
    if tensor.dtype != torch.int32:
        raise TypeError(f"{name} must use int32")
    if tensor.device != expected_device:
        raise ValueError(f"{name} must live on {expected_device}")
