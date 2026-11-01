# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Canonical Draft KV sources and generation-owned Graph read buffers.

Method policy and interval arithmetic are shared with the ordinary eager
runtime. This module owns only source geometry and persistent Graph backing;
Full write tables and Target state remain owned by the existing runner.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
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
    project_draft_window_interval,
    window_capacity_tokens,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_policy import (
    DraftWindowPolicyDecision,
    DraftWindowPolicyState,
)

if TYPE_CHECKING:
    from vllm_ascend.worker.v2.spec_decode.draft_window_eager import DraftWindowProposalContext


@dataclass(frozen=True, slots=True)
class FullDraftKVSource:
    """Borrowed, read-only references to canonical full BlockTables state.

    Canonical tables are already gathered into batch order by BlockTables.
    The Window owner reads them without changing Full write state.
    """

    logical_block_sizes: tuple[int, ...]
    kernel_block_sizes: tuple[int, ...]
    _block_tables_owner: Any = field(repr=False, compare=False)
    _input_block_tables: tuple[torch.Tensor, ...] = field(
        repr=False,
        compare=False,
    )
    _persistent_num_blocks_cpu: torch.Tensor = field(repr=False, compare=False)

    @property
    def num_groups(self) -> int:
        return len(self._input_block_tables)

    def input_block_table_row_width(self, global_group_id: int) -> int:
        self._validate_global_group_id(global_group_id)
        return self._input_block_tables[global_group_id].shape[1]

    def validate_owner_bindings(self) -> None:
        """Reject replacement of a canonical owner tensor after P2 binding."""
        current_tables = tuple(self._block_tables_owner.input_block_tables)
        if len(current_tables) != len(self._input_block_tables):
            raise RuntimeError("canonical input block table group count changed")
        for global_group_id, (current, bound) in enumerate(zip(current_tables, self._input_block_tables)):
            if current is not bound:
                raise RuntimeError(
                    f"canonical input block table {global_group_id} was replaced after FullSource binding"
                )
        if self._block_tables_owner.num_blocks.cpu is not self._persistent_num_blocks_cpu:
            raise RuntimeError("canonical persistent num_blocks.cpu was replaced after FullSource binding")
        current_gpu = self._block_tables_owner.num_blocks.gpu
        if (
            current_gpu.ndim != self._persistent_num_blocks_cpu.ndim
            or current_gpu.shape != self._persistent_num_blocks_cpu.shape
            or current_gpu.dtype != self._persistent_num_blocks_cpu.dtype
        ):
            raise RuntimeError("current num_blocks.gpu view does not match its canonical CPU owner")

    def _validate_global_group_id(self, global_group_id: int) -> None:
        if not 0 <= global_group_id < self.num_groups:
            raise IndexError(f"global group id {global_group_id} is outside [0, {self.num_groups})")


@dataclass(frozen=True, slots=True)
class DraftWindowReadDescriptor:
    """Read-only description of a row slice in the canonical full source."""

    global_group_id: int
    source_start_column: int
    first_token_offset: int
    required_columns: int
    safe_extent_tokens: int
    valid_num_blocks: int
    source_row_width: int

    @property
    def source_end_column(self) -> int:
        return self.source_start_column + self.required_columns


@dataclass(frozen=True, slots=True)
class DraftWindowSourceCapability:
    """Canonical read projection; it never materializes a window table."""

    policy: DraftWindowPolicyDecision
    runtime_route: str
    group: DraftWindowGroupRoute
    full_source: FullDraftKVSource
    expected_future_offset: int
    parallel_config: Any
    tensor_parallel_rank: int
    workspace_capacity_columns: int | None = None
    capacity_columns: int = field(init=False)
    window_size: int = field(init=False)

    def __post_init__(self) -> None:
        window = self.policy.effective_window_size
        if self.policy.state is not DraftWindowPolicyState.ELIGIBLE_ON or window is None:
            raise ValueError("Graph source requires an eligible Window policy")
        dense = self.group.status is DraftWindowProviderStatus.DIRECT_DENSE and self.group.native_window is None
        native_dflash2 = (
            self.runtime_route == "dflash2" and self.group.status is DraftWindowProviderStatus.NATIVE_SWA_CANDIDATE
        )
        if not dense and not native_dflash2:
            raise NotImplementedError(
                f"Draft Graph source cannot read gid={self.group.gid} "
                f"layout={self.group.status.value}: {self.group.reason}"
            )
        _require_nonnegative_python_int(self.expected_future_offset, "expected future offset")
        expected_offset = resolve_static_future_offset(
            normalized_method=self.policy.normalized_method,
            configured_k=self.expected_future_offset,
            uses_dynamic_k=False,
        )
        if expected_offset != self.expected_future_offset:
            raise ValueError("configured future offset does not match method semantics")
        self.full_source.validate_owner_bindings()
        _validate_source_geometry(
            group=self.group,
            logical_block_sizes=self.full_source.logical_block_sizes,
            kernel_block_sizes=self.full_source.kernel_block_sizes,
            num_groups=self.full_source.num_groups,
        )
        object.__setattr__(self, "window_size", window)
        span = self.group.table_column_span
        capacity_tokens = window_capacity_tokens(window, span)
        object.__setattr__(self, "capacity_columns", (capacity_tokens + span - 1) // span)

    def _validate_plan(self, plan: DraftWindowPlan) -> None:
        expected = compute_draft_window_plan(
            full_exact_end=plan.full_exact_end,
            future_offset=self.expected_future_offset,
            window_size=self.window_size,
            block_span_tokens=self.group.table_column_span,
        )
        if plan != expected:
            raise ValueError("scalar plan does not match the canonical method horizon, anchor and capacity")

    def describe(self, plan: DraftWindowPlan, *, valid_num_blocks: int) -> DraftWindowReadDescriptor:
        """Project the shared scalar plan onto the canonical batch-sorted table."""
        self._validate_plan(plan)
        return self._describe_interval(
            window_start=plan.window_start,
            window_visible_length=plan.window_visible_length,
            valid_num_blocks=valid_num_blocks,
        )

    def describe_frozen_step(
        self,
        proposal_plan: DraftWindowPlan,
        *,
        full_exact_end: int,
        valid_num_blocks: int,
    ) -> DraftWindowReadDescriptor:
        """Advance Eagle's actual end while retaining its proposal-frozen anchor."""
        if self.policy.normalized_method != "eagle3":
            raise ValueError("frozen per-step projection is defined only for Eagle3")
        self._validate_plan(proposal_plan)
        step_plan = proposal_plan.at_exact_end(full_exact_end)
        return self._describe_interval(
            window_start=step_plan.window_start,
            window_visible_length=step_plan.window_visible_length,
            valid_num_blocks=valid_num_blocks,
        )

    def _describe_interval(
        self,
        *,
        window_start: int,
        window_visible_length: int,
        valid_num_blocks: int,
    ) -> DraftWindowReadDescriptor:
        """Project a previously validated absolute interval onto FullSource."""
        _require_nonnegative_python_int(valid_num_blocks, "valid_num_blocks")
        span = self.group.table_column_span
        projection = project_draft_window_interval(
            window_start=window_start,
            full_exact_end=window_start + window_visible_length,
            block_span_tokens=span,
        )
        if projection.first_token_offset != 0:
            raise NotImplementedError(
                "MRV2 direct-prefix provider cannot consume a nonzero first-token "
                f"offset: gid={self.group.gid} "
                f"start={window_start} span={span} "
                f"offset={projection.first_token_offset}"
            )
        if projection.required_columns > self.capacity_columns:
            raise RuntimeError(
                "draft-window projection exceeds provider capacity: "
                f"required={projection.required_columns} "
                f"capacity={self.capacity_columns}"
            )
        source_row_width = self.full_source.input_block_table_row_width(self.group.gid)
        if valid_num_blocks > source_row_width:
            raise RuntimeError(
                "canonical valid block count exceeds source row width: "
                f"gid={self.group.gid} valid={valid_num_blocks} "
                f"row_width={source_row_width}"
            )
        if projection.source_end_column > valid_num_blocks:
            raise RuntimeError(
                "draft-window source shortage: "
                f"method={self.policy.normalized_method} "
                f"model={self.policy.target_model_type} "
                f"gid={self.group.gid} "
                f"start={projection.source_start_column} "
                f"required={projection.required_columns} "
                f"end={projection.source_end_column} valid={valid_num_blocks}"
            )
        return DraftWindowReadDescriptor(
            global_group_id=self.group.gid,
            source_start_column=projection.source_start_column,
            first_token_offset=projection.first_token_offset,
            required_columns=projection.required_columns,
            safe_extent_tokens=projection.safe_extent_tokens,
            valid_num_blocks=valid_num_blocks,
            source_row_width=source_row_width,
        )


def _normalize_method(method: str) -> str:
    return method.strip().lower()


def resolve_static_future_offset(
    *,
    normalized_method: str,
    configured_k: int,
    uses_dynamic_k: bool,
) -> int:
    """Return the Frozen method horizon offset, rejecting dynamic K."""
    if uses_dynamic_k:
        raise NotImplementedError("draft_window_size does not support dynamic speculative decoding")
    if normalized_method == "eagle3":
        return configured_k
    if normalized_method in {"dflash", "dspark"}:
        return 0
    raise NotImplementedError(
        f"no static draft-window future offset is defined for speculative method {normalized_method!r}"
    )


def validate_eligible_window_numbers(*, window_size: int, future_offset: int) -> None:
    """Apply the interim fail-closed numeric gate for eligible routes."""
    if window_size <= 0:
        raise ValueError("draft_window_size must be greater than 0 for an eligible MRV2 route")
    if future_offset < 0:
        raise ValueError("static future offset must be greater than or equal to 0")
    if window_size < future_offset:
        raise ValueError("draft_window_size must be at least the static future offset")


def build_full_draft_kv_source(
    *,
    group: DraftWindowGroupRoute,
    block_tables: Any,
) -> FullDraftKVSource:
    """Bind, but never copy or mutate, canonical full BlockTables tensors."""
    context = f"gid={group.gid} layout={group.status.value}"
    input_block_tables = tuple(block_tables.input_block_tables)
    logical_block_sizes = tuple(int(size) for size in block_tables.block_sizes)
    kernel_block_sizes = tuple(int(size) for size in block_tables.kernel_block_sizes)
    persistent_num_blocks = block_tables.num_blocks.gpu
    persistent_num_blocks_cpu = block_tables.num_blocks.cpu
    num_groups = len(input_block_tables)
    if persistent_num_blocks.ndim != 2:
        raise RuntimeError(f"{context}; reason=persistent num_blocks.gpu must be two-dimensional")
    if persistent_num_blocks_cpu.ndim != 2:
        raise RuntimeError(f"{context}; reason=persistent num_blocks.cpu must be two-dimensional")
    if persistent_num_blocks_cpu.device.type != "cpu":
        raise RuntimeError(f"{context}; reason=persistent num_blocks.cpu must live on CPU")
    if persistent_num_blocks_cpu.dtype != persistent_num_blocks.dtype:
        raise RuntimeError(f"{context}; reason=CPU/GPU persistent num_blocks dtype mismatch")
    if persistent_num_blocks_cpu.shape != persistent_num_blocks.shape:
        raise RuntimeError(f"{context}; reason=CPU/GPU persistent num_blocks shape mismatch")
    if not (len(logical_block_sizes) == len(kernel_block_sizes) == persistent_num_blocks.shape[0] == num_groups):
        raise RuntimeError(
            f"{context}; reason=canonical BlockTables group dimensions disagree: "
            f"tables={num_groups}, logical={len(logical_block_sizes)}, "
            f"kernel={len(kernel_block_sizes)}, "
            f"num_blocks={persistent_num_blocks.shape[0]}"
        )
    _validate_source_geometry(
        group=group,
        logical_block_sizes=logical_block_sizes,
        kernel_block_sizes=kernel_block_sizes,
        num_groups=num_groups,
    )
    for global_group_id, table in enumerate(input_block_tables):
        if table.ndim != 2:
            raise RuntimeError(f"{context}; reason=input_block_tables[{global_group_id}] must be two-dimensional")
        if table.shape[0] != persistent_num_blocks.shape[1]:
            raise RuntimeError(
                f"{context}; reason=input table {global_group_id} request capacity "
                f"{table.shape[0]} does not match persistent num_blocks "
                f"capacity {persistent_num_blocks.shape[1]}"
            )
    return FullDraftKVSource(
        logical_block_sizes=logical_block_sizes,
        kernel_block_sizes=kernel_block_sizes,
        _block_tables_owner=block_tables,
        _input_block_tables=input_block_tables,
        _persistent_num_blocks_cpu=persistent_num_blocks_cpu,
    )


def gather_valid_num_blocks_cpu(
    full_source: FullDraftKVSource,
    *,
    global_group_id: int,
    idx_mapping_np: np.ndarray,
    num_reqs: int,
    num_reqs_padded: int,
) -> torch.Tensor:
    """Gather valid counts from the canonical CPU owner in batch order.

    The accelerator view of ``UvaBackedTensor`` is a rotating transport and is
    not a stable freshness identity. P4 eager planning is host-side, so using
    the source-of-truth CPU owner avoids a device-to-host synchronization.
    """
    full_source.validate_owner_bindings()
    full_source._validate_global_group_id(global_group_id)
    if num_reqs < 0:
        raise ValueError("num_reqs must be greater than or equal to 0")
    if num_reqs_padded < num_reqs:
        raise ValueError("num_reqs_padded must be at least num_reqs")
    mapping = np.asarray(idx_mapping_np)
    if mapping.ndim != 1:
        raise ValueError("CPU idx_mapping must be one-dimensional")
    if not np.issubdtype(mapping.dtype, np.integer):
        raise TypeError("CPU idx_mapping must contain integer indices")
    if mapping.shape[0] < num_reqs:
        raise ValueError("num_reqs exceeds CPU idx_mapping capacity")
    persistent_counts = full_source._persistent_num_blocks_cpu[global_group_id]
    out = torch.zeros(num_reqs_padded, dtype=torch.int32, device="cpu")
    if num_reqs:
        indices = torch.from_numpy(mapping[:num_reqs].astype(np.int64, copy=False))
        torch.index_select(
            persistent_counts,
            0,
            indices,
            out=out[:num_reqs],
        )
    return out


def _validate_idx_mapping(idx_mapping: torch.Tensor) -> None:
    if idx_mapping.ndim != 1:
        raise ValueError("idx_mapping must be one-dimensional")
    if idx_mapping.dtype not in (torch.int32, torch.int64):
        raise TypeError("idx_mapping must use int32 or int64 indices")


def _shares_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Conservatively reject output views into any borrowed source storage."""
    if left.device != right.device:
        return False
    try:
        return left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
    except (AttributeError, RuntimeError):
        return left.data_ptr() == right.data_ptr()


def _validate_source_geometry(
    *,
    group: DraftWindowGroupRoute,
    logical_block_sizes: tuple[int, ...],
    kernel_block_sizes: tuple[int, ...],
    num_groups: int,
) -> None:
    context = f"gid={group.gid} layout={group.status.value}"
    global_group_id = group.gid
    if not (
        0 <= global_group_id < num_groups
        and global_group_id < len(logical_block_sizes)
        and global_group_id < len(kernel_block_sizes)
    ):
        raise RuntimeError(f"{context}; reason=inventory gid has no canonical source geometry")
    source_logical = logical_block_sizes[global_group_id]
    source_kernel = kernel_block_sizes[global_group_id]
    if source_logical != group.scheduler_block_size or source_kernel != group.table_column_span:
        raise RuntimeError(
            f"{context}; reason=source geometry logical={source_logical} "
            f"kernel={source_kernel} does not match inventory "
            f"logical={group.scheduler_block_size} kernel={group.table_column_span}"
        )


def _get_tensor_parallel_rank(parallel_config: Any) -> int:
    """Resolve process-local TP identity without inventing a rank.

    Unit/config-only callers may provide an explicit rank.  A live TP route
    must have an initialized vLLM TP group; failure to resolve it is a hard
    source/configuration error rather than a rank-zero fallback.
    """
    world_size = int(parallel_config.tensor_parallel_size)
    configured_rank = getattr(parallel_config, "tensor_parallel_rank", None)
    if configured_rank is not None:
        rank = int(configured_rank)
    elif world_size == 1:
        rank = 0
    else:
        try:
            from vllm.distributed import get_tp_group

            rank = int(get_tp_group().rank_in_group)
        except (AssertionError, AttributeError, RuntimeError) as error:
            raise RuntimeError("TP Draft Window route requires an initialized TP group rank") from error
    if rank < 0 or rank >= world_size:
        raise ValueError(f"TP rank {rank} is outside [0, {world_size})")
    return rank


def _build_draft_window_source_capability_for_context(
    *,
    policy: DraftWindowPolicyDecision,
    runtime_route: str,
    attn_groups: Sequence[Sequence[AttentionGroup]],
    block_tables: Any,
    kv_cache_config: Any,
    causal: bool | Mapping[int, bool],
    vllm_config: Any,
    dflash2_graph_contract: Any = None,
) -> DraftWindowSourceCapability:
    """Build source geometry after the caller validates its real Graph executor.

    Executor features do not turn a dense Draft cache into a sharded provider.
    Keep the actual context in the owner identity and check the source layout
    directly, without constructing an artificial eager capability decision.
    """
    if policy.state is not DraftWindowPolicyState.ELIGIBLE_ON or policy.effective_window_size is None:
        raise ValueError("Graph source requires an eligible Window policy")
    runtime_route = _normalize_method(runtime_route)
    expected_method = "dflash" if runtime_route == "dflash2" else runtime_route
    if runtime_route not in {"eagle3", "dflash", "dflash2", "dspark"} or expected_method != policy.normalized_method:
        raise ValueError("Graph source route differs from the actual method policy")
    native_windows: dict[int, int | None] = {}
    if runtime_route == "dflash2":
        from vllm_ascend.worker.v2.spec_decode.draft_window_dflash2_graph import DFlash2GraphContract

        if not isinstance(dflash2_graph_contract, DFlash2GraphContract):
            raise TypeError("DFlash2 Graph source requires its actual typed method contract")
        native_windows = dflash2_graph_contract.validate_source(
            attn_groups, block_tables, causal, policy.effective_window_size, dflash2_graph_contract.query_width
        )
    elif dflash2_graph_contract is not None:
        raise ValueError("DFlash2 source contract belongs to a different method")
    groups = inventory_draft_window_groups(
        attn_groups=attn_groups,
        kv_cache_config=kv_cache_config,
        block_tables=block_tables,
        parallel_config=vllm_config.parallel_config,
        full_retention_dflash2=dflash2_graph_contract is not None,
    )
    if len(groups) != 1:
        raise NotImplementedError(f"Draft Graph requires one executable consumer group, found {len(groups)}")
    group = groups[0]
    if runtime_route == "dflash2" and (
        group.gid not in native_windows or group.native_window != native_windows[group.gid]
    ):
        raise RuntimeError("backend inventory and DFlash2 native reader contract disagree")
    uses_dynamic_k = bool(vllm_config.speculative_config.uses_dynamic_speculative_decoding())
    if uses_dynamic_k:
        raise NotImplementedError("Draft Window Graph source requires static speculative decoding")
    full_source = build_full_draft_kv_source(
        group=group,
        block_tables=block_tables,
    )
    return DraftWindowSourceCapability(
        policy=policy,
        runtime_route=runtime_route,
        group=group,
        full_source=full_source,
        expected_future_offset=resolve_static_future_offset(
            normalized_method=policy.normalized_method,
            configured_k=int(vllm_config.speculative_config.num_speculative_tokens),
            uses_dynamic_k=uses_dynamic_k,
        ),
        parallel_config=vllm_config.parallel_config,
        tensor_parallel_rank=_get_tensor_parallel_rank(vllm_config.parallel_config),
        # Mixed/chunked rows may retain a full-resident row beside windowed
        # decode rows in the same stable read bank.
        workspace_capacity_columns=full_source.input_block_table_row_width(group.gid),
    )


class DraftWindowExecutionPhase(Enum):
    """Typed identity for a future live/capture binding point."""

    PROPOSAL = "proposal"
    STEP = "step"


@dataclass(frozen=True, slots=True, eq=False)
class DraftWindowGenerationIdentity:
    """A fresh consumer handle retaining its real proposal and selected shape.

    The proposal already freezes request order, row phases and mapping epoch.
    This handle adds only the current consumer step, read projection and bank
    geometry. Equal values from another runtime cannot impersonate the handle.
    """

    proposal: "DraftWindowProposalContext"
    padded_request_count: int
    read_descriptors: tuple[DraftWindowReadDescriptor, ...]
    idx_mapping_ptr: int
    phase: DraftWindowExecutionPhase
    step: int

    def __post_init__(self) -> None:
        from vllm_ascend.worker.v2.spec_decode.draft_window_eager import DraftWindowProposalContext

        if not isinstance(self.proposal, DraftWindowProposalContext):
            raise TypeError("generation requires the actual Draft Window proposal context")
        _require_nonnegative_python_int(self.padded_request_count, "padded request count")
        _require_nonnegative_python_int(self.step, "step")
        actual = len(self.proposal.ordered_request_ids)
        if self.padded_request_count < actual:
            raise ValueError("padded request count must be at least the actual request count")
        if len(self.read_descriptors) != actual:
            raise ValueError("read descriptors must match the actual request count")
        if not isinstance(self.phase, DraftWindowExecutionPhase):
            raise TypeError("phase must be a DraftWindowExecutionPhase")


class DraftWindowWorkspace:
    """Fixed-capacity, stable-address backing for one serialized read lane."""

    def __init__(
        self,
        *,
        max_num_reqs: int,
        table_capacity_columns: int,
        table_template: torch.Tensor,
        valid_count_template: torch.Tensor,
    ) -> None:
        if max_num_reqs <= 0:
            raise ValueError("workspace request capacity must be greater than 0")
        if table_capacity_columns <= 0:
            raise ValueError("workspace table capacity must be greater than 0")
        if table_template.ndim != 2 or valid_count_template.ndim != 1:
            raise ValueError("Draft workspace templates require a table and one-dimensional valid counts")
        if table_template.dtype != torch.int32 or valid_count_template.dtype != torch.int32:
            raise TypeError("Draft workspace tables and valid counts must use int32")
        if table_template.device != valid_count_template.device:
            raise ValueError("Draft workspace table and valid-count templates must share a device")
        self.max_num_reqs = max_num_reqs
        self.table_capacity_columns = table_capacity_columns
        self.window_block_table = table_template.new_zeros((max_num_reqs, table_capacity_columns))
        self.valid_num_blocks = valid_count_template.new_zeros(max_num_reqs)
        self.device_window_exact = torch.zeros(
            max_num_reqs,
            dtype=torch.int32,
            device=table_template.device,
        )
        self.cpu_window_exact = torch.zeros(max_num_reqs, dtype=torch.int32)
        self.cpu_window_upper = torch.zeros(max_num_reqs, dtype=torch.int32)
        self._bound_identity: DraftWindowGenerationIdentity | None = None
        self._active_table_columns: tuple[int, ...] = ()
        self._needs_clear = False

    @property
    def bound_identity(self) -> DraftWindowGenerationIdentity | None:
        return self._bound_identity

    def materialize(
        self,
        *,
        identity: DraftWindowGenerationIdentity,
        full_source: FullDraftKVSource,
        global_group_id: int,
        cpu_window_exact: torch.Tensor,
        cpu_window_upper: torch.Tensor,
        cpu_valid_num_blocks: torch.Tensor,
        preserve_on_failure: bool = False,
    ) -> None:
        """Rewrite all active and inert state from the canonical full source."""
        if self._bound_identity is not None:
            raise RuntimeError("workspace is already bound to an active generation")
        actual = len(identity.proposal.ordered_request_ids)
        padded = identity.padded_request_count
        if padded > self.max_num_reqs:
            raise RuntimeError(f"draft-window request capacity exceeded: padded={padded} capacity={self.max_num_reqs}")
        _validate_window_length_input(
            cpu_window_exact,
            name="CPU Window exact",
            padded_request_count=padded,
            expected_device=torch.device("cpu"),
        )
        _validate_window_length_input(
            cpu_valid_num_blocks,
            name="CPU valid block counts",
            padded_request_count=padded,
            expected_device=torch.device("cpu"),
        )
        _validate_window_length_input(
            cpu_window_upper,
            name="CPU Window upper",
            padded_request_count=padded,
            expected_device=torch.device("cpu"),
        )
        workspace_tensors = (
            self.window_block_table,
            self.valid_num_blocks,
            self.device_window_exact,
            self.cpu_window_exact,
            self.cpu_window_upper,
        )
        if any(
            _shares_storage(supplied, backing)
            for supplied in tuple(
                tensor
                for tensor in (
                    cpu_window_exact,
                    cpu_window_upper,
                    cpu_valid_num_blocks,
                )
                if tensor is not None
            )
            for backing in workspace_tensors
        ):
            raise ValueError("Window length inputs must not alias workspace backing")
        if actual and bool(torch.any(cpu_window_exact[:actual] < 0)):
            raise ValueError("Window exact lengths must be greater than or equal to 0")
        if actual and bool(torch.any(cpu_window_upper[:actual] < cpu_window_exact[:actual])):
            raise ValueError("CPU Window upper must be greater than or equal to exact")

        full_source.validate_owner_bindings()
        full_source._validate_global_group_id(global_group_id)
        source_table = full_source._input_block_tables[global_group_id]
        if source_table.device != self.window_block_table.device:
            raise ValueError("canonical source table and Window workspace must share a device")
        if source_table.dtype != self.window_block_table.dtype:
            raise TypeError("canonical source table and Window workspace must share a dtype")

        try:
            if self._needs_clear:
                self._zero_all()
            if actual:
                self.valid_num_blocks[:actual].copy_(cpu_valid_num_blocks[:actual])
            active_columns: list[int] = []
            for row, descriptor in enumerate(identity.read_descriptors):
                if descriptor.global_group_id != global_group_id:
                    raise ValueError("read descriptor global gid does not match the runtime source")
                if (
                    descriptor.source_start_column < 0
                    or descriptor.required_columns < 0
                    or descriptor.safe_extent_tokens < 0
                    or descriptor.valid_num_blocks < 0
                ):
                    raise ValueError("read descriptor extents must be greater than or equal to 0")
                if descriptor.first_token_offset != 0:
                    raise NotImplementedError("P3 supports only direct-prefix descriptors with zero token offset")
                if descriptor.source_row_width != source_table.shape[1]:
                    raise RuntimeError("read descriptor source row width is stale")
                if descriptor.required_columns > self.table_capacity_columns:
                    raise RuntimeError(
                        "draft-window table capacity exceeded: "
                        f"required={descriptor.required_columns} "
                        f"capacity={self.table_capacity_columns}"
                    )
                gathered_valid = int(cpu_valid_num_blocks[row].item())
                if gathered_valid > source_table.shape[1]:
                    raise RuntimeError("current valid block count exceeds the canonical source row width")
                if descriptor.valid_num_blocks != gathered_valid:
                    raise RuntimeError(
                        "read descriptor valid count is stale: "
                        f"row={row} descriptor={descriptor.valid_num_blocks} "
                        f"current={gathered_valid}"
                    )
                if descriptor.source_end_column > gathered_valid:
                    raise RuntimeError(
                        "draft-window source shortage during materialization: "
                        f"row={row} end={descriptor.source_end_column} "
                        f"valid={gathered_valid}"
                    )
                required = descriptor.required_columns
                if required:
                    self.window_block_table[row, :required].copy_(
                        source_table[
                            row,
                            descriptor.source_start_column : descriptor.source_end_column,
                        ]
                    )
                active_columns.append(required)
            if actual:
                self.device_window_exact[:actual].copy_(cpu_window_exact[:actual])
                self.cpu_window_exact[:actual].copy_(cpu_window_exact[:actual])
                self.cpu_window_upper[:actual].copy_(cpu_window_upper[:actual])
            self._active_table_columns = tuple(active_columns)
            self._bound_identity = identity
        except BaseException:
            if not preserve_on_failure:
                self._zero_all()
            raise

    def build_inputs(
        self, owner: "DraftWindowRuntimeState", identity: DraftWindowGenerationIdentity
    ) -> "DraftMetadataInputs":
        return DraftMetadataInputs(identity=identity, workspace=self, _owner=owner)

    def assert_bound(self, expected_identity: DraftWindowGenerationIdentity) -> None:
        if self._bound_identity != expected_identity:
            raise RuntimeError("workspace generation identity does not match the expected generation")

    def reset(self, expected_identity: DraftWindowGenerationIdentity, *, clear_buffers: bool = True) -> None:
        if self._bound_identity is not None:
            self.assert_bound(expected_identity)
        if clear_buffers:
            self._zero_all()
        else:
            self._bound_identity = None
            self._active_table_columns = ()
            self._needs_clear = True

    def _zero_all(self) -> None:
        self.window_block_table.zero_()
        self.valid_num_blocks.zero_()
        self.device_window_exact.zero_()
        self.cpu_window_exact.zero_()
        self.cpu_window_upper.zero_()
        self._active_table_columns = ()
        self._bound_identity = None
        self._needs_clear = False


@dataclass(frozen=True, slots=True, eq=False)
class DraftMetadataInputs:
    """One generation handle retaining its actual read bank and runtime owner.

    The bank owns all read tensors. A new handle on every materialization
    rejects stale references even when the next generation reuses that bank.
    Positions, query layout, write slots and Target metadata remain separate.
    """

    identity: DraftWindowGenerationIdentity
    workspace: DraftWindowWorkspace = field(repr=False, compare=False)
    _owner: "DraftWindowRuntimeState" = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        self.workspace.assert_bound(self.identity)

    @property
    def retired(self) -> bool:
        return self._owner._active_inputs is not self

    def validate_for(self, expected_identity: DraftWindowGenerationIdentity) -> None:
        if self.identity != expected_identity:
            raise RuntimeError("typed inputs generation identity does not match the expected generation")
        if self.retired:
            raise RuntimeError("cannot use retired Draft inputs")
        self.workspace.assert_bound(expected_identity)


class DraftWindowRuntimeState:
    """The single speculator-owned P3 lifecycle and retirement owner."""

    def __init__(
        self,
        *,
        capability: DraftWindowSourceCapability,
        workspace: DraftWindowWorkspace | None = None,
    ) -> None:
        self.capability = capability
        self._workspace = workspace
        self._source = capability.full_source
        self._metadata: Any = None
        self._current_workspace: DraftWindowWorkspace | None = None
        self._quarantine_on_failure = False
        self._active_inputs: DraftMetadataInputs | None = None
        self._last_mapping_epoch = -1
        self._last_retired_identity: DraftWindowGenerationIdentity | None = None
        self._quarantined_identity: DraftWindowGenerationIdentity | None = None

    @property
    def workspace(self) -> DraftWindowWorkspace:
        # Non-FULL consumers need a temporary read buffer; FULL Graph writes
        # directly to its capture bank and never allocates this scratch.
        if self._workspace is None:
            self._workspace = self.new_workspace()
        return self._workspace

    def new_workspace(self, num_reqs: int | None = None) -> DraftWindowWorkspace:
        capability = self.capability
        gid = capability.group.gid
        table = capability.full_source._input_block_tables[gid]
        return DraftWindowWorkspace(
            max_num_reqs=table.shape[0] if num_reqs is None else num_reqs,
            table_capacity_columns=capability.workspace_capacity_columns or capability.capacity_columns,
            table_template=table,
            valid_count_template=capability.full_source._block_tables_owner.num_blocks.gpu[gid],
        )

    @property
    def quarantined_identity(self) -> DraftWindowGenerationIdentity | None:
        return self._quarantined_identity

    def quarantine_generation(self, expected_identity: DraftWindowGenerationIdentity) -> None:
        """Keep possibly submitted Draft backing intact until process isolation.

        Unknown native partial submission cannot justify another device write
        or a completion claim. The live input handle intentionally blocks reuse.
        """
        inputs = self._active_inputs
        if inputs is None:
            raise RuntimeError("cannot quarantine a generation without live inputs")
        inputs.validate_for(expected_identity)
        self._metadata = None
        self._quarantined_identity = expected_identity

    def materialize(
        self,
        *,
        proposal: "DraftWindowProposalContext",
        idx_mapping: torch.Tensor,
        read_descriptors: Sequence[DraftWindowReadDescriptor],
        cpu_window_exact: torch.Tensor,
        cpu_window_upper: torch.Tensor,
        cpu_valid_num_blocks: torch.Tensor,
        num_reqs_padded: int,
        phase: DraftWindowExecutionPhase,
        step: int,
        quarantine_on_failure: bool = False,
        output_workspace: DraftWindowWorkspace | None = None,
    ) -> DraftMetadataInputs:
        """Materialize one consumer under the caller's actual proposal owner."""
        if self._quarantined_identity is not None:
            raise RuntimeError("draft-window backing is quarantined after uncertain Graph submission")
        _validate_idx_mapping(idx_mapping)
        identity = DraftWindowGenerationIdentity(
            proposal=proposal,
            padded_request_count=num_reqs_padded,
            read_descriptors=tuple(read_descriptors),
            idx_mapping_ptr=idx_mapping.data_ptr(),
            phase=phase,
            step=step,
        )
        if proposal.mapping_epoch <= self._last_mapping_epoch:
            raise ValueError(
                f"mapping epoch must strictly increase: current={proposal.mapping_epoch} "
                f"previous={self._last_mapping_epoch}"
            )
        self._last_retired_identity = None
        return self._materialize_identity(
            identity=identity,
            cpu_window_exact=cpu_window_exact,
            cpu_window_upper=cpu_window_upper,
            cpu_valid_num_blocks=cpu_valid_num_blocks,
            quarantine_on_failure=quarantine_on_failure,
            output_workspace=output_workspace,
        )

    def materialize_serialized_step(
        self,
        *,
        previous_identity: DraftWindowGenerationIdentity,
        proposal: "DraftWindowProposalContext",
        idx_mapping: torch.Tensor,
        read_descriptors: Sequence[DraftWindowReadDescriptor],
        cpu_window_exact: torch.Tensor,
        cpu_window_upper: torch.Tensor,
        cpu_valid_num_blocks: torch.Tensor,
        num_reqs_padded: int,
        step: int,
        quarantine_on_failure: bool = False,
        output_workspace: DraftWindowWorkspace | None = None,
    ) -> DraftMetadataInputs:
        """Continue the same real proposal, only after its preceding reader retires."""
        if self._active_inputs is not None:
            raise RuntimeError("cannot refresh a Draft Window while the previous reader is active")
        if self._last_retired_identity is not previous_identity:
            raise RuntimeError("serialized step does not continue the most recently retired generation")
        if proposal is not previous_identity.proposal or proposal.mapping_epoch != self._last_mapping_epoch:
            raise RuntimeError("serialized step proposal or request mapping changed")
        if self.capability.policy.normalized_method != "eagle3":
            raise RuntimeError("serialized per-step continuation is supported only for Eagle3")
        if step != previous_identity.step + 1:
            raise RuntimeError(
                "serialized Draft Window steps must strictly advance by one: "
                f"previous={previous_identity.step} current={step}"
            )
        _validate_idx_mapping(idx_mapping)
        if (
            self.capability.full_source is not self._source
            or idx_mapping.data_ptr() != previous_identity.idx_mapping_ptr
        ):
            raise RuntimeError("serialized step FullSource or request mapping changed within one proposal")
        identity = DraftWindowGenerationIdentity(
            proposal=proposal,
            padded_request_count=num_reqs_padded,
            read_descriptors=tuple(read_descriptors),
            idx_mapping_ptr=idx_mapping.data_ptr(),
            phase=DraftWindowExecutionPhase.STEP,
            step=step,
        )
        workspace = self.workspace if output_workspace is None else output_workspace
        if identity.padded_request_count > workspace.max_num_reqs:
            raise RuntimeError("serialized step padded request capacity exceeded")
        self._last_retired_identity = None
        return self._materialize_identity(
            identity=identity,
            cpu_window_exact=cpu_window_exact,
            cpu_window_upper=cpu_window_upper,
            cpu_valid_num_blocks=cpu_valid_num_blocks,
            quarantine_on_failure=quarantine_on_failure,
            output_workspace=output_workspace,
        )

    def _materialize_identity(
        self,
        *,
        identity: DraftWindowGenerationIdentity,
        cpu_window_exact: torch.Tensor,
        cpu_window_upper: torch.Tensor,
        cpu_valid_num_blocks: torch.Tensor,
        quarantine_on_failure: bool = False,
        output_workspace: DraftWindowWorkspace | None = None,
    ) -> DraftMetadataInputs:
        """Acquire and populate the one serialized backing for an identity."""
        if self._current_workspace is not None:
            raise RuntimeError("draft-window runtime still owns an active read bank")
        if self.capability.full_source is not self._source:
            raise RuntimeError("draft-window canonical source changed after runtime initialization")
        workspace = self.workspace if output_workspace is None else output_workspace
        self._current_workspace = workspace
        self._quarantine_on_failure = quarantine_on_failure
        self._last_mapping_epoch = max(self._last_mapping_epoch, identity.proposal.mapping_epoch)
        try:
            workspace.materialize(
                identity=identity,
                full_source=self._source,
                global_group_id=self.capability.group.gid,
                cpu_window_exact=cpu_window_exact,
                cpu_window_upper=cpu_window_upper,
                cpu_valid_num_blocks=cpu_valid_num_blocks,
                preserve_on_failure=quarantine_on_failure,
            )
            inputs = workspace.build_inputs(self, identity)
            self._active_inputs = inputs
            return inputs
        except BaseException:
            if quarantine_on_failure:
                # A native write may have been submitted before raising, even
                # though no typed inputs exist yet. Keep the workspace and
                # backing; only Host state may be changed before child isolation.
                self._quarantined_identity = identity
                raise
            self._finish_generation(identity)
            self._active_inputs = None
            raise

    def publish_metadata(
        self,
        inputs: DraftMetadataInputs,
        metadata: Any,
        *,
        target_metadata: Any,
    ) -> None:
        try:
            self._require_active_inputs(inputs)
            if self._metadata is not None:
                raise RuntimeError("cannot overwrite active draft metadata")
            from vllm_ascend.worker.v2.spec_decode.draft_window_graph_metadata import validate_draft_metadata_isolation

            validate_draft_metadata_isolation(metadata, target_metadata=target_metadata)
            self._metadata = metadata
        except BaseException:
            if self._active_inputs is inputs:
                self._handle_active_failure(inputs)
            raise

    def read_metadata(self, expected_identity: DraftWindowGenerationIdentity) -> Any:
        if self._active_inputs is None or self._metadata is None:
            raise RuntimeError("no active draft metadata")
        self._active_inputs.validate_for(expected_identity)
        return self._metadata

    def _handle_active_failure(self, inputs: DraftMetadataInputs) -> None:
        if self._quarantine_on_failure:
            self.quarantine_generation(inputs.identity)
        else:
            self._abort_active(inputs.identity)

    def retire_generation(self, expected_identity: DraftWindowGenerationIdentity) -> None:
        if self._quarantined_identity is not None:
            raise RuntimeError("cannot retire quarantined draft-window backing")
        inputs = self._active_inputs
        if inputs is None:
            raise RuntimeError("cannot use retired Draft inputs")
        inputs.validate_for(expected_identity)
        self._finish_generation(expected_identity)
        self._last_retired_identity = expected_identity

    def abort_generation(self, expected_identity: DraftWindowGenerationIdentity) -> None:
        if self._quarantined_identity is not None:
            raise RuntimeError("cannot abort quarantined draft-window backing")
        inputs = self._active_inputs
        if inputs is None:
            raise RuntimeError("cannot use retired Draft inputs")
        inputs.validate_for(expected_identity)
        self._abort_active(expected_identity)

    def _abort_active(self, identity: DraftWindowGenerationIdentity) -> None:
        if self._quarantined_identity is not None:
            raise RuntimeError("cannot clean up quarantined draft-window backing")
        inputs = self._active_inputs
        if inputs is None:
            return
        inputs.validate_for(identity)
        self._finish_generation(identity)
        self._last_retired_identity = None

    def _finish_generation(self, identity: DraftWindowGenerationIdentity) -> None:
        # Graph bank lifetime belongs to capture, not this Host generation.
        # Clear only at the next same-stream prepare, after the existing tail
        # dependency. Non-FULL scratch follows its serialized current stream.
        workspace = self._current_workspace
        assert workspace is not None
        workspace.reset(identity, clear_buffers=workspace is self._workspace)
        self._metadata = None
        self._active_inputs = None
        self._current_workspace = None

    def _require_active_inputs(self, inputs: DraftMetadataInputs) -> None:
        if self._active_inputs is not inputs:
            raise RuntimeError("typed inputs are not the active runtime generation")
        inputs.validate_for(inputs.identity)


def build_draft_window_runtime_state(
    capability: DraftWindowSourceCapability,
) -> DraftWindowRuntimeState:
    """Allocate P3 state only after P2 has closed all capability gates."""
    group = capability.group
    capability.full_source.validate_owner_bindings()
    source_table = capability.full_source._input_block_tables[group.gid]
    persistent_counts = capability.full_source._block_tables_owner.num_blocks.gpu[group.gid]
    if source_table.dtype != torch.int32 or persistent_counts.dtype != torch.int32:
        raise TypeError("P3 direct-prefix source tables and valid counts must use int32")
    if source_table.device != persistent_counts.device:
        raise ValueError("P3 source table and valid counts must share a device")
    table_capacity_columns = (
        capability.capacity_columns
        if capability.workspace_capacity_columns is None
        else capability.workspace_capacity_columns
    )
    if table_capacity_columns < capability.capacity_columns:
        raise ValueError("configured workspace capacity is smaller than the provider capacity")
    if table_capacity_columns > source_table.shape[1]:
        raise ValueError("configured workspace capacity exceeds the canonical source row width")
    return DraftWindowRuntimeState(
        capability=capability,
        workspace=None,
    )


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


def _require_nonnegative_python_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be a Python int")
    if value < 0:
        raise ValueError(f"{name} must be greater than or equal to 0")
