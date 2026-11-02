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
    project_draft_window_interval,
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
class DraftWindowSourceCapability:
    """Canonical read projection; it never materializes a window table."""

    policy: DraftWindowPolicyDecision
    runtime_route: str
    group: DraftWindowGroupRoute
    full_source: FullDraftKVSource
    expected_future_offset: int
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
                f"Draft Graph source cannot read gid={self.group.gid} layout={self.group.status.value}"
            )
        _require_nonnegative_python_int(self.expected_future_offset, "expected future offset")
        object.__setattr__(self, "window_size", window)
        span = self.group.table_column_span
        capacity_tokens = window_capacity_tokens(window, span)
        object.__setattr__(self, "capacity_columns", (capacity_tokens + span - 1) // span)


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
    runtime_route = runtime_route.strip().lower()
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
        # Mixed/chunked rows may retain a full-resident row beside windowed
        # decode rows in the same stable read bank.
        workspace_capacity_columns=full_source.input_block_table_row_width(group.gid),
    )


@dataclass(frozen=True, slots=True, eq=False)
class DraftWindowGenerationIdentity:
    """A fresh consumer handle retaining its real proposal and selected shape.

    The proposal already freezes request order, row phases and mapping epoch.
    This handle adds only the current consumer step and padded bank shape.
    Equal values from another runtime cannot impersonate the handle.
    """

    proposal: "DraftWindowProposalContext"
    padded_request_count: int
    idx_mapping_ptr: int
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


class DraftWindowWorkspace:
    """Fixed-capacity, stable-address backing for one serialized read lane."""

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
        self.max_num_reqs = max_num_reqs
        self.table_capacity_columns = table_capacity_columns
        self.window_block_table = table_template.new_zeros((max_num_reqs, table_capacity_columns))
        self.device_window_exact = torch.zeros(
            max_num_reqs,
            dtype=torch.int32,
            device=table_template.device,
        )
        self.cpu_window_exact = torch.zeros(max_num_reqs, dtype=torch.int32)
        self._bound_identity: DraftWindowGenerationIdentity | None = None
        self._needs_clear = False

    @property
    def bound_identity(self) -> DraftWindowGenerationIdentity | None:
        return self._bound_identity

    def materialize(
        self,
        *,
        identity: DraftWindowGenerationIdentity,
        source: DraftWindowSourceCapability,
        cpu_full_exact: torch.Tensor,
        cpu_valid_num_blocks: torch.Tensor,
        frozen_row_plans: tuple[DraftWindowPlan, ...] | None,
    ) -> None:
        """Rewrite all active and inert state from the canonical full source."""
        if self._bound_identity is not None:
            raise RuntimeError("workspace is already bound to an active generation")
        actual = len(identity.proposal.ordered_request_ids)
        padded = identity.padded_request_count
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

        full_source = source.full_source
        global_group_id = source.group.gid
        full_source.validate_owner_bindings()
        full_source._validate_global_group_id(global_group_id)
        source_table = full_source._input_block_tables[global_group_id]
        if source_table.device != self.window_block_table.device:
            raise ValueError("canonical source table and Window workspace must share a device")
        if source_table.dtype != self.window_block_table.dtype:
            raise TypeError("canonical source table and Window workspace must share a dtype")

        if self._needs_clear:
            self._zero_all()
        span = source.group.table_column_span
        for row, full_read in enumerate(identity.proposal.phase_plan.full_read_rows):
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
                projection = project_draft_window_interval(
                    window_start=plan.window_start, full_exact_end=exact, block_span_tokens=span
                )
                if projection.first_token_offset:
                    raise NotImplementedError("Draft Graph cannot read an intra-block Window start")
                start, required = projection.source_start_column, projection.required_columns
                visible = exact - plan.window_start
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
        self._bound_identity = identity

    def assert_bound(self, expected_identity: DraftWindowGenerationIdentity) -> None:
        if self._bound_identity != expected_identity:
            raise RuntimeError("workspace generation identity does not match the expected generation")

    def reset(self, expected_identity: DraftWindowGenerationIdentity) -> None:
        self.assert_bound(expected_identity)
        self._bound_identity = None
        self._needs_clear = True

    def _zero_all(self) -> None:
        self.window_block_table.zero_()
        self.device_window_exact.zero_()
        self.cpu_window_exact.zero_()
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
    _owner: "DraftWindowGraphBindingProvider" = field(repr=False, compare=False)
    metadata: Mapping[str, Any] | None = field(init=False, default=None, repr=False, compare=False)

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
        if self._owner.quarantined_identity is not None:
            raise RuntimeError("cannot use quarantined Draft inputs")
        self.workspace.assert_bound(expected_identity)

    def publish(self, metadata: Mapping[str, Any], *, target_metadata: Any) -> None:
        """Publish once on this actual generation, without a second owner store."""
        try:
            self.validate_for(self.identity)
            if self.metadata is not None:
                raise RuntimeError("cannot overwrite active draft metadata")
            from vllm_ascend.worker.v2.spec_decode.draft_window_graph_metadata import validate_draft_metadata_isolation

            validate_draft_metadata_isolation(metadata, target_metadata=target_metadata)
            object.__setattr__(self, "metadata", metadata)
        except BaseException:
            if self._owner._active_inputs is self and self._owner.quarantined_identity is None:
                self._owner.quarantine_generation(self.identity)
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


def _require_nonnegative_python_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be a Python int")
    if value < 0:
        raise ValueError(f"{name} must be greater than or equal to 0")
