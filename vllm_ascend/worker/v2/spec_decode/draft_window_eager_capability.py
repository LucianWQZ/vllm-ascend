# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Exact route identity retained by a Draft read-buffer owner."""

from dataclasses import dataclass, replace

from vllm_ascend.device.hardware_profile import AttentionBackendFamily
from vllm_ascend.worker.v2.spec_decode.draft_window import DraftWindowInventory, DraftWindowRuntimeContext


@dataclass(frozen=True, slots=True)
class P7EagerTopologyFingerprint:
    """Immutable route/topology identity for an eager Draft read carrier."""

    normalized_method: str
    runtime_route: str
    attention_backend_family: AttentionBackendFamily
    backend_impls: tuple[str, ...]
    builder_impls: tuple[str, ...]
    global_group_ids: tuple[int, ...]
    layouts: tuple[str, ...]
    block_geometry: tuple[tuple[int, int, int], ...]
    tensor_parallel_size: int
    data_parallel_size: int
    pipeline_parallel_size: int
    prefill_context_parallel_size: int
    decode_context_parallel_size: int
    sequence_parallel: bool
    dsa_context_parallel: bool
    flashcomm: bool
    graph_mode: str
    async_scheduling: bool
    prefix_caching: bool
    provider_version: int
    execution_lane: int
    tensor_parallel_rank: int = 0
    parallel_read_semantics: str = "replicated"

    def __post_init__(self) -> None:
        if not self.normalized_method or self.normalized_method != self.normalized_method.strip().lower():
            raise ValueError("topology method must be a non-empty normalized name")
        if not self.runtime_route or self.runtime_route != self.runtime_route.strip().lower():
            raise ValueError("topology runtime route must be a non-empty normalized name")
        if not isinstance(self.attention_backend_family, AttentionBackendFamily):
            raise TypeError("topology attention backend family must be typed")
        if self.provider_version <= 0:
            raise ValueError("topology provider version must be greater than zero")
        if self.execution_lane < 0:
            raise ValueError("topology execution lane must be nonnegative")
        topology_sizes = (
            self.tensor_parallel_size,
            self.data_parallel_size,
            self.pipeline_parallel_size,
            self.prefill_context_parallel_size,
            self.decode_context_parallel_size,
        )
        if any(size <= 0 for size in topology_sizes):
            raise ValueError("topology world sizes must be greater than zero")
        group_count = len(self.global_group_ids)
        if not (
            len(self.backend_impls)
            == len(self.builder_impls)
            == len(self.layouts)
            == len(self.block_geometry)
            == group_count
        ):
            raise ValueError("topology group fingerprints must have equal lengths")
        if any(gid < 0 for gid in self.global_group_ids):
            raise ValueError("topology global gids must be nonnegative")
        if not 0 <= self.tensor_parallel_rank < self.tensor_parallel_size:
            raise ValueError("topology TP rank must be within the TP world")
        if self.parallel_read_semantics not in {
            "replicated",
            "dcp_interleaved",
            "dsa_cp_derived",
        }:
            raise ValueError("topology parallel read semantics are not recognized")

    def for_execution_lane(
        self,
        execution_lane: int,
        *,
        async_scheduling: bool | None = None,
    ) -> "P7EagerTopologyFingerprint":
        """Return an explicit lane variant; never mutate the route identity."""
        if execution_lane < 0:
            raise ValueError("topology execution lane must be nonnegative")
        if async_scheduling is None:
            return replace(self, execution_lane=execution_lane)
        return replace(
            self,
            execution_lane=execution_lane,
            async_scheduling=async_scheduling,
        )


def _qualified_type_name(value: type[object]) -> str:
    return f"{value.__module__}.{value.__qualname__}"


def _build_topology_fingerprint(
    *,
    inventory: DraftWindowInventory,
    runtime_context: DraftWindowRuntimeContext,
    provider_version: int,
    execution_lane: int,
) -> P7EagerTopologyFingerprint:
    groups = inventory.groups
    return P7EagerTopologyFingerprint(
        normalized_method=inventory.method,
        runtime_route=inventory.runtime_route,
        attention_backend_family=runtime_context.attention_backend_family,
        backend_impls=tuple(_qualified_type_name(group.backend_class) for group in groups),
        builder_impls=tuple(_qualified_type_name(group.builder_class) for group in groups),
        global_group_ids=tuple(group.global_group_id for group in groups),
        layouts=tuple(group.layout_category.value for group in groups),
        block_geometry=tuple(
            (
                group.global_group_id,
                group.logical_block_size,
                group.kernel_block_size,
            )
            for group in groups
        ),
        tensor_parallel_size=runtime_context.tensor_parallel_size,
        data_parallel_size=runtime_context.data_parallel_size,
        pipeline_parallel_size=runtime_context.pipeline_parallel_size,
        prefill_context_parallel_size=runtime_context.prefill_context_parallel_size,
        decode_context_parallel_size=runtime_context.decode_context_parallel_size,
        sequence_parallel=runtime_context.sequence_parallel,
        dsa_context_parallel=runtime_context.dsa_context_parallel,
        flashcomm=runtime_context.flashcomm,
        graph_mode=runtime_context.graph_mode,
        async_scheduling=runtime_context.async_scheduling,
        prefix_caching=runtime_context.prefix_caching,
        provider_version=provider_version,
        execution_lane=execution_lane,
        tensor_parallel_rank=runtime_context.tensor_parallel_rank,
        parallel_read_semantics=(
            "dsa_cp_derived"
            if runtime_context.dsa_context_parallel
            else "dcp_interleaved"
            if runtime_context.decode_context_parallel_size > 1
            else "replicated"
        ),
    )
