# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inspect actual draft consumers before selecting a read-view provider.

Resolve every real consumer after ``set_attn`` without owning tensors.
Unknown or incompatible consumers fail before proposal work; successful
selection describes an addressing contract, not device qualification.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast


class DraftWindowProviderStatus(str, Enum):
    DIRECT_DENSE = "DIRECT_DENSE"
    NATIVE_SWA = "NATIVE_SWA"
    DIRECT_MLA = "DIRECT_MLA"


@dataclass(frozen=True)
class DraftWindowBackendTypes:
    """Concrete runtime classes; injected explicitly in CPU contract tests."""

    dense_backend: type
    dense_builder: type
    full_attention_spec: type
    uniform_spec: type[Any]
    storage_block_size: Callable[[Any], int]
    compression_ratio: Callable[[Any], int]
    mla_backend: type | None = None
    mla_builder: type | None = None
    mla_spec: type | None = None
    sliding_window_spec: type | None = None

    @classmethod
    def load(cls) -> "DraftWindowBackendTypes":
        # Keep device/backend imports out of CONFIG_OFF and policy evaluation.
        from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec, UniformTypeKVCacheSpecs

        from vllm_ascend.attention.attention_v1 import AscendAttentionBackend, AscendAttentionMetadataBuilder
        from vllm_ascend.attention.mla_v1 import AscendMLABackend, AscendMLAMetadataBuilder
        from vllm_ascend.core.kv_cache_interface import (
            AscendMLAAttentionSpec,
            get_kv_cache_compression_ratio,
            get_storage_block_size,
        )

        return cls(
            AscendAttentionBackend,
            AscendAttentionMetadataBuilder,
            FullAttentionSpec,
            UniformTypeKVCacheSpecs,
            get_storage_block_size,
            get_kv_cache_compression_ratio,
            mla_backend=AscendMLABackend,
            mla_builder=AscendMLAMetadataBuilder,
            mla_spec=AscendMLAAttentionSpec,
            sliding_window_spec=SlidingWindowSpec,
        )


@dataclass(frozen=True)
class DraftWindowGroupRoute:
    gid: int
    layers: tuple[str, ...]
    scheduler_block_size: int
    table_column_span: int
    status: DraftWindowProviderStatus
    native_window: int | None = None


def _select_read_provider(
    backend: type,
    builder: Any,
    specs: Sequence[Any],
    types: DraftWindowBackendTypes,
    *,
    uncompressed: bool,
    native_window: int | None,
    uniform_windows: bool,
    full_retention_dflash2: bool,
) -> DraftWindowProviderStatus:
    """Match one actual consumer, then apply its layout-specific restrictions."""
    status = DraftWindowProviderStatus
    selected: DraftWindowProviderStatus | None = None
    if not uncompressed:
        raise NotImplementedError("Compressed or packed cache geometry needs a separate read provider")

    def matches(backend_type: type | None, builder_type: type | None, spec_type: type | None) -> bool:
        return (
            backend is backend_type and type(builder) is builder_type and all(type(item) is spec_type for item in specs)
        )

    if full_retention_dflash2 and any(
        type(item) is not types.full_attention_spec or getattr(item, "attention_chunk_size", None) is not None
        for item in specs
    ):
        raise NotImplementedError("DFlash2 requires Full-retention FullAttentionSpec without chunked attention")

    native = matches(types.dense_backend, types.dense_builder, types.sliding_window_spec)
    if matches(types.dense_backend, types.dense_builder, types.full_attention_spec):
        if (
            all(getattr(item, "attention_chunk_size", None) is None for item in specs)
            and uniform_windows
            and not (full_retention_dflash2 and all(getattr(item, "sliding_window", None) for item in specs))
        ):
            selected = status.DIRECT_DENSE
        elif full_retention_dflash2:
            native = True
        else:
            raise NotImplementedError("Dense chunked or nonuniform native attention needs a separate provider")
    elif not native:
        if matches(types.mla_backend, types.mla_builder, types.mla_spec):
            if builder.flashmla_state is not None:
                raise NotImplementedError("FlashMLA persistent derived state needs a separate provider")
            selected = status.DIRECT_MLA
        if selected is status.DIRECT_MLA and any(
            item.model_version is not None
            or item.cache_sparse_sfa_c8
            or item.cache_sparse_sfa_turboquant
            or item.store_on_host
            or item.indexes_kv_by_block_stride
            for item in specs
        ):
            raise NotImplementedError("Packed, offloaded, or strided MLA caches need a separate provider")
    if native:
        if not uniform_windows or native_window is None or native_window <= 0:
            raise NotImplementedError("Native SWA group has inconsistent windows")
        selected = status.NATIVE_SWA
    if selected is None:
        raise NotImplementedError(f"No read-view provider for {backend.__name__}/{type(builder).__name__}")
    return selected


def inventory_draft_window_groups(
    *,
    attn_groups: Sequence[Sequence[Any]],
    kv_cache_config: Any,
    block_tables: Any,
    parallel_config: Any,
    types: DraftWindowBackendTypes | None = None,
    full_retention_dflash2: bool = False,
) -> tuple[DraftWindowGroupRoute, ...]:
    """Inventory every actual consumer without inspecting tensor contents.

    AscendBlockTables currently expands manager blocks into kernel columns.
    Thus ``kernel_block_sizes[gid]`` is the logical raw-token span of one
    table column; compressed physical rows are that span divided by ratio.
    No resident-set guarantee is inferred from capacity or valid counts.
    """
    if types is None:
        types = DraftWindowBackendTypes.load()
    if parallel_config.prefill_context_parallel_size != 1 or parallel_config.decode_context_parallel_size != 1:
        raise NotImplementedError("Context-parallel read coordinates need a separate provider")
    cache_groups = kv_cache_config.kv_cache_groups
    if not (
        len(attn_groups) == len(cache_groups) == len(block_tables.block_sizes) == len(block_tables.kernel_block_sizes)
    ):
        raise ValueError("Draft attention groups and block-table geometry do not match")
    routes = []
    for gid, (groups, cache_group) in enumerate(zip(attn_groups, cache_groups)):
        if not groups:
            # init_attn_backend preserves indices for target-only cache groups
            # after filtering by active_layer_names. They have no draft reader.
            continue
        spec = cache_group.kv_cache_spec
        specs = tuple(spec.kv_cache_specs.values()) if isinstance(spec, types.uniform_spec) else (spec,)
        if not specs:
            raise ValueError(f"Draft KV group {gid} has no cache spec")
        geometries = {
            (int(item.block_size), int(types.compression_ratio(item)), int(types.storage_block_size(item)))
            for item in specs
        }
        if len(geometries) != 1:
            raise ValueError(f"Draft KV group {gid} has inconsistent cache geometry")
        scheduler_span, ratio, storage_size = geometries.pop()
        if int(block_tables.block_sizes[gid]) != scheduler_span:
            raise ValueError(f"Draft KV group {gid} scheduler and canonical table spans disagree")
        column_span = int(block_tables.kernel_block_sizes[gid])
        if (
            min(scheduler_span, column_span, ratio, storage_size) <= 0
            or scheduler_span % column_span
            or column_span % ratio
        ):
            raise ValueError(f"Draft KV group {gid} has invalid block geometry")
        for group in groups:
            if group.kv_cache_group_id != gid or int(group.kv_cache_spec.block_size) != scheduler_span:
                raise ValueError(f"Draft KV group {gid} consumer and canonical source geometry disagree")
            if isinstance(spec, types.uniform_spec):
                try:
                    consumer_specs = tuple(spec.kv_cache_specs[name] for name in group.layer_names)
                except KeyError as error:
                    raise ValueError(f"Draft KV group {gid} omits an active consumer spec") from error
            else:
                consumer_specs = specs
            builder = group.get_metadata_builder(0)
            backend = group.backend
            windows = {getattr(item, "sliding_window", None) for item in consumer_specs}
            native_window = next(iter(windows)) if len(windows) == 1 else None
            try:
                status = _select_read_provider(
                    backend,
                    builder,
                    consumer_specs,
                    types,
                    uncompressed=ratio == 1 and storage_size == scheduler_span,
                    native_window=native_window,
                    uniform_windows=len(windows) == 1,
                    full_retention_dflash2=full_retention_dflash2,
                )
            except NotImplementedError as error:
                raise NotImplementedError(f"Draft Window gid={gid}: {error}") from error
            routes.append(
                DraftWindowGroupRoute(
                    gid=gid,
                    layers=tuple(group.layer_names),
                    scheduler_block_size=scheduler_span,
                    table_column_span=column_span,
                    status=status,
                    native_window=native_window,
                )
            )
    if not routes:
        raise ValueError("Draft Window requires at least one attention consumer")
    return tuple(routes)


def require_direct_prefix_routes(
    speculator: Any, *, window_size: int | None = None
) -> tuple[DraftWindowGroupRoute, ...]:
    """Admit actual consumers once, retaining only groups that need projection.

    Inventory tags describe layouts, not read qualification. Native consumers
    keep their original band and require the method's real causal/residency
    contract here. Each admitted route is the original inventory object.
    """
    config = speculator.attn_vllm_config
    routes = inventory_draft_window_groups(
        attn_groups=speculator.attn_groups,
        kv_cache_config=speculator.kv_cache_config,
        block_tables=speculator.block_tables,
        parallel_config=config.parallel_config,
        full_retention_dflash2=getattr(speculator, "_dflash2_full_kv_reference", False),
    )
    if {name for route in routes for name in route.layers} != set(speculator.draft_attn_layer_names):
        raise ValueError("Draft Window providers do not cover the actual draft attention layers")
    native = tuple(route for route in routes if route.status is DraftWindowProviderStatus.NATIVE_SWA)
    if native:
        if window_size is None or speculator.speculative_config.method != "dflash":
            raise NotImplementedError("Native SWA coverage is only certified for the DFlash extra-Window route")
        from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl

        scheduler = config.scheduler_config
        query_bound = max(
            scheduler.max_num_batched_tokens,
            scheduler.max_num_scheduled_tokens or 0,
            speculator.num_query_per_req,
        )
        future = speculator.num_speculative_steps
        if query_bound < 1 or future < 0:
            raise ValueError("Native SWA coverage requires a positive query bound")
        native_only = len(native) == len(routes)
        context = config.compilation_config.static_forward_context
        declared = getattr(speculator, "_group_causal", True)
        omitted: set[DraftWindowGroupRoute] = set()
        for route in native:
            # Inventory already established a positive, homogeneous native P.
            native_window = cast(int, route.native_window)
            causal = declared.get(route.gid) if isinstance(declared, dict) else declared
            if type(causal) is not bool:
                raise ValueError("Native SWA requires an explicit per-group causal declaration")
            if not causal:
                if not (
                    native_only and getattr(speculator.draft_model_config, "architectures", ()) == ["DFlash2DraftModel"]
                ):
                    raise NotImplementedError(
                        "Draft Window non-causal native groups require the DFlash2 resident provider"
                    )
                # sparse_mode=0 reads the complete rebased interval, including
                # future queries: it cannot inherit the causal-band proof.
                width = speculator.num_query_per_req
                if width < 1 or window_size + route.table_column_span - 1 + width > native_window:
                    raise NotImplementedError(
                        "DFlash2 non-causal draft_window_size plus column alignment and query width "
                        "must fit its resident native window"
                    )
            for name in route.layers:
                impl = getattr(context.get(name), "impl", None)
                if (
                    type(impl) is not AscendAttentionBackendImpl
                    or impl.sliding_window != native_window
                    or impl.sinks is not None
                    or impl.use_bnsd_kv_cache
                ):
                    raise NotImplementedError(f"Draft Window native SWA consumer {name} has no coverage certificate")
            # S <= max(E+f-W, 0); the first native query reads from
            # max(E-Q-P, 0). W >= P+Q+f contains that complete read set.
            # It then needs no projection beside Full groups. Native-only
            # batches still need their own builder for the proposal scope.
            if not native_only and window_size >= native_window + query_bound + future:
                omitted.add(route)
        routes = tuple(route for route in routes if route not in omitted)
    prefix = tuple(route for route in routes if route.status is not DraftWindowProviderStatus.NATIVE_SWA)
    if prefix:
        # Every group must represent any start aligned to the first span;
        # checking only the current request would miss later block boundaries.
        anchor = prefix[0].table_column_span
        for route in prefix:
            if anchor % route.table_column_span:
                raise NotImplementedError(
                    "Draft Window provider missing: "
                    f"gid={route.gid} span {route.table_column_span} cannot represent every start "
                    f"aligned to the first group span {anchor}"
                )
    return routes
