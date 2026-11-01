# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inspect actual draft consumers before selecting a read-view provider.

This inventory is created after ``set_attn`` and owns no tensors. A candidate
means the dense provider's addressing contract matches, not device validation.
Unknown consumers remain missing providers instead of being silently skipped.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any


class DraftWindowProviderStatus(str, Enum):
    DIRECT_DENSE = "DIRECT_DENSE"
    NATIVE_SWA_CANDIDATE = "NATIVE_SWA_CANDIDATE"
    NATIVE_SWA_INTERSECTION = "NATIVE_SWA_INTERSECTION"
    DIRECT_MLA = "DIRECT_MLA"
    DIRECT_DSA = "DIRECT_DSA"
    DIRECT_SFA = "DIRECT_SFA"
    DIRECT_SFA_INDEXER = "DIRECT_SFA_INDEXER"
    PROVIDER_MISSING = "PROVIDER_MISSING"


@dataclass(frozen=True)
class DraftWindowTopology:
    """Only the coordinates used by a Draft reader, not executor diagnostics."""

    prefill_context_parallel_size: int
    decode_context_parallel_size: int
    dsa_cp_effective: bool = False

    @classmethod
    def from_config(cls, config: Any) -> "DraftWindowTopology":
        parallel = config.parallel_config
        return cls(parallel.prefill_context_parallel_size, parallel.decode_context_parallel_size)


@dataclass(frozen=True)
class DraftWindowBackendTypes:
    """Concrete runtime classes; injected explicitly in CPU contract tests."""

    dense_backend: type
    dense_builder: type
    full_attention_spec: type
    uniform_spec: type[Any]
    context_parallel_builders: tuple[type, ...]
    storage_block_size: Callable[[Any], int]
    compression_ratio: Callable[[Any], int]
    mla_backend: type | None = None
    mla_builder: type | None = None
    mla_spec: type | None = None
    sfa_backend: type | None = None
    sfa_builder: type | None = None
    indexer_backend: type | None = None
    indexer_builder: type[Any] | None = None
    indexer_spec: type | None = None
    dsa_backend: type | None = None
    dsa_builder: type | None = None
    dsa_cp_builder: type | None = None
    dsa_spec: type | None = None
    sliding_window_spec: type | None = None

    @classmethod
    def load(cls) -> "DraftWindowBackendTypes":
        # Keep device/backend imports out of CONFIG_OFF and policy evaluation.
        from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec, UniformTypeKVCacheSpecs

        from vllm_ascend.attention.attention_v1 import AscendAttentionBackend, AscendAttentionMetadataBuilder
        from vllm_ascend.attention.context_parallel.dsa_cp import AscendDSACPMetadataBuilder
        from vllm_ascend.attention.context_parallel.sfa_cp import (
            AscendSFADSACPMetadataBuilder,
            AscendSFADSADCPMetadataBuilder,
        )
        from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder, AscendDSASWABackend
        from vllm_ascend.attention.indexer import AscendSFAIndexerBackend, AscendSFAIndexerMetadataBuilder
        from vllm_ascend.attention.mla_v1 import AscendMLABackend, AscendMLAMetadataBuilder
        from vllm_ascend.attention.sfa_v1 import AscendSFABackend, AscendSFAMetadataBuilder
        from vllm_ascend.core.kv_cache_interface import (
            AscendMLAAttentionSpec,
            AscendSFAIndexerCacheSpec,
            AscendSlidingWindowMLASpec,
            get_kv_cache_compression_ratio,
            get_storage_block_size,
        )

        return cls(
            AscendAttentionBackend,
            AscendAttentionMetadataBuilder,
            FullAttentionSpec,
            UniformTypeKVCacheSpecs,
            (AscendDSACPMetadataBuilder, AscendSFADSADCPMetadataBuilder, AscendSFADSACPMetadataBuilder),
            get_storage_block_size,
            get_kv_cache_compression_ratio,
            mla_backend=AscendMLABackend,
            mla_builder=AscendMLAMetadataBuilder,
            mla_spec=AscendMLAAttentionSpec,
            sfa_backend=AscendSFABackend,
            sfa_builder=AscendSFAMetadataBuilder,
            indexer_backend=AscendSFAIndexerBackend,
            indexer_builder=AscendSFAIndexerMetadataBuilder,
            indexer_spec=AscendSFAIndexerCacheSpec,
            dsa_backend=AscendDSASWABackend,
            dsa_builder=AscendDSAMetadataBuilder,
            dsa_cp_builder=AscendDSACPMetadataBuilder,
            dsa_spec=AscendSlidingWindowMLASpec,
            sliding_window_spec=SlidingWindowSpec,
        )


@dataclass(frozen=True)
class DraftWindowGroupRoute:
    gid: int
    layers: tuple[str, ...]
    scheduler_block_size: int
    table_column_span: int
    topology: DraftWindowTopology
    status: DraftWindowProviderStatus
    reason: str
    native_window: int | None = None


def _select_read_provider(
    backend: type,
    builder: Any,
    specs: Sequence[Any],
    types: DraftWindowBackendTypes,
    topology: DraftWindowTopology,
    *,
    uncompressed: bool,
    native_window: int | None,
    uniform_windows: bool,
    full_retention_dflash2: bool,
) -> tuple[DraftWindowProviderStatus, str]:
    """Match one actual consumer, then apply its layout-specific restrictions."""
    status = DraftWindowProviderStatus
    missing = status.PROVIDER_MISSING
    if not uncompressed:
        return missing, "Compressed or packed cache geometry needs a separate read provider"

    def matches(backend_type: type | None, builder_types: tuple[type | None, ...], spec_type: type | None) -> bool:
        return (
            backend_type is not None
            and backend is backend_type
            and type(builder) in builder_types
            and all(type(item) is spec_type for item in specs)
        )

    native = matches(types.dense_backend, (types.dense_builder,), types.sliding_window_spec)
    if matches(types.dense_backend, (types.dense_builder,), types.full_attention_spec):
        if (
            all(getattr(item, "attention_chunk_size", None) is None for item in specs)
            and uniform_windows
            and not (full_retention_dflash2 and all(getattr(item, "sliding_window", None) for item in specs))
        ):
            selected = status.DIRECT_DENSE
        elif full_retention_dflash2:
            native = True
        else:
            return missing, "Dense chunked or nonuniform native attention needs a separate provider"
    elif not native:
        selected = missing
        if matches(types.mla_backend, (types.mla_builder,), types.mla_spec):
            if builder.flashmla_state is not None:
                return missing, "FlashMLA persistent derived state needs a separate provider"
            selected = status.DIRECT_MLA
        elif matches(types.sfa_backend, (types.sfa_builder,), types.mla_spec):
            if builder.nope:
                return missing, "NoPE SFA has persistent sparse-kernel state requiring its own owner"
            if any(item.non_causal_multi_token_decode for item in specs):
                return missing, "Ordinary SFA kernels use causal sparse_mode=3 for multi-token queries"
            selected = status.DIRECT_SFA
        elif matches(types.dsa_backend, (types.dsa_builder, types.dsa_cp_builder), types.dsa_spec):
            hf = builder.model_config.hf_config
            if type(builder) is types.dsa_cp_builder:
                return missing, "Legacy DSA-CP is rejected by the MRV2 startup contract; no Window provider is enabled"
            if builder.compressor_ratio != 1:
                return missing, "Compressed DSA needs a separate provider"
            if builder._device_metadata_enabled:
                return missing, "Deferred DSA metadata needs an executor completion contract"
            if getattr(hf, "vision_n_layers", 0) > 0:
                return missing, "DSA vision ranges require absolute-to-window coordinate projection"
            if any(
                item.model_version is not None or item.indexes_kv_by_block_stride or item.bounded_replay
                for item in specs
            ):
                return missing, "Packed, strided, or replay DSA needs a separate read provider"
            if hf.sliding_window <= 0 or any(item.sliding_window != hf.sliding_window for item in specs):
                return missing, "DSA native-window kernel and resident cache geometry disagree"
            selected = status.DIRECT_DSA
        elif matches(types.indexer_backend, (types.indexer_builder,), types.indexer_spec):
            if builder.use_pcp or builder.use_dcp or builder.use_dsa_cp:
                return missing, "SFA indexer rank-local read coordinates need a separate provider"
            if any(item.cache_sparse_li_c4 or item.sfa_dcp_replicated_indexer_size != 1 for item in specs):
                return missing, "Packed C4 or replicated indexer pages need a separate provider"
            return status.DIRECT_SFA_INDEXER, ""
        if selected in (status.DIRECT_MLA, status.DIRECT_SFA) and any(
            item.model_version is not None
            or item.cache_sparse_sfa_c8
            or item.cache_sparse_sfa_turboquant
            or item.store_on_host
            or item.indexes_kv_by_block_stride
            for item in specs
        ):
            return missing, "Packed, offloaded, or strided MLA/SFA caches need a separate provider"
        if selected is missing:
            return missing, f"No read-view provider for {backend.__name__}/{type(builder).__name__}"
    if native:
        if not uniform_windows or native_window is None or native_window <= 0:
            return missing, "Native SWA group has inconsistent windows"
        selected = status.NATIVE_SWA_CANDIDATE
    if topology.prefill_context_parallel_size != 1 or topology.decode_context_parallel_size != 1:
        return missing, "Context-parallel read coordinates need a separate provider"
    return selected, ""


def inventory_draft_window_groups(
    *,
    attn_groups: Sequence[Sequence[Any]],
    kv_cache_config: Any,
    block_tables: Any,
    topology: DraftWindowTopology,
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
        spans = {int(item.block_size) for item in specs}
        ratios = {int(types.compression_ratio(item)) for item in specs}
        storage_sizes = {int(types.storage_block_size(item)) for item in specs}
        if len(spans) != 1 or len(ratios) != 1 or len(storage_sizes) != 1:
            raise ValueError(f"Draft KV group {gid} has inconsistent cache geometry")
        scheduler_span, ratio, storage_size = spans.pop(), ratios.pop(), storage_sizes.pop()
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
            # The target can enable DSA-CP while the draft is dense GQA.
            # Selection of the actual builder, not the global flag, decides
            # whether this reader uses legacy DSA-CP coordinates.
            route_topology = replace(
                topology,
                dsa_cp_effective=(
                    isinstance(builder, types.context_parallel_builders)
                    or (
                        types.indexer_builder is not None
                        and isinstance(builder, types.indexer_builder)
                        and builder.use_dsa_cp
                    )
                ),
            )
            windows = {getattr(item, "sliding_window", None) for item in consumer_specs}
            native_window = next(iter(windows)) if len(windows) == 1 else None
            status, reason = _select_read_provider(
                backend,
                builder,
                consumer_specs,
                types,
                route_topology,
                uncompressed=ratio == 1 and storage_size == scheduler_span,
                native_window=native_window,
                uniform_windows=len(windows) == 1,
                full_retention_dflash2=full_retention_dflash2,
            )
            routes.append(
                DraftWindowGroupRoute(
                    gid=gid,
                    layers=tuple(group.layer_names),
                    scheduler_block_size=scheduler_span,
                    table_column_span=column_span,
                    topology=route_topology,
                    status=status,
                    reason=reason,
                    native_window=native_window,
                )
            )
    return tuple(routes)


def require_draft_window_providers(routes: Sequence[DraftWindowGroupRoute]) -> None:
    """Fail before materializing any group if any required provider is missing."""
    if not routes:
        raise ValueError("Draft Window requires at least one attention consumer")
    missing = [
        route
        for route in routes
        if route.status in (DraftWindowProviderStatus.PROVIDER_MISSING, DraftWindowProviderStatus.NATIVE_SWA_CANDIDATE)
    ]
    if missing:
        details = "; ".join(f"gid={route.gid}: {route.reason}" for route in missing)
        raise NotImplementedError(f"Draft Window provider missing: {details}")


def describe_draft_window_routes(speculator: Any) -> tuple[DraftWindowGroupRoute, ...]:
    """Inspect initialized draft attention; never invoke from the OFF path."""
    config = speculator.attn_vllm_config
    topology = DraftWindowTopology.from_config(config)
    routes = inventory_draft_window_groups(
        attn_groups=speculator.attn_groups,
        kv_cache_config=speculator.kv_cache_config,
        block_tables=speculator.block_tables,
        topology=topology,
        full_retention_dflash2=getattr(speculator, "_dflash2_full_kv_reference", False),
    )
    actual_layers = {name for route in routes for name in route.layers}
    if actual_layers != set(speculator.draft_attn_layer_names):
        raise ValueError("Draft Window inventory does not cover the actual draft attention layers")
    return routes


def require_compatible_direct_prefix_routes(routes: Sequence[DraftWindowGroupRoute]) -> None:
    """Keep the first group's window alignment and project it into all groups.

    Requiring each group span to divide the first span ensures every future
    aligned start is expressible, without changing the window to an LCM or
    failing after an ordinary request crosses a block boundary.
    """
    require_draft_window_providers(routes)
    prefix_routes = tuple(
        route for route in routes if route.status is not DraftWindowProviderStatus.NATIVE_SWA_INTERSECTION
    )
    if not prefix_routes:
        # Native-band builders validate and rebase their own coordinates;
        # unlike a direct-prefix reader, they need no companion Full group.
        if routes:
            return
        raise NotImplementedError("Draft Window requires at least one certified read group")
    anchor_span = prefix_routes[0].table_column_span
    for route in prefix_routes:
        if anchor_span % route.table_column_span:
            raise NotImplementedError(
                "Draft Window provider missing: "
                f"gid={route.gid} span {route.table_column_span} cannot represent every start "
                f"aligned to the first group span {anchor_span}"
            )


def select_native_swa_subset_routes(
    routes: Sequence[DraftWindowGroupRoute],
    *,
    window_size: int,
    max_query_tokens: int,
    future_offset_bound: int,
    context: Any,
    dense_impl_type: type[Any],
    group_causal: dict[int, bool] | bool = True,
    noncausal_dflash2: bool = False,
    query_width: int | None = None,
) -> tuple[DraftWindowGroupRoute, ...]:
    """Keep a certified native SWA reader unchanged, never claim its residency.

    S <= max(E+f-W, 0). A native P-window's earliest query reads no
    earlier than max(E-Q-P, 0). W >= P+Q+f proves that its complete
    existing read set lies inside the requested extra Window. Smaller W
    selects a typed native-band intersection, whose original builder exact
    pair is reused after build. Neither route claims a resident Full prefix.
    """
    if max_query_tokens < 1 or future_offset_bound < 0:
        raise ValueError("Native SWA coverage requires a positive query bound")
    native_only = bool(routes) and all(
        route.status is DraftWindowProviderStatus.NATIVE_SWA_CANDIDATE for route in routes
    )
    projected = []
    for route in routes:
        if route.status is not DraftWindowProviderStatus.NATIVE_SWA_CANDIDATE:
            projected.append(route)
            continue
        native_window = route.native_window
        if native_window is None or native_window <= 0:
            raise NotImplementedError(f"Draft Window gid={route.gid} has no native-SWA geometry")
        causal = group_causal.get(route.gid) if isinstance(group_causal, dict) else group_causal
        if type(causal) is not bool:
            raise ValueError("Native SWA requires an explicit per-group causal declaration")
        if not causal:
            if not (noncausal_dflash2 and native_only):
                raise NotImplementedError("Draft Window non-causal native groups require the DFlash2 resident provider")
            # sparse_mode=0 reads the entire projected interval, including
            # future query positions. It must not inherit a causal-band
            # certificate or assume an evicted Full prefix is resident.
            if (
                query_width is None
                or query_width < 1
                or window_size + route.table_column_span - 1 + query_width > native_window
            ):
                raise NotImplementedError(
                    "DFlash2 non-causal draft_window_size plus column alignment and query width "
                    "must fit its resident native window"
                )
        for name in route.layers:
            impl = getattr(context.get(name), "impl", None)
            if (
                type(impl) is not dense_impl_type
                or impl.sliding_window != native_window
                or impl.sinks is not None
                or impl.use_bnsd_kv_cache
            ):
                raise NotImplementedError(f"Draft Window native SWA consumer {name} has no coverage certificate")
        if native_only or window_size < native_window + max_query_tokens + future_offset_bound:
            projected.append(
                replace(
                    route,
                    status=DraftWindowProviderStatus.NATIVE_SWA_INTERSECTION,
                    reason=(
                        "Typed causal native band intersected using its builder's current exact pair"
                        if causal
                        else "Typed DFlash2 unmasked resident interval using its builder's current exact pair"
                    ),
                )
            )
    if not projected:
        raise NotImplementedError("Draft Window requires at least one certified read group")
    return tuple(projected)


def require_direct_prefix_routes(
    speculator: Any, *, window_size: int | None = None
) -> tuple[DraftWindowGroupRoute, ...]:
    """Select all direct-prefix consumers before any proposal work is submitted."""
    routes = describe_draft_window_routes(speculator)
    if any(route.status is DraftWindowProviderStatus.NATIVE_SWA_CANDIDATE for route in routes):
        if window_size is None or speculator.speculative_config.method != "dflash":
            raise NotImplementedError("Native SWA coverage is only certified for the DFlash extra-Window route")
        from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl

        config = speculator.attn_vllm_config
        scheduler = config.scheduler_config
        query_bound = max(
            scheduler.max_num_batched_tokens,
            scheduler.max_num_scheduled_tokens or 0,
            speculator.num_query_per_req,
        )
        routes = select_native_swa_subset_routes(
            routes,
            window_size=window_size,
            max_query_tokens=query_bound,
            future_offset_bound=speculator.num_speculative_steps,
            context=config.compilation_config.static_forward_context,
            dense_impl_type=AscendAttentionBackendImpl,
            group_causal=getattr(speculator, "_group_causal", True),
            noncausal_dflash2=getattr(speculator.draft_model_config, "architectures", ()) == ["DFlash2DraftModel"],
            query_width=speculator.num_query_per_req,
        )
    if any(route.status is DraftWindowProviderStatus.DIRECT_SFA for route in routes):
        from vllm_ascend.worker.v2.spec_decode.draft_window_sfa import validate_sfa_read_consumers

        validate_sfa_read_consumers(speculator.attn_vllm_config.compilation_config.static_forward_context, routes)
    require_compatible_direct_prefix_routes(routes)
    return routes
