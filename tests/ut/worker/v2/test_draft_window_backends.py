# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for consumer identity and block-table units."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from vllm_ascend.worker.v2.spec_decode.draft_window_backends import (
    DraftWindowBackendTypes,
    DraftWindowProviderStatus,
    inventory_draft_window_groups,
    require_compatible_direct_prefix_routes,
    select_native_swa_subset_routes,
)


class DenseBackend:
    pass


class DenseBuilder:
    sliding_window: int | None
    sinks: object | None
    use_bnsd_kv_cache: bool


class DenseSubclass(DenseBuilder):
    pass


class CompressedBuilder:
    pass


class FullSpec:
    sliding_window: int | None
    attention_chunk_size: int | None

    def __init__(self, block_size=256, tokens_per_state=1):
        self.block_size = block_size
        self.tokens_per_state = tokens_per_state


class SlidingSpec(FullSpec):
    pass


class UniformSpec:
    def __init__(self, specs):
        self.kv_cache_specs = specs


@pytest.fixture
def types():
    return DraftWindowBackendTypes(
        dense_backend=DenseBackend,
        dense_builder=DenseBuilder,
        full_attention_spec=FullSpec,
        uniform_spec=UniformSpec,
        storage_block_size=lambda spec: spec.block_size // spec.tokens_per_state,
        compression_ratio=lambda spec: spec.tokens_per_state,
    )


@pytest.fixture
def parallel_config():
    return SimpleNamespace(prefill_context_parallel_size=1, decode_context_parallel_size=1)


def inventory(
    types,
    parallel_config,
    *,
    specs=None,
    builders=None,
    spans=None,
    table_spans=None,
    group_ids=None,
    full_retention_dflash2=False,
):
    specs = [FullSpec()] if specs is None else specs
    builders = [DenseBuilder() for _ in specs] if builders is None else builders
    spans = [128 for _ in specs] if spans is None else spans
    groups = [
        [
            SimpleNamespace(
                backend=DenseBackend,
                kv_cache_group_id=gid if group_ids is None else group_ids[gid],
                kv_cache_spec=next(iter(specs[gid].kv_cache_specs.values()))
                if isinstance(specs[gid], UniformSpec)
                else specs[gid],
                layer_names=list(specs[gid].kv_cache_specs)
                if isinstance(specs[gid], UniformSpec)
                else [f"draft.layers.{gid}"],
                get_metadata_builder=lambda index, builder=builder: builder,
            )
        ]
        if builder is not None
        else []
        for gid, builder in enumerate(builders)
    ]
    return inventory_draft_window_groups(
        attn_groups=groups,
        kv_cache_config=SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec) for spec in specs]),
        block_tables=SimpleNamespace(
            block_sizes=table_spans
            if table_spans is not None
            else [
                next(iter(spec.kv_cache_specs.values())).block_size
                if isinstance(spec, UniformSpec) and spec.kv_cache_specs
                else getattr(spec, "block_size", 0)
                for spec in specs
            ],
            kernel_block_sizes=spans,
        ),
        parallel_config=parallel_config,
        types=types,
        full_retention_dflash2=full_retention_dflash2,
    )


def test_dflash2_reference_full_allocation_keeps_native_read_provider(types, parallel_config):
    spec = FullSpec()
    spec.sliding_window = 2048
    kinds = replace(types, sliding_window_spec=SlidingSpec)
    (original,) = inventory(kinds, parallel_config, specs=[spec])
    assert original.status is DraftWindowProviderStatus.DIRECT_DENSE
    assert original.native_window == 2048  # Full retention is not Full attention.
    (reference,) = inventory(kinds, parallel_config, specs=[spec], full_retention_dflash2=True)
    assert reference.status is DraftWindowProviderStatus.NATIVE_SWA_CANDIDATE
    assert reference.native_window == 2048


def test_consumer_gid_must_match_its_canonical_source(types, parallel_config):
    with pytest.raises(ValueError, match="consumer and canonical source geometry"):
        inventory(types, parallel_config, group_ids=[1])


def test_cache_and_canonical_table_scheduler_spans_must_match(types, parallel_config):
    with pytest.raises(ValueError, match="scheduler and canonical table spans"):
        inventory(types, parallel_config, table_spans=[512])


def test_full_spec_with_chunked_attention_cannot_claim_dense_prefix_reads(types, parallel_config):
    spec = FullSpec()
    spec.attention_chunk_size = 1024
    with pytest.raises(NotImplementedError, match="Dense chunked"):
        inventory(types, parallel_config, specs=[spec])


def test_missing_native_window_is_a_missing_provider_not_an_implicit_full_reader(types, parallel_config):
    spec = SlidingSpec()
    spec.sliding_window = None
    with pytest.raises(NotImplementedError, match="inconsistent windows"):
        inventory(replace(types, sliding_window_spec=SlidingSpec), parallel_config, specs=[spec])


def test_kernel_column_span_is_not_scheduler_block_size(types, parallel_config):
    (route,) = inventory(types, parallel_config)
    assert route.scheduler_block_size == 256
    assert route.table_column_span == 128
    assert route.status is DraftWindowProviderStatus.DIRECT_DENSE
    require_compatible_direct_prefix_routes((route,))


def test_same_named_unregistered_builder_cannot_gain_dense_capability(types, parallel_config):
    # Strings do not certify an implementation, even when names match exactly.
    impostor = type(DenseBuilder.__name__, (), {})()
    with pytest.raises(NotImplementedError):
        inventory(types, parallel_config, builders=[impostor])


def test_subclass_and_native_window_require_their_own_provider(types, parallel_config):
    with pytest.raises(NotImplementedError):
        inventory(types, parallel_config, specs=[FullSpec(), SlidingSpec()], builders=[DenseSubclass(), DenseBuilder()])


def native_swa_bundle(types, parallel_config):
    spec = SlidingSpec()
    spec.sliding_window = 4096
    routes = inventory(replace(types, sliding_window_spec=SlidingSpec), parallel_config, specs=[spec, FullSpec()])
    impl = DenseBuilder()
    impl.sliding_window = 4096
    impl.sinks = None
    impl.use_bnsd_kv_cache = False
    context = {"draft.layers.0": SimpleNamespace(impl=impl)}
    return routes, impl, context


def test_native_swa_candidate_is_not_a_residency_or_read_provider(types, parallel_config):
    routes, _, _ = native_swa_bundle(types, parallel_config)
    assert routes[0].status is DraftWindowProviderStatus.NATIVE_SWA_CANDIDATE
    with pytest.raises(NotImplementedError, match="Native SWA"):
        require_compatible_direct_prefix_routes(routes)


def test_covered_swa_keeps_only_full_group_projection(types, parallel_config):
    routes, _, context = native_swa_bundle(types, parallel_config)
    selected = select_native_swa_subset_routes(
        routes,
        window_size=32768,
        max_query_tokens=2048,
        future_offset_bound=7,
        context=context,
        dense_impl_type=DenseBuilder,
    )
    assert selected == (routes[1],)
    require_compatible_direct_prefix_routes(selected)
    assert context["draft.layers.0"].impl.sliding_window == 4096


def test_smaller_extra_window_selects_native_intersection(types, parallel_config):
    routes, _, context = native_swa_bundle(types, parallel_config)
    selected = select_native_swa_subset_routes(
        routes,
        window_size=1024,
        max_query_tokens=8,
        future_offset_bound=7,
        context=context,
        dense_impl_type=DenseBuilder,
    )
    assert len(selected) == 2
    assert selected[0].status is DraftWindowProviderStatus.NATIVE_SWA_INTERSECTION
    assert selected[1] == routes[1]
    require_compatible_direct_prefix_routes(selected)


@pytest.mark.parametrize("window", [1024, 32768])
def test_native_only_dflash2_keeps_a_typed_builder_without_a_full_group(types, parallel_config, window):
    routes, _, context = native_swa_bundle(types, parallel_config)
    selected = select_native_swa_subset_routes(
        routes[:1],
        window_size=window,
        max_query_tokens=8,
        future_offset_bound=7,
        context=context,
        dense_impl_type=DenseBuilder,
    )
    assert len(selected) == 1
    assert selected[0].status is DraftWindowProviderStatus.NATIVE_SWA_INTERSECTION
    assert selected[0].native_window == 4096
    require_compatible_direct_prefix_routes(selected)


@pytest.mark.parametrize("allowed,window", [(True, 1024), (False, 1024), (True, 4096)])
def test_noncausal_native_only_dflash2_needs_explicit_resident_certificate(types, parallel_config, allowed, window):
    routes, _, context = native_swa_bundle(types, parallel_config)
    options = dict(
        window_size=window,
        max_query_tokens=512,
        future_offset_bound=7,
        context=context,
        dense_impl_type=DenseBuilder,
        group_causal={0: False},
        noncausal_dflash2=allowed,
        query_width=8,
    )
    if allowed and window == 1024:
        (route,) = select_native_swa_subset_routes(routes[:1], **options)
        assert route.status is DraftWindowProviderStatus.NATIVE_SWA_INTERSECTION
    else:
        with pytest.raises(NotImplementedError, match="non-causal|resident"):
            select_native_swa_subset_routes(routes[:1], **options)


def test_unmasked_rebased_read_covers_future_queries_without_fabricating_a_causal_band():
    for end in (520, 1151, 1152, 4095, 4096):
        span, window, width, native = 128, 1024, 8, 2048
        assert window + span - 1 + width <= native
        start = max(end - window, 0) // span * span
        assert start >= max(end - native + width, 0)
        # sparse_mode=0 keeps every valid key through E, including query
        # positions to the right of the first query; rebasing changes no mask.
        actual = {start + key for key in range(end - start)}
        assert actual == set(range(start, end))
        assert end - 1 in actual


@pytest.mark.parametrize("change", [{"sliding_window": 1024}, {"sinks": object()}, {"use_bnsd_kv_cache": True}])
def test_native_swa_actual_consumer_must_match_certificate(types, parallel_config, change):
    routes, impl, context = native_swa_bundle(types, parallel_config)
    for name, value in change.items():
        setattr(impl, name, value)
    with pytest.raises(NotImplementedError, match="coverage certificate"):
        select_native_swa_subset_routes(
            routes,
            window_size=32768,
            max_query_tokens=2048,
            future_offset_bound=7,
            context=context,
            dense_impl_type=DenseBuilder,
        )


@pytest.mark.parametrize("future", [0, 7])
def test_native_swa_coverage_against_absolute_read_sets(types, parallel_config, future):
    routes, _, context = native_swa_bundle(types, parallel_config)
    width, native_window = 8, 4096
    window = native_window + width + future
    select_native_swa_subset_routes(
        routes,
        window_size=window,
        max_query_tokens=width,
        future_offset_bound=future,
        context=context,
        dense_impl_type=DenseBuilder,
    )
    for end in (8, 1151, 1152, 4095, 4096, 4223, 33023):
        for block in (64, 128):
            start = max(end + future - window, 0) // block * block
            for query in range(end - width, end):
                native_read = set(range(max(query - native_window, 0), query + 1))
                assert native_read == {key for key in native_read if key >= start}


@pytest.mark.parametrize("ratio,span", [(4, 128), (128, 4096)])
def test_compressed_layout_cannot_gain_a_direct_provider(types, parallel_config, ratio, span):
    with pytest.raises(NotImplementedError, match="Compressed"):
        inventory(
            types, parallel_config, specs=[FullSpec(span * 2, ratio)], builders=[CompressedBuilder()], spans=[span]
        )


@pytest.mark.parametrize(
    "parallel",
    [
        {"prefill_context_parallel_size": 2},
        {"decode_context_parallel_size": 2},
    ],
)
def test_cp_needs_coordinate_provider(types, parallel_config, parallel):
    with pytest.raises(NotImplementedError, match="Context-parallel"):
        inventory(types, SimpleNamespace(**dict(vars(parallel_config), **parallel)))


def test_tp_and_effective_sp_are_separate_from_cp(types, parallel_config):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=16,
            use_sequence_parallel_moe=True,
            enable_expert_parallel=True,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        )
    )
    (route,) = inventory(types, config.parallel_config)
    assert route.status is DraftWindowProviderStatus.DIRECT_DENSE


def test_every_active_gid_is_retained_and_missing_provider_blocks_bundle(types, parallel_config):
    with pytest.raises(NotImplementedError, match="gid=2"):
        inventory(
            types,
            parallel_config,
            specs=[FullSpec(), FullSpec(), FullSpec()],
            builders=[DenseBuilder(), None, CompressedBuilder()],
        )


def test_uniform_wrapper_is_unwrapped_without_inventing_residency(types, parallel_config):
    (route,) = inventory(types, parallel_config, specs=[UniformSpec({"a": FullSpec(), "b": FullSpec()})])
    assert route.status is DraftWindowProviderStatus.DIRECT_DENSE


@pytest.mark.parametrize(
    "specs,spans",
    [
        ([FullSpec(256)], [96]),
        ([FullSpec(256, 3)], [128]),
        ([UniformSpec({"a": FullSpec(256), "b": FullSpec(512)})], [128]),
    ],
)
def test_invalid_geometry_is_not_zero_padded_or_repaired(types, parallel_config, specs, spans):
    with pytest.raises(ValueError, match="geometry"):
        inventory(types, parallel_config, specs=specs, spans=spans)


def test_empty_inventory_is_not_success():
    with pytest.raises(ValueError, match="at least one"):
        require_compatible_direct_prefix_routes(())


def test_unknown_second_gid_is_never_ignored(types, parallel_config):
    with pytest.raises(NotImplementedError, match="gid=1"):
        inventory(types, parallel_config, specs=[FullSpec(), FullSpec()], builders=[DenseBuilder(), object()])


@pytest.mark.parametrize("anchor,second", [(256, 128), (128, 64), (128, 128)])
def test_multi_group_projection_preserves_first_group_start(types, parallel_config, anchor, second):
    routes = inventory(types, parallel_config, specs=[FullSpec(256), FullSpec(256)], spans=[anchor, second])
    require_compatible_direct_prefix_routes(routes)
    # The logical read start is decided once using the first group's B.
    end, window, future = 2051, 1024, 8
    start = max(end + future - window, 0) // anchor * anchor
    columns = [start // route.table_column_span for route in routes]
    assert all(column * route.table_column_span == start for column, route in zip(columns, routes))
    assert [route.gid for route in routes] == [0, 1]


def test_incompatible_multi_group_alignment_rejected_before_request_arrives(types, parallel_config):
    routes = inventory(types, parallel_config, specs=[FullSpec(256), FullSpec(256)], spans=[64, 128])
    # S=128 happens to be representable in both groups, but S=64 is not.
    # Accepting only the first request would create a serving-time failure.
    assert [128 // route.table_column_span for route in routes] == [2, 1]
    with pytest.raises(NotImplementedError, match="first group span 64"):
        require_compatible_direct_prefix_routes(routes)


class MLABuilder:
    flashmla_state: object | None = None


class MLASpec(FullSpec):
    model_version = None
    cache_sparse_sfa_c8 = False
    cache_sparse_sfa_turboquant = False
    store_on_host = False
    indexes_kv_by_block_stride = False
    non_causal_multi_token_decode = False


def test_plain_mla_provider_requires_actual_builder_and_spec(types, parallel_config):
    mla_types = replace(
        types,
        mla_backend=DenseBackend,
        mla_builder=MLABuilder,
        mla_spec=MLASpec,
    )
    (route,) = inventory(mla_types, parallel_config, specs=[MLASpec()], builders=[MLABuilder()])
    assert route.status is DraftWindowProviderStatus.DIRECT_MLA
    require_compatible_direct_prefix_routes((route,))


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_version", "alternate-layout"),
        ("cache_sparse_sfa_c8", True),
        ("cache_sparse_sfa_turboquant", True),
        ("store_on_host", True),
        ("indexes_kv_by_block_stride", True),
    ],
)
def test_mla_special_layouts_do_not_inherit_plain_read_projection(types, parallel_config, field, value):
    mla_types = replace(types, mla_backend=DenseBackend, mla_builder=MLABuilder, mla_spec=MLASpec)
    spec = MLASpec()
    setattr(spec, field, value)
    with pytest.raises(NotImplementedError):
        inventory(mla_types, parallel_config, specs=[spec], builders=[MLABuilder()])


def test_flashmla_has_separate_persistent_state_contract(types, parallel_config):
    mla_types = replace(types, mla_backend=DenseBackend, mla_builder=MLABuilder, mla_spec=MLASpec)
    builder = MLABuilder()
    builder.flashmla_state = object()
    with pytest.raises(NotImplementedError, match="FlashMLA"):
        inventory(mla_types, parallel_config, specs=[MLASpec()], builders=[builder])


def test_small_window_native_intersection_gap(types, parallel_config):
    routes, _, context = native_swa_bundle(types, parallel_config)
    selected = select_native_swa_subset_routes(
        routes,
        window_size=1024,
        max_query_tokens=8,
        future_offset_bound=7,
        context=context,
        dense_impl_type=DenseBuilder,
    )
    assert len(selected) == 2
    assert selected[0].status is DraftWindowProviderStatus.NATIVE_SWA_INTERSECTION
    assert selected[1].gid == routes[1].gid


@pytest.mark.parametrize("field", ["prefill_context_parallel_size", "decode_context_parallel_size"])
@pytest.mark.parametrize("kind", ["dense", "mla", "native"])
def test_admitted_providers_use_the_actual_draft_parallel_contract(types, parallel_config, kind, field):
    candidates = {
        "dense": (types, FullSpec(), DenseBuilder()),
        "mla": (
            replace(types, mla_backend=DenseBackend, mla_builder=MLABuilder, mla_spec=MLASpec),
            MLASpec(),
            MLABuilder(),
        ),
        "native": (replace(types, sliding_window_spec=SlidingSpec), SlidingSpec(), DenseBuilder()),
    }
    kinds, spec, builder = candidates[kind]
    if kind == "native":
        spec.sliding_window = 128
    setattr(parallel_config, field, 2)
    with pytest.raises(NotImplementedError, match="Context-parallel"):
        inventory(kinds, parallel_config, specs=[spec], builders=[builder])


@pytest.mark.parametrize("consumer", ["SFA", "INDEXER", "DSA"])
def test_unqualified_sparse_draft_consumers_fail_before_proposal_work(types, parallel_config, consumer):
    from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder, AscendDSASWABackend
    from vllm_ascend.attention.indexer import AscendSFAIndexerBackend, AscendSFAIndexerMetadataBuilder
    from vllm_ascend.attention.sfa_v1 import AscendSFABackend, AscendSFAMetadataBuilder

    backend, builder_type = {
        "SFA": (AscendSFABackend, AscendSFAMetadataBuilder),
        "INDEXER": (AscendSFAIndexerBackend, AscendSFAIndexerMetadataBuilder),
        "DSA": (AscendDSASWABackend, AscendDSAMetadataBuilder),
    }[consumer]
    group = SimpleNamespace(
        backend=backend,
        kv_cache_group_id=0,
        kv_cache_spec=FullSpec(),
        layer_names=["draft.layer"],
        get_metadata_builder=lambda _: object.__new__(builder_type),
    )
    with pytest.raises(NotImplementedError, match="No read-view provider"):
        inventory_draft_window_groups(
            attn_groups=[[group]],
            kv_cache_config=SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=FullSpec())]),
            block_tables=SimpleNamespace(block_sizes=[256], kernel_block_sizes=[128]),
            parallel_config=parallel_config,
            types=types,
        )
