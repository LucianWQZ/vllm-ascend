# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for consumer identity and block-table units."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec, UniformTypeKVCacheSpecs

from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
)
from vllm_ascend.attention.mla_v1 import AscendMLABackend, AscendMLAMetadataBuilder
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.worker.v2.spec_decode.draft_window_backends import (
    DraftWindowProviderStatus,
    inventory_draft_window_groups,
    require_direct_prefix_routes,
)


class DenseSubclass(AscendAttentionMetadataBuilder):
    pass


class CompressedBuilder:
    pass


def dense_builder():
    return object.__new__(AscendAttentionMetadataBuilder)


def mla_builder():
    builder = object.__new__(AscendMLAMetadataBuilder)
    builder.flashmla_state = None
    return builder


def full_spec(block_size=256, tokens_per_state=1, **kwargs):
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=2,
        head_size=128,
        dtype=torch.bfloat16,
        tokens_per_state=tokens_per_state,
        **kwargs,
    )


def sliding_spec(**kwargs):
    return SlidingWindowSpec(
        block_size=256, num_kv_heads=2, head_size=128, dtype=torch.bfloat16, sliding_window=4096, **kwargs
    )


def mla_spec():
    return AscendMLAAttentionSpec(block_size=256, num_kv_heads=1, head_size=128, dtype=torch.bfloat16)


def uniform_spec(specs):
    return UniformTypeKVCacheSpecs(block_size=next(iter(specs.values())).block_size, kv_cache_specs=specs)


@pytest.fixture
def parallel_config():
    return SimpleNamespace(prefill_context_parallel_size=1, decode_context_parallel_size=1)


def inventory(
    parallel_config,
    *,
    specs=None,
    builders=None,
    spans=None,
    table_spans=None,
    group_ids=None,
    full_retention_dflash2=False,
    impl_changes=None,
    draft_layer_names=None,
):
    specs = [full_spec()] if specs is None else specs
    builders = [dense_builder() for _ in specs] if builders is None else builders
    spans = [128 for _ in specs] if spans is None else spans
    groups = [
        [
            SimpleNamespace(
                backend=AscendMLABackend if isinstance(builder, AscendMLAMetadataBuilder) else AscendAttentionBackend,
                kv_cache_group_id=gid if group_ids is None else group_ids[gid],
                kv_cache_spec=next(iter(specs[gid].kv_cache_specs.values()))
                if isinstance(specs[gid], UniformTypeKVCacheSpecs)
                else specs[gid],
                layer_names=list(specs[gid].kv_cache_specs)
                if isinstance(specs[gid], UniformTypeKVCacheSpecs)
                else [f"draft.layers.{gid}"],
                get_metadata_builder=lambda index, builder=builder: builder,
            )
        ]
        if builder is not None
        else []
        for gid, builder in enumerate(builders)
    ]
    context = {}
    for group_set in groups:
        for group in group_set:
            for name in group.layer_names:
                impl = object.__new__(AscendAttentionBackendImpl)
                impl.sliding_window = getattr(group.kv_cache_spec, "sliding_window", None)
                impl.sinks, impl.use_bnsd_kv_cache = None, False
                for field, value in (impl_changes or {}).items():
                    setattr(impl, field, value)
                context[name] = SimpleNamespace(impl=impl)
    owner = SimpleNamespace(
        attn_groups=groups,
        draft_attn_layer_names=set(context) if draft_layer_names is None else draft_layer_names,
        kv_cache_config=SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec) for spec in specs]),
        block_tables=SimpleNamespace(
            block_sizes=table_spans
            if table_spans is not None
            else [
                next(iter(spec.kv_cache_specs.values())).block_size
                if isinstance(spec, UniformTypeKVCacheSpecs) and spec.kv_cache_specs
                else getattr(spec, "block_size", 0)
                for spec in specs
            ],
            kernel_block_sizes=spans,
        ),
        attn_vllm_config=SimpleNamespace(
            parallel_config=parallel_config,
            compilation_config=SimpleNamespace(static_forward_context=context),
        ),
        _dflash2_full_kv_reference=full_retention_dflash2,
    )
    owner.vllm_config = owner.attn_vllm_config
    return inventory_draft_window_groups(owner)


def require_routes(
    routes,
    *,
    window_size=1024,
    max_query_tokens=8,
    future_offset_bound=7,
    context=None,
    group_causal=True,
    noncausal_dflash2=False,
    query_width=8,
):
    config = SimpleNamespace(
        parallel_config=None,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max_query_tokens, max_num_scheduled_tokens=None),
        compilation_config=SimpleNamespace(static_forward_context=context or {}),
    )
    owner = SimpleNamespace(
        vllm_config=config,
        attn_groups=(),
        kv_cache_config=None,
        block_tables=None,
        draft_attn_layer_names=[name for route in routes for name in route.layers],
        speculative_config=SimpleNamespace(method="dflash"),
        num_query_per_req=query_width,
        num_speculative_steps=future_offset_bound,
        _group_causal=group_causal,
        draft_model_config=SimpleNamespace(architectures=["DFlash2DraftModel"] if noncausal_dflash2 else []),
    )
    with (
        patch(
            "vllm_ascend.worker.v2.spec_decode.draft_window_backends.inventory_draft_window_groups", return_value=routes
        ),
    ):
        return require_direct_prefix_routes(owner, window_size=window_size)


def test_dflash2_reference_full_allocation_keeps_native_read_provider(parallel_config):
    spec = full_spec()
    spec = replace(spec, sliding_window=2048)
    (original,) = inventory(parallel_config, specs=[spec])
    assert original.status is DraftWindowProviderStatus.DIRECT_DENSE
    assert original.native_window == 2048  # Full retention is not Full attention.
    (reference,) = inventory(parallel_config, specs=[spec], full_retention_dflash2=True)
    assert reference.status is DraftWindowProviderStatus.NATIVE_SWA
    assert reference.native_window == 2048


@pytest.mark.parametrize("native", [None, 2048])
def test_dflash2_full_retention_keeps_full_and_native_consumers(parallel_config, native):
    spec = full_spec()
    spec = replace(spec, sliding_window=native)
    (route,) = inventory(parallel_config, specs=[spec], full_retention_dflash2=True)
    expected = DraftWindowProviderStatus.DIRECT_DENSE if native is None else DraftWindowProviderStatus.NATIVE_SWA
    assert route.status is expected
    assert route.native_window == native


@pytest.mark.parametrize("actual", [None, 1024, 4096])
def test_dflash2_inventory_compares_native_band_to_actual_consumer(parallel_config, actual):
    spec = replace(full_spec(), sliding_window=2048)
    with pytest.raises(NotImplementedError, match="inventory native band"):
        inventory(
            parallel_config,
            specs=[spec],
            full_retention_dflash2=True,
            impl_changes={"sliding_window": actual},
        )


@pytest.mark.parametrize("storage", ["sliding", "chunked", "wrapped_chunked"])
def test_dflash2_reference_cannot_admit_recycling_or_chunked_storage(parallel_config, storage):
    spec = sliding_spec() if storage == "sliding" else full_spec()
    spec = replace(spec, sliding_window=2048)
    if storage != "sliding":
        spec = replace(spec, attention_chunk_size=256)
    actual = uniform_spec({"draft.layers.0": spec}) if storage == "wrapped_chunked" else spec
    with pytest.raises(NotImplementedError, match="Full-retention FullAttentionSpec"):
        inventory(
            parallel_config,
            specs=[actual],
            full_retention_dflash2=True,
        )


@pytest.mark.parametrize(
    "dtype,changes,expected",
    [
        (torch.bfloat16, {}, True),
        (torch.float16, {}, True),
        (torch.float32, {}, False),
        (torch.bfloat16, {"sinks": object()}, False),
        (torch.bfloat16, {"use_bnsd_kv_cache": True}, False),
        (torch.bfloat16, {"sliding_window": 2048}, False),
    ],
)
def test_dense_read_uses_the_actual_consumer_and_spec_once(parallel_config, dtype, changes, expected):
    (route,) = inventory(parallel_config, specs=[replace(full_spec(), dtype=dtype)], impl_changes=changes)
    assert route.dense_read is expected


def test_inventory_is_the_single_actual_layer_coverage_boundary(parallel_config):
    with pytest.raises(ValueError, match="do not cover the actual draft"):
        inventory(parallel_config, draft_layer_names={"unknown.draft.reader"})


def test_consumer_gid_must_match_its_canonical_source(parallel_config):
    with pytest.raises(ValueError, match="consumer and canonical source geometry"):
        inventory(parallel_config, group_ids=[1])


def test_cache_and_canonical_table_scheduler_spans_must_match(parallel_config):
    with pytest.raises(ValueError, match="scheduler and canonical table spans"):
        inventory(parallel_config, table_spans=[512])


def test_full_spec_with_chunked_attention_cannot_claim_dense_prefix_reads(parallel_config):
    spec = full_spec()
    spec = replace(spec, attention_chunk_size=1024)
    with pytest.raises(NotImplementedError, match="Dense chunked"):
        inventory(parallel_config, specs=[spec])


def test_missing_native_window_is_a_missing_provider_not_an_implicit_full_reader(parallel_config):
    spec = sliding_spec()
    object.__setattr__(spec, "sliding_window", None)
    with pytest.raises(NotImplementedError, match="inconsistent windows"):
        inventory(parallel_config, specs=[spec])


def test_kernel_column_span_is_not_scheduler_block_size(parallel_config):
    (route,) = inventory(parallel_config)
    assert route.table_column_span == 128
    assert route.status is DraftWindowProviderStatus.DIRECT_DENSE
    require_routes((route,))


def test_same_named_unregistered_builder_cannot_gain_dense_capability(parallel_config):
    # Strings do not certify an implementation, even when names match exactly.
    impostor = type(AscendAttentionMetadataBuilder.__name__, (), {})()
    with pytest.raises(NotImplementedError):
        inventory(parallel_config, builders=[impostor])


def test_subclass_and_native_window_require_their_own_provider(parallel_config):
    with pytest.raises(NotImplementedError):
        inventory(
            parallel_config,
            specs=[full_spec(), sliding_spec()],
            builders=[object.__new__(DenseSubclass), dense_builder()],
        )


def native_swa_bundle(parallel_config):
    spec = sliding_spec()
    routes = inventory(parallel_config, specs=[spec, full_spec()])
    impl = object.__new__(AscendAttentionBackendImpl)
    impl.sliding_window = 4096
    impl.sinks = None
    impl.use_bnsd_kv_cache = False
    context = {"draft.layers.0": SimpleNamespace(impl=impl)}
    return routes, impl, context


def test_native_layout_does_not_bypass_actual_window_qualification(parallel_config):
    routes, _, _ = native_swa_bundle(parallel_config)
    assert routes[0].status is DraftWindowProviderStatus.NATIVE_SWA
    with pytest.raises(NotImplementedError, match="Native SWA"):
        require_routes(routes, window_size=None)


def test_covered_swa_keeps_only_full_group_projection(parallel_config):
    routes, _, context = native_swa_bundle(parallel_config)
    selected = require_routes(
        routes,
        window_size=32768,
        max_query_tokens=2048,
        future_offset_bound=7,
        context=context,
    )
    assert selected == (routes[1],)
    assert context["draft.layers.0"].impl.sliding_window == 4096


def test_smaller_extra_window_selects_native_intersection(parallel_config):
    routes, _, context = native_swa_bundle(parallel_config)
    selected = require_routes(
        routes,
        window_size=1024,
        max_query_tokens=8,
        future_offset_bound=7,
        context=context,
    )
    assert len(selected) == 2
    assert selected[0] == routes[0]
    assert selected[1] == routes[1]


@pytest.mark.parametrize("window", [1024, 32768])
def test_native_only_dflash2_keeps_a_typed_builder_without_a_full_group(parallel_config, window):
    routes, _, context = native_swa_bundle(parallel_config)
    selected = require_routes(
        routes[:1],
        window_size=window,
        max_query_tokens=8,
        future_offset_bound=7,
        context=context,
    )
    assert len(selected) == 1
    assert selected[0] == routes[0]
    assert selected[0].native_window == 4096


@pytest.mark.parametrize("allowed,window", [(True, 1024), (False, 1024), (True, 4096)])
def test_noncausal_native_only_dflash2_needs_explicit_resident_certificate(parallel_config, allowed, window):
    routes, _, context = native_swa_bundle(parallel_config)
    options = dict(
        window_size=window,
        max_query_tokens=512,
        future_offset_bound=7,
        context=context,
        group_causal={0: False},
        noncausal_dflash2=allowed,
        query_width=8,
    )
    if allowed and window == 1024:
        (route,) = require_routes(routes[:1], **options)
        assert route == routes[0]
    else:
        with pytest.raises(NotImplementedError, match="non-causal|resident"):
            require_routes(routes[:1], **options)


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
def test_native_swa_actual_consumer_must_match_certificate(parallel_config, change):
    routes, impl, context = native_swa_bundle(parallel_config)
    for name, value in change.items():
        setattr(impl, name, value)
    with pytest.raises(NotImplementedError, match="coverage certificate"):
        require_routes(
            routes,
            window_size=32768,
            max_query_tokens=2048,
            future_offset_bound=7,
            context=context,
        )


@pytest.mark.parametrize("future", [0, 7])
def test_native_swa_coverage_against_absolute_read_sets(parallel_config, future):
    routes, _, context = native_swa_bundle(parallel_config)
    width, native_window = 8, 4096
    window = native_window + width + future
    selected = require_routes(
        routes,
        window_size=window,
        max_query_tokens=width,
        future_offset_bound=future,
        context=context,
    )
    assert selected == (routes[1],)
    for end in (8, 1151, 1152, 4095, 4096, 4223, 33023):
        for block in (64, 128):
            start = max(end + future - window, 0) // block * block
            for query in range(end - width, end):
                native_read = set(range(max(query - native_window, 0), query + 1))
                assert native_read == {key for key in native_read if key >= start}


@pytest.mark.parametrize("ratio,span", [(4, 128), (128, 4096)])
def test_compressed_layout_cannot_gain_a_direct_provider(parallel_config, ratio, span):
    with pytest.raises(NotImplementedError, match="Compressed"):
        inventory(parallel_config, specs=[full_spec(span * 2, ratio)], builders=[CompressedBuilder()], spans=[span])


@pytest.mark.parametrize(
    "parallel",
    [
        {"prefill_context_parallel_size": 2},
        {"decode_context_parallel_size": 2},
    ],
)
def test_cp_needs_coordinate_provider(parallel_config, parallel):
    with pytest.raises(NotImplementedError, match="Context-parallel"):
        inventory(SimpleNamespace(**dict(vars(parallel_config), **parallel)))


def test_tp_and_effective_sp_are_separate_from_cp(parallel_config):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=16,
            use_sequence_parallel_moe=True,
            enable_expert_parallel=True,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        )
    )
    (route,) = inventory(config.parallel_config)
    assert route.status is DraftWindowProviderStatus.DIRECT_DENSE


def test_every_active_gid_is_retained_and_missing_provider_blocks_bundle(parallel_config):
    with pytest.raises(NotImplementedError, match="gid=2"):
        inventory(
            parallel_config,
            specs=[full_spec(), full_spec(), full_spec()],
            builders=[dense_builder(), None, CompressedBuilder()],
        )


def test_uniform_wrapper_is_unwrapped_without_inventing_residency(parallel_config):
    (route,) = inventory(parallel_config, specs=[uniform_spec({"a": full_spec(), "b": full_spec()})])
    assert route.status is DraftWindowProviderStatus.DIRECT_DENSE


@pytest.mark.parametrize(
    "specs,spans",
    [
        ([full_spec(256)], [96]),
        ([full_spec(256, 3)], [128]),
        ([uniform_spec({"a": full_spec(256), "b": full_spec(512)})], [128]),
    ],
)
def test_invalid_geometry_is_not_zero_padded_or_repaired(parallel_config, specs, spans):
    with pytest.raises(ValueError, match="geometry"):
        inventory(parallel_config, specs=specs, spans=spans)


def test_empty_inventory_is_not_success():
    with pytest.raises(ValueError, match="at least one"):
        inventory(
            SimpleNamespace(prefill_context_parallel_size=1, decode_context_parallel_size=1),
            specs=[],
            builders=[],
            spans=[],
        )


def test_unknown_second_gid_is_never_ignored(parallel_config):
    with pytest.raises(NotImplementedError, match="gid=1"):
        inventory(parallel_config, specs=[full_spec(), full_spec()], builders=[dense_builder(), object()])


@pytest.mark.parametrize("anchor,second", [(256, 128), (128, 64), (128, 128)])
def test_multi_group_projection_preserves_first_group_start(parallel_config, anchor, second):
    routes = inventory(parallel_config, specs=[full_spec(256), full_spec(256)], spans=[anchor, second])
    require_routes(routes)
    # The logical read start is decided once using the first group's B.
    end, window, future = 2051, 1024, 8
    start = max(end + future - window, 0) // anchor * anchor
    columns = [start // route.table_column_span for route in routes]
    assert all(column * route.table_column_span == start for column, route in zip(columns, routes))
    assert [route.gid for route in routes] == [0, 1]


def test_incompatible_multi_group_alignment_rejected_before_request_arrives(parallel_config):
    routes = inventory(parallel_config, specs=[full_spec(256), full_spec(256)], spans=[64, 128])
    # S=128 happens to be representable in both groups, but S=64 is not.
    # Accepting only the first request would create a serving-time failure.
    assert [128 // route.table_column_span for route in routes] == [2, 1]
    with pytest.raises(NotImplementedError, match="first group span 64"):
        require_routes(routes)


def test_plain_mla_provider_requires_actual_builder_and_spec(parallel_config):
    (route,) = inventory(parallel_config, specs=[mla_spec()], builders=[mla_builder()])
    assert route.status is DraftWindowProviderStatus.DIRECT_MLA
    require_routes((route,))


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
def test_mla_special_layouts_do_not_inherit_plain_read_projection(parallel_config, field, value):
    spec = mla_spec()
    spec = replace(spec, **{field: value})
    with pytest.raises(NotImplementedError):
        inventory(parallel_config, specs=[spec], builders=[mla_builder()])


def test_flashmla_has_separate_persistent_state_contract(parallel_config):
    builder = mla_builder()
    builder.flashmla_state = object()
    with pytest.raises(NotImplementedError, match="FlashMLA"):
        inventory(parallel_config, specs=[mla_spec()], builders=[builder])


def test_small_window_native_intersection_gap(parallel_config):
    routes, _, context = native_swa_bundle(parallel_config)
    selected = require_routes(
        routes,
        window_size=1024,
        max_query_tokens=8,
        future_offset_bound=7,
        context=context,
    )
    assert len(selected) == 2
    assert selected[0] == routes[0]
    assert selected[1].gid == routes[1].gid


@pytest.mark.parametrize("field", ["prefill_context_parallel_size", "decode_context_parallel_size"])
@pytest.mark.parametrize("kind", ["dense", "mla", "native"])
def test_admitted_providers_use_the_actual_draft_parallel_contract(parallel_config, kind, field):
    spec, builder = {
        "dense": (full_spec(), dense_builder()),
        "mla": (mla_spec(), mla_builder()),
        "native": (sliding_spec(), dense_builder()),
    }[kind]
    setattr(parallel_config, field, 2)
    with pytest.raises(NotImplementedError, match="Context-parallel"):
        inventory(parallel_config, specs=[spec], builders=[builder])


@pytest.mark.parametrize("consumer", ["SFA", "INDEXER", "DSA", "C8MXFP"])
def test_unqualified_draft_consumers_fail_before_proposal_work(parallel_config, consumer):
    from vllm_ascend.attention.attention_c8_mxfp import AscendC8MXFPAttentionBackend, AscendC8MXFPMetadataBuilder
    from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder, AscendDSASWABackend
    from vllm_ascend.attention.indexer import AscendSFAIndexerBackend, AscendSFAIndexerMetadataBuilder
    from vllm_ascend.attention.sfa_v1 import AscendSFABackend, AscendSFAMetadataBuilder

    backend, builder_type = {
        "SFA": (AscendSFABackend, AscendSFAMetadataBuilder),
        "INDEXER": (AscendSFAIndexerBackend, AscendSFAIndexerMetadataBuilder),
        "DSA": (AscendDSASWABackend, AscendDSAMetadataBuilder),
        "C8MXFP": (AscendC8MXFPAttentionBackend, AscendC8MXFPMetadataBuilder),
    }[consumer]
    group = SimpleNamespace(
        backend=backend,
        kv_cache_group_id=0,
        kv_cache_spec=full_spec(),
        layer_names=["draft.layer"],
        get_metadata_builder=lambda _: object.__new__(builder_type),
    )
    with pytest.raises(NotImplementedError, match="No read-view provider"):
        inventory_draft_window_groups(
            SimpleNamespace(
                attn_groups=[[group]],
                kv_cache_config=SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=full_spec())]),
                block_tables=SimpleNamespace(block_sizes=[256], kernel_block_sizes=[128]),
                vllm_config=SimpleNamespace(
                    parallel_config=parallel_config,
                    compilation_config=SimpleNamespace(static_forward_context={}),
                ),
            )
        )
