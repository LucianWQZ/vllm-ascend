# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Host contracts of real DFlash2 Graph inputs, without device execution.

These headers model fixed 544 allocations, not tensor contents or a numeric
oracle. Device capture, selector results and completion require device tests.
"""

from types import SimpleNamespace

import pytest


class TensorHeader:
    """Only metadata is legal: no values, allocation, copy or synchronization."""

    def __init__(self, shape, dtype="torch.int64"):
        self.shape, self.dtype, self.device = tuple(shape), dtype, "npu:0"
        self.address = id(self)

    def data_ptr(self):
        return self.address

    def stride(self):
        result, size = [], 1
        for dim in reversed(self.shape):
            result.append(size)
            size *= dim
        return tuple(reversed(result))

    def __getattr__(self, name):
        raise AssertionError(f"contract tried a nonmetadata tensor operation: {name}")


def dense_impl(native=2048):
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl

    impl = AscendAttentionBackendImpl.__new__(AscendAttentionBackendImpl)
    impl.sliding_window, impl.sinks, impl.use_bnsd_kv_cache = native, None, False
    if native is not None:
        impl._dflash2_native_swa_mask = TensorHeader((2048, 2048), "torch.int8")
    return impl


class Selector:
    top_k = 2

    def __call__(self):
        raise AssertionError("Host contract must not run the selector")


@pytest.fixture
def module():
    from vllm_ascend.worker.v2.spec_decode import draft_window_dflash2_graph

    return draft_window_dflash2_graph


@pytest.fixture
def owner():
    context = {"draft.0": SimpleNamespace(impl=dense_impl())}
    cfg = SimpleNamespace(
        method="dflash",
        enforce_eager=False,
        num_speculative_tokens=7,
        draft_sample_method="greedy",
        uses_dynamic_speculative_decoding=lambda: False,
        enable_adaptive_verification=False,
        draft_parallel_config=SimpleNamespace(tensor_parallel_size=2, use_sequence_parallel_moe=False),
    )
    model = SimpleNamespace(
        compute_candidates=lambda *_: None,
        model=SimpleNamespace(
            candidate_selector=Selector(),
            layers=[
                SimpleNamespace(attention_conv=SimpleNamespace(block_size=8), mlp_conv=SimpleNamespace(block_size=8))
            ],
        ),
    )
    compilation = SimpleNamespace(static_forward_context=context)
    return SimpleNamespace(
        speculative_config=cfg,
        draft_model_config=SimpleNamespace(architectures=["DFlash2DraftModel"]),
        vllm_config=SimpleNamespace(
            compilation_config=compilation,
            speculative_config=cfg,
            additional_config={},
            parallel_config=SimpleNamespace(
                tensor_parallel_size=2,
                data_parallel_size=1,
                pipeline_parallel_size=1,
                prefill_context_parallel_size=1,
                decode_context_parallel_size=1,
                use_sequence_parallel_moe=False,
            ),
        ),
        attn_vllm_config=SimpleNamespace(
            compilation_config=compilation,
            speculative_config=cfg,
            additional_config={},
            parallel_config=SimpleNamespace(
                tensor_parallel_size=2,
                data_parallel_size=1,
                pipeline_parallel_size=1,
                prefill_context_parallel_size=1,
                decode_context_parallel_size=1,
                use_sequence_parallel_moe=False,
            ),
        ),
        enable_adaptive_verification=False,
        use_fp64_gumbel=False,
        num_speculative_steps=7,
        num_query_per_req=8,
        top_k=2,
        max_num_reqs=4,
        vocab_size=8,
        _dflash2_full_kv_reference=True,
        _group_causal={0: False},
        model=model,
        block_tables=SimpleNamespace(
            block_sizes=[128],
            kernel_block_sizes=[128],
            blocks_per_kv_block=[1],
            input_block_tables=[TensorHeader((4, 1024), "torch.int32")],
        ),
        candidate_sampler=SimpleNamespace(
            num_steps=7,
            top_k=2,
        ),
        draft_logits=None,
        draft_attn_layer_names={"draft.0"},
    )


def route(native=2048):
    return SimpleNamespace(
        gid=0,
        native_window=native,
        table_column_span=128,
        layers=("draft.0",),
    )


def validate_source(module, owner, actual_route=None, *, window_size=1024):
    return module.validate_dflash2_graph_read_source(
        owner,
        actual_route or route(),
        window_size=window_size,
    )


def test_native_source_keeps_actual_mask_and_exact_boundary(module, owner):
    module.validate_dflash2_graph_source(owner)
    mask = owner.vllm_config.compilation_config.static_forward_context["draft.0"].impl._dflash2_native_swa_mask
    validate_source(module, owner)
    validate_source(module, owner, window_size=1913)
    with pytest.raises(ValueError, match="original noncausal native band"):
        validate_source(module, owner, window_size=1914)
    assert owner.vllm_config.compilation_config.static_forward_context["draft.0"].impl._dflash2_native_swa_mask is mask


def test_graph_factory_qualifies_the_real_dflash2_owner(monkeypatch, owner):
    from vllm_ascend.worker.v2.spec_decode import draft_window, draft_window_graph_runtime
    from vllm_ascend.worker.v2.spec_decode.draft_window_policy import resolve_draft_window_policy

    actual_route = route()
    monkeypatch.setattr(draft_window_graph_runtime, "validate_graph_context", lambda *_: None)
    monkeypatch.setattr(draft_window, "inventory_draft_window_groups", lambda **_: (actual_route,))
    kwargs = dict(
        policy=resolve_draft_window_policy(requested_window_size=1024, method="dflash", is_dflash2=True),
        runtime_route="dflash2",
        attn_groups=(),
        block_tables=owner.block_tables,
        kv_cache_config=None,
        vllm_config=owner.vllm_config,
        dflash2_graph_owner=owner,
    )
    source = draft_window._build_draft_window_source_capability_for_context(**kwargs)
    assert source.table is owner.block_tables.input_block_tables[0]
    owner.model.model.layers[0].attention_conv.block_size = 7
    with pytest.raises(ValueError, match="query boundary"):
        draft_window._build_draft_window_source_capability_for_context(**kwargs)
    kwargs["dflash2_graph_owner"] = None
    with pytest.raises(TypeError, match="actual model owner"):
        draft_window._build_draft_window_source_capability_for_context(**kwargs)


def test_full_only_source_host_contract_does_not_fake_native_model(module, owner):
    owner.vllm_config.compilation_config.static_forward_context["draft.0"].impl = dense_impl(None)
    module.validate_dflash2_graph_source(owner)
    validate_source(module, owner, route(None))


@pytest.mark.parametrize("sampling", ["greedy", "probabilistic"])
def test_actual_conditional_full_vocab_logits_allocation(module, owner, sampling):
    owner.speculative_config.draft_sample_method = sampling
    if sampling == "probabilistic":
        owner.draft_logits = TensorHeader((4, 7, 8), "torch.float32")
    module.validate_dflash2_graph_source(owner)
    owner.draft_logits = None if sampling == "probabilistic" else TensorHeader((4, 7, 8), "torch.float32")
    with pytest.raises(ValueError, match="full-vocab cache"):
        module.validate_dflash2_graph_source(owner)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (
            lambda x: setattr(x.speculative_config, "uses_dynamic_speculative_decoding", lambda: True),
            "static nonadaptive",
        ),
        (lambda x: setattr(x, "use_fp64_gumbel", True), "fp64 Gumbel"),
        (lambda x: setattr(x, "_draft_window_runtime", object()), "cannot coexist"),
        (lambda x: setattr(x, "_dflash2_full_kv_reference", False), "full-retention loader"),
        (lambda x: setattr(x.speculative_config, "enforce_eager", True), "typed eager owner"),
        (lambda x: setattr(x.model.model.layers[0].attention_conv, "block_size", 3), "query boundary"),
    ],
)
def test_malformed_capture_contract_is_rejected(module, owner, mutation, message):
    mutation(owner)
    with pytest.raises(ValueError, match=message):
        module.validate_dflash2_graph_source(owner)


@pytest.mark.parametrize("field,value", [("sinks", object()), ("use_bnsd_kv_cache", True)])
def test_unsupported_loaded_attention_fails_before_capture(module, owner, field, value):
    setattr(owner.vllm_config.compilation_config.static_forward_context["draft.0"].impl, field, value)
    with pytest.raises(ValueError, match="sinks/packed"):
        module.validate_dflash2_graph_source(owner)


def test_shared_inventory_native_band_and_actual_layers_are_checked(module, owner):
    module.validate_dflash2_graph_source(owner)
    with pytest.raises(ValueError, match="layer inventory"):
        validate_source(module, owner, route(1024))
    wrong = route()
    wrong.layers = ("draft.0", "draft.0")
    with pytest.raises(ValueError, match="layer inventory"):
        validate_source(module, owner, wrong)
    for declaration in (True, {0: True}, {1: False}):
        owner._group_causal = declaration
        with pytest.raises(ValueError, match="noncausal"):
            validate_source(module, owner)
    owner._group_causal = False
    validate_source(module, owner)


def test_actual_ascend_expanded_manager_blocks_use_kernel_span(module, owner):
    owner.block_tables.block_sizes = [512]
    owner.block_tables.blocks_per_kv_block = [4]
    actual_route = route()
    module.validate_dflash2_graph_source(owner)
    validate_source(module, owner, actual_route, window_size=1913)
    with pytest.raises(ValueError, match="original noncausal native band"):
        validate_source(module, owner, actual_route, window_size=1914)


def test_real_constructor_owns_expanded_complete_table_layout(monkeypatch):
    import torch

    from vllm_ascend.patch.worker.patch_v2 import patch_uva
    from vllm_ascend.worker.v2.block_table import AscendBlockTables

    original_zeros = torch.zeros

    def zeros_without_pinning(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return original_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", zeros_without_pinning)
    monkeypatch.setattr(patch_uva, "is_uva_available", lambda: True)
    monkeypatch.setattr(torch.ops._C_ascend, "get_npu_view_from_cpu_tensor", lambda value: value, raising=False)
    tables = AscendBlockTables(
        block_sizes=[512],
        kernel_block_sizes=[128],
        max_num_reqs=4,
        max_num_batched_tokens=32,
        max_num_blocks_per_group=[256],
        device=torch.device("cpu"),
    )
    assert tables.blocks_per_kv_block == [4]
    assert tables.input_block_tables[0].shape == (4, 1024)
    assert tables.input_block_tables[0].dtype == torch.int32
    assert tables.input_block_tables[0].stride() == (1024, 1)


def test_real_draft_attention_config_borrows_the_model_context():
    from vllm.config import AttentionConfig

    from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import AscendDFlash2Speculator

    owner = AscendDFlash2Speculator.__new__(AscendDFlash2Speculator)
    context = {"draft.0": object()}
    owner.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context=context),
        attention_config=AttentionConfig(),
    )
    owner.requires_non_causal = True
    derived = owner.attn_vllm_config
    assert derived.compilation_config.static_forward_context is context
    assert derived.attention_config.use_non_causal is True
    assert owner.vllm_config.attention_config.use_non_causal is False


@pytest.mark.parametrize(
    "field, value",
    [
        ("tensor_parallel_size", 4),
        ("data_parallel_size", 2),
        ("pipeline_parallel_size", 2),
        ("prefill_context_parallel_size", 2),
        ("decode_context_parallel_size", 2),
    ],
)
def test_first_graph_qualification_rejects_unvalidated_topologies(module, owner, field, value):
    setattr(owner.vllm_config.parallel_config, field, value)
    with pytest.raises(ValueError, match="actual Target/Draft TP2"):
        module.validate_dflash2_graph_config(owner.vllm_config)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda c: setattr(c.speculative_config.draft_parallel_config, "tensor_parallel_size", 1), "TP2"),
        (lambda c: setattr(c.speculative_config, "num_speculative_tokens", 3), "K7/Q8"),
        (lambda c: setattr(c.speculative_config, "enable_adaptive_verification", True), "K7/Q8"),
        (lambda c: c.additional_config.update(enable_dsa_cp=True), "FlashComm, SP"),
        (lambda c: c.additional_config.update(enable_flashcomm1=True), "FlashComm, SP"),
        (lambda c: setattr(c.parallel_config, "use_sequence_parallel_moe", True), "FlashComm, SP"),
        (
            lambda c: setattr(c.speculative_config.draft_parallel_config, "use_sequence_parallel_moe", True),
            "FlashComm, SP",
        ),
    ],
)
def test_first_graph_qualification_does_not_infer_k_or_features_from_name(module, owner, mutation, message):
    module.validate_dflash2_graph_config(owner.vllm_config)
    mutation(owner.vllm_config)
    with pytest.raises(ValueError, match=message):
        module.validate_dflash2_graph_config(owner.vllm_config)
