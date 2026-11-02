# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Host descriptors of real DFlash2 Graph inputs, without a device import.

These headers model fixed 544 allocations, not tensor contents or a numeric
oracle. Device capture, selector results and completion require device tests.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


class TensorHeader:
    """Only metadata is legal: no values, allocation, copy or synchronization."""

    def __init__(self, shape, dtype="torch.int64"):
        self.shape, self.dtype, self.device = tuple(shape), dtype, "npu:0"
        self.address = id(self)
        self.offset = 0
        self.contiguous = True

    def data_ptr(self):
        return self.address

    def untyped_storage(self):
        return self

    def storage_offset(self):
        return self.offset

    def stride(self):
        result, size = [], 1
        for dim in reversed(self.shape):
            result.append(size)
            size *= dim
        return tuple(reversed(result))

    def is_contiguous(self):
        return self.contiguous

    def __getattr__(self, name):
        raise AssertionError(f"contract tried a nonmetadata tensor operation: {name}")


class DenseImpl:
    def __init__(self, native=2048):
        self.sliding_window, self.sinks, self.use_bnsd_kv_cache = native, None, False
        if native is not None:
            self._dflash2_native_swa_mask = TensorHeader((2048, 2048), "torch.int8")


class Selector:
    top_k = 2

    def __call__(self):
        raise AssertionError("Host contract must not run the selector")


@pytest.fixture
def module(monkeypatch):
    path = Path(__file__).parents[3] / "vllm_ascend/worker/v2/spec_decode/draft_window_dflash2_graph.py"
    name = "_test_dflash2_graph_contract"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, loaded)
    spec.loader.exec_module(loaded)
    monkeypatch.setattr(loaded, "_attention_impl_type", lambda: DenseImpl)
    return loaded


@pytest.fixture
def owner():
    context = {"draft.0": SimpleNamespace(impl=DenseImpl())}
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
            scores=TensorHeader((4, 7, 2), "torch.float32"),
            cached_candidate_ids=TensorHeader((4, 7, 2)),
        ),
        sample_indices=TensorHeader((28,)),
        sample_pos=TensorHeader((28,)),
        sample_idx_mapping=TensorHeader((28,)),
        temperature=TensorHeader((4,), "torch.float32"),
        seeds=TensorHeader((4,)),
        draft_tokens=TensorHeader((4, 7)),
        draft_logits=None,
        draft_attn_layer_names={"draft.0"},
    )


def route(native=2048, *, scheduler=128):
    return SimpleNamespace(
        gid=0,
        native_window=native,
        scheduler_block_size=scheduler,
        table_column_span=128,
        layers=("draft.0",),
    )


def validate_source(contract, owner, actual_route=None, **kwargs):
    return contract.validate_source(
        actual_route or route(),
        causal=kwargs.pop("causal", {0: False}),
        window_size=kwargs.pop("window_size", 1024),
        **kwargs,
    )


def test_native_source_keeps_actual_mask_and_exact_boundary(module, owner):
    contract = module.validate_dflash2_graph_source(owner)
    mask = owner.vllm_config.compilation_config.static_forward_context["draft.0"].impl._dflash2_native_swa_mask
    validate_source(contract, owner)
    validate_source(contract, owner, window_size=1913)
    with pytest.raises(ValueError, match="original noncausal native band"):
        validate_source(contract, owner, window_size=1914)
    assert any(buffer is mask for buffer in contract.buffers)
    assert contract.native_windows == (("draft.0", 2048),)


def test_full_only_source_host_contract_does_not_fake_native_model(module, owner):
    owner.vllm_config.compilation_config.static_forward_context["draft.0"].impl = DenseImpl(None)
    contract = module.validate_dflash2_graph_source(owner)
    validate_source(contract, owner, route(None))


@pytest.mark.parametrize("sampling", ["greedy", "probabilistic"])
def test_actual_conditional_full_vocab_logits_allocation(module, owner, sampling):
    owner.speculative_config.draft_sample_method = sampling
    if sampling == "probabilistic":
        owner.draft_logits = TensorHeader((4, 7, 8), "torch.float32")
    contract = module.validate_dflash2_graph_source(owner)
    assert any(buffer is owner.draft_tokens for buffer in contract.buffers)
    assert any(buffer is owner.draft_logits for buffer in contract.buffers) is (sampling == "probabilistic")
    owner.draft_logits = None if sampling == "probabilistic" else TensorHeader((4, 7, 8), "torch.float32")
    with pytest.raises(ValueError, match="full-vocab cache"):
        module.validate_dflash2_graph_source(owner)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda x: setattr(x, "num_query_per_req", 5), "anchor query geometry"),
        (
            lambda x: setattr(x.speculative_config, "uses_dynamic_speculative_decoding", lambda: True),
            "static nonadaptive",
        ),
        (lambda x: setattr(x, "enable_adaptive_verification", True), "static nonadaptive"),
        (lambda x: setattr(x, "use_fp64_gumbel", True), "fp64 Gumbel"),
        (lambda x: setattr(x, "_draft_window_runtime", object()), "cannot coexist"),
        (lambda x: setattr(x, "_dflash2_full_kv_reference", False), "full-retention loader"),
        (lambda x: setattr(x.speculative_config, "enforce_eager", True), "typed eager owner"),
        (lambda x: setattr(x.model.model.layers[0].attention_conv, "block_size", 3), "query boundary"),
        (lambda x: setattr(x.candidate_sampler.scores, "shape", (4, 7, 3)), "shape/dtype"),
        (lambda x: setattr(x.sample_idx_mapping, "dtype", "torch.int32"), "shape/dtype"),
        (lambda x: setattr(x.seeds, "contiguous", False), "contiguous"),
        (
            lambda x: setattr(x.attn_vllm_config, "compilation_config", SimpleNamespace(static_forward_context={})),
            "static context",
        ),
    ],
)
def test_malformed_capture_contract_is_rejected(module, owner, mutation, message):
    mutation(owner)
    with pytest.raises(ValueError, match=message):
        module.validate_dflash2_graph_source(owner)


def test_contract_holds_actual_fixed_buffers_without_identity_mirrors(module, owner):
    contract = module.validate_dflash2_graph_source(owner)
    assert contract.buffers[0] is owner.candidate_sampler.scores
    assert any(buffer is owner.draft_tokens for buffer in contract.buffers)
    assert contract.native_windows == (("draft.0", 2048),)
    assert contract.query_width == 8
    assert not hasattr(contract, "validate_owner")


@pytest.mark.parametrize("field,value", [("sinks", object()), ("use_bnsd_kv_cache", True)])
def test_unsupported_loaded_attention_fails_before_capture(module, owner, field, value):
    setattr(owner.vllm_config.compilation_config.static_forward_context["draft.0"].impl, field, value)
    with pytest.raises(ValueError, match="sinks/packed"):
        module.validate_dflash2_graph_source(owner)


def test_shared_inventory_native_band_and_actual_layers_are_checked(module, owner):
    contract = module.validate_dflash2_graph_source(owner)
    with pytest.raises(ValueError, match="layer inventory"):
        validate_source(contract, owner, route(1024))
    wrong = route()
    wrong.layers = ("draft.0", "draft.0")
    with pytest.raises(ValueError, match="layer inventory"):
        validate_source(contract, owner, wrong)
    with pytest.raises(ValueError, match="noncausal"):
        validate_source(contract, owner, causal=True)


def test_actual_ascend_expanded_manager_blocks_use_kernel_span(module, owner):
    owner.block_tables.block_sizes = [512]
    owner.block_tables.blocks_per_kv_block = [4]
    actual_route = route(scheduler=512)
    contract = module.validate_dflash2_graph_source(owner)
    validate_source(contract, owner, actual_route, window_size=1913)
    with pytest.raises(ValueError, match="original noncausal native band"):
        validate_source(contract, owner, actual_route, window_size=1914)
    owner.block_tables.input_block_tables[0].shape = (4, 1023)
    with pytest.raises(ValueError, match="complete expanded"):
        validate_source(contract, owner, actual_route)


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
