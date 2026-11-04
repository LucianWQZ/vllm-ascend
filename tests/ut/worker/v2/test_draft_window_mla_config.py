# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
from contextlib import nullcontext
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from vllm_ascend.platform import _validate_draft_decode_context_parallel_config
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import _replicated_mla_draft_config


@pytest.mark.parametrize("window", [None, 1024])
@pytest.mark.parametrize("spec_eager,draft_eager", [(True, False), (False, True)])
def test_qualified_ordinary_mla_keeps_target_and_replicates_only_draft(window, spec_eager, draft_eager):
    draft = NS(use_mla=True, enforce_eager=draft_eager, hf_config=NS(architectures=["Glm5DSparkForCausalLM"]))
    speculative = NS(draft_model_config=draft, enforce_eager=spec_eager)
    parallel = NS(decode_context_parallel_size=16, prefill_context_parallel_size=1, tensor_parallel_size=16)
    config = NS(
        parallel_config=parallel, speculative_config=speculative, additional_config={"draft_window_size": window}
    )
    actual = _replicated_mla_draft_config(config)
    assert actual is not config and actual.parallel_config is not parallel
    assert parallel.decode_context_parallel_size == 16
    assert actual.parallel_config.decode_context_parallel_size == 1
    assert actual.parallel_config.prefill_context_parallel_size == 1
    assert actual.parallel_config.tensor_parallel_size == 16
    assert actual.speculative_config is speculative and speculative.draft_model_config is draft
    assert actual.additional_config is config.additional_config


@pytest.mark.parametrize(
    "mla,architecture,spec_eager,draft_eager,dcp",
    [
        (False, "Glm5DSparkForCausalLM", True, False, 16),
        (True, "OtherMLADraft", True, False, 16),
        (True, "Glm5DSparkForCausalLM", False, False, 16),
        (True, "Glm5DSparkForCausalLM", True, False, 1),
    ],
)
def test_other_draft_execution_configs_keep_upstream_identity(mla, architecture, spec_eager, draft_eager, dcp):
    draft = NS(use_mla=mla, enforce_eager=draft_eager, hf_config=NS(architectures=[architecture]))
    config = NS(
        parallel_config=NS(decode_context_parallel_size=dcp),
        speculative_config=NS(draft_model_config=draft, enforce_eager=spec_eager),
    )
    assert _replicated_mla_draft_config(config) is config
    assert config.parallel_config.decode_context_parallel_size == dcp


def gqa_draft_config(method, *, v2=True, draft_pcp=1, draft_dcp=1, dynamic=False):
    draft = NS(
        use_mla=False,
        model_arch_config=NS(total_num_attention_heads=64),
        get_total_num_kv_heads=lambda: 64,
    )
    return NS(
        use_v2_model_runner=v2,
        parallel_config=NS(tensor_parallel_size=16, prefill_context_parallel_size=1, decode_context_parallel_size=16),
        speculative_config=NS(
            method=method,
            draft_model_config=draft,
            draft_parallel_config=NS(
                tensor_parallel_size=16,
                prefill_context_parallel_size=draft_pcp,
                decode_context_parallel_size=draft_dcp,
            ),
            num_speculative_tokens_per_batch_size={1: 4} if dynamic else None,
        ),
    )


@pytest.mark.parametrize("method", ["dflash", "dspark"])
def test_replicated_gqa_draft_does_not_inherit_target_dcp_validation(method):
    config = gqa_draft_config(method)
    target, draft = config.parallel_config, config.speculative_config.draft_parallel_config
    _validate_draft_decode_context_parallel_config(config)
    assert config.parallel_config is target and target.decode_context_parallel_size == 16
    assert config.speculative_config.draft_parallel_config is draft
    assert draft.prefill_context_parallel_size == draft.decode_context_parallel_size == 1


@pytest.mark.parametrize(
    "method,v2,draft_pcp,draft_dcp",
    [
        ("dspark", False, 1, 1),
        ("eagle3", True, 1, 1),
        ("mtp", True, 1, 1),
        ("dspark", True, 1, 2),
        ("dflash", True, 2, 1),
    ],
)
def test_other_draft_dcp_routes_keep_the_original_head_validation(method, v2, draft_pcp, draft_dcp):
    config = gqa_draft_config(method, v2=v2, draft_pcp=draft_pcp, draft_dcp=draft_dcp)
    with pytest.raises(ValueError, match="tensor parallel size 16 must be greater than total num kv heads 64"):
        _validate_draft_decode_context_parallel_config(config)


def test_replicated_gqa_draft_keeps_dynamic_dcp_rejection():
    with pytest.raises(ValueError, match="Dynamic speculative decoding"):
        _validate_draft_decode_context_parallel_config(gqa_draft_config("dspark", dynamic=True))


@pytest.mark.parametrize(
    "mla,draft_dcp,target_dcp,filtered",
    [(True, 1, 16, True), (False, 1, 16, False), (True, 16, 16, False), (True, 1, 1, False)],
)
def test_only_replicated_mla_skips_target_context_draft_builders(monkeypatch, mla, draft_dcp, target_dcp, filtered):
    from vllm_ascend.patch.worker import patch_model_runner as patch

    speculator = Mock(spec=patch.upstream.DraftModelSpeculator)
    speculator.draft_attn_layer_names = {"draft"}
    speculator.draft_model_config = NS(use_mla=mla)
    speculator.dcp_size = draft_dcp
    cache_spec = NS(
        block_size=128,
        uses_slot_mapping=True,
        dcp_sharded=False,
        block_table_token_alignment=1,
        max_num_blocks_per_req=lambda *args: 16,
    )
    cache_config = NS(kv_cache_groups=[NS(layer_names=["target", "draft"], kv_cache_spec=cache_spec)])
    owner = NS(
        parallel_config=NS(cp_kv_cache_interleave_size=1, decode_context_parallel_size=target_dcp),
        speculator=speculator,
        max_model_len=2048,
        is_encoder_decoder=False,
        jit_warmup_registry=NS(activate=nullcontext),
        vllm_config=object(),
        device="cpu",
    )
    calls = []

    class ReachedOriginalBuilder(Exception):
        pass

    def stop_at_original_builder(*args, **kwargs):
        calls.append((args, kwargs))
        raise ReachedOriginalBuilder

    monkeypatch.setattr(patch.upstream, "init_attn_backend", stop_at_original_builder)
    with pytest.raises(ReachedOriginalBuilder):
        patch.initialize_kv_cache(owner, cache_config)
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert kwargs["active_layer_names"] == ({"target"} if filtered else None)
    assert kwargs["draft_layer_names"] is speculator.draft_attn_layer_names
    assert args[1] is owner.vllm_config
    assert owner.parallel_config.decode_context_parallel_size == target_dcp
    assert speculator.dcp_size == draft_dcp
    assert cache_config.kv_cache_groups[0].layer_names == ["target", "draft"]


@pytest.mark.parametrize("target_dcp,draft_dcp", [(16, 1), (16, 16), (1, 1)])
@pytest.mark.parametrize("parent_raises", [False, True])
def test_dflash_backend_selection_uses_actual_draft_context_and_restores_target(
    monkeypatch, target_dcp, draft_dcp, parent_raises
):
    from vllm.config import get_current_vllm_config, set_current_vllm_config

    from vllm_ascend.attention.attention_v1 import AscendAttentionBackend, AscendAttentionMetadataBuilder
    from vllm_ascend.attention.context_parallel.attention_cp import AscendAttentionDCPMetadataBuilder
    from vllm_ascend.worker.v2.spec_decode.dflash import speculator as module

    target = NS(parallel_config=NS(decode_context_parallel_size=target_dcp))
    draft = NS(parallel_config=NS(decode_context_parallel_size=draft_dcp))
    owner = module.AscendDFlashSpeculator.__new__(module.AscendDFlashSpeculator)
    owner.draft_kv_cache_group_ids, owner.max_num_tokens, owner.device = [0], 1, "cpu"
    monkeypatch.setattr(module.AscendDFlashSpeculator, "attn_vllm_config", property(lambda self: draft))
    inputs = tuple(object() for _ in range(5))
    seen = []

    class StopAtBoundary(Exception):
        pass

    def parent(actual_owner, *args):
        assert actual_owner is owner and args == inputs
        seen.append((get_current_vllm_config(), AscendAttentionBackend.get_builder_cls()))
        if parent_raises:
            raise StopAtBoundary("parent")

    def stop_before_allocation(*args, **kwargs):
        # A normal parent return must restore Target before child allocation.
        assert get_current_vllm_config() is target
        raise StopAtBoundary("allocation")

    monkeypatch.setattr(module.DFlashSpeculator, "set_attn", parent)
    monkeypatch.setattr(module.torch, "zeros", stop_before_allocation)
    target_builder = AscendAttentionDCPMetadataBuilder if target_dcp > 1 else AscendAttentionMetadataBuilder
    draft_builder = AscendAttentionDCPMetadataBuilder if draft_dcp > 1 else AscendAttentionMetadataBuilder
    with set_current_vllm_config(target):
        assert AscendAttentionBackend.get_builder_cls() is target_builder
        with pytest.raises(StopAtBoundary, match="parent" if parent_raises else "allocation"):
            owner.set_attn(*inputs)
        assert get_current_vllm_config() is target
        assert AscendAttentionBackend.get_builder_cls() is target_builder
    assert seen == [(draft, draft_builder)]
    assert target.parallel_config.decode_context_parallel_size == target_dcp
    assert draft.parallel_config.decode_context_parallel_size == draft_dcp
