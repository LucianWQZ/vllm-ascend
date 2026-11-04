# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
from types import SimpleNamespace as NS

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
