# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
from types import SimpleNamespace as NS

import pytest

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
