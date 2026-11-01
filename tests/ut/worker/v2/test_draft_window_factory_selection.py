# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Join actual runner selection to the actual MRV2 factory's policy boundary.

V1 selection is checked without constructing a V1 engine. Device constructors
are isolated; successful dispatch does not claim backend or runtime support.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


from tests.ut.worker.v2.test_draft_window_factory import _config, _constructor
from tests.ut.worker.v2.test_draft_window_factory import factory as factory


@pytest.fixture
def selector(monkeypatch, factory):
    # The selector itself runs unmodified. Only its configuration/environment
    # import seams are isolated, as in the factory source-contract fixture.
    vllm = ModuleType("vllm")
    vllm.__path__ = []
    envs = ModuleType("vllm.envs")
    vllm.__dict__["envs"] = envs
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.envs", envs)
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/mrv2_utils.py"
    spec = importlib.util.spec_from_file_location("_actual_window_runner_selector", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, envs


@pytest.mark.parametrize("override", [None, False, True], ids=["auto", "explicit-v1", "explicit-v2"])
@pytest.mark.parametrize(
    "case",
    [
        "off",
        "eagle-off",
        "dflash2-graph-off",
        "eligible-on",
        "mtp-forced-off",
        "v4-forced-off",
        "plain-eagle-on",
        "dflash2-eager-on",
        "dflash2-graph-on",
        "draft-graph-on",
        "dynamic-k-on",
    ],
)
def test_selector_and_factory_window_contract(selector, factory, monkeypatch, override, case):
    selection, envs = selector
    module, effective, _ = factory
    envs.VLLM_USE_V2_MODEL_RUNNER = override
    method, arch, route, cls = "dspark", "Qwen3DSparkModel", "dspark", "AscendDSparkSpeculator"
    window: int | str | None = (
        None if case.endswith("off") and case not in {"mtp-forced-off", "v4-forced-off"} else 1024
    )
    eager = case not in {"dflash2-graph-off", "dflash2-graph-on", "draft-graph-on"}
    if case in {"eagle-off", "plain-eagle-on"}:
        method, arch, route, cls = "eagle", "EagleDraftModel", "eagle", "AscendEagleSpeculator"
    elif case.startswith("dflash2"):
        method, arch, route, cls = "dflash", "DFlash2DraftModel", "dflash2", "AscendDFlash2Speculator"
    elif case == "mtp-forced-off":
        method, route, cls, window, eager = "mtp", "mtp", "AscendMTPSpeculator", "invalid", False
    if case == "v4-forced-off":
        config = _config(method="dspark", eager=False, model_type="deepseek_v4", draft_arch="DSparkDraftModel")
        window = "invalid"
    else:
        config = _config(method=method, eager=eager, draft_arch=arch)
    dynamic = case == "dynamic-k-on"
    config.speculative_config.uses_dynamic_speculative_decoding.return_value = dynamic
    if dynamic:
        config.speculative_config.num_speculative_tokens_per_batch_size = [[1, 8, 2]]
    effective.draft_window_size = window
    config.additional_config = {"draft_window_size": window}
    constructor = _constructor(monkeypatch, route, cls)
    selected_v2 = selection.use_v2_model_runner(config)
    expected_auto = case not in {"dflash2-graph-off", "dflash2-graph-on"}
    assert selected_v2 is (expected_auto if override is None else override)
    if not selected_v2:
        # Runner dispatch does not call the V2 factory on the V1 branch.
        constructor.assert_not_called()
        return
    rejected = case in {"plain-eagle-on", "dflash2-graph-on", "draft-graph-on", "dynamic-k-on"}
    if rejected:
        with pytest.raises(NotImplementedError):
            module.init_speculator(config, "cpu")
        constructor.assert_not_called()
    else:
        result = module.init_speculator(config, "cpu")
        assert result is constructor.return_value
        assert hasattr(result, "draft_window_policy") is (case in {"eligible-on", "dflash2-eager-on"})
        assert not hasattr(result, "_draft_window_runtime")
    assert config.compilation_config.mode == "target-graph"
