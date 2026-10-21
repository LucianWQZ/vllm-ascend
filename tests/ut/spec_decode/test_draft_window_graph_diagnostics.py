# SPDX-License-Identifier: Apache-2.0
"""Opt-in output references are not a new graph output/completion contract."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode

from vllm_ascend.worker.v2.spec_decode.draft_window_graph_diagnostics import DraftGraphOutputReferences
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator


def test_real_forward_default_off_and_retained_capture_references(monkeypatch):
    import vllm.v1.worker.gpu.spec_decode.dflash.speculator as upstream

    @contextmanager
    def context(*args, **kwargs):
        yield

    monkeypatch.setattr(upstream, "set_forward_context", context)
    owner = AscendDSparkSpeculator.__new__(AscendDSparkSpeculator)
    owner.vllm_config = object()
    owner.input_buffers = SimpleNamespace(input_ids=torch.arange(28), positions=torch.arange(28))

    class Model(torch.nn.Module):
        def forward(self, input_ids, positions, inputs_embeds):
            return torch.stack((input_ids, positions), 1).to(torch.bfloat16)

    owner.model = Model()
    assert owner._run_model(7, {}, {}, None).shape == (7, 2)
    assert not hasattr(owner, "_draft_graph_output_references")
    sink = owner._draft_graph_output_references = DraftGraphOutputReferences(owner)
    outputs = {}
    for n in (7, 14, 28):
        owner._run_model(n, {}, {}, None)  # eager warmup not captured
        assert n not in sink.outputs
        outputs[n] = owner._run_model(n, {}, {}, None, CUDAGraphMode.FULL)
    sink.freeze([7, 14, 28])
    assert all(sink.get(owner, n) is outputs[n] for n in outputs)
    with pytest.raises(RuntimeError, match="owner/scope"):
        sink.retain(owner, 7, outputs[7])
    with pytest.raises(RuntimeError, match="foreign"):
        sink.get(object(), 7)


def test_inventory_missing_bucket_and_wrong_output_rejected():
    owner = object()
    sink = DraftGraphOutputReferences(owner)
    with pytest.raises(RuntimeError, match="output contract"):
        sink.retain(owner, 7, None)
    sink.retain(owner, 7, torch.zeros(7, 4))
    with pytest.raises(RuntimeError, match="inventory"):
        sink.freeze([7, 14, 28])


def test_capture_configuration_gate_is_not_bypass(monkeypatch):
    import vllm_ascend.worker.v2.spec_decode.dspark.speculator as module

    owner = AscendDSparkSpeculator.__new__(AscendDSparkSpeculator)
    owner.vllm_config = SimpleNamespace(additional_config={"draft_window_graph_numeric_diagnostic": True})
    monkeypatch.setattr(module, "experimental_draft_window_graph_characterization_enabled", lambda: False)
    with pytest.raises(RuntimeError, match="experimental Graph gate"):
        owner.capture()
    assert not hasattr(owner, "_draft_graph_output_references")
