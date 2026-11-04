# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QuaRot DFlash loading: basis direction, load order and caller isolation."""

from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from vllm_ascend import utils
from vllm_ascend.models import qwen3_dspark


def load_draft(monkeypatch, model, config=None):
    from vllm_ascend.worker.v2.spec_decode.dflash import speculator as module

    monkeypatch.setattr(module, "disable_profiling_chunk_for_draft", lambda config: nullcontext())
    monkeypatch.setattr(module.DFlashSpeculator, "load_draft_model", lambda self, target, layers: model)
    owner = module.AscendDFlashSpeculator.__new__(module.AscendDFlashSpeculator)
    owner.vllm_config = config or SimpleNamespace()
    return owner.load_draft_model(object(), {"target.attention"})


def test_non_quarot_has_no_weight_or_alignment_state(monkeypatch):
    monkeypatch.setattr(utils, "get_rotation_path", lambda config: None)
    forbidden = Mock(side_effect=AssertionError("Non-QuaRot must not touch weights"))
    monkeypatch.setattr(qwen3_dspark, "align_draft_weights", forbidden)
    model = SimpleNamespace()
    assert load_draft(monkeypatch, model) is model
    assert vars(model) == {}
    forbidden.assert_not_called()


def test_loaded_aux_blocks_use_correct_rotation_and_keep_owned_vocab(monkeypatch):
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])  # Orthogonal, deliberately not symmetric.
    monkeypatch.setattr(utils, "get_rotation_path", lambda config: Path("rotation.safetensors"))
    monkeypatch.setattr(qwen3_dspark, "get_rotation_path", utils.get_rotation_path)
    rotation_loader = Mock(return_value=rotation)
    monkeypatch.setattr(qwen3_dspark, "get_rotation_matrix", rotation_loader)
    model = nn.Module()
    model.model = nn.Module()
    model.model.use_aux_hidden_state = True
    model.model.fc = nn.Linear(4, 2, bias=False)
    model.model.embed_tokens = nn.Embedding(4, 2)
    model.lm_head = nn.Linear(2, 4, bias=False)
    model.has_own_embed_tokens = model.has_own_lm_head = True
    original = torch.tensor([[1.0, 2.0, 3.0, 4.0], [-2.0, 3.0, 4.0, 1.0]])
    with torch.no_grad():
        model.model.fc.weight.copy_(original)
    embed, head = model.model.embed_tokens.weight.clone(), model.lm_head.weight.clone()
    hidden = torch.tensor([[1.0, 3.0, 2.0, 4.0], [-1.0, 2.0, 6.0, -2.0]])
    rotated = (hidden.view(2, 2, 2) @ rotation).reshape(2, 4)
    config = SimpleNamespace(model_config=SimpleNamespace(hf_text_config=SimpleNamespace()))
    assert load_draft(monkeypatch, model, config) is model
    torch.testing.assert_close(rotated @ model.model.fc.weight.T, hidden @ original.T, atol=0, rtol=0)
    torch.testing.assert_close(model.model.embed_tokens.weight, embed)
    torch.testing.assert_close(model.lm_head.weight, head)
    rotation_loader.assert_called_once()


def test_nonaux_retains_original_loading_path(monkeypatch):
    monkeypatch.setattr(utils, "get_rotation_path", lambda config: Path("rotation.safetensors"))
    forbidden = Mock(side_effect=AssertionError("Nonaux loading must keep its original path"))
    monkeypatch.setattr(qwen3_dspark, "align_draft_weights", forbidden)
    model = SimpleNamespace(model=SimpleNamespace(use_aux_hidden_state=False))
    assert load_draft(monkeypatch, model) is model
    assert vars(model.model) == {"use_aux_hidden_state": False}
    forbidden.assert_not_called()


@pytest.mark.parametrize("failure", ["missing_projection", "packed_weight", "loader_error"])
def test_invalid_or_failed_alignment_never_returns_a_loaded_model(monkeypatch, failure):
    monkeypatch.setattr(utils, "get_rotation_path", lambda config: Path("rotation.safetensors"))
    model = SimpleNamespace(model=SimpleNamespace(use_aux_hidden_state=True, fc=nn.Linear(4, 2, bias=False)))
    if failure == "missing_projection":
        del model.model.fc
    elif failure == "packed_weight":
        model.model.fc.weight = nn.Parameter(torch.ones(2, 4, dtype=torch.int8), requires_grad=False)
    else:
        monkeypatch.setattr(qwen3_dspark, "align_draft_weights", Mock(side_effect=RuntimeError("load failed")))
    with pytest.raises((ValueError, RuntimeError)):
        load_draft(monkeypatch, model)


@pytest.mark.parametrize("window", [None, 1024])
def test_mrv2_loader_aligns_after_original_load_and_context_restore(monkeypatch, window):
    from vllm_ascend.worker.v2.spec_decode.dflash import speculator as module

    events = []

    @contextmanager
    def profile_context(config):
        events.append("enter")
        yield
        events.append("exit")

    model = SimpleNamespace(model=SimpleNamespace(use_aux_hidden_state=True, fc=nn.Linear(4, 2, bias=False)))
    monkeypatch.setattr(module, "disable_profiling_chunk_for_draft", profile_context)

    def load(self, target, layers):
        assert events == ["enter"]
        events.append("load")
        return model

    monkeypatch.setattr(module.DFlashSpeculator, "load_draft_model", load)
    monkeypatch.setattr(utils, "get_rotation_path", lambda config: Path("rotation.safetensors"))
    monkeypatch.setattr(qwen3_dspark, "align_draft_weights", lambda loaded, projection, config: events.append("align"))
    speculator = module.AscendDFlashSpeculator.__new__(module.AscendDFlashSpeculator)
    speculator.vllm_config = SimpleNamespace(additional_config={"draft_window_size": window})
    assert speculator.load_draft_model(object(), {"target.attention"}) is model
    assert events == ["enter", "load", "exit", "align"]
