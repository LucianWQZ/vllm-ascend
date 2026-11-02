# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Private outputs must be owned before a launcher can partially submit.

Only the dispatch boundary is replaced here. Real device enqueue and event
completion require the corresponding NPU tests.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


class DispatchFailure(RuntimeError):
    pass


class RetentionFailure(RuntimeError):
    pass


@pytest.fixture
def operators(monkeypatch):
    fake_triton = SimpleNamespace(
        jit=lambda function: function,
        cdiv=lambda numerator, divisor: (numerator + divisor - 1) // divisor,
        next_power_of_2=lambda value: 1 << (value - 1).bit_length(),
    )
    monkeypatch.setitem(
        sys.modules, "vllm.triton_utils", SimpleNamespace(triton=fake_triton, tl=SimpleNamespace(constexpr=int))
    )
    root = Path(__file__).resolve().parents[4]
    path = root / "vllm_ascend/ops/triton/v2/spec_decode/draft_window.py"
    spec = importlib.util.spec_from_file_location("draft_window_submission_contract", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def invoke(operators, operation, retain):
    if operation == "gather":
        return operators.gather_draft_kv_fp32(
            torch.tensor([[1, 2, 3]]), torch.empty(2, 4, 1, 8), torch.empty(2, 4, 1, 8), retain=retain
        )
    raise AssertionError(f"Unknown read operation: {operation}")


@pytest.mark.parametrize("operation,outputs", [("gather", 2)])
def test_outputs_are_retained_before_dispatch_can_raise(monkeypatch, operators, operation, outputs):
    held: list[torch.Tensor] = []

    class PartialDispatch:
        def __getitem__(self, grid):
            def launch(*args):
                assert len(held) >= outputs
                # The output arguments belong to this real call, rather than
                # a previously retained allocation with an equal shape.
                expected = args[3:5]
                assert all(any(value is output for value in held) for output in expected)
                raise DispatchFailure("submission can fail after enqueue")

            return launch

    kernel = {"gather": "_gather_draft_kv_fp32"}
    monkeypatch.setattr(operators, kernel[operation], PartialDispatch())
    with pytest.raises(DispatchFailure):
        invoke(operators, operation, lambda *values: held.extend(values))
    assert len([value for value in held if isinstance(value, torch.Tensor)]) >= outputs


@pytest.mark.parametrize("operation", ["gather"])
def test_retention_failure_prevents_dispatch(monkeypatch, operators, operation):
    calls = []

    class Dispatch:
        def __getitem__(self, grid):
            calls.append(grid)
            return lambda *args: None

    def retain(*values):
        raise RetentionFailure("owner cannot accept new storage")

    kernel = {"gather": "_gather_draft_kv_fp32"}
    monkeypatch.setattr(operators, kernel[operation], Dispatch())
    with pytest.raises(RetentionFailure):
        invoke(operators, operation, retain)
    assert calls == []
