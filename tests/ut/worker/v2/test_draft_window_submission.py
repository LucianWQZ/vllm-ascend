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
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops.triton.v2.spec_decode.draft_window", module)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(get_npu_format=lambda cache: 0))
    return module


def invoke(operators, operation, retain, monkeypatch):
    from vllm_ascend.worker.v2.spec_decode.draft_window_dense import DenseDraftRead, gather_dense_draft_kv

    if operation != "gather":
        raise AssertionError(f"Unknown read operation: {operation}")
    device = SimpleNamespace(type="npu")

    class DeviceTensor:
        def __init__(self, tensor):
            self.tensor = tensor
            self.device = device

        def __getattr__(self, name):
            return getattr(self.tensor, name)

    native_empty = torch.empty
    slots = DeviceTensor(torch.tensor([[1, 2, 3]]))
    key = DeviceTensor(native_empty(2, 4, 1, 8, dtype=torch.bfloat16))
    value = DeviceTensor(native_empty(2, 4, 1, 8, dtype=torch.bfloat16))
    monkeypatch.setattr(torch, "empty", lambda *args, device=None, **kwargs: native_empty(*args, **kwargs))
    read = DenseDraftRead(slots, torch.tensor([3]), torch.zeros(1, 1, 1, 3, dtype=torch.bool), 3, 1, retain)
    return gather_dense_draft_kv(read, key, value)


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
        invoke(operators, operation, lambda *values: held.extend(values), monkeypatch)
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
        invoke(operators, operation, retain, monkeypatch)
    assert calls == []
