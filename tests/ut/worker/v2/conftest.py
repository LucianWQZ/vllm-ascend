# SPDX-License-Identifier: Apache-2.0
"""Shared fixture for the normal MRV2 factory and runner-selection contract."""

import ast
import importlib.util
import sys
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def factory(monkeypatch):
    from vllm_ascend import envs

    # The default entry contract must not depend on the test process runtime.
    monkeypatch.setattr(envs, "VLLM_ASCEND_EXPERIMENTAL_DRAFT_WINDOW_GRAPH", False)
    spec_dir = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/v2/spec_decode"
    # Execute the actual Ascend model-boundary predicate, not a test replica.
    utils_tree = ast.parse((spec_dir.parents[2] / "utils.py").read_text())
    predicate = next(
        node for node in utils_tree.body if isinstance(node, ast.FunctionDef) and node.name == "is_deepseek_v41"
    )
    utils = ModuleType("vllm_ascend.utils")
    utils.__dict__["Any"] = object
    exec(compile(ast.Module(body=[predicate], type_ignores=[]), "utils.py", "exec"), utils.__dict__)
    monkeypatch.setitem(sys.modules, "vllm_ascend.utils", utils)
    logging = ModuleType("vllm.logger")
    logging.__dict__["logger"] = Mock()
    monkeypatch.setitem(sys.modules, "vllm.logger", logging)
    config = ModuleType("vllm.config")
    CompilationMode = Enum("CompilationMode", ["NONE", "VLLM_COMPILE"])
    config.__dict__["CompilationMode"] = CompilationMode
    config.__dict__["VllmConfig"] = object
    monkeypatch.setitem(sys.modules, "vllm.config", config)
    ascend_config = ModuleType("vllm_ascend.ascend_config")
    effective = SimpleNamespace(draft_window_size=None)
    ascend_config.__dict__["get_ascend_config"] = lambda: effective
    monkeypatch.setitem(sys.modules, "vllm_ascend.ascend_config", ascend_config)
    name = "vllm_ascend.worker.v2.spec_decode.draft_window_policy"
    spec = importlib.util.spec_from_file_location(name, spec_dir / "draft_window_policy.py")
    assert spec is not None and spec.loader is not None
    policy = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, policy)
    spec.loader.exec_module(policy)
    spec = importlib.util.spec_from_file_location("_draft_window_factory_under_test", spec_dir / "__init__.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, effective, CompilationMode
