# SPDX-License-Identifier: Apache-2.0
"""Narrow CPU controls for the admitted dense Eagle AR capture descriptor."""

import ast
import copy
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from enum import Enum
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "vllm_ascend/worker/v2/spec_decode/target_dependent_ar/aclgraph.py"
CONFIG = ROOT / "vllm_ascend/ascend_config.py"

GUARD_PATH = ROOT / "vllm_ascend/worker/v2/spec_decode/draft_graph_review_limits.py"
guard_spec = importlib.util.spec_from_file_location("eagle_ar_support_test", GUARD_PATH)
assert guard_spec is not None and guard_spec.loader is not None
GUARD = importlib.util.module_from_spec(guard_spec)
sys.modules[guard_spec.name] = GUARD
guard_spec.loader.exec_module(GUARD)


class Mode(Enum):
    NONE = 0
    FULL = 1
    FULL_DECODE_ONLY = 2

    def decode_mode(self):
        return self.FULL if self is self.FULL_DECODE_ONLY else self


def function(path, name, namespace):
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def config(directory, maximum=1):
    model = copy.deepcopy(GUARD._CONSUMER_CONFIGS["eagle3"])
    path = Path(directory) / "config.json"
    path.write_text(json.dumps(model))
    hf = NS(**model)
    return NS(
        speculative_config=NS(
            method="eagle3",
            enforce_eager=False,
            num_speculative_tokens=3,
            uses_dynamic_speculative_decoding=lambda: False,
            enable_adaptive_verification=False,
            draft_model_config=NS(model=str(path.parent), hf_config=hf, hf_text_config=hf, dtype="torch.bfloat16"),
            draft_parallel_config=NS(
                tensor_parallel_size=4,
                data_parallel_size=1,
                use_sequence_parallel_moe=False,
                enable_expert_parallel=False,
            ),
        ),
        scheduler_config=NS(max_num_seqs=maximum, async_scheduling=True),
        parallel_config=NS(
            tensor_parallel_size=4,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            use_sequence_parallel_moe=True,
            enable_expert_parallel=True,
            all2all_backend="allgather_reducescatter",
        ),
        compilation_config=NS(
            cudagraph_mode=Mode.FULL_DECODE_ONLY, cudagraph_capture_sizes=[4, 8, 16], max_cudagraph_capture_size=16
        ),
        cache_config=NS(enable_prefix_caching=True),
        model_config=NS(max_model_len=4096),
        additional_config={},
    )


class EagleARCaptureTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.ns = dict(VllmConfig=object, CUDAGraphMode=Mode, copy=copy.copy, Any=object)
        geometry = function(CONFIG, "_eagle3_tp4_graph_review", self.ns)
        self.helper = function(SOURCE, "_eagle_ar_capture_config", self.ns)
        package = types.ModuleType("vllm_ascend")
        package.__dict__["envs"] = NS(VLLM_ASCEND_EXPERIMENTAL_DRAFT_WINDOW_GRAPH=True)
        module = types.ModuleType("vllm_ascend.ascend_config")
        module.__dict__["_eagle3_tp4_graph_review"] = geometry
        self.package = package
        self.modules = {
            "vllm_ascend": package,
            "vllm_ascend.ascend_config": module,
            "vllm_ascend.worker.v2.spec_decode.draft_graph_review_limits": GUARD,
        }

    def config(self, maximum=1):
        return config(self.directory.name, maximum)

    def call(self, conf, query=1, mode=Mode.FULL_DECODE_ONLY):
        with patch.dict(sys.modules, self.modules):
            return self.helper(conf, mode, query)

    def test_only_compilation_copy_keeps_target_first_and_real_topology(self):
        for maximum, expected in ((1, [1]), (2, [1, 2])):
            conf = self.config(maximum)
            before = list(conf.compilation_config.cudagraph_capture_sizes)
            actual = self.call(conf)
            self.assertIsNot(actual, conf)
            self.assertIsNot(actual.compilation_config, conf.compilation_config)
            self.assertEqual(actual.compilation_config.cudagraph_capture_sizes, expected)
            self.assertEqual(actual.compilation_config.max_cudagraph_capture_size, max(expected))
            self.assertEqual(conf.compilation_config.cudagraph_capture_sizes, before)
            self.assertIs(actual.parallel_config, conf.parallel_config)
            self.assertIs(actual.scheduler_config, conf.scheduler_config)
            self.assertIs(actual.parallel_config.use_sequence_parallel_moe, True)
            self.assertIs(self.call(conf, query=4), conf)

    def test_unchanged_off_eager_mtp_and_unadmitted_profiles(self):
        conf = self.config()
        self.assertIs(self.call(conf, mode=Mode.NONE), conf)
        for mutate in (
            lambda c: setattr(c.speculative_config, "method", "deepseek_mtp"),
            lambda c: setattr(c.speculative_config, "enforce_eager", True),
            lambda c: setattr(c.speculative_config, "num_speculative_tokens", 7),
        ):
            value = copy.deepcopy(conf)
            mutate(value)
            self.assertIs(self.call(value), value)
        value = self.config()
        value.parallel_config.tensor_parallel_size = 2
        with self.assertRaises(NotImplementedError):
            self.call(value)
        value = self.config()
        value.parallel_config.use_sequence_parallel_moe = False
        value.parallel_config.all2all_backend = "flashinfer_all2allv"
        self.assertIsNot(self.call(value), value)
        self.package.__dict__["envs"].VLLM_ASCEND_EXPERIMENTAL_DRAFT_WINDOW_GRAPH = False
        self.assertIs(self.call(conf), conf)

    def test_dp2_uses_actual_local_one_contract_without_changing_draft_dp(self):
        conf = self.config()
        conf.parallel_config.data_parallel_size = 2
        actual = self.call(conf)
        self.assertEqual(actual.compilation_config.cudagraph_capture_sizes, [1])
        self.assertIs(actual.parallel_config, conf.parallel_config)
        self.assertEqual(actual.parallel_config.data_parallel_size, 2)
        self.assertIs(actual.speculative_config.draft_parallel_config, conf.speculative_config.draft_parallel_config)
        self.assertEqual(actual.speculative_config.draft_parallel_config.data_parallel_size, 1)

    def test_no_invented_buckets_or_unknown_dense_layout(self):
        conf = self.config()
        for field in ("use_sequence_parallel_moe", "enable_expert_parallel"):
            value = copy.deepcopy(conf)
            setattr(value.speculative_config.draft_parallel_config, field, True)
            with self.assertRaises(NotImplementedError):
                self.call(value)
        conf.compilation_config.cudagraph_capture_sizes = [8, 16]
        with self.assertRaisesRegex(NotImplementedError, "Qualified eagle3"):
            self.call(conf)
        conf.compilation_config.cudagraph_capture_sizes = [True, 4]
        with self.assertRaises(NotImplementedError):
            self.call(conf)
        conf.compilation_config.cudagraph_capture_sizes = [4]
        conf.scheduler_config.max_num_seqs = True
        with self.assertRaises(NotImplementedError):
            self.call(conf)
        for maximum in (None, True, 0):
            conf = self.config()
            conf.compilation_config.max_cudagraph_capture_size = maximum
            with self.assertRaises(NotImplementedError):
                self.call(conf)

    def test_actual_constructor_adapts_before_parent_and_graph_param_registration(self):
        tree = ast.parse(SOURCE.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AutoRegressiveAclGraphManager")
        ctor = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        assignment = next(n for n in ctor.body if isinstance(n, ast.Assign))
        self.assertIsInstance(assignment.value, ast.Call)
        assert isinstance(assignment.value, ast.Call) and isinstance(assignment.value.func, ast.Name)
        self.assertEqual(assignment.value.func.id, "_eagle_ar_capture_config")
        parent_call = next(n for n in ctor.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call))
        assert isinstance(parent_call.value, ast.Call)
        self.assertLess(assignment.lineno, parent_call.lineno)


if __name__ == "__main__":
    unittest.main()
