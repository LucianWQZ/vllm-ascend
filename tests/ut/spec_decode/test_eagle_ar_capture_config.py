# SPDX-License-Identifier: Apache-2.0
"""Narrow CPU controls for the admitted dense Eagle AR capture descriptor."""

import ast
import copy
import importlib.util
import sys
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


def config(maximum=1):
    hf = NS(
        architectures=["LlamaForCausalLMEagle3"],
        hidden_size=2048,
        num_hidden_layers=1,
        head_dim=256,
        num_attention_heads=16,
        num_key_value_heads=2,
        fc_norm=True,
        use_sliding_window=False,
        sliding_window=None,
    )
    return NS(
        speculative_config=NS(
            method="eagle3",
            enforce_eager=False,
            num_speculative_tokens=3,
            uses_dynamic_speculative_decoding=lambda: False,
            draft_model_config=NS(hf_config=hf),
            draft_parallel_config=NS(
                tensor_parallel_size=4,
                data_parallel_size=1,
                use_sequence_parallel_moe=False,
                enable_expert_parallel=False,
            ),
        ),
        scheduler_config=NS(max_num_seqs=maximum),
        parallel_config=NS(
            tensor_parallel_size=4,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            use_sequence_parallel_moe=True,
        ),
        compilation_config=NS(
            cudagraph_mode=Mode.FULL_DECODE_ONLY, cudagraph_capture_sizes=[4, 8, 16], max_cudagraph_capture_size=16
        ),
        additional_config={},
        _ascend_draft_graph_support=GUARD.DraftGraphSupport(4096, 3, maximum, 1, "eagle3", 4),
    )


class EagleARCaptureTests(unittest.TestCase):
    def setUp(self):
        self.ns = dict(VllmConfig=object, CUDAGraphMode=Mode, copy=copy.copy, Any=object)
        geometry = function(CONFIG, "_eagle3_tp4_graph_review", self.ns)
        self.helper = function(SOURCE, "_eagle_ar_capture_config", self.ns)
        package = types.ModuleType("vllm_ascend")
        package.envs = NS(VLLM_ASCEND_EXPERIMENTAL_DRAFT_WINDOW_GRAPH=True)
        module = types.ModuleType("vllm_ascend.ascend_config")
        module._eagle3_tp4_graph_review = geometry
        self.package = package
        self.modules = {
            "vllm_ascend": package,
            "vllm_ascend.ascend_config": module,
            "vllm_ascend.worker.v2.spec_decode.draft_graph_review_limits": GUARD,
        }

    def call(self, conf, query=1, mode=Mode.FULL_DECODE_ONLY):
        with patch.dict(sys.modules, self.modules):
            return self.helper(conf, mode, query)

    def test_only_compilation_copy_keeps_target_first_and_real_topology(self):
        for maximum, expected in ((1, [1]), (2, [1, 2])):
            conf = config(maximum)
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
        conf = config()
        self.assertIs(self.call(conf, mode=Mode.NONE), conf)
        for mutate in (
            lambda c: setattr(c.speculative_config, "method", "deepseek_mtp"),
            lambda c: setattr(c.speculative_config, "enforce_eager", True),
            lambda c: setattr(c.speculative_config, "num_speculative_tokens", 7),
            lambda c: setattr(c.parallel_config, "data_parallel_size", 2),
            lambda c: setattr(c.parallel_config, "tensor_parallel_size", 2),
        ):
            value = copy.deepcopy(conf)
            mutate(value)
            self.assertIs(self.call(value), value)
        value = config()
        value.parallel_config.use_sequence_parallel_moe = False
        self.assertIsNot(self.call(value), value)
        self.package.envs.VLLM_ASCEND_EXPERIMENTAL_DRAFT_WINDOW_GRAPH = False
        self.assertIs(self.call(conf), conf)

    def test_dp2_uses_the_cached_local_one_contract_without_changing_draft_dp(self):
        conf = config()
        conf.parallel_config.data_parallel_size = 2
        conf._ascend_draft_graph_support = GUARD.DraftGraphSupport(4096, 3, 1, 2, "eagle3", 4)
        actual = self.call(conf)
        self.assertEqual(actual.compilation_config.cudagraph_capture_sizes, [1])
        self.assertIs(actual.parallel_config, conf.parallel_config)
        self.assertEqual(actual.parallel_config.data_parallel_size, 2)
        self.assertIs(actual.speculative_config.draft_parallel_config, conf.speculative_config.draft_parallel_config)
        self.assertEqual(actual.speculative_config.draft_parallel_config.data_parallel_size, 1)

    def test_no_invented_buckets_or_unknown_dense_layout(self):
        conf = config()
        for field in ("use_sequence_parallel_moe", "enable_expert_parallel"):
            value = copy.deepcopy(conf)
            setattr(value.speculative_config.draft_parallel_config, field, True)
            with self.assertRaises(NotImplementedError):
                self.call(value)
        conf.compilation_config.cudagraph_capture_sizes = [8, 16]
        with self.assertRaisesRegex(NotImplementedError, "no admitted"):
            self.call(conf)
        conf.compilation_config.cudagraph_capture_sizes = [True, 4]
        with self.assertRaises(ValueError):
            self.call(conf)
        conf.compilation_config.cudagraph_capture_sizes = [4]
        conf.scheduler_config.max_num_seqs = True
        with self.assertRaises(ValueError):
            self.call(conf)
        for maximum in (None, True, 0):
            conf = config()
            conf.compilation_config.max_cudagraph_capture_size = maximum
            with self.assertRaises(ValueError):
                self.call(conf)

    def test_actual_constructor_adapts_before_parent_and_graph_param_registration(self):
        tree = ast.parse(SOURCE.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AutoRegressiveAclGraphManager")
        ctor = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        assignment = next(n for n in ctor.body if isinstance(n, ast.Assign))
        self.assertIsInstance(assignment.value, ast.Call)
        self.assertEqual(assignment.value.func.id, "_eagle_ar_capture_config")
        parent_call = next(n for n in ctor.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call))
        self.assertLess(assignment.lineno, parent_call.lineno)


if __name__ == "__main__":
    unittest.main()
