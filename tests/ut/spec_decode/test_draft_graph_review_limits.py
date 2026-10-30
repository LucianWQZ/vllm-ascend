# SPDX-License-Identifier: Apache-2.0
"""Actual checkpoint review guards; no framework/device imports."""

from __future__ import annotations

import ast
import runpy
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import patch

from pydantic import TypeAdapter

SOURCE = Path(__file__).resolve().parents[3]
SPEC = SOURCE / "vllm_ascend/worker/v2/spec_decode"
MODULE = "vllm_ascend.worker.v2.spec_decode.draft_graph_review_limits"
GUARD = runpy.run_path(str(SPEC / "draft_graph_review_limits.py"))


def actual_function(path, name, namespace):
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name]
    tree = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    return namespace[name]


class ReviewBoundaryControls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.model = Path(self.tmp.name)
        self.data = b'{"fixed_test_config":true}\n'
        (self.model / "config.json").write_bytes(self.data)
        self.digest = GUARD["hashlib"].sha256(self.data).hexdigest()
        self.config = NS(
            speculative_config=NS(
                method="dflash",
                enforce_eager=False,
                draft_model_config=NS(
                    model=str(self.model),
                    hf_config=NS(layer_types=["full_attention"] * 8, use_sliding_window=False, sliding_window=None),
                ),
                num_speculative_tokens=7,
                enable_adaptive_verification=False,
                uses_dynamic_speculative_decoding=lambda: False,
                draft_parallel_config=NS(tensor_parallel_size=4),
            ),
            parallel_config=NS(
                tensor_parallel_size=4,
                data_parallel_size=1,
                pipeline_parallel_size=1,
                prefill_context_parallel_size=1,
                decode_context_parallel_size=1,
                enable_expert_parallel=True,
            ),
            scheduler_config=NS(max_num_seqs=1, async_scheduling=False),
            cache_config=NS(enable_prefix_caching=False),
            additional_config={"draft_window_size": 32768},
            compilation_config=NS(cudagraph_mode=NS(name="FULL_DECODE_ONLY")),
        )
        # Use the actual lightweight bool validator without importing the
        # framework's AscendConfig module into this guard-only test.
        config_module = ModuleType("vllm_ascend.ascend_config")
        config_module.validate_additional_config_bool = actual_function(
            SOURCE / "vllm_ascend/ascend_config.py",
            "validate_additional_config_bool",
            {"TypeAdapter": TypeAdapter},
        )
        config_patch = patch.dict(sys.modules, {"vllm_ascend.ascend_config": config_module})
        config_patch.start()
        self.addCleanup(config_patch.stop)
        self.resolve = GUARD["resolve_draft_graph_review_limit"]
        self.validate = GUARD["validate_draft_graph_review_requests"]

    def identified(self, name):
        return patch.dict(self.resolve.__globals__, {name: self.digest})

    def test_full8_one_actual_row_requires_qualified_padding_and_rejects_multirow(self):
        with self.identified("_FULL8_CONFIG"):
            for gate in (False, True):
                with (
                    self.subTest(gate=gate),
                    self.assertRaisesRegex(NotImplementedError, "PR017/vLLM544 pair is incomplete"),
                ):
                    self.resolve(self.config, experimental_graph_review=gate)
        # Keep the internal row guard for a later qualified entry; its shape
        # checks do not by themselves admit the ordinary checkpoint service.
        owner = NS(_draft_graph_review_limit="full8_local_active_at_most_one")
        self.validate(owner, NS(num_reqs=0, num_reqs_after_padding=1))
        self.validate(owner, NS(num_reqs=1, num_reqs_after_padding=1, global_num_reqs=1, client_concurrency=64))
        for padded in (None, 0, 2, 4, 8, True, "1"):
            with self.subTest(padded=padded), self.assertRaisesRegex(NotImplementedError, "actual=1/padded=1"):
                self.validate(owner, NS(num_reqs=1, num_reqs_after_padding=padded))
        for active in (2, 3, 4, 8):
            for padded in (1, 4, 8):
                with (
                    self.subTest(active=active, padded=padded),
                    self.assertRaisesRegex(NotImplementedError, "1 actual request per DP"),
                ):
                    self.validate(owner, NS(num_reqs=active, num_reqs_after_padding=padded))
        with self.assertRaisesRegex(RuntimeError, "unknown temporary"):
            self.validate(NS(_draft_graph_review_limit="full8_local_active_at_most_three"), NS(num_reqs=1))

    def test_full8_shared_config_rejects_unqualified_scheduling_before_constructor(self):
        validate = GUARD["validate_full8_single_request_review_config"]
        validate(self.config)
        cases = (
            (self.config.scheduler_config, "max_num_seqs", 2),
            (self.config.scheduler_config, "max_num_seqs", True),
            (self.config.scheduler_config, "async_scheduling", True),
            (self.config.cache_config, "enable_prefix_caching", True),
            (self.config.parallel_config, "tensor_parallel_size", 8),
            (self.config.parallel_config, "data_parallel_size", 2),
            (self.config.parallel_config, "pipeline_parallel_size", 2),
            (self.config.parallel_config, "prefill_context_parallel_size", 2),
            (self.config.parallel_config, "decode_context_parallel_size", 2),
            (self.config.speculative_config, "num_speculative_tokens", 8),
            (self.config.speculative_config, "enable_adaptive_verification", True),
            (self.config.parallel_config, "enable_expert_parallel", False),
        )
        for obj, key, bad in cases:
            original = getattr(obj, key)
            setattr(obj, key, bad)
            with self.subTest(key=key, bad=bad), self.assertRaises(NotImplementedError):
                validate(self.config)
            setattr(obj, key, original)
        for key, bad in (("draft_window_size", 1024), ("enable_dsa_cp", True), ("enable_flashcomm1", True)):
            old = self.config.additional_config.copy()
            self.config.additional_config[key] = bad
            with self.subTest(key=key), self.assertRaises(NotImplementedError):
                validate(self.config)
            self.config.additional_config = old

    def test_ordinary_dp1_startup_uses_the_exact_checkpoint_and_single_config_guard(self):
        module = ModuleType(MODULE)
        module.resolve_draft_graph_review_limit = self.resolve
        dense = actual_function(
            SOURCE / "vllm_ascend/ascend_config.py",
            "_dense_dp_graph_window_review",
            {"validate_additional_config_bool": lambda value, name: value},
        )
        with patch.dict(sys.modules, {MODULE: module}), self.identified("_FULL8_CONFIG"):
            with self.assertRaisesRegex(NotImplementedError, "PR017/vLLM544 pair is incomplete"):
                dense(self.config, 32768)
            self.config.scheduler_config.max_num_seqs = 2
            with self.assertRaisesRegex(NotImplementedError, "max_num_seqs=1"):
                dense(self.config, 32768)
            # Existing explicit numerical DP1 profile remains independent.
            self.config.additional_config["draft_window_graph_numeric_diagnostic"] = True
            self.assertTrue(dense(self.config, 32768))
            del self.config.additional_config["draft_window_graph_numeric_diagnostic"]
            self.config.scheduler_config.max_num_seqs = 1
            self.config.speculative_config.enforce_eager = True
            self.assertFalse(dense(self.config, 32768))
            self.config.speculative_config.enforce_eager = False
            self.assertFalse(dense(self.config, 1024))
            self.config.parallel_config.data_parallel_size = 2
            self.assertTrue(dense(self.config, 32768))
            self.config.parallel_config.data_parallel_size = 1
        # A dense model alone is not a checkpoint identity or DP1 admission.
        with patch.dict(sys.modules, {MODULE: module}):
            self.assertFalse(dense(self.config, 32768))

    def test_invalid_active_counts_do_not_coerce_or_disable_guard(self):
        owner = NS(_draft_graph_review_limit="full8_local_active_at_most_one")
        for active in (-1, True, 3.0, "3"):
            with self.assertRaises(ValueError):
                self.validate(owner, NS(num_reqs=active))

    def test_dummy_profile_and_capture_capacity_are_not_business_rows(self):
        owner = NS(_draft_graph_review_limit="full8_local_active_at_most_one")
        self.validate(owner, NS(num_reqs=16), dummy_run=True)
        self.validate(owner, NS(num_reqs=16), is_profile=True)
        self.validate(owner, NS(num_reqs=16, is_dummy=True))
        # Capture does not call propose/owner_scope; no capture hook is changed.
        self.validate(NS(), NS(num_reqs=16))

    def test_swa_exact_configuration_rejects_graph_but_not_eager_or_target_graph(self):
        with self.identified("_SWA6_CONFIG"):
            with self.assertRaisesRegex(NotImplementedError, "SWA6 checkpoint"):
                self.resolve(self.config)
            self.config.speculative_config.enforce_eager = True
            self.assertIsNone(self.resolve(self.config))
            self.config.speculative_config.enforce_eager = False
            self.config.compilation_config.cudagraph_mode = NS(name="NONE")
            self.assertIsNone(self.resolve(self.config))

    def test_eagle_ordinary_entry_pending_qualification_not_silent_eager(self):
        self.config.speculative_config.method = "eagle3"
        with self.identified("_EAGLE_CONFIG"):
            with self.assertRaisesRegex(NotImplementedError, "pending single-request qualification"):
                self.resolve(self.config)
            self.config.speculative_config.enforce_eager = True
            self.assertIsNone(self.resolve(self.config))

    def test_explicit_existing_numeric_diagnostic_retains_original_interface(self):
        for checkpoint in ("_SWA6_CONFIG", "_FULL8_CONFIG"):
            with self.subTest(checkpoint=checkpoint), self.identified(checkpoint):
                self.config.additional_config["draft_window_graph_numeric_diagnostic"] = True
                with self.assertRaises(NotImplementedError):
                    self.resolve(self.config)
                self.assertIsNone(self.resolve(self.config, experimental_graph_review=True))
                self.config.additional_config["draft_window_graph_numeric_diagnostic"] = "true"
                with self.assertRaises(TypeError):
                    self.resolve(self.config)
                self.config.additional_config.clear()
                self.config.additional_config["draft_window_size"] = 32768

    def test_different_config_and_other_methods_are_not_typewide_bans(self):
        self.assertIsNone(self.resolve(self.config))
        with self.identified("_SWA6_CONFIG"):
            self.config.speculative_config.method = "mtp"
            self.assertIsNone(self.resolve(self.config))
        (self.model / "config.json").unlink()
        self.config.speculative_config.method = "dflash"
        self.assertIsNone(self.resolve(self.config))

    def test_factory_rejects_before_constructor_and_binds_exact_owner(self):
        module = ModuleType(MODULE)
        module.resolve_draft_graph_review_limit = self.resolve
        ascend = ModuleType("vllm_ascend")
        ascend.envs = NS(VLLM_ASCEND_EXPERIMENTAL_DRAFT_WINDOW_GRAPH=False)
        calls = []
        owner = NS()
        namespace = {
            "_resolve_window_policy": lambda c: None,
            "_use_graph_window_owner": lambda c, p: False,
            "_init_speculator": lambda *a: calls.append("construct") or owner,
            "DraftWindowPolicyState": NS(FORCED_OFF=object()),
        }
        factory = actual_function(SPEC / "__init__.py", "init_speculator", namespace)
        with (
            patch.dict(sys.modules, {MODULE: module, "vllm_ascend": ascend}),
            self.identified("_SWA6_CONFIG"),
            self.assertRaises(NotImplementedError),
        ):
            factory(self.config, None)
        self.assertEqual(calls, [])
        self.config.scheduler_config.max_num_seqs = 2
        with (
            patch.dict(sys.modules, {MODULE: module, "vllm_ascend": ascend}),
            self.identified("_FULL8_CONFIG"),
            self.assertRaisesRegex(NotImplementedError, "max_num_seqs=1"),
        ):
            factory(self.config, None)
        self.assertEqual(calls, [])
        self.config.scheduler_config.max_num_seqs = 1
        with patch.dict(sys.modules, {MODULE: module, "vllm_ascend": ascend}), self.identified("_FULL8_CONFIG"):
            for gate in (False, True):
                ascend.envs.VLLM_ASCEND_EXPERIMENTAL_DRAFT_WINDOW_GRAPH = gate
                with (
                    self.subTest(gate=gate),
                    self.assertRaisesRegex(NotImplementedError, "PR017/vLLM544 pair is incomplete"),
                ):
                    factory(self.config, None)
            self.assertEqual(calls, [])
            self.config.speculative_config.enforce_eager = True
            self.assertIs(factory(self.config, None), owner)
        self.assertEqual(calls, ["construct"])
        self.assertFalse(hasattr(owner, "_draft_graph_review_limit"))

    def test_actual_proposal_boundary_checks_before_runtime_or_submission(self):
        module = ModuleType(MODULE)
        module.validate_draft_graph_review_requests = self.validate
        scope = actual_function(
            SPEC / "dflash/speculator.py", "_draft_window_owner_scope", {"nullcontext": nullcontext}
        )
        owner = NS(_draft_graph_review_limit="full8_local_active_at_most_one")
        with patch.dict(sys.modules, {MODULE: module}):
            for active in (2, 3, 4, 8):
                with self.assertRaises(NotImplementedError):
                    scope(owner, NS(num_reqs=active, num_reqs_after_padding=4))
            with scope(owner, NS(num_reqs=1, num_reqs_after_padding=1)):
                pass
            with scope(owner, NS(num_reqs=16), True, False):
                pass

    def test_actual_propose_rejects_before_input_batch_mutation(self):
        module = ModuleType(MODULE)
        module.validate_draft_graph_review_requests = self.validate
        actual_scope = actual_function(
            SPEC / "dflash/speculator.py", "_draft_window_owner_scope", {"nullcontext": nullcontext}
        )
        tree = ast.parse((SPEC / "dflash/speculator.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendDFlashSpeculator")
        variants = [n for n in ast.walk(cls) if isinstance(n, ast.FunctionDef) and n.name == "propose"]
        self.assertEqual(len(variants), 1)  # Fixed017/544 has one current public signature.
        for variant in variants:
            functions = ast.Module(
                body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), variant],
                type_ignores=[],
            )
            namespace = {"_draft_window_owner_scope": actual_scope}
            exec(compile(ast.fix_missing_locations(functions), str(SPEC / "dflash/speculator.py"), "exec"), namespace)
            owner = NS(_draft_graph_review_limit="full8_local_active_at_most_one", input_batch="unchanged")
            for batch in (NS(num_reqs=2, num_reqs_after_padding=2), NS(num_reqs=1, num_reqs_after_padding=2)):
                with patch.dict(sys.modules, {MODULE: module}), self.assertRaises(NotImplementedError):
                    namespace["propose"](owner, batch, *([None] * 10))
                self.assertEqual(owner.input_batch, "unchanged")

    def test_no_server_path_or_framework_import_in_guard(self):
        content = (SPEC / "draft_graph_review_limits.py").read_text()
        self.assertNotIn("/mnt/", content)
        for n in ast.walk(ast.parse(content)):
            if isinstance(n, ast.Import):
                self.assertTrue(all(a.name in ("hashlib",) for a in n.names))
            elif isinstance(n, ast.ImportFrom):
                self.assertIn(n.module, ("__future__", "pathlib", "dataclasses", "vllm_ascend.ascend_config"))
                if n.module == "vllm_ascend.ascend_config":
                    self.assertEqual([alias.name for alias in n.names], ["validate_additional_config_bool"])
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("torch_npu", sys.modules)
        self.assertNotIn("vllm", sys.modules)


if __name__ == "__main__":
    unittest.main()
