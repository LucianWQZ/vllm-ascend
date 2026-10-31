# SPDX-License-Identifier: Apache-2.0
"""Finite Full8 entry and real pre-query guards; no framework/device import."""

import ast
import hashlib
import runpy
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "vllm_ascend/worker/v2/spec_decode"
GUARD = runpy.run_path(str(SPEC / "draft_graph_review_limits.py"))


def actual_function(path, name, namespace):
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(
                    body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
                    type_ignores=[],
                )
            ),
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace[name]


class Full8FiniteEntry(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        model = Path(self.tmp.name)
        data = b'{"bounded_fixture":true}\n'
        (model / "config.json").write_bytes(data)
        self.g = GUARD.copy()
        self.resolve = GUARD["resolve_draft_graph_review_limit"]
        digest_patch = patch.dict(self.resolve.__globals__, {"_FULL8_CONFIG": hashlib.sha256(data).hexdigest()})
        digest_patch.start()
        self.addCleanup(digest_patch.stop)
        from pydantic import TypeAdapter

        module = ModuleType("vllm_ascend.ascend_config")
        module.validate_additional_config_bool = actual_function(
            ROOT / "vllm_ascend/ascend_config.py", "validate_additional_config_bool", {"TypeAdapter": TypeAdapter}
        )
        mocked = patch.dict(sys.modules, {"vllm_ascend.ascend_config": module})
        mocked.start()
        self.addCleanup(mocked.stop)
        self.config = NS(
            speculative_config=NS(
                method="dflash",
                enforce_eager=False,
                num_speculative_tokens=7,
                enable_adaptive_verification=False,
                uses_dynamic_speculative_decoding=lambda: False,
                draft_parallel_config=NS(tensor_parallel_size=4),
                draft_model_config=NS(model=str(model)),
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
            model_config=NS(max_model_len=40960),
            compilation_config=NS(cudagraph_mode=NS(name="FULL_DECODE_ONLY"), cudagraph_capture_sizes=[8]),
            additional_config=dict(draft_window_size=32768, draft_window_graph_replicated_tp_review=True),
        )

    def test_default_stays_closed_and_explicit_entry_is_exact(self):
        with self.assertRaisesRegex(NotImplementedError, "PR017/vLLM544 pair is incomplete"):
            self.resolve(self.config, experimental_graph_review=True)
        self.config.additional_config["draft_window_graph_full8_single_request_review"] = True
        self.assertEqual(self.resolve(self.config, experimental_graph_review=True), "full8_local_active_at_most_one")
        with self.assertRaises(NotImplementedError):
            self.resolve(self.config, experimental_graph_review=False)
        (Path(self.config.speculative_config.draft_model_config.model) / "config.json").write_text("{}")
        with self.assertRaises(NotImplementedError):
            self.resolve(self.config, experimental_graph_review=True)

    def test_wrong_topology_mode_capture_and_bool_rejected(self):
        self.config.additional_config["draft_window_graph_full8_single_request_review"] = True
        for obj, key, value in [
            (self.config.parallel_config, "data_parallel_size", 2),
            (self.config.parallel_config, "tensor_parallel_size", 2),
            (self.config.scheduler_config, "max_num_seqs", 2),
            (self.config.scheduler_config, "async_scheduling", True),
            (self.config.cache_config, "enable_prefix_caching", True),
            (self.config.compilation_config, "cudagraph_capture_sizes", [8, 16]),
            (self.config.model_config, "max_model_len", 8192),
            (self.config.speculative_config, "enforce_eager", True),
        ]:
            before = getattr(obj, key)
            setattr(obj, key, value)
            with self.subTest(key=key, value=value), self.assertRaises(NotImplementedError):
                self.resolve(self.config, experimental_graph_review=True)
            setattr(obj, key, before)
        for key, value in [
            ("draft_window_graph_full8_single_request_review", "not-a-bool"),
            ("draft_window_graph_eagle_single_request_review", True),
        ]:
            before = self.config.additional_config.get(key)
            self.config.additional_config[key] = value
            with self.subTest(key=key), self.assertRaises((TypeError, ValueError)):
                self.resolve(self.config, experimental_graph_review=True)
            if before is None:
                self.config.additional_config.pop(key)
            else:
                self.config.additional_config[key] = before

    def test_all4_ranks_refuse_actual_multirow_before_any_body(self):
        guard = GUARD["validate_draft_graph_review_requests"]
        body = []
        for active in (2, 3, 4, 8):
            rejected = []
            for rank in range(4):
                owner = NS(_draft_graph_review_limit="full8_local_active_at_most_one")
                try:
                    guard(owner, NS(num_reqs=active, num_reqs_after_padding=active))
                    body.append(rank)
                except NotImplementedError:
                    rejected.append(rank)
            self.assertEqual(rejected, list(range(4)))
        self.assertEqual(body, [])
        for padded in (2, 4, 8, None):
            with self.subTest(padded=padded), self.assertRaises(NotImplementedError):
                guard(
                    NS(_draft_graph_review_limit="full8_local_active_at_most_one"),
                    NS(num_reqs=1, num_reqs_after_padding=padded),
                )

    def test_real_draft_descriptor_guard_precedes_slots_and_builder(self):
        stage = GUARD["validate_full8_single_request_graph_stage"]
        owner = NS(_draft_graph_review_limit="full8_local_active_at_most_one")
        stage(owner, actual=1, padded=1, tokens=8)
        for n, p, q in ((2, 2, 16), (1, 2, 16), (1, 1, 16)):
            with self.subTest(shape=(n, p, q)), self.assertRaises(NotImplementedError):
                stage(owner, actual=n, padded=p, tokens=q)
        with self.assertRaises(TypeError):
            stage(owner, actual=True, padded=1, tokens=8)
        tree = ast.parse((SPEC / "dflash/speculator.py").read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_build_graph_uniform_metadata")
        stage_line = next(
            n.lineno
            for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "validate_full8_single_request_graph_stage"
        )
        slot_line = next(
            n.lineno
            for n in ast.walk(fn)
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "full_slots" for t in n.targets)
        )
        self.assertLess(stage_line, slot_line)

    def test_dummy_profile_and_zero_context_are_not_fake_query(self):
        owner = NS(_draft_graph_review_limit="full8_local_active_at_most_one")
        guard = GUARD["validate_draft_graph_review_requests"]
        guard(owner, NS(num_reqs=3, num_reqs_after_padding=4), dummy_run=True)
        guard(owner, NS(num_reqs=3, num_reqs_after_padding=4), is_profile=True)
        module = ModuleType("vllm_ascend.worker.v2.spec_decode.draft_graph_review_limits")
        module.validate_draft_graph_review_requests = guard
        with patch.dict(sys.modules, {module.__name__: module}):
            seen = []
            scope = actual_function(
                SPEC / "dflash/speculator.py",
                "_draft_window_owner_scope",
                {
                    "nullcontext": nullcontext,
                    "_draft_window_context_kv_scope": lambda o, b: (seen.append("context-only"), nullcontext())[1],
                },
            )
            owner._draft_window_runtime_state = NS()
            owner.num_speculative_steps = 7
            with scope(owner, NS(num_reqs=1, num_reqs_after_padding=2), num_speculative_tokens=0):
                pass
            self.assertEqual(seen, ["context-only"])
            with self.assertRaises(NotImplementedError):
                scope(owner, NS(num_reqs=1, num_reqs_after_padding=2), num_speculative_tokens=7)


if __name__ == "__main__":
    unittest.main()
