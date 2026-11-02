# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU protocol checks for the fixed 544 parallel-drafter ownership seam.

The real lifetime pool is used. Only the device completion token is a stand-in;
this does not claim device completion, model correctness or Graph qualification.
"""

from __future__ import annotations

import ast
import builtins
import importlib.util
import sys
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "vllm_ascend/worker/v2/spec_decode/dflash/speculator.py"
EAGER = ROOT / "vllm_ascend/worker/v2/spec_decode/draft_window_eager.py"
LIFETIME = ROOT / "vllm_ascend/worker/v2/spec_decode/draft_window_lifetime.py"


def load_lifetime():
    name = "_r8_pr017_parallel_owner_lifetime"
    spec = importlib.util.spec_from_file_location(name, LIFETIME)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LIFETIME_MODULE = load_lifetime()


def extract(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


def load_seam(completion):
    runtime = SimpleNamespace(_NPUCompletion=completion)
    name = "_parallel_owner_graph_support"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "vllm_ascend/worker/v2/spec_decode/draft_graph_review_limits.py"
    )
    assert spec is not None and spec.loader is not None
    guard_module = importlib.util.module_from_spec(spec)
    sys.modules[name] = guard_module
    spec.loader.exec_module(guard_module)
    original_import = builtins.__import__

    def importing(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "vllm_ascend.worker.v2.spec_decode.draft_window_lifetime":
            return LIFETIME_MODULE
        if name == "vllm_ascend.worker.v2.spec_decode.draft_window_runtime":
            return runtime
        if name == "vllm_ascend.worker.v2.spec_decode.draft_graph_review_limits":
            return guard_module
        return original_import(name, globals, locals, fromlist, level)

    namespace = {
        "contextmanager": contextmanager,
        "nullcontext": nullcontext,
        "__builtins__": dict(vars(builtins), __import__=importing),
    }
    extract(SPEC, {"_draft_window_context_kv_scope", "_draft_window_owner_scope"}, namespace)
    return namespace


class Completion:
    recorded = 0

    def __init__(self):
        type(self).recorded += 1

    def is_complete(self):
        return True

    def wait(self):
        return None


class ParallelOwnerTests(unittest.TestCase):
    def setUp(self):
        Completion.recorded = 0
        self.seam = load_seam(Completion)
        self.batch = SimpleNamespace(is_dummy=False)
        self.owner = SimpleNamespace(
            _draft_window_graph_binding_provider=SimpleNamespace(quarantined_identity=None),
            _draft_window_epoch_source=object(),
            _draft_window_bypass_active=False,
            _draft_window_proposal_context=None,
            vllm_config=SimpleNamespace(max_concurrent_batches=2),
            num_speculative_steps=8,
        )
        self.normal: list[tuple[Any, dict[str, Any]]] = []

        @contextmanager
        def normal_scope(batch, **kwargs):
            self.normal.append((batch, kwargs))
            yield

        self.owner._draft_window_proposal_scope = normal_scope

    def scope(self, tokens, **kwargs):
        return self.seam["_draft_window_owner_scope"](self.owner, self.batch, num_speculative_tokens=tokens, **kwargs)

    def test_zero_uses_real_completion_pool_without_graph_reader(self):
        with self.scope(0):
            self.assertTrue(self.owner._draft_window_context_kv_active)
        self.assertFalse(self.owner._draft_window_context_kv_active)
        self.assertEqual(Completion.recorded, 1)
        self.assertEqual(self.normal, [])
        self.assertFalse(self.owner._draft_window_context_kv_pool._poisoned)
        self.assertFalse(hasattr(self.owner, "_draft_window_runtime"))
        self.assertIsNone(self.owner._draft_window_proposal_context)

    def test_zero_original_error_quarantines_and_actual_normal_scope_rejects(self):
        first = RuntimeError("native context write stand-in")
        with (
            self.assertRaises(RuntimeError) as seen,
            self.scope(0),
        ):
            raise first
        self.assertIs(seen.exception, first)
        self.assertTrue(self.owner._draft_window_context_kv_pool._poisoned)
        self.assertTrue(self.owner._draft_window_graph_submission_uncertain)
        actual = {"contextmanager": contextmanager}
        extract(EAGER, {"_draft_window_proposal_scope"}, actual)
        self.owner._draft_window_proposal_scope = MethodType(actual["_draft_window_proposal_scope"], self.owner)
        with (
            self.assertRaisesRegex(RuntimeError, "quarantined"),
            self.scope(8),
        ):
            self.fail("normal Graph body must not execute")

    def test_normal_uncertain_or_quarantined_refuses_zero_before_native(self):
        for identity, uncertain in [(object(), False), (None, True)]:
            with self.subTest(uncertain=uncertain):
                self.owner._draft_window_graph_binding_provider.quarantined_identity = identity
                self.owner._draft_window_graph_submission_uncertain = uncertain
                with (
                    self.assertRaisesRegex(RuntimeError, "quarantined"),
                    self.scope(0),
                ):
                    self.fail("context write must not execute")
                self.assertFalse(hasattr(self.owner, "_draft_window_context_kv_pool"))
        self.assertEqual(Completion.recorded, 0)

    def test_completion_failure_is_original_and_shared_quarantine(self):
        first = RuntimeError("completion record stand-in")

        class BrokenCompletion:
            def __init__(self):
                raise first

        self.seam = load_seam(BrokenCompletion)
        with (
            self.assertRaises(RuntimeError) as seen,
            self.scope(0),
        ):
            pass
        self.assertIs(seen.exception, first)
        self.assertTrue(self.owner._draft_window_graph_submission_uncertain)
        self.assertTrue(self.owner._draft_window_context_kv_pool._poisoned)

    def test_zero_and_normal_proposals_cannot_overlap(self):
        with self.scope(0):
            with self.assertRaisesRegex(RuntimeError, "already active"):
                self.scope(8)
            with self.assertRaisesRegex(RuntimeError, "already active"):
                self.scope(0)
        self.owner._draft_window_proposal_context = object()
        with (
            self.assertRaisesRegex(RuntimeError, "already active"),
            self.scope(0),
        ):
            self.fail("nested context write must not execute")

    def test_normal_static_k_and_dummy_keep_existing_graph_scope(self):
        for tokens in (None, 8):
            with self.scope(tokens):
                pass
        for flag in ("dummy_run", "is_profile"):
            with self.scope(0, **{flag: True}):
                pass
        self.batch.is_dummy = True
        with self.scope(0):
            pass
        self.assertEqual(len(self.normal), 5)
        self.assertTrue(self.normal[-1][1]["dummy_run"])
        self.assertFalse(hasattr(self.owner, "_draft_window_context_kv_pool"))
        for invalid_tokens in (1, 7, 9, False, 0.0):
            with (
                self.subTest(tokens=invalid_tokens),
                self.assertRaisesRegex(RuntimeError, "static K"),
            ):
                self.scope(invalid_tokens)

    def test_public_typed_eager_keeps_context_only_flag_and_off_is_idle(self):
        del self.owner._draft_window_graph_binding_provider
        calls = []

        def proposal_context(*args, **kwargs):
            calls.append((args, kwargs))
            return nullcontext()

        self.owner._draft_window_runtime = SimpleNamespace(proposal_context=proposal_context)
        with self.scope(0):
            pass
        with self.scope(8):
            pass
        self.assertEqual([call[1]["context_kv_only"] for call in calls], [True, False])
        del self.owner._draft_window_runtime
        with self.scope(0):
            pass
        self.assertFalse(hasattr(self.owner, "_draft_window_context_kv_pool"))

    def test_full8_target_padding_is_not_a_zero_branch_draft_query(self):
        # Use the actual immutable support type required by the production guard.
        guard_name = "_parallel_owner_graph_support"
        self.owner._draft_graph_review_limit = sys.modules[guard_name].DraftGraphSupport(40960, 7, 1, 1, "dflash", 8)
        self.owner.max_model_len = 40960
        self.owner.num_query_per_req = 8
        self.owner.vllm_config.model_config = SimpleNamespace(max_model_len=40960)
        self.batch.num_reqs = 1
        self.batch.num_reqs_after_padding = 2
        self.batch.req_ids = ["a"]
        self.batch.idx_mapping_np = [0]
        self.batch.prefill_len_np = [33025]
        self.batch.draft_graph_request_bounds = ((33025, 33089),)
        with self.scope(0):
            pass
        self.assertEqual(self.normal, [])
        for tokens in (None, 8):
            with self.assertRaisesRegex(NotImplementedError, "admitted actual rows"):
                self.scope(tokens)
        self.batch.num_reqs_after_padding = 1
        with self.scope(8):
            pass
        # Context-only skips query padding, never the actual request capacity.
        self.batch.num_reqs = 3
        self.batch.num_reqs_after_padding = 4
        for tokens in (0, 8):
            with self.assertRaisesRegex(NotImplementedError, "admitted actual rows"):
                self.scope(tokens)
        self.assertEqual(len(self.normal), 1)

    def test_double_owner_is_rejected(self):
        self.owner._draft_window_runtime = object()
        with self.assertRaisesRegex(RuntimeError, "cannot share"):
            self.scope(0)

    def test_544_propose_interfaces_guard_first_and_forward_real_count(self):
        for relative in ("dflash/speculator.py", "dspark/speculator.py"):
            path = ROOT / "vllm_ascend/worker/v2/spec_decode" / relative
            tree = ast.parse(path.read_text())
            propose = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "propose"]
            self.assertEqual(len(propose), 1)
            function = propose[0]
            self.assertIn("dp_sync_state", [arg.arg for arg in function.args.args])
            self.assertIn("num_speculative_tokens", [arg.arg for arg in function.args.args])
            assignment = function.body[0]
            assert isinstance(assignment, ast.Assign) and isinstance(assignment.value, ast.Call)
            assert isinstance(assignment.value.func, ast.Name)
            self.assertEqual(assignment.value.func.id, "_draft_window_owner_scope")
            forwarded = [
                node
                for node in ast.walk(function)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "propose"
            ]
            self.assertEqual(len(forwarded), 1)
            count = [kw.value for kw in forwarded[0].keywords if kw.arg == "num_speculative_tokens"]
            self.assertEqual(len(count), 1)
            assert isinstance(count[0], ast.Name)
            self.assertEqual(count[0].id, "num_speculative_tokens")

    def test_capture_keeps_544_callback_and_graph_scope(self):
        path = ROOT / "vllm_ascend/worker/v2/spec_decode/dflash/aclgraph.py"
        tree = ast.parse(path.read_text())
        capture = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "capture")
        self.assertTrue(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "draft_window_dflash_capture_scope"
                for node in ast.walk(capture)
            )
        )
        values = [node for node in ast.walk(capture) if isinstance(node, ast.Dict)]
        self.assertTrue(
            any(
                any(isinstance(key, ast.Constant) and key.value == "prepare_forward" for key in node.keys)
                for node in values
            )
        )
        replay = next(
            node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "run_fullgraph"
        )
        guard = next(
            node
            for node in ast.walk(replay)
            if isinstance(node, ast.Raise)
            and isinstance(node.exc, ast.Call)
            and getattr(node.exc.func, "id", None) == "NotImplementedError"
        )
        fast_path = next(
            node
            for node in ast.walk(replay)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Call)
            and getattr(node.test.func, "id", None) == "all"
        )
        self.assertLess(guard.lineno, fast_path.lineno)


if __name__ == "__main__":
    unittest.main()
