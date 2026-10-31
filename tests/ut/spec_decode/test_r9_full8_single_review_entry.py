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


def request_batch(prompt=33025, output=64, padded=1):
    return NS(
        num_reqs=1,
        num_reqs_after_padding=padded,
        req_ids=["full8-A"],
        idx_mapping_np=[7],
        prefill_len_np=[prompt],
        draft_graph_review_request_rows=(("full8-A", 7, prompt, prompt + output, prompt),),
    )


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
            with scope(owner, request_batch(padded=2), num_speculative_tokens=0):
                pass
            self.assertEqual(seen, ["context-only"])
            with self.assertRaises(NotImplementedError):
                scope(owner, request_batch(padded=2), num_speculative_tokens=7)

    def test_authoritative_snapshot_and_finite_request_limits(self):
        owner = NS(_draft_graph_review_limit="full8_local_active_at_most_one")
        states = NS(
            req_id_to_index={"full8-A": 7},
            prompt_len=NS(np={7: 33025}),
            max_seq_len={7: 33089},
            prefill_len=NS(np={7: 33025}),
        )
        snapshot = GUARD["eagle_single_request_review_rows"]
        batch = request_batch()
        self.assertEqual(
            snapshot(owner, states, batch.req_ids, batch.idx_mapping_np), batch.draft_graph_review_request_rows
        )
        GUARD["validate_draft_graph_review_requests"](owner, batch)
        states.req_id_to_index["full8-A"] = 8
        with self.assertRaisesRegex(RuntimeError, "mapping changed"):
            snapshot(owner, states, batch.req_ids, batch.idx_mapping_np)
        for prompt, output in ((33024, 64), (33026, 64), (33025, 63), (33025, 65)):
            with self.subTest(prompt=prompt, output=output), self.assertRaisesRegex(NotImplementedError, "prompt33025"):
                GUARD["validate_draft_graph_review_requests"](owner, request_batch(prompt, output))

    def test_missing_conflicting_or_reordered_request_evidence_is_rejected(self):
        owner = NS(_draft_graph_review_limit="full8_local_active_at_most_one")
        for change in ("missing", "extra-row", "wrong-id", "wrong-index", "wrong-prefill", "bad-row-type"):
            batch = request_batch()
            if change == "missing":
                del batch.draft_graph_review_request_rows
            elif change == "extra-row":
                batch.draft_graph_review_request_rows *= 2
            elif change == "wrong-id":
                batch.req_ids = ["previous-B"]
            elif change == "wrong-index":
                batch.idx_mapping_np = [8]
            elif change == "wrong-prefill":
                batch.prefill_len_np = [33026]
            else:
                batch.draft_graph_review_request_rows = (("full8-A", True, 33025, 33089, 33025),)
            with self.subTest(change=change), self.assertRaisesRegex(RuntimeError, "Full8 review"):
                GUARD["validate_draft_graph_review_requests"](owner, batch)

    def test_preemption_and_context_only_do_not_bypass_original_request(self):
        owner = NS(_draft_graph_review_limit="full8_local_active_at_most_one", num_speculative_steps=7)
        owner._draft_window_runtime_state = NS()
        module = ModuleType("vllm_ascend.worker.v2.spec_decode.draft_graph_review_limits")
        module.validate_draft_graph_review_requests = GUARD["validate_draft_graph_review_requests"]
        seen = []
        scope = actual_function(
            SPEC / "dflash/speculator.py",
            "_draft_window_owner_scope",
            {
                "nullcontext": nullcontext,
                "_draft_window_context_kv_scope": lambda o, b: (seen.append("context-only"), nullcontext())[1],
            },
        )
        with patch.dict(sys.modules, {module.__name__: module}):
            with self.assertRaisesRegex(NotImplementedError, "prompt33025"):
                scope(owner, request_batch(prompt=33024, padded=4), num_speculative_tokens=0)
            self.assertEqual(seen, [])
            with scope(owner, request_batch(padded=4), num_speculative_tokens=0):
                pass
            self.assertEqual(seen, ["context-only"])
            GUARD["note_eagle_single_request_review_lifecycle"](
                owner, NS(finished_req_ids=set(), preempted_req_ids={"full8-A"})
            )
            with self.assertRaisesRegex(NotImplementedError, "preemption"):
                scope(owner, request_batch(padded=4), num_speculative_tokens=0)

    def test_actual_propose_refuses_bad_request_before_owner_or_body(self):
        module = ModuleType("vllm_ascend.worker.v2.spec_decode.draft_graph_review_limits")
        module.validate_draft_graph_review_requests = GUARD["validate_draft_graph_review_requests"]
        scope = actual_function(
            SPEC / "dflash/speculator.py", "_draft_window_owner_scope", {"nullcontext": nullcontext}
        )
        cls = next(
            n
            for n in ast.parse((SPEC / "dflash/speculator.py").read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == "AscendDFlashSpeculator"
        )
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "propose")
        calls = []
        namespace = {
            "_draft_window_owner_scope": scope,
            "build_attn_metadata_wrapper": nullcontext,
            "super": lambda: NS(propose=lambda *args, **kwargs: calls.append((args, kwargs))),
        }
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(
                        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), fn],
                        type_ignores=[],
                    )
                ),
                str(SPEC / "dflash/speculator.py"),
                "exec",
            ),
            namespace,
        )
        owner = NS(_draft_graph_review_limit="full8_local_active_at_most_one", input_batch="previous-owner")
        with patch.dict(sys.modules, {module.__name__: module}):
            for batch in (request_batch(prompt=33024), request_batch(output=63)):
                with self.assertRaises(NotImplementedError):
                    namespace["propose"](owner, batch, *([None] * 10))
                self.assertEqual(owner.input_batch, "previous-owner")
                self.assertEqual(calls, [])
            batch = request_batch()
            namespace["propose"](owner, batch, *([None] * 10))
            self.assertIs(owner.input_batch, batch)
            self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
