# SPDX-License-Identifier: Apache-2.0
"""Finite Full8 entry and real pre-query guards; no framework/device import."""

import ast
import hashlib
import runpy
import sys
import tempfile
import unittest
from contextlib import contextmanager, nullcontext
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
                draft_model_config=NS(model=str(model), hf_config=NS(max_position_embeddings=262144)),
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
            model_config=NS(max_model_len=40960, hf_config=NS(max_position_embeddings=262144)),
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
        self.config.scheduler_config.async_scheduling = True
        self.config.cache_config.enable_prefix_caching = True
        self.config.additional_config["enable_flashcomm1"] = True
        self.assertEqual(self.resolve(self.config, experimental_graph_review=True), "full8_local_active_at_most_one")
        self.config.scheduler_config.async_scheduling = False
        self.config.cache_config.enable_prefix_caching = False
        del self.config.additional_config["enable_flashcomm1"]
        for obj, key, value in [
            # This fixture lacks the mandatory TP-only Draft DP1 descriptor.
            (self.config.parallel_config, "data_parallel_size", 2),
            (self.config.parallel_config, "tensor_parallel_size", 2),
            (self.config.scheduler_config, "max_num_seqs", 2),
            (self.config.scheduler_config, "async_scheduling", 1),
            (self.config.cache_config, "enable_prefix_caching", "true"),
            (self.config.compilation_config, "cudagraph_capture_sizes", [8, 16]),
            (self.config.model_config, "max_model_len", 262145),
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

    def test_dp2_single_entry_retains_real_tp_only_draft_descriptor(self):
        self.config.additional_config["draft_window_graph_full8_single_request_review"] = True
        self.config.parallel_config.data_parallel_size = 2
        draft = self.config.speculative_config.draft_parallel_config
        draft.data_parallel_size = 1
        for window in (None, 32768):
            self.config.additional_config["draft_window_size"] = window
            self.assertEqual(
                self.resolve(self.config, experimental_graph_review=True), "full8_local_active_at_most_one"
            )
            self.config.scheduler_config.async_scheduling = True
            self.config.cache_config.enable_prefix_caching = True
            self.config.additional_config["enable_flashcomm1"] = True
            self.assertEqual(
                self.resolve(self.config, experimental_graph_review=True), "full8_local_active_at_most_one"
            )
            self.config.scheduler_config.async_scheduling = False
            self.config.cache_config.enable_prefix_caching = False
            del self.config.additional_config["enable_flashcomm1"]
        for value in (None, True, 0, 2, "1"):
            draft.data_parallel_size = value
            with self.subTest(draft_dp=value), self.assertRaises(NotImplementedError):
                self.resolve(self.config, experimental_graph_review=True)
        draft.data_parallel_size = 1
        for obj, key, value in (
            (self.config.parallel_config, "data_parallel_size", 3),
            (self.config.parallel_config, "data_parallel_size", True),
            (self.config.parallel_config, "enable_expert_parallel", False),
            (self.config.scheduler_config, "max_num_seqs", 2),
            (self.config.scheduler_config, "async_scheduling", 1),
            (self.config.cache_config, "enable_prefix_caching", "true"),
            (self.config.compilation_config, "cudagraph_capture_sizes", [8, 16]),
        ):
            old = getattr(obj, key)
            setattr(obj, key, value)
            with self.subTest(key=key, value=value), self.assertRaises(NotImplementedError):
                self.resolve(self.config, experimental_graph_review=True)
            setattr(obj, key, old)

    def test_dp2_replicas_keep_independent_request_rows_and_local_multirow_rejection(self):
        guard = GUARD["validate_draft_graph_review_requests"]
        for dp in range(2):
            for tp in range(4):
                owner = NS(
                    _draft_graph_review_limit="full8_local_active_at_most_one",
                    max_model_len=40960,
                    num_query_per_req=8,
                    vllm_config=NS(model_config=NS(max_model_len=40960)),
                )
                batch = request_batch()
                batch.req_ids = [f"full8-dp{dp}"]
                batch.draft_graph_review_request_rows = ((batch.req_ids[0], 7, 33025, 33089, 33025),)
                with self.subTest(dp=dp, tp=tp):
                    guard(owner, batch)
                    batch.draft_graph_review_request_rows = ((f"full8-dp{1 - dp}", 7, 33025, 33089, 33025),)
                    with self.assertRaisesRegex(RuntimeError, "identity/source"):
                        guard(owner, batch)
                    batch.num_reqs = 2
                    with self.assertRaisesRegex(NotImplementedError, "at most 1"):
                        guard(owner, batch)

    def test_dp2_cannot_select_the_dp1_two_request_entry(self):
        self.config.speculative_config.draft_parallel_config.data_parallel_size = 1
        self.config.additional_config["draft_window_graph_full8_two_request_review"] = True
        self.config.scheduler_config.max_num_seqs = 2
        self.config.compilation_config.cudagraph_capture_sizes = [8, 16]
        for window in (None, 32768):
            self.config.additional_config["draft_window_size"] = window
            for features in (False, True):
                self.config.scheduler_config.async_scheduling = features
                self.config.cache_config.enable_prefix_caching = features
                self.config.additional_config["enable_flashcomm1"] = features
                self.config.parallel_config.data_parallel_size = 1
                self.assertEqual(
                    self.resolve(self.config, experimental_graph_review=True), "full8_local_active_at_most_two"
                )
                self.config.parallel_config.data_parallel_size = 2
                with self.assertRaisesRegex(NotImplementedError, "DP2 one request only"):
                    self.resolve(self.config, experimental_graph_review=True)

    def test_two_request_entry_is_separate_and_uses_both_exact_banks(self):
        extra = self.config.additional_config
        extra["draft_window_graph_full8_two_request_review"] = True
        self.config.scheduler_config.max_num_seqs = 2
        self.config.compilation_config.cudagraph_capture_sizes = [8, 16]
        for window in (None, 32768):
            extra["draft_window_size"] = window
            self.assertEqual(
                self.resolve(self.config, experimental_graph_review=True), "full8_local_active_at_most_two"
            )
        for sizes in ([16], [8], [8, 16, 24], [True, 16]):
            self.config.compilation_config.cudagraph_capture_sizes = sizes
            with self.subTest(sizes=sizes), self.assertRaises(NotImplementedError):
                self.resolve(self.config, experimental_graph_review=True)
        self.config.compilation_config.cudagraph_capture_sizes = [8, 16]
        extra["draft_window_graph_full8_single_request_review"] = True
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            self.resolve(self.config, experimental_graph_review=True)
        del extra["draft_window_graph_full8_single_request_review"]
        for maximum in (1, 3, 8, True):
            self.config.scheduler_config.max_num_seqs = maximum
            with self.subTest(maximum=maximum), self.assertRaises(NotImplementedError):
                self.resolve(self.config, experimental_graph_review=True)

    def test_two_rows_use_distinct_original_identity_and_capacity(self):
        guard = GUARD["validate_draft_graph_review_requests"]
        owner = NS(
            _draft_graph_review_limit="full8_local_active_at_most_two",
            max_model_len=40960,
            num_query_per_req=8,
            vllm_config=NS(model_config=NS(max_model_len=40960)),
        )
        batch = NS(
            num_reqs=2,
            num_reqs_after_padding=2,
            req_ids=["long", "short"],
            idx_mapping_np=[7, 2],
            prefill_len_np=[33025, 1153],
            draft_graph_review_request_rows=(("long", 7, 33025, 33089, 33025), ("short", 2, 1153, 1217, 1153)),
        )
        guard(owner, batch)
        # A real reorder must move every authoritative row field together.
        batch.req_ids.reverse()
        batch.idx_mapping_np.reverse()
        batch.prefill_len_np.reverse()
        batch.draft_graph_review_request_rows = batch.draft_graph_review_request_rows[::-1]
        guard(owner, batch)
        saved = batch.draft_graph_review_request_rows
        batch.draft_graph_review_request_rows = (saved[1], saved[0])
        with self.assertRaisesRegex(RuntimeError, "identity/source"):
            guard(owner, batch)
        batch.draft_graph_review_request_rows = saved
        batch.idx_mapping_np[1] = 2
        batch.draft_graph_review_request_rows = (saved[0], ("long", 2, 33025, 33089, 33025))
        with self.assertRaisesRegex(RuntimeError, "distinct"):
            guard(owner, batch)
        batch.idx_mapping_np[1] = 7
        batch.draft_graph_review_request_rows = (saved[0], ("long", 7, 33025, 40953, 33025))
        with self.assertRaisesRegex(NotImplementedError, "capacity"):
            guard(owner, batch)
        batch.draft_graph_review_request_rows = saved
        owner._draft_graph_review_preempted_request_ids = {"long"}
        with self.assertRaisesRegex(NotImplementedError, "preemption"):
            guard(owner, batch)

    def test_two_request_graph_stages_reject_unqualified_padding_and_extra_rows(self):
        stage = GUARD["validate_full8_single_request_graph_stage"]
        owner = NS(_draft_graph_review_limit="full8_local_active_at_most_two")
        for counts in ((1, 1, 8), (2, 2, 16)):
            stage(owner, actual=counts[0], padded=counts[1], tokens=counts[2])
        for counts in ((0, 0, 0), (3, 3, 24), (8, 8, 64), (1, 2, 16), (2, 4, 32), (2, 2, 8)):
            with self.subTest(counts=counts), self.assertRaises(NotImplementedError):
                stage(owner, actual=counts[0], padded=counts[1], tokens=counts[2])
        with self.assertRaises(TypeError):
            stage(owner, actual=True, padded=1, tokens=8)

    def test_two_request_context_only_still_checks_request_rows(self):
        guard = GUARD["validate_draft_graph_review_requests"]
        owner = NS(
            _draft_graph_review_limit="full8_local_active_at_most_two",
            max_model_len=40960,
            num_query_per_req=8,
            vllm_config=NS(model_config=NS(max_model_len=40960)),
        )
        batch = NS(
            num_reqs=2,
            num_reqs_after_padding=4,
            req_ids=["a", "b"],
            idx_mapping_np=[0, 1],
            prefill_len_np=[127, 1153],
            draft_graph_review_request_rows=(("a", 0, 127, 191, 127), ("b", 1, 1153, 1217, 1153)),
        )
        guard(owner, batch, context_kv_only=True)
        with self.assertRaisesRegex(NotImplementedError, "padding"):
            guard(owner, batch)
        batch.req_ids[1] = "other"
        with self.assertRaisesRegex(RuntimeError, "identity/source"):
            guard(owner, batch, context_kv_only=True)

    def test_existing_dp_dummy_scope_is_not_a_business_generation(self):
        # Execute only the real Source bypass branch. No fake Graph metadata,
        # copied business generation, provider construction or device event.
        tree = ast.parse((SPEC / "draft_window_eager.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DraftWindowEagerMixin")
        node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_draft_window_proposal_scope")
        namespace = {"contextmanager": contextmanager}
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(SPEC / "draft_window_eager.py"), "exec"), namespace)
        scope = namespace["_draft_window_proposal_scope"]
        owner = NS(
            _draft_window_epoch_source=object(),
            _draft_window_async_pool=None,
            _draft_window_runtime_state=NS(quarantined_identity=None),
            _draft_window_bypass_active=False,
            _draft_window_proposal_context=None,
            _draft_window_graph_characterization=object(),
            dp_size=2,
        )
        with scope(owner, NS(is_dummy=True), dummy_run=True, is_profile=False):
            self.assertTrue(owner._draft_window_bypass_active)
            self.assertTrue(owner._draft_window_dp_dummy_active)
            self.assertFalse(owner._draft_window_prefill_bypass_active)
            self.assertIsNone(owner._draft_window_proposal_context)
        self.assertFalse(owner._draft_window_dp_dummy_active)
        self.assertFalse(owner._draft_window_bypass_active)
        with (
            self.assertRaisesRegex(ValueError, "actual submission"),
            scope(owner, NS(is_dummy=True), dummy_run=True, is_profile=False),
        ):
            raise ValueError("actual submission")
        self.assertTrue(owner._draft_window_graph_submission_uncertain)
        with (
            self.assertRaisesRegex(RuntimeError, "quarantined"),
            scope(owner, NS(is_dummy=True), dummy_run=True, is_profile=False),
        ):
            self.fail("uncertain dummy must not re-enter")

    def test_all4_ranks_refuse_actual_multirow_before_any_body(self):
        guard = GUARD["validate_draft_graph_review_requests"]
        body = []
        for active in (2, 3, 4, 8):
            rejected = []
            for rank in range(4):
                owner = NS(
                    _draft_graph_review_limit="full8_local_active_at_most_one",
                    max_model_len=40960,
                    num_query_per_req=8,
                    vllm_config=NS(model_config=NS(max_model_len=40960)),
                )
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
                    NS(
                        _draft_graph_review_limit="full8_local_active_at_most_one",
                        max_model_len=40960,
                        num_query_per_req=8,
                        vllm_config=NS(model_config=NS(max_model_len=40960)),
                    ),
                    NS(num_reqs=1, num_reqs_after_padding=padded),
                )

    def test_real_draft_descriptor_guard_precedes_slots_and_builder(self):
        stage = GUARD["validate_full8_single_request_graph_stage"]
        owner = NS(
            _draft_graph_review_limit="full8_local_active_at_most_one",
            max_model_len=40960,
            num_query_per_req=8,
            vllm_config=NS(model_config=NS(max_model_len=40960)),
        )
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
        owner = NS(
            _draft_graph_review_limit="full8_local_active_at_most_one",
            max_model_len=40960,
            num_query_per_req=8,
            vllm_config=NS(model_config=NS(max_model_len=40960)),
        )
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
        owner = NS(
            _draft_graph_review_limit="full8_local_active_at_most_one",
            max_model_len=40960,
            num_query_per_req=8,
            vllm_config=NS(model_config=NS(max_model_len=40960)),
        )
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
        for prompt, output in ((0, 64), (33025, 0), (40950, 3), (40825, 128)):
            with (
                self.subTest(prompt=prompt, output=output),
                self.assertRaisesRegex(NotImplementedError, "actual allocated model capacity"),
            ):
                GUARD["validate_draft_graph_review_requests"](owner, request_batch(prompt, output))

    def test_missing_conflicting_or_reordered_request_evidence_is_rejected(self):
        owner = NS(
            _draft_graph_review_limit="full8_local_active_at_most_one",
            max_model_len=40960,
            num_query_per_req=8,
            vllm_config=NS(model_config=NS(max_model_len=40960)),
        )
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
        owner = NS(
            _draft_graph_review_limit="full8_local_active_at_most_one",
            num_speculative_steps=7,
            max_model_len=40960,
            num_query_per_req=8,
            vllm_config=NS(model_config=NS(max_model_len=40960)),
        )
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
            with self.assertRaisesRegex(NotImplementedError, "actual allocated model capacity"):
                scope(owner, request_batch(prompt=40825, output=128, padded=4), num_speculative_tokens=0)
            self.assertEqual(seen, [])
            with scope(owner, request_batch(padded=4), num_speculative_tokens=0):
                pass
            self.assertEqual(seen, ["context-only"])
            GUARD["note_eagle_single_request_review_lifecycle"](
                owner, NS(finished_req_ids=set(), preempted_req_ids={"full8-A"})
            )
            with self.assertRaisesRegex(NotImplementedError, "preemption"):
                scope(owner, request_batch(padded=4), num_speculative_tokens=0)

    def test_capacity_cases_are_not_exact_prompt_output_allowlists(self):
        self.config.additional_config["draft_window_graph_full8_single_request_review"] = True
        owner = NS(
            _draft_graph_review_limit="full8_local_active_at_most_one",
            max_model_len=40960,
            num_query_per_req=8,
            vllm_config=NS(model_config=NS(max_model_len=40960)),
        )
        for window in (None, 32768):
            self.config.additional_config["draft_window_size"] = window
            self.assertEqual(
                self.resolve(self.config, experimental_graph_review=True), "full8_local_active_at_most_one"
            )
            for prompt, output in ((127, 3), (32767, 3), (32769, 128), (40824, 128)):
                with self.subTest(window=window, prompt=prompt, output=output):
                    GUARD["validate_draft_graph_review_requests"](owner, request_batch(prompt, output))
        self.config.model_config.max_model_len = 8192
        self.assertEqual(self.resolve(self.config, experimental_graph_review=True), "full8_local_active_at_most_one")
        owner.max_model_len = owner.vllm_config.model_config.max_model_len = 8192
        GUARD["validate_draft_graph_review_requests"](owner, request_batch(8181, 3))
        with self.assertRaisesRegex(NotImplementedError, "actual allocated model capacity"):
            GUARD["validate_draft_graph_review_requests"](owner, request_batch(8182, 3))

    def test_runtime_capacity_cannot_silently_drift_from_allocating_config(self):
        guard = GUARD["validate_draft_graph_review_requests"]
        for allocated, runtime in ((40960, 65536), (65536, 40960), (True, 40960)):
            owner = NS(
                _draft_graph_review_limit="full8_local_active_at_most_one",
                max_model_len=runtime,
                num_query_per_req=8,
                vllm_config=NS(model_config=NS(max_model_len=allocated)),
            )
            with (
                self.subTest(allocated=allocated, runtime=runtime),
                self.assertRaisesRegex(NotImplementedError, "actual allocated model capacity"),
            ):
                guard(owner, request_batch())

    def test_capacity_is_bounded_by_both_real_model_configs(self):
        self.config.additional_config["draft_window_graph_full8_single_request_review"] = True
        for config in (self.config.model_config, self.config.speculative_config.draft_model_config):
            before = config.hf_config.max_position_embeddings
            for value in (40959, None, True, "262144"):
                config.hf_config.max_position_embeddings = value
                with (
                    self.subTest(config=config, ceiling=value),
                    self.assertRaisesRegex(NotImplementedError, "both model position ceilings"),
                ):
                    self.resolve(self.config, experimental_graph_review=True)
            config.hf_config.max_position_embeddings = before
        for capacity in (None, True, "40960", 8, 262145):
            self.config.model_config.max_model_len = capacity
            with (
                self.subTest(capacity=capacity),
                self.assertRaisesRegex(NotImplementedError, "both model position ceilings"),
            ):
                self.resolve(self.config, experimental_graph_review=True)

    def test_window_off_zero_context_keeps_guard_without_creating_window_owner(self):
        module = ModuleType("vllm_ascend.worker.v2.spec_decode.draft_graph_review_limits")
        module.validate_draft_graph_review_requests = GUARD["validate_draft_graph_review_requests"]
        scope = actual_function(
            SPEC / "dflash/speculator.py", "_draft_window_owner_scope", {"nullcontext": nullcontext}
        )
        owner = NS(
            _draft_graph_review_limit="full8_local_active_at_most_one",
            max_model_len=40960,
            num_query_per_req=8,
            vllm_config=NS(model_config=NS(max_model_len=40960)),
        )
        with patch.dict(sys.modules, {module.__name__: module}):
            with scope(owner, request_batch(32769, 128, padded=4), num_speculative_tokens=0):
                pass
            self.assertFalse(hasattr(owner, "_draft_window_runtime_state"))
            with self.assertRaisesRegex(NotImplementedError, "actual=1/padded=1"):
                scope(owner, request_batch(32769, 128, padded=4), num_speculative_tokens=7)
            with self.assertRaisesRegex(NotImplementedError, "actual allocated model capacity"):
                scope(owner, request_batch(40825, 128, padded=4), num_speculative_tokens=0)

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
        owner = NS(
            _draft_graph_review_limit="full8_local_active_at_most_one",
            input_batch="previous-owner",
            max_model_len=40960,
            num_query_per_req=8,
            vllm_config=NS(model_config=NS(max_model_len=40960)),
        )
        with patch.dict(sys.modules, {module.__name__: module}):
            for batch in (request_batch(prompt=0), request_batch(prompt=40825, output=128)):
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
