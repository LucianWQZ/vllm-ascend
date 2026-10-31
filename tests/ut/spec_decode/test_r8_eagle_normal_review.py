import ast
import hashlib
import importlib.util
import json
import os
import sys
import types
import unittest
from pathlib import Path

SOURCE = Path(os.environ.get("MRV2_EAGLE_NORMAL_TEST_SOURCE", str(Path(__file__).resolve().parents[3]))).resolve()
REL = "vllm_ascend/worker/v2/spec_decode/draft_graph_review_limits.py"
spec = importlib.util.spec_from_file_location("_r8_eagle_normal_guard_test", SOURCE / REL)
M = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = M
spec.loader.exec_module(M)
N = types.SimpleNamespace


class Vector:
    def __init__(self, value, dtype="torch.int32", device="cpu"):
        self.value = value
        self.ndim = 1
        self.shape = (1,)
        self.dtype = dtype
        self.device = N(type=device)

    def __getitem__(self, index):
        return N(item=lambda: self.value)


def config():
    return N(
        speculative_config=N(
            method="eagle3",
            enforce_eager=False,
            num_speculative_tokens=3,
            uses_dynamic_speculative_decoding=lambda: False,
            enable_adaptive_verification=False,
            draft_parallel_config=N(
                tensor_parallel_size=4, enable_expert_parallel=False, use_sequence_parallel_moe=False
            ),
        ),
        parallel_config=N(
            tensor_parallel_size=4,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            enable_expert_parallel=True,
            all2all_backend="allgather_reducescatter",
            use_sequence_parallel_moe=True,
        ),
        scheduler_config=N(async_scheduling=False),
        model_config=N(max_model_len=4096),
        cache_config=N(enable_prefix_caching=False),
        compilation_config=N(cudagraph_mode=N(name="FULL_DECODE_ONLY")),
        additional_config=dict(draft_window_size=1024, draft_window_graph_replicated_tp_review=True),
    )


def objects(prompt=257, output=32, capacity=4096):
    owner = N(_draft_graph_review_limit=M.EagleSingleRequestReviewScope(max_model_len=capacity), max_model_len=capacity)
    batch = N(
        num_reqs=1,
        num_reqs_after_padding=1,
        req_ids=["actual-A"],
        idx_mapping_np=[7],
        prefill_len_np=[prompt],
        draft_graph_review_request_rows=(("actual-A", 7, prompt, prompt + output, prompt),),
    )
    owner.input_batch = batch
    return owner, batch


class Controls(unittest.TestCase):
    def test_authoritative_prompt_and_maximum(self):
        owner, batch = objects()
        states = N(
            req_id_to_index={"actual-A": 7}, prompt_len=N(np={7: 257}), max_seq_len={7: 289}, prefill_len=N(np={7: 257})
        )
        self.assertEqual(
            M.eagle_single_request_review_rows(owner, states, ["actual-A"], [7]), batch.draft_graph_review_request_rows
        )
        states.req_id_to_index["actual-A"] = 6
        with self.assertRaises(RuntimeError):
            M.eagle_single_request_review_rows(owner, states, ["actual-A"], [7])

    def test_valid_capacity_requests_and_guarded_maximum(self):
        for prompt, output in ((1, 1), (257, 32), (1021, 128), (2051, 128), (3965, 128), (4092, 1)):
            owner, batch = objects(prompt, output)
            M.validate_draft_graph_review_requests(owner, batch)
            upper = min(prompt + output + 3, owner.max_model_len)
            for value in (prompt, upper):
                M.validate_eagle_single_request_review_exact(owner, batch, Vector(value), num_reqs=1, padded=1, step=0)
            for value in (prompt - 1, upper + 1):
                with self.assertRaises(NotImplementedError):
                    M.validate_eagle_single_request_review_exact(
                        owner, batch, Vector(value), num_reqs=1, padded=1, step=1
                    )

    def test_reject_wrong_count_padding_capacity_resume(self):
        for change in (
            "active",
            "padded",
            "prompt",
            "output",
            "over-capacity",
            "owner-capacity",
            "resume",
            "stale-id",
            "stale-index",
        ):
            owner, batch = objects()
            if change == "active":
                batch.num_reqs = 2
            elif change == "padded":
                batch.num_reqs_after_padding = 2
            elif change == "prompt":
                batch.draft_graph_review_request_rows = (("actual-A", 7, 0, 32, 0),)
                batch.prefill_len_np = [0]
            elif change == "output":
                batch.draft_graph_review_request_rows = (("actual-A", 7, 257, 257, 257),)
            elif change == "over-capacity":
                batch.draft_graph_review_request_rows = (("actual-A", 7, 257, 4094, 257),)
            elif change == "owner-capacity":
                owner.max_model_len = 8192
            elif change == "resume":
                batch.draft_graph_review_request_rows = (("actual-A", 7, 257, 289, 258),)
                batch.prefill_len_np = [258]
            elif change == "stale-id":
                batch.req_ids = ["previous-B"]
            else:
                batch.idx_mapping_np = [6]
            with self.assertRaises((NotImplementedError, RuntimeError)):
                M.validate_draft_graph_review_requests(owner, batch)

    def test_dummy_and_capture_buckets_not_business_requests(self):
        owner, batch = objects()
        batch.num_reqs = 4
        batch.num_reqs_after_padding = 4
        M.validate_draft_graph_review_requests(owner, batch, dummy_run=True)
        M.validate_draft_graph_review_requests(owner, batch, is_profile=True)
        batch.is_dummy = True
        M.validate_draft_graph_review_requests(owner, batch)

    def test_each_graph_consumer_has_exact_shape(self):
        owner, batch = objects()
        for first, step, ordinal, tokens in ((True, 0, 0, 4), (False, 1, 0, 1), (False, 2, 1, 1)):
            inputs = N(actual_request_count=1, padded_request_count=1, identity=N(step=step))
            M.validate_eagle_single_request_graph_stage(owner, inputs, first=first, n=1, tokens=tokens, ordinal=ordinal)
            with self.assertRaises(NotImplementedError):
                M.validate_eagle_single_request_graph_stage(
                    owner, inputs, first=first, n=1, tokens=tokens + 1, ordinal=ordinal
                )
        with self.assertRaises(RuntimeError):
            M.validate_eagle_single_request_review_exact(
                owner, batch, Vector(257, device="npu"), num_reqs=1, padded=1, step=0
            )
        with self.assertRaises(RuntimeError):
            M.validate_eagle_single_request_review_exact(
                owner, batch, Vector(257, dtype="torch.float32"), num_reqs=1, padded=1, step=0
            )

    def test_construct_limits_fail_closed(self):
        c = config()
        self.assertIsInstance(
            M.validate_eagle_single_request_review_config(c, experimental_graph_review=True),
            M.EagleSingleRequestReviewScope,
        )
        c.additional_config.pop("draft_window_size")
        self.assertEqual(
            M.validate_eagle_single_request_review_config(c, experimental_graph_review=True).max_model_len, 4096
        )
        for capacity in (None, True, 0, 3, "4096"):
            c = config()
            c.model_config.max_model_len = capacity
            with self.subTest(capacity=capacity), self.assertRaises(ValueError):
                M.validate_eagle_single_request_review_config(c, experimental_graph_review=True)
        for window in (0, True, 1024.0, 1023, "1024"):
            c = config()
            c.additional_config["draft_window_size"] = window
            with self.subTest(window=window), self.assertRaises(ValueError):
                M.validate_eagle_single_request_review_config(c, experimental_graph_review=True)
        c = config()
        c.parallel_config.all2all_backend = "flashinfer_all2allv"
        c.parallel_config.use_sequence_parallel_moe = False
        self.assertIsInstance(
            M.validate_eagle_single_request_review_config(c, experimental_graph_review=True),
            M.EagleSingleRequestReviewScope,
        )
        for backend, sp in [
            ("allgather_reducescatter", False),
            ("flashinfer_all2allv", True),
            ("deepep_high_throughput", True),
            ("unknown", False),
        ]:
            c = config()
            c.parallel_config.all2all_backend = backend
            c.parallel_config.use_sequence_parallel_moe = sp
            with self.assertRaisesRegex(ValueError, "all2all_backend"):
                M.validate_eagle_single_request_review_config(c, experimental_graph_review=True)
        for target, key, value in [
            ("parallel_config", "tensor_parallel_size", 2),
            ("parallel_config", "data_parallel_size", 2),
            ("parallel_config", "prefill_context_parallel_size", 2),
            ("parallel_config", "decode_context_parallel_size", 2),
            ("parallel_config", "enable_expert_parallel", False),
            ("parallel_config", "use_sequence_parallel_moe", False),
            ("scheduler_config", "async_scheduling", True),
            ("cache_config", "enable_prefix_caching", True),
            ("speculative_config", "num_speculative_tokens", 4),
        ]:
            c = config()
            setattr(getattr(c, target), key, value)
            with self.assertRaises(ValueError):
                M.validate_eagle_single_request_review_config(c, experimental_graph_review=True)
        with self.assertRaises(ValueError):
            M.validate_eagle_single_request_review_config(config(), experimental_graph_review=False)

    def test_shared_and_eager_paths_remain_unchanged_when_no_scope(self):
        M.validate_draft_graph_review_requests(N(), N())
        M.validate_eagle_single_request_graph_stage(N(), None, first=None, n=None, tokens=None, ordinal=None)
        full = N(
            _draft_graph_review_limit="full8_local_active_at_most_one",
            max_model_len=40960,
            num_query_per_req=8,
            vllm_config=N(model_config=N(max_model_len=40960)),
        )
        M.validate_draft_graph_review_requests(
            full,
            N(
                num_reqs=1,
                num_reqs_after_padding=1,
                req_ids=["full8-A"],
                idx_mapping_np=[7],
                prefill_len_np=[33025],
                draft_graph_review_request_rows=(("full8-A", 7, 33025, 33089, 33025),),
            ),
        )
        with self.assertRaises(NotImplementedError):
            M.validate_draft_graph_review_requests(full, N(num_reqs=2, num_reqs_after_padding=2))

    def test_source_calls_have_no_new_device_operations(self):
        for rel in (
            REL,
            "vllm_ascend/worker/v2/spec_decode/draft_window_eagle3.py",
            "vllm_ascend/worker/v2/spec_decode/draft_window_eagle3_capture.py",
        ):
            tree = ast.parse((SOURCE / rel).read_text())
            if rel == REL:
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                        self.assertNotIn(node.func.attr, ("synchronize", "query", "cpu", "to", "copy_"))
        tree = ast.parse((SOURCE / "vllm_ascend/worker/v2/model_runner.py").read_text())
        self.assertTrue(
            any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "eagle_single_request_review_rows"
                for n in ast.walk(tree)
            )
        )

    def test_preempted_before_first_token_cannot_enter_graph(self):
        owner, batch = objects()
        M.note_eagle_single_request_review_lifecycle(owner, N(finished_req_ids=set(), preempted_req_ids={"actual-A"}))
        # Prompt/prefill lengths are still equal: this is exactly the old blind spot.
        self.assertEqual(batch.draft_graph_review_request_rows[0][2], batch.draft_graph_review_request_rows[0][4])
        with self.assertRaises(NotImplementedError):
            M.validate_draft_graph_review_requests(owner, batch)
        states = N(
            req_id_to_index={"actual-A": 7}, prompt_len=N(np={7: 257}), max_seq_len={7: 289}, prefill_len=N(np={7: 257})
        )
        with self.assertRaises(NotImplementedError):
            M.eagle_single_request_review_rows(owner, states, ["actual-A"], [7])
        # Explicit startup dummy/profile paths retain their existing bypass contract.
        M.validate_draft_graph_review_requests(owner, batch, dummy_run=True)
        M.validate_draft_graph_review_requests(owner, batch, is_profile=True)

    def test_finished_request_clears_only_its_preemption_record(self):
        owner, batch = objects()
        M.note_eagle_single_request_review_lifecycle(
            owner, N(finished_req_ids=set(), preempted_req_ids={"actual-A", "other-B"})
        )
        M.note_eagle_single_request_review_lifecycle(owner, N(finished_req_ids={"actual-A"}, preempted_req_ids=None))
        self.assertEqual(owner._draft_graph_review_preempted_request_ids, {"other-B"})
        M.validate_draft_graph_review_requests(owner, batch)
        untouched = N()
        M.note_eagle_single_request_review_lifecycle(
            untouched, N(finished_req_ids=set(), preempted_req_ids={"actual-A"})
        )
        self.assertFalse(hasattr(untouched, "_draft_graph_review_preempted_request_ids"))


def eagle_graph_protocol():
    """Execute actual Source ownership methods with no framework/device import."""
    import dataclasses
    from unittest.mock import patch

    base = SOURCE / "vllm_ascend/worker/v2/spec_decode"
    module = ast.parse((base / "draft_window_eagle3.py").read_text())
    scheduled = next(
        n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "_eagle3_graph_scheduled_consumers"
    )
    graph = ast.parse((base / "draft_window_eagle3_graph.py").read_text())
    nodes = [scheduled] + [
        n
        for n in graph.body
        if isinstance(n, ast.ClassDef) and n.name in ("Eagle3ScratchTransfer", "Eagle3GraphStagingOwner")
    ]
    states = N(BOUND=object(), UPDATED=object())
    ns = dict(
        __name__="_r8_eagle_protocol",
        dataclass=dataclasses.dataclass,
        DraftWindowGraphCarrierState=states,
        _current_stream_identity=lambda: 7,
    )
    sys.modules[ns["__name__"]] = types.ModuleType(ns["__name__"])
    tree = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), "<actual-eagle-owner-source>", "exec"), ns)
    helper_module = types.ModuleType("vllm_ascend.worker.v2.spec_decode.draft_window_eagle3")
    helper_module._eagle3_graph_scheduled_consumers = ns["_eagle3_graph_scheduled_consumers"]
    owner_type = ns["Eagle3GraphStagingOwner"]

    def make(scheduled_count, first_graph=False):
        runtime = N(
            capability=N(inventory=N(method="eagle3"), provider=N(expected_future_offset=3)),
            quarantined_identity=None,
            _active_inputs=None,
            _last_retired_identity=None,
        )
        provider = N(_runtime_state=runtime)
        owner = owner_type(runtime, provider, static_k=3, current_stream_identity=lambda: 7)
        with patch.dict(sys.modules, {helper_module.__name__: helper_module}):
            owner.begin(1, ("A",), first_pass_is_graph=first_graph, num_speculative_tokens=scheduled_count)
        return owner, runtime, provider

    ns["_source_helper_module"] = helper_module
    return ns, states, make


class Main544EagleControls(unittest.TestCase):
    def test_zero_still_requires_one_real_first_and_static_geometry(self):
        ns, states, make = eagle_graph_protocol()
        owner, runtime, _ = make(0)
        self.assertEqual((owner.static_k, owner._expected_steps), (3, 1))
        identity = N(step=0, normalized_method="eagle3", mapping_epoch=1, ordered_request_fingerprint=("A",))
        runtime._last_retired_identity = identity
        owner.adopt_eager_first_pass(identity)
        self.assertEqual(owner.finish(), ())
        self.assertEqual(owner.static_k, 3)
        owner, _, _ = make(0)
        with self.assertRaisesRegex(RuntimeError, "omitted a real"):
            owner.finish()
        self.assertTrue(owner.quarantined)

    def test_normal_k_requires_all_ar_and_wrong_zero_owner_rejected(self):
        _, _, make = eagle_graph_protocol()
        owner, runtime, _ = make(3)
        identity = N(step=0, normalized_method="eagle3", mapping_epoch=1, ordered_request_fingerprint=("A",))
        runtime._last_retired_identity = identity
        owner.adopt_eager_first_pass(identity)
        with self.assertRaisesRegex(RuntimeError, "omitted a real"):
            owner.finish()
        self.assertTrue(owner.quarantined)
        owner, runtime, _ = make(0)
        identity.ordered_request_fingerprint = ("stale",)
        runtime._last_retired_identity = identity
        with self.assertRaisesRegex(RuntimeError, "actual retired"):
            owner.adopt_eager_first_pass(identity)

    def test_partial_and_invalid_schedule_rejected_without_ownership(self):
        ns, _, make = eagle_graph_protocol()
        for scheduled in (-1, 1, 2, 4, True):
            with self.subTest(scheduled=scheduled), self.assertRaises(NotImplementedError):
                make(scheduled)
        self.assertEqual(ns["_eagle3_graph_scheduled_consumers"](3, None), 3)
        self.assertEqual(ns["_eagle3_graph_scheduled_consumers"](3, 0), 1)

    def test_zero_graph_first_needs_real_updated_bank_and_forbids_ar(self):
        ns, states, make = eagle_graph_protocol()
        route = N(EAGLE3_FIRST_PASS=object(), EAGLE3_AR_STEP=object())
        ns["DraftWindowGraphRoute"] = route

        def stage(owner, runtime, provider, step):
            identity = N(
                step=step,
                generation_id=step,
                mapping_epoch=1,
                normalized_method="eagle3",
                ordered_request_fingerprint=("A",),
            )
            buffers = (
                "window_block_table",
                "valid_num_blocks",
                "device_window_exact",
                "cpu_window_exact",
                "cpu_window_upper",
            )

            def buffer(address):
                return N(untyped_storage=lambda: N(data_ptr=lambda: address))

            inputs = N(
                identity=identity,
                validate_for=lambda actual: self.assertIs(actual, identity),
                **{name: buffer(index + 1) for index, name in enumerate(buffers)},
            )
            carrier = N(
                state=states.BOUND, read_inputs=N(**{name: buffer(index + 101) for index, name in enumerate(buffers)})
            )
            provider.bind_read_carrier = lambda **kwargs: carrier
            runtime.retire_generation = lambda actual: setattr(runtime, "_last_retired_identity", actual)
            capture = N(
                consumer_ordinal=0 if step == 0 else step - 1,
                fingerprint=N(
                    route=route.EAGLE3_FIRST_PASS if step == 0 else route.EAGLE3_AR_STEP,
                    static_k=3,
                    query_width=4 if step == 0 else 1,
                    graph_mode="FULL",
                ),
            )
            owner.bind_and_release_scratch(
                capture_identity=capture,
                inputs=inputs,
                metadata_builder=lambda read: None,
                target_metadata={},
                write_side_reference=None,
            )
            return carrier

        owner, runtime, provider = make(0, True)
        stage(owner, runtime, provider, 0)
        with self.assertRaisesRegex(RuntimeError, "UPDATE attribution"):
            owner.finish()
        owner, runtime, provider = make(0, True)
        carrier = stage(owner, runtime, provider, 0)
        carrier.state = states.UPDATED
        self.assertEqual(len(owner.finish()), 1)
        owner, runtime, provider = make(0, True)
        carrier = stage(owner, runtime, provider, 0)
        carrier.state = states.UPDATED
        with self.assertRaisesRegex(RuntimeError, "proposal/step"):
            stage(owner, runtime, provider, 1)
        self.assertTrue(owner.quarantined)

    def test_actual_zero_scope_completes_first_only_and_failed_scope_isolates(self):
        from contextlib import contextmanager
        from unittest.mock import patch

        self.assertNotIn("torch", sys.modules)
        ns, _, make = eagle_graph_protocol()
        path = SOURCE / "vllm_ascend/worker/v2/spec_decode/draft_window_eagle3.py"
        module = ast.parse(path.read_text())
        actual = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "Eagle3DraftWindowMixin")
        scope = next(
            n for n in actual.body if isinstance(n, ast.FunctionDef) and n.name == "_eagle3_draft_window_proposal_scope"
        )
        env = dict(
            contextmanager=contextmanager,
            _requires_prefill_baseline_bypass=lambda batch: False,
            _eagle3_graph_scheduled_consumers=ns["_eagle3_graph_scheduled_consumers"],
        )
        tree = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), scope],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(tree), "<actual-eagle-proposal-scope>", "exec"), env)

        def state():
            controller, runtime, _ = make(0)
            # Start the actual scope with a fresh, unowned controller.
            controller._proposal_epoch = controller._request_fingerprint = controller._stream = None
            controller._last_epoch = -1
            finished = []
            context = N(mapping_epoch=1, ordered_request_ids=("A",))
            owner = N(
                _eagle3_window_epoch_source=N(begin=lambda batch: context, finish=lambda c: finished.append(c)),
                _eagle3_window_static_k=3,
                _eagle3_window_expected_steps=3,
                _eagle3_graph_session=N(controller=controller),
                _draft_window_runtime_state=runtime,
            )
            return owner, controller, runtime, finished

        owner, controller, runtime, finished = state()
        with env["_eagle3_draft_window_proposal_scope"](
            owner, N(), dummy_run=False, is_profile=False, num_speculative_tokens=0
        ):
            self.assertEqual((owner._eagle3_window_scheduled_tokens, owner._eagle3_window_expected_steps), (0, 1))
            helper = ns["_source_helper_module"]
            with patch.dict(sys.modules, {helper.__name__: helper}):
                controller.begin(1, ("A",), first_pass_is_graph=False, num_speculative_tokens=0)
            identity = N(step=0, normalized_method="eagle3", mapping_epoch=1, ordered_request_fingerprint=("A",))
            runtime._last_retired_identity = identity
            controller.adopt_eager_first_pass(identity)
            owner._eagle3_window_next_step = 1
        self.assertEqual(len(finished), 1)
        self.assertFalse(controller.quarantined)
        self.assertIsNone(owner._eagle3_window_proposal_context)
        self.assertEqual((owner._eagle3_window_static_k, owner._eagle3_window_expected_steps), (3, 3))
        owner, controller, runtime, finished = state()
        with (
            self.assertRaisesRegex(RuntimeError, "without consuming all"),
            env["_eagle3_draft_window_proposal_scope"](
                owner, N(), dummy_run=False, is_profile=False, num_speculative_tokens=0
            ),
        ):
            pass
        self.assertTrue(controller.quarantined)
        self.assertTrue(owner._draft_window_graph_submission_uncertain)
        self.assertEqual(len(finished), 1)
        self.assertNotIn("torch", sys.modules)

    def test_skip_sample_forwarding_and_new_metadata_switch(self):
        path = SOURCE / "vllm_ascend/worker/v2/spec_decode/draft_window_eagle3_adapter.py"
        original = ast.parse(path.read_text())
        actual = next(
            n for n in original.body if isinstance(n, ast.ClassDef) and n.name == "Eagle3GraphIntegrationMixin"
        )
        selected = [
            n
            for n in actual.body
            if isinstance(n, ast.FunctionDef) and n.name in ("_prefill", "_configure_decode_metadata_update")
        ]

        class Parent:
            def _configure_decode_metadata_update(self):
                self.supports_decode_metadata_update = True

            def _prefill(self, *args, skip_sample=False):
                self.calls.append((args, skip_sample))
                if getattr(self, "raise_body", False):
                    raise RuntimeError("body failed")
                return "real-body-return"

        ns = dict(Parent=Parent, CUDAGraphMode=N(NONE="NONE"))
        cls = ast.ClassDef(
            name="ActualBridge",
            bases=[ast.Name(id="Parent", ctx=ast.Load())],
            keywords=[],
            body=selected,
            decorator_list=[],
        )
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                "<actual-eagle-adapter-methods>",
                "exec",
            ),
            ns,
        )
        owner = ns["ActualBridge"]()
        owner.calls = []
        owner._eagle3_window_is_active = lambda: False
        self.assertEqual(owner._prefill(1, 7, {}, "FullSlots", None, skip_sample=True), "real-body-return")
        self.assertTrue(owner.calls[-1][1])
        owner._configure_decode_metadata_update()
        self.assertTrue(owner.supports_decode_metadata_update)
        owner._draft_window_policy = object()
        owner._configure_decode_metadata_update()
        self.assertFalse(owner.supports_decode_metadata_update)
        owner._eagle3_window_is_active = lambda: True
        owner.replicated_pcp = False
        owner._eagle3_window_scheduled_tokens = 0
        identity = object()
        owner._require_eagle3_window_context = lambda: N(mapping_epoch=1, ordered_request_ids=("A",))
        recorded = []
        owner._eagle3_graph_session = N(
            controller=N(
                begin=lambda *a, **kw: recorded.append(("begin", kw)),
                adopt_eager_first_pass=lambda value: recorded.append(("adopt", value)),
            )
        )
        owner._require_eagle3_target_metadata = lambda value: value
        owner._materialize_eagle3_first_pass = lambda **kw: N(identity=identity)
        owner._build_eagle3_private_attn_metadata = lambda **kw: {"private": "canonical-derived"}
        owner.input_batch = N(query_start_loc_np="original-query")
        owner._finish_eagle3_window_step = lambda inputs, **kw: recorded.append(("finish", kw["succeeded"]))
        owner._prefill(1, 7, {}, "FullSlots", None, skip_sample=True)
        self.assertEqual(
            recorded,
            [
                ("begin", {"first_pass_is_graph": False, "num_speculative_tokens": 0}),
                ("finish", True),
                ("adopt", identity),
            ],
        )
        self.assertTrue(owner.calls[-1][1])
        self.assertEqual(owner.calls[-1][0][3], "FullSlots")
        owner.raise_body = True
        recorded.clear()
        with self.assertRaisesRegex(RuntimeError, "body failed"):
            owner._prefill(1, 7, {}, "FullSlots", None, skip_sample=True)
        self.assertEqual(recorded[-1], ("finish", False))
        self.assertFalse(any(row[0] == "adopt" for row in recorded))


if __name__ == "__main__":
    print(
        json.dumps(
            dict(
                actual_test_source=str(SOURCE),
                source_sha256=hashlib.sha256((SOURCE / REL).read_bytes()).hexdigest(),
                scope="STDLIB_PROTOCOL_AND_AST_ONLY_NO_FRAMEWORK",
            )
        )
    )
    unittest.main()
