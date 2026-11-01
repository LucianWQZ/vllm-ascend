# SPDX-License-Identifier: Apache-2.0
"""Shape and ownership guards, independent of the old experiment entry names."""

import ast
import copy
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/spec_decode/draft_graph_review_limits.py"
spec = importlib.util.spec_from_file_location("draft_graph_support_test", MODULE_PATH)
assert spec is not None and spec.loader is not None
GUARD = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = GUARD
spec.loader.exec_module(GUARD)


def actual_source_function(path, name, namespace, *, class_name=None):
    module = ast.parse(path.read_text())
    nodes = (
        module.body
        if class_name is None
        else next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == class_name).body
    )
    function = next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == name)
    tree = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    return namespace[name]


class DraftGraphSupportTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def config(self, *, method="eagle3", rows=1, dp=1, window=True):
        profile = "eagle3" if method == "eagle3" else "dflash_full8"
        model = copy.deepcopy(GUARD._QUALIFIED_MODEL_CONFIGS[profile])
        directory = Path(self.directory.name) / f"{method}-{rows}-{dp}-{window}"
        directory.mkdir(exist_ok=True)
        (directory / "config.json").write_text(json.dumps(model))
        parallel = dict(
            tensor_parallel_size=4,
            data_parallel_size=dp,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            enable_expert_parallel=True,
            all2all_backend="allgather_reducescatter",
            use_sequence_parallel_moe=True,
        )
        query = 4 if method == "eagle3" else 8
        k = query - 1
        hf = NS(**model)
        draft = NS(model=str(directory), hf_config=hf, hf_text_config=hf, dtype="torch.bfloat16")
        return NS(
            speculative_config=NS(
                method=method,
                enforce_eager=False,
                num_speculative_tokens=k,
                uses_dynamic_speculative_decoding=lambda: False,
                enable_adaptive_verification=False,
                draft_model_config=draft,
                draft_parallel_config=NS(
                    tensor_parallel_size=4,
                    data_parallel_size=1,
                    enable_expert_parallel=False,
                    use_sequence_parallel_moe=False,
                ),
            ),
            parallel_config=NS(**parallel),
            scheduler_config=NS(max_num_seqs=rows, async_scheduling=True),
            cache_config=NS(enable_prefix_caching=True),
            compilation_config=NS(
                cudagraph_mode=NS(name="FULL_DECODE_ONLY"),
                cudagraph_capture_sizes=[query * n for n in range(1, rows + 1)],
                max_cudagraph_capture_size=query * rows,
            ),
            model_config=NS(max_model_len=4096 if method == "eagle3" else 40960, hf_config=hf),
            additional_config={"draft_window_size": (1024 if method == "eagle3" else 32768) if window else None},
        )

    def resolve(self, config):
        return GUARD.resolve_draft_graph_review_limit(config, experimental_graph_review=True)

    def test_capacities_without_experiment_flags(self):
        for method in ("eagle3", "dflash"):
            for dp, rows in ((1, 1), (1, 2), (2, 1)):
                for window in (False, True):
                    with self.subTest(method=method, dp=dp, rows=rows, window=window):
                        scope = self.resolve(self.config(method=method, dp=dp, rows=rows, window=window))
                        self.assertEqual(
                            (scope.method, scope.data_parallel_size, scope.max_requests), (method, dp, rows)
                        )

    def test_dp2_local_pair_and_three_requests_stay_blocked(self):
        for method in ("eagle3", "dflash"):
            for dp, rows in ((2, 2), (1, 3)):
                with self.subTest(method=method, dp=dp, rows=rows), self.assertRaises(NotImplementedError):
                    self.resolve(self.config(method=method, dp=dp, rows=rows))

    def test_eager_and_native_off_do_not_read_or_validate_review_identity(self):
        config = self.config(window=False)
        config.additional_config["draft_window_graph_eagle_two_request_review"] = True
        config.speculative_config.enforce_eager = True
        with patch.object(Path, "read_text", side_effect=AssertionError("unexpected config IO")):
            self.assertIsNone(self.resolve(config))
            config.speculative_config.enforce_eager = False
            self.assertIsNone(GUARD.resolve_draft_graph_review_limit(config, experimental_graph_review=False))

    def test_startup_contract_is_reused_without_file_io(self):
        config = self.config(rows=2)
        scope = self.resolve(config)
        with patch.object(Path, "read_text", side_effect=AssertionError("duplicate startup parse")):
            self.assertIs(self.resolve(config), scope)

    def test_config_packaging_changes_do_not_change_support(self):
        config = self.config()
        path = Path(config.speculative_config.draft_model_config.model) / "config.json"
        model = json.loads(path.read_text())
        model.update(transformers_version="a packaging version", initializer_range=0.3, _name_or_path="elsewhere")
        path.write_text(json.dumps(model, indent=1))
        self.assertEqual(self.resolve(config).method, "eagle3")

    def test_unknown_semantics_and_changed_aux_or_heads_do_not_open(self):
        for field, value in (
            ("unrecognized_attention_mode", "new"),
            ("num_key_value_heads", 4),
            ("eagle_config", {"eagle_aux_hidden_state_layer_ids": [2, 18, 34]}),
            ("rope_parameters", {"rope_type": "default", "rope_theta": 10000}),
        ):
            config = self.config()
            path = Path(config.speculative_config.draft_model_config.model) / "config.json"
            model = json.loads(path.read_text())
            model[field] = value
            path.write_text(json.dumps(model))
            with self.subTest(field=field), self.assertRaises(NotImplementedError):
                self.resolve(config)

    def test_swa_counterexample_remains_blocked_with_window_off(self):
        config = self.config(method="dflash", window=False)
        path = Path(config.speculative_config.draft_model_config.model) / "config.json"
        path.write_text(json.dumps(GUARD._QUALIFIED_MODEL_CONFIGS["dflash_swa6"]))
        with self.assertRaisesRegex(NotImplementedError, "valid-output"):
            self.resolve(config)

    def test_diagnostic_flag_does_not_bypass_failed_model(self):
        config = self.config(method="dflash")
        path = Path(config.speculative_config.draft_model_config.model) / "config.json"
        path.write_text(json.dumps(GUARD._QUALIFIED_MODEL_CONFIGS["dflash_swa6"]))
        config.additional_config["draft_window_graph_numeric_diagnostic"] = True
        with self.assertRaises(NotImplementedError):
            self.resolve(config)

    def test_missing_bucket_and_wrong_draft_descriptor_are_rejected(self):
        for damage in ("bucket", "draft_dp", "draft_sp", "capacity"):
            config = self.config(rows=2)
            if damage == "bucket":
                config.compilation_config.cudagraph_capture_sizes = [4]
            elif damage == "draft_dp":
                config.speculative_config.draft_parallel_config.data_parallel_size = 2
            elif damage == "draft_sp":
                config.speculative_config.draft_parallel_config.use_sequence_parallel_moe = True
            else:
                config.model_config.max_model_len = 3
            with self.subTest(damage=damage), self.assertRaises(NotImplementedError):
                self.resolve(config)

    def batch(self, *, rows=1):
        records = tuple((f"request-{i}", i, 1021, 1149, 1021) for i in range(rows))
        return NS(
            num_reqs=rows,
            num_reqs_after_padding=rows,
            req_ids=[r[0] for r in records],
            idx_mapping_np=list(range(rows)),
            prefill_len_np=[1021] * rows,
            draft_graph_review_request_rows=records,
        )

    def test_actual_rows_and_request_identity_not_client_concurrency(self):
        scope = self.resolve(self.config(rows=2))
        owner = NS(_draft_graph_review_limit=scope, max_model_len=scope.max_model_len)
        batch = self.batch(rows=2)
        GUARD.validate_draft_graph_review_requests(owner, batch)
        batch.num_reqs_after_padding = 4
        with self.assertRaises(NotImplementedError):
            GUARD.validate_draft_graph_review_requests(owner, batch)
        batch.num_reqs_after_padding = 2
        batch.req_ids[1] = batch.req_ids[0]
        with self.assertRaises(NotImplementedError):
            GUARD.validate_draft_graph_review_requests(owner, batch)

    def test_preemption_state_is_kept_until_actual_finish(self):
        scope = self.resolve(self.config())
        owner = NS(_draft_graph_review_limit=scope, max_model_len=scope.max_model_len)
        batch = self.batch()
        GUARD.note_eagle_single_request_review_lifecycle(
            owner, NS(finished_req_ids=set(), preempted_req_ids={"request-0"})
        )
        with self.assertRaises(NotImplementedError):
            GUARD.validate_draft_graph_review_requests(owner, batch)
        GUARD.note_eagle_single_request_review_lifecycle(
            owner, NS(finished_req_ids={"request-0"}, preempted_req_ids=set())
        )
        GUARD.validate_draft_graph_review_requests(owner, batch)

    def test_normalized_model_conflict_is_not_hidden_by_original_config(self):
        config = self.config()
        config.speculative_config.draft_model_config.hf_config.head_dim = 128
        with self.assertRaisesRegex(NotImplementedError, "Normalized Draft head_dim"):
            self.resolve(config)

    def test_exact_lengths_follow_canonical_continuing_prefill_progress(self):
        class Exact:
            device = NS(type="cpu")
            ndim, shape, dtype = 1, (1,), "torch.int32"

            def __init__(self, value):
                self.value = value

            def __getitem__(self, index):
                return NS(item=lambda: self.value)

        scope = self.resolve(self.config())
        owner = NS(
            _draft_graph_review_limit=scope,
            max_model_len=scope.max_model_len,
            _eagle3_window_proposal_context=NS(phase_plan=NS(baseline_only_rows=(0,))),
        )
        batch = self.batch()
        batch.num_computed_prefill_tokens_np = [512]
        batch.num_scheduled_tokens = [128]
        for step in (0, 1, 2):
            GUARD.validate_eagle_single_request_review_exact(
                owner, batch, Exact(640 + step), num_reqs=1, padded=1, step=step
            )
        for value in (639, 642, 1021):
            with self.subTest(value=value), self.assertRaises(NotImplementedError):
                GUARD.validate_eagle_single_request_review_exact(
                    owner, batch, Exact(value), num_reqs=1, padded=1, step=0
                )
        batch.num_scheduled_tokens = [509]
        with self.assertRaises(RuntimeError):
            GUARD.validate_eagle_single_request_review_exact(owner, batch, Exact(1021), num_reqs=1, padded=1, step=0)

    def test_full8_request_lookahead_cannot_exceed_allocated_capacity(self):
        config = self.config(method="dflash")
        scope = self.resolve(config)
        owner = NS(
            _draft_graph_review_limit=scope,
            max_model_len=scope.max_model_len,
            num_query_per_req=8,
            vllm_config=config,
        )
        batch = self.batch()
        GUARD.validate_draft_graph_review_requests(owner, batch)
        batch.draft_graph_review_request_rows = (("request-0", 0, 1021, 40960, 1021),)
        with self.assertRaises(NotImplementedError):
            GUARD.validate_draft_graph_review_requests(owner, batch)

    def test_full8_descriptor_rejects_unqualified_padding_before_execution(self):
        scope = self.resolve(self.config(method="dflash", rows=2))
        owner = NS(_draft_graph_review_limit=scope)
        GUARD.validate_full8_single_request_graph_stage(owner, actual=1, padded=1, tokens=8)
        GUARD.validate_full8_single_request_graph_stage(owner, actual=2, padded=2, tokens=16)
        for shape in ((1, 2, 16), (2, 4, 32), (3, 3, 24)):
            with self.subTest(shape=shape), self.assertRaises(NotImplementedError):
                GUARD.validate_full8_single_request_graph_stage(
                    owner, actual=shape[0], padded=shape[1], tokens=shape[2]
                )

    def test_context_only_full8_keeps_request_guards_without_fake_query_shape(self):
        config = self.config(method="dflash")
        scope = self.resolve(config)
        owner = NS(
            _draft_graph_review_limit=scope,
            max_model_len=scope.max_model_len,
            num_query_per_req=8,
            num_speculative_steps=7,
            vllm_config=config,
            _draft_window_runtime_state=object(),
        )
        batch = self.batch()
        batch.num_reqs_after_padding = 2
        seen = []
        namespace = {
            "nullcontext": nullcontext,
            "_draft_window_context_kv_scope": lambda o, b: (seen.append("context-KV-only"), nullcontext())[1],
        }
        entry = actual_source_function(
            MODULE_PATH.with_name("dflash") / "speculator.py", "_draft_window_owner_scope", namespace
        )
        with patch.dict(sys.modules, {"vllm_ascend.worker.v2.spec_decode.draft_graph_review_limits": GUARD}):
            with entry(owner, batch, num_speculative_tokens=0):
                pass
            self.assertEqual(seen, ["context-KV-only"])
            with self.assertRaises(NotImplementedError):
                entry(owner, batch, num_speculative_tokens=7)
            batch.req_ids[0] = "wrong-original-request"
            with self.assertRaises(RuntimeError):
                entry(owner, batch, num_speculative_tokens=0)
        self.assertFalse(hasattr(owner, "_draft_window_runtime"))

    def test_actual_propose_rejects_bad_rows_before_replacing_owner(self):
        config = self.config(method="dflash")
        scope = self.resolve(config)
        old_batch = object()
        owner = NS(
            _draft_graph_review_limit=scope,
            max_model_len=scope.max_model_len,
            num_query_per_req=8,
            num_speculative_steps=7,
            vllm_config=config,
            input_batch=old_batch,
        )
        batch = self.batch(rows=2)
        path = MODULE_PATH.with_name("dflash") / "speculator.py"
        namespace = {"nullcontext": nullcontext}
        entry = actual_source_function(path, "_draft_window_owner_scope", namespace)
        propose = actual_source_function(
            path, "propose", {"_draft_window_owner_scope": entry}, class_name="AscendDFlashSpeculator"
        )
        with (
            patch.dict(sys.modules, {"vllm_ascend.worker.v2.spec_decode.draft_graph_review_limits": GUARD}),
            self.assertRaises(NotImplementedError),
        ):
            propose(owner, batch, *([None] * 10), num_speculative_tokens=7)
        self.assertIs(owner.input_batch, old_batch)


SOURCE = ROOT = MODULE_PATH.parents[4]
M = GUARD
N = NS


def _single_owner_batch(prompt=257, output=32, capacity=4096):
    scope = GUARD.DraftGraphSupport(capacity, 3, 1, 1, "eagle3", 4)
    owner = NS(_draft_graph_review_limit=scope, max_model_len=capacity)
    batch = NS(
        num_reqs=1,
        num_reqs_after_padding=1,
        req_ids=["actual-A"],
        idx_mapping_np=[7],
        prefill_len_np=[prompt],
        draft_graph_review_request_rows=(("actual-A", 7, prompt, prompt + output, prompt),),
    )
    owner.input_batch = batch
    return owner, batch


def _pair_owner_batch():
    owner = NS(_draft_graph_review_limit=GUARD.DraftGraphSupport(4096, 3, 2, 1, "eagle3", 4), max_model_len=4096)
    batch = NS(
        num_reqs=2,
        num_reqs_after_padding=2,
        req_ids=["request-A", "request-B"],
        idx_mapping_np=[7, 3],
        prefill_len_np=[1537, 513],
        draft_graph_review_request_rows=(("request-A", 7, 1537, 1569, 1537), ("request-B", 3, 513, 545, 513)),
    )
    owner.input_batch = batch
    return owner, batch


def actual_ar_replay_method():
    """Use the real manager entry, replacing no framework/device implementation."""
    path = SOURCE / "vllm_ascend/worker/v2/spec_decode/target_dependent_ar/aclgraph.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AutoRegressiveAclGraphManager")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "run_fullgraph")
    scope = dict(logger=N(info_once=lambda *args: None), use_updatable_graph=lambda backend: False)
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method]
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), "exec"), scope)
    return scope["run_fullgraph"]


class ActualGraphDescriptorControls(unittest.TestCase):
    def manager(self, owner, *, first=False, session=None):
        seen = []
        owner.attn_backend = "actual-test-backend"
        owner.draft_vllm_config = object()
        owner.build_draft_attn_metadatas = lambda *args: (seen.append(("metadata", args)), {})[1]
        if session is not None:
            owner._eagle3_graph_session = session
            owner._eagle3_window_is_active = lambda: True
        manager = N(
            speculator=owner,
            is_draft_model_prefill=first,
            update_stream=object(),
            _graph_replay=lambda *args: (seen.append(("replay", args)), "legacy-result")[1],
            _run_eagle_window_fullgraph=lambda *args: (seen.append(("window", args)), "window-result")[1],
        )
        return manager, seen

    def run_actual(self, manager, desc):
        from unittest.mock import patch

        with patch.dict(sys.modules, {"vllm_ascend.worker.v2.spec_decode.draft_graph_review_limits": M}):
            return actual_ar_replay_method()(manager, desc)

    def test_off_padded_ar_rejected_before_metadata_and_replay(self):
        owner, _ = _single_owner_batch()
        manager, seen = self.manager(owner)
        for n, tokens in ((4, 4), (1, 4), (2, 2)):
            with self.subTest(n=n, tokens=tokens), self.assertRaisesRegex(NotImplementedError, "padded1"):
                self.run_actual(manager, N(num_reqs=n, num_tokens=tokens))
            self.assertEqual(seen, [])
        self.assertEqual(self.run_actual(manager, N(num_reqs=1, num_tokens=1)), "legacy-result")
        self.assertEqual([v[0] for v in seen], ["metadata", "replay"])
        self.assertFalse(hasattr(owner, "_draft_window_runtime_state"))
        self.assertFalse(hasattr(owner, "_draft_window_runtime"))

    def test_first_q4_and_existing_on_identity_checks(self):
        owner, _ = _single_owner_batch()
        manager, seen = self.manager(owner, first=True)
        self.assertEqual(self.run_actual(manager, N(num_reqs=1, num_tokens=4)), "legacy-result")
        seen.clear()
        with self.assertRaisesRegex(NotImplementedError, "first Q4"):
            self.run_actual(manager, N(num_reqs=4, num_tokens=4))
        self.assertEqual(seen, [])
        for first, step, ordinal, tokens in ((True, 0, 0, 4), (False, 1, 0, 1), (False, 2, 1, 1)):
            inputs = N(actual_request_count=1, padded_request_count=1, identity=N(step=step))
            M.validate_eagle_single_request_graph_stage(owner, inputs, first=first, n=1, tokens=tokens, ordinal=ordinal)
            inputs.identity.step = 7
            with self.assertRaises(NotImplementedError):
                M.validate_eagle_single_request_graph_stage(
                    owner, inputs, first=first, n=1, tokens=tokens, ordinal=ordinal
                )

    def test_window_session_stays_on_its_existing_stronger_stage_path(self):
        owner, _ = _single_owner_batch()
        manager, seen = self.manager(owner, session=object())
        self.assertEqual(self.run_actual(manager, N(num_reqs=1, num_tokens=1)), "window-result")
        self.assertEqual([v[0] for v in seen], ["window"])

    def test_unscoped_and_real_source_dummy_keep_legacy_behavior(self):
        for owner in (
            N(),
            N(
                _draft_graph_review_limit=GUARD.DraftGraphSupport(4096, 3, 1, 1, "eagle3", 4),
                input_batch=N(is_dummy=True),
            ),
        ):
            manager, seen = self.manager(owner)
            self.assertEqual(self.run_actual(manager, N(num_reqs=4, num_tokens=4)), "legacy-result")
            self.assertEqual([v[0] for v in seen], ["metadata", "replay"])
        owner, batch = _single_owner_batch()
        batch.is_dummy = False
        batch.num_reqs = 2
        manager, seen = self.manager(owner)
        with self.assertRaises(NotImplementedError):
            self.run_actual(manager, N(num_reqs=1, num_tokens=1))
        self.assertEqual(seen, [])


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


class Vector:
    def __init__(self, values):
        self.values = values
        self.ndim = 1
        self.shape = (len(values),)
        self.dtype = "torch.int32"
        self.device = NS(type="cpu")

    def __getitem__(self, index):
        return NS(item=lambda: self.values[index])


class EagleMixedRequestControls(unittest.TestCase):
    def test_mixed_chunked_prefill_uses_canonical_progress_for_each_consumer(self):
        # C333 ordinary g0 failed with these saved SchedulerOutput counters:
        # long prompt 1537: 1152 computed + 128 scheduled = 1280; short prompt
        # 513: 384 + 129 = 513. These are Source-reconstructed lengths, not a
        # saved device/guard exact snapshot. Execute the actual Ascend CPU
        # length constructor on that state, without importing the model runner.
        path = ROOT / "vllm_ascend/worker/v2/model_runner.py"
        module = ast.parse(path.read_text())
        runner = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
        function = next(
            node for node in runner.body if isinstance(node, ast.FunctionDef) and node.name == "_update_seq_lens_cpu"
        )
        namespace = {}
        code = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(code), str(path), "exec"), namespace)
        owner, batch = _pair_owner_batch()
        ids = batch.req_ids
        cpu_lengths = [0, 0]
        snapshots = {7: 1152, 3: 384}
        runner = NS(
            speculator=object(),
            sync_spec_pp_cpu_counts=False,
            _can_skip_spec_cpu_length_sync=lambda: False,
            num_computed_tokens_event=NS(synchronize=lambda: None),
            num_computed_tokens_cpu=snapshots.copy(),
            req_states=NS(req_id_to_index=dict(zip(ids, batch.idx_mapping_np)), num_computed_tokens_cpu={}),
            input_buffers=NS(seq_lens_cpu=cpu_lengths),
        )
        scheduler = NS(
            # Scheduled-token dict order differs from request/source order.
            num_scheduled_tokens={ids[1]: 129, ids[0]: 128},
            scheduled_cached_reqs=NS(req_ids=ids),
        )
        namespace["_update_seq_lens_cpu"](runner, scheduler, ids)
        self.assertEqual(cpu_lengths, [1280, 513])
        batch.num_computed_prefill_tokens_np = [1152, 384]
        batch.num_scheduled_tokens = [128, 129]
        owner._eagle3_window_proposal_context = NS(phase_plan=NS(baseline_only_rows=(0,)))
        for step in (0, 1, 2):
            M.validate_eagle_single_request_review_exact(
                owner, batch, Vector([1280 + step, 513 + step]), num_reqs=2, padded=2, step=step
            )
            for values in ([1279 + step, 513 + step], [1281 + step, 513 + step], [1280 + step, 512]):
                with self.subTest(step=step, values=values), self.assertRaises(NotImplementedError):
                    M.validate_eagle_single_request_review_exact(
                        owner, batch, Vector(values), num_reqs=2, padded=2, step=step
                    )
        # Absent/changed phase cannot relabel a terminal/decode row as continuing.
        owner._eagle3_window_proposal_context = NS(phase_plan=NS(baseline_only_rows=()))
        with self.assertRaises(NotImplementedError):
            M.validate_eagle_single_request_review_exact(
                owner, batch, Vector(cpu_lengths), num_reqs=2, padded=2, step=0
            )
        owner._eagle3_window_proposal_context = NS(phase_plan=NS(baseline_only_rows=(1,)))
        with self.assertRaises(RuntimeError):
            M.validate_eagle_single_request_review_exact(
                owner, batch, Vector([1537, 513]), num_reqs=2, padded=2, step=0
            )

    def test_continuing_prefill_does_not_relax_identity_capacity_or_counter_contract(self):
        for damage in (
            "wrong-id",
            "bad-index",
            "capacity",
            "preempted",
            "missing",
            "fractional",
            "negative",
            "terminal",
            "zero",
        ):
            owner, batch = _pair_owner_batch()
            batch.num_computed_prefill_tokens_np = [1152, 384]
            batch.num_scheduled_tokens = [128, 129]
            owner._eagle3_window_proposal_context = NS(phase_plan=NS(baseline_only_rows=(0,)))
            if damage == "wrong-id":
                batch.req_ids[0] = "different-request"
            elif damage == "bad-index":
                batch.idx_mapping_np[0] = 6
            elif damage == "capacity":
                owner.max_model_len = 2048
            elif damage == "preempted":
                owner._draft_graph_review_preempted_request_ids = {batch.req_ids[0]}
            elif damage == "missing":
                del batch.num_scheduled_tokens
            elif damage == "fractional":
                batch.num_scheduled_tokens[0] = 128.5
            elif damage == "negative":
                batch.num_computed_prefill_tokens_np[0] = -1
            elif damage == "terminal":
                batch.num_scheduled_tokens[0] = 385
            else:
                batch.num_scheduled_tokens[0] = 0
            with self.subTest(damage=damage), self.assertRaises((RuntimeError, NotImplementedError)):
                M.validate_eagle_single_request_review_exact(
                    owner, batch, Vector([1280, 513]), num_reqs=2, padded=2, step=0
                )


class Controller:
    def __init__(self):
        self._proposal_epoch = None
        self._owner_stream = 17
        self._stream_identity = lambda: 17
        self.quarantined = False

    def require_reusable(self):
        if self.quarantined:
            raise RuntimeError("quarantined")

    def quarantine(self, _):
        self.quarantined = True


def idle_owner():
    provider = NS(inventory_method="eagle3")
    session = NS(controller=Controller(), provider=provider, _capture_call=None)
    batch = NS(is_dummy=True, num_reqs=1, num_reqs_after_padding=1)
    owner = NS(
        method="eagle3",
        dp_size=2,
        num_speculative_steps=3,
        max_num_reqs=1,
        input_batch=batch,
        vllm_config=NS(additional_config={}),
        _draft_graph_review_limit=GUARD.DraftGraphSupport(4096, 3, 1, 2, "eagle3", 4),
        update_stream=object(),
        _eagle3_graph_session=session,
        _eagle3_window_bypass_active=True,
        _eagle3_window_continuing_prefill=False,
        _draft_window_graph_binding_provider=provider,
        _draft_window_runtime_state=NS(_active_inputs=None),
    )
    return owner, batch, session


DP_MODULE_PATH = SOURCE / "vllm_ascend/worker/v2/spec_decode/draft_window_eagle3_dp.py"
dp_spec = importlib.util.spec_from_file_location("draft_graph_idle_test", DP_MODULE_PATH)
assert dp_spec is not None and dp_spec.loader is not None
DP = importlib.util.module_from_spec(dp_spec)
dp_spec.loader.exec_module(DP)
IDLE = vars(DP)


class EagleIdleOwnershipControls(unittest.TestCase):
    def test_actual_idle_uses_full_baseline_and_both_consumer_managers(self):
        owner, batch, session = idle_owner()
        with IDLE["eagle3_dp_dummy_scope"](owner, batch, dummy_run=True, is_profile=False):
            self.assertTrue(owner._eagle3_graph_dp_dummy_active)
            for first, tokens, count in ((True, 4, 1), (False, 1, 2)):
                manager = NS(speculator=owner, update_stream=owner.update_stream, is_draft_model_prefill=first)
                descriptor = NS(cg_mode=NS(name="FULL"), num_reqs=1, num_tokens=tokens)
                self.assertTrue(IDLE["validate_eagle_dp_dummy_replay"](owner, manager, descriptor, [{}] * count))
                with self.assertRaises(RuntimeError):
                    IDLE["validate_eagle_dp_dummy_replay"](owner, manager, descriptor, [{}] * (count + 1))
            self.assertIsNone(session.controller._proposal_epoch)
        self.assertFalse(owner._eagle3_graph_dp_dummy_active)
        self.assertFalse(session.controller.quarantined)
        # K0 business first-only is a genuine request, never dummy/idle.
        batch.is_dummy = False
        with IDLE["eagle3_dp_dummy_scope"](owner, batch, dummy_run=False, is_profile=False):
            self.assertFalse(owner._eagle3_graph_dp_dummy_active)

    def test_idle_failure_or_live_generation_cannot_reuse(self):
        owner, batch, session = idle_owner()
        with (
            self.assertRaisesRegex(RuntimeError, "injected"),
            IDLE["eagle3_dp_dummy_scope"](owner, batch, dummy_run=True, is_profile=False),
        ):
            raise RuntimeError("injected submission uncertainty")
        self.assertTrue(session.controller.quarantined)
        self.assertTrue(owner._draft_window_graph_submission_uncertain)
        self.assertFalse(owner._eagle3_graph_dp_dummy_active)
        for field in ("_eagle3_window_proposal_context", "_eagle3_window_live_inputs"):
            owner, batch, session = idle_owner()
            setattr(owner, field, object())
            with (
                self.assertRaises(RuntimeError),
                IDLE["eagle3_dp_dummy_scope"](owner, batch, dummy_run=True, is_profile=False),
            ):
                self.fail("live business generation entered dummy replay")
            self.assertTrue(session.controller.quarantined)
        owner, batch, session = idle_owner()
        owner._draft_window_runtime_state._active_inputs = object()
        with (
            self.assertRaises(RuntimeError),
            IDLE["eagle3_dp_dummy_scope"](owner, batch, dummy_run=True, is_profile=False),
        ):
            self.fail("live scratch entered dummy replay")
        owner, batch, session = idle_owner()
        with IDLE["eagle3_dp_dummy_scope"](owner, batch, dummy_run=True, is_profile=True):
            self.assertFalse(getattr(owner, "_eagle3_graph_dp_dummy_active", False))


if __name__ == "__main__":
    unittest.main()
