# SPDX-License-Identifier: Apache-2.0
"""Named TP4xDP2 local1 admission/consumer contracts, no framework or device."""

import ast
import copy
import hashlib
import runpy
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
BASE = runpy.run_path(str(Path(__file__).with_name("test_r8_eagle_normal_review.py")))
CAPTURE = runpy.run_path(str(Path(__file__).with_name("test_eagle_ar_capture_config.py")))
CONFIG = runpy.run_path(str(Path(__file__).with_name("test_r9_full8_single_review_entry.py")))
M = BASE["M"]
IDLE = runpy.run_path(str(ROOT / "vllm_ascend/worker/v2/spec_decode/draft_window_eagle3_dp.py"))


def config():
    value = BASE["config"]()
    value.speculative_config.draft_model_config = CAPTURE["config"]().speculative_config.draft_model_config
    value.parallel_config.data_parallel_size = 2
    value.speculative_config.draft_parallel_config.data_parallel_size = 1
    value.scheduler_config.max_num_seqs = 1
    value.compilation_config.cudagraph_mode = CAPTURE["Mode"].FULL_DECODE_ONLY
    value.compilation_config.cudagraph_capture_sizes = [4, 8, 16]
    value.compilation_config.max_cudagraph_capture_size = 16
    value.additional_config["draft_window_graph_eagle_dp2_review"] = True
    value.additional_config.pop("draft_window_graph_eagle_single_request_review", None)
    return value


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
        vllm_config=config(),
        update_stream=object(),
        _eagle3_graph_session=session,
        _eagle3_window_bypass_active=True,
        _eagle3_window_continuing_prefill=False,
        _draft_window_graph_binding_provider=provider,
        _draft_window_runtime_state=NS(_active_inputs=None),
    )
    return owner, batch, session


class EagleDP2Controls(unittest.TestCase):
    def test_scope_keeps_local1_and_independent_draft_dp1(self):
        for window in (None, 1024):
            value = config()
            value.additional_config["draft_window_size"] = window
            scope = M.validate_eagle_dp2_review_config(value, experimental_graph_review=True)
            self.assertEqual((scope.max_requests, scope.data_parallel_size), (1, 2))
        for component, field, bad in (
            ("parallel_config", "data_parallel_size", 1),
            ("parallel_config", "data_parallel_size", 3),
            ("parallel_config", "tensor_parallel_size", 2),
            ("scheduler_config", "max_num_seqs", 2),
            ("scheduler_config", "max_num_seqs", True),
            ("speculative_config", "num_speculative_tokens", 4),
            ("compilation_config", "cudagraph_capture_sizes", [8, 16]),
        ):
            value = config()
            setattr(getattr(value, component), field, bad)
            with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                M.validate_eagle_dp2_review_config(value, experimental_graph_review=True)
        value = config()
        value.speculative_config.draft_parallel_config.data_parallel_size = 2
        with self.assertRaises(ValueError):
            M.validate_eagle_dp2_review_config(value, experimental_graph_review=True)
        for flag in ("draft_window_graph_eagle_single_request_review", "draft_window_graph_eagle_two_request_review"):
            value = config()
            value.additional_config[flag] = True
            with self.assertRaises(ValueError):
                M.validate_eagle_dp2_review_config(value, experimental_graph_review=True)
        owner, batch = BASE["objects"]()
        owner._draft_graph_review_limit = M.EagleSingleRequestReviewScope(4096, data_parallel_size=2)
        M.validate_draft_graph_review_requests(owner, batch)
        batch.num_reqs = batch.num_reqs_after_padding = 2
        with self.assertRaises(NotImplementedError):
            M.validate_draft_graph_review_requests(owner, batch)

    def test_checkpoint_mutex_and_existing_numeric_meaning(self):
        from pydantic import TypeAdapter

        module = types.ModuleType("vllm_ascend.ascend_config")
        module.validate_additional_config_bool = CONFIG["actual_function"](
            ROOT / "vllm_ascend/ascend_config.py", "validate_additional_config_bool", {"TypeAdapter": TypeAdapter}
        )
        with tempfile.TemporaryDirectory() as directory:
            data = b'{"independent_dp2_guard_fixture":true}\n'
            file = Path(directory) / "config.json"
            file.write_bytes(data)
            value = config()
            value.speculative_config.draft_model_config = NS(model=directory)
            with (
                patch.dict(sys.modules, {"vllm_ascend.ascend_config": module}),
                patch.object(M, "_EAGLE_CONFIG", hashlib.sha256(data).hexdigest()),
            ):
                scope = M.resolve_draft_graph_review_limit(value, experimental_graph_review=True)
                self.assertEqual((scope.max_requests, scope.data_parallel_size), (1, 2))
                value.additional_config["draft_window_graph_numeric_diagnostic"] = True
                self.assertIsNone(M.resolve_draft_graph_review_limit(value, experimental_graph_review=True))
                value.additional_config["draft_window_graph_eagle_two_request_review"] = True
                with self.assertRaises(ValueError):
                    M.resolve_draft_graph_review_limit(value, experimental_graph_review=True)
                del value.additional_config["draft_window_graph_eagle_two_request_review"]
                file.write_text("{}")
                with self.assertRaises(NotImplementedError):
                    M.resolve_draft_graph_review_limit(value, experimental_graph_review=True)

    def test_actual_ar_helper_preserves_target_and_first_tp_dp(self):
        value = config()
        controls = CAPTURE["EagleARCaptureTests"]()
        controls.setUp()
        # The actual startup allowance, not a fake DP1 model or computed Graph output.
        controls.modules["vllm_ascend.worker.v2.spec_decode.draft_graph_review_limits"] = M
        controls.modules["vllm_ascend.ascend_config"].validate_additional_config_bool = (
            lambda x, _: x if type(x) is bool else (_ for _ in ()).throw(TypeError())
        )
        before = copy.deepcopy(value.compilation_config)
        actual = controls.call(value)
        self.assertEqual(actual.compilation_config.cudagraph_capture_sizes, [1])
        self.assertEqual(value.compilation_config.cudagraph_capture_sizes, before.cudagraph_capture_sizes)
        self.assertIs(actual.parallel_config, value.parallel_config)
        self.assertEqual(actual.parallel_config.data_parallel_size, 2)
        self.assertTrue(actual.parallel_config.use_sequence_parallel_moe)
        self.assertIs(controls.call(value, query=4), value)
        value.parallel_config.use_sequence_parallel_moe = False
        value.parallel_config.all2all_backend = "flashinfer_all2allv"
        self.assertEqual(controls.call(value).compilation_config.cudagraph_capture_sizes, [1])
        self.assertFalse(value.parallel_config.use_sequence_parallel_moe)
        value.additional_config["draft_window_graph_eagle_dp2_review"] = False
        self.assertIs(controls.call(value), value)

    def test_actual_fingerprint_only_allows_local1_eagle_consumers(self):
        source = ROOT / "vllm_ascend/worker/v2/spec_decode/draft_window_graph.py"
        tree = ast.parse(source.read_text())
        nodes = [
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name in ("DraftWindowGraphRoute", "DraftWindowGraphRouteFingerprint")
        ]
        assignment = next(n for n in tree.body if isinstance(n, ast.AnnAssign) and n.target.id == "_ROUTE_CONTRACTS")
        module = types.ModuleType("eagle_dp2_fingerprint_host_contract")
        module.__dict__.update(dataclass=dataclass, Enum=Enum, DraftWindowExecutionPhase=NS(PROPOSAL=0, STEP=1))
        with patch.dict(sys.modules, {module.__name__: module}):
            exec(compile(ast.Module(body=nodes + [assignment], type_ignores=[]), str(source), "exec"), module.__dict__)
            route = module.DraftWindowGraphRoute
            for first, query, selected in ((True, 4, route.EAGLE3_FIRST_PASS), (False, 1, route.EAGLE3_AR_STEP)):
                fields = dict(
                    method="eagle3",
                    route=selected,
                    graph_mode="FULL",
                    backend_impl="actual.FIA",
                    builder_impl="actual.Builder",
                    global_group_id=0,
                    layout="dense_gqa_direct_prefix",
                    capture_descriptor="eagle3:local1",
                    capture_bucket=(1, query),
                    query_width=query,
                    padded_request_count=1,
                    padded_token_count=query,
                    static_k=3,
                    window_size=1024,
                    logical_block_size=128,
                    kernel_block_size=128,
                    block_span_tokens=128,
                    tensor_parallel_size=4,
                    data_parallel_size=2,
                    pipeline_parallel_size=1,
                    prefill_context_parallel_size=1,
                    decode_context_parallel_size=1,
                    sequence_parallel=True,
                    dsa_context_parallel=False,
                    flashcomm=True,
                    async_scheduling=True,
                    prefix_caching=True,
                )
                for tp_rank in range(4):
                    actual = module.DraftWindowGraphRouteFingerprint(**fields, tensor_parallel_rank=tp_rank)
                    self.assertEqual(actual.data_parallel_size, 2)
                for field, bad in (
                    ("data_parallel_size", 3),
                    ("padded_request_count", 2),
                    ("tensor_parallel_size", 2),
                    ("dsa_context_parallel", True),
                ):
                    invalid = dict(fields, **{field: bad})
                    with self.subTest(first=first, field=field), self.assertRaises(ValueError):
                        module.DraftWindowGraphRouteFingerprint(**invalid)

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
