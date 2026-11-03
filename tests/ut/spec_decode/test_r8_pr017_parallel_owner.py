# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU protocol checks for the fixed 544 parallel-drafter ownership seam.

The real lifetime pool is used. Only the device completion token is a stand-in;
this does not claim device completion, model correctness or Graph qualification.
"""

from __future__ import annotations

import unittest
from contextlib import contextmanager, nullcontext
from types import MethodType, SimpleNamespace
from typing import Any
from unittest.mock import PropertyMock, patch

import numpy as np

from vllm_ascend.worker.v2.spec_decode import draft_window_runtime
from vllm_ascend.worker.v2.spec_decode.dflash import aclgraph as dflash_graph
from vllm_ascend.worker.v2.spec_decode.dflash import speculator as dflash
from vllm_ascend.worker.v2.spec_decode.draft_graph_review_limits import DraftGraphSupport
from vllm_ascend.worker.v2.spec_decode.draft_window_eager import DraftWindowGraphInputMixin
from vllm_ascend.worker.v2.spec_decode.draft_window_graph import DraftWindowGraphRoute
from vllm_ascend.worker.v2.spec_decode.draft_window_lifetime import DraftWindowOwnerPool
from vllm_ascend.worker.v2.spec_decode.dspark import speculator as dspark


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
        completion = patch.object(draft_window_runtime, "_NPUCompletion", Completion)
        completion.start()
        self.addCleanup(completion.stop)
        self.batch = SimpleNamespace(is_dummy=False)
        self.owner = SimpleNamespace(
            _draft_window_graph_binding_provider=SimpleNamespace(quarantined=False),
            _draft_window_proposal=None,
            _draft_window_bypass_active=False,
            vllm_config=SimpleNamespace(max_concurrent_batches=2),
            num_speculative_steps=8,
        )
        provider = self.owner._draft_window_graph_binding_provider
        provider.quarantine = lambda: setattr(provider, "quarantined", True)
        self.normal: list[tuple[Any, dict[str, Any]]] = []

        @contextmanager
        def normal_scope(batch, **kwargs):
            self.normal.append((batch, kwargs))
            yield

        self.owner._draft_window_proposal_scope = normal_scope

    def scope(self, tokens, **kwargs):
        return dflash._draft_window_owner_scope(self.owner, self.batch, num_speculative_tokens=tokens, **kwargs)

    def test_zero_uses_real_completion_pool_without_graph_reader(self):
        with self.scope(0):
            self.assertTrue(self.owner._draft_window_context_kv_active)
        self.assertFalse(self.owner._draft_window_context_kv_active)
        self.assertEqual(Completion.recorded, 1)
        self.assertEqual(self.normal, [])
        self.assertIsInstance(self.owner._draft_window_context_kv_pool, DraftWindowOwnerPool)
        self.assertFalse(self.owner._draft_window_context_kv_pool._poisoned)
        self.assertFalse(hasattr(self.owner, "_draft_window_runtime"))
        self.assertIsNone(self.owner._draft_window_proposal)

    def test_zero_original_error_quarantines_and_actual_normal_scope_rejects(self):
        first = RuntimeError("native context write stand-in")
        with (
            self.assertRaises(RuntimeError) as seen,
            self.scope(0),
        ):
            raise first
        self.assertIs(seen.exception, first)
        self.assertTrue(self.owner._draft_window_context_kv_pool._poisoned)
        self.assertTrue(self.owner._draft_window_graph_binding_provider.quarantined)
        self.owner._draft_window_proposal_scope = MethodType(
            DraftWindowGraphInputMixin._draft_window_proposal_scope, self.owner
        )
        with (
            self.assertRaisesRegex(RuntimeError, "quarantined"),
            self.scope(8),
        ):
            self.fail("normal Graph body must not execute")

    def test_quarantined_refuses_zero_before_native(self):
        self.owner._draft_window_graph_binding_provider.quarantine()
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

        broken = patch.object(draft_window_runtime, "_NPUCompletion", BrokenCompletion)
        broken.start()
        self.addCleanup(broken.stop)
        with (
            self.assertRaises(RuntimeError) as seen,
            self.scope(0),
        ):
            pass
        self.assertIs(seen.exception, first)
        self.assertTrue(self.owner._draft_window_graph_binding_provider.quarantined)
        self.assertTrue(self.owner._draft_window_context_kv_pool._poisoned)

    def test_zero_and_normal_proposals_cannot_overlap(self):
        with self.scope(0):
            with self.assertRaisesRegex(RuntimeError, "already active"):
                self.scope(8)
            with self.assertRaisesRegex(RuntimeError, "already active"):
                self.scope(0)
        self.owner._draft_window_proposal = object()
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
        self.owner._draft_graph_review_limit = DraftGraphSupport(40960, 7, 1, 1, "dflash", 8)
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

    def test_544_propose_guards_before_owner_change_and_forwards_real_count(self):
        for module, cls, parent in (
            (dflash, dflash.AscendDFlashSpeculator, dflash.DFlashSpeculator),
            (dspark, dspark.AscendDSparkSpeculator, dspark.DSparkSpeculator),
        ):
            with (
                self.subTest(method=cls.__name__),
                patch.object(
                    cls,
                    "attn_vllm_config",
                    new_callable=PropertyMock,
                    return_value=SimpleNamespace(parallel_config=object()),
                ),
            ):
                owner = object.__new__(cls)
                original_batch = owner.input_batch = object()
                owner.replicated_pcp = owner.enable_adaptive_verification = owner.use_dcp = False
                owner.input_buffers = SimpleNamespace(positions=object())
                owner.max_num_tokens = 1024
                batch = SimpleNamespace(is_prefilling_np=np.array([False]))
                rejected = RuntimeError("guard before replacing owner")
                with (
                    patch.object(module, "_draft_window_owner_scope", side_effect=rejected),
                    self.assertRaises(RuntimeError) as seen,
                ):
                    owner.propose(batch, *([None] * 10), num_speculative_tokens=7)
                self.assertIs(seen.exception, rejected)
                self.assertIs(owner.input_batch, original_batch)
                result = object()
                with (
                    patch.object(module, "_draft_window_owner_scope", return_value=nullcontext()),
                    patch.object(module, "build_attn_metadata_wrapper", return_value=nullcontext()),
                    patch.object(module, "build_attn_metadata_factory", return_value=nullcontext()),
                    patch.object(parent, "propose", return_value=result) as propose,
                ):
                    self.assertIs(owner.propose(batch, *([None] * 10), num_speculative_tokens=7), result)
                self.assertIs(owner.input_batch, batch)
                self.assertEqual(propose.call_args.kwargs["num_speculative_tokens"], 7)

    def test_capture_forwards_544_callback_and_window_scope(self):
        manager = object.__new__(dflash_graph.DFlashAclGraphManager)
        manager.speculator = object()
        manager._capture_descs = {}
        callback = object()
        with (
            patch.object(dflash_graph, "draft_window_dflash_capture_scope", return_value=nullcontext()) as scope,
            patch.object(dflash_graph, "communicator_switch", return_value=nullcontext()),
            patch.object(dflash_graph, "model_capture_wrapper", return_value=nullcontext()),
            patch.object(dflash_graph.DFlashCudaGraphManager, "capture") as capture,
        ):
            manager.capture(object(), object(), object(), [], object(), 4096, prepare_forward=callback)
        scope.assert_called_once()
        self.assertIs(capture.call_args.kwargs["prepare_forward"], callback)
        manager.speculator = SimpleNamespace(attn_backends={"draft": dflash_graph.AscendFlashAttentionBackend})
        with (
            patch.object(
                dflash_graph, "resolve_parallel_draft_window_graph_route", return_value=DraftWindowGraphRoute.DFLASH
            ),
            patch.object(dflash_graph, "use_updatable_graph", return_value=False),
            self.assertRaisesRegex(NotImplementedError, "updatable backend"),
        ):
            manager.run_fullgraph(SimpleNamespace(num_tokens=8))


if __name__ == "__main__":
    unittest.main()
