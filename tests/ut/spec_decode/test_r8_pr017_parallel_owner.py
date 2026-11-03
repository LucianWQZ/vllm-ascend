# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Actual provider and fixed544 K0 dispatch boundaries, without device execution.

Native preparation/KV computation are isolated only at their device boundary.
These Host tests do not claim numerical correctness, replay or device completion.
"""

from __future__ import annotations

import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, PropertyMock, patch

import numpy as np
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers

from vllm_ascend.worker.v2.spec_decode.dflash import aclgraph as dflash_graph
from vllm_ascend.worker.v2.spec_decode.dflash import speculator as dflash
from vllm_ascend.worker.v2.spec_decode.draft_graph_review_limits import DraftGraphSupport
from vllm_ascend.worker.v2.spec_decode.draft_window_backends import DraftWindowGroupRoute, DraftWindowProviderStatus
from vllm_ascend.worker.v2.spec_decode.draft_window_graph import DraftWindowGraphBindingProvider
from vllm_ascend.worker.v2.spec_decode.draft_window_policy import resolve_draft_window_policy
from vllm_ascend.worker.v2.spec_decode.dspark import speculator as dspark


class ParallelOwnerTests(unittest.TestCase):
    def setUp(self):
        event = patch.object(torch.npu, "Event", side_effect=AssertionError("K0 must not add an Event"))
        self.events = event.start()
        self.addCleanup(event.stop)
        self.stream = SimpleNamespace(npu_stream=17)
        stream = patch.object(torch.npu, "current_stream", return_value=self.stream)
        stream.start()
        self.addCleanup(stream.stop)
        self.batch = SimpleNamespace(is_dummy=False)
        tables = SimpleNamespace(
            input_block_tables=[torch.arange(8, dtype=torch.int32).reshape(2, 4)],
            num_blocks=SimpleNamespace(cpu=torch.full((1, 2), 4, dtype=torch.int32)),
        )
        self.provider = DraftWindowGraphBindingProvider(
            policy=resolve_draft_window_policy(requested_window_size=4, method="dflash"),
            group=DraftWindowGroupRoute(0, ("draft.layer",), 4, DraftWindowProviderStatus.DIRECT_DENSE),
            block_tables=tables,
        )
        self.owner = SimpleNamespace(
            _draft_window_graph_binding_provider=self.provider,
            _draft_window_bypass_active=False,
            vllm_config=SimpleNamespace(),
            num_speculative_steps=8,
        )
        self.normal: list[tuple[Any, dict[str, Any]]] = []

        @contextmanager
        def normal_scope(batch, **kwargs):
            self.normal.append((batch, kwargs))
            yield

        self.owner._draft_window_proposal_scope = normal_scope

    def scope(self, tokens, **kwargs):
        return dflash._draft_window_owner_scope(self.owner, self.batch, num_speculative_tokens=tokens, **kwargs)

    def test_actual_fixed_core_zero_writes_then_normal_binding(self):
        owner = object.__new__(dflash.AscendDFlashSpeculator)
        owner._draft_window_graph_binding_provider = self.provider
        owner._draft_window_bypass_active = False
        owner.num_speculative_steps, owner.num_query_per_req = 7, 8
        owner.max_model_len, owner.max_num_reqs, owner.max_num_tokens = 64, 2, 16
        owner.pcp_manager, owner._layer_group_idx = None, None
        owner.draft_kv_cache_group_id, owner.draft_kv_cache_group_ids = 0, [0]
        owner.dcp_size, owner.parallel_drafting_token_id, owner.sample_from_anchor = 1, 42, False
        owner.input_buffers = InputBuffers(2, 16, torch.device("cpu"))
        owner.context_positions = torch.zeros(16, dtype=torch.int64)
        owner._context_slot_mappings = torch.zeros((1, 16), dtype=torch.int32)
        owner.hidden_states = torch.zeros((16, 4))
        owner.draft_tokens = torch.full((2, 7), -1, dtype=torch.int64)
        owner.sample_indices = owner.sample_pos = owner.sample_idx_mapping = torch.zeros(16, dtype=torch.int64)
        owner.temperature, owner.seeds = torch.ones(2), torch.zeros(2, dtype=torch.int64)
        owner.block_tables = self.provider.block_tables
        owner.block_tables.slot_mappings = torch.full((1, 16), -1, dtype=torch.int32)
        owner.block_tables.block_sizes = [4]
        owner.block_tables.cp_rank, owner.block_tables.cp_interleave = 0, 1
        batch = InputBatch.make_dummy(1, 4, owner.input_buffers, is_padding=False)
        batch.positions.copy_(torch.arange(4))
        last_hidden = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        target = {"positions": batch.positions.clone(), "hidden": last_hidden.clone()}
        full_table = self.provider.table.clone()
        full_slots = owner.block_tables.slot_mappings.clone()
        descriptor = BatchExecutionDescriptor(CUDAGraphMode.FULL, num_tokens=8, num_reqs=1, uniform_token_count=8)
        bank = self.provider.register_capture_identity(descriptor=descriptor)
        bank_tensors = (bank.window_block_table, bank.device_window_exact, bank.cpu_window_exact)
        empty_bank = [value.clone() for value in bank_tensors]
        addresses = tuple(value.data_ptr() for value in (owner.hidden_states, owner.context_positions, *bank_tensors))
        canonical_kv = torch.zeros((16, 4))
        writes = []

        def prepare(*args):
            # Stand in only for the native preparation kernel. The actual
            # Ascend/core propose, persistent copy and context dispatch execute.
            self.assertTrue(owner._draft_window_context_kv_active)
            self.assertIs(args[9], batch)
            args[2][:4].copy_(batch.positions)
            args[3][:4].copy_(torch.arange(4, dtype=torch.int32))

        def store(hidden, positions, slots):
            self.assertEqual(hidden.data_ptr(), owner.hidden_states.data_ptr())
            self.assertEqual(positions.data_ptr(), owner.context_positions.data_ptr())
            self.assertEqual(slots.data_ptr(), owner._context_slot_mappings.data_ptr())
            canonical_kv.index_copy_(0, slots.to(torch.int64), hidden)
            writes.append(hidden.clone())

        owner.model = SimpleNamespace(precompute_and_store_context_kv=Mock(side_effect=store))
        args = (
            batch,
            {},
            {},
            last_hidden,
            None,
            *[torch.zeros(2, dtype=torch.int64) for _ in range(4)],
            torch.ones(2),
            torch.zeros(2, dtype=torch.int64),
        )
        with (
            patch.object(dflash.dflash_speculator, "prepare_dflash_inputs", side_effect=prepare) as preparation,
            patch.object(owner, "_build_uniform_attn_metadata", side_effect=AssertionError("K0 has no query metadata")),
            patch.object(owner, "_generate_draft", side_effect=AssertionError("K0 has no query forward")),
        ):
            for _ in range(2):
                result = owner.propose(*args, num_speculative_tokens=0)
                self.assertEqual(result.data_ptr(), owner.draft_tokens.data_ptr())
                self.assertIs(owner.input_batch, batch)
                self.assertFalse(owner._draft_window_context_kv_active)
            self.assertEqual(preparation.call_count, 2)
        self.assertEqual(len(writes), 2)
        self.assertTrue(torch.equal(canonical_kv[:4], last_hidden))
        self.assertTrue(torch.equal(batch.positions, target["positions"]))
        self.assertTrue(torch.equal(last_hidden, target["hidden"]))
        self.assertTrue(torch.equal(self.provider.table, full_table))
        self.assertTrue(torch.equal(owner.block_tables.slot_mappings, full_slots))
        for actual, before in zip(bank_tensors, empty_bank, strict=True):
            self.assertTrue(torch.equal(actual, before))
        self.assertEqual(
            addresses,
            tuple(value.data_ptr() for value in (owner.hidden_states, owner.context_positions, *bank_tensors)),
        )
        self.assertIsNone(self.provider.proposal)
        self.assertEqual(self.provider.get(None), ())
        self.assertFalse(hasattr(owner, "_draft_window_context_kv_pool"))
        self.assertFalse(hasattr(owner, "_draft_window_runtime"))
        self.events.assert_not_called()

        # Resume the actual next normal bank preparation and parameter binding.
        # Host state retirement is checked, not claimed to be device completion.
        with self.provider.proposal_scope((False,)):
            self.provider.prepare(
                bank,
                cpu_full_exact=torch.tensor([8], dtype=torch.int32),
                idx_mapping_np=np.array([0], dtype=np.intp),
                step=0,
            )
            self.provider.bind_current(descriptor)
            (params,) = self.provider.get(None)
            self.assertIs(params["block_table"], bank.window_block_table)
            self.assertEqual(params["actual_seq_lengths_kv"], [4])
            self.assertEqual(int(bank.window_block_table[0, 0]), 1)
            self.provider.finish_update()
        self.assertFalse(self.provider.quarantined)
        self.assertEqual(self.provider.get(None), ())
        self.assertTrue(torch.equal(self.provider.table, full_table))

        failure = RuntimeError("native context KV write failed")
        owner.model.precompute_and_store_context_kv.side_effect = failure
        with (
            patch.object(dflash.dflash_speculator, "prepare_dflash_inputs", side_effect=prepare),
            self.assertRaises(RuntimeError) as seen,
        ):
            owner.propose(*args, num_speculative_tokens=0)
        self.assertIs(seen.exception, failure)
        self.assertTrue(self.provider.quarantined)
        self.assertFalse(owner._draft_window_context_kv_active)
        with self.assertRaisesRegex(RuntimeError, "quarantined"):
            owner.propose(*args, num_speculative_tokens=7)
        self.events.assert_not_called()

    def test_quarantined_refuses_zero_before_native(self):
        self.owner._draft_window_graph_binding_provider.quarantine()
        with (
            self.assertRaisesRegex(RuntimeError, "quarantined"),
            self.scope(0),
        ):
            self.fail("context write must not execute")
        self.assertFalse(hasattr(self.owner, "_draft_window_context_kv_pool"))
        self.events.assert_not_called()

    def test_zero_changed_stream_is_rejected_before_native_write(self):
        # Only the device header/current-stream boundary is stood in for here;
        # the real provider enforces its established stream before any write.
        header = SimpleNamespace(device=SimpleNamespace(type="npu"))
        with patch.object(type(self.provider), "table", new_callable=PropertyMock, return_value=header):
            with self.scope(0):
                pass
            self.stream.npu_stream = 18
            with self.assertRaisesRegex(RuntimeError, "changed stream"), self.scope(0):
                self.fail("changed-stream context write must not execute")
        self.assertFalse(self.owner._draft_window_context_kv_active)
        self.assertFalse(self.provider.quarantined)
        self.events.assert_not_called()

    def test_zero_and_normal_proposals_cannot_overlap(self):
        with self.scope(0):
            with self.assertRaisesRegex(RuntimeError, "already active"):
                self.scope(8)
            with self.assertRaisesRegex(RuntimeError, "already active"):
                self.scope(0)
        with (
            self.provider.proposal_scope((False,)),
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
        manager.speculator = SimpleNamespace(
            attn_backends={"draft": dflash_graph.AscendFlashAttentionBackend},
            _draft_window_graph_binding_provider=object(),
        )
        with (
            patch.object(dflash_graph, "use_updatable_graph", return_value=False),
            self.assertRaisesRegex(NotImplementedError, "updatable backend"),
        ):
            manager.run_fullgraph(SimpleNamespace(num_tokens=8))

    def test_parallel_replay_retains_the_off_and_idle_dp_baseline_path(self):
        for idle_dp in (False, True):
            with self.subTest(idle_dp=idle_dp):
                manager = object.__new__(dflash_graph.DFlashAclGraphManager)
                backend, metadata, result = object(), object(), object()
                owner = manager.speculator = SimpleNamespace(
                    attn_backends={"draft": backend},
                    input_batch=SimpleNamespace(seq_lens_cpu_upper_bound=object()),
                    build_draft_attn_metadatas=Mock(return_value=metadata),
                )
                if idle_dp:
                    owner._draft_window_dp_dummy_active = True
                    owner._draft_window_graph_binding_provider = object()
                desc = SimpleNamespace(num_tokens=8, num_reqs=1)
                with (
                    patch.object(manager, "_graph_replay", return_value=result) as replay,
                    patch.object(dflash_graph, "use_updatable_graph", return_value=False),
                ):
                    self.assertIs(manager.run_fullgraph(desc), result)
                owner.build_draft_attn_metadatas.assert_called_once_with(1, owner.input_batch.seq_lens_cpu_upper_bound)
                replay.assert_called_once_with(desc, backend, 8, metadata)


if __name__ == "__main__":
    unittest.main()
