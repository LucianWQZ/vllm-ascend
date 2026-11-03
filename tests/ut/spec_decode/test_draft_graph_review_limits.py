# SPDX-License-Identifier: Apache-2.0
"""Shape and ownership guards, independent of the old experiment entry names."""

import copy
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import torch

from vllm_ascend.worker.v2.model_runner import NPUModelRunner
from vllm_ascend.worker.v2.spec_decode import draft_graph_review_limits as GUARD
from vllm_ascend.worker.v2.spec_decode import draft_window_eagle3_dp as DP
from vllm_ascend.worker.v2.spec_decode.dflash import speculator as dflash_module
from vllm_ascend.worker.v2.spec_decode.target_dependent_ar import aclgraph as ar_graph
from vllm_ascend.worker.v2.spec_decode.target_dependent_ar import speculator as ar_module


def runner_bounds(owner, batch):
    """Exercise the canonical runner boundary before using its private batch."""
    count = max(batch.idx_mapping_np, default=-1) + 1
    prompt, maximum = [0] * count, [0] * count
    for row, (start, end) in zip(batch.idx_mapping_np, batch.draft_graph_request_bounds, strict=True):
        prompt[row], maximum[row] = start, end
    state = NS(
        req_id_to_index=dict(zip(batch.req_ids, batch.idx_mapping_np, strict=True)),
        prompt_len=NS(np=prompt),
        prefill_len=NS(np=prompt),
        max_seq_len=maximum,
    )
    return GUARD.draft_graph_request_bounds(owner, state, batch.req_ids, batch.idx_mapping_np)


class DraftGraphSupportTest(unittest.TestCase):
    def config(self, *, method="eagle3", rows=1, dp=1, window=True):
        profile = "eagle3" if method == "eagle3" else "dflash_full8"
        model = copy.deepcopy(GUARD._CONSUMER_CONFIGS[profile])
        model["max_position_embeddings"] = 262144
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
        draft = NS(model="qualified-draft", hf_config=hf, hf_text_config=hf, dtype="torch.bfloat16")
        return NS(
            speculative_config=NS(
                method=method,
                enforce_eager=False,
                num_speculative_tokens=k,
                uses_dynamic_speculative_decoding=lambda: False,
                enable_adaptive_verification=False,
                draft_model_config=draft,
                target_model_config=NS(hf_text_config=NS(model_type="qwen3_moe")),
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
            model_config=NS(max_model_len=4096 if method == "eagle3" else 40960, hf_config=hf, hf_text_config=hf),
            additional_config={"draft_window_size": (1024 if method == "eagle3" else 32768) if window else None},
        )

    def resolve(self, config):
        return GUARD.resolve_draft_graph_review_limit(config, experimental_graph_review=True)

    def qualified_owner(self, config=None, **overrides):
        config = self.config() if config is None else config
        self.assertIs(self.resolve(config), True)
        speculative = config.speculative_config
        fields = dict(
            _draft_graph_review_limit=True,
            method=speculative.method,
            num_speculative_steps=speculative.num_speculative_tokens,
            max_num_reqs=config.scheduler_config.max_num_seqs,
            max_model_len=config.model_config.max_model_len,
            num_query_per_req=speculative.num_speculative_tokens + 1,
            vllm_config=config,
        )
        fields.update(overrides)
        return NS(**fields)

    def test_capacities_without_experiment_flags(self):
        for method in ("eagle3", "dflash"):
            for dp, rows in ((1, 1), (1, 2), (2, 1)):
                for window in (False, True):
                    with (
                        self.subTest(method=method, dp=dp, rows=rows, window=window),
                        patch.object(Path, "read_text", side_effect=AssertionError("startup must not reload config")),
                    ):
                        self.assertIs(self.resolve(self.config(method=method, dp=dp, rows=rows, window=window)), True)

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

    def test_startup_parser_does_not_add_fields_to_upstream_config(self):
        for method in ("eagle3", "dflash"):
            config = self.config(method=method, rows=2)
            fields = vars(config).copy()
            scope = self.resolve(config)
            self.assertEqual(vars(config), fields)
            self.assertEqual(self.resolve(config), scope)
            self.assertEqual(vars(config), fields)

    def test_unscoped_dflash_loading_and_dflash2_super_config_stay_unmodified(self):
        config = self.config(method="dflash")
        # Unscoped DFlash startup must not pollute copies used by the shared
        # DFlash/DFlash2 load_dflash_model path, even when support is None.
        for model in (
            {"architectures": ["GlmDFlashDraftModel"], "model_type": "glm_moe_dsa"},
            {"architectures": ["DFlash2DraftModel"], "model_type": "qwen3"},
        ):
            config.speculative_config.draft_model_config.hf_config = NS(**model)
            fields = vars(config).copy()
            # The real DFlash2 service selects method=dflash too.
            self.assertIsNone(self.resolve(config))
            self.assertEqual(vars(config), fields)
            # The upstream DFlash constructor makes a shallow VllmConfig copy.
            parent_config = copy.copy(config)
            self.assertEqual(vars(parent_config), fields)
            self.assertNotIn("_ascend_draft_graph_support", vars(parent_config))

    def test_config_packaging_changes_do_not_change_support(self):
        config = self.config()
        hf = config.speculative_config.draft_model_config.hf_config
        hf.transformers_version = "a packaging version"
        hf.initializer_range = 0.3
        hf.unused_export_note = "not read by the fixed model"
        with patch.object(Path, "read_text", side_effect=AssertionError("unexpected config IO")):
            self.assertIs(self.resolve(config), True)

    def test_model_loader_settings_do_not_redefine_window_geometry(self):
        for method in ("eagle3", "dflash"):
            config = self.config(method=method)
            hf = config.speculative_config.draft_model_config.hf_config
            expected = self.resolve(config)
            # These are parsed/executed by the model implementation. They do
            # not change row count, KV layout, window alignment or Graph bank.
            for field, value in (
                ("hidden_act", "gelu"),
                ("intermediate_size", 8192),
                ("rms_norm_eps", 1e-5),
                ("tie_word_embeddings", True),
                ("vocab_size", 256000),
                ("block_size", 32),
                ("num_target_layers", 2),
            ):
                setattr(hf, field, value)
                with self.subTest(method=method, field=field):
                    self.assertEqual(self.resolve(config), expected)

    def test_worker_wrapped_config_keeps_actual_consumer_geometry(self):
        from transformers import AutoConfig
        from vllm.transformers_utils.configs.eagle import EAGLEConfig

        for method, profile in (("dflash", "dflash_full8"), ("eagle3", "eagle3")):
            config = self.config(method=method)
            fields = copy.deepcopy(GUARD._CONSUMER_CONFIGS[profile])
            fields["max_position_embeddings"] = 262144
            inner = AutoConfig.for_model(**fields)
            wrapped = EAGLEConfig(inner, method=method, model_type="eagle")
            config.speculative_config.draft_model_config.hf_config = wrapped
            self.assertEqual(wrapped.model_type, "eagle")
            self.assertIs(self.resolve(config), True)
            # The model consumes outer effective fields. Unwrapping the whole
            # config would silently ignore this changed projection geometry.
            wrapped.hidden_size += 1
            with self.assertRaisesRegex(NotImplementedError, "hidden_size"):
                self.resolve(config)

    def test_glm_dflash_uses_target_feature_contract_not_qwen_full8_profile(self):
        from transformers import AutoConfig
        from vllm.transformers_utils.configs.eagle import EAGLEConfig

        config = self.config(method="dflash")
        speculative = config.speculative_config
        inner = AutoConfig.for_model(
            model_type="qwen3",
            architectures=["DFlashDraftModel"],
            hidden_size=6144,
            num_hidden_layers=5,
            layer_types=["full_attention"] * 5,
            target_hidden_size=6144,
            num_target_layers=78,
            dflash_config={"target_layer_ids": [1, 20, 38, 56, 75]},
        )
        hf = EAGLEConfig(inner, method="dflash", model_type="eagle")
        speculative.draft_model_config.hf_config = hf
        target_hf = NS(model_type="glm_moe_dsa", num_hidden_layers=78)
        speculative.target_model_config = NS(hf_text_config=target_hf, get_hidden_size=lambda: 6144)
        self.assertIsNone(self.resolve(config))
        # This export count is not consumed by the fixed DFlash model/worker;
        # the actual Target layer count below still bounds every auxiliary ID.
        hf.num_target_layers = 40
        self.assertIsNone(self.resolve(config))
        for field, bad in (
            ("target_hidden_size", 2048),
            ("layer_types", ["sliding_attention"] * 5),
            ("dflash_config", {"target_layer_ids": [1, 78]}),
            ("dflash_config", {"target_layer_ids": [1, 1]}),
        ):
            before = getattr(hf, field)
            setattr(hf, field, bad)
            with self.subTest(field=field, bad=bad), self.assertRaisesRegex(NotImplementedError, "GLM DFlash"):
                self.resolve(config)
            setattr(hf, field, before)
        # The same shared architecture does not open an unqualified Qwen draft.
        target_hf.model_type = "qwen3_moe"
        with self.assertRaisesRegex(NotImplementedError, "hidden_size"):
            self.resolve(config)

    def test_startup_reuses_resolved_qualification_without_reparsing(self):
        from vllm_ascend import envs
        from vllm_ascend.worker.v2.spec_decode import _use_graph_window_owner
        from vllm_ascend.worker.v2.spec_decode import draft_graph_review_limits as actual_guard

        config = self.config(rows=2)
        support = self.resolve(config)
        policy = NS(enabled=True)
        with (
            patch.object(envs, "VLLM_ASCEND_EXPERIMENTAL_DRAFT_WINDOW_GRAPH", True),
            patch.object(
                actual_guard, "resolve_draft_graph_review_limit", side_effect=AssertionError("already resolved")
            ),
        ):
            self.assertTrue(_use_graph_window_owner(config, policy, support))
            config.speculative_config.enforce_eager = True
            self.assertFalse(_use_graph_window_owner(config, policy, support))

    def test_factory_checks_static_graph_topology_before_source_creation(self):
        from vllm_ascend.ascend_config import validate_draft_window_graph_config
        from vllm_ascend.device import hardware_profile

        config = self.config()
        support = self.resolve(config)
        for field in ("pipeline_parallel_size", "prefill_context_parallel_size", "decode_context_parallel_size"):
            with self.subTest(field=field):
                setattr(config.parallel_config, field, 2)
                with self.assertRaisesRegex(RuntimeError, "PP=PCP=DCP=1"):
                    validate_draft_window_graph_config(config, support=support)
                setattr(config.parallel_config, field, 1)
        with patch.object(
            hardware_profile, "get_current_hardware_profile", return_value=NS(attention_backend_family=None)
        ), self.assertRaisesRegex(NotImplementedError, "STANDARD"):
            validate_draft_window_graph_config(config, support=support)

    def test_real_factory_passes_its_single_qualification_to_validation(self):
        from vllm_ascend import ascend_config, envs
        from vllm_ascend.worker.v2 import spec_decode as factory
        from vllm_ascend.worker.v2.spec_decode import draft_graph_review_limits as actual_guard

        config = self.config(rows=2)
        before = vars(config).copy()
        policy = NS(enabled=True, reason=None)
        owner = NS(_initialize_draft_window_source_capability=lambda: None)
        with (
            patch.object(envs, "VLLM_ASCEND_EXPERIMENTAL_DRAFT_WINDOW_GRAPH", True),
            patch.object(factory, "_resolve_window_policy", return_value=policy),
            patch.object(factory, "_init_speculator", return_value=owner),
            patch.object(factory, "logger", Mock()),
            patch.object(
                actual_guard, "resolve_draft_graph_review_limit", wraps=actual_guard.resolve_draft_graph_review_limit
            ) as resolve,
            patch.object(
                ascend_config,
                "validate_draft_window_graph_config",
                wraps=ascend_config.validate_draft_window_graph_config,
            ) as validate,
        ):
            self.assertIs(factory.init_speculator(config, None), owner)
            self.assertEqual(resolve.call_count, 1)
        self.assertIs(validate.call_args.kwargs["support"], owner._draft_graph_review_limit)
        self.assertIs(owner._draft_window_policy, policy)
        self.assertFalse(hasattr(owner, "draft_window_policy"))
        self.assertEqual(vars(config), before)

    def test_factory_rejects_an_existing_window_owner_before_graph_admission(self):
        from vllm_ascend import envs
        from vllm_ascend.worker.v2 import spec_decode as factory

        config = self.config()
        policy = NS(enabled=True, reason=None)
        for field in ("_draft_window_policy", "draft_window_policy", "_draft_window_runtime"):
            owner = NS(_initialize_draft_window_source_capability=lambda: None)
            setattr(owner, field, object())
            with (
                self.subTest(field=field),
                patch.object(envs, "VLLM_ASCEND_EXPERIMENTAL_DRAFT_WINDOW_GRAPH", True),
                patch.object(factory, "_resolve_window_policy", return_value=policy),
                patch.object(factory, "_init_speculator", return_value=owner),
                patch.object(factory, "logger", Mock()),
                self.assertRaisesRegex(RuntimeError, "mutually exclusive"),
            ):
                factory.init_speculator(config, None)

    def test_real_consumer_overrides_and_normalized_positions_stay_protected(self):
        for field, value in (("is_causal", False), ("target_hidden_size", 4096), ("use_aux_hidden_state", False)):
            config = self.config()
            setattr(config.speculative_config.draft_model_config.hf_config, field, value)
            with self.subTest(field=field), self.assertRaises(NotImplementedError):
                self.resolve(config)
        config = self.config()
        config.speculative_config.draft_model_config.hf_config.rope_parameters = {
            "rope_type": "default",
            "rope_theta": 1,
        }
        with self.assertRaisesRegex(NotImplementedError, "rope_parameters"):
            self.resolve(config)

    def test_changed_attention_aux_or_position_semantics_are_rejected(self):
        for field, value in (
            ("is_causal", True),
            ("num_key_value_heads", 4),
            ("eagle_config", {"eagle_aux_hidden_state_layer_ids": [2, 18, 34]}),
            ("rope_parameters", {"rope_type": "default", "rope_theta": 10000}),
        ):
            config = self.config()
            setattr(config.speculative_config.draft_model_config.hf_config, field, value)
            with self.subTest(field=field), self.assertRaises(NotImplementedError):
                self.resolve(config)

    def test_swa_counterexample_remains_blocked_with_window_off(self):
        config = self.config(method="dflash", window=False)
        hf = config.speculative_config.draft_model_config.hf_config
        hf.layer_types = ["sliding_attention"] * 5 + ["full_attention"]
        hf.sliding_window = 4096
        with self.assertRaisesRegex(NotImplementedError, "valid-output"):
            self.resolve(config)

    def test_diagnostic_flag_does_not_bypass_failed_model(self):
        config = self.config(method="dflash")
        hf = config.speculative_config.draft_model_config.hf_config
        hf.layer_types = ["sliding_attention"] * 5 + ["full_attention"]
        hf.sliding_window = 4096
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
            draft_graph_request_bounds=tuple((r[2], r[3]) for r in records),
            num_draft_tokens_per_req=None,
        )

    def test_actual_rows_and_request_identity_not_client_concurrency(self):
        owner = self.qualified_owner(self.config(rows=2))
        batch = self.batch(rows=2)
        GUARD.validate_draft_graph_review_requests(owner, batch)
        batch.num_reqs_after_padding = 4
        with self.assertRaises(NotImplementedError):
            GUARD.validate_draft_graph_review_requests(owner, batch)

    def test_preemption_state_is_kept_until_actual_finish(self):
        for method in ("eagle3", "dflash"):
            with self.subTest(method=method):
                owner = self.qualified_owner(self.config(method=method))
                runner = NPUModelRunner.__new__(NPUModelRunner)
                runner.speculator = owner
                batch = self.batch()

                def original_finish(output, *, current_owner=owner):
                    # Our actual runner hook records lifecycle before the
                    # original runner can update/reuse physical request rows.
                    self.assertEqual(
                        current_owner._draft_graph_review_preempted_request_ids,
                        output.preempted_req_ids - output.finished_req_ids,
                    )

                with patch.object(NPUModelRunner.__mro__[1], "finish_requests", side_effect=original_finish):
                    runner.finish_requests(NS(finished_req_ids=set(), preempted_req_ids={"request-0"}))
                    with self.assertRaises(NotImplementedError):
                        runner_bounds(owner, batch)
                    runner.finish_requests(NS(finished_req_ids={"request-0"}, preempted_req_ids=set()))
                self.assertEqual(runner_bounds(owner, batch), batch.draft_graph_request_bounds)
                GUARD.validate_draft_graph_review_requests(owner, batch)

    def test_lifecycle_ignores_routes_without_qualified_support(self):
        for owner in (
            NS(),
            NS(_draft_graph_review_limit=None),
        ):
            with self.subTest(owner=owner):
                # No scheduler fields may be read for an unrelated route.
                GUARD.note_eagle_single_request_review_lifecycle(owner, object())
                self.assertFalse(hasattr(owner, "_draft_graph_review_preempted_request_ids"))

    def test_bounds_use_actual_runner_mapping_and_reject_resumed_requests(self):
        import numpy as np

        owner = self.qualified_owner(self.config(rows=2))
        state = NS(
            req_id_to_index={"a": 2, "b": 0},
            prompt_len=NS(np=np.array([513, 0, 1537])),
            prefill_len=NS(np=np.array([513, 0, 1537])),
            max_seq_len=np.array([545, 0, 1569]),
        )
        state.num_computed_prefill_tokens = state.prefill_len.np.copy()
        runner: Any = object.__new__(NPUModelRunner)
        runner.req_states = state
        runner.decode_query_len, runner.use_pp, runner._uses_flashmla = 4, False, False
        runner.adaptive_verification, runner.pcp_manager = None, None
        scheduled = NS(
            num_scheduled_tokens={"b": 4, "a": 4},
            total_num_scheduled_tokens=8,
            scheduled_spec_decode_tokens={},
        )
        batch, _ = runner.gather_batch_req_state(scheduled, dummy_run=False)
        self.assertEqual(batch.req_ids, ["b", "a"])
        self.assertEqual(batch.idx_mapping_np.tolist(), [0, 2])
        self.assertEqual(
            GUARD.draft_graph_request_bounds(owner, state, batch.req_ids, batch.idx_mapping_np),
            ((513, 545), (1537, 1569)),
        )
        scheduled.num_scheduled_tokens = {"unknown": 4, "a": 4}
        with self.assertRaises(KeyError):
            runner.gather_batch_req_state(scheduled, dummy_run=False)
        state.prefill_len.np[2] += 1
        with self.assertRaises(NotImplementedError):
            GUARD.draft_graph_request_bounds(owner, state, ["a", "b"], [2, 0])
        state.prefill_len.np[2] -= 1
        owner._draft_graph_review_preempted_request_ids = {"b"}
        with self.assertRaises(NotImplementedError):
            GUARD.draft_graph_request_bounds(owner, state, ["a", "b"], [2, 0])

    def test_changed_normalized_head_dimension_stays_blocked(self):
        config = self.config()
        config.speculative_config.draft_model_config.hf_config.head_dim = 128
        with self.assertRaisesRegex(NotImplementedError, "consumer head_dim"):
            self.resolve(config)

    def test_exact_lengths_follow_canonical_continuing_prefill_progress(self):
        class Exact:
            def __init__(self, value):
                self.value = value

            def __getitem__(self, index):
                return NS(item=lambda: self.value)

        owner = self.qualified_owner(_draft_window_graph_binding_provider=NS(proposal=(True,)))
        batch = self.batch()
        batch.num_computed_prefill_tokens_np = np.array([512], dtype=np.int32)
        batch.num_scheduled_tokens = np.array([128], dtype=np.int32)
        batch.prefill_len_np = np.asarray(batch.prefill_len_np, dtype=np.int32)
        for step in (0, 1, 2):
            GUARD.validate_eagle_single_request_review_exact(
                owner, batch, Exact(640 + step), num_reqs=1, padded=1, step=step
            )
        for value in (639, 642, 1021):
            with self.subTest(value=value), self.assertRaises(NotImplementedError):
                GUARD.validate_eagle_single_request_review_exact(
                    owner, batch, Exact(value), num_reqs=1, padded=1, step=0
                )
        # The real proposal uses the same placeholder-aware prompt math.
        from vllm_ascend.worker.v2.spec_decode.draft_window_runtime import continuing_prefill_rows

        self.assertEqual(tuple(continuing_prefill_rows(batch, batch.num_reqs)), (True,))

    def test_full8_request_lookahead_cannot_exceed_allocated_capacity(self):
        config = self.config(method="dflash")
        owner = self.qualified_owner(config)
        batch = self.batch()
        GUARD.validate_draft_graph_review_requests(owner, batch)
        batch.draft_graph_request_bounds = ((1021, 40960),)
        with self.assertRaises(NotImplementedError):
            runner_bounds(owner, batch)

    def test_full8_capacity_uses_both_normalized_model_ceilings(self):
        for model in ("target", "draft"):
            config = self.config(method="dflash")
            actual = config.model_config if model == "target" else config.speculative_config.draft_model_config
            actual.hf_text_config = NS(max_position_embeddings=32768)
            with self.subTest(model=model), self.assertRaisesRegex(NotImplementedError, "both model position ceilings"):
                self.resolve(config)

    def test_full8_actual_metadata_boundary_rejects_unqualified_padding_before_materialize(self):
        from vllm_ascend.worker.v2.spec_decode.draft_window_eager import DraftWindowGraphInputMixin

        owner: Any = object.__new__(DraftWindowGraphInputMixin)
        owner.__dict__.update(vars(self.qualified_owner(self.config(method="dflash", rows=2))))
        provider = owner._draft_window_graph_binding_provider = NS(
            proposal=None, capture_identity_for=Mock(), prepare=Mock()
        )
        owner._draft_window_bypass_active = False
        owner.draft_attn_layer_names = {"draft.layer"}
        owner.num_query_per_req = 8
        owner.block_tables = NS(slot_mappings=torch.zeros((1, 32), dtype=torch.int32))
        for actual, padded in ((1, 1), (2, 2), (1, 2), (2, 4)):
            with self.subTest(actual=actual, padded=padded):
                batch = self.batch(rows=actual)
                GUARD.validate_draft_graph_review_requests(owner, batch)
                descriptor = NS(num_reqs=padded, num_tokens=padded * 8, cg_mode=NS(name="FULL"))
                provider.proposal = (False,) * actual
                owner.input_batch = NS(num_reqs=actual, idx_mapping_np=np.arange(actual))
                owner.input_buffers = NS(seq_lens=torch.zeros(padded, dtype=torch.int32))
                provider.prepare.reset_mock()
                provider.capture_identity_for.reset_mock()
                expected: Any = (
                    nullcontext() if actual == padded else self.assertRaisesRegex(RuntimeError, "uniform FULL")
                )
                with expected:
                    self.assertIsNone(
                        owner._build_uniform_attn_metadata(
                            descriptor,
                            actual,
                            8,
                            torch.zeros(actual, dtype=torch.int32),
                            8,
                        )
                    )
                if actual == padded:
                    provider.capture_identity_for.assert_called_once_with(descriptor=descriptor)
                    provider.prepare.assert_called_once()
                    self.assertIs(provider.prepare.call_args.kwargs["idx_mapping_np"], owner.input_batch.idx_mapping_np)
                    self.assertEqual(provider.prepare.call_args.kwargs["cpu_full_exact"].tolist(), [0] * padded)
                else:
                    provider.prepare.assert_not_called()
                    provider.capture_identity_for.assert_not_called()
        # Capacity is checked once before the proposal can change Graph ownership.
        with self.assertRaises(NotImplementedError):
            GUARD.validate_draft_graph_review_requests(owner, self.batch(rows=3))

    def test_context_only_full8_keeps_request_guards_without_fake_query_shape(self):
        config = self.config(method="dflash")
        owner = self.qualified_owner(config, _draft_window_graph_binding_provider=object())
        batch = self.batch()
        batch.num_reqs_after_padding = 2
        seen: list[str] = []

        def context_kv_scope(_owner):
            seen.append("context-KV-only")
            return nullcontext()

        with patch.object(dflash_module, "_draft_window_context_kv_scope", context_kv_scope):
            with dflash_module._draft_window_owner_scope(owner, batch, num_speculative_tokens=0):
                pass
            self.assertEqual(seen, ["context-KV-only"])
            with self.assertRaises(NotImplementedError):
                dflash_module._draft_window_owner_scope(owner, batch, num_speculative_tokens=7)
        self.assertFalse(hasattr(owner, "_draft_window_runtime"))

    def test_actual_propose_rejects_bad_rows_before_replacing_owner(self):
        config = self.config(method="dflash")
        old_batch = object()
        owner = object.__new__(dflash_module.AscendDFlashSpeculator)
        owner.__dict__.update(vars(self.qualified_owner(config, input_batch=old_batch)))
        batch = self.batch(rows=2)
        unused_tensor = torch.empty(0, device="cpu")
        with self.assertRaises(NotImplementedError):
            owner.propose(
                batch,
                attn_metadata={},
                slot_mappings={},
                last_hidden_states=unused_tensor,
                aux_hidden_states=None,
                num_sampled=unused_tensor,
                num_rejected=unused_tensor,
                last_sampled=unused_tensor,
                next_prefill_tokens=unused_tensor,
                temperature=unused_tensor,
                seeds=unused_tensor,
                num_speculative_tokens=7,
            )
        self.assertIs(owner.input_batch, old_batch)


def _single_owner_batch(prompt=257, output=32, capacity=4096):
    owner = NS(
        _draft_graph_review_limit=True, method="eagle3", max_model_len=capacity, num_speculative_steps=3, max_num_reqs=1
    )
    batch = NS(
        num_reqs=1,
        num_reqs_after_padding=1,
        req_ids=["actual-A"],
        idx_mapping_np=[7],
        prefill_len_np=[prompt],
        draft_graph_request_bounds=((prompt, prompt + output),),
    )
    owner.input_batch = batch
    return owner, batch


def _pair_owner_batch():
    owner = NS(
        _draft_graph_review_limit=True, method="eagle3", max_model_len=4096, num_speculative_steps=3, max_num_reqs=2
    )
    batch = NS(
        num_reqs=2,
        num_reqs_after_padding=2,
        req_ids=["request-A", "request-B"],
        idx_mapping_np=[7, 3],
        prefill_len_np=[1537, 513],
        draft_graph_request_bounds=((1537, 1569), (513, 545)),
    )
    owner.input_batch = batch
    return owner, batch


class ActualGraphDescriptorControls(unittest.TestCase):
    def manager(self, owner, *, first=False, session=None):
        seen: list[tuple[str, tuple[Any, ...]]] = []

        def build_metadata(*args):
            seen.append(("metadata", args))
            return {}

        def replay(*args):
            seen.append(("replay", args))
            return "legacy-result"

        def run_window(*args):
            seen.append(("window", args))
            return "window-result"

        owner.attn_backend = "actual-test-backend"
        owner.draft_vllm_config = object()
        owner.build_draft_attn_metadatas = build_metadata
        if session is not None:
            owner._eagle3_graph_session = session
            owner._eagle3_window_is_active = lambda: True
        manager = object.__new__(ar_graph.AutoRegressiveAclGraphManager)
        manager.__dict__.update(
            speculator=owner,
            is_draft_model_prefill=first,
            update_stream=object(),
            _graph_replay=replay,
            _run_eagle_window_fullgraph=run_window,
        )
        return manager, seen

    def run_actual(self, manager, desc):
        with patch.object(ar_graph, "use_updatable_graph", return_value=False):
            return manager.run_fullgraph(desc)

    def test_actual_propose_admits_rows_before_owner_change_and_off_replay(self):
        header, batch = _single_owner_batch()
        owner = object.__new__(ar_module.AscendTargetDependentARSpeculator)
        owner.__dict__.update(vars(header), method="eagle3", replicated_pcp=False, pcp_manager=None)
        old_batch = owner.input_batch = object()
        manager, seen = self.manager(owner)
        native_calls = []
        # The controlled parent returns at replay; no tensor operand is read.
        unused_tensor = torch.empty(0, device="cpu")

        def native_propose(actual, actual_batch, *args, **kwargs):
            native_calls.append(actual_batch)
            self.assertIs(actual, owner)
            return self.run_actual(manager, NS(num_reqs=1, num_tokens=1))

        with patch.object(ar_module.TargetDependentARSpeculator, "propose", native_propose):
            for damage in ("active", "padded", "missing-bounds", "row-bounds"):
                invalid = copy.deepcopy(batch)
                if damage == "active":
                    invalid.num_reqs = invalid.num_reqs_after_padding = 2
                elif damage == "padded":
                    invalid.num_reqs_after_padding = 2
                elif damage == "missing-bounds":
                    invalid.draft_graph_request_bounds = None
                else:
                    invalid.draft_graph_request_bounds += ((513, 545),)
                with self.subTest(damage=damage), self.assertRaises((RuntimeError, NotImplementedError)):
                    owner.propose(
                        invalid,
                        attn_metadata={},
                        slot_mappings={},
                        last_hidden_states=unused_tensor,
                        aux_hidden_states=None,
                        num_sampled=unused_tensor,
                        num_rejected=unused_tensor,
                        last_sampled=unused_tensor,
                        next_prefill_tokens=unused_tensor,
                        temperature=unused_tensor,
                        seeds=unused_tensor,
                    )
                self.assertIs(owner.input_batch, old_batch)
                self.assertEqual(native_calls, [])
                self.assertEqual(seen, [])
            result = owner.propose(
                batch,
                attn_metadata={},
                slot_mappings={},
                last_hidden_states=unused_tensor,
                aux_hidden_states=None,
                num_sampled=unused_tensor,
                num_rejected=unused_tensor,
                last_sampled=unused_tensor,
                next_prefill_tokens=unused_tensor,
                temperature=unused_tensor,
                seeds=unused_tensor,
            )
            self.assertEqual(result, "legacy-result")
        self.assertIs(owner.input_batch, batch)
        self.assertEqual(native_calls, [batch])
        self.assertEqual([kind for kind, _ in seen], ["metadata", "replay"])

    def test_off_padded_ar_rejected_before_metadata_and_replay(self):
        owner, _ = _single_owner_batch()
        manager, seen = self.manager(owner)
        for n, tokens in ((4, 4), (1, 4), (2, 2)):
            with self.subTest(n=n, tokens=tokens), self.assertRaisesRegex(NotImplementedError, "padded1"):
                self.run_actual(manager, NS(num_reqs=n, num_tokens=tokens))
            self.assertEqual(seen, [])
        self.assertEqual(self.run_actual(manager, NS(num_reqs=1, num_tokens=1)), "legacy-result")
        self.assertEqual([v[0] for v in seen], ["metadata", "replay"])
        self.assertFalse(hasattr(owner, "_draft_window_graph_binding_provider"))
        self.assertFalse(hasattr(owner, "_draft_window_runtime"))

    def test_first_q4_descriptor_checks_before_replay(self):
        owner, _ = _single_owner_batch()
        manager, seen = self.manager(owner, first=True)
        self.assertEqual(self.run_actual(manager, NS(num_reqs=1, num_tokens=4)), "legacy-result")
        seen.clear()
        with self.assertRaisesRegex(NotImplementedError, "first Q4"):
            self.run_actual(manager, NS(num_reqs=4, num_tokens=4))
        self.assertEqual(seen, [])

    def test_window_session_keeps_its_actual_materialize_and_binding_path(self):
        owner, _ = _single_owner_batch()
        manager, seen = self.manager(owner, session=object())
        self.assertEqual(self.run_actual(manager, NS(num_reqs=1, num_tokens=1)), "window-result")
        self.assertEqual([v[0] for v in seen], ["window"])

    def test_unscoped_and_real_source_dummy_keep_legacy_behavior(self):
        for owner in (
            NS(),
            NS(
                _draft_graph_review_limit=True,
                method="eagle3",
                num_speculative_steps=3,
                max_num_reqs=1,
                max_model_len=4096,
                input_batch=NS(is_dummy=True),
            ),
        ):
            manager, seen = self.manager(owner)
            self.assertEqual(self.run_actual(manager, NS(num_reqs=4, num_tokens=4)), "legacy-result")
            self.assertEqual([v[0] for v in seen], ["metadata", "replay"])
        owner, batch = _single_owner_batch()
        batch.is_dummy = False
        batch.num_reqs = 2
        manager, seen = self.manager(owner)
        with self.assertRaises(NotImplementedError):
            self.run_actual(manager, NS(num_reqs=1, num_tokens=1))
        self.assertEqual(seen, [])


class EagleMixedRequestControls(unittest.TestCase):
    def test_mixed_chunked_prefill_uses_canonical_progress_for_each_consumer(self):
        # C333 ordinary g0 failed with these saved SchedulerOutput counters:
        # long prompt 1537: 1152 computed + 128 scheduled = 1280; short prompt
        # 513: 384 + 129 = 513. These are Source-reconstructed lengths, not a
        # saved device/guard exact snapshot. Execute the actual Ascend CPU
        # length constructor on that state through the actual runner class.
        owner, batch = _pair_owner_batch()
        ids = batch.req_ids
        cpu_lengths = [0, 0]
        snapshots = {7: 1152, 3: 384}
        runner = object.__new__(NPUModelRunner)
        runner.__dict__.update(
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
        runner._update_seq_lens_cpu(scheduler, ids)
        self.assertEqual(cpu_lengths, [1280, 513])
        batch.num_computed_prefill_tokens_np = np.array([1152, 384], dtype=np.int32)
        batch.num_scheduled_tokens = np.array([128, 129], dtype=np.int32)
        batch.prefill_len_np = np.asarray(batch.prefill_len_np, dtype=np.int32)
        owner._draft_window_graph_binding_provider = NS(proposal=(True, False))
        for step in (0, 1, 2):
            GUARD.validate_eagle_single_request_review_exact(
                owner,
                batch,
                torch.tensor([1280 + step, 513 + step], dtype=torch.int32),
                num_reqs=2,
                padded=2,
                step=step,
            )
            for values in ([1279 + step, 513 + step], [1281 + step, 513 + step], [1280 + step, 512]):
                with self.subTest(step=step, values=values), self.assertRaises(NotImplementedError):
                    GUARD.validate_eagle_single_request_review_exact(
                        owner, batch, torch.tensor(values, dtype=torch.int32), num_reqs=2, padded=2, step=step
                    )
        # Absent/changed phase cannot relabel a terminal/decode row as continuing.
        owner._draft_window_graph_binding_provider.proposal = (False, False)
        with self.assertRaises(NotImplementedError):
            GUARD.validate_eagle_single_request_review_exact(
                owner, batch, torch.tensor(cpu_lengths, dtype=torch.int32), num_reqs=2, padded=2, step=0
            )
        from vllm_ascend.worker.v2.spec_decode.draft_window_runtime import continuing_prefill_rows

        batch.num_draft_tokens_per_req = None
        self.assertEqual(tuple(continuing_prefill_rows(batch, batch.num_reqs)), (True, False))

    def test_continuing_prefill_does_not_relax_identity_capacity_or_counter_contract(self):
        for damage in (
            "wrong-id",
            "bad-index",
            "capacity",
            "preempted",
            "missing",
            "negative",
            "terminal",
            "zero",
        ):
            owner, batch = _pair_owner_batch()
            batch.num_computed_prefill_tokens_np = np.array([1152, 384], dtype=np.int32)
            batch.num_scheduled_tokens = np.array([128, 129], dtype=np.int32)
            batch.prefill_len_np = np.asarray(batch.prefill_len_np, dtype=np.int32)
            batch.num_draft_tokens_per_req = None
            batch.is_prefilling_np, batch.has_prefill = [True, True], True
            if damage == "wrong-id":
                batch.req_ids[0] = "different-request"
            elif damage == "bad-index":
                batch.idx_mapping_np[0] = 6
            elif damage == "capacity":
                batch.draft_graph_request_bounds = ((1537, 4096), (513, 545))
            elif damage == "preempted":
                owner._draft_graph_review_preempted_request_ids = {batch.req_ids[0]}
            elif damage == "missing":
                del batch.num_scheduled_tokens
            elif damage == "negative":
                batch.num_computed_prefill_tokens_np[0] = -1
            elif damage == "terminal":
                batch.num_scheduled_tokens[0] = 385
            else:
                batch.num_scheduled_tokens[0] = 0
            with (
                self.subTest(damage=damage),
                self.assertRaises((RuntimeError, NotImplementedError, ValueError, TypeError, AttributeError)),
            ):
                if damage in ("wrong-id", "bad-index"):
                    # Identity is established when the actual runner builds the
                    # batch, not repeatedly reconstructed by each AR reader.
                    state = NS(req_id_to_index={"request-A": 7, "request-B": 3})
                    GUARD.draft_graph_request_bounds(owner, state, batch.req_ids, batch.idx_mapping_np)
                from vllm_ascend.worker.v2.spec_decode.draft_window_runtime import continuing_prefill_rows

                runner_bounds(owner, batch)
                owner._draft_window_graph_binding_provider = NS(
                    proposal=tuple(continuing_prefill_rows(batch, batch.num_reqs))
                )
                GUARD.validate_draft_graph_review_requests(owner, batch)
                GUARD.validate_eagle_single_request_review_exact(
                    owner, batch, torch.tensor([1280, 513], dtype=torch.int32), num_reqs=2, padded=2, step=0
                )


def idle_owner():
    provider = NS(method="eagle3", _active_bank=None, proposal=None, quarantined=False)
    session = NS(provider=provider, _capture_call=None)
    batch = NS(is_dummy=True, num_reqs=1, num_reqs_after_padding=1)
    owner = NS(
        method="eagle3",
        dp_size=2,
        num_speculative_steps=3,
        max_num_reqs=1,
        input_batch=batch,
        vllm_config=NS(additional_config={}),
        _draft_graph_review_limit=True,
        update_stream=object(),
        _eagle3_graph_session=session,
        _eagle3_window_bypass_active=True,
        _eagle3_window_continuing_prefill=False,
        _draft_window_graph_binding_provider=provider,
    )

    def require_reusable():
        if provider.quarantined:
            raise RuntimeError("quarantined")

    provider.require_reusable = require_reusable
    provider.quarantine = lambda: setattr(provider, "quarantined", True)
    return owner, batch, session


class EagleIdleOwnershipControls(unittest.TestCase):
    def test_actual_idle_uses_full_baseline_and_both_consumer_managers(self):
        owner, batch, session = idle_owner()
        with DP.eagle3_dp_dummy_scope(owner, batch, dummy_run=True, is_profile=False):
            self.assertTrue(owner._eagle3_graph_dp_dummy_active)
            for first, tokens, count in ((True, 4, 1), (False, 1, 2)):
                manager = NS(speculator=owner, update_stream=owner.update_stream, is_draft_model_prefill=first)
                descriptor = NS(cg_mode=NS(name="FULL"), num_reqs=1, num_tokens=tokens)
                self.assertTrue(DP.validate_eagle_dp_dummy_replay(owner, manager, descriptor, [{}] * count))
                with self.assertRaises(RuntimeError):
                    DP.validate_eagle_dp_dummy_replay(owner, manager, descriptor, [{}] * (count + 1))
            self.assertIsNone(owner._draft_window_graph_binding_provider.proposal)
        self.assertFalse(owner._eagle3_graph_dp_dummy_active)
        self.assertFalse(owner._draft_window_graph_binding_provider.quarantined)
        # K0 business first-only is a genuine request, never dummy/idle.
        batch.is_dummy = False
        with DP.eagle3_dp_dummy_scope(owner, batch, dummy_run=False, is_profile=False):
            self.assertFalse(owner._eagle3_graph_dp_dummy_active)

    def test_idle_failure_or_live_generation_cannot_reuse(self):
        owner, batch, session = idle_owner()
        with (
            self.assertRaisesRegex(RuntimeError, "injected"),
            DP.eagle3_dp_dummy_scope(owner, batch, dummy_run=True, is_profile=False),
        ):
            raise RuntimeError("injected submission uncertainty")
        self.assertTrue(owner._draft_window_graph_binding_provider.quarantined)
        self.assertFalse(owner._eagle3_graph_dp_dummy_active)
        with (
            self.assertRaisesRegex(RuntimeError, "quarantined"),
            DP.eagle3_dp_dummy_scope(owner, batch, dummy_run=True, is_profile=False),
        ):
            self.fail("uncertain idle work must reject the next Graph scope")
        owner, batch, session = idle_owner()
        owner._draft_window_graph_binding_provider.proposal = object()
        with (
            self.assertRaises(RuntimeError),
            DP.eagle3_dp_dummy_scope(owner, batch, dummy_run=True, is_profile=False),
        ):
            self.fail("live business generation entered dummy replay")
        self.assertTrue(owner._draft_window_graph_binding_provider.quarantined)
        owner, batch, session = idle_owner()
        owner._draft_window_graph_binding_provider._active_bank = object()
        with (
            self.assertRaises(RuntimeError),
            DP.eagle3_dp_dummy_scope(owner, batch, dummy_run=True, is_profile=False),
        ):
            self.fail("live scratch entered dummy replay")
        owner, batch, session = idle_owner()
        with DP.eagle3_dp_dummy_scope(owner, batch, dummy_run=True, is_profile=True):
            self.assertFalse(getattr(owner, "_eagle3_graph_dp_dummy_active", False))


if __name__ == "__main__":
    unittest.main()
