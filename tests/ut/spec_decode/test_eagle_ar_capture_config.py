# SPDX-License-Identifier: Apache-2.0
"""Narrow CPU controls for the admitted dense Eagle AR capture descriptor."""

import copy
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from vllm.config.compilation import CUDAGraphMode as Mode
from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.cudagraph_utils import SpeculatorCudaGraphManager

from vllm_ascend import envs
from vllm_ascend.worker.v2.spec_decode import draft_graph_review_limits as GUARD
from vllm_ascend.worker.v2.spec_decode.target_dependent_ar.aclgraph import (
    AutoRegressiveAclGraphManager,
    _eagle_ar_capture_config,
)


def config(maximum=1):
    model = copy.deepcopy(GUARD._CONSUMER_CONFIGS["eagle3"])
    hf = NS(**model)
    return NS(
        speculative_config=NS(
            method="eagle3",
            enforce_eager=False,
            num_speculative_tokens=3,
            uses_dynamic_speculative_decoding=lambda: False,
            enable_adaptive_verification=False,
            draft_model_config=NS(model="draft", hf_config=hf, hf_text_config=hf, dtype="torch.bfloat16"),
            draft_parallel_config=NS(
                tensor_parallel_size=4,
                data_parallel_size=1,
                use_sequence_parallel_moe=False,
                enable_expert_parallel=False,
            ),
        ),
        scheduler_config=NS(max_num_seqs=maximum, async_scheduling=True),
        parallel_config=NS(
            tensor_parallel_size=4,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            use_sequence_parallel_moe=True,
            enable_expert_parallel=True,
            all2all_backend="allgather_reducescatter",
        ),
        compilation_config=NS(
            cudagraph_mode=Mode.FULL_DECODE_ONLY, cudagraph_capture_sizes=[4, 8, 16], max_cudagraph_capture_size=16
        ),
        cache_config=NS(enable_prefix_caching=True),
        model_config=NS(max_model_len=4096),
        additional_config={},
    )


class EagleARCaptureTests(unittest.TestCase):
    def setUp(self):
        self.experimental = patch.object(envs, "VLLM_ASCEND_EXPERIMENTAL_DRAFT_WINDOW_GRAPH", True)
        self.experimental.start()
        self.addCleanup(self.experimental.stop)

    def config(self, maximum=1):
        return config(maximum)

    def call(self, conf, query=1, mode=Mode.FULL_DECODE_ONLY):
        return _eagle_ar_capture_config(conf, mode, query)

    def test_only_compilation_copy_keeps_target_first_and_real_topology(self):
        for maximum, expected in ((1, [1]), (2, [1, 2])):
            conf = self.config(maximum)
            before = list(conf.compilation_config.cudagraph_capture_sizes)
            actual = self.call(conf)
            self.assertIsNot(actual, conf)
            self.assertIsNot(actual.compilation_config, conf.compilation_config)
            self.assertEqual(actual.compilation_config.cudagraph_capture_sizes, expected)
            self.assertEqual(actual.compilation_config.max_cudagraph_capture_size, max(expected))
            self.assertEqual(conf.compilation_config.cudagraph_capture_sizes, before)
            self.assertIs(actual.parallel_config, conf.parallel_config)
            self.assertIs(actual.scheduler_config, conf.scheduler_config)
            self.assertIs(actual.parallel_config.use_sequence_parallel_moe, True)
            self.assertIs(self.call(conf, query=4), conf)

    def test_unchanged_off_eager_mtp_and_unadmitted_profiles(self):
        conf = self.config()
        self.assertIs(self.call(conf, mode=Mode.NONE), conf)
        for mutate in (
            lambda c: setattr(c.speculative_config, "method", "deepseek_mtp"),
            lambda c: setattr(c.speculative_config, "enforce_eager", True),
        ):
            value = copy.deepcopy(conf)
            mutate(value)
            self.assertIs(self.call(value), value)
        value = self.config()
        value.speculative_config.draft_model_config.hf_config.architectures = ["UnqualifiedEagle"]
        self.assertIs(self.call(value), value)
        value = self.config()
        value.parallel_config.tensor_parallel_size = 2
        with self.assertRaises(NotImplementedError):
            self.call(value)
        value = self.config()
        value.parallel_config.use_sequence_parallel_moe = False
        value.parallel_config.all2all_backend = "flashinfer_all2allv"
        self.assertIsNot(self.call(value), value)
        envs.VLLM_ASCEND_EXPERIMENTAL_DRAFT_WINDOW_GRAPH = False
        self.assertIs(self.call(conf), conf)

    def test_dp2_uses_actual_local_one_contract_without_changing_draft_dp(self):
        conf = self.config()
        conf.parallel_config.data_parallel_size = 2
        actual = self.call(conf)
        self.assertEqual(actual.compilation_config.cudagraph_capture_sizes, [1])
        self.assertIs(actual.parallel_config, conf.parallel_config)
        self.assertEqual(actual.parallel_config.data_parallel_size, 2)
        self.assertIs(actual.speculative_config.draft_parallel_config, conf.speculative_config.draft_parallel_config)
        self.assertEqual(actual.speculative_config.draft_parallel_config.data_parallel_size, 1)

    def test_real_config_constructors_own_integer_schema_before_capture(self):
        from vllm.config import CompilationConfig, ParallelConfig, SchedulerConfig, SpeculativeConfig

        conf = self.config()
        conf.parallel_config = ParallelConfig(
            tensor_parallel_size="4",
            data_parallel_size="1",
            enable_expert_parallel=True,
            all2all_backend="flashinfer_all2allv",
            distributed_executor_backend="mp",
        )
        conf.speculative_config.draft_parallel_config = SpeculativeConfig.create_draft_parallel_config(
            conf.parallel_config, 4, NS(is_moe=False)
        )
        conf.scheduler_config = SchedulerConfig(max_model_len=4096, is_encoder_decoder=False, max_num_seqs=True)
        conf.compilation_config = CompilationConfig(
            cudagraph_mode=Mode.FULL_DECODE_ONLY,
            cudagraph_capture_sizes=[True, "4"],
            max_cudagraph_capture_size="4",
        )
        self.assertIs(type(conf.parallel_config.tensor_parallel_size), int)
        self.assertIs(type(conf.parallel_config.data_parallel_size), int)
        self.assertIs(type(conf.scheduler_config.max_num_seqs), int)
        self.assertTrue(all(type(size) is int for size in conf.compilation_config.cudagraph_capture_sizes))
        self.assertIs(type(conf.compilation_config.max_cudagraph_capture_size), int)
        self.assertEqual(self.call(conf).compilation_config.cudagraph_capture_sizes, [1])
        with self.assertRaises(ValueError):
            SchedulerConfig(max_model_len=4096, is_encoder_decoder=False, max_num_seqs=1.5)
        with self.assertRaises(ValueError):
            CompilationConfig(cudagraph_capture_sizes=["not an integer"])

    def test_no_invented_buckets_or_unknown_dense_layout(self):
        conf = self.config()
        # The public factory already rejects these configurations before
        # constructing a speculator. Capture uses that same startup contract.
        for mutate in (
            lambda c: setattr(c.speculative_config, "num_speculative_tokens", 7),
            lambda c: setattr(c.speculative_config, "uses_dynamic_speculative_decoding", lambda: True),
            lambda c: setattr(c.parallel_config, "pipeline_parallel_size", 2),
        ):
            value = self.config()
            mutate(value)
            with self.assertRaises(NotImplementedError):
                self.call(value)
        for field in ("enable_expert_parallel",):
            value = copy.deepcopy(conf)
            setattr(value.speculative_config.draft_parallel_config, field, True)
            with self.assertRaises(NotImplementedError):
                self.call(value)
        conf.compilation_config.cudagraph_capture_sizes = [8, 16]
        with self.assertRaisesRegex(NotImplementedError, "Qualified eagle3"):
            self.call(conf)
        for maximum in (None, True, 0):
            conf = self.config()
            conf.compilation_config.max_cudagraph_capture_size = maximum
            with self.assertRaises(NotImplementedError):
                self.call(conf)

    def test_actual_constructor_passes_the_adapted_config_to_the_original_parent(self):
        conf, parent_configs = self.config(2), []

        def initialize(manager, value, *args, **kwargs):
            parent_configs.append(value)
            manager._capture_descs = {}

        with (
            patch.object(SpeculatorCudaGraphManager, "__init__", initialize),
            patch.object(SpeculatorCudaGraphManager, "needs_capture", return_value=False),
        ):
            manager = AutoRegressiveAclGraphManager(conf, "cpu", Mode.FULL_DECODE_ONLY, 1)
        self.assertEqual(parent_configs[0].compilation_config.cudagraph_capture_sizes, [1, 2])
        self.assertEqual(conf.compilation_config.cudagraph_capture_sizes, [4, 8, 16])
        self.assertFalse(manager.is_draft_model_prefill)
        self.assertIsNone(manager.speculator)


if __name__ == "__main__":
    unittest.main()
