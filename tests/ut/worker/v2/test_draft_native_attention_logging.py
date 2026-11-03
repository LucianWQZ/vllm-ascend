# SPDX-License-Identifier: Apache-2.0
"""Use the actual installed logger contracts; no model or device initialization."""

import logging
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import patch

from transformers.utils import logging as transformers_logging
from vllm import logger as vllm_logging
from vllm.distributed import parallel_state

from tests.ut.worker.v2.test_draft_window_native_attention import config, loaded, native


class Records(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


class DraftNativeAttentionLoggingTests(unittest.TestCase):
    def logger(self):
        logger = logging.getLogger("draft-native-test-" + uuid.uuid4().hex)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        records = Records()
        logger.addHandler(records)
        self.addCleanup(logger.removeHandler, records)
        return logger, records

    def actual_native_logger(self, name):
        self.assertIs(native.logger, vllm_logging.logger)
        logger = vllm_logging.init_logger(name)
        self.assertIs(type(logger), type(native.logger))
        return logger

    def test_transformers_stdlib_scope_fails_and_actual_native_adapter_accepts(self):
        logger, records = self.logger()
        # This is the installed Transformers class-level patch, not a mock
        # accepting arbitrary kwargs. The failing Source used this plain logger.
        self.assertIs(logger.info_once.__func__, transformers_logging.info_once)
        for scope in ("local", "global"):
            with self.subTest(scope=scope), self.assertRaisesRegex(TypeError, "scope"):
                logger.info_once("old native summary", scope=scope)
        logger = self.actual_native_logger(logger.name)
        self.assertIs(logger.info_once.__func__, vllm_logging._VllmLogger.info_once)
        self.assertIs(logger.warning_once.__func__, vllm_logging._VllmLogger.warning_once)
        with (
            patch.object(parallel_state, "_WORLD", SimpleNamespace(is_first_rank=True, local_rank=0)),
            patch.object(native, "logger", logger),
        ):
            owner = loaded(config(kinds=["sliding_attention"] * 2), [2048, 2048])
            owner.speculative_config = SimpleNamespace(enforce_eager=True)
            owner.draft_window_policy = 1024
            for _ in range(2):
                native.report_owner_draft_native_attention(owner)
        self.assertEqual(len(records.messages), 2)
        self.assertIn("native attention=SWA", records.messages[0])
        self.assertIn("additional draft_window_size=1024", records.messages[1])

    def test_global_scope_still_excludes_nonzero_rank_and_deduplicates_rank_zero(self):
        for rank in (0, 1, 3):
            with self.subTest(rank=rank):
                logger, records = self.logger()
                logger = self.actual_native_logger(logger.name)
                with (
                    patch.object(parallel_state, "_WORLD", SimpleNamespace(is_first_rank=rank == 0, local_rank=rank)),
                    patch.object(native, "logger", logger),
                ):
                    for _ in range(2):
                        # Distinct loaded owners share the logger's exact
                        # once-key without a second per-owner deduplication cache.
                        owner = loaded(config(kinds=["sliding_attention"] * 2), [2048, 2048])
                        owner.speculative_config = SimpleNamespace(enforce_eager=True)
                        owner.draft_window_policy = 1024
                        native.report_owner_draft_native_attention(owner)
                        native.report_owner_draft_native_attention(owner)
                self.assertEqual(len(records.messages), 2 if rank == 0 else 0)


if __name__ == "__main__":
    unittest.main()
