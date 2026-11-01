# SPDX-License-Identifier: Apache-2.0
"""Use the actual installed logger contracts; no model or device initialization."""

import ast
import logging
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from transformers.utils import logging as transformers_logging
from vllm import logger as vllm_logging
from vllm.distributed import parallel_state

from tests.ut.worker.v2.test_draft_window_native_attention import config, loaded, native

SOURCE = Path(__file__).resolve().parents[4]
AR = SOURCE / "vllm_ascend/worker/v2/spec_decode/target_dependent_ar/speculator.py"


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

    def actual_ar_logger(self, name):
        tree = ast.parse(AR.read_text())
        imports = [node for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == "vllm.logger"]
        self.assertTrue(any(alias.name == "logger" for node in imports for alias in node.names))
        self.assertFalse(
            any(
                isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "logger" for target in node.targets)
                for node in tree.body
            )
        )
        namespace = {}
        exec(compile(ast.Module(body=imports, type_ignores=[]), str(AR), "exec"), namespace)
        self.assertIs(namespace["logger"], vllm_logging.logger)
        # Keep each case's handler and once-key independent while exercising
        # the same installed adapter class as the shared production logger.
        logger = vllm_logging.init_logger(name)
        self.assertIs(type(logger), type(namespace["logger"]))
        return logger

    def test_transformers_stdlib_scope_fails_and_actual_ar_adapter_accepts(self):
        logger, records = self.logger()
        # This is the installed Transformers class-level patch, not a mock
        # accepting arbitrary kwargs. The failing Source used this plain logger.
        self.assertIs(logger.info_once.__func__, transformers_logging.info_once)
        for scope in ("local", "global"):
            with self.subTest(scope=scope), self.assertRaisesRegex(TypeError, "scope"):
                logger.info_once("old native summary", scope=scope)
        logger = self.actual_ar_logger(logger.name)
        self.assertIs(logger.info_once.__func__, vllm_logging._VllmLogger.info_once)
        self.assertIs(logger.warning_once.__func__, vllm_logging._VllmLogger.warning_once)
        with patch.object(parallel_state, "_WORLD", SimpleNamespace(is_first_rank=True, local_rank=0)):
            owner = loaded(config(kinds=["sliding_attention"] * 2), [2048, 2048])
            for _ in range(2):
                native.report_draft_native_attention(owner, logger=logger, window_size=1024)
        self.assertEqual(len(records.messages), 2)
        self.assertIn("native attention=SWA", records.messages[0])
        self.assertIn("additional draft_window_size=1024", records.messages[1])

    def test_global_scope_still_excludes_nonzero_rank_and_deduplicates_rank_zero(self):
        for rank in (0, 1, 3):
            with self.subTest(rank=rank):
                logger, records = self.logger()
                logger = self.actual_ar_logger(logger.name)
                with patch.object(parallel_state, "_WORLD", SimpleNamespace(is_first_rank=rank == 0, local_rank=rank)):
                    for _ in range(2):
                        # Distinct loaded owners share the logger's exact
                        # once-key without a second per-owner deduplication cache.
                        owner = loaded(config(kinds=["sliding_attention"] * 2), [2048, 2048])
                        native.report_draft_native_attention(owner, logger=logger, window_size=1024)
                        native.report_draft_native_attention(owner, logger=logger, window_size=1024)
                self.assertEqual(len(records.messages), 2 if rank == 0 else 0)


if __name__ == "__main__":
    unittest.main()
