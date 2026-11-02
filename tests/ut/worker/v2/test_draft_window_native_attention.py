# SPDX-License-Identifier: Apache-2.0
"""Host-only native attention inventory; no framework or device imports."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
MODULE = ROOT / "vllm_ascend/worker/v2/spec_decode/draft_window_native_attention.py"
spec = importlib.util.spec_from_file_location("_native_attention_host_test", MODULE)
assert spec is not None and spec.loader is not None
native = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = native
spec.loader.exec_module(native)

Dense = type("AscendAttentionBackendImpl", (), {})
Dense.__module__ = "vllm_ascend.attention.attention_v1"


def config(*, kinds=None, window=2048, architecture="DFlash2DraftModel", count=2, dflash=None) -> dict[str, Any]:
    result = {"architectures": [architecture], "num_hidden_layers": count, "sliding_window": window}
    if kinds is not None:
        result["layer_types"] = kinds
    if dflash is not None:
        result["dflash_config"] = dflash
    return result


def loaded(configuration, windows, *, causal=False, mask=True, order=None):
    context = {}
    names = []
    for index, window in enumerate(windows):
        name = f"draft.model.layers.{64 + index}.self_attn.attn"
        impl = Dense()
        impl.sliding_window = window
        if mask:
            impl._dflash2_native_swa_mask = object()
        context[name] = SimpleNamespace(impl=impl)
        names.append(name)
    groups = [SimpleNamespace(layer_names=[names[index]]) for index in (order or range(len(names)))]
    return SimpleNamespace(
        draft_model_config=SimpleNamespace(hf_config=configuration),
        attn_vllm_config=SimpleNamespace(compilation_config=SimpleNamespace(static_forward_context=context)),
        attn_groups=[groups],
        draft_attn_layer_names=set(names),
        _group_causal=causal,
    )


class Log:
    def __init__(self):
        self.info = []
        self.warnings = []

    def info_once(self, *args, **kwargs):
        if (args, kwargs) not in self.info:
            self.info.append((args, kwargs))

    def warning_once(self, *args, **kwargs):
        if (args, kwargs) not in self.warnings:
            self.warnings.append((args, kwargs))


class NativeAttentionTests(unittest.TestCase):
    def test_loaded_consumers_are_authoritative_after_loader_normalization(self):
        # The fixed loader resolves wrapper defaults/use_swa. The inventory
        # reads the resulting consumer, without implementing that loader again.
        cfg = config(kinds=["full_attention"] * 2, dflash={"use_swa": True})
        owner = loaded(cfg, [2048, 2048])
        owner.draft_model_config.model = "/does/not/need/config.json"
        owner.speculative_config = SimpleNamespace(enforce_eager=True)
        owner.draft_window_policy = SimpleNamespace(effective_window_size=1024)
        log = Log()
        with patch.object(Path, "read_text", side_effect=AssertionError("startup must not reload config")):
            for _ in range(2):
                self.assertEqual(native.report_owner_draft_native_attention(owner, log).kind, "SWA")
        self.assertEqual(len(log.info), 1)
        self.assertEqual(len(log.warnings), 1)
        self.assertFalse(hasattr(owner, "_draft_window_runtime"))

    def test_unknown_architecture_or_missing_count_does_not_become_full(self):
        for cfg in (config(architecture="UnknownDraft"), {}, config(count=0)):
            result = native.inspect_draft_native_attention(loaded(cfg, [None, None]))
            self.assertEqual(result.kind, "UNKNOWN")
            with self.assertRaises(NotImplementedError):
                native.require_known_draft_native_attention(result, window_size=1024, graph_requested=False)

    def test_actual_window_and_complete_layer_registration_are_required(self):
        for windows in ([0, 2048], [True, 2048], [2048], [None, None, None]):
            with self.subTest(windows=windows):
                result = native.inspect_draft_native_attention(loaded(config(), windows))
                self.assertEqual(result.kind, "UNKNOWN")
        owner = loaded(config(), [None, None])
        owner.attn_groups[0].append(owner.attn_groups[0][0])
        self.assertEqual(native.inspect_draft_native_attention(owner).kind, "UNKNOWN")

    def test_eagle_uses_its_actual_loaded_attention_not_qwen_flags(self):
        cfg = config(architecture="LlamaForCausalLMEagle3", count=1)
        cfg.update(model_type="llama", use_sliding_window=True)
        self.assertEqual(native.inspect_draft_native_attention(loaded(cfg, [None], causal=True)).kind, "FULL")
        self.assertEqual(native.inspect_draft_native_attention(loaded(cfg, [2048], causal=True)).kind, "SWA")

    def test_owner_actual_draft_none_mode_does_not_inherit_target_graph(self):
        owner = loaded(config(), [None, None])
        next(iter(owner.attn_vllm_config.compilation_config.static_forward_context.values())).impl = object()
        owner.speculative_config = SimpleNamespace(enforce_eager=False)
        owner.query_cudagraph_manager = SimpleNamespace(cudagraph_mode=SimpleNamespace(name="NONE"))
        owner.attn_vllm_config.compilation_config.cudagraph_mode = SimpleNamespace(name="FULL")
        log = Log()
        result = native.report_owner_draft_native_attention(owner, log)
        self.assertEqual(result.kind, "UNKNOWN")
        self.assertEqual(log.info[0][0][4], False)
        owner.query_cudagraph_manager.cudagraph_mode.name = "FULL"
        with self.assertRaises(NotImplementedError):
            native.report_owner_draft_native_attention(owner, log)

    def test_startup_inventory_preserves_baseline_off_unknown_graph(self):
        owner = loaded(config(architecture="UnclassifiedMTPDraft"), [None, None])
        owner.speculative_config = SimpleNamespace(method="mtp", enforce_eager=False)
        owner.query_cudagraph_manager = SimpleNamespace(cudagraph_mode=SimpleNamespace(name="FULL"))
        log = Log()
        self.assertEqual(native.report_owner_draft_native_attention(owner, log).kind, "UNKNOWN")
        self.assertEqual(log.info[0][0][4], True)
        owner.draft_window_policy = SimpleNamespace(effective_window_size=1024)
        with self.assertRaises(NotImplementedError):
            native.report_owner_draft_native_attention(owner, log)

    def test_full_swa_mixed_and_full_backing_not_attention(self):
        for kinds, windows, expected in (
            (["full_attention"] * 2, [None, None], "FULL"),
            (["sliding_attention"] * 2, [2048, 2048], "SWA"),
            (["full_attention", "sliding_attention"], [None, 2048], "MIXED"),
        ):
            owner = loaded(config(kinds=kinds), windows, order=[1, 0])
            # Deliberately Full-retained storage must not override the reader.
            owner.kv_cache_config = SimpleNamespace(kind="FullAttentionSpec")
            result = native.inspect_draft_native_attention(owner)
            self.assertEqual(result.kind, expected)
            self.assertTrue(result.actual_complete)

    def test_group_must_match_the_actual_loader_draft_layer_registration(self):
        owner = loaded(config(), [None, None])
        context = owner.attn_vllm_config.compilation_config.static_forward_context
        old = owner.attn_groups[0][1].layer_names[0]
        new = old.replace("layers.65.", "layers.66.")
        context[new] = context.pop(old)
        owner.attn_groups[0][1].layer_names = [new]
        self.assertEqual(native.inspect_draft_native_attention(owner).kind, "UNKNOWN")

    def test_missing_consumer_mask_and_unknown_implementation_rejected(self):
        cfg = config(kinds=["sliding_attention"] * 2)
        for owner in (loaded(cfg, [2048]), loaded(cfg, [2048, 2048], mask=False)):
            self.assertEqual(native.inspect_draft_native_attention(owner).kind, "UNKNOWN")
        owner = loaded(cfg, [2048, 2048])
        next(iter(owner.attn_vllm_config.compilation_config.static_forward_context.values())).impl = object()
        self.assertEqual(native.inspect_draft_native_attention(owner).kind, "UNKNOWN")

    def test_off_no_runtime_and_unknown_window_graph_fail_closed(self):
        owner = loaded(config(architecture="UnknownDraft"), [None, None])
        inventory = native.inspect_draft_native_attention(owner)
        native.require_known_draft_native_attention(inventory, window_size=None, graph_requested=False)
        for window, graph in ((1024, False), (None, True)):
            with self.assertRaises(NotImplementedError):
                native.require_known_draft_native_attention(inventory, window_size=window, graph_requested=graph)
        self.assertFalse(hasattr(owner, "_draft_window_runtime"))
        self.assertFalse(hasattr(owner, "_draft_window_runtime_state"))

    def test_summary_once_and_only_actual_narrowing_warns(self):
        for windows, window, expected in (
            ([2048, 2048], None, ()),
            ([2048, 2048], 1024, (2048,)),
            ([2048, 2048], 2048, ()),
            ([2048, 2048], 4096, ()),
            ([2048, 4096], 2048, (4096,)),
            ([4096, 4096], 1024, (4096,)),
        ):
            owner = loaded(config(kinds=["sliding_attention"] * 2), windows)
            owner.speculative_config = SimpleNamespace(enforce_eager=True)
            owner.draft_window_policy = SimpleNamespace(effective_window_size=window)
            log = Log()
            for _ in range(2):
                result = native.report_owner_draft_native_attention(owner, log)
            self.assertEqual(result.kind, "SWA")
            self.assertEqual(len(log.info), 1)
            self.assertEqual(len(log.warnings), bool(expected))
            if expected:
                self.assertEqual(log.warnings[0][0][1:], (expected, window))
            self.assertFalse(hasattr(owner, "_draft_window_runtime"))


if __name__ == "__main__":
    unittest.main()
