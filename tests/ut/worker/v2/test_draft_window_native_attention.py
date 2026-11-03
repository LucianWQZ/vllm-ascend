# SPDX-License-Identifier: Apache-2.0
"""CPU startup inventory of real attention implementations, without device work."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
from vllm_ascend.attention.mla_v1 import AscendMLAImpl
from vllm_ascend.worker.v2.spec_decode import draft_window_native_attention as native


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
        impl = object.__new__(AscendAttentionBackendImpl)
        impl.sliding_window = window
        if mask:
            impl._dflash2_native_swa_mask = object()
        context[name] = SimpleNamespace(impl=impl)
        names.append(name)
    groups = [SimpleNamespace(layer_names=[names[index]]) for index in (order or range(len(names)))]
    owner = SimpleNamespace(
        draft_model_config=SimpleNamespace(hf_config=SimpleNamespace(**configuration)),
        attn_vllm_config=SimpleNamespace(compilation_config=SimpleNamespace(static_forward_context=context)),
        attn_groups=[groups],
        draft_attn_layer_names=set(names),
        _group_causal=causal,
    )
    owner.vllm_config = owner.attn_vllm_config
    return owner


def report(owner, logger):
    with patch.object(native, "logger", logger):
        native.report_owner_draft_native_attention(owner)


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
                report(owner, log)
                self.assertEqual(log.info[0][0][1], "SWA")
        self.assertEqual(len(log.info), 1)
        self.assertEqual(len(log.warnings), 1)
        self.assertFalse(hasattr(owner, "_draft_window_runtime"))

    def test_unknown_architecture_or_missing_count_does_not_become_full(self):
        for cfg in (config(architecture="UnknownDraft"), {}, config(count=0)):
            owner = loaded(cfg, [None, None])
            owner.speculative_config = SimpleNamespace(enforce_eager=True)
            owner.draft_window_policy = SimpleNamespace(effective_window_size=1024)
            self.assertEqual(native.inspect_draft_native_attention(owner)[0], "UNKNOWN")
            with self.assertRaises(NotImplementedError, msg=str(cfg)):
                report(owner, Log())

    def test_actual_window_and_complete_layer_registration_are_required(self):
        for windows in ([0, 2048], [True, 2048], [2048], [None, None, None]):
            with self.subTest(windows=windows):
                result = native.inspect_draft_native_attention(loaded(config(), windows))
                self.assertEqual(result[0], "UNKNOWN")
        owner = loaded(config(), [None, None])
        owner.attn_groups[0].append(owner.attn_groups[0][0])
        self.assertEqual(native.inspect_draft_native_attention(owner)[0], "UNKNOWN")

    def test_eagle_uses_its_actual_loaded_attention_not_qwen_flags(self):
        cfg = config(architecture="LlamaForCausalLMEagle3", count=1)
        cfg.update(model_type="llama", use_sliding_window=True)
        self.assertEqual(native.inspect_draft_native_attention(loaded(cfg, [None], causal=True))[0], "FULL")
        self.assertEqual(native.inspect_draft_native_attention(loaded(cfg, [2048], causal=True))[0], "SWA")

    def test_real_plain_mla_is_full_without_a_native_window_operand(self):
        owner = loaded(config(architecture="Glm5DSparkForCausalLM"), [None, None])
        for layer in owner.attn_vllm_config.compilation_config.static_forward_context.values():
            layer.impl = object.__new__(AscendMLAImpl)
        self.assertEqual(native.inspect_draft_native_attention(owner)[0], "FULL")

    def test_startup_graph_request_uses_draft_flag_not_target_mode(self):
        owner = loaded(config(), [None, None])
        owner.speculative_config = SimpleNamespace(enforce_eager=True)
        owner.attn_vllm_config.compilation_config.cudagraph_mode = SimpleNamespace(name="FULL")
        log = Log()
        report(owner, log)
        self.assertEqual(log.info[0][0][4], False)
        owner.speculative_config.enforce_eager = False
        report(owner, log)
        self.assertEqual(log.info[-1][0][4], True)

    def test_startup_inventory_preserves_baseline_off_unknown_graph(self):
        owner = loaded(config(architecture="UnclassifiedMTPDraft"), [None, None])
        owner.speculative_config = SimpleNamespace(method="mtp", enforce_eager=False)
        owner.query_cudagraph_manager = SimpleNamespace(cudagraph_mode=SimpleNamespace(name="FULL"))
        log = Log()
        report(owner, log)
        self.assertEqual(log.info[0][0][1], "UNKNOWN")
        self.assertEqual(log.info[0][0][4], True)
        owner.draft_window_policy = SimpleNamespace(effective_window_size=1024)
        with self.assertRaises(NotImplementedError):
            report(owner, log)

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
            self.assertEqual(result[0], expected)
            self.assertNotEqual(result[0], "UNKNOWN")

    def test_group_must_match_the_actual_loader_draft_layer_registration(self):
        owner = loaded(config(), [None, None])
        context = owner.attn_vllm_config.compilation_config.static_forward_context
        old = owner.attn_groups[0][1].layer_names[0]
        new = old.replace("layers.65.", "layers.66.")
        context[new] = context.pop(old)
        owner.attn_groups[0][1].layer_names = [new]
        self.assertEqual(native.inspect_draft_native_attention(owner)[0], "UNKNOWN")

    def test_missing_consumer_mask_and_unknown_implementation_rejected(self):
        cfg = config(kinds=["sliding_attention"] * 2)
        for owner in (loaded(cfg, [2048]), loaded(cfg, [2048, 2048], mask=False)):
            self.assertEqual(native.inspect_draft_native_attention(owner)[0], "UNKNOWN")
        owner = loaded(cfg, [2048, 2048])
        next(iter(owner.attn_vllm_config.compilation_config.static_forward_context.values())).impl = object()
        self.assertEqual(native.inspect_draft_native_attention(owner)[0], "UNKNOWN")
        # A same-named class does not acquire the actual implementation's contract.
        impostor = type(AscendAttentionBackendImpl.__name__, (), {"sliding_window": None})
        impostor.__module__ = AscendAttentionBackendImpl.__module__
        next(iter(owner.attn_vllm_config.compilation_config.static_forward_context.values())).impl = impostor()
        self.assertEqual(native.inspect_draft_native_attention(owner)[0], "UNKNOWN")

    def test_off_no_runtime_and_unknown_window_graph_fail_closed(self):
        owner = loaded(config(), [2048, 2048], mask=False)
        owner.speculative_config = SimpleNamespace(enforce_eager=True)
        log = Log()
        report(owner, log)
        self.assertEqual(log.info[0][0][1], "UNKNOWN")
        for window, eager in ((1024, True), (None, False)):
            owner.draft_window_policy = SimpleNamespace(effective_window_size=window)
            owner.speculative_config.enforce_eager = eager
            with self.assertRaises(NotImplementedError):
                report(owner, log)
        self.assertFalse(log.warnings)  # Partial native windows cannot certify narrowing.
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
                report(owner, log)
            self.assertEqual(log.info[0][0][1], "SWA")
            self.assertEqual(len(log.info), 1)
            self.assertEqual(len(log.warnings), bool(expected))
            if expected:
                self.assertEqual(log.warnings[0][0][1:], (expected, window))
            self.assertFalse(hasattr(owner, "_draft_window_runtime"))


if __name__ == "__main__":
    unittest.main()
