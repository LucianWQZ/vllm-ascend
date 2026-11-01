# SPDX-License-Identifier: Apache-2.0
"""Host-only native attention inventory; no framework or device imports."""

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
MODULE = ROOT / "vllm_ascend/worker/v2/spec_decode/draft_window_native_attention.py"
spec = importlib.util.spec_from_file_location("_native_attention_host_test", MODULE)
native = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = native
spec.loader.exec_module(native)

Dense = type("AscendAttentionBackendImpl", (), {})
Dense.__module__ = "vllm_ascend.attention.attention_v1"


def config(*, kinds=None, window=2048, architecture="DFlash2DraftModel", count=2, dflash=None):
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
    def test_dflash_implementation_defaults_and_override(self):
        self.assertEqual(native.configured_native_windows(config()), (None, None))
        self.assertEqual(
            native.configured_native_windows(config(dflash={"use_swa": True, "swa_window_size": 4096})),
            (4096, 4096),
        )
        self.assertEqual(
            native.configured_native_windows(
                config(kinds=["full_attention", "full_attention"], dflash={"use_swa": True})
            ),
            (2048, 2048),
        )
        self.assertIsNone(native.configured_native_windows(config(architecture="UnknownDraft")))

    def test_qwen3_raw_defaults_match_normalization_before_model(self):
        raw = config(count=30)
        raw["model_type"] = "qwen3"
        self.assertEqual(native.configured_native_windows(raw, raw_config=True), (None,) * 30)
        raw["use_sliding_window"] = True
        self.assertEqual(native.configured_native_windows(raw, raw_config=True), (None,) * 28 + (2048,) * 2)
        raw["max_window_layers"] = 0
        raw.pop("sliding_window")
        self.assertEqual(native.configured_native_windows(raw, raw_config=True), (4096,) * 30)

    def test_legacy_dspark_wrapper_uses_nested_layer_declarations(self):
        # The deployed GLM DSpark config has this training-format wrapper;
        # core544 selects the Qwen3DSparkModel loader from the nested Qwen3 config.
        for kinds, windows, expected in (
            (["full_attention"] * 2, [None, None], "FULL"),
            (["sliding_attention"] * 2, [2048, 2048], "SWA"),
            (["full_attention", "sliding_attention"], [None, 2048], "MIXED"),
        ):
            nested = config(kinds=kinds, architecture="Qwen3DSparkModel")
            nested.pop("architectures")
            nested.update(model_type="qwen3", use_sliding_window=expected != "FULL")
            raw = dict(
                architectures=["DSparkDraftModel"], speculators_model_type="dspark", transformer_layer_config=nested
            )
            normalized = dict(nested, architectures=["Qwen3DSparkModel"])
            self.assertEqual(native.configured_native_windows(raw, raw_config=True), tuple(windows))
            result = native.inspect_draft_native_attention(loaded(normalized, windows), raw_config=raw)
            self.assertEqual(result.kind, expected)
            self.assertTrue(result.actual_complete)

    def test_legacy_dspark_unknown_wrapper_or_missing_layer_config_stays_unknown(self):
        raw = dict(
            architectures=["DSparkDraftModel"],
            speculators_model_type="dspark",
            transformer_layer_config=dict(
                model_type="qwen3",
                num_hidden_layers=2,
                layer_types=["full_attention"] * 2,
                use_sliding_window=False,
                sliding_window=None,
            ),
        )
        damaged = []
        for key, value in (
            ("architectures", ["UnknownDSparkDraft"]),
            ("speculators_model_type", "unknown"),
            ("transformer_layer_config", None),
        ):
            row = copy.deepcopy(raw)
            row[key] = value
            damaged.append(row)
        for key, value in (("model_type", "unknown"), ("num_hidden_layers", None)):
            row = copy.deepcopy(raw)
            row["transformer_layer_config"][key] = value
            damaged.append(row)
        for row in damaged:
            with self.subTest(row=row):
                self.assertIsNone(native.configured_native_windows(row, raw_config=True))

    def test_legacy_dspark_outer_conflicts_and_actual_disagreement_rejected(self):
        nested = dict(
            model_type="qwen3",
            num_hidden_layers=2,
            layer_types=["sliding_attention"] * 2,
            use_sliding_window=True,
            sliding_window=2048,
        )
        raw = dict(architectures=["DSparkDraftModel"], speculators_model_type="dspark", transformer_layer_config=nested)
        for key, value in (
            ("model_type", "other"),
            ("num_hidden_layers", 3),
            ("layer_types", ["full_attention"] * 2),
            ("use_sliding_window", False),
            ("sliding_window", 1024),
        ):
            row = copy.deepcopy(raw)
            row[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "wrapper.*disagree"):
                native.configured_native_windows(row, raw_config=True)
        normalized = dict(nested, architectures=["Qwen3DSparkModel"])
        # Matching duplicate fields are harmless; conflicting loaded readers are not.
        raw["num_hidden_layers"] = 2
        self.assertEqual(native.configured_native_windows(raw, raw_config=True), (2048, 2048))
        result = native.inspect_draft_native_attention(loaded(normalized, [None, None]), raw_config=raw)
        self.assertEqual(result.kind, "UNKNOWN")
        self.assertIn("actual Draft consumers and configured native attention inventory disagree", result.issues)

    def test_designated_llama_eagle3_actual_defaults(self):
        cfg = config(architecture="LlamaForCausalLMEagle3", count=1)
        cfg["model_type"] = "llama"
        # LlamaAttention does not infer SWA from this scalar or a Qwen flag.
        cfg["use_sliding_window"] = True
        self.assertEqual(native.configured_native_windows(cfg, raw_config=True), (None,))
        self.assertEqual(native.inspect_draft_native_attention(loaded(cfg, [None], causal=True)).kind, "FULL")
        cfg["layer_types"] = ["sliding_attention"]
        self.assertEqual(native.configured_native_windows(cfg), (2048,))
        cfg["architectures"] = ["OtherLlamaDraft"]
        self.assertIsNone(native.configured_native_windows(cfg))

    def test_owner_startup_helper_reads_only_local_draft_once(self):
        cfg = config(kinds=["sliding_attention"] * 2)
        cfg["model_type"] = "qwen3"
        cfg["use_sliding_window"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(cfg))
            owner = loaded(cfg, [2048, 2048])
            owner.draft_model_config.model = directory
            owner.speculative_config = SimpleNamespace(enforce_eager=True)
            owner.draft_window_policy = SimpleNamespace(effective_window_size=1024)
            log = Log()
            self.assertEqual(native.report_owner_draft_native_attention(owner, log).kind, "SWA")
            path.write_text("not JSON: a second startup call must use the original cached input")
            self.assertEqual(native.report_owner_draft_native_attention(owner, log).kind, "SWA")
            self.assertEqual(len(log.info), 1)
            self.assertEqual(len(log.warnings), 1)
            self.assertFalse(hasattr(owner, "_draft_window_runtime"))
            self.assertFalse(hasattr(owner, "_draft_window_runtime_state"))

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

    def test_invalid_config_does_not_become_full(self):
        for cfg in (
            config(kinds=["sliding_attention"]),
            config(kinds=["sliding_attention"] * 2, window=0),
            config(kinds=["other"] * 2),
        ):
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                native.configured_native_windows(cfg)

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

    def test_actual_layer_swap_and_raw_normalized_conflict_rejected(self):
        cfg = config(kinds=["full_attention", "sliding_attention"])
        wrong = native.inspect_draft_native_attention(loaded(cfg, [2048, None]))
        self.assertEqual(wrong.kind, "UNKNOWN")
        right = loaded(cfg, [None, 2048])
        conflict = native.inspect_draft_native_attention(right, raw_config=config())
        self.assertEqual(conflict.kind, "UNKNOWN")
        self.assertIn("raw and normalized Draft attention declarations disagree", conflict.issues)
        unknown_raw = native.inspect_draft_native_attention(right, raw_config=config(architecture="UnknownDraft"))
        self.assertEqual(unknown_raw.kind, "UNKNOWN")

    def test_missing_consumer_mask_and_unknown_implementation_rejected(self):
        cfg = config(kinds=["sliding_attention"] * 2)
        for owner in (loaded(cfg, [2048]), loaded(cfg, [2048, 2048], mask=False)):
            self.assertEqual(native.inspect_draft_native_attention(owner).kind, "UNKNOWN")
        owner = loaded(cfg, [2048, 2048])
        next(iter(owner.attn_vllm_config.compilation_config.static_forward_context.values())).impl = object()
        self.assertEqual(native.inspect_draft_native_attention(owner).kind, "UNKNOWN")

    def test_provider_native_window_must_match_actual_consumer(self):
        owner = loaded(config(kinds=["sliding_attention"] * 2), [2048, 2048])
        name = owner.attn_groups[0][0].layer_names[0]
        good = SimpleNamespace(layers=(name,), native_window=2048)
        bad = SimpleNamespace(layers=(name,), native_window=4096)
        self.assertEqual(native.inspect_draft_native_attention(owner, provider_routes=[good]).kind, "SWA")
        self.assertEqual(native.inspect_draft_native_attention(owner, provider_routes=[bad]).kind, "UNKNOWN")

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
        for window, warning in ((None, False), (1024, True), (2048, False), (4096, False)):
            owner = loaded(config(kinds=["sliding_attention"] * 2), [2048, 2048])
            log = Log()
            for _ in range(2):
                result = native.report_draft_native_attention(owner, logger=log, window_size=window)
            self.assertEqual(result.kind, "SWA")
            self.assertEqual(len(log.info), 1)
            self.assertEqual(len(log.warnings), int(warning))
            if warning:
                self.assertEqual(log.warnings[0][0][1:], ((2048,), 1024))
            self.assertFalse(hasattr(owner, "_draft_window_runtime"))

    def test_runtime_existing_certified_provider_warning_is_preserved(self):
        owner = SimpleNamespace()
        log = Log()
        result = native.report_draft_native_attention(
            owner, logger=log, window_size=1024, certified_native_windows={0: 2048}
        )
        self.assertEqual(result.kind, "UNKNOWN")
        self.assertEqual(len(log.warnings), 1)
        self.assertFalse(result.actual_complete)


if __name__ == "__main__":
    unittest.main()
