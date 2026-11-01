# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Host-only startup inventory of the Draft model's native attention.

KV retention is not an attention contract. In particular, DFlash2 keeps Full
KV backing while its actual consumers retain their native sliding windows.
This inventory neither creates a Window owner nor enables a read provider.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import regex as re

_MISSING = object()
_MAX_DRAFT_CONFIG_BYTES = 1024 * 1024
_EAGLE_ARCHITECTURE = "LlamaForCausalLMEagle3"
_DFLASH_ARCHITECTURES = frozenset({"DFlashDraftModel", "DFlash2DraftModel", "Qwen3DSparkModel", "Qwen3OmniDSparkModel"})
_DENSE_IMPLEMENTATIONS = frozenset(
    {
        "vllm_ascend.attention.attention_v1.AscendAttentionBackendImpl",
        "vllm_ascend.attention.mla_v1.AscendMLAImpl",
    }
)


def _get(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def _window(value: Any) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value <= 0:
        raise ValueError("native sliding_window must be a positive integer or None")
    return value


def _kind(windows: Sequence[int | None]) -> str:
    if not windows:
        return "UNKNOWN"
    if all(window is None for window in windows):
        return "FULL"
    return "SWA" if all(window is not None for window in windows) else "MIXED"


def _layer_windows(kinds: Any, count: int, window: Any, *, use_swa: bool = False) -> tuple[int | None, ...]:
    if kinds is not None and (
        not isinstance(kinds, (list, tuple))
        or len(kinds) != count
        or any(kind not in ("full_attention", "sliding_attention") for kind in kinds)
    ):
        raise ValueError("native layer_types must describe every Draft attention layer")
    if kinds is None or (use_swa and "sliding_attention" not in kinds):
        kinds = ["sliding_attention" if use_swa else "full_attention"] * count
    windows = tuple(_window(window) if kind == "sliding_attention" else None for kind in kinds)
    if any(kind == "sliding_attention" and value is None for kind, value in zip(kinds, windows)):
        raise ValueError("native sliding attention has no positive window")
    return windows


def configured_native_windows(config: Any, *, raw_config: bool = False) -> tuple[int | None, ...] | None:
    """Mirror the supported model implementations, not generic HF defaults.

    DFlash/DSpark's Qwen backbone resolves dflash_config.use_swa and
    swa_window_size before the per-layer declarations. A bare sliding_window
    field does not enable SWA. The named GLM MLA architecture is explicitly
    dense. Training-format DSpark wrappers use their own Qwen3 layer config.
    Unknown architectures deliberately have no implicit Full default.
    """
    if config is None:
        return None
    architectures = set(_get(config, "architectures", ()) or ())
    if raw_config and architectures == {"DSparkDraftModel"}:
        # Fixed core544 SpeculatorsConfig.build_vllm_speculative_config takes
        # transformer_layer_config, then update_dspark selects Qwen3DSparkModel
        # for this legacy training name. The outer wrapper is not a layer config.
        nested = _get(config, "transformer_layer_config")
        if (
            _get(config, "speculators_model_type") != "dspark"
            or not isinstance(nested, Mapping)
            or nested.get("model_type") != "qwen3"
        ):
            return None
        # The loader reads these declarations from the nested config only.
        # Do not silently accept contradictory or misplaced outer declarations.
        for key in (
            "model_type",
            "num_hidden_layers",
            "layer_types",
            "sliding_window",
            "use_sliding_window",
            "max_window_layers",
            "dflash_config",
        ):
            outer = _get(config, key, _MISSING)
            if outer is not _MISSING and outer != nested.get(key, _MISSING):
                raise ValueError(f"DSpark wrapper and transformer_layer_config disagree on {key}")
        config = dict(nested, architectures=["Qwen3DSparkModel"])
        architectures = {"Qwen3DSparkModel"}
    count = _get(config, "num_hidden_layers")
    if type(count) is not int or count <= 0:
        return None
    if architectures == {_EAGLE_ARCHITECTURE}:
        # The exercised checkpoint uses llama_eagle3.LlamaDecoderLayer, which
        # delegates to LlamaAttention. That implementation defaults to Full
        # and only reads sliding_window for a declared sliding_attention row.
        # Do not apply DFlash use_swa or Qwen normalization to this architecture.
        return _layer_windows(_get(config, "layer_types") or None, count, _get(config, "sliding_window"))
    if architectures == {"Glm5DSparkForCausalLM"}:
        return (None,) * count
    if len(architectures) != 1 or not architectures.issubset(_DFLASH_ARCHITECTURES):
        return None
    layer_types = _get(config, "layer_types")
    configured_window = _get(config, "sliding_window")
    if raw_config and _get(config, "model_type") == "qwen3":
        # Qwen3Config.__post_init__: these are the installed implementation's
        # defaults, not a general inference from a field named sliding_window.
        use_sliding = _get(config, "use_sliding_window", False)
        if type(use_sliding) is not bool:
            raise ValueError("Qwen3 use_sliding_window must be boolean")
        configured_window = _get(config, "sliding_window", 4096) if use_sliding else None
        if layer_types is None:
            first_sliding = _get(config, "max_window_layers", 28)
            if type(first_sliding) is not int or first_sliding < 0:
                raise ValueError("Qwen3 max_window_layers must be a nonnegative integer")
            layer_types = [
                "sliding_attention" if configured_window is not None and index >= first_sliding else "full_attention"
                for index in range(count)
            ]
    dflash = _get(config, "dflash_config", {}) or {}
    use_swa = _get(dflash, "use_swa", False)
    if type(use_swa) is not bool:
        raise ValueError("dflash_config.use_swa must be boolean")
    return _layer_windows(layer_types, count, _get(dflash, "swa_window_size", configured_window), use_swa=use_swa)


@dataclass(frozen=True)
class DraftNativeAttention:
    kind: str
    windows: tuple[int, ...]
    actual_layers: tuple[tuple[str, int | None], ...]
    sources: tuple[str, ...]
    issues: tuple[str, ...]
    actual_complete: bool

    def narrowed_windows(self, window_size: int | None) -> tuple[int, ...]:
        # Only observed/certified native consumers establish this warning.
        if window_size is None or not self.actual_complete:
            return ()
        return tuple(sorted({value for _, value in self.actual_layers if value is not None and window_size < value}))


def inspect_draft_native_attention(
    speculator: Any,
    *,
    raw_config: Any = None,
    provider_routes: Sequence[Any] = (),
) -> DraftNativeAttention:
    """Read actual Draft consumer metadata without querying a device/tensor.

    Call after the actual Draft attention groups and static context exist.
    raw_config, when available, must be the original Draft config, never the
    Target config. Incomplete/contradictory evidence stays UNKNOWN.
    """
    model = _get(speculator, "draft_model_config")
    normalized = _get(model, "hf_config")
    sources = []
    issues = []
    declarations = []
    for label, config in (("raw Draft config", raw_config), ("normalized Draft config", normalized)):
        if config is None:
            continue
        try:
            windows = configured_native_windows(config, raw_config=label == "raw Draft config")
        except ValueError as error:
            issues.append(f"{label}: {error}")
            continue
        if windows is not None:
            declarations.append(windows)
            sources.append(label)
        else:
            issues.append(f"{label} has an unknown native attention implementation/default contract")
    if len(declarations) == 2 and declarations[0] != declarations[1]:
        issues.append("raw and normalized Draft attention declarations disagree")
    config_windows = declarations[-1] if declarations else None
    config = _get(speculator, "attn_vllm_config")
    context = _get(_get(config, "compilation_config"), "static_forward_context", {}) or {}
    groups = _get(speculator, "attn_groups", ()) or ()
    layers: dict[str, int | None] = {}
    for gid, group_set in enumerate(groups):
        for group in group_set:
            for name in _get(group, "layer_names", ()):
                if name in layers:
                    issues.append(f"Draft consumer {name} is registered more than once")
                    continue
                impl = _get(context.get(name), "impl")
                identity = f"{type(impl).__module__}.{type(impl).__qualname__}"
                if identity not in _DENSE_IMPLEMENTATIONS:
                    issues.append(f"Draft consumer {name} has unknown native attention implementation {identity}")
                    continue
                value = _get(impl, "sliding_window", _MISSING)
                if identity.endswith(".AscendMLAImpl"):
                    # This dense implementation has no native window operand.
                    value = None
                if value is _MISSING:
                    issues.append(f"Draft consumer {name} has no native window descriptor")
                    continue
                try:
                    layers[name] = _window(value)
                    causal = _get(speculator, "_group_causal", _MISSING)
                    causal = causal.get(gid, _MISSING) if isinstance(causal, Mapping) else causal
                    if layers[name] is not None and type(causal) is not bool:
                        issues.append(f"Draft consumer {name} has no actual native causality descriptor")
                    if (
                        layers[name] is not None
                        and causal is False
                        and (_get(impl, "_dflash2_native_swa_mask") is None)
                    ):
                        issues.append(f"noncausal Draft consumer {name} has no native-SWA mask binding")
                except ValueError as error:
                    issues.append(f"Draft consumer {name}: {error}")
    if layers:
        sources.append("actual loaded Draft attention implementations")
    for route in provider_routes:
        native = _get(route, "native_window")
        if native is None:
            continue
        try:
            native = _window(native)
        except ValueError as error:
            issues.append(f"Draft provider: {error}")
            continue
        for name in _get(route, "layers", _get(route, "layer_names", ())):
            if name not in layers or layers[name] != native:
                issues.append(f"Draft provider and actual consumer {name} disagree on native window")
        sources.append("actual Draft provider native-window descriptor")
    # Actual layer names may carry a global Target-layer offset. Match the
    # complete contiguous Draft range in model order, not KV-group order.
    indexed = []
    for name, window in layers.items():
        match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
        if match is None:
            issues.append(f"Draft consumer {name} has no model-layer index")
        else:
            indexed.append((int(match.group(1)), name, window))
    indexed.sort()
    actual = tuple((name, window) for _, name, window in indexed)
    complete = bool(actual) and config_windows is not None and len(actual) == len(config_windows)
    if complete:
        first = indexed[0][0]
        if tuple(index for index, _, _ in indexed) != tuple(range(first, first + len(indexed))):
            issues.append("actual Draft model-layer indices are not a complete contiguous range")
        if tuple(window for _, window in actual) != config_windows:
            issues.append("actual Draft consumers and configured native attention inventory disagree")
    if not complete:
        issues.append("complete supported Draft config and actual consumer inventory are required")
    values = tuple(value for _, value in actual)
    kind = _kind(values) if complete and not issues else "UNKNOWN"
    return DraftNativeAttention(
        kind=kind,
        windows=tuple(sorted({value for value in values if value is not None})),
        actual_layers=actual,
        sources=tuple(dict.fromkeys(sources)),
        issues=tuple(dict.fromkeys(issues)),
        actual_complete=complete and not issues,
    )


def require_known_draft_native_attention(
    inventory: DraftNativeAttention, *, window_size: int | None, graph_requested: bool
) -> None:
    """A detection prerequisite, not a provider/Graph support certificate.

    Ordinary OFF eager remains legal and allocates no Window state. Existing
    method/provider guards must still admit every known Window/Graph route.
    """
    if window_size is None and not graph_requested:
        return
    if not inventory.actual_complete or inventory.kind == "UNKNOWN":
        raise NotImplementedError(
            "Draft Window/Graph requires a known native attention contract: " + "; ".join(inventory.issues)
        )


def report_draft_native_attention(
    speculator: Any,
    *,
    logger: Any,
    window_size: int | None,
    graph_requested: bool = False,
    raw_config: Any = None,
    provider_routes: Sequence[Any] = (),
    certified_native_windows: Mapping[int, int] | None = None,
) -> DraftNativeAttention:
    """Emit a once-per-loaded-model summary, without installing an owner."""
    inventory = inspect_draft_native_attention(speculator, raw_config=raw_config, provider_routes=provider_routes)
    logger.info_once(
        "Draft native attention=%s; native windows=%s; extra Draft window=%s; "
        "Draft Graph requested=%s; complete actual inventory=%s; issues=%s. "
        "Full KV retention does not imply Full attention.",
        inventory.kind,
        inventory.windows,
        window_size,
        graph_requested,
        inventory.actual_complete,
        inventory.issues,
        scope="global",
    )
    native_windows = set(inventory.narrowed_windows(window_size))
    # The existing typed eager provider has already certified these native
    # consumers. Keep its warning even when a test/older model omits config.
    if window_size is not None:
        native_windows.update(value for value in (certified_native_windows or {}).values() if window_size < value)
    if native_windows:
        logger.warning_once(
            "The Draft model contains native SWA layers with windows %s. "
            "The additional draft_window_size=%s narrows those layers and may reduce "
            "acceptance or throughput. Prefer the native window by default and "
            "enable an additional window only after measuring your workload.",
            tuple(sorted(native_windows)),
            window_size,
            scope="global",
        )
    return inventory


def report_owner_draft_native_attention(owner: Any, logger: Any) -> DraftNativeAttention:
    """Startup-only adapter for the actual speculator's ``set_attn`` seam.

    The optional original local Draft config is read at most once per owner;
    remote model IDs trigger no I/O/download. Read/parse failures become an
    UNKNOWN declaration and do not break ordinary Window-OFF eager. No runtime,
    provider, metadata tensor or device completion object is allocated here.
    """
    raw = _get(owner, "_draft_native_attention_raw_config", _MISSING)
    if raw is _MISSING:
        raw = None
        model_path = _get(_get(owner, "draft_model_config"), "model")
        if isinstance(model_path, (str, Path)):
            path = Path(model_path) / "config.json"
            try:
                if path.is_file():
                    if path.stat().st_size > _MAX_DRAFT_CONFIG_BYTES:
                        raise ValueError("original Draft config exceeds the startup size limit")
                    raw = json.loads(path.read_text(encoding="utf-8"))
                    if not isinstance(raw, dict):
                        raise ValueError("original Draft config must be an object")
            except (OSError, ValueError):
                # Supplying an explicit unknown declaration retains the failed
                # raw-source boundary in inspect(), rather than assuming Full.
                raw = {}
        owner._draft_native_attention_raw_config = raw
    policies = (_get(owner, "_draft_window_policy"), _get(owner, "draft_window_policy"))
    window = next((_get(policy, "effective_window_size") for policy in policies if policy is not None), None)
    speculative = _get(owner, "speculative_config")
    eager = (
        _get(speculative, "enforce_eager") is True or _get(_get(owner, "draft_model_config"), "enforce_eager") is True
    )
    managers = (
        _get(owner, "query_cudagraph_manager"),
        _get(owner, "prefill_cudagraph_manager"),
        _get(owner, "decode_cudagraph_manager"),
    )
    modes = tuple(_get(_get(manager, "cudagraph_mode"), "name") for manager in managers if manager is not None)
    # A real initialized manager is authoritative. Before manager creation this
    # is only the explicit Draft request, never the Target's graph selection.
    graph_requested = not eager and (any(mode != "NONE" for mode in modes) if modes else True)
    capability = _get(owner, "_draft_window_source_capability")
    routes = _get(_get(capability, "inventory"), "groups", ()) or ()
    runtime = _get(owner, "_draft_window_runtime")
    inventory = report_draft_native_attention(
        owner,
        logger=logger,
        window_size=window,
        graph_requested=graph_requested,
        raw_config=raw,
        provider_routes=routes,
        certified_native_windows=_get(runtime, "native_swa_intersections"),
    )
    # Detection must not turn an original Window-OFF Graph route (for example
    # built-in MTP) into a newly rejected configuration. Additional Window
    # owners always need the contract; DFlash2 Graph is itself new admission
    # in this change and needs it even when the additional Window is OFF.
    architectures = _get(_get(_get(owner, "draft_model_config"), "hf_config"), "architectures", ()) or ()
    new_dflash2_graph = graph_requested and tuple(architectures) == ("DFlash2DraftModel",)
    require_known_draft_native_attention(inventory, window_size=window, graph_requested=new_dflash2_graph)
    return inventory
