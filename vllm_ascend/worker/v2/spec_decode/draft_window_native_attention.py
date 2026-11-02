# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Host-only startup inventory of the Draft model's native attention.

KV retention is not an attention contract. In particular, DFlash2 keeps Full
KV backing while its actual consumers retain their native sliding windows.
This inventory neither creates a Window owner nor enables a read provider.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

_MISSING = object()
_DRAFT_ARCHITECTURES = frozenset(
    {
        "DFlashDraftModel",
        "DFlash2DraftModel",
        "Qwen3DSparkModel",
        "Qwen3OmniDSparkModel",
        "LlamaForCausalLMEagle3",
        "Glm5DSparkForCausalLM",
    }
)
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


@dataclass(frozen=True)
class DraftNativeAttention:
    kind: str
    windows: tuple[int, ...]
    issues: tuple[str, ...]

    @property
    def actual_complete(self) -> bool:
        return self.kind != "UNKNOWN"

    def narrowed_windows(self, window_size: int | None) -> tuple[int, ...]:
        # Only observed/certified native consumers establish this warning.
        if window_size is None or not self.actual_complete:
            return ()
        return tuple(sorted({value for value in self.windows if window_size < value}))


def inspect_draft_native_attention(speculator: Any) -> DraftNativeAttention:
    """Inspect loaded Draft consumers, after the real loader and ``set_attn``.

    Normalized architecture/layer count bounds the inventory. Native windows
    come from the executing implementations, never a KV storage spec or a
    second interpretation of the original model config/defaults.
    """
    normalized = _get(_get(speculator, "draft_model_config"), "hf_config")
    architectures: Sequence[str] = _get(normalized, "architectures", ()) or ()
    count = _get(normalized, "num_hidden_layers")
    issues = []
    known = len(architectures) == 1 and architectures[0] in _DRAFT_ARCHITECTURES
    if not known or type(count) is not int or count <= 0:
        issues.append("normalized Draft architecture or layer count has no supported implementation contract")
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
    # The loader already identifies the Draft layer set, including any Target
    # layer offset. Use that registration instead of parsing layer-name strings.
    declared = _get(speculator, "draft_attn_layer_names", ()) or ()
    complete = bool(layers) and known and type(count) is int and len(layers) == count
    if not complete or set(layers) != set(declared):
        issues.append("complete supported Draft architecture and actual consumer inventory are required")
    values = tuple(layers.values())
    kind = _kind(values) if complete and not issues else "UNKNOWN"
    return DraftNativeAttention(
        kind=kind,
        windows=tuple(sorted({value for value in values if value is not None})),
        issues=tuple(dict.fromkeys(issues)),
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


def report_owner_draft_native_attention(owner: Any, logger: Any) -> DraftNativeAttention:
    """Startup-only summary of the actual speculator's loaded attention.

    The fixed loader owns original/normalized configuration. No config file,
    runtime, metadata tensor or completion object is read or allocated here.
    """
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
    inventory = inspect_draft_native_attention(owner)
    logger.info_once(
        "Draft native attention=%s; native windows=%s; extra Draft window=%s; "
        "Draft Graph requested=%s; complete actual inventory=%s; issues=%s. "
        "Full KV retention does not imply Full attention.",
        inventory.kind,
        inventory.windows,
        window,
        graph_requested,
        inventory.actual_complete,
        inventory.issues,
        scope="global",
    )
    narrowed = inventory.narrowed_windows(window)
    if narrowed:
        logger.warning_once(
            "The Draft model contains native SWA layers with windows %s. "
            "The additional draft_window_size=%s narrows those layers and may reduce "
            "acceptance or throughput. Prefer the native window by default and "
            "enable an additional window only after measuring your workload.",
            narrowed,
            window,
            scope="global",
        )
    # Detection must not turn an original Window-OFF Graph route (for example
    # built-in MTP) into a newly rejected configuration. Additional Window
    # owners always need the contract; DFlash2 Graph is itself new admission
    # in this change and needs it even when the additional Window is OFF.
    architectures = _get(_get(_get(owner, "draft_model_config"), "hf_config"), "architectures", ()) or ()
    new_dflash2_graph = graph_requested and tuple(architectures) == ("DFlash2DraftModel",)
    require_known_draft_native_attention(inventory, window_size=window, graph_requested=new_dflash2_graph)
    return inventory
