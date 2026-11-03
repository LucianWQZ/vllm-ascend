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


def inspect_draft_native_attention(speculator: Any) -> DraftNativeAttention:
    """Inspect loaded Draft consumers, after the real loader and ``set_attn``.

    Normalized architecture/layer count bounds the inventory. Native windows
    come from the executing implementations, never a KV storage spec or a
    second interpretation of the original model config/defaults.
    """
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
    from vllm_ascend.attention.mla_v1 import AscendMLAImpl

    normalized = speculator.draft_model_config.hf_config
    architectures: Sequence[str] = getattr(normalized, "architectures", ()) or ()
    count = getattr(normalized, "num_hidden_layers", None)
    issues = []
    known = len(architectures) == 1 and architectures[0] in _DRAFT_ARCHITECTURES
    if not known or type(count) is not int or count <= 0:
        issues.append("normalized Draft architecture or layer count has no supported implementation contract")
    context = speculator.attn_vllm_config.compilation_config.static_forward_context
    declared_causal = getattr(speculator, "_group_causal", _MISSING)
    layers: dict[str, int | None] = {}
    for gid, group_set in enumerate(speculator.attn_groups):
        causal = declared_causal.get(gid, _MISSING) if isinstance(declared_causal, Mapping) else declared_causal
        for group in group_set:
            for name in group.layer_names:
                if name in layers:
                    issues.append(f"Draft consumer {name} is registered more than once")
                    continue
                impl = getattr(context.get(name), "impl", None)
                if type(impl) not in (AscendAttentionBackendImpl, AscendMLAImpl):
                    identity = f"{type(impl).__module__}.{type(impl).__qualname__}"
                    issues.append(f"Draft consumer {name} has unknown native attention implementation {identity}")
                    continue
                value = getattr(impl, "sliding_window", _MISSING)
                if type(impl) is AscendMLAImpl:
                    # This dense implementation has no native window operand.
                    value = None
                if value is _MISSING:
                    issues.append(f"Draft consumer {name} has no native window descriptor")
                    continue
                try:
                    layers[name] = _window(value)
                    if layers[name] is not None and type(causal) is not bool:
                        issues.append(f"Draft consumer {name} has no actual native causality descriptor")
                    if (
                        layers[name] is not None
                        and causal is False
                        and (getattr(impl, "_dflash2_native_swa_mask", None) is None)
                    ):
                        issues.append(f"noncausal Draft consumer {name} has no native-SWA mask binding")
                except ValueError as error:
                    issues.append(f"Draft consumer {name}: {error}")
    # The loader already identifies the Draft layer set, including any Target
    # layer offset. Use that registration instead of parsing layer-name strings.
    declared = speculator.draft_attn_layer_names
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


def report_owner_draft_native_attention(owner: Any, logger: Any) -> DraftNativeAttention:
    """Startup-only summary of the actual speculator's loaded attention.

    The fixed loader owns original/normalized configuration. No config file,
    runtime, metadata tensor or completion object is read or allocated here.
    """
    policies = (getattr(owner, "_draft_window_policy", None), getattr(owner, "draft_window_policy", None))
    window = next((policy.effective_window_size for policy in policies if policy is not None), None)
    speculative = owner.speculative_config
    eager = speculative.enforce_eager is True or getattr(owner.draft_model_config, "enforce_eager", False) is True
    # All real callers run here from set_attn, before the fixed runner calls
    # init_cudagraph_manager. Report the Draft request, not Target Graph mode
    # or a hypothetical already-created manager.
    graph_requested = not eager
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
    # Only observed/certified native consumers establish this warning.
    narrowed = (
        tuple(value for value in inventory.windows if window < value)
        if window is not None and inventory.actual_complete
        else ()
    )
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
    architectures = getattr(owner.draft_model_config.hf_config, "architectures", ()) or ()
    new_dflash2_graph = graph_requested and tuple(architectures) == ("DFlash2DraftModel",)
    if (window is not None or new_dflash2_graph) and not inventory.actual_complete:
        raise NotImplementedError(
            "Draft Window/Graph requires a known native attention contract: " + "; ".join(inventory.issues)
        )
    return inventory
