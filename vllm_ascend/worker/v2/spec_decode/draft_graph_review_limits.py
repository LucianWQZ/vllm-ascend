# SPDX-License-Identifier: Apache-2.0
"""Startup support and runtime shape bounds for qualified Draft Graph consumers.

One frozen contract is shared by config, the speculator and actual request
checks. It is not a numerical PASS receipt. Window OFF only enters this
contract when the new experimental Graph owner is explicitly selected.
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import index
from typing import TypeGuard


@dataclass(frozen=True, slots=True)
class DraftGraphSupport:
    max_model_len: int
    static_k: int
    max_requests: int
    data_parallel_size: int
    method: str
    query_tokens: int


# Keep the qualified attention/projection, auxiliary and position geometry.
# Model norm/activation/vocabulary settings belong to the actual model loader,
# not a second Window allowlist. Actual dtype/capacity are checked below.
_COMMON_CONSUMER_CONFIG = {
    "attention_bias": False,
    "hidden_size": 2048,
}
_CONSUMER_CONFIGS = {
    "dflash_full8": {
        **_COMMON_CONSUMER_CONFIG,
        **{
            "architectures": ["DFlashDraftModel"],
            "block_size": 16,
            "dflash_config": {"mask_token_id": 248070, "target_layer_ids": [1, 10, 19, 28, 37]},
            "head_dim": 128,
            "layer_types": ["full_attention"] * 8,
            "model_type": "qwen3",
            "num_attention_heads": 32,
            "num_hidden_layers": 8,
            "num_key_value_heads": 4,
            "num_target_layers": 40,
            "rope_parameters": {
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "factor": 64.0,
                "original_max_position_embeddings": 4096,
                "rope_type": "yarn",
                "rope_theta": 10000000,
            },
            "sliding_window": None,
            "use_sliding_window": False,
        },
    },
    "eagle3": {
        **_COMMON_CONSUMER_CONFIG,
        **{
            "architectures": ["LlamaForCausalLMEagle3"],
            "eagle_config": {"eagle_aux_hidden_state_layer_ids": [3, 19, 35]},
            "head_dim": 256,
            "model_type": "llama",
            "num_attention_heads": 16,
            "num_hidden_layers": 1,
            "num_key_value_heads": 2,
            "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
            "use_sliding_window": False,
            "eagle_aux_hidden_state_layer_ids": [3, 19, 35],
        },
    },
}
_FULL8_QUERY_TOKENS = 8


def _consumer_profile(speculative):
    """Use the same normalized configuration that constructs the actual model."""
    hf = speculative.draft_model_config.hf_config
    model_type = getattr(hf, "model_type", None)
    if model_type == "eagle":
        from vllm.transformers_utils.configs.eagle import EAGLEConfig

        # Fixed vLLM wraps DFlash and Eagle configs before worker creation.
        # Keep checking the outer, effective projection/attention fields;
        # only the architecture's model type belongs to the wrapped model.
        if isinstance(hf, EAGLEConfig) and hf.model is not None:
            model_type = hf.model.model_type
    architectures = getattr(hf, "architectures", ()) or ()
    if architectures == ["DFlashDraftModel"] and model_type == "qwen3":
        if (
            getattr(hf, "layer_types", None) == ["sliding_attention"] * 5 + ["full_attention"]
            and getattr(hf, "sliding_window", None) == 4096
        ):
            return "dflash_swa6"
        # DFlashDraftModel is also the fixed GLM draft's upstream model class.
        # Its target-feature projection is not the Qwen Full8 consumer.
        target = getattr(speculative, "target_model_config", None)
        target_hf = getattr(target, "hf_text_config", None)
        if target is not None and target_hf is not None and getattr(target_hf, "model_type", None) == "glm_moe_dsa":
            dflash = getattr(hf, "dflash_config", None)
            aux = dflash.get("target_layer_ids", ()) if isinstance(dflash, dict) else ()
            layers = getattr(hf, "num_hidden_layers", None)
            if not (
                type(layers) is int
                and layers > 0
                and getattr(hf, "layer_types", None) == ["full_attention"] * layers
                and getattr(hf, "sliding_window", None) is None
                and getattr(hf, "use_sliding_window", False) is False
                and getattr(hf, "target_hidden_size", None) == target.get_hidden_size()
                and getattr(hf, "num_target_layers", None) == target_hf.num_hidden_layers
                and isinstance(aux, list)
                and aux
                and all(type(layer) is int and 0 <= layer < target_hf.num_hidden_layers for layer in aux)
                and len(set(aux)) == len(aux)
            ):
                raise NotImplementedError("GLM DFlash Graph requires its Full draft and matching Target feature layout")
            return "glm_dflash"
        profile = "dflash_full8"
    elif architectures == ["LlamaForCausalLMEagle3"]:
        profile = "eagle3"
    else:
        return None
    expected = _CONSUMER_CONFIGS[profile]
    for field, value in expected.items():
        actual = model_type if field == "model_type" else getattr(hf, field, None)
        if field == "rope_parameters" and isinstance(actual, dict):
            actual = dict(actual)
            actual.pop("type", None)  # Transformers can retain this legacy alias.
        if actual != value:
            raise NotImplementedError(f"Draft Graph consumer {field} differs from its qualified configuration")
    # These actual model overrides change projection/auxiliary/attention work;
    # none belongs to the qualified consumer. Packaging keys do not enter here.
    if any(
        getattr(hf, field, None) is not None
        for field in (
            "is_causal",
            "target_hidden_size",
            "num_aux_hidden_states",
            "norm_before_fc",
            "use_aux_hidden_state",
            "add_swa_attention_sink_bias",
            "quantization_config",
        )
    ) or (profile == "dflash_full8" and getattr(hf, "eagle_config", None)):
        raise NotImplementedError("Draft Graph consumer overrides have not been qualified")
    return profile


def full8_graph_review_max_requests(limit):
    return limit.max_requests if isinstance(limit, DraftGraphSupport) and limit.method == "dflash" else None


def _is_eagle_support(limit: object) -> TypeGuard[DraftGraphSupport]:
    return isinstance(limit, DraftGraphSupport) and limit.method == "eagle3"


def resolve_draft_graph_review_limit(vllm_config, *, experimental_graph_review=False):
    """Parse startup support without changing the upstream configuration.

    The speculator owns the returned frozen contract. VllmConfig must contain
    only declared fields because upstream DFlash loading copies it with replace.
    """
    speculative = vllm_config.speculative_config
    if (
        speculative is None
        or speculative.method not in ("dflash", "eagle3")
        or speculative.enforce_eager
        or experimental_graph_review is not True
    ):
        return None
    mode = vllm_config.compilation_config.cudagraph_mode
    decode = getattr(mode, "decode_mode", None)
    effective = decode() if callable(decode) else mode
    if getattr(effective, "name", effective) == "NONE":
        return None
    profile = _consumer_profile(speculative)
    if profile == "dflash_swa6":
        raise NotImplementedError(
            "This six-layer native-SWA DFlash Graph combination has an unresolved valid-output counterexample; "
            "use the independently supported Draft eager path. No fallback is selected."
        )
    scope = None
    if profile in ("dflash_full8", "eagle3"):
        scope = _validate_qualified_graph_config(vllm_config, profile)
    elif profile != "glm_dflash":
        hf = speculative.draft_model_config.hf_config
        # Familiar names do not open an unknown normalized consumer geometry.
        architectures = getattr(hf, "architectures", ()) or ()
        if (speculative.method == "eagle3" and "LlamaForCausalLMEagle3" in architectures) or (
            speculative.method == "dflash"
            and "DFlashDraftModel" in architectures
            and getattr(hf, "model_type", None) == "qwen3"
        ):
            raise NotImplementedError("This dense Draft Graph model configuration has not been qualified")
    return scope


def _validate_qualified_graph_config(vllm_config, profile):
    speculative = vllm_config.speculative_config
    parallel = vllm_config.parallel_config
    draft_parallel = speculative.draft_parallel_config
    compilation = vllm_config.compilation_config
    scheduler = vllm_config.scheduler_config
    additional = vllm_config.additional_config or {}
    eagle = profile == "eagle3"
    method, k, window, query = ("eagle3", 3, 1024, 4) if eagle else ("dflash", 7, 32768, 8)
    dp, rows = parallel.data_parallel_size, scheduler.max_num_seqs
    target_phase = (getattr(parallel, "all2all_backend", None), getattr(parallel, "use_sequence_parallel_moe", None))
    target_phase_valid = not eagle or target_phase in (
        ("allgather_reducescatter", True),
        ("flashinfer_all2allv", False),
    )
    captures = compilation.cudagraph_capture_sizes
    maximum = vllm_config.model_config.max_model_len
    if (
        speculative.method != method
        or str(getattr(speculative.draft_model_config, "dtype", "")) != "torch.bfloat16"
        or speculative.uses_dynamic_speculative_decoding()
        or speculative.num_speculative_tokens != k
        or speculative.enable_adaptive_verification is not False
        or type(dp) is not int
        or dp not in (1, 2)
        or type(rows) is not int
        or rows not in (1, 2)
        or (dp == 2 and rows != 1)
        or parallel.tensor_parallel_size != 4
        or any(
            getattr(parallel, field) != 1
            for field in ("pipeline_parallel_size", "prefill_context_parallel_size", "decode_context_parallel_size")
        )
        or parallel.enable_expert_parallel is not True
        or getattr(draft_parallel, "tensor_parallel_size", None) != 4
        or getattr(draft_parallel, "data_parallel_size", None) != 1
        or (
            eagle
            and (
                getattr(draft_parallel, "enable_expert_parallel", None) is not False
                or getattr(draft_parallel, "use_sequence_parallel_moe", None) is not False
            )
        )
        or not target_phase_valid
        or getattr(compilation.cudagraph_mode, "name", None) != "FULL_DECODE_ONLY"
        or additional.get("draft_window_size") not in (None, window)
        or (additional.get("draft_window_size") is not None and type(additional["draft_window_size"]) is not int)
        or additional.get("enable_dsa_cp", False) is not False
        or type(additional.get("enable_flashcomm1", False)) is not bool
        or type(scheduler.async_scheduling) is not bool
        or type(vllm_config.cache_config.enable_prefix_caching) is not bool
        or not captures
        or any(type(size) is not int or size <= 0 for size in captures)
        or not {query * count for count in range(1, rows + 1)}.issubset(captures)
        or (not eagle and captures != [query * count for count in range(1, rows + 1)])
        or type(compilation.max_cudagraph_capture_size) is not int
        or compilation.max_cudagraph_capture_size < query * rows
        or type(maximum) is not int
        or maximum <= k
    ):
        raise NotImplementedError(
            f"Qualified {method} Draft Graph requires TP4, DP1 local capacity 1/2 or DP2 local capacity 1, "
            f"static K{k}, Window OFF/W{window}, exact admitted padded rows and corresponding Q{query} buckets; "
            "larger/unknown shapes remain protected. No eager fallback is selected."
        )
    if not eagle:
        _full8_review_capacity(vllm_config)
    return DraftGraphSupport(maximum, k, rows, dp, method, query)


def eagle_dp2_graph_review_enabled(owner):
    """Read the actual owner's startup contract; never reparse during capture."""
    scope = getattr(owner, "_draft_graph_review_limit", None)
    return _is_eagle_support(scope) and scope.data_parallel_size == 2


def note_eagle_single_request_review_lifecycle(owner, scheduler_output):
    """Keep only authoritative preemption IDs until real request completion.

    Fixed MRV2 resumption is re-emitted as scheduled_new_reqs, including the case
    where no output token exists yet. prefill_len==prompt_len cannot detect
    that case. No scheduler, request state, or device object is modified.
    """
    limit = getattr(owner, "_draft_graph_review_limit", None)
    if not _is_eagle_support(limit) and full8_graph_review_max_requests(limit) is None:
        return
    finished = scheduler_output.finished_req_ids
    preempted = scheduler_output.preempted_req_ids or set()
    if (
        not isinstance(finished, set)
        or not isinstance(preempted, set)
        or any(not isinstance(req_id, str) or not req_id for req_id in finished | preempted)
    ):
        raise TypeError("Eagle review requires the actual scheduler finished/preempted request ID sets")
    blocked = getattr(owner, "_draft_graph_review_preempted_request_ids", None)
    if blocked is None:
        blocked = set()
        owner._draft_graph_review_preempted_request_ids = blocked
    blocked.update(preempted)
    blocked.difference_update(finished)


def draft_graph_request_bounds(owner, req_states, req_ids, idx_mapping_np):
    """Borrow the runner's row order; copy only CPU limits absent from InputBatch."""
    if getattr(owner, "_draft_graph_review_limit", None) is None:
        return None
    bounds = []
    blocked = getattr(owner, "_draft_graph_review_preempted_request_ids", ())
    for req_id, raw_index in zip(req_ids, idx_mapping_np, strict=True):
        if req_id in blocked:
            raise NotImplementedError("Draft Graph does not qualify resumption after preemption")
        row = int(raw_index)
        if req_states.req_id_to_index.get(req_id) != row:
            raise RuntimeError("Draft Graph request mapping changed during input preparation")
        prompt = int(req_states.prompt_len.np[row])
        if int(req_states.prefill_len.np[row]) != prompt:
            raise NotImplementedError("Draft Graph does not qualify resumed request history")
        bounds.append((prompt, int(req_states.max_seq_len[row])))
    return tuple(bounds)


def _request_bounds(input_batch):
    bounds = input_batch.draft_graph_request_bounds
    if bounds is None:
        raise RuntimeError("Draft Graph lacks authoritative CPU request bounds")
    return bounds


def validate_eagle_single_request_review_exact(owner, input_batch, exact, *, num_reqs, padded, step):
    """Inspect the existing CPU exact owner, never an upper bound or new D2H."""
    scope = getattr(owner, "_draft_graph_review_limit", None)
    if not _is_eagle_support(scope):
        return
    rows = _request_bounds(input_batch)
    if (
        type(num_reqs) is not int
        or num_reqs != len(rows)
        or type(padded) is not int
        or padded != len(rows)
        or type(step) is not int
        or not 0 <= step < scope.static_k
        or exact.device.type != "cpu"
        or exact.ndim != 1
        or exact.shape[0] != len(rows)
        or str(exact.dtype) != "torch.int32"
    ):
        raise RuntimeError("Eagle review lost its exact admitted-row CPU consumer input")
    context = getattr(getattr(owner, "_eagle3_window_epoch_source", None), "_active", None)
    continuing_rows = context.phase_plan.baseline_only_rows if context is not None else ()
    for position, (prompt, maximum) in enumerate(rows):
        value = exact[position].item()
        lower = prompt
        upper = min(maximum + scope.static_k, scope.max_model_len)
        if position in continuing_rows:
            # The proposal epoch already classified and validated this row from
            # canonical InputBatch counters. In a mixed prefill, its first
            # consumer ends before the prompt; later Eagle AR consumers advance
            # that end by their step, while the row remains a Full-source read.
            try:
                computed_value = input_batch.num_computed_prefill_tokens_np[position]
                scheduled_value = input_batch.num_scheduled_tokens[position]
                if type(computed_value) is bool or type(scheduled_value) is bool:
                    raise TypeError("prefill counters must be integers, not booleans")
                computed = index(computed_value)
                scheduled = index(scheduled_value)
            except (AttributeError, IndexError, TypeError) as error:
                raise RuntimeError("Eagle continuing prefill requires its canonical row counters") from error
            end = computed + scheduled
            if not 0 <= computed < prompt or scheduled <= 0 or end >= prompt:
                raise RuntimeError("Eagle continuing prefill counters disagree with its frozen row phase")
            lower = upper = min(end + step, scope.max_model_len)
        if type(value) is not int or not lower <= value <= upper:
            raise NotImplementedError(
                "Eagle consumer exact length is outside its request and model capacity: "
                f"row={position}, step={step}, exact={value!r}, expected=[{lower}, {upper}]"
            )


def validate_eagle_single_request_graph_descriptor(owner, *, first, n, tokens):
    """Check the executed Graph shape, including Window OFF consumers."""
    scope = getattr(owner, "_draft_graph_review_limit", None)
    if not _is_eagle_support(scope):
        return
    batch = owner.input_batch
    rows = _request_bounds(batch)
    if type(first) is not bool or type(n) is not int or type(tokens) is not int:
        raise TypeError("Eagle review requires exact Graph descriptor types")
    if (
        not 1 <= n <= scope.max_requests
        or n != batch.num_reqs
        or n != batch.num_reqs_after_padding
        or n != len(rows)
        or tokens != n * (scope.static_k + 1 if first else 1)
    ):
        raise NotImplementedError(
            "Eagle review Graph requires padded1 or explicitly admitted padded2, first Q4 or AR Q1 before replay"
        )


def validate_eagle_single_request_graph_stage(owner, inputs, *, first, n, tokens, ordinal):
    scope = getattr(owner, "_draft_graph_review_limit", None)
    if not _is_eagle_support(scope):
        return
    validate_eagle_single_request_graph_descriptor(owner, first=first, n=n, tokens=tokens)
    if (
        len(inputs.identity.proposal.ordered_request_ids) != n
        or inputs.identity.padded_request_count != n
        or type(ordinal) is not int
        or ordinal not in ((0,) if first else (0, 1))
        or inputs.identity.step != (0 if first else ordinal + 1)
    ):
        raise NotImplementedError("Eagle review Graph must use first Q4 and exactly two AR Q1 consumers per request")


def _full8_review_capacity(vllm_config):
    """Use the allocated sequence capacity and both actual model ceilings."""
    target = vllm_config.model_config
    draft = vllm_config.speculative_config.draft_model_config
    capacity = getattr(target, "max_model_len", None)
    ceilings: list[int] = []
    for config in (target, draft):
        text = getattr(config, "hf_text_config", None)
        if text is None:
            text = getattr(config, "hf_config", None)
        ceiling = getattr(text, "max_position_embeddings", None)
        if type(ceiling) is not int or ceiling <= 0:
            raise NotImplementedError(
                "Full8 review requires an exact positive configured capacity within both model position ceilings"
            )
        ceilings.append(ceiling)
    if type(capacity) is not int or capacity <= _FULL8_QUERY_TOKENS or capacity > min(ceilings):
        raise NotImplementedError(
            "Full8 review requires an exact positive configured capacity within both model position ceilings"
        )
    return capacity


def validate_full8_single_request_graph_stage(owner, *, actual, padded, tokens):
    """Check the real Draft descriptor before metadata/native query submission."""
    maximum = full8_graph_review_max_requests(getattr(owner, "_draft_graph_review_limit", None))
    if maximum is None:
        return
    if type(actual) is not int or type(padded) is not int or type(tokens) is not int:
        raise TypeError("Full8 review requires exact integer Draft descriptor counts")
    if not 1 <= actual <= maximum or padded != actual or tokens != padded * _FULL8_QUERY_TOKENS:
        raise NotImplementedError(
            "Full8 review requires actual/padded1 or explicitly admitted actual/padded2, "
            "with eight query tokens per row; no fallback is selected"
        )


def validate_draft_graph_review_requests(
    owner, input_batch, *, dummy_run=False, is_profile=False, context_kv_only=False
):
    """Validate local effective requests before the Graph owner is changed."""
    scope = getattr(owner, "_draft_graph_review_limit", None)
    if scope is None or dummy_run or is_profile or getattr(input_batch, "is_dummy", False):
        return
    if not isinstance(scope, DraftGraphSupport):
        raise RuntimeError("unknown Draft Graph support contract")
    if scope.method == "dflash" and type(input_batch.num_reqs) is int and input_batch.num_reqs == 0:
        return
    if scope.method == "dflash" and (
        getattr(owner, "num_query_per_req", None) != scope.query_tokens
        or getattr(getattr(getattr(owner, "vllm_config", None), "model_config", None), "max_model_len", None)
        != scope.max_model_len
    ):
        raise RuntimeError("DFlash Graph owner lost its configured query/model capacity")
    rows = _request_bounds(input_batch)
    active, padded = input_batch.num_reqs, input_batch.num_reqs_after_padding
    if (
        type(active) is not int
        or not 1 <= active <= scope.max_requests
        or (not context_kv_only and (type(padded) is not int or padded != active))
        or len(rows) != active
        or len(input_batch.req_ids) != active
        or len(set(input_batch.req_ids)) != active
        or len(input_batch.idx_mapping_np) < active
        or len(set(input_batch.idx_mapping_np[:active])) != active
        or len(input_batch.prefill_len_np) < active
        or owner.max_model_len != scope.max_model_len
    ):
        raise NotImplementedError("Draft Graph requires admitted actual rows and their corresponding padded bank")
    blocked = getattr(owner, "_draft_graph_review_preempted_request_ids", ())
    lookahead = scope.static_k if scope.method == "eagle3" else scope.query_tokens
    for position, (prompt, maximum) in enumerate(rows):
        if input_batch.req_ids[position] in blocked:
            raise NotImplementedError("Draft Graph does not qualify resumption after preemption")
        if (
            not 0 < prompt < maximum
            or maximum + lookahead > scope.max_model_len
            or input_batch.prefill_len_np[position] != prompt
        ):
            raise NotImplementedError(
                "Draft Graph requires the original prompt/output budget and static query lookahead "
                "within allocated capacity; resumed requests are not qualified"
            )
