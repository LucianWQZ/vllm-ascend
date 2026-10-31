# SPDX-License-Identifier: Apache-2.0
"""Temporary, checkpoint-scoped Draft Graph review boundaries.

These are fail-closed review limits, not fallbacks or general backend rules.
The fixed config digest identifies a checkpoint configuration, not its weights.
It is read once at factory construction, never on the proposal hot path.
Default-off numeric diagnostics keep their existing private capture interface.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

_FULL8_CONFIG = "d2be70579f0b8e7b109e776dbf14bd88ac5be115d2dabbb20e6373988939f410"
_SWA6_CONFIG = "20784b063040f7871efec4bf63342e5cb941907925943540176dfbb8aaf186dc"
_EAGLE_CONFIG = "5b9242c2e8603da59527fb9e9dd85f3b5bf1e7beae48b251563bbd442a512c65"
_FULL8_QUERY_TOKENS = 8


@dataclass(frozen=True, slots=True)
class EagleSingleRequestReviewScope:
    """Capacity-bounded review entry; this object is not a PASS receipt."""

    max_model_len: int
    static_k: int = 3


def validate_eagle_single_request_review_config(vllm_config, *, experimental_graph_review):
    speculative = vllm_config.speculative_config
    parallel = vllm_config.parallel_config
    scheduler = vllm_config.scheduler_config
    cache = vllm_config.cache_config
    additional = vllm_config.additional_config or {}
    maximum = getattr(getattr(vllm_config, "model_config", None), "max_model_len", None)
    requested_window = additional.get("draft_window_size")
    draft_parallel = getattr(speculative, "draft_parallel_config", None)
    mode = getattr(vllm_config.compilation_config.cudagraph_mode, "name", None)
    # Ascend's actual patch_parallel_config predicate deliberately removes
    # core544's DP>1 restriction. Before AscendConfig normalizes disabled
    # FlashComm, Target TP4/EP therefore has allgather/SP=True; afterward the
    # flashinfer backend makes it False. Accept only these two exact phases.
    # Dense Draft EP is disabled by core544's create_draft_parallel_config;
    # its captured read geometry must remain independently non-SP.
    target_backend = getattr(parallel, "all2all_backend", None)
    target_sp = getattr(parallel, "use_sequence_parallel_moe", None)
    qualified_target_phase = (target_backend == "allgather_reducescatter" and target_sp is True) or (
        target_backend == "flashinfer_all2allv" and target_sp is False
    )
    if (
        experimental_graph_review is not True
        or speculative is None
        or speculative.method != "eagle3"
        or speculative.enforce_eager
        or speculative.uses_dynamic_speculative_decoding()
        or speculative.num_speculative_tokens != 3
        or getattr(draft_parallel, "tensor_parallel_size", None) != 4
        or getattr(draft_parallel, "use_sequence_parallel_moe", None) is not False
        or getattr(draft_parallel, "enable_expert_parallel", None) is not False
        or parallel.tensor_parallel_size != 4
        or parallel.data_parallel_size != 1
        or parallel.pipeline_parallel_size != 1
        or parallel.prefill_context_parallel_size != 1
        or parallel.decode_context_parallel_size != 1
        # The independent dense Draft source retains its separately
        # validated replicated read geometry; this is Target config only.
        or getattr(parallel, "enable_expert_parallel", False) is not True
        or not qualified_target_phase
        or additional.get("enable_dsa_cp", False) is not False
        or additional.get("enable_flashcomm1", False) is not False
        or scheduler.async_scheduling is not False
        or cache.enable_prefix_caching is not False
        or mode != "FULL_DECODE_ONLY"
        or requested_window not in (None, 1024)
        or (requested_window is not None and type(requested_window) is not int)
        or type(maximum) is not int
        or maximum <= 3
        or additional.get("draft_window_graph_replicated_tp_review") is not True
        or additional.get("draft_window_graph_numeric_diagnostic", False) is not False
        or speculative.enable_adaptive_verification is not False
    ):
        raise ValueError(
            "Eagle single-request review requires static K3 with Window OFF or W1024, "
            "a positive configured model capacity, TP4/DP1/PP1/PCP1/DCP1, "
            "the qualified Ascend Target EP/SP initialization phases and non-SP dense Draft, FULL_DECODE_ONLY, "
            "non-async/no-prefix/non-adaptive and the experimental Graph gate; "
            f"actual={{'gate': {experimental_graph_review!r}, 'method': {getattr(speculative, 'method', None)!r}, "
            f"'eager': {getattr(speculative, 'enforce_eager', None)!r}, "
            f"'K': {getattr(speculative, 'num_speculative_tokens', None)!r}, "
            f"'draft_tp': {getattr(draft_parallel, 'tensor_parallel_size', None)!r}, "
            f"'draft_EP': {getattr(draft_parallel, 'enable_expert_parallel', None)!r}, "
            f"'draft_SP': {getattr(draft_parallel, 'use_sequence_parallel_moe', None)!r}, "
            f"""'tp_dp_pp_pcp_dcp': {
                (
                    parallel.tensor_parallel_size,
                    parallel.data_parallel_size,
                    parallel.pipeline_parallel_size,
                    parallel.prefill_context_parallel_size,
                    parallel.decode_context_parallel_size,
                )
            }, """
            f"'EP': {getattr(parallel, 'enable_expert_parallel', None)!r}, "
            f"'all2all_backend': {target_backend!r}, 'SP': {target_sp!r}, "
            f"'mode': {mode!r}, 'async': {scheduler.async_scheduling!r}, 'prefix': {cache.enable_prefix_caching!r}, "
            f"'window': {additional.get('draft_window_size')!r}, "
            f"'replicated': {additional.get('draft_window_graph_replicated_tp_review')!r}, "
            f"'FC': {additional.get('enable_flashcomm1', False)!r}, "
            f"'DSACP': {additional.get('enable_dsa_cp', False)!r}, "
            f"'numeric': {additional.get('draft_window_graph_numeric_diagnostic', False)!r}, "
            f"'adaptive': {getattr(speculative, 'enable_adaptive_verification', None)!r}}}"
        )
    return EagleSingleRequestReviewScope(max_model_len=maximum)


def note_eagle_single_request_review_lifecycle(owner, scheduler_output):
    """Keep only authoritative preemption IDs until real request completion.

    a982 v2 resumption is re-emitted as scheduled_new_reqs, including the case
    where no output token exists yet. prefill_len==prompt_len cannot detect
    that case. No scheduler, request state, or device object is modified.
    """
    limit = getattr(owner, "_draft_graph_review_limit", None)
    if not isinstance(limit, EagleSingleRequestReviewScope) and limit != "full8_local_active_at_most_one":
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


def eagle_single_request_review_rows(owner, req_states, req_ids, idx_mapping_np):
    """Snapshot authoritative CPU limits for the finite Eagle and Full8 entries."""
    limit = getattr(owner, "_draft_graph_review_limit", None)
    if not isinstance(limit, EagleSingleRequestReviewScope) and limit != "full8_local_active_at_most_one":
        return None
    method = "Eagle" if isinstance(limit, EagleSingleRequestReviewScope) else "Full8"
    rows = []
    for req_id, raw_index in zip(req_ids, idx_mapping_np):
        if req_id in getattr(owner, "_draft_graph_review_preempted_request_ids", ()):
            raise NotImplementedError(f"{method} single-request review does not qualify resumption after preemption")
        index = int(raw_index)
        if req_states.req_id_to_index.get(req_id) != index:
            raise RuntimeError(f"{method} review request mapping changed during input preparation")
        rows.append(
            (
                req_id,
                index,
                int(req_states.prompt_len.np[index]),
                int(req_states.max_seq_len[index]),
                int(req_states.prefill_len.np[index]),
            )
        )
    if len(rows) != len(req_ids):
        raise RuntimeError(f"{method} review lacks complete authoritative request rows")
    return tuple(rows)


def _eagle_single_request_row(owner, input_batch):
    scope = owner._draft_graph_review_limit
    rows = getattr(input_batch, "draft_graph_review_request_rows", None)
    if (
        type(input_batch.num_reqs) is not int
        or input_batch.num_reqs != 1
        or input_batch.num_reqs_after_padding != 1
        or not isinstance(rows, tuple)
        or len(rows) != 1
    ):
        raise NotImplementedError("Eagle single-request review requires one actual request and one padded row")
    row = rows[0]
    if (
        not isinstance(row, tuple)
        or len(row) != 5
        or not isinstance(row[0], str)
        or any(type(value) is not int for value in row[1:])
        or row[0] != input_batch.req_ids[0]
        or row[1] < 0
        or row[1] != int(input_batch.idx_mapping_np[0])
        or row[4] != int(input_batch.prefill_len_np[0])
    ):
        raise RuntimeError("Eagle review request identity/source row differs")
    if row[0] in getattr(owner, "_draft_graph_review_preempted_request_ids", ()):
        raise NotImplementedError("Eagle single-request review does not qualify resumption after preemption")
    _, _, prompt, maximum, prefill = row
    if (
        type(getattr(owner, "max_model_len", None)) is not int
        or owner.max_model_len != scope.max_model_len
        or not 0 < prompt < maximum
        or maximum + scope.static_k > scope.max_model_len
        or prefill != prompt
    ):
        raise NotImplementedError(
            "Eagle single-request review requires positive prompt/max_tokens plus static-K lookahead "
            "within configured max_model_len; "
            "resumed/preempted prompt-plus-output prefills are not qualified"
        )
    return row


def validate_eagle_single_request_review_exact(owner, input_batch, exact, *, num_reqs, padded, step):
    """Inspect the existing CPU exact owner, never an upper bound or new D2H."""
    scope = getattr(owner, "_draft_graph_review_limit", None)
    if not isinstance(scope, EagleSingleRequestReviewScope):
        return
    _, _, prompt, maximum, _ = _eagle_single_request_row(owner, input_batch)
    if (
        num_reqs != 1
        or padded != 1
        or type(step) is not int
        or not 0 <= step < scope.static_k
        or exact.device.type != "cpu"
        or exact.ndim != 1
        or exact.shape[0] != 1
        or str(exact.dtype) != "torch.int32"
    ):
        raise RuntimeError("Eagle review lost its one-row CPU exact consumer input")
    value = exact[0].item()
    if type(value) is not int or not prompt <= value <= min(maximum + scope.static_k, scope.max_model_len):
        raise NotImplementedError("Eagle consumer exact length is outside its request and model capacity")


def validate_eagle_single_request_graph_stage(owner, inputs, *, first, n, tokens, ordinal):
    scope = getattr(owner, "_draft_graph_review_limit", None)
    if not isinstance(scope, EagleSingleRequestReviewScope):
        return
    _eagle_single_request_row(owner, owner.input_batch)
    if (
        inputs.actual_request_count != 1
        or inputs.padded_request_count != 1
        or n != 1
        or tokens != (4 if first else 1)
        or ordinal not in ((0,) if first else (0, 1))
        or inputs.identity.step != (0 if first else ordinal + 1)
    ):
        raise NotImplementedError("Eagle review Graph must use first Q4 and exactly two AR Q1 consumers, padded1")


def _full8_review_capacity(vllm_config):
    """Use the allocated sequence capacity and both actual model ceilings."""
    target = vllm_config.model_config
    draft = vllm_config.speculative_config.draft_model_config
    capacity = getattr(target, "max_model_len", None)
    ceilings = []
    for config in (target, draft):
        text = getattr(config, "hf_text_config", None)
        if text is None:
            text = getattr(config, "hf_config", None)
        ceilings.append(getattr(text, "max_position_embeddings", None))
    if (
        type(capacity) is not int
        or capacity <= _FULL8_QUERY_TOKENS
        or any(type(value) is not int or value <= 0 for value in ceilings)
        or capacity > min(ceilings)
    ):
        raise NotImplementedError(
            "Full8 review requires an exact positive configured capacity within both model position ceilings"
        )
    return capacity


def validate_full8_single_request_review_config(vllm_config):
    """Prevent unsupported batches using the config shared by every TP rank.

    The execution check still validates actual local rows. Limiting capacity
    here avoids letting a supported service schedule an unqualified TP/DP
    batch and then fail only after some ranks have entered the Draft graph.
    """
    speculative = vllm_config.speculative_config
    parallel = vllm_config.parallel_config
    scheduler = vllm_config.scheduler_config
    additional = vllm_config.additional_config or {}
    sizes = (
        parallel.tensor_parallel_size,
        parallel.data_parallel_size,
        parallel.pipeline_parallel_size,
        parallel.prefill_context_parallel_size,
        parallel.decode_context_parallel_size,
    )
    if (
        sizes != (4, 1, 1, 1, 1)
        or getattr(speculative.draft_parallel_config, "tensor_parallel_size", None) != 4
        or speculative.num_speculative_tokens != 7
        or speculative.enable_adaptive_verification is not False
        or getattr(parallel, "enable_expert_parallel", False) is not True
        or speculative.uses_dynamic_speculative_decoding()
        or type(scheduler.max_num_seqs) is not int
        or scheduler.max_num_seqs != 1
        or scheduler.async_scheduling is not False
        or vllm_config.cache_config.enable_prefix_caching is not False
        or additional.get("draft_window_size") not in (None, 32768)
        or additional.get("enable_dsa_cp", False) is not False
        or additional.get("enable_flashcomm1", False) is not False
        or getattr(vllm_config.compilation_config.cudagraph_mode, "name", None) != "FULL_DECODE_ONLY"
    ):
        raise NotImplementedError(
            "This Full8 checkpoint's temporary Draft Graph review permits only static K7, Window OFF or W32768, "
            "TP4/DP1/PP1/PCP1/DCP1 with EP, max_num_seqs=1, FULL_DECODE_ONLY, "
            "non-async/no-prefix/no-DSA-CP/no-FlashComm. Use that single-request configuration "
            "or the independently supported Draft eager route; no fallback is selected."
        )
    _full8_review_capacity(vllm_config)


def validate_full8_single_request_review_entry(vllm_config, *, experimental_graph_review):
    """Explicit experiment only; every TP rank checks the same fixed config."""
    from vllm_ascend.ascend_config import validate_additional_config_bool

    validate_full8_single_request_review_config(vllm_config)
    speculative = vllm_config.speculative_config
    additional = vllm_config.additional_config or {}
    enabled = validate_additional_config_bool(
        additional.get("draft_window_graph_full8_single_request_review", False),
        "additional_config.draft_window_graph_full8_single_request_review",
    )
    model = getattr(speculative.draft_model_config, "model", None)
    config_file = Path(model) / "config.json" if isinstance(model, (str, Path)) else None
    if (
        not enabled
        or experimental_graph_review is not True
        or speculative.method != "dflash"
        or speculative.enforce_eager is not False
        or additional.get("draft_window_graph_replicated_tp_review") is not True
        or additional.get("draft_window_graph_numeric_diagnostic", False) is not False
        or list(vllm_config.compilation_config.cudagraph_capture_sizes) != [8]
        or config_file is None
        or not config_file.is_file()
        or config_file.stat().st_size > 1 << 20
        or hashlib.sha256(config_file.read_bytes()).hexdigest() != _FULL8_CONFIG
    ):
        raise NotImplementedError(
            "Full8 ordinary review requires its exact checkpoint, explicit Full8 single-request "
            "and experimental Graph gates, validated model capacity and capture[8]; "
            "this is only a finite active1/padded1 qualification entry, not a PASS receipt."
        )
    return "full8_local_active_at_most_one"


def validate_full8_single_request_graph_stage(owner, *, actual, padded, tokens):
    """Check the real Draft descriptor before metadata/native query submission."""
    if getattr(owner, "_draft_graph_review_limit", None) != "full8_local_active_at_most_one":
        return
    if type(actual) is not int or type(padded) is not int or type(tokens) is not int:
        raise TypeError("Full8 review requires exact integer Draft descriptor counts")
    if (actual, padded, tokens) != (1, 1, 8):
        raise NotImplementedError("Full8 review only qualifies actual1/padded1 Draft Q8; no fallback is selected")


def _full8_single_request_row(owner, input_batch):
    """Reject unqualified original requests before a Full8 owner is changed."""
    rows = getattr(input_batch, "draft_graph_review_request_rows", None)
    if not isinstance(rows, tuple) or len(rows) != 1:
        raise RuntimeError("Full8 review requires one authoritative prompt/output request row")
    try:
        current = (
            input_batch.req_ids[0],
            int(input_batch.idx_mapping_np[0]),
            int(input_batch.prefill_len_np[0]),
        )
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as error:
        raise RuntimeError("Full8 review lacks its current request identity/source row") from error
    row = rows[0]
    if (
        not isinstance(row, tuple)
        or len(row) != 5
        or not isinstance(row[0], str)
        or any(type(value) is not int for value in row[1:])
        or (row[0], row[1], row[4]) != current
    ):
        raise RuntimeError("Full8 review request identity/source row differs")
    if row[0] in getattr(owner, "_draft_graph_review_preempted_request_ids", ()):
        raise NotImplementedError("Full8 single-request review does not qualify resumption after preemption")
    _, _, prompt, maximum, prefill = row
    capacity = getattr(owner, "max_model_len", None)
    configured_capacity = getattr(
        getattr(getattr(owner, "vllm_config", None), "model_config", None), "max_model_len", None
    )
    query = getattr(owner, "num_query_per_req", None)
    if (
        type(capacity) is not int
        or capacity <= _FULL8_QUERY_TOKENS
        or type(configured_capacity) is not int
        or capacity != configured_capacity
        or type(query) is not int
        or query != _FULL8_QUERY_TOKENS
        or prompt <= 0
        or maximum <= prompt
        or prefill != prompt
        or maximum + query > capacity
    ):
        raise NotImplementedError(
            "Full8 single-request review requires original positive prompt/output and "
            "prompt + max_tokens + Q8 within the actual allocated model capacity; "
            "resumed/preempted prompt-plus-output prefills are not qualified"
        )
    return row


def resolve_draft_graph_review_limit(vllm_config, *, experimental_graph_review=False):
    """Return a cached limit or reject the precisely identified Graph route."""
    speculative = vllm_config.speculative_config
    additional = getattr(vllm_config, "additional_config", None) or {}
    from vllm_ascend.ascend_config import validate_additional_config_bool

    finite_review = validate_additional_config_bool(
        additional.get("draft_window_graph_eagle_single_request_review", False),
        "additional_config.draft_window_graph_eagle_single_request_review",
    )
    full8_review = validate_additional_config_bool(
        additional.get("draft_window_graph_full8_single_request_review", False),
        "additional_config.draft_window_graph_full8_single_request_review",
    )
    if full8_review and finite_review:
        raise ValueError("Full8 and Eagle finite review entries are mutually exclusive")
    if full8_review:
        return validate_full8_single_request_review_entry(
            vllm_config, experimental_graph_review=experimental_graph_review
        )
    if finite_review:
        finite_scope = validate_eagle_single_request_review_config(
            vllm_config,
            experimental_graph_review=experimental_graph_review,
        )
    if speculative.method not in ("dflash", "eagle3") or speculative.enforce_eager is True:
        return None
    compilation = getattr(vllm_config, "compilation_config", None)
    mode = getattr(compilation, "cudagraph_mode", None)
    if mode is not None:
        decode_mode = getattr(mode, "decode_mode", None)
        effective = decode_mode() if callable(decode_mode) else mode
        if getattr(effective, "name", effective) == "NONE":
            return None
    model = getattr(speculative.draft_model_config, "model", None)
    if not isinstance(model, (str, Path)):
        if finite_review:
            raise ValueError("Eagle finite review requires one bounded local checkpoint config identity")
        return None
    config_file = Path(model) / "config.json"
    if not config_file.is_file():
        if finite_review:
            raise ValueError("Eagle finite review requires one bounded local checkpoint config identity")
        return None
    # The fixed manifests have small JSON configs. Do not read model tensors.
    if config_file.stat().st_size > 1 << 20:
        if finite_review:
            raise ValueError("Eagle finite review requires one bounded local checkpoint config identity")
        return None
    digest = hashlib.sha256(config_file.read_bytes()).hexdigest()
    if finite_review:
        if speculative.method != "eagle3" or digest != _EAGLE_CONFIG:
            raise NotImplementedError("Eagle single-request review requires its fixed checkpoint configuration")
        # A bounded ordinary qualification entry, not an unrestricted product
        # PASS. Actual request/exact/stage guards below retain this scope.
        return finite_scope
    diagnostic = additional.get("draft_window_graph_numeric_diagnostic", False)
    if type(diagnostic) is not bool:
        raise TypeError("draft_window_graph_numeric_diagnostic must be a boolean")
    if diagnostic and experimental_graph_review is True:
        # This changes no diagnostic oracle, ownership or tolerance. The
        # existing private gate still controls access to numeric capture.
        return None
    if speculative.method == "dflash" and digest == _FULL8_CONFIG:
        validate_full8_single_request_review_config(vllm_config)
        raise NotImplementedError(
            "This Full8 checkpoint's ordinary single-request Draft Graph qualification "
            "on the PR017/vLLM544 pair is incomplete, so this entry is not open. "
            "The explicit private numerical diagnostic and independently supported "
            "typed Draft eager route remain available; no fallback is selected."
        )
    if speculative.method == "dflash" and digest == _SWA6_CONFIG:
        raise NotImplementedError(
            "This SWA6 checkpoint's Draft Graph route is blocked by an unresolved "
            "valid-output numerical counterexample. Draft eager and Target Graph "
            "remain independent; no fallback is selected."
        )
    if speculative.method == "eagle3" and digest == _EAGLE_CONFIG:
        raise NotImplementedError(
            "This Eagle3 checkpoint's ordinary Draft Graph entry is pending "
            "single-request qualification on the integrated source/runtime. "
            "The previous four-request valid-output failure remains unresolved; "
            "no eager fallback is selected."
        )
    return None


def validate_draft_graph_review_requests(
    owner, input_batch, *, dummy_run=False, is_profile=False, context_kv_only=False
):
    """Check real local DP rows, not padded graph capacity or client load."""
    limit = getattr(owner, "_draft_graph_review_limit", None)
    if limit is None or dummy_run or is_profile or getattr(input_batch, "is_dummy", False):
        return
    if isinstance(limit, EagleSingleRequestReviewScope):
        _eagle_single_request_row(owner, input_batch)
        return
    if limit != "full8_local_active_at_most_one":
        raise RuntimeError("unknown temporary Draft Graph review boundary")
    active = input_batch.num_reqs
    if type(active) is not int or active < 0:
        raise ValueError("Draft Graph requires an exact nonnegative local active request count")
    if active > 1:
        raise NotImplementedError(
            "This Full8 Draft Graph checkpoint's temporary review boundary permits at most "
            "1 actual request per DP replica; active=2 and greater are blocked by an "
            "unresolved valid-output numerical counterexample. "
            "Set max_num_seqs=1 on the qualified TP4/DP1 service; Draft eager remains independent. "
            "Graph padding capacity is not the active request count."
        )
    if active == 1:
        padded = getattr(input_batch, "num_reqs_after_padding", None)
        if not context_kv_only and (type(padded) is not int or padded != 1):
            raise NotImplementedError(
                "This Full8 checkpoint's single-request Draft Graph review requires actual=1/padded=1; "
                "other padding shapes are not qualified. No fallback is selected."
            )
        _full8_single_request_row(owner, input_batch)
