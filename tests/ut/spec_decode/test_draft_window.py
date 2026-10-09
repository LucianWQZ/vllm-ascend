# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU-only contracts for the MRV2 speculative draft-window P1 layer."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch

import vllm_ascend.worker.v2.spec_decode as spec_decode_factory
from vllm_ascend.worker.v2.spec_decode.autoregressive.speculator import (
    AscendAutoRegressiveSpeculator,
)
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import (
    AscendDFlashSpeculator,
)
from vllm_ascend.worker.v2.spec_decode.draft_window import (
    DirectPrefixProjection,
    DraftWindowPolicyDecision,
    DraftWindowPolicyState,
    DraftWindowScalarPlan,
    compute_direct_prefix_capacity,
    compute_draft_window_scalar_plan,
    compute_legacy_anchor,
    compute_visible_length,
    compute_window_capacity_tokens,
    project_direct_prefix,
    resolve_draft_window_policy,
    resolve_static_future_offset,
    validate_eligible_window_numbers,
)
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import (
    AscendDSparkSpeculator,
)


def _speculative_config(
    method: str,
    *,
    configured_k: int = 3,
    dynamic_k: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        method=method,
        num_speculative_tokens=configured_k,
        uses_dynamic_speculative_decoding=lambda: dynamic_k,
    )


def _vllm_config(
    method: str,
    *,
    target_model_type: str = "qwen3",
    configured_k: int = 3,
    dynamic_k: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        speculative_config=_speculative_config(
            method,
            configured_k=configured_k,
            dynamic_k=dynamic_k,
        ),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=target_model_type)),
    )


def _baseline_vllm_config(
    method: str,
    *,
    use_dspark: bool = False,
    use_dflash: bool = False,
    use_eagle: bool = False,
    use_gemma4_mtp: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            method=method,
            use_dspark=lambda: use_dspark,
            use_dflash=lambda: use_dflash,
            use_eagle=lambda: use_eagle,
            use_gemma4_mtp=lambda: use_gemma4_mtp,
            use_step3p5_mtp=lambda: False,
            draft_model_config=SimpleNamespace(architectures=["DraftModel"]),
        )
    )


def _patch_factory(
    monkeypatch: pytest.MonkeyPatch,
    *,
    requested_window_size: int | None,
) -> tuple[Any, list[tuple[Any, torch.device]], list[Any], list[str]]:
    baseline_speculator = SimpleNamespace()
    baseline_calls: list[tuple[Any, torch.device]] = []
    configured_decisions: list[Any] = []
    warnings: list[str] = []

    monkeypatch.setattr(
        spec_decode_factory,
        "get_ascend_config",
        lambda: SimpleNamespace(draft_window_size=requested_window_size),
    )

    def init_baseline(config: Any, device: torch.device) -> Any:
        baseline_calls.append((config, device))
        return baseline_speculator

    monkeypatch.setattr(
        spec_decode_factory,
        "_init_speculator_baseline",
        init_baseline,
    )
    monkeypatch.setattr(
        spec_decode_factory,
        "_configure_eligible_draft_window",
        lambda speculator, decision: configured_decisions.append((speculator, decision)),
    )
    monkeypatch.setattr(
        spec_decode_factory.logger,
        "warning_once",
        lambda message: warnings.append(message),
    )
    return baseline_speculator, baseline_calls, configured_decisions, warnings


@pytest.mark.parametrize(
    ("config", "constructor_path"),
    [
        (
            _baseline_vllm_config("dspark", use_dspark=True, use_eagle=True),
            "vllm_ascend.worker.v2.spec_decode.dspark.speculator.AscendDSparkSpeculator",
        ),
        (
            _baseline_vllm_config("dflash", use_dflash=True, use_eagle=True),
            "vllm_ascend.worker.v2.spec_decode.dflash.speculator.AscendDFlashSpeculator",
        ),
        (
            _baseline_vllm_config("mtp", use_eagle=True),
            "vllm_ascend.worker.v2.spec_decode.mtp.speculator.AscendMTPSpeculator",
        ),
        (
            _baseline_vllm_config(
                "mtp",
                use_eagle=True,
                use_gemma4_mtp=True,
            ),
            "vllm_ascend.worker.v2.spec_decode.eagle.speculator.AscendEagleSpeculator",
        ),
        (
            _baseline_vllm_config("eagle3", use_eagle=True),
            "vllm_ascend.worker.v2.spec_decode.eagle.speculator.AscendEagleSpeculator",
        ),
    ],
)
def test_extracted_baseline_factory_preserves_dispatch_order(
    config: SimpleNamespace,
    constructor_path: str,
) -> None:
    device = torch.device("cpu")
    with patch(constructor_path) as constructor:
        assert spec_decode_factory._init_speculator_baseline(config, device) is constructor.return_value
        constructor.assert_called_once_with(config, device)


@pytest.mark.parametrize("method", ["mtp", "eagle", "eagle3", "dflash", "dspark"])
def test_none_resolves_config_off_without_validating_method(method: str) -> None:
    decision = resolve_draft_window_policy(
        requested_window_size=None,
        method=method,
        target_model_type="deepseek_v4",
    )

    assert decision == DraftWindowPolicyDecision(
        state=DraftWindowPolicyState.CONFIG_OFF,
        requested_window_size=None,
        effective_window_size=None,
        normalized_method=method,
        target_model_type="deepseek_v4",
        reason=None,
    )


@pytest.mark.parametrize("requested_window_size", [512, 0, -1, 2])
def test_mtp_forced_off_precedes_numeric_validation(
    requested_window_size: int,
) -> None:
    decision = resolve_draft_window_policy(
        requested_window_size=requested_window_size,
        method="mtp",
        target_model_type="deepseek_v4",
    )

    assert decision.state is DraftWindowPolicyState.FORCED_OFF
    assert decision.requested_window_size == requested_window_size
    assert decision.effective_window_size is None
    assert decision.reason == "mtp"


@pytest.mark.parametrize("requested_window_size", [512, 0, -1, 2])
def test_deepseek_v4_dspark_forced_off_precedes_numeric_validation(
    requested_window_size: int,
) -> None:
    decision = resolve_draft_window_policy(
        requested_window_size=requested_window_size,
        method="dspark",
        target_model_type="deepseek_v4_1",
    )

    assert decision.state is DraftWindowPolicyState.FORCED_OFF
    assert decision.effective_window_size is None
    assert decision.reason == "deepseek_v4_cotrained_dspark"


def test_deepseek_v41_model_type_is_in_forced_off_family() -> None:
    decision = resolve_draft_window_policy(
        requested_window_size=-1,
        method="dspark",
        target_model_type="deepseek_v41",
    )

    assert decision.state is DraftWindowPolicyState.FORCED_OFF
    assert decision.reason == "deepseek_v4_cotrained_dspark"


@pytest.mark.parametrize("method", ["eagle3", "dflash", "dspark"])
def test_supported_route_resolves_eligible(method: str) -> None:
    decision = resolve_draft_window_policy(
        requested_window_size=512,
        method=method,
        target_model_type="qwen3",
    )

    assert decision.state is DraftWindowPolicyState.ELIGIBLE_ON
    assert decision.effective_window_size == 512
    assert decision.reason is None


def test_plain_eagle_with_window_fails_closed() -> None:
    with pytest.raises(NotImplementedError, match="plain Eagle"):
        resolve_draft_window_policy(
            requested_window_size=512,
            method="eagle",
            target_model_type="llama",
        )


def test_policy_normalizes_method_and_explicit_model_family() -> None:
    decision = resolve_draft_window_policy(
        requested_window_size=-1,
        method="  DSPARK  ",
        target_model_type="  DeepSeek-V4_1  ",
    )

    assert decision.normalized_method == "dspark"
    assert decision.target_model_type == "deepseek_v4_1"
    assert decision.state is DraftWindowPolicyState.FORCED_OFF


def test_unknown_configured_method_fails_closed() -> None:
    with pytest.raises(NotImplementedError, match="unknown_method"):
        resolve_draft_window_policy(
            requested_window_size=512,
            method="unknown_method",
            target_model_type="qwen3",
        )


def test_policy_decision_is_immutable() -> None:
    decision = resolve_draft_window_policy(
        requested_window_size=512,
        method="eagle3",
        target_model_type="qwen3",
    )

    with pytest.raises(FrozenInstanceError):
        decision.effective_window_size = 1024  # type: ignore[misc]


def test_config_off_factory_preserves_baseline_and_emits_no_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline, baseline_calls, configured, warnings = _patch_factory(
        monkeypatch,
        requested_window_size=None,
    )
    config = _vllm_config("eagle")
    device = torch.device("cpu")

    def fail_validation(**_kwargs: Any) -> None:
        raise AssertionError("CONFIG_OFF must not run numeric validation")

    monkeypatch.setattr(
        spec_decode_factory,
        "validate_eligible_window_numbers",
        fail_validation,
    )

    assert spec_decode_factory.init_speculator(config, device) is baseline
    assert baseline_calls == [(config, device)]
    assert configured == []
    assert warnings == []
    assert not hasattr(baseline, "_draft_window_policy")


@pytest.mark.parametrize(
    ("method", "target_model_type", "requested_window_size", "warning_fragment"),
    [
        ("mtp", "qwen3", 512, "MTP"),
        ("mtp", "qwen3", -1, "MTP"),
        ("dspark", "deepseek_v4", 512, "DeepSeek-V4"),
        ("dspark", "deepseek_v4_1", -1, "DeepSeek-V4"),
    ],
)
def test_forced_off_factory_skips_all_window_downstream_work(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    target_model_type: str,
    requested_window_size: int,
    warning_fragment: str,
) -> None:
    baseline, baseline_calls, configured, warnings = _patch_factory(
        monkeypatch,
        requested_window_size=requested_window_size,
    )
    config = _vllm_config(
        method,
        target_model_type=target_model_type,
        configured_k=8,
    )
    device = torch.device("cpu")

    def fail_validation(**_kwargs: Any) -> None:
        raise AssertionError("FORCED_OFF must precede numeric validation")

    monkeypatch.setattr(
        spec_decode_factory,
        "validate_eligible_window_numbers",
        fail_validation,
    )

    assert spec_decode_factory.init_speculator(config, device) is baseline
    assert baseline_calls == [(config, device)]
    assert configured == []
    assert len(warnings) == 1
    assert warning_fragment in warnings[0]
    assert not hasattr(baseline, "_draft_window_policy")


@pytest.mark.parametrize(
    ("method", "configured_k", "expected_future_offset"),
    [
        ("eagle3", 3, 3),
        ("dflash", 8, 0),
        ("dspark", 8, 0),
    ],
)
def test_eligible_factory_validates_then_configures_speculator(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    configured_k: int,
    expected_future_offset: int,
) -> None:
    baseline, baseline_calls, configured, warnings = _patch_factory(
        monkeypatch,
        requested_window_size=512,
    )
    config = _vllm_config(method, configured_k=configured_k)
    device = torch.device("cpu")
    validation_calls: list[tuple[int, int]] = []
    real_validate = validate_eligible_window_numbers

    def record_validation(*, window_size: int, future_offset: int) -> None:
        validation_calls.append((window_size, future_offset))
        real_validate(window_size=window_size, future_offset=future_offset)

    monkeypatch.setattr(
        spec_decode_factory,
        "validate_eligible_window_numbers",
        record_validation,
    )

    assert spec_decode_factory.init_speculator(config, device) is baseline
    assert validation_calls == [(512, expected_future_offset)]
    assert baseline_calls == [(config, device)]
    assert len(configured) == 1
    assert configured[0][0] is baseline
    assert configured[0][1].state is DraftWindowPolicyState.ELIGIBLE_ON
    assert warnings == []


@pytest.mark.parametrize(
    ("window_size", "configured_k", "message"),
    [
        (0, 3, "greater than 0"),
        (-1, 3, "greater than 0"),
        (2, 3, "static future offset"),
    ],
)
def test_invalid_eligible_numbers_fail_before_speculator_construction(
    monkeypatch: pytest.MonkeyPatch,
    window_size: int,
    configured_k: int,
    message: str,
) -> None:
    _baseline, baseline_calls, configured, warnings = _patch_factory(
        monkeypatch,
        requested_window_size=window_size,
    )
    config = _vllm_config("eagle3", configured_k=configured_k)

    with pytest.raises(ValueError, match=message):
        spec_decode_factory.init_speculator(config, torch.device("cpu"))

    assert baseline_calls == []
    assert configured == []
    assert warnings == []


def test_dynamic_k_fails_before_speculator_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _baseline, baseline_calls, configured, warnings = _patch_factory(
        monkeypatch,
        requested_window_size=512,
    )
    config = _vllm_config("eagle3", configured_k=8, dynamic_k=True)

    with pytest.raises(NotImplementedError, match="dynamic speculative decoding"):
        spec_decode_factory.init_speculator(config, torch.device("cpu"))

    assert baseline_calls == []
    assert configured == []
    assert warnings == []


@pytest.mark.parametrize(
    ("owner_type", "method"),
    [
        (AscendAutoRegressiveSpeculator, "eagle3"),
        (AscendDFlashSpeculator, "dflash"),
        (AscendDSparkSpeculator, "dspark"),
    ],
)
def test_speculator_owner_accepts_one_eligible_policy(
    owner_type: type[Any],
    method: str,
) -> None:
    owner = object.__new__(owner_type)
    decision = resolve_draft_window_policy(
        requested_window_size=512,
        method=method,
        target_model_type="qwen3",
    )

    owner._configure_draft_window(decision)

    assert owner._draft_window_policy is decision
    with pytest.raises(RuntimeError, match="already configured"):
        owner._configure_draft_window(decision)


@pytest.mark.parametrize(
    "owner_type",
    [
        AscendAutoRegressiveSpeculator,
        AscendDFlashSpeculator,
        AscendDSparkSpeculator,
    ],
)
def test_speculator_owner_rejects_noneligible_policy(owner_type: type[Any]) -> None:
    owner = object.__new__(owner_type)
    decision = resolve_draft_window_policy(
        requested_window_size=None,
        method="eagle3",
        target_model_type="qwen3",
    )

    with pytest.raises(ValueError, match="ELIGIBLE_ON"):
        owner._configure_draft_window(decision)
    assert not hasattr(owner, "_draft_window_policy")


@pytest.mark.parametrize(
    ("full_exact_end", "future_offset", "window_size", "block_size", "expected"),
    [
        (100, 3, 128, 64, (0, 100, 191)),
        (189, 3, 64, 64, (128, 61, 127)),
        (190, 3, 64, 64, (128, 62, 127)),
        (130, 0, 64, 64, (64, 66, 127)),
        (4096, 3, 512, 16, (3584, 512, 527)),
        (100, 4, 3, 64, (64, 36, 66)),
    ],
)
def test_scalar_plan_matches_explicit_integer_oracles(
    full_exact_end: int,
    future_offset: int,
    window_size: int,
    block_size: int,
    expected: tuple[int, int, int],
) -> None:
    plan = compute_draft_window_scalar_plan(
        full_exact_end=full_exact_end,
        future_offset=future_offset,
        window_size=window_size,
        legacy_block_size=block_size,
    )

    expected_start, expected_visible, expected_capacity = expected
    assert plan == DraftWindowScalarPlan(
        full_exact_end=full_exact_end,
        full_horizon_end=full_exact_end + future_offset,
        window_start=expected_start,
        window_visible_length=expected_visible,
        window_capacity_tokens=expected_capacity,
    )


def test_negative_visible_length_fails_closed() -> None:
    assert (
        compute_legacy_anchor(
            full_context_length=63,
            future_offset=2,
            window_size=1,
            legacy_block_size=64,
        )
        == 64
    )
    with pytest.raises(ValueError, match="exceeds full exact end"):
        compute_visible_length(full_exact_end=63, window_start=64)
    with pytest.raises(ValueError, match="exceeds full exact end"):
        compute_draft_window_scalar_plan(
            full_exact_end=63,
            future_offset=2,
            window_size=1,
            legacy_block_size=64,
        )


def test_anchor_uses_legacy_block_alignment_not_lcm() -> None:
    assert (
        compute_legacy_anchor(
            full_context_length=300,
            future_offset=0,
            window_size=128,
            legacy_block_size=64,
        )
        == 128
    )
    assert (
        compute_legacy_anchor(
            full_context_length=1000,
            future_offset=0,
            window_size=128,
            legacy_block_size=64,
        )
        == 832
    )


def test_direct_prefix_projection_handles_nonzero_offset() -> None:
    projection = project_direct_prefix(
        window_start=63,
        window_visible_length=64,
        block_span_tokens=64,
    )

    assert projection == DirectPrefixProjection(
        source_start_column=0,
        first_token_offset=63,
        required_columns=2,
        safe_extent_tokens=65,
    )
    assert projection.source_end_column == 2


def test_empty_direct_prefix_projection_has_zero_extent() -> None:
    projection = project_direct_prefix(
        window_start=63,
        window_visible_length=0,
        block_span_tokens=64,
    )

    assert projection.required_columns == 0
    assert projection.safe_extent_tokens == 0
    assert projection.source_end_column == projection.source_start_column


@pytest.mark.parametrize(("window_size", "expected_columns"), [(64, 2), (100, 3)])
def test_direct_prefix_capacity_preserves_alignment_slack(
    window_size: int,
    expected_columns: int,
) -> None:
    assert (
        compute_window_capacity_tokens(
            window_size=window_size,
            legacy_block_size=64,
        )
        == window_size + 63
    )
    assert (
        compute_direct_prefix_capacity(
            window_size=window_size,
            legacy_block_size=64,
            block_span_tokens=64,
        )
        == expected_columns
    )


def test_same_absolute_anchor_projects_per_group_without_realigning() -> None:
    plan = compute_draft_window_scalar_plan(
        full_exact_end=1000,
        future_offset=0,
        window_size=256,
        legacy_block_size=128,
    )

    assert plan.window_start == 640
    span_128 = project_direct_prefix(
        window_start=plan.window_start,
        window_visible_length=plan.window_visible_length,
        block_span_tokens=128,
    )
    span_64 = project_direct_prefix(
        window_start=plan.window_start,
        window_visible_length=plan.window_visible_length,
        block_span_tokens=64,
    )
    assert span_128.source_start_column == 5
    assert span_64.source_start_column == 10


def test_scalar_plan_uses_exact_end_not_planning_upper_bound() -> None:
    plan = compute_draft_window_scalar_plan(
        full_exact_end=150,
        future_offset=0,
        window_size=100,
        legacy_block_size=64,
    )

    assert plan.window_start == 0
    assert plan.window_visible_length == 150


def test_static_future_offset_is_method_specific() -> None:
    assert (
        resolve_static_future_offset(
            normalized_method="eagle3",
            configured_k=3,
            uses_dynamic_k=False,
        )
        == 3
    )
    assert (
        resolve_static_future_offset(
            normalized_method="dflash",
            configured_k=8,
            uses_dynamic_k=False,
        )
        == 0
    )
    assert (
        resolve_static_future_offset(
            normalized_method="dspark",
            configured_k=8,
            uses_dynamic_k=False,
        )
        == 0
    )


def test_dynamic_future_offset_fails_closed() -> None:
    with pytest.raises(NotImplementedError, match="dynamic speculative decoding"):
        resolve_static_future_offset(
            normalized_method="eagle3",
            configured_k=8,
            uses_dynamic_k=True,
        )


@pytest.mark.parametrize(
    ("window_size", "future_offset", "message"),
    [(0, 0, "greater than 0"), (-1, 0, "greater than 0"), (2, 3, "static future offset")],
)
def test_numeric_validation_rejects_invalid_eligible_values(
    window_size: int,
    future_offset: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_eligible_window_numbers(
            window_size=window_size,
            future_offset=future_offset,
        )
