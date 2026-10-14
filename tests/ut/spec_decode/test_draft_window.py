# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU-only contracts for the MRV2 speculative draft-window P1/P2 layers."""

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)
from vllm.v1.worker.gpu import block_table as upstream_block_table
from vllm.v1.worker.utils import AttentionGroup

import vllm_ascend.worker.v2.spec_decode as spec_decode_factory
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionMetadataBuilder,
)
from vllm_ascend.attention.dsa_v1 import (
    AscendDSABackend,
    AscendDSAMetadataBuilder,
)
from vllm_ascend.attention.mla_v1 import (
    AscendMLABackend,
    AscendMLAMetadataBuilder,
)
from vllm_ascend.attention.sfa_v1 import (
    AscendSFABackend,
    AscendSFAMetadataBuilder,
)
from vllm_ascend.worker.v2 import block_table as ascend_block_table
from vllm_ascend.worker.v2.block_table import AscendBlockTables
from vllm_ascend.worker.v2.spec_decode.autoregressive.speculator import (
    AscendAutoRegressiveSpeculator,
)
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import (
    AscendDFlashSpeculator,
)
from vllm_ascend.worker.v2.spec_decode.draft_window import (
    DirectPrefixProjection,
    DraftWindowGroupInventory,
    DraftWindowInventory,
    DraftWindowLayoutCategory,
    DraftWindowPolicyDecision,
    DraftWindowPolicyState,
    DraftWindowReadDescriptor,
    DraftWindowRuntimeContext,
    DraftWindowScalarPlan,
    FullDraftKVSource,
    SingleGroupDirectPrefixProvider,
    build_draft_window_inventory,
    build_draft_window_runtime_context,
    build_draft_window_source_capability,
    build_full_draft_kv_source,
    compute_direct_prefix_capacity,
    compute_draft_window_scalar_plan,
    compute_legacy_anchor,
    compute_visible_length,
    compute_window_capacity_tokens,
    gather_valid_num_blocks,
    gather_valid_num_blocks_cpu,
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
    draft_architectures: tuple[str, ...] = ("DraftModel",),
) -> SimpleNamespace:
    return SimpleNamespace(
        method=method,
        num_speculative_tokens=configured_k,
        uses_dynamic_speculative_decoding=lambda: dynamic_k,
        draft_model_config=SimpleNamespace(architectures=list(draft_architectures)),
    )


def _vllm_config(
    method: str,
    *,
    target_model_type: str = "qwen3",
    configured_k: int = 3,
    dynamic_k: bool = False,
    draft_architectures: tuple[str, ...] = ("DraftModel",),
) -> SimpleNamespace:
    return SimpleNamespace(
        speculative_config=_speculative_config(
            method,
            configured_k=configured_k,
            dynamic_k=dynamic_k,
            draft_architectures=draft_architectures,
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
    monkeypatch.setattr(spec_decode_factory.logger, "warning", lambda message: warnings.append(message))
    monkeypatch.setattr(
        spec_decode_factory.logger,
        "warning_once",
        lambda _message: (_ for _ in ()).throw(
            AssertionError("forced-OFF warnings must be scoped to each factory-created speculator")
        ),
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


@pytest.mark.parametrize(
    ("config", "constructor_path"),
    [
        (
            _baseline_vllm_config("dspark", use_dspark=True, use_eagle=True),
            "vllm_ascend.worker.v2.spec_decode.dspark.speculator.AscendDSparkSpeculator",
        ),
        (
            _baseline_vllm_config("mtp", use_eagle=True),
            "vllm_ascend.worker.v2.spec_decode.mtp.speculator.AscendMTPSpeculator",
        ),
        (
            _baseline_vllm_config("eagle3", use_eagle=True),
            "vllm_ascend.worker.v2.spec_decode.eagle.speculator.AscendEagleSpeculator",
        ),
    ],
)
def test_public_config_off_factory_reaches_real_baseline_dispatch(
    config: SimpleNamespace,
    constructor_path: str,
) -> None:
    device = torch.device("cpu")
    baseline_speculator = object()
    with (
        patch(
            "vllm_ascend.worker.v2.spec_decode.get_ascend_config",
            return_value=SimpleNamespace(draft_window_size=None),
        ),
        patch(constructor_path, return_value=baseline_speculator) as constructor,
    ):
        assert spec_decode_factory.init_speculator(config, device) is baseline_speculator
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
        target_model_type="  DeepSeek_V4_1  ",
    )

    assert decision.normalized_method == "dspark"
    assert decision.target_model_type == "deepseek_v4_1"
    assert decision.state is DraftWindowPolicyState.FORCED_OFF


def test_model_type_normalization_does_not_rewrite_separators() -> None:
    decision = resolve_draft_window_policy(
        requested_window_size=512,
        method="dspark",
        target_model_type="DeepSeek-V4",
    )

    assert decision.target_model_type == "deepseek-v4"
    assert decision.state is DraftWindowPolicyState.ELIGIBLE_ON


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
    assert not hasattr(baseline, "_draft_window_graph_binding_provider")


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
    assert not hasattr(baseline, "_draft_window_runtime_state")
    assert not hasattr(baseline, "_draft_window_epoch_source")
    assert not hasattr(baseline, "_draft_window_graph_binding_provider")


def test_forced_off_warning_is_emitted_once_per_created_speculator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[str] = []
    created = [object(), object()]
    monkeypatch.setattr(
        spec_decode_factory,
        "get_ascend_config",
        lambda: SimpleNamespace(draft_window_size=512),
    )
    monkeypatch.setattr(
        spec_decode_factory,
        "_init_speculator_baseline",
        lambda _config, _device: created.pop(0),
    )
    monkeypatch.setattr(spec_decode_factory.logger, "warning", lambda message: warnings.append(message))
    monkeypatch.setattr(
        spec_decode_factory.logger,
        "warning_once",
        lambda _message: (_ for _ in ()).throw(
            AssertionError("process-global warning deduplication violates the instance-scoped contract")
        ),
    )
    config = _vllm_config("mtp")

    first = spec_decode_factory.init_speculator(config, torch.device("cpu"))
    second = spec_decode_factory.init_speculator(config, torch.device("cpu"))

    assert first is not second
    assert len(warnings) == 2
    assert all("MTP" in message for message in warnings)


def test_unknown_forced_off_reason_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _baseline, baseline_calls, configured, _warnings = _patch_factory(
        monkeypatch,
        requested_window_size=512,
    )
    monkeypatch.setattr(
        spec_decode_factory,
        "resolve_draft_window_policy",
        lambda **_kwargs: DraftWindowPolicyDecision(
            state=DraftWindowPolicyState.FORCED_OFF,
            requested_window_size=512,
            effective_window_size=None,
            normalized_method="future_method",
            target_model_type="future_model",
            reason="future_forced_off_reason",
        ),
    )

    with pytest.raises(RuntimeError, match="future_forced_off_reason"):
        spec_decode_factory.init_speculator(_vllm_config("mtp"), torch.device("cpu"))

    assert baseline_calls == []
    assert configured == []


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
        (300, 0, 64, 64, (192, 108, 127)),
        (300, 0, 65, 64, (192, 108, 128)),
        (300, 0, 66, 64, (192, 108, 129)),
        (300, 0, 127, 64, (128, 172, 190)),
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


def test_direct_prefix_capacity_covers_zero_offset_projection() -> None:
    capacity = compute_direct_prefix_capacity(
        window_size=32,
        legacy_block_size=96,
        block_span_tokens=48,
    )
    projection = project_direct_prefix(
        window_start=96,
        window_visible_length=127,
        block_span_tokens=48,
    )

    assert projection.first_token_offset == 0
    assert projection.required_columns <= capacity


def test_direct_prefix_capacity_rejects_non_default_geometry() -> None:
    projection = project_direct_prefix(
        window_start=64,
        window_visible_length=47,
        block_span_tokens=48,
    )

    assert projection.first_token_offset == 16
    assert projection.required_columns == 2
    with pytest.raises(ValueError, match="divide legacy block size"):
        compute_direct_prefix_capacity(
            window_size=32,
            legacy_block_size=16,
            block_span_tokens=48,
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


def _p2_policy(method: str = "eagle3") -> DraftWindowPolicyDecision:
    return resolve_draft_window_policy(
        requested_window_size=128,
        method=method,
        target_model_type="qwen3",
    )


def _p2_runtime_context(**changes: Any) -> DraftWindowRuntimeContext:
    context = DraftWindowRuntimeContext(
        graph_mode="NONE",
        async_scheduling=False,
        tensor_parallel_size=1,
        data_parallel_size=1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        sequence_parallel=False,
        dsa_context_parallel=False,
        flashcomm=False,
        prefix_caching=False,
    )
    return replace(context, **changes)


def _p2_vllm_config(
    *,
    configured_k: int = 64,
    additional_config: dict[str, Any] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            enforce_eager=True,
            num_speculative_tokens=configured_k,
            uses_dynamic_speculative_decoding=lambda: False,
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(name="FULL"),
            pass_config=SimpleNamespace(enable_sp=False),
        ),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            use_sequence_parallel_moe=False,
        ),
        additional_config=additional_config or {},
    )


def _full_attention_spec(block_size: int = 64) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=16,
        dtype=torch.float16,
    )


def _attention_group(
    *,
    global_group_id: int = 0,
    layer_names: tuple[str, ...] = ("draft.layers.0.self_attn",),
    block_size: int = 64,
    backend_class: type[Any] = AscendAttentionBackend,
    builder: Any | None = None,
    spec: Any | None = None,
) -> AttentionGroup:
    if builder is None:
        builder = object.__new__(AscendAttentionMetadataBuilder)
    return AttentionGroup(
        backend=backend_class,
        layer_names=list(layer_names),
        kv_cache_spec=spec or _full_attention_spec(block_size),
        kv_cache_group_id=global_group_id,
        metadata_builders=[builder],
    )


def _block_tables(
    *,
    block_sizes: tuple[int, ...] = (64,),
    kernel_block_sizes: tuple[int, ...] = (64,),
    row_width: int = 8,
    max_num_reqs: int = 6,
) -> SimpleNamespace:
    num_groups = len(block_sizes)
    return SimpleNamespace(
        block_sizes=list(block_sizes),
        kernel_block_sizes=list(kernel_block_sizes),
        input_block_tables=[
            torch.arange(max_num_reqs * row_width, dtype=torch.int32).reshape(max_num_reqs, row_width)
            for _ in range(num_groups)
        ],
        num_blocks=SimpleNamespace(
            gpu=torch.zeros((num_groups, max_num_reqs), dtype=torch.int32),
            cpu=torch.zeros((num_groups, max_num_reqs), dtype=torch.int32),
        ),
        cp_size=1,
        slot_mappings=torch.arange(32, dtype=torch.int32).reshape(num_groups, -1),
        compute_slot_mappings=MagicMock(name="compute_slot_mappings"),
    )


class _CpuOnlyBlockTableBuffer:
    """Minimal storage shim that lets the real BlockTables ctor run on CPU."""

    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device | None = None,
    ) -> None:
        self.gpu = torch.zeros(shape, dtype=dtype, device=device or torch.device("cpu"))
        self.cpu = torch.zeros(shape, dtype=dtype, device="cpu")
        self.np = self.cpu.numpy()


def _inventory(
    *,
    attn_groups: list[list[AttentionGroup]] | None = None,
    block_tables: SimpleNamespace | None = None,
    causal: bool | dict[int, bool] = True,
    runtime_route: str = "eagle3",
    runtime_context: DraftWindowRuntimeContext | None = None,
) -> DraftWindowInventory:
    if attn_groups is None:
        attn_groups = [[_attention_group()]]
    if block_tables is None:
        block_tables = _block_tables()
    return build_draft_window_inventory(
        policy=_p2_policy("dflash" if runtime_route == "dflash2" else runtime_route),
        runtime_route=runtime_route,
        attn_groups=attn_groups,
        block_tables=block_tables,
        causal=causal,
        runtime_context=runtime_context or _p2_runtime_context(),
    )


def _inventory_and_source(
    *,
    logical_block_size: int = 64,
    kernel_block_size: int = 64,
    row_width: int = 8,
) -> tuple[DraftWindowInventory, FullDraftKVSource, SimpleNamespace]:
    block_tables = _block_tables(
        block_sizes=(logical_block_size,),
        kernel_block_sizes=(kernel_block_size,),
        row_width=row_width,
    )
    inventory = _inventory(
        attn_groups=[
            [
                _attention_group(
                    block_size=logical_block_size,
                )
            ]
        ],
        block_tables=block_tables,
    )
    source = build_full_draft_kv_source(
        inventory=inventory,
        block_tables=block_tables,
        idx_mapping=torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32),
    )
    return inventory, source, block_tables


def test_p2_inventory_uses_global_gid_and_actual_runtime_classes() -> None:
    block_tables = _block_tables(
        block_sizes=(32, 64),
        kernel_block_sizes=(32, 64),
    )
    group = _attention_group(
        global_group_id=1,
        layer_names=("draft.layers.4.self_attn", "draft.layers.5.self_attn"),
    )

    inventory = _inventory(
        attn_groups=[[], [group]],
        block_tables=block_tables,
        causal={1: False},
    )

    assert inventory.empty_global_group_ids == (0,)
    assert inventory.consumer_global_group_ids == (1,)
    assert inventory.capability_failures == ()
    assert inventory.groups == (
        DraftWindowGroupInventory(
            global_group_id=1,
            local_consumer_index=0,
            layer_names=(
                "draft.layers.4.self_attn",
                "draft.layers.5.self_attn",
            ),
            backend_class=AscendAttentionBackend,
            builder_class=AscendAttentionMetadataBuilder,
            logical_block_size=64,
            kernel_block_size=64,
            block_span_tokens=64,
            causal=False,
            layout_category=DraftWindowLayoutCategory.DENSE_GQA_DIRECT_PREFIX,
            capability_failures=(),
        ),
    )


def test_p2_inventory_rejects_mixed_consumers_in_one_global_gid() -> None:
    class UnknownBackend:
        pass

    class UnknownBuilder:
        pass

    inventory = _inventory(
        attn_groups=[
            [
                _attention_group(),
                _attention_group(
                    layer_names=("draft.layers.1.self_attn",),
                    backend_class=UnknownBackend,
                    builder=UnknownBuilder(),
                ),
            ]
        ]
    )

    assert len(inventory.groups) == 2
    with pytest.raises(
        NotImplementedError,
        match=r"method=eagle3.*gid=0.*multiple executable consumers",
    ):
        inventory.require_single_group_direct_prefix()


def test_p2_inventory_rejects_homogeneous_multi_group_runtime() -> None:
    block_tables = _block_tables(
        block_sizes=(64, 64),
        kernel_block_sizes=(64, 64),
    )
    inventory = _inventory(
        attn_groups=[
            [_attention_group(global_group_id=0)],
            [
                _attention_group(
                    global_group_id=1,
                    layer_names=("draft.layers.1.self_attn",),
                )
            ],
        ],
        block_tables=block_tables,
    )

    assert inventory.consumer_global_group_ids == (0, 1)
    with pytest.raises(NotImplementedError, match="multi-group runtime"):
        inventory.require_single_group_direct_prefix()


def test_p2_inventory_unknown_layout_is_diagnostic() -> None:
    class UnknownBackend:
        pass

    class UnknownBuilder:
        pass

    inventory = _inventory(
        attn_groups=[
            [
                _attention_group(
                    backend_class=UnknownBackend,
                    builder=UnknownBuilder(),
                )
            ]
        ]
    )

    group = inventory.groups[0]
    assert group.layout_category is DraftWindowLayoutCategory.UNKNOWN
    assert group.block_span_tokens is None
    with pytest.raises(
        NotImplementedError,
        match=r"backend=UnknownBackend.*layout=unknown",
    ):
        inventory.require_single_group_direct_prefix()


@pytest.mark.parametrize(
    ("backend_class", "builder_class", "spec", "expected_category"),
    [
        (
            AscendMLABackend,
            AscendMLAMetadataBuilder,
            MLAAttentionSpec(
                block_size=64,
                num_kv_heads=1,
                head_size=16,
                dtype=torch.float16,
            ),
            DraftWindowLayoutCategory.MLA,
        ),
        (
            AscendDSABackend,
            AscendDSAMetadataBuilder,
            _full_attention_spec(),
            DraftWindowLayoutCategory.DSA,
        ),
        (
            AscendSFABackend,
            AscendSFAMetadataBuilder,
            _full_attention_spec(),
            DraftWindowLayoutCategory.SFA,
        ),
        (
            AscendAttentionBackend,
            AscendAttentionMetadataBuilder,
            FullAttentionSpec(
                block_size=64,
                num_kv_heads=1,
                head_size=16,
                dtype=torch.float16,
                sliding_window=256,
            ),
            DraftWindowLayoutCategory.NATIVE_OR_RECYCLING,
        ),
        (
            AscendAttentionBackend,
            AscendAttentionMetadataBuilder,
            SlidingWindowSpec(
                block_size=64,
                num_kv_heads=1,
                head_size=16,
                dtype=torch.float16,
                sliding_window=256,
            ),
            DraftWindowLayoutCategory.NATIVE_OR_RECYCLING,
        ),
    ],
)
def test_p2_inventory_fail_closes_unverified_backend_layouts(
    backend_class: type[Any],
    builder_class: type[Any],
    spec: Any,
    expected_category: DraftWindowLayoutCategory,
) -> None:
    inventory = _inventory(
        attn_groups=[
            [
                _attention_group(
                    backend_class=backend_class,
                    builder=object.__new__(builder_class),
                    spec=spec,
                )
            ]
        ]
    )

    assert inventory.groups[0].layout_category is expected_category
    assert inventory.groups[0].block_span_tokens is None
    with pytest.raises(NotImplementedError, match=expected_category.value):
        inventory.require_single_group_direct_prefix()


def test_p2_inventory_all_empty_groups_fail_closed() -> None:
    block_tables = _block_tables(
        block_sizes=(64, 64),
        kernel_block_sizes=(64, 64),
    )
    inventory = _inventory(
        attn_groups=[[], []],
        block_tables=block_tables,
    )

    assert inventory.empty_global_group_ids == (0, 1)
    with pytest.raises(NotImplementedError, match="no executable draft"):
        inventory.require_single_group_direct_prefix()


def test_p2_dflash2_is_inventory_only_and_not_a_runtime_candidate() -> None:
    inventory = _inventory(runtime_route="dflash2")

    assert inventory.runtime_route == "dflash2"
    assert inventory.groups[0].backend_class is AscendAttentionBackend
    with pytest.raises(
        NotImplementedError,
        match=r"method=dflash.*route=dflash2.*DFlash2",
    ):
        inventory.require_single_group_direct_prefix()


def test_p2_inventory_rejects_policy_runtime_route_mismatch() -> None:
    inventory = build_draft_window_inventory(
        policy=_p2_policy("dflash"),
        runtime_route="eagle3",
        attn_groups=[[_attention_group()]],
        block_tables=_block_tables(),
        causal=True,
        runtime_context=_p2_runtime_context(),
    )

    with pytest.raises(
        NotImplementedError,
        match=r"method=dflash.*route=eagle3.*does not match policy method",
    ):
        inventory.require_single_group_direct_prefix()


@pytest.mark.parametrize(
    ("context_change", "diagnostic"),
    [
        ({"graph_mode": "FULL"}, "Graph"),
        ({"async_scheduling": True}, "Async"),
        ({"tensor_parallel_size": 2}, "TP=2"),
        ({"data_parallel_size": 2}, "DP=2"),
        ({"pipeline_parallel_size": 2}, "PP=2"),
        ({"prefill_context_parallel_size": 2}, "PCP=2"),
        ({"decode_context_parallel_size": 2}, "DCP=2"),
        ({"sequence_parallel": True}, "sequence parallel"),
        ({"dsa_context_parallel": True}, "DSA-CP"),
        ({"flashcomm": True}, "FlashComm"),
    ],
)
def test_p2_inventory_fail_closes_unverified_runtime_modes(
    context_change: dict[str, Any],
    diagnostic: str,
) -> None:
    inventory = _inventory(
        runtime_context=_p2_runtime_context(**context_change),
    )

    with pytest.raises(NotImplementedError, match=diagnostic):
        inventory.require_single_group_direct_prefix()


def test_failure_free_inventory_invariant_uses_runtime_error() -> None:
    inventory = _inventory()
    impossible_group = replace(
        inventory.groups[0],
        layout_category=DraftWindowLayoutCategory.UNKNOWN,
        block_span_tokens=None,
        capability_failures=(),
    )
    forged_inventory = replace(
        inventory,
        groups=(impossible_group,),
        capability_failures=(),
    )

    with pytest.raises(RuntimeError, match="failure-free inventory"):
        forged_inventory.require_single_group_direct_prefix()


def test_full_source_references_canonical_owners_without_exposing_writes() -> None:
    inventory, source, block_tables = _inventory_and_source()
    table_before = block_tables.input_block_tables[0].clone()
    counts_before = block_tables.num_blocks.gpu.clone()
    slots_before = block_tables.slot_mappings.clone()

    assert source.references_input_block_table(
        0,
        block_tables.input_block_tables[0],
    )
    assert source.references_current_num_blocks_gpu_view(block_tables.num_blocks.gpu)
    assert source.references_persistent_num_blocks_cpu(block_tables.num_blocks.cpu)
    assert source.logical_block_sizes == (64,)
    assert source.kernel_block_sizes == (64,)
    assert source.source_identity.input_block_table_ptrs == (block_tables.input_block_tables[0].data_ptr(),)
    source.validate_owner_bindings()
    with pytest.raises(FrozenInstanceError):
        source.logical_block_sizes = (32,)  # type: ignore[misc]

    assert inventory.require_single_group_direct_prefix().global_group_id == 0
    torch.testing.assert_close(block_tables.input_block_tables[0], table_before)
    torch.testing.assert_close(block_tables.num_blocks.gpu, counts_before)
    torch.testing.assert_close(block_tables.slot_mappings, slots_before)
    block_tables.compute_slot_mappings.assert_not_called()


def test_full_source_contract_with_real_ascend_block_tables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        upstream_block_table,
        "StagedWriteTensor",
        _CpuOnlyBlockTableBuffer,
    )
    monkeypatch.setattr(
        upstream_block_table,
        "UvaBackedTensor",
        _CpuOnlyBlockTableBuffer,
    )
    monkeypatch.setattr(
        ascend_block_table.triton,
        "next_power_of_2",
        lambda value: 1 << (value - 1).bit_length(),
        raising=False,
    )
    block_tables = AscendBlockTables(
        block_sizes=[64],
        max_num_reqs=6,
        max_num_batched_tokens=32,
        max_num_blocks_per_group=[8],
        device=torch.device("cpu"),
        kernel_block_sizes=[64],
    )
    inventory = _inventory(block_tables=block_tables)

    source = build_full_draft_kv_source(
        inventory=inventory,
        block_tables=block_tables,
        idx_mapping=torch.tensor([0, 1], dtype=torch.int64),
    )

    assert source.references_input_block_table(
        0,
        block_tables.input_block_tables[0],
    )
    assert source.references_current_num_blocks_gpu_view(block_tables.num_blocks.gpu)
    assert source.references_persistent_num_blocks_cpu(block_tables.num_blocks.cpu)
    assert source.logical_block_sizes == tuple(block_tables.block_sizes)
    assert source.kernel_block_sizes == tuple(block_tables.kernel_block_sizes)
    assert block_tables.cp_size == 1


def test_full_source_distinguishes_stable_cpu_owner_from_rotating_gpu_view() -> None:
    _inventory_value, source, block_tables = _inventory_and_source()
    original_gpu_view = block_tables.num_blocks.gpu
    replacement_gpu_view = original_gpu_view.clone()

    block_tables.num_blocks.gpu = replacement_gpu_view

    assert source.references_current_num_blocks_gpu_view(replacement_gpu_view)
    assert not source.references_current_num_blocks_gpu_view(original_gpu_view)
    assert source.references_persistent_num_blocks_cpu(block_tables.num_blocks.cpu)
    source.validate_owner_bindings()


def test_full_source_detects_canonical_table_replacement() -> None:
    _inventory_value, source, block_tables = _inventory_and_source()
    block_tables.input_block_tables[0] = torch.zeros_like(block_tables.input_block_tables[0])

    with pytest.raises(RuntimeError, match="canonical input block table 0 was replaced"):
        source.validate_owner_bindings()


def test_full_source_binds_exact_current_batch_mapping_without_copying() -> None:
    inventory, _source, block_tables = _inventory_and_source()
    static_source = build_full_draft_kv_source(
        inventory=inventory,
        block_tables=block_tables,
    )
    current_idx_mapping = torch.tensor([3, 1], dtype=torch.int32)

    with pytest.raises(RuntimeError, match="bind the current batch idx_mapping"):
        gather_valid_num_blocks(
            static_source,
            global_group_id=0,
            num_reqs=2,
            num_reqs_padded=2,
        )

    batch_source = static_source.bind_batch(current_idx_mapping)
    assert not static_source.references_same_idx_mapping_tensor(current_idx_mapping)
    assert static_source.source_identity.idx_mapping_ptr is None
    assert batch_source.references_same_idx_mapping_tensor(current_idx_mapping)
    assert batch_source.source_identity.idx_mapping_ptr == current_idx_mapping.data_ptr()
    current_idx_mapping.copy_(torch.tensor([1, 3], dtype=torch.int32))
    assert batch_source.references_same_idx_mapping_tensor(current_idx_mapping)
    assert batch_source.references_input_block_table(
        0,
        block_tables.input_block_tables[0],
    )


def test_full_source_rejects_inventory_source_geometry_mismatch() -> None:
    inventory = _inventory()
    incompatible_block_tables = _block_tables(
        block_sizes=(64,),
        kernel_block_sizes=(32,),
    )

    with pytest.raises(
        RuntimeError,
        match=r"method=eagle3.*gid=0.*source geometry.*kernel=32.*inventory.*kernel=64",
    ):
        build_full_draft_kv_source(
            inventory=inventory,
            block_tables=incompatible_block_tables,
        )


def test_valid_counts_gather_persistent_rows_into_batch_order_and_clear_padding() -> None:
    inventory, _source, block_tables = _inventory_and_source()
    block_tables.num_blocks.gpu[0] = torch.tensor(
        [1, 4, 2, 6, 3, 5],
        dtype=torch.int32,
    )
    idx_mapping = torch.tensor([3, 1, 5, 0, 0, 0], dtype=torch.int32)
    source = build_full_draft_kv_source(
        inventory=inventory,
        block_tables=block_tables,
        idx_mapping=idx_mapping,
    )
    out = torch.full((6,), 99, dtype=torch.int32)

    gathered = gather_valid_num_blocks(
        source,
        global_group_id=0,
        num_reqs=3,
        num_reqs_padded=5,
        out=out,
    )

    assert gathered.data_ptr() == out.data_ptr()
    assert gathered.tolist() == [6, 4, 5, 0, 0]
    assert out.tolist() == [6, 4, 5, 0, 0, 0]


def test_cpu_valid_counts_follow_canonical_owner_and_rotated_gpu_view() -> None:
    inventory, _source, block_tables = _inventory_and_source()
    canonical_cpu = torch.tensor([[1, 4, 2, 6, 3, 5]], dtype=torch.int32)
    block_tables.num_blocks.cpu.copy_(canonical_cpu)
    source = build_full_draft_kv_source(
        inventory=inventory,
        block_tables=block_tables,
    )
    original_identity = source.source_identity

    # UvaBackedTensor rotates its accelerator view. The canonical CPU owner,
    # not that transient view pointer, is the stable source identity.
    block_tables.num_blocks.gpu = canonical_cpu.clone()
    source.validate_owner_bindings()
    assert source.source_identity == original_identity

    gathered = gather_valid_num_blocks_cpu(
        source,
        global_group_id=0,
        idx_mapping_np=np.array([3, 1, 5], dtype=np.intp),
        num_reqs=3,
        num_reqs_padded=5,
    )
    assert gathered.tolist() == [6, 4, 5, 0, 0]


def test_cpu_valid_counts_reject_replaced_canonical_owner() -> None:
    _inventory_value, source, block_tables = _inventory_and_source()
    block_tables.num_blocks.cpu = torch.zeros_like(block_tables.num_blocks.cpu)

    with pytest.raises(RuntimeError, match="canonical persistent num_blocks.cpu"):
        source.validate_owner_bindings()


@pytest.mark.parametrize("alias_kind", ["persistent_counts", "input_table", "idx_mapping"])
def test_valid_count_output_cannot_alias_borrowed_full_source(
    alias_kind: str,
) -> None:
    inventory, _source, block_tables = _inventory_and_source()
    block_tables.num_blocks.gpu[0] = torch.tensor(
        [1, 4, 2, 6, 3, 5],
        dtype=torch.int32,
    )
    idx_mapping = torch.tensor([3, 1, 5, 0, 0, 0], dtype=torch.int32)
    source = build_full_draft_kv_source(
        inventory=inventory,
        block_tables=block_tables,
        idx_mapping=idx_mapping,
    )
    canonical_table_before = block_tables.input_block_tables[0].clone()
    persistent_counts_before = block_tables.num_blocks.gpu.clone()
    idx_mapping_before = idx_mapping.clone()
    aliases = {
        "persistent_counts": block_tables.num_blocks.gpu[0],
        "input_table": block_tables.input_block_tables[0][0],
        "idx_mapping": idx_mapping,
    }

    with pytest.raises(ValueError, match="must not alias borrowed FullSource storage"):
        gather_valid_num_blocks(
            source,
            global_group_id=0,
            num_reqs=3,
            num_reqs_padded=5,
            out=aliases[alias_kind],
        )

    torch.testing.assert_close(
        block_tables.input_block_tables[0],
        canonical_table_before,
    )
    torch.testing.assert_close(
        block_tables.num_blocks.gpu,
        persistent_counts_before,
    )
    torch.testing.assert_close(idx_mapping, idx_mapping_before)


@pytest.mark.parametrize("idx_dtype", [torch.int32, torch.int64])
def test_valid_counts_follow_reorder_subset_and_never_reuse_stale_tail(
    idx_dtype: torch.dtype,
) -> None:
    inventory, _source, block_tables = _inventory_and_source()
    block_tables.num_blocks.gpu[0] = torch.tensor(
        [2, 7, 4, 1, 6, 3],
        dtype=torch.int32,
    )
    idx_mapping = torch.tensor([4, 1, 2, 0, 0, 0], dtype=idx_dtype)
    source = build_full_draft_kv_source(
        inventory=inventory,
        block_tables=block_tables,
        idx_mapping=idx_mapping,
    )
    out = torch.full((6,), 88, dtype=torch.int32)

    assert gather_valid_num_blocks(
        source,
        global_group_id=0,
        num_reqs=3,
        num_reqs_padded=4,
        out=out,
    ).tolist() == [6, 7, 4, 0]

    idx_mapping[:2] = torch.tensor([2, 4], dtype=idx_dtype)
    assert gather_valid_num_blocks(
        source,
        global_group_id=0,
        num_reqs=2,
        num_reqs_padded=3,
        out=out,
    ).tolist() == [4, 6, 0]
    assert out.tolist() == [4, 6, 0, 0, 0, 0]


def test_valid_count_does_not_infer_from_zero_block_ids() -> None:
    inventory, _source, block_tables = _inventory_and_source(row_width=4)
    block_tables.input_block_tables[0][0] = torch.tensor(
        [0, 9, 0, 0],
        dtype=torch.int32,
    )
    block_tables.num_blocks.gpu[0, 0] = 2
    source = build_full_draft_kv_source(
        inventory=inventory,
        block_tables=block_tables,
        idx_mapping=torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32),
    )

    valid = gather_valid_num_blocks(
        source,
        global_group_id=0,
        num_reqs=1,
        num_reqs_padded=1,
    )
    provider = SingleGroupDirectPrefixProvider(
        inventory=inventory,
        full_source=source,
        legacy_block_size=64,
        expected_future_offset=0,
    )
    descriptor = provider.describe(
        DraftWindowScalarPlan(
            full_exact_end=128,
            full_horizon_end=128,
            window_start=0,
            window_visible_length=128,
            window_capacity_tokens=191,
        ),
        valid_num_blocks=int(valid[0]),
    )

    assert descriptor.source_start_column == 0
    assert descriptor.required_columns == 2
    assert descriptor.valid_num_blocks == 2


def test_provider_returns_read_only_descriptor_at_exact_coverage_boundary() -> None:
    inventory, source, block_tables = _inventory_and_source(row_width=8)
    table_before = block_tables.input_block_tables[0].clone()
    provider = SingleGroupDirectPrefixProvider(
        inventory=inventory,
        full_source=source,
        legacy_block_size=64,
        expected_future_offset=64,
    )

    descriptor = provider.describe(
        DraftWindowScalarPlan(
            full_exact_end=192,
            full_horizon_end=256,
            window_start=128,
            window_visible_length=64,
            window_capacity_tokens=191,
        ),
        valid_num_blocks=3,
    )

    assert descriptor == DraftWindowReadDescriptor(
        global_group_id=0,
        source_start_column=2,
        first_token_offset=0,
        required_columns=1,
        safe_extent_tokens=64,
        valid_num_blocks=3,
        source_row_width=8,
    )
    assert descriptor.source_end_column == 3
    torch.testing.assert_close(block_tables.input_block_tables[0], table_before)


def test_provider_non_boundary_projection_uses_canonical_anchor() -> None:
    inventory, source, _block_tables_value = _inventory_and_source()
    provider = SingleGroupDirectPrefixProvider(
        inventory=inventory,
        full_source=source,
        legacy_block_size=64,
        expected_future_offset=0,
    )

    descriptor = provider.describe(
        DraftWindowScalarPlan(
            full_exact_end=257,
            full_horizon_end=257,
            window_start=128,
            window_visible_length=129,
            window_capacity_tokens=191,
        ),
        valid_num_blocks=5,
    )

    assert descriptor.source_start_column == 2
    assert descriptor.first_token_offset == 0
    assert descriptor.required_columns == 3
    assert descriptor.source_end_column == 5


def test_provider_allows_proven_span_dividing_legacy_geometry() -> None:
    inventory, source, _block_tables_value = _inventory_and_source(
        logical_block_size=64,
        kernel_block_size=32,
    )
    provider = SingleGroupDirectPrefixProvider(
        inventory=inventory,
        full_source=source,
        legacy_block_size=64,
        expected_future_offset=64,
    )

    descriptor = provider.describe(
        DraftWindowScalarPlan(
            full_exact_end=129,
            full_horizon_end=193,
            window_start=64,
            window_visible_length=65,
            window_capacity_tokens=191,
        ),
        valid_num_blocks=5,
    )

    assert descriptor.source_start_column == 2
    assert descriptor.first_token_offset == 0
    assert descriptor.required_columns == 3


def test_provider_rejects_noncanonical_plan_even_when_group_offset_is_zero() -> None:
    inventory, source, _block_tables_value = _inventory_and_source(
        logical_block_size=64,
        kernel_block_size=32,
    )
    provider = SingleGroupDirectPrefixProvider(
        inventory=inventory,
        full_source=source,
        legacy_block_size=64,
        expected_future_offset=0,
    )
    noncanonical_plan = DraftWindowScalarPlan(
        full_exact_end=96,
        full_horizon_end=96,
        window_start=32,
        window_visible_length=64,
        window_capacity_tokens=191,
    )

    with pytest.raises(ValueError, match=r"canonical P1 anchor.*32 != 0"):
        provider.describe(noncanonical_plan, valid_num_blocks=3)


def test_provider_rejects_span_not_dividing_legacy_without_realigning_anchor() -> None:
    inventory, source, _block_tables_value = _inventory_and_source(
        logical_block_size=96,
        kernel_block_size=48,
    )
    assert (
        project_direct_prefix(
            window_start=64,
            window_visible_length=48,
            block_span_tokens=48,
        ).first_token_offset
        == 16
    )

    with pytest.raises(NotImplementedError, match=r"span=48.*legacy_block_size=64"):
        SingleGroupDirectPrefixProvider(
            inventory=inventory,
            full_source=source,
            legacy_block_size=64,
            expected_future_offset=0,
        )


def test_provider_t10_source_shortage_is_a_hard_error() -> None:
    inventory, source, _block_tables_value = _inventory_and_source()
    provider = SingleGroupDirectPrefixProvider(
        inventory=inventory,
        full_source=source,
        legacy_block_size=64,
        expected_future_offset=0,
    )
    plan = DraftWindowScalarPlan(
        full_exact_end=384,
        full_horizon_end=384,
        window_start=256,
        window_visible_length=128,
        window_capacity_tokens=191,
    )

    with pytest.raises(
        RuntimeError,
        match=r"source shortage.*start=4.*required=2.*valid=5",
    ):
        provider.describe(plan, valid_num_blocks=5)

    descriptor = provider.describe(plan, valid_num_blocks=6)
    assert descriptor.source_end_column == descriptor.valid_num_blocks == 6


def test_provider_rejects_stale_table_tail_even_when_row_width_is_large() -> None:
    inventory, source, block_tables = _inventory_and_source(row_width=8)
    block_tables.input_block_tables[0][0, 5] = 777
    provider = SingleGroupDirectPrefixProvider(
        inventory=inventory,
        full_source=source,
        legacy_block_size=64,
        expected_future_offset=0,
    )
    plan = DraftWindowScalarPlan(
        full_exact_end=384,
        full_horizon_end=384,
        window_start=256,
        window_visible_length=128,
        window_capacity_tokens=191,
    )

    with pytest.raises(RuntimeError, match="source shortage"):
        provider.describe(plan, valid_num_blocks=5)


def test_provider_detects_canonical_owner_replacement_before_describing() -> None:
    inventory, source, block_tables = _inventory_and_source()
    provider = SingleGroupDirectPrefixProvider(
        inventory=inventory,
        full_source=source,
        legacy_block_size=64,
        expected_future_offset=0,
    )
    block_tables.input_block_tables[0] = torch.zeros_like(block_tables.input_block_tables[0])

    with pytest.raises(
        RuntimeError,
        match="canonical input block table 0 was replaced",
    ):
        provider.describe(
            compute_draft_window_scalar_plan(
                full_exact_end=128,
                future_offset=0,
                window_size=128,
                legacy_block_size=64,
            ),
            valid_num_blocks=2,
        )


@pytest.mark.parametrize(
    "owner_type",
    [
        AscendAutoRegressiveSpeculator,
        AscendDFlashSpeculator,
        AscendDSparkSpeculator,
    ],
)
def test_config_off_owner_does_not_enter_p2_inventory(
    owner_type: type[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = object.__new__(owner_type)
    module = __import__(owner_type.__module__, fromlist=["unused"])
    monkeypatch.setattr(
        module,
        "build_draft_window_source_capability",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("CONFIG_OFF/FORCED_OFF must not enter P2")),
    )
    monkeypatch.setattr(
        module,
        "build_draft_window_runtime_state",
        lambda _capability: (_ for _ in ()).throw(AssertionError("CONFIG_OFF/FORCED_OFF must not enter P3")),
    )

    owner._initialize_draft_window_source_capability()

    assert not hasattr(owner, "_draft_window_source_capability")
    assert not hasattr(owner, "_draft_window_runtime_state")
    assert not hasattr(owner, "_draft_window_epoch_source")
    assert not hasattr(owner, "_draft_window_graph_binding_provider")


@pytest.mark.parametrize(
    ("owner_type", "method", "runtime_route", "causal"),
    [
        (AscendAutoRegressiveSpeculator, "eagle3", "eagle3", True),
        (AscendDFlashSpeculator, "dflash", "dflash", {0: False}),
        (AscendDSparkSpeculator, "dspark", "dspark", {0: True}),
    ],
)
def test_eligible_owner_builds_p2_capability_from_set_attn_state(
    owner_type: type[Any],
    method: str,
    runtime_route: str,
    causal: bool | dict[int, bool],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = object.__new__(owner_type)
    owner._draft_window_policy = _p2_policy(method)
    owner.attn_groups = object()
    owner.block_tables = object()
    owner.vllm_config = object()
    if owner_type is not AscendAutoRegressiveSpeculator:
        owner._group_causal = causal
        owner._speculator_name = runtime_route.capitalize()
    expected_capability = object()
    expected_runtime_state = object()
    calls: list[dict[str, Any]] = []
    runtime_calls: list[object] = []
    module = __import__(owner_type.__module__, fromlist=["unused"])

    def build_capability(**kwargs: Any) -> object:
        calls.append(kwargs)
        return expected_capability

    monkeypatch.setattr(
        module,
        "build_draft_window_source_capability",
        build_capability,
    )
    monkeypatch.setattr(
        module,
        "build_draft_window_runtime_state",
        lambda capability: runtime_calls.append(capability) or expected_runtime_state,
    )
    p4b_initialize = None
    if owner_type is AscendAutoRegressiveSpeculator:
        p4b_initialize = MagicMock()
        monkeypatch.setattr(
            owner,
            "_initialize_eagle3_draft_window_state",
            p4b_initialize,
        )

    owner._initialize_draft_window_source_capability()

    assert owner._draft_window_source_capability is expected_capability
    assert owner._draft_window_runtime_state is expected_runtime_state
    assert len(calls) == 1
    assert runtime_calls == [expected_capability]
    assert calls[0]["policy"] is owner._draft_window_policy
    assert calls[0]["attn_groups"] is owner.attn_groups
    assert calls[0]["block_tables"] is owner.block_tables
    assert calls[0]["vllm_config"] is owner.vllm_config
    assert calls[0]["runtime_route"] == runtime_route
    assert calls[0]["causal"] == causal
    if owner_type in (AscendDFlashSpeculator, AscendDSparkSpeculator):
        assert hasattr(owner, "_draft_window_epoch_source")
    else:
        assert not hasattr(owner, "_draft_window_epoch_source")
        assert p4b_initialize is not None
        p4b_initialize.assert_called_once_with()


def test_dflash2_owner_passes_explicit_runtime_route_to_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = object.__new__(AscendDFlashSpeculator)
    owner._draft_window_policy = _p2_policy("dflash")
    owner._speculator_name = "DFlash2"
    owner._group_causal = True
    owner.attn_groups = object()
    owner.block_tables = object()
    owner.vllm_config = object()
    seen: dict[str, Any] = {}
    runtime_calls: list[object] = []

    def reject_after_inventory(**kwargs: Any) -> object:
        seen.update(kwargs)
        raise NotImplementedError("DFlash2 runtime Window remains gated")

    monkeypatch.setattr(
        "vllm_ascend.worker.v2.spec_decode.dflash.speculator.build_draft_window_source_capability",
        reject_after_inventory,
    )
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.spec_decode.dflash.speculator.build_draft_window_runtime_state",
        lambda capability: runtime_calls.append(capability),
    )

    with pytest.raises(NotImplementedError, match="DFlash2"):
        owner._initialize_draft_window_source_capability()

    assert seen["runtime_route"] == "dflash2"
    assert runtime_calls == []
    assert not hasattr(owner, "_draft_window_source_capability")
    assert not hasattr(owner, "_draft_window_runtime_state")
    assert not hasattr(owner, "_draft_window_epoch_source")


@pytest.mark.parametrize(
    "owner_type",
    [
        AscendAutoRegressiveSpeculator,
        AscendDFlashSpeculator,
        AscendDSparkSpeculator,
    ],
)
def test_p3_runtime_initialization_failure_leaves_no_partial_owner_state(
    owner_type: type[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = object.__new__(owner_type)
    owner._draft_window_policy = _p2_policy("eagle3")
    owner.attn_groups = object()
    owner.block_tables = object()
    owner.vllm_config = object()
    if owner_type is not AscendAutoRegressiveSpeculator:
        owner._group_causal = True
        owner._speculator_name = "Eagle3"
    module = __import__(owner_type.__module__, fromlist=["unused"])
    capability = object()
    monkeypatch.setattr(
        module,
        "build_draft_window_source_capability",
        lambda **_kwargs: capability,
    )
    monkeypatch.setattr(
        module,
        "build_draft_window_runtime_state",
        lambda _capability: (_ for _ in ()).throw(RuntimeError("allocation failed")),
    )

    with pytest.raises(RuntimeError, match="allocation failed"):
        owner._initialize_draft_window_source_capability()

    assert not hasattr(owner, "_draft_window_source_capability")
    assert not hasattr(owner, "_draft_window_runtime_state")
    assert not hasattr(owner, "_draft_window_epoch_source")


def test_runtime_context_uses_actual_eager_and_topology_configuration() -> None:
    vllm_config = _p2_vllm_config()

    context = build_draft_window_runtime_context(
        vllm_config,
        SimpleNamespace(cp_size=1),
    )

    assert context == _p2_runtime_context()


def test_runtime_context_flashcomm_env_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", "1")
    context = build_draft_window_runtime_context(
        _p2_vllm_config(),
        SimpleNamespace(cp_size=1),
    )

    assert context.flashcomm is True
    inventory = _inventory(runtime_context=context)
    with pytest.raises(NotImplementedError, match="FlashComm"):
        inventory.require_single_group_direct_prefix()


def test_runtime_context_prefix_cache_fails_before_source_and_workspace() -> None:
    vllm_config = _p2_vllm_config()
    vllm_config.cache_config.enable_prefix_caching = True
    context = build_draft_window_runtime_context(
        vllm_config,
        SimpleNamespace(cp_size=1),
    )

    assert context.prefix_caching is True
    inventory = _inventory(runtime_context=context)
    with pytest.raises(NotImplementedError, match="prefix cache"):
        inventory.require_single_group_direct_prefix()


def test_eagle3_provider_rejects_plan_offset_that_disagrees_with_configured_k() -> None:
    block_tables = _block_tables(row_width=8)
    capability = build_draft_window_source_capability(
        policy=_p2_policy("eagle3"),
        runtime_route="eagle3",
        attn_groups=[[_attention_group()]],
        block_tables=block_tables,
        causal=True,
        vllm_config=_p2_vllm_config(configured_k=64),
    )
    assert capability.eager_topology_fingerprint is not None
    assert capability.eager_topology_fingerprint.graph_mode == "NONE"
    assert capability.eager_topology_fingerprint.async_scheduling is False
    assert capability.eager_topology_fingerprint.global_group_ids == (0,)
    mismatched_plan = DraftWindowScalarPlan(
        full_exact_end=257,
        full_horizon_end=257,
        window_start=128,
        window_visible_length=129,
        window_capacity_tokens=191,
    )

    with pytest.raises(ValueError, match=r"configured.*future offset.*64"):
        capability.provider.describe(
            mismatched_plan,
            valid_num_blocks=5,
        )


def test_async_source_capability_stops_before_full_source_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vllm_config = _p2_vllm_config(configured_k=3)
    vllm_config.scheduler_config.async_scheduling = True

    def forbidden_full_source(**_kwargs: Any) -> Any:
        raise AssertionError("STATIC_READY async must not create FullSource")

    monkeypatch.setattr(
        "vllm_ascend.worker.v2.spec_decode.draft_window.build_full_draft_kv_source",
        forbidden_full_source,
    )

    with pytest.raises(NotImplementedError, match="STATIC_READY.*NPU completion"):
        build_draft_window_source_capability(
            policy=_p2_policy("eagle3"),
            runtime_route="eagle3",
            attn_groups=[[_attention_group()]],
            block_tables=_block_tables(),
            causal=True,
            vllm_config=vllm_config,
        )
