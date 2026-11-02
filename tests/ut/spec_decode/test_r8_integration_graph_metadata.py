# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU Graph metadata and source-owner contracts; no NPU execution.

These isolated CPU protocol tests establish exact-source/owner rejection;
model, numerical, capture and lifetime evidence remains separate.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture(params=["dspark", "dflash", "dflash2", "eagle3"])
def graph_source_initialization(request, monkeypatch):
    """Exercise actual owner initialization without loading a device model."""
    method = request.param
    from vllm_ascend.worker.v2.spec_decode.draft_window_eager import DraftWindowGraphInputMixin

    source, provider = object(), object()
    build_source, build_provider = Mock(return_value=source), Mock(return_value=provider)
    if method == "eagle3":
        from vllm_ascend.worker.v2.spec_decode import draft_window_eagle3_adapter, draft_window_eagle3_capture

        initialize = draft_window_eagle3_adapter.Eagle3GraphIntegrationMixin._initialize_draft_window_source_capability
        monkeypatch.setattr(draft_window_eagle3_capture, "build_eagle3_graph_source", build_source)
        monkeypatch.setattr(draft_window_eagle3_adapter, "DraftWindowGraphBindingProvider", build_provider)
    else:
        from vllm_ascend.worker.v2.spec_decode import draft_window_eager

        initialize = DraftWindowGraphInputMixin._initialize_draft_window_source_capability
        monkeypatch.setattr(draft_window_eager, "_build_draft_window_source_capability_for_context", build_source)
        monkeypatch.setattr(draft_window_eager, "DraftWindowGraphBindingProvider", build_provider)
    owner = SimpleNamespace(
        _draft_window_policy=object(),
        _speculator_name=method,
        attn_groups=object(),
        block_tables=object(),
        kv_cache_config=object(),
        _group_causal=False,
        vllm_config=object(),
        _initialize_draft_window_graph_inputs=Mock(),
        _initialize_eagle3_draft_window_state=Mock(),
    )
    return SimpleNamespace(
        owner=owner,
        initialize=initialize,
        build_source=build_source,
        build_provider=build_provider,
        source=source,
        provider=provider,
    )


def test_graph_owner_installs_only_its_admitted_source(graph_source_initialization):
    c = graph_source_initialization
    c.initialize(c.owner)
    c.build_source.assert_called_once()
    c.build_provider.assert_called_once_with(source=c.source)
    assert c.owner._draft_window_graph_binding_provider is c.provider
    assert not hasattr(c.owner, "_draft_window_source_capability")
    assert not hasattr(c.owner, "_draft_window_runtime_state")
    assert not hasattr(c.owner, "_draft_window_runtime")
    if c.owner._speculator_name != "eagle3":
        assert c.build_source.call_args.kwargs["dflash2_graph_owner"] is (
            c.owner if c.owner._speculator_name == "dflash2" else None
        )


def test_graph_source_rejection_has_no_eager_fallback_or_partial_owner(graph_source_initialization):
    c = graph_source_initialization
    failure = NotImplementedError("unsupported actual Graph source")
    c.build_source.side_effect = failure
    with pytest.raises(NotImplementedError) as seen:
        c.initialize(c.owner)
    assert seen.value is failure
    c.build_provider.assert_not_called()
    assert not hasattr(c.owner, "_draft_window_graph_binding_provider")


def test_graph_owner_reinitialization_rejects_before_source_build(graph_source_initialization):
    c = graph_source_initialization
    c.owner._draft_window_graph_binding_provider = c.provider
    with pytest.raises(RuntimeError, match="already initialized"):
        c.initialize(c.owner)
    c.build_source.assert_not_called()


def test_window_off_without_graph_policy_does_not_construct_source(graph_source_initialization):
    c = graph_source_initialization
    del c.owner._draft_window_policy
    c.initialize(c.owner)
    c.build_source.assert_not_called()
    c.build_provider.assert_not_called()
    assert not hasattr(c.owner, "_draft_window_graph_binding_provider")


@pytest.mark.parametrize(
    "field",
    [None, "pipeline_parallel_size", "prefill_context_parallel_size", "decode_context_parallel_size", "cp_size"],
)
def test_graph_initialization_validates_actual_unsharded_context_once(monkeypatch, field):
    from vllm_ascend.worker.v2.spec_decode import draft_window_graph_runtime as runtime

    monkeypatch.setattr(runtime, "draft_window_graph_enabled", lambda: True)
    monkeypatch.setattr(
        runtime,
        "get_current_hardware_profile",
        lambda: SimpleNamespace(attention_backend_family=runtime.AttentionBackendFamily.STANDARD),
    )
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(name="FULL_DECODE_ONLY")),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=16,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
    )
    tables = SimpleNamespace(cp_size=1)
    if field == "cp_size":
        tables.cp_size = 2
    elif field is not None:
        setattr(config.parallel_config, field, 2)
    if field is None:
        runtime.validate_graph_context(config, tables)
    else:
        with pytest.raises(RuntimeError, match="PP=PCP=DCP=1"):
            runtime.validate_graph_context(config, tables)


@pytest.fixture
def bridge():
    from vllm_ascend.worker.v2.spec_decode.draft_window_read import DraftReadView

    device = torch.tensor([17, 1025, 0, 0], dtype=torch.int32)
    exact = device.clone()
    table = torch.arange(8).reshape(4, 2)
    read = DraftReadView(table, device, exact, exact, 1025, num_reqs=4)
    common = SimpleNamespace(num_reqs=4, slot_mapping=object(), positions=object())
    read.apply(common)
    return SimpleNamespace(read=read, common=common)


def test_current_exact_uses_the_common_backend_read_interface_without_a_second_certificate(bridge):
    slots, positions = bridge.common.slot_mapping, bridge.common.positions
    assert bridge.common._draft_window_read is bridge.read
    assert bridge.read.exact_cpu_for(bridge.common) is bridge.read.seq_lens_cpu
    assert bridge.common.seq_lens_cpu_upper_bound is bridge.read.seq_lens_cpu
    assert bridge.common.seq_lens is bridge.read.seq_lens
    assert bridge.common.slot_mapping is slots and bridge.common.positions is positions


@pytest.mark.parametrize(
    "field", ["seq_lens", "seq_lens_cpu", "_seq_lens_cpu", "seq_lens_cpu_upper_bound", "block_table_tensor", "num_reqs"]
)
def test_replaced_read_operand_is_rejected_at_the_actual_backend_read_boundary(bridge, field):
    value = getattr(bridge.common, field)
    setattr(bridge.common, field, value.clone() if isinstance(value, torch.Tensor) else 3)
    with pytest.raises(ValueError, match="replaced after projection"):
        bridge.read.exact_cpu_for(bridge.common)


def test_a_different_host_upper_cannot_be_used_as_the_actual_exact_length(bridge):
    bridge.common.seq_lens_cpu = torch.tensor([24, 1040, 0, 0], dtype=torch.int32)
    with pytest.raises(ValueError, match="replaced after projection"):
        bridge.read.exact_cpu_for(bridge.common)


@pytest.mark.parametrize("active", [False, True])
def test_actual_dispatch_selects_the_capture_bank_once(monkeypatch, active):
    from vllm_ascend.worker.v2.spec_decode.dflash.aclgraph import DFlashAclGraphManager, DFlashCudaGraphManager

    descriptor = SimpleNamespace()
    bucket = object()
    select = Mock(return_value=bucket)
    provider = SimpleNamespace(capture_identity_for=select)
    monkeypatch.setattr(DFlashCudaGraphManager, "dispatch", lambda *args: descriptor)
    manager = object.__new__(DFlashAclGraphManager)
    manager.speculator = SimpleNamespace(
        _draft_window_graph_binding_provider=provider,
        _draft_window_proposal=object() if active else None,
    )
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.spec_decode.dflash.aclgraph.resolve_parallel_draft_window_graph_route",
        lambda owner: "dflash",
    )
    assert manager.dispatch(1, 8, 8, 0) is descriptor
    assert manager.speculator._draft_window_selected_metadata is (bucket if active else None)
    if active:
        select.assert_called_once_with(route="dflash", descriptor=descriptor)
    else:
        select.assert_not_called()
    manager.speculator = SimpleNamespace()
    assert manager.dispatch(1, 8, 8, 0) is descriptor
    assert not hasattr(manager.speculator, "_draft_window_selected_metadata")


@pytest.fixture
def full_metadata_dispatch(monkeypatch):
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

    from vllm_ascend.worker.v2.spec_decode import draft_window_graph_runtime as runtime
    from vllm_ascend.worker.v2.spec_decode.draft_window import DraftMetadataInputs
    from vllm_ascend.worker.v2.spec_decode.draft_window_graph import DraftWindowGraphBindingProvider

    proposal: Any = SimpleNamespace(full_read_rows=(False, False))
    provider: Any = object.__new__(DraftWindowGraphBindingProvider)
    provider._quarantined = False
    provider._source = SimpleNamespace(policy=SimpleNamespace(normalized_method="dflash"))
    bank: Any = SimpleNamespace(
        max_num_reqs=2,
        window_block_table=torch.ones((2, 2), dtype=torch.int32),
        cpu_window_exact=torch.tensor([5, 7], dtype=torch.int32),
        device_window_exact=torch.tensor([5, 7], dtype=torch.int32),
    )
    inputs = DraftMetadataInputs(proposal, 2, 0, bank, provider)
    provider._active_inputs = inputs
    descriptor = BatchExecutionDescriptor(CUDAGraphMode.FULL, num_tokens=16, num_reqs=2, uniform_token_count=8)
    capture = SimpleNamespace(descriptor=descriptor, route=SimpleNamespace())
    carrier = object()
    provider.bind_read_carrier = Mock(return_value=carrier)
    owner = SimpleNamespace(
        _draft_window_proposal=proposal,
        _draft_window_graph_binding_provider=provider,
        _draft_window_selected_metadata=capture,
        _group_causal=False,
        model_state=SimpleNamespace(attn_metadata={"target.layer": object()}),
    )
    return SimpleNamespace(
        owner=owner,
        inputs=inputs,
        descriptor=descriptor,
        provider=provider,
        carrier=carrier,
        prepare=runtime.prepare_parallel_graph,
    )


def test_actual_replay_uses_the_current_provider_bank_without_a_second_generation_ledger(full_metadata_dispatch):
    c = full_metadata_dispatch
    assert c.prepare(c.owner, c.descriptor) is c.carrier
    c.provider.bind_read_carrier.assert_called_once()
    assert c.provider.bind_read_carrier.call_args.kwargs["inputs"] is c.inputs
    assert not hasattr(c.owner, "_draft_window_live_inputs")
    assert not hasattr(c.owner, "_draft_window_deferred_metadata")


@pytest.mark.parametrize("damage", ["proposal", "descriptor", "published", "retired"])
def test_actual_replay_rejects_stale_rows_bucket_or_already_published_generation(full_metadata_dispatch, damage):
    c = full_metadata_dispatch
    if damage == "proposal":
        c.owner._draft_window_proposal = object()
    elif damage == "descriptor":
        c.owner._draft_window_selected_metadata = SimpleNamespace(descriptor=object())
    elif damage == "published":
        object.__setattr__(c.inputs, "params", {"old": object()})
    else:
        c.provider._active_inputs = None
    with pytest.raises(RuntimeError, match="proposal|bucket|unconsumed|generation|retired"):
        c.prepare(c.owner, c.descriptor)
    c.provider.bind_read_carrier.assert_not_called()
