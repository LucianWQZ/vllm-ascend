# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU Graph metadata and source-owner contracts; no NPU execution.

These isolated CPU protocol tests establish exact-source/owner rejection;
model, numerical, capture and lifetime evidence remains separate.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture(params=["dspark", "dflash", "dflash2", "eagle3"])
def graph_source_initialization(request, monkeypatch):
    """Exercise actual owner initialization without loading a device model."""
    method = request.param
    from vllm_ascend.worker.v2.spec_decode.draft_window_eager import DraftWindowGraphInputMixin

    provider = object()
    create_provider = Mock(return_value=provider)
    if method == "eagle3":
        from vllm_ascend.worker.v2.spec_decode import draft_window, draft_window_eagle3_adapter

        initialize = draft_window_eagle3_adapter.Eagle3GraphIntegrationMixin._initialize_draft_window_source_capability
        monkeypatch.setattr(draft_window, "create_draft_window_graph_provider", create_provider)
    else:
        from vllm_ascend.worker.v2.spec_decode import draft_window_eager

        initialize = DraftWindowGraphInputMixin._initialize_draft_window_source_capability
        monkeypatch.setattr(draft_window_eager, "create_draft_window_graph_provider", create_provider)
    owner = SimpleNamespace(
        _draft_window_policy=object(),
        _speculator_name=method,
        method=method,
        attn_groups=object(),
        block_tables=object(),
        kv_cache_config=object(),
        _group_causal=False,
        vllm_config=object(),
        _initialize_eagle3_draft_window_state=Mock(),
    )
    return SimpleNamespace(
        owner=owner,
        initialize=initialize,
        create_provider=create_provider,
        provider=provider,
    )


def test_graph_owner_installs_only_its_admitted_source(graph_source_initialization):
    c = graph_source_initialization
    c.initialize(c.owner)
    expected = dict(
        policy=c.owner._draft_window_policy,
        runtime_route=c.owner.method,
        attn_groups=c.owner.attn_groups,
        block_tables=c.owner.block_tables,
        kv_cache_config=c.owner.kv_cache_config,
        vllm_config=c.owner.vllm_config,
    )
    if c.owner.method != "eagle3":
        expected["dflash2_graph_owner"] = c.owner if c.owner.method == "dflash2" else None
    c.create_provider.assert_called_once_with(**expected)
    assert c.owner._draft_window_graph_binding_provider is c.provider
    assert not hasattr(c.owner, "_draft_window_source_capability")
    assert not hasattr(c.owner, "_draft_window_runtime_state")
    assert not hasattr(c.owner, "_draft_window_runtime")
    if c.owner._speculator_name != "eagle3":
        assert c.owner._draft_window_bypass_active is False
        assert c.owner._draft_window_prefill_bypass_active is False
        assert c.owner._draft_window_dp_dummy_active is False
        assert c.create_provider.call_args.kwargs["dflash2_graph_owner"] is (
            c.owner if c.owner._speculator_name == "dflash2" else None
        )


def test_graph_source_rejection_has_no_eager_fallback_or_partial_owner(graph_source_initialization):
    c = graph_source_initialization
    failure = NotImplementedError("unsupported actual Graph source")
    c.create_provider.side_effect = failure
    with pytest.raises(NotImplementedError) as seen:
        c.initialize(c.owner)
    assert seen.value is failure
    assert not hasattr(c.owner, "_draft_window_graph_binding_provider")


def test_graph_owner_reinitialization_rejects_before_source_build(graph_source_initialization):
    c = graph_source_initialization
    c.owner._draft_window_graph_binding_provider = c.provider
    with pytest.raises(RuntimeError, match="already initialized"):
        c.initialize(c.owner)
    c.create_provider.assert_not_called()


def test_window_off_without_graph_policy_does_not_construct_source(graph_source_initialization):
    c = graph_source_initialization
    del c.owner._draft_window_policy
    c.initialize(c.owner)
    c.create_provider.assert_not_called()
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


@pytest.mark.parametrize("upper", [1153, 1160])
def test_common_builder_consumes_the_window_exact_tensor_not_the_original_upper(upper):
    from vllm_ascend.attention.utils import AscendCommonAttentionMetadata, _select_seq_lens
    from vllm_ascend.worker.v2.spec_decode.draft_window_read import materialize_draft_read

    offsets = torch.tensor([0, 8, 16], dtype=torch.int32)
    exact = torch.tensor([1151, 1152], dtype=torch.int32)
    table = torch.arange(24, dtype=torch.int32).reshape(2, 12)
    slots = torch.arange(16, dtype=torch.int64)
    common = AscendCommonAttentionMetadata(
        query_start_loc=offsets,
        query_start_loc_cpu=offsets,
        seq_lens=exact,
        num_reqs=2,
        num_actual_tokens=16,
        max_query_len=8,
        max_seq_len=upper,
        block_table_tensor=table,
        slot_mapping=slots,
        seq_lens_cpu_upper_bound=torch.full((2,), upper, dtype=torch.int32),
    )
    materialize_draft_read(
        common,
        (12, 12),
        128,
        exact_cpu=exact,
        starts_cpu=torch.tensor([0, 128], dtype=torch.int32),
        retain=lambda *objects: None,
    )
    selected = _select_seq_lens(
        common,
        None,
        SimpleNamespace(parallel_drafting=True, use_dspark=lambda: False),
        SimpleNamespace(model_config=object()),
    )
    assert selected.data_ptr() == common._draft_window_exact_cpu.data_ptr()
    assert selected.data_ptr() == common.seq_lens_cpu_upper_bound.data_ptr()
    assert selected.tolist() == [1151, 1024]
    assert common.slot_mapping is slots and common.query_start_loc is offsets
    assert table.tolist() == [list(range(12)), list(range(12, 24))]


def test_dispatch_keeps_the_original_framework_selection_without_a_bank_mirror(monkeypatch):
    from vllm_ascend.worker.v2.spec_decode.dflash.aclgraph import DFlashAclGraphManager, DFlashCudaGraphManager

    descriptor = SimpleNamespace()
    monkeypatch.setattr(DFlashCudaGraphManager, "dispatch", lambda *args: descriptor)
    manager = object.__new__(DFlashAclGraphManager)
    for owner in (SimpleNamespace(), SimpleNamespace(_draft_window_graph_binding_provider=object())):
        manager.speculator = owner
        assert manager.dispatch(1, 8, 8, 0) is descriptor
        assert not hasattr(owner, "_draft_window_selected_metadata")


@pytest.mark.parametrize("wrong_descriptor", [False, True])
def test_real_metadata_preparation_selects_the_final_descriptor_once(monkeypatch, wrong_descriptor):
    from dataclasses import replace

    import numpy as np

    from tests.ut.worker.v2.test_draft_window_graph_buffers import graph_buffers
    from vllm_ascend.worker.v2.spec_decode.draft_window_eager import (
        DraftWindowGraphInputMixin,
    )

    c = graph_buffers.__wrapped__()
    owner = DraftWindowGraphInputMixin()
    owner._draft_window_graph_binding_provider, owner._draft_window_bypass_active = c.provider, False
    owner.block_tables, owner.num_query_per_req, owner.draft_attn_layer_names = c.owner, 1, {"draft.layer"}
    owner.input_batch = SimpleNamespace(num_reqs=2, idx_mapping_np=np.array([0, 1]))
    snapshot = Mock(return_value=torch.tensor([11, 14], dtype=torch.int32))

    class Inputs:
        @property
        def seq_lens(self):
            return snapshot()

    owner.input_buffers = Inputs()
    select = Mock(wraps=c.provider.capture_identity_for)
    monkeypatch.setattr(c.provider, "capture_identity_for", select)
    desc = replace(c.capture.descriptor, max_query_len=9) if wrong_descriptor else c.capture.descriptor
    context = (False, False)
    if wrong_descriptor:
        with pytest.raises(RuntimeError, match="runtime descriptor"), c.provider.proposal_scope(context):
            owner._build_attn_metadata(2, desc, np.array([0, 1, 2]), torch.tensor([99, 99]), 1)
        snapshot.assert_not_called()
    else:
        with c.provider.proposal_scope(context):
            assert owner._build_attn_metadata(2, desc, np.array([0, 1, 2]), torch.tensor([99, 99]), 1) is None
            assert c.provider._active_bank is c.capture
            assert c.capture.cpu_window_exact.tolist() == [7, 6]
            c.provider.bind_current(desc)
            c.provider.finish_update()
        snapshot.assert_called_once_with()
    select.assert_called_once_with(descriptor=desc)
    assert not hasattr(owner, "_draft_window_selected_metadata")
