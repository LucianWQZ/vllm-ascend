# SPDX-License-Identifier: Apache-2.0
"""The fixed DFlash dummy helper and real Ascend builder, on CPU operands."""

from types import MethodType
from types import SimpleNamespace as NS

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.spec_decode.dflash import cudagraph
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.attention_v1 import AscendAttentionBackend, AscendAttentionMetadataBuilder, AscendMetadata
from vllm_ascend.worker.v2.spec_decode.dflash import capture
from vllm_ascend.worker.v2.spec_decode.draft_window_backends import DraftWindowGroupRoute, DraftWindowProviderStatus
from vllm_ascend.worker.v2.spec_decode.draft_window_graph import DraftWindowGraphBindingProvider


@pytest.fixture
def contract(monkeypatch):
    # CPU pinning stands in for the unavailable DMA allocator. Both the real
    # dummy InputBatch/BlockTables methods and Ascend metadata computation run.
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda value: value)
    monkeypatch.setattr(cudagraph, "build_attn_metadata", capture.build_attn_metadata)
    buffers = object.__new__(InputBuffers)
    buffers.device = torch.device("cpu")
    buffers.seq_lens = torch.zeros(2, dtype=torch.int32)
    buffers.query_start_loc = torch.zeros(3, dtype=torch.int32)
    buffers.input_ids = torch.zeros(16, dtype=torch.int32)
    buffers.positions = torch.zeros(16, dtype=torch.int64)
    buffers.is_padding = torch.zeros(16, dtype=torch.bool)
    tables = NS(
        input_block_tables=[torch.full((2, 32), 77, dtype=torch.int32)],
        num_blocks=NS(cpu=torch.full((1, 2), 32, dtype=torch.int32)),
        slot_mappings=torch.full((1, 16), 91, dtype=torch.int64),
        cp_size=1,
        _num_dirty_rows=0,
    )
    for name in ("get_dummy_block_tables", "get_dummy_slot_mappings"):
        setattr(tables, name, MethodType(getattr(BlockTables, name), tables))
    spec = FullAttentionSpec(block_size=128, num_kv_heads=1, head_size=4, dtype=torch.bfloat16)
    builder = object.__new__(AscendAttentionMetadataBuilder)
    builder.device, builder.use_pcp = torch.device("cpu"), False
    builder.kv_cache_spec, builder.speculative_config = spec, NS(parallel_drafting=True, use_dspark=lambda: False)
    builder.model_config = NS(runner_type="generate", hf_text_config=NS(model_type="qwen3"))
    builder.vllm_config = NS(use_v2_model_runner=True, model_config=builder.model_config)
    builder.attn_mask_builder = AttentionMaskBuilder(torch.device("cpu"))
    builder.decode_threshold = 8
    builder.metadata_cls = AscendMetadata
    group = AttentionGroup(AscendAttentionBackend, ["draft.a", "draft.b"], spec, 0, metadata_builders=[builder])
    config = NS(kv_cache_groups=[NS(kv_cache_spec=spec, layer_names=group.layer_names)])
    route = DraftWindowGroupRoute(0, tuple(group.layer_names), 128, DraftWindowProviderStatus.DIRECT_DENSE)
    provider = DraftWindowGraphBindingProvider(policy=1024, group=route, block_tables=tables)
    bank = provider.register_capture_identity(
        descriptor=BatchExecutionDescriptor(CUDAGraphMode.FULL, 16, 2, uniform_token_count=8)
    )
    owner = NS(_draft_window_graph_binding_provider=provider, input_buffers=buffers)
    calls = []
    actual_build = builder.build

    def build(*args, **kwargs):
        calls.append(kwargs["common_attn_metadata"])
        return actual_build(*args, **kwargs)

    monkeypatch.setattr(builder, "build", build)
    return NS(
        buffers=buffers,
        tables=tables,
        builder=builder,
        groups=[[group]],
        config=config,
        owner=owner,
        provider=provider,
        bank=bank,
        calls=calls,
    )


def prepare(c, *, skip=False, causal=False):
    return cudagraph._prepare_dflash_inputs_to_capture(
        2, 16, c.buffers, c.tables, c.groups, c.config, 4096, skip, causal
    )


@pytest.mark.parametrize(
    "method,causal", [("dflash", False), ("dflash", True), ("dspark", {0: False}), ("dflash2", {0: False})]
)
def test_original_helper_builds_window_reads_once_and_preserves_full_inputs(contract, method, causal):
    from vllm_ascend.attention.attention_v1 import FIAParamProvider

    c = contract
    if method == "dspark":
        c.builder.speculative_config.use_dspark = lambda: True
        c.builder.model_config.hf_text_config.model_type = "glm_moe_dsa"
    original_prepare = cudagraph._prepare_dflash_inputs_to_capture
    off = prepare(c, causal=causal)
    full_gpu, query, slots = c.buffers.seq_lens, c.buffers.query_start_loc, c.tables.slot_mappings
    before = (
        full_gpu.clone(),
        query.clone(),
        slots.clone(),
        c.buffers.positions.clone(),
        c.tables.input_block_tables[0].clone(),
    )
    c.calls.clear()
    with capture.draft_window_dflash_capture_scope(c.owner):
        on = prepare(c, causal=causal)
    assert cudagraph._prepare_dflash_inputs_to_capture is original_prepare
    assert cudagraph.build_attn_metadata is capture.build_attn_metadata and len(c.calls) == 1
    layer = on.attn_metadata["draft.a"]
    assert layer is on.attn_metadata["draft.b"] and layer is not off.attn_metadata["draft.a"]
    assert layer.block_tables.data_ptr() == c.bank.window_block_table.data_ptr()
    assert layer.seq_lens_cpu.data_ptr() == c.bank.cpu_window_exact.data_ptr()
    assert layer.seq_lens_list == [1151, 1151] and layer.actual_seq_lengths_q == [8, 16]
    assert layer.seq_lens_gpu.data_ptr() == full_gpu.data_ptr()
    assert layer.query_start_loc_gpu.data_ptr() == query.data_ptr()
    assert layer.slot_mapping.data_ptr() == slots.data_ptr() == on.slot_mappings["draft.a"].data_ptr()
    assert layer.causal is bool(causal if isinstance(causal, bool) else causal[0])
    assert layer.attn_mask is off.attn_metadata["draft.a"].attn_mask
    assert layer.num_decodes == 2 and layer.num_prefills == 0 and layer.num_decode_tokens == 16
    params = FIAParamProvider("draft.a", 2048 if method == "dflash2" else None, is_draft_model=True).resolve(
        on.attn_metadata
    )
    assert params["actual_seq_lengths"] == [8, 16] and params["actual_seq_lengths_kv"] == [1151, 1151]
    assert params["block_table"] is c.bank.window_block_table
    for actual, expected in zip((full_gpu, query, slots, c.buffers.positions, c.tables.input_block_tables[0]), before):
        torch.testing.assert_close(actual, expected)
    assert c.provider._active_bank is None and not c.provider._pending
    assert not hasattr(c.buffers, "_ascend_draft_window_capture_context")


def test_context_only_skip_and_off_keep_the_original_helper_without_read_preparation(contract, monkeypatch):
    c = contract
    bank = c.bank.window_block_table.clone(), c.bank.cpu_window_exact.clone()
    with capture.draft_window_dflash_capture_scope(c.owner):
        result = prepare(c, skip=True)
    assert result.attn_metadata is None and not c.calls
    for actual, before in zip((c.bank.window_block_table, c.bank.cpu_window_exact), bank):
        torch.testing.assert_close(actual, before)
    raw = cudagraph.build_attn_metadata
    with capture.draft_window_dflash_capture_scope(NS()) as provider:
        assert provider is None and cudagraph.build_attn_metadata is raw
        result = prepare(c)
    assert len(c.calls) == 1 and result.attn_metadata["draft.a"].seq_lens_list == [8, 8]


def test_builder_failure_restores_alias_without_modifying_full_inputs(contract, monkeypatch):
    c = contract
    original = cudagraph.build_attn_metadata

    def fail(*args, **kwargs):
        raise RuntimeError("real builder failed")

    monkeypatch.setattr(c.builder, "build", fail)
    with pytest.raises(RuntimeError, match="real builder failed"), capture.draft_window_dflash_capture_scope(c.owner):
        prepare(c)
    assert cudagraph.build_attn_metadata is original
    assert c.buffers.seq_lens.tolist() == [8, 8] and c.buffers.query_start_loc.tolist() == [0, 8, 16]
    assert c.tables.slot_mappings.tolist() == [[-1] * 16]
    assert c.provider._active_bank is None and not c.provider._pending


def test_native_dflash2_mask_and_real_fia1_parameters_survive_the_builder_callback(contract, monkeypatch):
    from vllm.forward_context import ForwardContext, override_forward_context

    from vllm_ascend.attention import attention_v1

    c = contract
    with capture.draft_window_dflash_capture_scope(c.owner):
        metadata = prepare(c, causal={0: False}).attn_metadata["draft.a"]
    impl = object.__new__(attention_v1.AscendAttentionBackendImpl)
    impl.key_cache = impl.value_cache = torch.zeros((1, 128, 1, 4))
    impl._get_kv_cache_view = lambda key, value: (key, value, 128)
    impl.sliding_window, impl.enable_c8_quant = 2048, False
    mask = impl._dflash2_native_swa_mask = torch.empty((2048, 2048), dtype=torch.int8)
    impl._use_max_workspace_for_fia_graph = False
    impl._layer_name, impl.num_heads, impl.num_kv_heads, impl.scale = "draft.a", 1, 1, 0.5
    captured = []
    monkeypatch.setattr(attention_v1, "get_capture_resource", lambda *args: torch.empty(1))
    monkeypatch.setattr(attention_v1, "register_task", lambda operation, kwargs, provider: captured.append(kwargs))
    query = torch.zeros((16, 1, 4))
    output = torch.empty_like(query)
    context = ForwardContext(no_compile_layers={}, attn_metadata={"draft.a": metadata}, slot_mapping={})
    context.is_draft_model = True
    with override_forward_context(context):
        actual, tokens = impl.full_graph_fia(query, query, query, metadata, output)
    assert actual is output and tokens == 16 and len(captured) == 1
    kwargs = captured[0]
    assert kwargs["actual_seq_lengths"] == [8, 16]
    assert kwargs["actual_seq_lengths_kv"] == [1151, 1151]
    assert kwargs["block_table"] is c.bank.window_block_table
    assert kwargs["atten_mask"] is mask and kwargs["pre_tokens"] == kwargs["next_tokens"] == 2047
    assert kwargs["sparse_mode"] == 4 and metadata.seq_lens_gpu.tolist() == [8, 8]


def test_original_prepare_forward_uses_full_context_slots_and_target_anchor(contract):
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

    c = contract
    with capture.draft_window_dflash_capture_scope(c.owner):
        metadata = prepare(c, causal=False).attn_metadata
    target_query = torch.tensor([0, 5, 11], dtype=torch.int32)
    context_slots = torch.full((1, 16), -1, dtype=torch.int64)
    hidden, positions = torch.zeros((16, 4)), torch.arange(16)
    calls = []
    owner = NS(
        _num_graph_context_tokens=lambda n: n * 8,
        _aux_staging=None,
        target_input_buffers=NS(query_start_loc=target_query),
        _num_rejected=torch.tensor([1, 0]),
        _context_slot_mappings=context_slots,
        _layer_group_idx=None,
        hidden_states=hidden,
        context_positions=positions,
        supports_decode_metadata_update=False,
        model=NS(precompute_and_store_context_kv=lambda *args: calls.append(("write", args))),
        prepare_context_anchor=lambda *args: calls.append(("anchor", args)),
    )
    owner._precompute_context_kv = MethodType(DFlashSpeculator._precompute_context_kv, owner)
    DFlashSpeculator._prepare_graph_forward(owner, 2, metadata)
    assert [kind for kind, _ in calls] == ["anchor", "write"]
    assert calls[0][1][1] is target_query and calls[0][1][2] is owner._num_rejected
    written = calls[1][1]
    assert written[0].data_ptr() == hidden.data_ptr() and written[1].data_ptr() == positions.data_ptr()
    assert written[2].data_ptr() == context_slots.data_ptr() and written[2].tolist() == [-1] * 16
    assert target_query.tolist() == [0, 5, 11]
