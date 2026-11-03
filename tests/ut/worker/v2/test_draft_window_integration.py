# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Real speculator/metadata interfaces with only device boundaries isolated.

CPU ownership and signature checks do not claim engine or NPU execution.
"""

import inspect
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from vllm.config import AttentionConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator import TargetDependentARSpeculator

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.worker.v2 import attn_utils
from vllm_ascend.worker.v2.spec_decode import draft_window_read
from vllm_ascend.worker.v2.spec_decode.dflash import speculator as dflash_module
from vllm_ascend.worker.v2.spec_decode.dspark import speculator as dspark_module
from vllm_ascend.worker.v2.spec_decode.target_dependent_ar import speculator as ar_module


@pytest.fixture(params=["dflash", "dspark", "autoregressive"])
def contract(request, monkeypatch):
    """Instantiate real classes; isolate native proposal and final builder work."""
    module, cls, base = {
        "dflash": (dflash_module, dflash_module.AscendDFlashSpeculator, DFlashSpeculator),
        "dspark": (dspark_module, dspark_module.AscendDSparkSpeculator, DSparkSpeculator),
        "autoregressive": (ar_module, ar_module.AscendTargetDependentARSpeculator, TargetDependentARSpeculator),
    }[request.param]
    proposal_hook = Mock(return_value=object())
    proposal_signature = inspect.signature(base.propose)

    def native_propose(self, *args, **kwargs):
        bound = proposal_signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        return proposal_hook(self, bound.arguments)

    monkeypatch.setattr(base, "propose", native_propose)
    prefill_hook = Mock()
    if request.param == "autoregressive":
        prefill_signature = inspect.signature(base._prefill)

        def native_prefill(self, *args, **kwargs):
            bound = prefill_signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            return prefill_hook(self, bound.arguments)

        monkeypatch.setattr(base, "_prefill", native_prefill)
        monkeypatch.setattr(ar_module, "torch_gather_wrapper", nullcontext)
        monkeypatch.setattr(ar_module, "disable_target_pcp_for_replicated_draft", lambda _: nullcontext())
    monkeypatch.setattr(attn_utils._BUILD_ATTN_METADATA_MODULE, "build_attn_metadata", attn_utils.build_attn_metadata)
    monkeypatch.setattr(attn_utils, "ring_state_update_skipped", lambda: False)
    obj: Any = object.__new__(cls)
    full_table = torch.arange(8, dtype=torch.int32).view(1, 8)
    slots = torch.arange(3, dtype=torch.int64).view(1, 3)
    positions = torch.tensor([18, 19, 20])
    obj.input_buffers = SimpleNamespace(
        positions=positions, seq_lens=torch.tensor([21], dtype=torch.int32), query_start_loc=torch.tensor([0, 3])
    )
    obj.block_tables = SimpleNamespace(input_block_tables=[full_table], slot_mappings=slots)
    obj.draft_is_prefilling = torch.tensor([False])
    obj.max_model_len, obj.draft_max_seq_len, obj.dcp_size = 1024, 25, 1
    if request.param != "autoregressive":
        obj.num_query_per_req = 3
    obj.max_num_tokens, obj.max_num_reqs = 3, 1
    obj.attn_architecture, obj.use_dcp, obj.replicated_pcp = "GQA", False, False
    obj.enable_adaptive_verification = False
    obj.vllm_config = SimpleNamespace(parallel_config=object(), attention_config=AttentionConfig())
    obj.requires_non_causal = True
    obj.draft_vllm_config = obj.vllm_config
    obj.model_state = object()
    obj._sparse_kv_offload_metadata = SimpleNamespace(build_kwargs=Mock(return_value=None))
    obj.input_batch = SimpleNamespace(
        is_prefilling_np=np.array([False]),
        query_start_loc_np=np.array([0, 3]),
        seq_lens_cpu_upper_bound=torch.tensor([22]),
    )
    obj._get_seq_lens_cpu = Mock(return_value=torch.tensor([22]))
    obj.draft_attn_layer_names = {"draft.a", "draft.b"}
    obj._prepare_replicated_prefill_attn = lambda metadata, slots, *args, **kwargs: (metadata, slots)
    obj._prepare_draft_dcp_metadata_inputs = Mock(return_value=(None, torch.tensor([False])))
    backend = Mock()
    backend.consumes_pcp_context = False
    backend.build.side_effect = lambda **kwargs: kwargs["common_attn_metadata"]
    obj.attn_groups = [[SimpleNamespace(layer_names=["draft.a", "draft.b"], get_metadata_builder=lambda _: backend)]]
    obj.kv_cache_config = SimpleNamespace(kv_cache_groups=[object()])
    return SimpleNamespace(
        obj=obj,
        mode=CUDAGraphMode,
        module=module,
        backend=backend,
        base=base,
        proposal_hook=proposal_hook,
        method=request.param,
        table=full_table,
        slots=slots,
        positions=positions,
        read=draft_window_read,
        prefill_hook=prefill_hook,
    )


def _build(contract, **extra):
    return contract.obj._build_attn_metadata(
        1,
        SimpleNamespace(num_reqs=1, num_tokens=3, cg_mode=contract.mode.NONE),
        np.array([0, 3], dtype=np.int32),
        torch.tensor([22], dtype=torch.int32),
        3,
        **extra,
    )


def test_explicit_full_write_mapping_preserves_upstream_signature(contract):
    # This parameter was added upstream after the old Ascend override. Call the
    # actual override with it so a missing passthrough produces a real TypeError.
    assert "slot_mappings" in inspect.signature(contract.base._build_attn_metadata).parameters
    explicit_slots = torch.tensor([[70, 71, 72]])
    result = _build(contract, slot_mappings=explicit_slots)
    assert result["draft.a"].slot_mapping.untyped_storage().data_ptr() == explicit_slots.untyped_storage().data_ptr()
    assert result["draft.a"].slot_mapping.tolist() == [70, 71, 72]
    assert contract.slots.tolist() == [[0, 1, 2]]


@pytest.mark.parametrize("contract", ["dspark"], indirect=True)
@pytest.mark.parametrize("architecture", [None, "DSA", "QSA"])
def test_dspark_other_backends_preserve_native_metadata_dispatch(contract, monkeypatch, architecture):
    contract.obj.attn_architecture = architecture
    build = contract.obj._build_attn_metadata
    signature = inspect.signature(contract.base._build_attn_metadata)
    native = Mock(return_value=object())
    monkeypatch.setattr(contract.base, "_build_attn_metadata", native)
    monkeypatch.setattr(contract.module, "build_attn_metadata_factory", Mock(side_effect=AssertionError))
    slots = torch.tensor([[17, 18, 19]])
    local_lengths = torch.tensor([21])

    result = build(
        1,
        SimpleNamespace(num_reqs=1, num_tokens=3, cg_mode=contract.mode.NONE),
        np.array([0, 3], dtype=np.int32),
        torch.tensor([22], dtype=torch.int32),
        3,
        causal=False,
        dcp_local_seq_lens=local_lengths,
        slot_mappings=slots,
    )

    assert result is native.return_value
    call = signature.bind(contract.obj, *native.call_args.args, **native.call_args.kwargs).arguments
    assert call["num_reqs"] == 1 and call["step"] == 3
    assert call["causal"] is False
    assert call["slot_mappings"] is slots
    assert call["dcp_local_seq_lens"] is local_lengths
    assert call["query_start_loc_np"].tolist() == [0, 3]
    assert call["seq_lens_cpu_upper_bound"].tolist() == [22]
    assert call["batch_desc"].cg_mode is contract.mode.NONE


def test_callback_final_writer_updates_real_common_cpu_mirrors_and_keeps_full_operands(contract):
    retained: list[Any] = []
    read_retained: list[Any] = []
    called: list[Any] = []
    read = contract.read
    exact = torch.tensor([21], dtype=torch.int32)

    def callback(gid, common):
        called.append(gid)
        assert common.seq_lens_cpu.tolist() == [25]  # Upstream scheduling upper before projection.
        read.materialize_draft_read(
            common,
            (8,),
            4,
            exact_cpu=exact,
            starts_cpu=torch.tensor([12], dtype=torch.int32),
            retain=lambda *items: read_retained.extend(items),
        )

    runtime = SimpleNamespace(
        metadata_callback=Mock(return_value=callback),
        retain_metadata=retained.append,
        reads_active=True,
    )
    contract.obj._draft_window_runtime = runtime
    with attn_utils.build_attn_metadata_wrapper():
        result = _build(contract)
    common = result["draft.a"]
    assert called == [0]
    assert result["draft.b"] is common
    assert common.block_table_tensor.tolist() == [[3, 4, 5]]
    assert common.seq_lens_cpu is common._seq_lens_cpu is common._draft_window_exact_cpu
    assert common.seq_lens_cpu.tolist() == [9]
    # The completed exact snapshot is also the tight upper bound; do not
    # preserve a second optimistic-length projection after rebasing.
    assert common.seq_lens_cpu_upper_bound.tolist() == [9]
    assert common.seq_lens.tolist() == [9]
    assert common.positions.untyped_storage().data_ptr() == contract.positions.untyped_storage().data_ptr()
    assert common.slot_mapping.untyped_storage().data_ptr() == contract.slots.untyped_storage().data_ptr()
    assert contract.obj.input_buffers.seq_lens.tolist() == [21]
    assert contract.table.tolist() == [list(range(8))]
    assert retained == [result]
    assert any(value is common.block_table_tensor for value in read_retained)
    runtime.metadata_callback.assert_called_once_with(3, 1)
    contract.backend.build.assert_called_once()


def test_off_metadata_retains_full_read_and_write_storage(contract):
    result = _build(contract)
    common = result["draft.a"]
    assert common.block_table_tensor.untyped_storage().data_ptr() == contract.table.untyped_storage().data_ptr()
    assert (
        common.seq_lens.untyped_storage().data_ptr() == contract.obj.input_buffers.seq_lens.untyped_storage().data_ptr()
    )
    assert common.slot_mapping.untyped_storage().data_ptr() == contract.slots.untyped_storage().data_ptr()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("profiling", [False, True])
@pytest.mark.parametrize("num_speculative_tokens", [0, 3])
def test_propose_uses_real_upstream_signature_scope_and_dp_rules(contract, enabled, profiling, num_speculative_tokens):
    batch = SimpleNamespace(is_prefilling_np=np.array([False]), num_reqs=1)
    arguments = dict(
        input_batch=batch,
        attn_metadata=object(),
        slot_mappings=object(),
        last_hidden_states=object(),
        aux_hidden_states=[object()],
        num_sampled=object(),
        num_rejected=object(),
        last_sampled=object(),
        next_prefill_tokens=object(),
        temperature=object(),
        seeds=object(),
        dp_sync_state=object(),
        dummy_run=profiling,
        skip_attn_for_dummy_run=profiling,
        mm_inputs=object(),
        is_profile=profiling,
        num_speculative_tokens=num_speculative_tokens,
    )
    events = []

    @contextmanager
    def proposal_context(speculator, input_batch, dummy_run, is_profile, **kwargs):
        assert (speculator, input_batch, dummy_run, is_profile) == (contract.obj, batch, profiling, profiling)
        expected = (
            {"num_speculative_tokens": num_speculative_tokens}
            if contract.method == "autoregressive"
            else {"context_kv_only": num_speculative_tokens == 0}
        )
        assert kwargs == expected
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    if enabled:
        contract.obj._draft_window_runtime = SimpleNamespace(proposal_context=proposal_context)
    result = contract.obj.propose(**arguments)
    assert result is contract.proposal_hook.return_value
    passed = contract.proposal_hook.call_args.args[1]
    for name, value in arguments.items():
        reset_dp = name == "dp_sync_state" and profiling and contract.method != "autoregressive"
        assert passed[name] is (None if reset_dp else value)
    assert events == (["enter", "exit"] if enabled else [])


@pytest.mark.parametrize("contract", ["autoregressive"], indirect=True)
def test_eagle_first_pass_builds_fresh_draft_metadata_preserving_target_and_write_identity(contract):
    target_common = object()
    target_metadata = {"draft.a": target_common, "target.only": object()}
    full_write = {"draft.a": torch.tensor([31, 32, 33])}
    frozen_read = SimpleNamespace(seq_lens_cpu=torch.tensor([9]))
    new_metadata = {"draft.a": frozen_read}
    contract.obj._draft_window_runtime = SimpleNamespace(reads_active=True)
    contract.obj._build_attn_metadata = Mock(return_value=new_metadata)
    contract.obj._prefill(1, 3, target_metadata, full_write, None, contract.mode.NONE)
    passed = contract.prefill_hook.call_args.args[1]
    assert passed["attn_metadata"] == new_metadata
    assert passed["attn_metadata"]["draft.a"] is frozen_read
    assert passed["slot_mappings"] is full_write
    assert target_metadata["draft.a"] is target_common
    assert "target.only" in target_metadata
    build = contract.obj._build_attn_metadata.call_args.kwargs
    assert build["step"] == 0
    assert build["query_start_loc_np"] is contract.obj.input_batch.query_start_loc_np


@pytest.mark.parametrize("contract", ["autoregressive"], indirect=True)
def test_eagle_window_update_cannot_overwrite_inflight_read(contract):
    read = torch.tensor([9], dtype=torch.int32)
    metadata = {"draft.a": SimpleNamespace(seq_lens_cpu=read, seq_lens_list=[9])}
    contract.obj._draft_window_runtime = SimpleNamespace(reads_active=True)
    contract.obj._get_seq_lens_cpu.side_effect = AssertionError("Window updater read Full CPU lengths")
    contract.obj._update_decode_attn_metadata(metadata, 2, 1)
    assert metadata["draft.a"].seq_lens_cpu is read
    assert read.tolist() == [9]


@pytest.mark.parametrize("contract", ["autoregressive"], indirect=True)
def test_current_dense_eagle_uses_upstream_unfused_selection(contract):
    from vllm.v1.attention.backend import AttentionMetadataBuilder

    from vllm_ascend.attention.attention_v1 import AscendAttentionBackend, AscendAttentionMetadataBuilder

    assert AttentionMetadataBuilder.supports_draft_decode_metadata_update is False
    assert AscendAttentionMetadataBuilder.supports_draft_decode_metadata_update is False
    contract.obj.num_speculative_steps = 8
    contract.obj.attn_groups = [
        [
            SimpleNamespace(
                supports_draft_decode_metadata_update=AscendAttentionMetadataBuilder.supports_draft_decode_metadata_update,
                backend=AscendAttentionBackend,
            )
        ]
    ]
    DraftModelSpeculator._configure_decode_metadata_update(contract.obj)
    assert contract.obj.supports_decode_metadata_update is False


@pytest.mark.parametrize("contract", ["autoregressive"], indirect=True)
@pytest.mark.parametrize("enabled", [False, True])
def test_eagle_set_attn_installs_runtime_only_after_backend_selection(contract, monkeypatch, enabled):
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackend
    from vllm_ascend.worker.v2.spec_decode import draft_window_native_attention, draft_window_runtime

    signature = inspect.signature(DraftModelSpeculator.set_attn)
    hook = Mock()

    def set_attn(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        return hook(self, bound.arguments)

    monkeypatch.setattr(contract.base, "set_attn", set_attn)
    monkeypatch.setattr(ar_module, "set_current_vllm_config", lambda _: nullcontext())
    monkeypatch.setattr(ar_module, "_get_graph_update_backend", lambda _: AscendAttentionBackend)
    monkeypatch.setattr(draft_window_native_attention, "report_owner_draft_native_attention", Mock())
    runtime = Mock(return_value=object())
    monkeypatch.setattr(draft_window_runtime, "DraftWindowRuntime", runtime)
    policy = object()
    if enabled:
        contract.obj.draft_window_policy = policy
    arguments = [object() for _ in range(5)]
    contract.obj.set_attn(*arguments)
    assert list(hook.call_args.args[1].values())[1:] == arguments
    assert contract.obj.attn_backend is AscendAttentionBackend
    assert contract.obj.attn_architecture == "GQA"
    if enabled:
        runtime.assert_called_once_with(contract.obj, policy)
        assert contract.obj._draft_window_runtime is runtime.return_value
    else:
        runtime.assert_not_called()
        assert not hasattr(contract.obj, "_draft_window_runtime")


@pytest.mark.parametrize("contract", ["autoregressive"], indirect=True)
@pytest.mark.parametrize("consumer", ["sink", "c8"])
def test_eagle_window_first_pass_keeps_multitoken_phase_at_real_consumer(contract, monkeypatch, consumer):
    """Execute first-pass assembly and actual consumer dispatch/native arguments.

    Device kernels are recording stubs; real consumer methods and metadata
    types execute unchanged. This proves the host query-width
    contract, not NPU arithmetic or cache-write contents.
    """
    from vllm_ascend.attention import attention_v1

    state, metadata_type = attention_v1.AscendAttentionState, attention_v1.AscendMetadata
    native_calls = []

    def native(query, *args, **kwargs):
        native_calls.append((query, kwargs))
        return torch.ones_like(query), None

    monkeypatch.setattr(attention_v1, "_EXTRA_CTX", SimpleNamespace(capturing=False))
    monkeypatch.setattr(
        attention_v1,
        "torch_npu",
        SimpleNamespace(
            npu_fused_infer_attention_score=native,
            npu_fused_infer_attention_score_v2=native,
        ),
    )
    contract.obj.input_batch.is_prefilling_np[:] = True
    contract.obj.draft_is_prefilling[:] = True
    contract.obj._draft_window_runtime = SimpleNamespace(
        reads_active=True,
        metadata_callback=lambda *args: lambda gid, common: None,
        retain_metadata=lambda metadata: None,
    )

    def build_typed_metadata(**kwargs):
        common = kwargs["common_attn_metadata"]
        return metadata_type(
            attn_state=common.attn_state,
            num_actual_tokens=3,
            num_prefills=1,
            num_decodes=0,
            seq_lens=common.seq_lens,
            seq_lens_cpu=common.seq_lens_cpu,
            seq_lens_list=common.seq_lens_cpu.tolist(),
            actual_seq_lengths_q=common.query_start_loc_cpu[1:].tolist(),
            block_tables=common.block_table_tensor,
            slot_mapping=common.slot_mapping,
            model_runner_type="generate",
        )

    contract.backend.build.side_effect = build_typed_metadata
    with attn_utils.build_attn_metadata_wrapper():
        metadata = contract.obj._build_attn_metadata(
            1,
            SimpleNamespace(num_reqs=1, num_tokens=3, cg_mode=contract.mode.NONE),
            np.array([0, 3], dtype=np.int32),
            torch.tensor([22], dtype=torch.int32),
            0,
        )["draft.a"]
    query = torch.zeros(3, 1, 4)
    key = torch.zeros_like(query)
    output = torch.full_like(query, float("nan"))
    cache = torch.zeros(1, 128, 1, 4)
    impl = object.__new__(
        attention_v1.AscendAttentionBackendImpl if consumer == "sink" else attention_v1.AscendC8AttentionBackendImpl
    )
    for field, value in dict(
        sinks=torch.ones(1),
        sliding_window=None,
        num_heads=1,
        num_kv_heads=1,
        head_size=4,
        scale=0.5,
        attn_type=None,
        key_cache=cache,
        value_cache=cache,
        _get_fia_params=lambda *args: (cache, cache, 128, metadata.block_tables, metadata.seq_lens_list),
    ).items():
        setattr(impl, field, value)
    if consumer == "sink":
        impl.forward_fused_infer_attention(query, key, key, metadata, output)
        assert list(native_calls[0][1]["actual_seq_qlen"]) == [3]
    else:
        impl.vllm_config = SimpleNamespace(kv_transfer_config=None)
        impl._prepare_c8_scales = lambda *args: None
        impl._quantize_kv_to_int8 = lambda k, v, *args: (k, v)
        impl._reshape_and_cache = lambda q, k, v, *args: (q, k, v, None)
        impl._nz_5d_view = lambda value, *args: value
        impl._forward_c8_chunked_prefill = Mock(side_effect=lambda *args: output.fill_(1))
        layer = SimpleNamespace(layer_name="draft.a", _c8_k_aq_scale_nz_bnsd=None, _c8_v_aq_scale_nz_bnsd=None)
        impl.forward(layer, query, key, key, (cache, cache), metadata, output)
        assert not torch.isnan(output).any(), "DecodeOnly incorrectly consumed only one row of the prefill query"
        impl._forward_c8_chunked_prefill.assert_called_once()
        assert not native_calls
    assert metadata.attn_state is state.ChunkedPrefill
    assert metadata.actual_seq_lengths_q == [3]


@pytest.mark.parametrize("contract", ["autoregressive"], indirect=True)
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("prefilling", [False, True])
def test_eagle_replicated_prefill_builds_once_after_full_mapping_restore(contract, enabled, prefilling):
    """Run the real replicated helper and metadata assembly, not its fixture stub."""
    obj = contract.obj
    del obj._prepare_replicated_prefill_attn
    obj.replicated_pcp = True
    from vllm_ascend.worker.v2.input_batch import AscendInputBatch

    batch = object.__new__(AscendInputBatch)
    vars(batch).update(vars(obj.input_batch))
    obj.input_batch = batch
    obj.kv_cache_config.kv_cache_groups = [SimpleNamespace(layer_names=["draft.a", "draft.b"])]
    obj.input_batch.num_reqs = 1
    obj.input_batch.is_dummy = False
    obj.input_batch.idx_mapping = torch.tensor([0])
    obj.input_batch.query_start_loc = obj.input_buffers.query_start_loc
    obj.input_batch.positions = contract.positions
    obj.input_batch.is_prefilling_np[:] = prefilling
    obj.block_tables.gather_block_tables = Mock()
    restored_slots = torch.tensor([[31, 32, 33]])
    obj.block_tables.slot_mappings = restored_slots
    obj.block_tables.compute_slot_mappings = Mock(return_value=restored_slots)
    if enabled:
        obj._draft_window_runtime = SimpleNamespace(
            reads_active=True,
            metadata_callback=Mock(return_value=None),
            retain_metadata=Mock(),
        )
    target_value = object()
    target_metadata = {"draft.a": target_value}
    target_slots = {"draft.a": torch.tensor([7, 8, 9])}
    obj._prefill(1, 3, target_metadata, target_slots, None, contract.mode.NONE)
    obj.block_tables.gather_block_tables.assert_called_once_with(obj.input_batch.idx_mapping, num_reqs_padded=1)
    obj.block_tables.compute_slot_mappings.assert_called_once_with(
        obj.input_batch.idx_mapping,
        obj.input_batch.query_start_loc,
        obj.input_batch.positions,
        num_tokens_padded=3,
    )
    assert contract.backend.build.call_count == 1
    passed = contract.prefill_hook.call_args.args[1]
    metadata = passed["attn_metadata"]["draft.a"]
    assert metadata is not target_value
    assert metadata.slot_mapping.untyped_storage().data_ptr() == restored_slots.untyped_storage().data_ptr()
    assert (
        passed["slot_mappings"]["draft.a"].untyped_storage().data_ptr() == restored_slots.untyped_storage().data_ptr()
    )
    assert metadata.query_start_loc_cpu.tolist() == [0, 3]
    assert target_metadata["draft.a"] is target_value
    assert target_slots["draft.a"].tolist() == [7, 8, 9]
    expected = AscendAttentionState.ChunkedPrefill if enabled else AscendAttentionState.DecodeOnly
    assert metadata.attn_state is expected
    if enabled:
        obj._draft_window_runtime.metadata_callback.assert_called_once_with(0, 1)
        obj._draft_window_runtime.retain_metadata.assert_called_once()


@pytest.mark.parametrize("contract", ["dspark"], indirect=True)
@pytest.mark.parametrize("padded_rows", [1, 3])
def test_dspark_mla_reuses_prepared_exact_cpu_without_another_transfer(contract, monkeypatch, padded_rows):
    # Execute the actual DSpark helper, runtime, common metadata and proposal
    # owner. Only backend discovery/completion/native builder are isolated.
    from vllm_ascend.worker.v2.spec_decode import draft_window_policy, draft_window_runtime

    runtime_module = draft_window_runtime
    runtime_class = runtime_module.DraftWindowRuntime
    monkeypatch.setattr(runtime_class, "_validate_route", staticmethod(lambda spec: (0,)))
    completion = SimpleNamespace(record=lambda stream: None, query=lambda: False, synchronize=lambda: None)
    monkeypatch.setattr(torch.npu, "Event", lambda: completion)
    monkeypatch.setattr(torch.npu, "current_stream", lambda: object())
    obj = contract.obj
    del obj._prepare_draft_dcp_metadata_inputs  # Run the real completed CPU copy.
    obj.attn_architecture = "MLA"
    obj.input_buffers.seq_lens = torch.tensor([21] + [0] * (padded_rows - 1), dtype=torch.int32)
    obj.input_buffers.query_start_loc = torch.tensor([0, 3] + [3] * (padded_rows - 1), dtype=torch.int32)
    obj.block_tables.block_tables = [SimpleNamespace(gpu=contract.table)]
    obj.block_tables.num_blocks = SimpleNamespace(np=np.array([[8]], dtype=np.int32))
    obj.block_tables.kernel_block_sizes = [4]
    obj.vllm_config.max_concurrent_batches = 2
    obj.draft_model_config = SimpleNamespace(enforce_eager=True)
    obj.input_batch.num_reqs = 1
    obj.input_batch.req_ids = ["current-request"]
    obj.input_batch.idx_mapping_np = np.array([0])
    obj.input_batch.idx_mapping = torch.tensor([0])
    obj.input_batch.num_computed_prefill_tokens_np = np.array([16])
    obj.input_batch.prefill_len_np = np.array([16])
    obj.input_batch.num_scheduled_tokens = np.array([3])
    obj.input_batch.num_draft_tokens_per_req = np.array([2])
    runtime = runtime_class(
        obj, draft_window_policy.resolve_draft_window_policy(requested_window_size=8, method="dspark")
    )
    obj._draft_window_runtime = runtime
    original_to, original_copy = torch.Tensor.to, torch.Tensor.copy_
    window_cpu_copies, prepared_cpu_copies = [], []

    def observe_to(tensor, *args, **kwargs):
        if kwargs == {"device": "cpu", "dtype": torch.int32, "copy": True}:
            window_cpu_copies.append(tensor)
        return original_to(tensor, *args, **kwargs)

    def observe_copy(tensor, source, *args, **kwargs):
        if source.data_ptr() == obj.input_buffers.seq_lens.data_ptr() and source.shape == (1,):
            prepared_cpu_copies.append(tensor)
        return original_copy(tensor, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", observe_to)
    monkeypatch.setattr(torch.Tensor, "copy_", observe_copy)
    with runtime.proposal_context(obj, obj.input_batch):
        owner = runtime._owner
        result = obj._build_attn_metadata(
            1,
            SimpleNamespace(num_reqs=padded_rows, num_tokens=3, cg_mode=contract.mode.NONE),
            np.array([0, 3], dtype=np.int32),
            torch.tensor([30], dtype=torch.int32),  # Deliberately not exact.
            3,
        )
    assert len(prepared_cpu_copies) == 1  # Existing blocking helper copy remains.
    assert len(window_cpu_copies) == 0  # The feature adds no second exact copy.
    prepared = prepared_cpu_copies[0]
    assert any(isinstance(value, torch.Tensor) and value.data_ptr() == prepared.data_ptr() for value in owner.retained)
    assert owner.completion is not None
    assert owner.completion is completion
    common = result["draft.a"]
    assert common.seq_lens_cpu.tolist() == [9] + [0] * (padded_rows - 1)
    assert common.seq_lens.tolist() == [9] + [0] * (padded_rows - 1)
    assert common.slot_mapping.untyped_storage().data_ptr() == contract.slots.untyped_storage().data_ptr()
    assert obj.input_buffers.seq_lens.tolist() == [21] + [0] * (padded_rows - 1)
    assert contract.table.tolist() == [list(range(8))]
