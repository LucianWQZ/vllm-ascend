# SPDX-License-Identifier: Apache-2.0
"""Repair2: real group builder -> capture helper -> per-layer FIA observer."""

from types import SimpleNamespace

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor, CudaGraphManager
from vllm.v1.worker.gpu.spec_decode.dflash import cudagraph as upstream_dflash

from tests.ut.spec_decode.test_draft_window_graph_characterization import (
    _characterization_owner,
    _runtime_context,
)
from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl, AscendAttentionMetadataBuilder
from vllm_ascend.worker.v2.spec_decode.dflash.capture import (
    ascend_prepare_dflash_inputs_to_capture,
    draft_window_dflash_capture_scope,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_graph_characterization import (
    attach_parallel_graph_capture_operand_observer,
    install_parallel_draft_window_graph_characterization,
    record_parallel_graph_capture_operator_operand,
)

ATTRIBUTE = "_draft_window_graph_capture_operand_context"


def _real_builder(monkeypatch):
    # Only pinned-memory/device services are substituted. Both assembly and
    # AscendAttentionMetadataBuilder.build_for_cudagraph_capture/build are real.
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self, *a, **kw: self)
    builder = AscendAttentionMetadataBuilder.__new__(AscendAttentionMetadataBuilder)
    builder.pcp_enabled = False
    builder.pcp_size = 1
    builder.device = torch.device("cpu")
    builder.decode_threshold = 8
    builder.kv_cache_spec = SimpleNamespace()
    builder.speculative_config = SimpleNamespace(parallel_drafting=True)
    builder.model_config = SimpleNamespace(runner_type="generate")
    builder.attn_mask_builder = SimpleNamespace(get_attention_mask=lambda *a: None)
    return builder


def _capture_fixture(monkeypatch, *, separate=False):
    owner, descriptor = _characterization_owner()
    builder = _real_builder(monkeypatch)
    layers = ["draft.layers.0.self_attn", "draft.layers.1.self_attn"]
    groups = [SimpleNamespace(layer_names=layers, get_metadata_builder=lambda _: builder)]
    if separate:
        groups = [SimpleNamespace(layer_names=[layer], get_metadata_builder=lambda _: builder) for layer in layers]
    config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=layers)])
    tables = owner.block_tables
    tables.cp_size, tables.cp_rank, tables.cp_interleave = 1, 0, 1
    tables.get_dummy_block_tables = lambda count: tuple(t[:count] for t in tables.input_block_tables)
    tables.get_dummy_slot_mappings = lambda count: tables.slot_mappings[:, :count].fill_(-1)
    session = install_parallel_draft_window_graph_characterization(owner, runtime_context=_runtime_context())

    def build():
        return ascend_prepare_dflash_inputs_to_capture(
            4, 8, owner.input_buffers, tables, [groups], config, 512, False, True
        ).attn_metadata

    return owner, descriptor, session, build


@pytest.mark.parametrize("separate", [False, True])
def test_real_group_builder_shared_metadata_reaches_every_layer(monkeypatch, separate):
    owner, desc, session, build = _capture_fixture(monkeypatch, separate=separate)
    target = owner.model_state.attn_metadata
    with draft_window_dflash_capture_scope(owner, owner.input_buffers, {CUDAGraphMode.FULL: [desc]}, max_num_reqs=8):
        warmup = build()
        capture = build()
    values = list(capture.values())
    assert (values[0] is values[1]) is not separate
    assert list(warmup.values())[0] is not values[0]
    recorded_ids = dict(session.recorder.events[-1].layer_metadata_ids)
    assert recorded_ids == {layer: id(obj) for layer, obj in capture.items()}
    assert not any(hasattr(m, ATTRIBUTE) for m in target.values())
    identity = session.provider.registered_capture_identities(route=session.route)[0]
    # Same attachment is idempotent, not silently replaced; old closures retain
    # precisely their own immutable context even after the helper scope exits.
    before = getattr(values[0], ATTRIBUTE)
    attach_parallel_graph_capture_operand_observer(session.provider, identity=identity, metadata=capture)
    assert getattr(values[0], ATTRIBUTE) is before
    for layer, metadata in capture.items():
        record_parallel_graph_capture_operator_operand(
            metadata,
            block_table=metadata.block_tables,
            seq_lengths=metadata.seq_lens_list,
            layer_name=layer,
            num_tokens=8,
            handle=object(),
            event=object(),
            stream=object(),
            backend_impl=AscendAttentionBackendImpl,
        )
    assert [e.layer_name for e in session.recorder.operand_events] == list(capture)
    assert all(e.capture_descriptor == identity.fingerprint.capture_descriptor for e in session.recorder.operand_events)


def test_attach_conflict_is_atomic_and_retryable(monkeypatch):
    owner, desc, session, build = _capture_fixture(monkeypatch)
    with draft_window_dflash_capture_scope(owner, owner.input_buffers, {CUDAGraphMode.FULL: [desc]}, max_num_reqs=8):
        captured = build()
    identity = session.provider.registered_capture_identities(route=session.route)[0]
    clean = SimpleNamespace()
    conflict = SimpleNamespace(**{ATTRIBUTE: object()})
    with pytest.raises(RuntimeError):
        attach_parallel_graph_capture_operand_observer(
            session.provider, identity=identity, metadata={"clean": clean, "conflict": conflict}
        )
    assert not hasattr(clean, ATTRIBUTE)
    delattr(conflict, ATTRIBUTE)
    attach_parallel_graph_capture_operand_observer(
        session.provider, identity=identity, metadata={"clean": clean, "repaired": conflict}
    )
    assert getattr(clean, ATTRIBUTE).identity == identity
    assert getattr(next(iter(captured.values())), ATTRIBUTE).identity == identity


def test_attach_setattr_failure_rolls_back_and_preserves_old_context(monkeypatch):
    owner, desc, session, build = _capture_fixture(monkeypatch)
    identity = session.provider.registered_capture_identities(route=session.route)[0]

    class FailsOnce:
        fail = True

        def __setattr__(self, name, value):
            object.__setattr__(self, name, value)
            if name == ATTRIBUTE and self.fail:
                raise RuntimeError("injected setattr failure")

    first, last = SimpleNamespace(), FailsOnce()
    with pytest.raises(RuntimeError, match="injected"):
        attach_parallel_graph_capture_operand_observer(
            session.provider, identity=identity, metadata={"a": first, "b": last}
        )
    assert not hasattr(first, ATTRIBUTE) and not hasattr(last, ATTRIBUTE)
    last.fail = False
    attach_parallel_graph_capture_operand_observer(
        session.provider, identity=identity, metadata={"a": first, "b": last}
    )
    context = getattr(first, ATTRIBUTE)
    other, _, other_session, _ = _capture_fixture(monkeypatch)
    other_identity = other_session.provider.registered_capture_identities(route=other_session.route)[0]
    with pytest.raises(RuntimeError, match="conflicting"):
        attach_parallel_graph_capture_operand_observer(
            other_session.provider, identity=other_identity, metadata={"a": first}
        )
    assert getattr(first, ATTRIBUTE) is context
    with pytest.raises(RuntimeError, match="Target"):
        attach_parallel_graph_capture_operand_observer(
            session.provider, identity=identity, metadata={"a": first}, target_metadata={"target": first}
        )


def test_real_upstream_factory_warmup_capture_inventory_and_closures(monkeypatch):
    owner, _, _, _ = _capture_fixture(monkeypatch)
    # Fresh installation for the real [7,14,28] inventory, no fake generations.
    del owner._draft_window_graph_characterization
    del owner._draft_window_graph_binding_provider
    descriptors = [
        BatchExecutionDescriptor(cg_mode=CUDAGraphMode.FULL, num_reqs=n, num_tokens=n * 7, uniform_token_count=7)
        for n in (4, 2, 1)
    ]
    owner.query_cudagraph_manager._capture_descs = {CUDAGraphMode.FULL: descriptors}
    owner.num_query_per_req = 7
    owner.speculative_config.num_speculative_tokens = 7
    session = install_parallel_draft_window_graph_characterization(owner, runtime_context=_runtime_context())
    builder = _real_builder(monkeypatch)
    layers = ["draft.layer.0", "draft.layer.1"]
    groups = [[SimpleNamespace(layer_names=layers, get_metadata_builder=lambda _: builder)]]
    config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=layers)])
    manager = upstream_dflash.DFlashCudaGraphManager.__new__(upstream_dflash.DFlashCudaGraphManager)
    manager.max_num_reqs, manager.dp_size = 8, 1
    closures, objects, observed = [], [], []

    def capture_loop(_manager, create_forward_fn, _progress):
        # Substitute only the hardware graph scope, keep upstream closure factory.
        for desc in descriptors:
            for warmup in (True, False):
                forward = create_forward_fn(desc, warmup=warmup)
                closures.append(forward)
                forward(CUDAGraphMode.NONE)

    def forward(n, tokens, metadata, slots, dp, mode):
        obj = metadata[layers[0]]
        assert obj is metadata[layers[1]]
        objects.append(obj)
        context = getattr(obj, ATTRIBUTE)
        assert context.identity.fingerprint.capture_bucket == (n, tokens)
        for layer in layers:
            record_parallel_graph_capture_operator_operand(
                obj,
                block_table=obj.block_tables,
                seq_lengths=obj.seq_lens_list,
                layer_name=layer,
                num_tokens=tokens,
                handle=object(),
                event=object(),
                stream=object(),
                backend_impl=AscendAttentionBackendImpl,
            )
            observed.append((n, tokens, layer))

    monkeypatch.setattr(CudaGraphManager, "capture", capture_loop)
    monkeypatch.setattr(upstream_dflash, "_prepare_dflash_inputs_to_capture", ascend_prepare_dflash_inputs_to_capture)
    with draft_window_dflash_capture_scope(
        owner, owner.input_buffers, {CUDAGraphMode.FULL: descriptors}, max_num_reqs=8
    ):
        manager.capture(forward, owner.input_buffers, owner.block_tables, groups, config, 512, True)
    assert len({id(obj) for obj in objects}) == 6
    assert len(observed) == 12
    # A retained old closure still uses its old bucket after subsequent attaches.
    closures[0](CUDAGraphMode.NONE)
    assert observed[-2:] == [(4, 28, layer) for layer in layers]
    old_context = getattr(objects[0], ATTRIBUTE)
    other_identity = session.provider.registered_capture_identities(route=session.route)[0]
    assert other_identity != old_context.identity
    with pytest.raises(RuntimeError, match="conflicting"):
        attach_parallel_graph_capture_operand_observer(
            session.provider, identity=other_identity, metadata={"old": objects[0]}
        )
    assert getattr(objects[0], ATTRIBUTE) is old_context


def test_observer_absent_is_noop(monkeypatch):
    metadata = SimpleNamespace()
    attach_parallel_graph_capture_operand_observer(
        SimpleNamespace(observer=None), identity=None, metadata={"a": metadata}
    )
    record_parallel_graph_capture_operator_operand(
        metadata,
        block_table=None,
        seq_lengths=None,
        layer_name=None,
        num_tokens=0,
        handle=None,
        event=None,
        stream=None,
        backend_impl=AscendAttentionBackendImpl,
    )
    assert vars(metadata) == {}


def test_operator_callback_does_not_copy_device_or_write_files(monkeypatch):
    import vllm_ascend.worker.v2.spec_decode.draft_window_graph_characterization as module

    owner, desc, session, build = _capture_fixture(monkeypatch)
    with draft_window_dflash_capture_scope(owner, owner.input_buffers, {CUDAGraphMode.FULL: [desc]}, max_num_reqs=8):
        metadata = build()

    def forbidden(*args, **kwargs):
        pytest.fail("D2H/I/O inside capture observer")

    monkeypatch.setattr(session.recorder, "write_artifact", forbidden)
    monkeypatch.setattr(module, "_tensor_values", forbidden)
    monkeypatch.setattr(module, "_tensor_rows", forbidden)
    for layer, obj in metadata.items():
        record_parallel_graph_capture_operator_operand(
            obj,
            block_table=obj.block_tables,
            seq_lengths=obj.seq_lens,
            layer_name=layer,
            num_tokens=8,
            handle=None,
            event=None,
            stream=None,
            backend_impl=AscendAttentionBackendImpl,
        )
    assert len(session.recorder.operand_events) == 2
    assert all(e.handle_identity is None for e in session.recorder.operand_events)
    assert all(e.length_values == () for e in session.recorder.operand_events)
    obj = next(iter(metadata.values()))
    with pytest.raises(RuntimeError, match="tensor length operand"):
        record_parallel_graph_capture_operator_operand(
            obj,
            block_table=obj.block_tables,
            seq_lengths=obj.seq_lens.clone(),
            layer_name="bad",
            num_tokens=8,
            handle=None,
            event=None,
            stream=None,
            backend_impl=AscendAttentionBackendImpl,
        )


def test_observer_off_keeps_candidate_provider_and_metadata_contract(monkeypatch):
    from tests.ut.spec_decode.test_draft_window_graph_carrier import _materialize
    from tests.ut.spec_decode.test_draft_window_graph_characterization import (
        _install_metadata_builder,
        _publish_eager_generation,
    )
    from vllm_ascend.worker.v2.spec_decode.draft_window_graph_characterization import (
        prepare_parallel_draft_window_graph_characterization,
        record_parallel_graph_replay_submission,
    )

    monkeypatch.setenv("VLLM_ASCEND_EXPERIMENTAL_DRAFT_WINDOW_GRAPH_OBSERVE", "0")
    owner, desc, session, build = _capture_fixture(monkeypatch)
    assert session.provider.observer is None
    with draft_window_dflash_capture_scope(owner, owner.input_buffers, {CUDAGraphMode.FULL: [desc]}, max_num_reqs=8):
        assert not any(hasattr(m, ATTRIBUTE) for m in build().values())
    inputs = _materialize(owner._draft_window_runtime_state, mapping_epoch=10)
    _publish_eager_generation(owner, inputs)
    _install_metadata_builder(owner)
    carrier = prepare_parallel_draft_window_graph_characterization(owner, desc)
    assert carrier is not None and carrier.metadata
    monkeypatch.setattr(torch.npu, "current_stream", lambda: pytest.fail("observer-off event"))
    record_parallel_graph_replay_submission(owner)
    snapshot = session.recorder.snapshot()
    assert not any(snapshot[k] for k in ("events", "operand_events", "g4_records", "lifecycle_events"))


def test_artifact_io_failure_is_reported_without_masking_model_error(monkeypatch, tmp_path):
    owner, _, session, _ = _capture_fixture(monkeypatch)
    session.recorder._artifact_path = tmp_path / "no-such-directory" / "artifact.json"
    try:
        raise ValueError("original model failure")
    except ValueError as error:
        session.recorder.write_artifact()
        assert str(error) == "original model failure"
    assert "FileNotFoundError" in session.recorder.snapshot()["artifact_errors"][0]


def test_real_builder_to_shared_metadata_to_real_fia_call_per_layer(monkeypatch):
    import vllm_ascend.attention.attention_v1 as backend
    from tests.ut.spec_decode.test_draft_window_graph_characterization import _FakeExternalEvent

    owner, desc, session, build = _capture_fixture(monkeypatch)
    with draft_window_dflash_capture_scope(owner, owner.input_buffers, {CUDAGraphMode.FULL: [desc]}, max_num_reqs=8):
        metadata = build()
    graph_params = SimpleNamespace(workspaces={8: torch.zeros(1)}, events={8: []}, handles={8: []}, attn_params={8: []})
    monkeypatch.setattr(backend, "get_draft_graph_params", lambda: graph_params)
    monkeypatch.setattr(backend, "weak_ref_tensors", lambda value: value)
    monkeypatch.setattr(backend.torch_npu.npu, "current_stream", lambda: "host-stream-stub")
    monkeypatch.setattr(torch.npu, "ExternalEvent", _FakeExternalEvent)
    monkeypatch.setattr(torch.npu, "graph_task_group_begin", lambda _: None)
    monkeypatch.setattr(torch.npu, "graph_task_group_end", lambda _: object())
    calls = []
    monkeypatch.setattr(
        backend.torch_npu, "npu_fused_infer_attention_score", SimpleNamespace(out=lambda **kw: calls.append(kw))
    )
    monkeypatch.setattr(backend, "_EXTRA_CTX", SimpleNamespace(is_draft_model=True, is_draft_model_prefill=False))
    for layer_name, obj in metadata.items():
        impl = AscendAttentionBackendImpl.__new__(AscendAttentionBackendImpl)
        impl.num_heads, impl.num_kv_heads, impl.head_size, impl.scale = 1, 1, 4, 1.0
        impl.sliding_window, impl.enable_c8_quant = None, False
        impl._use_max_workspace_for_fia_graph = False
        impl._use_layer_aware_fia_graph_replay = False
        impl.kv_sharing_target_layer_name = None
        impl._layer_name = layer_name
        impl._get_fia_params = lambda key, value, m: (key, value, 64, m.block_tables, m.seq_lens_list)
        tensors = [torch.zeros((8, 1, 4)) for _ in range(4)]
        impl.full_graph_fia(tensors[0], tensors[1], tensors[2], obj, tensors[3])
    assert len(calls) == 2
    assert [e.layer_name for e in session.recorder.operand_events] == list(metadata)
    for call, event in zip(calls, session.recorder.operand_events):
        assert call["block_table"].data_ptr() == event.actual_table_address
        assert tuple(call["actual_seq_lengths_kv"]) == event.length_values
    assert not any(hasattr(obj, ATTRIBUTE) for obj in owner.model_state.attn_metadata.values())


def test_no_blocking_observation_between_replay_and_update(monkeypatch):
    import inspect

    import vllm_ascend.worker.v2.spec_decode.draft_window_graph_characterization as module
    from tests.ut.spec_decode.test_draft_window_graph_carrier import _materialize
    from tests.ut.spec_decode.test_draft_window_graph_characterization import (
        _install_metadata_builder,
        _publish_eager_generation,
    )
    from vllm_ascend.worker.v2.spec_decode.dflash.aclgraph import DFlashAclGraphManager
    from vllm_ascend.worker.v2.spec_decode.draft_window_graph import (
        DraftWindowGraphStage,
        consume_draft_window_graph_metadata,
    )

    owner, desc, session, _ = _capture_fixture(monkeypatch)
    inputs = _materialize(owner._draft_window_runtime_state, mapping_epoch=10)
    _publish_eager_generation(owner, inputs)
    _install_metadata_builder(owner)
    module.prepare_parallel_draft_window_graph_characterization(owner, desc)
    consume_draft_window_graph_metadata(owner, route=session.route, expected_consumer_count=1)
    monkeypatch.setattr(module, "_tensor_rows", lambda *a: pytest.fail("blocking D2H before UPDATE"))
    monkeypatch.setattr(module, "_tensor_values", lambda *a: pytest.fail("blocking D2H before UPDATE"))
    for stage in (DraftWindowGraphStage.REPLAY, DraftWindowGraphStage.CONSUMER, DraftWindowGraphStage.UPDATE):
        session.provider._observe_submitted(stage)
    source = inspect.getsource(DFlashAclGraphManager.run_fullgraph)
    assert source.index("record_parallel_graph_output(") > source.index("update_full_graph_params(")
