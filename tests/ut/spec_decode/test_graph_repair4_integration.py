# SPDX-License-Identifier: Apache-2.0
"""CPU integration of the real graph-manager return with the G4 observer.

The upstream replay method is never replaced. Only native replay and the NPU
UPDATE/forward-context boundary are substituted; preparation, metadata
consumption/validation, binding, and recording execute production code.
"""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from dataclasses import replace
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor, CudaGraphManager
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager

import vllm_ascend.worker.v2.spec_decode.dflash.aclgraph as manager_module
import vllm_ascend.worker.v2.spec_decode.draft_window_graph_characterization as observer
from tests.ut.spec_decode.test_draft_window_capture_observer import _real_builder
from tests.ut.spec_decode.test_draft_window_graph_carrier import (
    _descriptor as _read_descriptor,
)
from tests.ut.spec_decode.test_draft_window_graph_carrier import (
    _materialize,
)
from tests.ut.spec_decode.test_draft_window_graph_characterization import (
    _characterization_owner,
    _install_metadata_builder,
    _publish_eager_generation,
    _runtime_context,
)
from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
from vllm_ascend.worker.v2.spec_decode.dflash.aclgraph import DFlashAclGraphManager
from vllm_ascend.worker.v2.spec_decode.dflash.capture import (
    ascend_prepare_dflash_inputs_to_capture,
    draft_window_dflash_capture_scope,
)
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator
from vllm_ascend.worker.v2.spec_decode.draft_window import DraftWindowExecutionPhase
from vllm_ascend.worker.v2.spec_decode.draft_window_eager import DraftWindowEagerMixin
from vllm_ascend.worker.v2.spec_decode.draft_window_graph import (
    DraftWindowGraphBindingProvider,
)

QUERY_WIDTH = 7
LAYER = "draft.layers.0.self_attn"


@contextmanager
def _cpu_forward_context(*_args, **_kwargs):
    yield


def _manager(monkeypatch, owner, descriptors, calls):
    manager = DFlashAclGraphManager.__new__(DFlashAclGraphManager)
    manager.speculator = owner
    manager.device = torch.device("cpu")
    manager.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(decode_context_parallel_size=1))
    manager.update_stream = SimpleNamespace(wait_stream=lambda _stream: None)
    manager.graphs = {desc: SimpleNamespace(replay=lambda: calls.append("native_replay")) for desc in descriptors}
    owner.dp_size = 1
    owner.input_batch.seq_lens_cpu_upper_bound = torch.zeros(8, dtype=torch.int32)
    owner.build_draft_attn_metadatas = MethodType(AscendDFlashSpeculator.build_draft_attn_metadatas, owner)
    monkeypatch.setattr(manager_module, "set_forward_context", _cpu_forward_context)
    monkeypatch.setattr(manager_module, "get_forward_context", lambda: object())
    monkeypatch.setattr(
        manager_module,
        "_EXTRA_CTX",
        SimpleNamespace(is_draft_model=False, is_draft_model_prefill=False),
    )
    monkeypatch.setattr(
        AscendAttentionBackendImpl,
        "update_graph_params",
        staticmethod(lambda *_args, **_kwargs: calls.append("native_update")),
    )
    return manager


def test_real_upstream_none_return_is_not_hashed(monkeypatch):
    """Repair3 RED: actual inherited run returns None after native replay."""
    assert DFlashCudaGraphManager.run_fullgraph is CudaGraphManager.run_fullgraph
    assert inspect.getsourcefile(CudaGraphManager.run_fullgraph).endswith("vllm/v1/worker/gpu/cudagraph_utils.py")
    owner, desc = _characterization_owner()
    session = observer.install_parallel_draft_window_graph_characterization(owner, runtime_context=_runtime_context())
    inputs = _materialize(owner._draft_window_runtime_state, mapping_epoch=10)
    _publish_eager_generation(owner, inputs)
    _install_metadata_builder(owner)
    calls = []
    manager = _manager(monkeypatch, owner, [desc], calls)
    assert session.recorder.enabled
    assert manager.run_fullgraph(desc) is None
    assert calls == ["native_replay", "native_update"]
    record = session.recorder.g4_records[-1]
    assert record.generation_id == inputs.identity.generation_id
    assert record.draft_output_hash is None
    assert record.draft_output_status == "OUTPUT_NOT_CAPTURED"
    assert record.draft_output_reason == "GRAPH_MANAGER_RETURNS_NONE"


def _lifetime_fixture(monkeypatch, **geometry):
    """Reuse source/capture fixtures; all owner integration methods are real."""
    template, _ = _characterization_owner(**geometry)
    owner = AscendDFlashSpeculator.__new__(AscendDFlashSpeculator)
    owner.__dict__.update(vars(template))
    owner.num_query_per_req = QUERY_WIDTH
    owner.speculative_config = SimpleNamespace(
        num_speculative_tokens=QUERY_WIDTH if geometry.get("method") == "dspark" else QUERY_WIDTH - 1
    )
    owner.arange = torch.arange(9, dtype=torch.int32)
    owner._initialize_draft_window_eager_state()
    descriptors = [
        BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.FULL,
            num_tokens=n * QUERY_WIDTH,
            num_reqs=n,
            uniform_token_count=QUERY_WIDTH,
        )
        for n in (1, 2, 4)
    ]
    owner.query_cudagraph_manager._capture_descs = {CUDAGraphMode.FULL: descriptors}
    builder = _real_builder(monkeypatch)
    owner.attn_groups = [
        [
            SimpleNamespace(
                layer_names=[LAYER],
                get_metadata_builder=lambda _: builder,
            )
        ]
    ]
    owner.kv_cache_config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=[LAYER])])
    tables = owner.block_tables
    tables.cp_size, tables.cp_rank, tables.cp_interleave = 1, 0, 1
    tables.get_dummy_block_tables = lambda n: tuple(t[:n] for t in tables.input_block_tables)
    tables.get_dummy_slot_mappings = lambda n: tables.slot_mappings[:, :n].fill_(-1)
    session = observer.install_parallel_draft_window_graph_characterization(owner, runtime_context=_runtime_context())
    # Genuine capture adapter + metadata assembly + Ascend builder.
    with draft_window_dflash_capture_scope(
        owner, owner.input_buffers, {CUDAGraphMode.FULL: descriptors}, max_num_reqs=8
    ):
        for desc in descriptors:
            state = ascend_prepare_dflash_inputs_to_capture(
                desc.num_reqs,
                desc.num_tokens,
                owner.input_buffers,
                tables,
                owner.attn_groups,
                owner.kv_cache_config,
                512,
                False,
                True,
            )
            assert state.attn_metadata[LAYER].block_tables.shape[0] == desc.num_reqs
    tables.slot_mappings.copy_(torch.arange(tables.slot_mappings.numel(), dtype=torch.int32).reshape(1, -1))
    calls = []
    manager = _manager(monkeypatch, owner, descriptors, calls)
    # Audit the exact methods this harness will execute, including scope cleanup.
    assert owner._draft_window_proposal_scope.__func__ is DraftWindowEagerMixin._draft_window_proposal_scope
    assert owner._build_ascend_draft_attn_metadata.__func__ is DraftWindowEagerMixin._build_ascend_draft_attn_metadata
    assert owner.build_draft_attn_metadatas.__func__ is AscendDFlashSpeculator.build_draft_attn_metadatas
    assert DFlashCudaGraphManager.run_fullgraph is CudaGraphManager.run_fullgraph
    return owner, session, manager, {d.num_reqs: d for d in descriptors}, calls


def _batch(owner, active):
    owner.input_batch = SimpleNamespace(
        num_reqs=active,
        req_ids=tuple(f"request-{i}" for i in range(active)),
        idx_mapping_np=np.arange(1, active + 1, dtype=np.int32),
        num_computed_prefill_tokens_np=np.full(active, 64, dtype=np.int32),
        num_scheduled_tokens=np.ones(active, dtype=np.int32),
        prefill_len_np=np.full(active, 64, dtype=np.int32),
        is_prefilling_np=np.zeros(active, dtype=np.bool_),
        has_prefill=False,
        seq_lens_cpu_upper_bound=torch.full((active,), 128, dtype=torch.int32),
    )


def _publish_in_scope(owner, desc, *, start=0):
    batch = owner.input_batch
    context = owner._draft_window_proposal_context
    padded = desc.num_reqs
    exact = torch.tensor([64] * batch.num_reqs + [0] * (padded - batch.num_reqs), dtype=torch.int32)
    inputs = owner._draft_window_runtime_state.materialize(
        mapping_epoch=context.mapping_epoch,
        ordered_request_ids=context.ordered_request_ids,
        idx_mapping=torch.from_numpy(batch.idx_mapping_np),
        read_descriptors=tuple(_read_descriptor(start=start) for _ in range(batch.num_reqs)),
        device_window_exact=exact.clone(),
        cpu_window_exact=exact,
        cpu_window_upper=exact.clone(),
        num_reqs_padded=padded,
        phase=DraftWindowExecutionPhase.PROPOSAL,
        step=0,
        execution_lane=0,
    )
    owner.input_buffers.query_start_loc[: padded + 1].copy_(torch.arange(padded + 1, dtype=torch.int32) * QUERY_WIDTH)
    owner.input_buffers.positions[: desc.num_tokens].copy_(torch.arange(desc.num_tokens))
    _publish_eager_generation(owner, inputs)
    # The helper publishes eager metadata; retain the REAL scope's context.
    owner._draft_window_proposal_context = context
    return inputs


def test_multi_generation_integration_real_scope(monkeypatch):
    """MULTI_GENERATION_INTEGRATION: 1 -> 2 -> 4 -> 1, then 3 active / 4 padded."""
    owner, session, manager, descriptors, calls = _lifetime_fixture(monkeypatch)
    target = owner.model_state.attn_metadata
    target_table = target[LAYER].block_tables.clone()
    generations, backing, first_table = [], {}, None
    for active, padded, start in ((1, 1, 0), (2, 2, 1), (4, 4, 1), (1, 1, 0), (3, 4, 0)):
        _batch(owner, active)
        desc = descriptors[padded]
        with owner._draft_window_proposal_scope(owner.input_batch, dummy_run=False, is_profile=False):
            inputs = _publish_in_scope(owner, desc, start=start)
            generations.append(inputs.identity.generation_id)
            assert manager.run_fullgraph(desc) is None
            (carrier,) = session.provider.submitted_carriers
            assert carrier.runtime_identity.generation_id == generations[-1]
            assert carrier.backend_attribution is not None
            metadata = carrier.metadata[LAYER]
            assert metadata.block_tables is carrier.read_inputs.window_block_table
            assert metadata.seq_lens.data_ptr() == carrier.read_inputs.device_window_exact.data_ptr()
            assert metadata.actual_seq_lengths_q == [QUERY_WIDTH * (i + 1) for i in range(padded)]
            assert metadata.seq_lens_list == [64] * active + [0] * (padded - active)
            pointer = metadata.block_tables.data_ptr()
            assert backing.setdefault(padded, pointer) == pointer
            if len(generations) == 1:
                first_table = metadata.block_tables.clone()
            elif len(generations) == 4:
                torch.testing.assert_close(metadata.block_tables, first_table)
            if active < padded:
                assert torch.count_nonzero(metadata.block_tables[active:]) == 0
                assert torch.count_nonzero(metadata.seq_lens[active:]) == 0
            record = session.recorder.g4_records[-1]
            assert record.generation_id == generations[-1]
            assert record.draft_output_hash is None
            assert record.draft_output_status == "OUTPUT_NOT_CAPTURED"
            assert record.draft_output_reason == "GRAPH_MANAGER_RETURNS_NONE"
            assert owner.model_state.attn_metadata is target
            torch.testing.assert_close(target[LAYER].block_tables, target_table)
        # Retirement is exclusively production scope __exit__, never manual.
        assert inputs.lease.state.value == "retired"
        assert owner._draft_window_live_inputs is None
        assert owner._draft_window_proposal_context is None
        with pytest.raises(RuntimeError):
            inputs.validate_for(inputs.identity)
    assert len(set(generations)) == 5
    assert calls == ["native_replay", "native_update"] * 5
    snapshot = session.recorder.snapshot()
    retired = [e for e in snapshot["lifecycle_events"] if e["stage"] == "proposal_retired"]
    assert [e["generation_id"] for e in retired] == generations
    assert all(e["event_complete"] is None for e in retired)
    assert [r["generation_id"] for r in snapshot["g4_records"]] == generations
    print("MULTI_GENERATION_INTEGRATION=PASS; real proposal-scope retirement; tokens=7,14,28; requests=1,2,4,1,3pad4")


@pytest.fixture
def submitted_generation(monkeypatch):
    owner, desc = _characterization_owner()
    session = observer.install_parallel_draft_window_graph_characterization(owner, runtime_context=_runtime_context())
    inputs = _materialize(owner._draft_window_runtime_state, mapping_epoch=10)
    _publish_eager_generation(owner, inputs)
    _install_metadata_builder(owner)
    manager = _manager(monkeypatch, owner, [desc], [])
    assert manager.run_fullgraph(desc) is None
    return owner, session, inputs


@pytest.mark.parametrize("output", [None, (), [], "", torch.empty(0), [torch.ones(1), None]])
def test_optional_output_status_and_reason(submitted_generation, output):
    owner, session, _ = submitted_generation
    observer.record_parallel_graph_output(owner, output)
    record = session.recorder.g4_records[-1]
    assert record.draft_output_hash is None
    assert record.draft_output_status == "OUTPUT_NOT_CAPTURED"
    assert record.draft_output_reason == (
        "GRAPH_MANAGER_RETURNS_NONE" if output is None else "EMPTY_OR_UNSUPPORTED_OUTPUT"
    )


def test_valid_tensor_is_hashed_and_device_errors_escape(submitted_generation, monkeypatch):
    owner, session, _ = submitted_generation
    output = torch.tensor([1.0, 2.0])
    observer.record_parallel_graph_output(owner, output)
    record = session.recorder.g4_records[-1]
    assert record.draft_output_status == "HASHED"
    assert record.draft_output_reason is None
    assert len(record.draft_output_hash) == 64

    def failed_transfer(*_args, **_kwargs):
        raise RuntimeError("tensor device access failed")

    monkeypatch.setattr(torch.Tensor, "to", failed_transfer)
    with pytest.raises(RuntimeError, match="tensor device access failed"):
        observer.record_parallel_graph_output(owner, output)
    assert session.recorder.g4_records[-1] == record


@pytest.mark.parametrize("mutation", ["missing", "stale", "wrong_identity"])
def test_output_identity_errors_precede_tensor_reads(submitted_generation, monkeypatch, mutation):
    owner, session, inputs = submitted_generation
    if mutation == "missing":
        session.provider.discard_bound_carriers()
    elif mutation == "stale":
        session.provider.discard_bound_carriers()
        owner._draft_window_runtime_state.retire_generation(inputs.identity)
    else:
        (carrier,) = session.provider.submitted_carriers
        carrier.runtime_identity = replace(carrier.runtime_identity, generation_id=999999)

    def unsafe_read(*_args, **_kwargs):
        pytest.fail("output tensor read before identity validation")

    before = session.recorder.g4_records
    monkeypatch.setattr(torch.Tensor, "detach", unsafe_read)
    with pytest.raises(RuntimeError, match="one (submitted|G4) generation"):
        observer.record_parallel_graph_output(owner, torch.ones(1))
    assert session.recorder.g4_records == before


def test_observer_off_precedes_submitted_carrier_validation(submitted_generation, monkeypatch):
    owner, session, _ = submitted_generation
    session.recorder.enabled = False
    before = session.recorder.g4_records

    def forbidden(*_args):
        pytest.fail("disabled observer touched carrier identities or tensor")

    monkeypatch.setattr(DraftWindowGraphBindingProvider, "submitted_carriers", property(forbidden))
    monkeypatch.setattr(torch.Tensor, "detach", forbidden)
    observer.record_parallel_graph_output(owner, torch.ones(1))
    assert session.recorder.g4_records == before
