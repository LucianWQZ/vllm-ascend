# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Shared query construction and Eagle's real eager-reader boundary."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator
from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator import TargetDependentARSpeculator

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.worker.v2.spec_decode import draft_window_eager
from vllm_ascend.worker.v2.spec_decode.draft_window_eagle3 import (
    Eagle3DraftWindowMixin,
    _eagle3_graph_scheduled_consumers,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_eagle3_adapter import Eagle3GraphIntegrationMixin
from vllm_ascend.worker.v2.spec_decode.draft_window_eagle3_capture import Eagle3GraphSession
from vllm_ascend.worker.v2.spec_decode.eagle.speculator import AscendEagleSpeculator
from vllm_ascend.worker.v2.spec_decode.target_dependent_ar.speculator import AscendTargetDependentARSpeculator


def test_eagle_integration_keeps_the_existing_execution_mro():
    class PreviousBases(Eagle3DraftWindowMixin, AscendTargetDependentARSpeculator, EagleSpeculator):
        pass

    assert AscendEagleSpeculator.__mro__ == (
        AscendEagleSpeculator,
        Eagle3GraphIntegrationMixin,
        *PreviousBases.__mro__[1:],
    )
    assert AscendEagleSpeculator.__init__ is AscendTargetDependentARSpeculator.__init__


@pytest.mark.parametrize("fail", [False, True])
def test_ordinary_eager_forward_uses_the_existing_method_once(monkeypatch, fail):
    calls = []
    metadata = object()

    def forward(self, *args):
        calls.append(("forward", args))
        if fail:
            raise ValueError("native forward failed")

    monkeypatch.setattr(TargetDependentARSpeculator, "_generate_draft", forward)

    class Owner(Eagle3GraphIntegrationMixin):
        def _update_decode_attn_metadata(self, *args):
            calls.append(("update", args))

    owner: Any = object.__new__(Owner)
    args = (2, 4, metadata, object(), object())
    if fail:
        with pytest.raises(ValueError, match="native forward failed"):
            owner._generate_draft(*args)
    else:
        assert owner._generate_draft(*args) is None
    assert calls[0][0] == "forward" and calls[0][1][:5] == args
    assert [event[0] for event in calls] == ["forward"] + ([] if fail else ["update"])


def test_graph_owner_keeps_normal_prefill_decode_metadata_and_updates(monkeypatch):
    for name in ("_build_attn_metadata", "_update_decode_attn_metadata"):
        assert getattr(Eagle3GraphIntegrationMixin, name) is getattr(AscendTargetDependentARSpeculator, name)
    owner = object.__new__(Eagle3GraphIntegrationMixin)
    expected = object()

    def prefill(current, *args, **kwargs):
        assert current is owner and current._eagle3_window_native_prefill_active
        assert args == (1, 4) and kwargs == {"skip_sample": True}
        return expected

    monkeypatch.setattr(AscendTargetDependentARSpeculator, "_prefill", prefill)
    assert owner._prefill(1, 4, skip_sample=True) is expected
    assert not owner._eagle3_window_native_prefill_active


@pytest.mark.parametrize("step,has_prefill", [(0, True), (0, False), (1, False)])
def test_graph_owner_rejects_unintended_eager_business_forward(monkeypatch, step, has_prefill):
    class Owner(Eagle3GraphIntegrationMixin):
        def _eagle3_window_is_active(self):
            return True

    monkeypatch.setattr(
        AscendTargetDependentARSpeculator,
        "_run_model",
        lambda *args, **kwargs: pytest.fail("business Graph scope reached an ordinary eager forward"),
    )
    owner: Any = object.__new__(Owner)
    owner._eagle3_graph_session = SimpleNamespace(_capture_call=None)
    owner._eagle3_window_next_step = step
    owner.input_batch = SimpleNamespace(has_prefill=has_prefill)
    with pytest.raises(RuntimeError, match="FULL Graph managers"):
        owner._run_model(4, {}, None, None)


@pytest.mark.parametrize("scheduled", [-1, 1, 2, 4, True])
def test_eagle_partial_schedule_is_not_dynamic_k(scheduled):
    with pytest.raises(NotImplementedError):
        _eagle3_graph_scheduled_consumers(3, scheduled)


@pytest.fixture
def proposal(monkeypatch):
    owner: Any = Eagle3DraftWindowMixin()
    owner.method, owner.num_speculative_steps = "eagle3", 3
    owner.speculative_config = SimpleNamespace(uses_dynamic_speculative_decoding=lambda: False)
    runtime: Any = SimpleNamespace(
        source=SimpleNamespace(
            policy=SimpleNamespace(normalized_method="eagle3"),
            window_size=4,
            group=SimpleNamespace(gid=0, table_column_span=4),
            block_tables=SimpleNamespace(num_blocks=SimpleNamespace(cpu=torch.tensor([[4]], dtype=torch.int32))),
        ),
        _active_inputs=None,
        quarantined=False,
        retired=None,
        pending=[],
        require_reusable=lambda: None,
    )

    def retire(inputs):
        assert runtime._active_inputs is inputs
        runtime.retired, runtime._active_inputs = inputs, None

    runtime.retire_generation = retire
    runtime.quarantine_generation = lambda inputs: setattr(runtime, "quarantined", True)

    def finish():
        if runtime._active_inputs is not None or runtime.pending:
            raise RuntimeError("unfinished Graph consumer")

    runtime.finish_proposal = finish
    runtime.prepare_workspace = lambda capture: capture.workspace
    owner._draft_window_graph_binding_provider = runtime
    owner._initialize_eagle3_draft_window_state()
    batch = SimpleNamespace(
        num_reqs=1,
        req_ids=["request"],
        idx_mapping_np=np.array([0]),
        num_computed_prefill_tokens_np=np.array([10]),
        prefill_len_np=np.array([10]),
        num_scheduled_tokens=np.array([1]),
        num_draft_tokens_per_req=None,
        is_prefilling_np=np.array([False]),
        has_prefill=False,
    )
    session: Any = object.__new__(Eagle3GraphSession)
    session.owner = owner
    session.provider = runtime
    owner._eagle3_graph_session = session
    owner.model_state = SimpleNamespace(attn_metadata={})
    owner.input_batch = batch
    owner.max_model_len = 32
    owner.input_buffers = SimpleNamespace(seq_lens=torch.tensor([8], dtype=torch.int32))
    batch.seq_lens_np = np.array([10], dtype=np.int32)
    owner.draft_vllm_config = SimpleNamespace(parallel_config=object())
    batch.query_start_loc_np = np.array([0, 4], dtype=np.int32)
    batch.attn_state = AscendAttentionState.DecodeOnly
    bank = SimpleNamespace(
        window_block_table=torch.tensor([[1, 2]], dtype=torch.int32),
        device_window_exact=torch.tensor([5], dtype=torch.int32),
        cpu_window_exact=torch.tensor([5], dtype=torch.int32),
    )
    from vllm_ascend.worker.v2.spec_decode.draft_window_graph import DraftWindowGraphRoute

    session.resolve = lambda *, first, n, tokens, ordinal: SimpleNamespace(
        route=DraftWindowGraphRoute.EAGLE3_FIRST_PASS if first else DraftWindowGraphRoute.EAGLE3_AR_STEP,
        descriptor=SimpleNamespace(num_reqs=n, num_tokens=tokens),
        consumer_ordinal=ordinal,
        workspace=bank,
    )
    session.provider.bind_read_carrier = lambda **kwargs: runtime.pending.append(kwargs["inputs"])

    def inputs(step, **changes):
        context = owner._require_eagle3_window_context()
        values = dict(proposal=context, step=step)
        values.update(changes)
        values.setdefault("padded_request_count", 1)
        current = SimpleNamespace(**values)
        current.validate = lambda: runtime._active_inputs is current or pytest.fail("stale inputs")
        runtime._active_inputs = current
        return current

    runtime.materialize = lambda **kwargs: inputs(kwargs["step"], padded_request_count=kwargs["num_reqs_padded"])
    return owner, batch, session, runtime, inputs


def test_real_proposal_zero_uses_first_reader_and_keeps_static_k(proposal):
    owner, batch, session, runtime, inputs = proposal
    with owner._eagle3_draft_window_proposal_scope(batch, dummy_run=False, is_profile=False, num_speculative_tokens=0):
        assert owner._eagle3_window_expected_steps == 1
        session.stage(session.resolve(first=True, n=1, tokens=4, ordinal=0), 0)
        assert runtime.retired.step == 0
        runtime.pending.clear()
    assert runtime.retired is not None
    assert owner.num_speculative_steps == owner._eagle3_window_expected_steps == 3
    assert not runtime.pending


def test_real_proposal_keeps_each_submitted_consumer_until_update(proposal):
    owner, batch, session, runtime, inputs = proposal
    with owner._eagle3_draft_window_proposal_scope(batch, dummy_run=False, is_profile=False):
        for step in range(3):
            if step == 1:
                owner._snapshot_eagle3_decode_step_one(num_reqs=1, num_reqs_padded=1)
            session.stage(
                session.resolve(first=step == 0, n=1, tokens=4 if step == 0 else 1, ordinal=max(0, step - 1)), step
            )
            current = runtime.retired
            assert owner._eagle3_window_next_step == step + 1
            assert runtime.retired is current
            assert runtime._active_inputs is None
            assert runtime.pending == [current]
            runtime.pending.clear()
    assert not runtime.pending
    assert not getattr(owner, "_draft_window_graph_submission_uncertain", False)


def test_business_window_proposal_requires_the_real_installed_graph_manager(proposal):
    owner, batch, session, runtime, inputs = proposal
    del owner._eagle3_graph_session
    with (
        pytest.raises(RuntimeError, match="installed Graph manager"),
        owner._eagle3_draft_window_proposal_scope(batch, dummy_run=False, is_profile=False),
    ):
        pytest.fail("a business request reached the old private eager owner")
    assert owner._eagle3_window_proposal is None
    assert runtime._active_inputs is None


@pytest.mark.parametrize("damage", ["missing_reader", "missing_update", "wrong_batch", "wrong_step"])
def test_real_proposal_rejects_unconsumed_or_misattributed_reader(proposal, damage):
    owner, batch, session, runtime, inputs = proposal
    with (
        pytest.raises(RuntimeError),
        owner._eagle3_draft_window_proposal_scope(batch, dummy_run=False, is_profile=False, num_speculative_tokens=0),
    ):
        if damage != "missing_reader":
            if damage == "wrong_batch":
                owner.input_batch = SimpleNamespace(**vars(batch))
            session.stage(session.resolve(first=True, n=1, tokens=4, ordinal=0), 1 if damage == "wrong_step" else 0)
    assert owner._draft_window_graph_submission_uncertain
    assert owner._eagle3_window_proposal is None


@pytest.mark.parametrize("first_padding", [1, 2])
def test_first_and_ar_share_one_materialization_path_without_moving_the_anchor(proposal, monkeypatch, first_padding):
    from vllm_ascend.worker.v2.spec_decode import draft_window_eagle3

    owner, batch, session, runtime, _ = proposal
    owner.input_batch, owner.max_model_len = batch, 32
    owner.input_buffers = SimpleNamespace(seq_lens=torch.tensor([8], dtype=torch.int32))
    batch.seq_lens_np = np.array([10], dtype=np.int64)
    batch.seq_lens_cpu_upper_bound = torch.tensor([12], dtype=torch.int32)
    batch.idx_mapping = torch.tensor([0], dtype=torch.int64)
    runtime.source.window_size = 4
    runtime.source.group = SimpleNamespace(gid=0, table_column_span=4)
    calls = []
    monkeypatch.setattr(
        draft_window_eagle3, "gather_valid_num_blocks_cpu", lambda *args, **kwargs: torch.tensor([4], dtype=torch.int32)
    )

    def materialize(**kwargs):
        calls.append(kwargs)
        current = SimpleNamespace(proposal=kwargs["proposal"], step=kwargs["step"])
        current.validate = lambda: runtime._active_inputs is current
        runtime._active_inputs = current
        return current

    runtime.materialize = materialize
    with owner._eagle3_draft_window_proposal_scope(batch, dummy_run=False, is_profile=False):
        frozen = None
        for step in range(3):
            if step == 1:
                owner._snapshot_eagle3_decode_step_one(num_reqs=1, num_reqs_padded=1)
            padded = first_padding if step == 0 else 1
            session.stage(
                session.resolve(
                    first=step == 0, n=padded, tokens=4 * padded if step == 0 else padded, ordinal=max(0, step - 1)
                ),
                step,
            )
            frozen = frozen or owner._eagle3_window_proposal_plan
            assert owner._eagle3_window_proposal_plan is frozen
            assert frozen[0].window_start == 8
            runtime.pending.clear()
    assert [call["cpu_full_exact"].tolist() for call in calls] == [[10] + [0] * (first_padding - 1), [8], [9]]
    assert all(call["frozen_row_plans"] is frozen for call in calls)


def test_eagle_startup_rejects_no_usable_capture_buckets():
    stream = object()
    manager = SimpleNamespace(update_stream=stream, _capture_descs={})
    owner = SimpleNamespace(
        update_stream=stream,
        prefill_cudagraph_manager=manager,
        decode_cudagraph_manager=manager,
        num_speculative_steps=3,
        _draft_window_graph_binding_provider=SimpleNamespace(source=SimpleNamespace(group=object()), _carriers={}),
    )
    with pytest.raises(RuntimeError, match="nonempty capture bucket"):
        Eagle3GraphSession(owner)


@pytest.mark.parametrize("first,tokens,ordinal", [(True, 4, 0), (False, 1, 0), (False, 1, 1)])
def test_eagle_capture_lookup_keeps_each_real_shape_and_consumer(first, tokens, ordinal):
    from vllm_ascend.worker.v2.spec_decode.draft_window_graph import DraftWindowGraphRoute

    session: Any = object.__new__(Eagle3GraphSession)
    first_route = DraftWindowGraphRoute.EAGLE3_FIRST_PASS
    ar_route = DraftWindowGraphRoute.EAGLE3_AR_STEP
    captures = {
        (first_route, 1, 4): (object(),),
        (ar_route, 1, 1): (object(), object()),
    }

    def lookup(*, route, num_reqs, num_tokens):
        try:
            return captures[route, num_reqs, num_tokens]
        except KeyError:
            raise RuntimeError("Eagle3 Window has no exact capture bank") from None

    session.provider = SimpleNamespace(capture_banks_for=lookup)
    expected = captures[(first_route if first else ar_route, 1, tokens)][ordinal]
    assert session.resolve(first=first, n=1, tokens=tokens, ordinal=ordinal) is expected
    with pytest.raises(RuntimeError, match="no exact capture bank"):
        session.resolve(first=first, n=2, tokens=tokens, ordinal=ordinal)
    with pytest.raises(RuntimeError, match="no exact capture bank"):
        session.resolve(first=first, n=1, tokens=tokens + 1, ordinal=ordinal)


def test_eagle_capture_borrows_full_query_write_fields_and_owns_only_reads():
    query = torch.tensor([0, 4], dtype=torch.int32)
    slots, positions = torch.arange(4), torch.arange(8, 12)
    original = SimpleNamespace(slot_mapping=slots, positions=positions, seq_lens_list=[12])
    metadata = {"layer": original, "shared": original, "empty": None}
    bank = SimpleNamespace(
        window_block_table=torch.tensor([[2, 3]], dtype=torch.int32),
        device_window_exact=torch.tensor([5], dtype=torch.int32),
        cpu_window_exact=torch.tensor([5], dtype=torch.int32),
    )
    session: Any = object.__new__(Eagle3GraphSession)
    session.owner = SimpleNamespace(max_model_len=32, input_buffers=SimpleNamespace(query_start_loc=query))
    session.provider = SimpleNamespace(
        source=SimpleNamespace(window_size=4, group=SimpleNamespace(table_column_span=2)),
        startup_capture_workspace=lambda identity: bank,
    )
    capture = SimpleNamespace(descriptor=SimpleNamespace(num_reqs=1))
    session._capture_call = iter((capture,))
    result = session.capture_metadata(metadata)
    assert result["layer"] is result["shared"]
    layer = result["layer"]
    assert layer is not original and result["empty"] is None
    assert layer.block_tables is bank.window_block_table
    assert layer.seq_lens is bank.device_window_exact and layer.seq_lens_cpu is bank.cpu_window_exact
    assert layer.seq_lens_list == [5] and original.seq_lens_list == [12]
    assert layer.slot_mapping is slots and layer.positions is positions
    assert layer.query_start_loc.data_ptr() == query.data_ptr()
    assert not hasattr(original, "block_tables") and not hasattr(original, "query_start_loc")


@pytest.fixture
def parallel_hook():
    from vllm_ascend.worker.v2.spec_decode.draft_window_eager import DraftWindowGraphInputMixin

    class Baseline:
        def _build_attn_metadata(self, *args, **kwargs):
            return (args, kwargs)

    class Owner(DraftWindowGraphInputMixin, Baseline):
        pass

    owner: Any = Owner()
    owner._draft_window_graph_binding_provider = object()
    owner._draft_window_bypass_active = False
    owner.draft_attn_layer_names, owner.num_query_per_req = {"draft.layer"}, 8
    owner.block_tables = SimpleNamespace(slot_mappings=torch.arange(32, dtype=torch.int64).reshape(1, 32))
    inputs, calls = object(), []

    def materialize(**kwargs):
        calls.append(kwargs)
        return inputs

    owner._materialize_draft_window_read_inputs = materialize
    desc = SimpleNamespace(cg_mode=SimpleNamespace(name="FULL"), num_reqs=2, num_tokens=16)
    owner._draft_window_selected_metadata = SimpleNamespace(descriptor=desc)
    return owner, desc, inputs, calls


@pytest.mark.parametrize("bypass", [False, True])
def test_parallel_metadata_hook_retains_the_existing_off_and_prefill_path(parallel_hook, bypass):
    owner, desc, _, calls = parallel_hook
    if bypass:
        owner._draft_window_bypass_active = True
    else:
        del owner._draft_window_graph_binding_provider
    args = (1, desc, np.array([0, 8], dtype=np.int32), torch.tensor([16]), 8)
    original_args, original_kwargs = owner._build_attn_metadata(*args)
    assert original_args[:5] == args
    assert original_args[5:] == (True, None)
    assert not original_kwargs and not calls


def test_parallel_metadata_hook_prepares_one_bank_without_a_temporary_metadata_build(parallel_hook):
    owner, desc, inputs, calls = parallel_hook
    result = owner._build_attn_metadata(
        1,
        desc,
        np.array([0, 8], dtype=np.int32),
        torch.tensor([16]),
        8,
        slot_mappings=owner.block_tables.slot_mappings[:, :16],
    )
    assert result is None and not hasattr(owner, "_draft_window_live_inputs")
    assert calls == [{"num_reqs": 1, "num_reqs_padded": 2}]


@pytest.mark.parametrize("damage", ["mode", "query", "step", "tokens", "selection", "full_slots"])
def test_parallel_metadata_hook_rejects_bad_actual_inputs_before_bank_materialization(parallel_hook, damage):
    owner, desc, _, calls = parallel_hook
    query, step, slots = np.array([0, 8], dtype=np.int32), 8, owner.block_tables.slot_mappings[:, :16]
    if damage == "mode":
        desc.cg_mode.name = "NONE"
    elif damage == "query":
        query[-1] = 7
    elif damage == "step":
        step = 7
    elif damage == "tokens":
        desc.num_tokens = 15
    elif damage == "selection":
        owner._draft_window_selected_metadata = None
    elif damage == "full_slots":
        slots = slots.clone()
    with pytest.raises(RuntimeError, match="actual uniform FULL dispatch|canonical Full write slots"):
        owner._build_attn_metadata(1, desc, query, torch.tensor([16]), step, slot_mappings=slots)
    assert not calls


@pytest.mark.parametrize("scheduled", [0, 3])
def test_eagle_stage_uses_the_bank_without_rebuilding_or_reading_target_metadata(proposal, scheduled):
    owner, batch, session, runtime, _ = proposal
    target, draft_config = object(), object()
    owner.model_state, owner.draft_vllm_config = target, draft_config
    del batch.attn_state
    with owner._eagle3_draft_window_proposal_scope(
        batch, dummy_run=False, is_profile=False, num_speculative_tokens=scheduled
    ):
        for step in range(1 if scheduled == 0 else scheduled):
            if step == 1:
                owner._snapshot_eagle3_decode_step_one(num_reqs=1, num_reqs_padded=1)
            capture = session.resolve(first=step == 0, n=1, tokens=4 if step == 0 else 1, ordinal=max(0, step - 1))
            session.stage(capture, step)
            assert runtime.retired.step == step
            assert owner.model_state is target and owner.draft_vllm_config is draft_config
            runtime.pending.clear()


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE])
@pytest.mark.parametrize("skip_sample", [False, True])
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("padding", ["none", "tensor", "extra_request"])
def test_official_nonuniform_prefill_uses_window_first_bank_without_an_eager_owner(
    monkeypatch, mode, skip_sample, fail, padding
):
    from vllm_ascend.attention.attention_v1 import AscendAttentionState

    class Owner(Eagle3GraphIntegrationMixin):
        def _eagle3_window_is_active(self):
            return True

    owner: Any = object.__new__(Owner)
    events, sampled = [], []
    query = torch.tensor([0, 3, 7], dtype=torch.int32)
    slots = torch.arange(7, dtype=torch.int64) + 80
    absolute = torch.arange(7, dtype=torch.int64) + 70
    target_layer = SimpleNamespace(
        block_tables=torch.full((2, 4), 91, dtype=torch.int32),
        seq_lens=torch.tensor([77, 88], dtype=torch.int32),
        seq_lens_gpu=torch.tensor([77, 88], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([77, 88], dtype=torch.int32),
        seq_lens_list=[77, 88],
        actual_seq_lengths_q=[3, 7],
        query_start_loc=query,
        positions=absolute,
        slot_mapping=slots,
        qfa_metadata_cache={},
        num_actual_tokens=7,
        attn_state=AscendAttentionState.PrefillNoCache,
    )
    target = {"draft": target_layer}
    slot_map = {"draft": slots}
    original_target_table = target_layer.block_tables.clone()
    bank = SimpleNamespace(
        window_block_table=torch.tensor([[1, 2, 0, 0], [5, 6, 7, 8]], dtype=torch.int32),
        device_window_exact=torch.tensor([5, 11], dtype=torch.int32),
        cpu_window_exact=torch.tensor([5, 11], dtype=torch.int32),
    )
    capture = object()
    pending = SimpleNamespace(workspace=bank)
    provider = SimpleNamespace(_active_inputs=None)

    def resolve(**kwargs):
        assert kwargs == dict(first=True, n=2, tokens=8, ordinal=0)
        return capture

    def prepare(actual):
        assert actual is capture
        events.append("prepare")
        return bank

    def materialize(**kwargs):
        assert kwargs == dict(step=0, num_reqs=2, num_reqs_padded=2, output_workspace=bank)
        provider._active_inputs = pending
        events.append("materialize")
        return pending

    def retire(actual):
        assert actual is pending and provider._active_inputs is pending
        events.append("retire")
        provider._active_inputs = None

    provider.prepare_workspace, provider.retire_generation = prepare, retire
    owner._draft_window_graph_binding_provider = provider
    owner._eagle3_graph_session = SimpleNamespace(_capture_call=None, resolve=resolve)
    owner._eagle3_window_next_step, owner.num_speculative_steps = 0, 3
    owner.input_batch = SimpleNamespace(
        num_reqs=2,
        has_prefill=True,
        idx_mapping=torch.tensor([1, 0]),
        query_start_loc_np=np.array([0, 3, 7], dtype=np.int32),
    )

    def gather(mapping, *, num_reqs_padded):
        assert mapping is owner.input_batch.idx_mapping and num_reqs_padded == 2
        events.append("gather_full")

    owner.block_tables = SimpleNamespace(gather_block_tables=gather)
    owner._materialize_eagle3_step = materialize
    owner.model_state = SimpleNamespace(attn_metadata=target)
    owner._quarantine_eagle3_window = lambda: events.append("quarantine")
    owner._prepare_replicated_prefill_attn = lambda metadata, mappings, *args, **kwargs: (metadata, mappings)
    owner.draft_attn_layer_names, owner.pcp_manager = {"draft"}, None
    model_tokens = 8 if padding == "tensor" else 7
    positions = torch.cat((absolute, torch.zeros(1, dtype=absolute.dtype))) if padding == "tensor" else absolute.clone()
    owner.input_buffers = SimpleNamespace(positions=positions)
    if padding == "extra_request":
        target_layer.actual_seq_lengths_q.append(8)
        target_layer.seq_lens_list.append(1)
    owner.last_token_indices, owner.idx_mapping = torch.tensor([2, 6]), torch.tensor([1, 0])
    owner.draft_tokens = torch.zeros((2, 3), dtype=torch.int64)
    owner.hidden_states = torch.zeros((2, 8))
    owner.sample_src_positions = torch.zeros(2, dtype=torch.int64)
    owner.temperature = owner.seeds = owner.current_draft_step = owner.draft_logits = object()
    output = torch.arange(model_tokens * 8, dtype=torch.float32).reshape(model_tokens, 8)

    def forward(current, num_tokens, metadata, mappings, across, actual_mode, mm_inputs):
        assert current is owner and num_tokens == model_tokens and actual_mode is mode and mappings is slot_map
        assert provider._active_inputs is pending
        layer = metadata["draft"]
        assert metadata is not target and layer is not target_layer
        assert layer.block_tables is bank.window_block_table
        assert layer.seq_lens is layer.seq_lens_gpu is bank.device_window_exact
        assert layer.seq_lens_cpu is bank.cpu_window_exact and layer.seq_lens_list == [5, 11]
        assert layer.attn_state is AscendAttentionState.ChunkedPrefill
        assert layer.slot_mapping is slots and layer.query_start_loc is query and layer.positions is absolute
        assert layer.actual_seq_lengths_q == [3, 7]
        assert layer.actual_seq_lengths_q is not target_layer.actual_seq_lengths_q
        assert layer.seq_lens_list is not target_layer.seq_lens_list
        assert target_layer.seq_lens_list == [77, 88]
        assert target_layer.attn_state is AscendAttentionState.PrefillNoCache
        torch.testing.assert_close(target_layer.block_tables, original_target_table)
        events.append("native")
        if fail:
            raise ValueError("native prefill failed")
        return output, output

    def sample(hidden, *args):
        sampled.append(hidden.clone())
        return torch.tensor([21, 22], dtype=torch.int64)

    owner.sample_draft = sample
    monkeypatch.setattr(AscendTargetDependentARSpeculator, "_run_model", forward)
    if padding == "extra_request":
        with pytest.raises(NotImplementedError, match="additional padding requests"):
            owner._prefill(2, model_tokens, target, slot_map, None, mode, skip_sample=skip_sample)
        assert events == [] and provider._active_inputs is None and not sampled
        assert target_layer.actual_seq_lengths_q == [3, 7, 8]
        return
    if fail:
        with pytest.raises(ValueError, match="native prefill failed"):
            owner._prefill(2, model_tokens, target, slot_map, None, mode, skip_sample=skip_sample)
        assert events == ["prepare", "gather_full", "materialize", "native", "quarantine"]
        assert provider._active_inputs is pending and owner._eagle3_window_next_step == 0
        assert not sampled
    else:
        owner._prefill(2, model_tokens, target, slot_map, None, mode, skip_sample=skip_sample)
        assert events == ["prepare", "gather_full", "materialize", "native", "retire"]
        assert provider._active_inputs is None and owner._eagle3_window_next_step == 1
        if skip_sample:
            assert not sampled and not bool(owner.draft_tokens.any())
        else:
            torch.testing.assert_close(sampled[0], output[[2, 6]])
            assert owner.draft_tokens[:, 0].tolist() == [21, 22]
    assert not hasattr(owner, "_draft_window_runtime")
    assert owner._eagle3_window_native_prefill_active is False
    assert target_layer.slot_mapping is slots and target_layer.query_start_loc is query
    torch.testing.assert_close(absolute, torch.arange(7) + 70)


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE])
@pytest.mark.parametrize("full_end,start", [(1025, 0), (2051, 1024)])
def test_dp_synchronized_native_first_without_local_prefill(monkeypatch, mode, full_end, start):
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

    from vllm_ascend.attention.attention_v1 import AscendAttentionState
    from vllm_ascend.worker.v2.spec_decode.draft_window import DraftWindowGraphSource
    from vllm_ascend.worker.v2.spec_decode.draft_window_backends import DraftWindowGroupRoute, DraftWindowProviderStatus
    from vllm_ascend.worker.v2.spec_decode.draft_window_graph import (
        DraftWindowGraphBindingProvider,
        DraftWindowGraphRoute,
    )
    from vllm_ascend.worker.v2.spec_decode.draft_window_policy import resolve_draft_window_policy

    owner: Any = object.__new__(Eagle3GraphIntegrationMixin)
    owner.method, owner.num_speculative_steps, owner.max_model_len = "eagle3", 3, 4096
    owner.speculative_config = SimpleNamespace(uses_dynamic_speculative_decoding=lambda: False)
    table = torch.arange(10, 42, dtype=torch.int32).reshape(1, 32)
    valid_blocks = (full_end + 127) // 128
    counts = torch.tensor([[valid_blocks]], dtype=torch.int32)
    owner.block_tables = SimpleNamespace(
        input_block_tables=[table],
        num_blocks=SimpleNamespace(cpu=counts, gpu=counts.clone()),
        block_sizes=[128],
        kernel_block_sizes=[128],
        gather_block_tables=lambda mapping, **kwargs: None,
    )
    group = DraftWindowGroupRoute(0, ("draft",), 128, 128, DraftWindowProviderStatus.DIRECT_DENSE)
    source = DraftWindowGraphSource(
        policy=resolve_draft_window_policy(requested_window_size=1024, method="eagle3"),
        group=group,
        block_tables=owner.block_tables,
    )
    provider = owner._draft_window_graph_binding_provider = DraftWindowGraphBindingProvider(source=source)
    capture = provider.register_capture_identity(
        route=DraftWindowGraphRoute.EAGLE3_FIRST_PASS,
        descriptor=BatchExecutionDescriptor(CUDAGraphMode.FULL, 4, 1, uniform_token_count=4),
    )
    bank = provider.startup_capture_workspace(capture)
    owner._eagle3_graph_session = SimpleNamespace(_capture_call=None, resolve=lambda **kwargs: capture)
    owner._initialize_eagle3_draft_window_state()
    owner.input_batch = SimpleNamespace(
        num_reqs=1,
        req_ids=["decoded-local-request"],
        idx_mapping_np=np.array([0]),
        idx_mapping=torch.tensor([0]),
        num_computed_prefill_tokens_np=np.array([full_end - 4]),
        prefill_len_np=np.array([1021]),
        num_scheduled_tokens=np.array([4]),
        num_draft_tokens_per_req=None,
        is_prefilling_np=np.array([False]),
        has_prefill=False,
        query_start_loc_np=np.array([0, 4], dtype=np.int32),
        seq_lens_np=np.array([full_end]),
    )
    query = torch.tensor([0, 4], dtype=torch.int32)
    positions = torch.arange(full_end - 4, full_end, dtype=torch.int64)
    slots = table[0, positions // 128].to(torch.int64) * 128 + positions % 128
    target_layer = SimpleNamespace(
        block_tables=table,
        seq_lens=torch.tensor([full_end], dtype=torch.int32),
        seq_lens_list=[full_end],
        actual_seq_lengths_q=[4],
        num_actual_tokens=4,
        query_start_loc=query,
        positions=positions,
        slot_mapping=slots,
        attn_state=AscendAttentionState.DecodeOnly,
    )
    target = {"draft": target_layer}
    owner.input_buffers = SimpleNamespace(positions=torch.cat((positions.clone(), torch.zeros(4, dtype=torch.int64))))
    owner.last_token_indices, owner.idx_mapping = torch.tensor([3]), torch.tensor([0])
    owner.draft_tokens, owner.hidden_states = torch.zeros((1, 3), dtype=torch.int64), torch.zeros((1, 2))
    owner.sample_src_positions = torch.zeros(1, dtype=torch.int64)
    owner.temperature = owner.seeds = owner.current_draft_step = owner.draft_logits = None
    owner.sample_draft = lambda *args: torch.tensor([23])
    owner._prepare_replicated_prefill_attn = lambda metadata, mappings, *args, **kwargs: (metadata, mappings)
    owner.model_state = SimpleNamespace(attn_metadata=target)
    owner.draft_attn_layer_names, owner.pcp_manager = {"draft"}, None
    observed = []

    def forward(current, num_tokens, metadata, mappings, across, actual_mode, mm_inputs):
        layer = metadata["draft"]
        assert actual_mode is mode and num_tokens == 8 and current._eagle3_window_native_prefill_active
        assert layer.actual_seq_lengths_q == [4] and layer.num_actual_tokens == 4
        assert layer.attn_state is AscendAttentionState.ChunkedPrefill
        assert layer.block_tables is bank.window_block_table
        assert bank.cpu_window_exact.tolist() == [full_end - start]
        assert bank.device_window_exact.tolist() == [full_end - start]
        assert bank.window_block_table[0, :9].tolist() == table[0, start // 128 : valid_blocks].tolist()
        assert owner._eagle3_window_proposal_plan[0].window_start == start
        assert owner._eagle3_window_proposal_plan[0].full_exact_end == full_end
        assert layer.query_start_loc is query and layer.positions is positions and layer.slot_mapping is slots
        assert mappings["draft"] is slots and target_layer.attn_state is AscendAttentionState.DecodeOnly
        torch.testing.assert_close(current.input_buffers.positions[:4], positions)
        observed.append(provider._active_inputs)
        return torch.zeros((8, 2)), torch.zeros((8, 2))

    monkeypatch.setattr(AscendTargetDependentARSpeculator, "_run_model", forward)
    owner._eagle3_window_proposal = draft_window_eager.DraftWindowProposalContext.from_batch(owner.input_batch)
    try:
        owner._prefill(1, 8, target, {"draft": slots}, None, mode)
    finally:
        owner._eagle3_window_proposal = None
    assert len(observed) == 1
    assert observed[0].proposal.full_read_rows == (False,)
    assert provider._active_inputs is None and not owner._eagle3_window_native_prefill_active
    assert owner._eagle3_window_next_step == 1 and owner.draft_tokens[0, 0] == 23
    assert bank.cpu_window_exact.tolist() == [full_end - start]  # Retirement did not clear live backing.
    torch.testing.assert_close(table, torch.arange(10, 42, dtype=torch.int32).reshape(1, 32))
    torch.testing.assert_close(target_layer.positions, positions)
    assert target_layer.seq_lens_list == [full_end] and target_layer.query_start_loc is query


def test_terminal_prefill_rejects_a_query_outside_the_frozen_window_before_writing_bank(proposal, monkeypatch):
    from vllm_ascend.worker.v2.spec_decode import draft_window_eagle3

    owner, batch, session, runtime, _ = proposal
    owner.input_batch, owner.max_model_len = batch, 32
    batch.num_computed_prefill_tokens_np = np.array([0])
    batch.prefill_len_np = batch.num_scheduled_tokens = batch.seq_lens_np = np.array([10])
    batch.is_prefilling_np, batch.has_prefill = np.array([True]), True
    batch.query_start_loc_np = np.array([0, 10], dtype=np.int32)
    runtime.source.window_size = 4
    runtime.source.group = SimpleNamespace(gid=0, table_column_span=4)
    monkeypatch.setattr(
        draft_window_eagle3,
        "gather_valid_num_blocks_cpu",
        lambda *args, **kwargs: pytest.fail("unsupported query reached bank materialization"),
    )
    with (
        pytest.raises(NotImplementedError, match="query begins before its frozen read window"),
        owner._eagle3_draft_window_proposal_scope(batch, dummy_run=False, is_profile=False),
    ):
        owner._materialize_eagle3_step(step=0, num_reqs=1, num_reqs_padded=1, output_workspace=object())
    assert runtime._active_inputs is None


def test_mixed_prefill_query_guard_keeps_full_rows_and_checks_only_actual_window_rows(proposal, monkeypatch):
    from vllm_ascend.worker.v2.spec_decode import draft_window_eagle3

    owner, batch, session, runtime, _ = proposal
    owner.input_batch, owner.max_model_len = batch, 32
    batch.num_reqs, batch.req_ids = 2, ["continuing", "terminal"]
    batch.idx_mapping_np, batch.idx_mapping = np.array([0, 1]), torch.tensor([0, 1])
    batch.num_computed_prefill_tokens_np = np.array([10, 8])
    batch.prefill_len_np, batch.num_scheduled_tokens = np.array([100, 10]), np.array([10, 2])
    batch.seq_lens_np, batch.query_start_loc_np = np.array([20, 10]), np.array([0, 10, 12], dtype=np.int32)
    batch.is_prefilling_np, batch.has_prefill = np.array([True, True]), True
    runtime.source.window_size = 4
    runtime.source.group = SimpleNamespace(gid=0, table_column_span=4)
    monkeypatch.setattr(
        draft_window_eagle3, "gather_valid_num_blocks_cpu", lambda *args, **kwargs: torch.tensor([5, 3, 0, 0])
    )
    calls = []

    def materialize(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace()

    runtime.materialize = materialize
    context = owner._eagle3_window_proposal = draft_window_eager.DraftWindowProposalContext.from_batch(batch)
    try:
        owner._materialize_eagle3_step(step=0, num_reqs=2, num_reqs_padded=4, output_workspace=object())
    finally:
        owner._eagle3_window_proposal = None
    assert context.full_read_rows == (True, False)
    assert len(calls) == 1 and calls[0]["cpu_full_exact"].tolist() == [20, 10, 0, 0]
    assert [plan.window_visible_length for plan in calls[0]["frozen_row_plans"]] == [4, 2]


@pytest.mark.parametrize("values", [np.array([10.5]), np.array([-1]), np.array([], dtype=np.int32)])
def test_eagle_rejects_invalid_canonical_lengths_before_writing_a_bank(proposal, monkeypatch, values):
    from vllm_ascend.worker.v2.spec_decode import draft_window_eagle3

    owner, batch, _, runtime, _ = proposal
    owner.input_batch = batch
    batch.seq_lens_np = values
    monkeypatch.setattr(
        draft_window_eagle3, "gather_valid_num_blocks_cpu", lambda *args, **kwargs: pytest.fail("invalid bank write")
    )
    with pytest.raises(ValueError), owner._eagle3_draft_window_proposal_scope(batch, dummy_run=False, is_profile=False):
        owner._materialize_eagle3_step(step=0, num_reqs=1, num_reqs_padded=1, output_workspace=object())
    assert runtime._active_inputs is None


@pytest.mark.parametrize("consumers", [1, 2])
@pytest.mark.parametrize("visited", [0, 1, 2, 3])
def test_capture_scope_selects_banks_once_and_requires_every_real_consumer(consumers, visited):
    from vllm_ascend.worker.v2.spec_decode.draft_window_graph import DraftWindowGraphRoute

    session: Any = object.__new__(Eagle3GraphSession)
    session._capture_call = None
    session.owner = SimpleNamespace(
        max_model_len=32,
        input_buffers=SimpleNamespace(query_start_loc=torch.tensor([0, 4], dtype=torch.int32)),
    )
    captures = tuple(
        SimpleNamespace(
            descriptor=SimpleNamespace(num_reqs=1),
            workspace=SimpleNamespace(
                window_block_table=torch.tensor([[index, index + 1]], dtype=torch.int32),
                device_window_exact=torch.tensor([5 + index], dtype=torch.int32),
                cpu_window_exact=torch.tensor([5 + index], dtype=torch.int32),
            ),
        )
        for index in range(consumers)
    )
    selections = []

    def select(**kwargs):
        selections.append(kwargs)
        return captures

    session.provider = SimpleNamespace(
        source=SimpleNamespace(window_size=4, group=SimpleNamespace(table_column_span=2)),
        capture_banks_for=select,
        startup_capture_workspace=lambda capture: capture.workspace,
    )
    full = SimpleNamespace(slot_mapping=torch.tensor([10]), seq_lens_list=[99])

    def execute():
        with session.capture_call(first=consumers == 1, n=1, tokens=4 if consumers == 1 else 1):
            for index in range(visited):
                private = session.capture_metadata({"layer": full})["layer"]
                assert private.block_tables is captures[index].workspace.window_block_table
                assert private.slot_mapping is full.slot_mapping
                assert full.seq_lens_list == [99]

    if visited == consumers:
        execute()
    else:
        with pytest.raises(RuntimeError, match="consumer"):
            execute()
    assert session._capture_call is None
    assert selections == [
        dict(
            route=DraftWindowGraphRoute.EAGLE3_FIRST_PASS if consumers == 1 else DraftWindowGraphRoute.EAGLE3_AR_STEP,
            num_reqs=1,
            num_tokens=4 if consumers == 1 else 1,
        )
    ]
