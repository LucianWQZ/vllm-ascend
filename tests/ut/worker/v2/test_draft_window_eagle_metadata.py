# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Shared query construction and Eagle's real eager-reader boundary."""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator
from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator import TargetDependentARSpeculator

from vllm_ascend.worker.v2.spec_decode import draft_window_eager
from vllm_ascend.worker.v2.spec_decode.draft_window_eager import AscendDraftAttentionReadInputs
from vllm_ascend.worker.v2.spec_decode.draft_window_eagle3 import (
    Eagle3DraftWindowMixin,
    _eagle3_graph_scheduled_consumers,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_eagle3_adapter import Eagle3GraphIntegrationMixin
from vllm_ascend.worker.v2.spec_decode.draft_window_eagle3_capture import Eagle3GraphSession
from vllm_ascend.worker.v2.spec_decode.draft_window_graph import DraftWindowGraphCarrierState
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


@pytest.fixture
def assembly(monkeypatch):
    table = torch.arange(16, dtype=torch.int32).reshape(4, 4)
    lengths = torch.tensor([7, 9, 0, 0], dtype=torch.int32)
    read = AscendDraftAttentionReadInputs((table,), lengths.clone(), lengths, lengths.clone(), 9)
    owner = SimpleNamespace(
        arange_np=np.arange(5, dtype=np.int32),
        num_query_per_req=8,
        attn_groups=object(),
        block_tables=SimpleNamespace(
            input_block_tables=[torch.full_like(table, 99)],
            slot_mappings=torch.arange(32).reshape(1, 32),
            cp_size=1,
        ),
        input_buffers=SimpleNamespace(
            query_start_loc=torch.arange(5, dtype=torch.int32),
            positions=torch.arange(32, dtype=torch.int64),
        ),
        input_batch=SimpleNamespace(is_prefilling_np=np.array([False] * 4)),
        vllm_config=SimpleNamespace(parallel_config=object()),
        model_state=SimpleNamespace(attn_metadata={"target": SimpleNamespace(attn_state="decode")}),
        kv_cache_config=object(),
    )
    from vllm_ascend.worker.v2 import attn_utils

    owner.assembly_calls = []

    def builder(**kwargs):
        owner.assembly_calls.append(kwargs)
        return {"draft": SimpleNamespace(**kwargs)}

    monkeypatch.setattr(attn_utils, "build_attn_metadata", builder)
    return owner, read


@pytest.mark.parametrize(
    "actual,width,query,expected,max_query,tokens",
    [
        (1, None, None, [0, 8, 8, 8, 8], 8, 8),
        (4, 8, None, [0, 8, 16, 24, 32], 8, 32),
        (2, 1, np.array([0, 3, 7], dtype=np.int32), [0, 3, 7, 7, 7], 4, 7),
        (1, 1, None, [0, 1, 1, 1, 1], 1, 1),
    ],
)
def test_shared_assembler_preserves_uniform_full_capture_and_nonuniform_queries(
    assembly, actual, width, query, expected, max_query, tokens
):
    owner, read = assembly
    result = draft_window_eager.build_draft_window_attn_metadata(
        owner,
        num_reqs=actual,
        num_reqs_padded=4,
        num_tokens_padded=tokens,
        read_inputs=read,
        query_width=width,
        causal=True,
        query_start_loc_np=query,
        dcp_local_seq_lens=None,
    )
    assert len(owner.assembly_calls) == 1
    assert result["draft"].query_start_loc.data_ptr() == owner.input_buffers.query_start_loc.data_ptr()
    assert result["draft"].query_start_loc.shape == (5,)
    result = owner.assembly_calls[0]
    assert result["positions"].data_ptr() == owner.input_buffers.positions.data_ptr()
    assert result["attn_state"] == "decode"
    assert result["parallel_config"] is owner.vllm_config.parallel_config
    assert result["query_start_loc_cpu"].tolist() == expected
    assert result["max_query_len"] == max_query
    assert result["draft_window_read_inputs"] is read
    assert result["seq_lens"] is read.seq_lens
    assert result["seq_lens_cpu"] is read.seq_lens_cpu_exact
    assert result["seq_lens_cpu_upper_bound"] is read.seq_lens_cpu_upper_bound
    assert result["block_tables"][0] is read.block_tables[0]
    slots = owner.block_tables.slot_mappings
    assert result["slot_mappings"].data_ptr() == slots.data_ptr()
    assert result["slot_mappings"].shape == (1, tokens)
    assert torch.equal(owner.block_tables.input_block_tables[0], torch.full_like(read.block_tables[0], 99))


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


def test_graph_owner_keeps_normal_prefill_decode_metadata_and_updates():
    for name in ("_prefill", "_build_attn_metadata", "_update_decode_attn_metadata"):
        assert getattr(Eagle3GraphIntegrationMixin, name) is getattr(AscendTargetDependentARSpeculator, name)


def test_graph_owner_rejects_unintended_eager_business_forward(monkeypatch):
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
    with pytest.raises(RuntimeError, match="FULL Graph managers"):
        owner._run_model(4, {}, None, None)


@pytest.mark.parametrize("scheduled", [-1, 1, 2, 4, True])
def test_eagle_partial_schedule_is_not_dynamic_k(scheduled):
    with pytest.raises(NotImplementedError):
        _eagle3_graph_scheduled_consumers(3, scheduled)


@pytest.fixture
def proposal():
    owner: Any = Eagle3DraftWindowMixin()
    owner.method, owner.num_speculative_steps = "eagle3", 3
    owner.speculative_config = SimpleNamespace(uses_dynamic_speculative_decoding=lambda: False)
    runtime: Any = SimpleNamespace(
        source=SimpleNamespace(policy=SimpleNamespace(normalized_method="eagle3"), expected_future_offset=3),
        _active_inputs=None,
        quarantined_identity=None,
        _last_retired_identity=None,
        require_reusable=lambda: None,
    )

    def retire(identity):
        assert runtime._active_inputs.identity is identity
        runtime._last_retired_identity, runtime._active_inputs = identity, None

    runtime.retire_generation = retire
    runtime.quarantine_generation = lambda identity: setattr(runtime, "quarantined_identity", identity)
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
    session.owner, session._carriers = owner, []
    session.provider = runtime
    owner._eagle3_graph_session = session
    owner._require_eagle3_target_metadata = lambda: {}
    session.resolve = lambda **kwargs: object()
    session.provider.bind_read_carrier = lambda **kwargs: SimpleNamespace(state=DraftWindowGraphCarrierState.BOUND)

    def inputs(step, **changes):
        context = owner._require_eagle3_window_context()
        values = dict(proposal=context, step=step)
        values.update(changes)
        identity = SimpleNamespace(**values)
        current = SimpleNamespace(identity=identity, retired=False)
        current.validate_for = lambda actual: actual is identity or pytest.fail("replaced identity")
        runtime._active_inputs = owner._eagle3_window_live_inputs = current
        return current

    return owner, batch, session, runtime, inputs


def test_real_proposal_zero_uses_first_reader_and_keeps_static_k(proposal):
    owner, batch, session, runtime, inputs = proposal
    with owner._eagle3_draft_window_proposal_scope(batch, dummy_run=False, is_profile=False, num_speculative_tokens=0):
        assert owner._eagle3_window_expected_steps == 1
        current = inputs(0)
        session.stage(current, first=True, n=1, tokens=4, ordinal=0)
        session._carriers[-1].state = DraftWindowGraphCarrierState.UPDATED
    assert runtime._last_retired_identity is current.identity
    assert owner.num_speculative_steps == owner._eagle3_window_expected_steps == 3
    assert not session._carriers


def test_real_proposal_keeps_each_submitted_consumer_until_update(proposal):
    owner, batch, session, runtime, inputs = proposal
    with owner._eagle3_draft_window_proposal_scope(batch, dummy_run=False, is_profile=False):
        for step in range(3):
            current = inputs(step)
            session.stage(current, first=step == 0, n=1, tokens=4 if step == 0 else 1, ordinal=max(0, step - 1))
            assert owner._eagle3_window_next_step == step + 1
            assert runtime._last_retired_identity is current.identity
            assert owner._eagle3_window_live_inputs is None
            assert len(session._carriers) == step + 1
            session._carriers[-1].state = DraftWindowGraphCarrierState.UPDATED
    assert not session._carriers
    assert not getattr(owner, "_draft_window_graph_submission_uncertain", False)


@pytest.mark.parametrize(
    "damage", ["missing_reader", "missing_update", "wrong_request", "cloned_proposal", "wrong_step", "wrong_ordinal"]
)
def test_real_proposal_rejects_unconsumed_or_misattributed_reader(proposal, damage):
    owner, batch, session, runtime, inputs = proposal
    with (
        pytest.raises(RuntimeError),
        owner._eagle3_draft_window_proposal_scope(batch, dummy_run=False, is_profile=False, num_speculative_tokens=0),
    ):
        if damage != "missing_reader":
            context = owner._require_eagle3_window_context()
            changes = {}
            if damage == "wrong_request":
                changes["proposal"] = replace(
                    context,
                    ordered_request_ids=("stale",),
                )
            elif damage == "cloned_proposal":
                changes["proposal"] = replace(context)
            current = inputs(1 if damage == "wrong_step" else 0, **changes)
            session.stage(current, first=True, n=1, tokens=4, ordinal=1 if damage == "wrong_ordinal" else 0)
    assert owner._draft_window_graph_submission_uncertain
    assert owner._eagle3_window_proposal_context is None
    if damage in ("wrong_request", "cloned_proposal", "wrong_step", "wrong_ordinal"):
        assert runtime.quarantined_identity is current.identity


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
    runtime.source.full_source = object()
    calls = []
    monkeypatch.setattr(
        draft_window_eagle3, "gather_valid_num_blocks_cpu", lambda *args, **kwargs: torch.tensor([4], dtype=torch.int32)
    )

    def plan(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            read_descriptors=(),
            cpu_window_exact=kwargs["cpu_full_exact"],
            cpu_window_upper=kwargs["cpu_apply_time_full_upper"],
            cpu_valid_num_blocks=kwargs["cpu_valid_num_blocks"],
        )

    monkeypatch.setattr(draft_window_eagle3, "build_phase_aware_materialization_plan", plan)

    def materialize(**kwargs):
        identity = SimpleNamespace(proposal=kwargs["proposal"], step=kwargs["step"])
        current = SimpleNamespace(identity=identity, retired=False)
        current.validate_for = lambda expected: expected is identity
        runtime._active_inputs = current
        return current

    runtime.materialize = materialize
    with owner._eagle3_draft_window_proposal_scope(batch, dummy_run=False, is_profile=False):
        frozen = None
        for step in range(3):
            if step == 1:
                owner._snapshot_eagle3_decode_step_one(num_reqs=1, num_reqs_padded=1)
            current = owner._materialize_eagle3_step(
                step=step,
                num_reqs=1,
                num_reqs_padded=first_padding if step == 0 else 1,
                seq_lens_cpu_upper_bound=batch.seq_lens_cpu_upper_bound,
            )
            frozen = frozen or owner._eagle3_window_proposal_plan
            assert owner._eagle3_window_proposal_plan is frozen
            assert frozen[0].window_start == 8
            session.stage(current, first=step == 0, n=1, tokens=4 if step == 0 else 1, ordinal=max(0, step - 1))
            session._carriers[-1].state = DraftWindowGraphCarrierState.UPDATED
    assert [call["cpu_full_exact"].tolist() for call in calls] == [[10] + [0] * (first_padding - 1), [8], [9]]
    assert [call["cpu_apply_time_full_upper"].tolist() for call in calls] == [
        [12] + [0] * (first_padding - 1),
        [13],
        [14],
    ]
    assert all(call["frozen_row_plans"] is frozen for call in calls)


@pytest.mark.parametrize("first,tokens,ordinal", [(True, 4, 0), (False, 1, 0), (False, 1, 1)])
def test_eagle_capture_lookup_keeps_each_real_shape_and_consumer(first, tokens, ordinal):
    from vllm_ascend.worker.v2.spec_decode.draft_window_graph import DraftWindowGraphRoute

    session: Any = object.__new__(Eagle3GraphSession)
    first_route = DraftWindowGraphRoute.EAGLE3_FIRST_PASS
    ar_route = DraftWindowGraphRoute.EAGLE3_AR_STEP
    session._captures = {
        (first_route, 1, 4, 0): object(),
        (ar_route, 1, 1, 0): object(),
        (ar_route, 1, 1, 1): object(),
    }
    expected = session._captures[(first_route if first else ar_route, 1, tokens, ordinal)]
    assert session.resolve(first=first, n=1, tokens=tokens, ordinal=ordinal) is expected
    with pytest.raises(RuntimeError, match="no exact capture bank"):
        session.resolve(first=first, n=2, tokens=tokens, ordinal=ordinal)
    with pytest.raises(RuntimeError, match="no exact capture bank"):
        session.resolve(first=first, n=1, tokens=tokens + 1, ordinal=ordinal)


@pytest.fixture
def parallel_hook(monkeypatch):
    from vllm_ascend.worker.v2.spec_decode import draft_window_graph_metadata
    from vllm_ascend.worker.v2.spec_decode.draft_window_eager import DraftWindowGraphInputMixin

    class Baseline:
        def _build_attn_metadata(self, *args, **kwargs):
            return (args, kwargs)

    class Owner(DraftWindowGraphInputMixin, Baseline):
        pass

    owner: Any = Owner()
    owner._draft_window_graph_binding_provider = object()
    owner._draft_window_bypass_active = False
    owner._draft_window_selected_metadata = object()
    owner.draft_attn_layer_names, owner.num_query_per_req = {"draft.layer"}, 8
    owner.block_tables = SimpleNamespace(slot_mappings=torch.arange(32, dtype=torch.int64).reshape(1, 32))
    inputs, calls = object(), []

    def materialize(**kwargs):
        calls.append(kwargs)
        return inputs

    owner._materialize_draft_window_read_inputs = materialize
    monkeypatch.setattr(draft_window_graph_metadata, "defer_parallel_graph_metadata", lambda *args, **kwargs: True)
    desc = SimpleNamespace(cg_mode=SimpleNamespace(name="FULL"), num_reqs=2, num_tokens=16)
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
    assert result is None and owner._draft_window_live_inputs is inputs
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
