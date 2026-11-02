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
        input_buffers=SimpleNamespace(query_start_loc=torch.arange(5, dtype=torch.int32)),
        kv_cache_config=object(),
    )
    monkeypatch.setattr(draft_window_eager.upstream_speculator_module, "build_attn_metadata", lambda **kwargs: kwargs)
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
    session.write_reference = lambda identity, target: None
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
    assert owner._eagle3_window_static_k == owner._eagle3_window_expected_steps == 3
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
