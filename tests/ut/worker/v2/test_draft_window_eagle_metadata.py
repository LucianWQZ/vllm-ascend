# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Shared query construction and Eagle's real eager-reader boundary."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator

from vllm_ascend.worker.v2.spec_decode import draft_window_eager
from vllm_ascend.worker.v2.spec_decode.draft_window_eager import AscendDraftAttentionReadInputs
from vllm_ascend.worker.v2.spec_decode.draft_window_eagle3 import Eagle3DraftWindowMixin
from vllm_ascend.worker.v2.spec_decode.draft_window_eagle3_adapter import Eagle3GraphIntegrationMixin
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


@pytest.mark.parametrize("active,fail", [(False, False), (False, True), (True, False), (True, True)])
def test_eager_forward_is_delegated_once_and_reader_retires_after_it(monkeypatch, active, fail):
    calls = []
    metadata = object()
    inputs = SimpleNamespace(identity=object())

    def forward(self, *args):
        calls.append(("forward", args))
        if fail:
            raise ValueError("native forward failed")
        return "result"

    monkeypatch.setattr(AscendTargetDependentARSpeculator, "_generate_draft", forward)

    class Owner(Eagle3GraphIntegrationMixin):
        def _eagle3_window_is_active(self):
            return active

        def _finish_eagle3_window_step(self, current, *, succeeded):
            assert current is inputs
            calls.append(("retire", succeeded))

    owner: Any = object.__new__(Owner)
    owner._eagle3_window_live_inputs = inputs
    owner._draft_window_runtime_state = SimpleNamespace(read_metadata=lambda identity: metadata)
    args = (2, 4, metadata, object(), object())
    if fail:
        with pytest.raises(ValueError, match="native forward failed"):
            owner._generate_draft(*args)
    else:
        assert owner._generate_draft(*args) == "result"
    assert calls[0][0] == "forward" and calls[0][1][:5] == args
    assert calls[1:] == ([("retire", not fail)] if active else [])
