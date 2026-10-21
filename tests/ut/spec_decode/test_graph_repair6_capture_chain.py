# SPDX-License-Identifier: Apache-2.0
"""Real DSpark.capture -> Ascend/DFlash/Cuda manager -> real builder/forward.

Only native graph, communicator, context and sampling primitives are replaced.
The upstream closure is NOT changed to pass FULL: it passes NONE during capture.
"""

from types import SimpleNamespace

import torch
from vllm.config.compilation import CUDAGraphMode

from tests.ut.spec_decode.test_graph_repair4_integration import _lifetime_fixture
from tests.ut.spec_decode.test_graph_repair6_contracts import context
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator


def capture_owner(monkeypatch, enabled=True):
    import vllm.v1.worker.gpu.cudagraph_utils as base
    import vllm.v1.worker.gpu.spec_decode.dflash.cudagraph as capture
    import vllm.v1.worker.gpu.spec_decode.dflash.speculator as upstream

    import vllm_ascend.worker.v2.spec_decode.dflash.aclgraph as manager_module
    import vllm_ascend.worker.v2.spec_decode.dspark.speculator as spark
    from vllm_ascend.worker.v2.spec_decode.dflash.capture import ascend_prepare_dflash_inputs_to_capture

    owner, session, manager, descs, calls = _lifetime_fixture(
        monkeypatch, block_size=128, window_size=1024, row_width=32, method="dspark"
    )
    owner.__class__ = AscendDSparkSpeculator
    del owner.build_draft_attn_metadatas
    owner.query_cudagraph_manager = manager
    owner.vllm_config = SimpleNamespace(additional_config={"draft_window_graph_numeric_diagnostic": enabled})
    from vllm_ascend.ascend_config import AscendConfig

    # Validate the real schema before crossing the native-only Host boundary.
    AscendConfig(sparse_kv_offload_config=None, **owner.vllm_config.additional_config)
    owner.device = torch.device("cpu")
    owner.num_speculative_steps = 7
    owner.max_model_len = 4096
    owner._speculator_name = "DSpark"
    owner.sample_from_anchor = True
    owner._draft_topk = None
    owner.enable_adaptive_verification = False
    owner._anchor_idx = torch.zeros(8, dtype=torch.int64)
    owner._sample_logits = lambda logits, *a: logits.argmax(-1)
    owner.input_buffers.input_ids = torch.zeros(64, dtype=torch.int64)
    for name in ("sample_indices", "sample_pos", "sample_idx_mapping", "sample_col"):
        setattr(owner, name, torch.zeros(56, dtype=torch.int64))
    owner.temperature = owner.seeds = torch.zeros(8)
    owner.draft_logits = torch.zeros(56, 4)
    owner.draft_tokens = torch.zeros(8, 7, dtype=torch.int64)
    owner.sample_draft = lambda hidden, *a: torch.zeros(len(hidden), dtype=torch.int64)
    owner.draft_attn_layer_names = {"draft.layers.0.self_attn"}
    cache = torch.zeros(257, 128, 1, 1, dtype=torch.bfloat16)

    class Backbone(torch.nn.Module):
        def forward(self, input_ids, positions, inputs_embeds):
            return torch.stack((input_ids, positions), 1).to(torch.bfloat16) + cache[1, 0, 0, 0]

        def compute_draft_logits(self, hidden):
            return hidden

        def markov_embed(self, prev):
            return torch.zeros(len(prev), 2, dtype=torch.bfloat16)

        def markov_bias(self, value):
            return value

    owner.model = Backbone()
    manager.graphs = {}
    manager.max_num_reqs = 8
    manager.dp_size = 1
    manager._capture_descs = {CUDAGraphMode.FULL: list(reversed(list(descs.values())))}
    manager._max_full_descs_to_capture = None
    manager._capture_mem_samples = None
    manager.use_breakable_cg = False
    manager.pool = None
    monkeypatch.setattr(spark, "experimental_draft_window_graph_characterization_enabled", lambda: True)
    monkeypatch.setattr(manager_module, "communicator_switch", context)
    monkeypatch.setattr(manager_module, "model_capture_wrapper", context)
    monkeypatch.setattr(upstream, "set_forward_context", context)
    monkeypatch.setattr(capture, "_prepare_dflash_inputs_to_capture", ascend_prepare_dflash_inputs_to_capture)
    monkeypatch.setattr(base, "graph_capture", context)
    monkeypatch.setattr(base, "is_global_first_rank", lambda: False)
    monkeypatch.setattr(base, "set_graph_pool_id", lambda _: None)
    monkeypatch.setattr(base, "current_platform", SimpleNamespace(graph_pool_handle=lambda: 0))
    monkeypatch.setattr(base, "current_stream", lambda: None)
    monkeypatch.setattr(
        base, "get_offloader", lambda: SimpleNamespace(sync_prev_onload=lambda: None, join_after_forward=lambda: None)
    )
    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: SimpleNamespace(replay=lambda: None))
    monkeypatch.setattr(torch.cuda, "graph", context)
    owner.capture()
    return owner, session, manager, descs, cache


def test_real_capture_sink_entry_uses_actual_none_mode(monkeypatch):
    owner, session, manager, descs, _ = capture_owner(monkeypatch)
    sink = owner._draft_graph_output_references
    assert sink.frozen and set(sink.outputs) == {7, 14, 28}
    assert len(manager.graphs) == 3 and len(session.provider.registered_capture_identities(route=session.route)) == 3
    assert all(sink.get(owner, n).shape == (n, 2) for n in (7, 14, 28))


def test_capture_diagnostic_default_off_has_no_sink(monkeypatch):
    owner, _, manager, _, _ = capture_owner(monkeypatch, False)
    assert not hasattr(owner, "_draft_graph_output_references")
    assert len(manager.graphs) == 3
