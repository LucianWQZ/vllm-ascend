# SPDX-License-Identifier: Apache-2.0
"""Same-Window continuous reference on an EXCLUSIVE disposable MRV2 fixture.

This is deliberately serialized, not normal-overlap evidence. It restores all
Draft query read/write blocks before an eager backbone reference, then restores
the actual Graph post-state before returning. Target state is never restored or
swapped. No sampling route is changed. The original manager still returns None.
"""

from __future__ import annotations

import copy
from types import MethodType

import torch

from tools.p7x_graph_repair6_storage import atomic_json, require

ATOL = 0.0
RTOL = 0.0


def independent_window(full, exact, valid, active, padded, width):
    require(full.ndim == 2 and len(exact) == len(valid) == active and active <= padded, "invalid canonical geometry")
    table = torch.zeros((padded, width), dtype=full.dtype, device=full.device)
    lengths, rows = [], []
    for r, end in enumerate(exact):
        start = max(end - 1024, 0) // 128
        stop = (end + 127) // 128
        require(0 <= start <= stop <= valid[r] <= full.shape[1] and stop - start <= width, "canonical shortage")
        table[r, : stop - start] = full[r, start:stop]
        lengths.append(end - start * 128)
        rows.append(dict(S=start * 128, E=end, V=lengths[-1], columns=[start, stop]))
    return table, lengths + [0] * (padded - active), rows


class ContinuousReference:
    def __init__(self, runner, scheduler, output, identity):
        require(
            not scheduler.requests and not scheduler.running and not scheduler.waiting,
            "exclusive empty engine required",
        )
        self.runner, self.owner = runner, runner.speculator
        self.manager = self.owner.query_cudagraph_manager
        self.sink = getattr(self.owner, "_draft_graph_output_references", None)
        require(self.sink is not None and self.sink.frozen, "capture output sink not installed before capture")
        self.output, self.identity, self.records = output, identity, []
        self.original = self.manager.run_fullgraph
        self.saved = ("run_fullgraph" in vars(self.manager), vars(self.manager).get("run_fullgraph"))
        self.poisoned = False
        self.current_case = None

    def __enter__(self):
        from tools.p7x_graph_repair6_storage import CollectionWriter

        self.writer = CollectionWriter(
            self.output,
            dict(
                identity=self.identity,
                scope="DIAGNOSTIC_SERIALIZED_BACKBONE_REFERENCE",
                normal_overlap="UNPROVEN",
                physical_read="OPEN",
                tolerance=dict(atol=ATOL, rtol=RTOL),
            ),
            max_bytes=4 << 30,
        )
        probe = self

        def run(_manager, desc):
            require(not probe.poisoned, "diagnostic failure quarantined")
            try:
                return probe._run(desc)
            except BaseException:
                probe.poisoned = True
                raise

        self.manager.run_fullgraph = MethodType(run, self.manager)
        return self

    def _run(self, desc):
        from vllm.config.compilation import CUDAGraphMode
        from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer

        s = self.owner
        require(
            s.sample_from_anchor and s.num_query_per_req == 7 and desc.cg_mode == CUDAGraphMode.FULL,
            "only frozen DSpark anchor FULL query7",
        )
        nr = s.input_batch.num_reqs
        require(
            self.current_case is not None and self.current_case["active"] == nr, "numeric fixture case not associated"
        )
        # Complete query pre-state, after eager context-KV production, before
        # Graph submit. This D2H is diagnostic and never between replay/UPDATE.
        self.manager.update_stream.synchronize()
        torch.npu.current_stream().synchronize()
        full = s.block_tables.input_block_tables[0][:nr].detach().cpu().clone()
        exact = s.input_buffers.seq_lens[:nr].detach().cpu().tolist()
        valid = s.block_tables.num_blocks.cpu[0, :nr].tolist()
        context = self.runner.compilation_config.static_forward_context
        caches = {
            f"{name}/{i}": kv
            for name in sorted(s.draft_attn_layer_names)
            for i, kv in enumerate(context[name].kv_cache)
        }
        slots = s.block_tables.slot_mappings[0, : desc.num_tokens].detach().cpu().clone()
        selected = sorted(
            set(full[r, : valid[r]].tolist()[c] for r in range(nr) for c in range(valid[r]))
            | {int(v) // 128 for v in slots.tolist() if v >= 0}
            | {0}
        )
        size = sum(len(selected) * t[0].numel() * t.element_size() for t in caches.values())
        require(size * 2 <= 1 << 30, "BLOCKED_RESOURCE: numeric pre/post device snapshots exceed 1 GiB")
        indices = torch.tensor(selected, dtype=torch.int64, device=s.device)
        before = {k: t.index_select(0, indices) for k, t in caches.items()}
        inputs = {k: getattr(s.input_buffers, k)[: desc.num_tokens].clone() for k in ("input_ids", "positions")}
        returned = self.original(desc)
        require(returned is None, "Graph manager return contract changed")
        self.manager.update_stream.synchronize()
        torch.npu.current_stream().synchronize()
        live = s._draft_window_live_inputs
        require(live is not None, "generation retired before numeric observation")
        graph_hidden = self.sink.get(s, desc.num_tokens).detach().cpu().clone()
        require(graph_hidden.dtype in (torch.float16, torch.bfloat16), "continuous reference dtype not declared")
        after = {k: t.index_select(0, indices) for k, t in caches.items()}
        # consume_metadata is a one-shot BOUND -> SUBMITTED transition. The
        # reference must inspect that exact submitted carrier, not consume twice.
        carriers = s._draft_window_graph_characterization.provider.submitted_carriers
        require(
            len(carriers) == 1
            and s._draft_window_graph_characterization.provider.require_runtime_identity(live.identity) is carriers[0],
            "missing exact submitted generation for reference",
        )
        actual = carriers[0].metadata
        width = next(iter(actual.values())).block_tables.shape[1]
        table, lengths, rows = independent_window(full, exact, valid, nr, desc.num_reqs, width)
        metadata = {}
        for name, m in actual.items():
            require(
                torch.equal(m.block_tables.detach().cpu(), table) and m.seq_lens_list == lengths,
                "production Window differs from independent canonical oracle",
            )
            ref = copy.copy(m)
            ref.block_tables = table.to(s.device)
            ref.seq_lens = torch.tensor(lengths, dtype=m.seq_lens.dtype, device=s.device)
            ref.seq_lens_list = list(lengths)
            metadata[name] = ref
        for key, t in caches.items():
            t.index_copy_(0, indices, before[key])
        # Model backbone only; does not alter greedy/Markov sampling or tokens.
        try:
            reference = s._run_model(
                desc.num_tokens,
                metadata,
                build_slot_mappings_by_layer(s.block_tables.slot_mappings[:, : desc.num_tokens], s.kv_cache_config),
                None,
                CUDAGraphMode.NONE,
            )
            torch.npu.current_stream().synchronize()
            reference = reference.detach().cpu().clone()
        finally:
            for key, t in caches.items():
                t.index_copy_(0, indices, after[key])
            torch.npu.current_stream().synchronize()
        require(
            all(torch.equal(getattr(s.input_buffers, k)[: desc.num_tokens], v) for k, v in inputs.items()),
            "reference mutated input/positions",
        )
        from tools.p7x_graph_repair6 import metrics

        a, b = graph_hidden[: nr * 7], reference[: nr * 7]
        measured = metrics(a, b)
        measured.update(atol=ATOL, rtol=RTOL)
        record = dict(
            generation=live.identity.generation_id,
            epoch=live.identity.mapping_epoch,
            case_identity=dict(self.current_case),
            descriptor=[desc.num_reqs, desc.num_tokens],
            active=nr,
            canonical_table=full.tolist(),
            exact=exact,
            valid=valid,
            oracle=rows,
            selected_blocks=selected,
            pre_kv=before,
            graph_post_kv=after,
            inputs=inputs,
            graph=graph_hidden,
            reference=reference,
            metrics=measured,
            status="PASS_SCOPED_CONTINUOUS"
            if measured["finite"] and torch.allclose(a, b, atol=ATOL, rtol=RTOL)
            else "NUMERIC_MISMATCH",
        )
        record["identity"] = dict(self.identity, generation=record["generation"], epoch=record["epoch"])
        self.writer.start(record["identity"])
        self.writer.finish(record)
        self.records.append({k: record[k] for k in ("generation", "epoch", "status", "metrics", "case_identity")})
        return returned

    def __exit__(self, kind, *_):
        if self.saved[0]:
            self.manager.run_fullgraph = self.saved[1]
        else:
            delattr(self.manager, "run_fullgraph")
        self.writer.close("EXECUTION_FAILED" if self.poisoned or kind else "COMPLETE")
        atomic_json(
            self.output.with_name("numeric-summary.json"),
            dict(
                samples=len(self.records),
                statuses=[r["status"] for r in self.records],
                normal_overlap="UNPROVEN",
                device_execution="EXECUTED",
            ),
        )
