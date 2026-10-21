# SPDX-License-Identifier: Apache-2.0
"""Same-Window continuous reference on an EXCLUSIVE disposable MRV2 fixture.

This is deliberately serialized, not normal-overlap evidence. It restores all
Draft query read/write blocks before an eager backbone reference, then restores
the actual Graph post-state before returning. Target state is never restored or
swapped. No sampling route is changed. The original manager still returns None.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
from types import MethodType

import torch

from tools.p7x_graph_repair6_storage import atomic_json, require
from tools.p7x_graph_repair6_submission import CompletedDiagnosticContractError, NumericMismatch, SubmissionState

ATOL = 0.0
RTOL = 0.0


def independent_window(full, exact, valid, active, padded, width):
    require(full.ndim == 2 and len(exact) == len(valid) == active and active <= padded, "invalid canonical geometry")
    table = torch.zeros((padded, width), dtype=full.dtype, device=full.device)
    lengths, rows = [], []
    for r, end in enumerate(exact):
        require(
            type(end) is int and end >= 7 and type(valid[r]) is int and valid[r] > 0, "invalid canonical exact/valid"
        )
        start = max(end - 1024, 0) // 128
        stop = (end + 127) // 128
        require(0 <= start <= stop <= valid[r] <= full.shape[1] and stop - start <= width, "canonical shortage")
        table[r, : stop - start] = full[r, start:stop]
        lengths.append(end - start * 128)
        rows.append(dict(S=start * 128, E=end, V=lengths[-1], columns=[start, stop]))
    return table, lengths + [0] * (padded - active), rows


class ContinuousReference:
    def __init__(self, runner, scheduler, output, identity, *, provenance=None):
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
        self.failed = False
        self.submission = SubmissionState()
        self.provenance = provenance or {}
        self.current_case = None

    def __enter__(self):
        from tools.p7x_graph_repair5_target import case_plan
        from tools.p7x_graph_repair6_storage import CHUNK, CollectionWriter, require_space

        context = self.runner.compilation_config.static_forward_context
        self.provenance = dict(
            self.provenance,
            draft_layers=sorted(self.owner.draft_attn_layer_names),
            kv_inventory={
                f"{n}/{i}": dict(shape=list(t.shape), dtype=str(t.dtype))
                for n in sorted(self.owner.draft_attn_layer_names)
                for i, t in enumerate(context[n].kv_cache)
            },
            hidden_contract=dict(
                width=self.sink.get(self.owner, 7).shape[1], dtype=str(self.sink.get(self.owner, 7).dtype)
            ),
        )
        # Independent branch budget, computed from the real Draft cache geometry.
        # Four snapshots plus inputs/output; no assumed compression or dedup.
        cases = case_plan(self.provenance.get("profile", "target4"))
        limit = self.provenance.get("sample_limit")
        if limit is not None:
            require(limit == 3, "numeric sentinel must cover A/B/A_prime")
            cases = cases[:limit]
        bytes_per_block = sum(
            t[0].numel() * t.element_size()
            for n in sorted(self.owner.draft_attn_layer_names)
            for t in context[n].kv_cache
        )
        samples = [
            dict(
                scenario=c["scenario"],
                bytes=4 * bytes_per_block * (1 + sum((e + 7 + 127) // 128 for e in c["ends"])) + CHUNK,
            )
            for c in cases
        ]
        required = sum(c["bytes"] for c in samples) + CHUNK
        self.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(
            self.output.with_name("numeric-resource-plan.json"),
            dict(
                samples=samples,
                required_free_bytes=required,
                scope="NUMERIC_ADDITIONAL_TO_TARGET_ARM",
                device_snapshot_limit=1 << 30,
                compression_assumed=False,
            ),
        )
        require_space(self.output.parent, required)
        self.writer = CollectionWriter(
            self.output,
            dict(
                numeric_schema="repair6-continuous-v2",
                identity=self.identity,
                provenance=self.provenance,
                scope="DIAGNOSTIC_SERIALIZED_BACKBONE_REFERENCE",
                normal_overlap="UNPROVEN",
                physical_read="OPEN",
                tolerance=dict(atol=ATOL, rtol=RTOL),
            ),
            max_bytes=required,
        )
        probe = self

        def run(_manager, desc):
            require(not probe.failed, "diagnostic branch already failed; no next generation")
            try:
                return probe._run(desc)
            except BaseException as error:
                probe.failed = True
                probe.submission.failed(error)
                probe.poisoned = probe.submission.pending
                try:
                    atomic_json(
                        probe.output.with_name("numeric-failure.json"),
                        dict(identity=probe.identity, submission=probe.submission.snapshot()),
                    )
                except Exception as export_error:
                    error.add_note(f"Host failure export also failed: {export_error!r}")
                if not probe.poisoned and isinstance(error, ValueError):
                    raise CompletedDiagnosticContractError(str(error)) from error
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
        self.submission.submit("PRESTATE_DRAIN_PENDING")
        self.manager.update_stream.synchronize()
        torch.npu.current_stream().synchronize()
        self.submission.complete("PRESTATE_DRAIN_COMPLETE")
        self.submission.submit("PRESTATE_SNAPSHOT_PENDING")
        full = s.block_tables.input_block_tables[0][: desc.num_reqs].detach().cpu().clone()
        exact = s.input_buffers.seq_lens[:nr].detach().cpu().tolist()
        mapping = torch.as_tensor(s.input_batch.idx_mapping_np[:nr], dtype=torch.int64)
        valid = s.block_tables.num_blocks.cpu[0].index_select(0, mapping).tolist()
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
        inputs["slot_mapping"] = slots
        inputs["query_start_loc"] = s.input_buffers.query_start_loc[: desc.num_reqs + 1].detach().cpu().clone()
        self.submission.submit("GRAPH_REPLAY_UPDATE_PENDING")
        returned = self.original(desc)
        require(returned is None, "Graph manager return contract changed")
        self.manager.update_stream.synchronize()
        torch.npu.current_stream().synchronize()
        self.submission.complete("GRAPH_DIAGNOSTIC_COMPLETE")
        live = s._draft_window_live_inputs
        require(live is not None, "generation retired before numeric observation")
        graph_hidden = self.sink.get(s, desc.num_tokens).detach().cpu().clone()
        require(graph_hidden.dtype in (torch.float16, torch.bfloat16), "continuous reference dtype not declared")
        self.submission.submit("POSTGRAPH_OBSERVATION_PENDING")
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

        def metadata_snapshot(values):
            return {
                name: dict(
                    table=m.block_tables.detach().cpu().clone(),
                    table_ptr=m.block_tables.data_ptr(),
                    exact_ptr=m.seq_lens.data_ptr(),
                    exact=m.seq_lens.detach().cpu().clone(),
                    upper=list(m.seq_lens_list),
                    query=list(m.actual_seq_lengths_q),
                    slot=m.slot_mapping.detach().cpu().clone(),
                    causal=m.causal,
                )
                for name, m in values.items()
            }

        actual_snapshot = metadata_snapshot(actual)
        reference_snapshot = metadata_snapshot(metadata)
        self.submission.submit("REFERENCE_RESTORE_AND_FORWARD_PENDING")
        for key, t in caches.items():
            t.index_copy_(0, indices, before[key])
        reference_before = {k: t.index_select(0, indices) for k, t in caches.items()}
        reference_inputs = {
            k: getattr(s.input_buffers, k)[: desc.num_tokens].clone() for k in ("input_ids", "positions")
        }
        reference_inputs["slot_mapping"] = s.block_tables.slot_mappings[0, : desc.num_tokens].clone()
        reference_inputs["query_start_loc"] = s.input_buffers.query_start_loc[: desc.num_reqs + 1].clone()
        # Model backbone only; does not alter greedy/Markov sampling or tokens.
        reference = s._run_model(
            desc.num_tokens,
            metadata,
            build_slot_mappings_by_layer(s.block_tables.slot_mappings[:, : desc.num_tokens], s.kv_cache_config),
            None,
            CUDAGraphMode.NONE,
        )
        torch.npu.current_stream().synchronize()
        self.submission.complete("REFERENCE_DIAGNOSTIC_COMPLETE")
        reference = reference.detach().cpu().clone()
        reference_after = {k: t.index_select(0, indices) for k, t in caches.items()}
        # No unconditional finally restore: a partially submitted native failure
        # must leave the disposable child quarantined, without further device work.
        self.submission.submit("GRAPH_POSTSTATE_RESTORE_PENDING")
        for key, t in caches.items():
            t.index_copy_(0, indices, after[key])
        torch.npu.current_stream().synchronize()
        self.submission.complete("GRAPH_POSTSTATE_RESTORE_COMPLETE")
        self.submission.submit("POSTSTATE_VALIDATION_AND_EXPORT_PENDING")
        require(
            all(
                torch.equal(getattr(s.input_buffers, k)[: desc.num_tokens], inputs[k])
                for k in ("input_ids", "positions")
            ),
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
            reference_post_kv=reference_after,
            reference_pre_kv=reference_before,
            reference_inputs=reference_inputs,
            idx_mapping=mapping.tolist(),
            actual_metadata=actual_snapshot,
            reference_metadata=reference_snapshot,
            inventory={k: dict(shape=list(t.shape), dtype=str(t.dtype)) for k, t in caches.items()},
            hidden_contract=dict(width=self.sink.get(s, desc.num_tokens).shape[1], dtype=str(graph_hidden.dtype)),
            target_descriptor=[self.current_case["padded"], self.current_case["padded"] * 8],
            provider=id(s._draft_window_graph_characterization.provider),
            backing=asdict(carriers[0].capture_identity.backing_identity),
            inputs=inputs,
            graph=graph_hidden,
            reference=reference,
            metrics=measured,
            status="PASS_SCOPED_CONTINUOUS"
            if measured["finite"]
            and torch.allclose(a, b, atol=ATOL, rtol=RTOL)
            and all(torch.equal(after[k], reference_after[k]) for k in caches)
            else "NUMERIC_MISMATCH",
        )
        record["identity"] = dict(self.identity, generation=record["generation"], epoch=record["epoch"])
        from vllm_ascend.worker.v2.spec_decode.draft_window_graph_characterization import _json_safe

        record["runtime_identity"] = _json_safe(live.identity)
        self.writer.start(record["identity"])
        self.writer.finish(record)
        # Writer's blocking CPU copies have completed; drain the diagnostic
        # current stream before classifying a mismatch as safe Host-only exit.
        torch.npu.current_stream().synchronize()
        self.submission.complete("EVIDENCE_EXPORT_COMPLETE")
        self.records.append({k: record[k] for k in ("generation", "epoch", "status", "metrics", "case_identity")})
        if record["status"] == "NUMERIC_MISMATCH":
            raise NumericMismatch("NUMERIC_MISMATCH: durable completed sample; stop dependent branch")
        return returned

    def __exit__(self, kind, *_):
        if self.saved[0]:
            self.manager.run_fullgraph = self.saved[1]
        else:
            delattr(self.manager, "run_fullgraph")
        self.writer.close("EXECUTION_FAILED" if self.failed or kind else "COMPLETE")
        atomic_json(
            self.output.with_name("numeric-summary.json"),
            dict(
                samples=len(self.records),
                identity=self.identity,
                numeric_schema="repair6-continuous-v2",
                submission=self.submission.snapshot(),
                execution=self.writer.manifest["execution"],
                statuses=[r["status"] for r in self.records],
                normal_overlap="UNPROVEN",
                device_execution="EXECUTED",
            ),
        )
