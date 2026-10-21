# SPDX-License-Identifier: Apache-2.0
"""Observe normal MRV2 order with bounded device copies, no per-generation wait.

The extra copies and end-of-campaign events are diagnostic. They do not prove
unobserved normal overlap or task-update effective generation. A failed child
is quarantined: no attempted cleanup wait on an incompletely submitted graph.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import asdict
from types import MethodType

import torch

from tools.p7x_graph_repair6_storage import atomic_json, require
from tools.p7x_graph_repair6_submission import SubmissionState


class NormalOrderProbe:
    def __init__(self, spec, output, identity, max_generations=128):
        self.spec, self.manager, self.output, self.identity = spec, spec.query_cudagraph_manager, output, identity
        self.max_generations = max_generations
        self.records, self.pending, self.saved = [], [], []
        self.submission = SubmissionState()
        self.poisoned = False

    def _wrap(self, owner, name, fn):
        self.saved.append((owner, name, name in vars(owner), vars(owner).get(name)))
        setattr(owner, name, MethodType(fn, owner))

    def __enter__(self):
        from vllm_ascend.worker.v2.spec_decode.draft_window_graph_characterization import _json_safe

        probe, original = self, self.manager.run_fullgraph

        def run(_manager, desc):
            require(len(probe.records) < probe.max_generations, "normal observation capacity reached before submission")
            start = time.monotonic_ns()
            probe.submission.submit("NORMAL_GRAPH_AND_OBSERVATION_COPY_PENDING")
            try:
                result = original(desc)
            except BaseException as error:
                probe.submission.failed(error)
                probe.poisoned = True
                raise
            live = probe.spec._draft_window_live_inputs
            require(live is not None, "live generation absent at normal-order boundary")
            nr = probe.spec.input_batch.num_reqs
            session = probe.spec._draft_window_graph_characterization
            carriers = session.provider.submitted_carriers
            require(len(carriers) == 1, "normal observation needs unique carrier")
            # Enqueued AFTER replay+UPDATE submission, before next generation.
            # Only current-stream device copy; no D2H/query/synchronize here.
            probe.pending.append(probe.spec.draft_tokens[:nr].clone())
            probe.records.append(
                dict(
                    generation=live.identity.generation_id,
                    epoch=live.identity.mapping_epoch,
                    descriptor=[desc.num_reqs, desc.num_tokens],
                    active=nr,
                    start_ns=start,
                    manager_return_ns=time.monotonic_ns(),
                    current_stream=int(torch.npu.current_stream().npu_stream),
                    update_stream=int(probe.manager.update_stream.npu_stream),
                    provider=id(session.provider),
                    backing=asdict(carriers[0].capture_identity.backing_identity),
                    window_descriptors=[asdict(d) for d in live.identity.read_descriptor_fingerprint],
                    runtime_binding=_json_safe(carriers[0].runtime_identity),
                    retired=False,
                    completion="UNPROVEN",
                )
            )
            return result

        scope_original = self.spec._draft_window_proposal_scope

        @contextmanager
        def scope(_owner, *a, **kw):
            before = len(probe.records)
            with scope_original(*a, **kw):
                yield
            if len(probe.records) > before:
                probe.records[-1].update(
                    retired=True,
                    retirement_ns=time.monotonic_ns(),
                    live_inputs_cleared=probe.spec._draft_window_live_inputs is None,
                )

        self._wrap(self.manager, "run_fullgraph", run)
        self._wrap(self.spec, "_draft_window_proposal_scope", scope)
        return self

    def __exit__(self, kind, *_):
        self.poisoned = self.poisoned or (kind is not None and self.submission.pending)
        for owner, name, existed, old in reversed(self.saved):
            if existed:
                setattr(owner, name, old)
            else:
                delattr(owner, name)
        complete = None
        if kind is None and not self.poisoned:
            # One final drain, not a hidden per-generation synchronization.
            try:
                update, consumer = torch.npu.Event(), torch.npu.Event()
                update.record(self.manager.update_stream)
                consumer.record(torch.npu.current_stream())
                update.synchronize()
                consumer.synchronize()
                self.submission.complete("FINAL_DIAGNOSTIC_DRAIN_COMPLETE")
            except BaseException as error:
                self.submission.failed(error)
                self.poisoned = True
                atomic_json(
                    self.output,
                    dict(
                        identity=self.identity,
                        status="QUARANTINED",
                        records=self.records,
                        submission=self.submission.snapshot(),
                        completion=None,
                    ),
                )
                raise
            complete = dict(
                source="ordinary_Event_final_campaign_drain",
                update_event=id(update),
                consumer_event=id(consumer),
                clock="Host identity, not device timestamp",
            )
            for record, tokens in zip(self.records, self.pending):
                record.update(tokens=tokens.cpu().tolist(), completion="FINAL_OBSERVATION_COPY_DRAINED")
        atomic_json(
            self.output,
            dict(
                schema="repair6-normal-order-v1",
                identity=self.identity,
                records=self.records,
                status="QUARANTINED" if kind else "OBSERVED_NORMAL_ORDER_WITH_DEVICE_COPIES",
                completion=complete,
                submission=self.submission.snapshot(),
                update_effective_generation="UNPROVEN",
                normal_unobserved_reuse="UNPROVEN",
                impact="extra device copy per generation; one final event drain; no per-generation waits",
            ),
        )
