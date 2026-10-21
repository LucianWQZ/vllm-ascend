# SPDX-License-Identifier: Apache-2.0
"""Explicit diagnostic instrumentation on the REAL MRV2 speculator instance.

No capture callbacks, globals, ExternalEvent.query, or production scheduling
changes. A normal diagnostic records ordinary events AFTER replay and UPDATE,
then waits both before sampling draft_tokens and allowing the next proposal.
This intentionally serializes diagnostics and cannot certify uninstrumented
overlap. A partially submitted abort quarantines the entire disposable process.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import MethodType
from typing import Any

import torch


class MRV2LifecycleProbe:
    def __init__(self, speculator: Any, output: Path, identity: dict, *, abort_at: str = "none", campaign=None):
        if abort_at not in ("none", "before_replay", "after_replay", "after_update", "after_completion"):
            raise ValueError("unknown abort point")
        self.owner, self.manager = speculator, speculator.query_cudagraph_manager
        self.output, self.identity, self.abort_at = output, dict(identity), abort_at
        self.records: list[dict] = []
        self.saved: list[tuple] = []
        self.current: dict | None = None
        self.consumer_event = None
        self.poisoned = False
        self.campaign = campaign

    def _flush(self):
        temporary = self.output.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "schema": "repair5-real-mrv2-lifecycle-v1",
                    "identity": self.identity,
                    "records": self.records,
                    "scope": "actual MRV2 proposal/Graph/update/retirement with diagnostic ordinary events",
                    "impact": "serialized device completion, NOT normal overlap/performance",
                    "normal_unsynchronized_reuse": "UNPROVEN",
                    "abort_at": self.abort_at,
                    "quarantined": self.poisoned,
                    "clock_domain": "host_monotonic_ns_NOT_device",
                    "draft_layer_names": sorted(self.owner.draft_attn_layer_names),
                    "event_producer": "MRV2LifecycleProbe:ordinary_Event_after_replay_and_update:v2",
                    "recorder": self.owner._draft_window_graph_characterization.recorder.snapshot(),
                },
                indent=2,
            )
            + "\n"
        )
        temporary.replace(self.output)
        if self.poisoned and self.campaign is not None:
            partial = dict(
                self.campaign,
                status="ABORT_PARTIAL",
                snapshot=self.owner._draft_window_graph_characterization.recorder.snapshot(),
            )
            self.output.with_name("abort-campaign.json").write_text(json.dumps(partial, indent=2) + "\n")

    def _abort(self, point: str):
        if self.abort_at != point:
            return
        self.poisoned = True
        if self.current is not None:
            self.current.update(abort=point, reuse="FORBIDDEN_PROCESS_QUARANTINE")
        self._flush()
        # No destructor/shutdown waits on a graph whose event may not be updated.
        os._exit(73)

    def _install(self, owner, name, fn):
        previous = (owner, name, name in vars(owner), vars(owner).get(name))
        setattr(owner, name, MethodType(fn, owner))
        self.saved.append(previous)

    def __enter__(self):
        try:
            return self._enter()
        except BaseException:
            self._restore()
            raise

    def _enter(self):
        session = getattr(self.owner, "_draft_window_graph_characterization", None)
        if session is None or not session.recorder.enabled:
            raise ValueError("real lifecycle requires experimental Window provider and observer")
        probe = self
        for desc, graph in self.manager.graphs.items():
            original = graph.replay

            def replay(_graph, original=original, desc=desc):
                if probe.poisoned:
                    raise RuntimeError("quarantined graph cannot replay")
                carriers = session.provider.submitted_carriers
                if len(carriers) != 1 or carriers[0].runtime_identity is None:
                    raise RuntimeError("no unique bound runtime generation before replay")
                carrier = carriers[0]
                identity = carrier.runtime_identity
                record = {
                    "generation": identity.generation_id,
                    "epoch": identity.mapping_epoch,
                    "provider": id(session.provider),
                    "descriptor": [desc.num_reqs, desc.num_tokens],
                    "backing": asdict(carrier.capture_identity.backing_identity),
                    "before_replay_ns": time.monotonic_ns(),
                    "completion": "UNPROVEN",
                    "retired": False,
                }
                # Keep JSON native. Actual addresses and source lineage are also
                # linked from the characterization G4 record by generation.
                probe.current = record
                probe.records.append(record)
                probe._abort("before_replay")
                original()
                probe.consumer_event = torch.npu.Event()
                stream = torch.npu.current_stream()
                probe.consumer_event.record(stream)
                record.update(
                    replay_return_ns=time.monotonic_ns(),
                    consumer_stream=int(stream.npu_stream),
                    consumer_event=id(probe.consumer_event),
                )
                probe._abort("after_replay")

            self._install(graph, "replay", replay)

        original_run = self.manager.run_fullgraph

        def run(_manager, desc):
            if probe.poisoned or (probe.current and probe.current["completion"] != "DIAGNOSTIC_COMPLETE"):
                raise RuntimeError("unsafe diagnostic backing reuse rejected")
            result = original_run(desc)  # prepare/bind/replay/consumer/UPDATE order unchanged
            if probe.current is None or probe.consumer_event is None:
                raise RuntimeError("manager did not execute instrumented graph")
            record = probe.current
            record["update_return_ns"] = time.monotonic_ns()
            probe._abort("after_update")
            update_event = torch.npu.Event()
            update_event.record(probe.manager.update_stream)
            update_event.synchronize()
            probe.consumer_event.synchronize()
            if update_event.query() is not True or probe.consumer_event.query() is not True:
                raise RuntimeError("ordinary completion event not complete")
            record.update(
                completion="DIAGNOSTIC_COMPLETE",
                complete_ns=time.monotonic_ns(),
                update_event=id(update_event),
                update_stream=int(probe.manager.update_stream.npu_stream),
                consumer_event_kind="torch.npu.Event",
                update_event_kind="torch.npu.Event",
                consumer_observation="record_after_replay_then_synchronize_then_query",
                update_observation="record_on_update_stream_then_synchronize_then_query",
            )
            count = probe.owner.input_batch.num_reqs
            tokens = probe.owner.draft_tokens[:count]
            record.update(
                active_requests=count, token_shape=list(tokens.shape), draft_tokens=tokens.detach().cpu().tolist()
            )
            probe._abort("after_completion")
            probe._flush()
            return result

        original_scope = self.owner._draft_window_proposal_scope

        @contextmanager
        def scope(_owner, *args, **kwargs):
            if probe.poisoned:
                raise RuntimeError("quarantined diagnostic cannot acquire another proposal")
            previous = len(probe.records)
            try:
                with original_scope(*args, **kwargs):
                    yield
            except BaseException as error:
                from tools.p7x_graph_repair6_submission import CompletedDiagnosticContractError, NumericMismatch

                completed = (
                    len(probe.records) > previous and probe.records[-1].get("completion") == "DIAGNOSTIC_COMPLETE"
                )
                probe.poisoned = not (
                    completed and isinstance(error, (NumericMismatch, CompletedDiagnosticContractError))
                )
                probe._flush()
                raise
            if len(probe.records) > previous:
                record = probe.records[-1]
                record.update(
                    retired=True,
                    retirement_return_ns=time.monotonic_ns(),
                    live_inputs_cleared=probe.owner._draft_window_live_inputs is None,
                )
                probe._flush()

        self._install(self.manager, "run_fullgraph", run)
        self._install(self.owner, "_draft_window_proposal_scope", scope)
        return self

    def __exit__(self, *_exc):
        try:
            self._flush()
        finally:
            self._restore()

    def _restore(self):
        for owner, name, existed, old in reversed(self.saved):
            if existed:
                setattr(owner, name, old)
            else:
                delattr(owner, name)
        self.saved.clear()


def check_lifecycle(document: dict) -> dict:
    """Low-level structure ONLY; final acceptance requires the CLI external join."""
    result: dict[str, Any] = {
        "status": "INVALID_EVIDENCE",
        "normal_overlap": "UNPROVEN",
        "update_effective_generation": "UNPROVEN",
    }
    try:
        if document["schema"] != "repair5-real-mrv2-lifecycle-v1" or not document["records"]:
            raise ValueError("missing lifecycle records")
        if document["quarantined"] or document["abort_at"] != "none":
            result["status"] = "QUARANTINED_NOT_NORMAL_PASS"
            return result
        if not all(
            document["identity"].get(k) is not None
            for k in (
                "run_id",
                "case_id",
                "attempt_id",
                "source_sha",
                "source_tree",
                "process_id",
                "engine_id",
                "driver_mode",
            )
        ):
            raise ValueError("missing run/source/process identity")
        if document["event_producer"] != "MRV2LifecycleProbe:ordinary_Event_after_replay_and_update:v2":
            raise ValueError("unknown completion observation producer")
        seen = set()
        previous = None
        provider = None
        generation = epoch = -1
        backings: dict[tuple[int, ...], dict] = {}
        for record in document["records"]:
            key = (record["provider"], record["generation"], record["epoch"])
            if key in seen:
                raise ValueError("duplicate generation")
            seen.add(key)
            if provider is None:
                provider = record["provider"]
            if (
                record["provider"] != provider
                or type(record["generation"]) is not int
                or type(record["epoch"]) is not int
                or record["generation"] <= generation
                or record["epoch"] <= epoch
            ):
                raise ValueError("provider/generation/backing mismatch")
            generation, epoch = record["generation"], record["epoch"]
            desc = record["descriptor"]
            if desc not in ([1, 7], [2, 14], [4, 28]):
                raise ValueError("unsupported Draft descriptor")
            backing = record["backing"]
            if type(backing["owner_id"]) is not int or backing["owner_id"] <= 0:
                raise ValueError("missing backing owner")
            for name in (
                "window_block_table",
                "valid_num_blocks",
                "device_window_exact",
                "cpu_window_exact",
                "cpu_window_upper",
            ):
                value = backing[name]
                if (
                    type(value["data_ptr"]) is not int
                    or value["data_ptr"] <= 0
                    or type(value["storage_data_ptr"]) is not int
                    or value["storage_data_ptr"] <= 0
                    or not value["dtype"]
                    or not value["device"]
                    or not value["shape"]
                    or len(value["stride"]) != len(value["shape"])
                    or any(type(n) is not int or n <= 0 for n in value["shape"])
                    or type(value["storage_offset"]) is not int
                    or value["storage_offset"] < 0
                ):
                    raise ValueError("invalid tensor backing")
            key_desc = tuple(desc)
            if key_desc in backings and backings[key_desc] != backing:
                raise ValueError("descriptor backing changed")
            backings[key_desc] = backing
            if (
                backing["window_block_table"]["shape"][0] != desc[0]
                or backing["device_window_exact"]["shape"] != [desc[0]]
                or backing["cpu_window_exact"]["shape"] != [desc[0]]
                or backing["cpu_window_upper"]["shape"] != [desc[0]]
            ):
                raise ValueError("backing shape differs from descriptor")
            for kind in ("consumer", "update"):
                if (
                    type(record[f"{kind}_event"]) is not int
                    or record[f"{kind}_event"] <= 0
                    or type(record[f"{kind}_stream"]) is not int
                    or record[f"{kind}_stream"] < 0
                    or record[f"{kind}_event_kind"] != "torch.npu.Event"
                ):
                    raise ValueError("missing ordinary event/stream provenance")
            if (
                record["consumer_observation"] != "record_after_replay_then_synchronize_then_query"
                or record["update_observation"] != "record_on_update_stream_then_synchronize_then_query"
            ):
                raise ValueError("unknown completion observation")
            active = record["active_requests"]
            if (
                type(active) is not int
                or not 0 < active <= desc[0]
                or record["token_shape"] != [active, 7]
                or len(record["draft_tokens"]) != active
                or any(len(row) != 7 or any(type(t) is not int or t < 0 for t in row) for row in record["draft_tokens"])
            ):
                raise ValueError("active requests/token geometry mismatch")
            if (
                record["completion"] != "DIAGNOSTIC_COMPLETE"
                or not record["retired"]
                or not record["live_inputs_cleared"]
                or not record["draft_tokens"]
            ):
                raise ValueError("missing completion/result/retirement")
            times = [
                record[k]
                for k in (
                    "before_replay_ns",
                    "replay_return_ns",
                    "update_return_ns",
                    "complete_ns",
                    "retirement_return_ns",
                )
            ]
            if (
                any(type(t) is not int or t <= 0 for t in times)
                or times != sorted(times)
                or (previous is not None and times[0] <= previous)
            ):
                raise ValueError("invalid diagnostic order/reuse")
            previous = times[-1]
        result.update(status="STRUCTURE_VALID_NOT_EXECUTION_ACCEPTED", generations=len(seen))
    except (KeyError, TypeError, ValueError) as error:
        result["error"] = str(error)
    return result


def check_abort(document: dict, launcher: dict, expected: dict) -> dict:
    """Externally observed process quarantine, NOT successful device completion."""
    result = {"status": "INVALID_EVIDENCE", "device_completion": "UNPROVEN", "reuse": "FORBIDDEN"}
    try:
        identity = document["identity"]
        fields = ("run_id", "case_id", "attempt_id", "source_sha", "source_tree", "driver_mode")
        if any(identity[k] != launcher[k] or identity[k] != expected["acceptance_identity"][k] for k in fields):
            raise ValueError("abort identity mismatch")
        if launcher["argv"] != expected["argv"] or launcher["pid"] != identity["process_id"]:
            raise ValueError("abort launcher/child mismatch")
        if launcher["returncode"] != 73 or launcher["signal"] is not None or launcher["timeout"]:
            raise ValueError("not the controlled exit")
        point = expected["argv"][expected["argv"].index("--repair5-abort-at") + 1]
        record = document["records"][-1]
        if (
            not document["quarantined"]
            or document["abort_at"] != point
            or record["abort"] != point
            or record["reuse"] != "FORBIDDEN_PROCESS_QUARANTINE"
        ):
            raise ValueError("wrong quarantine/abort stage")
        required = {
            "before_replay": ["before_replay_ns"],
            "after_replay": ["before_replay_ns", "replay_return_ns"],
            "after_update": ["before_replay_ns", "replay_return_ns", "update_return_ns"],
            "after_completion": ["before_replay_ns", "replay_return_ns", "update_return_ns", "complete_ns"],
        }[point]
        if any(type(record[k]) is not int for k in required):
            raise ValueError("incomplete stage evidence")
        result.update(status="EXPECTED_QUARANTINE_OBSERVED", point=point)
    except (KeyError, TypeError, ValueError, IndexError) as error:
        result["error"] = str(error)
    return result
