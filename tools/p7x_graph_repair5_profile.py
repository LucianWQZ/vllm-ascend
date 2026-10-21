# SPDX-License-Identifier: Apache-2.0
"""Opt-in request-stage profiler. No is_profile dispatch flag; no device polling.

Only instance-owned host wrappers are used. UPDATE attribution comes from the
native trace: the enclosing manager range is explicitly REPLAY_PLUS_UPDATE.
No claim of task operands, physical reads, or completion is made.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import MethodType

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != root / "tools"]
    sys.path.insert(0, str(root))


def preflight(torch_npu):
    p = getattr(torch_npu, "profiler", None)
    result = {"status": "CAPABILITY_UNVERIFIED", "apis": {}, "version": str(torch_npu.__version__)}
    for name in ("profile", "tensorboard_trace_handler"):
        fn = getattr(p, name, None)
        if not callable(fn):
            result.update(status="BLOCKED", error=f"torch_npu.profiler.{name} missing")
            return result
        try:
            result["apis"][name] = {"signature": str(inspect.signature(fn)), "source": inspect.getsource(fn)}
        except (TypeError, OSError, ValueError):
            result["apis"][name] = {"signature": None, "source": None, "reason": "NATIVE_OR_UNAVAILABLE"}
    activity = getattr(p, "ProfilerActivity", None)
    if not all(hasattr(activity, n) for n in ("CPU", "NPU")):
        result.update(status="BLOCKED", error="CPU/NPU activities missing")
        return result
    result["status"] = "APIS_AVAILABLE_DEVICE_NOT_RUN"
    return result


class RequestProfiler:
    def __init__(self, runner, output, identity, *, enabled, diagnostic=False, detailed=False):
        import torch

        self.torch, self.runner, self.output = torch, runner, Path(output)
        self.identity, self.enabled, self.diagnostic = identity, enabled, diagnostic
        if detailed and (not enabled or diagnostic):
            raise ValueError("detailed profile requires a separate enabled normal-path experiment")
        self.detailed = detailed
        self.saved, self.calls, self.groups = [], [], []
        self.profiler = None
        self.active = False
        self.current_group = None
        self.installations = {}
        self.deadline = time.monotonic() + 900
        self.device = str(runner.device) if hasattr(runner, "device") else "UNAVAILABLE"

    def _wrap(self, owner, name, label, descriptor=None):
        original = getattr(owner, name)
        self.saved.append((owner, name, name in vars(owner), vars(owner).get(name)))
        probe = self
        self.installations[label] = dict(installed=True, method=name, owner=type(owner).__name__)

        def wrapped(_self, *args, **kwargs):
            start = time.monotonic_ns()
            live_before = getattr(probe.runner.speculator, "_draft_window_live_inputs", None)
            with probe.torch.profiler.record_function("repair5/" + label):
                result = original(*args, **kwargs)
            # CPU-only record AFTER the wrapped operation. No device tensor read.
            probe.calls.append(
                dict(
                    label=label,
                    start_ns=start,
                    end_ns=time.monotonic_ns(),
                    descriptor=descriptor,
                    active=probe.active,
                    request_group=probe.current_group,
                    clock_domain="HOST_MONOTONIC_NS",
                    boundary="HOST_CALL_NOT_DEVICE_COMPLETION",
                )
            )
            identity = getattr(result, "identity", None) or getattr(live_before, "identity", None)
            if identity is None and label == "Draft_retire_host" and args:
                identity = args[0]
            if identity is not None and hasattr(identity, "generation_id"):
                probe.calls[-1].update(generation=identity.generation_id, mapping_epoch=identity.mapping_epoch)
            if args and hasattr(args[0], "num_tokens"):
                probe.calls[-1]["descriptor"] = [args[0].num_reqs, args[0].num_tokens]
            if label == "Draft_REPLAY_PLUS_UPDATE":
                live = getattr(probe.runner.speculator, "_draft_window_live_inputs", None)
                if live is not None:
                    # Immutable host descriptors; no tensor read, device query or
                    # observer enabled. This establishes actual planned S>0.
                    probe.calls[-1]["window_descriptors"] = [
                        asdict(d) for d in live.identity.read_descriptor_fingerprint
                    ]
                    probe.calls[-1]["generation"] = live.identity.generation_id
            return result

        setattr(owner, name, MethodType(wrapped, owner))

    def __enter__(self):
        spec = self.runner.speculator
        if not self.diagnostic and getattr(spec, "_draft_window_graph_characterization", None) is not None:
            if spec._draft_window_graph_characterization.recorder.enabled:
                raise ValueError("normal profiling requires heavy observer OFF")
        try:
            self.window_active = getattr(spec, "_draft_window_graph_characterization", None) is not None
            for label in ("Draft_materialize", "Draft_metadata_bind", "Draft_UPDATE_CALL", "Draft_retire_host"):
                self.installations[label] = dict(installed=False, reason="METHOD_OR_RUNTIME_NOT_PRESENT")
            self._wrap(spec, "propose", "Draft_propose")
            for method, label in (
                ("_materialize_draft_window_read_inputs", "Draft_materialize"),
                ("build_draft_attn_metadatas", "Draft_metadata_bind"),
            ):
                if hasattr(spec, method):
                    self._wrap(spec, method, label)
            if hasattr(spec.query_cudagraph_manager, "_update_full_graph_params"):
                self._wrap(spec.query_cudagraph_manager, "_update_full_graph_params", "Draft_UPDATE_CALL")
            runtime = getattr(spec, "_draft_window_runtime_state", None)
            if runtime is not None:
                self._wrap(runtime, "retire_generation", "Draft_retire_host")
            for name, manager in (("Target", self.runner.cudagraph_manager), ("Draft", spec.query_cudagraph_manager)):
                self._wrap(manager, "run_fullgraph", name + "_REPLAY_PLUS_UPDATE")
                for desc, graph in manager.graphs.items():
                    self._wrap(graph, "replay", name + "_REPLAY_SUBMIT", [desc.num_reqs, desc.num_tokens])
            return self
        except BaseException:
            self._restore()
            raise

    def begin_active(self):
        if self.enabled:
            import torch_npu

            self.capability = preflight(torch_npu)
            if self.capability["status"] != "APIS_AVAILABLE_DEVICE_NOT_RUN":
                raise RuntimeError(self.capability)
            if self.detailed:
                parameters = inspect.signature(torch_npu.profiler.profile).parameters
                if not all(k in parameters for k in ("record_shapes", "with_stack")):
                    raise RuntimeError("CAPABILITY_UNVERIFIED: detailed profile keyword signature unavailable")
            self.profiler = torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                record_shapes=self.detailed,
                profile_memory=False,
                with_stack=self.detailed,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(self.output / "trace")),
            )
            self.profiler.__enter__()
        self.active = True

    @contextmanager
    def group(self, label):
        if time.monotonic() > self.deadline:
            raise TimeoutError("profiling request-stage budget exceeded")
        start = time.monotonic_ns()
        self.current_group = label
        with self.torch.profiler.record_function("repair5/request/" + label):
            yield
        self.groups.append(dict(label=label, active=self.active, start_ns=start, end_ns=time.monotonic_ns()))
        self.current_group = None
        if self.active and self.profiler is not None:
            self.profiler.step()
        if time.monotonic() > self.deadline:
            raise TimeoutError("profiling request-stage budget exceeded")

    def _restore(self):
        for owner, name, existed, value in reversed(self.saved):
            if existed:
                setattr(owner, name, value)
            else:
                delattr(owner, name)
        self.saved.clear()

    def __exit__(self, *exc):
        try:
            if self.profiler is not None:
                self.profiler.__exit__(*exc)
        finally:
            self._restore()
            active_replays = [c for c in self.calls if c["active"] and c["label"] == "Draft_REPLAY_SUBMIT"]
            payload = dict(
                schema="repair5-profile-v1",
                identity=self.identity,
                python_worker_pid=os.getpid(),
                device=self.device,
                wrapper_pid=None,
                profiler_enabled=self.enabled,
                profile_configuration=dict(record_shapes=self.detailed, with_stack=self.detailed, profile_memory=False),
                detailed_diagnostic=self.detailed,
                capability=getattr(self, "capability", None),
                groups=self.groups,
                calls=self.calls,
                host_subrange_coverage=self.subrange_coverage(),
                status="GRAPH_REPLAY_OBSERVED" if active_replays else "OPEN_NO_DRAFT_REPLAY",
                scope="DIAGNOSTIC_SERIALIZED" if self.diagnostic else "NORMAL_OBSERVER_OFF",
                clock_domain="HOST_MONOTONIC_NS_ONLY",
                initialization_capture="EXCLUDED",
                warmup="first request group once before begin_active",
                update_host_range="Draft_UPDATE_CALL when instance seam exists; submission not completion",
                physical_read="NOT_PROVEN",
                completion="NOT_PROVEN",
            )
            payload["host_statistics"] = host_statistics(self.calls)
            payload["raw_exports"] = [
                str(p.relative_to(self.output)) for p in sorted((self.output / "trace").rglob("*")) if p.is_file()
            ]
            (self.output / "profile.json").write_text(json.dumps(payload, indent=2) + "\n")

    def subrange_coverage(self):
        replays = [c for c in self.calls if c["active"] and c["label"] == "Draft_REPLAY_PLUS_UPDATE"]
        result = {}
        for label, installation in self.installations.items():
            if label not in ("Draft_materialize", "Draft_metadata_bind", "Draft_UPDATE_CALL", "Draft_retire_host"):
                continue
            calls = [c for c in self.calls if c["active"] and c["label"] == label]
            expected = (
                len(replays) if self.window_active or label in ("Draft_metadata_bind", "Draft_UPDATE_CALL") else 0
            )
            expected_gens = {c["generation"] for c in replays if "generation" in c}
            actual_gens = {c["generation"] for c in calls if "generation" in c}
            missing = sorted(expected_gens - actual_gens) if self.window_active else []
            result[label] = dict(
                installation,
                expected_minimum=expected,
                actual=len(calls),
                missing_generations=missing,
                status="MISSING"
                if expected and (not installation["installed"] or len(calls) < expected or missing)
                else "OBSERVED"
                if calls
                else "NOT_APPLICABLE_WINDOW_OFF"
                if not self.window_active
                else "NOT_EXECUTED",
                expected_rule="minimum one per observed Draft graph; extra context/prefill calls retained",
                timing_scope="nested host ranges; do not sum as device elapsed",
            )
        return result


def host_statistics(calls):
    groups = defaultdict(list)
    for c in calls:
        if c["active"]:
            groups[c["label"]].append((c["end_ns"] - c["start_ns"]) / 1e6)
    result = {}
    for name, values in groups.items():
        ordered = sorted(values)
        result[name] = dict(
            samples=len(values),
            sum_ms=sum(values),
            mean_ms=sum(values) / len(values),
            p50_ms=ordered[(len(values) - 1) // 2],
            p95_ms=ordered[min(len(values) - 1, int(len(values) * 0.95))],
            scope="HOST_SUBMISSION_NOT_DEVICE_COMPLETION",
        )
    return result


def analyze_csv(paths):
    """Stream CSV metrics, retain field availability; never infer missing flows.

    Duration sums are NOT elapsed or non-overlapped device occupancy.
    """
    files, totals = [], defaultdict(lambda: {"samples": 0, "duration_us_sum": 0.0})
    for root in paths:
        for path in sorted(Path(root).rglob("*.csv")):
            with path.open(newline="", encoding="utf-8-sig") as stream:
                reader = csv.DictReader(stream)
                fields = reader.fieldnames or []
                duration = next((x for x in ("Duration(us)", "Duration (us)", "Duration") if x in fields), None)
                names = next((x for x in ("Name", "Op Name", "OP Type", "Op Type") if x in fields), None)
                rows = 0
                nonempty = dict.fromkeys(fields, 0)
                for row in reader:
                    rows += 1
                    for k in fields:
                        nonempty[k] += bool(row.get(k))
                    name = (row.get(names, "") if names else "").lower()
                    category = (
                        "copy"
                        if any(x in name for x in ("memcpy", "memcopy"))
                        else "wait_sync"
                        if any(x in name for x in ("wait", "synchron"))
                        else "update"
                        if "update" in name
                        else "replay"
                        if "replay" in name
                        else "UNRESOLVED_KERNEL_OR_API"
                    )
                    # An unlabeled Duration column has no established unit.
                    # Inventory it, but never relabel it as microseconds.
                    if duration in ("Duration(us)", "Duration (us)") and row.get(duration):
                        try:
                            value = float(row[duration])
                            if value >= 0 and value != float("inf"):
                                bucket = totals[f"{path}:{category}"]
                                bucket["samples"] += 1
                                bucket["duration_us_sum"] += value
                        except ValueError:
                            pass
                files.append(
                    dict(
                        path=str(path),
                        fields=fields,
                        nonempty=nonempty,
                        rows=rows,
                        duration_units="UNVERIFIED" if duration == "Duration" else "us",
                    )
                )
    return dict(
        status="FIELDS_ANALYZED" if files else "OPEN_NO_CSV",
        files=files,
        totals=dict(totals),
        device_idle="UNRESOLVED_WITHOUT_CALIBRATED_TIMELINE",
        target_draft_kernel_attribution="UNRESOLVED_UNLESS_EXPLICIT_FLOW_ASSOCIATION",
        stream_context_flow="FIELDS_INVENTORIED_NOT_INFERRED",
        stage="request-only profiling; prefill/decode attribution requires actual ranges",
        warning="overlapping duration sums are not end-to-end elapsed; host/device clocks not joined",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "analyze"))
    parser.add_argument("--input", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        try:
            import torch_npu

            result = preflight(torch_npu)
        except ImportError as error:
            result = {"status": "BLOCKED", "reason": str(error)}
    else:
        result = analyze_csv(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 2 if result["status"] in ("BLOCKED", "OPEN_NO_CSV") else 0


if __name__ == "__main__":
    raise SystemExit(main())
