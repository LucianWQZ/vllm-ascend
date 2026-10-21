#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Full Chrome trace flow association; no adjacency-based kernel attribution."""

from __future__ import annotations

import argparse
import bisect
import json
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path


def stamp(e):
    return Decimal(str(e["ts"]))


def flow_key(e):
    # Chrome id2.local is process scoped; global and legacy id join processes.
    ident = e.get("id2", e.get("id"))
    local = isinstance(ident, dict) and "local" in ident
    return (
        e.get("cat"),
        e.get("name") if e.get("name") in ("torch_to_npu", "enqueue_to_dequeue") else None,
        json.dumps(ident, sort_keys=True),
        e.get("pid") if local else None,
        e.get("scope"),
    )


class Intervals:
    def __init__(self, events, predicate):
        self.by_thread = defaultdict(list)
        for i, e in enumerate(events):
            if e.get("ph") == "X" and "ts" in e and predicate(e):
                self.by_thread[(e.get("pid"), e.get("tid"))].append((stamp(e), stamp(e) + Decimal(str(e["dur"])), i))
        self.index = {}
        for key, rows in self.by_thread.items():
            rows.sort()
            maximum, prefix = Decimal("-Infinity"), []
            for _, end, _ in rows:
                maximum = max(maximum, end)
                prefix.append(maximum)
            self.index[key] = ([r[0] for r in rows], prefix, rows)

    def containing(self, e):
        key = (e.get("pid"), e.get("tid"))
        if key not in self.index:
            return []
        starts, prefix, rows = self.index[key]
        at = stamp(e)
        j = bisect.bisect_right(starts, at) - 1
        matches = []
        while j >= 0 and prefix[j] >= at:
            start, end, i = rows[j]
            if start <= at <= end:
                matches.append((end - start, i))
            j -= 1
        return [i for _, i in sorted(matches)]


def analyze(events):
    flows = defaultdict(list)
    for i, e in enumerate(events):
        if e.get("ph") in ("s", "f", "t"):
            flows[flow_key(e)].append(i)
    intervals = Intervals(events, lambda e: True)
    ranges = Intervals(events, lambda e: e.get("name", "").startswith(("repair5/", "repair6/")))
    joined = []
    counts = Counter()
    by_category = defaultdict(Counter)
    for key, indices in flows.items():
        starts = [i for i in indices if events[i]["ph"] == "s"]
        ends = [i for i in indices if events[i]["ph"] == "f"]
        status = (
            "MATCHED"
            if len(starts) == len(ends) == 1 and len(indices) == 2
            else ("UNMATCHED" if not starts or not ends else "AMBIGUOUS")
        )
        counts[status] += 1
        by_category[str(key[0])][status] += 1
        row = dict(key=list(key), status=status, event_indices=indices)
        if status == "MATCHED":
            s, f = events[starts[0]], events[ends[0]]
            row.update(
                source_intervals=intervals.containing(s),
                destination_intervals=intervals.containing(f),
                host_ranges=ranges.containing(s),
                destination_ranges=ranges.containing(f),
            )
        joined.append(row)
    replays = []
    for i, e in enumerate(events):
        if e.get("ph") == "X" and e.get("name") == "repair5/Draft_REPLAY_SUBMIT":
            enclosing = ranges.containing(e)
            managers = [j for j in enclosing if events[j].get("name") == "repair5/Draft_REPLAY_PLUS_UPDATE"]
            links = [j for j, r in enumerate(joined) if r["status"] == "MATCHED" and i in r["host_ranges"]]
            managerlinks = [
                j
                for j, r in enumerate(joined)
                if r["status"] == "MATCHED" and any(k in r["host_ranges"] for k in managers)
            ]
            replays.append(
                dict(
                    event_index=i,
                    manager_indices=managers,
                    direct_flow_indices=links,
                    replay_plus_update_flow_indices=managerlinks,
                    status="EXPLICIT_FLOW_ASSOCIATED" if links else "UNRESOLVED_ASSOCIATION",
                    update_vs_replay_kernel_split="UNRESOLVED",
                )
            )
    # A small fragment is chosen by the entire manager range and explicit flow
    # destinations, not the first N events. Full event indices remain traceable.
    fragment_indices = set()
    if replays:
        chosen = max(replays, key=lambda r: len(r["direct_flow_indices"]))
        hosts = chosen["manager_indices"] or [chosen["event_index"]]
        for h in hosts:
            event = events[h]
            lo = stamp(event)
            hi = lo + Decimal(str(event["dur"]))
            fragment_indices.update(
                i
                for i, e in enumerate(events)
                if "ts" in e
                and e.get("pid") == event.get("pid")
                and e.get("tid") == event.get("tid")
                and lo <= stamp(e) <= hi
            )
        for j in chosen["replay_plus_update_flow_indices"]:
            fragment_indices.update(joined[j]["event_indices"])
            fragment_indices.update(joined[j]["destination_intervals"])
    return dict(
        schema="repair6-flow-analysis-v1",
        event_count=len(events),
        ph=dict(Counter(e.get("ph") for e in events)),
        flow_counts=dict(counts),
        by_category={k: dict(v) for k, v in by_category.items()},
        flows=joined,
        replays=replays,
        coverage=dict(total_replays=len(replays), direct_matched=sum(bool(r["direct_flow_indices"]) for r in replays)),
        clock_contract="trace-export timestamps only; decimal microseconds; no Host monotonic_ns join",
        device_idle="NOT_ESTABLISHED; kernel-uncovered intervals are not whole-device idle",
        physical_read="NOT_EXPOSED_BY_AVAILABLE_EXPORT",
        task_operand="NOT_EXPOSED_BY_AVAILABLE_EXPORT",
        fragment=[dict(original_event_index=i, event=events[i]) for i in sorted(fragment_indices)],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.trace.stat().st_size > 512 << 20:
        raise ValueError("BLOCKED_RESOURCE: JSON trace exceeds declared 512MiB parser limit")
    data = json.loads(a.trace.read_text())
    events = data["traceEvents"] if isinstance(data, dict) else data
    result = analyze(events)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
