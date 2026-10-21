#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Committed, bounded executor for ONE explicit frozen case; no implicit campaign.

launch_argv is the audited external watchdog for a device child. Comparison
commands have their own timeouts (not the old uniform 300 seconds). Nonzero
execution, comparison timeout and numerical failure remain separate records.
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.p7x_graph_repair6_watchdog import stop_owned_tree


def execute(argv, directory, timeout):
    directory.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    with (directory / "stdout.log").open("wb") as out, (directory / "stderr.log").open("wb") as err:
        process = subprocess.Popen(argv, stdout=out, stderr=err, start_new_session=True)
        timed_out = False
        interrupted = False
        terminated = []
        try:
            rc = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            rc, terminated = stop_owned_tree(process)
        except KeyboardInterrupt:
            interrupted = True
            rc, terminated = stop_owned_tree(process)
    result = dict(
        argv=argv,
        pid=process.pid,
        returncode=rc,
        signal=-rc if rc < 0 else None,
        timeout=timed_out,
        interrupted=interrupted,
        owned_processes_terminated=terminated,
        elapsed_seconds=time.monotonic() - start,
        max_rss_children=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        status="COMPARISON_TIMEOUT" if timed_out else "EXECUTION_FAILED" if rc else "COMMAND_COMPLETED_NOT_GATE_PASS",
    )
    (directory / "execution.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def run_case(plan, case_id, root):
    case = plan["cases"][case_id]
    if plan.get("publication_status") != "VERIFIED":
        raise ValueError("PRE_PUBLICATION plan cannot execute; verify published source and regenerate")
    if plan.get("schema") == "repair7-case-plan-v1" and case.get("requires_device_admission"):
        from tools.p7x_graph_repair7_device import validate_admission

        try:
            validate_admission(plan, case_id)
        except (ValueError, KeyError, TypeError, OSError) as error:
            directory = root / case_id
            directory.mkdir(parents=True, exist_ok=False)
            result = dict(
                case_id=case_id,
                status="BLOCKED_DEVICE_MAPPING",
                dependency_satisfied=False,
                primary_failure=str(error),
                records=[],
                device_started=False,
                source_identity=[plan["run_id"], plan["commit"], plan["tree"]],
            )
            (directory / "runner-result.json").write_text(json.dumps(result, indent=2) + "\n")
            return result
    records = []
    for dependency in case.get("dependencies", []):
        path = root / dependency / "runner-result.json"
        if not path.exists() or not json.loads(path.read_text()).get("dependency_satisfied"):
            raise ValueError("dependency not accepted: " + dependency)
        previous = json.loads(path.read_text())
        if plan.get("schema") == "repair7-case-plan-v1":
            if previous.get("source_identity") != [plan["run_id"], plan["commit"], plan["tree"]]:
                raise ValueError("dependency belongs to a different source/run")
    for dependency in case.get("attempt_dependencies", []):
        path = root / dependency / "runner-result.json"
        if not path.exists():
            raise ValueError("comparison must wait for both arms to finish attempts")
        if json.loads(path.read_text()).get("source_identity") != [plan["run_id"], plan["commit"], plan["tree"]]:
            raise ValueError("arm attempt belongs to a different source/run")
    from tools.p7x_graph_repair6_storage import require_space

    root.mkdir(parents=True, exist_ok=True)
    try:
        require_space(root, case.get("resource_policy", {}).get("minimum_launch_free_bytes", 0))
    except ValueError as error:
        result = dict(
            case_id=case_id,
            records=[],
            dependency_satisfied=False,
            status="BLOCKED_RESOURCE",
            reason=str(error),
            source_identity=[plan.get("run_id"), plan.get("commit"), plan.get("tree")],
        )
        (root / case_id).mkdir(exist_ok=False)
        (root / case_id / "runner-result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    primary = case.get("launch_argv", case["argv"])
    kind = "read_only_acceptance" if case.get("selection", "").startswith("READ_ONLY_") else "launch"
    commands = [dict(argv=primary, timeout=case["timeout_seconds"] + 120, kind=kind)]
    commands += case.get("validation_commands", [])
    for i, command in enumerate(commands):
        row = execute(command["argv"], root / case_id / f"command-{i:02}", command["timeout"])
        row["kind"] = command["kind"]
        records.append(row)
        # Producer failure does not erase a complete prefix. Independent
        # read-only comparator still runs; normal dependent execution stops.
        if (row["timeout"] or row.get("interrupted")) and command["kind"] == "launch":
            break
    expected_launch = 73 if case_id.startswith("ABORT_") else 0
    # An expected abort is accepted only together with its dedicated external
    # lifecycle checker. Never ignore arbitrary nonzero producer return codes.
    ok = bool(records) and all(
        r["returncode"] == (expected_launch if i == 0 else 0) and not r["timeout"] for i, r in enumerate(records)
    )
    if expected_launch:
        ok = ok and len(records) > 1 and any("check-abort" in r["argv"] for r in records[1:])
    result = dict(
        case_id=case_id,
        records=records,
        dependency_satisfied=ok,
        status="SCOPED_COMMANDS_ACCEPTED"
        if ok
        else "COMPARISON_TIMEOUT"
        if any(r["timeout"] for r in records)
        else "NUMERIC_MISMATCH"
        if any(r["kind"] != "launch" and r["returncode"] == 3 for r in records)
        else "INVALID_EVIDENCE"
        if any(r["kind"] != "launch" and r["returncode"] == 4 for r in records)
        else "BLOCKED_RESOURCE"
        if any(r["returncode"] == 5 for r in records)
        else "EXECUTION_FAILED"
        if records[0]["kind"] == "launch" and records[0]["returncode"] != expected_launch
        else "INCOMPLETE_COVERAGE",
    )
    first = records[0]
    acceptance = result["status"]
    primary = next(
        (
            dict(command_index=i, **r)
            for i, r in enumerate(records)
            if r["timeout"] or r.get("interrupted") or r["returncode"] != (expected_launch if i == 0 else 0)
        ),
        None,
    )
    producer_failed = first["kind"] == "launch" and first["returncode"] != expected_launch
    if first.get("interrupted"):
        result["status"] = "USER_INTERRUPTED"
    elif producer_failed:
        result["status"] = (
            "EXECUTION_TIMEOUT"
            if first["timeout"]
            else "SUBMISSION_UNCERTAIN_QUARANTINED"
            if first["returncode"] == 74
            else "NUMERIC_MISMATCH"
            if first["returncode"] == 3
            else "BLOCKED_RESOURCE"
            if first["returncode"] == 5
            else "EXECUTION_FAILED"
        )
    result.update(
        primary_failure=primary,
        producer_status=first["status"],
        artifact_status="SEE_READ_ONLY_ACCEPTANCE_RECORDS",
        acceptance_status=acceptance,
        dependency_status="SATISFIED" if ok else "BLOCKED",
        source_identity=[plan.get("run_id"), plan.get("commit"), plan.get("tree")],
    )
    (root / case_id / "runner-result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--case-id", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = run_case(json.loads(a.plan.read_text()), a.case_id, a.output)
    from tools.p7x_graph_repair6 import verdict_exit

    return 0 if r["dependency_satisfied"] else verdict_exit(r["status"])


if __name__ == "__main__":
    raise SystemExit(main())
