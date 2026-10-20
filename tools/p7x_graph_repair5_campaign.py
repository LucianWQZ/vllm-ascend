#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate bounded Repair5 commands. Does NOT execute any device experiment.

Use the resulting argv arrays or commands.sh AFTER publication and independent
network source verification. Dependency failures stop descendants, not unrelated
private fixtures. No trace rerun is scheduled without a new capability.
"""

import argparse
import hashlib
import json
import shlex
from pathlib import Path

FIXTURE_SHA = "1c485bc43a6e8f6d0d83c1dc6ebfd0fb9bb1f31a8ad9e498ef1d3e9d562e7cd4"


def build_plan(
    repo: Path, inputs: Path, output: Path, vllm: Path, python: str, sha: str, tree: str, run_id: str
) -> dict:
    # Freeze the original fixture/request bytes and generate both the supervised
    # child argv and the independent joint-checker's expected identity.
    if any(len(v) != 40 or any(c not in "0123456789abcdef" for c in v) for v in (sha, tree)):
        raise ValueError("exact source SHA required")
    fixture = inputs / "fixture_manifest.json"
    if hashlib.sha256(fixture.read_bytes()).hexdigest() != FIXTURE_SHA:
        raise ValueError("original fixture manifest bytes changed")
    for name in ("smoke", "combined", "mixed", "g4-long-group1", "g4-long-group2"):
        data = json.loads((inputs / "requests" / f"{name}.json").read_text())
        if not data or any(r["max_tokens"] != 16 for r in data):
            raise ValueError("frozen request set/16 tokens required")
    driver = str(repo / "tools/p7x_graph_repair2.py")
    repair5 = str(repo / "tools/p7x_graph_repair5.py")
    acceptance = str(repo / "tools/p7x_graph_repair4_acceptance.py")
    cases: dict = {}
    cases["P0"] = {
        "dependencies": [],
        "timeout_seconds": 60,
        "argv": [python, repair5, "capabilities", "--output", str(output / "P0.json")],
        "acceptance": "execution APIs available, exact source/import/fixture check still required",
    }
    for case, mode, requests, flags, dependencies in (
        ("A0", "A0", "smoke", [], ["P0"]),
        ("A1", "A1", "smoke", [], ["P0"]),
        ("B0", "B", "smoke", [], ["P0", "A0", "A1"]),
        ("B1", "B", "combined", [], ["B0"]),
        ("B2", "B", "mixed", [], ["B0"]),
        ("O0", "B-observer-off", "smoke", [], ["B0"]),
        ("O1", "B-observer-off", "combined", [], ["B0"]),
        ("G4_OFF_short", "A0", "smoke", ["--target-probe"], ["A0"]),
        ("G4_ON_short", "B", "smoke", ["--target-probe"], ["B0"]),
        ("G4_OFF_long1", "A0", "g4-long-group1", ["--target-probe"], ["A0"]),
        ("G4_ON_long1", "B", "g4-long-group1", ["--target-probe"], ["B0"]),
        ("G4_OFF_long2", "A0", "g4-long-group2", ["--target-probe"], ["A0"]),
        ("G4_ON_long2", "B", "g4-long-group2", ["--target-probe"], ["B0"]),
        ("G4_OFF_mixed", "A0", "mixed", ["--target-probe"], ["A0"]),
        ("G4_ON_mixed", "B", "mixed", ["--target-probe"], ["B0"]),
        ("G4F_OFF", "A0", "smoke", ["--repair5-controlled-target", "--repair5-controlled-mrv2"], ["A0"]),
        (
            "G4F_ON",
            "B",
            "smoke",
            ["--repair5-controlled-target", "--repair5-controlled-mrv2", "--repair5-lifecycle"],
            ["B0", "C2"],
        ),
        ("LIFE", "B", "smoke", ["--repair5-lifecycle"], ["B0", "C2"]),
    ):
        request_path = inputs / "requests" / f"{requests}.json"
        directory = output / case
        child = [
            python,
            driver,
            "run",
            "--case",
            mode,
            "--case-id",
            case,
            "--run-id",
            run_id,
            "--attempt",
            "1",
            "--expected-sha",
            sha,
            "--vllm-repo",
            str(vllm),
            "--target-model",
            "/mnt/share/wql/Qwen3-8B",
            "--draft-model",
            "/mnt/share/dspark_qwen3_8b_block7",
            "--fixture-manifest",
            str(fixture),
            "--requests",
            str(request_path),
            "--output",
            str(directory),
        ] + flags
        launcher = output / "launchers" / case
        timeout = 3600 if case.startswith("G4F") else 1800
        argv = [
            python,
            driver,
            "launch",
            "--output",
            str(launcher),
            "--run-id",
            run_id,
            "--case-id",
            case,
            "--attempt",
            "1",
            "--mode",
            mode,
            "--timeout",
            str(timeout),
            "--",
        ] + child
        cases[case] = {
            "argv": child,
            "launch_argv": argv,
            "dependencies": dependencies,
            "timeout_seconds": timeout,
            "acceptance_identity": {
                "run_id": run_id,
                "case_id": case,
                "attempt_id": "1",
                "driver_mode": mode,
                "source_sha": sha,
                "source_tree": tree,
            },
            "case_id": case,
            "driver_mode": mode,
            "run_id": run_id,
            "attempt_id": "1",
            "source_sha": sha,
            "requests_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
            "request_path": str(request_path),
            "outputs": str(directory),
            "acceptance": (
                "controlled full collection: 4-row Target bucket may be BLOCKED; not normal request joint"
                if case.startswith("G4F")
                else "official check-joint + case-specific checker"
            ),
        }
        if not case.startswith("G4F"):
            cases[case]["check_argv"] = [
                python,
                acceptance,
                "check-joint",
                str(directory / "campaign.json"),
                "--launcher",
                str(launcher / "launcher.json"),
                "--requests",
                str(request_path),
                "--plan",
                str(output / "case-plan.json"),
                "--case-id",
                case,
                "--output",
                str(directory / "joint.json"),
            ]
    cases["B0"]["additional_checks"] = [
        [python, acceptance, "check", str(output / "B0/campaign.json"), "--output", str(output / "B0/acceptance.json")]
    ]
    cases["C2"] = {
        "dependencies": ["P0"],
        "timeout_seconds": 180,
        "argv": [
            python,
            str(repo / "tools/p7x_graph_repair4_fixture.py"),
            "run",
            "--run-id",
            run_id,
            "--attempt",
            "C2-1",
            "--output",
            str(output / "C2"),
            "--timeout",
            "180",
        ],
        "acceptance": "private FIA oracle/checker only; cannot substitute real MRV2",
    }
    for point in ("before_replay", "after_replay", "after_update", "after_completion"):
        name = f"ABORT_{point}"
        entry = json.loads(json.dumps(cases["LIFE"]))
        entry["dependencies"] = ["LIFE"]
        entry["case_id"] = name
        entry.pop("check_argv")
        entry["acceptance_identity"]["case_id"] = name
        for key in ("argv", "launch_argv"):
            entry[key] = [
                a.replace(str(output / "LIFE"), str(output / name)).replace(
                    str(output / "launchers/LIFE"), str(output / "launchers" / name)
                )
                if str(output) in a
                else name
                if a == "LIFE"
                else a
                for a in entry[key]
            ]
            entry[key] += ["--repair5-abort-at", point]
        entry["acceptance"] = (
            "child exit=73, signal=null, timeout=false, matching lifecycle identity and exact abort point; "
            "quarantine not normal completion PASS. exit74 is unexpected native failure"
        )
        entry["check_argv"] = [
            python,
            repair5,
            "check-abort",
            "--input",
            str(output / name / "mrv2-lifecycle.json"),
            "--launcher",
            str(output / "launchers" / name / "launcher.json"),
            "--plan",
            str(output / "case-plan.json"),
            "--case-id",
            name,
            "--output",
            str(output / name / "abort-check.json"),
        ]
        cases[name] = entry
    cases["G4F_ON"]["additional_checks"] = [
        [
            python,
            repair5,
            "compare-target",
            str(output / "G4F_OFF/controlled-target.pt"),
            str(output / "G4F_ON/controlled-target.pt"),
            "--output",
            str(output / "paired-target.json"),
        ],
        [
            python,
            repair5,
            "check-mrv2-aba",
            str(output / "G4F_ON/controlled-target.pt"),
            "--output",
            str(output / "real-mrv2-aba.json"),
        ],
    ]
    cases["LIFE"]["additional_checks"] = [
        [
            python,
            repair5,
            "check-lifecycle",
            str(output / "LIFE/mrv2-lifecycle.json"),
            "--output",
            str(output / "LIFE/lifecycle-check.json"),
        ]
    ]
    # Emit executable references in dependency order, including the independent
    # private C2 before its consumers. A failed branch must not erase its siblings.
    ordered: dict = {}
    while len(ordered) < len(cases):
        ready = [
            name
            for name, case in cases.items()
            if name not in ordered and all(parent in ordered for parent in case["dependencies"])
        ]
        if not ready:
            raise ValueError("missing/cyclic campaign dependency")
        ordered.update((name, cases[name]) for name in ready)
    return {
        "schema": "repair5-case-plan-v1",
        "run_id": run_id,
        "commit": sha,
        "tree": tree,
        "publication_required": True,
        "device_execution": "NOT_RUN",
        "cases": ordered,
        "trace": "NO_REPEAT_WITHOUT_NEW_FIELDS; task operands/physical reads require external capability",
        "limits": {
            "max_records_controlled": 11,
            "per_checkpoint_bytes": 536870912,
            "controlled_output_disk_budget_bytes": 17179869184,
            "watchdog_required": True,
        },
        "public_graph_gate": "STILL_FROZEN",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "inputs", "output", "vllm"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("python", "sha", "tree", "run-id"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    plan = build_plan(
        args.repo.resolve(),
        args.inputs.resolve(),
        args.output.resolve(),
        args.vllm.resolve(),
        args.python,
        args.sha,
        args.tree,
        args.run_id,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "case-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    lines = [
        "# PRE_PUBLICATION: verify published exact source before running selected commands.",
        "# These are individual commands, NOT an unattended success-tolerant campaign.",
        "# Honor case-plan dependencies; confirmed native failures stop descendants.",
    ]
    for name, case in plan["cases"].items():
        lines.extend(
            [f"# {name}; depends on {case['dependencies']}", shlex.join(case.get("launch_argv", case["argv"]))]
        )
        if "check_argv" in case:
            lines.append(shlex.join(case["check_argv"]))
        lines.extend(shlex.join(c) for c in case.get("additional_checks", []))
    (args.output / "commands.sh").write_text("\n".join(lines) + "\n")
    print(args.output / "case-plan.json")


if __name__ == "__main__":
    main()
