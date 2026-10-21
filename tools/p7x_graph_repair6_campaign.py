#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate Repair6 dependency branches, never execute or publish them."""

import argparse
import copy
import shlex
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).parent.resolve()]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.p7x_graph_repair5_campaign import build_plan as previous
from tools.p7x_graph_repair6_storage import atomic_json


def build_plan(repo, inputs, output, vllm, python, sha, tree, run_id):
    plan = previous(repo, inputs, output, vllm, python, sha, tree, run_id)
    cases = plan["cases"]
    tool = str(repo / "tools/p7x_graph_repair6.py")

    def derived(source, name, flags):
        case = copy.deepcopy(cases[source])
        for field in ("argv", "launch_argv"):
            case[field] = [a.replace(source, name) for a in case[field]] + flags
        case["case_id"] = case["acceptance_identity"]["case_id"] = name
        case["outputs"] = str(output / name)
        case.pop("check_argv", None)
        case.pop("additional_checks", None)
        cases[name] = case
        return case

    def joint(name, kind, file):
        return [
            python,
            tool,
            "check-joint",
            "--kind",
            kind,
            "--artifact",
            str(output / name / file),
            "--campaign",
            str(output / name / "campaign.json"),
            "--launcher",
            str(output / "launchers" / name / "launcher.json"),
            "--plan",
            str(output / "case-plan.json"),
            "--case-id",
            name,
            "--output",
            str(output / name / (kind + "-joint.json")),
        ]

    for prefix in ("G4F", "TARGET4"):
        for arm in ("OFF", "ON"):
            name = prefix + "_" + arm
            case = cases[name]
            for key in ("argv", "launch_argv"):
                case[key] += ["--repair6-target-store"]
            case["capture_profile"] = "target4" if prefix == "TARGET4" else "base"
            case["check_argv"] = joint(name, "target", "controlled-target.json")
            case["additional_checks"] = []
            case["dependencies"] = ["PREFIX_" + arm]
            # A new binary output format/producer requires renewed execution.
            case["selection"] = "REQUIRED_STORAGE_FULL_TARGET"

    for arm in ("OFF", "ON"):
        name = "PREFIX_" + arm
        case = derived("G4F_" + arm, name, ["--repair6-target-prefix", "1"])
        case["dependencies"] = ["A0"] if arm == "OFF" else ["B0"]
        case["prefix_samples"] = 1
        case["check_argv"] = joint(name, "target-prefix", "controlled-target.json")
        case["selection"] = "REQUIRED_RESOURCE_AND_STORAGE_PREFLIGHT"
    case = derived("TARGET4_ON", "NUMERIC_TARGET4", ["--repair6-numeric-reference"])
    case["dependencies"] = ["PREFIX_ON"]
    case["check_argv"] = joint("NUMERIC_TARGET4", "numeric", "numeric-reference.json")
    case["additional_checks"] = [joint("NUMERIC_TARGET4", "target", "controlled-target.json")]
    case["selection"] = "REQUIRED_DIAGNOSTIC_CONTINUOUS_REFERENCE_NOT_NORMAL_OVERLAP"
    case = derived("O1", "NORMAL_ORDER", ["--repair6-normal-lifecycle"])
    case["dependencies"] = ["B0"]
    case["check_argv"] = joint("NORMAL_ORDER", "normal", "normal-lifecycle.json")
    case["selection"] = "REQUIRED_NORMAL_ORDER_WITH_EXTRA_DEVICE_COPIES_NOT_REUSE_PROOF"
    case = derived("PROFILE_ON_long_normal", "PROFILE_ON_long_detailed", ["--repair6-profile-detail"])
    case["dependencies"] = ["PROFILE_ON_long_normal"]
    original = cases["PROFILE_ON_long_normal"]
    case["check_argv"] = [
        a.replace("PROFILE_ON_long_normal", "PROFILE_ON_long_detailed") for a in original["check_argv"]
    ]
    case["additional_checks"] = [
        [a.replace("PROFILE_ON_long_normal", "PROFILE_ON_long_detailed") for a in command]
        for command in original["additional_checks"]
    ]
    case["selection"] = "CONDITIONAL_SEPARATE_SHAPES_STACK_DIAGNOSTIC"

    for prefix in ("G4F", "TARGET4"):
        cases[prefix + "_COMPARE"] = dict(
            dependencies=["P0"],
            timeout_seconds=3600,
            argv=[
                python,
                tool,
                "compare-target",
                "--off",
                str(output / (prefix + "_OFF") / "controlled-target.json"),
                "--on",
                str(output / (prefix + "_ON") / "controlled-target.json"),
                "--timeout",
                "3300",
                "--output",
                str(output / (prefix + "-paired.json")),
            ],
            selection="READ_ONLY_AFTER_BOTH_ATTEMPTS_ALSO_ON_PRODUCER_FAILURE",
        )
    for name, case in cases.items():
        case.setdefault("selection", "HISTORICAL_OR_CONDITIONAL_NOT_AUTOMATIC_RERUN")
        if name in ("P0", "A0", "A1", "B0"):
            case["selection"] = "REQUIRED_EXACT_SOURCE_ANCHOR"
        if name.startswith("PROFILE_") and "detailed" not in name:
            case["selection"] = (
                "AFFECTED_NEW_HOST_SUBRANGES_NORMAL_COST" if name != "PROFILE_LIFE" else "HISTORICAL_DIAGNOSTIC"
            )
        # Normalize every command into the repository-owned runner's schema.
        checks = ([case["check_argv"]] if "check_argv" in case else []) + case.get("additional_checks", [])
        case["validation_commands"] = [dict(argv=c, timeout=3300, kind="read_only_acceptance") for c in checks]
    ordered = {}
    while len(ordered) < len(cases):
        ready = [
            name
            for name, case in cases.items()
            if name not in ordered and all(d in ordered for d in case["dependencies"])
        ]
        if not ready:
            raise ValueError("missing or cyclic Repair6 dependency")
        ordered.update((name, cases[name]) for name in ready)
    plan["cases"] = ordered
    plan.update(
        schema="repair6-case-plan-v1",
        publication_status="NOT_PUBLISHED",
        code_state="PRE_PUBLICATION",
        all_g1_g5_closure=False,
        limits=dict(
            collection="runtime geometry budget plus atomic/raw reserve",
            comparator_seconds=3300,
            per_tensor_chunk_bytes=8 << 20,
            normal_generations=128,
        ),
        update_effective_generation="UNPROVEN; no new task semantics inferred",
    )
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "inputs", "output", "vllm"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("python", "sha", "tree", "run-id"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    plan = build_plan(a.repo, a.inputs, a.output, a.vllm, a.python, a.sha, a.tree, a.run_id)
    atomic_json(a.output / "case-plan.json", plan)
    commands = ["# PRE_PUBLICATION: commands do not authorize execution before verified publication."]
    for name in plan["cases"]:
        commands.append(
            shlex.join(
                [
                    a.python,
                    str(a.repo / "tools/p7x_graph_repair6_runner.py"),
                    "--plan",
                    str(a.output / "case-plan.json"),
                    "--case-id",
                    name,
                    "--output",
                    str(a.output / "runner"),
                ]
            )
        )
    (a.output / "commands.sh").write_text("\n".join(commands) + "\n")


if __name__ == "__main__":
    main()
