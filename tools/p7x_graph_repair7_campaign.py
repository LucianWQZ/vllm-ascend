#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Repair7 scoped plan. No publication or device authorization is inferred."""

import argparse
import copy
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).parent.resolve()]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.p7x_graph_repair6_campaign import build_plan as previous
from tools.p7x_graph_repair6_storage import atomic_json


def build_plan(repo, inputs, output, vllm, python, sha, tree, run_id):
    plan = previous(repo, inputs, output, vllm, python, sha, tree, run_id)
    cases = plan["cases"]
    sentinel = copy.deepcopy(cases["NUMERIC_TARGET4"])
    for field in ("argv", "launch_argv"):
        sentinel[field] = [v.replace("NUMERIC_TARGET4", "NUMERIC_SENTINEL") for v in sentinel[field]]
        sentinel[field] += ["--repair6-target-prefix", "3"]
    sentinel["outputs"] = str(output / "NUMERIC_SENTINEL")
    sentinel["case_id"] = sentinel["acceptance_identity"]["case_id"] = "NUMERIC_SENTINEL"
    sentinel["numeric_samples"] = 3
    sentinel["validation_commands"] = [copy.deepcopy(sentinel["validation_commands"][0])]
    sentinel["validation_commands"][0]["argv"] = [
        v.replace("NUMERIC_TARGET4", "NUMERIC_SENTINEL") for v in sentinel["validation_commands"][0]["argv"]
    ]
    sentinel["check_argv"] = sentinel["validation_commands"][0]["argv"]
    sentinel["additional_checks"] = []
    sentinel["dependencies"] = ["B0"]
    sentinel["resource_policy"]["geometry_preflight"] = "THREE_SAMPLE_TARGET_PLUS_DRAFT"
    cases["NUMERIC_SENTINEL"] = sentinel
    cases["NUMERIC_TARGET4"]["dependencies"] = ["NUMERIC_SENTINEL"]
    config_argv = [
        python,
        str(repo / "tools/p7x_graph_repair7.py"),
        "config-check",
        "--framework",
        "--plan",
        str(output / "case-plan.json"),
        "--output",
        str(output / "Config_Contract_Matrix.json"),
    ]
    cases["CONFIG"] = dict(
        argv=config_argv,
        timeout_seconds=300,
        dependencies=[],
        validation_commands=[],
        selection="MODEL_FREE_FRAMEWORK_CONFIG",
    )
    for name, case in cases.items():
        model = "run" in case["argv"] and any(Path(a).name == "p7x_graph_repair2.py" for a in case["argv"])
        case["requires_device_admission"] = model or name in ("P0", "C2", "CONFIG")
        if model:
            case["dependencies"] = list(dict.fromkeys(["CONFIG", *case["dependencies"]]))
        if name in ("G4F_COMPARE", "TARGET4_COMPARE"):
            prefix = name.removesuffix("_COMPARE")
            case["attempt_dependencies"] = [prefix + "_OFF", prefix + "_ON"]
    ordered = {}
    while len(ordered) < len(cases):
        ready = {
            name: case
            for name, case in cases.items()
            if name not in ordered and all(dep in ordered for dep in case.get("dependencies", []))
        }
        if not ready:
            raise ValueError("cyclic Repair7 dependency")
        ordered.update(ready)
    plan["cases"] = ordered
    plan.update(
        schema="repair7-case-plan-v1",
        publication_status="NOT_PUBLISHED",
        authorized_host_phy_ids=list(range(8)),
        tp=1,
        device_admission_receipt=None,
        device_admission_sha256=None,
        resume_policy="new attempt; no inherited success without exact source/run evidence",
        readiness="PRE_PUBLICATION; model-free CONFIG still requires authorized runtime mapping",
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


if __name__ == "__main__":
    main()
