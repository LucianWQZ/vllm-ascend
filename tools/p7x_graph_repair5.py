#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Repair5 no-model capability preflight and deterministic diagnostic tools.

No command enables the public graph gate. Device commands never fall back to CPU.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
# Direct execution must not import tools/bisect.py instead of stdlib bisect.
if __name__ == "__main__":
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != ROOT / "tools"]
    sys.path.insert(0, str(ROOT))


def load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"tool unavailable: {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def capability(torch: Any, torch_npu: Any) -> dict:
    """Execution requirements and optional introspection have separate statuses."""
    fixture = load_tool("p7x_graph_repair4_fixture")
    paths = (
        "NPUGraph",
        "graph",
        "current_stream",
        "Stream",
        "Event",
        "graph_task_group_begin",
        "graph_task_group_end",
        "graph_task_update_begin",
        "graph_task_update_end",
        "NPUGraph.replay",
        "Event.record",
        "Event.synchronize",
        "Event.query",
        "ExternalEvent.wait",
        "ExternalEvent.reset",
        "ExternalEvent.record",
    )
    apis = {}
    missing = []
    for path in paths:
        value = getattr(torch, "npu", None)
        for part in path.split("."):
            value = getattr(value, part, None)
        if not callable(value):
            missing.append(path)
            apis[path] = {"status": "CORE_CAPABILITY_MISSING"}
        else:
            apis[path] = fixture.api_source(value)
    result = {
        "schema": "repair5-capability-v1",
        "apis": apis,
        "missing": missing,
        "model_loaded": False,
        "device_execution": "NOT_RUN",
        "public_graph_gate": "STILL_FROZEN",
    }
    try:
        result["runtime"] = fixture.runtime_evidence(torch, torch_npu)
    except (AttributeError, ValueError, ImportError, RuntimeError, OSError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
        missing.append("runtime/version/schema")
    result["status"] = "CORE_CAPABILITY_MISSING" if missing else "EXECUTION_APIS_AVAILABLE"
    result["provenance_limited"] = [p for p, v in apis.items() if v["status"] == "PROVENANCE_LIMITED"]
    result["native_semantics"] = "UNPROVEN"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("capabilities", help="no model load, no Graph submission")
    cap.add_argument("--output", type=Path, required=True)
    plan = sub.add_parser("target-plan", help="predeclared controlled Target states; not device evidence")
    plan.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("compare-target", help="identity-based complete paired Target comparison")
    check.add_argument("left", type=Path)
    check.add_argument("right", type=Path)
    check.add_argument("--output", type=Path, required=True)
    lifecycle = sub.add_parser("check-lifecycle", help="diagnostic serialized MRV2 path, not normal overlap proof")
    lifecycle.add_argument("input", type=Path)
    lifecycle.add_argument("--output", type=Path, required=True)
    aba = sub.add_parser("check-mrv2-aba", help="controlled real speculator evidence, distinct from private FIA")
    aba.add_argument("input", type=Path)
    aba.add_argument("--output", type=Path, required=True)
    abort = sub.add_parser("check-abort", help="external launcher + actual lifecycle quarantine")
    for name in ("input", "launcher", "plan", "output"):
        abort.add_argument("--" + name, type=Path, required=True)
    abort.add_argument("--case-id", required=True)
    joint = sub.add_parser("check-diagnostic", help="external frozen-plan/launcher/campaign + diagnostic")
    joint.add_argument("--kind", choices=("target", "aba", "lifecycle", "abort", "profile"), required=True)
    for name in ("input", "campaign", "launcher", "plan", "output"):
        joint.add_argument("--" + name, type=Path, required=True)
    joint.add_argument("--case-id", required=True)
    args = parser.parse_args()
    if args.command == "check-diagnostic":
        evidence = load_tool("p7x_graph_repair5_evidence")
        import torch

        document = (
            torch.load(args.input, weights_only=True, map_location="cpu")
            if args.kind in ("target", "aba")
            else json.loads(args.input.read_text())
        )
        campaign = json.loads(args.campaign.read_text())
        launcher = json.loads(args.launcher.read_text())
        expected = json.loads(args.plan.read_text())["cases"][args.case_id]
        execution = evidence.audit_execution(
            document, campaign, launcher, expected, args.input, abort=args.kind == "abort"
        )
        result = {"status": execution["status"], "execution": execution}
        if execution["status"] == "EXECUTION_ASSOCIATED":
            if args.kind in ("target", "aba"):
                target = load_tool("p7x_graph_repair5_target")
                # The Target self-check establishes completeness/structure; it
                # does NOT replace the separately executed OFF/ON comparison.
                checked = (
                    target.check_collection(document) if args.kind == "target" else target.check_mrv2_aba(document)
                )
                result.update(status=checked["status"], diagnostic=checked)
                result["scope"] = "EXECUTION_ASSOCIATED_SINGLE_ARM; OFF/ON paired comparison remains required"
                required_profile = "target4" if "--repair5-capture-profile" in expected["argv"] else "base"
                if (
                    document.get("capture_profile") != required_profile
                    or campaign.get("capture_profile") != required_profile
                    or not str(document.get("execution_device", "")).startswith("npu")
                    or campaign.get("outputs") != []
                    or document.get("device_verification") != "EXECUTED_SCOPED_NOT_GATE_CLOSURE"
                ):
                    result.update(status="INVALID_EVIDENCE", error="profile/device/controlled completion mismatch")
            elif args.kind == "abort":
                checked = load_tool("p7x_graph_repair5_lifecycle").check_abort(document, launcher, expected)
                result.update(status=checked["status"], diagnostic=checked)
                if checked["status"] == "EXPECTED_QUARANTINE_OBSERVED":
                    try:
                        result["association"] = evidence.correlate_abort(document, campaign["snapshot"])
                    except (KeyError, ValueError, TypeError) as error:
                        result.update(status="INVALID_EVIDENCE", error=str(error))
            elif args.kind == "profile":
                normal = evidence.audit_normal_requests(campaign, launcher, expected)
                result.update(requests=normal, status="OPEN")
                valid = (
                    document["status"] == "GRAPH_REPLAY_OBSERVED"
                    and normal["status"] == "PASS"
                    and document["python_worker_pid"] == launcher["pid"]
                )
                if "_ON_long_" in args.case_id:
                    valid = valid and any(
                        d.get("source_start_column", 0) > 0
                        for c in document["calls"]
                        if c["active"]
                        for d in c.get("window_descriptors", [])
                    )
                if document["profiler_enabled"]:
                    valid = valid and bool(document["raw_exports"])
                    valid = valid and all((args.input.parent / p).is_file() for p in document["raw_exports"])
                if valid:
                    result["status"] = "PASS_SCOPED_PROFILING"
                result["physical_read"] = "OPEN"
            else:
                checked = load_tool("p7x_graph_repair5_lifecycle").check_lifecycle(document)
                normal = evidence.audit_normal_requests(campaign, launcher, expected)
                result.update(diagnostic=checked, requests=normal, status="INVALID_EVIDENCE")
                if normal["status"] == "FAIL":
                    result["status"] = "FAIL"
                if checked["status"] == "STRUCTURE_VALID_NOT_EXECUTION_ACCEPTED" and normal["status"] == "PASS":
                    try:
                        result["association"] = evidence.correlate_lifecycle(
                            document, campaign["snapshot"], expected, launcher
                        )
                        result["status"] = "PASS_DIAGNOSTIC_SERIALIZED"
                    except (KeyError, ValueError, TypeError) as error:
                        result["error"] = str(error)
        code = (
            0
            if result["status"]
            in (
                "PASS",
                "PASS_SCOPED_NUMERIC",
                "PASS_DIAGNOSTIC_SERIALIZED",
                "EXPECTED_QUARANTINE_OBSERVED",
                "PASS_SCOPED_PROFILING",
                "SINGLE_ARM_STRUCTURE_VALID",
            )
            else 2
        )
    elif args.command == "capabilities":
        try:
            import torch
            import torch_npu

            result = capability(torch, torch_npu)
        except ImportError as error:
            result = {"status": "CORE_CAPABILITY_MISSING", "error": str(error), "device_execution": "NOT_RUN"}
        code = 0 if result["status"] == "EXECUTION_APIS_AVAILABLE" else 2
    elif args.command == "check-abort":
        result = load_tool("p7x_graph_repair5_lifecycle").check_abort(
            json.loads(args.input.read_text()),
            json.loads(args.launcher.read_text()),
            json.loads(args.plan.read_text())["cases"][args.case_id],
        )
        result["scope"] = "LOW_LEVEL_ONLY_USE_check-diagnostic_FOR_FINAL_EXECUTION"
        code = 2
    elif args.command == "check-lifecycle":
        result = load_tool("p7x_graph_repair5_lifecycle").check_lifecycle(json.loads(args.input.read_text()))
        result["scope"] = "LOW_LEVEL_STRUCTURE_ONLY_USE_check-diagnostic"
        code = 2
    else:
        target = load_tool("p7x_graph_repair5_target")
        if args.command == "target-plan":
            result = {"schema": target.SCHEMA, "cases": target.case_plan(), "device_execution": "NOT_RUN"}
            code = 0
        elif args.command == "check-mrv2-aba":
            import torch

            result = target.check_mrv2_aba(torch.load(args.input, weights_only=True, map_location="cpu"))
            result["scope"] = "LOW_LEVEL_ONLY_USE_check-diagnostic_FOR_FINAL_EXECUTION"
            code = 2
        else:
            import torch

            result = target.compare_collections(
                torch.load(args.left, weights_only=True, map_location="cpu"),
                torch.load(args.right, weights_only=True, map_location="cpu"),
            )
            evidence = load_tool("p7x_graph_repair5_evidence")
            result["source_artifacts"] = {str(p): evidence.file_sha(p) for p in (args.left, args.right)}
            result["scope"] = "NUMERIC_COMPARISON_ONLY; both arms require check-diagnostic execution acceptance"
            code = 0 if result["status"] == "PASS" else 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
