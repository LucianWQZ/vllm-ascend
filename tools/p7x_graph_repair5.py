#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Repair5 no-model capability preflight and deterministic diagnostic tools.

No command enables the public graph gate. Device commands never fall back to CPU.
"""

from __future__ import annotations

import argparse
import hashlib
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
    args = parser.parse_args()
    if args.command == "capabilities":
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
        code = 0 if result["status"] == "EXPECTED_QUARANTINE_OBSERVED" else 2
    elif args.command == "check-lifecycle":
        result = load_tool("p7x_graph_repair5_lifecycle").check_lifecycle(json.loads(args.input.read_text()))
        code = 0 if result["status"] == "PASS_DIAGNOSTIC_SERIALIZED" else 2
    else:
        target = load_tool("p7x_graph_repair5_target")
        if args.command == "target-plan":
            result = {"schema": target.SCHEMA, "cases": target.case_plan(), "device_execution": "NOT_RUN"}
            code = 0
        elif args.command == "check-mrv2-aba":
            import torch

            result = target.check_mrv2_aba(torch.load(args.input, weights_only=True, map_location="cpu"))
            code = 0 if result["status"] == "PASS_SCOPED_NUMERIC" else 2
        else:
            import torch

            result = target.compare_collections(
                torch.load(args.left, weights_only=True, map_location="cpu"),
                torch.load(args.right, weights_only=True, map_location="cpu"),
            )
            result["source_artifacts"] = {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.left, args.right)
            }
            code = 0 if result["status"] == "PASS" else 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
