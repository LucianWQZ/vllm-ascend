#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Server-side helper for the P7-X base Graph G1-G5 campaign.

The runtime writes the raw JSON artifact.  This tool independently computes a
row oracle or audits that artifact.  It does not enable the feature, launch a
model, modify a gate, or claim physical NPU reads from host observations.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Executing a script places its directory first on sys.path.  This repository
# has a ``tools/bisect`` package, which would otherwise shadow Python's stdlib
# ``bisect`` once torch imports ``random``.  Make the repository root the
# explicit import root so this server handoff tool works as a direct script.
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path = [entry for entry in sys.path if os.path.abspath(entry or os.curdir) != _TOOLS_DIR]
sys.path.insert(0, os.path.dirname(_TOOLS_DIR))

_ORACLE_PATH = os.path.join(
    os.path.dirname(_TOOLS_DIR),
    "vllm_ascend",
    "worker",
    "v2",
    "spec_decode",
    "draft_window_graph_oracle.py",
)
_ORACLE_SPEC = importlib.util.spec_from_file_location(
    "_p7x_draft_window_graph_oracle",
    _ORACLE_PATH,
)
if _ORACLE_SPEC is None or _ORACLE_SPEC.loader is None:
    raise RuntimeError("unable to load the independent Graph oracle")
_ORACLE_MODULE = importlib.util.module_from_spec(_ORACLE_SPEC)
sys.modules[_ORACLE_SPEC.name] = _ORACLE_MODULE
_ORACLE_SPEC.loader.exec_module(_ORACLE_MODULE)

audit_characterization_snapshot = _ORACLE_MODULE.audit_characterization_snapshot
compute_graph_characterization_oracle = _ORACLE_MODULE.compute_graph_characterization_oracle


def _parse_ints(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error


def _write_result(result: dict[str, Any], output: Path | None) -> None:
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(payload, end="")
    else:
        output.write_text(payload, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    oracle = subparsers.add_parser("oracle")
    oracle.add_argument("--full-table", required=True, type=_parse_ints)
    oracle.add_argument("--horizon", required=True, type=int)
    oracle.add_argument("--exact", required=True, type=int)
    oracle.add_argument("--upper", required=True, type=int)
    oracle.add_argument("--window", required=True, type=int)
    oracle.add_argument("--legacy-block-size", required=True, type=int)
    oracle.add_argument("--block-span", required=True, type=int)
    oracle.add_argument("--query-width", required=True, type=int)
    oracle.add_argument("--valid-block-count", required=True, type=int)
    oracle.add_argument("--output", type=Path)

    audit = subparsers.add_parser("audit")
    audit.add_argument("artifact", type=Path)
    audit.add_argument("--output", type=Path)

    args = parser.parse_args()
    if args.command == "oracle":
        result = compute_graph_characterization_oracle(
            full_table=args.full_table,
            full_horizon_end=args.horizon,
            full_exact_end=args.exact,
            full_upper_end=args.upper,
            window_size=args.window,
            legacy_block_size=args.legacy_block_size,
            block_span_tokens=args.block_span,
            query_width=args.query_width,
            valid_block_count=args.valid_block_count,
        )
        _write_result(asdict(result), args.output)
        return 0

    snapshot = json.loads(args.artifact.read_text(encoding="utf-8"))
    result = audit_characterization_snapshot(snapshot)
    _write_result(result, args.output)
    return 0 if result["ready_for_device_campaign"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
