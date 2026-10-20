#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Internal DSpark TP1 characterization driver/checker, NOT a support gate.

No model, credential, server path or raw device evidence is packaged here.
Run --help. Only `run` imports vLLM and executes a model. Other commands use
stdlib, including the independent integer oracle. No global synchronization,
source patch, active table edit, or automatic trace-performance claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import MethodType
from typing import Any

# tools/bisect must not shadow the Python standard library when launching directly.
ROOT = Path(__file__).resolve().parents[1]
sys.path = [p for p in sys.path if Path(p or ".").resolve() != ROOT / "tools"]
sys.path.insert(0, str(ROOT))
PINNED_VLLM = "84030bbe3d74d99bad477a3d2e37a973ccd8865c"
PINNED_VLLM_TREE = "f3e320f8ae2e9a30e57a3c8dccfac94fcae93bee"
SECTIONS = ("events", "operand_events", "g4_records", "lifecycle_events")
IDENTITY_KEYS = ("run_id", "case_id", "attempt_id", "source_sha", "source_tree", "vllm_sha", "engine_id", "process_id")


def oracle(**kwargs):
    """Load independent math without importing the accelerator runtime."""
    path = ROOT / "vllm_ascend/worker/v2/spec_decode/draft_window_graph_oracle.py"
    spec = importlib.util.spec_from_file_location("_repair2_integer_oracle", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.compute_graph_characterization_oracle(**kwargs)


def _generation(record):
    return (record.get("generation_id"), record.get("mapping_epoch"), record.get("capture_descriptor"))


def _require(condition, message):
    # Evidence checks must have the same semantics under python -O.
    if not condition:
        raise ValueError(message)


def _invalid_evidence(message):
    return {
        "EVIDENCE_INTEGRITY": "INVALID_EVIDENCE",
        "HOST_CONTRACTS": "FAIL",
        "DEVICE_FUNCTIONAL_GATES": {
            "fixed_inventory_replay": "NOT_RUN",
            "G2_A_B_A": "OPEN",
            "G4_controlled_target_numeric": "OPEN",
        },
        "DEVICE_EVIDENCE_LEVELS": dict.fromkeys(
            ("operator_boundary", "task_internal", "physical_read", "completion_and_effective_generation"), "OPEN"
        ),
        "errors": [message],
        "missing": [],
    }


def _unique_generations(records, label, errors):
    grouped = {}
    for record in records:
        grouped.setdefault(_generation(record), []).append(record)
    unique = {}
    for key, matches in grouped.items():
        if len(matches) != 1:
            errors.append(f"duplicate {label} generation {key}")
        else:
            unique[key] = matches[0]
    return unique


def _strict_runtime_order(records):
    """Parallel Base proposals within ONE recorder/provider; gaps are legal.

    This is not the Eagle3 per-step (shared epoch) protocol or a device clock.
    """
    for field in ("sequence", "generation_id", "mapping_epoch"):
        values = [r.get(field) for r in records]
        if any(type(v) is not int or v < 0 for v in values) or any(a >= b for a, b in zip(values, values[1:])):
            return False
    return True


def audit_campaign(document: dict[str, Any]) -> dict[str, Any]:
    """Malformed evidence still produces a machine-readable, non-PASS report."""
    try:
        return _audit_campaign(document)
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as error:
        return _invalid_evidence(f"malformed evidence: {type(error).__name__}: {error}")


def _audit_campaign(document: dict[str, Any]) -> dict[str, Any]:
    """Check integrity, then associations and independent math; never infer L3/L5/L6.

    The four result namespaces are deliberately separate. A failed-run archive
    can have valid integrity, while every functional gate remains NOT_RUN.
    """
    errors: list[str] = []
    missing: list[str] = []
    _require(isinstance(document, dict), "campaign root must be an object")
    identity, snap = document.get("identity", {}), document.get("snapshot", {})
    _require(isinstance(identity, dict) and isinstance(snap, dict), "identity/snapshot must be objects")
    if document.get("schema") != "p7x-repair2-campaign-v1":
        errors.append("unsupported campaign schema")
    if any(identity.get(key) is None or identity.get(key) == "" for key in IDENTITY_KEYS):
        errors.append("missing run/case/attempt/source/engine identity")
    for key in ("source_sha", "source_tree", "vllm_sha"):
        value = identity.get(key)
        if not isinstance(value, str) or len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
            errors.append(f"invalid {key}")
    if identity.get("vllm_sha") != PINNED_VLLM:
        errors.append("wrong pinned vLLM")
    if snap.get("schema") != "mrv2-draft-window-graph-characterization-v2":
        errors.append("missing/invalid snapshot schema")
    if snap.get("process_id") != identity.get("process_id"):
        errors.append("cross-process evidence")
    if not snap.get("recorder_id") or not snap.get("provider_identity"):
        errors.append("missing recorder/provider identity")
    if snap.get("artifact_errors"):
        errors.append("recorder I/O/serialization failed")
    sections = {}
    for name in SECTIONS:
        value = snap.get(name)
        if not isinstance(value, list) or any(not isinstance(r, dict) for r in value):
            errors.append(f"invalid {name}")
            value = []
        sections[name] = value
    records = [r for section in sections.values() for r in section]
    expected_scope = {key: snap.get(key) for key in ("recorder_id", "process_id", "provider_identity")}
    if any(r.get("record_scope") != expected_scope for r in records):
        errors.append("cross-run/engine/provider record scope or missing record scope")
    sequences = [r.get("sequence") for r in records]
    if any(type(v) is not int for v in sequences) or len(sequences) != len(set(sequences)):
        errors.append("missing/duplicate sequence")
    for event in sections["operand_events"]:
        if event.get("provider_identity") != snap.get("provider_identity"):
            errors.append("cross-provider operator evidence")
        required = (
            "stage",
            "capture_descriptor",
            "backend_impl",
            "consumer",
            "layer_name",
            "actual_table_address",
            "expected_table_address",
            "table_shape",
            "table_dtype",
            "cpu_exact",
        )
        if any(event.get(key) is None for key in required):
            errors.append("missing required operator association")
        if event.get("actual_table_address") != event.get("expected_table_address"):
            errors.append("operator/backing address mismatch")
        if event.get("evidence_level") != "HOST_OPERATOR_BOUNDARY_FACT":
            errors.append("unrecognized operator evidence level")
        shape = event.get("table_shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(type(v) is not int or v <= 0 for v in shape)
            or not event.get("cpu_exact")
            or len(event["cpu_exact"]) != shape[0]
        ):
            errors.append("empty/invalid actual operator geometry")
    result: dict[str, Any] = {
        "EVIDENCE_INTEGRITY": "INVALID_EVIDENCE" if errors else "PASS",
        "HOST_CONTRACTS": "OPEN",
        "DEVICE_FUNCTIONAL_GATES": {
            "fixed_inventory_replay": "NOT_RUN",
            "G2_A_B_A": "OPEN",
            "G4_controlled_target_numeric": "OPEN",
        },
        "DEVICE_EVIDENCE_LEVELS": {
            "operator_boundary": "OPEN",
            "task_internal": "OPEN",
            "physical_read": "OPEN",
            "completion_and_effective_generation": "OPEN",
        },
        "errors": errors,
        "missing": missing,
        "scope": "No device trace inference from Python records; public Graph gate remains frozen",
    }
    if errors:
        return result
    events, operands = sections["events"], sections["operand_events"]
    bound = sorted((e for e in events if e.get("stage") == "bound"), key=lambda r: r["sequence"])
    by_generation = _unique_generations(bound, "bound", errors)
    g4_by_generation = _unique_generations(sections["g4_records"], "G4", errors)
    if not _strict_runtime_order(bound):
        errors.append("bound generation/mapping_epoch must strictly increase in recorder sequence order")
    for key in by_generation.keys() - g4_by_generation.keys():
        missing.append(f"unique G4 record missing for bound {key}")
    for key in g4_by_generation.keys() - by_generation.keys():
        errors.append(f"orphan G4 record {key}")
    captures = [e for e in events if e.get("stage") == "capture"]
    capture_backings: dict[str, tuple[Any, Any]] = {}
    for capture in captures:
        descriptor = capture.get("capture_descriptor")
        _require(isinstance(descriptor, str) and bool(descriptor), "capture descriptor must be a nonempty string")
        backing = (capture.get("route"), capture.get("backing_table_address"))
        if descriptor in capture_backings and capture_backings[descriptor] != backing:
            errors.append("capture descriptor changed backing/route within one provider scope")
        capture_backings[descriptor] = backing

    def capture_for(record):
        # Warmup and capture can legitimately repeat a descriptor. Resolve the
        # most recent preceding helper snapshot, not an arbitrary dict overwrite.
        matching = [
            e
            for e in captures
            if e.get("capture_descriptor") == record.get("capture_descriptor") and e["sequence"] < record["sequence"]
        ]
        return max(matching, key=lambda e: e["sequence"]) if matching else None

    def same_backing(record, address, label):
        if (
            type(address) is not int
            or address <= 0
            or type(record.get("backing_table_address")) is not int
            or record.get("backing_table_address") != address
        ):
            errors.append(f"{label} backing association mismatch")

    for record in bound:
        capture = capture_for(record)
        if capture is None:
            missing.append(f"capture backing snapshot missing {_generation(record)}")
        else:
            same_backing(capture, record.get("backing_table_address"), "bound/capture")
            if capture.get("route") != record.get("route"):
                errors.append("bound/capture route mismatch")
    required_stages = ("bound", "replay", "consumer", "update")
    complete = set()
    for key, record in by_generation.items():
        if any(v is None for v in key):
            errors.append("bound record has no generation/epoch/descriptor")
            continue
        matching = [e for e in events if _generation(e) == key]
        ordered = [[e["sequence"] for e in matching if e.get("stage") == stage] for stage in required_stages]
        if any(len(items) != 1 for items in ordered):
            missing.append(f"incomplete source timeline {key}")
        elif [items[0] for items in ordered] != sorted(items[0] for items in ordered):
            errors.append(f"source timeline order mismatch {key}")
        else:
            complete.add(key)
        for event in matching:
            if event.get("stage") in ("live", "replay", "consumer", "update"):
                if event.get("values_sampled_at_sequence") != record["sequence"]:
                    errors.append("cached source values have wrong BOUND sample reference")
                for field in ("route", "backing_table_address", "table_values", "exact_lengths", "upper_lengths"):
                    if event.get(field) != record.get(field):
                        errors.append(f"cached source/BOUND {field} mismatch")
    expected_layers = document.get("draft_layer_names", [])
    if not expected_layers:
        missing.append("actual Draft layer inventory absent")
    invocation_keys = set()
    for event in operands:
        if expected_layers and event.get("layer_name") not in expected_layers:
            errors.append("operator layer not in actual Draft inventory")
        if event.get("handle_identity") is not None:
            invocation_key = (
                event.get("stage"),
                *_generation(event),
                event.get("layer_name"),
                event["handle_identity"],
            )
            if invocation_key in invocation_keys:
                errors.append("duplicate operator task/layer association")
            invocation_keys.add(invocation_key)
    for key in complete:
        seen = {e.get("layer_name") for e in operands if _generation(e) == key and e.get("stage") == "update_operator"}
        if not expected_layers or set(expected_layers) != seen:
            missing.append(f"incomplete per-layer update {key}")
        captured = {
            e.get("layer_name")
            for e in operands
            if e.get("capture_descriptor") == key[2] and e.get("stage") == "capture_operator"
        }
        if set(expected_layers) != captured:
            missing.append(f"incomplete per-layer capture {key[2]}")
    for event in operands:
        capture = capture_for(event)
        if capture is None:
            missing.append(f"operator has no preceding capture backing {event.get('capture_descriptor')}")
        else:
            for field in ("expected_table_address", "actual_table_address", "captured_table_address"):
                same_backing(capture, event.get(field), f"operator/capture {field}")
            if capture.get("route") != event.get("route"):
                errors.append("operator/capture route mismatch")
        if event.get("stage") == "capture_operator":
            if event.get("generation_id") is not None or event.get("mapping_epoch") is not None:
                errors.append("startup capture must not claim a runtime generation")
            if capture and event.get("cpu_exact") != capture.get("exact_lengths"):
                errors.append("capture operator/helper exact mismatch")
        elif event.get("stage") != "update_operator":
            errors.append("unknown operator stage")
        if event.get("stage") == "update_operator" and _generation(event) not in by_generation:
            errors.append("orphan update operator")
        elif event.get("stage") == "update_operator":
            bound_record = by_generation[_generation(event)]
            for field in ("expected_table_address", "actual_table_address", "captured_table_address"):
                same_backing(bound_record, event.get(field), f"operator/bound {field}")
            if event["sequence"] <= bound_record["sequence"] or event.get("route") != bound_record.get("route"):
                errors.append("operator/bound sequence or route mismatch")
            if event.get("cpu_exact") != bound_record.get("exact_lengths"):
                errors.append("operator exact/generation lag")
            if event.get("length_kind") == "host_sequence" and event.get("length_values") != event.get("cpu_exact"):
                errors.append("actual operator length differs from typed exact")
    oracle_results = []
    for record in g4_by_generation.values():
        key = _generation(record)
        if key not in by_generation:
            errors.append("orphan G4 record")
            continue
        raw = record.get("runtime_oracle_inputs")
        if not raw or any(
            raw.get(k) is None for k in ("full_exact", "full_upper", "valid_counts", "source_identity", "mapping")
        ):
            missing.append(f"raw canonical oracle inputs absent {key}")
            continue
        try:
            count = len(raw["full_exact"])
            _require(
                count > 0
                and all(len(raw[k]) == count for k in ("full_upper", "valid_counts", "mapping", "request_ids")),
                "raw active row counts disagree",
            )
            _require(
                raw["source_identity"]["input_block_table_ptrs"][0] == record["canonical_table_address"],
                "canonical source identity mismatch",
            )
            source_event = by_generation[key]
            same_backing(source_event, record.get("window_table_address"), "G4/bound")
            _require(record["sequence"] > source_event["sequence"], "G4 must follow BOUND")
            _require(record.get("route") == source_event.get("route"), "G4/bound route mismatch")
            tables = (source_event["table_values"], record["window_table_values"])
            for table in tables:
                _require(isinstance(table, list) and len(table) >= count, "missing active table rows")
                _require(all(isinstance(row, list) and row for row in table), "invalid table rows")
                _require(all(v == 0 for row in table[count:] for v in row), "nonzero padded table row")
            for field in ("exact_lengths", "upper_lengths"):
                _require(len(source_event[field]) >= count, "missing active length rows")
                _require(all(v == 0 for v in source_event[field][count:]), "nonzero padded length row")
            for row, exact in enumerate(raw["full_exact"]):
                expected = oracle(
                    full_table=record["canonical_table_values"][row],
                    full_horizon_end=exact,
                    full_exact_end=exact,
                    full_upper_end=raw["full_upper"][row],
                    window_size=raw["window_size"],
                    legacy_block_size=raw["legacy_block_size"],
                    block_span_tokens=raw["block_span_tokens"],
                    query_width=raw["query_width"],
                    valid_block_count=raw["valid_counts"][row],
                )
                # Active required columns must independently match BOTH samples.
                # Extra zero capacity/padded rows may differ in snapshot shape.
                for table in tables:
                    _require(
                        table[row][: expected.required_columns] == list(expected.expected_block_ids),
                        "BOUND/G4 active table differs from canonical oracle",
                    )
                    _require(all(v == 0 for v in table[row][expected.required_columns :]), "nonzero active table tail")
                _require(source_event["exact_lengths"][row] == expected.visible_length, "exact length mismatch")
                _require(source_event["upper_lengths"][row] == expected.window_upper, "upper length mismatch")
                oracle_results.append(
                    {
                        "generation": key,
                        "row": row,
                        "H": exact,  # Base parallel route has future_offset=0.
                        "S": expected.window_start,
                        "E": exact,
                        "V": expected.visible_length,
                        "columns": expected.expected_columns,
                    }
                )
        except (KeyError, TypeError, IndexError, ValueError, RuntimeError) as error:
            errors.append(f"independent runtime oracle mismatch {key}: {error}")
    if bound and len(sections["g4_records"]) != len(bound):
        missing.append("G4/bound cardinality mismatch")
    # Actual graphs and replay records, never request counts, establish coverage.
    required_buckets = {(1, 7), (2, 14), (4, 28)}
    inventory = {tuple(v) for v in document.get("draft_graph_keys", [])}
    covered = set()
    for n, tokens in required_buckets:
        if any(f"reqs={n};tokens={tokens};" in str(key[2]) for key in complete):
            covered.add((n, tokens))
    result["actual_replay_buckets"] = sorted(covered)
    result["required_replay_buckets"] = sorted(required_buckets)
    result["missing_replay_buckets"] = sorted(required_buckets - covered)
    result["missing_inventory_buckets"] = sorted(required_buckets - inventory)
    result["independent_oracle_rows"] = oracle_results
    result["HOST_A_B_A_CONTENT"] = check_aba(bound, sections["g4_records"])
    if not bound or not operands or not oracle_results:
        missing.append("nonempty bound/operator/raw-oracle evidence required")
    if not errors and not missing:
        result["HOST_CONTRACTS"] = "PASS"
        result["DEVICE_EVIDENCE_LEVELS"]["operator_boundary"] = "PASS"
    if errors:
        result["EVIDENCE_INTEGRITY"] = "INVALID_EVIDENCE"
        result["HOST_CONTRACTS"] = "FAIL"
    elif document.get("startup_complete") and document.get("status") == "OK":
        result["DEVICE_FUNCTIONAL_GATES"]["fixed_inventory_replay"] = (
            "PASS" if required_buckets <= inventory and required_buckets <= covered and not missing else "OPEN"
        )
    return result


def check_aba(bound, full_records):
    """Same recorder/engine was checked by audit_campaign, not inferred from ptrs.

    This is a host/content association, NOT a device happens-before proof.
    """
    if len(bound) < 3:
        return "OPEN"
    if any(type(r.get("sequence")) is not int for r in bound):
        return "FAIL"
    ordered = sorted(bound, key=lambda r: r["sequence"])
    if not _strict_runtime_order(ordered):
        return "FAIL"
    a, b, last = ordered[-3:]
    errors = []
    source = _unique_generations(full_records, "G4", errors)
    if errors:
        return "FAIL"
    selected = [source.get(_generation(r)) for r in (a, b, last)]
    if any(r is None for r in selected):
        return "OPEN"
    scope = a.get("record_scope")
    if not isinstance(scope, dict) or any(
        scope.get(k) is None for k in ("recorder_id", "process_id", "provider_identity")
    ):
        return "FAIL"
    if any(r.get("record_scope") != scope for r in (a, b, last, *selected)):
        return "FAIL"
    raw = [r.get("runtime_oracle_inputs") for r in selected]
    if any(not r for r in raw):
        return "OPEN"
    for key in ("capture_descriptor", "backing_table_address"):
        if not a.get(key) or not (a[key] == b.get(key) == last.get(key)):
            return "OPEN"
    for key in ("table_values", "exact_lengths", "upper_lengths"):
        if not a.get(key) or a[key] != last.get(key):
            return "OPEN"
    if a["table_values"] == b.get("table_values"):
        return "OPEN"
    if raw[0].get("source_identity") != raw[2].get("source_identity"):
        return "OPEN"
    for key in ("full_exact", "full_upper", "valid_counts", "mapping"):
        if raw[0].get(key) is None or raw[0][key] != raw[2].get(key):
            return "OPEN"
    return "PASS"


def compare_full_write(left, right):
    """Compare aligned semantic state across runs; numeric ptr equality is local.

    alignment_sha256 must cover Target tokens, positions, full KV state and
    query/request layout from a controlled fixture, not just prompt/seed.
    """
    if not left.get("alignment_sha256") or left.get("alignment_sha256") != right.get("alignment_sha256"):
        return {"status": "OPEN", "reason": "controlled Target input/history/KV alignment missing"}
    fields = (
        "canonical_table_values",
        "slot_mapping_values",
        "position_values",
        "query_layout_values",
        "target_length_values",
    )
    if any(left.get(k) in (None, [], {}) or right.get(k) in (None, [], {}) for k in fields):
        return {"status": "OPEN", "reason": "write/Target values not collected"}
    return {
        "status": "PASS" if all(left[k] == right[k] for k in fields) else "FAIL",
        "scope": "aligned full write/Target semantic state only; cross-process pointers not compared",
    }


def compare_target_numeric(left, right):
    required = ("aligned_input_state_sha256", "dtype", "values", "atol", "rtol", "tolerance_reason")
    if any(left.get(k) is None or right.get(k) is None for k in required):
        return {"status": "OPEN", "reason": "missing controlled input/KV/position/numeric evidence"}
    if any(left[k] != right[k] for k in ("aligned_input_state_sha256", "dtype", "atol", "rtol", "tolerance_reason")):
        return {"status": "OPEN", "reason": "comparison points/tolerances are not aligned"}
    a, b = left["values"], right["values"]
    if not a or len(a) != len(b) or left["atol"] < 0 or left["rtol"] < 0:
        return {"status": "INVALID_EVIDENCE", "reason": "empty/invalid numeric comparison"}
    okay = all(
        math.isfinite(x) and math.isfinite(y) and abs(x - y) <= left["atol"] + left["rtol"] * abs(y)
        for x, y in zip(a, b)
    )
    return {"status": "PASS" if okay else "FAIL", "max_abs_error": max(abs(x - y) for x, y in zip(a, b))}


def analyze_campaign(document):
    """Derived analysis only: retain the official check, scan the entire sequence."""
    audit = audit_campaign(document)
    rows = audit.get("independent_oracle_rows", [])
    boundaries = {
        "S_eq_0": any(r["S"] == 0 for r in rows),
        "S_gt_0": any(r["S"] > 0 for r in rows),
        "non_block_end": any(r["E"] % 128 for r in rows),
    }
    for end in (32, 1024, 1025, 1151, 1152, 1153, 2049):
        boundaries[f"runtime_E_{end}"] = any(r["E"] == end for r in rows)
    candidates = []
    if audit["EVIDENCE_INTEGRITY"] == "PASS":
        snap = document["snapshot"]
        bound = sorted((r for r in snap["events"] if r.get("stage") == "bound"), key=lambda r: r["sequence"])
        groups = {}
        for record in bound:
            key = (record["capture_descriptor"], record["backing_table_address"])
            groups.setdefault(key, []).append(record)
        # Bounded diagnostic scan: report incomplete search, never infer absence
        # from a budget-exhausted prefix. Each candidate includes raw sequence IDs.
        examined, exhausted = 0, False
        for records in groups.values():
            for i, a in enumerate(records):
                for j in range(i + 2, len(records)):
                    examined += 1
                    if examined > 100000:
                        exhausted = True
                        break
                    last = records[j]
                    if any(a[k] != last[k] for k in ("table_values", "exact_lengths", "upper_lengths")):
                        continue
                    b = next((r for r in records[i + 1 : j] if r["table_values"] != a["table_values"]), None)
                    if b and check_aba([a, b, last], snap["g4_records"]) == "PASS":
                        candidates.append(
                            {
                                "sequences": [a["sequence"], b["sequence"], last["sequence"]],
                                "descriptor": a["capture_descriptor"],
                                "backing": a["backing_table_address"],
                                "generations": [_generation(r) for r in (a, b, last)],
                            }
                        )
                if exhausted:
                    break
            if exhausted:
                break
    else:
        exhausted = False
    return {
        "official_check": audit,
        "requested": document.get("outputs", []),
        "observed_runtime_rows": rows,
        "boundaries": {k: "COVERED_HOST_ORACLE" if v else "NOT_REACHED" for k, v in boundaries.items()},
        "aba_scan": {
            "status": "HOST_CANDIDATE" if candidates else "OPEN",
            "candidates": candidates,
            "scan_complete": not exhausted,
            "device_refresh": "OPEN",
        },
    }


def compare_probe_sets(left, right, left_dir, right_dir):
    """Strict whole-collection comparison, never a FULL_TARGET equivalence claim.

    Raw paths resolve beside each delivered JSON (not arbitrary server paths).
    Repeated hashes remain distinct ordered samples; missing/extra samples fail.
    """
    result = {
        "write_semantics": "OPEN",
        "head_only_numeric": "OPEN",
        "full_target_numeric": "OPEN",
        "pairs": [],
        "errors": [],
        "tolerance": {"atol": 0, "rtol": 0, "reason": "identical head input, exact diagnostic comparison"},
    }
    if left.get("errors") or right.get("errors"):
        result["errors"].append("probe collection errors")
        return result

    def tensor(value):
        _require(isinstance(value, dict), "missing tensor snapshot")
        _require(all(value.get(k) is not None for k in ("shape", "dtype", "values")), "missing tensor fields")
        _require(
            isinstance(value["shape"], list) and all(type(n) is int and n >= 0 for n in value["shape"]),
            "invalid tensor shape",
        )

        def validate(data, shape):
            if shape:
                _require(isinstance(data, list) and len(data) == shape[0], "shape/value mismatch")
                for child in data:
                    validate(child, shape[1:])
            else:
                _require(isinstance(data, (int, float)) and math.isfinite(data), "nonfinite/nonnumeric value")

        validate(value["values"], value["shape"])
        return {k: value[k] for k in ("shape", "dtype", "values")}

    def state(value):
        _require(isinstance(value, dict), "missing phase snapshot")
        layers = value.get("target_layers")
        _require(isinstance(layers, dict) and bool(layers), "missing Target layers")
        _require(bool(value.get("canonical_tables")), "missing canonical tables")
        return {
            "canonical_tables": [tensor(t) for t in value["canonical_tables"]],
            **{k: tensor(value.get(k)) for k in ("slots", "positions", "query")},
            "target_layers": {k: {f: tensor(v.get(f)) for f in ("table", "exact")} for k, v in layers.items()},
        }

    for kind, status in (("full_write_target", "write_semantics"), ("target_logits_head", "head_only_numeric")):
        a = [r for r in left.get("records", []) if r.get("kind") == kind]
        b = [r for r in right.get("records", []) if r.get("kind") == kind]
        result[kind + "_counts"] = [len(a), len(b)]
        if not a or not b:
            continue
        if len(a) != len(b):
            result[status] = "FAIL"
            result["errors"].append(f"{kind} sample cardinality differs")
            continue
        try:
            for index in range(len(a)):
                x, y = a[index], b[index]
                if kind == "full_write_target":
                    for phase in ("before", "after"):
                        _require(
                            state(x.get(phase)) == state(y.get(phase)), f"sample {index} {phase} semantic mismatch"
                        )
                else:
                    import torch

                    tensors = []
                    pair = {"index": index, "files": []}
                    for item, root in ((x, left_dir), (y, right_dir)):
                        path = Path(root) / Path(item["path"]).name
                        _require(path.is_file(), f"RAW_NOT_DELIVERED: {path.name}")
                        data = torch.load(path, map_location="cpu", weights_only=True)
                        for name, shape in (("hidden_states", "input_shape"), ("logits", "output_shape")):
                            t = data[name]
                            _require(isinstance(t, torch.Tensor) and t.numel() > 0, "empty/non-tensor raw input")
                            _require(
                                list(t.shape) == item[shape] and bool(torch.isfinite(t).all()),
                                "shape/nonfinite raw tensor",
                            )
                        _require(str(data["logits"].dtype) == item["dtype"], "dtype mismatch")
                        digest = hashlib.sha256(
                            data["hidden_states"].contiguous().view(torch.uint8).numpy().tobytes()
                        ).hexdigest()
                        _require(digest == item["input_sha256"], "input hash mismatch")
                        pair["files"].append(
                            {
                                "path": str(path),
                                "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                "input_sha256": digest,
                                "output_sha256": hashlib.sha256(
                                    data["logits"].contiguous().view(torch.uint8).numpy().tobytes()
                                ).hexdigest(),
                            }
                        )
                        tensors.append(data)
                    for key in ("hidden_states", "logits"):
                        p, q = tensors[0][key], tensors[1][key]
                        _require(
                            p.dtype == q.dtype and p.shape == q.shape and torch.equal(p, q), f"{key} differs at {index}"
                        )
                    result["pairs"].append(pair)
            result[status] = "PASS"
        except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
            result[status] = "OPEN" if "RAW_NOT_DELIVERED" in str(error) else "FAIL"
            result["errors"].append(str(error))
    return result


def launch_case(args):
    """External supervisor: no model imports, no fabricated campaign on SIGSEGV."""
    args.output.mkdir(parents=True, exist_ok=False)
    argv = args.argv[1:] if args.argv and args.argv[0] == "--" else args.argv
    _require(bool(argv), "launcher requires a child argv")
    record = {
        "run_id": args.run_id,
        "case_id": args.case_id,
        "driver_mode": args.mode,
        "source_sha": _git(ROOT, "rev-parse", "HEAD"),
        "source_tree": _git(ROOT, "rev-parse", "HEAD^{tree}"),
        "attempt_id": args.attempt,
        "argv": argv,
        "status": "STARTING",
        "returncode": None,
        "signal": None,
        "timeout": False,
        "host_start_ns": time.monotonic_ns(),
    }
    if "--requests" in argv:
        request_path = Path(argv[argv.index("--requests") + 1])
        record["requests_sha256"] = hashlib.sha256(request_path.read_bytes()).hexdigest()
    path = args.output / "launcher.json"
    _write(path, record)
    with (args.output / "stdout.log").open("wb") as stdout, (args.output / "stderr.log").open("wb") as stderr:
        try:
            child = subprocess.Popen(argv, stdout=stdout, stderr=stderr, start_new_session=True)
        except OSError as error:
            record.update(status="NOT_STARTED", error=str(error), returncode=127)
            _write(path, record)
            return 127
        record.update(pid=child.pid, status="RUNNING")
        _write(path, record)
        try:
            code = child.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            record["timeout"] = True
            os.killpg(child.pid, signal.SIGTERM)
            try:
                code = child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                code = child.wait()
    record.update(returncode=code, signal=-code if code < 0 else None, status="EXITED", host_end_ns=time.monotonic_ns())
    _write(path, record)
    return 124 if record["timeout"] else code if code >= 0 else 128 - code


def _write(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _manager_inventory(manager):
    if manager is None:
        return {"descriptors": None, "graphs": None}

    def describe(d):
        return {
            "mode": d.cg_mode.name,
            "num_reqs": d.num_reqs,
            "num_tokens": d.num_tokens,
            "uniform_token_count": d.uniform_token_count,
        }

    return {
        "descriptors": [describe(d) for group in manager._capture_descs.values() for d in group],
        "graphs": [describe(d) for d in manager.graphs],
    }


class TargetBoundaryProbe:
    """Test-driver-only, instance-scoped observation outside graph capture.

    Installed AFTER startup capture. The original methods/return values remain
    untouched, including on errors. This probe adds D2H and is explicitly NOT
    PERFORMANCE REPRESENTATIVE. A logits-head equality is not whole-Target/KV
    equality; the latter still requires a controlled aligned fixture.
    """

    def __init__(self, runner, output):
        self.runner, self.output = runner, output
        self.records, self.errors = [], []
        self._saved = []

    def _safe(self, callback):
        try:
            if len(self.records) >= 256:
                raise RuntimeError("bounded Target probe exhausted")
            callback()
        except Exception as error:
            if len(self.errors) < 8:
                self.errors.append(repr(error))

    def _snapshot(self):
        runner, spec = self.runner, self.runner.speculator

        def tensor(t):
            if t is None:
                return None
            return {
                "ptr": t.data_ptr(),
                "shape": list(t.shape),
                "dtype": str(t.dtype),
                "values": t.detach().cpu().tolist(),
            }

        target = runner.model_state.attn_metadata
        return {
            "target_root": id(target),
            "target_layers": {
                name: {
                    "object": id(m),
                    "table": tensor(getattr(m, "block_tables", None)),
                    "exact": tensor(getattr(m, "seq_lens", None)),
                }
                for name, m in target.items()
                if m is not None
            },
            "canonical_tables": [tensor(t) for t in spec.block_tables.input_block_tables],
            "slots": tensor(spec.block_tables.slot_mappings),
            "positions": tensor(spec.input_buffers.positions),
            "query": tensor(spec.input_buffers.query_start_loc),
        }

    def install(self):
        probe = self
        spec, model = self.runner.speculator, self.runner.model
        original_propose, original_logits = spec.propose, model.compute_logits

        def propose(_self, *args, **kwargs):
            record = {
                "kind": "full_write_target",
                "before": None,
                "after": None,
                "host_start_ns": time.monotonic_ns(),
                "alignment_warning": (
                    "pre/post proposal are different full-prepare phases; compare A0/B at aligned phases"
                ),
            }
            probe._safe(lambda: record.update(before=probe._snapshot()))
            try:
                return original_propose(*args, **kwargs)
            finally:
                probe._safe(lambda: record.update(after=probe._snapshot()))
                record["host_end_ns"] = time.monotonic_ns()
                probe._safe(lambda: probe.records.append(record))

        def logits(_self, *args, **kwargs):
            result = original_logits(*args, **kwargs)

            def collect():
                hidden = args[0] if args else kwargs.get("hidden_states")
                if hidden is None or result is None:
                    raise RuntimeError("Target logits boundary unavailable")
                import torch

                host_input, host_output = hidden.detach().cpu().contiguous(), result.detach().cpu().contiguous()
                digest = hashlib.sha256(host_input.view(torch.uint8).numpy().tobytes()).hexdigest()
                path = probe.output / f"target-logits-{len(probe.records):05}.pt"
                torch.save({"hidden_states": host_input, "logits": host_output}, path)
                probe.records.append(
                    {
                        "kind": "target_logits_head",
                        "path": str(path),
                        "input_sha256": digest,
                        "input_shape": list(host_input.shape),
                        "dtype": str(host_output.dtype),
                        "output_shape": list(host_output.shape),
                        "full_target_kv_alignment": None,
                        "scope": "actual compute_logits operands; NOT full Target state equivalence",
                    }
                )

            probe._safe(collect)
            return result

        for owner, name, replacement in ((spec, "propose", propose), (model, "compute_logits", logits)):
            self._saved.append((owner, name, vars(owner).get(name), name in vars(owner)))
            setattr(owner, name, MethodType(replacement, owner))

    def close(self):
        for owner, name, value, existed in reversed(self._saved):
            if existed:
                setattr(owner, name, value)
            else:
                delattr(owner, name)
        self._saved.clear()
        _write(
            self.output / "target-boundary.json",
            {
                "records": self.records,
                "errors": self.errors,
                "impact": "D2H; NOT PERFORMANCE REPRESENTATIVE; no controlled full-KV equivalence claim",
            },
        )


def run_case(args):
    """Fixed startup/serving smoke, with no hidden capture-size downgrade.

    Source/model verification precedes model loading. The in-process executor
    is the same audited diagnostic architecture as repair1, not a new transport.
    A/B/A requests are labelled attempts, never asserted semantic equality.
    """
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    identity = {
        "run_id": args.run_id,
        "case_id": getattr(args, "case_id", None) or args.case,
        "driver_mode": args.case,
        "attempt_id": args.attempt,
        "source_sha": _git(ROOT, "rev-parse", "HEAD"),
        "source_tree": _git(ROOT, "rev-parse", "HEAD^{tree}"),
        "vllm_sha": _git(args.vllm_repo, "rev-parse", "HEAD"),
        "engine_id": None,
        "process_id": os.getpid(),
    }
    if identity["source_sha"] != args.expected_sha or _git(ROOT, "status", "--porcelain"):
        raise RuntimeError("candidate SHA/clean source mismatch")
    if (
        identity["vllm_sha"] != PINNED_VLLM
        or _git(args.vllm_repo, "rev-parse", "HEAD^{tree}") != PINNED_VLLM_TREE
        or _git(args.vllm_repo, "status", "--porcelain")
    ):
        raise RuntimeError("pinned vLLM SHA/tree/clean mismatch")
    request_bytes = args.requests.read_bytes() if args.requests else None
    requests = (
        json.loads(request_bytes)
        if request_bytes is not None
        else [{"label": f"smoke-{n}", "prompts": ["Hello, my name is"] * n, "max_tokens": 16} for n in (1, 2, 4)]
    )
    request_manifest = {
        "sha256": hashlib.sha256(request_bytes).hexdigest() if request_bytes is not None else None,
        "entries": requests,
    }
    fixtures = json.loads(args.fixture_manifest.read_text())
    for item in fixtures["files"]:
        path = Path(item["path"])
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != item["sha256"]:
            raise RuntimeError(f"fixture hash mismatch: {path}")
    if len(fixtures["files"]) < 9:
        raise RuntimeError("fixture manifest must include target config/index/5 shards and draft config/weights")
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_ASCEND_EXPERIMENTAL_DRAFT_WINDOW_GRAPH"] = "1" if args.case.startswith("B") else "0"
    os.environ["VLLM_ASCEND_EXPERIMENTAL_DRAFT_WINDOW_GRAPH_OBSERVE"] = "0" if args.case == "B-observer-off" else "1"
    os.environ["VLLM_ASCEND_EXPERIMENTAL_DRAFT_WINDOW_GRAPH_ARTIFACT"] = str(output / "partial.json")
    import vllm
    from vllm import LLM, SamplingParams

    import vllm_ascend

    record = {
        "schema": "p7x-repair2-campaign-v1",
        "identity": identity,
        "startup_complete": False,
        "status": "STARTED",
        "argv": sys.argv,
        "fixture_manifest": fixtures,
        "imports": {"vllm": vllm.__file__, "vllm_ascend": vllm_ascend.__file__},
        "observer_enabled": args.case != "B-observer-off",
        "outputs": [],
        "request_manifest": request_manifest,
        "controlled_target_numeric": None,
        "task_trace": None,
        "physical_read_trace": None,
    }
    llm = session = target_probe = full_target_probe = lifecycle_probe = None
    additional = {} if args.case == "A0" else {"draft_window_size": 1024}
    config = dict(
        model=args.target_model,
        max_model_len=4096,
        enforce_eager=False,
        async_scheduling=False,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.80,
        max_num_seqs=4,
        max_num_batched_tokens=4096,
        seed=0,
        speculative_config={
            "model": args.draft_model,
            "method": "dspark",
            "num_speculative_tokens": 7,
            "enable_adaptive_verification": False,
        },
        additional_config=additional,
        disable_log_stats=True,
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [7, 14, 28]},
    )
    record["llm_kwargs"] = config
    _write(output / "progress.json", {"identity": identity, "phase": "BEFORE_STARTUP"})
    try:
        if getattr(args, "repair5_controlled_target", False) or getattr(args, "repair5_lifecycle", False):
            tool_spec = importlib.util.spec_from_file_location("_repair5", ROOT / "tools/p7x_graph_repair5.py")
            if tool_spec is None or tool_spec.loader is None:
                raise RuntimeError("Repair5 capability tool missing")
            repair5 = importlib.util.module_from_spec(tool_spec)
            tool_spec.loader.exec_module(repair5)
            import torch
            import torch_npu

            capability = repair5.capability(torch, torch_npu)
            _write(output / "capability.json", capability)
            if capability["status"] != "EXECUTION_APIS_AVAILABLE":
                raise RuntimeError("CORE_CAPABILITY_MISSING before model load")
        llm = LLM(**config)
        core = llm.llm_engine.engine_core.engine_core
        runner = core.model_executor.driver_worker.worker.model_runner
        spec = runner.speculator
        identity["engine_id"] = id(core)
        record["model_runner"] = f"{type(runner).__module__}.{type(runner).__name__}"
        record["speculator"] = f"{type(spec).__module__}.{type(spec).__name__}"
        if (
            record["model_runner"] != "vllm_ascend.worker.v2.model_runner.NPUModelRunner"
            or record["speculator"] != "vllm_ascend.worker.v2.spec_decode.dspark.speculator.AscendDSparkSpeculator"
        ):
            raise RuntimeError("INVALID_RUN_WRONG_MODEL_RUNNER_OR_SPECULATOR")
        record["target_inventory"] = _manager_inventory(runner.cudagraph_manager)
        record["draft_inventory"] = _manager_inventory(spec.query_cudagraph_manager)
        record["draft_graph_keys"] = [[d["num_reqs"], d["num_tokens"]] for d in record["draft_inventory"]["graphs"]]
        record["draft_layer_names"] = list(spec.draft_attn_layer_names)
        record["startup_complete"] = True
        _write(
            output / "progress.json",
            {
                "identity": identity,
                "phase": "STARTUP_COMPLETE",
                "draft_inventory": record["draft_inventory"],
                "draft_layer_names": record["draft_layer_names"],
            },
        )
        session = getattr(spec, "_draft_window_graph_characterization", None)
        record["provider_installed"] = hasattr(spec, "_draft_window_graph_binding_provider")
        if args.case == "A0" and (session is not None or record["provider_installed"]):
            raise RuntimeError("OFF allocated Window state")
        if args.case.startswith("B"):
            if session is None:
                raise RuntimeError("experimental Graph session missing; no eager fallback accepted")
            if set(map(tuple, record["draft_graph_keys"])) != {(1, 7), (2, 14), (4, 28)}:
                raise RuntimeError("fixed capture inventory not realized")
            fingerprints = [i.fingerprint for i in session.provider.registered_capture_identities(route=session.route)]
            if any(
                f.query_width != 7 or f.static_k != 7 or f.window_size != 1024 or f.block_span_tokens != 128
                for f in fingerprints
            ):
                raise RuntimeError("runtime fixture geometry differs from frozen Base")
        if args.target_probe:
            target_probe = TargetBoundaryProbe(runner, output)
            target_probe.install()
        if getattr(args, "repair5_lifecycle", False):
            lifecycle = repair5.load_tool("p7x_graph_repair5_lifecycle")
            lifecycle_probe = lifecycle.MRV2LifecycleProbe(
                spec, output / "mrv2-lifecycle.json", identity, abort_at=args.repair5_abort_at
            )
            lifecycle_probe.__enter__()
        if getattr(args, "repair5_controlled_target", False):
            controlled = repair5.load_tool("p7x_graph_repair5_target")
            fixture = controlled.ControlledTarget(
                runner,
                fixture_id=hashlib.sha256(args.fixture_manifest.read_bytes()).hexdigest(),
                arm="OFF" if args.case == "A0" else "ON",
                scheduler=core.scheduler,
                identity=identity,
            )
            fixture.execute(output / "controlled-target.pt", include_draft=args.repair5_controlled_mrv2)
            record["controlled_target_numeric"] = {
                "artifact": "controlled-target.pt",
                "kind": "FULL_TARGET_CONTROLLED_SYNTHETIC_KV",
                "status": "EXECUTED_SCOPED_NOT_GATE_CLOSURE",
            }
            # A disposable engine is never returned to serving after diagnostic KV staging.
            requests = []
        if getattr(args, "full_target_probe", False):
            adapter_spec = importlib.util.spec_from_file_location(
                "_repair4_full_target", ROOT / "tools/p7x_graph_repair4_target.py"
            )
            if adapter_spec is None or adapter_spec.loader is None:
                raise RuntimeError("FULL target adapter module unavailable")
            adapter_module = importlib.util.module_from_spec(adapter_spec)
            adapter_spec.loader.exec_module(adapter_module)
            fixture_id = hashlib.sha256(
                json.dumps(
                    {"files": fixtures["files"], "source": identity["source_sha"], "vllm": identity["vllm_sha"]},
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            full_target_probe = adapter_module.LiveTargetAdapter(
                runner,
                fixture_id=fixture_id,
                max_bytes=args.full_target_max_bytes,
                max_records=args.full_target_max_records,
            )
            full_target_probe.__enter__()
        # Fixed 1/2/4 requests are attempts; checker requires actual descriptor events.
        for request in requests:
            _write(
                output / "progress.json",
                {"identity": identity, "phase": "BEFORE_GENERATE", "request": request["label"]},
            )
            start = time.monotonic_ns()
            results = llm.generate(
                request["prompts"],
                SamplingParams(temperature=0.0, seed=0, ignore_eos=True, max_tokens=request["max_tokens"]),
                use_tqdm=False,
            )
            record["outputs"].append(
                {
                    "label": request["label"],
                    "prompts": request["prompts"],
                    "host_start_ns": start,
                    "host_end_ns": time.monotonic_ns(),
                    "prompt_token_ids": [r.prompt_token_ids for r in results],
                    "token_ids": [r.outputs[0].token_ids for r in results],
                }
            )
            _write(
                output / "progress.json", {"identity": identity, "phase": "AFTER_GENERATE", "request": request["label"]}
            )
        record["status"] = "OK"
    except BaseException as error:
        record.update(
            status="EXCEPTION",
            exception_type=type(error).__name__,
            exception_message=str(error),
            traceback=traceback.format_exc(),
        )
        # Preserve original exception. Cleanup/artifact errors are separate fields.
    finally:
        if lifecycle_probe is not None:
            try:
                lifecycle_probe.__exit__(None, None, None)
            except Exception as error:
                record["lifecycle_probe_error"] = repr(error)
        if full_target_probe is not None:
            try:
                full_target_probe.__exit__(None, None, None)
                full_target_probe.save(output / "target-full-forward.pt")
                record["full_target_probe"] = {
                    "artifact": "target-full-forward.pt",
                    "status": "ALIGNMENT_REQUIRED",
                    "controlled_pair_generated": False,
                    "live_readiness": "NOT_ESTABLISHED",
                    "record_statuses": [r["status"] for r in full_target_probe.records],
                    "skipped_for_limit": full_target_probe.skipped_for_limit,
                    "max_bytes_per_record": full_target_probe.max_bytes,
                    "max_records": full_target_probe.max_records,
                    "scope": "actual target forward; complete declared read/write KV blocks; no restore",
                    "update_effective_generation": "UNKNOWN",
                    "impact": "stream synchronization and D2H; NOT PERFORMANCE REPRESENTATIVE",
                }
            except Exception as error:
                record["full_target_probe_error"] = repr(error)
        if target_probe is not None:
            try:
                target_probe.close()
            except Exception as error:
                record["target_probe_error"] = repr(error)
        try:
            if session is not None:
                record["snapshot"] = session.recorder.snapshot()
            elif (output / "partial.json").is_file():
                record["snapshot"] = json.loads((output / "partial.json").read_text())
        except Exception as error:
            record["artifact_collection_error"] = repr(error)
        _write(output / "campaign.json", record)
        if lifecycle_probe is not None and lifecycle_probe.poisoned:
            # Diagnostic child only: uncertain submission is never reused or
            # fed to normal engine teardown. External launch/watchdog owns it.
            record["diagnostic_quarantine"] = "PROCESS_EXIT_74_NO_REUSE"
            _write(output / "campaign.json", record)
            os._exit(74)
        if llm is not None:
            try:
                llm.llm_engine.engine_core.shutdown()
            except Exception as error:
                record["cleanup_error"] = repr(error)
                _write(output / "campaign.json", record)
    if args.case == "A1":
        return (
            0
            if record.get("exception_type") == "NotImplementedError"
            and "GRAPH_FROZEN" in record.get("exception_message", "")
            else 2
        )
    return 0 if record["status"] == "OK" else 2


def trace_probe(output):
    """Bounded discovery only. Never equate installed/options with adequate fields."""
    report = {
        "status": "CAPABILITY_UNVERIFIED",
        "tools": {},
        "required_export_fields": [
            "device/task id",
            "stream id",
            "launch/update relation",
            "actual operand address",
            "descriptor shape/dtype",
            "completion",
            "physical block/token reads if available",
        ],
        "physical_read_note": "Memory/MemoryAccess names do not prove block read-set capability",
    }
    for name in ("msprof", "msprobe", "npu-smi"):
        executable = shutil.which(name)
        entry = {
            "installed": executable is not None,
            "path": executable,
            "capture_executed": False,
            "required_fields_exported": None,
            "sufficient_for_gate": None,
        }
        if executable:
            try:
                result = subprocess.run([executable, "--help"], capture_output=True, text=True, timeout=20)
                entry.update(help_exit=result.returncode, help_stdout=result.stdout, help_stderr=result.stderr)
            except (OSError, subprocess.TimeoutExpired) as error:
                entry["error"] = str(error)
        report["tools"][name] = entry
    _write(output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("artifact", type=Path)
    analyze.add_argument("--output", required=True, type=Path)
    probes = sub.add_parser("compare-probes")
    probes.add_argument("left", type=Path)
    probes.add_argument("right", type=Path)
    probes.add_argument("--output", required=True, type=Path)
    launch = sub.add_parser("launch")
    launch.add_argument("--output", type=Path, required=True)
    for name in ("run-id", "case-id", "attempt", "mode"):
        launch.add_argument(f"--{name}", required=True)
    launch.add_argument("--timeout", type=int, default=1800)
    launch.add_argument("argv", nargs=argparse.REMAINDER)
    check = sub.add_parser("check")
    check.add_argument("artifact", type=Path)
    check.add_argument("--output", required=True, type=Path)
    compare = sub.add_parser("compare-target")
    compare.add_argument("left", type=Path)
    compare.add_argument("right", type=Path)
    compare.add_argument("--output", required=True, type=Path)
    write = sub.add_parser("compare-write")
    write.add_argument("left", type=Path)
    write.add_argument("right", type=Path)
    write.add_argument("--output", required=True, type=Path)
    probe = sub.add_parser("trace-probe")
    probe.add_argument("--output", required=True, type=Path)
    fixtures = sub.add_parser("make-requests")
    fixtures.add_argument("--target-model", required=True)
    fixtures.add_argument("--output", required=True, type=Path)
    fixtures.add_argument("--profile", choices=("smoke", "boundaries", "combined"), default="combined")
    run = sub.add_parser("run")
    run.add_argument("--case", required=True, choices=("A0", "A1", "B", "B-observer-off"))
    run.add_argument("--case-id", help="actual experiment ID; --case remains the driver mode")
    for name in ("run-id", "attempt", "expected-sha", "target-model", "draft-model"):
        run.add_argument(f"--{name}", required=True)
    for name in ("vllm-repo", "fixture-manifest", "output"):
        run.add_argument(f"--{name}", required=True, type=Path)
    run.add_argument("--requests", type=Path)
    run.add_argument("--target-probe", action="store_true", help="extra D2H after capture; not for timing")
    run.add_argument(
        "--repair5-controlled-target",
        action="store_true",
        help="exclusive no-request engine; deterministic actual Target Graph/KV fixture",
    )
    run.add_argument(
        "--repair5-controlled-mrv2",
        action="store_true",
        help="also invoke actual speculator.propose from each controlled Target state",
    )
    run.add_argument(
        "--repair5-lifecycle",
        action="store_true",
        help="ordinary-event serialized diagnostic on real MRV2 path; observer ON required",
    )
    run.add_argument(
        "--repair5-abort-at",
        default="none",
        choices=("none", "before_replay", "after_replay", "after_update", "after_completion"),
    )
    run.add_argument(
        "--full-target-probe", action="store_true", help="actual target FULL forward/KV checkpoints after capture"
    )
    run.add_argument("--full-target-max-bytes", type=int, default=536870912, help="hard per-record CPU payload budget")
    run.add_argument(
        "--full-target-max-records", type=int, default=4, help="hard retained-record bound; omissions reported"
    )
    args = parser.parse_args()
    if args.command == "run":
        if args.repair5_controlled_mrv2 and not (
            args.repair5_controlled_target and (args.case == "A0" or args.repair5_lifecycle)
        ):
            parser.error("controlled MRV2 requires controlled Target; Window ON also requires lifecycle")
        if args.repair5_abort_at != "none" and not args.repair5_lifecycle:
            parser.error("abort requires lifecycle in disposable child supervised by launch")
    if args.command == "launch":
        return launch_case(args)
    if args.command == "compare-probes":
        result = compare_probe_sets(
            json.loads(args.left.read_text()), json.loads(args.right.read_text()), args.left.parent, args.right.parent
        )
        _write(args.output, result)
        return 0 if result["write_semantics"] == result["head_only_numeric"] == "PASS" else 2
    if args.command == "analyze":
        result = analyze_campaign(json.loads(args.artifact.read_text()))
        _write(args.output, result)
        return 0 if result["official_check"]["EVIDENCE_INTEGRITY"] == "PASS" else 2
    if args.command == "make-requests":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.target_model, local_files_only=True)
        unit = tokenizer.encode("Hello world. ", add_special_tokens=False)
        if not unit:
            raise RuntimeError("empty tokenizer fixture")
        requests = (
            []
            if args.profile == "boundaries"
            else [{"label": f"smoke-{n}", "prompts": ["Hello, my name is"] * n, "max_tokens": 16} for n in (1, 2, 4)]
        )
        for length in () if args.profile == "smoke" else (32, 1024, 1025, 1151, 1152, 1153, 2049, 32):
            tokens = (unit * ((length + len(unit) - 1) // len(unit)))[:length]
            requests.append(
                {
                    "label": f"prompt-length-{length}-attempt-{len(requests)}",
                    "prompts": [{"prompt_token_ids": tokens}],
                    "max_tokens": 16,
                }
            )
        _write(args.output, requests)
        return 0  # E/H are runtime facts, not inferred from these prompt lengths.
    if args.command == "run":
        return run_case(args)
    if args.command == "trace-probe":
        trace_probe(args.output)
        return 0  # discovery completed; capability remains unverified
    if args.command in ("compare-target", "compare-write"):
        compare_fn = compare_target_numeric if args.command == "compare-target" else compare_full_write
        result = compare_fn(json.loads(args.left.read_text()), json.loads(args.right.read_text()))
        _write(args.output, result)
        return 0 if result["status"] == "PASS" else 2
    try:
        result = audit_campaign(json.loads(args.artifact.read_text()))
    except (OSError, ValueError) as error:
        result = _invalid_evidence(f"cannot decode artifact: {error}")
    _write(args.output, result)
    return 0 if result["DEVICE_FUNCTIONAL_GATES"]["fixed_inventory_replay"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
