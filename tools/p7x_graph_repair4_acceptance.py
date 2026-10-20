#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Authoritative Repair4 acceptance wrapper; stdlib only, no runtime producer.

Commands (JSON on stdout, optional --output; non-acceptance exits 2):
  check campaign.json [--run-id R --case-id B0 --attempt-id A]
  trace-fields export.csv-or-directory ... --run-id R --case-id T0|T1 --attempt-id A
  check-target off.json on.json

`check` accepts the Repair2 campaign schema. Repair2's audit remains a necessary
integrity/host check; the legacy presence-only snapshot audit is never used.
Only case_id=B0 is subject to the 1/2/4 normal-output acceptance requirements;
other cases return B0=NOT_APPLICABLE (exit 2), without running those requirements.
Null graph returns/hashes leave numeric gates OPEN, independently of B0 normal
generation success. Host retirement is not device completion evidence.

Controlled fixture schema `p7x-repair4-full-target-v1` (checker input only):
  identity: nonempty run_id/case_id/attempt_id strings; fixture_id; model_sha256;
  target_layer_names; selected_blocks: {layer: [nonnegative block IDs]};
  block_span_tokens; tolerance: {atol, rtol, reason}; nonempty ordered records.
Each record has a unique sample_id, entrypoint='model.forward', executed=true,
call_id=snapshot_call_id, snapshot_consistent=true, before_phase=
'target_forward_pre', after_phase='target_forward_post', before/after states,
aligned_input_state_sha256 (SHA256 of canonical JSON of before), forward_output.
Canonical JSON uses sort_keys=True, separators=(',', ':'), allow_nan=False.
States contain token_ids, positions, query_start_loc, sequence_lengths,
slot_mapping (integer tensors), unique request_ids, and target_layers. Each
layer has an integer block_table and kv: {str(block_id): {owner, version,
reuse_epoch, key, value}}. Every selected block is required before AND after;
selection must cover every visible/read/write block in both phases. This small
schema excludes shared-owner blocks. KV tensors include every token in each
block, with leading dimension block_span_tokens. Tensors are {shape, dtype,
values}, with nested finite JSON values and explicit int32/int64 or floating dtypes.
Before states align exactly; after metadata aligns exactly; after KV and forward
outputs use the predeclared tolerance. Missing hashes remain OPEN. A fixture
PASS is SUPPLIED_EVIDENCE_ONLY. See tools/p7x_graph_repair4_target.py for the
separate actual forward checkpoint producer and its checkpoint comparison CLI.
This JSON checker does not verify its live aligned pair: NOT_VERIFIED, G4F BLOCKED.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IDENTITY_KEYS = ("run_id", "case_id", "attempt_id")
REQUIRED_BUCKETS = ((1, 7), (2, 14), (4, 28))
INTEGER_DTYPES = ("int32", "int64")
FLOAT_DTYPES = ("float16", "bfloat16", "float32", "float64")


def _require(condition, reason):
    # Deliberately not assert: python -O must retain every evidence check.
    if not condition:
        raise ValueError(reason)


def _identity(value):
    _require(isinstance(value, dict), "identity must be an object")
    _require(
        all(isinstance(value.get(k), str) and value[k].strip() for k in IDENTITY_KEYS),
        "missing run/case/attempt identity",
    )
    return {k: value[k] for k in IDENTITY_KEYS}


def _repair2():
    spec = importlib.util.spec_from_file_location("_repair4_repair2", ROOT / "tools/p7x_graph_repair2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _producer_status():
    return {
        "G4F": "BLOCKED",
        "checker": "SUPPLIED_EVIDENCE_ONLY",
        "producer": "tools/p7x_graph_repair4_target.py",
        "live_aligned_pair": "NOT_VERIFIED",
        "reason": "See target.py for the actual forward checkpoint producer; live aligned pair NOT_VERIFIED here.",
    }


def _device_levels():
    return dict.fromkeys(
        ("operator_boundary", "task_internal", "physical_read", "completion_and_effective_generation"), "OPEN"
    )


def _sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _normal_outputs(document):
    _require(
        document.get("status") == "OK" and document.get("startup_complete") is True,
        "normal generation did not complete successfully",
    )
    outputs = document.get("outputs")
    _require(isinstance(outputs, list) and outputs, "missing/null normal outputs")
    counts, labels = set(), set()
    for output in outputs:
        _require(isinstance(output, dict), "invalid normal output")
        label = output.get("label")
        _require(isinstance(label, str) and label and label not in labels, "missing/duplicate output label")
        labels.add(label)
        start, end = output.get("host_start_ns"), output.get("host_end_ns")
        _require(type(start) is int and type(end) is int and 0 <= start < end, "invalid output interval")
        arrays = [output.get(k) for k in ("prompt_token_ids", "token_ids")]
        for array in arrays:
            _require(isinstance(array, list) and array, "missing/null output token evidence")
            _require(
                all(isinstance(row, list) and row and all(type(t) is int and t >= 0 for t in row) for row in array),
                "invalid/empty output token evidence",
            )
        _require(len(arrays[0]) == len(arrays[1]), "prompt/output cardinality mismatch")
        counts.add(len(arrays[0]))
    _require({1, 2, 4} <= counts, "normal output request counts 1/2/4 incomplete")
    return {"status": "PASS", "batches": len(outputs), "request_counts": sorted(counts)}


def _runtime_buckets(document):
    counts = Counter()
    for event in document["snapshot"]["events"]:
        if event.get("stage") != "replay":
            continue
        descriptor = event.get("capture_descriptor")
        _require(isinstance(descriptor, str), "missing replay descriptor")
        fields = {}
        for item in descriptor.split(";"):
            key, separator, value = item.partition("=")
            _require(separator and key and value and key not in fields, "ambiguous replay descriptor")
            fields[key] = value
        _require(fields.get("mode") == "FULL" and fields.get("uniform") == "7", "wrong actual replay mode/width")
        bucket = next(
            (b for b in REQUIRED_BUCKETS if fields.get("reqs") == str(b[0]) and fields.get("tokens") == str(b[1])), None
        )
        _require(bucket is not None, "unexpected actual replay bucket")
        counts[bucket] += 1
    _require(set(counts) == set(REQUIRED_BUCKETS), "actual runtime replay buckets 1,7 / 2,14 / 4,28 incomplete")
    return {"status": "PASS", "replay_counts": {f"{n},{tokens}": counts[n, tokens] for n, tokens in REQUIRED_BUCKETS}}


def _retirement(document):
    snap = document["snapshot"]
    bounds = [r for r in snap["events"] if r.get("stage") == "bound"]
    generations = {(r["generation_id"], r["mapping_epoch"]) for r in bounds}
    _require(len(generations) == len(bounds) and len(generations) >= 2, "multiple unique runtime generations required")
    layers = document.get("draft_layer_names")
    _require(
        isinstance(layers, list)
        and layers
        and all(isinstance(v, str) and v for v in layers)
        and len(set(layers)) == len(layers),
        "invalid actual Draft layer inventory",
    )
    lifecycle = snap["lifecycle_events"]
    for event in lifecycle:
        _require((event.get("generation_id"), event.get("mapping_epoch")) in generations, "orphan lifecycle generation")
    for generation, epoch in sorted(generations):

        def matching(records, generation=generation, epoch=epoch):
            return [r for r in records if (r.get("generation_id"), r.get("mapping_epoch")) == (generation, epoch)]

        events = matching(snap["events"])
        operands = [r for r in matching(snap["operand_events"]) if r.get("stage") == "update_operator"]
        life = matching(lifecycle)
        _require(
            Counter(r.get("layer_name") for r in operands) == Counter(layers),
            f"incomplete/duplicate per-layer UPDATE for generation {generation}",
        )
        submitted = [r for r in life if r.get("stage") == "replay_submitted"]
        retired = [r for r in life if r.get("stage") == "proposal_retired"]
        updates = [r for r in life if r.get("stage") == "update_event_recorded"]
        _require(
            len(submitted) == len(retired) == 1 and len(updates) == len(operands),
            f"incomplete/duplicate retirement timeline for generation {generation}",
        )
        replay = [r for r in events if r.get("stage") == "replay"]
        _require(
            len(replay) == 1 and replay[0]["sequence"] < submitted[0]["sequence"],
            "submission must follow actual replay",
        )
        used = set()
        for operand in operands:
            event_id, stream = operand.get("event_identity"), operand.get("stream_identity")
            _require(
                type(event_id) is int and event_id > 0 and type(stream) is int and stream > 0,
                "UPDATE event/stream association missing",
            )
            matches = [
                r
                for r in updates
                if r.get("event_identity") == event_id
                and r.get("stream_identity") == stream
                and operand["sequence"] < r["sequence"]
            ]
            _require(len(matches) == 1 and matches[0]["sequence"] not in used, "UPDATE event association mismatch")
            _require(submitted[0]["sequence"] < operand["sequence"], "UPDATE must follow replay submission")
            used.add(matches[0]["sequence"])
        _require(retired[0].get("lease_state") == "retired", "proposal lease was not retired")
        _require(
            retired[0]["sequence"] > max(r["sequence"] for r in [*events, *operands, *updates, *submitted]),
            "retirement precedes replay/UPDATE",
        )
    return {"status": "PASS", "generations": len(generations), "scope": "host proposal retirement only"}


def audit_campaign(document, expected_identity=None):
    """Authoritative B0 decision; retain Repair2 checks without inheriting PASS."""
    result = {
        "schema": "p7x-repair4-acceptance-v1",
        "identity": None,
        "EVIDENCE_INTEGRITY": "INVALID_EVIDENCE",
        "HOST_CONTRACTS": "OPEN",
        "DEVICE_FUNCTIONAL_GATES": {"B0": "FAIL"},
        "DEVICE_EVIDENCE_LEVELS": _device_levels(),
        "NUMERIC_GATES": {"draft_output": "OPEN", "target_output": "OPEN", "G4_full_target": "OPEN"},
        "PRODUCER_STATUS": _producer_status(),
        "OUTPUT_EVIDENCE": {},
        "HOST_OBSERVATIONS": {},
        "errors": [],
        "missing": [],
    }
    try:
        _require(isinstance(document, dict), "campaign must be an object")
        identity = _identity(document.get("identity"))
        result["identity"] = identity
        if expected_identity is not None:
            _require(identity == _identity(expected_identity), "unexpected run/case/attempt identity")
        legacy = _repair2().audit_campaign(document)
        result["repair2_audit"] = legacy
        result["EVIDENCE_INTEGRITY"] = legacy["EVIDENCE_INTEGRITY"]
        result["HOST_CONTRACTS"] = legacy["HOST_CONTRACTS"]
        result["errors"].extend(legacy["errors"])
        result["missing"].extend(legacy["missing"])
        result["DEVICE_EVIDENCE_LEVELS"]["operator_boundary"] = legacy["DEVICE_EVIDENCE_LEVELS"]["operator_boundary"]
        for name in ("actual_replay_buckets", "required_replay_buckets", "missing_replay_buckets"):
            result[name] = legacy.get(name, [])
        snap = document.get("snapshot")
        _require(isinstance(snap, dict), "snapshot must be an object")
        records = snap.get("g4_records")
        _require(isinstance(records, list) and all(isinstance(r, dict) for r in records), "invalid G4 records")
        for record in records:
            for key in (
                "output_status",
                "output_reason",
                "draft_output_status",
                "draft_output_reason",
                "target_output_status",
                "target_output_reason",
            ):
                _require(record.get(key) is None or isinstance(record[key], str), f"invalid {key}")
        for kind in ("draft", "target"):
            present = sum(_sha256(r.get(f"{kind}_output_hash")) for r in records)
            result["OUTPUT_EVIDENCE"][f"{kind}_hashes"] = "PRESENT" if records and present == len(records) else "OPEN"
            result["OUTPUT_EVIDENCE"][f"{kind}_hash_count"] = present
        result["OUTPUT_EVIDENCE"]["records"] = [
            {
                k: r.get(k)
                for k in (
                    "generation_id",
                    "mapping_epoch",
                    "output_status",
                    "output_reason",
                    "draft_output_status",
                    "draft_output_reason",
                    "target_output_status",
                    "target_output_reason",
                )
            }
            for r in records
        ]
        result["OUTPUT_EVIDENCE"]["numeric_reason"] = (
            "Missing/null hashes remain OPEN; even present hashes alone do not establish aligned numeric comparison."
        )
        if identity["case_id"] != "B0":
            result["DEVICE_FUNCTIONAL_GATES"]["B0"] = "NOT_APPLICABLE"
            return result
        for name, check in (
            ("normal_outputs", _normal_outputs),
            ("runtime_replays", _runtime_buckets),
            ("retirement", _retirement),
        ):
            try:
                result["HOST_OBSERVATIONS"][name] = check(document)
            except (KeyError, TypeError, ValueError, IndexError, AttributeError) as error:
                result["HOST_OBSERVATIONS"][name] = {"status": "FAIL", "reason": str(error)}
                result["errors"].append(str(error))
        if (
            not result["errors"]
            and legacy["HOST_CONTRACTS"] == "PASS"
            and legacy["DEVICE_FUNCTIONAL_GATES"]["fixed_inventory_replay"] == "PASS"
        ):
            result["DEVICE_FUNCTIONAL_GATES"]["B0"] = "PASS"
    except (KeyError, TypeError, ValueError, IndexError, AttributeError, OverflowError) as error:
        result["EVIDENCE_INTEGRITY"] = "INVALID_EVIDENCE"
        result["errors"].append(str(error))
    return result


def trace_fields(paths, identity):
    """Inventory actual CSV exports for ONE trace case; never infer device proof."""
    result = {
        "schema": "p7x-repair4-trace-fields-v1",
        "identity": None,
        "status": "OPEN",
        "files": [],
        "DEVICE_EVIDENCE_LEVELS": _device_levels(),
        "errors": [],
        "identity_binding": "CALLER_SUPPLIED; CSV exports do not establish campaign identity",
        "scope": "CSV field availability only; T0 and T1 are separate; shapes are not operands or physical reads",
    }
    try:
        result["identity"] = _identity(identity)
        _require(identity["case_id"] in ("T0", "T1"), "trace case must be explicitly T0 or T1")
        files = []
        for item in paths:
            path = Path(item)
            files.extend(sorted(path.rglob("*.csv")) if path.is_dir() else [path])
        _require(files, "no exported CSV files")
        _require(len({p.resolve() for p in files}) == len(files), "duplicate CSV path")
        for path in files:
            with path.open(newline="", encoding="utf-8-sig") as stream:
                reader = csv.reader(stream, strict=True)
                headers = next(reader, [])
                _require(
                    headers and all(h.strip() for h in headers) and len(set(headers)) == len(headers),
                    f"missing/duplicate CSV headers: {path}",
                )
                counts = dict.fromkeys(headers, 0)
                rows = attention = 0
                for row in reader:
                    if not row:
                        continue
                    _require(len(row) == len(headers), f"CSV row width mismatch: {path}:{reader.line_num}")
                    rows += 1
                    for field, value in zip(headers, row):
                        counts[field] += bool(value.strip())
                    named = dict(zip(headers, row))
                    attention += any("attention" in named.get(k, "").lower() for k in ("Op Name", "OP Type", "Op Type"))
            result["files"].append(
                {
                    "path": str(path.resolve()),
                    "headers": headers,
                    "row_count": rows,
                    "nonempty_counts": counts,
                    "attention_rows": attention,
                }
            )
        result["status"] = "PASS" if sum(f["row_count"] for f in result["files"]) else "OPEN"
    except (OSError, UnicodeError, csv.Error, TypeError, ValueError) as error:
        result["status"] = "INVALID_EVIDENCE"
        result["errors"].append(str(error))
    return result


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _tensor(value, integer=False):
    _require(isinstance(value, dict), "missing tensor")
    shape, dtype = value.get("shape"), value.get("dtype")
    _require(isinstance(shape, list) and all(type(n) is int and n > 0 for n in shape), "invalid/empty tensor shape")
    _require(dtype in (INTEGER_DTYPES if integer else INTEGER_DTYPES + FLOAT_DTYPES), "unsupported tensor dtype")
    flat = []

    def visit(data, dims):
        if dims:
            _require(isinstance(data, list) and len(data) == dims[0], "tensor shape/value mismatch")
            for child in data:
                visit(child, dims[1:])
        else:
            _require(type(data) in (int, float) and math.isfinite(data), "nonfinite/nonnumeric tensor")
            if dtype in INTEGER_DTYPES:
                bits = 32 if dtype == "int32" else 64
                _require(type(data) is int and -(2 ** (bits - 1)) <= data < 2 ** (bits - 1), "invalid integer tensor")
            flat.append(data)

    visit(value.get("values"), shape)
    return flat


def _state(state, document):
    _require(isinstance(state, dict), "missing before/after state")
    names = ("token_ids", "positions", "query_start_loc", "sequence_lengths", "slot_mapping")
    data = {name: _tensor(state.get(name), integer=True) for name in names}
    _require(all(len(state[name]["shape"]) == 1 for name in names), "state layout tensors must be vectors")
    requests = state.get("request_ids")
    _require(
        isinstance(requests, list)
        and requests
        and all(isinstance(r, str) and r for r in requests)
        and len(set(requests)) == len(requests),
        "invalid request layout",
    )
    n, tokens = len(requests), len(data["token_ids"])
    _require(len(data["positions"]) == len(data["slot_mapping"]) == tokens, "token/position/slot count mismatch")
    _require(all(x >= 0 for name in names for x in data[name]), "negative state value")
    query = data["query_start_loc"]
    lengths = data["sequence_lengths"]
    _require(
        len(query) == n + 1 and query[0] == 0 and query[-1] == tokens and all(a < b for a, b in zip(query, query[1:])),
        "invalid query/request alignment",
    )
    _require(len(lengths) == n and all(v > 0 for v in lengths), "invalid request lengths")
    layers = state.get("target_layers")
    _require(
        isinstance(layers, dict) and set(layers) == set(document["target_layer_names"]), "incomplete Target layers"
    )
    span = document["block_span_tokens"]
    covered = {}
    for name, layer in layers.items():
        _require(isinstance(layer, dict), "invalid layer state")
        table, kv = layer.get("block_table"), layer.get("kv")
        table_values = _tensor(table, integer=True)
        _require(
            len(table["shape"]) == 2 and table["shape"][0] == n and all(v >= 0 for v in table_values),
            "invalid block table geometry",
        )
        selected = document["selected_blocks"][name]
        _require(isinstance(kv, dict) and set(kv) == {str(b) for b in selected}, "incomplete selected KV blocks")
        for block in kv.values():
            _require(isinstance(block, dict) and block.get("owner") in requests, "missing KV ownership")
            _require(
                all(type(block.get(k)) is int and block[k] >= 0 for k in ("version", "reuse_epoch")),
                "missing KV version/reuse state",
            )
            _tensor(block.get("key"))
            _tensor(block.get("value"))
            _require(
                block["key"]["shape"] == block["value"]["shape"] and block["key"]["dtype"] == block["value"]["dtype"],
                "KV key/value geometry mismatch",
            )
            _require(
                block["key"]["shape"] and block["key"]["shape"][0] == span, "KV tensor must contain a complete block"
            )
        visible = set()
        for row, request in enumerate(requests):
            needed = (lengths[row] + span - 1) // span
            _require(needed <= table["shape"][1], "block table does not cover history")
            blocks = table["values"][row][:needed]
            _require(set(blocks) <= set(selected), "selected blocks do not cover visible history")
            _require(all(kv[str(b)]["owner"] == request for b in blocks), "KV owner/request mismatch")
            visible.update(blocks)
            for index in range(query[row], query[row + 1]):
                position = data["positions"][index]
                _require(position < lengths[row], "position outside request history")
                block = table["values"][row][position // span]
                _require(
                    data["slot_mapping"][index] == block * span + position % span,
                    "slot/position/block mapping mismatch",
                )
        covered[name] = visible
    return covered


def _target_document(document):
    _require(
        isinstance(document, dict) and document.get("schema") == "p7x-repair4-full-target-v1",
        "unsupported controlled Target schema",
    )
    _identity(document.get("identity"))
    _require(isinstance(document.get("fixture_id"), str) and document["fixture_id"], "missing fixture identity")
    _require(_sha256(document.get("model_sha256")), "missing model hash")
    layers, selection = document.get("target_layer_names"), document.get("selected_blocks")
    _require(
        isinstance(layers, list)
        and layers
        and all(isinstance(v, str) and v for v in layers)
        and len(set(layers)) == len(layers),
        "invalid Target layer inventory",
    )
    _require(isinstance(selection, dict) and set(selection) == set(layers), "missing selected-block inventory")
    for blocks in selection.values():
        _require(
            isinstance(blocks, list)
            and blocks
            and all(type(b) is int and b >= 0 for b in blocks)
            and len(set(blocks)) == len(blocks),
            "invalid selected blocks",
        )
    _require(type(document.get("block_span_tokens")) is int and document["block_span_tokens"] > 0, "invalid block span")
    tolerance = document.get("tolerance")
    _require(
        isinstance(tolerance, dict) and isinstance(tolerance.get("reason"), str) and tolerance["reason"].strip(),
        "missing tolerance reason",
    )
    _require(
        all(
            type(tolerance.get(k)) in (int, float) and math.isfinite(tolerance[k]) and tolerance[k] >= 0
            for k in ("atol", "rtol")
        ),
        "invalid/nonfinite tolerance",
    )
    records = document.get("records")
    _require(isinstance(records, list) and records, "missing actual forward records")
    ids, calls, missing_hash = set(), set(), False
    for record in records:
        _require(isinstance(record, dict), "invalid Target record")
        sample, call = record.get("sample_id"), record.get("call_id")
        _require(isinstance(sample, str) and sample and sample not in ids, "missing/duplicate sample identity")
        _require(isinstance(call, str) and call and call not in calls, "missing/duplicate forward call identity")
        ids.add(sample)
        calls.add(call)
        _require(
            record.get("entrypoint") == "model.forward" and record.get("executed") is True,
            "actual model.forward required; compute_logits is head-only",
        )
        _require(
            record.get("before_phase") == "target_forward_pre" and record.get("after_phase") == "target_forward_post",
            "wrong Target phase",
        )
        _require(
            record.get("snapshot_call_id") == call and record.get("snapshot_consistent") is True,
            "snapshot-to-forward association absent/unsafe",
        )
        before, after = _state(record.get("before"), document), _state(record.get("after"), document)
        for field in ("token_ids", "positions", "query_start_loc", "request_ids", "sequence_lengths", "slot_mapping"):
            _require(
                record["before"][field] == record["after"][field], "forward input layout changed between snapshots"
            )
        _require(
            all(before[layer] | after[layer] == set(selection[layer]) for layer in layers),
            "selected blocks must equal complete before/after coverage",
        )
        for layer in layers:
            for block in selection[layer]:
                a, b = [record[phase]["target_layers"][layer]["kv"][str(block)] for phase in ("before", "after")]
                _require(
                    a["owner"] == b["owner"] and a["reuse_epoch"] == b["reuse_epoch"] and a["version"] <= b["version"],
                    "KV ownership/reuse/version changed during forward",
                )
                for kind in ("key", "value"):
                    _require(
                        a[kind]["shape"] == b[kind]["shape"] and a[kind]["dtype"] == b[kind]["dtype"],
                        "KV tensor geometry changed during forward",
                    )
        _tensor(record.get("forward_output"))
        digest = record.get("aligned_input_state_sha256")
        if digest in (None, ""):
            missing_hash = True
        else:
            _require(_sha256(digest) and digest == _digest(record["before"]), "aligned input hash mismatch")
    return missing_hash


def _numeric_tensor(left, right, tolerance):
    a, b = _tensor(left), _tensor(right)
    if left["shape"] != right["shape"] or left["dtype"] != right["dtype"]:
        return False
    atol, rtol = tolerance["atol"], tolerance["rtol"]
    limits = [atol + rtol * abs(y) for y in b]
    return all(math.isfinite(limit) and abs(x - y) <= limit for x, y, limit in zip(a, b, limits))


def compare_full_target(left, right):
    """Validate supplied fixtures, with producer availability reported separately."""
    result = {
        "schema": "p7x-repair4-target-comparison-v1",
        "status": "OPEN",
        **_producer_status(),
        "scope": "SUPPLIED_EVIDENCE_ONLY; live aligned pair NOT_VERIFIED; no device acceptance",
        "errors": [],
    }
    try:
        missing_left, missing_right = _target_document(left), _target_document(right)
        result["identities"] = [_identity(d["identity"]) for d in (left, right)]
        _require(result["identities"][0] != result["identities"][1], "same run/case/attempt cannot be both sides")
        for key in (
            "fixture_id",
            "model_sha256",
            "target_layer_names",
            "selected_blocks",
            "block_span_tokens",
            "tolerance",
        ):
            _require(left[key] == right[key], f"unaligned {key}")
        a, b = left["records"], right["records"]
        _require([r["sample_id"] for r in a] == [r["sample_id"] for r in b], "sample cardinality/order mismatch")
        if missing_left or missing_right:
            result["reason"] = "missing aligned input hash; numeric completeness remains OPEN"
            return result
        result["status"] = "PASS"
        for x, y in zip(a, b):
            _require(
                x["before"] == y["before"], "before token/position/query/request/KV states are not exactly aligned"
            )
            if not _numeric_tensor(x["forward_output"], y["forward_output"], left["tolerance"]):
                result["status"] = "FAIL"
                result["errors"].append(f"forward output differs: {x['sample_id']}")
            xs, ys = x["after"], y["after"]
            _require(
                {k: v for k, v in xs.items() if k != "target_layers"}
                == {k: v for k, v in ys.items() if k != "target_layers"},
                "after metadata mismatch",
            )
            for layer in left["target_layer_names"]:
                xl, yl = xs["target_layers"][layer], ys["target_layers"][layer]
                _require(xl["block_table"] == yl["block_table"], "after block tables differ")
                for block in xl["kv"]:
                    p, q = xl["kv"][block], yl["kv"][block]
                    _require(
                        all(p[k] == q[k] for k in ("owner", "version", "reuse_epoch")), "after KV lineage mismatch"
                    )
                    if not all(_numeric_tensor(p[k], q[k], left["tolerance"]) for k in ("key", "value")):
                        result["status"] = "FAIL"
                        result["errors"].append(f"after KV differs: {x['sample_id']} {layer} block {block}")
        result["pairs"] = len(a)
    except (KeyError, TypeError, ValueError, IndexError, AttributeError, OverflowError) as error:
        result["status"] = "INVALID_EVIDENCE"
        result["errors"].append(str(error))
    return result


def _read_document(path):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    return json.loads(path.read_text(), object_pairs_hook=unique_object, parse_constant=invalid_constant)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="authoritative B0 acceptance using Repair2 integrity/math checks")
    check.add_argument("campaign", type=Path)
    trace = sub.add_parser("trace-fields", help="inspect actual exported CSV headers and row counts for T0 OR T1")
    trace.add_argument("paths", nargs="+", type=Path)
    target = sub.add_parser("check-target", help="SUPPLIED_EVIDENCE_ONLY; live aligned pair NOT_VERIFIED, G4F BLOCKED")
    target.add_argument("left", type=Path)
    target.add_argument("right", type=Path)
    for command in (check, trace):
        for key in IDENTITY_KEYS:
            command.add_argument("--" + key.replace("_", "-"), required=command is trace)
    for command in (check, trace, target):
        command.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            expected = {k: getattr(args, k) for k in IDENTITY_KEYS}
            result = audit_campaign(_read_document(args.campaign), expected if any(expected.values()) else None)
            passed = result["DEVICE_FUNCTIONAL_GATES"]["B0"] == "PASS"
        elif args.command == "trace-fields":
            result = trace_fields(args.paths, {k: getattr(args, k) for k in IDENTITY_KEYS})
            passed = result["status"] == "PASS"
        else:
            result = compare_full_target(_read_document(args.left), _read_document(args.right))
            passed = result["status"] == "PASS"
    except (OSError, ValueError) as error:
        result, passed = (
            {
                "status": "INVALID_EVIDENCE",
                "errors": [str(error)],
                "PRODUCER_STATUS": _producer_status(),
            },
            False,
        )
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
