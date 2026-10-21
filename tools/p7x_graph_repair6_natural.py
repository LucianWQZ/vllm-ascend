#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Unique logical HEAD_ONLY pairing. Legacy lacks business identity: stays OPEN.

The separate ordinal audit reproduces historical comparisons without treating
indices or matching hidden hashes as proof of full Target/KV alignment.
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

if __name__ == "__main__":
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).parent.resolve()]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from tools.p7x_graph_repair6 import metrics, verdict_exit
from tools.p7x_graph_repair6_storage import atomic_json, digest, require


def records(root):
    doc = json.loads((root / "target-boundary.json").read_text())
    out = []
    for ordinal, r in enumerate(x for x in doc["records"] if x["kind"] == "target_logits_head"):
        path = root / Path(r["path"]).name
        value = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
        hidden = value["hidden_states"].contiguous()
        require(
            hashlib.sha256(memoryview(hidden.view(torch.uint8).numpy())).hexdigest() == r["input_sha256"],
            "raw hidden hash mismatch",
        )
        out.append(dict(ordinal=ordinal, record=r, values=value, path=str(path), sha256=digest(path)))
    return out, doc["errors"]


def compare(a_root, b_root):
    a, ae = records(a_root)
    b, be = records(b_root)
    indices = []
    missing = []
    for arm, rows in (("OFF", a), ("ON", b)):
        index = defaultdict(list)
        for r in rows:
            identity = r["record"].get("logical_identity")
            if not identity or not identity.get("request_group"):
                missing.append(dict(arm=arm, ordinal=r["ordinal"], path=r["path"], reason="NO_BUSINESS_IDENTITY"))
            else:
                index[json.dumps(identity, sort_keys=True)].append(r)
        indices.append(index)
    pairs = []
    for key in sorted(indices[0].keys() | indices[1].keys()):
        x, y = indices[0].get(key, []), indices[1].get(key, [])
        row = dict(identity=json.loads(key), counts=[len(x), len(y)], status="UNPAIRED_OR_AMBIGUOUS")
        if len(x) == len(y) == 1:
            row.update(measure(x[0], y[0]))
        pairs.append(row)
    ordinal = []
    for i in range(max(len(a), len(b))):
        row = dict(ordinal=i, status="UNPAIRED")
        if i < min(len(a), len(b)):
            row.update(measure(a[i], b[i]))
        ordinal.append(row)
    return dict(
        schema="repair6-natural-head-association-v1",
        counts=[len(a), len(b)],
        pairs=pairs,
        missing_identity=missing,
        producer_errors=[ae, be],
        historical_ordinal_audit=ordinal,
        samples=[
            dict(
                arm=arm,
                ordinal=r["ordinal"],
                path=r["path"],
                sha256=r["sha256"],
                logical_identity=r["record"].get("logical_identity"),
            )
            for arm, rows in (("OFF", a), ("ON", b))
            for r in rows
        ],
        status="NUMERIC_MISMATCH"
        if any(r["status"] == "ALIGNED_HEAD_NUMERIC_FAIL" for r in pairs)
        else "OPEN_FULL_KV_PRESTATE_NOT_CAPTURED",
        full_target="NOT_PROVEN_USE_CONTROLLED_FIXTURE",
        original_verdict="PRESERVED_NOT_OVERWRITTEN",
    )


def measure(x, y):
    a, b = x["values"], y["values"]
    same = (
        a["hidden_states"].shape == b["hidden_states"].shape
        and a["hidden_states"].dtype == b["hidden_states"].dtype
        and torch.equal(a["hidden_states"], b["hidden_states"])
    )
    row = dict(
        status="ALIGNMENT_PRECONDITION_NOT_MET",
        input_equal=same,
        full_kv_pre_equal=None,
        paths=[x["path"], y["path"]],
        hashes=[x["sha256"], y["sha256"]],
    )
    if same:
        m = metrics(a["logits"], b["logits"])
        row.update(
            metrics=m,
            status="ALIGNED_HEAD_NUMERIC_PASS" if m["finite"] and m["max_abs"] == 0 else "ALIGNED_HEAD_NUMERIC_FAIL",
        )
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("off", "on", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    try:
        result = compare(a.off, a.on)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        result = dict(status="INVALID_EVIDENCE", error=str(error))
    atomic_json(a.output, result)
    return verdict_exit(result["status"])


if __name__ == "__main__":
    raise SystemExit(main())
