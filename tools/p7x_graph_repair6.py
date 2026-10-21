#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Repair6 bounded evidence comparison. Partial evidence is never campaign PASS."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __name__ == "__main__":
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).parent.resolve()]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from tools.p7x_graph_repair5 import load_tool
from tools.p7x_graph_repair6_storage import CollectionReader, atomic_json, require


def compare(left, right, *, timeout=1800, progress=None):
    """One sample pair at a time; mmap payloads, preserve entire declared plan."""
    started = time.monotonic()
    result = dict(
        status="INVALID_EVIDENCE",
        pairs=[],
        execution=[],
        numerical_scope="CONTROLLED_FULL_TARGET",
        completion="UNPROVEN",
        physical_read="OPEN",
        campaign="INCOMPLETE_COVERAGE",
    )
    try:
        a, b = CollectionReader(left), CollectionReader(right)
        result["execution"] = [a.execution, b.execution]
        tool = load_tool("p7x_graph_repair5_target")
        for reader in (a, b):
            expected = [tool.sample_identity(c, reader.header["fixture_id"]) for c in reader.header["cases"]]
            require(
                [s["identity"] for s in reader.samples] == expected[: len(reader.samples)], "not an ordered plan prefix"
            )
        # Validate header identities with the authoritative comparator, including
        # profile, source, inventory, tolerances. Never shorten its declared plan.
        base = tool.compare_collections(dict(a.header, records=[]), dict(b.header, records=[]))
        require(base["status"] == "OPEN", base.get("error", "invalid collection headers"))
        for index, case in enumerate(a.header["cases"]):
            require(time.monotonic() - started < timeout, "COMPARISON_TIMEOUT")
            identity = tool.sample_identity(case, a.header["fixture_id"])
            pair = dict(identity=identity, status="INCOMPLETE_COVERAGE")
            if index < min(len(a.samples), len(b.samples)):
                x, y = a.record(index), b.record(index)
                if x["status"] == y["status"] == "CAPTURED_COMPLETE":
                    partial = tool.compare_collections(dict(a.header, records=[x]), dict(b.header, records=[y]))
                    pair = next(p for p in partial["pairs"] if p["identity"] == identity)
                    if partial.get("error"):
                        pair.update(status="INVALID_EVIDENCE", error=partial["error"])
                    pair["metrics"] = {field: metrics(x[field], y[field]) for field in ("hidden", "logits")}
                else:
                    pair.update(status="EXECUTION_FAILED", sample_states=[x["status"], y["status"]])
                del x, y
            result["pairs"].append(pair)
            if progress:
                atomic_json(progress, dict(result, elapsed_seconds=time.monotonic() - started))
        statuses = [p["status"] for p in result["pairs"]]
        result["complete_pairs"] = statuses.count("PASS")
        result["status"] = (
            "NUMERIC_MISMATCH"
            if "FAIL" in statuses
            else "INVALID_EVIDENCE"
            if "INVALID_EVIDENCE" in statuses
            else "SAMPLE_COMPLETE"
            if statuses and all(s == "PASS" for s in statuses)
            else "ALIGNMENT_PRECONDITION_NOT_MET"
            if "ALIGNMENT_PRECONDITION_NOT_MET" in statuses
            else "PARTIAL_PREFIX_VALID"
            if statuses and statuses[0] == "PASS"
            else "INCOMPLETE_COVERAGE"
        )
        result["campaign"] = (
            "REQUIRES_EXTERNAL_EXECUTION_JOIN" if result["status"] == "SAMPLE_COMPLETE" else "INCOMPLETE_COVERAGE"
        )
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, IndexError) as error:
        result["error"] = str(error)
        result["status"] = "COMPARISON_TIMEOUT" if str(error) == "COMPARISON_TIMEOUT" else "INVALID_EVIDENCE"
        if any(p["status"] == "FAIL" for p in result["pairs"]):
            result["status"] = "NUMERIC_MISMATCH"
    result["elapsed_seconds"] = time.monotonic() - started
    return result


def metrics(x, y):
    require(x.shape == y.shape and x.dtype == y.dtype, "numeric geometry mismatch")
    result = dict(shape=list(x.shape), dtype=str(x.dtype), finite=True, max_abs=0.0, max_rel=0.0, atol=0.0, rtol=0.0)
    for start in range(0, x.numel(), 1 << 20):
        a, b = (t.reshape(-1)[start : start + (1 << 20)].float() for t in (x, y))
        finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        result["finite"] &= finite
        if finite:
            d = (a - b).abs()
            result["max_abs"] = max(result["max_abs"], float(d.max()))
            result["max_rel"] = max(result["max_rel"], float((d / b.abs().clamp_min(1e-12)).max()))
    return result


def joint(artifact, campaign, launcher, expected, kind):
    from tools.p7x_graph_repair5_evidence import audit_execution, audit_normal_requests

    # Execution identity is independent of numeric/sample validity. A valid
    # payload cannot authorize an unjoined, timed-out or failed device child.
    result = dict(status="INVALID_EVIDENCE", normal_reuse="UNPROVEN", effective_generation="UNPROVEN")
    try:
        if kind == "normal":
            document = json.loads(Path(artifact).read_text())
        else:
            reader = CollectionReader(artifact)
            document = reader.header
        result["execution"] = audit_execution(document, campaign, launcher, expected, artifact)
        if result["execution"]["status"] != "EXECUTION_ASSOCIATED":
            result["status"] = result["execution"]["status"]
            return result
        if kind == "normal":
            result["requests"] = audit_normal_requests(campaign, launcher, expected)
            require(result["requests"]["status"] == "PASS", "normal request execution incomplete")
            require(
                document["schema"] == "repair6-normal-order-v1" and document["completion"] is not None,
                "normal observation incomplete",
            )
            records = document["records"]
            require(document["status"] == "OBSERVED_NORMAL_ORDER_WITH_DEVICE_COPIES", "quarantined normal run")
            completion = document["completion"]
            require(
                completion["source"] == "ordinary_Event_final_campaign_drain"
                and all(type(completion[k]) is int and completion[k] > 0 for k in ("update_event", "consumer_event")),
                "missing final ordinary-event provenance",
            )
            previous = (-1, -1)
            backings = {}
            for r in records:
                require(r["generation"] > previous[0] and r["epoch"] > previous[1], "stale generation/epoch")
                previous = r["generation"], r["epoch"]
                require(
                    r["descriptor"] in ([1, 7], [2, 14], [4, 28]) and 0 < r["active"] <= r["descriptor"][0],
                    "wrong descriptor",
                )
                require(len(r["tokens"]) == r["active"] and all(len(t) == 7 for t in r["tokens"]), "incomplete tokens")
                require(all(type(t) is int and t >= 0 for row in r["tokens"] for t in row), "invalid token values")
                require(
                    r["retired"] and r["live_inputs_cleared"] and r["completion"] == "FINAL_OBSERVATION_COPY_DRAINED",
                    "retirement/copy incomplete",
                )
                require(
                    launcher["host_start_ns"]
                    <= r["start_ns"]
                    <= r["manager_return_ns"]
                    <= r["retirement_ns"]
                    <= launcher["host_end_ns"],
                    "time association",
                )
                require(
                    type(r["current_stream"]) is int and type(r["update_stream"]) is int, "missing stream provenance"
                )
                backing = r["backing"]["window_block_table"]
                require(backing["data_ptr"] > 0 and backing["shape"][0] == r["descriptor"][0], "bad backing geometry")
                require(type(r["backing"]["owner_id"]) is int and r["backing"]["owner_id"] > 0, "backing owner absent")
                for field in (
                    "window_block_table",
                    "valid_num_blocks",
                    "device_window_exact",
                    "cpu_window_exact",
                    "cpu_window_upper",
                ):
                    value = r["backing"][field]
                    require(
                        value["data_ptr"] > 0
                        and value["storage_data_ptr"] > 0
                        and value["dtype"]
                        and value["device"]
                        and value["shape"]
                        and all(type(n) is int and n > 0 for n in value["shape"])
                        and len(value["shape"]) == len(value["stride"])
                        and value["storage_offset"] >= 0,
                        "invalid full backing structure",
                    )
                    require(value["shape"][0] == r["descriptor"][0], "backing/descriptor row mismatch")
                key = tuple(r["descriptor"])
                require(key not in backings or backings[key] == (r["provider"], r["backing"]), "backing drift")
                backings[key] = (r["provider"], r["backing"])
            require(
                all(sum(r["descriptor"] == d for r in records) >= 2 for d in ([1, 7], [2, 14], [4, 28])),
                "normal three-bucket/multigeneration coverage incomplete",
            )
            result["status"] = "OBSERVED_NORMAL_ORDER_NOT_REUSE_PROOF"
        elif kind == "numeric":
            tool = load_tool("p7x_graph_repair5_target")
            expected_cases = tool.case_plan(expected.get("capture_profile", "base"))
            rows = [reader.record(i) for i in range(len(reader.samples))]
            require(0 < len(rows) <= len(expected_cases), "numeric sample coverage empty/extra")
            statuses = []
            previous = (-1, -1)
            for r, c in zip(rows, expected_cases):
                require(
                    r["case_identity"]["scenario"] == c["scenario"] and r["case_identity"]["step"] == c["step"],
                    "numeric plan identity mismatch",
                )
                require(r["generation"] > previous[0] and r["epoch"] > previous[1], "numeric stale generation")
                previous = r["generation"], r["epoch"]
                require(
                    r["identity"] == dict(document["identity"], generation=r["generation"], epoch=r["epoch"]),
                    "numeric source/generation mismatch",
                )
                require(
                    r["active"] == len(c["ends"]) and r["descriptor"] in ([1, 7], [2, 14], [4, 28]),
                    "numeric active/descriptor mismatch",
                )
                require(
                    r["descriptor"][0] == (1 if r["active"] == 1 else 2 if r["active"] == 2 else 4),
                    "Target padding mistaken for Draft padding",
                )
                require(r["pre_kv"].keys() == r["graph_post_kv"].keys() and bool(r["pre_kv"]), "missing KV pre-state")
                require(r["inputs"]["input_ids"].shape[0] == r["descriptor"][1], "numeric input coverage")
                n = r["active"] * 7
                measured = metrics(r["graph"][:n], r["reference"][:n])
                require(
                    r["graph"].shape == r["reference"].shape and r["graph"].shape[0] == r["descriptor"][1],
                    "numeric output bucket mismatch",
                )
                statuses.append("PASS" if measured["finite"] and measured["max_abs"] == 0 else "FAIL")
            result.update(
                status="NUMERIC_MISMATCH"
                if "FAIL" in statuses
                else "PASS_DIAGNOSTIC_CONTINUOUS_REFERENCE"
                if len(rows) == len(expected_cases)
                else "INCOMPLETE_COVERAGE",
                samples=statuses,
            )
            result["serialized_generation_discriminator"] = generation_discriminator(rows)
        else:
            tool = load_tool("p7x_graph_repair5_target")
            expected_cases = tool.case_plan(document.get("capture_profile", "base"))
            required = expected.get("prefix_samples") if kind == "target-prefix" else len(expected_cases)
            require(
                document["cases"] == expected_cases and len(reader.samples) == required,
                "Target sample plan incomplete",
            )
            rows = []
            for i, c in enumerate(expected_cases[:required]):
                r = reader.record(i)
                require(r["identity"] == tool.sample_identity(c, document["fixture_id"]), "Target sample identity")
                if r["status"] == "CAPTURED_COMPLETE":
                    tool.validate_controlled_record(r)
                    rows.append("SAMPLE_COMPLETE")
                else:
                    rows.append(r["status"])
            result.update(
                status=(
                    "PREFIX_STRUCTURE_VALID_NOT_CAMPAIGN_PASS"
                    if kind == "target-prefix"
                    else "SINGLE_ARM_STRUCTURE_VALID"
                )
                if all(v == "SAMPLE_COMPLETE" for v in rows)
                else "INCOMPLETE_COVERAGE",
                samples=rows,
            )
    except (KeyError, TypeError, ValueError, OSError, RuntimeError) as error:
        result["error"] = str(error)
    return result


def generation_discriminator(rows):
    """Observed numerical matching only, not a task-update or normal-overlap proof."""
    result = []
    for i, row in enumerate(rows):
        current = row["reference"][: row["active"] * 7]
        actual = row["graph"][: row["active"] * 7]
        candidates = {"current": bool(torch.equal(actual, current))}
        if i and rows[i - 1]["reference"].shape == row["reference"].shape:
            previous = rows[i - 1]["reference"][: row["active"] * 7]
            candidates["previous"] = bool(torch.equal(actual, previous))
            candidates["references_distinguishable"] = not torch.equal(current, previous)
        result.append(
            dict(
                generation=row["generation"],
                candidates=candidates,
                task_update_effective_generation="UNPROVEN",
                scope="DIAGNOSTIC_SERIALIZED_OUTPUT",
            )
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("compare-target")
    p.add_argument("--off", type=Path, required=True)
    p.add_argument("--on", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--timeout", type=float, default=1800)
    j = sub.add_parser("check-joint")
    j.add_argument("--kind", choices=("target", "target-prefix", "numeric", "normal"), required=True)
    for name in ("artifact", "campaign", "launcher", "plan", "output"):
        j.add_argument("--" + name, type=Path, required=True)
    j.add_argument("--case-id", required=True)
    b = sub.add_parser("target-budget")
    b.add_argument("--config", type=Path, required=True)
    b.add_argument("--profile", choices=("base", "target4"), default="base")
    b.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.command == "compare-target":
        result = compare(args.off, args.on, timeout=args.timeout, progress=args.output.with_suffix(".progress.json"))
    elif args.command == "check-joint":
        result = joint(
            args.artifact,
            json.loads(args.campaign.read_text()),
            json.loads(args.launcher.read_text()),
            json.loads(args.plan.read_text())["cases"][args.case_id],
            args.kind,
        )
    else:
        from tools.p7x_graph_repair6_storage import resource_plan

        c = json.loads(args.config.read_text())
        result = resource_plan(
            load_tool("p7x_graph_repair5_target").case_plan(args.profile),
            layers=c["num_hidden_layers"],
            kv_heads=c["num_key_value_heads"],
            head_dim=c.get("head_dim", c["hidden_size"] // c["num_attention_heads"]),
            element_bytes=2,
            hidden_size=c["hidden_size"],
            vocab_size=c["vocab_size"],
            mask_bytes=2048**2,
        )
        result.update(
            status="ESTIMATE_NOT_DEVICE_VERIFIED",
            dtype_assumption="2-byte FP16/BF16; runtime validates actual geometry",
            config=str(args.config),
        )
    atomic_json(args.output, result)
    return (
        0
        if result["status"]
        in (
            "SAMPLE_COMPLETE",
            "SINGLE_ARM_STRUCTURE_VALID",
            "PREFIX_STRUCTURE_VALID_NOT_CAMPAIGN_PASS",
            "PASS_DIAGNOSTIC_CONTINUOUS_REFERENCE",
            "OBSERVED_NORMAL_ORDER_NOT_REUSE_PROOF",
            "ESTIMATE_NOT_DEVICE_VERIFIED",
        )
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
