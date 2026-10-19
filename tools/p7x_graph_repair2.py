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


def audit_campaign(document: dict[str, Any]) -> dict[str, Any]:
    """Check integrity, then associations and independent math; never infer L3/L5/L6.

    The four result namespaces are deliberately separate. A failed-run archive
    can have valid integrity, while every functional gate remains NOT_RUN.
    """
    if not __debug__:
        raise RuntimeError("evidence checker must not run with Python -O (assertions disabled)")
    errors: list[str] = []
    missing: list[str] = []
    identity, snap = document.get("identity", {}), document.get("snapshot", {})
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
    bound = [e for e in events if e.get("stage") == "bound"]
    by_generation = {_generation(e): e for e in bound}
    if len(by_generation) != len(bound):
        errors.append("duplicate bound generation")
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
    expected_layers = document.get("draft_layer_names", [])
    if not expected_layers:
        missing.append("actual Draft layer inventory absent")
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
        if event.get("stage") == "update_operator" and _generation(event) not in by_generation:
            errors.append("orphan update operator")
        elif event.get("stage") == "update_operator":
            bound_record = by_generation[_generation(event)]
            if event.get("cpu_exact") != bound_record.get("exact_lengths"):
                errors.append("operator exact/generation lag")
            if event.get("length_kind") == "host_sequence" and event.get("length_values") != event.get("cpu_exact"):
                errors.append("actual operator length differs from typed exact")
    oracle_results = []
    for record in sections["g4_records"]:
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
            assert count > 0 and all(
                len(raw[k]) == count for k in ("full_upper", "valid_counts", "mapping", "request_ids")
            )
            assert raw["source_identity"]["input_block_table_ptrs"][0] == record["canonical_table_address"]
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
                actual = record["window_table_values"][row]
                assert actual[: expected.required_columns] == list(expected.expected_block_ids)
                assert all(v == 0 for v in actual[expected.required_columns :])
                source_event = by_generation[key]
                assert source_event["exact_lengths"][row] == expected.visible_length
                assert source_event["upper_lengths"][row] == expected.window_upper
                oracle_results.append(
                    {
                        "generation": key,
                        "row": row,
                        "S": expected.window_start,
                        "E": exact,
                        "V": expected.visible_length,
                        "columns": expected.expected_columns,
                    }
                )
        except (AssertionError, KeyError, TypeError, IndexError, ValueError, RuntimeError) as error:
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
    a, b, last = bound[-3:]
    source = {_generation(r): r.get("runtime_oracle_inputs") for r in full_records}
    raw = [source.get(_generation(r)) for r in (a, b, last)]
    if any(not r for r in raw):
        return "OPEN"
    if (
        len({r.get("generation_id") for r in (a, b, last)}) != 3
        or len({r.get("mapping_epoch") for r in (a, b, last)}) != 3
    ):
        return "FAIL"
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
        "case_id": args.case,
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
        "controlled_target_numeric": None,
        "task_trace": None,
        "physical_read_trace": None,
    }
    llm = session = target_probe = None
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
    try:
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
        # Fixed 1/2/4 requests are attempts; checker requires actual descriptor events.
        requests = (
            json.loads(args.requests.read_text())
            if args.requests
            else [{"label": f"smoke-{n}", "prompts": ["Hello, my name is"] * n, "max_tokens": 16} for n in (1, 2, 4)]
        )
        for request in requests:
            start = time.monotonic_ns()
            results = llm.generate(
                request["prompts"],
                SamplingParams(temperature=0.0, seed=0, ignore_eos=True, max_tokens=request["max_tokens"]),
                use_tqdm=False,
            )
            record["outputs"].append(
                {
                    "label": request["label"],
                    "host_start_ns": start,
                    "host_end_ns": time.monotonic_ns(),
                    "prompt_token_ids": [r.prompt_token_ids for r in results],
                    "token_ids": [r.outputs[0].token_ids for r in results],
                }
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
    run = sub.add_parser("run")
    run.add_argument("--case", required=True, choices=("A0", "A1", "B", "B-observer-off"))
    for name in ("run-id", "attempt", "expected-sha", "target-model", "draft-model"):
        run.add_argument(f"--{name}", required=True)
    for name in ("vllm-repo", "fixture-manifest", "output"):
        run.add_argument(f"--{name}", required=True, type=Path)
    run.add_argument("--requests", type=Path)
    run.add_argument("--target-probe", action="store_true", help="extra D2H after capture; not for timing")
    args = parser.parse_args()
    if args.command == "make-requests":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.target_model, local_files_only=True)
        unit = tokenizer.encode("Hello world. ", add_special_tokens=False)
        if not unit:
            raise RuntimeError("empty tokenizer fixture")
        requests = []
        for length in (32, 1024, 1025, 1151, 1152, 1153, 2049, 32):
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
    result = audit_campaign(json.loads(args.artifact.read_text()))
    _write(args.output, result)
    return 0 if result["DEVICE_FUNCTIONAL_GATES"]["fixed_inventory_replay"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
