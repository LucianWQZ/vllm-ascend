# SPDX-License-Identifier: Apache-2.0
"""Exclusive, disposable-engine Target Graph fixture (not natural request replay).

Installed only immediately after LLM initialization, before any request enters
the scheduler. It uses the actual captured Target buffers and ModelAclGraphManager,
never swaps graph-bound addresses. It stages deterministic synthetic KV (not
prefill-derived KV), calls the complete Target Graph, and snapshots every mapped
read/write block. Every unselected block is also streamed through a digest before
and after the call to detect unauthorized writes without storing the entire HBM.

The process MUST exit after the fixture: no allocator/request is allowed to use
its diagnostic contents. No restore of partial live request state is attempted.
OFF and ON are independent disposable engines, with different physical block
placement but the same logical state. This proves only the declared controlled
states, not arbitrary model histories or unsynchronized serving.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCHEMA = "repair5-controlled-target-v1"
BLOCK = 128
ATOL = 0.0
RTOL = 0.0


def _base():
    spec = importlib.util.spec_from_file_location(
        "_repair5_target_base", Path(__file__).with_name("p7x_graph_repair4_target.py")
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("target checkpoint tool missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def case_plan(profile="base") -> list[dict]:
    # Query width is TARGET's K+1=8, not Draft's seven. No bucket is manufactured.
    require(profile in ("base", "target4"), "unknown capture profile")
    cases = [
        {"scenario": name, "step": i, "ends": ends, "padded": padded, "query_width": 8}
        for i, (name, ends, padded) in enumerate(
            (
                ("A", [1152], 1),
                ("B", [1280], 1),
                ("A_prime", [1152], 1),
                ("short", [32], 1),
                ("long", [1152], 1),
                ("boundary", [1024], 1),
                ("nonaligned", [1153], 1),
                ("mixed", [32, 1153], 2),
                ("padding", [32], 2),
                ("return-short", [32], 1),
                ("three-active-four-padded", [32, 1025, 1153], 4),
            )
        )
    ]
    if profile == "target4":
        cases.append(
            dict(scenario="four-active", step=len(cases), ends=[32, 1024, 1152, 1153], padded=4, query_width=8)
        )
    return cases


def sample_identity(case: dict, fixture_id: str) -> dict:
    return {
        "scenario": case["scenario"],
        "logical_requests": [f"row-{r}" for r in range(len(case["ends"]))],
        "phase": "CONTROLLED_DECODE_SYNTHETIC_PRE_KV",
        "positions": [list(range(e - case["query_width"], e)) for e in case["ends"]],
        "query_layout": [case["query_width"] * i for i in range(case["padded"] + 1)],
        "active": len(case["ends"]),
        "padded": case["padded"],
        "step": case["step"],
        "fixture_id": fixture_id,
    }


def outside_digest(caches: dict, selected: dict) -> dict:
    """Read every outside block, bounded temporary CPU memory; diagnostic only."""
    result = {}
    for name, tensor in caches.items():
        included = set(selected[name].tolist())
        digest = hashlib.sha256()
        # Preserve the v1 byte stream exactly: ascending unselected blocks.
        # One bounded contiguous D2H per chunk instead of one per block.
        chunk_blocks = max(1, (8 * 1024 * 1024) // (tensor[0].numel() * tensor.element_size()))
        for start in range(0, tensor.shape[0], chunk_blocks):
            stop = min(start + chunk_blocks, tensor.shape[0])
            if all(b in included for b in range(start, stop)):
                continue
            host = tensor[start:stop].detach().cpu().contiguous()
            for b in range(start, stop):
                if b not in included:
                    digest.update(memoryview(host[b - start].view(torch.uint8).numpy()))
        result[name] = {"sha256": digest.hexdigest(), "blocks": tensor.shape[0] - len(included)}
    return result


def logical_kv(snapshot: dict, block_owners: dict[int, tuple]) -> dict:
    """Retain complete blocks, including overhang; no sampled equal-only subset."""
    result = {}
    for name, entry in snapshot["tensors"].items():
        require(entry["block_ids"] is not None, "missing physical/logical block mapping")
        result[name] = {
            json.dumps(block_owners[physical]): entry["data"][i] for i, physical in enumerate(entry["block_ids"])
        }
    return result


class ControlledTarget:
    def __init__(
        self,
        runner: Any,
        *,
        fixture_id: str,
        arm: str,
        scheduler: Any,
        identity: dict,
        profile="base",
        bounded_store=False,
        numeric_probe=None,
        sample_limit=None,
    ):
        require(arm in ("OFF", "ON"), "invalid arm")
        # Explicit scheduler emptiness is necessary; this is not a live checkpoint.
        require(
            not scheduler.requests and not scheduler.running and not scheduler.waiting,
            "controlled fixture requires an empty, disposable engine",
        )
        self.base = _base()
        self.runner, self.fixture_id, self.arm = runner, fixture_id, arm
        require(
            all(
                identity.get(k) is not None
                for k in ("run_id", "case_id", "attempt_id", "source_sha", "source_tree", "process_id")
            ),
            "missing run/source identity",
        )
        self.identity = dict(identity)
        self.profile = profile
        self.cases = case_plan(profile)
        self.probe = self.base.LiveTargetAdapter(runner, fixture_id=fixture_id, max_records=len(self.cases))
        self.probe._validate_runner()
        self.inventory = self.base.target_inventory(runner)
        self.records: list[dict] = []
        self.deadline = time.monotonic() + 3300
        self.disk_budget = 16 * 1024**3
        self.bounded_store = bounded_store
        self.writer: Any = None  # optional, lazily imported versioned evidence writer
        self.persisted = 0
        self.numeric_probe = numeric_probe
        require(sample_limit is None or (bounded_store and 0 < sample_limit <= len(self.cases)), "invalid prefix limit")
        self.sample_limit = sample_limit

    def _budget(self, output):
        require(time.monotonic() < self.deadline, "controlled diagnostic time budget exceeded")
        require(
            sum(p.stat().st_size for p in output.parent.glob("*") if p.is_file()) <= self.disk_budget,
            "controlled diagnostic disk budget exceeded",
        )

    def _save(self, result, output):
        """Enforce byte/time limits during serialization, not only afterwards."""
        if self.writer is not None:
            if len(self.records) > self.persisted:
                self.writer.finish(self.records[-1])
                self.persisted += 1
                # Completed values are durable; retain only small status records.
                self.records[-1] = {k: self.records[-1][k] for k in ("identity", "status", "generation")}
                self.probe.records.clear()
            return
        self._budget(output)
        used = sum(p.stat().st_size for p in output.parent.glob("*") if p.is_file())
        remaining = self.disk_budget - used
        deadline = self.deadline
        partial = output.with_suffix(".pt.partial")
        with partial.open("wb") as stream:

            class BoundedWriter:
                def __init__(self):
                    self.total = 0

                def write(self, data):
                    if self.total + len(data) > remaining or time.monotonic() >= deadline:
                        raise OSError("controlled serialization budget exceeded; partial retained")
                    count = stream.write(data)
                    self.total += count
                    return count

                def flush(self):
                    stream.flush()

            torch.save(result, BoundedWriter())
        partial.replace(output)

    def _stage(self, case: dict):
        from vllm.config.compilation import CUDAGraphMode

        from vllm_ascend.ops.rotary_embedding import update_cos_sin
        from vllm_ascend.worker.v2.input_batch import AscendInputBatch

        r = self.runner
        padded, active, width = case["padded"], len(case["ends"]), case["query_width"]
        matches = [
            d
            for d in r.cudagraph_manager.graphs
            if d.cg_mode.name == "FULL" and d.num_reqs == padded and d.num_tokens == padded * width
        ]
        require(len(matches) == 1, f"CAPTURE_BUCKET_ABSENT: target ({padded},{padded * width})")
        desc = matches[0]
        batch = AscendInputBatch.make_dummy(padded, padded * width, r.input_buffers, max_query_len=width)
        batch.is_dummy = False
        batch.req_ids = [f"row-{i}" for i in range(active)]
        batch.num_reqs, batch.num_tokens = active, active * width
        batch.num_scheduled_tokens = np.full(active, width, dtype=np.int32)
        batch.num_computed_tokens_np = np.array(case["ends"], dtype=np.int32) - width
        batch.num_computed_prefill_tokens_np = batch.num_computed_tokens_np.copy()
        batch.prefill_len_np = np.zeros(active, dtype=np.int32)
        batch.is_prefilling_np = np.zeros(active, dtype=np.bool_)
        batch.query_start_loc_np = np.arange(padded + 1, dtype=np.int32) * width
        batch.idx_mapping_np = np.arange(active, dtype=np.int64)
        batch.idx_mapping = batch.idx_mapping[:active]
        batch.input_ids.fill_(17)
        batch.is_padding.fill_(True)
        batch.is_padding[: active * width] = False
        batch.positions.zero_()
        ends = case["ends"] + [1] * (padded - active)
        batch.seq_lens.copy_(torch.tensor(ends, device=batch.seq_lens.device))
        batch.seq_lens_np[:] = ends
        batch.seq_lens_cpu_upper_bound[:] = torch.tensor(ends)
        for row, end in enumerate(case["ends"]):
            batch.positions[row * width : (row + 1) * width] = torch.arange(end - width, end, device=r.device)
            batch.input_ids[row * width : (row + 1) * width] += row
        update_cos_sin(batch.positions)
        require(len(r.block_tables.input_block_tables) == 1, "single group only")
        table, slots = r.block_tables.input_block_tables[0], r.block_tables.slot_mappings[0]
        table.zero_()
        slots.fill_(-1)
        # All allocations belong to this no-request child. Different arm offsets
        # exercise the logical mapping, without changing pointer ownership.
        next_block = 1 if self.arm == "OFF" else 65
        owners: dict[int, tuple[str, int]] = {0: ("padding", 0)}
        for row, end in enumerate(case["ends"]):
            columns = (end + 7 + BLOCK - 1) // BLOCK
            physical = list(range(next_block, next_block + columns))
            table[row, :columns] = torch.tensor(physical, device=table.device)
            owners.update({b: (f"row-{row}", col) for col, b in enumerate(physical)})
            for q, pos in enumerate(range(end - width, end)):
                slots[row * width + q] = physical[pos // BLOCK] * BLOCK + pos % BLOCK
            next_block += columns
        context = r.compilation_config.static_forward_context
        for layer_index, name in enumerate(self.inventory["target_layers"] + self.inventory["draft_layers"]):
            for part, tensor in enumerate(context[name].kv_cache):
                require(
                    tensor.ndim == 4 and tensor.shape[1] == BLOCK and next_block <= tensor.shape[0],
                    "incompatible or insufficient fixture KV allocation",
                )
                # Defined complete pre-state. No model prefill history is claimed.
                tensor.zero_()
                for physical_id, (logical_row, col) in owners.items():
                    value = (
                        0.0
                        if logical_row == "padding"
                        else (int(logical_row.split("-")[1]) + col + part + layer_index % 3) / 128
                    )
                    tensor[physical_id].fill_(value)
        counts = r.block_tables.num_blocks
        counts.cpu.zero_()
        for row, end in enumerate(case["ends"]):
            counts.cpu[0, row] = (end + 7 + BLOCK - 1) // BLOCK
        counts.gpu.copy_(counts.cpu)
        r.model_state.prepare_attn(
            batch,
            CUDAGraphMode.FULL,
            (table[:padded],),
            r.block_tables.slot_mappings[:, : padded * width],
            r.attn_groups,
            r.kv_cache_config,
        )
        return desc, batch, owners

    def execute(self, output: Path, *, include_draft: bool = False) -> dict:
        result = {
            "schema": SCHEMA,
            "capture_profile": self.profile,
            "identity": self.identity,
            "execution_device": str(self.runner.device),
            "arm": self.arm,
            "fixture_id": self.fixture_id,
            "inventory": self.inventory,
            "cases": self.cases,
            "records": self.records,
            "tolerance": {"atol": ATOL, "rtol": RTOL, "basis": "predeclared exact deterministic Target contract"},
            "scope": "actual Target Graph, synthetic logical pre-KV; disposable empty engine",
            "impact": "stream synchronization, complete KV initialization and D2H; NOT PERFORMANCE REPRESENTATIVE",
            "device_verification": "IN_PROGRESS",
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        if self.bounded_store:
            from tools.p7x_graph_repair6_storage import CollectionWriter, atomic_json, require_space, resource_plan

            context = self.runner.compilation_config.static_forward_context
            tensors = [t for name in self.inventory["target_layers"] for t in context[name].kv_cache]
            geometries = {(tuple(t.shape[1:]), t.element_size()) for t in tensors}
            require(len(geometries) == 1, "heterogeneous KV geometry outside Base")
            (block, heads, dim), size = geometries.pop()
            require(block == BLOCK, "wrong block geometry")
            cfg = self.runner.model.config
            masks = [getattr(m, "attn_mask", None) for m in self.runner.model_state.attn_metadata.values()]
            mask_bytes = max((m.numel() * m.element_size() for m in masks if m is not None), default=0)
            budget = resource_plan(
                self.cases,
                layers=len(self.inventory["target_layers"]),
                kv_heads=heads,
                head_dim=dim,
                element_bytes=size,
                hidden_size=cfg.hidden_size,
                vocab_size=cfg.vocab_size,
                mask_bytes=mask_bytes,
            )
            atomic_json(output.with_name("target-resource-plan.json"), budget)
            require_space(output.parent, budget["off_on_with_raw_and_atomic_bytes"])
            self.probe.max_bytes = max(row["checkpoint"] for row in budget["samples"])
            self.disk_budget = budget["one_arm_bytes"]
            header = {k: v for k, v in result.items() if k != "records"}
            self.writer = CollectionWriter(output, header, max_bytes=self.disk_budget, timeout=3300)
            self.probe.tensor_sink = self.writer.store.put
        try:
            for case in self.cases:
                if self.sample_limit is not None and len(self.records) >= self.sample_limit:
                    break
                self._budget(output)
                started = time.monotonic()
                record = {
                    "identity": sample_identity(case, self.fixture_id),
                    "status": "PREPARING",
                    "generation": case["step"],
                    "arm": self.arm,
                }
                self.records.append(record)
                if self.writer is not None:
                    self.writer.start(record["identity"])
                try:
                    self.probe._synchronize()
                    desc, batch, owners = self._stage(case)
                    record["target_geometry"] = {
                        "descriptor": [desc.num_reqs, desc.num_tokens],
                        "active": batch.num_reqs,
                        "padded": desc.num_reqs,
                        "query_layout": batch.query_start_loc_np.tolist(),
                    }
                    record["timings_seconds"] = {"stage": time.monotonic() - started}
                    inputs, access, slots, blocks, reads = self.probe._boundary(desc, batch)
                    hash_start = time.monotonic()
                    record["outside_before"] = outside_digest(access(), blocks)
                    record["timings_seconds"]["outside_hash_before"] = time.monotonic() - hash_start
                    # Calls the actual ModelAclGraphManager, not compute_logits.
                    forward_start = time.monotonic()
                    hidden = self.probe.run(
                        lambda desc=desc: self.runner.cudagraph_manager.run_fullgraph(desc),
                        inputs=inputs,
                        kv_access=access,
                        write_slots=slots,
                        synchronize=self.probe._synchronize,
                        origin="LIVE_ASCEND_TARGET_FULL_GRAPH",
                        block_ids=blocks,
                        declared_reads=reads,
                    )
                    record["timings_seconds"]["forward_with_checkpoint_and_sync"] = time.monotonic() - forward_start
                    checkpoint = self.probe.records[-1]
                    require(checkpoint["status"] == "CAPTURED", "incomplete Target checkpoint")
                    record["checkpoint"] = checkpoint
                    record["block_owners"] = owners
                    record["logical_pre_kv"] = logical_kv(checkpoint["kv_before"], owners)
                    record["logical_post_kv"] = logical_kv(checkpoint["kv_after"], owners)
                    hash_start = time.monotonic()
                    record["outside_after"] = outside_digest(access(), blocks)
                    record["timings_seconds"]["outside_hash_after"] = time.monotonic() - hash_start
                    actual_hidden = hidden[0] if isinstance(hidden, tuple) else hidden
                    record["hidden"] = actual_hidden[: batch.num_tokens].detach().cpu().clone()
                    record["logits"] = (
                        self.runner.model.compute_logits(actual_hidden[: batch.num_tokens]).detach().cpu()
                    )
                    record["status"] = "CAPTURED_COMPLETE"
                    if include_draft:
                        if self.numeric_probe is not None:
                            self.numeric_probe.current_case = record["identity"]
                        spec = self.runner.speculator
                        target_root = self.runner.model_state.attn_metadata
                        require(
                            spec.sample_from_anchor and spec.num_query_per_req == 7,
                            "real MRV2 control requires DSpark K=query7 anchor",
                        )
                        nr = batch.num_reqs
                        maximum = spec.max_num_reqs
                        device = self.runner.device
                        # Actual propose prepares all query/sampling state, eager
                        # context KV, P3/P6 materialization, Graph/update and retire.
                        result_tokens = spec.propose(
                            batch,
                            self.runner.model_state.attn_metadata,
                            {name: slots[f"{name}/K"] for name in self.inventory["target_layers"]},
                            actual_hidden,
                            list(hidden[1]) if isinstance(hidden, tuple) else None,
                            torch.ones(nr, dtype=torch.int32, device=device),
                            torch.zeros(nr, dtype=torch.int32, device=device),
                            torch.full((maximum,), 17, dtype=torch.int64, device=device),
                            torch.full((maximum,), 17, dtype=torch.int64, device=device),
                            torch.zeros(maximum, device=device),
                            torch.zeros(maximum, dtype=torch.int64, device=device),
                        )
                        if self.arm == "OFF":
                            # No Window provider/event probe is installed for
                            # CONFIG_OFF. Complete both actual baseline streams
                            # AFTER propose/update; diagnostic-only serialization.
                            spec.update_stream.synchronize()
                            torch.npu.current_stream().synchronize()
                        # The optional lifecycle probe proves its own diagnostic
                        # completion before propose returns; no lease is extended.
                        record["draft_tokens"] = result_tokens.detach().cpu().clone()
                        record["draft_semantics"] = f"Window {self.arm}; draft values NOT compared OFF/ON"
                        session = getattr(spec, "_draft_window_graph_characterization", None)
                        record["draft_snapshot"] = None if session is None else session.recorder.snapshot()
                        if session is not None:
                            record["draft_geometry"] = draft_geometry(spec, record["draft_snapshot"], nr)
                        require(
                            self.runner.model_state.attn_metadata is target_root, "Draft replaced Target metadata owner"
                        )
                        hash_start = time.monotonic()
                        outside_after_draft = outside_digest(access(), blocks)
                        record["timings_seconds"]["outside_hash_after_draft"] = time.monotonic() - hash_start
                        record["target_after_draft"] = {
                            "kv": self.probe.snapshot(access(), [self.probe.max_bytes], blocks),
                            "inputs": self.probe.snapshot(inputs, [self.probe.max_bytes]),
                            "outside": outside_after_draft,
                            "root_same": True,
                            "draft_slots": spec.block_tables.slot_mappings[0, : nr * 7].detach().cpu().clone(),
                            "draft_positions": spec.input_buffers.positions[: nr * 7].detach().cpu().clone(),
                            "draft_query": spec.input_buffers.query_start_loc[: nr + 1].detach().cpu().clone(),
                            "slot_phase": (
                                "baseline prepare_dflash_inputs rewrites full query slots AFTER Target forward"
                            ),
                        }
                except ValueError as error:
                    record.update(status="BLOCKED", error=str(error))
                    if not str(error).startswith("CAPTURE_BUCKET_ABSENT:"):
                        raise
                except BaseException as error:
                    record.update(status="EXECUTION_FAILED", error=f"{type(error).__name__}: {error}")
                    raise
                finally:
                    record.setdefault("timings_seconds", {})["before_dump"] = time.monotonic() - started
                    self._flush_preserving_error(result, output)
            result["device_verification"] = "EXECUTED_SCOPED_NOT_GATE_CLOSURE"
        finally:
            self._flush_preserving_error(result, output)
            if self.writer is not None:
                self.writer.close(result["device_verification"])
        return result

    def _flush_preserving_error(self, result, output):
        original = sys.exc_info()[1]
        try:
            self._save(result, output)
        except Exception as error:
            if original is None:
                raise
            original.add_note(f"Secondary evidence flush failure: {type(error).__name__}: {error}")


def draft_geometry(spec, snapshot, active):
    """Use the executed binding, not Target padding or a guessed descriptor."""
    g4 = snapshot["g4_records"][-1]
    bound = [e for e in snapshot["events"] if e["stage"] == "bound" and e["generation_id"] == g4["generation_id"]]
    require(len(bound) == 1, "missing unique Draft binding")
    parts = dict(p.split("=", 1) for p in bound[0]["capture_descriptor"].split(";"))
    padded, tokens = int(parts["reqs"]), int(parts["tokens"])
    return {
        "descriptor": [padded, tokens],
        "active": active,
        "padded": padded,
        "query_layout": spec.input_buffers.query_start_loc[: padded + 1].detach().cpu().tolist(),
        "generation": g4["generation_id"],
        "epoch": g4["mapping_epoch"],
        "source_identity": g4["runtime_oracle_inputs"]["source_identity"],
    }


def _tensor_equal(a: Any, b: Any) -> bool:
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (
            isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and a.shape == b.shape
            and a.dtype == b.dtype
            and bool(torch.isfinite(a).all() and torch.isfinite(b).all())
            and torch.equal(a, b)
        )
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_tensor_equal(a[k], b[k]) for k in a)
    return a == b


def validate_controlled_record(record: dict) -> dict:
    """Recompute all alignment/writer prerequisites from payloads, not labels."""
    base = _base()
    checkpoint = record["checkpoint"]
    require(checkpoint["status"] == "CAPTURED", "incomplete checkpoint")
    for name in ("inputs", "output", "kv_before", "kv_after", "write_slots"):
        base._validate_snapshot(checkpoint[name])
    data = base._snapshot_value(checkpoint["inputs"])
    identity = record["identity"]
    active, padded = identity["active"], identity["padded"]
    width = 8
    require(record["generation"] == identity["step"], "generation/plan step mismatch")
    require(
        data["num_reqs"] == active
        and data["num_tokens"] == active * width
        and data["descriptor"][:2] == [padded * width, padded],
        "batch geometry mismatch",
    )
    require(list(data["req_ids"]) == identity["logical_requests"], "logical requests differ")
    require(data["query_start_loc"].tolist() == identity["query_layout"], "query layout differs")
    expected_positions = sum(identity["positions"], []) + [0] * ((padded - active) * width)
    require(data["positions"].tolist() == expected_positions, "actual positions differ from plan")
    require(
        data["input_ids"][: active * width].tolist() == [17 + row for row in range(active) for _ in range(width)],
        "actual tokens differ from plan",
    )
    require(
        data["is_padding"].tolist() == [False] * (active * width) + [True] * ((padded - active) * width),
        "padding differs from plan",
    )
    owners = record["block_owners"]
    require(len(set(tuple(v) for v in owners.values())) == len(owners), "aliased logical block owner")
    names = data["target_layer_names"]
    require(set(data["layers"]) == set(names), "layer metadata mismatch")
    semantics = {}
    for name in names:
        m = data["layers"][name]
        require(
            m["causal"] is True and m["actual_seq_lengths_q"] == identity["query_layout"][1:],
            "causal/query operand differs from plan",
        )
        lengths = [positions[-1] + 1 for positions in identity["positions"]] + [1] * (padded - active)
        require(m["seq_lens_list"] == lengths, "actual read lengths differ")
        actual_slots = base._snapshot_value(checkpoint["write_slots"])[f"{name}/K"].tolist()
        require(
            m["slot_mapping"].tolist() == actual_slots[: active * width],
            "metadata slot view is not the full canonical active write view",
        )
        expected_slots = []
        logical_rows = []
        for row, length in enumerate(lengths):
            count = (length + BLOCK - 1) // BLOCK
            physical = m["block_tables"][row, :count].tolist()
            logical = [tuple(owners[p]) for p in physical]
            expected_logical = [(f"row-{row}", c) for c in range(count)] if row < active else [("padding", 0)]
            require(logical == expected_logical, "wrong read table/request owner")
            logical_rows.append(logical)
            if row < active:
                expected_slots += [physical[pos // BLOCK] * BLOCK + pos % BLOCK for pos in identity["positions"][row]]
            else:
                expected_slots += [-1] * width
        require(actual_slots == expected_slots, "wrong physical write slots")
        valid = [slot for slot in actual_slots if slot >= 0]
        require(len(valid) == len(set(valid)), "duplicate writer slot")
        semantics[name] = {
            "table": logical_rows,
            "lengths": lengths,
            "query": m["actual_seq_lengths_q"],
            "causal": m["causal"],
        }
    # The sampling payload must actually be the full-forward result captured
    # by the checkpointer, not a cached hidden tensor from an earlier generation.
    outputs = checkpoint["output"]["tensors"]
    hidden_entry = outputs.get("") or outputs.get("/0")
    require(
        hidden_entry is not None and _tensor_equal(hidden_entry["data"][: active * width], record["hidden"]),
        "hidden sample differs from actual full-forward output",
    )
    require(
        bool(torch.isfinite(record["logits"]).all()) and record["logits"].shape[0] == active * width,
        "invalid/incomplete logits",
    )
    return semantics


def check_collection(document):
    result = {"status": "INVALID_EVIDENCE", "numeric_off_on": "NOT_ASSESSED", "records": []}
    try:
        require(document["schema"] == SCHEMA, "wrong collection schema")
        cases = case_plan(document.get("capture_profile", "base"))
        require(document["cases"] == cases and len(document["records"]) == len(cases), "incomplete plan")
        for case, record in zip(cases, document["records"]):
            require(record["identity"] == sample_identity(case, document["fixture_id"]), "wrong sample identity")
            if record["status"] == "BLOCKED":
                result["records"].append("BLOCKED")
                continue
            require(record["status"] == "CAPTURED_COMPLETE", "producer incomplete")
            validate_controlled_record(record)
            result["records"].append("VALID")
        result["status"] = "OPEN" if "BLOCKED" in result["records"] else "SINGLE_ARM_STRUCTURE_VALID"
    except (KeyError, TypeError, ValueError) as error:
        result["error"] = str(error)
    return result


def compare_collections(a: dict, b: dict) -> dict:
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "INVALID_EVIDENCE",
        "pairs": [],
        "physical_read": "OPEN",
        "completion": "UNPROVEN",
    }
    try:
        require(a["schema"] == b["schema"] == SCHEMA and a["arm"] == "OFF" and b["arm"] == "ON", "schema/arm mismatch")
        profile = a.get("capture_profile", "base")
        require(profile == b.get("capture_profile", "base"), "cross-profile comparison")
        cases = case_plan(profile)
        require(a["fixture_id"] == b["fixture_id"] and a["cases"] == b["cases"] == cases, "plan/fixture mismatch")
        require(
            all(
                a["identity"][k] == b["identity"][k] and a["identity"][k]
                for k in ("run_id", "source_sha", "source_tree")
            ),
            "cross-run/source mixture",
        )
        result["execution_devices"] = [a["execution_device"], b["execution_device"]]
        require(
            a["tolerance"] == b["tolerance"] and a["tolerance"]["atol"] == ATOL and a["tolerance"]["rtol"] == RTOL,
            "tolerance changed",
        )
        require(a["inventory"] == b["inventory"], "model inventory mismatch")
        expected = {json.dumps(sample_identity(c, a["fixture_id"]), sort_keys=True) for c in cases}
        indices = []
        for doc in (a, b):
            index = {}
            for record in doc["records"]:
                key = json.dumps(record["identity"], sort_keys=True)
                require(key in expected and key not in index, "extra/duplicate sample identity")
                index[key] = record
            indices.append(index)
        all_statuses = []
        for key in sorted(expected):
            pair = {"identity": json.loads(key), "status": "INSUFFICIENT_COVERAGE"}
            x, y = (i.get(key) for i in indices)
            if x and y:
                if x["status"] != "CAPTURED_COMPLETE" or y["status"] != "CAPTURED_COMPLETE":
                    pair["status"] = "PRODUCER_BLOCKED"
                else:
                    # Full raw checkpoint retains internal physical slot/table/KV
                    # validation. Cross-run pairing uses declared logical ownership.
                    base = _base()
                    semantics_equal = validate_controlled_record(x) == validate_controlled_record(y)
                    for rec in (x, y):
                        base._validate_read_coverage(rec["checkpoint"])
                        recomputed = logical_kv(rec["checkpoint"]["kv_before"], rec["block_owners"])
                        require(_tensor_equal(recomputed, rec["logical_pre_kv"]), "forged logical pre-KV")
                        require(
                            _tensor_equal(
                                logical_kv(rec["checkpoint"]["kv_after"], rec["block_owners"]), rec["logical_post_kv"]
                            ),
                            "forged logical post-KV",
                        )
                    if not semantics_equal or not _tensor_equal(x["logical_pre_kv"], y["logical_pre_kv"]):
                        pair["status"] = "ALIGNMENT_PRECONDITION_NOT_MET"
                        all_statuses.append(pair["status"])
                        result["pairs"].append(pair)
                        continue
                    errors = []
                    for field in ("hidden", "logits", "logical_post_kv"):
                        if not _tensor_equal(x[field], y[field]):
                            errors.append(field)
                    for rec in (x, y):
                        if rec["outside_before"] != rec["outside_after"]:
                            errors.append("unauthorized outside-block write")
                        coverage = base._write_coverage(
                            rec["checkpoint"]["kv_before"],
                            rec["checkpoint"]["kv_after"],
                            rec["checkpoint"]["write_slots"],
                        )
                        if any(v["changed_outside_write_slots"] for v in coverage.values()):
                            errors.append("unauthorized within-block write")
                        if "target_after_draft" in rec:
                            after = rec["target_after_draft"]
                            if (
                                not after["root_same"]
                                or after["outside"] != rec["outside_after"]
                                or base._differences(rec["checkpoint"]["kv_after"], after["kv"])
                            ):
                                errors.append("Draft polluted Target/write state")
                            original = base._snapshot_value(rec["checkpoint"]["inputs"])
                            current = base._snapshot_value(after["inputs"])
                            for field in ("input_ids", "positions", "query_start_loc"):
                                require(_tensor_equal(original[field], current[field]), f"Draft changed Target {field}")
                            full_table = next(iter(original["layers"].values()))["block_tables"]
                            expected_slots, expected_pos = [], []
                            for row, positions in enumerate(rec["identity"]["positions"]):
                                query_positions = list(range(positions[-1] + 1, positions[-1] + 8))
                                expected_pos += query_positions
                                expected_slots += [
                                    int(full_table[row, p // 128]) * 128 + p % 128 for p in query_positions
                                ]
                            require(
                                after["draft_slots"].tolist() == expected_slots
                                and after["draft_positions"].tolist() == expected_pos
                                and after["draft_query"].tolist()
                                == [i * 7 for i in range(rec["identity"]["active"] + 1)],
                                "Draft canonical full write/query contract failed",
                            )
                    pair.update(status="FAIL" if errors else "PASS", errors=errors)
            all_statuses.append(pair["status"])
            result["pairs"].append(pair)
        result["confirmed_failures"] = all_statuses.count("FAIL")
        result["status"] = (
            "FAIL" if "FAIL" in all_statuses else ("PASS" if all(s == "PASS" for s in all_statuses) else "OPEN")
        )
        result["window_interference_coverage"] = (
            "COLLECTED"
            if all("target_after_draft" in r for r in b["records"] if r["status"] == "CAPTURED_COMPLETE")
            else "OPEN_WINDOW_CONSUMER_NOT_EXECUTED_IN_PAIRED_RUN"
        )
    except (KeyError, TypeError, ValueError) as error:
        result["error"] = str(error)
        result["status"] = "FAIL" if any(p["status"] == "FAIL" for p in result["pairs"]) else "INVALID_EVIDENCE"
    return result


def check_mrv2_aba(document: dict) -> dict:
    """No greedy-token equality is promoted to hidden-state/read equivalence."""
    result: dict[str, Any] = {"status": "INVALID_EVIDENCE", "physical_read": "OPEN", "normal_overlap": "UNPROVEN"}
    try:
        require(document["schema"] == SCHEMA and document["arm"] == "ON", "not controlled Window ON")
        cases = case_plan(document.get("capture_profile", "base"))
        require(document["cases"] == cases, "incomplete declared plan")
        records = document["records"]
        require(len(records) == len(cases), "missing/extra generation")
        require(
            [r["identity"] for r in records] == [sample_identity(c, document["fixture_id"]) for c in cases],
            "wrong scenario order/identity",
        )
        a, b, again = records[:3]
        generations: list[int] = []
        backing: list[int] = []
        provider = None
        previous_epoch = -1
        for record in records:
            if record["status"] == "BLOCKED":
                require(record["error"].startswith("CAPTURE_BUCKET_ABSENT:"), "unexpected blocked generation")
                continue
            require(record["status"] == "CAPTURED_COMPLETE", "incomplete full Target precondition")
            validate_controlled_record(record)
            snapshot = record["draft_snapshot"]
            g4 = snapshot["g4_records"][-1]
            require(provider in (None, snapshot["provider_identity"]), "cross-provider mixture")
            provider = snapshot["provider_identity"]
            require(g4["mapping_epoch"] > previous_epoch, "stale mapping epoch")
            previous_epoch = g4["mapping_epoch"]
            if generations:
                require(g4["generation_id"] > generations[-1], "stale generation")
            generations.append(g4["generation_id"])
            backing.append(g4["window_table_address"])
            require(record["draft_tokens"].numel() > 0, "no actual propose result")
            raw = g4["runtime_oracle_inputs"]
            target_inputs = _base()._snapshot_value(record["checkpoint"]["inputs"])
            canonical = next(iter(target_inputs["layers"].values()))["block_tables"].tolist()
            geometry = record["draft_geometry"]
            active, padded = geometry["active"], geometry["padded"]
            require(
                active == record["identity"]["active"]
                and padded in (1, 2, 4)
                and padded == next(n for n in (1, 2, 4) if n >= active)
                and geometry["descriptor"] == [padded, padded * 7],
                "wrong executed Draft geometry",
            )
            require(
                geometry["generation"] == g4["generation_id"] and geometry["epoch"] == g4["mapping_epoch"],
                "Draft geometry generation mismatch",
            )
            require(
                geometry["query_layout"][: active + 1] == [i * 7 for i in range(active + 1)],
                "Draft active query layout differs",
            )
            require(
                g4["canonical_table_values"][:active] == canonical[:active],
                "Draft source differs from controlled canonical Target table",
            )
            require(raw["source_identity"] is not None, "missing canonical provenance")
            require(geometry["source_identity"] == raw["source_identity"], "Draft lineage mismatch")
            require(raw["request_ids"] == record["identity"]["logical_requests"], "wrong logical request")
            ends = [p[-1] + 1 + 7 for p in record["identity"]["positions"]]
            require(
                raw["full_exact"] == ends and raw["full_upper"] == ends,
                "controlled zero-rejection producer length differs",
            )
            expected_exact = []
            for row, end in enumerate(ends):
                start_column = max(end - 1024, 0) // 128
                end_column = (end + 127) // 128
                require(raw["valid_counts"][row] >= end_column, "canonical shortage")
                expected = g4["canonical_table_values"][row][start_column:end_column]
                actual = g4["window_table_values"][row]
                require(
                    actual == expected + [0] * (len(actual) - len(expected)), "Window differs from canonical oracle"
                )
                expected_exact.append(end - start_column * 128)
            events = [
                e for e in snapshot["events"] if e["stage"] == "bound" and e["generation_id"] == g4["generation_id"]
            ]
            require(len(events) == 1 and events[0]["mapping_epoch"] == g4["mapping_epoch"], "missing/duplicate bind")
            require(
                all(not any(row) for row in g4["window_table_values"][active:padded]), "nonzero padded Draft read table"
            )
            lengths = expected_exact + [0] * (padded - len(ends))
            require(
                events[0]["exact_lengths"] == lengths and events[0]["upper_lengths"] == lengths,
                "stale/wrong window length",
            )
        require(generations[0] < generations[1] < generations[2], "stale actual MRV2 generation")
        require(len(set(backing[:3])) == 1, "ABA backing changed")
        require(_tensor_equal(a["logical_pre_kv"], again["logical_pre_kv"]), "A prime pre-KV not restored")
        require(_tensor_equal(a["draft_tokens"], again["draft_tokens"]), "A prime actual result differs")
        result["actual_generations"] = generations
        # Discrete greedy results may be identical despite distinct contexts.
        # This is explicitly inconclusive, never an automatic ABA PASS.
        result["status"] = (
            "PASS_SCOPED_NUMERIC"
            if not _tensor_equal(a["draft_tokens"], b["draft_tokens"])
            else "OPEN_NONDISCRIMINATING_DRAFT_TOKENS"
        )
        result["scope"] = "actual MRV2 propose tokens; not continuous hidden oracle or physical read proof"
    except (KeyError, TypeError, ValueError, IndexError) as error:
        result["error"] = str(error)
    return result
