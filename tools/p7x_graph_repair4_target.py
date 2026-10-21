#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
"""Bounded full-forward/KV checkpoints; never restores or edits live tensors.

Live use, AFTER startup capture, in the existing single-worker driver:

    with LiveTargetAdapter(runner, fixture_id="weights-manifest:fixed-case") as probe:
        ...  # existing driver executes its independently controlled requests
    probe.save("target-checkpoints.pt")

The adapter wraps only this runner's prepare_inputs and target run_fullgraph.
It calls the original forward once, preserves its result/exception, and takes
complete mapped-block CPU copies around it. No compute_logits call, scheduler edit,
allocator edit, capture rewrite, extra replay, or cache restore is performed.
The fixture driver must supply identical histories; comparison checks actual
inputs and complete declared read/write pre-forward KV rather than trusting
the fixture_id. Blocks outside that footprint are explicitly unobserved.
The ordinary generation driver produces attempts, not a deterministic pair:
its status is ALIGNMENT_REQUIRED until complete observed pre-state matches.
No automatic fixed-token fixture or full-ready assertion is provided.
Comparison requires identical physical block IDs; it does not normalize
equivalent logical histories stored in different physical layouts.

Only dense, unquantized Qwen3 TP/DP/PP/PCP/DCP=1, non-async FULL target replay
is admitted. Unsupported/incomplete observations are BLOCKED, never PASS.
Host synchronization and D2H make this unsuitable for timing. Even a matching
pair does not establish physical reads or torch-npu update-effective timing.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Direct execution must not let tools/bisect shadow the standard library.
sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]

import argparse
import importlib
import json
import math
from collections.abc import Callable, Mapping
from enum import Enum
from types import MethodType
from typing import Any

import torch

SCHEMA = "p7x-repair4-full-forward-v1"
PINNED_VLLM = "84030bbe3d74d99bad477a3d2e37a973ccd8865c"
DEFAULT_MAX_BYTES = 512 * 1024 * 1024
REQUIRED_INPUTS = {"input_ids", "positions", "query_start_loc", "seq_lens"}


class EvidenceError(ValueError):
    """The boundary cannot supply the complete declared evidence."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def _class_path(obj: Any) -> str:
    return f"{type(obj).__module__}.{type(obj).__name__}"


def _view(tensor: torch.Tensor) -> tuple:
    return (str(tensor.device), tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype))


def _flatten(value: Any, path: str = "") -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    tensors: dict[str, torch.Tensor] = {}
    literals: dict[str, Any] = {}
    if isinstance(value, torch.Tensor):
        _require(value.layout == torch.strided and value.numel() > 0, f"empty/non-strided tensor: {path}")
        tensors[path] = value
    elif isinstance(value, Mapping):
        _require(all(isinstance(k, str) for k in value), f"non-string mapping keys: {path}")
        literals[path] = ("dict", tuple(sorted(value)))
        for key in sorted(value):
            escaped = key.replace("~", "~0").replace("/", "~1")
            ts, ls = _flatten(value[key], f"{path}/{escaped}")
            tensors.update(ts)
            literals.update(ls)
    elif isinstance(value, (list, tuple)):
        literals[path] = (type(value).__name__, len(value))
        for index, child in enumerate(value):
            ts, ls = _flatten(child, f"{path}/{index}")
            tensors.update(ts)
            literals.update(ls)
    elif isinstance(value, Enum):
        literals[path] = (_class_path(value), value.name)
    elif value is None or type(value) in (str, int, float, bool):
        _require(not isinstance(value, float) or math.isfinite(value), f"nonfinite literal: {path}")
        literals[path] = (type(value).__name__, value)
    else:
        raise EvidenceError(f"unsupported input at {path}: {_class_path(value)}")
    return tensors, literals


def _snapshot(
    value: Any, budget: list[int], block_ids: Mapping[str, torch.Tensor] | None = None, *, tensor_sink=None
) -> dict:
    tensors, literals = _flatten(value)
    selections: dict[str, torch.Tensor] = {}
    if block_ids is not None:
        selections, _ = _flatten(block_ids)
        _require(set(selections) == set(tensors), "incomplete selected-block inventory")
    # A group metadata object is shared by multiple real layer keys. Account
    # and copy an identical view only once; retain every field and alias label.
    keys = {
        name: (_view(t), t.storage_offset(), tuple(selections[name].tolist()) if name in selections else None)
        for name, t in tensors.items()
    }
    unique = {keys[name]: name for name in tensors}
    size = sum(
        (selections[name].numel() * t[0].numel() if name in selections else t.numel()) * t.element_size()
        for name in unique.values()
        for t in (tensors[name],)
    )
    _require(size <= budget[0], f"complete checkpoint needs {size} bytes; remaining budget {budget[0]}")
    budget[0] -= size
    aliases: dict[tuple[str, int], int] = {}
    entries: dict[str, Any] = {}
    copies: dict[tuple, torch.Tensor] = {}
    for name, tensor in tensors.items():
        storage = tensor.untyped_storage()
        storage_key = (str(tensor.device), storage.data_ptr())
        alias = aliases.setdefault(storage_key, len(aliases))
        selected = None
        payload = tensor
        if name in selections:
            ids = selections[name]
            _require(ids.device.type == "cpu" and ids.dtype == torch.int64 and ids.ndim == 1, "invalid block ids")
            selected = tuple(ids.tolist())
            _require(bool(selected) and list(selected) == sorted(set(selected)), "empty/duplicate/unsorted block ids")
            _require(selected[0] >= 0 and selected[-1] < tensor.shape[0], "selected block out of bounds")
            if tensor_sink is None:
                payload = tensor.index_select(0, ids.to(tensor.device))
        if keys[name] not in copies:
            copies[keys[name]] = (
                tensor_sink(tensor, selected)
                if tensor_sink is not None
                else payload.detach().to(device="cpu", copy=True).contiguous()
            )
        copied = copies[keys[name]]
        entries[name] = {
            "data": copied,
            "shape": tuple(copied.shape),
            "source_shape": tuple(tensor.shape),
            "block_ids": selected,
            "stride": tuple(tensor.stride()),
            "offset": tensor.storage_offset(),
            "dtype": str(tensor.dtype),
            "device_type": tensor.device.type,
            "alias": alias,
            "address": tensor.data_ptr(),
        }
    return {"literals": literals, "tensors": entries}


def _validate_snapshot(snapshot: dict) -> None:
    _require(set(snapshot) == {"literals", "tensors"}, "invalid checkpoint schema")
    for name, entry in snapshot["tensors"].items():
        data = entry["data"]
        _require(isinstance(data, torch.Tensor) and data.device.type == "cpu", f"missing CPU payload: {name}")
        _require(data.numel() > 0 and tuple(data.shape) == entry["shape"], f"invalid payload shape: {name}")
        _require(str(data.dtype) == entry["dtype"], f"invalid payload dtype: {name}")
        _require(bool(torch.isfinite(data).all()), f"nonfinite payload: {name}")


def _snapshot_value(snapshot: dict, path: str = "") -> Any:
    """Recover primitive structure without executing serialized Python objects."""
    if path in snapshot["tensors"]:
        return snapshot["tensors"][path]["data"]
    kind, value = snapshot["literals"][path]
    if kind == "dict":
        return {key: _snapshot_value(snapshot, path + "/" + key.replace("~", "~0").replace("/", "~1")) for key in value}
    if kind in ("tuple", "list"):
        return [_snapshot_value(snapshot, f"{path}/{index}") for index in range(value)]
    return value


def _differences(left: dict, right: dict, *, atol: float = 0, rtol: float = 0) -> list[str]:
    _validate_snapshot(left)
    _validate_snapshot(right)
    errors = []
    if left["literals"] != right["literals"]:
        errors.append("literal/layout mismatch")
    a, b = left["tensors"], right["tensors"]
    if set(a) != set(b):
        return errors + ["tensor inventory mismatch"]
    for name in a:
        fields = ("shape", "source_shape", "block_ids", "stride", "offset", "dtype", "device_type", "alias")
        if any(a[name][key] != b[name][key] for key in fields):
            errors.append(f"tensor geometry/alias mismatch: {name}")
            continue
        x, y = a[name]["data"], b[name]["data"]
        equal = torch.allclose(x, y, atol=atol, rtol=rtol) if x.is_floating_point() else torch.equal(x, y)
        if not equal:
            errors.append(f"tensor values differ: {name}")
    return errors


def _write_coverage(before: dict, after: dict, slots: dict) -> dict:
    """Compare every element, including blocks outside the declared write slots."""
    _validate_snapshot(before)
    _validate_snapshot(after)
    _validate_snapshot(slots)
    a, b, s = before["tensors"], after["tensors"], slots["tensors"]
    _require(bool(a) and set(a) == set(b) == set(s), "incomplete KV/write-slot inventory")
    result = {}
    for key, entry in a.items():
        _require(
            all(
                entry[f] == b[key][f]
                for f in ("shape", "source_shape", "block_ids", "stride", "offset", "dtype", "alias", "address")
            ),
            f"KV backing replaced during forward: {key}",
        )
        x, y, allowed = entry["data"], b[key]["data"], s[key]["data"]
        _require(x.ndim == 4, "KV coverage requires [blocks, block_size, heads, head_dim]")
        _require(allowed.ndim == 1 and allowed.dtype in (torch.int32, torch.int64), "invalid write slots")
        count = x.shape[0] * x.shape[1]
        physical_count = entry["source_shape"][0] * x.shape[1]
        _require(bool(((allowed >= -1) & (allowed < physical_count)).all()), "out-of-range write slot")
        allowed = allowed[allowed >= 0].long()
        if entry["block_ids"] is not None:
            mapping = {block: index for index, block in enumerate(entry["block_ids"])}
            _require(all(int(slot) // x.shape[1] in mapping for slot in allowed), "write block omitted from checkpoint")
            allowed = torch.tensor(
                [mapping[int(slot) // x.shape[1]] * x.shape[1] + int(slot) % x.shape[1] for slot in allowed],
                dtype=torch.int64,
            )
        changed = (x != y).reshape(count, -1).any(dim=1)
        permitted = torch.zeros(count, dtype=torch.bool)
        permitted[allowed] = True
        result[key] = {
            "total_elements": x.numel(),
            "total_slots": count,
            "changed_slots": int(changed.sum()),
            "changed_outside_write_slots": int((changed & ~permitted).sum()),
            "outside_selected_blocks": "UNOBSERVED" if entry["block_ids"] is not None else "COVERED",
        }
    return result


def select_mapped_blocks(
    block_table: torch.Tensor, lengths: list[int], slots: torch.Tensor, *, num_blocks: int, block_size: int
) -> torch.Tensor:
    """Union every declared read block (including padding) with every write block."""
    table, writes = block_table.detach().cpu(), slots.detach().cpu()
    _require(table.ndim == 2 and table.dtype in (torch.int32, torch.int64), "invalid read table")
    _require(bool(lengths) and len(lengths) <= table.shape[0], "read lengths/table rows mismatch")
    _require(writes.ndim == 1 and writes.dtype in (torch.int32, torch.int64), "invalid write-slot tensor")
    _require(block_size > 0 and num_blocks > 0, "invalid block geometry")
    blocks = set()
    for row, length in enumerate(lengths):
        _require(type(length) is int and length >= 0, "invalid declared read length")
        width = (length + block_size - 1) // block_size
        _require(width <= table.shape[1], "read table does not cover full sequence")
        blocks.update(table[row, :width].tolist())
    _require(bool(((writes >= -1) & (writes < num_blocks * block_size)).all()), "invalid physical write slot")
    blocks.update((writes[writes >= 0] // block_size).tolist())
    _require(bool(blocks) and min(blocks) >= 0 and max(blocks) < num_blocks, "missing/out-of-range mapped blocks")
    return torch.tensor(sorted(blocks), dtype=torch.int64)


def _validate_read_coverage(record: dict) -> None:
    if record["kv_scope"] == "ALL_BOUND_KV_TENSORS":
        _require(
            all(e["block_ids"] is None for e in record["kv_before"]["tensors"].values()),
            "partial blocks labelled as all-bound KV",
        )
        return
    _require(record["kv_scope"] == "FULL_DECLARED_READ_WRITE_BLOCKS", "unknown KV coverage domain")
    _validate_snapshot(record["declared_reads"])
    reads = _snapshot_value(record["declared_reads"])
    _require(set(reads) == set(record["kv_keys"]), "incomplete declared KV read inventory")
    for name in record["kv_keys"]:
        key = "/" + name.replace("~", "~0").replace("/", "~1")
        entry = record["kv_before"]["tensors"][key]
        selected = select_mapped_blocks(
            reads[name]["table"],
            reads[name]["lengths"],
            record["write_slots"]["tensors"][key]["data"],
            num_blocks=entry["source_shape"][0],
            block_size=entry["source_shape"][1],
        )
        _require(tuple(selected.tolist()) == entry["block_ids"], f"incomplete mapped KV read/write coverage: {name}")
    if record["origin"] == "LIVE_ASCEND_TARGET_FULL_GRAPH":
        inputs = _snapshot_value(record["inputs"])
        expected = {f"{name}/{part}" for name in inputs["target_layer_names"] for part in ("K", "V")}
        _require(expected == set(record["kv_keys"]), "incomplete live target layer KV inventory")
        for name, read in reads.items():
            metadata = inputs["layers"][name.rsplit("/", 1)[0]]
            _require(
                torch.equal(read["table"], metadata["block_tables"]) and read["lengths"] == metadata["seq_lens_list"],
                "read coverage manifest differs from actual target update metadata",
            )


class FullForwardCheckpointer:
    """Reusable explicit-callable runner. A caller-supplied callable is not NPU proof."""

    def __init__(self, *, fixture_id: str, max_bytes: int = DEFAULT_MAX_BYTES, max_records: int = 4, tensor_sink=None):
        _require(bool(fixture_id), "fixture_id must identify the model weights and controlled case")
        _require(max_bytes > 0 and max_records > 0, "checkpoint limits must be positive")
        self.fixture_id, self.max_bytes, self.max_records = fixture_id, max_bytes, max_records
        self.records: list[dict] = []
        self.skipped_for_limit = 0
        self.tensor_sink = tensor_sink

    def snapshot(self, value, budget, block_ids=None):
        return _snapshot(value, budget, block_ids, tensor_sink=self.tensor_sink)

    def run(
        self,
        full_forward: Callable[[], Any],
        *,
        inputs: Mapping[str, Any],
        kv_access: Callable[[], Mapping[str, torch.Tensor]],
        write_slots: Mapping[str, torch.Tensor],
        synchronize: Callable[[], None],
        origin: str = "EXPLICIT_FULL_FORWARD",
        block_ids: Mapping[str, torch.Tensor] | None = None,
        declared_reads: Mapping[str, Any] | None = None,
    ) -> Any:
        """Execute exactly once; only the supplied forward may write live tensors.

        synchronize must cover all relevant writers/readers. For live FULL
        replay it is called before submission and after the ORIGINAL manager
        returns (which has already submitted the backend task updates).
        """
        _require(len(self.records) < self.max_records, "checkpoint record limit reached")
        _require(inputs.keys() >= REQUIRED_INPUTS, "missing complete forward input fields")
        budget = [self.max_bytes]
        synchronize()
        kv = dict(kv_access())
        _require(bool(kv) and set(kv) == set(write_slots), "missing KV/write-slot coverage")
        _require(all(isinstance(t, torch.Tensor) and t.ndim == 4 for t in kv.values()), "unsupported KV layout")
        before: dict[str, Any] = {
            "schema": SCHEMA,
            "fixture_id": self.fixture_id,
            "origin": origin,
            "case_kind": "NATURAL_OBSERVATION"
            if origin == "LIVE_ASCEND_TARGET_FULL_GRAPH"
            else "EXPLICIT_CALLABLE_OBSERVATION",
            "status": "PREPARED",
            "kv_keys": sorted(kv),
            "inputs": self.snapshot(inputs, budget),
            "kv_before": self.snapshot(kv, budget, block_ids),
            "write_slots": self.snapshot(write_slots, budget),
            "update_effective_generation": "UNKNOWN",
            "physical_read_coverage": "UNPROVEN",
            "kv_scope": "FULL_DECLARED_READ_WRITE_BLOCKS" if block_ids is not None else "ALL_BOUND_KV_TENSORS",
            "declared_reads": self.snapshot(declared_reads, budget),
        }
        _validate_read_coverage(before)
        # Reserve the complete second KV copy before running the model.
        reserve = sum(e["data"].numel() * e["data"].element_size() for e in before["kv_before"]["tensors"].values())
        _require(reserve < budget[0], "budget cannot hold post-forward KV and output")
        self.records.append(before)
        try:
            output = full_forward()
        except BaseException as error:
            before.update(status="FORWARD_FAILED", error=f"{type(error).__name__}: {error}")
            raise
        try:
            synchronize()
            _require(isinstance(output, (torch.Tensor, tuple, list)), "full forward returned no tensor output")
            before["output"] = self.snapshot(output, budget)
            _require(bool(before["output"]["tensors"]), "empty full-forward output")
            before["kv_after"] = self.snapshot(dict(kv_access()), budget, block_ids)
            before["write_coverage"] = _write_coverage(before["kv_before"], before["kv_after"], before["write_slots"])
            before["status"] = "CAPTURED"
        except EvidenceError as error:
            # Observation failure must not replace the successful model result.
            before.update(status="BLOCKED", error=f"{type(error).__name__}: {error}")
        except BaseException as error:
            before.update(status="CORE_RUNTIME_FAILED", error=f"{type(error).__name__}: {error}")
            raise
        return output

    def save(self, path: str | Path) -> None:
        torch.save({"schema": SCHEMA, "records": self.records, "skipped_for_limit": self.skipped_for_limit}, path)


def compare_records(left: dict, right: dict, *, atol: float = 0, rtol: float = 0, tolerance_reason: str) -> dict:
    result: dict[str, Any] = {
        "observed_forward_pair": "OPEN",
        "live_full_target": "OPEN",
        "controlled_case": "NOT_ESTABLISHED",
        "update_effective_generation": "UNKNOWN",
        "physical_read_coverage": "UNPROVEN",
        "alignment": "ALIGNMENT_REQUIRED",
        "live_readiness": "NOT_ESTABLISHED",
        "comparison_scope": "OBSERVED_FORWARD_PAIR_ONLY",
        "block_mapping_comparison": "EXACT_PHYSICAL_IDS_NO_LOGICAL_REMAP",
        "errors": [],
    }
    try:
        _require(all(math.isfinite(v) and v >= 0 for v in (atol, rtol)), "invalid tolerances")
        _require(bool(tolerance_reason.strip()), "tolerance justification required")
        for record in (left, right):
            _require(record["schema"] == SCHEMA and record["status"] == "CAPTURED", "incomplete full-forward record")
            _require(bool(record["kv_keys"]), "empty KV inventory")
            expected = {"/" + k.replace("~", "~0").replace("/", "~1") for k in record["kv_keys"]}
            for field in ("kv_before", "kv_after", "write_slots"):
                _require(set(record[field]["tensors"]) == expected, f"incomplete {field}")
            _require(bool(record["output"]["tensors"]), "missing actual-forward output")
            _validate_read_coverage(record)
        _require(left["fixture_id"] == right["fixture_id"], "fixture/model identities differ")
        _require(left["origin"] == right["origin"], "different forward boundaries")
        _require(
            left["case_kind"] == right["case_kind"]
            and left["case_kind"] in ("NATURAL_OBSERVATION", "EXPLICIT_CALLABLE_OBSERVATION"),
            "predeclared controlled cases are not implemented by this observer",
        )
        result["case_kind"] = left["case_kind"]
        _require(left["kv_scope"] == right["kv_scope"], "different KV coverage domains")
        alignment = _differences(left["inputs"], right["inputs"])
        alignment += _differences(left["kv_before"], right["kv_before"])
        alignment += _differences(left["write_slots"], right["write_slots"])
        if alignment:
            result["errors"] = ["unaligned inputs/KV: " + message for message in alignment]
            return result
        result["alignment"] = "MATCHED_COMPLETE_DECLARED_PRE_KV"
        errors = _differences(left["output"], right["output"], atol=atol, rtol=rtol)
        errors += _differences(left["kv_after"], right["kv_after"], atol=atol, rtol=rtol)
        for record in (left, right):
            coverage = _write_coverage(record["kv_before"], record["kv_after"], record["write_slots"])
            if any(entry["changed_outside_write_slots"] for entry in coverage.values()):
                errors.append("KV changed outside the declared full write slots")
        result.update(
            observed_forward_pair="FAIL" if errors else "PASS",
            errors=errors,
            atol=atol,
            rtol=rtol,
            tolerance_reason=tolerance_reason,
            kv_scope=left["kv_scope"],
            outside_selected_blocks="UNOBSERVED"
            if left["kv_scope"] == "FULL_DECLARED_READ_WRITE_BLOCKS"
            else "COVERED",
        )
        # Neither an origin label nor NPU-typed payload metadata proves a
        # predeclared controlled case. Even matched live observations stay OPEN.
    except (EvidenceError, KeyError, TypeError, AttributeError, RuntimeError) as error:
        result.update(observed_forward_pair="INVALID_EVIDENCE", errors=[str(error)])
    return result


class LiveTargetAdapter(FullForwardCheckpointer):
    """Instance-only observer for the pinned MRV2 Qwen3 dense target graph seam."""

    def __init__(self, runner: Any, *, fixture_id: str, **kwargs: Any):
        super().__init__(fixture_id=fixture_id, **kwargs)
        self.runner = runner
        self._saved: list[tuple] = []
        self._batch = None
        self._busy = False

    def _validate_runner(self) -> None:
        r = self.runner
        _require(
            _class_path(r) == "vllm_ascend.worker.v2.model_runner.NPUModelRunner"
            and _class_path(r.cudagraph_manager) == "vllm_ascend.worker.v2.aclgraph_utils.ModelAclGraphManager",
            "requires the live Ascend target ModelAclGraphManager, not the draft manager",
        )
        _require(
            _class_path(r.model) == "vllm.model_executor.models.qwen3.Qwen3ForCausalLM", "only dense Qwen3 admitted"
        )
        _require(
            _class_path(r.model_state) == "vllm_ascend.worker.v2.model_states.default.AscendModelState",
            "custom target model inputs are not inventoried",
        )
        _require(torch.device(r.device).type == "npu", "live adapter requires NPU; CPU tests are not live readiness")
        p = r.vllm_config.parallel_config
        _require(
            all(
                getattr(p, key) == 1
                for key in (
                    "tensor_parallel_size",
                    "data_parallel_size",
                    "pipeline_parallel_size",
                    "prefill_context_parallel_size",
                    "decode_context_parallel_size",
                )
            ),
            "only single-rank target checkpointing is inventoried",
        )
        _require(not r.vllm_config.scheduler_config.async_scheduling, "async runner ownership unsupported")
        _require(
            not r.uses_inputs_embeds
            and r.model_state.rope_state is None
            and r.vllm_config.quant_config is None
            and r.lora_config is None
            and r.adaptive_verification is None
            and r.pcp_manager is None
            and r.kvpp.scheduler is None,
            "embeddings/custom positions/quantization/LoRA/adaptive/PCP/KVPP unsupported",
        )
        _require(r.cudagraph_manager.model_runner is r, "target graph manager owner mismatch")
        _require(bool(r.cudagraph_manager.graphs), "install only after completed startup capture")
        _require(r.update_stream is not None, "missing graph update stream")

    def _synchronize(self) -> None:
        # Never called between graph replay and its backend update submission.
        self.runner.update_stream.synchronize()
        torch.npu.current_stream().synchronize()

    def _boundary(self, desc: Any, batch: Any) -> tuple[dict, Callable, dict, dict, dict]:
        r = self.runner
        _require(
            desc.cg_mode.name == "FULL" and desc in r.cudagraph_manager.graphs, "uncaptured target FULL descriptor"
        )
        _require(desc.num_active_loras == 0 and desc.num_ubatches == 1, "LoRA/microbatch unsupported")
        _require(batch is not None and not batch.is_dummy, "no fresh actual prepare_inputs batch")
        _require(
            batch.num_tokens_after_padding == desc.num_tokens and batch.num_reqs_after_padding == desc.num_reqs,
            "live batch/descriptor mismatch",
        )
        n, q = desc.num_tokens, desc.num_reqs
        _require(n > 0 and q > 0 and batch.num_tokens > 0, "empty batch")
        for name in ("input_ids", "positions", "is_padding"):
            _require(
                _view(getattr(batch, name)) == _view(getattr(r.input_buffers, name)[:n]),
                f"{name} is not the captured input buffer view",
            )
        inventory = target_inventory(r)
        names = inventory["target_layers"]
        context = r.compilation_config.static_forward_context
        metadata = r.model_state.attn_metadata
        _require(set(names) <= metadata.keys(), "missing target attention metadata")
        slots, layer_inputs, block_ids, declared_reads = {}, {}, {}, {}
        group_ids = {
            name: gid for gid, group in enumerate(r.kv_cache_config.kv_cache_groups) for name in group.layer_names
        }
        for name in names:
            layer, m = context[name], metadata[name]
            impl = layer.impl
            _require(
                _class_path(impl) == "vllm_ascend.attention.attention_v1.AscendAttentionBackendImpl",
                f"unmapped backend: {name}",
            )
            _require(
                not impl.use_bnsd_kv_cache
                and not impl.enable_c8_quant
                and impl.sliding_window is None
                and impl.sinks is None
                and impl.kv_sharing_target_layer_name is None,
                f"unsupported target KV layout/attention variant: {name}",
            )
            _require(m.causal and m.num_actual_tokens == batch.num_tokens, "target token/causality mismatch")
            _require(len(m.seq_lens_list) == len(m.actual_seq_lengths_q), "incomplete padded FIA read lengths")
            _require(
                _view(m.slot_mapping[: batch.num_tokens])
                == _view(r.block_tables.slot_mappings[group_ids[name], : batch.num_tokens]),
                "write slots are not the canonical captured slot-mapping view",
            )
            _require(isinstance(layer.kv_cache, (tuple, list)) and len(layer.kv_cache) == 2, "missing dense K/V pair")
            for label, tensor, cached in zip(("K", "V"), layer.kv_cache, (impl.key_cache, impl.value_cache)):
                _require(
                    isinstance(cached, torch.Tensor) and _view(tensor) == _view(cached), "stale backend KV binding"
                )
                _require(tensor.ndim == 4 and tensor.is_contiguous(), "unsupported KV storage")
                canonical_slots = r.block_tables.slot_mappings[group_ids[name], :n]
                _require(bool((canonical_slots[batch.num_tokens :] == -1).all()), "padding can write live KV slots")
                slots[f"{name}/{label}"] = canonical_slots
                block_ids[f"{name}/{label}"] = select_mapped_blocks(
                    m.block_tables,
                    m.seq_lens_list,
                    slots[f"{name}/{label}"],
                    num_blocks=tensor.shape[0],
                    block_size=tensor.shape[1],
                )
                declared_reads[f"{name}/{label}"] = {"table": m.block_tables, "lengths": m.seq_lens_list}
            layer_inputs[name] = {
                key: getattr(m, key)
                for key in (
                    "block_tables",
                    "slot_mapping",
                    "seq_lens",
                    "seq_lens_list",
                    "actual_seq_lengths_q",
                    "query_start_loc",
                    "attn_mask",
                    "attn_state",
                    "causal",
                )
            }

        def kv_access() -> dict:
            return {
                f"{name}/{label}": tensor for name in names for label, tensor in zip(("K", "V"), context[name].kv_cache)
            }

        # Match EVERY captured FIA K/V read view to the complete live target
        # inventory. No layer-name/order guess, and no partial KV sampling.
        params = importlib.import_module("vllm_ascend.compilation.acl_graph").get_graph_params()
        captured = params.attn_params[n]
        _require(len(captured) == len(names) == len(params.handles[n]) == len(params.events[n]), "incomplete FIA tasks")
        expected = {}
        for name in names:
            k, v = context[name].kv_cache
            signature = (_view(k.view(k.shape[0], k.shape[1], -1)), _view(v.view(v.shape[0], v.shape[1], -1)))
            _require(signature not in expected, "shared target KV needs separate writer attribution")
            expected[signature] = name
        seen = set()
        for param in captured:
            # full_graph_fia: 17 base operands, four quantization slots, layer.
            # full_graph_fia_v2/PA and future tuple revisions are not this schema.
            _require(isinstance(param, tuple) and len(param) == 22, "unknown captured FIA parameter schema")
            _require(
                all(isinstance(param[i], torch.Tensor) for i in (0, 1, 2, 3, 11, 12))
                and (param[4] is None or isinstance(param[4], torch.Tensor)),
                "invalid captured FIA tensor fields",
            )
            signature = (_view(param[1]), _view(param[2]))
            _require(signature in expected and signature not in seen, "captured KV read inventory mismatch")
            name = expected[signature]
            impl = context[name].impl
            _require(
                param[5] == context[name].kv_cache[0].shape[1]
                and param[8:11] == (impl.num_kv_heads, impl.num_heads, impl.scale)
                and param[16] is None
                and all(value is None for value in param[17:21])
                and param[21] in (None, name),
                "captured FIA geometry/quantization/layer schema mismatch",
            )
            _require(
                all(
                    isinstance(param[i], list) and param[i] and all(type(v) is int and v >= 0 for v in param[i])
                    for i in (6, 7)
                )
                and len(param[6]) == len(param[7])
                and param[7][-1] == n
                and all(type(param[i]) is int for i in (13, 14, 15)),
                "invalid captured FIA sequence/sparse fields",
            )
            seen.add(signature)
        _require(seen == expected.keys(), "uncaptured target KV layer")
        inputs = {
            "input_ids": batch.input_ids,
            "positions": batch.positions,
            "query_start_loc": batch.query_start_loc,
            "seq_lens": batch.seq_lens,
            "is_padding": batch.is_padding,
            "idx_mapping": batch.idx_mapping,
            "req_ids": tuple(batch.req_ids),
            "num_tokens": batch.num_tokens,
            "num_reqs": batch.num_reqs,
            "descriptor": (n, q, desc.uniform_token_count, desc.max_query_len),
            "num_computed_tokens": torch.as_tensor(batch.num_computed_tokens_np),
            "layers": layer_inputs,
            "selected_block_ids": block_ids,
            "target_layer_names": names,
            "pinned_vllm_contract": PINNED_VLLM,
            "inventory": inventory,
        }
        return inputs, kv_access, slots, block_ids, declared_reads

    def __enter__(self) -> LiveTargetAdapter:
        _require(not self._saved, "adapter already installed")
        self._validate_runner()
        r, adapter = self.runner, self
        prepare, forward = r.prepare_inputs, r.cudagraph_manager.run_fullgraph

        def prepare_inputs(_owner, *args, **kwargs):
            result = prepare(*args, **kwargs)
            adapter._batch = result
            return result

        def run_fullgraph(_owner, desc):
            batch, adapter._batch = adapter._batch, None
            if adapter._busy:
                raise EvidenceError("concurrent/reentrant target observer")
            adapter._busy = True
            called = False

            def actual_forward():
                nonlocal called
                called = True
                return forward(desc)

            try:
                if len(adapter.records) >= adapter.max_records:
                    adapter.skipped_for_limit += 1
                    return actual_forward()
                adapter._synchronize()
                inputs, kv_access, slots, block_ids, declared_reads = adapter._boundary(desc, batch)
                return adapter.run(
                    actual_forward,
                    inputs=inputs,
                    kv_access=kv_access,
                    write_slots=slots,
                    synchronize=adapter._synchronize,
                    origin="LIVE_ASCEND_TARGET_FULL_GRAPH",
                    block_ids=block_ids,
                    declared_reads=declared_reads,
                )
            except (EvidenceError, AttributeError, KeyError) as error:
                if called:
                    raise  # Preserve the original forward's exception.
                if len(adapter.records) < adapter.max_records:
                    adapter.records.append(
                        {"schema": SCHEMA, "status": "BLOCKED", "error": f"{type(error).__name__}: {error}"}
                    )
                return actual_forward()  # An unsupported observation never skips a real model call.
            finally:
                adapter._busy = False

        for owner, name, replacement in (
            (r, "prepare_inputs", prepare_inputs),
            (r.cudagraph_manager, "run_fullgraph", run_fullgraph),
        ):
            self._saved.append((owner, name, name in vars(owner), vars(owner).get(name)))
            setattr(owner, name, MethodType(replacement, owner))
        return self

    def __exit__(self, *_exc) -> None:
        for owner, name, existed, value in reversed(self._saved):
            if existed:
                setattr(owner, name, value)
            else:
                delattr(owner, name)
        self._saved.clear()
        self._batch = None


def target_inventory(runner: Any) -> dict:
    """Derive execution ownership from model objects, not the shared registry size.

    Qwen3ForCausalLM owns its Attention modules. The registry and AttentionGroups
    also contain the independently constructed draft model. Captured K/V tasks
    are checked separately by _boundary; module membership alone is not capture.
    """
    context = runner.compilation_config.static_forward_context
    configured_list = [n for g in runner.kv_cache_config.kv_cache_groups for n in g.layer_names]
    grouped = {n for groups in runner.attn_groups for g in groups for n in g.layer_names}
    _require(len(configured_list) == len(set(configured_list)), "duplicate configured layer")
    configured = set(configured_list)
    _require(grouped == configured and configured <= context.keys(), "registry/config/group mismatch")
    modules = {id(module) for module in runner.model.modules()}
    names = {name for name in configured if id(context[name]) in modules}
    draft = set(getattr(runner.speculator, "draft_attn_layer_names", ()))
    _require(bool(names) and not names & draft, "empty target or draft bound to target model")
    _require(names | draft == configured, "unowned configured layer or missing target module")
    _require(len({id(context[name]) for name in names}) == len(names), "aliased target layer registration")
    return {
        "registered_layers": sorted(context),
        "configured_layers": sorted(configured),
        "draft_layers": sorted(draft),
        "target_layers": sorted(names),
        "ownership_basis": "actual_target_model_module_identity",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--atol", type=float, default=0)
    parser.add_argument("--rtol", type=float, default=0)
    parser.add_argument("--tolerance-reason", required=True)
    parser.add_argument("--output", type=Path, help="write the same comparison JSON printed to stdout")
    args = parser.parse_args()
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "observed_forward_pair": "OPEN",
        "live_full_target": "OPEN",
        "controlled_case": "NOT_ESTABLISHED",
        "live_readiness": "NOT_ESTABLISHED",
        "alignment": "ALIGNMENT_REQUIRED",
        "update_effective_generation": "UNKNOWN",
        "block_mapping_comparison": "EXACT_PHYSICAL_IDS_NO_LOGICAL_REMAP",
        "pairs": [],
        "confirmed_failure_count": 0,
        "evidence_complete": False,
    }
    try:
        a, b = (torch.load(path, map_location="cpu", weights_only=True) for path in (args.left, args.right))
        _require(a.get("schema") == b.get("schema") == SCHEMA, "invalid collection schema")
        omissions = bool(
            a.get("skipped_for_limit")
            or b.get("skipped_for_limit")
            or not a.get("records")
            or len(a["records"]) != len(b.get("records", []))
        )
        if omissions:
            report["reason"] = "missing/unpaired records or bounded collection omissions"
        results = [
            compare_records(x, y, atol=args.atol, rtol=args.rtol, tolerance_reason=args.tolerance_reason)
            for x, y in zip(a.get("records", []), b.get("records", []))
        ]
        statuses = [result["observed_forward_pair"] for result in results]
        report["pairs"] = results
        report["confirmed_failure_count"] = statuses.count("FAIL")
        report["evidence_complete"] = bool(results) and not omissions and all(s in ("PASS", "FAIL") for s in statuses)
        report["evidence_issues"] = {
            "invalid_pairs": statuses.count("INVALID_EVIDENCE"),
            "open_pairs": statuses.count("OPEN"),
            "collection_omissions": omissions,
        }
        # A proven aligned difference survives unrelated missing/invalid evidence.
        candidates = set(statuses) | ({"OPEN"} if omissions or not results else set())
        report["observed_forward_pair"] = next(
            status for status in ("FAIL", "INVALID_EVIDENCE", "OPEN", "PASS") if status in candidates
        )
        if report["evidence_complete"]:
            report["alignment"] = "MATCHED_COMPLETE_DECLARED_PRE_KV"
    except Exception as error:
        report.update(observed_forward_pair="INVALID_EVIDENCE", reason=f"{type(error).__name__}: {error}")
    payload = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        _require(
            args.output.resolve() not in (args.left.resolve(), args.right.resolve()), "output would overwrite input"
        )
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["observed_forward_pair"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
