# SPDX-License-Identifier: Apache-2.0
"""Versioned, bounded tensor evidence. No pickle execution or live restoration.

Tensor bytes are streamed in <=8 MiB pieces to immutable content-addressed
files. JSON contains view/alias provenance, chunk offsets and hashes, never
tensor values. Readers mmap CPU payloads and validate each chunk before use.
Incomplete sample manifests remain visible and cannot authorize a PASS.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
import weakref
from pathlib import Path

import torch

SCHEMA = "repair6-tensor-store-v1"
CHUNK = 8 << 20


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(CHUNK), b""):
            h.update(data)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def safe_path(root, name):
    p = Path(name)
    require(not p.is_absolute() and ".." not in p.parts, "unsafe evidence path")
    target = (root / p).resolve()
    require(target.is_relative_to(root.resolve()), "evidence path escapes root")
    return target


class TensorStore:
    def __init__(self, root, *, max_bytes=32 << 30, timeout=3600, chunk_bytes=CHUNK, readonly=False):
        self.root = Path(root)
        self.readonly = readonly
        if readonly:
            require(self.root.is_dir(), "missing tensor store")
        else:
            self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes, self.deadline = max_bytes, time.monotonic() + timeout
        self.chunk_bytes = chunk_bytes
        require(chunk_bytes > 0 and max_bytes > 0, "invalid storage budget")
        self.used = sum(p.stat().st_size for p in self.root.glob("*.bin"))
        self.refs = {}

    def _budget(self, additional=0):
        require(time.monotonic() < self.deadline, "storage deadline exceeded")
        require(self.used + additional <= self.max_bytes, "BLOCKED_RESOURCE: store bytes")

    @staticmethod
    def key(t):
        return (t.untyped_storage().data_ptr(), t.storage_offset(), tuple(t.shape), tuple(t.stride()), str(t.dtype))

    def put(self, tensor, selected=None):
        require(not self.readonly, "read-only evidence store")
        require(tensor.layout == torch.strided and tensor.numel() > 0, "invalid tensor")
        shape = tuple(tensor.shape) if selected is None else (len(selected), *tensor.shape[1:])
        total = math.prod(shape) * tensor.element_size()
        self._budget(total)
        temporary = self.root / f"tensor-{os.getpid()}.partial"
        require(not temporary.exists(), "unfinished tensor write; use a new attempt")
        hasher, chunks, offset = hashlib.sha256(), [], 0
        # No all-KV device index_select and no whole-tensor CPU copy. Preserve
        # deterministic row/element order even when logical blocks are sparse.
        require(tensor.is_contiguous(), "streaming store requires contiguous view")
        pieces = (tensor[i].reshape(-1) for i in selected) if selected is not None else (tensor.reshape(-1),)
        if tensor.ndim == 0:
            require(selected is None, "cannot select scalar blocks")
        step = max(1, self.chunk_bytes // tensor.element_size())
        with temporary.open("xb") as f:
            for row in pieces:
                for start in range(0, row.numel(), step):
                    self._budget(total)
                    cpu = row[start : start + step].detach().to("cpu", copy=True).contiguous()
                    data = memoryview(cpu.view(torch.uint8).numpy())
                    f.write(data)
                    hasher.update(data)
                    chunks.append(dict(offset=offset, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
                    offset += len(data)
        require(offset == total, "incomplete tensor write")
        sha = hasher.hexdigest()
        target = self.root / (sha + ".bin")
        if target.exists():
            require(target.stat().st_size == total and digest(target) == sha, "content-address collision/corruption")
            temporary.unlink()
        else:
            temporary.replace(target)
            self.used += total
        ref = dict(path=target.name, bytes=total, sha256=sha, chunks=chunks, shape=shape, dtype=str(tensor.dtype))
        mapped = self.read_tensor(ref)
        self.refs[id(mapped)] = (weakref.ref(mapped), ref)
        return mapped

    def encode(self, value):
        if isinstance(value, torch.Tensor):
            saved = self.refs.get(id(value))
            ref = saved[1] if saved is not None and saved[0]() is value else None
            if ref is None:
                mapped = self.put(value)
                ref = self.refs[id(mapped)][1]
            return {"tensor_ref": ref}
        if isinstance(value, dict):
            return {"mapping": [[self.encode(k), self.encode(v)] for k, v in value.items()]}
        if isinstance(value, (tuple, list)):
            return {"tuple" if isinstance(value, tuple) else "list": [self.encode(v) for v in value]}
        require(value is None or type(value) in (int, float, str, bool), "unsupported evidence object")
        require(not isinstance(value, float) or math.isfinite(value), "nonfinite metadata")
        return value

    def read_tensor(self, ref):
        path = safe_path(self.root, ref["path"])
        dtype = getattr(torch, ref["dtype"].removeprefix("torch."), None)
        require(isinstance(dtype, torch.dtype), "unknown dtype")
        n = math.prod(ref["shape"])
        require(n > 0 and n * torch.empty((), dtype=dtype).element_size() == ref["bytes"], "tensor geometry mismatch")
        require(path.stat().st_size == ref["bytes"], "tensor truncated")
        offset, h = 0, hashlib.sha256()
        with path.open("rb") as f:
            for c in ref["chunks"]:
                require(c["offset"] == offset and 0 < c["bytes"] <= self.chunk_bytes, "invalid chunk coverage")
                data = f.read(c["bytes"])
                require(
                    len(data) == c["bytes"] and hashlib.sha256(data).hexdigest() == c["sha256"], "chunk hash mismatch"
                )
                h.update(data)
                offset += len(data)
        require(offset == ref["bytes"] and h.hexdigest() == ref["sha256"], "missing chunks/whole hash mismatch")
        return torch.from_file(str(path), shared=False, size=n, dtype=dtype).reshape(ref["shape"])

    def decode(self, value):
        if not isinstance(value, dict):
            return value
        require(len(value) == 1, "unknown encoded object")
        if "tensor_ref" in value:
            return self.read_tensor(value["tensor_ref"])
        if "mapping" in value:
            pairs = [(self.decode(k), self.decode(v)) for k, v in value["mapping"]]
            require(len(dict(pairs)) == len(pairs), "duplicate mapping key")
            return dict(pairs)
        if "tuple" in value:
            return tuple(self.decode(v) for v in value["tuple"])
        require("list" in value, "unknown encoded type")
        return [self.decode(v) for v in value["list"]]


class CollectionWriter:
    def __init__(self, path, header, **limits):
        self.path = Path(path)
        require(not self.path.exists(), "collection exists; use new attempt")
        self.store = TensorStore(self.path.parent / (self.path.stem + "-tensors"), **limits)
        self.manifest = dict(
            schema=SCHEMA, header=header, samples=[], execution="IN_PROGRESS", tensor_root=self.store.root.name
        )
        self._flush()

    def _flush(self):
        atomic_json(self.path, self.manifest)

    def start(self, identity):
        require(all(s["identity"] != identity for s in self.manifest["samples"]), "duplicate sample")
        self.manifest["samples"].append(dict(identity=identity, status="PREPARING", path=None))
        self._flush()

    def finish(self, record):
        sample = self.manifest["samples"][-1]
        require(sample["status"] == "PREPARING" and record["identity"] == sample["identity"], "wrong sample finish")
        name = f"{self.path.stem}-sample-{len(self.manifest['samples']) - 1:03}.json"
        path = self.path.parent / name
        payload = {k: v for k, v in record.items() if k not in ("logical_pre_kv", "logical_post_kv")}
        # Those maps contain views of the checkpoint payload, not new evidence.
        # Reconstruct them from the retained physical/logical ownership mapping.
        payload["derived_logical_kv"] = "FROM_COMPLETE_CHECKPOINT_AND_BLOCK_OWNERS"
        atomic_json(path, self.store.encode(payload))
        sample.update(status=record["status"], path=name, bytes=path.stat().st_size, sha256=digest(path))
        self._flush()

    def close(self, execution):
        self.manifest["execution"] = execution
        self._flush()


class CollectionReader:
    def __init__(self, path):
        self.path = Path(path)
        self.legacy = self.path.suffix == ".pt"
        if self.legacy:
            self.document = torch.load(self.path, map_location="cpu", weights_only=True, mmap=True)
            require(self.document["schema"] == "repair5-controlled-target-v1", "unknown legacy schema")
            self.header = {k: v for k, v in self.document.items() if k != "records"}
            self.samples = [dict(identity=r["identity"], status=r["status"]) for r in self.document["records"]]
            self.execution = "LEGACY_PARTIAL_OR_COMPLETE_SEE_LAUNCHER"
        else:
            m = json.loads(self.path.read_text())
            require(m["schema"] == SCHEMA, "unknown store version")
            self.header, self.samples, self.execution = m["header"], m["samples"], m["execution"]
            self.store = TensorStore(safe_path(self.path.parent, m["tensor_root"]), readonly=True)
        ids = [json.dumps(s["identity"], sort_keys=True) for s in self.samples]
        require(len(ids) == len(set(ids)), "duplicate sample identity")

    def record(self, index):
        if self.legacy:
            return self.document["records"][index]
        s = self.samples[index]
        require(s["path"] is not None, "sample incomplete")
        path = safe_path(self.path.parent, s["path"])
        require(path.stat().st_size == s["bytes"] and digest(path) == s["sha256"], "sample hash mismatch")
        r = self.store.decode(json.loads(path.read_text()))
        require(r["identity"] == s["identity"] and r["status"] == s["status"], "manifest/sample mismatch")
        if r.get("checkpoint", {}).get("status") == "CAPTURED":
            for field, target in (("kv_before", "logical_pre_kv"), ("kv_after", "logical_post_kv")):
                r[target] = {
                    name: {json.dumps(r["block_owners"][b]): e["data"][i] for i, b in enumerate(e["block_ids"])}
                    for name, e in r["checkpoint"][field]["tensors"].items()
                }
        return r


def resource_plan(cases, *, layers, kv_heads, head_dim, element_bytes, hidden_size, vocab_size, mask_bytes):
    """Conservative no-content-dedup bound, derived from complete block union.

    Four KV copies allow before/after/after-draft plus serializer/mmap margin.
    Shared mask appears once in each of the two metadata checkpoints.
    """
    rows = []
    for c in cases:
        blocks = sum((e + 127) // 128 for e in c["ends"]) + int(c["padded"] > len(c["ends"]))
        kv = layers * 2 * blocks * 128 * kv_heads * head_dim * element_bytes
        outputs = c["padded"] * 8 * (hidden_size * 8 + vocab_size) * element_bytes
        logical = 4 * kv + 2 * mask_bytes + outputs + (8 << 20)
        rows.append(
            dict(
                scenario=c["scenario"],
                blocks=blocks,
                kv_one_copy=kv,
                checkpoint=2 * kv + mask_bytes + outputs + (8 << 20),
                sample_bound=logical,
            )
        )
    total = sum(r["sample_bound"] for r in rows)
    return dict(
        samples=rows,
        one_arm_bytes=total,
        off_on_with_raw_and_atomic_bytes=4 * total + CHUNK,
        host_temporary_bytes=CHUNK,
        device_copy_temporary_bytes=CHUNK,
        mapped_working_set="OS paged; per-layer comparator; not a guaranteed process RSS bound",
    )


def require_space(path, needed):
    available = shutil.disk_usage(path).free
    require(
        available >= needed, f"BLOCKED_RESOURCE: need {needed}, available {available}, deficit {needed - available}"
    )
