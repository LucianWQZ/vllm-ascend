#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed, administrator-assisted physical device admission.

inspect never initializes torch/NPU. A mapping receipt must be supplied by the
host/container allocation owner, with raw evidence hashes. This is not automatic
discovery of host physical IDs from container logical numbers.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

if __name__ == "__main__":
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).parent.resolve()]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.p7x_graph_repair5_evidence import file_sha as digest
from tools.p7x_graph_repair5_evidence import require


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def inspect():
    nodes = {}
    for p in Path("/dev").glob("davinci*"):
        st = p.stat()
        nodes[str(p)] = dict(major=os.major(st.st_rdev), minor=os.minor(st.st_rdev))
    result = dict(
        pid=os.getpid(),
        parent_pid=os.getppid(),
        time_ns=time.time_ns(),
        nodes=nodes,
        visibility={k: os.environ.get(k) for k in ("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES")},
        namespace=os.readlink("/proc/self/ns/mnt") if Path("/proc/self/ns/mnt").exists() else None,
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        if Path("/proc/sys/kernel/random/boot_id").exists()
        else None,
    )
    try:
        p = subprocess.run(["npu-smi", "info", "-m"], capture_output=True, text=True, timeout=15)
        result["mapping_command"] = dict(
            argv=["npu-smi", "info", "-m"], rc=p.returncode, stdout=p.stdout, stderr=p.stderr
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        result["mapping_error"] = str(error)
    result["status"] = "READ_ONLY_INVENTORY_NOT_ADMISSION"
    return result


def validate_receipt(receipt, current, run_id, *, now=None, require_probe=True):
    now = time.time() if now is None else now
    require(receipt["schema"] == "repair7-device-receipt-v1", "unknown receipt schema")
    require(receipt["run_id"] == run_id and receipt["allocation_owner"].strip(), "allocation identity")
    require(receipt["issued_at"] <= now < receipt["expires_at"] <= receipt["issued_at"] + 3600, "expired receipt")
    require(type(receipt["host_phy_id"]) is int and 0 <= receipt["host_phy_id"] <= 7, "unauthorized physical device")
    require(receipt["tp"] == 1 and len(receipt["allocated_nodes"]) == 1, "TP1 requires one allocated chip")
    require(current["namespace"] is not None and current["boot_id"] is not None, "unknown host/container identity")
    for key in ("namespace", "boot_id", "visibility"):
        require(receipt[key] == current[key], "mapping changed: " + key)
    for node, geometry in receipt["allocated_nodes"].items():
        require(current["nodes"].get(node) == geometry, "device node changed")
    require(receipt["mapping_evidence"] and receipt["resource_evidence"], "missing owner mapping/resource evidence")
    for row in receipt["mapping_evidence"] + receipt["resource_evidence"]:
        require(digest(row["path"]) == row["sha256"], "changed admission evidence")
    if require_probe:
        probe = receipt["probe"]
        require(probe["host_pid"] > 0 and probe["container_pid"] > 0 and probe["command"], "missing own probe PID")
        require(probe["start_ns"] < probe["end_ns"] and probe["host_phy_id"] == receipt["host_phy_id"], "probe mapping")
        require(probe["status"] == "EXITED" and probe["returncode"] == 0, "probe incomplete")
        require(probe["host_process_evidence"] and probe["pid_namespace_evidence"], "cannot link probe to host PID")
        for row in (probe["host_process_evidence"], probe["pid_namespace_evidence"]):
            require(digest(row["path"]) == row["sha256"], "probe evidence changed")
    return dict(
        status="ADMITTED_BY_OWNER_MAPPING_RECEIPT",
        physical_id=receipt["host_phy_id"],
        proof_boundary="owner supplied mapping/PID evidence; not independently inferred from logical ID",
    )


def validate_admission(plan, case_id):
    require(
        bool(plan.get("device_admission_receipt")) and bool(plan.get("device_admission_sha256")),
        "BLOCKED_DEVICE_MAPPING: no current admission receipt",
    )
    path = Path(plan["device_admission_receipt"])
    require(digest(path) == plan["device_admission_sha256"], "receipt changed")
    receipt = json.loads(path.read_text())
    result = validate_receipt(receipt, inspect(), plan["run_id"])
    require(receipt["source_sha"] == plan["commit"] and receipt["case_id"] == case_id, "new case needs fresh admission")
    require(receipt["source_tree"] == plan["tree"], "receipt source tree mismatch")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("inspect", "check", "probe"))
    p.add_argument("--receipt", type=Path)
    p.add_argument("--run-id")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    inventory = inspect()
    if a.command == "inspect":
        result = inventory
    else:
        if not a.receipt or not a.run_id:
            p.error("receipt and run-id required")
        receipt = json.loads(a.receipt.read_text())
        result = validate_receipt(receipt, inventory, a.run_id, require_probe=a.command != "probe")
        if a.command == "probe":
            # Only AFTER read-only allocation mapping is attested. No model.
            import torch
            import torch_npu  # noqa: F401

            result.update(pid=os.getpid(), command=sys.argv, start_ns=time.time_ns(), inventory=inventory)
            atomic_json(a.output, dict(result, status="PROBE_STARTED_NEEDS_HOST_PID_CORRELATION"))
            value = torch.ones(1024, device="npu:0")
            result.update(
                value=float(value.sum().cpu()),
                end_ns=time.time_ns(),
                status="PROBE_FINISHED_NEEDS_HOST_PID_CORRELATION",
            )
    atomic_json(a.output, result)


if __name__ == "__main__":
    main()
