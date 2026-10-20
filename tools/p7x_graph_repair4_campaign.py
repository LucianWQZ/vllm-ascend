# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Repair4 plan/receiver tools; no device execution or publication on import.

Commands are argv arrays, never shell-evaluated. Execution uses the bounded
Repair2 launcher. A generated plan is not permission to run an unpublished SHA.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight(handoff: dict, vllm: Path, *, require_npu: bool = False) -> dict:
    root = Path(__file__).resolve().parents[1]

    def git(repo: Path, *args: str) -> str:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

    repositories = {}
    for name, repo, sha, tree in (
        ("ascend", root, handoff["commit"], handoff["tree"]),
        ("vllm", vllm, handoff["vllm"]["commit"], handoff["vllm"]["tree"]),
    ):
        actual = {
            "head": git(repo, "rev-parse", "HEAD"),
            "tree": git(repo, "rev-parse", "HEAD^{tree}"),
            "status": git(repo, "status", "--porcelain", "--untracked-files=all"),
        }
        if actual != {"head": sha, "tree": tree, "status": ""}:
            raise ValueError(f"{name} exact/clean check failed: {actual}")
        repositories[name] = actual
    imports = {}
    for name in (
        "vllm",
        "vllm_ascend.worker.v2.model_runner",
        "vllm_ascend.worker.v2.spec_decode.dspark.speculator",
        "vllm_ascend.worker.v2.spec_decode.dflash.aclgraph",
        "vllm_ascend.worker.v2.spec_decode.draft_window_graph_characterization",
    ):
        module = importlib.import_module(name)
        if module.__file__ is None:
            raise ValueError(f"import has no source file: {name}")
        path = Path(module.__file__).resolve()
        repo = vllm if name == "vllm" else root
        relative = name.replace(".", "/") + ("/__init__.py" if name == "vllm" else ".py")
        expected = subprocess.check_output(["git", "-C", str(repo), "show", f"HEAD:{relative}"])
        if hashlib.sha256(expected).hexdigest() != digest(path):
            raise ValueError(f"import content differs from pinned source: {name}")
        imports[name] = {"path": str(path), "sha256": digest(path)}
    for name, expected in handoff["tools_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError(f"tool hash mismatch: {name}")
    npu: str | dict = "NOT_REQUESTED"
    if require_npu:
        import torch
        import torch_npu

        if not torch.npu.is_available():
            raise RuntimeError("NPU unavailable; no CPU fallback")
        npu = {"torch": torch.__version__, "torch_npu": torch_npu.__version__, "device": torch.npu.get_device_name()}
    return {
        "source": repositories,
        "imports": imports,
        "npu": npu,
        "device_functional_gates": "NOT_RUN",
        "publication_verified_by_this_command": False,
    }


def make_plan(handoff: dict, inputs: Path, output: Path, vllm: Path, run_id: str) -> dict:
    """Expand the frozen recipe without loading models or touching an allocator."""
    sha = handoff["commit"]
    fixture = inputs / "fixture_manifest.json"
    if digest(fixture) != handoff["fixture_manifest_sha256"]:
        raise ValueError("fixture manifest mismatch")
    recipes: dict[str, tuple[str, str, bool | str, list[str]]] = {
        "A0": ("A0", "smoke.json", False, []),
        "A1": ("A1", "smoke.json", False, ["A0"]),
        "B0": ("B", "smoke.json", False, ["A0", "A1"]),
        "B1": ("B", "combined.json", False, ["B0"]),
        "B2": ("B", "mixed.json", False, ["B0"]),
        "O0": ("B-observer-off", "smoke.json", False, ["A0"]),
        "O1": ("B-observer-off", "combined.json", False, ["A0"]),
        "T0": ("B-observer-off", "smoke.json", False, ["A0"]),
        "T1": ("B", "smoke.json", False, ["B0"]),
    }
    for name in ("short", "long1", "long2", "mixed"):
        request = {
            "short": "smoke.json",
            "long1": "g4-long-group1.json",
            "long2": "g4-long-group2.json",
            "mixed": "mixed.json",
        }[name]
        recipes[f"G4-OFF-{name}"] = ("A0", request, True, ["A0"])
        recipes[f"G4-ON-{name}"] = ("B", request, True, ["B0"])
        recipes[f"G4F-OFF-{name}"] = ("A0", request, "full", ["A0"])
        recipes[f"G4F-ON-{name}"] = ("B", request, "full", ["B0"])
    cases = {}
    for case, (mode, request, probe, dependencies) in recipes.items():
        request_path = inputs / "requests" / request
        if not request_path.is_file():
            raise ValueError(f"missing frozen requests: {request}")
        argv = [
            sys.executable,
            "tools/p7x_graph_repair2.py",
            "run",
            "--case",
            mode,
            "--case-id",
            case,
            "--run-id",
            run_id,
            "--attempt",
            "1",
            "--expected-sha",
            sha,
            "--vllm-repo",
            str(vllm),
            "--fixture-manifest",
            str(fixture),
            "--target-model",
            "/mnt/share/wql/Qwen3-8B",
            "--draft-model",
            "/mnt/share/dspark_qwen3_8b_block7",
            "--requests",
            str(request_path),
            "--output",
            str(output / case),
        ]
        if probe == "full":
            argv += ["--full-target-probe", "--full-target-max-records", "8"]
        elif probe:
            argv += ["--target-probe"]
        cases[case] = {
            "argv": argv,
            "acceptance_identity": {
                "run_id": run_id,
                "case_id": case,
                "attempt_id": "1",
                "driver_mode": mode,
                "source_sha": sha,
                "source_tree": handoff["tree"],
            },
            "depends_on": dependencies,
            "launcher_argv": [
                sys.executable,
                "tools/p7x_graph_repair2.py",
                "launch",
                "--output",
                str(output / f"{case}-launcher"),
                "--run-id",
                run_id,
                "--case-id",
                case,
                "--attempt",
                "1",
                "--mode",
                mode,
                "--timeout",
                "1800",
                "--",
                *argv,
            ],
            "requests_sha256": digest(request_path),
            "joint_acceptance_argv": [
                sys.executable,
                "tools/p7x_graph_repair4_acceptance.py",
                "check-joint",
                str(output / case / "campaign.json"),
                "--launcher",
                str(output / f"{case}-launcher" / "launcher.json"),
                "--requests",
                str(request_path),
                "--plan",
                "REPLACE_WITH_SAVED_PLAN_PATH",
                "--case-id",
                case,
                "--output",
                str(output / f"{case}-joint.json"),
            ],
            "status": "NOT_RUN",
            "time_limit_seconds": 1800,
            "evidence_limit": (
                "complete selected KV/full-forward checkpoint attempts; aligned pair must be proven"
                if probe == "full"
                else "HEAD_ONLY for target-probe; NOT FULL_TARGET"
            ),
        }
    for case, abort_at in (
        ("C2", "none"),
        ("ABORT-before", "before_replay"),
        ("ABORT-inflight", "after_replay"),
        ("ABORT-update", "after_update"),
        ("ABORT-complete", "after_completion"),
    ):
        cases[case] = {
            "argv": [
                sys.executable,
                "tools/p7x_graph_repair4_fixture.py",
                "run",
                "--run-id",
                run_id,
                "--attempt",
                case,
                "--output",
                str(output / case),
                "--timeout",
                "180",
                "--abort-at",
                abort_at,
            ],
            "depends_on": ["P0"],
            "status": "NOT_RUN",
            "time_limit_seconds": 180,
            "evidence_limit": "private P3/P6/FIA fixture; not complete MRV2; diagnostic synchronized",
        }
    return {
        "schema_version": "repair4-campaign-plan-v1",
        "commit": sha,
        "tree": handoff["tree"],
        "run_id": run_id,
        "publication_status": handoff["publication_status"],
        "execution_authorized": False,
        "precondition": "separate publication + independent network checkout verification required",
        "cases": cases,
        "stop_rule": (
            "core binding/device error stops dependent cases; independent private fixture only if device healthy"
        ),
        "limits": {
            "max_case_output_bytes": 2147483648,
            "minimum_free_disk_gib": 20,
            "shared_device_contention": "STOP",
            "device_family": "A3",
        },
    }


def pack(root: Path, output: Path) -> None:
    """Package a bounded campaign directory with a receiver-verifiable index."""
    root, output = root.resolve(), output.resolve()
    if output.is_relative_to(root) or output.exists():
        raise ValueError("output must be new and outside input directory")
    files = sorted(p for p in root.rglob("*") if p.is_file())
    if not files or any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("empty input or symlink is not accepted")
    if any(p.name == "REPAIR4_PACKAGE_INDEX.json" for p in files):
        raise ValueError("reserved index name present")
    if sum(p.stat().st_size for p in files) > 8 * 1024**3:
        raise ValueError("package exceeds 8 GiB; split explicitly and retain indexed original bytes")
    index = {str(p.relative_to(root)): {"bytes": p.stat().st_size, "sha256": digest(p)} for p in files}
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, str(p.relative_to(root)))
        z.writestr("REPAIR4_PACKAGE_INDEX.json", json.dumps(index, indent=2))


def verify(path: Path) -> dict:
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or z.testzip() is not None:
            raise ValueError("duplicate member or invalid ZIP CRC")
        if any(Path(n).is_absolute() or ".." in Path(n).parts for n in names):
            raise ValueError("unsafe member path")
        index = json.loads(z.read("REPAIR4_PACKAGE_INDEX.json"))
        if set(index) != set(names) - {"REPAIR4_PACKAGE_INDEX.json"}:
            raise ValueError("unindexed or missing members")
        for name, entry in index.items():
            value = z.read(name)
            if len(value) != entry["bytes"] or hashlib.sha256(value).hexdigest() != entry["sha256"]:
                raise ValueError(f"member mismatch: {name}")
            if name.endswith(".json"):
                json.loads(value)
    return {
        "integrity": "VERIFIED",
        "members": len(index),
        "sha256": digest(path),
        "device_correctness": "NOT_ASSESSED",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    for key in ("handoff", "inputs", "output-dir", "vllm-repo", "output"):
        plan.add_argument(f"--{key}", type=Path, required=True)
    plan.add_argument("--run-id", required=True)
    packing = sub.add_parser("pack")
    packing.add_argument("--root", type=Path, required=True)
    packing.add_argument("--output", type=Path, required=True)
    checking = sub.add_parser("verify")
    checking.add_argument("package", type=Path)
    pre = sub.add_parser("preflight")
    pre.add_argument("--handoff", type=Path, required=True)
    pre.add_argument("--vllm-repo", type=Path, required=True)
    pre.add_argument("--output", type=Path, required=True)
    pre.add_argument("--require-npu", action="store_true")
    args = parser.parse_args()
    if args.command == "plan":
        result = make_plan(
            json.loads(args.handoff.read_text()),
            args.inputs.resolve(),
            args.output_dir.resolve(),
            args.vllm_repo.resolve(),
            args.run_id,
        )
        for entry in result["cases"].values():
            if "joint_acceptance_argv" in entry:
                entry["joint_acceptance_argv"] = [
                    str(args.output.resolve()) if a == "REPLACE_WITH_SAVED_PLAN_PATH" else a
                    for a in entry["joint_acceptance_argv"]
                ]
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    elif args.command == "pack":
        pack(args.root, args.output)
    elif args.command == "preflight":
        result = preflight(json.loads(args.handoff.read_text()), args.vllm_repo, require_npu=args.require_npu)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    else:
        print(json.dumps(verify(args.package), indent=2))


if __name__ == "__main__":
    main()
