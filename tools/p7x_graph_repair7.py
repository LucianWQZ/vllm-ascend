#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Repair7 model-free configuration and full Target external execution join."""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).parent.resolve()]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.p7x_graph_repair2 import build_llm_kwargs, configure_run_environment, parse_args
from tools.p7x_graph_repair6_storage import CollectionReader, atomic_json, digest, require


def config_matrix(plan, *, framework=False):
    """Use the actual child CLI and LLM kwargs; framework mode loads config, not LLM."""
    try:
        from vllm_ascend.ascend_config import AscendConfig
    except (ImportError, RuntimeError, OSError) as error:
        return dict(status="ENVIRONMENT_BLOCKED", rows=[], error=str(error), framework="NOT_RUN")

    results = []
    for name, case in plan["cases"].items():
        argv = case["argv"]
        if not any(Path(a).name == "p7x_graph_repair2.py" for a in argv) or "run" not in argv:
            continue
        row = dict(case_id=name, status="INVALID_CONFIG", framework="NOT_RUN")
        try:
            args = parse_args(argv[argv.index("run") :])
            config = build_llm_kwargs(args)
            row["kwargs"] = config
            row["config_sha256"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
            typed = AscendConfig(sparse_kv_offload_config=None, **config["additional_config"])
            row["numeric_enabled"] = typed.draft_window_graph_numeric_diagnostic
            if framework:
                configure_run_environment(args, args.output)
                from vllm.engine.arg_utils import EngineArgs

                # Real VllmConfig platform validation invokes init_ascend_config.
                engine = EngineArgs(**config).create_engine_config()
                from vllm_ascend.ascend_config import init_ascend_config

                effective = init_ascend_config(engine)
                require(effective.draft_window_graph_numeric_diagnostic == row["numeric_enabled"], "config drift")
                row["framework"] = "ENGINE_ARGS_PLATFORM_FACTORY_VALIDATED_NO_MODEL_LOAD"
            row["status"] = "FRAMEWORK_VALID" if framework else "SCHEMA_VALID_NOT_FRAMEWORK_PASS"
        except (ValueError, TypeError, RuntimeError, ImportError, OSError) as error:
            row["error"] = f"{type(error).__name__}: {error}"
        results.append(row)
    require(results, "empty configuration matrix")
    return dict(
        status="INVALID_CONFIG" if any(r["status"] == "INVALID_CONFIG" for r in results) else "CONFIGS_VALID",
        rows=results,
        unique_configurations=len({r.get("config_sha256") for r in results}),
        boundary="real CLI / kwargs / Ascend schema; EngineArgs + factory only when --framework",
        device_execution="NOT_RUN",
    )


def target_join(root, plan, prefix, device):
    """Recompute tensors and external execution; never trust a prior PASS JSON.

    Relocation changes only the local evidence path check, not frozen argv or
    original identity. Device is a separately supplied provenance statement,
    not a tool assertion that host physical ownership has been verified.
    """
    from tools.p7x_graph_repair5_evidence import audit_execution
    from tools.p7x_graph_repair6 import compare

    result = dict(status="INVALID_EVIDENCE", execution=[], device=device, authorization="NOT_VERIFIED")
    try:
        require(prefix in ("G4F", "TARGET4"), "unknown profile")
        paths = []
        arm_identities = []
        for arm, mode in (("OFF", "A0"), ("ON", "B")):
            name = prefix + "_" + arm
            expected = copy.deepcopy(plan["cases"][name])
            require(expected["acceptance_identity"]["driver_mode"] == mode, "arm substitution")
            artifact = root / name / "controlled-target.json"
            reader = CollectionReader(artifact)
            require(reader.header["arm"] == arm, "artifact arm substitution")
            require(
                reader.header["capture_profile"] == ("base" if prefix == "G4F" else "target4"),
                "artifact/profile mismatch",
            )
            arm_identities.append(
                [reader.header["identity"][key] for key in ("run_id", "attempt_id", "source_sha", "source_tree")]
            )
            campaign = json.loads((root / name / "campaign.json").read_text())
            launcher = json.loads((root / "launchers" / name / "launcher.json").read_text())
            original = expected["outputs"]
            expected["outputs"] = str(artifact.parent)
            execution = audit_execution(reader.header, campaign, launcher, expected, artifact)
            execution.update(case_id=name, original_output=original, relocated_output=str(artifact.parent))
            result["execution"].append(execution)
            require(reader.header["identity"]["run_id"] == plan["run_id"], "run mismatch")
            require(reader.header["identity"]["source_sha"] == plan["commit"], "source mismatch")
            require(digest(artifact) == execution.get("artifact_sha256", digest(artifact)), "artifact changed")
            paths.append(artifact)
        require(arm_identities[0] == arm_identities[1], "cross-run/attempt/source arms")
        comparison = compare(*paths)
        result["comparison"] = comparison
        result["artifact_sha256"] = [digest(p) for p in paths]
        result["source_sha"] = plan["commit"]
        result["status"] = (
            "NUMERIC_MISMATCH"
            if comparison["status"] == "NUMERIC_MISMATCH"
            else "INVALID_EVIDENCE"
            if any(x["status"] == "INVALID_EVIDENCE" for x in result["execution"])
            else "SCOPED_TARGET_EXECUTION_AND_NUMERIC_VALID"
            if comparison["status"] == "SAMPLE_COMPLETE"
            and all(x["status"] == "EXECUTION_ASSOCIATED" for x in result["execution"])
            else "PARTIAL_PREFIX_VALID"
            if comparison["status"] == "PARTIAL_PREFIX_VALID"
            else "INVALID_EVIDENCE"
        )
        result["authorized_device_acceptance"] = False
        result["authorization_reason"] = (
            "Requires current attempt device admission; historical segment not transferable"
        )
        result["remaining"] = ["normal reuse", "UPDATE generation", "task operand", "physical read"]
    except (KeyError, ValueError, TypeError, OSError) as error:
        result["error"] = str(error)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("config-check")
    c.add_argument("--plan", type=Path, required=True)
    c.add_argument("--framework", action="store_true")
    c.add_argument("--output", type=Path, required=True)
    j = sub.add_parser("target-join")
    for name in ("root", "plan", "device", "output"):
        j.add_argument("--" + name, type=Path, required=True)
    j.add_argument("--profile", choices=("G4F", "TARGET4"), required=True)
    a = parser.parse_args()
    plan = json.loads(a.plan.read_text())
    if a.command == "config-check":
        result = config_matrix(plan, framework=a.framework)
    else:
        import torch

        torch.set_num_threads(2)
        result = target_join(a.root, plan, a.profile, json.loads(a.device.read_text()))
    atomic_json(a.output, result)
    return 0 if result["status"] in ("CONFIGS_VALID", "SCOPED_TARGET_EXECUTION_AND_NUMERIC_VALID") else 2


if __name__ == "__main__":
    raise SystemExit(main())
