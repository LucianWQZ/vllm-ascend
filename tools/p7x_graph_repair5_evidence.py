# SPDX-License-Identifier: Apache-2.0
"""External execution association for diagnostics, never a device proof."""

import hashlib
import importlib.util
from pathlib import Path


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def audit_execution(document, campaign, launcher, expected, artifact, *, abort=False):
    """Join a producer artifact to the frozen argv and external child outcome.

    Source labels are necessary but insufficient: file bytes, child PID, complete
    plan and normal/aborted execution must also agree. No numerical verdict here.
    """
    result = {"status": "INVALID_EVIDENCE", "numeric_status": "NOT_ASSESSED"}
    try:
        fields = ("run_id", "case_id", "attempt_id", "source_sha", "source_tree", "driver_mode")
        identity = document["identity"]
        frozen = expected["acceptance_identity"]
        for field in fields:
            require(
                identity[field] == campaign["identity"][field] == launcher[field] == frozen[field],
                f"identity mismatch: {field}",
            )
        require(
            all(
                len(identity[k]) == 40 and all(c in "0123456789abcdef" for c in identity[k])
                for k in ("source_sha", "source_tree")
            ),
            "invalid exact source",
        )
        require(
            type(launcher["pid"]) is int
            and launcher["pid"] > 0
            and launcher["pid"] == identity["process_id"] == campaign["identity"]["process_id"],
            "child PID mismatch",
        )
        require(
            identity["engine_id"] == campaign["identity"]["engine_id"] and identity["engine_id"] is not None,
            "engine mismatch",
        )
        require(launcher["argv"] == expected["argv"], "child argv differs from frozen plan")
        require(
            expected["launch_argv"][expected["launch_argv"].index("--") + 1 :] == expected["argv"],
            "launch/child argv mismatch",
        )
        path = Path(artifact).resolve()
        require(path.parent == Path(expected["outputs"]).resolve(), "artifact outside declared output")
        require(
            campaign["request_manifest"]["sha256"] == expected["requests_sha256"] == launcher["requests_sha256"],
            "request manifest mismatch",
        )
        if not abort:
            indexed = campaign["diagnostic_artifacts"][path.name]
            require(
                indexed["sha256"] == file_sha(path) and indexed["bytes"] == path.stat().st_size,
                "artifact bytes/hash mismatch",
            )
        else:
            # os._exit intentionally prevents campaign finalization. The probe
            # embeds its contemporaneous recorder in the abort artifact; the
            # external checker hashes those bytes, never invents a final campaign.
            require(campaign["status"] == "ABORT_PARTIAL", "missing abort partial")
            result["artifact_sha256"] = file_sha(path)
        require(
            type(launcher["host_start_ns"]) is int
            and type(launcher["host_end_ns"]) is int
            and 0 < launcher["host_start_ns"] < launcher["host_end_ns"],
            "launcher interval",
        )
        result["status"] = "FAIL"
        require(
            launcher["status"] == "EXITED"
            and launcher["signal"] is None
            and launcher["timeout"] is False
            and launcher["returncode"] == (73 if abort else 0),
            "child failed/timed out/signalled",
        )
        require(campaign["startup_complete"] is True, "startup incomplete")
        if not abort:
            require(campaign["status"] == "OK", "campaign failed")
            require(
                not any(
                    campaign.get(k)
                    for k in (
                        "cleanup_error",
                        "artifact_collection_error",
                        "full_target_probe_error",
                        "target_probe_error",
                        "profile_probe_error",
                        "lifecycle_probe_error",
                        "diagnostic_artifact_error",
                    )
                ),
                "campaign cleanup/export failure",
            )
        result.update(status="EXECUTION_ASSOCIATED", artifact_sha256=file_sha(path))
    except (KeyError, TypeError, ValueError, OSError, IndexError) as error:
        result["error"] = str(error)
    return result


def audit_normal_requests(campaign, launcher, expected):
    path = Path(__file__).with_name("p7x_graph_repair4_acceptance.py")
    spec = importlib.util.spec_from_file_location("_repair5_joint", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    flat = dict(expected["acceptance_identity"], argv=expected["argv"], requests_sha256=expected["requests_sha256"])
    return module.audit_joint(campaign, launcher, Path(expected["request_path"]).read_bytes(), flat)


def correlate_lifecycle(document, snapshot, expected=None, launcher=None):
    """Strict one-to-one producer/recorder join. IDs remain host provenance."""
    require(snapshot["process_id"] == document["identity"]["process_id"], "recorder PID mismatch")
    require(snapshot["observer_enabled"] is True and not snapshot["artifact_errors"], "recorder incomplete")
    bound = [r for r in snapshot["events"] if r["stage"] == "bound"]
    records = document["records"]
    require(len(bound) == len(records) > 0, "truncated/extra lifecycle generations")
    layers = document["draft_layer_names"]
    require(layers and len(layers) == len(set(layers)), "missing/duplicate execution layers")
    if expected is not None:
        coverage = expected["lifecycle_coverage"]
        for desc in coverage["descriptors"]:
            require(
                sum(r["descriptor"] == desc for r in records) >= coverage["minimum_generations_per_descriptor"],
                "declared replay/generation coverage missing",
            )
    for record, binding in zip(records, bound):
        generation, epoch = record["generation"], record["epoch"]
        require(
            binding["generation_id"] == generation and binding["mapping_epoch"] == epoch,
            "bound/lifecycle ordering or identity mismatch",
        )
        require(record["provider"] == snapshot["provider_identity"], "provider mismatch")
        desc = record["descriptor"]
        text = binding["capture_descriptor"]
        fields = dict(p.split("=", 1) for p in text.split(";"))
        require(
            fields["tokens"] == str(desc[1])
            and fields["reqs"] == str(desc[0])
            and fields["mode"] == "FULL"
            and fields["uniform"] == "7",
            "descriptor mismatch",
        )
        require(
            binding["backing_table_address"] == record["backing"]["window_block_table"]["data_ptr"],
            "bound backing mismatch",
        )
        if launcher is not None:
            require(
                launcher["host_start_ns"]
                <= record["before_replay_ns"]
                <= record["retirement_return_ns"]
                <= launcher["host_end_ns"],
                "lifecycle outside child lifetime",
            )
        g4 = [r for r in snapshot["g4_records"] if r["generation_id"] == generation]
        require(
            len(g4) == 1
            and g4[0]["mapping_epoch"] == epoch
            and len(g4[0]["runtime_oracle_inputs"]["request_ids"]) == record["active_requests"],
            "active request/canonical generation mismatch",
        )
        lifecycle = [e for e in snapshot["lifecycle_events"] if e["generation_id"] == generation]
        stages = [e["stage"] for e in lifecycle]
        require(
            stages.count("replay_submitted") == stages.count("proposal_retired") == 1,
            "missing/duplicate replay or retirement",
        )
        require(stages.count("update_event_recorded") == len(layers), "incomplete update lifecycle")
        require(all(e["mapping_epoch"] == epoch for e in lifecycle), "lifecycle epoch mismatch")
        updates = [
            e
            for e in snapshot["operand_events"]
            if e["stage"] == "update_operator" and e["generation_id"] == generation
        ]
        require(
            len(updates) == len(layers) and {u["layer_name"] for u in updates} == set(layers),
            "missing/duplicate/wrong layer update",
        )
        require(
            all(
                u["mapping_epoch"] == epoch
                and u["capture_descriptor"] == text
                and u["provider_identity"] == record["provider"]
                and u["actual_table_address"] == binding["backing_table_address"]
                for u in updates
            ),
            "update association mismatch",
        )
        selected = [binding, *lifecycle, *updates, g4[0]]
        require(
            all(
                x["record_scope"]
                == {
                    "recorder_id": snapshot["recorder_id"],
                    "process_id": snapshot["process_id"],
                    "provider_identity": snapshot["provider_identity"],
                }
                for x in selected
            ),
            "cross-recorder mixture",
        )
        replay = next(e for e in lifecycle if e["stage"] == "replay_submitted")
        retire = next(e for e in lifecycle if e["stage"] == "proposal_retired")
        require(
            binding["sequence"] < replay["sequence"] < min(u["sequence"] for u in updates) < retire["sequence"],
            "replay/update/retire ordering invalid",
        )
    return {"status": "RECORDER_ASSOCIATED", "generations": len(records)}


def correlate_abort(document, snapshot):
    """Partial submission has stage-specific evidence, never normal completion."""
    require(document["recorder"] == snapshot, "abort/campaign snapshots differ")
    require(snapshot["process_id"] == document["identity"]["process_id"], "abort recorder PID")
    require(snapshot["observer_enabled"] is True and not snapshot["artifact_errors"], "abort recorder incomplete")
    record = document["records"][-1]
    require(record["provider"] == snapshot["provider_identity"], "abort provider mismatch")
    generation, epoch = record["generation"], record["epoch"]
    bound = [e for e in snapshot["events"] if e["stage"] == "bound"]
    require(len(bound) == len(document["records"]) > 0, "abort generation coverage mismatch")
    binding = bound[-1]
    require(
        binding["generation_id"] == generation and binding["mapping_epoch"] == epoch,
        "abort bound generation mismatch",
    )
    fields = dict(p.split("=", 1) for p in binding["capture_descriptor"].split(";"))
    require(
        record["descriptor"] == [int(fields["reqs"]), int(fields["tokens"])]
        and fields["mode"] == "FULL"
        and fields["uniform"] == "7",
        "abort descriptor mismatch",
    )
    require(
        binding["backing_table_address"] == record["backing"]["window_block_table"]["data_ptr"],
        "abort backing mismatch",
    )
    updates = [
        e for e in snapshot["operand_events"] if e["stage"] == "update_operator" and e["generation_id"] == generation
    ]
    if document["abort_at"] in ("after_update", "after_completion"):
        layers = document["draft_layer_names"]
        require(
            len(updates) == len(layers) > 0 and {e["layer_name"] for e in updates} == set(layers),
            "abort update coverage mismatch",
        )
    for event in [binding, *updates]:
        require(
            event["mapping_epoch"] == epoch
            and event["capture_descriptor"] == binding["capture_descriptor"]
            and event["record_scope"]
            == {
                "recorder_id": snapshot["recorder_id"],
                "process_id": snapshot["process_id"],
                "provider_identity": record["provider"],
            },
            "abort recorder association mismatch",
        )
    return {"status": "PARTIAL_RECORDER_ASSOCIATED", "device_completion": "UNPROVEN"}
