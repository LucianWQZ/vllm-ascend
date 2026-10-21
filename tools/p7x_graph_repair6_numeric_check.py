# SPDX-License-Identifier: Apache-2.0
"""Sequential continuous evidence validation, independent of the producer oracle.

Only the current sample's KV is mapped. Cross-generation comparisons retain
bounded output tensors and content digests, never whole collections of KV.
This validator cannot prove task completion or physical memory transactions.
"""

import hashlib
import json

import torch

from tools.p7x_graph_repair6_storage import digest, require


def tensor(value, shape=None, dtype=None):
    require(isinstance(value, torch.Tensor) and value.numel() > 0, "missing real tensor")
    require(shape is None or list(value.shape) == list(shape), "tensor shape mismatch")
    require(dtype is None or str(value.dtype) == dtype, "tensor dtype mismatch")
    flat = value.reshape(-1)
    for offset in range(0, flat.numel(), 1 << 20):
        require(bool(torch.isfinite(flat[offset : offset + (1 << 20)]).all()), "nonfinite tensor")
    return value


def fingerprint(value):
    h = hashlib.sha256()

    def feed(v):
        if isinstance(v, torch.Tensor):
            h.update(str((v.dtype, tuple(v.shape))).encode())
            flat = v.reshape(-1)
            for start in range(0, flat.numel(), 1 << 18):
                h.update(memoryview(flat[start : start + (1 << 18)].contiguous().view(torch.uint8).numpy()))
        elif isinstance(v, dict):
            for k in sorted(v):
                h.update(str(k).encode())
                feed(v[k])
        else:
            h.update(json.dumps(v, sort_keys=True).encode())

    feed(value)
    return h.hexdigest()


def validate_sample(r, header, case):
    from tools.p7x_graph_repair5_target import sample_identity
    from tools.p7x_graph_repair6 import metrics

    provenance = header["provenance"]
    require(r["case_identity"] == sample_identity(case, provenance["fixture_id"]), "full case identity mismatch")
    active = len(case["ends"])
    padded = 1 if active == 1 else 2 if active == 2 else 4
    tokens = padded * 7
    require(r["active"] == active and r["descriptor"] == [padded, tokens], "Draft descriptor/active mismatch")
    require(r["target_descriptor"] == [case["padded"], case["padded"] * 8], "Target descriptor mismatch")
    require(
        r["identity"] == dict(header["identity"], generation=r["generation"], epoch=r["epoch"]),
        "source/generation mismatch",
    )
    runtime = r["runtime_identity"]
    require(
        runtime["generation_id"] == r["generation"] and runtime["mapping_epoch"] == r["epoch"],
        "runtime identity mismatch",
    )
    require(
        runtime["actual_request_count"] == active and runtime["padded_request_count"] == padded,
        "runtime row ownership mismatch",
    )
    require(
        len(runtime["ordered_request_fingerprint"]) == active
        and len(set(runtime["ordered_request_fingerprint"])) == active,
        "missing logical request ownership",
    )
    require(
        runtime["provider_id"] > 0 and runtime["source_identity"] and r["provider"] > 0, "missing canonical lineage"
    )
    source = runtime["source_identity"]
    require(
        source["block_tables_owner_id"] > 0
        and source["persistent_num_blocks_ptr"] > 0
        and len(source["input_block_table_ptrs"]) == 1
        and source["input_block_table_ptrs"][0] > 0,
        "invalid FullSource ownership",
    )
    require(r["exact"] == [e + 7 for e in case["ends"]], "DSpark anchor exact != Target end + actual query7")
    full = r["canonical_table"]
    require(
        type(full) is list and len(full) == padded and full and len({len(row) for row in full}) == 1,
        "canonical geometry",
    )
    require(all(type(b) is int and b >= 0 for row in full for b in row), "invalid physical block id")
    require(len(r["valid"]) == active, "valid-count coverage")
    layers = provenance["draft_layers"]
    require(layers and len(layers) == len(set(layers)), "missing layer inventory")
    require(
        set(r["actual_metadata"]) == set(r["reference_metadata"]) == set(layers), "metadata layer inventory mismatch"
    )
    width = tensor(r["actual_metadata"][layers[0]]["table"]).shape[1]
    expected = torch.zeros((padded, width), dtype=torch.int32)
    lengths, oracle, block_union = [], [], {0}
    for row, e in enumerate(r["exact"]):
        require(type(e) is int and e >= 7 and type(r["valid"][row]) is int, "invalid exact/valid")
        first = max(0, e - 1024) // 128
        end_column = (e + 127) // 128
        count = r["valid"][row]
        require(0 <= first < end_column <= count <= len(full[row]) and end_column - first <= width, "source shortage")
        require(len(set(full[row][:count])) == count, "repeated canonical blocks")
        expected[row, : end_column - first] = torch.tensor(full[row][first:end_column], dtype=torch.int32)
        lengths.append(e - first * 128)
        oracle.append(dict(S=first * 128, E=e, V=lengths[-1], columns=[first, end_column]))
        block_union.update(full[row][:count])
    require(r["oracle"] == oracle, "claimed oracle differs from independent recomputation")
    descriptors = runtime["read_descriptor_fingerprint"]
    require(len(descriptors) == active, "runtime read descriptor coverage")
    for row, (d, o) in enumerate(zip(descriptors, oracle)):
        require(
            d["global_group_id"] == 0
            and d["first_token_offset"] == 0
            and d["source_start_column"] == o["S"] // 128
            and d["required_columns"] == o["columns"][1] - o["columns"][0]
            and d["valid_num_blocks"] == r["valid"][row]
            and d["source_row_width"] == len(full[row]),
            "runtime descriptor differs from canonical",
        )
    lengths += [0] * (padded - active)
    inputs = r["inputs"]
    require(
        len(r["idx_mapping"]) == active
        and len(set(r["idx_mapping"])) == active
        and all(type(i) is int and i >= 0 for i in r["idx_mapping"]),
        "request row mapping invalid",
    )
    require(
        set(r["reference_inputs"]) == set(inputs) == {"input_ids", "positions", "slot_mapping", "query_start_loc"},
        "reference input coverage",
    )
    require(
        all(torch.equal(tensor(inputs[k]), tensor(r["reference_inputs"][k])) for k in inputs),
        "Graph/reference input mismatch",
    )
    ids = tensor(inputs["input_ids"], [tokens], "torch.int64")
    positions = tensor(inputs["positions"], [tokens], "torch.int64")
    slots = tensor(inputs["slot_mapping"], [tokens])
    query = tensor(inputs["query_start_loc"], [padded + 1])
    require(query.tolist() == [i * 7 for i in range(padded + 1)], "query layout mismatch")
    require(bool((ids[: active * 7] >= 0).all()), "invalid input token")
    for row, e in enumerate(r["exact"]):
        expected_positions = list(range(e - 7, e))
        require(positions[row * 7 : (row + 1) * 7].tolist() == expected_positions, "absolute positions mismatch")
        expected_slots = [full[row][p // 128] * 128 + p % 128 for p in expected_positions]
        require(slots[row * 7 : (row + 1) * 7].tolist() == expected_slots, "noncanonical write slot")
    require(all(s == -1 for s in slots[active * 7 :].tolist()), "padded writes not masked")
    block_union.update(s // 128 for s in slots.tolist() if s >= 0)
    require(r["selected_blocks"] == sorted(block_union), "selected read/write/padding blocks incomplete")
    inventory = provenance["kv_inventory"]
    require(set(inventory) == {f"{n}/{i}" for n in layers for i in (0, 1)}, "K/V inventory incomplete")
    require(r["inventory"] == inventory, "model KV inventory drift")
    for name, shape in inventory.items():
        require(len(shape["shape"]) == 4 and shape["shape"][1] == 128, "Base KV geometry mismatch")
        require(max(block_union) < shape["shape"][0], "block outside cache")
        for field in ("pre_kv", "reference_pre_kv", "graph_post_kv", "reference_post_kv"):
            require(set(r[field]) == set(inventory), "missing/extra KV component")
            tensor(r[field][name], [len(block_union), *shape["shape"][1:]], shape["dtype"])
        require(torch.equal(r["pre_kv"][name], r["reference_pre_kv"][name]), "Graph/reference KV prestate mismatch")
    kv_equal = all(torch.equal(r["graph_post_kv"][n], r["reference_post_kv"][n]) for n in inventory)
    for metadata in (r["actual_metadata"], r["reference_metadata"]):
        for m in metadata.values():
            require(
                torch.equal(tensor(m["table"], [padded, width], "torch.int32"), expected),
                "actual/reference table mismatch",
            )
            require(
                tensor(m["exact"], [padded]).tolist() == lengths and m["upper"] == lengths,
                "actual/reference lengths mismatch",
            )
            require(
                m["query"] == [i * 7 for i in range(1, padded + 1)] and m["causal"] is True, "query/causal mismatch"
            )
            require(torch.equal(tensor(m["slot"], [tokens]), slots), "metadata write identity values mismatch")
    backing = r["backing"]
    require(type(backing["owner_id"]) is int and backing["owner_id"] > 0, "backing owner absent")
    for field in (
        "window_block_table",
        "valid_num_blocks",
        "device_window_exact",
        "cpu_window_exact",
        "cpu_window_upper",
    ):
        b = backing[field]
        require(
            type(b["data_ptr"]) is int
            and b["data_ptr"] > 0
            and b["storage_data_ptr"] > 0
            and b["shape"][0] == padded
            and len(b["shape"]) == len(b["stride"])
            and b["storage_offset"] >= 0
            and b["device"]
            and b["dtype"],
            "invalid typed backing",
        )
    for name in layers:
        a, b = r["actual_metadata"][name], r["reference_metadata"][name]
        require(
            a["table_ptr"] == backing["window_block_table"]["data_ptr"]
            and a["exact_ptr"] == backing["device_window_exact"]["data_ptr"],
            "actual metadata not persistent carrier",
        )
        require(
            b["table_ptr"] != a["table_ptr"] and b["exact_ptr"] != a["exact_ptr"], "reference aliases tested carrier"
        )
    hidden = provenance["hidden_contract"]
    require(
        r["hidden_contract"] == hidden and type(hidden["width"]) is int and hidden["width"] > 0, "hidden width contract"
    )
    require(hidden["dtype"] in ("torch.bfloat16", "torch.float16"), "unsupported hidden dtype")
    for field in ("graph", "reference"):
        tensor(r[field], [tokens, hidden["width"]], hidden["dtype"])
    measured = metrics(r["graph"][: active * 7], r["reference"][: active * 7])
    failed = measured["max_abs"] != 0 or not kv_equal or r["status"] == "NUMERIC_MISMATCH"
    require(r["status"] in ("PASS_SCOPED_CONTINUOUS", "NUMERIC_MISMATCH"), "unknown sample status")
    return dict(
        status="NUMERIC_MISMATCH" if failed else "PASS",
        metrics=measured,
        kv_equal=kv_equal,
        prestate=fingerprint(dict(inputs=inputs, kv=r["pre_kv"], full=full, exact=r["exact"])),
        output=r["graph"][: active * 7].clone(),
        reference=r["reference"][: active * 7].clone(),
    )


def check_collection(reader, campaign, expected):
    from tools.p7x_graph_repair5_target import case_plan

    result = dict(
        status="INVALID_EVIDENCE",
        samples=[],
        evidence_errors=[],
        aba="OPEN",
        normal_reuse="UNPROVEN",
        task_update_effective_generation="UNPROVEN",
        retained_state="current KV + A/previous output summaries only",
    )
    previous = None
    anchor = None
    confirmed_failure = any(s["status"] == "NUMERIC_MISMATCH" for s in reader.samples)
    try:
        h = reader.header
        require(h["numeric_schema"] == "repair6-continuous-v2", "legacy numeric evidence lacks required observations")
        require(h["tolerance"] == dict(atol=0.0, rtol=0.0), "tolerance changed")
        provenance = h["provenance"]
        require(provenance["profile"] == expected["capture_profile"] == campaign["capture_profile"], "profile mismatch")
        require(provenance["config_sha256"] == campaign["config_sha256"], "configuration mismatch")
        require(provenance["fixture_id"] == expected["fixture_sha256"], "fixture mismatch")
        require(
            provenance["draft_layers"] == campaign["numeric_contract"]["draft_layers"]
            and provenance == campaign["numeric_contract"],
            "campaign/model contract mismatch",
        )
        cases = case_plan(provenance["profile"])
        require(0 < len(reader.samples) <= len(cases), "empty/extra numeric samples")
        summary_path = reader.path.with_name("numeric-summary.json")
        summary = json.loads(summary_path.read_text())
        indexed = campaign["diagnostic_artifacts"][summary_path.name]
        require(
            digest(summary_path) == indexed["sha256"] and summary_path.stat().st_size == indexed["bytes"],
            "summary bytes/hash mismatch",
        )
        require(
            summary["identity"] == h["identity"] and summary["execution"] == reader.execution,
            "summary identity/execution mismatch",
        )
        require(
            summary["statuses"] == [s["status"] for s in reader.samples] and summary["samples"] == len(reader.samples),
            "summary sample coverage mismatch",
        )
        backings = {}
        snapshot = campaign["snapshot"]
        require(
            snapshot["process_id"] == h["identity"]["process_id"]
            and snapshot["observer_enabled"]
            and not snapshot["artifact_errors"],
            "recorder identity/integrity",
        )
        bound = [e for e in snapshot["events"] if e["stage"] == "bound"]
        require(len(bound) == len(reader.samples), "recorder/numeric generation coverage")
        for i, c in enumerate(cases[: len(reader.samples)]):
            require(not reader.incomplete(i), "incomplete numeric sample")
            r = reader.record(i)
            require(
                type(r["generation"]) is int and type(r["epoch"]) is int and r["generation"] >= 0 and r["epoch"] >= 0,
                "invalid generation/epoch",
            )
            require(
                previous is None or r["generation"] > previous["generation"] and r["epoch"] > previous["epoch"],
                "generation/epoch rollback",
            )
            key = tuple(r["descriptor"])
            require(r["backing"]["window_block_table"]["data_ptr"] > 0, "missing backing")
            association = (r["provider"], r["backing"])
            require(key not in backings or backings[key] == association, "persistent backing identity drift")
            backings[key] = association
            checked = validate_sample(r, h, c)
            require(r["provider"] == snapshot["provider_identity"], "recorder provider mismatch")
            events = [e for e in snapshot["events"] if e["generation_id"] == r["generation"]]
            for stage in ("bound", "replay", "consumer", "update"):
                selected = [e for e in events if e["stage"] == stage]
                require(
                    len(selected) == 1 and selected[0]["mapping_epoch"] == r["epoch"],
                    "missing/duplicate recorder stage",
                )
                event = selected[0]
                parts = dict(p.split("=", 1) for p in event["capture_descriptor"].split(";"))
                require([int(parts["reqs"]), int(parts["tokens"])] == r["descriptor"], "recorder bucket mismatch")
                require(
                    event["backing_table_address"] == r["backing"]["window_block_table"]["data_ptr"],
                    "recorder backing mismatch",
                )
            retire = [
                e
                for e in snapshot["lifecycle_events"]
                if e["generation_id"] == r["generation"] and e["stage"] == "proposal_retired"
            ]
            require(len(retire) == 1 and retire[0]["mapping_epoch"] == r["epoch"], "missing retirement association")
            confirmed_failure |= checked["status"] == "NUMERIC_MISMATCH"
            row = dict(
                generation=r["generation"],
                epoch=r["epoch"],
                scenario=c["scenario"],
                status=checked["status"],
                metrics=checked["metrics"],
                kv_equal=checked["kv_equal"],
                current_match=torch.equal(checked["output"], checked["reference"]),
            )
            if previous is not None:
                row["previous_match"] = torch.equal(checked["output"], previous["reference"])
                row["references_distinguishable"] = not torch.equal(checked["reference"], previous["reference"])
            if c["scenario"] == "A":
                anchor = checked
            if c["scenario"] == "A_prime":
                result["aba"] = (
                    "PASS_SCOPED"
                    if anchor is not None
                    and previous is not None
                    and (
                        anchor["prestate"] == checked["prestate"]
                        and torch.equal(anchor["output"], checked["output"])
                        and not torch.equal(anchor["output"], previous["output"])
                    )
                    else "OPEN_PRESTATE_OR_DISCRIMINATION"
                )
            result["samples"].append(row)
            previous = dict(checked, generation=r["generation"], epoch=r["epoch"])
            del r
        require(reader.execution == "COMPLETE", "collection execution failed/incomplete")
        result["status"] = (
            "PASS_DIAGNOSTIC_CONTINUOUS_REFERENCE" if len(reader.samples) == len(cases) else "INCOMPLETE_COVERAGE"
        )
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, IndexError) as error:
        result["evidence_errors"].append(str(error))
    if confirmed_failure:
        result["status"] = "NUMERIC_MISMATCH"
    return result
