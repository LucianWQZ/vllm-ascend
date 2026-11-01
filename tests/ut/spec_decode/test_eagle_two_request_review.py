# SPDX-License-Identifier: Apache-2.0
"""Exact n1/n2 ordinary Eagle guards; CPU Source contracts, no framework imports."""

import ast
import copy
import hashlib
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
BASE = runpy.run_path(str(Path(__file__).with_name("test_r8_eagle_normal_review.py")))
M = BASE["M"]
CONFIG = runpy.run_path(str(Path(__file__).with_name("test_r9_full8_single_review_entry.py")))


class Vector:
    def __init__(self, values):
        self.values = values
        self.ndim = 1
        self.shape = (len(values),)
        self.dtype = "torch.int32"
        self.device = NS(type="cpu")

    def __getitem__(self, index):
        return NS(item=lambda: self.values[index])


def config():
    value = BASE["config"]()
    value.scheduler_config.max_num_seqs = 2
    value.compilation_config.cudagraph_capture_sizes = [4, 8, 16]
    value.compilation_config.max_cudagraph_capture_size = 16
    value.additional_config["draft_window_graph_eagle_two_request_review"] = True
    return value


def objects(active=2):
    ids = ["request-A", "request-B"][:active]
    prompts = [1537, 513][:active]
    indices = [7, 3][:active]
    owner = NS(_draft_graph_review_limit=M.EagleSingleRequestReviewScope(4096, max_requests=2), max_model_len=4096)
    batch = NS(
        num_reqs=active,
        num_reqs_after_padding=active,
        req_ids=ids,
        idx_mapping_np=indices,
        prefill_len_np=prompts,
        draft_graph_review_request_rows=tuple(
            (req, index, prompt, prompt + 32, prompt) for req, index, prompt in zip(ids, indices, prompts)
        ),
    )
    owner.input_batch = batch
    return owner, batch


class EagleTwoRequestControls(unittest.TestCase):
    def test_mixed_chunked_prefill_uses_canonical_progress_for_each_consumer(self):
        # C333 ordinary g0 failed with these saved SchedulerOutput counters:
        # long prompt 1537: 1152 computed + 128 scheduled = 1280; short prompt
        # 513: 384 + 129 = 513. These are Source-reconstructed lengths, not a
        # saved device/guard exact snapshot. Execute the actual Ascend CPU
        # length constructor on that state, without importing the model runner.
        path = ROOT / "vllm_ascend/worker/v2/model_runner.py"
        module = ast.parse(path.read_text())
        runner = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
        function = next(
            node for node in runner.body if isinstance(node, ast.FunctionDef) and node.name == "_update_seq_lens_cpu"
        )
        namespace = {}
        code = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(code), str(path), "exec"), namespace)
        owner, batch = objects()
        ids = batch.req_ids
        cpu_lengths = [0, 0]
        snapshots = {7: 1152, 3: 384}
        runner = NS(
            speculator=object(),
            sync_spec_pp_cpu_counts=False,
            _can_skip_spec_cpu_length_sync=lambda: False,
            num_computed_tokens_event=NS(synchronize=lambda: None),
            num_computed_tokens_cpu=snapshots.copy(),
            req_states=NS(req_id_to_index=dict(zip(ids, batch.idx_mapping_np)), num_computed_tokens_cpu={}),
            input_buffers=NS(seq_lens_cpu=cpu_lengths),
        )
        scheduler = NS(
            # Scheduled-token dict order differs from request/source order.
            num_scheduled_tokens={ids[1]: 129, ids[0]: 128},
            scheduled_cached_reqs=NS(req_ids=ids),
        )
        namespace["_update_seq_lens_cpu"](runner, scheduler, ids)
        self.assertEqual(cpu_lengths, [1280, 513])
        batch.num_computed_prefill_tokens_np = [1152, 384]
        batch.num_scheduled_tokens = [128, 129]
        owner._eagle3_window_proposal_context = NS(phase_plan=NS(baseline_only_rows=(0,)))
        for step in (0, 1, 2):
            M.validate_eagle_single_request_review_exact(
                owner, batch, Vector([1280 + step, 513 + step]), num_reqs=2, padded=2, step=step
            )
            for values in ([1279 + step, 513 + step], [1281 + step, 513 + step], [1280 + step, 512]):
                with self.subTest(step=step, values=values), self.assertRaises(NotImplementedError):
                    M.validate_eagle_single_request_review_exact(
                        owner, batch, Vector(values), num_reqs=2, padded=2, step=step
                    )
        # Absent/changed phase cannot relabel a terminal/decode row as continuing.
        owner._eagle3_window_proposal_context = NS(phase_plan=NS(baseline_only_rows=()))
        with self.assertRaises(NotImplementedError):
            M.validate_eagle_single_request_review_exact(
                owner, batch, Vector(cpu_lengths), num_reqs=2, padded=2, step=0
            )
        owner._eagle3_window_proposal_context = NS(phase_plan=NS(baseline_only_rows=(1,)))
        with self.assertRaises(RuntimeError):
            M.validate_eagle_single_request_review_exact(
                owner, batch, Vector([1537, 513]), num_reqs=2, padded=2, step=0
            )

    def test_continuing_prefill_does_not_relax_identity_capacity_or_counter_contract(self):
        for damage in (
            "wrong-id",
            "bad-index",
            "capacity",
            "preempted",
            "missing",
            "fractional",
            "negative",
            "terminal",
            "zero",
        ):
            owner, batch = objects()
            batch.num_computed_prefill_tokens_np = [1152, 384]
            batch.num_scheduled_tokens = [128, 129]
            owner._eagle3_window_proposal_context = NS(phase_plan=NS(baseline_only_rows=(0,)))
            if damage == "wrong-id":
                batch.req_ids[0] = "different-request"
            elif damage == "bad-index":
                batch.idx_mapping_np[0] = 6
            elif damage == "capacity":
                owner.max_model_len = 2048
            elif damage == "preempted":
                owner._draft_graph_review_preempted_request_ids = {batch.req_ids[0]}
            elif damage == "missing":
                del batch.num_scheduled_tokens
            elif damage == "fractional":
                batch.num_scheduled_tokens[0] = 128.5
            elif damage == "negative":
                batch.num_computed_prefill_tokens_np[0] = -1
            elif damage == "terminal":
                batch.num_scheduled_tokens[0] = 385
            else:
                batch.num_scheduled_tokens[0] = 0
            with self.subTest(damage=damage), self.assertRaises((RuntimeError, NotImplementedError)):
                M.validate_eagle_single_request_review_exact(
                    owner, batch, Vector([1280, 513]), num_reqs=2, padded=2, step=0
                )

    def test_exact_config_and_default_single_remain_separate(self):
        for window in (None, 1024):
            value = config()
            value.additional_config["draft_window_size"] = window
            scope = M.validate_eagle_two_request_review_config(value, experimental_graph_review=True)
            self.assertEqual((scope.max_requests, scope.max_model_len), (2, 4096))
            value.scheduler_config.async_scheduling = True
            value.cache_config.enable_prefix_caching = True
            value.additional_config["enable_flashcomm1"] = True
            self.assertEqual(M.validate_eagle_two_request_review_config(value, experimental_graph_review=True), scope)
        for obj, field, bad in (
            ("scheduler_config", "max_num_seqs", 1),
            ("scheduler_config", "max_num_seqs", 3),
            ("scheduler_config", "max_num_seqs", True),
            ("parallel_config", "data_parallel_size", 2),
            ("parallel_config", "tensor_parallel_size", 2),
            ("speculative_config", "num_speculative_tokens", 4),
            ("compilation_config", "cudagraph_capture_sizes", [4, 16]),
            ("compilation_config", "max_cudagraph_capture_size", 4),
        ):
            value = config()
            setattr(getattr(value, obj), field, bad)
            with self.subTest(field=field, value=bad), self.assertRaises(ValueError):
                M.validate_eagle_two_request_review_config(value, experimental_graph_review=True)
        for key in ("draft_window_graph_eagle_single_request_review", "draft_window_graph_numeric_diagnostic"):
            value = config()
            value.additional_config[key] = True
            with self.assertRaises(ValueError):
                M.validate_eagle_two_request_review_config(value, experimental_graph_review=True)
        owner, batch = objects()
        owner._draft_graph_review_limit = M.EagleSingleRequestReviewScope(4096)
        with self.assertRaises(NotImplementedError):
            M.validate_draft_graph_review_requests(owner, batch)
        original, one = BASE["objects"]()
        M.validate_draft_graph_review_requests(original, one)
        M.validate_eagle_single_request_review_exact(original, one, Vector([257]), num_reqs=1, padded=1, step=0)
        M.validate_eagle_single_request_graph_descriptor(original, first=True, n=1, tokens=4)
        M.validate_eagle_single_request_graph_descriptor(original, first=False, n=1, tokens=1)
        self.assertEqual(
            M.validate_eagle_single_request_review_config(
                BASE["config"](), experimental_graph_review=True
            ).max_requests,
            1,
        )

    def test_each_authoritative_row_and_exact_are_independent(self):
        owner, batch = objects()
        M.validate_draft_graph_review_requests(owner, batch)
        M.validate_eagle_single_request_review_exact(owner, batch, Vector([1537, 513]), num_reqs=2, padded=2, step=0)
        for change in ("wrong-id", "duplicate-id", "index-alias", "capacity", "resume", "padding", "three"):
            value, current = copy.deepcopy((owner, batch))
            value.input_batch = current
            rows = list(current.draft_graph_review_request_rows)
            if change == "wrong-id":
                current.req_ids[1] = "other-B"
            elif change == "duplicate-id":
                current.req_ids[1] = current.req_ids[0]
            elif change == "index-alias":
                rows[1] = (rows[1][0], 7, *rows[1][2:])
                current.idx_mapping_np[1] = 7
            elif change == "capacity":
                rows[1] = (rows[1][0], 3, 513, 4094, 513)
            elif change == "resume":
                rows[1] = (rows[1][0], 3, 513, 545, 514)
                current.prefill_len_np[1] = 514
            elif change == "padding":
                current.num_reqs_after_padding = 4
            else:
                current.num_reqs = 3
            current.draft_graph_review_request_rows = tuple(rows)
            with self.subTest(change=change), self.assertRaises((NotImplementedError, RuntimeError)):
                M.validate_draft_graph_review_requests(value, current)
        with self.assertRaises(NotImplementedError):
            M.validate_eagle_single_request_review_exact(
                owner, batch, Vector([1537, 512]), num_reqs=2, padded=2, step=1
            )
        with self.assertRaises(RuntimeError):
            M.validate_eagle_single_request_review_exact(owner, batch, Vector([1537]), num_reqs=2, padded=2, step=1)

    def test_first_and_two_ar_stages_reject_wrong_banks_or_steps(self):
        for active in (1, 2):
            owner, _ = objects(active)
            for first, ordinal, step, query in ((True, 0, 0, 4), (False, 0, 1, 1), (False, 1, 2, 1)):
                inputs = NS(actual_request_count=active, padded_request_count=active, identity=NS(step=step))
                M.validate_eagle_single_request_graph_stage(
                    owner, inputs, first=first, n=active, tokens=query * active, ordinal=ordinal
                )
                with self.assertRaises(NotImplementedError):
                    M.validate_eagle_single_request_graph_stage(
                        owner, inputs, first=first, n=active, tokens=query * active + 1, ordinal=ordinal
                    )
                inputs.identity.step = 4
                with self.assertRaises(NotImplementedError):
                    M.validate_eagle_single_request_graph_stage(
                        owner, inputs, first=first, n=active, tokens=query * active, ordinal=ordinal
                    )

    def test_finished_and_preempted_rows_keep_request_ownership(self):
        owner, batch = objects()
        M.note_eagle_single_request_review_lifecycle(owner, NS(finished_req_ids=set(), preempted_req_ids={"request-B"}))
        with self.assertRaises(NotImplementedError):
            M.validate_draft_graph_review_requests(owner, batch)
        M.note_eagle_single_request_review_lifecycle(owner, NS(finished_req_ids={"request-B"}, preempted_req_ids=None))
        M.validate_draft_graph_review_requests(owner, batch)
        # n2 -> n1 -> new n2 uses new authoritative request rows, not recycled IDs.
        owner.input_batch = objects(1)[1]
        M.validate_draft_graph_review_requests(owner, owner.input_batch)
        new = objects()[1]
        new.req_ids[1] = "request-C"
        row = new.draft_graph_review_request_rows[1]
        new.draft_graph_review_request_rows = (new.draft_graph_review_request_rows[0], ("request-C", *row[1:]))
        owner.input_batch = new
        M.validate_draft_graph_review_requests(owner, new)

    def test_off_actual_manager_checks_before_metadata(self):
        controls = BASE["ActualGraphDescriptorControls"]()
        owner, _ = objects()
        manager, seen = controls.manager(owner)
        self.assertEqual(controls.run_actual(manager, NS(num_reqs=2, num_tokens=2)), "legacy-result")
        self.assertEqual([value[0] for value in seen], ["metadata", "replay"])
        seen.clear()
        with self.assertRaises(NotImplementedError):
            controls.run_actual(manager, NS(num_reqs=4, num_tokens=4))
        self.assertEqual(seen, [])
        self.assertFalse(hasattr(owner, "_draft_window_runtime"))

    def test_resolver_keeps_fixed_checkpoint_mutex_and_default_closed(self):
        from pydantic import TypeAdapter

        module = ModuleType("vllm_ascend.ascend_config")
        module.validate_additional_config_bool = CONFIG["actual_function"](
            ROOT / "vllm_ascend/ascend_config.py", "validate_additional_config_bool", {"TypeAdapter": TypeAdapter}
        )
        with tempfile.TemporaryDirectory() as directory:
            data = b'{"independent_guard_fixture":true}\n'
            file = Path(directory) / "config.json"
            file.write_bytes(data)
            value = config()
            value.speculative_config.draft_model_config = NS(model=directory)
            with (
                patch.dict(sys.modules, {"vllm_ascend.ascend_config": module}),
                patch.object(M, "_EAGLE_CONFIG", hashlib.sha256(data).hexdigest()),
            ):
                self.assertEqual(
                    M.resolve_draft_graph_review_limit(value, experimental_graph_review=True).max_requests, 2
                )
                for key in (
                    "draft_window_graph_eagle_single_request_review",
                    "draft_window_graph_full8_two_request_review",
                ):
                    value.additional_config[key] = True
                    with self.assertRaises(ValueError):
                        M.resolve_draft_graph_review_limit(value, experimental_graph_review=True)
                    del value.additional_config[key]
                file.write_text("{}")
                with self.assertRaises(NotImplementedError):
                    M.resolve_draft_graph_review_limit(value, experimental_graph_review=True)
                file.write_bytes(data)
                del value.additional_config["draft_window_graph_eagle_two_request_review"]
                with self.assertRaises(NotImplementedError):
                    M.resolve_draft_graph_review_limit(value, experimental_graph_review=True)


if __name__ == "__main__":
    unittest.main()
