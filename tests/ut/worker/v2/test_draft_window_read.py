# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from vllm_ascend.attention.utils import AscendCommonAttentionMetadata, _select_seq_lens
from vllm_ascend.worker.v2.spec_decode.draft_window_read import (
    materialize_draft_read,
)


class TestCanonicalRead(unittest.TestCase):
    def setUp(self):
        self.table = torch.arange(24, dtype=torch.int32).reshape(3, 8)
        self.owner = SimpleNamespace(
            block_tables=[SimpleNamespace(gpu=self.table)],
            num_blocks=SimpleNamespace(np=np.array([[8, 8, 8]])),
            kernel_block_sizes=[4],
        )
        self.mapping = np.array([2, 0])
        self.retained: list[torch.Tensor] = []

    def common(self, table, exact, upper, padded_rows=3):
        return AscendCommonAttentionMetadata(
            query_start_loc=torch.arange(padded_rows + 1, dtype=torch.int32),
            query_start_loc_cpu=torch.arange(padded_rows + 1, dtype=torch.int32),
            seq_lens=exact,
            num_reqs=padded_rows,
            num_actual_tokens=padded_rows,
            max_query_len=1,
            max_seq_len=int(upper.max()),
            block_table_tensor=table,
            slot_mapping=torch.arange(padded_rows, dtype=torch.int64),
            seq_lens_cpu_upper_bound=upper,
        )

    def read(self, ends=(21, 11), starts=(12, 4), upper=(24, 15), padded_rows=3):
        exact = torch.tensor(ends, dtype=torch.int32)
        common = self.common(self.table[self.mapping.tolist()], exact, torch.tensor(upper), padded_rows)
        materialize_draft_read(
            common,
            self.owner.num_blocks.np[0, self.mapping].tolist(),
            4,
            exact_cpu=exact,
            starts_cpu=torch.tensor(starts),
            retain=lambda *values: self.retained.extend(values),
        )
        return common

    def test_projection_rejects_invalid_compact_geometry(self):
        for table, counts, span in (
            (self.table.flatten(), (8, 8), 4),
            (self.table.float(), (8, 8), 4),
            (self.table[:1], (8, 8), 4),
            (self.table, (9, 8), 4),
            (self.table, (-1, 8), 4),
            (self.table, (8, 8), 0),
        ):
            with self.subTest(table=table.shape, counts=counts, span=span), self.assertRaises(ValueError):
                materialize_draft_read(
                    self.common(table, torch.tensor([21, 11]), torch.tensor([24, 15])),
                    counts,
                    span,
                    exact_cpu=torch.tensor([21, 11]),
                    starts_cpu=torch.tensor([12, 4]),
                    retain=lambda *values: None,
                )

    def test_canonical_projection_tightens_upper_and_preserves_padding(self):
        view = self.read()
        self.assertEqual(view.block_table_tensor.tolist(), [[19, 20, 21], [1, 2, 0], [0, 0, 0]])
        self.assertEqual(view.seq_lens_cpu.tolist(), [9, 7, 0])
        self.assertIs(view.seq_lens_cpu_upper_bound, view.seq_lens_cpu)
        self.assertEqual(view.max_seq_len, 9)
        self.assertNotEqual(
            view.block_table_tensor.untyped_storage().data_ptr(), self.table.untyped_storage().data_ptr()
        )
        self.assertEqual(self.table.tolist(), torch.arange(24).reshape(3, 8).tolist())

    def test_projection_copies_retain_actual_source_and_destination(self):
        retained: list[torch.Tensor] = []
        copied: list[tuple[torch.Tensor, torch.Tensor]] = []
        original_copy = torch.Tensor.copy_

        def copy(destination, source, *args, **kwargs):
            if kwargs.get("non_blocking"):
                self.assertTrue(any(value is source for value in retained))
                self.assertTrue(any(value is destination for value in retained))
                copied.append((source, destination))
            return original_copy(destination, source, *args, **kwargs)

        exact = torch.tensor([21, 11], dtype=torch.int32)
        common = self.common(self.table[self.mapping.tolist()], exact, exact + 3)
        with patch.object(torch.Tensor, "copy_", copy):
            materialize_draft_read(
                common,
                self.owner.num_blocks.np[0, self.mapping].tolist(),
                4,
                exact_cpu=exact,
                starts_cpu=torch.tensor([12, 4], dtype=torch.int32),
                retain=lambda *values: retained.extend(values),
            )
        self.assertEqual(len(copied), 3)  # Two active-row D2D copies and one exact-length H2D.
        self.assertEqual(copied[-1][0].dtype, torch.int32)
        self.assertEqual(common.seq_lens.tolist(), [9, 7, 0])
        self.assertTrue(any(value is common.block_table_tensor for value in retained))
        self.assertTrue(any(value is common.seq_lens for value in retained))

    def test_retention_failure_prevents_first_table_copy(self):
        calls = []

        def retain(*values):
            calls.append(values)
            if len(calls) == 2:
                raise RuntimeError("retain destination failed")

        exact = torch.tensor([21, 11], dtype=torch.int32)
        with patch.object(torch.Tensor, "copy_") as copy:
            with self.assertRaisesRegex(RuntimeError, "retain destination"):
                materialize_draft_read(
                    self.common(self.table[self.mapping.tolist()], exact, exact + 3),
                    self.owner.num_blocks.np[0, self.mapping].tolist(),
                    4,
                    exact_cpu=exact,
                    starts_cpu=torch.tensor([12, 4], dtype=torch.int32),
                    retain=retain,
                )
            copy.assert_not_called()

    def test_reorder_in_place_and_same_row_replacement(self):
        first = self.read()
        self.mapping[:] = [0, 2]
        self.table[0] = torch.arange(100, 108)
        second = self.read()
        self.assertEqual(second.block_table_tensor[0].tolist(), [103, 104, 105])
        self.assertEqual(first.block_table_tensor[0].tolist(), [19, 20, 21])

    def test_valid_count_cannot_be_replaced_by_capacity(self):
        self.owner.num_blocks.np[0, 2] = 5
        with self.assertRaisesRegex(ValueError, "does not cover"):
            self.read()

    def test_upper_is_not_used_as_exact(self):
        self.owner.num_blocks.np[0, 2] = 6
        view = self.read(upper=(31, 15))
        self.assertEqual(view.seq_lens.tolist(), [9, 7, 0])
        self.assertEqual(view.seq_lens_cpu_upper_bound.tolist(), [9, 7, 0])

    def test_current_prepared_rows_preserve_reorder_and_strided_sources(self):
        for strided in (False, True):
            with self.subTest(strided=strided):
                table = self.table if not strided else torch.arange(48, dtype=torch.int32).reshape(3, 16)[:, ::2]
                before = table.clone()
                # The framework's compact input can itself be strided.
                compact = table[[2, 0]]
                if strided:
                    storage = torch.zeros((2, 16), dtype=torch.int32)
                    storage[:, ::2] = compact
                    compact = storage[:, ::2]
                exact = torch.tensor([21, 11], dtype=torch.int32)
                common = self.common(compact, exact, torch.tensor([24, 15]))
                materialize_draft_read(
                    common,
                    (8, 8),
                    4,
                    exact_cpu=exact,
                    starts_cpu=torch.tensor([12, 4]),
                    retain=lambda *values: self.retained.extend(values),
                )
                expected = torch.stack(
                    (
                        before[2, 3:6],
                        torch.cat((before[0, 1:3], torch.zeros(1, dtype=table.dtype))),
                        torch.zeros(3, dtype=table.dtype),
                    )
                )
                torch.testing.assert_close(common.block_table_tensor, expected, rtol=0, atol=0)
                torch.testing.assert_close(table, before, rtol=0, atol=0)
                self.assertEqual(common.seq_lens.tolist(), [9, 7, 0])

    def test_prepared_compact_table_requires_active_request_extent(self):
        with self.assertRaisesRegex(ValueError, "compact column extent"):
            materialize_draft_read(
                self.common(self.table[:1], torch.tensor([21, 11]), torch.tensor([24, 15])),
                (8, 8),
                4,
                exact_cpu=torch.tensor([21, 11]),
                starts_cpu=torch.tensor([12, 4]),
                retain=lambda *values: None,
            )

    def test_direct_common_projection_preserves_query_and_write_identity(self):
        slots, positions, query = object(), object(), object()
        exact = torch.tensor([21, 11], dtype=torch.int32)
        common = self.common(self.table[self.mapping.tolist()], exact, exact + 3)
        common.slot_mapping, common.positions, common.query_start_loc = slots, positions, query
        materialize_draft_read(
            common,
            (8, 8),
            4,
            exact_cpu=exact,
            starts_cpu=torch.tensor([12, 4]),
            retain=lambda *values: self.retained.extend(values),
        )
        self.assertIs(common.slot_mapping, slots)
        self.assertIs(common.positions, positions)
        self.assertIs(common.query_start_loc, query)

    def test_reject_unrepresentable_start(self):
        with self.assertRaisesRegex(ValueError, "intra-block"):
            self.read(starts=(13, 4))

    def test_invalid_exact_upper_and_start_ordering_is_rejected(self):
        for kwargs in ({"upper": (20, 15)}, {"starts": (24, 4)}, {"ends": (-1, 11)}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "ordering"):
                self.read(**kwargs)

    def test_active_rows_cannot_exceed_the_actual_common_padded_extent(self):
        with self.assertRaisesRegex(ValueError, "padded row extent"):
            self.read(padded_rows=1)

    def test_continuing_full_row_and_window_row_share_batch(self):
        view = self.read(starts=(0, 4))
        self.assertEqual(view.seq_lens.tolist(), [21, 7, 0])
        self.assertEqual(view.block_table_tensor[0, :6].tolist(), [16, 17, 18, 19, 20, 21])

    def select_parallel_lengths(self, common):
        return _select_seq_lens(
            common,
            kv_cache_spec=None,
            speculative_config=SimpleNamespace(parallel_drafting=True, use_dspark=lambda: False),
            vllm_config=SimpleNamespace(model_config=object()),
        )

    def test_parallel_builder_reuses_the_current_window_exact_snapshot(self):
        common = self.read(padded_rows=2)
        selected = self.select_parallel_lengths(common)
        # Generic parallel drafting otherwise reselects device lengths and
        # synchronizes again at the GQA builder's tolist().
        self.assertEqual(selected.data_ptr(), common.seq_lens_cpu.data_ptr())
        self.assertNotEqual(selected.data_ptr(), common.seq_lens.data_ptr())
        self.assertEqual(selected.tolist(), [9, 7])

    def test_parallel_off_does_not_trust_an_uncertified_cpu_mirror(self):
        common = SimpleNamespace(
            num_reqs=2,
            seq_lens=torch.tensor([9, 7], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([99, 99], dtype=torch.int32),
            _seq_lens_cpu=torch.tensor([99, 99], dtype=torch.int32),
        )
        self.assertIs(self.select_parallel_lengths(common), common.seq_lens)
        self.assertFalse(hasattr(common, "_draft_window_exact_cpu"))

    @patch("vllm_ascend.attention.utils._is_glm_model", new=lambda config: True)
    def test_glm_dspark_off_retains_official_cpu_choice_without_copy(self):
        for use_v2 in (False, True):
            with self.subTest(use_v2=use_v2):
                common = SimpleNamespace(
                    num_reqs=1,
                    seq_lens=torch.tensor([1151], dtype=torch.int32),
                    _seq_lens_cpu=torch.tensor([1152], dtype=torch.int32),
                    seq_lens_cpu=None,
                )
                with patch.object(torch.Tensor, "to", side_effect=AssertionError("OFF must not add an exact copy")):
                    selected = _select_seq_lens(
                        common,
                        None,
                        SimpleNamespace(parallel_drafting=True, use_dspark=lambda: True),
                        SimpleNamespace(model_config=object(), use_v2_model_runner=use_v2),
                    )
                self.assertEqual(selected.data_ptr(), common._seq_lens_cpu.data_ptr())
                self.assertEqual(selected.tolist(), [1152])
                self.assertFalse(hasattr(common, "_draft_window_exact_cpu"))

    @patch("vllm_ascend.attention.utils._is_glm_model", new=lambda config: True)
    def test_glm_dspark_on_full_and_window_reads_keep_current_exact(self):
        for starts in ((0, 4), (12, 4)):
            with self.subTest(starts=starts):
                common = self.read(starts=starts, padded_rows=2)
                selected = _select_seq_lens(
                    common,
                    None,
                    SimpleNamespace(parallel_drafting=True, use_dspark=lambda: True),
                    SimpleNamespace(model_config=object(), use_v2_model_runner=True),
                )
                self.assertEqual(selected.data_ptr(), common.seq_lens_cpu.data_ptr())
                self.assertEqual(selected.tolist(), [21 - starts[0], 7])
                self.assertEqual(selected.tolist(), common.seq_lens_cpu_upper_bound[:2].tolist())
                self.assertNotEqual(selected.tolist(), [24 - starts[0], 11])

    def test_snapshot_marker_is_the_actual_bound_cpu_tensor(self):
        common = self.read(padded_rows=2)
        self.assertIs(common._draft_window_exact_cpu, common.seq_lens_cpu)
        self.assertIs(common._draft_window_exact_cpu, common._seq_lens_cpu)
        self.assertEqual(common.seq_lens.tolist(), common._draft_window_exact_cpu.tolist())

    def test_compact_kernel_columns_reach_the_original_mla_builder(self):
        from vllm_ascend.attention.mla_v1 import AscendMLAMetadata, AscendMLAMetadataBuilder
        from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec

        # Manager blocks are 256 tokens; the Ascend table is expanded to
        # kernel-addressed 128-token columns before official batch gathering.
        spec = AscendMLAAttentionSpec(block_size=256, num_kv_heads=1, head_size=128, dtype=torch.bfloat16)
        compact = torch.arange(32, dtype=torch.int32).reshape(2, 16)
        full = compact.clone()
        exact = torch.tensor([1151, 1408], dtype=torch.int32)
        slots, positions = torch.arange(16), torch.arange(16) + 1135
        common = AscendCommonAttentionMetadata(
            num_reqs=2,
            seq_lens=exact,
            seq_lens_cpu_upper_bound=exact + 7,
            block_table_tensor=compact,
            max_seq_len=int((exact + 7).max()),
            slot_mapping=slots,
            positions=positions,
            query_start_loc_cpu=torch.tensor([0, 8, 16]),
            query_start_loc=torch.tensor([0, 8, 16]),
            pcp_context=None,
            pcp_cache_group_idx=None,
            context_parallel_metadata=None,
            num_actual_tokens=16,
            num_input_tokens=16,
            max_query_len=8,
            graph_pad_size=-1,
            attn_state=object(),
            causal=False,
        )
        materialize_draft_read(
            common,
            (9, 11),
            spec.block_size // 2,
            exact_cpu=exact,
            starts_cpu=torch.tensor([0, 384]),
            retain=lambda *values: None,
        )
        builder = object.__new__(AscendMLAMetadataBuilder)
        builder.flashmla_state, builder.use_pcp, builder.use_dcp = None, False, False
        builder.decode_threshold = 8
        builder.metadata_cls = AscendMLAMetadata
        builder.model_config = SimpleNamespace(get_head_size=lambda: 128)
        builder.attn_mask_builder = SimpleNamespace(get_splitfuse_attn_mask=lambda: None)
        # This Host check covers the original builder's real common-field
        # selection. Derived native operands are separately device-qualified.
        builder.build_decode_metadata = lambda *args: None
        with patch("vllm_ascend.attention.utils.is_pd_decode_recompute_scheduler_enabled", return_value=False):
            result = builder.build(0, common)
        self.assertEqual(result.block_tables.tolist(), [full[0, :9].tolist(), full[1, 3:11].tolist() + [0]])
        self.assertEqual(result.seq_lens.tolist(), [1151, 1024])
        self.assertIs(result.slot_mapping.untyped_storage(), slots.untyped_storage())
        self.assertIs(common.positions, positions)
        self.assertTrue(torch.equal(compact, full))


if __name__ == "__main__":
    unittest.main()
