# SPDX-License-Identifier: Apache-2.0
"""Native causal SWA retained beside one independently windowed Full group.

This reviewed path keeps native readers on their original absolute tables,
lengths and slots. It is valid only when the extra Window contains every
native causal query's band, including the parallel query-width offset. A
narrower Window still needs an explicit native interval projection; it must
never silently inherit this passthrough. No Target, cache layout or kernel is
changed here, and no completion or physical-read assertion is manufactured.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

from vllm_ascend.attention.attention_v1 import AscendAttentionBackend, AscendAttentionMetadataBuilder

if TYPE_CHECKING:
    from vllm_ascend.worker.v2.spec_decode.draft_window import DraftWindowGenerationIdentity


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def contains_native_band(window_size: int, native_window: int, query_width: int) -> bool:
    """S=floor(max(E-W,0)/B)*B must precede the earliest query's band.

    For Q parallel queries the earliest position is E-Q, so its native lower
    bound is max(E-Q-native+1,0). W>=native+Q-1 is a sufficient bound independent
    of the request's alignment. Merely checking W>=native is insufficient.
    """
    if any(type(v) is not int or v <= 0 for v in (window_size, native_window, query_width)):
        raise ValueError("positive integer Window/native/query widths required")
    return window_size >= native_window + query_width - 1


@dataclass(frozen=True, slots=True)
class NativeSWAGraphGroup:
    gid: int
    layers: tuple[str, ...]
    native_window: int
    table_ptr: int
    table_shape: tuple[int, ...]
    table_dtype: torch.dtype


@dataclass(frozen=True, slots=True)
class NativeSWAFullSnapshot:
    identity: DraftWindowGenerationIdentity
    cpu_exact: torch.Tensor
    cpu_upper: torch.Tensor
    device_exact: torch.Tensor

    def validate(self, expected: Any, plan: NativeSWAGraphPlan) -> None:
        _require(self.identity == expected, "native SWA full lengths belong to another generation/epoch")
        n = expected.padded_request_count
        _require(
            self.cpu_exact.device.type == self.cpu_upper.device.type == "cpu"
            and self.cpu_exact.dtype == self.cpu_upper.dtype == self.device_exact.dtype == torch.int32
            and self.cpu_exact.shape == self.cpu_upper.shape == self.device_exact.shape == (n,),
            "native SWA requires owned CPU exact/upper and original device lengths for this bucket",
        )
        _require(
            self.device_exact.data_ptr() == plan.device_lengths_ptr
            and self.device_exact.device == plan.device_lengths_device,
            "native SWA device full-length owner changed",
        )
        _require(
            bool(torch.all(self.cpu_exact >= 0))
            and bool(torch.all(self.cpu_upper >= self.cpu_exact))
            and bool(torch.all(self.cpu_exact[expected.actual_request_count :] == 0))
            and bool(torch.all(self.cpu_upper[expected.actual_request_count :] == 0)),
            "native SWA exact/upper or padded lengths violate the current generation",
        )


@dataclass(frozen=True, slots=True)
class NativeSWAGraphPlan:
    full_gid: int
    full_layers: tuple[str, ...]
    native_groups: tuple[NativeSWAGraphGroup, ...]
    source_owner_id: int
    input_owner_id: int
    device_lengths_ptr: int
    device_lengths_device: torch.device
    query_width: int
    window_size: int

    @property
    def native_layers(self) -> tuple[str, ...]:
        return tuple(name for group in self.native_groups for name in group.layers)

    def filter_groups(self, groups: Sequence[Sequence[Any]], *, native: bool) -> list[list[Any]]:
        wanted = set(self.native_layers if native else self.full_layers)
        selected = [[group for group in members if set(group.layer_names) <= wanted] for members in groups]
        actual = [name for members in selected for group in members for name in group.layer_names]
        _require(
            len(actual) == len(set(actual)) == len(wanted) and set(actual) == wanted,
            "native SWA/Full partition changed after source admission",
        )
        gids = {name: group.gid for group in self.native_groups for name in group.layers}
        gids.update({name: self.full_gid for name in self.full_layers})
        _require(
            all(
                group.kv_cache_group_id == gid == gids[name]
                for gid, members in enumerate(selected)
                for group in members
                for name in group.layer_names
            ),
            "native SWA group index changed after admission",
        )
        return selected

    def window_metadata(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        _require(
            set(metadata) == set(self.full_layers + self.native_layers),
            "native SWA Graph metadata lacks a reader or contains a foreign layer",
        )
        return {name: metadata[name] for name in self.full_layers}

    def validate_native_metadata(
        self,
        metadata: Mapping[str, Any],
        *,
        source: Any,
        snapshot: NativeSWAFullSnapshot,
        expected: Any,
        write_side: Any,
    ) -> None:
        self.window_metadata(metadata)
        snapshot.validate(expected, self)
        _require(id(source) == self.source_owner_id, "native SWA canonical table owner changed")
        for group in self.native_groups:
            table = source.input_block_tables[group.gid]
            _require(
                table.data_ptr() == group.table_ptr
                and tuple(table.shape) == group.table_shape
                and table.dtype == group.table_dtype,
                "native SWA canonical table allocation changed",
            )
            for name in group.layers:
                value = metadata[name]
                _require(
                    value is not None
                    and value.causal is True
                    and value.block_tables.data_ptr() == table.data_ptr()
                    and value.block_tables.shape == (expected.padded_request_count, table.shape[1])
                    and value.block_tables.stride() == table.stride()
                    and value.block_tables.dtype == table.dtype
                    and value.seq_lens.data_ptr() == snapshot.device_exact.data_ptr()
                    and value.seq_lens_cpu is snapshot.cpu_exact
                    and value.seq_lens_list == snapshot.cpu_exact.tolist(),
                    "native SWA reader lost absolute table/exact/causal metadata",
                )
                _require(
                    value.slot_mapping.data_ptr() == source.slot_mappings[group.gid].data_ptr()
                    and value.slot_mapping.shape == write_side.slot_mapping.shape
                    and value.slot_mapping.dtype == source.slot_mappings.dtype
                    and value.positions.data_ptr() == write_side.positions.data_ptr()
                    and value.positions.shape == write_side.positions.shape
                    and value.query_start_loc.data_ptr() == write_side.query_start_loc.data_ptr()
                    and value.query_start_loc.shape == write_side.query_start_loc.shape
                    and value.actual_seq_lengths_q == list(write_side.actual_seq_lengths_q)
                    and value.num_actual_tokens == write_side.num_actual_tokens,
                    "native SWA Full write slots/absolute positions/query layout changed",
                )


def partition_source(
    *,
    attn_groups: Sequence[Sequence[Any]],
    block_tables: Any,
    input_buffers: Any,
    causal: bool | Mapping[int, bool],
    window_size: int,
    query_width: int,
) -> tuple[Sequence[Sequence[Any]], NativeSWAGraphPlan | None]:
    """Inventory actual consumers before selecting the one extra-Window group."""
    executable = [(gid, g) for gid, members in enumerate(attn_groups) for g in members]
    native = [(gid, g) for gid, g in executable if getattr(g.kv_cache_spec, "sliding_window", None) is not None]
    if not native:
        return attn_groups, None
    _require(
        input_buffers is not None and type(causal) is not bool,
        "native SWA Graph requires actual private inputs and per-group causality",
    )
    full = [
        (gid, g)
        for gid, g in executable
        if type(g.kv_cache_spec) is FullAttentionSpec
        and g.kv_cache_spec.sliding_window is None
        and g.kv_cache_spec.attention_chunk_size is None
    ]
    _require(
        len(full) == 1 and len(full) + len(native) == len(executable),
        "native SWA reviewed route requires one Full group and explicitly inventoried native groups",
    )
    all_layers = [name for _, g in executable for name in g.layer_names]
    _require(all_layers and len(all_layers) == len(set(all_layers)), "native SWA/Full layer owners overlap")
    for gid, g in executable:
        _require(
            g.kv_cache_group_id == gid
            and g.layer_names
            and g.backend is AscendAttentionBackend
            and type(g.get_metadata_builder()) is AscendAttentionMetadataBuilder,
            "native SWA requires actual per-gid dense FIA consumers/builders",
        )
        logical, kernel = int(block_tables.block_sizes[gid]), int(block_tables.kernel_block_sizes[gid])
        _require(
            g.kv_cache_spec.block_size == logical
            and kernel > 0
            and logical % kernel == 0
            and block_tables.input_block_tables[gid].ndim == 2
            and block_tables.input_block_tables[gid].stride(-1) == 1,
            "native SWA canonical block geometry differs from its consumer",
        )
    _require(
        block_tables.cp_size == 1 and block_tables.cp_rank == 0,
        "native SWA passthrough has no context-parallel interval contract",
    )
    groups = []
    for gid, g in native:
        spec = g.kv_cache_spec
        _require(
            type(spec) in (SlidingWindowSpec, FullAttentionSpec)
            and causal[gid] is True
            and contains_native_band(window_size, spec.sliding_window, query_width),
            "narrow native SWA Window needs explicit interval projection; passthrough is unsafe",
        )
        table = block_tables.input_block_tables[gid]
        groups.append(
            NativeSWAGraphGroup(
                gid, tuple(g.layer_names), spec.sliding_window, table.data_ptr(), tuple(table.shape), table.dtype
            )
        )
    gid, g = full[0]
    _require(
        causal[gid] is False and input_buffers.seq_lens.ndim == 1 and input_buffers.seq_lens.dtype == torch.int32,
        "native SWA review requires the separate noncausal Full group and prepared int32 lengths",
    )
    plan = NativeSWAGraphPlan(
        gid,
        tuple(g.layer_names),
        tuple(groups),
        id(block_tables),
        id(input_buffers),
        input_buffers.seq_lens.data_ptr(),
        input_buffers.seq_lens.device,
        query_width,
        window_size,
    )
    return plan.filter_groups(attn_groups, native=False), plan


def build_group_metadata(owner: Any, *, read_inputs: Any, builder_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Use the existing assembler twice with distinct, explicit read operands."""
    from vllm_ascend.worker.v2.spec_decode.draft_window_eager import AscendDraftAttentionReadInputs

    plan = owner._draft_window_source_capability.native_swa_graph
    snapshot = getattr(owner, "_draft_window_native_full_snapshot", None)
    inputs = owner._draft_window_live_inputs
    _require(
        isinstance(plan, NativeSWAGraphPlan)
        and isinstance(snapshot, NativeSWAFullSnapshot)
        and id(owner.input_buffers) == plan.input_owner_id,
        "native SWA current full snapshot missing",
    )
    snapshot.validate(inputs.identity, plan)
    n = inputs.padded_request_count
    absolute = AscendDraftAttentionReadInputs(
        block_tables=tuple(t[:n] for t in owner.block_tables.input_block_tables),
        seq_lens=snapshot.device_exact,
        seq_lens_cpu_exact=snapshot.cpu_exact,
        seq_lens_cpu_upper_bound=snapshot.cpu_upper,
        max_seq_len=int(snapshot.cpu_upper.max().item()),
    )
    full = owner._build_ascend_draft_attn_metadata(
        read_inputs=read_inputs, attn_groups=plan.filter_groups(owner.attn_groups, native=False), **builder_kwargs
    )
    native = owner._build_ascend_draft_attn_metadata(
        read_inputs=absolute,
        attn_groups=plan.filter_groups(owner.attn_groups, native=True),
        **dict(builder_kwargs, causal=owner._group_causal),
    )
    _require(
        set(full) == set(plan.full_layers) and set(native) == set(plan.native_layers),
        "native SWA metadata assembler returned missing/foreign groups",
    )
    return {**native, **full}
