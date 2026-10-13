# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Serialized eager consumer for the MRV2 Draft Window read plane.

This module is intentionally limited to the P4a dense, single-group route. It
connects P1/P2/P3 read state through an Ascend-owned metadata-assembly seam,
while leaving positions, query layout, slot mappings, and Target metadata
untouched. The upstream vLLM checkout remains unmodified.
"""

import copy
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch
import vllm.v1.worker.gpu.spec_decode.speculator as upstream_speculator_module
from vllm.v1.worker.gpu.cp_utils import maybe_prepare_dcp_local_seq_lens
from vllm.v1.worker.gpu.input_batch import InputBatch

from vllm_ascend.worker.v2.spec_decode.draft_window import (
    DraftMetadataInputs,
    DraftWindowExecutionPhase,
    DraftWindowGenerationIdentity,
    DraftWindowLeaseState,
    DraftWindowReadDescriptor,
    DraftWindowRuntimeState,
    DraftWindowSourceCapability,
    SingleGroupDirectPrefixProvider,
    compute_draft_window_scalar_plan,
    gather_valid_num_blocks_cpu,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_graph import (
    DraftWindowGraphStage,
    observe_draft_window_graph_source,
)


@dataclass(frozen=True, slots=True)
class AscendDraftAttentionReadInputs:
    """Ascend-owned, read-only transport for one Draft metadata build.

    The repository-local seam cannot override positions, query layout, or slot
    mappings. CPU exact and upper lengths are already final for this build.
    """

    block_tables: tuple[torch.Tensor, ...]
    seq_lens: torch.Tensor
    seq_lens_cpu_exact: torch.Tensor
    seq_lens_cpu_upper_bound: torch.Tensor
    max_seq_len: int
    dcp_local_seq_lens: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.max_seq_len < 0:
            raise ValueError("Draft attention max sequence length must be non-negative")
        for name, tensor in (
            ("device sequence lengths", self.seq_lens),
            ("CPU exact sequence lengths", self.seq_lens_cpu_exact),
            ("CPU upper sequence lengths", self.seq_lens_cpu_upper_bound),
        ):
            if tensor.ndim != 1:
                raise ValueError(f"Draft attention {name} must be one-dimensional")
            if tensor.dtype != torch.int32:
                raise TypeError(f"Draft attention {name} must use int32")
        if self.seq_lens_cpu_exact.device.type != "cpu":
            raise ValueError("Draft attention CPU exact sequence lengths must live on CPU")
        if self.seq_lens_cpu_upper_bound.device.type != "cpu":
            raise ValueError("Draft attention CPU upper sequence lengths must live on CPU")
        if self.seq_lens.shape != self.seq_lens_cpu_exact.shape:
            raise ValueError("Draft attention device and CPU exact lengths must have the same shape")
        if self.seq_lens_cpu_exact.shape != self.seq_lens_cpu_upper_bound.shape:
            raise ValueError("Draft attention CPU exact and upper lengths must have the same shape")


@dataclass(frozen=True, slots=True)
class DraftWindowProposalContext:
    """Caller-owned identity for one real outer ``propose`` invocation."""

    mapping_epoch: int
    input_batch_id: int
    ordered_request_ids: tuple[str, ...]
    idx_mapping_fingerprint: tuple[int, ...]


class DraftWindowProposalEpochSource:
    """Issue exactly one monotonic mapping epoch per real proposal."""

    def __init__(self) -> None:
        self._next_epoch = 0
        self._active: DraftWindowProposalContext | None = None

    def begin(self, input_batch: InputBatch | Any) -> DraftWindowProposalContext:
        if self._active is not None:
            raise RuntimeError("a draft-window proposal epoch is already active")
        request_ids, mapping = _proposal_fingerprint(input_batch)
        context = DraftWindowProposalContext(
            mapping_epoch=self._next_epoch,
            input_batch_id=id(input_batch),
            ordered_request_ids=request_ids,
            idx_mapping_fingerprint=mapping,
        )
        self._next_epoch += 1
        self._active = context
        return context

    def validate(
        self,
        expected: DraftWindowProposalContext,
        input_batch: InputBatch | Any,
    ) -> None:
        if self._active is not expected:
            raise RuntimeError("stale proposal epoch is not the active generation")
        if id(input_batch) != expected.input_batch_id:
            raise RuntimeError("proposal input batch object changed within one epoch")
        request_ids, mapping = _proposal_fingerprint(input_batch)
        if request_ids != expected.ordered_request_ids:
            raise RuntimeError("proposal request ordering changed within one epoch")
        if mapping != expected.idx_mapping_fingerprint:
            raise RuntimeError("proposal idx_mapping changed within one epoch")

    def finish(self, expected: DraftWindowProposalContext) -> None:
        if self._active is not expected:
            raise RuntimeError("stale proposal epoch cannot finish the active generation")
        self._active = None


@dataclass(frozen=True, slots=True)
class DraftWindowMaterializationPlan:
    """Host-side P4a plan whose tensors share one proposal generation."""

    read_descriptors: tuple[DraftWindowReadDescriptor, ...]
    cpu_window_exact: torch.Tensor
    cpu_window_upper: torch.Tensor
    cpu_valid_num_blocks: torch.Tensor


def build_eager_window_materialization_plan(
    *,
    provider: SingleGroupDirectPrefixProvider,
    cpu_full_exact: torch.Tensor,
    cpu_apply_time_full_upper: torch.Tensor,
    cpu_valid_num_blocks: torch.Tensor,
    actual_request_count: int,
    padded_request_count: int,
) -> DraftWindowMaterializationPlan:
    """Build descriptors and exact/upper Window lengths from one CPU snapshot."""
    if actual_request_count < 0:
        raise ValueError("actual request count must be greater than or equal to 0")
    if padded_request_count < actual_request_count:
        raise ValueError("padded request count must be at least the actual request count")
    for name, tensor in (
        ("full exact", cpu_full_exact),
        ("apply-time full upper", cpu_apply_time_full_upper),
        ("valid block counts", cpu_valid_num_blocks),
    ):
        if tensor.device.type != "cpu":
            raise ValueError(f"CPU {name} must live on CPU")
        if tensor.dtype != torch.int32:
            raise TypeError(f"CPU {name} must use int32")
        if tensor.ndim != 1 or tensor.shape[0] != padded_request_count:
            raise ValueError(f"CPU {name} must match the padded request count")

    cpu_window_exact = torch.zeros(padded_request_count, dtype=torch.int32)
    cpu_window_upper = torch.zeros(padded_request_count, dtype=torch.int32)
    descriptors: list[DraftWindowReadDescriptor] = []
    for row in range(actual_request_count):
        full_exact = int(cpu_full_exact[row].item())
        full_upper = int(cpu_apply_time_full_upper[row].item())
        valid_num_blocks = int(cpu_valid_num_blocks[row].item())
        if full_exact < 0:
            raise ValueError("prepared full exact length must be non-negative")
        if full_upper < full_exact:
            raise ValueError("apply-time full upper must be greater than or equal to prepared exact")
        scalar_plan = compute_draft_window_scalar_plan(
            full_exact_end=full_exact,
            future_offset=0,
            window_size=provider.inventory.window_size,
            legacy_block_size=provider.legacy_block_size,
        )
        descriptor = provider.describe(
            scalar_plan,
            valid_num_blocks=valid_num_blocks,
        )
        visible = scalar_plan.window_visible_length
        if descriptor.safe_extent_tokens < visible:
            raise RuntimeError("provider safe extent is shorter than Window exact")
        raw_upper = max(
            full_upper - scalar_plan.window_start,
            visible,
        )
        window_upper = min(raw_upper, descriptor.safe_extent_tokens)
        if window_upper < visible:
            raise RuntimeError("Window upper is shorter than Window exact")
        cpu_window_exact[row] = visible
        cpu_window_upper[row] = window_upper
        descriptors.append(descriptor)

    return DraftWindowMaterializationPlan(
        read_descriptors=tuple(descriptors),
        cpu_window_exact=cpu_window_exact,
        cpu_window_upper=cpu_window_upper,
        cpu_valid_num_blocks=cpu_valid_num_blocks,
    )


class DraftWindowEagerMixin:
    """Shared P4a integration for Ascend DFlash and eligible DSpark."""

    _draft_window_runtime_state: DraftWindowRuntimeState
    _draft_window_source_capability: DraftWindowSourceCapability
    _draft_window_epoch_source: DraftWindowProposalEpochSource
    _draft_window_proposal_context: DraftWindowProposalContext | None
    _draft_window_live_inputs: DraftMetadataInputs | None
    _draft_window_bypass_active: bool
    input_batch: InputBatch
    input_buffers: Any
    block_tables: Any
    model_state: Any
    draft_attn_layer_names: set[str] | None
    arange: torch.Tensor
    attn_groups: Any
    kv_cache_config: Any
    num_query_per_req: int
    max_model_len: int

    def _initialize_draft_window_eager_state(self) -> None:
        if not hasattr(self, "_draft_window_runtime_state"):
            raise RuntimeError("P4a eager state requires initialized P3 runtime state")
        if hasattr(self, "_draft_window_epoch_source"):
            raise RuntimeError("P4a eager state is already initialized")
        self._draft_window_epoch_source = DraftWindowProposalEpochSource()
        self._draft_window_proposal_context: DraftWindowProposalContext | None = None
        self._draft_window_live_inputs: DraftMetadataInputs | None = None
        self._draft_window_bypass_active = False

    @contextmanager
    def _draft_window_proposal_scope(
        self,
        input_batch: InputBatch,
        *,
        dummy_run: bool,
        is_profile: bool,
    ) -> Iterator[None]:
        epoch_source: DraftWindowProposalEpochSource | None = getattr(
            self,
            "_draft_window_epoch_source",
            None,
        )
        if epoch_source is None:
            yield
            return
        bypass_for_prefill = False
        if not dummy_run and not is_profile:
            bypass_for_prefill = _requires_prefill_baseline_bypass(input_batch)
        if dummy_run or is_profile or bypass_for_prefill:
            if self._draft_window_bypass_active or self._draft_window_proposal_context is not None:
                raise RuntimeError("a draft-window proposal scope is already active")
            self._draft_window_bypass_active = True
            try:
                # Startup/profile and pure-prefill proposals exercise the
                # unmodified upstream metadata path without allocating a
                # production Window generation. Mixed prefill/decode remains
                # unsupported and is rejected by the classifier above.
                yield
            finally:
                self._draft_window_bypass_active = False
            return
        context = epoch_source.begin(input_batch)
        self._draft_window_proposal_context = context
        succeeded = False
        try:
            yield
            succeeded = True
        finally:
            live_inputs = self._draft_window_live_inputs
            try:
                if live_inputs is not None and live_inputs.lease.state is not DraftWindowLeaseState.RETIRED:
                    if succeeded:
                        self._draft_window_runtime_state.retire_generation(live_inputs.identity)
                    else:
                        self._draft_window_runtime_state.abort_generation(live_inputs.identity)
                elif succeeded:
                    raise RuntimeError("eligible P4a proposal completed without consuming Draft Window inputs")
            finally:
                self._draft_window_live_inputs = None
                self._draft_window_proposal_context = None
                epoch_source.finish(context)

    def _snapshot_prepared_draft_seq_lens(
        self,
        num_reqs_padded: int,
    ) -> torch.Tensor:
        """Perform the one controlled D2H already present in the baseline builder."""
        return self.input_buffers.seq_lens[:num_reqs_padded].to(
            device="cpu",
            dtype=torch.int32,
            copy=True,
        )

    def _materialize_draft_window_read_inputs(
        self,
        *,
        num_reqs: int,
        num_reqs_padded: int,
    ) -> DraftMetadataInputs:
        context = self._require_current_draft_window_proposal()
        input_batch = self.input_batch
        self._draft_window_epoch_source.validate(context, input_batch)
        if num_reqs != input_batch.num_reqs:
            raise RuntimeError("metadata request count does not match proposal epoch")
        if num_reqs_padded < num_reqs:
            raise ValueError("padded request count must be at least actual requests")

        cpu_full_exact = self._snapshot_prepared_draft_seq_lens(num_reqs_padded)
        cpu_apply_time_full_upper = torch.zeros(
            num_reqs_padded,
            dtype=torch.int32,
        )
        if num_reqs:
            torch.add(
                input_batch.seq_lens_cpu_upper_bound[:num_reqs],
                self.num_query_per_req,
                out=cpu_apply_time_full_upper[:num_reqs],
            )
            cpu_apply_time_full_upper[:num_reqs].clamp_(max=self.max_model_len)
        capability = self._draft_window_source_capability
        cpu_valid_num_blocks = gather_valid_num_blocks_cpu(
            capability.full_source,
            global_group_id=capability.provider.global_group_id,
            idx_mapping_np=input_batch.idx_mapping_np,
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs_padded,
        )
        plan = build_eager_window_materialization_plan(
            provider=capability.provider,
            cpu_full_exact=cpu_full_exact,
            cpu_apply_time_full_upper=cpu_apply_time_full_upper,
            cpu_valid_num_blocks=cpu_valid_num_blocks,
            actual_request_count=num_reqs,
            padded_request_count=num_reqs_padded,
        )
        # Recheck freshness after the controlled length snapshot and host plan.
        self._draft_window_epoch_source.validate(context, input_batch)
        return self._draft_window_runtime_state.materialize(
            mapping_epoch=context.mapping_epoch,
            ordered_request_ids=context.ordered_request_ids,
            idx_mapping=input_batch.idx_mapping,
            read_descriptors=plan.read_descriptors,
            device_window_exact=None,
            cpu_window_exact=plan.cpu_window_exact,
            cpu_window_upper=plan.cpu_window_upper,
            cpu_valid_num_blocks=plan.cpu_valid_num_blocks,
            num_reqs_padded=num_reqs_padded,
            phase=DraftWindowExecutionPhase.PROPOSAL,
            step=0,
            execution_lane=0,
        )

    def _validated_draft_attention_read_inputs(
        self,
        inputs: DraftMetadataInputs,
        expected_identity: DraftWindowGenerationIdentity,
    ) -> AscendDraftAttentionReadInputs:
        """The single formal P4a validation and transport-narrowing boundary."""
        inputs.validate_for(expected_identity)
        observe_draft_window_graph_source(self, DraftWindowGraphStage.LIVE)
        global_group_id = expected_identity.global_group_id
        block_tables = tuple(
            (inputs.window_block_table if gid == global_group_id else full_table[: inputs.padded_request_count])
            for gid, full_table in enumerate(self.block_tables.input_block_tables)
        )
        active_upper = inputs.cpu_window_upper[: inputs.actual_request_count]
        max_seq_len = int(active_upper.max().item()) if active_upper.numel() else 0
        return AscendDraftAttentionReadInputs(
            block_tables=block_tables,
            seq_lens=inputs.device_window_exact,
            seq_lens_cpu_exact=inputs.cpu_window_exact,
            seq_lens_cpu_upper_bound=inputs.cpu_window_upper,
            max_seq_len=max_seq_len,
        )

    def _build_draft_attn_metadata(
        self,
        num_reqs: int,
        num_reqs_padded: int,
        num_tokens_padded: int,
        seq_lens_cpu_upper_bound: torch.Tensor,
        step: int,
        num_query_per_req: int | None = None,
        causal: bool | Mapping[int, bool] = False,
        query_start_loc_np: np.ndarray | None = None,
        dcp_local_seq_lens: torch.Tensor | None = None,
    ) -> dict[str, Any] | None:
        runtime_state = getattr(self, "_draft_window_runtime_state", None)
        if runtime_state is None or getattr(self, "_draft_window_bypass_active", False):
            return super()._build_draft_attn_metadata(  # type: ignore[misc]
                num_reqs,
                num_reqs_padded,
                num_tokens_padded,
                seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                step=step,
                num_query_per_req=num_query_per_req,
                causal=causal,
                query_start_loc_np=query_start_loc_np,
                dcp_local_seq_lens=dcp_local_seq_lens,
            )
        if not self.draft_attn_layer_names:
            return None
        if num_query_per_req is not None:
            raise AssertionError("DFlash/DSpark query width must come from the speculator instance")
        if step != self.num_query_per_req:
            raise RuntimeError("P4a apply-time step does not match the actual query width")
        target_metadata = self._require_nonempty_target_metadata()
        inputs = self._materialize_draft_window_read_inputs(
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs_padded,
        )
        # Register ownership immediately. The outer proposal scope is the one
        # exception/retirement owner if validation, building, or publishing
        # fails after the workspace has been materialized.
        self._draft_window_live_inputs = inputs
        live_read_inputs = self._validated_draft_attention_read_inputs(
            inputs,
            inputs.identity,
        )
        metadata = self._build_ascend_draft_attn_metadata(
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs_padded,
            num_tokens_padded=num_tokens_padded,
            read_inputs=live_read_inputs,
            causal=causal,
            query_start_loc_np=query_start_loc_np,
            dcp_local_seq_lens=dcp_local_seq_lens,
        )
        if metadata is None:
            raise RuntimeError("eligible P4a route produced no Draft metadata")
        if self.model_state.attn_metadata is not target_metadata:
            raise RuntimeError("Target metadata owner changed during Draft metadata build")
        metadata = _isolate_draft_metadata_from_target(
            metadata,
            target_metadata,
        )
        runtime_state.publish_metadata(
            inputs,
            metadata,
            target_metadata=target_metadata,
        )
        return runtime_state.read_metadata(inputs.identity)

    def _build_ascend_draft_attn_metadata(
        self,
        *,
        num_reqs: int,
        num_reqs_padded: int,
        num_tokens_padded: int,
        read_inputs: AscendDraftAttentionReadInputs,
        causal: bool | Mapping[int, bool],
        query_start_loc_np: np.ndarray | None,
        dcp_local_seq_lens: torch.Tensor | None,
    ) -> dict[str, Any]:
        """Vendor-local copy of the upstream metadata-assembly seam.

        Only query layout and builder kwargs are reproduced. Proposal logic,
        input preparation, slot calculation, forward, and sampling remain in
        upstream. ``UPSTREAM_DRIFT_GUARD`` tests pin the source contract.
        """
        if query_start_loc_np is not None:
            raise NotImplementedError("MRV2 draft-window P4a requires the uniform DFlash/DSpark query layout")
        query_start_loc_cpu = torch.clamp(self.arange[: num_reqs_padded + 1], max=num_reqs) * self.num_query_per_req
        max_query_len = self.num_query_per_req

        if len(read_inputs.block_tables) != len(self.block_tables.input_block_tables):
            raise ValueError("Draft attention read block-table group count does not match the configured cache groups")
        if read_inputs.seq_lens.shape[0] != num_reqs_padded:
            raise ValueError("Draft attention read lengths do not match the padded request count")
        if dcp_local_seq_lens is None:
            dcp_local_seq_lens = read_inputs.dcp_local_seq_lens
        if dcp_local_seq_lens is None and self.block_tables.cp_size > 1:
            dcp_local_seq_lens = maybe_prepare_dcp_local_seq_lens(
                self.input_buffers.dcp_local_seq_lens,
                read_inputs.seq_lens,
                num_reqs,
                self.block_tables.cp_size,
                self.block_tables.cp_rank,
                self.block_tables.cp_interleave,
            )

        builder_kwargs = dict(
            attn_groups=self.attn_groups,
            num_reqs=num_reqs_padded,
            num_tokens=num_tokens_padded,
            query_start_loc_gpu=self.input_buffers.query_start_loc[: num_reqs_padded + 1],
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=read_inputs.seq_lens,
            seq_lens_cpu=read_inputs.seq_lens_cpu_exact,
            dcp_local_seq_lens=(None if dcp_local_seq_lens is None else dcp_local_seq_lens[:num_reqs_padded]),
            max_seq_len=read_inputs.max_seq_len,
            block_tables=list(read_inputs.block_tables),
            slot_mappings=self.block_tables.slot_mappings[:, :num_tokens_padded],
            kv_cache_config=self.kv_cache_config,
            causal=causal,
            seq_lens_cpu_upper_bound=read_inputs.seq_lens_cpu_upper_bound,
        )
        return self._call_ascend_draft_attn_metadata_builder(
            read_inputs=read_inputs,
            builder_kwargs=builder_kwargs,
        )

    def _call_ascend_draft_attn_metadata_builder(
        self,
        *,
        read_inputs: AscendDraftAttentionReadInputs,
        builder_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        del read_inputs
        # Existing Ascend scopes replace this function with the Ascend builder
        # (and, for DSpark, the positions/is_prefilling factory). Window state
        # is carried only by explicit kwargs; no global/context transport is
        # introduced here.
        return upstream_speculator_module.build_attn_metadata(**builder_kwargs)

    def _require_nonempty_target_metadata(self) -> Mapping[Any, Any]:
        target_metadata = self.model_state.attn_metadata
        if not isinstance(target_metadata, Mapping) or not target_metadata:
            raise ValueError("Window ON requires non-empty Target metadata from model_state.attn_metadata")
        return target_metadata

    def _require_current_draft_window_proposal(
        self,
    ) -> DraftWindowProposalContext:
        context = self._draft_window_proposal_context
        if context is None:
            raise RuntimeError("Draft Window metadata build has no active proposal epoch")
        return context


def _proposal_fingerprint(
    input_batch: InputBatch | Any,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    num_reqs = int(input_batch.num_reqs)
    request_ids = tuple(input_batch.req_ids[:num_reqs])
    if len(request_ids) != num_reqs:
        raise ValueError("proposal request IDs do not match num_reqs")
    mapping = np.asarray(input_batch.idx_mapping_np)
    if mapping.ndim != 1 or mapping.shape[0] < num_reqs:
        raise ValueError("proposal CPU idx_mapping does not cover active requests")
    if not np.issubdtype(mapping.dtype, np.integer):
        raise TypeError("proposal CPU idx_mapping must contain integers")
    return request_ids, tuple(int(value) for value in mapping[:num_reqs])


def _requires_prefill_baseline_bypass(
    input_batch: InputBatch | Any,
) -> bool:
    """Classify the pure-prefill phase that must use baseline full reads.

    Upstream invokes the speculator after every target step, including every
    chunk of a long prompt and the chunk that completes it. A batch containing
    only prefill requests uses the baseline full-context metadata path and
    allocates no Window generation. A mixed prefill/decode batch remains outside
    the supported route and fails before epoch/workspace creation.
    """
    num_reqs = int(input_batch.num_reqs)
    is_prefilling = np.asarray(input_batch.is_prefilling_np)[:num_reqs]
    if is_prefilling.ndim != 1 or is_prefilling.shape[0] != num_reqs:
        raise ValueError("prefill classification does not cover active requests")
    has_prefill = bool(input_batch.has_prefill)
    any_prefill = bool(np.any(is_prefilling))
    if has_prefill != any_prefill:
        raise RuntimeError("InputBatch has_prefill disagrees with is_prefilling_np")
    if not any_prefill:
        return False
    if not bool(np.all(is_prefilling)):
        raise NotImplementedError("MRV2 draft-window P4a does not support mixed prefill/decode proposals")
    return True


def _isolate_draft_metadata_from_target(
    draft_metadata: Any,
    target_metadata: Any,
) -> Any:
    """Clone only Draft-side mutable references that alias real Target state."""
    target_object_ids, target_tensor_storage, target_numpy_arrays = _collect_mutable_identity(target_metadata)
    memo: dict[int, Any] = {}

    def isolate(value: Any) -> Any:
        if value is None or isinstance(
            value,
            (
                bool,
                int,
                float,
                complex,
                str,
                bytes,
                type,
                Enum,
                torch.dtype,
                torch.device,
            ),
        ):
            return value
        value_id = id(value)
        if value_id in memo:
            return memo[value_id]
        if isinstance(value, torch.Tensor):
            if value_id in target_object_ids or _tensor_storage_key(value) in target_tensor_storage:
                cloned = value.clone()
                memo[value_id] = cloned
                return cloned
            return value
        if isinstance(value, np.ndarray):
            if value_id in target_object_ids or any(
                np.shares_memory(value, target_array) for target_array in target_numpy_arrays
            ):
                cloned_array = value.copy()
                memo[value_id] = cloned_array
                return cloned_array
            return value
        if isinstance(value, dict):
            result = {} if value_id in target_object_ids else value
            memo[value_id] = result
            items = list(value.items())
            if result is value:
                for key, item in items:
                    value[key] = isolate(item)
            else:
                result.update((key, isolate(item)) for key, item in items)
            return result
        if isinstance(value, list):
            result_list: list[Any] = [] if value_id in target_object_ids else value
            memo[value_id] = result_list
            isolated_items = [isolate(item) for item in value]
            result_list[:] = isolated_items
            return result_list
        if isinstance(value, tuple):
            result_tuple = tuple(isolate(item) for item in value)
            memo[value_id] = result_tuple
            return result_tuple
        if isinstance(value, set):
            result_set = {isolate(item) for item in value}
            if value_id not in target_object_ids:
                value.clear()
                value.update(result_set)
                result_set = value
            memo[value_id] = result_set
            return result_set
        if is_dataclass(value) and not isinstance(value, type):
            # Always operate on a shallow Draft-side copy. A mutable dataclass
            # may itself be shared with Target; mutating it before discovering
            # that alias would corrupt Target metadata. object.__setattr__ also
            # handles frozen/slotted dataclass copies without touching Target.
            result_dataclass = copy.copy(value)
            memo[value_id] = result_dataclass
            for data_field in fields(value):
                object.__setattr__(
                    result_dataclass,
                    data_field.name,
                    isolate(getattr(value, data_field.name)),
                )
            return result_dataclass
        values = getattr(value, "__dict__", None)
        result_object = copy.copy(value) if value_id in target_object_ids else value
        memo[value_id] = result_object
        if isinstance(values, dict):
            for name, item in list(values.items()):
                setattr(result_object, name, isolate(item))
        for owner_type in type(value).__mro__:
            slots = owner_type.__dict__.get("__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for name in slots:
                if name in ("__dict__", "__weakref__"):
                    continue
                try:
                    setattr(result_object, name, isolate(getattr(value, name)))
                except (AttributeError, TypeError):
                    continue
        return result_object

    return isolate(draft_metadata)


def _collect_mutable_identity(
    value: Any,
) -> tuple[set[int], set[tuple[str, int]], tuple[np.ndarray, ...]]:
    object_ids: set[int] = set()
    tensor_storage: set[tuple[str, int]] = set()
    numpy_arrays: list[np.ndarray] = []
    seen: set[int] = set()

    def visit(current: Any) -> None:
        if current is None or isinstance(
            current,
            (
                bool,
                int,
                float,
                complex,
                str,
                bytes,
                type,
                Enum,
                torch.dtype,
                torch.device,
            ),
        ):
            return
        current_id = id(current)
        if current_id in seen:
            return
        seen.add(current_id)
        if isinstance(current, torch.Tensor):
            object_ids.add(current_id)
            tensor_storage.add(_tensor_storage_key(current))
            return
        if isinstance(current, np.ndarray):
            object_ids.add(current_id)
            numpy_arrays.append(current)
            return
        if isinstance(current, Mapping):
            object_ids.add(current_id)
            for key, item in current.items():
                visit(key)
                visit(item)
            return
        if isinstance(current, (list, set, bytearray, tuple)):
            if not isinstance(current, tuple):
                object_ids.add(current_id)
            for item in current:
                visit(item)
            return
        if is_dataclass(current) and not isinstance(current, type):
            object_ids.add(current_id)
            for data_field in fields(current):
                visit(getattr(current, data_field.name))
            return
        values = getattr(current, "__dict__", None)
        if isinstance(values, dict):
            object_ids.add(current_id)
            for item in values.values():
                visit(item)
        for owner_type in type(current).__mro__:
            slots = owner_type.__dict__.get("__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for name in slots:
                if name in ("__dict__", "__weakref__"):
                    continue
                try:
                    visit(getattr(current, name))
                except AttributeError:
                    continue

    visit(value)
    return object_ids, tensor_storage, tuple(numpy_arrays)


def _tensor_storage_key(tensor: torch.Tensor) -> tuple[str, int]:
    return str(tensor.device), tensor.untyped_storage().data_ptr()
