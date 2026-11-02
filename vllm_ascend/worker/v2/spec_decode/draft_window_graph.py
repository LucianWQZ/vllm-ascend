# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Persistent Draft Graph read banks and runtime binding validation.

Startup capture fixes addresses. Each proposal updates only its private read
banks and metadata; canonical Full write tables, slots, and absolute positions
remain unchanged. The bank owner retains current generation and request
identity through the actual typed Graph UPDATE entrance.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

from vllm_ascend.worker.v2.spec_decode.draft_window import (
    DraftMetadataInputs,
    DraftWindowGenerationIdentity,
    DraftWindowReadDescriptor,
    DraftWindowSourceCapability,
    DraftWindowWorkspace,
    _validate_idx_mapping,
)

if TYPE_CHECKING:
    from vllm_ascend.worker.v2.spec_decode.draft_window_eager import DraftWindowProposalContext


class DraftWindowGraphRoute(Enum):
    """Method-specific graph source route; never a generic support boolean."""

    EAGLE3_FIRST_PASS = "eagle3_first_pass"
    EAGLE3_AR_STEP = "eagle3_ar_step"
    DFLASH = "dflash"
    DSPARK = "dspark"


class DraftWindowGraphCarrierState(Enum):
    """Lifecycle of one persistent per-consumer carrier backing."""

    EMPTY = "empty"
    BOUND = "bound"
    SUBMITTED = "submitted"
    UPDATED = "updated"


@dataclass(frozen=True, slots=True)
class DraftWindowGraphTensorIdentity:
    """Address/geometry fingerprint; deliberately not a freshness token."""

    storage_data_ptr: int
    data_ptr: int
    storage_offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    device: str


@dataclass(frozen=True, slots=True)
class DraftWindowGraphCaptureIdentity:
    """One real manager descriptor and its method-specific consumer."""

    route: DraftWindowGraphRoute
    descriptor: BatchExecutionDescriptor
    consumer_ordinal: int


@dataclass(slots=True)
class DraftWindowGraphReadCarrier:
    """One persistent capture operand rebound to explicit runtime metadata."""

    capture_identity: DraftWindowGraphCaptureIdentity
    workspace: DraftWindowWorkspace
    backing_snapshot: tuple[DraftWindowGraphTensorIdentity, ...]
    inputs: DraftMetadataInputs | None = None
    state: DraftWindowGraphCarrierState = DraftWindowGraphCarrierState.EMPTY


_ROUTE_METHODS = {
    DraftWindowGraphRoute.EAGLE3_FIRST_PASS: "eagle3",
    DraftWindowGraphRoute.EAGLE3_AR_STEP: "eagle3",
    DraftWindowGraphRoute.DFLASH: "dflash",
    DraftWindowGraphRoute.DSPARK: "dspark",
}


def _tensor_identity(tensor: torch.Tensor) -> DraftWindowGraphTensorIdentity:
    try:
        storage_data_ptr = tensor.untyped_storage().data_ptr()
    except (AttributeError, RuntimeError):
        storage_data_ptr = tensor.data_ptr()
    return DraftWindowGraphTensorIdentity(
        storage_data_ptr=storage_data_ptr,
        data_ptr=tensor.data_ptr(),
        storage_offset=tensor.storage_offset(),
        shape=tuple(tensor.shape),
        stride=tuple(tensor.stride()),
        dtype=str(tensor.dtype),
        device=str(tensor.device),
    )


def _bank_snapshot(backing: Any) -> tuple[DraftWindowGraphTensorIdentity, ...]:
    """Check the stable Graph operands, including offset/stride."""
    return tuple(
        _tensor_identity(getattr(backing, name))
        for name in (
            "window_block_table",
            "device_window_exact",
            "cpu_window_exact",
            "cpu_window_upper",
        )
    )


def _validate_route(
    route: DraftWindowGraphRoute,
    generation: DraftWindowGenerationIdentity,
    method: str,
) -> None:
    if not isinstance(route, DraftWindowGraphRoute):
        raise TypeError("graph route must be a DraftWindowGraphRoute")
    if method != _ROUTE_METHODS[route]:
        raise ValueError(f"graph route method mismatch: route={route.value} method={method}")
    if route is DraftWindowGraphRoute.EAGLE3_AR_STEP:
        if generation.step <= 0:
            raise ValueError(f"graph route {route.value} requires an autoregressive step")
    elif generation.step != 0:
        raise ValueError(f"graph route {route.value} requires proposal step 0")


class DraftWindowGraphBindingProvider:
    """Own stable capture banks and bind each current runtime generation."""

    def __init__(self, *, source: DraftWindowSourceCapability) -> None:
        if not isinstance(source, DraftWindowSourceCapability):
            raise TypeError("Graph binding provider requires its actual Draft source")
        self._source = source
        group = source.group
        source.full_source.validate_owner_bindings()
        source_table = source.full_source._input_block_tables[group.gid]
        persistent_counts = source.full_source._block_tables_owner.num_blocks.gpu[group.gid]
        if source_table.dtype != torch.int32 or persistent_counts.dtype != torch.int32:
            raise TypeError("Draft Graph source tables and valid counts must use int32")
        if source_table.device != persistent_counts.device:
            raise ValueError("Draft Graph source table and valid counts must share a device")
        table_capacity_columns = (
            source.capacity_columns if source.workspace_capacity_columns is None else source.workspace_capacity_columns
        )
        if table_capacity_columns < source.capacity_columns:
            raise ValueError("configured workspace capacity is smaller than the provider capacity")
        if table_capacity_columns > source_table.shape[1]:
            raise ValueError("configured workspace capacity exceeds the canonical source row width")
        self._tp_identity = self._capture_topology()
        if source.group.read_semantics != "replicated" or any(
            getattr(source.parallel_config, name) != 1
            for name in ("pipeline_parallel_size", "prefill_context_parallel_size", "decode_context_parallel_size")
        ):
            raise ValueError("Draft Graph read banks require replicated reads and PP=PCP=DCP=1")
        self._workspace: DraftWindowWorkspace | None = None
        self._quarantine_on_failure = False
        self._active_inputs: DraftMetadataInputs | None = None
        self._last_mapping_epoch = -1
        self._last_retired_identity: DraftWindowGenerationIdentity | None = None
        self._quarantined_identity: DraftWindowGenerationIdentity | None = None
        self._carriers: dict[DraftWindowGraphCaptureIdentity, DraftWindowGraphReadCarrier] = {}
        self._submitted_carriers: tuple[DraftWindowGraphReadCarrier, ...] = ()
        self._bound_carriers: dict[DraftWindowGraphRoute, dict[int, DraftWindowGraphReadCarrier]] = {}
        self._validated_graphs: dict[Any, DraftWindowGraphRoute] = {}

    @property
    def source(self) -> DraftWindowSourceCapability:
        return self._source

    @property
    def workspace(self) -> DraftWindowWorkspace:
        # Non-FULL consumers need a temporary read buffer; FULL Graph writes
        # directly to its capture bank and never allocates this scratch.
        if self._workspace is None:
            self._workspace = self.new_workspace()
        return self._workspace

    def new_workspace(self, num_reqs: int | None = None) -> DraftWindowWorkspace:
        capability = self.source
        gid = capability.group.gid
        table = capability.full_source._input_block_tables[gid]
        return DraftWindowWorkspace(
            max_num_reqs=table.shape[0] if num_reqs is None else num_reqs,
            table_capacity_columns=capability.workspace_capacity_columns or capability.capacity_columns,
            table_template=table,
        )

    @property
    def quarantined_identity(self) -> DraftWindowGenerationIdentity | None:
        return self._quarantined_identity

    def quarantine_generation(self, expected_identity: DraftWindowGenerationIdentity) -> None:
        """Keep possibly submitted Draft backing intact until process isolation.

        Unknown native partial submission cannot justify another device write
        or a completion claim. The live input handle intentionally blocks reuse.
        """
        inputs = self._active_inputs
        if inputs is None:
            raise RuntimeError("cannot quarantine a generation without live inputs")
        inputs.validate_for(expected_identity)
        self._quarantined_identity = expected_identity

    def materialize(
        self,
        *,
        proposal: DraftWindowProposalContext,
        idx_mapping: torch.Tensor,
        read_descriptors: Sequence[DraftWindowReadDescriptor],
        cpu_window_exact: torch.Tensor,
        cpu_window_upper: torch.Tensor,
        cpu_valid_num_blocks: torch.Tensor,
        num_reqs_padded: int,
        step: int,
        previous_identity: DraftWindowGenerationIdentity | None = None,
        quarantine_on_failure: bool = False,
        output_workspace: DraftWindowWorkspace | None = None,
    ) -> DraftMetadataInputs:
        """Populate a new proposal or the next retired Eagle consumer's bank."""
        if self._quarantined_identity is not None:
            raise RuntimeError("draft-window backing is quarantined after uncertain Graph submission")
        if self._active_inputs is not None:
            raise RuntimeError("Draft Graph provider still owns an active read bank")
        _validate_idx_mapping(idx_mapping)
        if previous_identity is None:
            if proposal.mapping_epoch <= self._last_mapping_epoch:
                raise ValueError(
                    f"mapping epoch must strictly increase: current={proposal.mapping_epoch} "
                    f"previous={self._last_mapping_epoch}"
                )
        else:
            if self._last_retired_identity is not previous_identity:
                raise RuntimeError("serialized step does not continue the most recently retired generation")
            if proposal is not previous_identity.proposal or proposal.mapping_epoch != self._last_mapping_epoch:
                raise RuntimeError("serialized step proposal or request mapping changed")
            if self.source.policy.normalized_method != "eagle3":
                raise RuntimeError("serialized per-step continuation is supported only for Eagle3")
            if step != previous_identity.step + 1:
                raise RuntimeError(
                    "serialized Draft Window steps must strictly advance by one: "
                    f"previous={previous_identity.step} current={step}"
                )
            if idx_mapping.data_ptr() != previous_identity.idx_mapping_ptr:
                raise RuntimeError("serialized step request mapping changed within one proposal")
        identity = DraftWindowGenerationIdentity(
            proposal=proposal,
            padded_request_count=num_reqs_padded,
            read_descriptors=tuple(read_descriptors),
            idx_mapping_ptr=idx_mapping.data_ptr(),
            step=step,
        )
        workspace = self.workspace if output_workspace is None else output_workspace
        self._quarantine_on_failure = quarantine_on_failure
        self._last_mapping_epoch = max(self._last_mapping_epoch, identity.proposal.mapping_epoch)
        try:
            workspace.materialize(
                identity=identity,
                full_source=self.source.full_source,
                global_group_id=self.source.group.gid,
                cpu_window_exact=cpu_window_exact,
                cpu_window_upper=cpu_window_upper,
                cpu_valid_num_blocks=cpu_valid_num_blocks,
                preserve_on_failure=quarantine_on_failure,
            )
            self._last_retired_identity = None
            inputs = DraftMetadataInputs(identity=identity, workspace=workspace, _owner=self)
            self._active_inputs = inputs
            return inputs
        except BaseException:
            if quarantine_on_failure:
                # A native write may have been submitted before raising, even
                # though no typed inputs exist yet. Keep the workspace and
                # backing; only Host state may be changed before child isolation.
                self._quarantined_identity = identity
                raise
            workspace.reset(identity, clear_buffers=workspace is self._workspace)
            self._active_inputs = None
            raise

    def _handle_active_failure(self, inputs: DraftMetadataInputs) -> None:
        if self._quarantine_on_failure:
            self.quarantine_generation(inputs.identity)
        else:
            self._abort_active(inputs.identity)

    def retire_generation(self, expected_identity: DraftWindowGenerationIdentity) -> None:
        if self._quarantined_identity is not None:
            raise RuntimeError("cannot retire quarantined draft-window backing")
        inputs = self._active_inputs
        if inputs is None:
            raise RuntimeError("cannot use retired Draft inputs")
        inputs.validate_for(expected_identity)
        self._finish_generation(expected_identity)
        self._last_retired_identity = expected_identity

    def abort_generation(self, expected_identity: DraftWindowGenerationIdentity) -> None:
        if self._quarantined_identity is not None:
            raise RuntimeError("cannot abort quarantined draft-window backing")
        inputs = self._active_inputs
        if inputs is None:
            raise RuntimeError("cannot use retired Draft inputs")
        inputs.validate_for(expected_identity)
        self._abort_active(expected_identity)

    def _abort_active(self, identity: DraftWindowGenerationIdentity) -> None:
        if self._quarantined_identity is not None:
            raise RuntimeError("cannot clean up quarantined draft-window backing")
        inputs = self._active_inputs
        if inputs is None:
            return
        inputs.validate_for(identity)
        self._finish_generation(identity)
        self._last_retired_identity = None

    def _finish_generation(self, identity: DraftWindowGenerationIdentity) -> None:
        # Graph bank lifetime belongs to capture, not this Host generation.
        # Clear only at the next same-stream prepare, after the existing tail
        # dependency. Non-FULL scratch follows its serialized current stream.
        inputs = self._active_inputs
        assert inputs is not None
        workspace = inputs.workspace
        workspace.reset(identity, clear_buffers=workspace is self._workspace)
        self._active_inputs = None

    @property
    def submitted_carriers(self) -> tuple[DraftWindowGraphReadCarrier, ...]:
        return self._submitted_carriers

    def registered_capture_identities(
        self,
        *,
        route: DraftWindowGraphRoute,
    ) -> tuple[DraftWindowGraphCaptureIdentity, ...]:
        """Return immutable startup identities for one explicitly typed route."""
        if not isinstance(route, DraftWindowGraphRoute):
            raise TypeError("capture identity lookup requires a graph route")
        return tuple(
            sorted(
                (
                    carrier.capture_identity
                    for carrier in self._carriers.values()
                    if carrier.capture_identity.route is route
                ),
                key=lambda identity: (
                    identity.consumer_ordinal,
                    (identity.descriptor.num_reqs, identity.descriptor.num_tokens),
                ),
            )
        )

    def startup_capture_workspace(
        self,
        capture_identity: DraftWindowGraphCaptureIdentity,
    ) -> DraftWindowWorkspace:
        """Return the real persistent bank before any business generation."""
        carrier = self._require_registered_capture(capture_identity)
        if carrier.state in (DraftWindowGraphCarrierState.BOUND, DraftWindowGraphCarrierState.SUBMITTED):
            raise RuntimeError("startup capture carrier is already bound to a runtime generation")
        return carrier.workspace

    @property
    def method(self) -> str:
        """Return the P2-normalized method owned by this provider."""
        return self.source.policy.normalized_method

    def register_capture_identity(
        self,
        *,
        route: DraftWindowGraphRoute,
        descriptor: BatchExecutionDescriptor,
        consumer_ordinal: int = 0,
    ) -> DraftWindowGraphCaptureIdentity:
        """Bind the manager's exact frozen descriptor to one persistent bank."""
        if not isinstance(route, DraftWindowGraphRoute) or _ROUTE_METHODS[route] != self.method:
            raise ValueError("capture route does not match the actual Draft method")
        if type(descriptor) is not BatchExecutionDescriptor:
            raise TypeError("capture requires the actual BatchExecutionDescriptor")
        if descriptor.cg_mode.name != "FULL":
            raise ValueError("Draft Window read capture requires a FULL attention descriptor")
        if descriptor.num_active_loras or descriptor.num_ubatches != 1:
            raise NotImplementedError("Draft Window capture does not support LoRA or microbatch variants")
        num_reqs = descriptor.num_reqs
        if type(num_reqs) is not int or num_reqs < 1 or type(consumer_ordinal) is not int or consumer_ordinal < 0:
            raise ValueError("capture request count and consumer ordinal must be valid integers")
        if type(descriptor.num_tokens) is not int or descriptor.num_tokens < 1 or descriptor.num_tokens % num_reqs:
            raise ValueError("capture token count must be positive and divisible by request count")
        query_width = descriptor.num_tokens // num_reqs
        if descriptor.uniform_token_count is not None and descriptor.uniform_token_count != query_width:
            raise ValueError("capture query width differs from the actual descriptor")
        if route is DraftWindowGraphRoute.EAGLE3_AR_STEP and query_width != 1:
            raise ValueError("Eagle3 AR capture requires query width 1")
        self._validate_source_contract()
        identity = DraftWindowGraphCaptureIdentity(route, descriptor, consumer_ordinal)
        if identity in self._carriers:
            raise RuntimeError("capture consumer identity is already registered")
        source = self.source.full_source
        if num_reqs > source._input_block_tables[self.source.group.gid].shape[0]:
            raise RuntimeError("capture request bucket exceeds source capacity")
        backing = self.new_workspace(num_reqs)
        backing._needs_clear = True
        self._carriers[identity] = DraftWindowGraphReadCarrier(
            capture_identity=identity,
            workspace=backing,
            backing_snapshot=_bank_snapshot(backing),
        )
        return identity

    def prepare_workspace(self, capture_identity: DraftWindowGraphCaptureIdentity) -> DraftWindowWorkspace:
        """Select the final capture bank before any canonical read is written.

        The existing Graph tail-notify/current-stream dependency orders reuse.
        UPDATED is only Host submission attribution, never device completion.
        """
        carrier = self._require_registered_capture(capture_identity)
        if carrier.state in (DraftWindowGraphCarrierState.BOUND, DraftWindowGraphCarrierState.SUBMITTED):
            raise RuntimeError("cannot overwrite an active Draft Window graph read carrier")
        self.require_reusable()
        if _bank_snapshot(carrier.workspace) != carrier.backing_snapshot:
            raise RuntimeError("Graph capture bank address or tensor geometry changed")
        return carrier.workspace

    def require_reusable(self) -> None:
        """Keep every proposal on its established completion stream."""
        if self.quarantined_identity is not None:
            raise RuntimeError("Draft Window bank is quarantined after uncertain submission")
        source = self.source.full_source._input_block_tables[self.source.group.gid]
        if source.device.type != "cpu":
            stream = int(torch.npu.current_stream().npu_stream)
            previous = getattr(self, "_owner_stream", stream)
            if stream != previous:
                raise RuntimeError("Draft Window bank reuse changed stream without a completion dependency")
            self._owner_stream = stream

    def bind_read_carrier(
        self,
        *,
        capture_identity: DraftWindowGraphCaptureIdentity,
        inputs: DraftMetadataInputs,
        metadata_builder: Callable[
            [DraftWindowWorkspace],
            Mapping[str, Any],
        ],
        target_metadata: Any,
    ) -> DraftWindowGraphReadCarrier:
        """Publish this generation against its selected persistent Graph bank."""
        if not isinstance(capture_identity, DraftWindowGraphCaptureIdentity):
            raise TypeError("graph read carrier requires a capture identity")
        carrier = self._require_registered_capture(capture_identity)
        if carrier.state in (
            DraftWindowGraphCarrierState.BOUND,
            DraftWindowGraphCarrierState.SUBMITTED,
        ):
            raise RuntimeError("cannot overwrite an active Draft Window graph read carrier")
        if not callable(metadata_builder):
            raise TypeError("graph metadata builder must be callable")
        generation = inputs.identity
        inputs.validate_for(generation)
        if inputs._owner is not self:
            raise RuntimeError("graph carrier inputs do not belong to the provider runtime")
        _validate_route(capture_identity.route, generation, self.method)
        self._validate_source_contract()
        if generation.padded_request_count != capture_identity.descriptor.num_reqs:
            raise RuntimeError("capture/runtime padded request count mismatch")
        self.source.full_source.validate_owner_bindings()
        if inputs.workspace is not carrier.workspace:
            raise RuntimeError("Graph generation was not materialized into its exact capture bank")
        metadata = metadata_builder(carrier.workspace)
        if not isinstance(metadata, Mapping) or not metadata:
            raise RuntimeError("explicit Draft Window graph carrier produced no Draft metadata")
        self._validate_metadata_uses_read_carrier(metadata, carrier.workspace)
        inputs.publish(metadata, target_metadata=target_metadata)

        bound = self._bound_carriers.setdefault(capture_identity.route, {})
        ordinal = capture_identity.consumer_ordinal
        if ordinal in bound:
            raise RuntimeError("graph read carrier consumer ordinal is already bound")
        if bound:
            previous = next(iter(bound.values())).inputs
            if previous is None or previous.identity.proposal is not generation.proposal:
                raise RuntimeError("graph read carriers do not belong to the same actual proposal")
        bound[ordinal] = carrier
        carrier.inputs = inputs
        carrier.state = DraftWindowGraphCarrierState.BOUND
        self._submitted_carriers = ()
        return carrier

    def consume_metadata(
        self,
        *,
        route: DraftWindowGraphRoute,
        expected_consumer_count: int,
    ) -> tuple[Mapping[str, Any], ...]:
        """Submit exactly one explicit carrier set; baseline metadata is forbidden."""
        if not isinstance(route, DraftWindowGraphRoute):
            raise TypeError("graph metadata consumption requires a DraftWindowGraphRoute")
        if not isinstance(expected_consumer_count, int) or expected_consumer_count <= 0:
            raise ValueError("expected graph consumer count must be positive")
        bound = self._bound_carriers.get(route, {})
        if len(bound) != expected_consumer_count or set(bound) != set(range(expected_consumer_count)):
            raise RuntimeError("graph Window provider requires one explicit read carrier for every consumer")
        carriers = tuple(bound[index] for index in range(expected_consumer_count))
        metadata = []
        for carrier in carriers:
            if carrier.state is not DraftWindowGraphCarrierState.BOUND or carrier.inputs is None:
                raise RuntimeError("graph read carrier has no bound runtime generation")
            if carrier.inputs.metadata is None:
                raise RuntimeError("graph read carrier has no Draft-private metadata")
            metadata.append(carrier.inputs.metadata)
            carrier.state = DraftWindowGraphCarrierState.SUBMITTED
        del self._bound_carriers[route]
        self._submitted_carriers = carriers
        return tuple(metadata)

    def validate_graph(self, graph: Any, *, route: DraftWindowGraphRoute) -> None:
        """Check the captured consumer contract once, before its first replay.

        UpdatableGraph.resolve_tasks binds its own captured tasks by provider
        and consumer ordinal. It preserves operation, handle and event by
        construction, so runtime does not authenticate those objects again.
        """
        if graph in self._validated_graphs:
            if self._validated_graphs[graph] is not route:
                raise RuntimeError("Draft Window graph route changed after capture")
            return
        import torch_npu

        from vllm_ascend.attention.attention_v1 import FIAParamProvider

        if not graph.tasks:
            raise RuntimeError("Draft Window Graph captured no attention consumer")
        for task in graph.tasks:
            if (
                not isinstance(task.provider, FIAParamProvider)
                or not task.provider.is_draft_model
                or task.provider.layer_name not in self.source.group.layers
                or task.operation != torch_npu.npu_fused_infer_attention_score.out
            ):
                raise RuntimeError("Draft Window Graph requires its supported Draft FIA consumer")
        self._validated_graphs[graph] = route

    def finish_update(self, *, route: DraftWindowGraphRoute) -> None:
        """Mark successful Host UPDATE; stream/event order still owns reuse."""
        for carrier in self._require_submitted_route(route):
            carrier.state = DraftWindowGraphCarrierState.UPDATED
        self._submitted_carriers = ()

    def _require_registered_capture(
        self,
        capture_identity: DraftWindowGraphCaptureIdentity,
    ) -> DraftWindowGraphReadCarrier:
        carrier = self._carriers.get(capture_identity)
        if carrier is None or carrier.capture_identity is not capture_identity:
            raise RuntimeError("capture identity/bucket is not registered by this Draft provider")
        if carrier.workspace.max_num_reqs != capture_identity.descriptor.num_reqs:
            raise RuntimeError("Graph capture bank request geometry changed")
        return carrier

    def _capture_topology(self) -> tuple[int, int]:
        parallel = self.source.parallel_config
        if parallel is None:
            raise RuntimeError("Graph read banks require their actual parallel configuration")
        world, rank = parallel.tensor_parallel_size, self.source.tensor_parallel_rank
        if type(world) is not int or world < 1 or type(rank) is not int or not 0 <= rank < world:
            raise RuntimeError("Graph TP world/rank must be explicit integers within range")
        return world, rank

    def _validate_source_contract(self) -> None:
        if self._capture_topology() != self._tp_identity:
            raise RuntimeError("Draft Graph parallel configuration or TP rank changed after capture")

    @staticmethod
    def _validate_metadata_uses_read_carrier(
        metadata: Mapping[str, Any],
        workspace: DraftWindowWorkspace,
    ) -> None:
        """Validate only Draft-private read carriers, never full write state."""
        identity = _tensor_identity
        expected_table_storage = identity(workspace.window_block_table).storage_data_ptr
        expected_device_storage = identity(workspace.device_window_exact).storage_data_ptr
        expected_cpu_storage = identity(workspace.cpu_window_exact).storage_data_ptr
        expected_lengths = workspace.cpu_window_exact.tolist()
        for layer_name, layer_metadata in metadata.items():
            if layer_metadata is None:
                continue
            block_tables = getattr(layer_metadata, "block_tables", None)
            seq_lens = getattr(layer_metadata, "seq_lens", None)
            seq_lens_cpu = getattr(layer_metadata, "seq_lens_cpu", None)
            seq_lens_list = getattr(layer_metadata, "seq_lens_list", None)
            if not isinstance(block_tables, torch.Tensor) or (
                identity(block_tables).storage_data_ptr != expected_table_storage
            ):
                raise RuntimeError(f"Draft graph metadata {layer_name!r} does not use carrier block tables")
            if not isinstance(seq_lens, torch.Tensor) or (
                identity(seq_lens).storage_data_ptr != expected_device_storage
            ):
                raise RuntimeError(f"Draft graph metadata {layer_name!r} does not use carrier device lengths")
            if not isinstance(seq_lens_cpu, torch.Tensor) or (
                identity(seq_lens_cpu).storage_data_ptr != expected_cpu_storage
            ):
                raise RuntimeError(f"Draft graph metadata {layer_name!r} does not use carrier CPU lengths")
            if not isinstance(seq_lens_list, list) or seq_lens_list[: len(expected_lengths)] != expected_lengths:
                raise RuntimeError(f"Draft graph metadata {layer_name!r} does not use carrier exact length values")

    def _require_submitted_route(
        self,
        route: DraftWindowGraphRoute,
    ) -> tuple[DraftWindowGraphReadCarrier, ...]:
        if not self._submitted_carriers:
            raise RuntimeError("no explicit Draft Window graph read carrier batch was submitted")
        if any(carrier.capture_identity.route is not route for carrier in self._submitted_carriers):
            raise RuntimeError("submitted graph read carrier route mismatch")
        return self._submitted_carriers


def resolve_parallel_draft_window_graph_route(
    owner: Any,
) -> DraftWindowGraphRoute | None:
    """Resolve DFlash/DSpark from the provider's actual method policy.

    The class-name/debug label on a speculator is not a policy or capability
    source.  When no provider is installed this remains an exact OFF-path
    no-op.  A provider for any non-parallel method is a hard route mismatch.
    """
    provider = getattr(owner, "_draft_window_graph_binding_provider", None)
    if provider is None:
        return None
    if not isinstance(provider, DraftWindowGraphBindingProvider):
        raise TypeError("invalid instance-owned Draft Window graph binding provider")
    method = provider.method
    if method == "dflash":
        return DraftWindowGraphRoute.DFLASH
    if method == "dspark":
        return DraftWindowGraphRoute.DSPARK
    raise RuntimeError(f"parallel Draft Window graph manager cannot consume method {method!r}")


def validate_draft_window_graph_capture(
    owner: Any,
    *,
    capture_context: Any = None,
) -> None:
    """Require the startup context belonging to this instance and provider."""
    provider = getattr(owner, "_draft_window_graph_binding_provider", None)
    if provider is None:
        return
    if not isinstance(provider, DraftWindowGraphBindingProvider):
        raise TypeError("invalid instance-owned Draft Window graph binding provider")
    if capture_context is None:
        raise RuntimeError("Draft Window Graph capture has no owner-bound persistent read context")
    validate = getattr(capture_context, "validate_for_provider", None)
    if not callable(validate):
        raise TypeError("invalid Draft Window startup capture context")
    validate(provider)


def consume_draft_window_graph_metadata(
    owner: Any,
    *,
    route: DraftWindowGraphRoute,
    expected_consumer_count: int,
) -> tuple[Mapping[str, Any], ...] | None:
    """Return explicit carrier metadata, or preserve exact OFF behavior."""
    provider = getattr(owner, "_draft_window_graph_binding_provider", None)
    if provider is None:
        return None
    if not isinstance(provider, DraftWindowGraphBindingProvider):
        raise TypeError("invalid instance-owned Draft Window graph binding provider")
    return provider.consume_metadata(
        route=route,
        expected_consumer_count=expected_consumer_count,
    )
