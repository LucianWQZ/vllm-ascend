# SPDX-License-Identifier: Apache-2.0
"""Default-off Eagle3 captured device copies for independent numerical replay.

These extra D2D/index-select operators change the captured diagnostic program.
They are not a normal-overlap, native operand or physical-read observation.
No Host/device copy, synchronization, event query or I/O occurs in callbacks.
The external disposable producer updates indices before replay, drains after
UPDATE, and runs the independent eager reference. Normal Graph has no copies.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MethodType
from typing import Any

import torch


@dataclass(slots=True)
class Eagle3CapturedState:
    indices: torch.Tensor
    before: dict[str, Any] | None = None
    after: dict[str, torch.Tensor] | None = None
    before_kv_owned: dict[str, torch.Tensor] | None = None
    outputs_owned: tuple[torch.Tensor, torch.Tensor] | None = None
    legacy_capture_pool_outputs: tuple[torch.Tensor, torch.Tensor] | None = None
    # Captured clone/index_select nodes retain raw source addresses, not Python
    # ownership. Keep exact sources alive for this captured program's lifetime.
    capture_sources: tuple[torch.Tensor, ...] = ()
    attention_query_owned: torch.Tensor | None = None
    attention_output_owned: torch.Tensor | None = None
    attention_capture_parameters: dict | None = None
    attention_mask_source: torch.Tensor | None = None


def _attention_tensor_header(tensor):
    if tensor is None:
        return None
    npu_format = None
    if tensor.device.type in ("npu", "privateuseone"):
        # Tensor descriptor metadata only; no synchronize, event query,
        # transfer or operator launch. CPU fixtures leave this field None.
        import torch_npu

        npu_format = int(torch_npu.get_npu_format(tensor))
    return dict(
        shape=list(tensor.shape),
        stride=list(tensor.stride()),
        dtype=str(tensor.dtype),
        device=str(tensor.device),
        address=tensor.data_ptr(),
        storage_offset=tensor.storage_offset(),
        npu_format=npu_format,
    )


def eagle_fia_capture_parameter_header(row):
    """The fixed Ascend capture tuple; startup lengths are not replay values."""
    if len(row) != 22 or any(value is not None for value in row[17:21]):
        raise RuntimeError("Eagle attention discriminator requires the fixed unquantized FIA tuple")
    return dict(
        scope="CAPTURE_CALL_BINDING_NOT_EXECUTION_OCCURRENCE;STARTUP_LENGTHS_NOT_CURRENT_REPLAY",
        query=_attention_tensor_header(row[0]),
        key=_attention_tensor_header(row[1]),
        value=_attention_tensor_header(row[2]),
        table=_attention_tensor_header(row[3]),
        mask=_attention_tensor_header(row[4]),
        block_size=row[5],
        startup_kv_lengths=list(row[6]),
        startup_query_lengths=list(row[7]),
        kv_heads=row[8],
        query_heads=row[9],
        scale=row[10],
        output=_attention_tensor_header(row[11]),
        lse=_attention_tensor_header(row[12]),
        sparse_mode=row[13],
        pre_tokens=row[14],
        next_tokens=row[15],
        sliding_window=row[16],
        layer=row[21],
        input_layout="TND",
    )


class Eagle3AttentionDiscriminator:
    """Optional per-consumer D2D query/output copies, never normal execution.

    Q is produced inside the captured model. Copy it immediately before FIA
    into destinations allocated outside capture, keyed by the complete
    descriptor/consumer identity. No capture-pool clone is a history owner.
    Headers describe the fixed capture call; current table/exact values come
    from the separate completed-generation snapshots, not startup constants.
    """

    def __init__(self, snapshots, impl):
        if getattr(impl, "sinks", None) is not None or getattr(impl, "enable_c8_quant", False):
            raise RuntimeError("Eagle discriminator requires actual no-sink BF16 FIA")
        self.snapshots, self.impl = snapshots, impl
        self.capture_identity = None
        self.reference_active = False
        self.reference = None
        self.query_shape = (impl.num_heads, impl.head_size)
        for identity, state in snapshots.states.items():
            shape = (identity.fingerprint.padded_token_count, *self.query_shape)
            state.attention_query_owned = torch.empty(
                shape, dtype=snapshots.owner.hidden_states.dtype, device=snapshots.owner.device
            )
            state.attention_output_owned = torch.empty_like(state.attention_query_owned)
        original_forward = impl.forward_fused_infer_attention
        original_params = impl._get_fia_params
        original_full = impl.full_graph_fia
        probe = self

        def parameters(_impl, key, value, metadata, kv_cache=None):
            result = original_params(key, value, metadata, kv_cache)
            if probe.reference_active:
                k, v, block, table, lengths = result
                if probe.reference is None:
                    raise RuntimeError("reference FIA parameters have no current query")
                probe.reference["parameter_calls"].append(
                    dict(
                        key=_attention_tensor_header(k),
                        value=_attention_tensor_header(v),
                        block_size=block,
                        table=_attention_tensor_header(table),
                        kv_lengths=list(lengths),
                        query_lengths=list(metadata.actual_seq_lengths_q),
                        causal=metadata.causal,
                        attn_state=getattr(metadata.attn_state, "name", str(metadata.attn_state)),
                        query_heads=_impl.num_heads,
                        kv_heads=_impl.num_kv_heads,
                        head_size=_impl.head_size,
                        scale=_impl.scale,
                        sliding_window=_impl.sliding_window,
                        input_layout="TND",
                        scope="ACTUAL_EAGER_BACKEND_ARGUMENTS_BEFORE_PINNED_DEVICEOPERATOR;NOT_NATIVE_EXECUTION_OPERANDS",
                    )
                )
                probe.reference["mask_source"] = metadata.attn_mask
            return result

        def full(_impl, query, key, value, metadata, output, layer=None):
            if probe.capture_identity is None:
                return original_full(query, key, value, metadata, output, layer)
            from vllm_ascend.attention import attention_v1 as attention

            params = (
                attention.get_draft_graph_prefill_params()
                if attention._EXTRA_CTX.is_draft_model_prefill
                else attention.get_draft_graph_params()
            )
            tokens = probe.capture_identity.fingerprint.padded_token_count
            previous = len(params.attn_params[tokens])
            result = original_full(query, key, value, metadata, output, layer)
            rows = params.attn_params[tokens]
            if len(rows) != previous + 1:
                raise RuntimeError("ambiguous actual Eagle FIA capture tuple")
            state = snapshots.states[probe.capture_identity]
            state.attention_capture_parameters = eagle_fia_capture_parameter_header(rows[-1])
            # The persistent capture carrier already owns this mask. Retain
            # the same source for a post-drain value export, with no D2H here.
            state.attention_mask_source = metadata.attn_mask
            return result

        def forward(_impl, query, key, value, metadata, output, kv_cache=None):
            state = None if probe.capture_identity is None else snapshots.states[probe.capture_identity]
            if state is not None:
                destination = state.attention_query_owned
                if query.numel() != destination.numel() or query.dtype != destination.dtype:
                    raise RuntimeError("actual Eagle FIA query differs from declared local geometry")
                destination.copy_(query.reshape(destination.shape))
            if probe.reference_active:
                probe.reference = dict(
                    query=query.clone(),
                    output=None,
                    parameter_calls=[],
                    mask_source=None,
                    query_header=_attention_tensor_header(query),
                )
            result = original_forward(query, key, value, metadata, output, kv_cache)
            if state is not None:
                destination = state.attention_output_owned
                if result.numel() != destination.numel() or result.dtype != destination.dtype:
                    raise RuntimeError("actual Eagle FIA output differs from declared local geometry")
                destination.copy_(result.reshape(destination.shape))
            if probe.reference_active:
                probe.reference["output"] = result.clone()
            return result

        impl._get_fia_params = MethodType(parameters, impl)
        impl.full_graph_fia = MethodType(full, impl)
        impl.forward_fused_infer_attention = MethodType(forward, impl)

    def begin_reference(self):
        if self.capture_identity is not None or self.reference_active:
            raise RuntimeError("overlapping Eagle attention diagnostic scopes")
        self.reference, self.reference_active = None, True

    def end_reference(self):
        self.reference_active = False

    def completed_values(self, identity):
        """Caller must have completed the existing diagnostic Graph/ref drains."""
        state = self.snapshots.states[identity]
        if state.attention_capture_parameters is None or self.reference is None or self.reference["output"] is None:
            raise RuntimeError("incomplete Eagle attention discriminator")
        if len(self.reference["parameter_calls"]) != 1:
            raise RuntimeError("Eagle reference did not make one actual FIA parameter call")

        def cpu(t):
            return None if t is None else t.detach().cpu().clone()

        return dict(
            graph_query=cpu(state.attention_query_owned),
            graph_output=cpu(state.attention_output_owned),
            graph_capture_parameters=state.attention_capture_parameters,
            graph_mask_post_drain=cpu(state.attention_mask_source),
            reference_query=cpu(self.reference["query"]),
            reference_output=cpu(self.reference["output"]),
            reference_parameters=self.reference["parameter_calls"],
            reference_query_header=self.reference["query_header"],
            reference_mask_post_drain=cpu(self.reference["mask_source"]),
            scope="INTRUSIVE_CAPTURED_D2D_AND_POST_DRAIN_VALUES;NOT_NORMAL_TIMING_OR_PHYSICAL_READ",
        )


class Eagle3CaptureSnapshots:
    def __init__(self, owner: Any, captures: Mapping, *, max_device_bytes: int = 2 << 30) -> None:
        self.owner = owner
        if getattr(owner.attn_backend, "forward_includes_kv_cache_update", None) is not True:
            raise RuntimeError("Eagle3 numeric requires metadata-owned Ascend KV writes")
        context = owner.draft_vllm_config.compilation_config.static_forward_context
        layers = sorted(owner.draft_attn_layer_names)
        if not layers or any(name not in context for name in layers):
            raise RuntimeError("Eagle3 numeric capture has no complete Draft layer inventory")
        self.caches = {f"{name}/{i}": tensor for name in layers for i, tensor in enumerate(context[name].kv_cache)}
        if not self.caches or any(not isinstance(t, torch.Tensor) or t.ndim < 2 for t in self.caches.values()):
            raise RuntimeError("Eagle3 numeric capture requires initialized real Draft KV tensors")
        self.layers = tuple(layers)
        self.states = {}
        from vllm_ascend.ascend_config import validate_additional_config_bool

        config = getattr(getattr(owner, "vllm_config", None), "additional_config", {})
        attention_enabled = validate_additional_config_bool(
            config.get("draft_window_graph_eagle_attention_probe", False),
            "additional_config.draft_window_graph_eagle_attention_probe",
        )
        impl = None
        if attention_enabled:
            if len(layers) != 1:
                raise RuntimeError("Eagle attention discriminator requires its declared single Draft layer")
            impl = context[layers[0]].impl
        groups = {identity.fingerprint.global_group_id for identity in captures.values()}
        if len(groups) != 1:
            raise RuntimeError("Eagle3 numeric profile requires one actual global cache group")
        self.global_group_id = next(iter(groups))
        width = owner.block_tables.input_block_tables[self.global_group_id].shape[1]
        per_block = sum(t[0].numel() * t.element_size() for t in self.caches.values())
        needed = sum(
            (identity.fingerprint.padded_request_count * width + 1) * per_block * 3 for identity in captures.values()
        )
        output_bytes = sum(
            identity.fingerprint.padded_token_count
            * owner.hidden_states.shape[-1]
            * owner.hidden_states.element_size()
            * 2
            for identity in captures.values()
        )
        needed += output_bytes
        if attention_enabled:
            needed += sum(
                identity.fingerprint.padded_token_count
                * impl.num_heads
                * impl.head_size
                * owner.hidden_states.element_size()
                * 2
                for identity in captures.values()
            )
        if needed > max_device_bytes:
            raise ValueError("BLOCKED_RESOURCE: Eagle3 capture copies exceed declared 2 GiB device budget")
        self.hbm_free_before_capture = None
        if owner.device.type in ("npu", "privateuseone"):
            self.hbm_free_before_capture = torch.npu.mem_get_info(owner.device)[0]
            if self.hbm_free_before_capture < needed + (2 << 30):
                raise ValueError("BLOCKED_RESOURCE: insufficient HBM for diagnostic copies and 2 GiB margin")
        self.device_bytes_bound = needed
        for identity in captures.values():
            n = identity.fingerprint.padded_request_count
            count = n * width + 1
            self.states[identity] = Eagle3CapturedState(
                indices=torch.zeros(count, dtype=torch.int64, device=owner.device),
                # The model returns capture-pool temporaries. Retaining the
                # Python tuple does not prevent later nodes reusing storage.
                outputs_owned=tuple(
                    torch.empty(
                        (identity.fingerprint.padded_token_count, owner.hidden_states.shape[-1]),
                        dtype=owner.hidden_states.dtype,
                        device=owner.device,
                    )
                    for _ in range(2)
                ),
                # Prestate is consumed after the entire merged AR Graph, too.
                # It needs the same outside-capture ownership as poststate.
                before_kv_owned={
                    key: torch.empty((count, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
                    for key, tensor in self.caches.items()
                },
                # These destinations must survive every operator in the
                # merged AR Graph. Allocate them before model capture so
                # diagnostic poststates do not depend on capture-pool output
                # reuse. They remain instance-owned for the capture lifetime.
                after={
                    key: torch.empty((count, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
                    for key, tensor in self.caches.items()
                },
            )
        self.attention_probe = Eagle3AttentionDiscriminator(self, impl) if attention_enabled else None

    def before_model(self, identity: Any, metadata: Mapping, slots: Mapping) -> None:
        if identity is None:
            return
        owner, state = self.owner, self.states[identity]
        if self.attention_probe is not None:
            self.attention_probe.capture_identity = identity
        if state.before_kv_owned is None or set(state.before_kv_owned) != set(self.caches):
            raise RuntimeError("Eagle3 prestate has no preallocated complete KV destinations")
        fp = identity.fingerprint
        n, tokens = fp.padded_request_count, fp.padded_token_count
        if not isinstance(slots, Mapping) or not set(self.layers) <= set(slots):
            raise RuntimeError("Eagle3 captured numeric input is missing actual full slot operands")
        if not set(self.layers) <= set(metadata):
            raise RuntimeError("Eagle3 captured numeric input is missing actual layer metadata")
        # In particular builder.query_start_loc may be an otherwise unused,
        # freshly allocated GPU auxiliary tensor. Keeping only clone outputs
        # lets its allocation be reused before the diagnostic Graph replays.
        # These are references only: no new copy/sync/query and no normal-path
        # installation. Owners are bounded by the frozen capture inventory.
        sources = [
            owner.input_buffers.input_ids,
            owner.input_buffers.positions,
            owner.hidden_states,
            owner.input_buffers.seq_lens,
            owner.input_buffers.query_start_loc,
            owner.target_input_buffers.positions,
            owner.last_token_indices,
            owner.block_tables.input_block_tables[fp.global_group_id],
            state.indices,
            *self.caches.values(),
        ]
        for name in self.layers:
            layer = metadata[name]
            sources.extend((layer.slot_mapping, layer.block_tables, layer.seq_lens, layer.query_start_loc, slots[name]))
        if any(not isinstance(tensor, torch.Tensor) for tensor in sources):
            raise RuntimeError("captured numeric copy source must be an owned tensor")
        state.capture_sources = tuple(sources)
        for key, tensor in self.caches.items():
            torch.index_select(tensor, 0, state.indices, out=state.before_kv_owned[key])
        state.before = dict(
            inputs={
                "input_ids": owner.input_buffers.input_ids[:tokens].clone(),
                "positions": owner.input_buffers.positions[:tokens].clone(),
                "hidden_states": owner.hidden_states[:tokens].clone(),
                "full_exact": owner.input_buffers.seq_lens[:n].clone(),
                "query_start_loc": owner.input_buffers.query_start_loc[: n + 1].clone(),
                # First-pass keeps the Target Full write grid, while upstream
                # only refreshes the accepted Draft input prefix. Rejected
                # tail positions may retain old values and are not a new
                # source for recomputing the Full write slots.
                "target_positions": owner.target_input_buffers.positions[:tokens].clone(),
                "last_token_indices": owner.last_token_indices[:n].clone(),
            },
            canonical_table=owner.block_tables.input_block_tables[fp.global_group_id][:n].clone(),
            # This backend writes KV through metadata.slot_mapping. The
            # ForwardContext argument is retained separately, not mislabeled.
            actual_slots={name: metadata[name].slot_mapping[:tokens].clone() for name in self.layers},
            forward_context_slots={name: slots[name][:tokens].clone() for name in self.layers},
            bank_tables={name: metadata[name].block_tables.clone() for name in self.layers},
            bank_device_exact={name: metadata[name].seq_lens.clone() for name in self.layers},
            metadata_query={name: metadata[name].query_start_loc.clone() for name in self.layers},
            kv=state.before_kv_owned,
            # Preserve the former capture-pool allocation as a diagnostic
            # counterfactual. It is never the eager restore/oracle source.
            legacy_capture_pool_kv={key: tensor.index_select(0, state.indices) for key, tensor in self.caches.items()},
        )

    def after_model(self, identity: Any) -> None:
        if identity is not None:
            state = self.states[identity]
            if state.after is None or set(state.after) != set(self.caches):
                raise RuntimeError("Eagle3 poststate has no preallocated complete KV destinations")
            for key, tensor in self.caches.items():
                torch.index_select(tensor, 0, state.indices, out=state.after[key])
            if self.attention_probe is not None:
                self.attention_probe.capture_identity = None

    def retain_output(self, identity: Any, output: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Capture immediate D2D copies into preallocated diagnostic owners."""
        state = self.states[identity]
        owned = state.outputs_owned
        if (
            owned is None
            or not isinstance(output, tuple)
            or len(output) != 2
            or any(
                not isinstance(source, torch.Tensor)
                or source.shape != target.shape
                or source.dtype != target.dtype
                or source.device != target.device
                for source, target in zip(output, owned)
            )
        ):
            raise RuntimeError("Eagle3 numeric output differs from the declared two-hidden-tensor contract")
        for source, target in zip(output, owned):
            target.copy_(source)
        # Keep the old view for a same-Graph counterfactual; never use it as
        # the oracle or as the independent eager reference's input.
        state.legacy_capture_pool_outputs = output
        return owned

    def set_indices_before_replay(self, identities: tuple, selected: list[int]) -> None:
        """External diagnostic only, never invoked from a capture callback."""
        if not selected or selected != sorted(set(selected)) or selected[0] < 0:
            raise ValueError("Eagle3 numeric selection must be a sorted unique physical block set")
        if any(selected[-1] >= tensor.shape[0] for tensor in self.caches.values()):
            raise ValueError("Eagle3 numeric selection exceeds actual Draft KV capacity")
        for identity in identities:
            indices = self.states[identity].indices
            if len(selected) > indices.numel():
                raise ValueError("Eagle3 numeric selection exceeds declared capture capacity")
            values = torch.zeros(indices.numel(), dtype=torch.int64)
            values[: len(selected)] = torch.tensor(selected, dtype=torch.int64)
            indices.copy_(values)
