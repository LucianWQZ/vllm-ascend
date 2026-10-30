# SPDX-License-Identifier: Apache-2.0
"""Opt-in retained capture outputs; not a transport or completion mechanism.

Default mode retains references only. The separately selected owned mode
records explicit D2D copies, changing diagnostic timing and allocation. Both
are diagnostic and never normal-performance or reader-completion evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MethodType

import torch


def _tensor_header(value):
    if value is None:
        return None
    return dict(
        shape=list(value.shape),
        stride=list(value.stride()),
        dtype=str(value.dtype),
        storage_offset=value.storage_offset(),
        data_ptr=value.data_ptr(),
    )


def owned_attention_probe_layer(owner):
    """Select one actual reader for an explicitly owned diagnostic capture.

    SWA keeps its original sole Full reader. A non-SWA parallel drafter uses
    its first named noncausal reader, allowing a first-divergence comparison.
    This does not enable, replace or change an attention implementation.
    """
    plan = owner._draft_window_source_capability.native_swa_graph
    if plan is not None:
        if len(plan.full_layers) != 1:
            raise RuntimeError("owned numeric attention probe requires one admitted Full reader")
        return plan.full_layers[0]
    names = tuple(sorted(owner.draft_attn_layer_names))
    causal = owner._group_causal
    flags = tuple(causal.values()) if isinstance(causal, Mapping) else (causal,)
    if not names or not flags or any(value is not False for value in flags):
        raise RuntimeError("owned numeric attention probe requires actual noncausal readers")
    return names[0]


def _metadata_header(value):
    state = getattr(value, "attn_state", None)
    return dict(
        attn_state=getattr(state, "name", str(state)),
        causal=value.causal,
        seq_lens_list=list(value.seq_lens_list),
        actual_seq_lengths_q=list(value.actual_seq_lengths_q),
        num_actual_tokens=value.num_actual_tokens,
        num_prefills=value.num_prefills,
        num_decodes=value.num_decodes,
        max_query_len=value.max_query_len,
        table=_tensor_header(value.block_tables),
        mask=_tensor_header(value.attn_mask),
    )


class DraftGraphOutputReferences:
    def __init__(self, owner, *, owned=False):
        self.owner = owner
        self.owned = owned
        self.raw_outputs = {}
        self.attention = {}
        self.reference_active = False
        self.reference_attention = None
        self._capture_params = None
        self._reference_params = None
        self.outputs: dict[int, torch.Tensor] = {}
        self.frozen = False
        self.collecting = False

    def retain(self, owner, num_tokens, output):
        if owner is not self.owner or self.frozen:
            raise RuntimeError("numeric capture owner/scope mismatch")
        if not isinstance(output, torch.Tensor) or output.ndim != 2 or output.shape[0] != num_tokens:
            raise RuntimeError("unknown Draft model continuous output contract")
        # Pinned CudaGraphManager invokes warmup then capture with mode NONE.
        # During the owner's capture scope, the last invocation per bucket is
        # the captured allocation. Freeze occurs only after capture succeeds.
        self.raw_outputs[num_tokens] = output
        self.outputs[num_tokens] = output.clone() if self.owned else output

    def install_attention_probe(self, impl):
        """One named Draft layer, outside task-update calls; default mode untouched."""
        if not self.owned or hasattr(self, "_attention_impl"):
            raise RuntimeError("owned single-layer attention probe scope required")
        self._attention_impl = impl
        decoder, parameters = impl.forward_fused_infer_attention, impl._get_fia_params
        sink = self

        def params(_impl, key, value, metadata, kv_cache=None):
            result = parameters(key, value, metadata, kv_cache)
            if sink.collecting or sink.reference_active:
                k, v, block, table, lengths = result
                row = dict(
                    key=_tensor_header(k),
                    value=_tensor_header(v),
                    block_size=block,
                    table=_tensor_header(table),
                    kv_lengths=list(lengths) if isinstance(lengths, (list, tuple)) else _tensor_header(lengths),
                    metadata=_metadata_header(metadata),
                )
                if sink.reference_active:
                    sink._reference_params = row
                else:
                    sink._capture_params = row
            return result

        def forward(_impl, query, key, value, metadata, output, kv_cache=None):
            if sink.reference_active:
                sink.reference_attention = None
                sink._reference_params = None
            # Preserve the actual pre-call query even if a decoder uses an
            # aliasing output. These copies exist only in the owned diagnostic.
            saved_query = query.clone() if sink.collecting or sink.reference_active else None
            result = decoder(query, key, value, metadata, output, kv_cache)
            if sink.collecting or sink.reference_active:
                row = dict(
                    query=saved_query,
                    output=result.clone(),
                    metadata=_metadata_header(metadata),
                    source_query=_tensor_header(query),
                    source_output=_tensor_header(result),
                    parameters=sink._reference_params if sink.reference_active else sink._capture_params,
                    parameter_observation=(
                        "EAGER_CURRENT_FIA_PARAMETER_HEADERS"
                        if sink.reference_active
                        else "CAPTURE_TIME_HEADERS_NOT_RUNTIME_TASK_OPERANDS"
                    ),
                )
                if sink.reference_active:
                    sink.reference_attention = row
                else:
                    sink.attention[int(query.shape[0])] = row
            return result

        impl._get_fia_params = MethodType(params, impl)
        impl.forward_fused_infer_attention = MethodType(forward, impl)

    def raw(self, owner, num_tokens):
        self.get(owner, num_tokens)
        return self.raw_outputs[num_tokens]

    def freeze(self, captured_tokens):
        if set(self.outputs) != set(captured_tokens):
            raise RuntimeError("numeric output inventory differs from actual captured graphs")
        self.frozen = True

    def get(self, owner, num_tokens):
        if owner is not self.owner or not self.frozen or num_tokens not in self.outputs:
            raise RuntimeError("missing or foreign captured output")
        return self.outputs[num_tokens]
