# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Eager proposal integration for resident direct-prefix Draft KV reads.

An enabled runtime is installed only after the actual Draft consumers have
been checked. Target graph selection and Full write/query operands are owned
by the existing runner. Window projections and built Draft metadata are owned
by a proposal until its final consumer event completes.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch

from vllm_ascend.worker.v2.spec_decode.draft_window_lifetime import (
    DraftWindowOwner,
    DraftWindowOwnerPool,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_plan import (
    DraftWindowPlan,
    compute_draft_window_plan,
    freeze_eagle_window_plan,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_policy import DraftWindowPolicyDecision
from vllm_ascend.worker.v2.spec_decode.draft_window_read import DirectPrefixSource, materialize_direct_prefix


class _NPUCompletion:
    def __init__(self) -> None:
        # Construction and recording both execute under the owner pool's
        # failure guard. No fallback token can substitute for either failure.
        self.event = torch.npu.Event()
        self.event.record(torch.npu.current_stream())

    def is_complete(self) -> bool:
        return bool(self.event.query())

    def wait(self) -> None:
        self.event.synchronize()


def continuing_prefill_rows(input_batch: Any, num_reqs: int) -> np.ndarray:
    """Prompt progress excludes speculative placeholders in scheduled tokens.

    This uses the same new-token count as MRV2 gather_batch_req_state. Decode
    rows already have computed_prefill >= prefill_len. A terminal chunk may
    therefore use a Window even while the input row is marked is_prefilling.
    """
    computed, prompt, scheduled = (
        np.asarray(values)[:num_reqs]
        for values in (
            input_batch.num_computed_prefill_tokens_np,
            input_batch.prefill_len_np,
            input_batch.num_scheduled_tokens,
        )
    )
    if any(array.shape != (num_reqs,) for array in (computed, prompt, scheduled)):
        raise ValueError("Draft Window phase state does not cover the active requests")
    if any(not np.issubdtype(array.dtype, np.integer) for array in (computed, prompt, scheduled)):
        raise TypeError("Draft Window phase counters must contain integers")
    if any(bool(np.any(array < 0)) for array in (computed, prompt, scheduled)):
        raise ValueError("Draft Window phase counters must be nonnegative")
    drafts = input_batch.num_draft_tokens_per_req
    if drafts is not None:
        drafts = np.asarray(drafts)[:num_reqs]
        if drafts.shape != (num_reqs,) or not np.issubdtype(drafts.dtype, np.integer):
            raise ValueError("Draft placeholders must contain one integer per active request")
        if bool(np.any(drafts < 0)):
            raise ValueError("Draft placeholders must be nonnegative")
        scheduled = scheduled - drafts
    if np.any(scheduled < 0):
        raise ValueError("Speculative placeholders exceed scheduled tokens")
    return computed + scheduled < prompt


class DraftWindowRuntime:
    """One speculator's bounded ownership and fresh metadata projection seam."""

    def __init__(self, speculator: Any, policy: DraftWindowPolicyDecision) -> None:
        if not policy.enabled or policy.effective_window_size is None:
            raise ValueError("An OFF Draft Window policy must not allocate a runtime")
        self.method = policy.normalized_method
        self.static_k = speculator.num_speculative_steps if self.method == "eagle3" else None
        if self.static_k is not None and policy.effective_window_size < 2 * self.static_k + 1:
            raise ValueError("Eagle3 draft_window_size must cover first-pass query width plus the proposal horizon")
        if self.method == "eagle3" and speculator.supports_decode_metadata_update:
            raise NotImplementedError("Eagle3 Draft Window requires per-step metadata preparation")
        # The model config inherits the Target flag. All three Ascend
        # speculators also force their graph managers to NONE for the explicit
        # speculative override, allowing Target Graph with Draft eager.
        if not (speculator.draft_model_config.enforce_eager or speculator.speculative_config.enforce_eager):
            raise ValueError("Draft Window requires an eager Draft model; Target graph mode is independent")
        self.speculator = speculator
        self.window_size = policy.effective_window_size
        self.mla_layer_names: tuple[str, ...] = ()
        self.native_swa_intersections: dict[int, int] = {}
        self.native_swa_causal: dict[int, bool] = {}
        self._dense_reads = False
        self.gids = self._validate_route(speculator)
        read_groups = self.gids or tuple(self.native_swa_intersections)
        if not read_groups:
            raise NotImplementedError("Draft Window requires at least one certified read group")
        self.anchor_block_span = int(speculator.block_tables.kernel_block_sizes[read_groups[0]])
        self._native_swa_built: set[int] = set()
        self.pool = DraftWindowOwnerPool(int(speculator.vllm_config.max_concurrent_batches))
        self._owner: DraftWindowOwner | None = None
        self._input_batch: Any = None
        self._bypass = False
        self._build_count = 0
        self._eagle_plans: tuple[DraftWindowPlan, ...] | None = None
        self._next_step = 0
        self._sources: dict[int, DirectPrefixSource] = {}

    @property
    def reads_active(self) -> bool:
        return self._owner is not None and not self._bypass

    def _validate_route(self, speculator: Any) -> tuple[int, ...]:
        # Imported lazily: provider inventory depends on the installed NPU
        # backends, while phase/owner tests need no device import boundary.
        from vllm_ascend.ascend_config import get_ascend_config
        from vllm_ascend.worker.v2.spec_decode.draft_window_backends import (
            DraftWindowProviderStatus,
            require_direct_prefix_routes,
        )

        config = speculator.attn_vllm_config
        if config.parallel_config.pipeline_parallel_size != 1:
            raise NotImplementedError("Draft Window pipeline parallelism requires a distributed reader provider")
        transfer = config.kv_transfer_config
        if (
            (transfer is not None and transfer.kv_connector is not None)
            or config.cache_config.kv_offloading_size
            or get_ascend_config().sparse_kv_offload_config.enabled
        ):
            raise NotImplementedError("Draft Window offloaded/transferred KV requires a resident-set provider")
        # For this FullAttentionSpec route the scheduler retains every active
        # request block through execution. No circular/native eviction,
        # offload, transfer or CP projection is admitted by this provider.
        # Standard GQA builders launch attention and read-derived preparation
        # on the current stream. Ordinary TP collectives consume hidden states,
        # not Window metadata. Ordinary Flashcomm/SP keeps the baseline query
        # padding and Full write coordinates; actual CP builders are excluded
        # by the provider inventory rather than a target-global CP switch.
        routes = require_direct_prefix_routes(speculator, window_size=self.window_size)
        self.native_swa_intersections = {}
        for route in routes:
            if route.status is DraftWindowProviderStatus.NATIVE_SWA:
                if route.native_window is None:
                    raise ValueError("A native SWA provider must identify its native window")
                self.native_swa_intersections[route.gid] = route.native_window
        if self.native_swa_intersections:
            if self.window_size < speculator.num_query_per_req:
                raise ValueError("DFlash draft_window_size must cover its query width")
            declared = getattr(speculator, "_group_causal", True)
            self.native_swa_causal = {
                gid: declared[gid] if isinstance(declared, dict) else declared for gid in self.native_swa_intersections
            }
            routes = tuple(route for route in routes if route.gid not in self.native_swa_intersections)
        if (
            routes
            and self.method in ("eagle3", "dflash", "dspark")
            and all(route.status is DraftWindowProviderStatus.DIRECT_DENSE for route in routes)
        ):
            from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl

            specs = [speculator.kv_cache_config.kv_cache_groups[route.gid].kv_cache_spec for route in routes]
            specs = [item for spec in specs for item in getattr(spec, "kv_cache_specs", {"": spec}).values()]
            context = config.compilation_config.static_forward_context
            implementations = [getattr(context.get(name), "impl", None) for route in routes for name in route.layers]
            self._dense_reads = all(spec.dtype in (torch.float16, torch.bfloat16) for spec in specs) and all(
                type(impl) is AscendAttentionBackendImpl
                and not impl.use_bnsd_kv_cache
                and impl.sinks is None
                and impl.sliding_window is None
                for impl in implementations
            )
        self.mla_layer_names = tuple(
            name for route in routes if route.status is DraftWindowProviderStatus.DIRECT_MLA for name in route.layers
        )
        return tuple(dict.fromkeys(route.gid for route in routes))

    @contextmanager
    def proposal_context(
        self,
        speculator: Any,
        input_batch: Any,
        dummy_run: bool = False,
        is_profile: bool = False,
        *,
        context_kv_only: bool = False,
        num_speculative_tokens: int | None = None,
    ) -> Iterator[None]:
        if speculator is not self.speculator:
            raise RuntimeError("Draft Window runtime belongs to another speculator")
        if self._owner is not None or self._bypass:
            raise RuntimeError("Draft Window proposal scope is already active")
        if context_kv_only and self.method not in ("dflash", "dspark"):
            raise ValueError("Context-only Draft KV updates require a parallel drafter")
        expected_eagle_steps = self.static_k
        if num_speculative_tokens is not None:
            if (
                self.method != "eagle3"
                or type(num_speculative_tokens) is not int
                or self.static_k is None
                or not 0 <= num_speculative_tokens <= self.static_k
            ):
                raise ValueError("Eagle3 scheduled draft count must stay within its configured static K")
            # A zero-draft AR proposal still runs prefill attention to update
            # KV, unlike the parallel drafter's context-only write branch.
            # Keep the configured K for window geometry; only completion's
            # expected number of consumed steps follows this real invocation.
            expected_eagle_steps = max(1, num_speculative_tokens)
        if dummy_run or is_profile or getattr(input_batch, "is_dummy", False):
            self._bypass = True
            try:
                yield
            finally:
                self._bypass = False
            return
        with self.pool.operation(_NPUCompletion) as owner:
            self._owner = owner
            self._input_batch = input_batch
            self._build_count = 0
            self._eagle_plans = None
            self._next_step = 0
            self._sources.clear()
            try:
                # Preparation and builders enqueue copies/kernels before model
                # forward. Even a later host error must retain this proposal.
                if self.native_swa_intersections:
                    self._native_swa_built.clear()
                yield
                # DFlash/DSpark keep context KV current when the scheduler
                # requests zero drafts. That upstream branch writes KV but
                # never executes draft attention. Retain completion protection
                # for its writes without demanding a nonexistent read view.
                if context_kv_only and self._build_count:
                    raise RuntimeError("Context-only Draft KV update unexpectedly consumed a read projection")
                if not context_kv_only and not self._build_count:
                    raise RuntimeError("Draft Window proposal did not consume a read projection")
                if self.method == "eagle3" and self._next_step != expected_eagle_steps:
                    raise RuntimeError("Eagle3 proposal did not consume every configured draft step")
            finally:
                self._owner = None
                self._input_batch = None
                self._eagle_plans = None
                self._sources.clear()

    def _source(self, gid: int, num_reqs: int) -> DirectPrefixSource:
        # The runner allocates canonical pages before propose. AR changes KV
        # contents/lengths, not this table owner or its allocated columns.
        if gid not in self._sources:
            rows = self._input_batch.idx_mapping_np[:num_reqs]
            self._sources[gid] = DirectPrefixSource.from_block_tables(self.speculator.block_tables, gid, rows)
        return self._sources[gid]

    def metadata_callback(
        self,
        step: int,
        num_reqs: int,
        *,
        prepared_exact_cpu: torch.Tensor | None = None,
        prepared_exact_device: torch.Tensor | None = None,
    ):
        """Own one metadata build; each group only chooses its read layout.

        Builders receive fresh common metadata in this proposal's batch order.
        Validate that boundary once, before either native Full, projected Host
        exact, or device-exact reads can mutate it. All three routes retire at
        the same proposal completion event.
        """
        if self._bypass:
            return None
        owner, batch = self._owner, self._input_batch
        if owner is None:
            raise RuntimeError("Draft Window metadata has no active proposal")
        if self.method != "eagle3" and step != self.speculator.num_query_per_req:
            raise ValueError("Parallel Draft metadata step must match its actual query width")
        if self.method == "eagle3" and (step != self._next_step or step >= self.static_k):
            raise ValueError("Eagle3 metadata must advance exactly one configured step")
        # The fixed propose call keeps this InputBatch's CPU request mapping
        # unchanged through every synchronous first/AR metadata preparation.
        if batch.num_reqs != num_reqs or len(batch.req_ids) != num_reqs:
            raise RuntimeError("Draft Window request mapping does not cover the active preparation")
        if (prepared_exact_cpu is None) != (prepared_exact_device is None):
            raise ValueError("Prepared exact CPU lengths require their device input")
        if prepared_exact_cpu is not None:
            if (
                prepared_exact_cpu.device.type != "cpu"
                or prepared_exact_cpu.dtype != torch.int32
                or prepared_exact_cpu.shape != (num_reqs,)
                or not prepared_exact_cpu.is_contiguous()
            ):
                raise ValueError("Prepared exact CPU lengths must be one contiguous int32 value per request")
            assert prepared_exact_device is not None
            if (
                prepared_exact_device.dtype != torch.int32
                or prepared_exact_device.ndim != 1
                or prepared_exact_device.numel() < num_reqs
                or not prepared_exact_device.is_contiguous()
            ):
                raise ValueError("Prepared exact device input must cover the active int32 request prefix")
            self.pool.retain(owner, prepared_exact_cpu)
        project = None
        seen: set[int] = set()
        full_exact = None

        def apply(gid: int, common: Any) -> None:
            nonlocal full_exact, project
            if gid not in self.gids:
                return
            if self._owner is not owner or gid in seen:
                raise RuntimeError(
                    "Draft Window callback belongs to a stale proposal or its group was already projected"
                )
            if common.num_reqs < num_reqs or common.seq_lens.ndim != 1 or common.seq_lens.numel() < num_reqs:
                raise ValueError("Draft Window metadata does not cover the active requests")
            if full_exact is None:
                full_exact = common.seq_lens
            if common.seq_lens.data_ptr() != full_exact.data_ptr():
                raise ValueError("Draft Window groups must share current preparation lengths")
            if hasattr(common, "_draft_window_read") or hasattr(common, "_draft_window_device_read"):
                raise ValueError("Draft Window requires fresh, unprojected common metadata")
            if project is None:
                project = self._select_projection(
                    step,
                    num_reqs,
                    common,
                    prepared_exact_cpu=prepared_exact_cpu,
                    prepared_exact_device=prepared_exact_device,
                )
            project(gid, common)
            seen.add(gid)
            if len(seen) == len(self.gids):
                self._build_count += 1
                if self.method == "eagle3":
                    self._next_step += 1

        return apply

    def _select_projection(
        self,
        step: int,
        num_reqs: int,
        common: Any,
        *,
        prepared_exact_cpu: torch.Tensor | None,
        prepared_exact_device: torch.Tensor | None,
    ):
        dense = self._dense_reads
        continuing = continuing_prefill_rows(self._input_batch, num_reqs)
        if self.method == "eagle3" and continuing.any():
            drafts = self._input_batch.num_draft_tokens_per_req
            if (
                dense
                and continuing.all()
                and (drafts is None or not drafts[:num_reqs].any())
                and getattr(self.speculator, "advance_draft_positions", False)
                and not getattr(self.speculator, "replicated_pcp", False)
            ):
                return self._full_prefill_metadata_callback(step, num_reqs, common)
            dense = dense and not (
                self._input_batch.is_prefilling_np[:num_reqs].all()
                or not getattr(self.speculator, "advance_draft_positions", False)
                or getattr(self.speculator, "replicated_pcp", False)
            )
        if self.method == "eagle3" and step == 0:
            prefilling = self._input_batch.is_prefilling_np[:num_reqs]
            dense = dense and (
                not prefilling.all()
                and bool(getattr(self.speculator, "advance_draft_positions", False))
                and not getattr(self.speculator, "replicated_pcp", False)
                if prefilling.any()
                else int(common.query_start_loc_cpu[: num_reqs + 1].diff().max()) <= self.static_k + 1
            )
        elif self.method != "eagle3" and dense and self._parallel_full_prefill(num_reqs, continuing):
            dense = False
        return self._host_metadata_callback(
            step,
            num_reqs,
            common,
            continuing,
            prepared_exact_cpu=prepared_exact_cpu,
            prepared_exact_device=prepared_exact_device,
            dense=dense,
        )

    def _full_prefill_metadata_callback(self, step: int, num_reqs: int, common: Any):
        owner = self._owner
        assert owner is not None
        full_exact = common.seq_lens
        self.pool.retain(owner, full_exact)
        if step == 0:
            exact, live = self._eagle_first_pass_lengths(full_exact, num_reqs)
            if not torch.equal(exact, live):
                raise ValueError("Full prompt metadata cannot include rejected query padding")
        else:
            exact = full_exact[:num_reqs].to(device="cpu", dtype=torch.int32, copy=True)
        self.pool.retain(owner, exact)

        def apply(gid: int, common: Any) -> None:
            exact_cpu = torch.zeros(common.num_reqs, dtype=torch.int32, device="cpu")
            exact_cpu[:num_reqs] = exact
            common.seq_lens_cpu = common._seq_lens_cpu = exact_cpu
            self.pool.retain(owner, common, exact_cpu, common.seq_lens, common.block_table_tensor)

        return apply

    def _eagle_first_pass_lengths(self, native_extent: torch.Tensor, num_reqs: int):
        """Snapshot native Q extent and initialized live KV end together.

        Rejected first-pass query padding belongs to Q's causal alignment,
        not the readable KV interval. The frozen CPU plan uses the last valid
        query position, while the native extent retains its query coordinates.
        """
        positions = self.speculator.input_buffers.positions
        indices = self.speculator.last_token_indices[:num_reqs]
        if (
            positions.ndim != 1
            or indices.shape != (num_reqs,)
            or positions.device != native_extent.device
            or indices.device != native_extent.device
            or positions.dtype not in (torch.int32, torch.int64)
            or indices.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("Eagle3 Window requires current one-dimensional prepared positions and last query indices")
        owner = self._owner
        if owner is None:
            raise RuntimeError("Eagle3 length snapshot has no active Draft Window proposal")
        indices = indices.to(dtype=torch.int64)
        self.pool.retain(owner, indices)
        last_positions = positions.index_select(0, indices)
        self.pool.retain(owner, last_positions)
        live_end = last_positions.to(dtype=torch.int32)
        self.pool.retain(owner, live_end)
        live_end.add_(1)
        pair = torch.stack((native_extent[:num_reqs], live_end))
        self.pool.retain(owner, pair)
        snapshot = pair.to(device="cpu", dtype=torch.int32, copy=True)
        self.pool.retain(owner, snapshot)
        extent_cpu, live_cpu = snapshot[0], snapshot[1]
        if bool(((live_cpu <= 0) | (live_cpu > extent_cpu) | (extent_cpu - live_cpu > self.static_k)).any()):
            raise ValueError("Eagle3 prepared live queries disagree with the current native padding extent")
        return extent_cpu, live_cpu

    def _parallel_full_prefill(self, num_reqs: int, continuing: np.ndarray) -> bool:
        """Keep large ongoing prompts on their native Full consumer.

        This chooses a consumer, never an exact length. Unknown layouts keep
        the existing private read path; all lengths come from actual device E.
        """
        batch = self._input_batch
        drafts = batch.num_draft_tokens_per_req
        scheduled = batch.num_scheduled_tokens[:num_reqs]
        boundaries = getattr(batch, "query_start_loc_np", None)
        return bool(
            continuing.all()
            and batch.is_prefilling_np[:num_reqs].all()
            and not getattr(self.speculator, "replicated_pcp", False)
            and (drafts is None or not drafts[:num_reqs].any())
            and (scheduled > 0).all()
            and (batch.num_computed_prefill_tokens_np[:num_reqs] + scheduled <= self.speculator.max_model_len).all()
            and boundaries is not None
            and len(boundaries) >= num_reqs + 1
            and np.array_equal(np.diff(boundaries[: num_reqs + 1]), scheduled)
        )

    def _host_metadata_callback(
        self,
        step: int,
        num_reqs: int,
        common: Any,
        continuing: np.ndarray,
        *,
        prepared_exact_cpu: torch.Tensor | None = None,
        prepared_exact_device: torch.Tensor | None = None,
        dense: bool = False,
    ):
        """Project current prepared lengths; CPU reuse requires an explicit pair.

        The caller owns a fresh exact snapshot in this active proposal and
        request order, completed from its paired device input. It must not
        modify that snapshot afterward. A common CPU field alone is not an
        exactness certificate. The proposal retains the supplied snapshot.
        """
        owner, batch = self._owner, self._input_batch
        assert owner is not None
        full_exact_input = common.seq_lens
        self.pool.retain(owner, full_exact_input)
        if self.method == "eagle3" and step == 0:
            exact, proposal_exact = self._eagle_first_pass_lengths(full_exact_input, num_reqs)
        elif prepared_exact_cpu is not None:
            assert prepared_exact_device is not None
            if (
                full_exact_input.data_ptr() != prepared_exact_device.data_ptr()
                or full_exact_input.device != prepared_exact_device.device
                or full_exact_input.dtype != prepared_exact_device.dtype
                or full_exact_input.ndim != 1
                or full_exact_input.numel() < num_reqs
                or not full_exact_input.is_contiguous()
            ):
                raise ValueError("Prepared exact CPU lengths belong to another device input")
            exact = prepared_exact_cpu
        else:
            # One exact D2H per build, shared across groups. The explicit
            # DSpark MLA snapshot avoids this duplicate copy; other
            # routes still synchronize here. An upper is never exact.
            exact = full_exact_input[:num_reqs].to(device="cpu", dtype=torch.int32, copy=True)
        if self.method == "eagle3":
            if self._eagle_plans is None:
                if step > 0:
                    raise RuntimeError("Eagle3 AR metadata has no frozen first-pass plan")
                assert proposal_exact is not None
                self._eagle_plans = tuple(
                    freeze_eagle_window_plan(
                        proposal_exact_end=int(end),
                        static_k=self.static_k,
                        max_rejected_tokens=self.static_k,
                        window_size=self.window_size,
                        block_span_tokens=self.anchor_block_span,
                    )
                    for end in proposal_exact
                )
            starts_list = [
                0
                if continuing[row] or (step == 0 and batch.is_prefilling_np[row])
                else plan.at_exact_end(int(exact[row])).window_start
                for row, plan in enumerate(self._eagle_plans)
            ]
        else:
            starts_list = [
                0
                if continuing[row]
                else compute_draft_window_plan(
                    full_exact_end=int(exact[row]),
                    window_size=self.window_size,
                    block_span_tokens=self.anchor_block_span,
                ).window_start
                for row in range(num_reqs)
            ]
        starts = torch.tensor(starts_list, dtype=torch.int32)

        def apply(gid: int, common_metadata: Any) -> None:
            source = self._source(gid, num_reqs)
            retain = lambda *objects: self.pool.retain(owner, *objects)
            view: Any
            if dense:
                from vllm_ascend.worker.v2.spec_decode.draft_window_dense import prepare_dense_draft_read

                first = self.method == "eagle3" and step == 0
                boundaries = common_metadata.query_start_loc_cpu[: num_reqs + 1]
                widths = boundaries.diff()
                query_width = int(widths.max())
                ragged = not bool((widths == query_width).all())
                view = prepare_dense_draft_read(
                    source,
                    exact_device=common_metadata.seq_lens,
                    exact_cpu=proposal_exact if first else exact,
                    query_ends_cpu=exact,
                    row_indices=batch.idx_mapping[:num_reqs],
                    upper_cpu=common_metadata.seq_lens_cpu_upper_bound[:num_reqs],
                    window_size=self.window_size,
                    anchor_span=self.anchor_block_span,
                    query_width=query_width,
                    causal=common_metadata.causal,
                    frozen_starts=starts,
                    full_rows_cpu=torch.tensor(
                        continuing | (batch.is_prefilling_np[:num_reqs] if first else False), device="cpu"
                    ),
                    query_start_loc_cpu=boundaries if ragged else None,
                    query_start_loc_device=common_metadata.query_start_loc[: num_reqs + 1] if ragged else None,
                    retain=retain,
                )
            else:
                view = materialize_direct_prefix(
                    source,
                    exact_cpu=exact,
                    exact_device=common_metadata.seq_lens,
                    upper_cpu=common_metadata.seq_lens_cpu_upper_bound[:num_reqs],
                    starts_cpu=starts,
                    padded_rows=common_metadata.num_reqs,
                    row_indices=batch.idx_mapping[:num_reqs],
                    retain=retain,
                )
            retain(view, common_metadata)
            view.apply(common_metadata)

        return apply

    def metadata_builder_callback(self, step: int):
        """Select a proposal-owned builder at the existing metadata extension point."""
        if self._bypass or not self.native_swa_intersections:
            return None
        owner = self._owner
        if owner is None:
            raise RuntimeError("Draft Window builder has no active proposal")

        def select(gid, builder, layer_names):
            if gid in self.native_swa_intersections:
                from vllm_ascend.worker.v2.spec_decode.draft_window_swa import NativeSWABuilder

                if self._owner is not owner:
                    raise RuntimeError("Native SWA builder selection belongs to a stale proposal")
                owned = NativeSWABuilder(
                    builder,
                    self,
                    owner,
                    gid,
                    self.native_swa_intersections[gid],
                    causal=self.native_swa_causal.get(gid, True),
                )
                self.pool.retain(owner, owned)
                return owned
            return builder

        return select

    def retain_metadata(self, metadata: Any) -> None:
        """Keep the final builder result, including newly allocated derived state."""
        if self._bypass:
            return
        if self._owner is None:
            raise RuntimeError("Draft Window metadata has no active proposal")
        if self.mla_layer_names:
            from vllm_ascend.worker.v2.spec_decode.draft_window_mla import freeze_mla_read_metadata

            owner = self._owner
            rotary_reads = freeze_mla_read_metadata(
                metadata, self.mla_layer_names, retain=lambda *objects: self.pool.retain(owner, *objects)
            )
            self.pool.retain(owner, *rotary_reads)
        self.pool.retain(self._owner, metadata)
