# SPDX-License-Identifier: Apache-2.0
"""Host wrapping/schema tests; native profiler and Graph execution substituted."""

import json
import sys
from types import SimpleNamespace

import pytest

from tests.ut.spec_decode.test_graph_repair5_contracts import load
from tests.ut.spec_decode.test_graph_repair5_review import actual_candidates


@pytest.mark.parametrize("enabled", [False, True])
def test_request_profiler_active_window_real_dispatch_no_eager_flag(monkeypatch, tmp_path, enabled):
    tool = load("p7x_graph_repair5_profile")
    calls = []

    class NativeProfiler:
        def __init__(self, **kwargs):
            assert kwargs["record_shapes"] is False and kwargs["profile_memory"] is False

        def __enter__(self):
            calls.append("profiler_start")
            return self

        def __exit__(self, *exc):
            calls.append("profiler_stop")

        def step(self):
            calls.append("step")

    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(
            __version__="HOST_STUB",
            profiler=SimpleNamespace(
                profile=NativeProfiler,
                tensorboard_trace_handler=lambda _: lambda p: None,
                ProfilerActivity=SimpleNamespace(CPU="CPU", NPU="NPU"),
            ),
        ),
    )
    manager = actual_candidates(7, [7, 14, 28])
    # Native executions substituted, real dispatch not substituted.
    manager.graphs = {
        d: SimpleNamespace(replay=lambda: calls.append("native_replay"))
        for d in manager._capture_descs[next(iter(manager._capture_descs))]
    }
    manager.run_fullgraph = lambda d: manager.graphs[d].replay()
    spec = SimpleNamespace(query_cudagraph_manager=manager)

    def propose(*args, **kwargs):
        assert not kwargs.get("is_profile")
        manager.run_fullgraph(manager.dispatch(1, 7, 7, 0))

    spec.propose = propose
    target = SimpleNamespace(run_fullgraph=lambda: None, graphs={})
    runner = SimpleNamespace(speculator=spec, cudagraph_manager=target)
    original = spec.propose
    with tool.RequestProfiler(runner, tmp_path, {"pid": 1}, enabled=enabled) as profile:
        with profile.group("warmup"):
            spec.propose()
        profile.begin_active()
        with profile.group("active"):
            spec.propose()
    assert spec.propose is original
    assert calls.count("native_replay") == 2
    assert ("profiler_start" in calls) is enabled
    data = json.loads((tmp_path / "profile.json").read_text())
    assert data["status"] == "GRAPH_REPLAY_OBSERVED"
    assert data["host_statistics"]["Draft_REPLAY_SUBMIT"]["samples"] == 1
    assert data["physical_read"] == "NOT_PROVEN"


def test_profiling_rejects_heavy_observer(tmp_path):
    tool = load("p7x_graph_repair5_profile")
    runner = SimpleNamespace(
        speculator=SimpleNamespace(
            _draft_window_graph_characterization=SimpleNamespace(recorder=SimpleNamespace(enabled=True))
        )
    )
    with pytest.raises(ValueError, match="observer OFF"):
        tool.RequestProfiler(runner, tmp_path, {}, enabled=True).__enter__()


def test_streaming_csv_analysis_does_not_add_overlapping_durations(tmp_path):
    tool = load("p7x_graph_repair5_profile")
    (tmp_path / "tasks.csv").write_text("Name,Duration(us),Stream ID\nMemcpy,10,1\nWait,5,2\n")
    result = tool.analyze_csv([tmp_path])
    assert result["files"][0]["nonempty"]["Stream ID"] == 2
    assert result["device_idle"].startswith("UNRESOLVED")
    assert result["totals"][f"{tmp_path / 'tasks.csv'}:copy"]["duration_us_sum"] == 10
    (tmp_path / "unknown.csv").write_text("Name,Duration\nMemcpy,1000\n")
    result = tool.analyze_csv([tmp_path])
    assert not any("unknown.csv" in key for key in result["totals"])


def test_streaming_pack_verify_and_corruption(tmp_path):
    tool = load("p7x_graph_repair5_pack")
    source = tmp_path / "source"
    source.mkdir()
    (source / "payload.bin").write_bytes(b"x" * (5 * 1024 * 1024))
    (source / "identity.json").write_text('{"source":"host_test"}')
    path = tmp_path / "evidence.zip"
    assert tool.pack(source, path)["status"] == "VERIFIED"
    assert tool.verify(path)["members"] == 4
    with pytest.raises(ValueError):
        tool.safe_name("../outside")
