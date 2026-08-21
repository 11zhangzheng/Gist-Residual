from time import sleep

import pytest

from fidmem.costs.tracker import CostTracker, amortized_total


def test_amortized_total_counts_first_cache_generation() -> None:
    total = amortized_total(
        base_gpu_s=80.0,
        online_gpu_s=12.0,
        answer_gpu_s=3.0,
        query_count=4,
    )

    assert total == 35.0


def test_amortized_total_rejects_no_queries() -> None:
    with pytest.raises(ValueError, match="query_count"):
        amortized_total(80.0, 12.0, 3.0, 0)


def test_measure_records_cpu_wall_time_and_usage() -> None:
    tracker = CostTracker(cuda_module=None)

    with tracker.measure(
        operation="visual_encode",
        cache_status="miss",
        frames=8,
        tokens={"visual": 24, "text": 7},
    ) as measurement:
        sleep(0.001)

    record = measurement.record
    assert record.operation == "visual_encode"
    assert record.gpu_seconds == 0.0
    assert record.wall_seconds > 0.0
    assert record.input_frames == 8
    assert record.visual_tokens == 24
    assert record.text_tokens == 7
    assert record.peak_memory_bytes == 0
    assert record.cache_status == "miss"
    assert record.device_name == "cpu"


def test_aggregate_groups_costs_by_requested_dimensions() -> None:
    tracker = CostTracker(cuda_module=None)
    with tracker.measure(
        "retrieve",
        "hit",
        0,
        3,
        video_id="video-a",
        question_id="question-1",
        action_type="SEARCH_GIST",
    ):
        pass
    with tracker.measure(
        "retrieve",
        "hit",
        0,
        5,
        video_id="video-a",
        question_id="question-1",
        action_type="SEARCH_GIST",
    ):
        pass

    summary = tracker.aggregate()
    assert summary[("video-a", "question-1", "SEARCH_GIST", "hit")]["count"] == 2
    assert summary[("video-a", "question-1", "SEARCH_GIST", "hit")]["text_tokens"] == 8

def test_measure_uses_cuda_events_and_peak_memory_when_available() -> None:
    class Event:
        def record(self) -> None:
            pass

        def elapsed_time(self, start: "Event") -> float:
            return 125.0

    class FakeCuda:
        def __init__(self) -> None:
            self.synchronize_calls = 0

        def is_available(self) -> bool:
            return True

        def synchronize(self) -> None:
            self.synchronize_calls += 1

        def Event(self, enable_timing: bool) -> "Event":
            assert enable_timing
            return Event()

        def max_memory_allocated(self) -> int:
            return 4096

        def get_device_name(self) -> str:
            return "fake-cuda"

    cuda = FakeCuda()
    tracker = CostTracker(cuda_module=cuda)

    with tracker.measure("answer", "miss", 2, (9, 4)) as measurement:
        pass

    assert cuda.synchronize_calls == 2
    assert measurement.record.gpu_seconds == 0.125
    assert measurement.record.peak_memory_bytes == 4096


def test_cuda_measurement_uses_start_event_and_includes_trailing_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    from fidmem.costs import tracker as tracker_module

    calls: list[str] = []

    class Event:
        def __init__(self, name: str) -> None:
            self.name = name

        def record(self) -> None:
            calls.append(f"{self.name}.record")

        def elapsed_time(self, end: "Event") -> float:
            assert self.name == "start"
            assert end.name == "end"
            return 125.0

    class FakeCuda:
        def __init__(self) -> None:
            self.event_count = 0

        def is_available(self) -> bool:
            return True

        def synchronize(self) -> None:
            calls.append("sync")

        def Event(self, enable_timing: bool) -> "Event":
            assert enable_timing
            self.event_count += 1
            return Event("start" if self.event_count == 1 else "end")

        def max_memory_allocated(self) -> int:
            return 4096

        def get_device_name(self) -> str:
            return "fake-cuda"

    ticks = iter((1_000, 2_000))
    monkeypatch.setattr(tracker_module, "perf_counter_ns", lambda: next(ticks))
    with CostTracker(cuda_module=FakeCuda()).measure("answer", "miss", 2, (9, 4)) as measurement:
        pass

    assert calls == ["sync", "start.record", "end.record", "sync"]
    assert measurement.record.gpu_seconds == 0.125
    assert measurement.record.wall_seconds == 0.000001


def test_aggregate_amortizes_base_and_recharges_visual_per_question() -> None:
    class Event:
        def __init__(self, elapsed_ms: float) -> None:
            self.elapsed_ms = elapsed_ms

        def record(self) -> None:
            pass

        def elapsed_time(self, end: "Event") -> float:
            return self.elapsed_ms

    class FakeCuda:
        def __init__(self) -> None:
            self.elapsed_ms = iter((80_000.0, 0.0, 5_000.0, 0.0, 80_000.0, 0.0, 5_000.0, 0.0))

        def is_available(self) -> bool:
            return True

        def synchronize(self) -> None:
            pass

        def Event(self, enable_timing: bool) -> "Event":
            return Event(next(self.elapsed_ms))

        def max_memory_allocated(self) -> int:
            return 0

        def get_device_name(self) -> str:
            return "fake-cuda"

    tracker = CostTracker(cuda_module=FakeCuda())
    for question_id in ("q1", "q2"):
        with tracker.measure("base", "miss", 0, 0, video_id="v1", question_id=question_id, action_type="BUILD", cost_component="base"):
            pass
        with tracker.measure("visual", "miss", 4, (10, 0), video_id="v1", question_id=question_id, action_type="VERIFY", cost_component="visual"):
            pass

    summary = tracker.aggregate(query_counts={"v1": 2})
    assert summary[("v1", "q1", "BUILD", "miss")]["gpu_seconds"] == 40.0
    assert summary[("v1", "q2", "BUILD", "miss")]["gpu_seconds"] == 40.0
    assert summary[("v1", "q1", "VERIFY", "miss")]["gpu_seconds"] == 5.0
    assert summary[("v1", "q2", "VERIFY", "miss")]["gpu_seconds"] == 5.0

try:
    import torch
except ImportError:
    torch = None


@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="CUDA unavailable")
def test_measure_uses_real_cuda_events() -> None:
    tracker = CostTracker(cuda_module=torch.cuda)
    with tracker.measure("cuda-smoke", "miss", 1, 0) as measurement:
        torch.zeros(1, device="cuda")
    assert measurement.record.gpu_seconds >= 0.0
    assert measurement.record.device_name != "cpu"

def test_aggregate_rejects_base_cost_without_video_query_count() -> None:
    tracker = CostTracker(cuda_module=None)
    with tracker.measure("base", "miss", 0, 0, video_id="v1", question_id="q1", action_type="BUILD", cost_component="base"):
        pass
    with pytest.raises(ValueError, match="query count"):
        tracker.aggregate()