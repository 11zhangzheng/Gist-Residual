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
