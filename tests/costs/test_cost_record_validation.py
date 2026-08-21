from __future__ import annotations

import pytest

from fidmem.costs.tracker import CostRecord


def _values() -> dict[str, object]:
    return {
        "operation": "op",
        "gpu_seconds": 0.0,
        "wall_seconds": 0.1,
        "input_frames": 0,
        "visual_tokens": 0,
        "text_tokens": 0,
        "peak_memory_bytes": 0,
        "cache_status": "miss",
        "device_name": "cpu",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gpu_seconds", float("inf")),
        ("gpu_seconds", float("nan")),
        ("wall_seconds", float("inf")),
        ("wall_seconds", float("nan")),
        ("gpu_seconds", -0.1),
        ("wall_seconds", -0.1),
    ],
)
def test_cost_record_rejects_non_finite_or_negative_durations(field: str, value: float) -> None:
    values = _values()
    values[field] = value
    with pytest.raises(ValueError, match=field):
        CostRecord(**values)


@pytest.mark.parametrize("field", ["input_frames", "visual_tokens", "text_tokens", "peak_memory_bytes"])
def test_cost_record_rejects_negative_counts(field: str) -> None:
    values = _values()
    values[field] = -1
    with pytest.raises(ValueError, match=field):
        CostRecord(**values)
