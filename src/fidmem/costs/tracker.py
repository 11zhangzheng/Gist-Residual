"""End-to-end resource accounting for fidelity-memory operations."""
from __future__ import annotations
from collections import defaultdict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from math import isfinite
from time import perf_counter_ns
from typing import Any
_AUTO_CUDA = object()
@dataclass(frozen=True)
class CostRecord:
    operation: str
    gpu_seconds: float
    wall_seconds: float
    input_frames: int
    visual_tokens: int
    text_tokens: int
    peak_memory_bytes: int
    cache_status: str
    device_name: str

    def __post_init__(self) -> None:
        self.validate_values()

    def validate_values(self) -> None:
        for field_name in ("gpu_seconds", "wall_seconds"):
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
                raise ValueError(f"{field_name} must be finite and non-negative")
        for field_name in ("input_frames", "visual_tokens", "text_tokens", "peak_memory_bytes"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
@dataclass
class CostMeasurement:
    record: CostRecord | None = None
@dataclass(frozen=True)
class _TrackedRecord:
    record: CostRecord
    video_id: str | None
    question_id: str | None
    action_type: str | None
    cost_component: str
def amortized_total(base_gpu_s: float, online_gpu_s: float, answer_gpu_s: float, query_count: int) -> float:
    if query_count < 1:
        raise ValueError("query_count must be at least one")
    return base_gpu_s / query_count + online_gpu_s + answer_gpu_s
class CostTracker:
    def __init__(self, cuda_module: Any = _AUTO_CUDA) -> None:
        self._cuda = self._resolve_cuda(cuda_module)
        self._records: list[_TrackedRecord] = []
    @staticmethod
    def _resolve_cuda(cuda_module: Any) -> Any | None:
        if cuda_module is None:
            return None
        if cuda_module is _AUTO_CUDA:
            try:
                import torch
            except ImportError:
                return None
            cuda_module = torch.cuda
        try:
            return cuda_module if cuda_module.is_available() else None
        except (AttributeError, RuntimeError):
            return None
    @staticmethod
    def _token_counts(tokens: int | tuple[int, int] | Mapping[str, int]) -> tuple[int, int]:
        if isinstance(tokens, int):
            return 0, tokens
        if isinstance(tokens, tuple):
            if len(tokens) != 2:
                raise ValueError("token tuple must contain visual and text counts")
            return tokens
        return tokens.get("visual", 0), tokens.get("text", 0)
    @contextmanager
    def measure(self, operation: str, cache_status: str, frames: int, tokens: int | tuple[int, int] | Mapping[str, int], *, video_id: str | None = None, question_id: str | None = None, action_type: str | None = None, cost_component: str = "online") -> Iterator[CostMeasurement]:
        if frames < 0:
            raise ValueError("frames must be non-negative")
        visual_tokens, text_tokens = self._token_counts(tokens)
        if visual_tokens < 0 or text_tokens < 0:
            raise ValueError("token counts must be non-negative")
        if cost_component not in {"base", "online", "answer", "visual"}:
            raise ValueError("unknown cost_component")
        measurement = CostMeasurement()
        start_event: Any | None = None
        end_event: Any | None = None
        if self._cuda is not None:
            self._cuda.synchronize()
            reset_peak = getattr(self._cuda, "reset_peak_memory_stats", None)
            if callable(reset_peak):
                reset_peak()
            start_event = self._cuda.Event(enable_timing=True)
            end_event = self._cuda.Event(enable_timing=True)
            start_event.record()
        wall_start_ns = perf_counter_ns()
        try:
            yield measurement
        finally:
            if self._cuda is None:
                gpu_seconds, peak_memory_bytes, device_name = 0.0, 0, "cpu"
            else:
                end_event.record()
                self._cuda.synchronize()
                gpu_seconds = start_event.elapsed_time(end_event) / 1_000
                peak_memory_bytes = int(self._cuda.max_memory_allocated())
                device_name = str(self._cuda.get_device_name())
            wall_seconds = (perf_counter_ns() - wall_start_ns) / 1_000_000_000
            record = CostRecord(operation, gpu_seconds, wall_seconds, frames, visual_tokens, text_tokens, peak_memory_bytes, cache_status, device_name)
            measurement.record = record
            self._records.append(_TrackedRecord(record, video_id, question_id, action_type, cost_component))
    def aggregate(self, query_counts: Mapping[str, int] | None = None) -> dict[tuple[str | None, str | None, str | None, str], dict[str, float | int]]:
        totals: dict[tuple[str | None, str | None, str | None, str], dict[str, float | int]] = defaultdict(lambda: {"count": 0, "gpu_seconds": 0.0, "wall_seconds": 0.0, "input_frames": 0, "visual_tokens": 0, "text_tokens": 0, "peak_memory_bytes": 0})
        for tracked in self._records:
            record = tracked.record
            key = (tracked.video_id, tracked.question_id, tracked.action_type, record.cache_status)
            total = totals[key]
            if tracked.cost_component == 'base':
                if query_counts is None or tracked.video_id not in query_counts:
                    raise ValueError('base costs require a per-video query count')
                divisor = query_counts[tracked.video_id]
            else:
                divisor = 1
            if divisor < 1:
                raise ValueError("base costs require a positive per-video query count")
            total["count"] += 1
            total["gpu_seconds"] += record.gpu_seconds / divisor
            total["wall_seconds"] += record.wall_seconds / divisor
            total["input_frames"] += record.input_frames / divisor
            total["visual_tokens"] += record.visual_tokens / divisor
            total["text_tokens"] += record.text_tokens / divisor
            total["peak_memory_bytes"] = max(total["peak_memory_bytes"], record.peak_memory_bytes)
        return dict(totals)