"""Deterministic, inexpensive event boundaries from shots and ASR pauses."""

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Segment:
    """A contiguous interval on a video's timeline, expressed in seconds."""

    start_sec: float
    end_sec: float

    def __post_init__(self) -> None:
        if self.start_sec < 0 or self.end_sec < self.start_sec:
            raise ValueError("segment boundaries must be non-negative and ordered")

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


def _unique_times(times: Iterable[float], duration_sec: float) -> tuple[float, ...]:
    return tuple(
        sorted({float(time) for time in times if 0.0 < float(time) < duration_sec})
    )


def _merged_shot_boundaries(
    shots: Iterable[float], duration_sec: float, min_sec: float
) -> tuple[float, ...]:
    """Drop shot cuts that would retain an event shorter than ``min_sec``."""
    retained: list[float] = []
    previous = 0.0
    for boundary in _unique_times(shots, duration_sec):
        if boundary - previous >= min_sec:
            retained.append(boundary)
            previous = boundary
    return tuple(retained)


def segment_timestamps(
    duration_sec: float,
    *,
    shots: Iterable[float] = (),
    speech_breaks: Iterable[float] = (),
    min_sec: float = 8.0,
    max_sec: float = 40.0,
) -> tuple[Segment, ...]:
    """Split a timeline without overlap or gaps, preferring ASR pause boundaries.

    Short consecutive shots are folded together.  At every valid boundary the
    earliest ASR pause is preferred; otherwise the latest retained shot is
    chosen.  A deterministic hard cut prevents intervals longer than
    ``max_sec`` while preserving a final interval of at least ``min_sec``.
    """
    duration = float(duration_sec)
    if duration <= 0:
        raise ValueError("duration_sec must be positive")
    if min_sec <= 0 or max_sec < min_sec:
        raise ValueError("require 0 < min_sec <= max_sec")

    pauses = _unique_times(speech_breaks, duration)
    shots_merged = _merged_shot_boundaries(shots, duration, min_sec)
    segments: list[Segment] = []
    start = 0.0

    while duration - start > max_sec:
        upper = min(start + max_sec, duration - min_sec)
        lower = start + min_sec
        valid_pauses = [time for time in pauses if lower <= time <= upper]
        if valid_pauses:
            end = valid_pauses[0]
        else:
            valid_shots = [time for time in shots_merged if lower <= time <= upper]
            end = valid_shots[-1] if valid_shots else upper
        segments.append(Segment(start, end))
        start = end

    segments.append(Segment(start, duration))
    return tuple(segments)
