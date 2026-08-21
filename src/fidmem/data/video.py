"""Thin deterministic wrappers around the bundled ffmpeg executable."""

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Iterable

from imageio_ffmpeg import get_ffmpeg_exe, read_frames

from .segmentation import Segment, segment_timestamps


@dataclass(frozen=True)
class VideoProbe:
    path: Path
    duration_sec: float
    width: int
    height: int
    fps: float


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise RuntimeError(f"required video tool is unavailable: {command[0]}") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip()
        raise RuntimeError(f"video command failed: {detail}") from error


def _parse_fps(value: str) -> float:
    numerator, separator, denominator = value.partition("/")
    if not separator:
        return float(numerator)
    denominator_value = float(denominator)
    if denominator_value == 0:
        raise ValueError("ffprobe returned a zero frame-rate denominator")
    return float(numerator) / denominator_value


def probe_video(path: str | Path) -> VideoProbe:
    """Read duration, dimensions, and frame rate with bundled ffmpeg."""
    video_path = Path(path)
    reader = read_frames(video_path)
    try:
        metadata = next(reader)
    finally:
        reader.close()
    width, height = metadata["source_size"]
    return VideoProbe(
        path=video_path,
        duration_sec=float(metadata["duration"]),
        width=int(width),
        height=int(height),
        fps=float(metadata["fps"]),
    )


def sample_frames(
    path: str | Path,
    timestamps_sec: Iterable[float],
    output_dir: str | Path,
) -> tuple[Path, ...]:
    """Extract one JPEG per timestamp using stable, timestamped filenames."""
    video_path = Path(path)
    duration_sec = probe_video(video_path).duration_sec
    timestamps = tuple(float(timestamp) for timestamp in timestamps_sec)
    if any(timestamp < 0 or timestamp >= duration_sec for timestamp in timestamps):
        raise ValueError("frame timestamps must lie within the probed video duration")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    frames: list[Path] = []
    for index, seconds in enumerate(timestamps):
        frame = destination / f"frame_{index:03d}_{round(seconds * 1000):010d}.jpg"
        _run(
            [
                get_ffmpeg_exe(),
                "-y",
                "-ss",
                f"{seconds:.6f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(frame),
            ]
        )
        if not frame.is_file() or frame.stat().st_size == 0:
            raise RuntimeError(
                f"frame extraction produced no output for timestamp {seconds:.6f}"
            )
        frames.append(frame)
    return tuple(frames)


def segment_video(
    path: str | Path,
    *,
    shots: Iterable[float] = (),
    speech_breaks: Iterable[float] = (),
    min_sec: float = 8.0,
    max_sec: float = 40.0,
) -> tuple[Segment, ...]:
    """Probe a video and return deterministic cheap event boundaries."""
    return segment_timestamps(
        probe_video(path).duration_sec,
        shots=shots,
        speech_breaks=speech_breaks,
        min_sec=min_sec,
        max_sec=max_sec,
    )
