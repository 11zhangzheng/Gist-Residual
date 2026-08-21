from pathlib import Path
import subprocess

import fidmem.data.video as video
import pytest

from fidmem.data.video import probe_video, sample_frames, segment_video


FIXTURE = Path(__file__).parents[1] / "fixtures" / "tiny_video.mp4"


def test_tiny_video_can_be_probed_sampled_and_segmented(tmp_path: Path) -> None:
    """A regression against breaking the deterministic ingest fixture."""
    metadata = probe_video(FIXTURE)

    assert metadata.duration_sec == pytest.approx(4.0, abs=0.1)
    assert (metadata.width, metadata.height) == (320, 240)
    assert metadata.fps == pytest.approx(10.0)

    frames = sample_frames(FIXTURE, (0.0, 1.5, 3.0), tmp_path / "frames")

    assert len(frames) == 3
    assert all(frame.is_file() for frame in frames)

    events = segment_video(FIXTURE, min_sec=1, max_sec=3)

    assert events[0].start_sec == 0.0
    assert events[-1].end_sec == pytest.approx(4.0, abs=0.1)


def test_sample_frames_rejects_a_timestamp_beyond_video_duration(tmp_path: Path) -> None:
    """A regression against accepting a seek past the decoded video timeline."""
    with pytest.raises(ValueError, match="duration"):
        sample_frames(FIXTURE, (4.1,), tmp_path / "frames")


def test_sample_frames_rejects_a_successful_command_without_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A regression against returning paths for frames ffmpeg did not create."""

    def successful_command_without_output(
        command: list[str],
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(video, "_run", successful_command_without_output)

    with pytest.raises(RuntimeError, match="no output"):
        video.sample_frames(FIXTURE, (1.0,), tmp_path / "frames")
