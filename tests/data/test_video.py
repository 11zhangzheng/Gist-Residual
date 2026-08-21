from pathlib import Path

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
