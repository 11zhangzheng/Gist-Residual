from pathlib import Path

import duckdb
import pytest

from fidmem.data.leakage import LeakageAuditor, VideoAsset


def test_audit_reports_a_copied_file_as_a_hash_duplicate(tmp_path: Path) -> None:
    """A regression against missing byte-identical videos with different IDs."""
    source = tmp_path / "train.mp4"
    source.write_bytes(b"same-video-bytes")
    copied = tmp_path / "eval.mp4"
    copied.write_bytes(source.read_bytes())
    output = tmp_path / "leakage.parquet"

    report = LeakageAuditor(output).audit(
        (VideoAsset("train-video", source),),
        (VideoAsset("eval-video", copied),),
    )

    assert [(finding.kind, finding.train_video_id, finding.eval_video_id) for finding in report.findings] == [
        ("hash_duplicate", "train-video", "eval-video"),
    ]
    assert duckdb.sql("SELECT kind FROM read_parquet(?)", params=[str(output)]).fetchall() == [
        ("hash_duplicate",),
    ]


def test_audit_reports_cosine_near_duplicates_after_id_and_hash_checks(
    tmp_path: Path,
) -> None:
    """A regression against silently accepting visually near-identical videos."""
    train_path = tmp_path / "train.mp4"
    train_path.write_bytes(b"train")
    eval_path = tmp_path / "eval.mp4"
    eval_path.write_bytes(b"eval")
    output = tmp_path / "leakage.parquet"

    report = LeakageAuditor(output).audit(
        (VideoAsset("train-video", train_path, ((1.0, 0.0),) * 8),),
        (VideoAsset("eval-video", eval_path, ((0.99, 0.01),) * 8),),
    )

    assert len(report.findings) == 1
    assert report.findings[0].kind == "near_duplicate"
    assert report.findings[0].cosine_similarity >= 0.985


def test_audit_normalizes_ids_before_comparing_them(tmp_path: Path) -> None:
    """A regression against allowing cosmetic video-ID variants across splits."""
    train_path = tmp_path / "train.mp4"
    train_path.write_bytes(b"one")
    eval_path = tmp_path / "eval.mp4"
    eval_path.write_bytes(b"two")
    output = tmp_path / "leakage.parquet"

    report = LeakageAuditor(output).audit(
        (VideoAsset("Scene 01.MP4", train_path),),
        (VideoAsset("scene-01", eval_path),),
    )

    assert [finding.kind for finding in report.findings] == ["id_duplicate"]


def test_audit_rejects_precomputed_embeddings_without_eight_frames(
    tmp_path: Path,
) -> None:
    """A regression against producing an audit from a non-protocol frame count."""
    train_path = tmp_path / "train.mp4"
    train_path.write_bytes(b"one")
    eval_path = tmp_path / "eval.mp4"
    eval_path.write_bytes(b"two")

    with pytest.raises(ValueError, match="eight"):
        LeakageAuditor(tmp_path / "leakage.parquet").audit(
            (VideoAsset("train", train_path, ((1.0, 0.0),)),),
            (VideoAsset("eval", eval_path, ((0.0, 1.0),)),),
        )


@pytest.mark.parametrize("count", (1, 7, 9))
def test_audit_rejects_provider_embedding_counts_other_than_eight(
    tmp_path: Path, count: int
) -> None:
    """A regression against accepting a malformed provider embedding batch."""
    source = Path(__file__).parents[1] / "fixtures" / "tiny_video.mp4"
    altered = tmp_path / "altered.mp4"
    altered.write_bytes(source.read_bytes() + b"\0")
    embeddings = tuple((1.0, 0.0) for _ in range(count))

    with pytest.raises(ValueError, match="exactly eight"):
        LeakageAuditor(
            tmp_path / "leakage.parquet",
            embedding_provider=lambda _path, _frames: embeddings,
        ).audit(
            (VideoAsset("train", source),),
            (VideoAsset("eval", altered),),
        )


@pytest.mark.parametrize(
    ("train_id", "eval_id"),
    (("\u732b", "\u72d7"), ("---", "___")),
)
def test_audit_does_not_collide_distinct_unicode_or_empty_normalized_ids(
    tmp_path: Path, train_id: str, eval_id: str
) -> None:
    """A regression against non-ASCII IDs collapsing into an ID duplicate."""
    train_path = tmp_path / "train.mp4"
    train_path.write_bytes(b"train")
    eval_path = tmp_path / "eval.mp4"
    eval_path.write_bytes(b"eval")

    report = LeakageAuditor(tmp_path / "leakage.parquet").audit(
        (VideoAsset(train_id, train_path),),
        (VideoAsset(eval_id, eval_path),),
    )

    assert report.findings == ()
