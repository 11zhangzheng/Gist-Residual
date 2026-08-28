"""Engineering-evidence-only LongTVQA Source Gate tests."""

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from fidmem.assets.longtvqa import (
    METADATA_FILES,
    build_human_audit_manifest,
    build_manifests,
    validate_human_audit_result,
    verify_metadata,
    verify_raw_videos,
)
from fidmem.data.video import VideoProbe
from fidmem.production.manifests import validate_split_isolation


REVISION = "a" * 40


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _metadata(root: Path, *, videos: int = 24, questions_per_video: int = 5) -> None:
    root.mkdir()
    rows = [
        {
            "qid": f"q{video:03d}-{question}",
            "video_id": f"episode-{video:03d}",
            "question": "engineering fixture only",
            "answer": "A",
            "options": ["A", "B", "C", "D"],
            "question_type": "fixture",
            "timestamp": [1.0, 2.0],
        }
        for video in range(videos)
        for question in range(questions_per_video)
    ]
    midpoint = len(rows) // 2
    _jsonl(root / "LongTVQA_train.jsonl", rows[:midpoint])
    _jsonl(root / "LongTVQA_val.jsonl", rows[midpoint:])
    subtitle_rows = [
        {"video_id": f"episode-{video:03d}", "text": "fixture"}
        for video in range(videos)
    ]
    _jsonl(root / "LongTVQA_subtitles_clip_level.jsonl", subtitle_rows)
    _jsonl(root / "LongTVQA_subtitles_episode_level.jsonl", subtitle_rows)
    (root / "README.md").write_text("engineering fixture only", encoding="utf-8")


def _videos(root: Path, count: int = 24) -> None:
    root.mkdir()
    for index in range(count):
        (root / f"episode-{index:03d}.mp4").write_bytes(f"fixture-{index}".encode())


def _probe(path: str | Path) -> VideoProbe:
    source = Path(path)
    return VideoProbe(path=source, duration_sec=60.0, width=16, height=16, fps=1.0)


def test_metadata_and_human_audit_remain_engineering_pending(tmp_path: Path) -> None:
    metadata_root = tmp_path / "metadata"
    _metadata(metadata_root)
    parsed = verify_metadata(metadata_root, immutable_revision=REVISION)
    assert parsed.report.status == "VERIFIED"
    assert parsed.report.qa_unconstructible_count == 0
    assert parsed.report.files == METADATA_FILES
    audit = build_human_audit_manifest(parsed, seed="fixture-seed")
    assert audit.status == "PENDING_HUMAN_AUDIT"
    assert len(audit.items) == 100
    assert audit.evidence_class == "engineering"
    result = tmp_path / "human-result.json"
    result.write_text(json.dumps({"status": "PENDING"}), encoding="utf-8")
    with pytest.raises(ValueError, match="not COMPLETED"):
        validate_human_audit_result(audit, result)


def test_missing_or_corrupt_video_fails_source_gate(tmp_path: Path) -> None:
    metadata_root = tmp_path / "metadata"
    video_root = tmp_path / "videos"
    _metadata(metadata_root)
    _videos(video_root, count=23)
    parsed = verify_metadata(metadata_root, immutable_revision=REVISION)

    def corrupt_probe(path: str | Path) -> VideoProbe:
        if Path(path).stem == "episode-000":
            raise RuntimeError("fixture corruption")
        return _probe(path)

    report = verify_raw_videos(
        parsed,
        video_root,
        probe=corrupt_probe,
        decode=lambda _path, _timestamp: None,
    )
    assert report.status == "FAIL"
    assert report.missing_video_ids == ("episode-023",)
    assert report.corrupt_video_ids == ("episode-000",)


def test_duplicate_video_content_identity_fails_source_gate(tmp_path: Path) -> None:
    metadata_root = tmp_path / "metadata"
    video_root = tmp_path / "videos"
    _metadata(metadata_root)
    _videos(video_root)
    (video_root / "episode-001.mp4").write_bytes(
        (video_root / "episode-000.mp4").read_bytes()
    )
    parsed = verify_metadata(metadata_root, immutable_revision=REVISION)
    report = verify_raw_videos(
        parsed, video_root, probe=_probe, decode=lambda _path, _timestamp: None
    )
    assert report.status == "FAIL"
    assert report.duplicate_video_ids == ("episode-000", "episode-001")


def test_builds_existing_video_disjoint_manifests_and_fixed_selections(
    tmp_path: Path,
) -> None:
    metadata_root = tmp_path / "metadata"
    video_root = tmp_path / "videos"
    _metadata(metadata_root)
    _videos(video_root)
    parsed = verify_metadata(metadata_root, immutable_revision=REVISION)
    video_report = verify_raw_videos(
        parsed,
        video_root,
        probe=_probe,
        decode=lambda _path, _timestamp: None,
    )
    assert video_report.status == "PASS"
    assignments = {"episode-000": "development", "episode-001": "holdout"}
    assignments.update({f"episode-{index:03d}": "canary" for index in range(2, 4)})
    assignments.update({f"episode-{index:03d}": "oracle" for index in range(4, 24)})
    payload = {
        "schema_version": 1,
        "split_policy_id": "engineering-fixture-only",
        "status": "FROZEN",
        "split_unit": "video_id",
        "video_groups": assignments,
        "selections": {
            "canary": {"count": 10, "seed": "fixture-canary"},
            "oracle": {"count": 100, "seed": "fixture-oracle"},
        },
    }
    policy_path = tmp_path / "policy.yaml"
    OmegaConf.save(config=OmegaConf.create(payload), f=policy_path)
    videos, questions, dataset, canary, oracle = build_manifests(
        parsed, video_report, split_policy_path=policy_path
    )
    validate_split_isolation(videos, questions)
    assert dataset.video_manifest_sha256 == videos.manifest_sha256
    assert len(canary.question_ids) == 10
    assert len(oracle.question_ids) == 100


def test_unfrozen_split_policy_fails_closed(tmp_path: Path) -> None:
    metadata_root = tmp_path / "metadata"
    video_root = tmp_path / "videos"
    _metadata(metadata_root)
    _videos(video_root)
    parsed = verify_metadata(metadata_root, immutable_revision=REVISION)
    video_report = verify_raw_videos(
        parsed, video_root, probe=_probe, decode=lambda _path, _timestamp: None
    )
    policy = tmp_path / "policy.yaml"
    policy.write_text("status: RESEARCH_OWNER_DECISION_REQUIRED\n", encoding="utf-8")
    with pytest.raises(ValueError, match="RESEARCH_OWNER_DECISION_REQUIRED"):
        build_manifests(parsed, video_report, split_policy_path=policy)
