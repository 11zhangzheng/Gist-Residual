from __future__ import annotations

import pytest
from pydantic import ValidationError

from fidmem.production.manifests import (
    DatasetManifest,
    QuestionManifest,
    QuestionManifestRecord,
    VideoManifest,
    VideoManifestRecord,
    select_questions_deterministically,
    selection_rank_sha256,
    validate_split_isolation,
)


def test_partial_dataset_manifest_requires_selection_identity() -> None:
    with pytest.raises(ValidationError, match="subset selection"):
        DatasetManifest(
            dataset_name="MME-Benchmarks/Video-MME-v2",
            dataset_version="6e4bebb03202e1ddbf3d37703e560e51c5aa2d64",
            dataset_scope="PARTIAL_DATASET_PILOT",
            source_metadata_sha256="1" * 64,
            source_archive_index_sha256="2" * 64,
            subset_selection_manifest_sha256=None,
            selected_video_count=45,
            selected_question_count=180,
            available_video_count=800,
            available_question_count=3200,
            split_policy_id="videomme-v2-pilot-split-v1",
            split_policy_sha256="3" * 64,
            video_manifest_sha256="4" * 64,
            question_manifest_sha256="5" * 64,
        )


def test_dataset_manifest_selected_counts_cannot_exceed_available_counts() -> None:
    with pytest.raises(ValidationError, match="selected video count exceeds"):
        DatasetManifest(
            dataset_name="MME-Benchmarks/Video-MME-v2",
            dataset_version="6e4bebb03202e1ddbf3d37703e560e51c5aa2d64",
            dataset_scope="PARTIAL_DATASET_PILOT",
            source_metadata_sha256="1" * 64,
            source_archive_index_sha256="2" * 64,
            subset_selection_manifest_sha256="6" * 64,
            selected_video_count=801,
            selected_question_count=180,
            available_video_count=800,
            available_question_count=3200,
            split_policy_id="videomme-v2-pilot-split-v1",
            split_policy_sha256="3" * 64,
            video_manifest_sha256="4" * 64,
            question_manifest_sha256="5" * 64,
        )


def test_full_dataset_manifest_forbids_selection_identity() -> None:
    with pytest.raises(ValidationError, match="full dataset forbids"):
        DatasetManifest(
            dataset_name="MME-Benchmarks/Video-MME-v2",
            dataset_version="6e4bebb03202e1ddbf3d37703e560e51c5aa2d64",
            dataset_scope="FULL_DATASET",
            source_metadata_sha256="1" * 64,
            source_archive_index_sha256="2" * 64,
            subset_selection_manifest_sha256="6" * 64,
            selected_video_count=800,
            selected_question_count=3200,
            available_video_count=800,
            available_question_count=3200,
            split_policy_id="videomme-v2-pilot-split-v1",
            split_policy_sha256="3" * 64,
            video_manifest_sha256="4" * 64,
            question_manifest_sha256="5" * 64,
        )


def video(video_id: str, group: str) -> VideoManifestRecord:
    return VideoManifestRecord(
        video_id=video_id,
        content_sha256="a" * 64,
        uri=f"videos/{video_id}.mp4",
        duration_seconds=60.0,
        group=group,
    )


def question(
    question_id: str,
    video_id: str,
    group: str,
    *,
    gold_answer_sha256: str | None = None,
    ground_truth_scope: str = "none",
) -> QuestionManifestRecord:
    return QuestionManifestRecord(
        question_id=question_id,
        video_id=video_id,
        record_sha256="b" * 64,
        question_types=("visual",),
        group=group,
        gold_answer_sha256=gold_answer_sha256,
        ground_truth_scope=ground_truth_scope,
    )


def video_manifest(*records: VideoManifestRecord) -> VideoManifest:
    return VideoManifest(dataset_name="dataset", dataset_version="rev", records=records)


def question_manifest(*records: QuestionManifestRecord) -> QuestionManifest:
    return QuestionManifest(
        dataset_name="dataset", dataset_version="rev", records=records
    )


def test_video_cannot_cross_development_and_holdout() -> None:
    videos = video_manifest(video("v1", "development"), video("v1", "holdout"))

    with pytest.raises(ValueError, match="video_id.*multiple experiment groups"):
        validate_split_isolation(videos, question_manifest())


def test_questions_inherit_their_video_group() -> None:
    with pytest.raises(ValueError, match="question split differs"):
        validate_split_isolation(
            video_manifest(video("v1", "canary")),
            question_manifest(question("q1", "v1", "oracle")),
        )


def test_duplicate_question_ids_are_rejected() -> None:
    videos = video_manifest(video("v1", "canary"))
    questions = question_manifest(
        question("q1", "v1", "canary"),
        question("q1", "v1", "canary"),
    )

    with pytest.raises(ValueError, match="duplicate question_id"):
        validate_split_isolation(videos, questions)


def test_gold_answer_is_rejected_outside_oracle_or_evaluation() -> None:
    with pytest.raises(ValueError, match="gold answer"):
        question("q1", "v1", "canary", gold_answer_sha256="c" * 64)


def test_selection_is_stable_and_does_not_read_gold_answer() -> None:
    videos = video_manifest(video("v1", "oracle"), video("v2", "oracle"))
    without_gold = question_manifest(
        question("q1", "v1", "oracle"),
        question("q2", "v1", "oracle"),
        question("q3", "v2", "oracle"),
    )
    with_gold = question_manifest(
        question(
            "q1",
            "v1",
            "oracle",
            gold_answer_sha256="c" * 64,
            ground_truth_scope="oracle",
        ),
        question("q2", "v1", "oracle"),
        question("q3", "v2", "oracle"),
    )

    first = select_questions_deterministically(
        videos, without_gold, group="oracle", count=2, seed="r002-v1"
    )
    second = select_questions_deterministically(
        videos, with_gold, group="oracle", count=2, seed="r002-v1"
    )

    assert first.question_ids == second.question_ids
    assert len(first.question_ids) == 2
    assert first.selection_sha256 != "0" * 64


def test_selection_uses_structured_canonical_ranking_with_golden_digest() -> None:
    videos = video_manifest(video("v1", "canary"), video("v2", "canary"))
    questions = question_manifest(
        question("q1", "v1", "canary"),
        question("q2", "v2", "canary"),
    )

    selected = select_questions_deterministically(
        videos, questions, group="canary", count=1, seed="r002-v1"
    )

    assert selection_rank_sha256("r002-v1", "v1", "q1") == (
        "c5b7c215ae15a3094475f0f280ea43e5c3c80400f93605e14dba69db04c00164"
    )
    assert selection_rank_sha256("r002-v1", "v2", "q2") == (
        "a5809759bdec40b099ec07350928a19efa56c685cd8366a3398a73ef1bfa2623"
    )
    assert selected.question_ids == ("q2",)
    assert selected.selection_sha256 != "0" * 64
