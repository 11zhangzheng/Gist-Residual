"""Dataset-neutral manifests, split isolation, and deterministic selection."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fidmem.production.authority import canonical_sha256

ExperimentGroup = Literal["development", "canary", "oracle", "holdout"]
GroundTruthScope = Literal["none", "oracle", "evaluation"]
DatasetScope = Literal["PARTIAL_DATASET_PILOT", "FULL_DATASET"]
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class VideoManifestRecord(_FrozenModel):
    video_id: str = Field(min_length=1)
    content_sha256: str = Field(pattern=_SHA256_PATTERN)
    uri: str = Field(min_length=1)
    duration_seconds: float = Field(gt=0)
    group: ExperimentGroup


class QuestionManifestRecord(_FrozenModel):
    question_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    record_sha256: str = Field(pattern=_SHA256_PATTERN)
    question_types: tuple[str, ...] = Field(min_length=1)
    group: ExperimentGroup
    gold_answer_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    ground_truth_scope: GroundTruthScope = "none"

    @model_validator(mode="after")
    def ground_truth_is_explicitly_scoped(self) -> Self:
        if self.gold_answer_sha256 is None and self.ground_truth_scope != "none":
            raise ValueError("ground truth scope requires a gold answer identity")
        if self.gold_answer_sha256 is not None:
            if self.ground_truth_scope not in {"oracle", "evaluation"}:
                raise ValueError("gold answer is allowed only for Oracle/evaluation")
            if self.group == "oracle" and self.ground_truth_scope != "oracle":
                raise ValueError("Oracle questions require Oracle ground truth scope")
            if self.group == "holdout" and self.ground_truth_scope != "evaluation":
                raise ValueError(
                    "holdout questions require evaluation ground truth scope"
                )
            if self.group in {"development", "canary"}:
                raise ValueError("gold answer is forbidden for development/canary")
        return self


class VideoManifest(_FrozenModel):
    schema_version: Literal[1] = 1
    dataset_name: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    records: tuple[VideoManifestRecord, ...]

    @property
    def manifest_sha256(self) -> str:
        return canonical_sha256(self.model_dump(mode="json"))


class QuestionManifest(_FrozenModel):
    schema_version: Literal[1] = 1
    dataset_name: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    records: tuple[QuestionManifestRecord, ...]

    @property
    def manifest_sha256(self) -> str:
        return canonical_sha256(self.model_dump(mode="json"))


class DatasetManifest(_FrozenModel):
    schema_version: Literal[2] = 2
    dataset_name: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    dataset_scope: DatasetScope
    source_metadata_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_archive_index_sha256: str = Field(pattern=_SHA256_PATTERN)
    subset_selection_manifest_sha256: str | None = Field(
        pattern=_SHA256_PATTERN
    )
    selected_video_count: int = Field(ge=0)
    selected_question_count: int = Field(ge=0)
    available_video_count: int = Field(ge=0)
    available_question_count: int = Field(ge=0)
    split_policy_id: str = Field(min_length=1)
    split_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    video_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    question_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def dataset_scope_is_bound_to_provenance(self) -> Self:
        if self.selected_video_count > self.available_video_count:
            raise ValueError("selected video count exceeds available video count")
        if self.selected_question_count > self.available_question_count:
            raise ValueError("selected question count exceeds available question count")
        if (
            self.dataset_scope == "PARTIAL_DATASET_PILOT"
            and self.subset_selection_manifest_sha256 is None
        ):
            raise ValueError("partial dataset requires a subset selection identity")
        if (
            self.dataset_scope == "FULL_DATASET"
            and self.subset_selection_manifest_sha256 is not None
        ):
            raise ValueError("full dataset forbids a subset selection identity")
        return self


class SelectionManifest(_FrozenModel):
    schema_version: Literal[1] = 1
    group: ExperimentGroup
    seed: str = Field(min_length=1)
    source_video_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_question_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    question_ids: tuple[str, ...]
    video_ids: tuple[str, ...]
    selection_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def selection_hash_matches(self) -> Self:
        payload = self.model_dump(mode="json", exclude={"selection_sha256"})
        if canonical_sha256(payload) != self.selection_sha256:
            raise ValueError("selection_sha256 does not match selection content")
        return self


def validate_split_isolation(
    videos: VideoManifest,
    questions: QuestionManifest,
) -> None:
    if (videos.dataset_name, videos.dataset_version) != (
        questions.dataset_name,
        questions.dataset_version,
    ):
        raise ValueError("question/video manifest dataset identity differs")

    assignments: dict[str, set[str]] = {}
    seen_video_rows: set[tuple[str, str]] = set()
    for record in videos.records:
        row = (record.video_id, record.group)
        if row in seen_video_rows:
            raise ValueError(f"duplicate video manifest row: {record.video_id}")
        seen_video_rows.add(row)
        assignments.setdefault(record.video_id, set()).add(record.group)
    for video_id, groups in assignments.items():
        if len(groups) != 1:
            raise ValueError(
                f"video_id {video_id} appears in multiple experiment groups"
            )

    seen_questions: set[str] = set()
    for record in questions.records:
        if record.question_id in seen_questions:
            raise ValueError(f"duplicate question_id: {record.question_id}")
        seen_questions.add(record.question_id)
        groups = assignments.get(record.video_id)
        if groups is None:
            raise ValueError(f"question references unknown video_id: {record.video_id}")
        video_group = next(iter(groups))
        if record.group != video_group:
            raise ValueError(
                f"question split differs from video split: {record.question_id}"
            )


def selection_rank_sha256(seed: str, video_id: str, question_id: str) -> str:
    return canonical_sha256(
        {
            "question_id": question_id,
            "seed": seed,
            "video_id": video_id,
        }
    )


def select_questions_deterministically(
    videos: VideoManifest,
    questions: QuestionManifest,
    *,
    group: ExperimentGroup,
    count: int,
    seed: str,
) -> SelectionManifest:
    if count <= 0:
        raise ValueError("selection count must be positive")
    if not seed:
        raise ValueError("selection seed must not be blank")
    validate_split_isolation(videos, questions)
    eligible = [record for record in questions.records if record.group == group]
    if count > len(eligible):
        raise ValueError("selection count exceeds eligible questions")

    def rank(record: QuestionManifestRecord) -> tuple[str, str, str]:
        digest = selection_rank_sha256(seed, record.video_id, record.question_id)
        return digest, record.video_id, record.question_id

    chosen = sorted(eligible, key=rank)[:count]
    payload = {
        "schema_version": 1,
        "group": group,
        "seed": seed,
        "source_video_manifest_sha256": videos.manifest_sha256,
        "source_question_manifest_sha256": questions.manifest_sha256,
        "question_ids": tuple(record.question_id for record in chosen),
        "video_ids": tuple(sorted({record.video_id for record in chosen})),
    }
    return SelectionManifest(
        **payload,
        selection_sha256=canonical_sha256(payload),
    )
