"""Pinned Video-MME-v2 metadata and subtitle verification helpers."""

from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal, Self
from zipfile import BadZipFile, ZipFile

import duckdb
from pydantic import BaseModel, ConfigDict, Field, model_validator

from fidmem.production.authority import canonical_sha256


DATASET_ID = "MME-Benchmarks/Video-MME-v2"
FROZEN_REVISION = "6e4bebb03202e1ddbf3d37703e560e51c5aa2d64"
METADATA_FILES = ("README.md", "subtitle.zip", "test.parquet")
_EXPECTED_COLUMNS = (
    "video_id",
    "url",
    "group_type",
    "group_structure",
    "question_id",
    "question",
    "options",
    "answer",
    "level",
    "second_head",
    "third_head",
)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class MetadataFileIdentity(_FrozenModel):
    path: str = Field(min_length=1)
    size: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class VideoMMEQuestion(_FrozenModel):
    video_id: str = Field(pattern=r"^\d{3}$")
    url: str = Field(min_length=1)
    group_type: str = ""
    group_structure: str = ""
    question_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    options: tuple[str, ...] = Field(min_length=1)
    answer: str = Field(min_length=1)
    level: str = ""
    second_head: str = ""
    third_head: str = ""
    question_types: tuple[str, ...] = Field(min_length=1)


class MetadataVerificationReport(_FrozenModel):
    schema_version: Literal[1] = 1
    evidence_class: Literal["engineering"] = "engineering"
    dataset_id: Literal[DATASET_ID] = DATASET_ID
    immutable_revision: Literal[FROZEN_REVISION] = FROZEN_REVISION
    metadata_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    question_count: int = Field(gt=0)
    video_count: int = Field(gt=0)
    files: tuple[str, ...]
    file_identities: tuple[MetadataFileIdentity, ...]
    status: Literal["VERIFIED"] = "VERIFIED"


class ParsedVideoMME(_FrozenModel):
    questions: tuple[VideoMMEQuestion, ...]
    video_ids: tuple[str, ...]
    report: MetadataVerificationReport


class HumanAuditItem(_FrozenModel):
    question_id: str = Field(min_length=1)
    video_id: str = Field(pattern=r"^\d{3}$")
    status: Literal["PENDING_HUMAN_AUDIT"] = "PENDING_HUMAN_AUDIT"


class HumanAuditManifest(_FrozenModel):
    schema_version: Literal[1] = 1
    evidence_class: Literal["engineering"] = "engineering"
    dataset_id: Literal[DATASET_ID] = DATASET_ID
    source_metadata_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: str = Field(min_length=1)
    required_items: Literal[100] = 100
    items: tuple[HumanAuditItem, ...] = Field(min_length=100, max_length=100)
    status: Literal["PENDING_HUMAN_AUDIT"] = "PENDING_HUMAN_AUDIT"

    @model_validator(mode="after")
    def question_ids_are_unique(self) -> Self:
        if len({item.question_id for item in self.items}) != self.required_items:
            raise ValueError("Video-MME-v2 human audit requires 100 unique question IDs")
        return self

    @property
    def manifest_sha256(self) -> str:
        return canonical_sha256(self.model_dump(mode="json"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nonblank(value: Any, *, field: str) -> str:
    if value is None or not str(value).strip():
        raise ValueError(f"Video-MME-v2 {field} must be nonblank")
    return str(value).strip()


def _parse_options(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("Video-MME-v2 options must be nonblank")
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                value = (text,)
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("Video-MME-v2 options must be a nonempty sequence")
    options = tuple(_nonblank(option, field="option") for option in value)
    return options


def _question_types(row: dict[str, Any]) -> tuple[str, ...]:
    labels = tuple(
        str(row[name]).strip()
        for name in ("group_type", "level", "second_head", "third_head")
        if row[name] is not None and str(row[name]).strip()
    )
    return labels or ("videomme-v2-unlabeled",)


def _read_questions(parquet_path: Path) -> tuple[dict[str, Any], ...]:
    connection = duckdb.connect()
    try:
        cursor = connection.execute(
            "SELECT * FROM read_parquet(?)", [str(parquet_path)]
        )
        columns = tuple(item[0] for item in cursor.description)
        if columns != _EXPECTED_COLUMNS:
            raise ValueError(
                "Video-MME-v2 Parquet columns differ from the official snapshot"
            )
        return tuple(
            dict(zip(columns, values, strict=True)) for values in cursor.fetchall()
        )
    finally:
        connection.close()


def _verify_subtitles(path: Path, video_ids: set[str]) -> None:
    try:
        with ZipFile(path) as archive:
            infos = tuple(archive.infolist())
    except BadZipFile as error:
        raise ValueError("Video-MME-v2 subtitle ZIP is invalid") from error
    names = tuple(info.filename for info in infos)
    expected = tuple(f"{video_id}.jsonl" for video_id in sorted(video_ids))
    if (
        len(names) != len(set(names))
        or tuple(sorted(names)) != expected
        or any("/" in name or "\\" in name for name in names)
    ):
        raise ValueError("Video-MME-v2 subtitle/video IDs differ")


def verify_metadata(
    root: str | Path,
    *,
    immutable_revision: str,
    _expected_question_count: int = 3200,
    _expected_video_count: int = 800,
) -> ParsedVideoMME:
    """Parse and fail-closed verify the pinned metadata-only snapshot."""
    if immutable_revision != FROZEN_REVISION:
        raise ValueError("Video-MME-v2 immutable revision differs from frozen immutable revision")
    directory = Path(root)
    actual_names = {path.name for path in directory.iterdir()} if directory.is_dir() else set()
    expected_names = set(METADATA_FILES)
    if actual_names - expected_names:
        raise ValueError("Video-MME-v2 metadata contains unexpected metadata files")
    missing = [name for name in METADATA_FILES if not (directory / name).is_file()]
    if missing:
        raise ValueError(f"Video-MME-v2 metadata snapshot is incomplete: {missing}")

    rows = _read_questions(directory / "test.parquet")
    questions = tuple(
        VideoMMEQuestion(
            video_id=_nonblank(row["video_id"], field="video_id"),
            url=_nonblank(row["url"], field="url"),
            group_type=str(row["group_type"] or "").strip(),
            group_structure=str(row["group_structure"] or "").strip(),
            question_id=_nonblank(row["question_id"], field="question_id"),
            question=_nonblank(row["question"], field="question"),
            options=_parse_options(row["options"]),
            answer=_nonblank(row["answer"], field="answer"),
            level=str(row["level"] or "").strip(),
            second_head=str(row["second_head"] or "").strip(),
            third_head=str(row["third_head"] or "").strip(),
            question_types=_question_types(row),
        )
        for row in rows
    )
    question_ids = tuple(question.question_id for question in questions)
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Video-MME-v2 metadata contains duplicate question IDs")
    video_ids = tuple(sorted({question.video_id for question in questions}))
    counts = Counter(question.video_id for question in questions)
    if any(count != 4 for count in counts.values()):
        raise ValueError("Video-MME-v2 every video must have exactly four questions")
    if len(rows) != _expected_question_count:
        raise ValueError("Video-MME-v2 question count differs from the official snapshot")
    if len(video_ids) != _expected_video_count:
        raise ValueError("Video-MME-v2 video count differs from the official snapshot")
    _verify_subtitles(directory / "subtitle.zip", set(video_ids))
    file_identities = tuple(
        MetadataFileIdentity(
            path=name,
            size=(directory / name).stat().st_size,
            sha256=_file_sha256(directory / name),
        )
        for name in METADATA_FILES
    )
    report = MetadataVerificationReport(
        metadata_sha256=canonical_sha256(
            [identity.model_dump(mode="json") for identity in file_identities]
        ),
        question_count=len(questions),
        video_count=len(video_ids),
        files=METADATA_FILES,
        file_identities=file_identities,
    )
    return ParsedVideoMME(questions=questions, video_ids=video_ids, report=report)


def build_human_audit_manifest(
    metadata: ParsedVideoMME,
    selected_video_ids: tuple[str, ...],
    seed: str,
    count: int = 100,
) -> HumanAuditManifest:
    """Build the deterministic, pending-only 100-question human audit."""
    if count != 100:
        raise ValueError("Video-MME-v2 human audit requires exactly 100 items")
    selected = tuple(selected_video_ids)
    if len(selected) != len(set(selected)):
        raise ValueError("selected video IDs must be unique")
    known = set(metadata.video_ids)
    if unknown := set(selected) - known:
        raise ValueError(f"selected video IDs are absent from metadata: {sorted(unknown)}")
    ranked = sorted(
        (question for question in metadata.questions if question.video_id in selected),
        key=lambda question: (
            canonical_sha256(
                {
                    "seed": seed,
                    "video_id": question.video_id,
                    "question_id": question.question_id,
                }
            ),
            question.question_id,
        ),
    )
    if len(ranked) < count:
        raise ValueError("not enough selected Video-MME-v2 questions for human audit")
    return HumanAuditManifest(
        source_metadata_sha256=metadata.report.metadata_sha256,
        seed=seed,
        items=tuple(
            HumanAuditItem(question_id=question.question_id, video_id=question.video_id)
            for question in ranked[:count]
        ),
    )


def validate_human_audit_result(
    manifest: HumanAuditManifest, result_path: str | Path
) -> None:
    """Validate a completed audit result without modifying it."""
    path = Path(result_path)
    if not path.is_file():
        raise ValueError("Video-MME-v2 human audit result is missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("Video-MME-v2 human audit result is invalid JSON") from error
    if not isinstance(payload, dict) or payload.get("status") != "COMPLETED":
        raise ValueError("Video-MME-v2 human audit is not COMPLETED")
    if payload.get("human_audit_manifest_sha256") != manifest.manifest_sha256:
        raise ValueError("Video-MME-v2 human audit result does not bind the pending manifest")
    if not str(payload.get("reviewer_identity", "")).strip():
        raise ValueError("Video-MME-v2 human audit result lacks reviewer identity")
    if not str(payload.get("completed_at", "")).strip():
        raise ValueError("Video-MME-v2 human audit result lacks completion identity")
    items = payload.get("items")
    if not isinstance(items, list) or len(items) != manifest.required_items:
        raise ValueError("Video-MME-v2 human audit has the wrong completed item count")
    if any(not isinstance(item, dict) for item in items):
        raise ValueError("Video-MME-v2 human audit items must be objects")
    observed_ids = tuple(str(item.get("question_id", "")) for item in items)
    if len(observed_ids) != len(set(observed_ids)):
        raise ValueError("Video-MME-v2 human audit result contains duplicate question IDs")
    expected = {item.question_id for item in manifest.items}
    if set(observed_ids) != expected:
        raise ValueError("Video-MME-v2 human audit completed-item identities differ from manifest")
    if any(item.get("outcome") != "PASS" for item in items):
        raise ValueError("Video-MME-v2 human audit contains a non-PASS outcome")
