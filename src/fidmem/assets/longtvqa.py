"""LongTVQA metadata, approved-video, Source Gate, and manifest preparation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field

from fidmem.data.video import VideoProbe, probe_video, sample_frames
from fidmem.production.authority import canonical_json_bytes, canonical_sha256
from fidmem.production.manifests import (
    DatasetManifest,
    QuestionManifest,
    QuestionManifestRecord,
    SelectionManifest,
    VideoManifest,
    VideoManifestRecord,
    select_questions_deterministically,
    validate_split_isolation,
)

DATASET_ID = "longvideoagent/LongTVQA"
METADATA_FILES = (
    "LongTVQA_train.jsonl",
    "LongTVQA_val.jsonl",
    "LongTVQA_subtitles_clip_level.jsonl",
    "LongTVQA_subtitles_episode_level.jsonl",
    "README.md",
)
VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".avi"}


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class MetadataVerificationReport(_FrozenModel):
    schema_version: Literal[1] = 1
    evidence_class: Literal["engineering"] = "engineering"
    dataset_id: Literal[DATASET_ID] = DATASET_ID
    immutable_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    metadata_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    question_count: int = Field(gt=0)
    video_count: int = Field(gt=0)
    clip_subtitle_video_count: int = Field(gt=0)
    episode_subtitle_video_count: int = Field(gt=0)
    qa_constructible_count: int = Field(ge=0)
    qa_unconstructible_count: int = Field(ge=0)
    files: tuple[str, ...]
    status: Literal["VERIFIED"] = "VERIFIED"


class HumanAuditItem(_FrozenModel):
    question_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    source_split: str = Field(min_length=1)
    source_timestamp: object | None = None


class HumanAuditManifest(_FrozenModel):
    schema_version: Literal[1] = 1
    evidence_class: Literal["engineering"] = "engineering"
    dataset_id: Literal[DATASET_ID] = DATASET_ID
    seed: str = Field(min_length=1)
    required_items: int = Field(ge=100)
    items: tuple[HumanAuditItem, ...] = Field(min_length=100)
    status: Literal["PENDING_HUMAN_AUDIT"] = "PENDING_HUMAN_AUDIT"

    @property
    def manifest_sha256(self) -> str:
        return canonical_sha256(self.model_dump(mode="json"))


class RawVideoVerificationReport(_FrozenModel):
    schema_version: Literal[1] = 1
    evidence_class: Literal["engineering"] = "engineering"
    dataset_id: Literal[DATASET_ID] = DATASET_ID
    approved_video_root: str = Field(min_length=1)
    expected_video_count: int = Field(gt=0)
    verified_video_count: int = Field(ge=0)
    random_decode_required: int = Field(ge=20)
    random_decode_completed: int = Field(ge=0)
    missing_video_ids: tuple[str, ...]
    corrupt_video_ids: tuple[str, ...]
    duplicate_video_ids: tuple[str, ...]
    missing_qa_mapping: tuple[str, ...]
    missing_subtitle_mapping: tuple[str, ...]
    video_records: tuple[VideoManifestRecord, ...]
    status: Literal["PASS", "FAIL"]


@dataclass(frozen=True)
class ParsedLongTVQA:
    questions: tuple[dict[str, Any], ...]
    clip_video_ids: frozenset[str]
    episode_video_ids: frozenset[str]
    report: MetadataVerificationReport


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path.name}:{line_number} must be an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"metadata file is empty: {path.name}")
    return rows


def _first(row: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return None


def question_id(row: dict[str, Any]) -> str:
    value = _first(row, ("question_id", "qid", "id", "uid"))
    if value is None:
        raise ValueError("LongTVQA question lacks a stable question ID")
    return str(value)


def video_id(row: dict[str, Any]) -> str:
    value = _first(
        row,
        ("video_id", "video", "episode_id", "episode", "vid", "show_episode"),
    )
    if value is None:
        raise ValueError("LongTVQA record lacks an episode/video ID")
    return Path(str(value)).stem


def _subtitle_ids(rows: list[dict[str, Any]]) -> frozenset[str]:
    return frozenset(video_id(row) for row in rows)


def _qa_constructible(row: dict[str, Any]) -> bool:
    prompt = _first(row, ("question", "query", "prompt"))
    answer = _first(row, ("answer", "correct_answer", "label"))
    options = _first(row, ("options", "choices", "candidates"))
    if options is None:
        options = [row.get(key) for key in ("A", "B", "C", "D") if row.get(key)]
    return bool(prompt and answer is not None and options)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_metadata(root: str | Path, *, immutable_revision: str) -> ParsedLongTVQA:
    directory = Path(root)
    missing = [name for name in METADATA_FILES if not (directory / name).is_file()]
    if missing:
        raise ValueError(f"LongTVQA metadata snapshot is incomplete: {missing}")
    questions: list[dict[str, Any]] = []
    for split, filename in (
        ("train", "LongTVQA_train.jsonl"),
        ("val", "LongTVQA_val.jsonl"),
    ):
        for row in _read_jsonl(directory / filename):
            questions.append({**row, "_source_split": split})
    ids = [question_id(row) for row in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("LongTVQA metadata contains duplicate question IDs")
    clip_ids = _subtitle_ids(
        _read_jsonl(directory / "LongTVQA_subtitles_clip_level.jsonl")
    )
    episode_ids = _subtitle_ids(
        _read_jsonl(directory / "LongTVQA_subtitles_episode_level.jsonl")
    )
    videos = frozenset(video_id(row) for row in questions)
    report = MetadataVerificationReport(
        immutable_revision=immutable_revision,
        metadata_sha256=canonical_sha256(
            [
                {"path": name, "sha256": _file_sha256(directory / name)}
                for name in METADATA_FILES
            ]
        ),
        question_count=len(questions),
        video_count=len(videos),
        clip_subtitle_video_count=len(clip_ids),
        episode_subtitle_video_count=len(episode_ids),
        qa_constructible_count=sum(_qa_constructible(row) for row in questions),
        qa_unconstructible_count=sum(not _qa_constructible(row) for row in questions),
        files=METADATA_FILES,
    )
    return ParsedLongTVQA(tuple(questions), clip_ids, episode_ids, report)


def build_human_audit_manifest(
    metadata: ParsedLongTVQA,
    *,
    seed: str,
    count: int = 100,
) -> HumanAuditManifest:
    if count < 100:
        raise ValueError("human timestamp audit requires at least 100 items")
    if len(metadata.questions) < count:
        raise ValueError("not enough LongTVQA questions for the human audit")
    ranked = sorted(
        metadata.questions,
        key=lambda row: (
            canonical_sha256(
                {
                    "seed": seed,
                    "video_id": video_id(row),
                    "question_id": question_id(row),
                }
            ),
            question_id(row),
        ),
    )[:count]
    return HumanAuditManifest(
        seed=seed,
        required_items=count,
        items=tuple(
            HumanAuditItem(
                question_id=question_id(row),
                video_id=video_id(row),
                source_split=str(row["_source_split"]),
                source_timestamp=_first(
                    row,
                    ("timestamp", "timestamps", "time", "start_end", "relevant_window"),
                ),
            )
            for row in ranked
        ),
    )


def validate_human_audit_result(
    manifest: HumanAuditManifest, result_path: str | Path
) -> None:
    path = Path(result_path)
    if not path.is_file():
        raise ValueError("human timestamp audit result is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "COMPLETED":
        raise ValueError("human timestamp audit is not COMPLETED")
    if payload.get("human_audit_manifest_sha256") != manifest.manifest_sha256:
        raise ValueError("human audit result does not bind the pending manifest")
    if not str(payload.get("reviewer_identity", "")).strip():
        raise ValueError("human audit result lacks reviewer identity")
    if not str(payload.get("completed_at", "")).strip():
        raise ValueError("human audit result lacks completion timestamp")
    items = payload.get("items")
    if not isinstance(items, list) or len(items) < manifest.required_items:
        raise ValueError("human timestamp audit has fewer than 100 completed items")
    expected = {item.question_id for item in manifest.items}
    observed = {
        str(item.get("question_id")) for item in items if isinstance(item, dict)
    }
    if observed != expected:
        raise ValueError("human audit completed-item identities differ from manifest")
    if any(item.get("outcome") != "PASS" for item in items):
        raise ValueError("human timestamp audit contains a non-PASS outcome")


def _default_decode(path: Path, timestamp: float) -> None:
    with tempfile.TemporaryDirectory(prefix="fidmem-longtvqa-decode-") as directory:
        sample_frames(path, (timestamp,), directory)


def verify_raw_videos(
    metadata: ParsedLongTVQA,
    approved_video_root: str | Path,
    *,
    random_decode_required: int = 20,
    decode_seed: str = "longtvqa-source-gate-v1",
    probe: Callable[[str | Path], VideoProbe] = probe_video,
    decode: Callable[[Path, float], None] = _default_decode,
) -> RawVideoVerificationReport:
    if random_decode_required < 20:
        raise ValueError("Source Gate requires random decode of at least 20 episodes")
    root = Path(approved_video_root).resolve()
    if not root.is_dir():
        raise ValueError(f"approved LongTVQA video root is missing: {root}")
    by_stem: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
            by_stem.setdefault(path.stem, []).append(path)
    expected = sorted({video_id(row) for row in metadata.questions})
    duplicate_ids = {key for key, paths in by_stem.items() if len(paths) > 1}
    missing = tuple(video for video in expected if video not in by_stem)
    probes: dict[str, VideoProbe] = {}
    corrupt: list[str] = []
    records: list[VideoManifestRecord] = []
    for identity in expected:
        paths = by_stem.get(identity, [])
        if len(paths) != 1:
            continue
        path = paths[0]
        try:
            observed = probe(path)
            if observed.duration_sec <= 0:
                raise ValueError("non-positive duration")
            probes[identity] = observed
            records.append(
                VideoManifestRecord(
                    video_id=identity,
                    content_sha256=_file_sha256(path),
                    uri=str(path),
                    duration_seconds=observed.duration_sec,
                    group="development",
                )
            )
        except (OSError, RuntimeError, ValueError):
            corrupt.append(identity)
    identities_by_hash: dict[str, list[str]] = {}
    for record in records:
        identities_by_hash.setdefault(record.content_sha256, []).append(record.video_id)
    for identities in identities_by_hash.values():
        if len(identities) > 1:
            duplicate_ids.update(identities)
    duplicate = tuple(sorted(duplicate_ids))
    ranked = sorted(
        probes,
        key=lambda identity: (
            canonical_sha256({"seed": decode_seed, "video_id": identity}),
            identity,
        ),
    )[:random_decode_required]
    decoded = 0
    for identity in ranked:
        try:
            decode(Path(probes[identity].path), probes[identity].duration_sec / 2)
            decoded += 1
        except (OSError, RuntimeError, ValueError):
            corrupt.append(identity)
    missing_subtitles = tuple(
        sorted(
            identity
            for identity in expected
            if identity not in metadata.clip_video_ids
            and identity not in metadata.episode_video_ids
        )
    )
    passed = (
        not (missing or corrupt or duplicate or missing_subtitles)
        and decoded >= random_decode_required
    )
    return RawVideoVerificationReport(
        approved_video_root=str(root),
        expected_video_count=len(expected),
        verified_video_count=len(records),
        random_decode_required=random_decode_required,
        random_decode_completed=decoded,
        missing_video_ids=missing,
        corrupt_video_ids=tuple(sorted(set(corrupt))),
        duplicate_video_ids=duplicate,
        missing_qa_mapping=(),
        missing_subtitle_mapping=missing_subtitles,
        video_records=tuple(sorted(records, key=lambda row: row.video_id)),
        status="PASS" if passed else "FAIL",
    )


def build_manifests(
    metadata: ParsedLongTVQA,
    videos: RawVideoVerificationReport,
    *,
    split_policy_path: str | Path,
) -> tuple[
    VideoManifest,
    QuestionManifest,
    DatasetManifest,
    SelectionManifest,
    SelectionManifest,
]:
    if videos.status != "PASS":
        raise ValueError("raw-video Source Gate must PASS before manifest construction")
    policy_path = Path(split_policy_path)
    raw = OmegaConf.to_container(OmegaConf.load(policy_path), resolve=True)
    if not isinstance(raw, dict) or raw.get("status") != "FROZEN":
        raise ValueError("LongTVQA split policy is RESEARCH_OWNER_DECISION_REQUIRED")
    assignments = raw.get("video_groups")
    if not isinstance(assignments, dict):
        raise ValueError("frozen split policy requires explicit video_groups")
    expected = {record.video_id for record in videos.video_records}
    if set(assignments) != expected:
        raise ValueError("split policy must assign every and only verified video_id")
    video_manifest = VideoManifest(
        dataset_name=DATASET_ID,
        dataset_version=metadata.report.immutable_revision,
        records=tuple(
            record.model_copy(update={"group": str(assignments[record.video_id])})
            for record in videos.video_records
        ),
    )
    group_by_video = {
        record.video_id: record.group for record in video_manifest.records
    }
    questions: list[QuestionManifestRecord] = []
    for row in metadata.questions:
        identity = video_id(row)
        group = group_by_video[identity]
        answer = _first(row, ("answer", "correct_answer", "label"))
        gold_hash = None
        scope = "none"
        if group in {"oracle", "holdout"}:
            if answer is None:
                raise ValueError(
                    f"question {question_id(row)} lacks required gold answer"
                )
            gold_hash = hashlib.sha256(str(answer).encode("utf-8")).hexdigest()
            scope = "oracle" if group == "oracle" else "evaluation"
        source_types = _first(
            row, ("question_types", "question_type", "type", "category")
        )
        types = (
            tuple(str(item) for item in source_types)
            if isinstance(source_types, list)
            else (str(source_types or "longtvqa-unlabeled"),)
        )
        questions.append(
            QuestionManifestRecord(
                question_id=question_id(row),
                video_id=identity,
                record_sha256=canonical_sha256(row),
                question_types=types,
                group=group,
                gold_answer_sha256=gold_hash,
                ground_truth_scope=scope,
            )
        )
    question_manifest = QuestionManifest(
        dataset_name=DATASET_ID,
        dataset_version=metadata.report.immutable_revision,
        records=tuple(questions),
    )
    validate_split_isolation(video_manifest, question_manifest)
    policy_sha = _file_sha256(policy_path)
    canary_count = int(raw["selections"]["canary"]["count"])
    oracle_count = int(raw["selections"]["oracle"]["count"])
    if not 10 <= canary_count <= 20:
        raise ValueError("production canary selection must contain 10-20 questions")
    if oracle_count != 100:
        raise ValueError(
            "production Oracle selection must contain exactly 100 questions"
        )
    dataset_manifest = DatasetManifest(
        dataset_name=DATASET_ID,
        dataset_version=metadata.report.immutable_revision,
        split_policy_id=str(raw["split_policy_id"]),
        split_policy_sha256=policy_sha,
        video_manifest_sha256=video_manifest.manifest_sha256,
        question_manifest_sha256=question_manifest.manifest_sha256,
    )
    canary = select_questions_deterministically(
        video_manifest,
        question_manifest,
        group="canary",
        count=canary_count,
        seed=str(raw["selections"]["canary"]["seed"]),
    )
    oracle = select_questions_deterministically(
        video_manifest,
        question_manifest,
        group="oracle",
        count=oracle_count,
        seed=str(raw["selections"]["oracle"]["seed"]),
    )
    return video_manifest, question_manifest, dataset_manifest, canary, oracle


def atomic_write_model(path: str | Path, value: BaseModel) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json_bytes(value.model_dump(mode="json")))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
