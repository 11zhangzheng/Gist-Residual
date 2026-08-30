"""Pinned Video-MME-v2 metadata and subtitle verification helpers."""

from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter
from collections.abc import Callable
from contextlib import closing
from pathlib import PurePosixPath
from pathlib import Path
import re
from typing import Any, BinaryIO, Literal, Self
from zipfile import BadZipFile, ZipFile, ZipInfo

import duckdb
from pydantic import BaseModel, ConfigDict, Field, model_validator

from fidmem.production.authority import canonical_sha256


DATASET_ID = "MME-Benchmarks/Video-MME-v2"
FROZEN_REVISION = "6e4bebb03202e1ddbf3d37703e560e51c5aa2d64"
METADATA_FILES = ("README.md", "subtitle.zip", "test.parquet")
OFFICIAL_ARCHIVE_PATHS = tuple(f"videos/{number:03d}.zip" for number in range(1, 41))
OFFICIAL_VIDEO_IDS = tuple(f"{number:03d}" for number in range(800))
POOL_SEED = "videomme-v2-partial-pilot-pool-v1"
POOL_ALGORITHM = "videomme-v2-archive-aware-hash-v1"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
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


class OfficialFileIdentity(_FrozenModel):
    """Pinned identity of one official Video-MME-v2 ZIP sibling."""

    path: str = Field(min_length=1)
    size: int = Field(gt=0)
    upstream_sha256: str = Field(pattern=_SHA256_PATTERN)


class ArchiveMemberIdentity(_FrozenModel):
    """A media member observed from a ZIP central directory only."""

    archive_path: str = Field(min_length=1)
    video_id: str = Field(pattern=r"^\d{3}$")
    member_path: str = Field(min_length=1)
    crc32: int = Field(ge=0, le=0xFFFFFFFF)
    compressed_size: int = Field(ge=0)
    uncompressed_size: int = Field(gt=0)


class ArchiveIndex(_FrozenModel):
    """Canonical, hash-bound inventory of official archive central directories."""

    schema_version: Literal[1] = 1
    dataset_id: Literal[DATASET_ID] = DATASET_ID
    immutable_revision: Literal[FROZEN_REVISION] = FROZEN_REVISION
    archives: tuple[OfficialFileIdentity, ...] = Field(min_length=1)
    members: tuple[ArchiveMemberIdentity, ...] = Field(min_length=1)
    archive_index_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def canonical_contents_and_hash_match(self) -> Self:
        archive_paths = tuple(archive.path for archive in self.archives)
        if archive_paths != tuple(sorted(archive_paths)) or len(archive_paths) != len(
            set(archive_paths)
        ):
            raise ValueError("archive index archive paths must be unique and canonical")
        member_keys = tuple(
            (member.archive_path, member.member_path) for member in self.members
        )
        if member_keys != tuple(sorted(member_keys)) or len(member_keys) != len(
            set(member_keys)
        ):
            raise ValueError("archive index members must be unique and canonical")
        video_ids = tuple(member.video_id for member in self.members)
        if len(video_ids) != len(set(video_ids)):
            raise ValueError("archive index contains duplicate video IDs")
        payload = self.model_dump(mode="json", exclude={"archive_index_sha256"})
        if canonical_sha256(payload) != self.archive_index_sha256:
            raise ValueError("archive_index_sha256 does not match archive index content")
        return self

    @property
    def video_ids(self) -> tuple[str, ...]:
        return tuple(member.video_id for member in self.members)


class PilotSelectionManifest(_FrozenModel):
    """Question-independent deterministic archive-aware Video-MME-v2 pilot."""

    schema_version: Literal[1] = 1
    dataset_id: Literal[DATASET_ID] = DATASET_ID
    immutable_revision: Literal[FROZEN_REVISION] = FROZEN_REVISION
    source_metadata_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_archive_index_sha256: str = Field(pattern=_SHA256_PATTERN)
    pool_seed: Literal[POOL_SEED] = POOL_SEED
    pool_algorithm: Literal[POOL_ALGORITHM] = POOL_ALGORITHM
    available_video_count: int = Field(gt=0)
    selected_video_count: int = Field(gt=0)
    selected_archive_paths: tuple[str, ...] = Field(min_length=1)
    selected_video_ids: tuple[str, ...] = Field(min_length=1)
    selection_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def selection_is_canonical_and_hash_bound(self) -> Self:
        if self.selected_video_count != len(self.selected_video_ids):
            raise ValueError("selected video count differs from selected video IDs")
        if self.selected_video_count > self.available_video_count:
            raise ValueError("selected video count exceeds available video count")
        if self.selected_archive_paths != tuple(sorted(self.selected_archive_paths)):
            raise ValueError("selected archive paths must be canonical")
        if len(self.selected_video_ids) != len(set(self.selected_video_ids)):
            raise ValueError("selected video IDs must be unique")
        payload = self.model_dump(mode="json", exclude={"selection_sha256"})
        if canonical_sha256(payload) != self.selection_sha256:
            raise ValueError("selection_sha256 does not match pilot selection content")
        return self


def _validate_official_archive_identity(
    identity: OfficialFileIdentity,
    *,
    expected_archive_paths: tuple[str, ...],
) -> None:
    if identity.path not in expected_archive_paths:
        raise ValueError("Video-MME-v2 archive path is not an official pinned archive")
    if identity.size <= 0:
        raise ValueError("Video-MME-v2 official archive size must be positive")
    if not re.fullmatch(_SHA256_PATTERN, identity.upstream_sha256):
        raise ValueError("Video-MME-v2 official archive lacks a lowercase upstream SHA-256")


def _safe_media_member(info: ZipInfo, identity: OfficialFileIdentity) -> ArchiveMemberIdentity:
    member_path = PurePosixPath(info.filename)
    if (
        not info.filename
        or "\\" in info.filename
        or member_path.is_absolute()
        or ".." in member_path.parts
        or member_path.parent != PurePosixPath(".")
    ):
        raise ValueError("unsafe ZIP member path")
    if info.is_dir() or member_path.suffix != ".mp4":
        raise ValueError("non-MP4 ZIP member")
    if not re.fullmatch(r"\d{3}", member_path.stem):
        raise ValueError("Video-MME-v2 ZIP member has an invalid video ID")
    # POSIX mode type bits identify symlinks without ever opening a member payload.
    if (info.external_attr >> 16) & 0o170000 == 0o120000:
        raise ValueError("unsafe ZIP member symlink")
    return ArchiveMemberIdentity(
        archive_path=identity.path,
        video_id=member_path.stem,
        member_path=info.filename,
        crc32=info.CRC,
        compressed_size=info.compress_size,
        uncompressed_size=info.file_size,
    )


def inspect_zip_central_directory(
    stream: BinaryIO, official_identity: OfficialFileIdentity
) -> tuple[ArchiveMemberIdentity, ...]:
    """Read only a seekable ZIP central directory, never an MP4 payload."""
    try:
        stream.seek(0, 2)
        observed_size = stream.tell()
        stream.seek(0)
        if observed_size != official_identity.size:
            raise ValueError("Video-MME-v2 archive size differs from official identity")
        with ZipFile(stream) as archive:
            members = tuple(
                _safe_media_member(info, official_identity) for info in archive.infolist()
            )
    except BadZipFile as error:
        raise ValueError("Video-MME-v2 official archive is not a valid ZIP") from error
    if not members:
        raise ValueError("Video-MME-v2 official archive has no MP4 members")
    return tuple(sorted(members, key=lambda member: member.member_path))


def _validate_metadata_coverage(
    metadata_video_ids: tuple[str, ...], archive_video_ids: tuple[str, ...]
) -> None:
    metadata_ids = tuple(metadata_video_ids)
    if len(metadata_ids) != len(set(metadata_ids)):
        raise ValueError("Video-MME-v2 metadata contains duplicate video IDs")
    metadata_set = set(metadata_ids)
    archive_set = set(archive_video_ids)
    if unexpected := sorted(archive_set - metadata_set):
        raise ValueError(
            f"Video-MME-v2 archive video IDs are absent from metadata: {unexpected}"
        )
    if missing := sorted(metadata_set - archive_set):
        raise ValueError(
            f"Video-MME-v2 metadata video IDs are missing from archives: {missing}"
        )


def build_archive_index(
    file_identities: tuple[OfficialFileIdentity, ...],
    opener: Callable[[OfficialFileIdentity], BinaryIO],
    *,
    metadata_video_ids: tuple[str, ...] | None = None,
    _expected_archive_paths: tuple[str, ...] = OFFICIAL_ARCHIVE_PATHS,
) -> ArchiveIndex:
    """Build a canonical official archive index through injected seekable streams."""
    expected_paths = tuple(_expected_archive_paths)
    identities = tuple(sorted(file_identities, key=lambda identity: identity.path))
    if tuple(identity.path for identity in identities) != expected_paths:
        raise ValueError("Video-MME-v2 archive siblings differ from the pinned official set")
    for identity in identities:
        _validate_official_archive_identity(
            identity, expected_archive_paths=expected_paths
        )
    members: list[ArchiveMemberIdentity] = []
    for identity in identities:
        with closing(opener(identity)) as stream:
            members.extend(inspect_zip_central_directory(stream, identity))
    members_tuple = tuple(
        sorted(members, key=lambda member: (member.archive_path, member.member_path))
    )
    video_ids = tuple(member.video_id for member in members_tuple)
    if len(video_ids) != len(set(video_ids)):
        raise ValueError("Video-MME-v2 archive index contains duplicate video IDs")
    if metadata_video_ids is not None:
        _validate_metadata_coverage(metadata_video_ids, video_ids)
    elif expected_paths == OFFICIAL_ARCHIVE_PATHS:
        _validate_metadata_coverage(
            tuple(f"{number:03d}" for number in range(800)), video_ids
        )
    payload = {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "immutable_revision": FROZEN_REVISION,
        "archives": [identity.model_dump(mode="json") for identity in identities],
        "members": [member.model_dump(mode="json") for member in members_tuple],
    }
    return ArchiveIndex(
        **payload,
        archive_index_sha256=canonical_sha256(payload),
    )


def load_official_archive_identities() -> tuple[OfficialFileIdentity, ...]:
    """Load the exact pinned remote ZIP sibling identities, without downloading them."""
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise RuntimeError("huggingface_hub is required for Video-MME-v2 archives") from error
    info = HfApi().dataset_info(
        repo_id=DATASET_ID,
        revision=FROZEN_REVISION,
        files_metadata=True,
    )
    siblings = {str(item.rfilename): item for item in (info.siblings or ())}
    if set(OFFICIAL_ARCHIVE_PATHS) - set(siblings):
        raise ValueError("Video-MME-v2 official archive siblings are missing")
    archive_siblings = {
        path
        for path in siblings
        if path.startswith("videos/") and path.endswith(".zip")
    }
    if archive_siblings - set(OFFICIAL_ARCHIVE_PATHS):
        raise ValueError("Video-MME-v2 remote contains unexpected official archive siblings")
    identities: list[OfficialFileIdentity] = []
    for path in OFFICIAL_ARCHIVE_PATHS:
        sibling = siblings[path]
        lfs = getattr(sibling, "lfs", None)
        sha256 = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        try:
            identity = OfficialFileIdentity(
                path=path,
                size=getattr(sibling, "size", None),
                upstream_sha256=sha256,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("Video-MME-v2 official archive identity is invalid") from error
        _validate_official_archive_identity(
            identity, expected_archive_paths=OFFICIAL_ARCHIVE_PATHS
        )
        identities.append(identity)
    return tuple(identities)


def open_official_archive(identity: OfficialFileIdentity) -> BinaryIO:
    """Open an official archive through the pinned Hub filesystem range reader."""
    _validate_official_archive_identity(
        identity, expected_archive_paths=OFFICIAL_ARCHIVE_PATHS
    )
    try:
        from huggingface_hub import HfFileSystem
    except ImportError as error:
        raise RuntimeError("huggingface_hub is required for Video-MME-v2 archives") from error
    path = f"datasets/{DATASET_ID}@{FROZEN_REVISION}/{identity.path}"
    return HfFileSystem(token=False).open(path, "rb", block_size=1024 * 1024)


def _pilot_rank(video_id: str) -> tuple[str, str]:
    return (
        canonical_sha256(
            {"algorithm": POOL_ALGORITHM, "seed": POOL_SEED, "video_id": video_id}
        ),
        video_id,
    )


def select_pilot(
    metadata: ParsedVideoMME,
    archive_index: ArchiveIndex,
    count: int = 45,
) -> PilotSelectionManifest:
    """Select a deterministic archive-aware pilot without reading question content."""
    if count <= 0:
        raise ValueError("pilot selection count must be positive")
    if tuple(metadata.video_ids) != OFFICIAL_VIDEO_IDS:
        raise ValueError(
            "Video-MME-v2 pilot selection requires the exact full source population"
        )
    _validate_metadata_coverage(metadata.video_ids, archive_index.video_ids)
    if count > len(archive_index.video_ids):
        raise ValueError("pilot selection count exceeds available videos")
    archive_by_video = {
        member.video_id: member.archive_path for member in archive_index.members
    }
    members_by_archive: dict[str, list[str]] = {}
    for member in archive_index.members:
        members_by_archive.setdefault(member.archive_path, []).append(member.video_id)
    selected_archives: set[str] = set()
    covered: set[str] = set()
    for video_id in sorted(metadata.video_ids, key=_pilot_rank):
        if video_id in covered:
            continue
        archive_path = archive_by_video[video_id]
        selected_archives.add(archive_path)
        covered.update(members_by_archive[archive_path])
        if len(covered) >= count:
            break
    selected_video_ids = tuple(sorted(covered, key=_pilot_rank)[:count])
    payload = {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "immutable_revision": FROZEN_REVISION,
        "source_metadata_sha256": metadata.report.metadata_sha256,
        "source_archive_index_sha256": archive_index.archive_index_sha256,
        "pool_seed": POOL_SEED,
        "pool_algorithm": POOL_ALGORITHM,
        "available_video_count": len(archive_index.video_ids),
        "selected_video_count": len(selected_video_ids),
        "selected_archive_paths": tuple(sorted(selected_archives)),
        "selected_video_ids": selected_video_ids,
    }
    return PilotSelectionManifest(
        **payload,
        selection_sha256=canonical_sha256(payload),
    )


def full_scope_media(archive_index: ArchiveIndex) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return only the canonical full media scope; full scope has no subset hash."""
    archive_paths = tuple(archive.path for archive in archive_index.archives)
    video_ids = tuple(sorted(archive_index.video_ids))
    expected_video_ids = tuple(f"{number:03d}" for number in range(800))
    if archive_paths != OFFICIAL_ARCHIVE_PATHS or video_ids != expected_video_ids:
        raise ValueError("Video-MME-v2 full scope requires all 40 official archives and 800 videos")
    return archive_paths, video_ids


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
