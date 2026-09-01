"""Engineering-fixture tests for the pinned Video-MME-v2 metadata adapter."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sys
from io import BufferedReader
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

import duckdb
import pytest

from fidmem.assets.videomme_v2 import (
    ArchiveIndex,
    ArchiveMemberIdentity,
    DATASET_ID,
    DownloadPlan,
    FROZEN_REVISION,
    HumanAuditManifest,
    METADATA_FILES,
    OfficialFileIdentity,
    OFFICIAL_ARCHIVE_PATHS,
    OFFICIAL_VIDEO_IDS,
    POOL_ALGORITHM,
    POOL_SEED,
    PilotSelectionManifest,
    check_download_capacity,
    download_pinned_file,
    extract_selected_media,
    prepare_videos,
    build_human_audit_manifest,
    build_archive_index,
    build_manifests,
    build_pilot_split,
    full_scope_media,
    load_official_archive_identities,
    open_official_archive,
    select_pilot,
    validate_human_audit_result,
    verify_metadata,
    verify_raw_videos,
    write_dataset_preparation,
    prepare_e01,
    _verify_prepared_files,
)
from fidmem.data.video import VideoProbe
from fidmem.production.authority import canonical_sha256


def _rows(*, videos: int = 25) -> list[tuple[object, ...]]:
    return [
        (
            f"{video:03d}",
            f"https://example.invalid/{video:03d}",
            "perception",
            "single-choice",
            f"q-{video:03d}-{question}",
            "Engineering fixture question",
            json.dumps(["A", "B", "C", "D"]),
            "A",
            "easy",
            "visual",
            "detail",
        )
        for video in range(1, videos + 1)
        for question in range(4)
    ]


def _metadata(root: Path, *, rows: list[tuple[object, ...]] | None = None) -> None:
    root.mkdir()
    values = _rows() if rows is None else rows
    columns = (
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
    connection = duckdb.connect()
    try:
        connection.execute(
            "CREATE TABLE questions ("
            "video_id VARCHAR, url VARCHAR, group_type VARCHAR, "
            "group_structure VARCHAR, question_id VARCHAR, question VARCHAR, "
            "options VARCHAR, answer VARCHAR, level VARCHAR, "
            "second_head VARCHAR, third_head VARCHAR)"
        )
        connection.executemany(
            "INSERT INTO questions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", values
        )
        connection.execute(
            "COPY questions TO ? (FORMAT PARQUET)", [str(root / "test.parquet")]
        )
    finally:
        connection.close()
    video_ids = {str(row[0]) for row in values}
    with ZipFile(root / "subtitle.zip", "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("subtitle/", b"")
        for video_id in sorted(video_ids):
            archive.writestr(f"subtitle/{video_id}.jsonl", json.dumps({"text": "fixture"}) + "\n")
    (root / "README.md").write_text("Engineering fixture only.\n", encoding="utf-8")
    assert tuple(path.name for path in sorted(root.iterdir())) == tuple(sorted(METADATA_FILES))


def _verify(
    root: Path, *, expected_question_count: int = 100, expected_video_count: int = 25
):
    return verify_metadata(
        root,
        immutable_revision=FROZEN_REVISION,
        _expected_question_count=expected_question_count,
        _expected_video_count=expected_video_count,
    )


def _completed_audit_payload(audit: HumanAuditManifest) -> dict[str, object]:
    return {
        "status": "COMPLETED",
        "human_audit_manifest_sha256": audit.manifest_sha256,
        "reviewer_identity": "reviewer@example.test",
        "completed_at": "2026-08-30T00:00:00Z",
        "items": [
            {"question_id": item.question_id, "outcome": "PASS"}
            for item in audit.items
        ],
    }


def _write_audit_result(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_pilot_split_is_exact_deterministic_and_video_disjoint(tmp_path: Path) -> None:
    root = tmp_path / "metadata"
    _metadata(root, rows=_rows(videos=45))
    parsed = _verify(root, expected_question_count=180, expected_video_count=45)
    selected = tuple(f"{item:03d}" for item in range(1, 46))

    split = build_pilot_split(parsed, selected)
    repeated = build_pilot_split(parsed, tuple(reversed(selected)))

    assert split == repeated
    assert {group: len(ids) for group, ids in split.video_groups.items()} == {
        "oracle": 25,
        "canary": 4,
        "holdout": 4,
        "development": 12,
    }
    groups = tuple(set(ids) for ids in split.video_groups.values())
    assert set.union(*groups) == set(selected)
    assert all(left.isdisjoint(right) for index, left in enumerate(groups) for right in groups[index + 1 :])


def test_source_gate_verifies_unique_selected_videos_and_midpoint_decodes(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    videos = raw / "videos"
    subtitles = raw / "subtitles"
    videos.mkdir(parents=True)
    subtitles.mkdir()
    selected = tuple(f"{item:03d}" for item in range(1, 46))
    for index, video_id in enumerate(selected):
        (videos / f"{video_id}.mp4").write_bytes(f"video-{index}".encode())
        (subtitles / f"{video_id}.jsonl").write_text('{"text":"fixture"}\n')
    decoded: list[str] = []

    report = verify_raw_videos(
        selected,
        raw,
        probe=lambda path: VideoProbe(Path(path), 60.0, 384, 216, 24.0),
        decode=lambda path, timestamp: decoded.append(f"{path.stem}:{timestamp}"),
    )

    assert report.status == "PASS"
    assert report.expected_video_count == report.verified_video_count == 45
    assert report.random_decode_required == report.random_decode_completed == 20
    assert len(decoded) == 20


def test_source_gate_fails_closed_for_missing_or_duplicate_content(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    videos = raw / "videos"
    subtitles = raw / "subtitles"
    videos.mkdir(parents=True)
    subtitles.mkdir()
    selected = tuple(f"{item:03d}" for item in range(1, 46))
    for video_id in selected:
        (videos / f"{video_id}.mp4").write_bytes(b"same")
        (subtitles / f"{video_id}.jsonl").write_text("{}\n")
    (videos / "044.mp4").unlink()

    report = verify_raw_videos(
        selected,
        raw,
        probe=lambda path: VideoProbe(Path(path), 1.0, 1, 1, 1.0),
        decode=lambda path, timestamp: None,
    )

    assert report.status == "FAIL"
    assert report.missing_video_ids == ("044",)
    assert report.duplicate_video_ids


@pytest.mark.parametrize(
    "probe",
    (
        VideoProbe(Path("ignored"), 1.0, 0, 1, 1.0),
        VideoProbe(Path("ignored"), 1.0, 1, 0, 1.0),
        VideoProbe(Path("ignored"), 1.0, 1, 1, 0.0),
    ),
)
def test_source_gate_rejects_nonpositive_geometry_or_fps(
    tmp_path: Path, probe: VideoProbe
) -> None:
    raw = tmp_path / "raw"
    (raw / "videos").mkdir(parents=True)
    (raw / "subtitles").mkdir()
    selected = tuple(f"{item:03d}" for item in range(1, 46))
    for index, video_id in enumerate(selected):
        (raw / "videos" / f"{video_id}.mp4").write_bytes(
            f"unique-{index}".encode()
        )
        (raw / "subtitles" / f"{video_id}.jsonl").write_text("{}\n")

    report = verify_raw_videos(
        selected,
        raw,
        probe=lambda path: VideoProbe(
            Path(path), probe.duration_sec, probe.width, probe.height, probe.fps
        ),
        decode=lambda path, timestamp: None,
    )

    assert report.status == "FAIL"
    assert report.corrupt_video_ids == selected


def test_manifests_bind_pilot_provenance_and_gold_scope(tmp_path: Path) -> None:
    metadata_root = tmp_path / "metadata"
    _metadata(metadata_root, rows=_rows(videos=45))
    parsed = _verify(metadata_root, expected_question_count=180, expected_video_count=45)
    selected = tuple(f"{item:03d}" for item in range(1, 46))
    raw = tmp_path / "raw"
    (raw / "videos").mkdir(parents=True)
    (raw / "subtitles").mkdir()
    for index, video_id in enumerate(selected):
        (raw / "videos" / f"{video_id}.mp4").write_bytes(f"video-{index}".encode())
        (raw / "subtitles" / f"{video_id}.jsonl").write_text("{}\n")
    videos = verify_raw_videos(
        selected,
        raw,
        probe=lambda path: VideoProbe(Path(path), 60.0, 1, 1, 1.0),
        decode=lambda path, timestamp: None,
    )
    selection = SimpleNamespace(
        selection_sha256="2" * 64,
        source_archive_index_sha256="3" * 64,
        available_video_count=800,
        selected_video_ids=selected,
    )

    video_manifest, question_manifest, dataset_manifest, canary, oracle, split = build_manifests(
        parsed, videos, selection
    )

    assert len(video_manifest.records) == 45
    assert len(question_manifest.records) == 180
    assert dataset_manifest.dataset_scope == "PARTIAL_DATASET_PILOT"
    assert dataset_manifest.selected_video_count == 45
    assert dataset_manifest.selected_question_count == 180
    assert len(canary.question_ids) == 16
    assert len(oracle.question_ids) == 100
    assert set(canary.video_ids).isdisjoint(oracle.video_ids)
    assert all(record.gold_answer_sha256 is None for record in question_manifest.records if record.group in {"development", "canary"})
    assert all(record.gold_answer_sha256 for record in question_manifest.records if record.group in {"oracle", "holdout"})
    assert split.policy_sha256 == dataset_manifest.split_policy_sha256
    source_policy = (
        Path(__file__).resolve().parents[2]
        / "configs/experiment_stacks/videomme_v2_pilot_split_policy.yaml"
    )
    assert split.source_policy_sha256 == hashlib.sha256(
        source_policy.read_bytes()
    ).hexdigest()


def test_dataset_preparation_is_pending_and_formal_e01_requires_real_audit(tmp_path: Path) -> None:
    metadata_root = tmp_path / "metadata"
    _metadata(metadata_root, rows=_rows(videos=45))
    parsed = _verify(metadata_root, expected_question_count=180, expected_video_count=45)
    selected = tuple(f"{item:03d}" for item in range(1, 46))
    raw = tmp_path / "raw"
    (raw / "videos").mkdir(parents=True)
    (raw / "subtitles").mkdir()
    for index, video_id in enumerate(selected):
        (raw / "videos" / f"{video_id}.mp4").write_bytes(f"video-{index}".encode())
        (raw / "subtitles" / f"{video_id}.jsonl").write_text("{}\n")
    videos = verify_raw_videos(
        selected, raw,
        probe=lambda path: VideoProbe(Path(path), 60.0, 1, 1, 1.0),
        decode=lambda path, timestamp: None,
    )
    archive_index = _full_population_archive_index(_even_archive_members())
    member_archives = {
        member.video_id: member.archive_path for member in archive_index.members
    }
    selection_payload = {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "immutable_revision": FROZEN_REVISION,
        "source_metadata_sha256": parsed.report.metadata_sha256,
        "source_archive_index_sha256": archive_index.archive_index_sha256,
        "pool_seed": POOL_SEED,
        "pool_algorithm": POOL_ALGORITHM,
        "available_video_count": 800,
        "selected_video_count": 45,
        "selected_archive_paths": tuple(
            sorted({member_archives[video_id] for video_id in selected})
        ),
        "selected_video_ids": selected,
    }
    selection = PilotSelectionManifest(
        **selection_payload,
        selection_sha256=canonical_sha256(selection_payload),
    )
    output = tmp_path / "preparation"

    checked = write_dataset_preparation(
        parsed, videos, archive_index, selection, output, check=True
    )
    assert checked["status"] == "PENDING_HUMAN_AUDIT"
    assert not output.exists()

    payload = write_dataset_preparation(parsed, videos, archive_index, selection, output)

    assert payload["status"] == "PENDING_HUMAN_AUDIT"
    assert (output / "archive_index.json").is_file()
    assert (output / "split_policy.json").is_file()
    assert (output / "dataset_manifest.json").is_file()
    with pytest.raises(ValueError, match="human audit result is missing"):
        prepare_e01(output, tmp_path / "missing-audit.json", check=True)


def test_metadata_parses_official_shape_and_hashes_ordered_files(tmp_path: Path) -> None:
    root = tmp_path / "metadata"
    _metadata(root)

    parsed = _verify(root)

    assert parsed.report.status == "VERIFIED"
    assert parsed.report.question_count == 100
    assert parsed.report.video_count == 25
    assert parsed.report.files == METADATA_FILES
    assert tuple(record.path for record in parsed.report.file_identities) == METADATA_FILES
    assert parsed.questions[0].question_types == ("perception", "easy", "visual", "detail")


def test_metadata_rejects_duplicate_question_ids(tmp_path: Path) -> None:
    root = tmp_path / "metadata"
    rows = _rows()
    rows[1] = (*rows[1][:4], rows[0][4], *rows[1][5:])
    _metadata(root, rows=rows)

    with pytest.raises(ValueError, match="duplicate question IDs"):
        _verify(root)


def test_metadata_rejects_nonofficial_video_id_set_at_full_count(
    tmp_path: Path,
) -> None:
    root = tmp_path / "metadata"
    rows = _rows(videos=800)
    rows = [
        (("000", *row[1:]) if row[0] == "800" else row)
        for row in rows
    ]
    _metadata(root, rows=rows)

    with pytest.raises(ValueError, match="exact official video ID set"):
        _verify(root, expected_question_count=3200, expected_video_count=800)


def test_metadata_rejects_video_without_exactly_four_questions(tmp_path: Path) -> None:
    root = tmp_path / "metadata"
    _metadata(root, rows=_rows()[:-1])

    with pytest.raises(ValueError, match="exactly four questions"):
        _verify(root)


@pytest.mark.parametrize("field", ["options", "answer"])
def test_metadata_rejects_blank_options_or_answers(tmp_path: Path, field: str) -> None:
    root = tmp_path / "metadata"
    rows = _rows()
    index = {"options": 6, "answer": 7}[field]
    rows[0] = (*rows[0][:index], "  ", *rows[0][index + 1 :])
    _metadata(root, rows=rows)

    with pytest.raises(ValueError, match=field):
        _verify(root)


def test_metadata_rejects_subtitle_video_mismatches(tmp_path: Path) -> None:
    root = tmp_path / "metadata"
    _metadata(root)
    with ZipFile(root / "subtitle.zip", "a", compression=ZIP_DEFLATED) as archive:
        archive.writestr("999.jsonl", '{"text":"unexpected"}\n')

    with pytest.raises(ValueError, match="subtitle/video IDs differ"):
        _verify(root)


def test_metadata_rejects_unexpected_files_and_wrong_revision(tmp_path: Path) -> None:
    root = tmp_path / "metadata"
    _metadata(root)
    (root / "notes.txt").write_text("unexpected", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected metadata files"):
        _verify(root)
    (root / "notes.txt").unlink()
    with pytest.raises(ValueError, match="frozen immutable revision"):
        verify_metadata(
            root,
            immutable_revision="f" * 40,
            _expected_question_count=100,
            _expected_video_count=25,
        )


def test_metadata_allows_only_huggingface_housekeeping_directory(tmp_path: Path) -> None:
    root = tmp_path / "metadata"
    _metadata(root)
    housekeeping = root / ".cache" / "huggingface"
    housekeeping.mkdir(parents=True)
    (housekeeping / ".gitignore").write_text("*\n")

    assert _verify(root).report.status == "VERIFIED"


def test_human_audit_is_pending_and_binds_exact_selected_question_ids(
    tmp_path: Path,
) -> None:
    root = tmp_path / "metadata"
    _metadata(root)
    parsed = _verify(root)

    audit = build_human_audit_manifest(
        parsed,
        selected_video_ids=tuple(f"{item:03d}" for item in range(1, 26)),
        seed="fixture-seed",
    )

    assert audit.status == "PENDING_HUMAN_AUDIT"
    assert audit.required_items == 100
    assert len(audit.items) == 100
    assert {item.question_id for item in audit.items} == {
        item.question_id for item in parsed.questions
    }
    result = tmp_path / "human-audit.json"
    _write_audit_result(result, _completed_audit_payload(audit))
    assert validate_human_audit_result(audit, result) is None


def test_human_audit_manifest_rejects_duplicate_question_ids(tmp_path: Path) -> None:
    root = tmp_path / "metadata"
    _metadata(root)
    audit = build_human_audit_manifest(
        _verify(root),
        selected_video_ids=tuple(f"{item:03d}" for item in range(1, 26)),
        seed="fixture-seed",
    )

    payload = audit.model_dump(mode="json")
    payload["items"] = [audit.items[0].model_dump(mode="json")] * 100

    with pytest.raises(ValueError, match="unique question IDs"):
        HumanAuditManifest.model_validate(payload)


def test_human_audit_rejects_duplicate_result_ids_for_a_forged_manifest(
    tmp_path: Path,
) -> None:
    root = tmp_path / "metadata"
    _metadata(root)
    audit = build_human_audit_manifest(
        _verify(root),
        selected_video_ids=tuple(f"{item:03d}" for item in range(1, 26)),
        seed="fixture-seed",
    )
    forged = audit.model_copy(update={"items": (audit.items[0],) * 100})
    result = tmp_path / "human-audit.json"
    _write_audit_result(result, _completed_audit_payload(forged))

    with pytest.raises(ValueError, match="duplicate question IDs"):
        validate_human_audit_result(forged, result)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("unbound", "bind"),
        ("non_pass", "non-PASS"),
        ("duplicate", "duplicate question IDs"),
        ("missing", "wrong completed item count"),
        ("blank_reviewer", "reviewer identity"),
        ("blank_completion", "completion identity"),
    ],
)
def test_human_audit_rejects_invalid_bound_results(
    tmp_path: Path, mutation: str, message: str
) -> None:
    root = tmp_path / "metadata"
    _metadata(root)
    audit = build_human_audit_manifest(
        _verify(root),
        selected_video_ids=tuple(f"{item:03d}" for item in range(1, 26)),
        seed="fixture-seed",
    )
    payload = _completed_audit_payload(audit)
    items = payload["items"]
    assert isinstance(items, list)
    if mutation == "unbound":
        payload["human_audit_manifest_sha256"] = "0" * 64
    elif mutation == "non_pass":
        items[0]["outcome"] = "FAIL"
    elif mutation == "duplicate":
        items[-1]["question_id"] = items[0]["question_id"]
    elif mutation == "missing":
        payload["items"] = items[:-1]
    elif mutation == "blank_reviewer":
        payload["reviewer_identity"] = "  "
    else:
        payload["completed_at"] = "  "
    result = tmp_path / "human-audit.json"
    _write_audit_result(result, payload)

    with pytest.raises(ValueError, match=message):
        validate_human_audit_result(audit, result)


def test_human_audit_only_uses_selected_video_ids(tmp_path: Path) -> None:
    root = tmp_path / "metadata"
    _metadata(root, rows=_rows(videos=30))
    parsed = _verify(root, expected_question_count=120, expected_video_count=30)
    selected_video_ids = tuple(f"{item:03d}" for item in range(5, 30))

    audit = build_human_audit_manifest(
        parsed, selected_video_ids=selected_video_ids, seed="fixture-seed"
    )

    assert len(audit.items) == 100
    assert {item.video_id for item in audit.items} == set(selected_video_ids)


def _archive_fixtures(
    root: Path,
    *,
    archives: int = 3,
    members_per_archive: int = 20,
    extra_members: dict[str, tuple[str, ...]] | None = None,
) -> tuple[OfficialFileIdentity, ...]:
    """Create ZIP engineering fixtures; their identities stand in for upstream LFS data."""
    root.mkdir()
    identities = []
    for archive_number in range(1, archives + 1):
        archive_path = f"videos/{archive_number:03d}.zip"
        local_path = root / f"{archive_number:03d}.zip"
        with ZipFile(local_path, "w", compression=ZIP_DEFLATED) as archive:
            first_video = (archive_number - 1) * members_per_archive + 1
            for video_number in range(first_video, first_video + members_per_archive):
                archive.writestr(f"{video_number:03d}.mp4", b"engineering-mp4")
            for member in (extra_members or {}).get(archive_path, ()):
                archive.writestr(member, b"unsafe")
        identities.append(
            OfficialFileIdentity(
                path=archive_path,
                size=local_path.stat().st_size,
                upstream_sha256=f"{archive_number:064x}",
            )
        )
    return tuple(identities)


def _fixture_opener(root: Path):
    def open_archive(identity: OfficialFileIdentity) -> BufferedReader:
        return (root / Path(identity.path).name).open("rb")

    return open_archive


def test_archive_index_records_only_safe_canonical_members_and_exact_metadata_coverage(
    tmp_path: Path,
) -> None:
    metadata_root = tmp_path / "metadata"
    _metadata(metadata_root, rows=_rows(videos=60))
    metadata = _verify(metadata_root, expected_question_count=240, expected_video_count=60)
    archive_root = tmp_path / "archives"
    identities = _archive_fixtures(archive_root)

    index = build_archive_index(
        identities,
        _fixture_opener(archive_root),
        metadata_video_ids=metadata.video_ids,
        _expected_archive_paths=tuple(identity.path for identity in identities),
    )

    assert tuple(archive.path for archive in index.archives) == tuple(
        identity.path for identity in identities
    )
    assert index.video_ids == tuple(f"{video:03d}" for video in range(1, 61))
    assert tuple(member.member_path for member in index.members[:20]) == tuple(
        f"{video:03d}.mp4" for video in range(1, 21)
    )
    assert index.archive_index_sha256 == build_archive_index(
        tuple(reversed(identities)),
        _fixture_opener(archive_root),
        metadata_video_ids=metadata.video_ids,
        _expected_archive_paths=tuple(identity.path for identity in identities),
    ).archive_index_sha256


@pytest.mark.parametrize(
    ("extra_member", "message"),
    [
        ("../escape.mp4", "unsafe ZIP member path"),
        ("/absolute.mp4", "unsafe ZIP member path"),
        (r"..\escape.mp4", "unsafe ZIP member path"),
        (r"C:\escape.mp4", "unsafe ZIP member path"),
        (r"\\server\share\escape.mp4", "unsafe ZIP member path"),
        ("not-a-video.txt", "non-MP4 ZIP member"),
    ],
)
def test_archive_index_rejects_unsafe_or_non_mp4_members(
    tmp_path: Path, extra_member: str, message: str
) -> None:
    archive_root = tmp_path / "archives"
    identities = _archive_fixtures(
        archive_root, extra_members={"videos/001.zip": (extra_member,)}
    )

    with pytest.raises(ValueError, match=message):
        build_archive_index(
            identities,
            _fixture_opener(archive_root),
            metadata_video_ids=tuple(f"{video:03d}" for video in range(1, 61)),
            _expected_archive_paths=tuple(identity.path for identity in identities),
        )


def test_archive_index_rejects_duplicate_members_and_nonmatching_identities(
    tmp_path: Path,
) -> None:
    archive_root = tmp_path / "archives"
    identities = _archive_fixtures(archive_root)
    with ZipFile(archive_root / "002.zip", "a", compression=ZIP_DEFLATED) as archive:
        archive.writestr("001.mp4", b"duplicate-video")
    identities = (
        identities[0],
        identities[1].model_copy(update={"size": (archive_root / "002.zip").stat().st_size}),
        identities[2],
    )

    with pytest.raises(ValueError, match="duplicate video IDs"):
        build_archive_index(
            identities,
            _fixture_opener(archive_root),
            metadata_video_ids=tuple(f"{video:03d}" for video in range(1, 61)),
            _expected_archive_paths=tuple(identity.path for identity in identities),
        )

    clean_root = tmp_path / "clean-archives"
    clean_identities = _archive_fixtures(clean_root)
    with pytest.raises(ValueError, match="upstream SHA-256"):
        build_archive_index(
            (clean_identities[0].model_construct(
                path=clean_identities[0].path,
                size=clean_identities[0].size,
                upstream_sha256="",
            ), *clean_identities[1:]),
            _fixture_opener(clean_root),
            metadata_video_ids=tuple(f"{video:03d}" for video in range(1, 61)),
            _expected_archive_paths=tuple(identity.path for identity in clean_identities),
        )
    with pytest.raises(ValueError, match="archive size differs"):
        build_archive_index(
            (clean_identities[0].model_copy(update={"size": clean_identities[0].size + 1}), *clean_identities[1:]),
            _fixture_opener(clean_root),
            metadata_video_ids=tuple(f"{video:03d}" for video in range(1, 61)),
            _expected_archive_paths=tuple(identity.path for identity in clean_identities),
        )


def test_archive_index_requires_exact_metadata_coverage(tmp_path: Path) -> None:
    archive_root = tmp_path / "archives"
    identities = _archive_fixtures(archive_root)
    expected_paths = tuple(identity.path for identity in identities)

    with pytest.raises(ValueError, match="absent from metadata"):
        build_archive_index(
            identities,
            _fixture_opener(archive_root),
            metadata_video_ids=tuple(f"{video:03d}" for video in range(59)),
            _expected_archive_paths=expected_paths,
        )
    with pytest.raises(ValueError, match="missing from archives"):
        build_archive_index(
            identities,
            _fixture_opener(archive_root),
            metadata_video_ids=tuple(f"{video:03d}" for video in range(61)),
            _expected_archive_paths=expected_paths,
        )


def test_pilot_selection_is_archive_aware_deterministic_and_question_independent(
) -> None:
    class MetadataWithoutQuestionContents:
        video_ids = OFFICIAL_VIDEO_IDS
        report = SimpleNamespace(metadata_sha256="1" * 64)

        @property
        def questions(self) -> tuple[object, ...]:
            raise AssertionError("pilot selection must not inspect questions or answers")

    index = _full_population_archive_index(_even_archive_members())
    first = select_pilot(MetadataWithoutQuestionContents(), index)
    second = select_pilot(MetadataWithoutQuestionContents(), index)

    assert first.pool_algorithm == POOL_ALGORITHM
    assert first.pool_seed == POOL_SEED
    assert first.selected_video_ids == second.selected_video_ids
    assert first.selected_archive_paths == second.selected_archive_paths
    assert first.selection_sha256 == second.selection_sha256
    assert len(first.selected_video_ids) == 45
    assert set(first.selected_video_ids).issubset(
        {member.video_id for member in index.members}
    )


def _full_population_archive_index(
    members_by_archive: tuple[tuple[str, ...], ...]
) -> ArchiveIndex:
    assert len(members_by_archive) == len(OFFICIAL_ARCHIVE_PATHS)
    assert tuple(video_id for ids in members_by_archive for video_id in ids) == OFFICIAL_VIDEO_IDS
    archives = tuple(
        OfficialFileIdentity(
            path=archive_path,
            size=archive_number,
            upstream_sha256=f"{archive_number:064x}",
        )
        for archive_number, archive_path in enumerate(OFFICIAL_ARCHIVE_PATHS, start=1)
    )
    members = tuple(
        ArchiveMemberIdentity(
            archive_path=archive_path,
            video_id=video_id,
            member_path=f"{video_id}.mp4",
            crc32=archive_number,
            compressed_size=1,
            uncompressed_size=1,
        )
        for archive_number, (archive_path, video_ids) in enumerate(
            zip(OFFICIAL_ARCHIVE_PATHS, members_by_archive, strict=True), start=1
        )
        for video_id in video_ids
    )
    payload = {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "immutable_revision": FROZEN_REVISION,
        "archives": [archive.model_dump(mode="json") for archive in archives],
        "members": [member.model_dump(mode="json") for member in members],
    }
    return ArchiveIndex(
        **payload, archive_index_sha256=canonical_sha256(payload)
    )


def test_formal_e01_revalidates_archive_selection_and_manifest_chain(tmp_path: Path) -> None:
    metadata_root = tmp_path / "metadata"
    _metadata(metadata_root, rows=_rows(videos=800))
    metadata = _verify(metadata_root, expected_question_count=3200, expected_video_count=800)
    archive_index = _full_population_archive_index(_even_archive_members())
    selection = select_pilot(metadata, archive_index)
    raw = tmp_path / "raw"
    (raw / "videos").mkdir(parents=True)
    (raw / "subtitles").mkdir()
    for index, video_id in enumerate(selection.selected_video_ids):
        (raw / "videos" / f"{video_id}.mp4").write_bytes(f"video-{index}".encode())
        (raw / "subtitles" / f"{video_id}.jsonl").write_text("{}\n")
    videos = verify_raw_videos(
        selection.selected_video_ids,
        raw,
        probe=lambda path: VideoProbe(Path(path), 60.0, 1, 1, 1.0),
        decode=lambda path, timestamp: None,
    )
    preparation = tmp_path / "preparation"
    write_dataset_preparation(metadata, videos, archive_index, selection, preparation)
    audit = HumanAuditManifest.model_validate_json(
        (preparation / "human_audit_manifest.json").read_text()
    )
    audit_result = tmp_path / "audit-result.json"
    _write_audit_result(audit_result, _completed_audit_payload(audit))

    e01_probe = lambda path: VideoProbe(Path(path), 60.0, 1, 1, 1.0)
    assert prepare_e01(
        preparation, audit_result, check=True, probe=e01_probe
    )["source_gate"] == "PASS"

    selected_video = Path(videos.video_records[0].uri)
    original_video = selected_video.read_bytes()
    selected_video.write_bytes(original_video + b"tampered")
    with pytest.raises(ValueError, match="current raw video identity"):
        prepare_e01(preparation, audit_result, check=True, probe=e01_probe)
    selected_video.write_bytes(original_video)

    archive_payload = json.loads((preparation / "archive_index.json").read_text())
    archive_payload["archives"][0]["size"] += 1
    (preparation / "archive_index.json").write_text(json.dumps(archive_payload))
    with pytest.raises(ValueError, match="archive_index_sha256"):
        prepare_e01(preparation, audit_result, check=True, probe=e01_probe)


def _even_archive_members() -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(f"{video:03d}" for video in range(start, start + 20))
        for start in range(1, 801, 20)
    )


def _uneven_full_archive_members() -> tuple[tuple[str, ...], ...]:
    sizes = (101, 7, 39) + (17,) * 36 + (41,)
    assert len(sizes) == 40 and sum(sizes) == 800
    start = 1
    members = []
    for size in sizes:
        members.append(tuple(f"{video:03d}" for video in range(start, start + size)))
        start += size
    return tuple(members)


def _independent_pilot_expected(
    members_by_archive: tuple[tuple[str, ...], ...], count: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    archive_by_video = {
        video_id: f"videos/{archive_number:03d}.zip"
        for archive_number, video_ids in enumerate(members_by_archive, start=1)
        for video_id in video_ids
    }
    rank = lambda video_id: (
        canonical_sha256(
            {"algorithm": POOL_ALGORITHM, "seed": POOL_SEED, "video_id": video_id}
        ),
        video_id,
    )
    covered: set[str] = set()
    selected_archives: set[str] = set()
    for video_id in sorted(archive_by_video, key=rank):
        if video_id not in covered:
            archive_path = archive_by_video[video_id]
            selected_archives.add(archive_path)
            archive_number = int(Path(archive_path).stem)
            covered.update(members_by_archive[archive_number - 1])
        if len(covered) >= count:
            break
    return tuple(sorted(selected_archives)), tuple(sorted(covered, key=rank)[:count])


def test_pilot_selection_matches_independent_uneven_archive_hash_ranking(
) -> None:
    members_by_archive = _uneven_full_archive_members()
    metadata = SimpleNamespace(
        video_ids=OFFICIAL_VIDEO_IDS, report=SimpleNamespace(metadata_sha256="2" * 64)
    )
    index = _full_population_archive_index(members_by_archive)

    selected = select_pilot(metadata, index)
    expected_archives, expected_videos = _independent_pilot_expected(
        members_by_archive, count=45
    )

    assert selected.selected_archive_paths == expected_archives
    assert selected.selected_video_ids == expected_videos


def test_pilot_selection_requires_canonical_full_population_without_a_public_bypass(
) -> None:
    index = _full_population_archive_index(_even_archive_members())
    unordered_metadata = SimpleNamespace(
        video_ids=tuple(reversed(OFFICIAL_VIDEO_IDS)),
        report=SimpleNamespace(metadata_sha256="3" * 64),
    )

    with pytest.raises(ValueError, match="full source population"):
        select_pilot(unordered_metadata, index)
    with pytest.raises(TypeError, match="_expected_video_ids"):
        select_pilot(  # type: ignore[call-arg]
            SimpleNamespace(
                video_ids=OFFICIAL_VIDEO_IDS,
                report=SimpleNamespace(metadata_sha256="4" * 64),
            ),
            index,
            _expected_video_ids=OFFICIAL_VIDEO_IDS,
        )


def _official_hub_siblings(*, extra_archive: bool = False) -> tuple[object, ...]:
    paths = OFFICIAL_ARCHIVE_PATHS + (("videos/041.zip",) if extra_archive else ())
    return tuple(
        SimpleNamespace(
            rfilename=path,
            size=100 + number,
            lfs=SimpleNamespace(sha256=f"{number:064x}"),
        )
        for number, path in enumerate(paths, start=1)
    )


def test_remote_archive_helpers_use_exact_pinned_hub_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    stream = object()

    class FakeHfApi:
        def dataset_info(self, **kwargs: object) -> object:
            calls["dataset_info"] = kwargs
            return SimpleNamespace(siblings=_official_hub_siblings())

    class FakeHfFileSystem:
        def __init__(self, *, token: bool) -> None:
            calls["token"] = token

        def open(self, path: str, mode: str, *, block_size: int) -> object:
            calls["open"] = (path, mode, block_size)
            return stream

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=FakeHfApi, HfFileSystem=FakeHfFileSystem),
    )

    identities = load_official_archive_identities()
    opened = open_official_archive(identities[0])

    assert calls["dataset_info"] == {
        "repo_id": DATASET_ID,
        "revision": FROZEN_REVISION,
        "files_metadata": True,
    }
    assert tuple(identity.path for identity in identities) == OFFICIAL_ARCHIVE_PATHS
    assert calls["token"] is False
    assert calls["open"] == (
        f"datasets/{DATASET_ID}@{FROZEN_REVISION}/{OFFICIAL_ARCHIVE_PATHS[0]}",
        "rb",
        1024 * 1024,
    )
    assert opened is stream


def test_remote_archive_identity_loading_rejects_unexpected_video_zip_siblings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeHfApi:
        def dataset_info(self, **kwargs: object) -> object:
            return SimpleNamespace(siblings=_official_hub_siblings(extra_archive=True))

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(HfApi=FakeHfApi)
    )

    with pytest.raises(ValueError, match="unexpected official archive siblings"):
        load_official_archive_identities()


def test_full_scope_media_has_all_pinned_archives_and_videos_without_a_selection_hash(
    tmp_path: Path,
) -> None:
    archive_root = tmp_path / "archives"
    identities = _archive_fixtures(archive_root, archives=40)
    index = build_archive_index(identities, _fixture_opener(archive_root))

    archive_paths, video_ids = full_scope_media(index)

    assert archive_paths == tuple(f"videos/{archive:03d}.zip" for archive in range(1, 41))
    assert video_ids == tuple(f"{video:03d}" for video in range(1, 801))
    assert not hasattr((archive_paths, video_ids), "selection_sha256")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _download_identity(content: bytes, path: str = "videos/001.zip") -> OfficialFileIdentity:
    return OfficialFileIdentity(
        path=path,
        size=len(content),
        upstream_sha256=_sha256_bytes(content),
    )


def test_download_resumes_partial_sibling_from_its_exact_size(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    content = b"0123456789abcdef"
    identity = _download_identity(content)
    destination = tmp_path / "001.zip"
    partial = destination.with_suffix(".partial")
    partial.write_bytes(content[:7])
    calls: list[tuple[str, int, int]] = []

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            hf_hub_url=lambda **kwargs: (
                f"https://example.invalid/{kwargs['repo_id']}/{kwargs['filename']}"
            )
        ),
    )

    def getter(url: str, stream: object, resume_size: int, expected_size: int) -> None:
        calls.append((url, resume_size, expected_size))
        stream.write(content[resume_size:])

    returned = download_pinned_file(identity, destination, resume=True, http_getter=getter)

    assert returned == destination
    assert destination.read_bytes() == content
    assert not partial.exists()
    assert calls == [
        (
            f"https://example.invalid/{DATASET_ID}/videos/001.zip",
            7,
            len(content),
        )
    ]


def test_download_logs_local_path_and_byte_state_without_remote_credentials(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    content = b"logged-payload"
    identity = _download_identity(content)
    destination = tmp_path / "001.zip"
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            hf_hub_url=lambda **_kwargs: "https://user:secret@example.invalid/archive"
        ),
    )

    def getter(_url: str, stream: object, resume_size: int, _size: int) -> None:
        stream.write(content[resume_size:])

    with caplog.at_level(logging.INFO, logger="fidmem.assets.videomme_v2"):
        download_pinned_file(identity, destination, resume=True, http_getter=getter)

    messages = "\n".join(caplog.messages)
    assert str(destination) in messages
    assert f"expected_bytes={len(content)}" in messages
    assert "state=VERIFIED" in messages
    assert "secret" not in messages


def test_download_reuses_completed_hash_matching_file_without_network(
    tmp_path: Path,
) -> None:
    content = b"already-complete"
    identity = _download_identity(content)
    destination = tmp_path / "001.zip"
    destination.write_bytes(content)
    before = destination.stat().st_mtime_ns

    def forbidden_getter(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("verified completed download must be reused")

    assert download_pinned_file(
        identity, destination, resume=True, http_getter=forbidden_getter
    ) == destination
    assert destination.stat().st_mtime_ns == before


def test_download_promotes_completed_hash_matching_partial_without_network(
    tmp_path: Path,
) -> None:
    content = b"completed-partial"
    identity = _download_identity(content)
    destination = tmp_path / "001.zip"
    partial = destination.with_suffix(".partial")
    partial.write_bytes(content)

    def forbidden_getter(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("completed verified partial must not use the network")

    returned = download_pinned_file(
        identity, destination, resume=True, http_getter=forbidden_getter
    )

    assert returned.read_bytes() == content
    assert not partial.exists()


def test_download_hash_mismatch_preserves_existing_final_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected = b"new-official-payload"
    identity = _download_identity(expected)
    destination = tmp_path / "001.zip"
    destination.write_bytes(b"last-visible-final")
    partial = destination.with_suffix(".partial")
    partial.write_bytes(b"bad")
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_url=lambda **_kwargs: "https://example.invalid/archive"),
    )

    def bad_getter(
        _url: str, stream: object, resume_size: int, _expected_size: int
    ) -> None:
        stream.write(b"x" * (len(expected) - resume_size))

    with pytest.raises(ValueError, match="SHA-256"):
        download_pinned_file(identity, destination, resume=True, http_getter=bad_getter)

    assert destination.read_bytes() == b"last-visible-final"
    assert partial.exists()


def test_download_rejects_oversized_partial_before_network(tmp_path: Path) -> None:
    identity = _download_identity(b"short")
    destination = tmp_path / "001.zip"
    destination.with_suffix(".partial").write_bytes(b"too-long")

    def forbidden_getter(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("oversized partial must fail before network")

    with pytest.raises(ValueError, match="partial.*larger"):
        download_pinned_file(
            identity, destination, resume=True, http_getter=forbidden_getter
        )


def _download_plan(*, archive_bytes: int, extracted_bytes: int) -> DownloadPlan:
    archive = OfficialFileIdentity(
        path="videos/001.zip", size=archive_bytes, upstream_sha256="1" * 64
    )
    member = ArchiveMemberIdentity(
        archive_path=archive.path,
        video_id="000",
        member_path="000.mp4",
        crc32=1,
        compressed_size=1,
        uncompressed_size=extracted_bytes,
    )
    return DownloadPlan(
        scope="pilot",
        archives=(archive,),
        selected_members=(member,),
        archive_bytes_remaining=archive_bytes,
    )


def test_capacity_requires_remaining_archives_selected_mp4s_and_20_gib(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    margin = 20 * 1024**3
    plan = _download_plan(archive_bytes=13, extracted_bytes=29)
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=margin + 41, used=0, free=margin + 41),
    )

    with pytest.raises(ValueError, match="insufficient.*space"):
        check_download_capacity(plan, tmp_path)

    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=margin + 42, used=0, free=margin + 42),
    )
    assert check_download_capacity(plan, tmp_path) is None


def _single_archive_index(
    archive_path: Path,
    *,
    member_names: tuple[str, ...] = ("000.mp4", "001.mp4"),
) -> ArchiveIndex:
    with ZipFile(archive_path) as archive:
        infos = {info.filename: info for info in archive.infolist()}
    identity = OfficialFileIdentity(
        path="videos/001.zip",
        size=archive_path.stat().st_size,
        upstream_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
    )
    members = tuple(
        ArchiveMemberIdentity(
            archive_path=identity.path,
            video_id=Path(name.rstrip("/")).stem,
            member_path=name,
            crc32=infos[name].CRC,
            compressed_size=infos[name].compress_size,
            uncompressed_size=infos[name].file_size,
        )
        for name in member_names
    )
    payload = {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "immutable_revision": FROZEN_REVISION,
        "archives": [identity.model_dump(mode="json")],
        "members": [member.model_dump(mode="json") for member in members],
    }
    return ArchiveIndex(**payload, archive_index_sha256=canonical_sha256(payload))


def _subtitle_fixture(path: Path, video_ids: tuple[str, ...]) -> None:
    with ZipFile(path, "w", compression=ZIP_STORED) as archive:
        archive.writestr("subtitle/", b"")
        for video_id in video_ids:
            archive.writestr(f"subtitle/{video_id}.jsonl", f'{{"video_id":"{video_id}"}}\n')


def test_extract_writes_only_selected_mp4s_and_subtitles_atomically(
    tmp_path: Path,
) -> None:
    archive_root = tmp_path / "archives"
    archive_root.mkdir()
    archive_path = archive_root / "001.zip"
    video_ids = tuple(f"{number:03d}" for number in range(1, 61))
    with ZipFile(archive_path, "w", compression=ZIP_STORED) as archive:
        for video_id in video_ids:
            archive.writestr(f"{video_id}.mp4", f"mp4-{video_id}".encode())
    index = _single_archive_index(archive_path, member_names=tuple(f"{v}.mp4" for v in video_ids))
    subtitle_zip = tmp_path / "subtitle.zip"
    _subtitle_fixture(subtitle_zip, video_ids)
    selected = video_ids[:45]
    video_root = tmp_path / "videos"

    paths = extract_selected_media(
        selected, index, archive_root, video_root, subtitle_zip
    )

    assert paths == tuple(video_root / f"{video_id}.mp4" for video_id in selected)
    assert tuple(path.name for path in sorted(video_root.glob("*.mp4"))) == tuple(
        f"{video_id}.mp4" for video_id in selected
    )
    assert tuple(
        path.name for path in sorted((tmp_path / "subtitles").glob("*.jsonl"))
    ) == tuple(f"{video_id}.jsonl" for video_id in selected)
    assert not tuple(tmp_path.rglob("*.tmp"))


def test_verify_prepared_files_accepts_official_nested_subtitle_layout(
    tmp_path: Path,
) -> None:
    archive_root = tmp_path / "raw" / "archives"
    archive_root.mkdir(parents=True)
    archive_path = archive_root / "001.zip"
    with ZipFile(archive_path, "w", compression=ZIP_STORED) as archive:
        archive.writestr("001.mp4", b"official-video")
    index = _single_archive_index(archive_path, member_names=("001.mp4",))
    subtitle_zip = tmp_path / "metadata" / "subtitle.zip"
    subtitle_zip.parent.mkdir()
    _subtitle_fixture(subtitle_zip, ("001",))
    extract_selected_media(
        ("001",), index, archive_root, tmp_path / "raw" / "videos", subtitle_zip
    )
    plan = DownloadPlan(
        scope="full",
        archives=index.archives,
        selected_members=index.members,
        archive_bytes_remaining=0,
    )

    archives, videos, subtitles = _verify_prepared_files(
        plan, tmp_path / "raw", subtitle_zip
    )

    assert archives == (archive_path,)
    assert videos == (tmp_path / "raw" / "videos" / "001.mp4",)
    assert subtitles == (tmp_path / "raw" / "subtitles" / "001.jsonl",)


def test_extract_reuses_output_only_while_recorded_size_and_hash_match(
    tmp_path: Path,
) -> None:
    archive_root = tmp_path / "archives"
    archive_root.mkdir()
    archive_path = archive_root / "001.zip"
    with ZipFile(archive_path, "w", compression=ZIP_STORED) as archive:
        archive.writestr("000.mp4", b"official-video")
    index = _single_archive_index(archive_path, member_names=("000.mp4",))
    subtitle_zip = tmp_path / "subtitle.zip"
    _subtitle_fixture(subtitle_zip, ("000",))
    video_root = tmp_path / "videos"
    output = extract_selected_media(
        ("000",), index, archive_root, video_root, subtitle_zip
    )[0]
    first_mtime = output.stat().st_mtime_ns

    extract_selected_media(("000",), index, archive_root, video_root, subtitle_zip)
    assert output.stat().st_mtime_ns == first_mtime

    output.write_bytes(b"same-size-wrong")
    os.utime(output, ns=(first_mtime, first_mtime))
    extract_selected_media(("000",), index, archive_root, video_root, subtitle_zip)
    assert output.read_bytes() == b"official-video"


def test_extract_never_trusts_a_symlinked_reuse_identity_record(
    tmp_path: Path,
) -> None:
    archive_root = tmp_path / "archives"
    archive_root.mkdir()
    archive_path = archive_root / "001.zip"
    with ZipFile(archive_path, "w", compression=ZIP_STORED) as archive:
        archive.writestr("000.mp4", b"official-video")
    index = _single_archive_index(archive_path, member_names=("000.mp4",))
    subtitle_zip = tmp_path / "subtitle.zip"
    _subtitle_fixture(subtitle_zip, ("000",))
    video_root = tmp_path / "videos"
    output = extract_selected_media(
        ("000",), index, archive_root, video_root, subtitle_zip
    )[0]
    record_path = output.with_suffix(".mp4.identity.json")
    forged = json.loads(record_path.read_text(encoding="utf-8"))
    output.write_bytes(b"forged-content")
    forged["size"] = len(b"forged-content")
    forged["sha256"] = hashlib.sha256(b"forged-content").hexdigest()
    outside_record = tmp_path / "outside-record.json"
    outside_record.write_text(json.dumps(forged), encoding="utf-8")
    record_path.unlink()
    record_path.symlink_to(outside_record)

    extract_selected_media(("000",), index, archive_root, video_root, subtitle_zip)

    assert output.read_bytes() == b"official-video"
    assert not record_path.is_symlink()


@pytest.mark.parametrize(
    ("unsafe_name", "message"),
    [
        ("../000.mp4", "unsafe ZIP member path"),
        ("/000.mp4", "unsafe ZIP member path"),
        ("000.mp4/", "directory.*media"),
    ],
)
def test_extract_rejects_unsafe_or_disguised_local_archive_members(
    tmp_path: Path, unsafe_name: str, message: str
) -> None:
    archive_root = tmp_path / "archives"
    archive_root.mkdir()
    archive_path = archive_root / "001.zip"
    with ZipFile(archive_path, "w", compression=ZIP_STORED) as archive:
        archive.writestr(unsafe_name, b"unsafe")
    index = _single_archive_index(archive_path, member_names=(unsafe_name,))
    subtitle_zip = tmp_path / "subtitle.zip"
    _subtitle_fixture(subtitle_zip, ("000",))

    with pytest.raises(ValueError, match=message):
        extract_selected_media(("000",), index, archive_root, tmp_path / "videos", subtitle_zip)


def test_extract_rejects_symlink_and_unexpected_selected_names(tmp_path: Path) -> None:
    archive_root = tmp_path / "archives"
    archive_root.mkdir()
    archive_path = archive_root / "001.zip"
    link = ZipInfo("000.mp4")
    link.external_attr = (0o120777 << 16)
    with ZipFile(archive_path, "w", compression=ZIP_STORED) as archive:
        archive.writestr(link, b"target")
    index = _single_archive_index(archive_path, member_names=("000.mp4",))
    subtitle_zip = tmp_path / "subtitle.zip"
    _subtitle_fixture(subtitle_zip, ("000",))

    with pytest.raises(ValueError, match="symlink"):
        extract_selected_media(("000",), index, archive_root, tmp_path / "videos", subtitle_zip)

    clean_path = archive_root / "001.zip"
    with ZipFile(clean_path, "w", compression=ZIP_STORED) as archive:
        archive.writestr("000.mp4", b"video")
    clean_index = _single_archive_index(clean_path, member_names=("000.mp4",))
    with pytest.raises(ValueError, match="selected video IDs"):
        extract_selected_media(
            ("001",), clean_index, archive_root, tmp_path / "videos", subtitle_zip
        )


def test_extract_rejects_directory_mode_disguised_as_mp4(tmp_path: Path) -> None:
    archive_root = tmp_path / "archives"
    archive_root.mkdir()
    archive_path = archive_root / "001.zip"
    disguised = ZipInfo("000.mp4")
    disguised.external_attr = 0o040755 << 16
    with ZipFile(archive_path, "w", compression=ZIP_STORED) as archive:
        archive.writestr(disguised, b"directory-content")
    index = _single_archive_index(archive_path, member_names=("000.mp4",))
    subtitle_zip = tmp_path / "subtitle.zip"
    _subtitle_fixture(subtitle_zip, ("000",))

    with pytest.raises(ValueError, match="directory.*media"):
        extract_selected_media(
            ("000",), index, archive_root, tmp_path / "videos", subtitle_zip
        )


def test_extract_crc_failure_never_replaces_verified_output(tmp_path: Path) -> None:
    archive_root = tmp_path / "archives"
    archive_root.mkdir()
    archive_path = archive_root / "001.zip"
    payload = b"crc-protected-payload"
    with ZipFile(archive_path, "w", compression=ZIP_STORED) as archive:
        archive.writestr("000.mp4", payload)
    raw = bytearray(archive_path.read_bytes())
    payload_offset = raw.index(payload)
    raw[payload_offset] ^= 0xFF
    archive_path.write_bytes(raw)
    index = _single_archive_index(archive_path, member_names=("000.mp4",))
    subtitle_zip = tmp_path / "subtitle.zip"
    _subtitle_fixture(subtitle_zip, ("000",))
    video_root = tmp_path / "videos"
    video_root.mkdir()
    final = video_root / "000.mp4"
    final.write_bytes(b"last-verified-output")

    with pytest.raises(ValueError, match="CRC"):
        extract_selected_media(("000",), index, archive_root, video_root, subtitle_zip)

    assert final.read_bytes() == b"last-verified-output"


def test_prepare_check_plans_exact_pilot_and_full_scopes_without_downloading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    index = _full_population_archive_index(_even_archive_members())
    metadata = SimpleNamespace(
        video_ids=OFFICIAL_VIDEO_IDS, report=SimpleNamespace(metadata_sha256="2" * 64)
    )
    monkeypatch.setattr(
        "fidmem.assets.videomme_v2.load_official_archive_identities",
        lambda: index.archives,
    )
    monkeypatch.setattr(
        "fidmem.assets.videomme_v2.build_archive_index", lambda *_args, **_kwargs: index
    )
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=100 * 1024**3, used=0, free=100 * 1024**3),
    )

    pilot = prepare_videos(
        metadata,
        tmp_path / "pilot-raw",
        tmp_path / "pilot-cache",
        subtitle_zip=tmp_path / "subtitle.zip",
        scope="pilot",
        check=True,
        resume=False,
        verify_only=False,
    )
    full = prepare_videos(
        metadata,
        tmp_path / "full-raw",
        tmp_path / "full-cache",
        subtitle_zip=tmp_path / "subtitle.zip",
        scope="full",
        check=True,
        resume=False,
        verify_only=False,
    )

    expected_pilot = select_pilot(metadata, index)
    assert pilot.status == "CHECKED"
    assert pilot.archive_index.archive_index_sha256 == index.archive_index_sha256
    assert pilot.plan.video_ids == expected_pilot.selected_video_ids
    assert tuple(item.path for item in pilot.plan.archives) == expected_pilot.selected_archive_paths
    assert len(full.plan.archives) == 40
    assert len(full.plan.video_ids) == 800
    assert full.selection is None


def test_prepare_capacity_fails_before_downloader_is_invoked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    index = _full_population_archive_index(_even_archive_members())
    metadata = SimpleNamespace(
        video_ids=OFFICIAL_VIDEO_IDS, report=SimpleNamespace(metadata_sha256="3" * 64)
    )
    monkeypatch.setattr(
        "fidmem.assets.videomme_v2.load_official_archive_identities",
        lambda: index.archives,
    )
    monkeypatch.setattr(
        "fidmem.assets.videomme_v2.build_archive_index", lambda *_args, **_kwargs: index
    )
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=1, used=0, free=1),
    )

    def forbidden_download(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("capacity must be checked before download")

    monkeypatch.setattr(
        "fidmem.assets.videomme_v2.download_pinned_file", forbidden_download
    )
    with pytest.raises(ValueError, match="insufficient.*space"):
        prepare_videos(
            metadata,
            tmp_path / "raw",
            tmp_path / "cache",
            subtitle_zip=tmp_path / "subtitle.zip",
            scope="pilot",
            check=False,
            resume=True,
            verify_only=False,
        )


@pytest.mark.parametrize("partial_kind", ["symlink", "complete_hash_mismatch"])
def test_prepare_check_rejects_unsafe_or_invalid_existing_partial(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, partial_kind: str
) -> None:
    index = _full_population_archive_index(_even_archive_members())
    metadata = SimpleNamespace(
        video_ids=OFFICIAL_VIDEO_IDS, report=SimpleNamespace(metadata_sha256="6" * 64)
    )
    selection = select_pilot(metadata, index)
    first_archive = next(
        archive
        for archive in index.archives
        if archive.path == selection.selected_archive_paths[0]
    )
    raw_root = tmp_path / "raw"
    archive_root = raw_root / "archives"
    archive_root.mkdir(parents=True)
    partial = archive_root / f"{Path(first_archive.path).stem}.partial"
    if partial_kind == "symlink":
        outside = tmp_path / "outside-partial"
        outside.write_bytes(b"x")
        partial.symlink_to(outside)
        message = "partial archive is unsafe"
    else:
        partial.write_bytes(b"x" * first_archive.size)
        message = "completed partial archive SHA-256"
    monkeypatch.setattr(
        "fidmem.assets.videomme_v2.load_official_archive_identities",
        lambda: index.archives,
    )
    monkeypatch.setattr(
        "fidmem.assets.videomme_v2.build_archive_index", lambda *_args, **_kwargs: index
    )
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=100 * 1024**3, used=0, free=100 * 1024**3),
    )

    with pytest.raises(ValueError, match=message):
        prepare_videos(
            metadata,
            raw_root,
            tmp_path / "cache",
            subtitle_zip=tmp_path / "subtitle.zip",
            scope="pilot",
            check=True,
            resume=False,
            verify_only=False,
        )


def test_prepare_verify_only_uses_recorded_plan_and_never_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    index = _full_population_archive_index(_even_archive_members())
    metadata = SimpleNamespace(
        video_ids=OFFICIAL_VIDEO_IDS, report=SimpleNamespace(metadata_sha256="4" * 64)
    )
    monkeypatch.setattr(
        "fidmem.assets.videomme_v2.load_official_archive_identities",
        lambda: index.archives,
    )
    monkeypatch.setattr(
        "fidmem.assets.videomme_v2.build_archive_index", lambda *_args, **_kwargs: index
    )
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=100 * 1024**3, used=0, free=100 * 1024**3),
    )
    raw_root = tmp_path / "raw"
    cache_root = tmp_path / "cache"
    prepare_videos(
        metadata,
        raw_root,
        cache_root,
        subtitle_zip=tmp_path / "subtitle.zip",
        scope="pilot",
        check=True,
        resume=False,
        verify_only=False,
    )

    def forbidden_network(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("verify-only must not use the network")

    monkeypatch.setattr(
        "fidmem.assets.videomme_v2.load_official_archive_identities", forbidden_network
    )
    monkeypatch.setattr(
        "fidmem.assets.videomme_v2.open_official_archive", forbidden_network
    )
    with pytest.raises(ValueError, match="missing.*archive"):
        prepare_videos(
            metadata,
            raw_root,
            cache_root,
            subtitle_zip=tmp_path / "subtitle.zip",
            scope="pilot",
            check=False,
            resume=False,
            verify_only=True,
        )


def test_prepare_resume_uses_recorded_plan_without_reindexing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    index = _full_population_archive_index(_even_archive_members())
    metadata = SimpleNamespace(
        video_ids=OFFICIAL_VIDEO_IDS, report=SimpleNamespace(metadata_sha256="7" * 64)
    )
    monkeypatch.setattr(
        "fidmem.assets.videomme_v2.load_official_archive_identities",
        lambda: index.archives,
    )
    monkeypatch.setattr(
        "fidmem.assets.videomme_v2.build_archive_index", lambda *_args, **_kwargs: index
    )
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=100 * 1024**3, used=0, free=100 * 1024**3),
    )
    raw_root = tmp_path / "raw"
    cache_root = tmp_path / "cache"
    subtitle_zip = tmp_path / "subtitle.zip"
    prepare_videos(
        metadata,
        raw_root,
        cache_root,
        subtitle_zip=subtitle_zip,
        scope="pilot",
        check=True,
        resume=False,
        verify_only=False,
    )

    def forbidden_network(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("resume must reuse the recorded archive index")

    monkeypatch.setattr(
        "fidmem.assets.videomme_v2.load_official_archive_identities", forbidden_network
    )
    monkeypatch.setattr(
        "fidmem.assets.videomme_v2.build_archive_index", forbidden_network
    )
    monkeypatch.setattr(
        "fidmem.assets.videomme_v2.download_pinned_file",
        lambda _identity, destination, **_kwargs: Path(destination),
    )
    monkeypatch.setattr(
        "fidmem.assets.videomme_v2.extract_selected_media",
        lambda video_ids, _index, _archives, video_root, _subtitles: tuple(
            Path(video_root) / f"{video_id}.mp4" for video_id in video_ids
        ),
    )

    result = prepare_videos(
        metadata,
        raw_root,
        cache_root,
        subtitle_zip=subtitle_zip,
        scope="pilot",
        check=False,
        resume=True,
        verify_only=False,
    )

    assert result.status == "PREPARED"
    assert result.archive_index == index


def test_prepare_verify_only_rejects_plan_identity_tampering_before_media_use(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    index = _full_population_archive_index(_even_archive_members())
    metadata = SimpleNamespace(
        video_ids=OFFICIAL_VIDEO_IDS, report=SimpleNamespace(metadata_sha256="5" * 64)
    )
    monkeypatch.setattr(
        "fidmem.assets.videomme_v2.load_official_archive_identities",
        lambda: index.archives,
    )
    monkeypatch.setattr(
        "fidmem.assets.videomme_v2.build_archive_index", lambda *_args, **_kwargs: index
    )
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=100 * 1024**3, used=0, free=100 * 1024**3),
    )
    raw_root = tmp_path / "raw"
    cache_root = tmp_path / "cache"
    prepare_videos(
        metadata,
        raw_root,
        cache_root,
        subtitle_zip=tmp_path / "subtitle.zip",
        scope="pilot",
        check=True,
        resume=False,
        verify_only=False,
    )
    state_path = cache_root / "videomme-v2-pilot-download-plan.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["plan"]["archives"][0]["upstream_sha256"] = "0" * 64
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ValueError, match="download plan.*archive index"):
        prepare_videos(
            metadata,
            raw_root,
            cache_root,
            subtitle_zip=tmp_path / "subtitle.zip",
            scope="pilot",
            check=False,
            resume=False,
            verify_only=True,
        )
