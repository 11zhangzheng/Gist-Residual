"""Engineering-fixture tests for the pinned Video-MME-v2 metadata adapter."""

from __future__ import annotations

import json
from io import BufferedReader
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import duckdb
import pytest

from fidmem.assets.videomme_v2 import (
    FROZEN_REVISION,
    HumanAuditManifest,
    METADATA_FILES,
    OfficialFileIdentity,
    POOL_ALGORITHM,
    POOL_SEED,
    build_human_audit_manifest,
    build_archive_index,
    full_scope_media,
    select_pilot,
    validate_human_audit_result,
    verify_metadata,
)


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
        for video in range(videos)
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
        for video_id in sorted(video_ids):
            archive.writestr(f"{video_id}.jsonl", json.dumps({"text": "fixture"}) + "\n")
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


def test_human_audit_is_pending_and_binds_exact_selected_question_ids(
    tmp_path: Path,
) -> None:
    root = tmp_path / "metadata"
    _metadata(root)
    parsed = _verify(root)

    audit = build_human_audit_manifest(
        parsed,
        selected_video_ids=tuple(f"{item:03d}" for item in range(25)),
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
        selected_video_ids=tuple(f"{item:03d}" for item in range(25)),
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
        selected_video_ids=tuple(f"{item:03d}" for item in range(25)),
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
        selected_video_ids=tuple(f"{item:03d}" for item in range(25)),
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
            first_video = (archive_number - 1) * members_per_archive
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
    assert index.video_ids == tuple(f"{video:03d}" for video in range(60))
    assert tuple(member.member_path for member in index.members[:20]) == tuple(
        f"{video:03d}.mp4" for video in range(20)
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
            metadata_video_ids=tuple(f"{video:03d}" for video in range(60)),
            _expected_archive_paths=tuple(identity.path for identity in identities),
        )


def test_archive_index_rejects_duplicate_members_and_nonmatching_identities(
    tmp_path: Path,
) -> None:
    archive_root = tmp_path / "archives"
    identities = _archive_fixtures(archive_root)
    with ZipFile(archive_root / "002.zip", "a", compression=ZIP_DEFLATED) as archive:
        archive.writestr("000.mp4", b"duplicate-video")
    identities = (
        identities[0],
        identities[1].model_copy(update={"size": (archive_root / "002.zip").stat().st_size}),
        identities[2],
    )

    with pytest.raises(ValueError, match="duplicate video IDs"):
        build_archive_index(
            identities,
            _fixture_opener(archive_root),
            metadata_video_ids=tuple(f"{video:03d}" for video in range(60)),
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
            metadata_video_ids=tuple(f"{video:03d}" for video in range(60)),
            _expected_archive_paths=tuple(identity.path for identity in clean_identities),
        )
    with pytest.raises(ValueError, match="archive size differs"):
        build_archive_index(
            (clean_identities[0].model_copy(update={"size": clean_identities[0].size + 1}), *clean_identities[1:]),
            _fixture_opener(clean_root),
            metadata_video_ids=tuple(f"{video:03d}" for video in range(60)),
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

    class MetadataWithoutQuestionContents:
        video_ids = metadata.video_ids
        report = metadata.report

        @property
        def questions(self) -> tuple[object, ...]:
            raise AssertionError("pilot selection must not inspect questions or answers")

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


def test_full_scope_media_has_all_pinned_archives_and_videos_without_a_selection_hash(
    tmp_path: Path,
) -> None:
    archive_root = tmp_path / "archives"
    identities = _archive_fixtures(archive_root, archives=40)
    index = build_archive_index(identities, _fixture_opener(archive_root))

    archive_paths, video_ids = full_scope_media(index)

    assert archive_paths == tuple(f"videos/{archive:03d}.zip" for archive in range(1, 41))
    assert video_ids == tuple(f"{video:03d}" for video in range(800))
    assert not hasattr((archive_paths, video_ids), "selection_sha256")
