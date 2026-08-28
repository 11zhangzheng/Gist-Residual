from __future__ import annotations

from pathlib import Path

import pytest

from fidmem.actions.environment import ActionObservation, OperationMetadata
from fidmem.costs.tracker import CostRecord
from fidmem.production.canary import validate_production_run
from fidmem.production.authority import canonical_json, canonical_sha256
from fidmem.production.manifests import (
    QuestionManifest,
    SelectionManifest,
    VideoManifest,
)
from fidmem.experiments.observation_import import (
    ObservationImportRecord,
    _cache_manifest,
    import_production_observations,
)
from fidmem.types import ActionInstance, ActionType
from tests.production.test_observation_provenance import (
    _production_payload,
    _sealed_authority,
    _write_jsonl,
)


def _visual_record(authority, question_id: str) -> ObservationImportRecord:
    payload = _production_payload(authority)
    payload["question_id"] = question_id
    payload["model_id"] = "org/visual-model"
    payload["action"] = ActionInstance(
        ActionType.VERIFY_VISUAL, "e1", "low"
    ).model_dump(mode="json")

    def cost(scope: str, cache_status: str) -> CostRecord:
        return CostRecord(
            operation=scope,
            gpu_seconds=0.1,
            wall_seconds=0.2,
            input_frames=12 if scope == "event_observation" else 0,
            visual_tokens=4,
            text_tokens=2,
            peak_memory_bytes=2048,
            cache_status=cache_status,
            device_name="cuda:0",
        )

    payload["observation"] = ActionObservation(
        action_type=ActionType.VERIFY_VISUAL,
        target_event_id="e1",
        operation_metadata=(
            OperationMetadata(
                scope="event_observation",
                cache_status="miss",
                amortizable=True,
                input_frames=12,
                cost_record=cost("event_observation", "miss"),
            ),
            OperationMetadata(
                scope="question_verification",
                cache_status="miss",
                amortizable=False,
                cost_record=cost("question_verification", "miss"),
            ),
        ),
    ).model_dump(mode="json")
    return ObservationImportRecord.model_validate(payload)


def test_visual_cache_reuses_event_level_but_isolates_question_level(tmp_path) -> None:
    _, authority = _sealed_authority(tmp_path)
    manifest = _cache_manifest(
        [_visual_record(authority, "q1"), _visual_record(authority, "q2")],
        authority.authority_sha256,
    )
    event_entries = [
        item for item in manifest["entries"] if item["scope"] == "event_observation"
    ]
    question_entries = [
        item for item in manifest["entries"] if item["scope"] == "question_verification"
    ]

    assert len({item["cache_key"] for item in event_entries}) == 1
    assert {item["question_id"] for item in event_entries} == {None}
    assert len({item["cache_key"] for item in question_entries}) == 2
    assert {item["question_id"] for item in question_entries} == {"q1", "q2"}


def test_validation_report_reconciles_raw_cost_and_provenance(tmp_path) -> None:
    authority_path, authority = _sealed_authority(tmp_path)
    source = tmp_path / "provider.jsonl"
    destination = tmp_path / "run"
    _write_jsonl(source, [_production_payload(authority)])
    import_production_observations(
        source,
        destination,
        authority_path=authority_path,
        resume=False,
        run_id="R002-canary",
    )

    report = validate_production_run(destination, authority_path=authority_path)

    assert report["production_provenance_valid"] is True
    assert report["cost_reconciliation_passed"] is True
    assert report["raw_cost_record_count"] == 1
    assert report["atomic_observation_count"] == 1
    assert report["missing_observation_count"] == 0
    assert report["duplicate_collision_count"] == 0
    assert report["provider_model_device_identity_consistent"] is True
    assert report["cross_question_cache_isolation_valid"] is True


def test_validation_fails_closed_when_cache_envelope_is_missing(tmp_path) -> None:
    authority_path, authority = _sealed_authority(tmp_path)
    source = tmp_path / "provider.jsonl"
    destination = tmp_path / "run"
    _write_jsonl(source, [_production_payload(authority)])
    result = import_production_observations(
        source,
        destination,
        authority_path=authority_path,
        resume=False,
        run_id="R002-canary",
    )
    cache_files = tuple(Path(result["cache_root"]).glob("*.json"))
    assert len(cache_files) == 1
    cache_files[0].unlink()

    with pytest.raises(ValueError, match="Authority-bound cache envelope is missing"):
        validate_production_run(destination, authority_path=authority_path)


def test_e03_selection_manifest_must_contain_ten_to_twenty_questions(tmp_path) -> None:
    authority_path, authority = _sealed_authority(tmp_path)
    source = tmp_path / "provider.jsonl"
    destination = tmp_path / "run"
    _write_jsonl(source, [_production_payload(authority)])
    import_production_observations(
        source,
        destination,
        authority_path=authority_path,
        resume=False,
        run_id="R002-canary",
    )
    questions = QuestionManifest.model_validate_json(
        (tmp_path / authority.dataset.question_manifest_path).read_text(
            encoding="utf-8"
        )
    )
    videos = VideoManifest.model_validate_json(
        (tmp_path / authority.dataset.video_manifest_path).read_text(encoding="utf-8")
    )
    payload = {
        "schema_version": 1,
        "group": "canary",
        "seed": "engineering-fixture",
        "source_video_manifest_sha256": videos.manifest_sha256,
        "source_question_manifest_sha256": questions.manifest_sha256,
        "question_ids": ("q1",),
        "video_ids": ("v1",),
    }
    selection = SelectionManifest(**payload, selection_sha256=canonical_sha256(payload))
    selection_path = tmp_path / "canary-selection.json"
    selection_path.write_text(
        canonical_json(selection.model_dump(mode="json")), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="10-20 questions"):
        validate_production_run(
            destination,
            authority_path=authority_path,
            selection_manifest_path=selection_path,
        )
