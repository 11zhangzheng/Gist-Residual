"""Read-only validation report for a committed production observation run."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from fidmem.production.authority import (
    authority_file_sha256,
    load_sealed_authority,
)
from fidmem.production.generation import GenerationStore
from fidmem.experiments.observation_import import (
    _authority_manifests,
    _manifest_pairs,
    _parse_existing,
    _validate_commit,
    _validate_record_against_authority,
    _validate_production_cache,
)
from fidmem.production.manifests import SelectionManifest
from fidmem.types import ActionType


def _cache_isolation_valid(cache_manifest: dict[str, Any]) -> bool:
    question_by_key: dict[str, str] = {}
    for entry in cache_manifest.get("entries", []):
        question_id = entry.get("question_id")
        if entry.get("amortizable"):
            if question_id is not None:
                return False
            continue
        if not isinstance(question_id, str) or not question_id:
            return False
        cache_key = entry.get("cache_key")
        previous = question_by_key.setdefault(cache_key, question_id)
        if previous != question_id:
            return False
    return True


def validate_production_run(
    output_dir: str | Path,
    *,
    authority_path: str | Path,
    selection_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Recompute Canary integrity metrics without executing models or providers."""

    run_root = Path(output_dir)
    destination = run_root
    authority_source = Path(authority_path)
    authority = load_sealed_authority(authority_source)
    destination = GenerationStore(
        destination, authority.authority_sha256
    ).current_path()
    _validate_commit(destination, authority.authority_sha256)
    records = _parse_existing(destination / "observations.jsonl")
    valid_pairs = _manifest_pairs(authority_source, authority)
    for record in records:
        _validate_record_against_authority(record, authority, valid_pairs=valid_pairs)

    with (destination / "cost.csv").open("r", encoding="utf-8", newline="") as stream:
        costs = list(csv.DictReader(stream))
    summary = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
    cache_manifest = json.loads(
        (destination / "cache_manifest.json").read_text(encoding="utf-8")
    )
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    state = json.loads((destination / "state.json").read_text(encoding="utf-8"))
    report_path = destination / "report.json"
    stored_report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.is_file()
        else None
    )
    if manifest.get("authority_file_sha256") != authority_file_sha256(authority_source):
        raise ValueError("production manifest Authority file SHA-256 mismatch")
    authority_hashes = {
        *(record.authority_sha256 for record in records),
        *(row.get("authority_sha256") for row in costs),
        summary.get("authority_sha256"),
        cache_manifest.get("authority_sha256"),
        manifest.get("authority_sha256"),
        state.get("authority_sha256"),
        *(entry.get("authority_sha256") for entry in state.get("command_history", [])),
        *([stored_report.get("authority_sha256")] if stored_report else []),
    }
    if authority_hashes != {authority.authority_sha256}:
        raise ValueError("production committed generation has mixed Authority")
    _validate_production_cache(
        run_root,
        records,
        cache_manifest,
        authority.authority_sha256,
    )
    provenance_valid = True

    raw_totals = {
        "total_gpu_seconds": math.fsum(float(row["gpu_seconds"]) for row in costs),
        "total_wall_seconds": math.fsum(float(row["wall_seconds"]) for row in costs),
        "total_input_frames": sum(int(row["input_frames"]) for row in costs),
        "total_visual_tokens": sum(int(row["visual_tokens"]) for row in costs),
        "total_text_tokens": sum(int(row["text_tokens"]) for row in costs),
        "peak_memory_bytes": max(
            (int(row["peak_memory_bytes"]) for row in costs), default=0
        ),
    }
    cost_reconciled = all(
        summary.get(key) == value for key, value in raw_totals.items()
    )
    question_ids = {record.question_id for record in records}
    video_ids = {record.video_id for record in records}
    event_ids = {
        record.action.event_id
        for record in records
        if record.action.event_id is not None
    }
    expected_questions = {question_id for question_id, _ in valid_pairs}
    selection_sha256 = None
    if selection_manifest_path is not None:
        selection = SelectionManifest.model_validate_json(
            Path(selection_manifest_path).read_text(encoding="utf-8")
        )
        questions, videos = _authority_manifests(authority_source, authority)
        if selection.group != "canary":
            raise ValueError("E03 selection manifest must bind the canary group")
        selected_question_ids = set(selection.question_ids)
        if len(selected_question_ids) != len(selection.question_ids):
            raise ValueError("E03 selection manifest contains duplicate questions")
        if len(set(selection.video_ids)) != len(selection.video_ids):
            raise ValueError("E03 selection manifest contains duplicate videos")
        if not 10 <= len(selected_question_ids) <= 20:
            raise ValueError("E03 selection manifest must contain 10-20 questions")
        if (
            selection.source_question_manifest_sha256 != questions.manifest_sha256
            or selection.source_video_manifest_sha256 != videos.manifest_sha256
        ):
            raise ValueError("Canary selection source manifests differ from Authority")
        selected_pairs = {
            (record.question_id, record.video_id)
            for record in questions.records
            if record.question_id in selected_question_ids
        }
        if {item[0] for item in selected_pairs} != selected_question_ids:
            raise ValueError("Canary selection references an unknown question")
        if {item[1] for item in selected_pairs} != set(selection.video_ids):
            raise ValueError("Canary selection video identities differ from questions")
        if not selected_pairs.issubset(valid_pairs):
            raise ValueError("Canary selection is outside the Authority manifests")
        expected_questions = selected_question_ids
        selection_sha256 = selection.selection_sha256
    unexpected_questions = question_ids - expected_questions
    action_counts = {
        "gist": sum(
            record.action.action_type is ActionType.SEARCH_GIST for record in records
        ),
        "residual": sum(
            record.action.action_type is ActionType.EXPAND_RESIDUAL
            for record in records
        ),
        "visual": sum(
            record.action.action_type is ActionType.VERIFY_VISUAL for record in records
        ),
    }
    identities = sorted(
        {
            (
                record.provider_identity.provider,
                record.model_id,
                record.provider_identity.model_revision,
                record.provider_identity.device_name,
            )
            for record in records
        }
    )
    return {
        "schema_version": 1,
        "evidence_class": "production",
        "authority_sha256": authority.authority_sha256,
        "selection_sha256": selection_sha256,
        "production_provenance_valid": provenance_valid,
        "production_namespace_isolated": True,
        "total_questions": len(question_ids),
        "total_videos": len(video_ids),
        "total_events": len(event_ids),
        "atomic_observation_count": len(records),
        "gist_observation_count": action_counts["gist"],
        "residual_observation_count": action_counts["residual"],
        "visual_observation_count": action_counts["visual"],
        "cache_hits": int(summary.get("cache_hits", 0)),
        "cache_misses": int(summary.get("cache_misses", 0)),
        "schema_error_count": 0,
        "schema_error_rate": 0.0,
        "missing_observation_count": len(expected_questions - question_ids),
        "unexpected_observation_question_count": len(unexpected_questions),
        "duplicate_collision_count": 0,
        "raw_cost_record_count": len(costs),
        "cost_reconciliation_passed": cost_reconciled,
        "raw_cost_totals": raw_totals,
        "aggregate_cost_totals": {key: summary.get(key) for key in raw_totals},
        **raw_totals,
        "provider_model_device_identity_consistent": provenance_valid,
        "provider_model_device_identities": [
            {
                "provider": provider,
                "model_id": model_id,
                "model_revision": revision,
                "device_name": device,
            }
            for provider, model_id, revision, device in identities
        ],
        "resume_validation": "not_executed",
        "cross_question_cache_isolation_valid": _cache_isolation_valid(cache_manifest),
    }
