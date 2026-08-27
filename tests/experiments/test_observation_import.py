from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from fidmem.actions.environment import ActionObservation, OperationMetadata
from fidmem.costs.tracker import CostRecord
from fidmem.types import ActionInstance, ActionType, RouterState
from fidmem.experiments.observation_import import (
    ObservationImportRecord,
    ProviderIdentity,
    import_observations,
)


def _state() -> RouterState:
    return RouterState(
        question="What color?",
        options=("blue", "red"),
        evidence=(),
        action_history=(),
        remaining_budget=20,
        candidate_event_ids=(),
        candidate_fidelity_levels={},
        context_frontiers={},
        cost_preference=0,
    )


def _cost(*, gpu_seconds: float = 0.25) -> CostRecord:
    return CostRecord(
        operation="residual",
        gpu_seconds=gpu_seconds,
        wall_seconds=0.5,
        input_frames=12,
        visual_tokens=24,
        text_tokens=8,
        peak_memory_bytes=1024,
        cache_status="miss",
        device_name="cuda:0",
    )


def _payload(
    *,
    with_cost: bool = True,
    question_id: str = "q1",
    event_id: str = "e1",
    gpu_seconds: float = 0.25,
) -> dict[str, object]:
    metadata = OperationMetadata(
        scope="residual",
        cache_status="miss",
        amortizable=True,
        cost_record=_cost(gpu_seconds=gpu_seconds) if with_cost else None,
    )
    return {
        "schema_version": 1,
        "question_id": question_id,
        "video_id": "v1",
        "provider_identity": {
            "provider": "local-fixture",
            "model_revision": "model-v1",
            "decode_config": {"temperature": 0},
            "device_name": "cuda:0",
        },
        "state": _state().model_dump(mode="json"),
        "action": ActionInstance(ActionType.EXPAND_RESIDUAL, event_id, None).model_dump(
            mode="json"
        ),
        "observation": ActionObservation(
            action_type=ActionType.EXPAND_RESIDUAL,
            target_event_id=event_id,
            operation_metadata=(metadata,),
        ).model_dump(mode="json"),
    }


def _write_jsonl(path: Path, payloads: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(payload, ensure_ascii=False) + "\n" for payload in payloads),
        encoding="utf-8",
    )


def test_record_accepts_authoritative_measured_observation() -> None:
    record = ObservationImportRecord.model_validate(_payload())
    assert record.schema_version == 1
    assert record.evidence_class == "engineering"
    assert record.authority_sha256 is None
    assert record.provider_identity.provider == "local-fixture"
    assert len(record.cost_records) == 1
    assert len(record.record_id) == 64


def test_engineering_record_rejects_authority_hash() -> None:
    payload = _payload()
    payload["authority_sha256"] = "a" * 64

    with pytest.raises(ValueError, match="engineering.*Authority"):
        ObservationImportRecord.model_validate(payload)


def test_record_rejects_non_stop_without_authoritative_cost() -> None:
    with pytest.raises(ValueError, match="authoritative cost"):
        ObservationImportRecord.model_validate(_payload(with_cost=False))


def test_import_writes_canonical_records_and_resume_is_byte_stable(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "provider.jsonl"
    output_dir = tmp_path / "run"
    _write_jsonl(
        input_path,
        [
            _payload(question_id="q1", event_id="e1"),
            _payload(question_id="q2", event_id="e2", gpu_seconds=0.75),
        ],
    )

    first = import_observations(input_path, output_dir, resume=False)
    observations_path = output_dir / "observations.jsonl"
    first_bytes = observations_path.read_bytes()
    rows = [json.loads(line) for line in first_bytes.decode("utf-8").splitlines()]
    assert not (output_dir.parent / "production-cache").exists()

    assert first["cache_hits"] == 0
    assert first["cache_misses"] == 2
    assert [row["question_id"] for row in rows] == ["q1", "q2"]
    assert all(len(row["record_id"]) == 64 for row in rows)

    resumed = import_observations(input_path, output_dir, resume=True)

    assert resumed["cache_hits"] == 2
    assert resumed["cache_misses"] == 0
    assert observations_path.read_bytes() == first_bytes


def test_resume_rejects_existing_record_id_with_changed_content(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "provider.jsonl"
    output_dir = tmp_path / "run"
    _write_jsonl(input_path, [_payload()])
    import_observations(input_path, output_dir, resume=False)
    observations_path = output_dir / "observations.jsonl"
    existing = json.loads(observations_path.read_text(encoding="utf-8"))
    existing["question_id"] = "tampered-question"
    observations_path.write_text(
        json.dumps(existing, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="record_id.*content"):
        import_observations(input_path, output_dir, resume=True)


def test_import_emits_measured_cost_summary_and_manifest_artifacts(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "provider.jsonl"
    config_path = tmp_path / "pilot.yaml"
    output_dir = tmp_path / "run"
    config_bytes = b"dataset: pilot\n"
    config_path.write_bytes(config_bytes)
    _write_jsonl(
        input_path,
        [
            _payload(question_id="q1", event_id="e1", gpu_seconds=0.25),
            _payload(question_id="q2", event_id="e2", gpu_seconds=0.75),
        ],
    )
    input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()

    result = import_observations(
        input_path,
        output_dir,
        resume=False,
        run_id="R002",
        config_path=config_path,
    )

    assert set(result["artifacts"]) == {
        "observations",
        "costs",
        "summary",
        "manifest",
    }
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary == {
        "schema_version": 1,
        "record_count": 2,
        "cost_record_count": 2,
        "cache_hits": 0,
        "cache_misses": 2,
        "total_gpu_seconds": 1.0,
        "p90_gpu_seconds": 0.75,
        "total_wall_seconds": 1.0,
        "total_input_frames": 24,
        "total_visual_tokens": 48,
        "total_text_tokens": 16,
        "peak_memory_bytes": 1024,
    }

    with (output_dir / "cost.csv").open("r", encoding="utf-8", newline="") as stream:
        cost_rows = list(csv.DictReader(stream))
    assert len(cost_rows) == 2
    assert {row["question_id"] for row in cost_rows} == {"q1", "q2"}
    assert {row["gpu_seconds"] for row in cost_rows} == {"0.25", "0.75"}
    assert all(row["provider"] == "local-fixture" for row in cost_rows)
    assert all(row["cache_status"] == "miss" for row in cost_rows)

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["run_id"] == "R002"
    assert manifest["input_path"] == str(input_path.resolve())
    assert manifest["input_sha256"] == input_hash
    assert manifest["config_path"] == str(config_path.resolve())
    assert manifest["config_sha256"] == hashlib.sha256(config_bytes).hexdigest()
    assert manifest["provider_identities"] == [
        {
            "provider": "local-fixture",
            "model_revision": "model-v1",
            "decode_config": {"temperature": 0},
            "device_name": "cuda:0",
        }
    ]
    assert manifest["artifacts"] == {
        "observations": str((output_dir / "observations.jsonl").resolve()),
        "costs": str((output_dir / "cost.csv").resolve()),
        "summary": str((output_dir / "summary.json").resolve()),
        "manifest": str((output_dir / "manifest.json").resolve()),
    }


def test_invalid_import_preserves_existing_artifacts(tmp_path: Path) -> None:
    input_path = tmp_path / "provider.jsonl"
    config_path = tmp_path / "pilot.yaml"
    output_dir = tmp_path / "run"
    config_path.write_text("dataset: pilot\n", encoding="utf-8")
    _write_jsonl(input_path, [_payload()])
    import_observations(
        input_path,
        output_dir,
        resume=False,
        run_id="R002",
        config_path=config_path,
    )
    artifact_names = (
        "observations.jsonl",
        "cost.csv",
        "summary.json",
        "manifest.json",
    )
    before = {name: (output_dir / name).read_bytes() for name in artifact_names}
    _write_jsonl(input_path, [_payload(with_cost=False)])

    with pytest.raises(ValueError, match="authoritative cost"):
        import_observations(
            input_path,
            output_dir,
            resume=False,
            run_id="R002",
            config_path=config_path,
        )

    assert {name: (output_dir / name).read_bytes() for name in artifact_names} == before


def test_record_rejects_empty_decode_configuration() -> None:
    payload = _payload()
    provider = dict(payload["provider_identity"])
    provider["decode_config"] = {}
    payload["provider_identity"] = provider

    with pytest.raises(ValueError, match="decode_config"):
        ObservationImportRecord.model_validate(payload)


@pytest.mark.parametrize("field", ("provider", "model_revision", "device_name"))
def test_record_rejects_blank_provider_identity(field: str) -> None:
    payload = _payload()
    provider = dict(payload["provider_identity"])
    provider[field] = "   "
    payload["provider_identity"] = provider

    with pytest.raises(ValueError, match=field):
        ObservationImportRecord.model_validate(payload)
