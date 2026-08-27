from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from fidmem.experiments.observation_import import ObservationImportRecord
from fidmem.production.generation import GenerationStore
from fidmem.production.authority import canonical_sha256, seal_authority
from fidmem.production.provenance import AuthorityBoundCache
from fidmem.production.observation_import import (
    ProductionObservationImportRecord,
    import_production_observations,
)
from tests.experiments.test_observation_import import _payload, _write_jsonl
from fidmem.storage.cache import ContentAddressedCache
from tests.production.helpers import complete_draft, fake_gpu_runtime, fake_repository


def test_production_uses_the_canonical_observation_record_schema() -> None:
    assert ProductionObservationImportRecord is ObservationImportRecord


def _sealed_authority(tmp_path: Path):
    path = tmp_path / "authority.json"
    sealed = seal_authority(
        complete_draft(tmp_path),
        output_path=path,
        project_root=tmp_path,
        runtime_probe=fake_gpu_runtime,
        repository_probe=fake_repository,
    )
    return path, sealed


def _production_payload(authority) -> dict[str, object]:
    payload = _payload()
    payload["provider_identity"] = {
        "provider": "approved-provider",
        "model_revision": "0123456789abcdef0123456789abcdef01234567",
        "decode_config": {"temperature": 0.0, "max_tokens": 128},
        "device_name": "cuda:0",
    }
    raw_response = {"request_id": "provider-request-1", "text": "real output"}
    payload.update(
        {
            "evidence_class": "production",
            "authority_sha256": authority.authority_sha256,
            "model_id": "org/residual-model",
            "config_sha256": canonical_sha256(
                authority.observation_configurations.model_dump(mode="json")
            ),
            "raw_response": raw_response,
            "raw_response_sha256": canonical_sha256(raw_response),
        }
    )
    return payload


def test_production_import_rejects_engineering_record(tmp_path: Path) -> None:
    authority_path, _ = _sealed_authority(tmp_path)
    source = tmp_path / "provider.jsonl"
    _write_jsonl(source, [_payload()])

    with pytest.raises(ValueError, match="production record.*authority"):
        import_production_observations(
            source,
            tmp_path / "run",
            authority_path=authority_path,
            resume=False,
        )


def test_production_artifacts_are_bound_and_resume_is_byte_stable(
    tmp_path: Path,
) -> None:
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
    names = (
        "observations.jsonl",
        "state.json",
        "cost.csv",
        "summary.json",
        "manifest.json",
        "cache_manifest.json",
        "COMMITTED.json",
    )
    active = GenerationStore(destination, authority.authority_sha256).current_path()
    before = {name: (active / name).read_bytes() for name in names}

    assert result["authority_sha256"] == authority.authority_sha256
    assert set(result["artifacts"]) == {
        "observations",
        "costs",
        "summary",
        "manifest",
        "cache_manifest",
        "state",
        "committed",
    }
    summary = json.loads((active / "summary.json").read_text(encoding="utf-8"))
    assert summary["evidence_class"] == "production"
    assert summary["authority_sha256"] == authority.authority_sha256
    assert summary["total_gpu_seconds"] == 0.25
    with (active / "cost.csv").open("r", encoding="utf-8", newline="") as stream:
        costs = list(csv.DictReader(stream))
    assert {row["authority_sha256"] for row in costs} == {authority.authority_sha256}
    assert {row["evidence_class"] for row in costs} == {"production"}

    cache_manifest = json.loads(
        (active / "cache_manifest.json").read_text(encoding="utf-8")
    )
    cache_key = cache_manifest["entries"][0]["cache_key"]
    bound_cache = AuthorityBoundCache(ContentAddressedCache(result["cache_root"]))
    cached = bound_cache.get_bound(
        cache_key, expected_authority_sha256=authority.authority_sha256
    )
    canonical_row = json.loads(
        (active / "observations.jsonl").read_text(encoding="utf-8")
    )
    assert cached["record_id"] == canonical_row["record_id"]

    resumed = import_production_observations(
        source,
        destination,
        authority_path=authority_path,
        resume=True,
        run_id="R002-canary",
    )

    assert resumed["cache_hits"] == 1
    assert resumed["cache_misses"] == 0
    assert resumed["cache_root"] == result["cache_root"]
    assert {name: (active / name).read_bytes() for name in names} == before


def test_resume_rejects_different_authority_before_writes(tmp_path: Path) -> None:
    first_root = tmp_path / "authority-a"
    second_root = tmp_path / "authority-b"
    first_root.mkdir()
    second_root.mkdir()
    first_path, first = _sealed_authority(first_root)
    second_draft = complete_draft(second_root)
    content = "Different frozen prompt."
    changed = second_draft.prompts.residual_generation.model_copy(
        update={
            "content": content,
            "sha256": hashlib.sha256(content.encode()).hexdigest(),
        }
    )
    second_draft = second_draft.model_copy(
        update={
            "prompts": second_draft.prompts.model_copy(
                update={"residual_generation": changed}
            )
        }
    )
    second_path = second_root / "authority.json"
    seal_authority(
        second_draft,
        output_path=second_path,
        project_root=second_root,
        runtime_probe=fake_gpu_runtime,
        repository_probe=fake_repository,
    )
    source = tmp_path / "provider.jsonl"
    destination = tmp_path / "run"
    _write_jsonl(source, [_production_payload(first)])
    import_production_observations(
        source,
        destination,
        authority_path=first_path,
        resume=False,
    )
    store = GenerationStore(destination, first.authority_sha256)
    active = store.current_path()
    before_pointer = store.pointer_path.read_bytes()
    before = {path.name: path.read_bytes() for path in active.iterdir()}

    with pytest.raises(ValueError, match="different Authority"):
        import_production_observations(
            source,
            destination,
            authority_path=second_path,
            resume=True,
        )

    assert store.pointer_path.read_bytes() == before_pointer
    assert {path.name: path.read_bytes() for path in active.iterdir()} == before
