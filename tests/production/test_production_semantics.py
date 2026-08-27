from __future__ import annotations

import json

from fidmem.production.authority import seal_authority
from fidmem.production.generation import GenerationStore
from fidmem.experiments.observation_import import (
    _cache_manifest,
    _cost_rows,
    _summary,
    import_production_observations,
)
from tests.production.helpers import complete_draft, fake_gpu_runtime, fake_repository
from tests.production.test_canary_validation import _visual_record
from tests.production.test_observation_provenance import (
    _production_payload,
    _write_jsonl,
)


def test_cache_key_includes_visual_budget(tmp_path) -> None:
    authority = seal_authority(
        complete_draft(tmp_path),
        output_path=tmp_path / "authority.json",
        project_root=tmp_path,
        runtime_probe=fake_gpu_runtime,
        repository_probe=fake_repository,
    )
    low = _visual_record(authority, "q1")
    high = low.model_copy(
        update={"action": low.action.model_copy(update={"visual_budget": "high"})}
    )
    manifest = _cache_manifest([low, high], authority.authority_sha256)

    assert len({entry["cache_key"] for entry in manifest["entries"]}) == 4


def test_summary_cache_counts_come_from_raw_cost_records(tmp_path) -> None:
    authority = seal_authority(
        complete_draft(tmp_path),
        output_path=tmp_path / "authority.json",
        project_root=tmp_path,
        runtime_probe=fake_gpu_runtime,
        repository_probe=fake_repository,
    )
    record = _visual_record(authority, "q1")
    rows = _cost_rows([record])

    summary = _summary(
        [record],
        rows,
        authority.authority_sha256,
        cache_hits=99,
        cache_misses=99,
    )

    assert summary["cache_hits"] == 0
    assert summary["cache_misses"] == 2
    assert summary["resume_record_hits"] == 99
    assert summary["resume_record_misses"] == 99


def test_nested_authority_path_resolves_project_relative_manifests(tmp_path) -> None:
    authority_dir = tmp_path / "artifacts" / "authorities"
    authority_path = authority_dir / "authority.json"
    authority = seal_authority(
        complete_draft(tmp_path),
        output_path=authority_path,
        project_root=tmp_path,
        runtime_probe=fake_gpu_runtime,
        repository_probe=fake_repository,
    )
    source = tmp_path / "provider.jsonl"
    destination = tmp_path / "run"
    _write_jsonl(source, [_production_payload(authority)])

    import_production_observations(
        source,
        destination,
        authority_path=authority_path,
        resume=False,
    )
    active = GenerationStore(destination, authority.authority_sha256).current_path()
    manifest = json.loads((active / "manifest.json").read_text(encoding="utf-8"))

    assert isinstance(manifest["provider_identities"][0], dict)
