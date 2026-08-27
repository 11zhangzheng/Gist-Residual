from __future__ import annotations

import json
import shutil

import pytest

from fidmem.production.authority import (
    AuthorityValidationError,
    authority_file_sha256,
    CostContract,
    ProductionAuthorityDraft,
    load_sealed_authority,
    seal_authority,
    validate_authority_draft,
)
from tests.production.helpers import (
    complete_draft,
    fake_cpu_runtime,
    fake_gpu_runtime,
    fake_repository,
)


def validate(draft, tmp_path, runtime=fake_gpu_runtime):
    return validate_authority_draft(
        draft,
        project_root=tmp_path,
        runtime_probe=runtime,
        repository_probe=fake_repository,
    )


def test_incomplete_authority_fails_closed(tmp_path) -> None:
    report = validate(ProductionAuthorityDraft(), tmp_path)

    assert report.production_ready is False
    assert "required_section_missing" in report.error_codes


@pytest.mark.parametrize(
    "bad_id",
    [
        "text-1b-2b",
        "shared-frozen-vlm",
        "frozen-answerer/v1",
        "REPLACE_WITH_64_HEX",
        "latest",
        "main",
        "offline-smoke-v1",
    ],
)
def test_placeholder_or_mutable_model_identity_is_rejected(tmp_path, bad_id) -> None:
    draft = complete_draft(tmp_path)
    changed = draft.models.residual_model.model_copy(update={"canonical_id": bad_id})
    draft = draft.model_copy(
        update={"models": draft.models.model_copy(update={"residual_model": changed})}
    )

    report = validate(draft, tmp_path)

    assert "model_identity_not_immutable" in report.error_codes


def test_mutated_prompt_is_rejected(tmp_path) -> None:
    draft = complete_draft(tmp_path)
    changed = draft.prompts.residual_generation.model_copy(
        update={"content": "mutated"}
    )
    draft = draft.model_copy(
        update={
            "prompts": draft.prompts.model_copy(update={"residual_generation": changed})
        }
    )

    assert "prompt_hash_mismatch" in validate(draft, tmp_path).error_codes


def test_mutated_config_is_rejected(tmp_path) -> None:
    draft = complete_draft(tmp_path)
    changed = draft.observation_configurations.retrieval.model_copy(
        update={"content": {"name": "changed", "version": 2}}
    )
    configs = draft.observation_configurations.model_copy(update={"retrieval": changed})
    draft = draft.model_copy(update={"observation_configurations": configs})

    assert "config_hash_mismatch" in validate(draft, tmp_path).error_codes


def test_runtime_without_gpu_is_rejected(tmp_path) -> None:
    report = validate(complete_draft(tmp_path), tmp_path, fake_cpu_runtime)

    assert "production_gpu_missing" in report.error_codes


def test_mutated_manifest_is_rejected(tmp_path) -> None:
    draft = complete_draft(tmp_path)
    (tmp_path / draft.dataset.video_manifest_path).write_text("{}\n", encoding="utf-8")

    assert "manifest_hash_mismatch" in validate(draft, tmp_path).error_codes


def test_repository_identity_mismatch_is_rejected(tmp_path) -> None:
    draft = complete_draft(tmp_path)
    wrong = draft.repository.model_copy(update={"source_tree_sha256": "9" * 64})

    report = validate(draft.model_copy(update={"repository": wrong}), tmp_path)

    assert "repository_identity_mismatch" in report.error_codes


def test_cost_schema_mismatch_is_rejected(tmp_path) -> None:
    draft = complete_draft(tmp_path)
    wrong = draft.cost.model_copy(update={"schema_sha256": "9" * 64})

    assert (
        "cost_schema_mismatch"
        in validate(draft.model_copy(update={"cost": wrong}), tmp_path).error_codes
    )


def test_seal_is_canonical_and_detects_tampering(tmp_path) -> None:
    path = tmp_path / "PRODUCTION_AUTHORITY.json"
    sealed = seal_authority(
        complete_draft(tmp_path),
        output_path=path,
        project_root=tmp_path,
        runtime_probe=fake_gpu_runtime,
        repository_probe=fake_repository,
    )

    assert load_sealed_authority(path) == sealed
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["prompts"]["residual_generation"]["content"] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="authority_sha256"):
        load_sealed_authority(path)


def test_failed_seal_preserves_existing_authority(tmp_path) -> None:
    path = tmp_path / "PRODUCTION_AUTHORITY.json"
    path.write_bytes(b"existing-authority")

    with pytest.raises(AuthorityValidationError):
        seal_authority(
            ProductionAuthorityDraft(),
            output_path=path,
            project_root=tmp_path,
            runtime_probe=fake_gpu_runtime,
            repository_probe=fake_repository,
        )

    assert path.read_bytes() == b"existing-authority"


def test_hosted_model_rejects_unverifiable_artifact_hash(tmp_path) -> None:
    draft = complete_draft(tmp_path)
    changed = draft.models.residual_model.model_copy(
        update={"artifact_sha256": "0" * 64}
    )
    models = draft.models.model_copy(update={"residual_model": changed})

    report = validate(draft.model_copy(update={"models": models}), tmp_path)

    assert "model_identity_evidence_missing" in report.error_codes


def test_authority_semantic_and_file_identity_survive_relocation(tmp_path) -> None:
    first = tmp_path / "authority-a.json"
    second = tmp_path / "nested" / "authority-b.json"
    sealed = seal_authority(
        complete_draft(tmp_path),
        output_path=first,
        project_root=tmp_path,
        runtime_probe=fake_gpu_runtime,
        repository_probe=fake_repository,
    )
    second.parent.mkdir()
    shutil.copyfile(first, second)

    assert load_sealed_authority(second).authority_sha256 == sealed.authority_sha256
    assert authority_file_sha256(first) == authority_file_sha256(second)
    assert first.resolve() != second.resolve()
