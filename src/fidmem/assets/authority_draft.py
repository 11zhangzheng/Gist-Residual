"""Build the existing ProductionAuthorityDraft from verified stack inputs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from fidmem.assets.resolver import AssetLock, assert_verified_lock, load_asset_lock
from fidmem.production import authority as authority_schema
from fidmem.production.authority import (
    CanonicalConfigIdentity,
    CostContract,
    DatasetIdentity,
    ModelIdentities,
    ModelIdentity,
    ObservationConfigurations,
    ProductionAuthorityDraft,
    PromptIdentities,
    PromptIdentity,
    canonical_json_bytes,
    canonical_sha256,
    probe_repository,
    production_cost_schema_sha256,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(value, dict):
        raise ValueError(f"Authority input config must be a mapping: {path}")
    return value


def _project_relative(root: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Authority input must be inside project root: {resolved}"
        ) from exc


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _prompt_identities(
    raw: dict[str, Any]
) -> tuple[PromptIdentities | None, list[str]]:
    values = raw.get("prompts", {})
    unresolved = [
        f"prompts.{role}"
        for role, item in values.items()
        if not isinstance(item, dict) or item.get("status") != "FROZEN"
    ]
    if unresolved:
        return None, unresolved
    identities = {}
    for role, item in values.items():
        content = str(item["content"])
        identities[role] = PromptIdentity(
            name=str(item["name"]),
            version=str(item["version"]),
            content=content,
            sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        )
    return PromptIdentities.model_validate(identities), []


def _observation_identities(
    raw: dict[str, Any],
) -> tuple[ObservationConfigurations | None, list[str]]:
    values = raw.get("observation_configurations", {})
    unresolved = [
        f"observation_configurations.{role}"
        for role, item in values.items()
        if not isinstance(item, dict) or item.get("status") != "FROZEN"
    ]
    if unresolved:
        return None, unresolved
    identities = {
        role: CanonicalConfigIdentity(
            version=str(item["version"]),
            content=dict(item["content"]),
            sha256=canonical_sha256(item["content"]),
        )
        for role, item in values.items()
    }
    return ObservationConfigurations.model_validate(identities), []


def _model_identities(
    lock: AssetLock,
    runtime_raw: dict[str, Any],
    *,
    project_root: Path,
    evidence_root: Path,
) -> tuple[ModelIdentities | None, list[str]]:
    values = runtime_raw.get("model_runtime_settings", {})
    roles = tuple(ModelIdentities.model_fields)
    unresolved = [
        f"model_runtime_settings.{role}"
        for role in roles
        if not isinstance(values.get(role), dict)
        or values[role].get("status") != "FROZEN"
    ]
    if unresolved:
        return None, unresolved
    physical_evidence: dict[str, tuple[Path, str]] = {}
    for asset_id in sorted(set(lock.logical_roles[role] for role in roles)):
        entry = lock.physical_assets[asset_id]
        payload = {
            "schema_version": 1,
            "asset_lock_sha256": lock.lock_sha256,
            "physical_asset_id": asset_id,
            "repo_id": entry.repo_id,
            "immutable_revision": entry.immutable_revision,
            "local_snapshot_path": entry.local_snapshot_path,
            "local_snapshot_sha256": entry.local_snapshot_sha256,
        }
        path = evidence_root / f"{asset_id}.identity.json"
        _atomic_json(path, payload)
        physical_evidence[asset_id] = (path, _file_sha256(path))
    models: dict[str, ModelIdentity] = {}
    for role in roles:
        asset_id = lock.logical_roles[role]
        entry = lock.physical_assets[asset_id]
        evidence_path, evidence_sha = physical_evidence[asset_id]
        relative = _project_relative(project_root, evidence_path)
        models[role] = ModelIdentity(
            provider="huggingface-local",
            canonical_id=entry.repo_id,
            immutable_revision=str(entry.immutable_revision),
            identity_kind="local_artifact",
            identity_evidence_path=relative,
            identity_evidence_sha256=evidence_sha,
            artifact_sha256=evidence_sha,
            local_snapshot_path=relative,
            local_snapshot_sha256=evidence_sha,
            dtype=str(entry.dtype or "not-applicable"),
            runtime_settings=dict(values[role]["content"]),
        )
    return ModelIdentities.model_validate(models), []


def build_authority_draft(
    *,
    project_root: str | Path,
    asset_lock_path: str | Path,
    manifests_root: str | Path,
    split_policy_path: str | Path,
    prompt_config_path: str | Path,
    observation_config_path: str | Path,
    evidence_root: str | Path,
) -> tuple[ProductionAuthorityDraft, tuple[str, ...]]:
    root = Path(project_root).resolve()
    lock = load_asset_lock(asset_lock_path)
    assert_verified_lock(lock, reverify=True)
    manifests = Path(manifests_root)
    dataset_path = manifests / "dataset_manifest.json"
    video_path = manifests / "video_manifest.json"
    question_path = manifests / "question_manifest.json"
    split_path = Path(split_policy_path)
    for path in (dataset_path, video_path, question_path, split_path):
        if not path.is_file():
            raise ValueError(f"Authority input is missing: {path}")
    dataset_payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    prompts, prompt_issues = _prompt_identities(_load_yaml(Path(prompt_config_path)))
    observation_raw = _load_yaml(Path(observation_config_path))
    observations, observation_issues = _observation_identities(observation_raw)
    models, model_issues = _model_identities(
        lock,
        observation_raw,
        project_root=root,
        evidence_root=Path(evidence_root),
    )
    cost = CostContract(
        cost_record_schema_version="1",
        cost_accounting_version="fidmem-cost-v1",
        units=dict(authority_schema._REQUIRED_COST_UNITS),
        aggregation_semantics=dict(authority_schema._REQUIRED_AGGREGATION),
        schema_sha256=production_cost_schema_sha256(),
    )
    draft = ProductionAuthorityDraft(
        repository=probe_repository(root),
        dataset=DatasetIdentity(
            dataset_name=str(dataset_payload["dataset_name"]),
            dataset_version=str(dataset_payload["dataset_version"]),
            split="development+canary+oracle+holdout",
            split_policy_id=str(dataset_payload["split_policy_id"]),
            split_policy_path=_project_relative(root, split_path),
            split_policy_sha256=_file_sha256(split_path),
            dataset_manifest_path=_project_relative(root, dataset_path),
            dataset_manifest_sha256=_file_sha256(dataset_path),
            question_manifest_path=_project_relative(root, question_path),
            question_manifest_sha256=_file_sha256(question_path),
            video_manifest_path=_project_relative(root, video_path),
            video_manifest_sha256=_file_sha256(video_path),
        ),
        models=models,
        prompts=prompts,
        observation_configurations=observations,
        runtime=None,
        cost=cost,
    )
    unresolved = tuple(
        sorted((*prompt_issues, *observation_issues, *model_issues, "runtime"))
    )
    return draft, unresolved


def write_authority_draft(path: str | Path, draft: ProductionAuthorityDraft) -> None:
    _atomic_json(Path(path), draft.model_dump(mode="json"))
