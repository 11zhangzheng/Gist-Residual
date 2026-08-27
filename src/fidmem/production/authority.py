"""Versioned Production Authority schemas and canonical identity helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_GIT_COMMIT_PATTERN = r"^[0-9a-f]{40}$"


def _json_value(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def canonical_json_bytes(value: object) -> bytes:
    """Return the one canonical UTF-8 byte representation for identities."""
    serialized = json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return serialized.encode("utf-8")


def canonical_json(value: object) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RepositoryIdentity(_FrozenModel):
    git_commit: str = Field(pattern=_GIT_COMMIT_PATTERN)
    dirty_worktree: bool
    source_tree_sha256: str = Field(pattern=_SHA256_PATTERN)
    repository_root_name: str = Field(min_length=1)


class DatasetIdentity(_FrozenModel):
    dataset_name: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    split: str = Field(min_length=1)
    split_policy_id: str = Field(min_length=1)
    split_policy_path: str = Field(min_length=1)
    split_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    dataset_manifest_path: str = Field(min_length=1)
    dataset_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    question_manifest_path: str = Field(min_length=1)
    question_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    video_manifest_path: str = Field(min_length=1)
    video_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)


class ModelIdentity(_FrozenModel):
    provider: str = Field(min_length=1)
    canonical_id: str = Field(min_length=1)
    immutable_revision: str = Field(min_length=1)
    identity_kind: Literal["local_artifact", "provider_revision"]
    identity_evidence_path: str = Field(min_length=1)
    identity_evidence_sha256: str = Field(pattern=_SHA256_PATTERN)
    artifact_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    local_snapshot_path: str | None = None
    local_snapshot_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    dtype: str = Field(min_length=1)
    runtime_settings: dict[str, Any] = Field(min_length=1)

    @model_validator(mode="after")
    def immutable_identity_is_explicit(self) -> Self:
        if self.identity_kind == "provider_revision":
            if (
                self.local_snapshot_path is not None
                or self.local_snapshot_sha256 is not None
            ):
                raise ValueError("provider revision must not claim a local snapshot")
            if self.artifact_sha256 is not None:
                raise ValueError(
                    "provider revision must use verifiable identity evidence, "
                    "not an arbitrary artifact_sha256"
                )
            return self
        required = (
            self.artifact_sha256,
            self.local_snapshot_path,
            self.local_snapshot_sha256,
        )
        if any(value is None for value in required):
            raise ValueError(
                "local artifact identity requires snapshot path and verified hashes"
            )
        if (
            self.identity_evidence_path != self.local_snapshot_path
            or self.identity_evidence_sha256 != self.local_snapshot_sha256
        ):
            raise ValueError("local artifact evidence must be the checkpoint artifact")
        return self


class ModelIdentities(_FrozenModel):
    gist_text_encoder: ModelIdentity
    gist_visual_encoder: ModelIdentity
    residual_model: ModelIdentity
    visual_model: ModelIdentity
    answerer: ModelIdentity
    embedding_model: ModelIdentity


class PromptIdentity(_FrozenModel):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    content: str = Field(min_length=1)
    sha256: str = Field(pattern=_SHA256_PATTERN)

    def verify_content_hash(self) -> Self:
        actual = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        if actual != self.sha256:
            raise ValueError("prompt sha256 does not match canonical content")
        return self


class PromptIdentities(_FrozenModel):
    gist_summary: PromptIdentity
    residual_generation: PromptIdentity
    visual_event: PromptIdentity
    visual_question: PromptIdentity
    answerer_template: PromptIdentity


class CanonicalConfigIdentity(_FrozenModel):
    version: str = Field(min_length=1)
    content: dict[str, Any] = Field(min_length=1)
    sha256: str = Field(pattern=_SHA256_PATTERN)

    def verify_content_hash(self) -> Self:
        if canonical_sha256(self.content) != self.sha256:
            raise ValueError("config sha256 does not match canonical content")
        return self


class ObservationConfigurations(_FrozenModel):
    segmentation: CanonicalConfigIdentity
    frame_sampling: CanonicalConfigIdentity
    retrieval: CanonicalConfigIdentity
    observation_budget: CanonicalConfigIdentity


class GPUIdentity(_FrozenModel):
    name: str = Field(min_length=1)
    uuid: str = Field(min_length=1)


class RuntimeIdentity(_FrozenModel):
    machine_identity: str = Field(min_length=1)
    gpu_count: int = Field(ge=0)
    gpus: tuple[GPUIdentity, ...]
    driver_version: str = Field(min_length=1)
    cuda_version: str = Field(min_length=1)
    pytorch_version: str = Field(min_length=1)
    python_version: str = Field(min_length=1)
    inference_backend: str = Field(min_length=1)
    inference_backend_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def gpu_count_matches_records(self) -> Self:
        if self.gpu_count != len(self.gpus):
            raise ValueError("gpu_count must match GPU identity records")
        return self


class CostContract(_FrozenModel):
    cost_record_schema_version: str = Field(min_length=1)
    cost_accounting_version: str = Field(min_length=1)
    units: dict[str, str] = Field(min_length=1)
    aggregation_semantics: dict[str, str] = Field(min_length=1)
    schema_sha256: str = Field(pattern=_SHA256_PATTERN)


class ProductionAuthorityDraft(_FrozenModel):
    schema_version: Literal[1] = 1
    lifecycle: Literal["draft"] = "draft"
    production_ready: Literal[False] = False
    repository: RepositoryIdentity | None = None
    dataset: DatasetIdentity | None = None
    models: ModelIdentities | None = None
    prompts: PromptIdentities | None = None
    observation_configurations: ObservationConfigurations | None = None
    runtime: RuntimeIdentity | None = None
    cost: CostContract | None = None


class SealedProductionAuthority(_FrozenModel):
    schema_version: Literal[1] = 1
    lifecycle: Literal["sealed"] = "sealed"
    production_ready: Literal[True] = True
    repository: RepositoryIdentity
    dataset: DatasetIdentity
    models: ModelIdentities
    prompts: PromptIdentities
    observation_configurations: ObservationConfigurations
    runtime: RuntimeIdentity
    cost: CostContract
    authority_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def authority_hash_matches_content(self) -> Self:
        payload = self.model_dump(mode="json", exclude={"authority_sha256"})
        if canonical_sha256(payload) != self.authority_sha256:
            raise ValueError("authority_sha256 does not match canonical content")
        return self


def load_sealed_authority(path: str | Path) -> SealedProductionAuthority:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return SealedProductionAuthority.model_validate(payload)


class AuthorityValidationIssue(_FrozenModel):
    code: str = Field(min_length=1)
    path: str = Field(min_length=1)
    message: str = Field(min_length=1)


class AuthorityValidationReport(_FrozenModel):
    production_ready: bool
    issues: tuple[AuthorityValidationIssue, ...]
    repository_identity: RepositoryIdentity | None = None
    runtime_identity: RuntimeIdentity | None = None

    @property
    def error_codes(self) -> tuple[str, ...]:
        return tuple(issue.code for issue in self.issues)


class AuthorityValidationError(ValueError):
    """Raised when a draft cannot be sealed without violating a gate."""


RuntimeProbe = Callable[[], RuntimeIdentity]
RepositoryProbe = Callable[[Path], RepositoryIdentity]


_REQUIRED_COST_UNITS = {
    "gpu_seconds": "seconds",
    "wall_seconds": "seconds",
    "input_frames": "count",
    "visual_tokens": "count",
    "text_tokens": "count",
    "peak_memory_bytes": "bytes",
}
_REQUIRED_AGGREGATION = {
    "gpu_seconds": "sum",
    "wall_seconds": "sum",
    "input_frames": "sum",
    "visual_tokens": "sum",
    "text_tokens": "sum",
    "peak_memory_bytes": "max",
    "cache_hits": "count",
    "cache_misses": "count",
    "amortizable_event_work": "charge_once_per_authority_cache_key",
}
_PLACEHOLDER_MARKERS = (
    "replace_with",
    "placeholder",
    "text-1b-2b",
    "shared-frozen-vlm",
    "frozen-answerer",
    "offline-smoke",
    "deterministic-smoke",
    "synthetic",
)
_MUTABLE_IDENTITIES = {"latest", "main", "master", "v1"}


def production_cost_schema_sha256() -> str:
    from fidmem.costs.tracker import CostRecord

    payload = {
        "schema_version": 1,
        "fields": [
            {"name": item.name, "type": str(item.type)} for item in fields(CostRecord)
        ],
    }
    return canonical_sha256(payload)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def authority_file_sha256(path: str | Path) -> str:
    """Hash the exact serialized Authority artifact bytes."""
    return _file_sha256(Path(path))


def _run_git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


_EXECUTION_SOURCE_PREFIXES = ("src/", "configs/")
_EXECUTION_ROOT_FILES = {
    "pyproject.toml",
    "setup.cfg",
    "setup.py",
    "requirements.txt",
    "requirements-dev.txt",
    "poetry.lock",
    "uv.lock",
    "environment.yml",
    "environment.yaml",
}
_GENERATED_PARTS = {
    ".aris",
    ".pytest_cache",
    "__pycache__",
    "artifacts",
    "build",
    "cache",
    "caches",
    "dist",
    "logs",
    "refine-logs",
    "reports",
}
_GENERATED_SUFFIXES = (".log", ".pyc", ".pyo", ".tmp", ".temp", "~")


def _normalized_source_path(relative: str) -> str:
    return relative.replace("\\", "/").lstrip("./")


def is_execution_affecting_source_path(relative: str) -> bool:
    normalized = _normalized_source_path(relative)
    parts = tuple(part.casefold() for part in normalized.split("/"))
    name = parts[-1] if parts else ""
    if not normalized or normalized == "PRODUCTION_AUTHORITY.json":
        return False
    if any(part in _GENERATED_PARTS for part in parts):
        return False
    if name.startswith(".") or name.endswith(_GENERATED_SUFFIXES):
        return False
    return normalized in _EXECUTION_ROOT_FILES or normalized.startswith(
        _EXECUTION_SOURCE_PREFIXES
    )


def source_tree_sha256(project_root: str | Path, candidates: object) -> str:
    root = Path(project_root).resolve()
    inventory: list[dict[str, object]] = []
    for raw_relative in sorted(str(item) for item in candidates):
        relative = _normalized_source_path(raw_relative)
        if not is_execution_affecting_source_path(relative):
            continue
        path = root / relative
        inventory.append(
            {
                "path": relative,
                "sha256": _file_sha256(path) if path.is_file() else None,
            }
        )
    return canonical_sha256(inventory)


def probe_repository(project_root: str | Path) -> RepositoryIdentity:
    root = Path(project_root).resolve()
    commit = _run_git(root, "rev-parse", "HEAD").decode().strip()
    status = _run_git(root, "status", "--porcelain=v1", "-z")
    listed = _run_git(root, "ls-files", "-co", "--exclude-standard", "-z")
    candidates = (
        raw_name.decode("utf-8", errors="surrogateescape")
        for raw_name in listed.split(b"\0")
        if raw_name
    )
    return RepositoryIdentity(
        git_commit=commit,
        dirty_worktree=bool(status),
        source_tree_sha256=source_tree_sha256(root, candidates),
        repository_root_name=root.name,
    )


def _nvidia_rows() -> list[tuple[str, str, str]]:
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    rows = []
    for line in output.splitlines():
        pieces = tuple(piece.strip() for piece in line.split(","))
        if len(pieces) == 3:
            rows.append(pieces)
    return rows


def probe_runtime() -> RuntimeIdentity:
    import torch

    rows = _nvidia_rows()
    gpus = tuple(GPUIdentity(name=row[0], uuid=row[1]) for row in rows)
    driver = rows[0][2] if rows else "unavailable"
    backend = os.environ.get("FIDMEM_INFERENCE_BACKEND", "unconfigured")
    backend_version = os.environ.get("FIDMEM_INFERENCE_BACKEND_VERSION", "unavailable")
    return RuntimeIdentity(
        machine_identity=socket.gethostname() or platform.node() or "unknown-host",
        gpu_count=len(gpus),
        gpus=gpus,
        driver_version=driver,
        cuda_version=str(torch.version.cuda or "unavailable"),
        pytorch_version=str(torch.__version__),
        python_version=platform.python_version(),
        inference_backend=backend,
        inference_backend_version=backend_version,
    )


def _is_immutable_identity(model: ModelIdentity) -> bool:
    values = (
        model.provider.casefold(),
        model.canonical_id.casefold(),
        model.immutable_revision.casefold(),
    )
    if any(marker in value for value in values for marker in _PLACEHOLDER_MARKERS):
        return False
    if any(value in _MUTABLE_IDENTITIES for value in values):
        return False
    return len(model.immutable_revision) >= 12


def _resolved_input(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Authority input path escapes project root") from exc
    return path


def validate_authority_draft(
    draft: ProductionAuthorityDraft,
    *,
    project_root: str | Path,
    runtime_probe: RuntimeProbe = probe_runtime,
    repository_probe: RepositoryProbe = probe_repository,
) -> AuthorityValidationReport:
    from fidmem.production.manifests import (
        DatasetManifest,
        QuestionManifest,
        VideoManifest,
        validate_split_isolation,
    )

    root = Path(project_root).resolve()
    issues: list[AuthorityValidationIssue] = []

    def issue(code: str, path: str, message: str) -> None:
        issues.append(AuthorityValidationIssue(code=code, path=path, message=message))

    actual_repository: RepositoryIdentity | None = None
    actual_runtime: RuntimeIdentity | None = None
    try:
        actual_repository = repository_probe(root)
    except Exception as exc:
        issue("repository_probe_failed", "repository", str(exc))
    try:
        actual_runtime = runtime_probe()
    except Exception as exc:
        issue("runtime_probe_failed", "runtime", str(exc))

    required = (
        "repository",
        "dataset",
        "models",
        "prompts",
        "observation_configurations",
        "cost",
    )
    for name in required:
        if getattr(draft, name) is None:
            issue(
                "required_section_missing", name, f"required section is missing: {name}"
            )

    if draft.repository is not None and actual_repository is not None:
        if draft.repository != actual_repository:
            issue(
                "repository_identity_mismatch",
                "repository",
                "declared repository identity differs from the current source tree",
            )

    if actual_runtime is not None:
        unavailable = {"unavailable", "unconfigured", "unknown"}
        if actual_runtime.gpu_count < 1 or not actual_runtime.gpus:
            issue(
                "production_gpu_missing", "runtime.gpus", "no production GPU is visible"
            )
        if (
            actual_runtime.driver_version.casefold() in unavailable
            or actual_runtime.cuda_version.casefold() in unavailable
            or actual_runtime.inference_backend.casefold() in unavailable
            or actual_runtime.inference_backend_version.casefold() in unavailable
            or any(
                gpu.uuid.casefold().startswith("unavailable")
                for gpu in actual_runtime.gpus
            )
        ):
            issue(
                "runtime_identity_incomplete",
                "runtime",
                "driver, CUDA, backend, and GPU UUID identities must be complete",
            )
        if draft.runtime is not None and draft.runtime != actual_runtime:
            issue(
                "runtime_identity_mismatch",
                "runtime",
                "draft runtime differs from the executing host",
            )

    if draft.models is not None:
        for role in draft.models.__class__.model_fields:
            model = getattr(draft.models, role)
            if not _is_immutable_identity(model):
                issue(
                    "model_identity_not_immutable",
                    f"models.{role}",
                    "model identity is mutable, synthetic, smoke, or a template marker",
                )
            try:
                evidence_path = _resolved_input(root, model.identity_evidence_path)
                if (
                    not evidence_path.is_file()
                    or _file_sha256(evidence_path) != model.identity_evidence_sha256
                ):
                    raise ValueError("model identity evidence hash differs")
                if model.identity_kind == "provider_revision":
                    if model.artifact_sha256 is not None:
                        raise ValueError(
                            "provider revision cannot use artifact_sha256 as proof"
                        )
                    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
                    expected = {
                        "identity_kind": "provider_revision",
                        "provider": model.provider,
                        "canonical_id": model.canonical_id,
                        "immutable_revision": model.immutable_revision,
                    }
                    if evidence != expected:
                        raise ValueError(
                            "provider revision evidence content differs from model identity"
                        )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                issue(
                    "model_identity_evidence_missing",
                    f"models.{role}",
                    str(exc),
                )
            if model.local_snapshot_path is not None:
                try:
                    snapshot = _resolved_input(root, model.local_snapshot_path)
                    actual = _file_sha256(snapshot)
                    if (
                        actual != model.local_snapshot_sha256
                        or actual != model.artifact_sha256
                    ):
                        issue(
                            "model_artifact_mismatch",
                            f"models.{role}",
                            "local snapshot does not match model artifact identity",
                        )
                except (OSError, ValueError) as exc:
                    issue("model_artifact_mismatch", f"models.{role}", str(exc))

    if draft.prompts is not None:
        for role in draft.prompts.__class__.model_fields:
            try:
                getattr(draft.prompts, role).verify_content_hash()
            except ValueError as exc:
                issue("prompt_hash_mismatch", f"prompts.{role}", str(exc))

    if draft.observation_configurations is not None:
        for role in draft.observation_configurations.__class__.model_fields:
            try:
                getattr(draft.observation_configurations, role).verify_content_hash()
            except ValueError as exc:
                issue(
                    "config_hash_mismatch",
                    f"observation_configurations.{role}",
                    str(exc),
                )

    if draft.dataset is not None:
        dataset = draft.dataset
        inputs = {
            "split_policy": (dataset.split_policy_path, dataset.split_policy_sha256),
            "dataset_manifest": (
                dataset.dataset_manifest_path,
                dataset.dataset_manifest_sha256,
            ),
            "question_manifest": (
                dataset.question_manifest_path,
                dataset.question_manifest_sha256,
            ),
            "video_manifest": (
                dataset.video_manifest_path,
                dataset.video_manifest_sha256,
            ),
        }
        resolved: dict[str, Path] = {}
        for name, (relative, expected) in inputs.items():
            try:
                path = _resolved_input(root, relative)
                resolved[name] = path
                if not path.is_file() or _file_sha256(path) != expected:
                    issue(
                        "manifest_hash_mismatch",
                        f"dataset.{name}",
                        f"{name} file is missing or its SHA-256 differs",
                    )
            except (OSError, ValueError) as exc:
                issue("manifest_hash_mismatch", f"dataset.{name}", str(exc))
        if all(name in resolved and resolved[name].is_file() for name in inputs):
            try:
                video_manifest = VideoManifest.model_validate_json(
                    resolved["video_manifest"].read_text(encoding="utf-8")
                )
                question_manifest = QuestionManifest.model_validate_json(
                    resolved["question_manifest"].read_text(encoding="utf-8")
                )
                dataset_manifest = DatasetManifest.model_validate_json(
                    resolved["dataset_manifest"].read_text(encoding="utf-8")
                )
                validate_split_isolation(video_manifest, question_manifest)
                if (
                    dataset_manifest.video_manifest_sha256
                    != video_manifest.manifest_sha256
                    or dataset_manifest.question_manifest_sha256
                    != question_manifest.manifest_sha256
                    or dataset_manifest.split_policy_sha256
                    != dataset.split_policy_sha256
                    or dataset_manifest.split_policy_id != dataset.split_policy_id
                ):
                    issue(
                        "dataset_manifest_mismatch",
                        "dataset.dataset_manifest",
                        "dataset manifest does not bind split/question/video identities",
                    )
            except Exception as exc:
                issue("dataset_manifest_invalid", "dataset", str(exc))

    if draft.cost is not None:
        if draft.cost.schema_sha256 != production_cost_schema_sha256():
            issue(
                "cost_schema_mismatch",
                "cost.schema_sha256",
                "CostRecord schema identity differs from production code",
            )
        if draft.cost.units != _REQUIRED_COST_UNITS:
            issue("cost_units_mismatch", "cost.units", "cost units are not canonical")
        if draft.cost.aggregation_semantics != _REQUIRED_AGGREGATION:
            issue(
                "cost_aggregation_mismatch",
                "cost.aggregation_semantics",
                "cost aggregation semantics are not canonical",
            )

    ordered = tuple(
        sorted(issues, key=lambda item: (item.code, item.path, item.message))
    )
    return AuthorityValidationReport(
        production_ready=not ordered,
        issues=ordered,
        repository_identity=actual_repository,
        runtime_identity=actual_runtime,
    )


def seal_authority(
    draft: ProductionAuthorityDraft,
    *,
    output_path: str | Path,
    project_root: str | Path,
    runtime_probe: RuntimeProbe = probe_runtime,
    repository_probe: RepositoryProbe = probe_repository,
) -> SealedProductionAuthority:
    report = validate_authority_draft(
        draft,
        project_root=project_root,
        runtime_probe=runtime_probe,
        repository_probe=repository_probe,
    )
    if not report.production_ready:
        codes = ", ".join(report.error_codes)
        raise AuthorityValidationError(
            f"Production Authority validation failed: {codes}"
        )
    if report.repository_identity is None or report.runtime_identity is None:
        raise AuthorityValidationError(
            "Production Authority probes did not return identities"
        )

    payload = {
        "schema_version": 1,
        "lifecycle": "sealed",
        "production_ready": True,
        "repository": report.repository_identity.model_dump(mode="json"),
        "dataset": draft.dataset.model_dump(mode="json"),
        "models": draft.models.model_dump(mode="json"),
        "prompts": draft.prompts.model_dump(mode="json"),
        "observation_configurations": draft.observation_configurations.model_dump(
            mode="json"
        ),
        "runtime": report.runtime_identity.model_dump(mode="json"),
        "cost": draft.cost.model_dump(mode="json"),
    }
    sealed = SealedProductionAuthority.model_validate(
        {**payload, "authority_sha256": canonical_sha256(payload)}
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical_json(sealed.model_dump(mode="json")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        load_sealed_authority(temporary_name)
        os.replace(temporary_name, destination)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return sealed
