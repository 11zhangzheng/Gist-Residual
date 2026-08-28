"""Fail-closed Hugging Face resolution, download, verification, and lock lifecycle."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fidmem.assets.stack import ExperimentStack
from fidmem.production.authority import canonical_json_bytes, canonical_sha256

RESOLVER_VERSION = "fidmem.asset-resolver/v1"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_COMMIT_PATTERN = r"^[0-9a-f]{40}$"


class AssetState(str, Enum):
    UNRESOLVED = "UNRESOLVED"
    RESOLVED = "RESOLVED"
    DOWNLOADING = "DOWNLOADING"
    DOWNLOADED = "DOWNLOADED"
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AssetLockEntry(_FrozenModel):
    repo_id: str = Field(min_length=1)
    repo_type: Literal["model", "dataset"]
    immutable_revision: str | None = Field(default=None, pattern=_COMMIT_PATTERN)
    state: AssetState
    backend: str = Field(min_length=1)
    dtype: str | None = None
    expected_files: tuple[str, ...] = ()
    local_snapshot_path: str | None = None
    local_snapshot_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    resolved_at: str | None = None
    verified_at: str | None = None
    failure: str | None = None

    @model_validator(mode="after")
    def lifecycle_fields_are_consistent(self) -> Self:
        if self.state not in {AssetState.UNRESOLVED, AssetState.FAILED}:
            if self.immutable_revision is None or self.resolved_at is None:
                raise ValueError(
                    f"{self.state.value} asset requires resolved immutable identity"
                )
        if self.state in {AssetState.DOWNLOADED, AssetState.VERIFIED}:
            if self.local_snapshot_path is None:
                raise ValueError(
                    f"{self.state.value} asset requires a local snapshot path"
                )
        if self.state is AssetState.VERIFIED:
            if self.local_snapshot_sha256 is None or self.verified_at is None:
                raise ValueError("VERIFIED asset requires recomputed snapshot identity")
        elif self.verified_at is not None:
            raise ValueError("only VERIFIED assets may carry verified_at")
        if self.state is AssetState.FAILED and not self.failure:
            raise ValueError("FAILED asset requires a failure reason")
        return self


class AssetLock(_FrozenModel):
    schema_version: Literal[1] = 1
    stack_id: str = Field(min_length=1)
    resolver_version: Literal[RESOLVER_VERSION] = RESOLVER_VERSION
    huggingface_hub_version: str | None = None
    generated_at: str = Field(min_length=1)
    logical_roles: dict[str, str] = Field(min_length=1)
    physical_assets: dict[str, AssetLockEntry] = Field(min_length=1)
    lock_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def lock_hash_and_mappings_match(self) -> Self:
        payload = self.model_dump(mode="json", exclude={"lock_sha256"})
        if canonical_sha256(payload) != self.lock_sha256:
            raise ValueError("asset lock hash mismatch")
        unknown = set(self.logical_roles.values()) - set(self.physical_assets)
        if unknown:
            raise ValueError(
                f"asset lock has unknown physical mappings: {sorted(unknown)}"
            )
        return self

    @classmethod
    def create(
        cls,
        *,
        stack_id: str,
        generated_at: str,
        logical_roles: dict[str, str],
        physical_assets: dict[str, AssetLockEntry],
        huggingface_hub_version: str | None,
    ) -> "AssetLock":
        payload = {
            "schema_version": 1,
            "stack_id": stack_id,
            "resolver_version": RESOLVER_VERSION,
            "huggingface_hub_version": huggingface_hub_version,
            "generated_at": generated_at,
            "logical_roles": logical_roles,
            "physical_assets": {
                key: value.model_dump(mode="json")
                for key, value in sorted(physical_assets.items())
            },
        }
        return cls(**payload, lock_sha256=canonical_sha256(payload))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_asset_lock(path: str | Path) -> AssetLock:
    return AssetLock.model_validate_json(Path(path).read_text(encoding="utf-8"))


def write_asset_lock(path: str | Path, lock: AssetLock) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json_bytes(lock.model_dump(mode="json")))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def initial_lock(stack: ExperimentStack) -> AssetLock:
    now = utc_now()
    entries = {
        asset_id: AssetLockEntry(
            repo_id=asset.repo_id,
            repo_type=asset.repo_type,
            immutable_revision=asset.immutable_revision,
            state=(
                AssetState.RESOLVED
                if asset.immutable_revision is not None
                else AssetState.UNRESOLVED
            ),
            backend=asset.backend,
            dtype=asset.dtype,
            resolved_at=now if asset.immutable_revision is not None else None,
        )
        for asset_id, asset in stack.physical_assets.items()
    }
    return AssetLock.create(
        stack_id=stack.stack_id,
        generated_at=now,
        logical_roles=dict(stack.logical_roles),
        physical_assets=entries,
        huggingface_hub_version=None,
    )


def snapshot_sha256(root: str | Path) -> str:
    directory = Path(root)
    if not directory.is_dir():
        raise ValueError(f"snapshot directory does not exist: {directory}")
    rows: list[dict[str, object]] = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        relative = path.relative_to(directory).as_posix()
        if relative.startswith(".cache/") or relative.endswith(".lock"):
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        rows.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    if not rows:
        raise ValueError("snapshot contains no verifiable files")
    return canonical_sha256(rows)


def verify_entry(entry: AssetLockEntry) -> AssetLockEntry:
    if entry.immutable_revision is None:
        raise ValueError("cannot verify an unresolved asset")
    if entry.local_snapshot_path is None:
        raise ValueError("asset has no local snapshot path")
    root = Path(entry.local_snapshot_path)
    missing = [name for name in entry.expected_files if not (root / name).is_file()]
    if missing:
        raise ValueError(f"snapshot is incomplete; missing files: {missing}")
    digest = snapshot_sha256(root)
    return entry.model_copy(
        update={
            "state": AssetState.VERIFIED,
            "local_snapshot_sha256": digest,
            "verified_at": utc_now(),
            "failure": None,
        }
    )


def assert_verified_lock(lock: AssetLock, *, reverify: bool = True) -> None:
    for asset_id, entry in lock.physical_assets.items():
        if entry.state is not AssetState.VERIFIED:
            raise ValueError(f"asset {asset_id} is not VERIFIED")
        if (
            reverify
            and verify_entry(entry).local_snapshot_sha256 != entry.local_snapshot_sha256
        ):
            raise ValueError(f"asset {asset_id} snapshot hash differs from lock")


def storage_roots(environment: dict[str, str] | None = None) -> dict[str, Path]:
    values = environment or dict(os.environ)
    names = (
        "FIDMEM_DATA_ROOT",
        "FIDMEM_MODEL_ROOT",
        "FIDMEM_CACHE_ROOT",
        "FIDMEM_ARTIFACT_ROOT",
    )
    missing = [name for name in names if not values.get(name)]
    if missing:
        raise ValueError(f"required storage roots are missing: {missing}")
    return {name: Path(values[name]).expanduser().resolve() for name in names}


def check_storage_roots(
    roots: dict[str, Path], *, min_free_gb: float = 20.0
) -> dict[str, dict[str, object]]:
    report: dict[str, dict[str, object]] = {}
    for name, path in roots.items():
        existing = path
        while not existing.exists() and existing != existing.parent:
            existing = existing.parent
        if not existing.exists():
            raise ValueError(f"no existing parent for {name}: {path}")
        if not os.access(existing, os.W_OK):
            raise ValueError(f"storage root is not writable: {name}={path}")
        free_gb = shutil.disk_usage(existing).free / (1024**3)
        if free_gb < min_free_gb:
            raise ValueError(f"insufficient free space for {name}: {free_gb:.2f} GiB")
        report[name] = {"path": str(path), "free_gb": round(free_gb, 3)}
    return report


def resolve_entry(
    entry: AssetLockEntry,
    *,
    info_loader: Callable[[str, str], tuple[str, tuple[str, ...]]],
) -> AssetLockEntry:
    revision, files = info_loader(entry.repo_id, entry.repo_type)
    if not re.fullmatch(_COMMIT_PATTERN, revision):
        raise ValueError("remote resolver did not return a full immutable commit SHA")
    return entry.model_copy(
        update={
            "immutable_revision": revision,
            "state": AssetState.RESOLVED,
            "expected_files": tuple(sorted(set(files))),
            "resolved_at": utc_now(),
            "failure": None,
        }
    )


def huggingface_info_loader(
    repo_id: str, repo_type: str
) -> tuple[str, tuple[str, ...]]:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for remote resolution") from exc
    api = HfApi()
    info: Any = (
        api.dataset_info(repo_id=repo_id)
        if repo_type == "dataset"
        else api.model_info(repo_id=repo_id)
    )
    revision = str(info.sha)
    files = tuple(str(item.rfilename) for item in (info.siblings or ()))
    return revision, files


def snapshot_download_entry(
    entry: AssetLockEntry,
    *,
    destination: Path,
    cache_root: Path,
) -> AssetLockEntry:
    if entry.immutable_revision is None:
        raise ValueError("download requires a resolved immutable revision")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for downloads") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=entry.repo_id,
        repo_type=entry.repo_type,
        revision=entry.immutable_revision,
        local_dir=destination,
        cache_dir=cache_root,
    )
    return entry.model_copy(
        update={
            "state": AssetState.DOWNLOADED,
            "local_snapshot_path": str(destination.resolve()),
            "local_snapshot_sha256": None,
            "verified_at": None,
            "failure": None,
        }
    )
