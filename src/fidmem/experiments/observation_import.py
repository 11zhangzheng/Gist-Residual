"""Import and validate provider-generated observation atoms."""

from __future__ import annotations

import csv
import io
import math
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from fidmem.actions.environment import ActionObservation
from fidmem.costs.tracker import CostRecord
from fidmem.production.authority import (
    SealedProductionAuthority,
    authority_file_sha256,
    canonical_json,
    canonical_sha256,
    load_sealed_authority,
)
from fidmem.production.generation import GenerationStore
from fidmem.production.manifests import QuestionManifest, VideoManifest
from fidmem.production.provenance import AuthorityBoundCache
from fidmem.storage.cache import ContentAddressedCache
from fidmem.types import ActionInstance, ActionType, RouterState


def _canonical_json(value: object) -> str:
    return canonical_json(value)


class ProviderIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    provider: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    decode_config: dict[str, Any] = Field(min_length=1)
    device_name: str = Field(min_length=1)

    @field_validator("provider", "model_revision", "device_name")
    @classmethod
    def identity_strings_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("provider identity strings must not be blank")
        return value


class ObservationImportRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    schema_version: Literal[1] = 1
    evidence_class: Literal["engineering", "production"] = "engineering"
    authority_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    model_id: str | None = Field(default=None, min_length=1)
    config_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    raw_response: Any = None
    raw_response_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    question_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    provider_identity: ProviderIdentity
    state: RouterState
    action: ActionInstance
    observation: ActionObservation

    @model_validator(mode="after")
    def validate_observation_contract(self) -> "ObservationImportRecord":
        if self.observation.action_type is not self.action.action_type:
            raise ValueError("action and observation action types must match")
        if self.observation.target_event_id != self.action.event_id:
            raise ValueError("action and observation target event ids must match")
        if self.action.action_type is not ActionType.STOP and not self.cost_records:
            raise ValueError(
                "non-STOP observation requires authoritative cost metadata"
            )
        production_fields = (
            self.authority_sha256,
            self.model_id,
            self.config_sha256,
            self.raw_response_sha256,
        )
        if self.evidence_class == "engineering":
            if any(value is not None for value in production_fields):
                raise ValueError(
                    "engineering observation must not carry Authority data"
                )
            if self.raw_response is not None:
                raise ValueError(
                    "engineering observation must not carry production raw response"
                )
            return self
        if any(value is None for value in production_fields):
            raise ValueError(
                "production observation requires Authority, model, config, and raw response hashes"
            )
        if self.raw_response is None:
            raise ValueError("production observation requires a raw provider response")
        if canonical_sha256(self.raw_response) != self.raw_response_sha256:
            raise ValueError("raw_response_sha256 does not match raw provider response")
        for metadata in self.observation.operation_metadata:
            cost = metadata.cost_record
            if cost is None:
                continue
            if cost.cache_status != metadata.cache_status:
                raise ValueError(
                    "CostRecord cache status differs from operation metadata"
                )
            if cost.device_name != self.provider_identity.device_name:
                raise ValueError(
                    "CostRecord device differs from provider device identity"
                )
        return self

    @property
    def cost_records(self) -> tuple[CostRecord, ...]:
        return tuple(
            metadata.cost_record
            for metadata in self.observation.operation_metadata
            if metadata.cost_record is not None
        )

    @property
    def record_id(self) -> str:
        return canonical_sha256(self.model_dump(mode="json"))


def _canonical_row(record: ObservationImportRecord) -> dict[str, Any]:
    return {
        "record_id": record.record_id,
        **record.model_dump(mode="json"),
    }


def _parse_input(path: Path) -> list[ObservationImportRecord]:
    if not path.is_file():
        raise ValueError(f"observation input does not exist: {path}")
    records: list[ObservationImportRecord] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid observation JSON on line {line_number}"
                ) from exc
            record = ObservationImportRecord.model_validate(payload)
            if record.record_id in seen:
                raise ValueError(
                    f"duplicate observation record_id on line {line_number}"
                )
            seen.add(record.record_id)
            records.append(record)
    if not records:
        raise ValueError("observation input is empty")
    return records


def _parse_existing(path: Path) -> list[ObservationImportRecord]:
    if not path.exists():
        return []
    records: list[ObservationImportRecord] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid canonical observation JSON on line {line_number}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError("canonical observation row must be an object")
            stored_id = payload.pop("record_id", None)
            if not isinstance(stored_id, str) or not stored_id:
                raise ValueError("canonical observation row lacks record_id")
            record = ObservationImportRecord.model_validate(payload)
            if stored_id != record.record_id:
                raise ValueError("record_id does not match canonical content")
            if stored_id in seen:
                raise ValueError("duplicate record_id in canonical observations")
            seen.add(stored_id)
            records.append(record)
    return records


def _atomic_write_jsonl(
    path: Path,
    records: list[ObservationImportRecord],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(_canonical_json(_canonical_row(record)))
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


_COST_FIELDNAMES = (
    "record_id",
    "metadata_index",
    "evidence_class",
    "authority_sha256",
    "model_id",
    "config_sha256",
    "question_id",
    "video_id",
    "action_type",
    "event_id",
    "provider",
    "model_revision",
    "provider_device_name",
    "scope",
    "amortizable",
    "operation",
    "gpu_seconds",
    "wall_seconds",
    "input_frames",
    "visual_tokens",
    "text_tokens",
    "peak_memory_bytes",
    "cache_status",
    "device_name",
)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _cost_rows(records: list[ObservationImportRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: item.record_id):
        for metadata_index, metadata in enumerate(
            record.observation.operation_metadata
        ):
            cost = metadata.cost_record
            if cost is None:
                continue
            rows.append(
                {
                    "record_id": record.record_id,
                    "metadata_index": metadata_index,
                    "evidence_class": record.evidence_class,
                    "authority_sha256": record.authority_sha256 or "",
                    "model_id": record.model_id or "",
                    "config_sha256": record.config_sha256 or "",
                    "question_id": record.question_id,
                    "video_id": record.video_id,
                    "action_type": record.action.action_type.value,
                    "event_id": record.action.event_id or "",
                    "provider": record.provider_identity.provider,
                    "model_revision": record.provider_identity.model_revision,
                    "provider_device_name": record.provider_identity.device_name,
                    "scope": metadata.scope,
                    "amortizable": metadata.amortizable,
                    "operation": cost.operation,
                    "gpu_seconds": cost.gpu_seconds,
                    "wall_seconds": cost.wall_seconds,
                    "input_frames": cost.input_frames,
                    "visual_tokens": cost.visual_tokens,
                    "text_tokens": cost.text_tokens,
                    "peak_memory_bytes": cost.peak_memory_bytes,
                    "cache_status": cost.cache_status,
                    "device_name": cost.device_name,
                }
            )
    return rows


def _cost_csv(rows: list[dict[str, Any]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=_COST_FIELDNAMES,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _summary(
    records: list[ObservationImportRecord],
    rows: list[dict[str, Any]],
    authority_sha256: str | None = None,
    *,
    cache_hits: int,
    cache_misses: int,
) -> dict[str, Any]:
    gpu_seconds = sorted(float(row["gpu_seconds"]) for row in rows)
    p90_index = math.ceil(0.9 * len(gpu_seconds)) - 1 if gpu_seconds else None
    production = bool(records and records[0].evidence_class == "production")
    if production:
        observed = {record.authority_sha256 for record in records}
        if authority_sha256 is None or observed != {authority_sha256}:
            raise ValueError("production summary Authority mismatch")
        raw_cache_hits = sum(row["cache_status"] == "hit" for row in rows)
        raw_cache_misses = sum(row["cache_status"] == "miss" for row in rows)

    return {
        "schema_version": 1,
        "record_count": len(records),
        **(
            {"evidence_class": "production", "authority_sha256": authority_sha256}
            if production
            else {}
        ),
        "cost_record_count": len(rows),
        "cache_hits": raw_cache_hits if production else cache_hits,
        "cache_misses": raw_cache_misses if production else cache_misses,
        **(
            {"resume_record_hits": cache_hits, "resume_record_misses": cache_misses}
            if production
            else {}
        ),
        "total_gpu_seconds": math.fsum(gpu_seconds),
        "p90_gpu_seconds": gpu_seconds[p90_index] if p90_index is not None else 0.0,
        "total_wall_seconds": math.fsum(float(row["wall_seconds"]) for row in rows),
        "total_input_frames": sum(int(row["input_frames"]) for row in rows),
        "total_visual_tokens": sum(int(row["visual_tokens"]) for row in rows),
        "total_text_tokens": sum(int(row["text_tokens"]) for row in rows),
        "peak_memory_bytes": max(
            (int(row["peak_memory_bytes"]) for row in rows),
            default=0,
        ),
    }


def _provider_identities(
    records: list[ObservationImportRecord],
) -> list[dict[str, Any]]:
    identities = {
        _canonical_json(
            record.provider_identity.model_dump(mode="json")
        ): record.provider_identity.model_dump(mode="json")
        for record in records
    }
    return [identities[key] for key in sorted(identities)]


def _write_artifacts(
    *,
    source: Path,
    config_source: Path | None,
    destination: Path,
    run_id: str,
    records: list[ObservationImportRecord],
    cache_hits: int,
    cache_misses: int,
) -> dict[str, str]:
    paths = {
        "observations": (destination / "observations.jsonl").resolve(),
        "costs": (destination / "cost.csv").resolve(),
        "summary": (destination / "summary.json").resolve(),
        "manifest": (destination / "manifest.json").resolve(),
    }
    rows = _cost_rows(records)
    summary = _summary(
        records,
        rows,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
    )
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "input_path": str(source.resolve()),
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "config_path": str(config_source.resolve()) if config_source else None,
        "config_sha256": (
            hashlib.sha256(config_source.read_bytes()).hexdigest()
            if config_source
            else None
        ),
        "provider_identities": _provider_identities(records),
        "artifacts": {key: str(path) for key, path in paths.items()},
    }
    _atomic_write_text(paths["costs"], _cost_csv(rows))
    _atomic_write_text(
        paths["summary"],
        _canonical_json(summary) + "\n",
    )
    _atomic_write_text(
        paths["manifest"],
        _canonical_json(manifest) + "\n",
    )
    return {key: str(path) for key, path in paths.items()}


def import_observations(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    resume: bool,
    run_id: str = "engineering-import",
    config_path: str | Path | None = None,
    authority_path: str | Path | None = None,
    failure_hook: Any | None = None,
) -> dict[str, Any]:
    source = Path(input_path)
    if authority_path is not None:
        return _import_production_mode(
            input_path,
            output_dir,
            authority_path=authority_path,
            resume=resume,
            run_id=run_id,
            failure_hook=failure_hook,
        )
    destination = Path(output_dir)
    config_source = Path(config_path) if config_path is not None else None
    if config_source is not None and not config_source.is_file():
        raise ValueError(f"config input does not exist: {config_source}")
    incoming = _parse_input(source)
    observations_path = destination / "observations.jsonl"
    existing = _parse_existing(observations_path) if resume else []
    existing_by_id = {record.record_id: record for record in existing}

    cache_hits = 0
    new_records: list[ObservationImportRecord] = []
    for record in incoming:
        if record.record_id in existing_by_id:
            cache_hits += 1
        else:
            new_records.append(record)

    merged = [*existing, *new_records] if resume else incoming
    cache_misses = len(new_records) if resume else len(incoming)
    if not (resume and not new_records):
        _atomic_write_jsonl(observations_path, merged)
    artifacts = _write_artifacts(
        source=source,
        config_source=config_source,
        destination=destination,
        run_id=run_id,
        records=merged,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
    )

    return {
        "record_count": len(merged),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "observations_path": str(observations_path),
        "artifacts": artifacts,
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_candidates(
    authority: SealedProductionAuthority, action_type: ActionType
) -> tuple[Any, ...]:
    if action_type is ActionType.SEARCH_GIST:
        return (
            authority.models.gist_text_encoder,
            authority.models.gist_visual_encoder,
        )
    if action_type in {ActionType.EXPAND_RESIDUAL, ActionType.EXPAND_CONTEXT}:
        return (authority.models.residual_model,)
    if action_type is ActionType.VERIFY_VISUAL:
        return (authority.models.visual_model,)
    return (authority.models.answerer,)


def _validate_record_against_authority(
    record: ObservationImportRecord,
    authority: SealedProductionAuthority,
    *,
    valid_pairs: set[tuple[str, str]],
) -> None:
    if record.authority_sha256 != authority.authority_sha256:
        raise ValueError("production record authority mismatch")
    expected_config = canonical_sha256(
        authority.observation_configurations.model_dump(mode="json")
    )
    if record.config_sha256 != expected_config:
        raise ValueError("production record config_sha256 differs from Authority")
    candidates = _model_candidates(authority, record.action.action_type)
    if not any(
        record.provider_identity.provider == model.provider
        and record.model_id == model.canonical_id
        and record.provider_identity.model_revision == model.immutable_revision
        and record.provider_identity.decode_config == model.runtime_settings
        for model in candidates
    ):
        raise ValueError(
            "provider/model/revision/decode identity differs from Authority"
        )
    devices = {
        value
        for index, gpu in enumerate(authority.runtime.gpus)
        for value in (gpu.name, gpu.uuid, f"cuda:{index}")
    }
    if record.provider_identity.device_name not in devices:
        raise ValueError("provider device identity differs from Authority runtime")
    if (record.question_id, record.video_id) not in valid_pairs:
        raise ValueError("production observation is missing from frozen manifests")


def _manifest_pairs(
    authority_path: Path, authority: SealedProductionAuthority
) -> set[tuple[str, str]]:
    authority_parent = authority_path.resolve().parent
    relative_inputs = {
        authority.dataset.split_policy_path: authority.dataset.split_policy_sha256,
        authority.dataset.dataset_manifest_path: authority.dataset.dataset_manifest_sha256,
        authority.dataset.question_manifest_path: authority.dataset.question_manifest_sha256,
        authority.dataset.video_manifest_path: authority.dataset.video_manifest_sha256,
    }
    root = next(
        (
            candidate
            for candidate in (authority_parent, *authority_parent.parents)
            if all(
                (candidate / relative).is_file()
                and _file_sha256(candidate / relative) == expected
                for relative, expected in relative_inputs.items()
            )
        ),
        None,
    )
    if root is None:
        raise ValueError(
            "Authority manifest files are missing or differ from their sealed hashes"
        )
    questions = QuestionManifest.model_validate_json(
        (root / authority.dataset.question_manifest_path).read_text(encoding="utf-8")
    )
    videos = VideoManifest.model_validate_json(
        (root / authority.dataset.video_manifest_path).read_text(encoding="utf-8")
    )
    allowed_videos = {record.video_id for record in videos.records}
    return {
        (record.question_id, record.video_id)
        for record in questions.records
        if record.video_id in allowed_videos
    }


def _parse_source(
    path: Path,
    authority: SealedProductionAuthority,
    *,
    valid_pairs: set[tuple[str, str]],
) -> list[ObservationImportRecord]:
    if not path.is_file():
        raise ValueError(f"observation input does not exist: {path}")
    records: list[ObservationImportRecord] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            record = ObservationImportRecord.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(
                f"production record on line {line_number} lacks required authority provenance"
            ) from exc
        _validate_record_against_authority(record, authority, valid_pairs=valid_pairs)
        if record.record_id in seen:
            raise ValueError("duplicate production observation record_id")
        seen.add(record.record_id)
        records.append(record)
    if not records:
        raise ValueError("production observation input is empty")
    return records


def _artifact_paths(destination: Path) -> dict[str, Path]:
    return {
        "observations": (destination / "observations.jsonl").resolve(),
        "costs": (destination / "cost.csv").resolve(),
        "summary": (destination / "summary.json").resolve(),
        "manifest": (destination / "manifest.json").resolve(),
        "cache_manifest": (destination / "cache_manifest.json").resolve(),
        "committed": (destination / "COMMITTED.json").resolve(),
        "state": (destination / "state.json").resolve(),
    }


def _validate_commit(destination: Path, expected_authority: str) -> None:
    marker = json.loads((destination / "COMMITTED.json").read_text(encoding="utf-8"))
    if marker.get("authority_sha256") != expected_authority:
        raise ValueError("existing run uses a different Authority")
    for name, expected in marker.get("artifact_sha256", {}).items():
        path = destination / name
        if not path.is_file() or _file_sha256(path) != expected:
            raise ValueError("committed production artifact hash mismatch")


def _cache_manifest(
    records: list[ObservationImportRecord], authority_sha256: str
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    keyed: dict[str, tuple[str, str]] = {}
    for record in records:
        for index, metadata in enumerate(record.observation.operation_metadata):
            identity = {
                "authority_sha256": authority_sha256,
                "video_id": record.video_id,
                "event_id": record.action.event_id,
                "scope": metadata.scope,
                "model_id": record.model_id,
                "model_revision": record.provider_identity.model_revision,
                "config_sha256": record.config_sha256,
                "visual_budget": record.action.visual_budget,
            }
            if not metadata.amortizable:
                identity["question_id"] = record.question_id
            cache_key = canonical_sha256(identity)
            signature = (record.raw_response_sha256, metadata.scope)
            if cache_key in keyed and keyed[cache_key] != signature:
                raise ValueError("production cache key collision")
            keyed[cache_key] = signature
            entries.append(
                {
                    "cache_key": cache_key,
                    "record_id": record.record_id,
                    "metadata_index": index,
                    "scope": metadata.scope,
                    "amortizable": metadata.amortizable,
                    "question_id": None if metadata.amortizable else record.question_id,
                }
            )
    return {
        "schema_version": 1,
        "evidence_class": "production",
        "authority_sha256": authority_sha256,
        "entries": sorted(
            entries, key=lambda item: (item["cache_key"], item["record_id"])
        ),
    }


def _production_cache_root(destination: Path, authority_sha256: str) -> Path:
    if (
        destination.parent.name == "runs"
        and destination.parent.parent.name == authority_sha256
    ):
        return destination.parent.parent / "cache"
    return destination.parent / "production-cache" / authority_sha256


def _cache_entry_payload(
    record: ObservationImportRecord,
    entry: dict[str, Any],
    authority_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "evidence_class": "production",
        "authority_sha256": authority_sha256,
        "record_id": record.record_id,
        "metadata_index": entry["metadata_index"],
        "scope": entry["scope"],
        "amortizable": entry["amortizable"],
        "question_id": entry["question_id"],
        "raw_response_sha256": record.raw_response_sha256,
    }


def _bind_production_cache(
    destination: Path,
    records: list[ObservationImportRecord],
    cache_manifest: dict[str, Any],
    authority_sha256: str,
) -> Path:
    cache_root = _production_cache_root(destination, authority_sha256)
    cache = AuthorityBoundCache(ContentAddressedCache(cache_root))
    by_record_id = {record.record_id: record for record in records}
    for entry in cache_manifest["entries"]:
        record = by_record_id[entry["record_id"]]
        payload = _cache_entry_payload(record, entry, authority_sha256)
        existing = cache.get_bound(
            entry["cache_key"],
            expected_authority_sha256=authority_sha256,
        )
        if existing is None:
            cache.put_bound(
                entry["cache_key"],
                payload,
                authority_sha256=authority_sha256,
            )
        elif existing != payload:
            raise ValueError("production cache key collision")
    return cache_root


def _validate_production_cache(
    destination: Path,
    records: list[ObservationImportRecord],
    cache_manifest: dict[str, Any],
    authority_sha256: str,
) -> Path:
    cache_root = _production_cache_root(destination, authority_sha256)
    if not cache_root.is_dir():
        raise ValueError("production Authority-bound cache is missing")
    cache = AuthorityBoundCache(ContentAddressedCache(cache_root))
    by_record_id = {record.record_id: record for record in records}
    for entry in cache_manifest["entries"]:
        record = by_record_id.get(entry["record_id"])
        if record is None:
            raise ValueError("production cache manifest references a missing record")
        expected = _cache_entry_payload(record, entry, authority_sha256)
        observed = cache.get_bound(
            entry["cache_key"],
            expected_authority_sha256=authority_sha256,
        )
        if observed is None:
            raise ValueError("production Authority-bound cache envelope is missing")
        if observed != expected:
            raise ValueError("production cache envelope differs from manifest")
    return cache_root


def _write_stage(
    stage: Path,
    destination: Path,
    source: Path,
    authority_path: Path,
    authority: SealedProductionAuthority,
    records: list[ObservationImportRecord],
    *,
    run_id: str,
    cache_hits: int,
    cache_misses: int,
) -> dict[str, Path]:
    paths = _artifact_paths(destination)
    rows = _cost_rows(records)
    summary = _summary(
        records,
        rows,
        authority.authority_sha256,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
    )
    observations = "".join(
        canonical_json(
            {"record_id": record.record_id, **record.model_dump(mode="json")}
        )
        + "\n"
        for record in records
    )
    cache_manifest = _cache_manifest(records, authority.authority_sha256)
    identity_by_key = {
        canonical_json(
            record.provider_identity.model_dump(mode="json")
        ): record.provider_identity.model_dump(mode="json")
        for record in records
    }
    manifest = {
        "schema_version": 1,
        "evidence_class": "production",
        "authority_path": str(authority_path.resolve()),
        "authority_sha256": authority.authority_sha256,
        "authority_file_sha256": authority_file_sha256(authority_path),
        "run_id": run_id,
        "input_path": str(source.resolve()),
        "input_sha256": _file_sha256(source),
        "provider_identities": [
            identity_by_key[key] for key in sorted(identity_by_key)
        ],
        "artifacts": {name: str(path) for name, path in paths.items()},
    }
    contents = {
        "observations.jsonl": observations,
        "cost.csv": _cost_csv(rows),
        "summary.json": canonical_json(summary) + "\n",
        "manifest.json": canonical_json(manifest) + "\n",
        "cache_manifest.json": canonical_json(cache_manifest) + "\n",
        "state.json": canonical_json(
            {
                "schema_version": 1,
                "evidence_class": "production",
                "authority_sha256": authority.authority_sha256,
                "command_history": [],
            }
        )
        + "\n",
    }
    for name, content in contents.items():
        path = stage / name
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    marker = {
        "schema_version": 1,
        "evidence_class": "production",
        "authority_sha256": authority.authority_sha256,
        "artifact_sha256": {
            name: _file_sha256(stage / name) for name in sorted(contents)
        },
    }
    (stage / "COMMITTED.json").write_text(
        canonical_json(marker) + "\n", encoding="utf-8", newline="\n"
    )
    _parse_existing(stage / "observations.jsonl")
    _validate_commit(stage, authority.authority_sha256)
    return paths


def _import_production_mode(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    authority_path: str | Path,
    resume: bool,
    run_id: str = "production-import",
    failure_hook: Any | None = None,
) -> dict[str, Any]:
    source = Path(input_path)
    destination = Path(output_dir)
    authority_source = Path(authority_path)
    authority = load_sealed_authority(authority_source)
    store = GenerationStore(destination, authority.authority_sha256)
    active: Path | None = None
    if store.pointer_path.exists():
        active = store.current_path()
        _validate_commit(active, authority.authority_sha256)
        if not resume:
            raise ValueError("production run already exists; use --resume")
    elif resume and destination.exists() and any(destination.iterdir()):
        raise ValueError("production resume requires a valid CURRENT pointer")

    valid_pairs = _manifest_pairs(authority_source, authority)
    incoming = _parse_source(source, authority, valid_pairs=valid_pairs)
    existing = (
        _parse_existing(active / "observations.jsonl") if active is not None else []
    )
    for record in existing:
        _validate_record_against_authority(record, authority, valid_pairs=valid_pairs)
    existing_ids = {record.record_id for record in existing}
    new_records = [
        record for record in incoming if record.record_id not in existing_ids
    ]
    cache_hits = len(incoming) - len(new_records)
    cache_misses = len(new_records)
    if resume and not new_records and active is not None:
        cache_root = _bind_production_cache(
            destination,
            existing,
            _cache_manifest(existing, authority.authority_sha256),
            authority.authority_sha256,
        )
        return {
            "record_count": len(existing),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "authority_sha256": authority.authority_sha256,
            "generation_id": active.name,
            "cache_root": str(cache_root.resolve()),
            "artifacts": {
                name: str(path) for name, path in _artifact_paths(active).items()
            },
        }

    merged = [*existing, *new_records]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.serialization.", dir=destination.parent
    ) as temporary:
        stage = Path(temporary)
        _write_stage(
            stage,
            destination,
            source,
            authority_source,
            authority,
            merged,
            run_id=run_id,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
        )
        contents = {
            name: (stage / name).read_bytes()
            for name in (
                "observations.jsonl",
                "cost.csv",
                "summary.json",
                "manifest.json",
                "cache_manifest.json",
                "state.json",
            )
        }
        cache_root = _bind_production_cache(
            destination,
            merged,
            _cache_manifest(merged, authority.authority_sha256),
            authority.authority_sha256,
        )
        active = store.publish(contents, failure_hook=failure_hook)

    _validate_commit(active, authority.authority_sha256)
    return {
        "record_count": len(merged),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "authority_sha256": authority.authority_sha256,
        "generation_id": active.name,
        "cache_root": str(cache_root.resolve()),
        "artifacts": {
            name: str(path) for name, path in _artifact_paths(active).items()
        },
    }


def import_production_observations(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    authority_path: str | Path,
    resume: bool,
    run_id: str = "production-import",
    failure_hook: Any | None = None,
) -> dict[str, Any]:
    """Compatibility entry point routed through the sole canonical importer."""

    return import_observations(
        input_path,
        output_dir,
        authority_path=authority_path,
        resume=resume,
        run_id=run_id,
        failure_hook=failure_hook,
    )
