"""Experiment Stack v1 executor contract producing canonical importer JSONL."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fidmem.actions.environment import ActionObservation, OperationMetadata
from fidmem.assets.resolver import AssetLock, assert_verified_lock, load_asset_lock
from fidmem.assets.stack import ExperimentStack, load_experiment_stack
from fidmem.costs.tracker import CostRecord
from fidmem.experiments.observation_import import (
    ObservationImportRecord,
    ProviderIdentity,
)
from fidmem.production.authority import (
    SealedProductionAuthority,
    canonical_json,
    canonical_json_bytes,
    canonical_sha256,
    load_sealed_authority,
)
from fidmem.types import ActionInstance, ActionType, RouterState


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())


class ExecutionRequest(_FrozenModel):
    schema_version: Literal[1] = 1
    authority_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    evidence_class: Literal["engineering", "production"]
    question_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    model_role: Literal[
        "gist_text_encoder",
        "gist_visual_encoder",
        "residual_model",
        "visual_model",
        "answerer",
        "embedding_model",
    ]
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    state: RouterState
    action: ActionInstance
    input_payload: dict[str, object] = Field(min_length=1)

    @model_validator(mode="after")
    def production_requires_authority(self) -> "ExecutionRequest":
        if self.evidence_class == "production" and self.authority_sha256 is None:
            raise ValueError("production request requires authority_sha256")
        if self.evidence_class == "engineering" and self.authority_sha256 is not None:
            raise ValueError("engineering request must not bind Production Authority")
        return self

    @property
    def request_key(self) -> str:
        return canonical_sha256(self.model_dump(mode="json"))


class MeasuredOperation(_FrozenModel):
    measurement_source: Literal["runtime-measured"]
    scope: Literal[
        "search_gist",
        "residual",
        "context",
        "event_observation",
        "question_verification",
    ]
    amortizable: bool
    cache_status: Literal["hit", "miss"]
    operation: str = Field(min_length=1)
    gpu_seconds: float = Field(ge=0)
    wall_seconds: float = Field(ge=0)
    input_frames: int = Field(ge=0)
    visual_tokens: int = Field(ge=0)
    text_tokens: int = Field(ge=0)
    peak_memory_bytes: int = Field(ge=0)
    device_name: str = Field(min_length=1)

    def metadata(self) -> OperationMetadata:
        cost = CostRecord(
            operation=self.operation,
            gpu_seconds=self.gpu_seconds,
            wall_seconds=self.wall_seconds,
            input_frames=self.input_frames,
            visual_tokens=self.visual_tokens,
            text_tokens=self.text_tokens,
            peak_memory_bytes=self.peak_memory_bytes,
            cache_status=self.cache_status,
            device_name=self.device_name,
        )
        return OperationMetadata(
            scope=self.scope,
            cache_status=self.cache_status,
            amortizable=self.amortizable,
            input_frames=self.input_frames,
            visual_tokens=self.visual_tokens,
            text_tokens=self.text_tokens,
            cost_record=cost,
        )


class ProviderExecutionResult(_FrozenModel):
    request_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider: str = Field(min_length=1)
    device_name: str = Field(min_length=1)
    decode_config: dict[str, object] = Field(min_length=1)
    raw_response: object
    observation: ActionObservation
    measured_operations: tuple[MeasuredOperation, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def devices_and_metadata_are_consistent(self) -> "ProviderExecutionResult":
        if any(
            item.device_name != self.device_name for item in self.measured_operations
        ):
            raise ValueError("measured operation device differs from provider device")
        if self.observation.operation_metadata:
            raise ValueError("backend observation must not pre-populate cost metadata")
        return self


class StackV1Backend(Protocol):
    """Real backends implement these methods using only frozen local snapshots."""

    def check(self, request: ExecutionRequest) -> None: ...

    def execute(self, request: ExecutionRequest) -> ProviderExecutionResult: ...


def _expected_scopes(action: ActionInstance) -> tuple[str, ...]:
    return {
        ActionType.SEARCH_GIST: ("search_gist",),
        ActionType.EXPAND_RESIDUAL: ("residual",),
        ActionType.EXPAND_CONTEXT: ("context",),
        ActionType.VERIFY_VISUAL: ("event_observation", "question_verification"),
        ActionType.STOP: (),
    }[action.action_type]


def canonical_import_record(
    request: ExecutionRequest, result: ProviderExecutionResult
) -> ObservationImportRecord:
    if result.request_key != request.request_key:
        raise ValueError("provider result request identity mismatch")
    if tuple(item.scope for item in result.measured_operations) != _expected_scopes(
        request.action
    ):
        raise ValueError("measured operation scopes differ from action contract")
    observation = result.observation.model_copy(
        update={
            "action_type": request.action.action_type,
            "target_event_id": request.action.event_id,
            "operation_metadata": tuple(
                item.metadata() for item in result.measured_operations
            ),
        }
    )
    raw_hash = canonical_sha256(result.raw_response)
    production = request.evidence_class == "production"
    return ObservationImportRecord(
        evidence_class=request.evidence_class,
        authority_sha256=request.authority_sha256 if production else None,
        model_id=request.model_id if production else None,
        config_sha256=request.config_sha256 if production else None,
        raw_response=result.raw_response if production else None,
        raw_response_sha256=raw_hash if production else None,
        question_id=request.question_id,
        video_id=request.video_id,
        provider_identity=ProviderIdentity(
            provider=result.provider,
            model_revision=request.model_revision,
            decode_config=dict(result.decode_config),
            device_name=result.device_name,
        ),
        state=request.state,
        action=request.action,
        observation=observation,
    )


def check_stack_assets(
    *,
    stack_path: str | Path,
    lock_path: str | Path,
    authority_path: str | Path | None,
) -> tuple[ExperimentStack, AssetLock, SealedProductionAuthority | None]:
    stack = load_experiment_stack(stack_path)
    lock = load_asset_lock(lock_path)
    if stack.stack_id != lock.stack_id or stack.logical_roles != lock.logical_roles:
        raise ValueError("Experiment Stack and asset lock identities differ")
    assert_verified_lock(lock, reverify=True)
    authority = load_sealed_authority(authority_path) if authority_path else None
    if authority is not None:
        for role in authority.models.__class__.model_fields:
            locked = lock.physical_assets[lock.logical_roles[role]]
            model = getattr(authority.models, role)
            if (model.canonical_id, model.immutable_revision) != (
                locked.repo_id,
                locked.immutable_revision,
            ):
                raise ValueError(
                    f"Authority model identity differs from asset lock: {role}"
                )
    return stack, lock, authority


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _load_raw(path: Path) -> ObservationImportRecord:
    payload = json.loads(path.read_text(encoding="utf-8"))
    request_key = payload.pop("request_key", None)
    record = ObservationImportRecord.model_validate(payload)
    if request_key != path.stem:
        raise ValueError("raw provider record request key differs from filename")
    return record


def _validate_existing_request(
    request: ExecutionRequest, record: ObservationImportRecord
) -> None:
    expected = {
        "evidence_class": request.evidence_class,
        "authority_sha256": request.authority_sha256,
        "model_id": (
            request.model_id if request.evidence_class == "production" else None
        ),
        "config_sha256": (
            request.config_sha256 if request.evidence_class == "production" else None
        ),
        "question_id": request.question_id,
        "video_id": request.video_id,
        "model_revision": request.model_revision,
        "state": request.state,
        "action": request.action,
    }
    observed = {
        "evidence_class": record.evidence_class,
        "authority_sha256": record.authority_sha256,
        "model_id": record.model_id,
        "config_sha256": record.config_sha256,
        "question_id": record.question_id,
        "video_id": record.video_id,
        "model_revision": record.provider_identity.model_revision,
        "state": record.state,
        "action": record.action,
    }
    if observed != expected:
        raise ValueError("resumed provider record differs from request identity")


def execute_batch(
    requests: tuple[ExecutionRequest, ...],
    *,
    backend: StackV1Backend,
    output_dir: str | Path,
    resume: bool,
    check_only: bool,
) -> dict[str, object]:
    if not requests:
        raise ValueError("provider request batch is empty")
    if len({item.request_key for item in requests}) != len(requests):
        raise ValueError("provider request batch contains duplicate identities")
    destination = Path(output_dir)
    raw_root = destination / "raw"
    existing: dict[str, ObservationImportRecord] = {}
    if raw_root.exists():
        if not resume and any(raw_root.glob("*.json")):
            raise ValueError("provider output exists; use --resume")
        existing = {path.stem: _load_raw(path) for path in raw_root.glob("*.json")}
    cache_hits = 0
    generated = 0
    records: dict[str, ObservationImportRecord] = {}
    for request in requests:
        backend.check(request)
        if request.request_key in existing:
            _validate_existing_request(request, existing[request.request_key])
            cache_hits += 1
            records[request.request_key] = existing[request.request_key]
            continue
        if check_only:
            continue
        result = backend.execute(request)
        record = canonical_import_record(request, result)
        payload = {
            "request_key": request.request_key,
            **record.model_dump(mode="json"),
        }
        _atomic_bytes(
            raw_root / f"{request.request_key}.json", canonical_json_bytes(payload)
        )
        records[request.request_key] = record
        generated += 1
    if not check_only:
        lines = "".join(
            canonical_json(records[key].model_dump(mode="json")) + "\n"
            for key in sorted(records)
        )
        _atomic_bytes(destination / "provider.jsonl", lines.encode("utf-8"))
    return {
        "status": "CHECK_PASSED" if check_only else "COMPLETED",
        "request_count": len(requests),
        "resume_hits": cache_hits,
        "generated": generated,
        "provider_jsonl": str((destination / "provider.jsonl").resolve()),
    }
