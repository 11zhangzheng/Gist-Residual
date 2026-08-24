"""Identity-bound, atomic and resumable multi-round DAgger workflow."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from decimal import Decimal
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from fidmem.actions.environment import EnvironmentTransition, MemoryEnvironment
from fidmem.agent.runner import RouterPolicy
from fidmem.oracle.labels import CostNormalization
from fidmem.router.dataset import OracleBCDataset, OracleBCRecord
from fidmem.types import RouterState

from .dagger_core import (
    CachedUtilityGraph,
    Deviation,
    DeviationAuthority,
    ForbiddenObservationGenerator,
    PolicyIdentity,
    _evaluate_dev,
    collect_deviations,
    policy_identity,
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("refusing to replace an artifact symlink")
    handle = tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class Task8DaggerProvenance(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    asset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    group_assignment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    longroute_example_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Task9DaggerProvenance(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    normalization_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    normalization: CostNormalization
    preference_set_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def normalization_identity_matches(self) -> "Task9DaggerProvenance":
        digest = _sha256_bytes(
            _canonical_json(self.normalization.model_dump(mode="json")).encode("utf-8")
        )
        if digest != self.normalization_manifest_hash:
            raise ValueError("Task 9 normalization identity mismatch")
        return self


class Task10DaggerProvenance(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_record_ids: tuple[str, ...] = Field(min_length=1)


class DAggerQuestionContext(BaseModel):
    """Question authority reconstructed from cached replay and Task 8/9/10."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    question_id: str = Field(min_length=1)
    video_group_id: str = Field(min_length=1)
    initial_state: RouterState
    initial_replay_transitions: tuple[EnvironmentTransition, ...] = ()
    state: RouterState
    environment: MemoryEnvironment
    dataset: OracleBCDataset
    observation_snapshot_id: str = Field(min_length=1)
    snapshot: CachedUtilityGraph
    task8: Task8DaggerProvenance
    task9: Task9DaggerProvenance
    task10: Task10DaggerProvenance

    @model_validator(mode="after")
    def authority_matches_dataset(self) -> "DAggerQuestionContext":
        if type(self.environment.executor) is not ForbiddenObservationGenerator:
            raise ValueError(
                "DAgger environment must inject exact ForbiddenObservationGenerator"
            )
        current = self.initial_state
        for persisted in self.initial_replay_transitions:
            if persisted.state != current:
                raise ValueError("initial replay transition state chain mismatch")
            cached = self.snapshot.get(current, persisted.action)
            if cached is None or cached != persisted.observation:
                raise ValueError("initial replay transition is absent from cache")
            replayed = self.environment.replay(current, persisted.action, cached)
            if replayed != persisted:
                raise ValueError("initial replay transition is not authoritative")
            current = replayed.next_state
        if current != self.state:
            raise ValueError("initial replay does not derive context state")
        if tuple(self.state.action_history) != tuple(
            transition.action for transition in self.initial_replay_transitions
        ):
            raise ValueError("context action history is not replay-derived")
        if self.dataset.identity != self.task10.dataset_identity:
            raise ValueError("Task 10 dataset identity does not match context dataset")
        if tuple(record.record_id for record in self.dataset.records) != (
            self.task10.base_record_ids
        ):
            raise ValueError("Task 10 base record set does not match context dataset")
        matches = tuple(
            record
            for record in self.dataset.records
            if record.question_id == self.question_id
            and record.video_id == self.video_group_id
            and record.state == self.state
            and record.observation_snapshot_id == self.observation_snapshot_id
        )
        if len(matches) != 1:
            raise ValueError(
                "context question/state/snapshot must match exactly one base record"
            )
        provenance = matches[0].provenance
        expected_task8 = Task8DaggerProvenance(
            dataset_manifest_hash=provenance.dataset_manifest_hash,
            source_manifest_hash=provenance.source_manifest_hash,
            asset_sha256=provenance.asset_sha256,
            group_assignment_sha256=provenance.group_assignment_sha256,
            longroute_example_sha256=provenance.longroute_example_sha256,
        )
        expected_task9 = Task9DaggerProvenance(
            normalization_manifest_hash=provenance.normalization_manifest_hash,
            normalization=provenance.normalization,
            preference_set_hash=provenance.preference_set_hash,
        )
        if self.task8 != expected_task8:
            raise ValueError("Task 8 provenance does not match base record")
        if self.task9 != expected_task9:
            raise ValueError("Task 9 provenance does not match base record")
        if self.state.cost_preference != provenance.selected_preference:
            raise ValueError("context preference does not match Task 9 provenance")
        return self

    @classmethod
    def from_record(
        cls,
        *,
        record: OracleBCRecord,
        dataset: OracleBCDataset,
        environment: MemoryEnvironment,
        snapshot: CachedUtilityGraph,
        initial_state: RouterState | None = None,
        initial_replay_transitions: Sequence[EnvironmentTransition] = (),
    ) -> "DAggerQuestionContext":
        transitions = tuple(initial_replay_transitions)
        root = record.state if initial_state is None else initial_state
        if record.state.action_history and not transitions:
            raise ValueError(
                "pre-acquired Task 10 record requires authoritative initial replay"
            )
        provenance = record.provenance
        return cls(
            question_id=record.question_id,
            video_group_id=record.video_id,
            initial_state=root,
            initial_replay_transitions=transitions,
            state=record.state,
            environment=environment,
            dataset=dataset,
            observation_snapshot_id=record.observation_snapshot_id,
            snapshot=snapshot,
            task8=Task8DaggerProvenance(
                dataset_manifest_hash=provenance.dataset_manifest_hash,
                source_manifest_hash=provenance.source_manifest_hash,
                asset_sha256=provenance.asset_sha256,
                group_assignment_sha256=provenance.group_assignment_sha256,
                longroute_example_sha256=provenance.longroute_example_sha256,
            ),
            task9=Task9DaggerProvenance(
                normalization_manifest_hash=provenance.normalization_manifest_hash,
                normalization=provenance.normalization,
                preference_set_hash=provenance.preference_set_hash,
            ),
            task10=Task10DaggerProvenance(
                dataset_identity=dataset.identity,
                base_record_ids=tuple(record.record_id for record in dataset.records),
            ),
        )

    @property
    def base_record(self) -> OracleBCRecord:
        return next(
            record
            for record in self.dataset.records
            if record.question_id == self.question_id
            and record.video_id == self.video_group_id
            and record.state == self.state
            and record.observation_snapshot_id == self.observation_snapshot_id
        )

    @property
    def environment_identity(self) -> str:
        payload = {
            "events": tuple(
                event.model_dump(mode="json")
                for event in self.environment.canonical_events
            ),
            "costs": self.environment.costs.model_dump(mode="json"),
            "action_semantics_version": self.environment.action_semantics_version,
            "executor_identity": (
                "fidmem.router.dagger_core.ForbiddenObservationGenerator/v1"
            ),
        }
        return _sha256_bytes(_canonical_json(payload).encode("utf-8"))

    @property
    def identity(self) -> str:
        payload = {
            "question_id": self.question_id,
            "video_group_id": self.video_group_id,
            "initial_state": self.initial_state.model_dump(mode="json"),
            "initial_replay_transitions": tuple(
                item.model_dump(mode="json") for item in self.initial_replay_transitions
            ),
            "state": self.state.model_dump(mode="json"),
            "environment_identity": self.environment_identity,
            "observation_snapshot_id": self.observation_snapshot_id,
            "snapshot_identity": self.snapshot.identity.model_dump(mode="json"),
            "task8": self.task8.model_dump(mode="json"),
            "task9": self.task9.model_dump(mode="json"),
            "task10": self.task10.model_dump(mode="json"),
        }
        return _sha256_bytes(_canonical_json(payload).encode("utf-8"))

    @property
    def deviation_authority(self) -> DeviationAuthority:
        return DeviationAuthority(
            context_identity=self.identity,
            video_group_id=self.video_group_id,
            observation_snapshot_id=self.observation_snapshot_id,
            base_dataset_identity=self.dataset.identity,
            base_record_id=self.base_record.record_id,
            dataset_manifest_hash=self.task8.dataset_manifest_hash,
            source_manifest_hash=self.task8.source_manifest_hash,
            asset_sha256=self.task8.asset_sha256,
            group_assignment_sha256=self.task8.group_assignment_sha256,
            longroute_example_sha256=self.task8.longroute_example_sha256,
            normalization_manifest_hash=self.task9.normalization_manifest_hash,
            preference_set_hash=self.task9.preference_set_hash,
        )


def select_train_question_subset(
    contexts: Sequence[DAggerQuestionContext],
    *,
    fraction: float,
    seed: int,
) -> tuple[DAggerQuestionContext, ...]:
    """Canonical hash selection independent of input/worker/world ordering."""

    records = tuple(contexts)
    if not records:
        raise ValueError("train question contexts must be non-empty")
    if not math.isfinite(fraction) or fraction <= 0 or fraction > 1:
        raise ValueError("train question subset fraction must be in (0, 1]")
    identities = tuple((item.question_id, item.video_group_id) for item in records)
    if len(set(identities)) != len(identities):
        raise ValueError("question/video contexts must be unique")

    def order(item: DAggerQuestionContext) -> tuple[str, str, str]:
        digest = _sha256_bytes(
            f"{seed}|{item.question_id}|{item.video_group_id}|{item.identity}".encode(
                "utf-8"
            )
        )
        return digest, item.question_id, item.video_group_id

    ordered = tuple(sorted(records, key=order))
    count = max(1, int(math.ceil(len(ordered) * fraction)))
    return ordered[:count]


def _path_has_symlink_component(path: Path) -> bool:
    absolute = path.absolute()
    candidates = (absolute, *absolute.parents)
    return any(candidate.is_symlink() for candidate in candidates)


def _paths_overlap(first: Path, second: Path) -> bool:
    return (
        first == second or first.is_relative_to(second) or second.is_relative_to(first)
    )


class DAggerConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    artifact_root: Path = Path("artifacts/dagger")
    bootstrap_checkpoint: Path = Path("bootstrap.pt")
    generations_dir: Path = Path("generations")
    current_pointer: Path = Path("current.json")
    min_rounds: int = Field(default=2, ge=2, le=3)
    max_rounds: int = Field(default=3, ge=2, le=3)
    beam_size: int = Field(default=8, ge=1)
    max_depth: int = Field(default=5, ge=1)
    utility_gain_threshold: float = Field(default=0.005, ge=0)
    regret_improvement_ratio: float = Field(default=0.02, ge=0)
    train_question_subset_fraction: float = Field(default=1.0, gt=0, le=1)
    train_question_subset_seed: int = 2026
    budget_bin_width: float = Field(default=1.0, gt=0)

    @field_validator(
        "artifact_root",
        "bootstrap_checkpoint",
        "generations_dir",
        "current_pointer",
        mode="before",
    )
    @classmethod
    def paths_have_strict_types(cls, value: object) -> object:
        if not isinstance(value, (str, Path)) or (
            isinstance(value, str) and not value.strip()
        ):
            raise ValueError("DAgger paths must be non-empty strings or Path values")
        return value

    def resolve_path(self, value: Path, *, directory: bool = False) -> Path:
        root = self.artifact_root.resolve()
        if value.is_absolute() or ".." in value.parts:
            raise ValueError("DAgger paths must be canonical relative paths")
        target = (root / value).resolve()
        if not target.is_relative_to(root) or target == root:
            raise ValueError("DAgger artifact paths must stay under artifact_root")
        if directory and target.suffix:
            raise ValueError("DAgger directory path must not have a suffix")
        return target

    @model_validator(mode="after")
    def paths_are_contained(self) -> "DAggerConfig":
        if self.min_rounds > self.max_rounds:
            raise ValueError("min_rounds cannot exceed max_rounds")
        unresolved_root = self.artifact_root.absolute()
        if _path_has_symlink_component(unresolved_root):
            raise ValueError("artifact_root must not use a symlink alias")
        if unresolved_root.exists() and not unresolved_root.is_dir():
            raise ValueError("artifact_root must be a directory")
        bootstrap = self.resolve_path(self.bootstrap_checkpoint)
        generations = self.resolve_path(self.generations_dir, directory=True)
        current = self.resolve_path(self.current_pointer)
        for unresolved in (
            unresolved_root / self.bootstrap_checkpoint,
            unresolved_root / self.generations_dir,
            unresolved_root / self.current_pointer,
        ):
            if _path_has_symlink_component(unresolved):
                raise ValueError("DAgger artifact paths must not use symlink aliases")
        if generations.exists() and not generations.is_dir():
            raise ValueError("generations_dir must be a directory")
        for key_file in (bootstrap, current):
            if key_file.exists() and not key_file.is_file():
                raise ValueError("DAgger key artifact paths must be files")
        if bootstrap == current:
            raise ValueError(
                "bootstrap checkpoint and current pointer must be distinct"
            )
        if _paths_overlap(generations, bootstrap) or _paths_overlap(
            generations, current
        ):
            raise ValueError("generations_dir must not overlap key artifact paths")
        return self

    @property
    def bootstrap_path(self) -> Path:
        return self.resolve_path(self.bootstrap_checkpoint)


@dataclass(frozen=True)
class PolicyTrainingResult:
    policy: RouterPolicy
    checkpoint_path: Path
    checkpoint_sha256: str
    aggregated_dataset_identity: str


class PolicyTrainer(Protocol):
    def dataset_identity(
        self, *, base_dataset: OracleBCDataset, deviations: tuple[Deviation, ...]
    ) -> str:
        ...

    def train(
        self,
        *,
        round_number: int,
        base_dataset: OracleBCDataset,
        deviations: tuple[Deviation, ...],
        new_deviations: tuple[Deviation, ...],
        source_policy_checkpoint: Path,
        output_checkpoint: Path,
    ) -> PolicyTrainingResult:
        ...

    def load_policy(
        self,
        *,
        checkpoint: Path,
        base_dataset: OracleBCDataset,
        deviations: tuple[Deviation, ...],
    ) -> RouterPolicy:
        ...


class ArtifactReference(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def canonical_relative(self) -> "ArtifactReference":
        candidate = Path(self.path)
        if (
            candidate.is_absolute()
            or ".." in candidate.parts
            or candidate.as_posix() != self.path
            or len(candidate.parts) != 1
        ):
            raise ValueError("artifact reference must be one canonical relative path")
        return self


class RoundMetrics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    dev_utility: float
    cost_regret: float = Field(ge=0)


class RoundThresholds(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    utility_gain: float = Field(ge=0)
    regret_improvement_ratio: float = Field(ge=0)


class DAggerRoundManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    schema_version: Literal[3] = 3
    run_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    round_number: int = Field(ge=1, le=3)
    generation: str = Field(min_length=1)
    previous_generation_id: str | None
    previous_generation_manifest_sha256: str | None
    manifest_path: Literal["manifest.json"] = "manifest.json"
    status: Literal["completed", "stopped"]
    stop_reason: Literal["continue", "threshold_not_met", "max_rounds"]
    source_policy: ArtifactReference
    source_policy_identity: PolicyIdentity
    checkpoint: ArtifactReference
    checkpoint_policy_identity: PolicyIdentity
    seen_keys: ArtifactReference
    dev_artifact: ArtifactReference
    deviation_artifact: ArtifactReference
    base_dataset_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    aggregated_dataset_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    train_subset_question_ids: tuple[str, ...] = Field(min_length=1)
    train_subset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_identities: tuple[str, ...] = Field(min_length=1)
    seen_key_count: int = Field(ge=0)
    deviation_count: int = Field(ge=0)
    new_deviation_count: int = Field(ge=0)
    thresholds: RoundThresholds
    metrics: RoundMetrics
    budget_bin_width: float = Field(gt=0)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def self_hash_matches(self) -> "DAggerRoundManifest":
        if self.round_number == 1:
            if (
                self.previous_generation_id is not None
                or self.previous_generation_manifest_sha256 is not None
            ):
                raise ValueError("round one must not claim a previous generation")
        elif (
            self.previous_generation_id is None
            or self.previous_generation_manifest_sha256 is None
        ):
            raise ValueError("later rounds must bind a previous generation")
        if self.previous_generation_id is not None:
            previous = Path(self.previous_generation_id)
            if (
                previous.is_absolute()
                or ".." in previous.parts
                or previous.as_posix() != self.previous_generation_id
            ):
                raise ValueError("previous generation identity is not canonical")
        if self.previous_generation_manifest_sha256 is not None and (
            len(self.previous_generation_manifest_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.previous_generation_manifest_sha256
            )
        ):
            raise ValueError("previous generation manifest identity is invalid")
        expected = _sha256_bytes(
            _canonical_json(
                self.model_dump(mode="json", exclude={"manifest_sha256"})
            ).encode("utf-8")
        )
        if expected != self.manifest_sha256:
            raise ValueError("round manifest self hash mismatch")
        return self


class DAggerRunResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    run_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["stopped"]
    resumed: bool
    final_checkpoint: Path
    manifests: tuple[DAggerRoundManifest, ...] = Field(min_length=1)
    durability_warnings: tuple[str, ...] = ()


class _SeenKeysArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal[2] = 2
    run_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    round_number: int = Field(ge=1)
    keys: tuple[str, ...]
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_keys_and_hash(self) -> "_SeenKeysArtifact":
        if self.keys != tuple(sorted(set(self.keys))) or any(
            len(key) != 64
            or any(character not in "0123456789abcdef" for character in key)
            for key in self.keys
        ):
            raise ValueError("seen keys must be sorted unique SHA-256 values")
        _validate_self_hash(self, "artifact_sha256", "seen key artifact")
        return self


class _DeviationArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal[2] = 2
    run_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    round_number: int = Field(ge=1)
    deviations: tuple[Deviation, ...]
    new_state_keys: tuple[str, ...]
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_contents(self) -> "_DeviationArtifact":
        keys = tuple(item.state_key for item in self.deviations)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("deviations must be canonical and unique")
        if self.new_state_keys != tuple(sorted(set(self.new_state_keys))) or any(
            key not in set(keys) for key in self.new_state_keys
        ):
            raise ValueError("new deviation keys are invalid")
        _validate_self_hash(self, "artifact_sha256", "deviation artifact")
        return self


class _ContextMetric(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    context_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    utility: float
    cost_regret: float = Field(ge=0)


class _DevArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    schema_version: Literal[2] = 2
    run_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    round_number: int = Field(ge=1)
    context_metrics: tuple[_ContextMetric, ...]
    metrics: RoundMetrics
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_metrics(self) -> "_DevArtifact":
        identities = tuple(item.context_identity for item in self.context_metrics)
        if identities != tuple(sorted(set(identities))) or not identities:
            raise ValueError("dev context metrics must be sorted unique")
        expected = RoundMetrics(
            dev_utility=math.fsum(item.utility for item in self.context_metrics)
            / len(self.context_metrics),
            cost_regret=math.fsum(item.cost_regret for item in self.context_metrics)
            / len(self.context_metrics),
        )
        if self.metrics != expected:
            raise ValueError("dev aggregate metrics mismatch")
        _validate_self_hash(self, "artifact_sha256", "dev artifact")
        return self


class _CurrentPointer(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal[1] = 1
    run_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    round_number: int = Field(ge=1)
    generation: str = Field(min_length=1)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pointer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_hash(self) -> "_CurrentPointer":
        _validate_self_hash(self, "pointer_sha256", "current pointer")
        return self


def _validate_self_hash(model: BaseModel, field: str, label: str) -> None:
    expected = _sha256_bytes(
        _canonical_json(model.model_dump(mode="json", exclude={field})).encode("utf-8")
    )
    if getattr(model, field) != expected:
        raise ValueError(f"{label} self hash mismatch")


def _sealed(model_type: type[BaseModel], **payload: object) -> BaseModel:
    body = {"schema_version": 2, **payload}
    provisional = model_type.model_construct(**body, artifact_sha256="0" * 64)
    canonical = provisional.model_dump(mode="json", exclude={"artifact_sha256"})
    digest = _sha256_bytes(_canonical_json(canonical).encode("utf-8"))
    return model_type(**body, artifact_sha256=digest)


def _write_model(path: Path, model: BaseModel) -> None:
    _atomic_write(
        path,
        (_canonical_json(model.model_dump(mode="json")) + "\n").encode("utf-8"),
    )


def _build_manifest(**payload: object) -> DAggerRoundManifest:
    normalized = {
        key: value.model_dump(mode="json") if isinstance(value, BaseModel) else value
        for key, value in payload.items()
    }
    body = {"schema_version": 3, **normalized}
    digest = _sha256_bytes(_canonical_json(body).encode("utf-8"))
    return DAggerRoundManifest(**body, manifest_sha256=digest)


def _generation_name(round_number: int) -> str:
    return f"round-{round_number:04d}"


def _generation_relative(config: DAggerConfig, round_number: int) -> str:
    return (config.generations_dir / _generation_name(round_number)).as_posix()


@dataclass(frozen=True)
class GenerationCommitOutcome:
    generation: Path
    committed: Literal[True] = True
    durable: bool = True
    durability_warning: str | None = None


class CommittedGenerationVerificationError(RuntimeError):
    """The pointer crossed the commit boundary but verification failed."""


class DaggerRoundStore:
    """Same-root transactional store with current pointer published last."""

    def __init__(self, config: DAggerConfig) -> None:
        self.config = config
        self.root = config.artifact_root.resolve()
        self.generations = config.resolve_path(config.generations_dir, directory=True)
        self.current = config.resolve_path(config.current_pointer)
        self.root.mkdir(parents=True, exist_ok=True)
        self.generations.mkdir(parents=True, exist_ok=True)

    def _round_path(self, round_number: int) -> Path:
        return self.generations / _generation_name(round_number)

    def _staging_path(self, round_number: int) -> Path:
        return self.generations / f".{_generation_name(round_number)}.staging"

    def recover(self) -> None:
        current_round = 0
        if self.current.exists():
            current_round = self.load_pointer().round_number
        for candidate in self.generations.iterdir():
            if candidate.name.startswith(".round-") and candidate.name.endswith(
                ".staging"
            ):
                self._remove(candidate)
        for round_number in range(current_round + 1, self.config.max_rounds + 1):
            orphan = self._round_path(round_number)
            if orphan.exists():
                self._remove(orphan)

    def _remove(self, candidate: Path) -> None:
        resolved_parent = candidate.parent.resolve()
        if resolved_parent != self.generations or not (
            candidate.name.startswith(".round-") or candidate.name.startswith("round-")
        ):
            raise ValueError("refusing to remove an unsafe DAgger generation")
        if candidate.is_symlink():
            candidate.unlink()
        elif candidate.is_dir():
            shutil.rmtree(candidate)
        elif candidate.exists():
            candidate.unlink()

    def begin(self, round_number: int) -> Path:
        staging = self._staging_path(round_number)
        final = self._round_path(round_number)
        if final.exists():
            raise ValueError("immutable DAgger generation already exists")
        if staging.exists():
            self._remove(staging)
        staging.mkdir()
        return staging

    def rollback(self, staging: Path) -> None:
        if staging.exists():
            self._remove(staging)

    def _pointer_for(self, manifest: DAggerRoundManifest) -> _CurrentPointer:
        pointer_body = {
            "schema_version": 1,
            "run_identity": manifest.run_identity,
            "round_number": manifest.round_number,
            "generation": manifest.generation,
            "manifest_sha256": manifest.manifest_sha256,
        }
        return _CurrentPointer(
            **pointer_body,
            pointer_sha256=_sha256_bytes(_canonical_json(pointer_body).encode("utf-8")),
        )

    def _pointer_matches(self, expected: _CurrentPointer) -> bool:
        try:
            return self.load_pointer() == expected
        except (OSError, ValueError):
            return False

    def _verify_committed_generation(
        self,
        final: Path,
        manifest: DAggerRoundManifest,
        pointer: _CurrentPointer,
    ) -> None:
        if not self._pointer_matches(pointer):
            raise ValueError("current pointer does not identify committed generation")
        manifest_path = final / "manifest.json"
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise ValueError("committed manifest is missing or unsafe")
        loaded = DAggerRoundManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        if loaded != manifest:
            raise ValueError("committed manifest does not match publication")
        for reference, name in (
            (manifest.source_policy, "source.pt"),
            (manifest.checkpoint, "checkpoint.pt"),
            (manifest.seen_keys, "seen.json"),
            (manifest.dev_artifact, "dev.json"),
            (manifest.deviation_artifact, "deviations.json"),
        ):
            _artifact_path(final, reference, name)

    def _warning_outcome(
        self,
        *,
        final: Path,
        manifest: DAggerRoundManifest,
        pointer: _CurrentPointer,
        error: BaseException,
    ) -> GenerationCommitOutcome:
        try:
            self._verify_committed_generation(final, manifest, pointer)
        except Exception as verification_error:
            raise CommittedGenerationVerificationError(
                "current pointer was replaced, but committed generation verification failed"
            ) from verification_error
        return GenerationCommitOutcome(
            generation=final,
            durable=False,
            durability_warning=(
                f"round {manifest.round_number} committed but root durability "
                f"confirmation failed: {error}"
            ),
        )

    def commit(
        self, staging: Path, manifest: DAggerRoundManifest
    ) -> GenerationCommitOutcome:
        final = self._round_path(manifest.round_number)
        _write_model(staging / "manifest.json", manifest)
        for child in staging.iterdir():
            if child.is_symlink() or not child.is_file():
                raise ValueError("generation contains an unsafe artifact")
            with child.open("rb+") as handle:
                os.fsync(handle.fileno())
        _fsync_directory(staging)
        try:
            os.replace(staging, final)
        except Exception:
            if final.exists() and not staging.exists():
                self._remove(final)
            raise
        committed = False
        pointer_temporary: Path | None = None
        try:
            _fsync_directory(self.generations)
            pointer = self._pointer_for(manifest)
            if self.current.is_symlink():
                raise ValueError("refusing to replace a current pointer symlink")
            handle = tempfile.NamedTemporaryFile(
                dir=self.root, prefix=f".{self.current.name}.", delete=False
            )
            pointer_temporary = Path(handle.name)
            with handle:
                handle.write(
                    (_canonical_json(pointer.model_dump(mode="json")) + "\n").encode(
                        "utf-8"
                    )
                )
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.replace(pointer_temporary, self.current)
            except Exception as error:
                if self._pointer_matches(pointer):
                    committed = True
                    return self._warning_outcome(
                        final=final,
                        manifest=manifest,
                        pointer=pointer,
                        error=error,
                    )
                raise
            committed = True
            try:
                _fsync_directory(self.root)
            except Exception as error:
                return self._warning_outcome(
                    final=final,
                    manifest=manifest,
                    pointer=pointer,
                    error=error,
                )
            return GenerationCommitOutcome(generation=final)
        finally:
            if pointer_temporary is not None:
                pointer_temporary.unlink(missing_ok=True)
            if not committed and final.exists():
                self._remove(final)

    def load_pointer(self) -> _CurrentPointer:
        if not self.current.is_file() or self.current.is_symlink():
            raise ValueError("current pointer is missing or unsafe")
        pointer = _CurrentPointer.model_validate_json(
            self.current.read_text(encoding="utf-8")
        )
        expected = _generation_relative(self.config, pointer.round_number)
        if pointer.generation != expected:
            raise ValueError("current pointer target is invalid")
        return pointer


def _fsync_directory(path: Path) -> None:
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _artifact_path(
    generation: Path, reference: ArtifactReference, expected_name: str
) -> Path:
    if reference.path != expected_name or generation.is_symlink():
        raise ValueError("generation artifact path is not canonical")
    target = generation / reference.path
    if (
        not target.is_file()
        or target.is_symlink()
        or not target.resolve().is_relative_to(generation.resolve())
    ):
        raise ValueError("generation artifact is missing or unsafe")
    if _sha256_file(target) != reference.sha256:
        raise ValueError("generation artifact identity mismatch")
    return target


def _artifact_reference(path: Path) -> ArtifactReference:
    return ArtifactReference(path=path.name, sha256=_sha256_file(path))


def _run_identity(
    *,
    contexts: Sequence[DAggerQuestionContext],
    dev_contexts: Sequence[DAggerQuestionContext],
    config: DAggerConfig,
    source_policy_identity: PolicyIdentity,
) -> str:
    payload = {
        "train_contexts": tuple(sorted(item.identity for item in contexts)),
        "dev_contexts": tuple(sorted(item.identity for item in dev_contexts)),
        "config": config.model_dump(mode="json"),
        "source_policy_identity": source_policy_identity.model_dump(mode="json"),
    }
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def _average_dev_metrics(
    contexts: Sequence[DAggerQuestionContext],
    *,
    policy: RouterPolicy,
    config: DAggerConfig,
) -> tuple[RoundMetrics, tuple[_ContextMetric, ...]]:
    entries: list[_ContextMetric] = []
    for context in sorted(contexts, key=lambda item: item.identity):
        utility, regret = _evaluate_dev(
            (context.state,),
            policy=policy,
            environment=context.environment,
            utility_graph=context.snapshot,
            normalization=context.task9.normalization,
            beam_size=config.beam_size,
            max_depth=config.max_depth,
        )
        entries.append(
            _ContextMetric(
                context_identity=context.identity,
                utility=utility,
                cost_regret=regret,
            )
        )
    metrics = RoundMetrics(
        dev_utility=math.fsum(item.utility for item in entries) / len(entries),
        cost_regret=math.fsum(item.cost_regret for item in entries) / len(entries),
    )
    return metrics, tuple(entries)


def _stop_decision(
    round_number: int,
    metrics: RoundMetrics,
    previous: RoundMetrics | None,
    config: DAggerConfig,
) -> tuple[
    Literal["completed", "stopped"],
    Literal["continue", "threshold_not_met", "max_rounds"],
]:
    if round_number >= config.max_rounds:
        return "stopped", "max_rounds"
    if round_number < config.min_rounds:
        return "completed", "continue"
    if previous is None:
        raise ValueError("round stop decision requires prior metrics")
    utility_gain = Decimal(str(metrics.dev_utility)) - Decimal(
        str(previous.dev_utility)
    )
    regret_improvement = Decimal(str(previous.cost_regret)) - Decimal(
        str(metrics.cost_regret)
    )
    regret_threshold = Decimal(str(config.regret_improvement_ratio)) * Decimal(
        str(previous.cost_regret)
    )
    continues = utility_gain >= Decimal(str(config.utility_gain_threshold)) or (
        previous.cost_regret > 0 and regret_improvement >= regret_threshold
    )
    return (
        ("completed", "continue")
        if continues
        else (
            "stopped",
            "threshold_not_met",
        )
    )


def _load_generation(
    *,
    store: DaggerRoundStore,
    round_number: int,
    run_identity: str,
    dataset: OracleBCDataset,
    trainer: PolicyTrainer,
    subset_ids: tuple[str, ...],
    subset_hash: str,
    context_ids: tuple[str, ...],
    dev_context_ids: tuple[str, ...],
    source_identity: PolicyIdentity,
    previous_metrics: RoundMetrics | None,
    previous_manifest: DAggerRoundManifest | None,
    previous_deviation_keys: set[str],
) -> tuple[DAggerRoundManifest, tuple[Deviation, ...], set[str], RouterPolicy]:
    generation = store._round_path(round_number)
    if not generation.is_dir() or generation.is_symlink():
        raise ValueError("committed generation is missing or unsafe")
    manifest_path = generation / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("round manifest is missing or unsafe")
    manifest = DAggerRoundManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    expected_generation = _generation_relative(store.config, round_number)
    expected_previous_id = (
        None if previous_manifest is None else previous_manifest.generation
    )
    expected_previous_hash = (
        None if previous_manifest is None else previous_manifest.manifest_sha256
    )
    if (
        manifest.previous_generation_id != expected_previous_id
        or manifest.previous_generation_manifest_sha256 != expected_previous_hash
    ):
        raise ValueError("round manifest predecessor chain mismatch")
    if (
        manifest.round_number != round_number
        or manifest.generation != expected_generation
        or manifest.run_identity != run_identity
        or manifest.base_dataset_identity != dataset.identity
        or manifest.train_subset_question_ids != subset_ids
        or manifest.train_subset_sha256 != subset_hash
        or manifest.context_identities != context_ids
        or manifest.source_policy_identity != source_identity
        or manifest.budget_bin_width != store.config.budget_bin_width
    ):
        raise ValueError("round manifest derived identity mismatch")
    expected_thresholds = RoundThresholds(
        utility_gain=store.config.utility_gain_threshold,
        regret_improvement_ratio=store.config.regret_improvement_ratio,
    )
    if manifest.thresholds != expected_thresholds:
        raise ValueError("round manifest threshold mismatch")
    _artifact_path(generation, manifest.source_policy, "source.pt")
    if manifest.source_policy.sha256 != (
        manifest.source_policy_identity.checkpoint_sha256
    ):
        raise ValueError("source policy content identity mismatch")
    seen_path = _artifact_path(generation, manifest.seen_keys, "seen.json")
    seen = _SeenKeysArtifact.model_validate_json(seen_path.read_text(encoding="utf-8"))
    if (
        seen.run_identity != run_identity
        or seen.round_number != round_number
        or len(seen.keys) != manifest.seen_key_count
    ):
        raise ValueError("seen key artifact derived fields mismatch")
    deviation_path = _artifact_path(
        generation, manifest.deviation_artifact, "deviations.json"
    )
    deviations_artifact = _DeviationArtifact.model_validate_json(
        deviation_path.read_text(encoding="utf-8")
    )
    deviations = deviations_artifact.deviations
    if (
        deviations_artifact.run_identity != run_identity
        or deviations_artifact.round_number != round_number
        or len(deviations) != manifest.deviation_count
        or len(deviations_artifact.new_state_keys) != manifest.new_deviation_count
    ):
        raise ValueError("deviation artifact derived fields mismatch")
    current_deviation_keys = {item.state_key for item in deviations}
    if not previous_deviation_keys.issubset(current_deviation_keys):
        raise ValueError("deviation chain removed a predecessor deviation")
    expected_new_keys = tuple(sorted(current_deviation_keys - previous_deviation_keys))
    if deviations_artifact.new_state_keys != expected_new_keys:
        raise ValueError("new deviation keys must equal the predecessor set difference")
    if any(item.state_key not in set(seen.keys) for item in deviations):
        raise ValueError("deviation keys are absent from seen artifact")
    dev_path = _artifact_path(generation, manifest.dev_artifact, "dev.json")
    dev = _DevArtifact.model_validate_json(dev_path.read_text(encoding="utf-8"))
    if (
        dev.run_identity != run_identity
        or dev.round_number != round_number
        or dev.metrics != manifest.metrics
        or tuple(item.context_identity for item in dev.context_metrics)
        != dev_context_ids
    ):
        raise ValueError("dev artifact derived fields mismatch")
    status, reason = _stop_decision(
        round_number, manifest.metrics, previous_metrics, store.config
    )
    if manifest.status != status or manifest.stop_reason != reason:
        raise ValueError("round manifest stop decision mismatch")
    aggregate_identity = trainer.dataset_identity(
        base_dataset=dataset, deviations=deviations
    )
    if aggregate_identity != manifest.aggregated_dataset_identity:
        raise ValueError("aggregated dataset identity mismatch")
    checkpoint = _artifact_path(generation, manifest.checkpoint, "checkpoint.pt")
    if (
        manifest.checkpoint.sha256
        != manifest.checkpoint_policy_identity.checkpoint_sha256
    ):
        raise ValueError("checkpoint policy content identity mismatch")
    policy = trainer.load_policy(
        checkpoint=checkpoint, base_dataset=dataset, deviations=deviations
    )
    if policy_identity(policy, checkpoint) != manifest.checkpoint_policy_identity:
        raise ValueError("loaded policy identity mismatch")
    return manifest, deviations, set(seen.keys), policy


def run_dagger(
    *,
    train_contexts: Sequence[DAggerQuestionContext],
    dev_contexts: Sequence[DAggerQuestionContext],
    initial_policy: RouterPolicy,
    trainer: PolicyTrainer,
    config: DAggerConfig,
    source_policy_checkpoint: Path | None = None,
) -> DAggerRunResult:
    train = tuple(train_contexts)
    dev = tuple(dev_contexts)
    if not train or not dev:
        raise ValueError("DAgger requires non-empty train and dev contexts")
    dataset = train[0].dataset
    if any(item.dataset.identity != dataset.identity for item in (*train, *dev)):
        raise ValueError("all DAgger contexts must share one Task 10 dataset")
    normalizations = {item.task9.normalization_manifest_hash for item in (*train, *dev)}
    if len(normalizations) != 1:
        raise ValueError("DAgger contexts must share one Task 9 normalization")
    bootstrap = config.bootstrap_path
    if (
        source_policy_checkpoint is not None
        and source_policy_checkpoint.resolve() != bootstrap
    ):
        raise ValueError("source policy must equal configured bootstrap checkpoint")
    bootstrap_identity = policy_identity(initial_policy, bootstrap)
    validated_bootstrap = trainer.load_policy(
        checkpoint=bootstrap,
        base_dataset=dataset,
        deviations=(),
    )
    if policy_identity(validated_bootstrap, bootstrap) != bootstrap_identity:
        raise ValueError("initial policy does not match validated bootstrap checkpoint")
    selected = select_train_question_subset(
        train,
        fraction=config.train_question_subset_fraction,
        seed=config.train_question_subset_seed,
    )
    subset_ids = tuple(item.question_id for item in selected)
    subset_hash = _sha256_bytes(_canonical_json(subset_ids).encode("utf-8"))
    context_ids = tuple(item.identity for item in selected)
    dev_context_ids = tuple(sorted(item.identity for item in dev))
    run_identity = _run_identity(
        contexts=selected,
        dev_contexts=dev,
        config=config,
        source_policy_identity=bootstrap_identity,
    )
    store = DaggerRoundStore(config)
    store.recover()
    manifests: list[DAggerRoundManifest] = []
    durability_warnings: list[str] = []
    deviations: tuple[Deviation, ...] = ()
    seen: set[str] = set()
    policy = validated_bootstrap
    current_checkpoint = bootstrap
    current_identity = bootstrap_identity
    previous_metrics: RoundMetrics | None = None
    resumed = False
    if store.current.exists():
        pointer = store.load_pointer()
        if pointer.run_identity != run_identity:
            raise ValueError("current pointer run identity mismatch")
        expected_identity = bootstrap_identity
        for round_number in range(1, pointer.round_number + 1):
            manifest, deviations, seen, policy = _load_generation(
                store=store,
                round_number=round_number,
                run_identity=run_identity,
                dataset=dataset,
                trainer=trainer,
                subset_ids=subset_ids,
                subset_hash=subset_hash,
                context_ids=context_ids,
                dev_context_ids=dev_context_ids,
                source_identity=expected_identity,
                previous_metrics=previous_metrics,
                previous_manifest=manifests[-1] if manifests else None,
                previous_deviation_keys={item.state_key for item in deviations},
            )
            manifests.append(manifest)
            expected_identity = manifest.checkpoint_policy_identity
            previous_metrics = manifest.metrics
        last = manifests[-1]
        if (
            pointer.generation != last.generation
            or pointer.manifest_sha256 != last.manifest_sha256
        ):
            raise ValueError("current pointer does not match its round manifest")
        current_checkpoint = store._round_path(last.round_number) / "checkpoint.pt"
        current_identity = last.checkpoint_policy_identity
        resumed = True
        if last.status == "stopped":
            return DAggerRunResult(
                run_identity=run_identity,
                status="stopped",
                resumed=True,
                final_checkpoint=current_checkpoint,
                manifests=tuple(manifests),
                durability_warnings=tuple(durability_warnings),
            )
    start_round = len(manifests) + 1
    for round_number in range(start_round, config.max_rounds + 1):
        round_seen = set(seen)
        new: list[Deviation] = []
        for context in selected:
            new.extend(
                collect_deviations(
                    (context.state,),
                    policy=policy,
                    environment=context.environment,
                    utility_graph=context.snapshot,
                    normalization=context.task9.normalization,
                    question_ids=(context.question_id,),
                    initial_replay_transitions=(context.initial_replay_transitions,),
                    authorities=(context.deviation_authority,),
                    seen_keys=round_seen,
                    budget_bin_width=config.budget_bin_width,
                    beam_size=config.beam_size,
                    max_depth=config.max_depth,
                )
            )
        by_key = {item.state_key: item for item in deviations}
        for item in new:
            by_key.setdefault(item.state_key, item)
        round_deviations = tuple(by_key[key] for key in sorted(by_key))
        staging = store.begin(round_number)
        try:
            shutil.copyfile(current_checkpoint, staging / "source.pt")
            source_ref = _artifact_reference(staging / "source.pt")
            if source_ref.sha256 != current_identity.checkpoint_sha256:
                raise ValueError("source policy checkpoint identity mismatch")
            output_checkpoint = staging / "checkpoint.pt"
            trained = trainer.train(
                round_number=round_number,
                base_dataset=dataset,
                deviations=round_deviations,
                new_deviations=tuple(new),
                source_policy_checkpoint=staging / "source.pt",
                output_checkpoint=output_checkpoint,
            )
            if trained.checkpoint_path.resolve() != output_checkpoint.resolve():
                raise ValueError("trainer returned an unexpected checkpoint path")
            checkpoint_ref = _artifact_reference(output_checkpoint)
            if checkpoint_ref.sha256 != trained.checkpoint_sha256:
                raise ValueError("trainer checkpoint identity mismatch")
            trained_identity = policy_identity(trained.policy, output_checkpoint)
            aggregate_identity = trainer.dataset_identity(
                base_dataset=dataset, deviations=round_deviations
            )
            if aggregate_identity != trained.aggregated_dataset_identity:
                raise ValueError("trainer aggregated dataset identity mismatch")
            metrics, entries = _average_dev_metrics(
                dev, policy=trained.policy, config=config
            )
            status, stop_reason = _stop_decision(
                round_number, metrics, previous_metrics, config
            )
            seen_artifact = _sealed(
                _SeenKeysArtifact,
                run_identity=run_identity,
                round_number=round_number,
                keys=tuple(sorted(round_seen)),
            )
            _write_model(staging / "seen.json", seen_artifact)
            deviation_artifact = _sealed(
                _DeviationArtifact,
                run_identity=run_identity,
                round_number=round_number,
                deviations=round_deviations,
                new_state_keys=tuple(sorted(item.state_key for item in new)),
            )
            _write_model(staging / "deviations.json", deviation_artifact)
            dev_artifact = _sealed(
                _DevArtifact,
                run_identity=run_identity,
                round_number=round_number,
                context_metrics=entries,
                metrics=metrics,
            )
            _write_model(staging / "dev.json", dev_artifact)
            manifest = _build_manifest(
                run_identity=run_identity,
                round_number=round_number,
                generation=_generation_relative(config, round_number),
                previous_generation_id=(
                    manifests[-1].generation if manifests else None
                ),
                previous_generation_manifest_sha256=(
                    manifests[-1].manifest_sha256 if manifests else None
                ),
                manifest_path="manifest.json",
                status=status,
                stop_reason=stop_reason,
                source_policy=source_ref,
                source_policy_identity=current_identity,
                checkpoint=checkpoint_ref,
                checkpoint_policy_identity=trained_identity,
                seen_keys=_artifact_reference(staging / "seen.json"),
                dev_artifact=_artifact_reference(staging / "dev.json"),
                deviation_artifact=_artifact_reference(staging / "deviations.json"),
                base_dataset_identity=dataset.identity,
                aggregated_dataset_identity=aggregate_identity,
                train_subset_question_ids=subset_ids,
                train_subset_sha256=subset_hash,
                context_identities=context_ids,
                seen_key_count=len(round_seen),
                deviation_count=len(round_deviations),
                new_deviation_count=len(new),
                thresholds=RoundThresholds(
                    utility_gain=config.utility_gain_threshold,
                    regret_improvement_ratio=config.regret_improvement_ratio,
                ),
                metrics=metrics,
                budget_bin_width=config.budget_bin_width,
            )
            outcome = store.commit(staging, manifest)
        except Exception:
            store.rollback(staging)
            raise
        final = outcome.generation
        if outcome.durability_warning is not None:
            durability_warnings.append(outcome.durability_warning)
        manifests.append(manifest)
        deviations = round_deviations
        seen = round_seen
        policy = trained.policy
        current_checkpoint = final / "checkpoint.pt"
        current_identity = trained_identity
        previous_metrics = metrics
        if status == "stopped":
            return DAggerRunResult(
                run_identity=run_identity,
                status="stopped",
                resumed=resumed,
                final_checkpoint=current_checkpoint,
                manifests=tuple(manifests),
                durability_warnings=tuple(durability_warnings),
            )
    raise RuntimeError("DAgger round loop ended without a stop decision")


__all__ = [
    "ArtifactReference",
    "DAggerConfig",
    "DAggerQuestionContext",
    "DAggerRoundManifest",
    "DAggerRunResult",
    "DaggerRoundStore",
    "PolicyTrainer",
    "PolicyTrainingResult",
    "Task10DaggerProvenance",
    "Task8DaggerProvenance",
    "Task9DaggerProvenance",
    "run_dagger",
    "select_train_question_subset",
]
