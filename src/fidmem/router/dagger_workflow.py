"""Identity-bound, atomic and resumable multi-round DAgger workflow."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fidmem.actions.environment import MemoryEnvironment
from fidmem.agent.runner import RouterPolicy
from fidmem.oracle.labels import CostNormalization
from fidmem.router.dataset import OracleBCDataset, OracleBCRecord
from fidmem.types import RouterState

from .dagger_core import (
    CachedUtilityGraph,
    DaggerRoundResult,
    Deviation,
    ForbiddenObservationGenerator,
    _evaluate_dev,
    _should_continue,
    collect_deviations,
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
    """Strict question, cache snapshot and Task 8/9/10 authority bundle."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    question_id: str = Field(min_length=1)
    video_group_id: str = Field(min_length=1)
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
        if not isinstance(
            getattr(self.environment, "_executor", None),
            ForbiddenObservationGenerator,
        ):
            raise ValueError(
                "DAgger environment must inject ForbiddenObservationGenerator"
            )
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
    ) -> "DAggerQuestionContext":
        provenance = record.provenance
        return cls(
            question_id=record.question_id,
            video_group_id=record.video_id,
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
    def identity(self) -> str:
        payload = {
            "question_id": self.question_id,
            "video_group_id": self.video_group_id,
            "state": self.state.model_dump(mode="json"),
            "observation_snapshot_id": self.observation_snapshot_id,
            "snapshot_identity": self.snapshot.identity.model_dump(mode="json"),
            "observation_identity": self.snapshot.observation_identity.model_dump(
                mode="json"
            ),
            "evaluator_identity": self.snapshot.evaluator_identity.model_dump(
                mode="json"
            ),
            "task8": self.task8.model_dump(mode="json"),
            "task9": self.task9.model_dump(mode="json"),
            "task10": self.task10.model_dump(mode="json"),
        }
        return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


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


class DAggerConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    artifact_root: Path = Path("artifacts/dagger")
    max_rounds: int = Field(default=3, ge=2, le=3)
    beam_size: int = Field(default=8, ge=1)
    max_depth: int = Field(default=5, ge=1)
    utility_gain_threshold: float = Field(default=0.005, ge=0)
    regret_improvement_ratio: float = Field(default=0.02, ge=0)
    train_question_subset_fraction: float = Field(default=1.0, gt=0, le=1)
    train_question_subset_seed: int = 2026
    budget_bin_width: float = Field(default=1.0, gt=0)
    seen_keys_path: Path = Path("seen-keys.json")
    deviation_artifact_path: Path = Path("deviations.json")
    manifest_dir: Path = Path("manifests")
    checkpoint_dir: Path = Path("checkpoints")

    def resolve_path(self, value: Path, *, directory: bool = False) -> Path:
        root = self.artifact_root.resolve()
        target = value.resolve() if value.is_absolute() else (root / value).resolve()
        if not target.is_relative_to(root) or target == root:
            raise ValueError("DAgger artifact paths must stay under artifact_root")
        if directory and target.suffix:
            raise ValueError("DAgger directory path must not have a suffix")
        return target

    @model_validator(mode="after")
    def paths_are_contained(self) -> "DAggerConfig":
        self.resolve_path(self.seen_keys_path)
        self.resolve_path(self.deviation_artifact_path)
        self.resolve_path(self.manifest_dir, directory=True)
        self.resolve_path(self.checkpoint_dir, directory=True)
        return self


@dataclass(frozen=True)
class PolicyTrainingResult:
    policy: RouterPolicy
    checkpoint_path: Path
    checkpoint_sha256: str
    aggregated_dataset_identity: str


class PolicyTrainer(Protocol):
    """Injected trainer boundary; production adapter invokes Task 10 train_bc."""

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

    schema_version: Literal[1] = 1
    run_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    round_number: int = Field(ge=1, le=3)
    status: Literal["completed", "stopped"]
    stop_reason: Literal["continue", "threshold_not_met", "max_rounds"]
    source_policy: ArtifactReference
    checkpoint: ArtifactReference
    base_dataset_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    aggregated_dataset_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    train_subset_question_ids: tuple[str, ...] = Field(min_length=1)
    train_subset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_identities: tuple[str, ...] = Field(min_length=1)
    seen_keys_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seen_key_count: int = Field(ge=0)
    deviation_artifact: ArtifactReference
    deviation_count: int = Field(ge=0)
    new_deviation_count: int = Field(ge=0)
    thresholds: RoundThresholds
    metrics: RoundMetrics
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def self_hash_matches(self) -> "DAggerRoundManifest":
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


class _SeenKeysArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    run_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
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
        expected = _sha256_bytes(
            _canonical_json(
                self.model_dump(mode="json", exclude={"artifact_sha256"})
            ).encode("utf-8")
        )
        if expected != self.artifact_sha256:
            raise ValueError("seen key artifact self hash mismatch")
        return self


def _seen_artifact(run_identity: str, keys: set[str]) -> _SeenKeysArtifact:
    base = {
        "schema_version": 1,
        "run_identity": run_identity,
        "keys": tuple(sorted(keys)),
    }
    return _SeenKeysArtifact(
        **base,
        artifact_sha256=_sha256_bytes(_canonical_json(base).encode("utf-8")),
    )


def _write_model(path: Path, model: BaseModel) -> None:
    _atomic_write(
        path,
        (_canonical_json(model.model_dump(mode="json")) + "\n").encode("utf-8"),
    )


def _deviation_path(config: DAggerConfig, round_number: int) -> Path:
    base = config.resolve_path(config.deviation_artifact_path)
    return base.with_name(f"{base.stem}-round-{round_number}{base.suffix}")


def _manifest_path(config: DAggerConfig, round_number: int) -> Path:
    return config.resolve_path(config.manifest_dir, directory=True) / (
        f"round-{round_number}.json"
    )


def _checkpoint_path(config: DAggerConfig, round_number: int) -> Path:
    return config.resolve_path(config.checkpoint_dir, directory=True) / (
        f"policy-round-{round_number}.pt"
    )


def _write_deviations(path: Path, deviations: Sequence[Deviation]) -> str:
    payload = {
        "schema_version": 1,
        "deviations": tuple(
            deviation.model_dump(mode="json") for deviation in deviations
        ),
    }
    encoded = (_canonical_json(payload) + "\n").encode("utf-8")
    _atomic_write(path, encoded)
    return _sha256_bytes(encoded)


def _load_deviations(path: Path, expected_sha256: str) -> tuple[Deviation, ...]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("deviation artifact is missing or not a regular file")
    if _sha256_file(path) != expected_sha256:
        raise ValueError("deviation artifact identity mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or set(payload)
        != {
            "schema_version",
            "deviations",
        }
        or payload["schema_version"] != 1
    ):
        raise ValueError("deviation artifact schema is invalid")
    return tuple(Deviation.model_validate(item) for item in payload["deviations"])


def _build_manifest(**payload: object) -> DAggerRoundManifest:
    normalized = {
        key: value.model_dump(mode="json") if isinstance(value, BaseModel) else value
        for key, value in payload.items()
    }
    base = {"schema_version": 1, **normalized}
    digest = _sha256_bytes(_canonical_json(base).encode("utf-8"))
    return DAggerRoundManifest(**base, manifest_sha256=digest)


def _run_identity(
    *,
    contexts: Sequence[DAggerQuestionContext],
    dev_contexts: Sequence[DAggerQuestionContext],
    config: DAggerConfig,
    source_policy_sha256: str,
) -> str:
    payload = {
        "train_contexts": tuple(sorted(item.identity for item in contexts)),
        "dev_contexts": tuple(sorted(item.identity for item in dev_contexts)),
        "config": config.model_dump(mode="json"),
        "source_policy_sha256": source_policy_sha256,
    }
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def _load_existing_manifests(
    config: DAggerConfig,
    *,
    run_identity: str,
    base_dataset_identity: str,
    subset_ids: tuple[str, ...],
    context_ids: tuple[str, ...],
) -> tuple[DAggerRoundManifest, ...]:
    manifests: list[DAggerRoundManifest] = []
    for round_number in range(1, config.max_rounds + 1):
        path = _manifest_path(config, round_number)
        if not path.exists():
            break
        manifest = DAggerRoundManifest.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        if manifest.round_number != round_number:
            raise ValueError("round manifest sequence is invalid")
        if manifest.run_identity != run_identity:
            raise ValueError("round manifest run identity mismatch")
        if manifest.base_dataset_identity != base_dataset_identity:
            raise ValueError("round manifest dataset identity mismatch")
        if manifest.train_subset_question_ids != subset_ids:
            raise ValueError("round manifest train subset identity mismatch")
        if manifest.context_identities != context_ids:
            raise ValueError("round manifest context identity mismatch")
        checkpoint = Path(manifest.checkpoint.path)
        deviations = Path(manifest.deviation_artifact.path)
        if (
            not checkpoint.is_file()
            or checkpoint.is_symlink()
            or _sha256_file(checkpoint) != manifest.checkpoint.sha256
        ):
            raise ValueError("round checkpoint identity mismatch")
        _load_deviations(deviations, manifest.deviation_artifact.sha256)
        if manifests and manifest.source_policy != manifests[-1].checkpoint:
            raise ValueError("round source policy chain is invalid")
        manifests.append(manifest)
    return tuple(manifests)


def _average_dev_metrics(
    contexts: Sequence[DAggerQuestionContext],
    *,
    policy: RouterPolicy,
    config: DAggerConfig,
) -> tuple[float, float]:
    utilities: list[float] = []
    regrets: list[float] = []
    for context in contexts:
        utility, regret = _evaluate_dev(
            (context.state,),
            policy=policy,
            environment=context.environment,
            utility_graph=context.snapshot,
            normalization=context.task9.normalization,
            beam_size=config.beam_size,
            max_depth=config.max_depth,
        )
        utilities.append(utility)
        regrets.append(regret)
    return sum(utilities) / len(utilities), sum(regrets) / len(regrets)


def run_dagger(
    *,
    train_contexts: Sequence[DAggerQuestionContext],
    dev_contexts: Sequence[DAggerQuestionContext],
    initial_policy: RouterPolicy,
    source_policy_checkpoint: Path,
    trainer: PolicyTrainer,
    config: DAggerConfig,
) -> DAggerRunResult:
    """Run or resume an identity-checked DAgger correction workflow."""

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
    source = source_policy_checkpoint.resolve()
    if not source.is_file() or source.is_symlink():
        raise ValueError("source policy checkpoint must be a regular file")
    source_hash = _sha256_file(source)
    selected = select_train_question_subset(
        train,
        fraction=config.train_question_subset_fraction,
        seed=config.train_question_subset_seed,
    )
    subset_ids = tuple(item.question_id for item in selected)
    subset_hash = _sha256_bytes(_canonical_json(subset_ids).encode("utf-8"))
    context_ids = tuple(item.identity for item in selected)
    run_identity = _run_identity(
        contexts=selected,
        dev_contexts=dev,
        config=config,
        source_policy_sha256=source_hash,
    )
    existing = _load_existing_manifests(
        config,
        run_identity=run_identity,
        base_dataset_identity=dataset.identity,
        subset_ids=subset_ids,
        context_ids=context_ids,
    )
    seen_path = config.resolve_path(config.seen_keys_path)
    deviations: tuple[Deviation, ...] = ()
    seen: set[str] = set()
    policy = initial_policy
    current_source = source
    current_source_hash = source_hash
    previous: DaggerRoundResult | None = None
    resumed = bool(existing)
    if existing:
        last = existing[-1]
        deviations = _load_deviations(
            Path(last.deviation_artifact.path), last.deviation_artifact.sha256
        )
        if not seen_path.is_file() or seen_path.is_symlink():
            raise ValueError("seen key artifact is missing on resume")
        seen_artifact = _SeenKeysArtifact.model_validate_json(
            seen_path.read_text(encoding="utf-8")
        )
        if (
            seen_artifact.run_identity != run_identity
            or seen_artifact.artifact_sha256 != last.seen_keys_sha256
            or len(seen_artifact.keys) != last.seen_key_count
        ):
            raise ValueError("seen key artifact identity mismatch")
        seen = set(seen_artifact.keys)
        previous = DaggerRoundResult(
            round_number=last.round_number,
            deviations=(),
            dev_utility=last.metrics.dev_utility,
            cost_regret=last.metrics.cost_regret,
            should_continue=last.status == "completed",
            seen_keys=seen_artifact.keys,
        )
        current_source = Path(last.checkpoint.path)
        current_source_hash = last.checkpoint.sha256
        if last.status == "stopped":
            return DAggerRunResult(
                run_identity=run_identity,
                status="stopped",
                resumed=True,
                final_checkpoint=current_source,
                manifests=existing,
            )
        policy = trainer.load_policy(
            checkpoint=current_source,
            base_dataset=dataset,
            deviations=deviations,
        )

    manifests = list(existing)
    start_round = len(existing) + 1
    for round_number in range(start_round, config.max_rounds + 1):
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
                    seen_keys=seen,
                    budget_bin_width=config.budget_bin_width,
                    beam_size=config.beam_size,
                    max_depth=config.max_depth,
                )
            )
        by_key = {item.state_key: item for item in deviations}
        for item in new:
            by_key.setdefault(item.state_key, item)
        deviations = tuple(by_key[key] for key in sorted(by_key))
        seen_artifact = _seen_artifact(run_identity, seen)
        _write_model(seen_path, seen_artifact)

        output_checkpoint = _checkpoint_path(config, round_number)
        trained = trainer.train(
            round_number=round_number,
            base_dataset=dataset,
            deviations=deviations,
            new_deviations=tuple(new),
            source_policy_checkpoint=current_source,
            output_checkpoint=output_checkpoint,
        )
        actual_checkpoint = trained.checkpoint_path.resolve()
        if actual_checkpoint != output_checkpoint.resolve():
            raise ValueError("trainer returned an unexpected checkpoint path")
        if (
            not actual_checkpoint.is_file()
            or actual_checkpoint.is_symlink()
            or _sha256_file(actual_checkpoint) != trained.checkpoint_sha256
        ):
            raise ValueError("trainer checkpoint identity mismatch")
        dev_utility, cost_regret = _average_dev_metrics(
            dev, policy=trained.policy, config=config
        )
        should_continue = _should_continue(
            round_number,
            dev_utility,
            cost_regret,
            previous,
            utility_gain_threshold=config.utility_gain_threshold,
            regret_improvement_ratio=config.regret_improvement_ratio,
        )
        if round_number >= config.max_rounds:
            should_continue = False
            stop_reason: Literal[
                "continue", "threshold_not_met", "max_rounds"
            ] = "max_rounds"
        elif not should_continue:
            stop_reason = "threshold_not_met"
        else:
            stop_reason = "continue"
        status: Literal["completed", "stopped"] = (
            "completed" if should_continue else "stopped"
        )
        deviation_path = _deviation_path(config, round_number)
        deviation_sha = _write_deviations(deviation_path, deviations)
        manifest = _build_manifest(
            run_identity=run_identity,
            round_number=round_number,
            status=status,
            stop_reason=stop_reason,
            source_policy=ArtifactReference(
                path=str(current_source), sha256=current_source_hash
            ),
            checkpoint=ArtifactReference(
                path=str(actual_checkpoint), sha256=trained.checkpoint_sha256
            ),
            base_dataset_identity=dataset.identity,
            aggregated_dataset_identity=trained.aggregated_dataset_identity,
            train_subset_question_ids=subset_ids,
            train_subset_sha256=subset_hash,
            context_identities=context_ids,
            seen_keys_sha256=seen_artifact.artifact_sha256,
            seen_key_count=len(seen),
            deviation_artifact=ArtifactReference(
                path=str(deviation_path), sha256=deviation_sha
            ),
            deviation_count=len(deviations),
            new_deviation_count=len(new),
            thresholds=RoundThresholds(
                utility_gain=config.utility_gain_threshold,
                regret_improvement_ratio=config.regret_improvement_ratio,
            ),
            metrics=RoundMetrics(dev_utility=dev_utility, cost_regret=cost_regret),
        )
        _write_model(_manifest_path(config, round_number), manifest)
        manifests.append(manifest)
        previous = DaggerRoundResult(
            round_number=round_number,
            deviations=(),
            dev_utility=dev_utility,
            cost_regret=cost_regret,
            should_continue=should_continue,
            seen_keys=tuple(sorted(seen)),
        )
        policy = trained.policy
        current_source = actual_checkpoint
        current_source_hash = trained.checkpoint_sha256
        if not should_continue:
            return DAggerRunResult(
                run_identity=run_identity,
                status="stopped",
                resumed=resumed,
                final_checkpoint=current_source,
                manifests=tuple(manifests),
            )
    raise RuntimeError("DAgger round loop ended without a stop decision")


__all__ = [
    "ArtifactReference",
    "DAggerConfig",
    "DAggerQuestionContext",
    "DAggerRoundManifest",
    "DAggerRunResult",
    "PolicyTrainer",
    "PolicyTrainingResult",
    "Task10DaggerProvenance",
    "Task8DaggerProvenance",
    "Task9DaggerProvenance",
    "run_dagger",
    "select_train_question_subset",
]
