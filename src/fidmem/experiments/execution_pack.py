"""Fail-closed orchestration primitives for the paper experiment execution pack.

This module never performs model inference itself. It validates frozen inputs,
upstream gates, GPU selection, and run identity before invoking an explicitly
configured command without a shell.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from omegaconf import OmegaConf

from fidmem.production.authority import (
    canonical_json_bytes,
    canonical_sha256,
    load_sealed_authority,
    probe_repository,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PLACEHOLDER_MARKERS = (
    "RESEARCH_OWNER_DECISION_REQUIRED",
    "REPLACE_WITH",
    "TEMPLATE",
    "NOT_PRODUCTION",
    "UNRESOLVED",
)


class CheckFailure(RuntimeError):
    """A preflight condition failed and execution must not start."""


class LifecycleStatus(str, Enum):
    PREPARED = "PREPARED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _atomic_json(path: Path, payload: object) -> None:
    _atomic_bytes(path, canonical_json_bytes(payload))


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_experiment_config(
    path: str | Path, *, _seen: frozenset[Path] = frozenset()
) -> dict[str, Any]:
    """Load a composed YAML config with deterministic recursive ``extends``."""

    source = Path(path).resolve()
    if source in _seen:
        raise ValueError(f"experiment config inheritance cycle at {source}")
    if not source.is_file():
        raise FileNotFoundError(source)
    raw = OmegaConf.to_container(OmegaConf.load(source), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("experiment config root must be a mapping")
    parent_value = raw.pop("extends", None)
    if parent_value is None:
        return dict(raw)
    parents = [parent_value] if isinstance(parent_value, str) else parent_value
    if not isinstance(parents, list) or not all(
        isinstance(item, str) for item in parents
    ):
        raise ValueError("extends must be a path or list of paths")
    merged: dict[str, Any] = {}
    for parent in parents:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = source.parent / parent_path
        merged = _deep_merge(
            merged,
            load_experiment_config(parent_path, _seen=_seen | {source}),
        )
    return _deep_merge(merged, raw)


@dataclass(frozen=True)
class ExperimentSpec:
    id: str
    purpose: str
    evidence_class: str
    dependencies: tuple[str, ...]
    required_gates: tuple[str, ...]
    produces_gates: tuple[str, ...]
    config_path: str
    script_path: str
    phase: str
    gpu_required: bool
    resource_class: str
    resumable: bool
    dataset_split: str = "RESEARCH_OWNER_DECISION_REQUIRED"
    model_roles: tuple[str, ...] = ()
    expected_inputs: tuple[str, ...] = ()
    expected_outputs: tuple[str, ...] = ()
    metrics: tuple[str, ...] = ()
    paper_role: str = "RESEARCH_OWNER_DECISION_REQUIRED"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExperimentSpec":
        required = {
            "id",
            "purpose",
            "evidence_class",
            "dependencies",
            "required_gates",
            "produces_gates",
            "config_path",
            "script_path",
            "phase",
            "gpu_required",
            "resource_class",
            "resumable",
        }
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(f"experiment record missing fields: {missing}")
        return cls(
            id=str(value["id"]),
            purpose=str(value["purpose"]),
            evidence_class=str(value["evidence_class"]),
            dependencies=tuple(str(item) for item in value["dependencies"]),
            required_gates=tuple(str(item) for item in value["required_gates"]),
            produces_gates=tuple(str(item) for item in value["produces_gates"]),
            config_path=str(value["config_path"]),
            script_path=str(value["script_path"]),
            phase=str(value["phase"]),
            gpu_required=bool(value["gpu_required"]),
            resource_class=str(value["resource_class"]),
            resumable=bool(value["resumable"]),
            dataset_split=str(
                value.get("dataset_split", "RESEARCH_OWNER_DECISION_REQUIRED")
            ),
            model_roles=tuple(str(item) for item in value.get("model_roles", ())),
            expected_inputs=tuple(
                str(item) for item in value.get("expected_inputs", ())
            ),
            expected_outputs=tuple(
                str(item) for item in value.get("expected_outputs", ())
            ),
            metrics=tuple(str(item) for item in value.get("metrics", ())),
            paper_role=str(value.get("paper_role", "RESEARCH_OWNER_DECISION_REQUIRED")),
        )


@dataclass(frozen=True)
class ExperimentRegistry:
    schema_version: int
    protocol_version: str
    gates: Mapping[str, Mapping[str, Any]]
    experiments: tuple[ExperimentSpec, ...]
    source_path: Path

    def experiment(self, experiment_id: str) -> ExperimentSpec:
        matches = [item for item in self.experiments if item.id == experiment_id]
        if len(matches) != 1:
            raise CheckFailure(f"unknown or duplicate experiment ID: {experiment_id}")
        return matches[0]


def load_registry(path: str | Path) -> ExperimentRegistry:
    source = Path(path).resolve()
    raw = OmegaConf.to_container(OmegaConf.load(source), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("registry root must be a mapping")
    experiments = raw.get("experiments")
    gates = raw.get("gates")
    if not isinstance(experiments, list) or not isinstance(gates, dict):
        raise ValueError("registry requires experiment list and gate mapping")
    return ExperimentRegistry(
        schema_version=int(raw.get("schema_version", 0)),
        protocol_version=str(raw.get("protocol_version", "")),
        gates={str(key): dict(value) for key, value in gates.items()},
        experiments=tuple(ExperimentSpec.from_mapping(item) for item in experiments),
        source_path=source,
    )


def validate_registry(
    registry: ExperimentRegistry, *, project_root: str | Path
) -> list[str]:
    root = Path(project_root).resolve()
    issues: list[str] = []
    if registry.schema_version != 1:
        issues.append("registry schema_version must be 1")
    if not registry.protocol_version:
        issues.append("registry protocol_version is required")
    ids = [item.id for item in registry.experiments]
    if len(ids) != len(set(ids)):
        issues.append("experiment IDs must be unique")
    id_set = set(ids)
    for item in registry.experiments:
        for dependency in item.dependencies:
            if dependency not in id_set:
                issues.append(f"{item.id}: unknown dependency {dependency}")
        for gate in (*item.required_gates, *item.produces_gates):
            if gate not in registry.gates:
                issues.append(f"{item.id}: unknown gate {gate}")
        config_path = root / item.config_path
        if not config_path.is_file():
            issues.append(f"{item.id}: missing config {item.config_path}")
        else:
            try:
                config = load_experiment_config(config_path)
                if config.get("experiment_id") != item.id:
                    issues.append(f"{item.id}: config experiment_id mismatch")
                if config.get("protocol_version") != registry.protocol_version:
                    issues.append(f"{item.id}: config protocol_version mismatch")
            except (OSError, ValueError) as exc:
                issues.append(f"{item.id}: invalid config: {exc}")
        script_path = root / item.script_path
        if not script_path.is_file():
            issues.append(f"{item.id}: missing script {item.script_path}")
        else:
            script_text = script_path.read_text(encoding="utf-8")
            if not script_text.startswith("#!/usr/bin/env bash"):
                issues.append(f"{item.id}: script lacks bash shebang")
            if item.id not in script_text:
                issues.append(f"{item.id}: script does not bind its registry ID")
    visiting: set[str] = set()
    visited: set[str] = set()
    by_id = {item.id: item for item in registry.experiments}
    for item in registry.experiments:
        for dependency in item.dependencies:
            if dependency not in by_id:
                continue
            produced = set(by_id[dependency].produces_gates)
            if not produced.intersection(item.required_gates):
                issues.append(
                    f"{item.id}: dependency {dependency} has no enforced gate edge"
                )

    def visit(experiment_id: str) -> None:
        if experiment_id in visiting:
            issues.append(f"dependency cycle includes {experiment_id}")
            return
        if experiment_id in visited or experiment_id not in by_id:
            return
        visiting.add(experiment_id)
        for dependency in by_id[experiment_id].dependencies:
            visit(dependency)
        visiting.remove(experiment_id)
        visited.add(experiment_id)

    for experiment_id in ids:
        visit(experiment_id)
    for gate_id, gate in registry.gates.items():
        producer = str(gate.get("producer", ""))
        if producer not in id_set:
            issues.append(f"gate {gate_id}: unknown producer {producer}")
        elif gate_id not in by_id[producer].produces_gates:
            issues.append(f"gate {gate_id}: producer does not declare the gate")
        threshold_path = root / "configs" / "experiments" / "gates" / f"{gate_id}.yaml"
        if not threshold_path.is_file():
            issues.append(f"gate {gate_id}: threshold file is missing")
        else:
            try:
                threshold_data = OmegaConf.to_container(
                    OmegaConf.load(threshold_path), resolve=True
                )
                if not isinstance(threshold_data, dict):
                    issues.append(f"gate {gate_id}: thresholds must be a mapping")
            except (OSError, ValueError) as exc:
                issues.append(f"gate {gate_id}: invalid thresholds: {exc}")
    return sorted(set(issues))


@dataclass(frozen=True)
class GateRecord:
    schema_version: int
    gate_id: str
    experiment_id: str
    run_id: str
    status: str
    protocol_version: str
    config_sha256: str
    result_sha256: str
    authority_sha256: str | None
    checks: Mapping[str, Any]
    thresholds: Mapping[str, Any]
    decided_at: str
    gate_sha256: str

    @classmethod
    def create(
        cls,
        *,
        gate_id: str,
        experiment_id: str,
        run_id: str,
        status: str,
        protocol_version: str,
        config_sha256: str,
        result_sha256: str,
        authority_sha256: str | None,
        checks: Mapping[str, Any],
        thresholds: Mapping[str, Any],
    ) -> "GateRecord":
        if status not in {"PASS", "FAIL"}:
            raise ValueError("gate status must be PASS or FAIL")
        for name, value in (
            ("config_sha256", config_sha256),
            ("result_sha256", result_sha256),
        ):
            if not _SHA256_RE.fullmatch(value):
                raise ValueError(f"{name} must be lowercase SHA-256")
        if authority_sha256 is not None and not _SHA256_RE.fullmatch(authority_sha256):
            raise ValueError("authority_sha256 must be lowercase SHA-256")
        payload = {
            "schema_version": 1,
            "gate_id": gate_id,
            "experiment_id": experiment_id,
            "run_id": run_id,
            "status": status,
            "protocol_version": protocol_version,
            "config_sha256": config_sha256,
            "result_sha256": result_sha256,
            "authority_sha256": authority_sha256,
            "checks": dict(checks),
            "thresholds": dict(thresholds),
            "decided_at": _now(),
        }
        return cls(**payload, gate_sha256=canonical_sha256(payload))

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "gate_id": self.gate_id,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "status": self.status,
            "protocol_version": self.protocol_version,
            "config_sha256": self.config_sha256,
            "result_sha256": self.result_sha256,
            "authority_sha256": self.authority_sha256,
            "checks": dict(self.checks),
            "thresholds": dict(self.thresholds),
            "decided_at": self.decided_at,
        }

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        payload = {**self.payload(), "gate_sha256": self.gate_sha256}
        if destination.exists():
            existing = json.loads(destination.read_text(encoding="utf-8"))
            if existing != payload:
                raise ValueError("gate artifact already exists with different identity")
            return
        _atomic_json(destination, payload)

    @classmethod
    def load(cls, path: str | Path) -> "GateRecord":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        record = cls(**raw)
        if canonical_sha256(record.payload()) != record.gate_sha256:
            raise CheckFailure(f"gate artifact hash mismatch: {record.gate_id}")
        return record


def parse_gpu_selection(value: str | None) -> tuple[int, ...]:
    if value is None or value == "":
        return ()
    parts = value.split(",")
    if any(not part.isdigit() for part in parts):
        raise ValueError("GPU selection must be comma-separated non-negative indices")
    indices = tuple(int(part) for part in parts)
    if len(indices) != len(set(indices)):
        raise ValueError("GPU selection must not contain duplicates")
    return indices


def probe_gpus() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.free,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except OSError:
        return []
    if completed.returncode != 0:
        return []
    devices: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 6:
            continue
        devices.append(
            {
                "index": int(fields[0]),
                "uuid": fields[1],
                "name": fields[2],
                "free_vram_mb": int(fields[3]),
                "total_vram_mb": int(fields[4]),
                "utilization_percent": int(fields[5]),
            }
        )
    return devices


def _is_placeholder(value: object) -> bool:
    return isinstance(value, str) and any(
        marker in value.upper() for marker in _PLACEHOLDER_MARKERS
    )


def _nested_value(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise CheckFailure(f"required config field is missing: {path}")
        current = current[part]
    return current


def resolve_environment_references(value: Any) -> Any:
    """Resolve exact ``env:NAME`` leaves before hashing a run config."""

    if isinstance(value, str) and value.startswith("env:"):
        name = value.removeprefix("env:")
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
            raise CheckFailure(f"invalid environment reference: {value}")
        resolved = os.environ.get(name)
        if not resolved:
            raise CheckFailure(f"required environment variable is missing: {name}")
        return resolved
    if isinstance(value, Mapping):
        return {
            key: resolve_environment_references(item) for key, item in value.items()
        }
    if isinstance(value, list):
        return [resolve_environment_references(item) for item in value]
    return value


def _default_executor(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> int:
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            env=dict(env),
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    return completed.returncode


class ExperimentRunner:
    def __init__(
        self,
        *,
        registry_path: str | Path,
        project_root: str | Path,
        gate_root: str | Path | None = None,
        output_root: str | Path | None = None,
        executor: Callable[..., int] = _default_executor,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.registry = load_registry(registry_path)
        self.gate_root = Path(
            gate_root or self.project_root / "artifacts" / "experiment-gates"
        ).resolve()
        self.output_root = Path(
            output_root or self.project_root / "artifacts" / "experiments"
        ).resolve()
        self.executor = executor

    def _authority(
        self,
        config: Mapping[str, Any],
        loader: Callable[[str | Path], Any],
    ) -> tuple[str | None, str | None]:
        value = config.get("production_authority")
        if value in (None, ""):
            return None, None
        if _is_placeholder(value):
            raise CheckFailure("production identity contains a placeholder")
        path = Path(str(value))
        if not path.is_absolute():
            path = self.project_root / path
        if loader is load_sealed_authority and not path.is_file():
            raise CheckFailure(f"Production Authority is missing: {path}")
        try:
            authority = loader(path)
        except (OSError, ValueError) as exc:
            raise CheckFailure(f"Production Authority is invalid: {exc}") from exc
        if isinstance(authority, Mapping):
            digest = authority.get("authority_sha256")
        else:
            digest = getattr(authority, "authority_sha256", None)
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise CheckFailure("Production Authority semantic identity is invalid")
        authority_repository = (
            authority.get("repository")
            if isinstance(authority, Mapping)
            else getattr(authority, "repository", None)
        )
        if authority_repository is not None and (self.project_root / ".git").exists():
            observed_repository = probe_repository(self.project_root)
            expected_repository = (
                authority_repository
                if isinstance(authority_repository, Mapping)
                else authority_repository.model_dump(mode="json")
            )
            if observed_repository.model_dump(mode="json") != dict(expected_repository):
                raise CheckFailure(
                    "Production Authority source identity differs from current repository"
                )
        return str(path.resolve()), digest

    def _required_gates(self, spec: ExperimentSpec) -> dict[str, str]:
        observed: dict[str, str] = {}
        for gate_id in spec.required_gates:
            path = self.gate_root / f"{gate_id}.json"
            if not path.is_file():
                raise CheckFailure(f"required gate {gate_id} is missing: FAIL_CLOSED")
            gate = GateRecord.load(path)
            if gate.protocol_version != self.registry.protocol_version:
                raise CheckFailure(f"gate {gate_id} protocol version mismatch")
            if gate.status != "PASS":
                raise CheckFailure(f"required gate {gate_id} has status {gate.status}")
            if gate.gate_id != gate_id:
                raise CheckFailure(f"gate identity mismatch for {gate_id}")
            observed[gate_id] = gate.gate_sha256
        return observed

    def check(
        self,
        experiment_id: str,
        *,
        config_path: str | Path | None = None,
        gpus: str | None = None,
        run_id: str | None = None,
        resume: bool = False,
        gpu_probe: Callable[[], list[dict[str, Any]]] = probe_gpus,
        authority_loader: Callable[[str | Path], Any] = load_sealed_authority,
    ) -> dict[str, Any]:
        issues = validate_registry(self.registry, project_root=self.project_root)
        if issues:
            raise CheckFailure("registry is inconsistent: " + "; ".join(issues))
        spec = self.registry.experiment(experiment_id)
        config_source = Path(config_path or self.project_root / spec.config_path)
        if not config_source.is_absolute():
            config_source = self.project_root / config_source
        config = resolve_environment_references(load_experiment_config(config_source))
        if config.get("experiment_id") != experiment_id:
            raise CheckFailure("config experiment_id does not match registry")
        execution = config.get("execution", {})
        if not isinstance(execution, Mapping):
            raise CheckFailure("execution config must be a mapping")
        if spec.phase == "router_training" and bool(
            execution.get("may_generate_observations", False)
        ):
            raise CheckFailure("Router training cannot regenerate observations")
        upstream_gates = self._required_gates(spec)
        authority_path, authority_sha256 = self._authority(config, authority_loader)
        if spec.evidence_class in {"production", "paper"} and authority_sha256 is None:
            raise CheckFailure("Production Authority is required")
        for field in config.get("required_fields", []):
            value = _nested_value(config, str(field))
            if value in (None, "") or _is_placeholder(value):
                raise CheckFailure(
                    f"required field is unresolved or placeholder: {field}"
                )
        selected_gpus = parse_gpu_selection(gpus)
        if spec.gpu_required and not selected_gpus:
            raise CheckFailure("explicit --gpus selection is required")
        resources = config.get("resources", {})
        if not isinstance(resources, Mapping):
            raise CheckFailure("resources config must be a mapping")
        devices = gpu_probe() if spec.gpu_required else []
        by_index = {int(item["index"]): item for item in devices}
        minimum_vram = int(resources.get("min_free_vram_mb", 0))
        selected_devices: list[dict[str, Any]] = []
        for index in selected_gpus:
            device = by_index.get(index)
            if device is None:
                raise CheckFailure(f"GPU {index} is not visible")
            if int(device.get("free_vram_mb", 0)) < minimum_vram:
                raise CheckFailure(
                    f"GPU {index} has insufficient free VRAM: "
                    f"{device.get('free_vram_mb')} < {minimum_vram} MiB"
                )
            selected_devices.append(dict(device))
        minimum_disk = float(resources.get("min_free_disk_gb", 0))
        disk_target = self.output_root
        existing_target = disk_target
        while (
            not existing_target.exists() and existing_target != existing_target.parent
        ):
            existing_target = existing_target.parent
        free_gb = shutil.disk_usage(existing_target).free / (1024**3)
        if free_gb < minimum_disk:
            raise CheckFailure(
                f"insufficient disk space: {free_gb:.2f} < {minimum_disk:.2f} GiB"
            )
        for variable in config.get("required_environment", []):
            if not os.environ.get(str(variable)):
                raise CheckFailure(
                    f"required environment variable is missing: {variable}"
                )
        for raw_path in config.get("required_paths", []):
            path = Path(str(raw_path))
            if _is_placeholder(str(path)):
                raise CheckFailure(f"required path is a placeholder: {path}")
            if not path.is_absolute():
                path = self.project_root / path
            if not path.exists():
                raise CheckFailure(f"required path is missing: {path}")
        if spec.phase == "router_training":
            cache_value = _nested_value(config, "inputs.observation_cache")
            if _is_placeholder(cache_value):
                raise CheckFailure(
                    "Router requires frozen observations; placeholder found"
                )
            cache_path = Path(str(cache_value))
            if not cache_path.is_absolute():
                cache_path = self.project_root / cache_path
            if not cache_path.exists():
                raise CheckFailure(
                    "Router requires frozen observations; cache is missing"
                )
        command = execution.get("command")
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(item, str) and item for item in command)
        ):
            raise CheckFailure("execution.command must be a non-empty argument list")
        if any(_is_placeholder(item) for item in command):
            raise CheckFailure("execution command contains an unresolved placeholder")
        source = config.get("source", {})
        observed_repository = (
            probe_repository(self.project_root)
            if (self.project_root / ".git").exists()
            else None
        )
        if (
            isinstance(source, Mapping)
            and bool(source.get("require_clean", False))
            and observed_repository is not None
            and observed_repository.dirty_worktree
        ):
            raise CheckFailure("Git worktree is dirty")
        requested_commit = (
            source.get("git_commit") if isinstance(source, Mapping) else None
        )
        if (
            requested_commit
            and not _is_placeholder(requested_commit)
            and observed_repository is not None
            and observed_repository.git_commit != requested_commit
        ):
            raise CheckFailure("Git commit differs from frozen config")
        effective_run_id = run_id or f"{experiment_id}-{_now().replace(':', '')}"
        if not _RUN_ID_RE.fullmatch(effective_run_id):
            raise CheckFailure("run ID contains unsupported characters")
        candidate_run = self.output_root / experiment_id / effective_run_id
        if resume and not candidate_run.is_dir():
            raise CheckFailure("--resume requires an existing run namespace")
        if not resume and candidate_run.exists():
            raise CheckFailure(
                "run namespace already exists; use --resume or a new run ID"
            )
        config_sha256 = canonical_sha256(config)
        return {
            "status": "CHECK_PASSED",
            "experiment_id": experiment_id,
            "protocol_version": self.registry.protocol_version,
            "config_path": str(config_source.resolve()),
            "config_sha256": config_sha256,
            "authority_path": authority_path,
            "authority_sha256": authority_sha256,
            "source_identity": (
                observed_repository.model_dump(mode="json")
                if observed_repository is not None
                else None
            ),
            "selected_gpus": list(selected_gpus),
            "selected_devices": selected_devices,
            "free_disk_gb": round(free_gb, 3),
            "upstream_gates": upstream_gates,
            "execution_command": list(command),
            "required_outputs": list(config.get("outputs", {}).get("required", [])),
            "run_id": effective_run_id,
        }

    def _status(self, run_dir: Path, status: LifecycleStatus, **extra: Any) -> None:
        payload = {
            "schema_version": 1,
            "status": status.value,
            "updated_at": _now(),
            **extra,
        }
        _atomic_json(run_dir / "STATUS.json", payload)
        _atomic_bytes(run_dir / status.value, b"")

    def execute_preflighted(
        self,
        preflight: Mapping[str, Any],
        *,
        run_id: str,
        resume: bool,
    ) -> Path:
        if not _RUN_ID_RE.fullmatch(run_id):
            raise CheckFailure("run ID contains unsupported characters")
        experiment_id = str(preflight["experiment_id"])
        run_dir = self.output_root / experiment_id / run_id
        metadata_path = run_dir / "metadata.json"
        identity = {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "run_id": run_id,
            "protocol_version": preflight.get("protocol_version"),
            "config_path": str(preflight.get("config_path")),
            "config_sha256": preflight["config_sha256"],
            "authority_path": preflight.get("authority_path"),
            "authority_sha256": preflight.get("authority_sha256"),
            "source_identity": preflight.get("source_identity"),
            "execution_command": list(preflight["execution_command"]),
            "selected_gpus": list(preflight.get("selected_gpus", [])),
            "selected_devices": list(preflight.get("selected_devices", [])),
            "upstream_gates": dict(preflight.get("upstream_gates", {})),
        }
        if run_dir.exists():
            if not resume:
                raise CheckFailure("run directory exists; use --resume")
            if not metadata_path.is_file():
                raise CheckFailure("resume requires immutable metadata.json")
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            for field in (
                "experiment_id",
                "run_id",
                "protocol_version",
                "config_path",
                "config_sha256",
                "authority_path",
                "authority_sha256",
                "source_identity",
                "execution_command",
                "selected_gpus",
                "selected_devices",
                "upstream_gates",
            ):
                if existing.get(field) != identity.get(field):
                    label = "config identity" if field == "config_sha256" else field
                    raise CheckFailure(f"resume {label} mismatch")
            status = json.loads((run_dir / "STATUS.json").read_text(encoding="utf-8"))
            if status.get("status") == LifecycleStatus.COMPLETED.value:
                return run_dir
        else:
            if resume:
                raise CheckFailure("--resume requires an existing run namespace")
            run_dir.mkdir(parents=True, exist_ok=False)
            identity["prepared_at"] = _now()
            identity["exact_invocation"] = list(os.sys.argv)
            _atomic_json(metadata_path, identity)
            config = load_experiment_config(str(preflight["config_path"]))
            snapshot = canonical_json_bytes(config)
            if hashlib.sha256(snapshot).hexdigest() != preflight["config_sha256"]:
                raise CheckFailure("config snapshot differs from preflight identity")
            _atomic_bytes(run_dir / "config.snapshot.json", snapshot)
            _atomic_json(
                run_dir / "upstream-gates.snapshot.json", identity["upstream_gates"]
            )
            self._status(run_dir, LifecycleStatus.PREPARED)
        self._status(run_dir, LifecycleStatus.RUNNING, started_at=_now())
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(item) for item in identity["selected_gpus"]
        )
        env["FIDMEM_RUN_DIR"] = str(run_dir.resolve())
        env["FIDMEM_EXPERIMENT_ID"] = experiment_id
        env["FIDMEM_CONFIG_SNAPSHOT"] = str(
            (run_dir / "config.snapshot.json").resolve()
        )
        try:
            return_code = self.executor(
                identity["execution_command"],
                cwd=self.project_root,
                env=env,
                stdout_path=run_dir / "stdout.log",
                stderr_path=run_dir / "stderr.log",
            )
            if return_code != 0:
                self._status(
                    run_dir,
                    LifecycleStatus.FAILED,
                    ended_at=_now(),
                    return_code=int(return_code),
                )
                return run_dir
            missing = [
                item
                for item in preflight.get("required_outputs", [])
                if not (run_dir / str(item)).is_file()
            ]
            if missing:
                self._status(
                    run_dir,
                    LifecycleStatus.FAILED,
                    ended_at=_now(),
                    return_code=0,
                    reason=f"required outputs missing: {missing}",
                )
                return run_dir
            self._status(
                run_dir,
                LifecycleStatus.COMPLETED,
                ended_at=_now(),
                return_code=0,
            )
            return run_dir
        except BaseException as exc:
            self._status(
                run_dir,
                LifecycleStatus.FAILED,
                ended_at=_now(),
                reason=f"{type(exc).__name__}: {exc}",
            )
            raise

    def run(
        self,
        experiment_id: str,
        *,
        config_path: str | Path | None,
        gpus: str | None,
        run_id: str | None,
        resume: bool,
    ) -> Path:
        preflight = self.check(
            experiment_id,
            config_path=config_path,
            gpus=gpus,
            run_id=run_id,
            resume=resume,
        )
        return self.execute_preflighted(
            preflight,
            run_id=str(preflight["run_id"]),
            resume=resume,
        )
