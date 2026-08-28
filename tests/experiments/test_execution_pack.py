from __future__ import annotations

import json
from pathlib import Path

import pytest

from fidmem.experiments.execution_pack import (
    CheckFailure,
    ExperimentRunner,
    GateRecord,
    LifecycleStatus,
    load_experiment_config,
    load_registry,
    parse_gpu_selection,
    validate_registry,
)
from fidmem.production.authority import canonical_sha256


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _registry(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "registry.yaml",
        """
schema_version: 1
protocol_version: test-v1
gates:
  canary: {producer: E03}
  oracle: {producer: E04}
experiments:
  - id: E03
    purpose: Canary
    evidence_class: production
    dependencies: []
    required_gates: []
    produces_gates: [canary]
    config_path: configs/e03.yaml
    script_path: scripts/03.sh
    phase: observations
    gpu_required: true
    resource_class: unknown_until_canary
    resumable: true
  - id: E04
    purpose: Oracle
    evidence_class: production
    dependencies: [E03]
    required_gates: [canary]
    produces_gates: [oracle]
    config_path: configs/e04.yaml
    script_path: scripts/04.sh
    phase: oracle
    gpu_required: true
    resource_class: unknown_until_canary
    resumable: true
  - id: E10
    purpose: Router
    evidence_class: paper
    dependencies: [E04]
    required_gates: [oracle]
    produces_gates: []
    config_path: configs/e10.yaml
    script_path: scripts/10.sh
    phase: router_training
    gpu_required: true
    resource_class: unknown_until_oracle
    resumable: true
""".strip()
        + "\n",
    )


def _config(tmp_path: Path, experiment_id: str, *, authority: str = "missing") -> Path:
    return _write(
        tmp_path / "configs" / f"{experiment_id.lower()}.yaml",
        f"""
schema_version: 1
protocol_version: test-v1
experiment_id: {experiment_id}
production_authority: {authority}
source:
  require_clean: false
resources:
  min_free_disk_gb: 0
  min_free_vram_mb: 1
inputs:
  observation_cache: RESEARCH_OWNER_DECISION_REQUIRED
execution:
  command: [python, -c, "print('should not run during check')"]
  may_generate_observations: false
outputs:
  required: [results/result.json]
""".strip()
        + "\n",
    )


def _script_files(tmp_path: Path) -> None:
    for name, experiment_id in (("03", "E03"), ("04", "E04"), ("10", "E10")):
        _write(
            tmp_path / "scripts" / f"{name}.sh",
            f"#!/usr/bin/env bash\n# {experiment_id}\n",
        )
    for experiment_id in ("E03", "E04", "E10"):
        path = tmp_path / "configs" / f"{experiment_id.lower()}.yaml"
        if not path.exists():
            _config(tmp_path, experiment_id)
    for gate_id in ("canary", "oracle"):
        _write(
            tmp_path / "configs" / "experiments" / "gates" / f"{gate_id}.yaml",
            "{}\n",
        )


def _gate(root: Path, gate_id: str, experiment_id: str, status: str) -> Path:
    record = GateRecord.create(
        gate_id=gate_id,
        experiment_id=experiment_id,
        run_id=f"{experiment_id}-run",
        status=status,
        protocol_version="test-v1",
        config_sha256="a" * 64,
        result_sha256="b" * 64,
        authority_sha256="c" * 64,
        checks={"engineering_test": True},
        thresholds={"frozen": True},
    )
    path = root / f"{gate_id}.json"
    record.write(path)
    return path


def test_registry_dependency_and_gate_consistency(tmp_path: Path) -> None:
    registry_path = _registry(tmp_path)
    _script_files(tmp_path)
    for experiment_id in ("E03", "E04", "E10"):
        _config(tmp_path, experiment_id)
    registry = load_registry(registry_path)
    assert validate_registry(registry, project_root=tmp_path) == []


def test_failed_canary_blocks_oracle(tmp_path: Path) -> None:
    registry_path = _registry(tmp_path)
    _script_files(tmp_path)
    config = _config(tmp_path, "E04", authority="authority.json")
    _write(tmp_path / "authority.json", "{}\n")
    _gate(tmp_path / "gates", "canary", "E03", "FAIL")
    runner = ExperimentRunner(
        registry_path=registry_path,
        project_root=tmp_path,
        gate_root=tmp_path / "gates",
    )
    with pytest.raises(CheckFailure, match="canary.*FAIL"):
        runner.check("E04", config_path=config, gpus="0", gpu_probe=lambda: [])


def test_failed_oracle_blocks_router_before_cache_check(tmp_path: Path) -> None:
    registry_path = _registry(tmp_path)
    _script_files(tmp_path)
    config = _config(tmp_path, "E10", authority="authority.json")
    _gate(tmp_path / "gates", "oracle", "E04", "FAIL")
    runner = ExperimentRunner(
        registry_path=registry_path,
        project_root=tmp_path,
        gate_root=tmp_path / "gates",
    )
    with pytest.raises(CheckFailure, match="oracle.*FAIL"):
        runner.check("E10", config_path=config, gpus="0", gpu_probe=lambda: [])


def test_missing_authority_fails_closed(tmp_path: Path) -> None:
    registry_path = _registry(tmp_path)
    _script_files(tmp_path)
    config = _config(tmp_path, "E03")
    runner = ExperimentRunner(registry_path=registry_path, project_root=tmp_path)
    with pytest.raises(CheckFailure, match="Production Authority"):
        runner.check("E03", config_path=config, gpus="0", gpu_probe=lambda: [])


def test_router_cannot_regenerate_observations(tmp_path: Path) -> None:
    registry_path = _registry(tmp_path)
    _script_files(tmp_path)
    config = _config(tmp_path, "E10", authority="authority.json")
    text = config.read_text(encoding="utf-8").replace(
        "may_generate_observations: false", "may_generate_observations: true"
    )
    config.write_text(text, encoding="utf-8")
    runner = ExperimentRunner(registry_path=registry_path, project_root=tmp_path)
    with pytest.raises(CheckFailure, match="Router.*observations"):
        runner.check("E10", config_path=config, gpus="0", gpu_probe=lambda: [])


def test_check_never_calls_executor(tmp_path: Path) -> None:
    called = False

    def executor(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("executor must not run")

    registry_path = _registry(tmp_path)
    _script_files(tmp_path)
    config = _config(tmp_path, "E03")
    runner = ExperimentRunner(
        registry_path=registry_path,
        project_root=tmp_path,
        executor=executor,
    )
    with pytest.raises(CheckFailure):
        runner.check("E03", config_path=config, gpus="0", gpu_probe=lambda: [])
    assert called is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [("0", (0,)), ("0,2", (0, 2)), ("2,0", (2, 0)), ("", ())],
)
def test_gpu_selection_parsing(value: str, expected: tuple[int, ...]) -> None:
    assert parse_gpu_selection(value) == expected


@pytest.mark.parametrize("value", ["-1", "0,0", "gpu0", "0,"])
def test_invalid_gpu_selection_rejected(value: str) -> None:
    with pytest.raises(ValueError):
        parse_gpu_selection(value)


def test_missing_gpu_fails_without_execution(tmp_path: Path) -> None:
    registry_path = _registry(tmp_path)
    _script_files(tmp_path)
    config = _config(tmp_path, "E03", authority="authority.json")
    runner = ExperimentRunner(registry_path=registry_path, project_root=tmp_path)
    with pytest.raises(CheckFailure, match="GPU 0"):
        runner.check(
            "E03",
            config_path=config,
            gpus="0",
            gpu_probe=lambda: [],
            authority_loader=lambda _path: {"authority_sha256": "c" * 64},
        )


def test_resume_identity_and_failed_lifecycle(tmp_path: Path) -> None:
    registry_path = _registry(tmp_path)
    _script_files(tmp_path)
    config = _config(tmp_path, "E03", authority="authority.json")
    output_root = tmp_path / "runs"
    runner = ExperimentRunner(
        registry_path=registry_path,
        project_root=tmp_path,
        output_root=output_root,
        executor=lambda *_args, **_kwargs: 9,
    )
    preflight = {
        "experiment_id": "E03",
        "config_path": str(config.resolve()),
        "config_sha256": canonical_sha256(load_experiment_config(config)),
        "authority_sha256": "c" * 64,
        "selected_gpus": [0],
        "upstream_gates": {},
        "execution_command": ["false"],
    }
    run_dir = runner.execute_preflighted(preflight, run_id="same", resume=False)
    status = json.loads((run_dir / "STATUS.json").read_text(encoding="utf-8"))
    assert status["status"] == LifecycleStatus.FAILED.value
    assert not (run_dir / "COMPLETED").exists()
    with pytest.raises(CheckFailure, match="config identity"):
        runner.execute_preflighted(
            {**preflight, "config_sha256": "e" * 64},
            run_id="same",
            resume=True,
        )


def test_config_snapshot_is_byte_consistent(tmp_path: Path) -> None:
    base = _write(
        tmp_path / "base.yaml",
        "schema_version: 1\nresources: {min_free_disk_gb: 0}\n",
    )
    child = _write(
        tmp_path / "child.yaml",
        f"extends: {base.name}\nexperiment_id: E00\n",
    )
    loaded = load_experiment_config(child)
    assert loaded["schema_version"] == 1
    assert loaded["experiment_id"] == "E00"
    assert "extends" not in loaded


def test_placeholder_production_identity_is_rejected(tmp_path: Path) -> None:
    registry_path = _registry(tmp_path)
    _script_files(tmp_path)
    config = _config(tmp_path, "E03", authority="RESEARCH_OWNER_DECISION_REQUIRED")
    runner = ExperimentRunner(registry_path=registry_path, project_root=tmp_path)
    with pytest.raises(CheckFailure, match="placeholder"):
        runner.check("E03", config_path=config, gpus="0", gpu_probe=lambda: [])
