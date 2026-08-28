from pathlib import Path

from fidmem.experiments.execution_pack import (
    ExperimentRunner,
    LifecycleStatus,
    load_experiment_config,
)
from fidmem.production.authority import canonical_sha256


def test_successful_run_records_provenance_and_resume_is_noop(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("schema_version: 1\nexperiment_id: E00\n", encoding="utf-8")
    script = tmp_path / "run.sh"
    script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        """
schema_version: 1
protocol_version: v1
gates: {}
experiments:
  - id: E00
    purpose: test
    evidence_class: engineering
    dependencies: []
    required_gates: []
    produces_gates: []
    config_path: config.yaml
    script_path: run.sh
    phase: setup
    gpu_required: false
    resource_class: cpu
    resumable: true
""".strip()
        + "\n",
        encoding="utf-8",
    )
    calls = 0

    def executor(_command, *, env, stdout_path, stderr_path, **_kwargs):
        nonlocal calls
        calls += 1
        run_dir = Path(env["FIDMEM_RUN_DIR"])
        result = run_dir / "results" / "result.json"
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_text('{"engineering_test":true}\n', encoding="utf-8")
        stdout_path.write_text("engineering test\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return 0

    runner = ExperimentRunner(
        registry_path=registry,
        project_root=tmp_path,
        output_root=tmp_path / "runs",
        executor=executor,
    )
    preflight = {
        "experiment_id": "E00",
        "protocol_version": "v1",
        "config_path": str(config.resolve()),
        "config_sha256": canonical_sha256(load_experiment_config(config)),
        "authority_path": None,
        "authority_sha256": None,
        "source_identity": None,
        "selected_gpus": [],
        "selected_devices": [],
        "upstream_gates": {},
        "execution_command": ["engineering-test"],
        "required_outputs": ["results/result.json"],
    }
    run_dir = runner.execute_preflighted(preflight, run_id="run", resume=False)
    assert (run_dir / LifecycleStatus.COMPLETED.value).is_file()
    assert not (run_dir / LifecycleStatus.FAILED.value).exists()
    assert (run_dir / "metadata.json").is_file()
    assert (run_dir / "config.snapshot.json").is_file()
    assert (run_dir / "upstream-gates.snapshot.json").is_file()
    assert calls == 1
    assert runner.execute_preflighted(preflight, run_id="run", resume=True) == run_dir
    assert calls == 1
