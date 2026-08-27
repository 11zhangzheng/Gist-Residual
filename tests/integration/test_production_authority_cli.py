from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf

from fidmem.cli import main
from fidmem.production.authority import seal_authority
from tests.production.helpers import complete_draft, fake_gpu_runtime, fake_repository
from fidmem.production.generation import GenerationStore
from tests.production.test_observation_provenance import (
    _production_payload,
    _write_jsonl,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _sealed_authority(tmp_path: Path):
    path = tmp_path / "authority.json"
    sealed = seal_authority(
        complete_draft(tmp_path),
        output_path=path,
        project_root=tmp_path,
        runtime_probe=fake_gpu_runtime,
        repository_probe=fake_repository,
    )
    return path, sealed


def test_authority_validate_template_fails_closed_with_issue_codes(
    capsys,
) -> None:
    template = PROJECT_ROOT / "configs/production/authority.example.yaml"

    assert main(["authority-validate", "--draft", str(template)]) == 2
    result = json.loads(capsys.readouterr().out)

    assert result["production_ready"] is False
    assert "required_section_missing" in result["error_codes"]
    assert result["evidence_class"] == "engineering"


def test_authority_seal_failure_writes_no_output(tmp_path: Path, capsys) -> None:
    output = tmp_path / "PRODUCTION_AUTHORITY.json"

    assert (
        main(
            [
                "authority-seal",
                "--draft",
                str(PROJECT_ROOT / "configs/production/authority.example.yaml"),
                "--output",
                str(output),
            ]
        )
        == 2
    )

    assert not output.exists()
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"


def test_production_import_and_report_use_authority_namespace(
    tmp_path: Path, capsys
) -> None:
    authority_path, authority = _sealed_authority(tmp_path)
    source = tmp_path / "provider.jsonl"
    _write_jsonl(source, [_production_payload(authority)])
    artifact_root = tmp_path / "artifacts"
    common = [
        "--config",
        str(PROJECT_ROOT / "configs/base.yaml"),
        "--artifact-root",
        str(artifact_root),
        "--run-id",
        "canary",
        "--production-authority",
        str(authority_path),
    ]

    assert main(["build-observations", *common, "--input-jsonl", str(source)]) == 0
    imported = json.loads(capsys.readouterr().out)
    run_root = (
        artifact_root / "production" / authority.authority_sha256 / "runs" / "canary"
    )
    assert imported["evidence_class"] == "production"
    assert imported["authority_sha256"] == authority.authority_sha256
    assert (run_root / "CURRENT.json").is_file()
    store = GenerationStore(run_root, authority.authority_sha256)
    active = store.current_path()
    state = json.loads((active / "state.json").read_text(encoding="utf-8"))
    assert state["authority_sha256"] == authority.authority_sha256
    assert {entry["authority_sha256"] for entry in state["command_history"]} == {
        authority.authority_sha256
    }

    assert main(["report", *common]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["production_provenance_valid"] is True
    assert report["cost_reconciliation_passed"] is True
    assert report["accuracy_primary_gate"] is False

    active = store.current_path()
    stored_report = json.loads((active / "report.json").read_text(encoding="utf-8"))
    assert stored_report["authority_sha256"] == authority.authority_sha256


def test_frozen_oracle_pilot_values_are_unchanged() -> None:
    config = OmegaConf.to_container(
        OmegaConf.load(PROJECT_ROOT / "configs/experiment/oracle_pilot.yaml"),
        resolve=True,
    )["oracle_pilot"]

    assert (
        config["question_count"],
        config["beam_size"],
        config["max_depth"],
        config["exhaustive_subset_size"],
        config["stability_state_count"],
        config["stability_repeats"],
        config["flip_rate_threshold"],
    ) == (100, 8, 5, 20, 100, 3, 0.02)
