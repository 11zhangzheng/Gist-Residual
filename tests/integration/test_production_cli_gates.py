from __future__ import annotations

import json
from pathlib import Path

import pytest

from fidmem.cli import main
from fidmem.production.authority import seal_authority
from tests.production.helpers import complete_draft, fake_gpu_runtime, fake_repository
from tests.production.test_observation_provenance import (
    _production_payload,
    _write_jsonl,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_invalid_production_authority_returns_machine_readable_block(
    tmp_path, capsys
) -> None:
    invalid = tmp_path / "invalid-authority.json"
    invalid.write_text("{}\n", encoding="utf-8")

    assert (
        main(
            [
                "report",
                "--config",
                str(PROJECT_ROOT / "configs/base.yaml"),
                "--artifact-root",
                str(tmp_path / "artifacts"),
                "--run-id",
                "canary",
                "--production-authority",
                str(invalid),
            ]
        )
        == 2
    )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "blocked"
    assert result["evidence_class"] == "production"


def test_report_records_successful_noop_resume_gate(tmp_path, capsys) -> None:
    authority_path = tmp_path / "authority.json"
    authority = seal_authority(
        complete_draft(tmp_path),
        output_path=authority_path,
        project_root=tmp_path,
        runtime_probe=fake_gpu_runtime,
        repository_probe=fake_repository,
    )
    source = tmp_path / "provider.jsonl"
    _write_jsonl(source, [_production_payload(authority)])
    common = [
        "--config",
        str(PROJECT_ROOT / "configs/base.yaml"),
        "--artifact-root",
        str(tmp_path / "artifacts"),
        "--run-id",
        "canary",
        "--production-authority",
        str(authority_path),
    ]

    assert main(["build-observations", *common, "--input-jsonl", str(source)]) == 0
    capsys.readouterr()
    assert (
        main(
            [
                "build-observations",
                *common,
                "--input-jsonl",
                str(source),
                "--resume",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert main(["report", *common]) == 0
    report = json.loads(capsys.readouterr().out)

    assert report["resume_validation"] is True


@pytest.mark.parametrize(
    "command_args",
    (
        ("build-observations",),
        ("build-observations", "--dry-run", "--input-jsonl", "unused.jsonl"),
        ("ingest", "--dry-run"),
        ("build-gist", "--dry-run"),
    ),
)
def test_engineering_paths_cannot_enter_production_namespace(
    tmp_path, capsys, command_args
) -> None:
    authority_path = tmp_path / "authority.json"
    seal_authority(
        complete_draft(tmp_path),
        output_path=authority_path,
        project_root=tmp_path,
        runtime_probe=fake_gpu_runtime,
        repository_probe=fake_repository,
    )
    artifact_root = tmp_path / "artifacts"

    assert (
        main(
            [
                *command_args,
                "--config",
                str(PROJECT_ROOT / "configs/base.yaml"),
                "--artifact-root",
                str(artifact_root),
                "--run-id",
                "pollution-regression",
                "--production-authority",
                str(authority_path),
            ]
        )
        == 2
    )

    blocked = json.loads(capsys.readouterr().out)
    assert blocked["status"] == "blocked"
    assert blocked["evidence_class"] == "production"
    assert not artifact_root.exists()
