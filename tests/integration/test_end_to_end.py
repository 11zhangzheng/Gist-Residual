from __future__ import annotations

import json
from pathlib import Path

from fidmem.cli import main


def test_cli_dry_run_and_report(tmp_path: Path, capsys) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "retrieval:\n  top_k: 5\noracle:\n  max_depth: 5\n  beam_size: 8\n"
        "visual:\n  low_frames: 12\n  high_frames: 32\nbudget:\n"
        "  a800_gpu_hours: 800\n  v100_gpu_hours: 200\n",
        encoding="utf-8",
    )
    assert main(["evaluate", "--config", str(config), "--artifact-root", str(tmp_path), "--run-id", "e2e", "--dry-run"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["dry_run"] is True
    assert main(["report", "--config", str(config), "--artifact-root", str(tmp_path), "--run-id", "e2e"]) == 0
    report_path = tmp_path / "development" / "runs" / "e2e" / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["run_id"] == "e2e"
    assert report["config_sha256"]
    assert report["execution_status"] == "completed"
    assert [entry["command"] for entry in report["command_history"]] == ["evaluate"]
