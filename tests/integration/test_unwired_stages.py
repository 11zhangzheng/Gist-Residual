from __future__ import annotations

import json
from pathlib import Path

import pytest

from fidmem.cli import main


def _config(path: Path) -> Path:
    path.write_text(
        "retrieval:\n  top_k: 5\noracle:\n  max_depth: 5\n  beam_size: 8\n"
        "visual:\n  low_frames: 12\n  high_frames: 32\nbudget:\n"
        "  a800_gpu_hours: 800\n  v100_gpu_hours: 200\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    "stage",
    ("build-oracle", "train-router", "run-dagger", "evaluate"),
)
def test_unwired_stage_fails_closed_but_dry_run_succeeds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], stage: str
) -> None:
    config = _config(tmp_path / "config.yaml")
    common = [
        stage,
        "--config",
        str(config),
        "--artifact-root",
        str(tmp_path),
        "--run-id",
        stage,
    ]

    assert main(common) == 2
    blocked_output = json.loads(capsys.readouterr().out)
    assert blocked_output["blocked"] is True
    assert blocked_output["status"] == "blocked"
    assert "not wired to a real implementation" in blocked_output["reason"]

    state_path = tmp_path / "development" / "runs" / stage / "state.json"
    blocked_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert blocked_state["blocked"] is True
    assert blocked_state["status"] == "blocked"
    assert blocked_state["execution_status"] == "blocked"
    assert blocked_state["reason"] == blocked_output["reason"]
    assert blocked_state["command_history"][0]["blocked"] is True

    assert main(common + ["--dry-run"]) == 0
    dry_run_output = json.loads(capsys.readouterr().out)
    assert dry_run_output["dry_run"] is True

    final_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert len(final_state["command_history"]) == 2
    assert final_state["command_history"][0]["status"] == "blocked"
    assert final_state["command_history"][1]["dry_run"] is True
