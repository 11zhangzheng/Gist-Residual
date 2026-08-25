from __future__ import annotations

import json
from pathlib import Path

from fidmem.cli import main


def test_observation_resume_does_not_recharge_completed_items(tmp_path: Path, capsys) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "retrieval:\n  top_k: 5\noracle:\n  max_depth: 5\n  beam_size: 8\n"
        "visual:\n  low_frames: 12\n  high_frames: 32\nbudget:\n"
        "  a800_gpu_hours: 800\n  v100_gpu_hours: 200\n",
        encoding="utf-8",
    )
    common = ["build-observations", "--config", str(config), "--artifact-root", str(tmp_path), "--run-id", "resume", "--observations", "3"]
    main(common)
    capsys.readouterr()
    first = json.loads((tmp_path / "runs" / "resume" / "state.json").read_text(encoding="utf-8"))
    main(common + ["--resume"])
    capsys.readouterr()
    second = json.loads((tmp_path / "runs" / "resume" / "state.json").read_text(encoding="utf-8"))
    assert second["completed_observations"] == 3
    assert second.get("observation_cost", 0.0) == first.get("observation_cost", 0.0)
