from __future__ import annotations

import json
from pathlib import Path

import pytest

from fidmem.actions.environment import ActionObservation, OperationMetadata
from fidmem.cli import main
from fidmem.costs.tracker import CostRecord
from fidmem.types import ActionInstance, ActionType, RouterState


def _config(path: Path) -> Path:
    path.write_text(
        "retrieval:\n  top_k: 5\noracle:\n  max_depth: 5\n  beam_size: 8\n"
        "visual:\n  low_frames: 12\n  high_frames: 32\nbudget:\n"
        "  a800_gpu_hours: 800\n  v100_gpu_hours: 200\n",
        encoding="utf-8",
    )
    return path


def _payload(*, with_cost: bool = True) -> dict[str, object]:
    state = RouterState(
        question="What color?",
        options=("blue", "red"),
        evidence=(),
        action_history=(),
        remaining_budget=20,
        candidate_event_ids=(),
        candidate_fidelity_levels={},
        context_frontiers={},
        cost_preference=0,
    )
    cost = CostRecord(
        operation="residual",
        gpu_seconds=0.25,
        wall_seconds=0.5,
        input_frames=12,
        visual_tokens=24,
        text_tokens=8,
        peak_memory_bytes=1024,
        cache_status="miss",
        device_name="cuda:0",
    )
    metadata = OperationMetadata(
        scope="residual",
        cache_status="miss",
        amortizable=True,
        cost_record=cost if with_cost else None,
    )
    return {
        "schema_version": 1,
        "question_id": "q1",
        "video_id": "v1",
        "provider_identity": {
            "provider": "local-fixture",
            "model_revision": "model-v1",
            "decode_config": {"temperature": 0},
            "device_name": "cuda:0",
        },
        "state": state.model_dump(mode="json"),
        "action": ActionInstance(ActionType.EXPAND_RESIDUAL, "e1", None).model_dump(
            mode="json"
        ),
        "observation": ActionObservation(
            action_type=ActionType.EXPAND_RESIDUAL,
            target_event_id="e1",
            operation_metadata=(metadata,),
        ).model_dump(mode="json"),
    }


def _write_input(path: Path, *, with_cost: bool = True) -> None:
    path.write_text(
        json.dumps(_payload(with_cost=with_cost), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def test_build_observations_imports_authoritative_jsonl(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _config(tmp_path / "config.yaml")
    input_path = tmp_path / "provider.jsonl"
    _write_input(input_path)
    common = [
        "build-observations",
        "--config",
        str(config),
        "--artifact-root",
        str(tmp_path),
        "--run-id",
        "provider-import",
        "--input-jsonl",
        str(input_path),
    ]

    assert main(common) == 0
    output = json.loads(capsys.readouterr().out)
    run_root = tmp_path / "development" / "runs" / "provider-import"
    assert output["mode"] == "provider_import"
    assert output["status"] == "completed"
    assert set(output["artifacts"]) == {
        "observations",
        "costs",
        "summary",
        "manifest",
    }
    assert all(
        (run_root / name).is_file()
        for name in ("observations.jsonl", "cost.csv", "summary.json", "manifest.json")
    )
    state = json.loads((run_root / "state.json").read_text(encoding="utf-8"))
    assert state["execution_status"] == "completed"
    assert state["last_command"]["record_count"] == 1
    assert (
        main(
            [
                "report",
                "--config",
                str(config),
                "--artifact-root",
                str(tmp_path),
                "--run-id",
                "provider-import",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["costs"]["cost_record_count"] == 1
    assert report["costs"]["total_gpu_seconds"] == 0.25


def test_build_observations_failure_preserves_complete_import(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _config(tmp_path / "config.yaml")
    input_path = tmp_path / "provider.jsonl"
    _write_input(input_path)
    common = [
        "build-observations",
        "--config",
        str(config),
        "--artifact-root",
        str(tmp_path),
        "--run-id",
        "provider-import",
        "--input-jsonl",
        str(input_path),
    ]
    assert main(common) == 0
    capsys.readouterr()
    run_root = tmp_path / "development" / "runs" / "provider-import"
    artifact_names = (
        "observations.jsonl",
        "cost.csv",
        "summary.json",
        "manifest.json",
    )
    before = {name: (run_root / name).read_bytes() for name in artifact_names}
    _write_input(input_path, with_cost=False)

    assert main(common + ["--resume"]) == 2
    blocked = json.loads(capsys.readouterr().out)
    assert blocked["mode"] == "provider_import"
    assert blocked["status"] == "blocked"
    assert "authoritative cost" in blocked["reason"]
    assert {name: (run_root / name).read_bytes() for name in artifact_names} == before
    state = json.loads((run_root / "state.json").read_text(encoding="utf-8"))
    assert state["execution_status"] == "blocked"
    assert len(state["command_history"]) == 2
