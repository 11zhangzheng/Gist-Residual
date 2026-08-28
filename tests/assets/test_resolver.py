"""Engineering-evidence-only tests for the asset lock lifecycle."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from fidmem.assets.resolver import (
    AssetLock,
    AssetLockEntry,
    AssetState,
    assert_verified_lock,
    initial_lock,
    resolve_entry,
    snapshot_sha256,
    verify_entry,
)
from fidmem.assets.stack import load_experiment_stack


ROOT = Path(__file__).resolve().parents[2]


def test_initial_lock_preserves_unresolved_asset_and_reuse() -> None:
    lock = initial_lock(
        load_experiment_stack(ROOT / "configs/experiment_stacks/gist_residual_v1.yaml")
    )
    assert lock.physical_assets["qwen3_vl_8b_instruct"].state is AssetState.UNRESOLVED
    assert lock.logical_roles["residual_model"] == lock.logical_roles["visual_model"]
    with pytest.raises(ValueError, match="not VERIFIED"):
        assert_verified_lock(lock)


def test_lock_tampering_is_detected() -> None:
    lock = initial_lock(
        load_experiment_stack(ROOT / "configs/experiment_stacks/gist_residual_v1.yaml")
    )
    payload = lock.model_dump(mode="json")
    payload["logical_roles"]["answerer"] = "bge_m3"
    with pytest.raises(ValidationError, match="hash mismatch"):
        AssetLock.model_validate(payload)


def test_resolution_requires_full_commit() -> None:
    entry = AssetLockEntry(
        repo_id="owner/repo", repo_type="model", state="UNRESOLVED", backend="test"
    )
    with pytest.raises(ValueError, match="full immutable"):
        resolve_entry(
            entry, info_loader=lambda _repo, _type: ("main", ("config.json",))
        )


def test_incomplete_snapshot_is_rejected(tmp_path: Path) -> None:
    entry = AssetLockEntry(
        repo_id="owner/repo",
        repo_type="model",
        immutable_revision="a" * 40,
        state="DOWNLOADED",
        backend="test",
        expected_files=("config.json", "weights.bin"),
        local_snapshot_path=str(tmp_path),
        resolved_at="2026-08-27T00:00:00+00:00",
    )
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        verify_entry(entry)


def test_verification_recomputes_snapshot_identity(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"test": True}), encoding="utf-8")
    entry = AssetLockEntry(
        repo_id="owner/repo",
        repo_type="model",
        immutable_revision="a" * 40,
        state="DOWNLOADED",
        backend="test",
        expected_files=("config.json",),
        local_snapshot_path=str(tmp_path),
        resolved_at="2026-08-27T00:00:00+00:00",
    )
    verified = verify_entry(entry)
    assert verified.state is AssetState.VERIFIED
    assert verified.local_snapshot_sha256 == snapshot_sha256(tmp_path)
