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
    reconcile_lock,
    resolve_entry,
    snapshot_sha256,
    verify_entry,
)
from fidmem.assets.stack import ExperimentStack, load_experiment_stack


ROOT = Path(__file__).resolve().parents[2]
REVISION = "a" * 40
REMOTE_FILES = ("README.md", "subtitle.zip", "test.parquet", "videos/001.zip")


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


def test_dataset_resolution_keeps_only_required_remote_files() -> None:
    dataset_entry = AssetLockEntry(
        repo_id="owner/dataset",
        repo_type="dataset",
        state="UNRESOLVED",
        backend="huggingface_hub",
    )

    resolved = resolve_entry(
        dataset_entry,
        info_loader=lambda _repo, _type: (REVISION, REMOTE_FILES),
        required_files=("README.md", "subtitle.zip", "test.parquet"),
    )

    assert resolved.expected_files == ("README.md", "subtitle.zip", "test.parquet")
    assert "videos/001.zip" not in resolved.expected_files


def test_dataset_resolution_rejects_missing_required_remote_file() -> None:
    dataset_entry = AssetLockEntry(
        repo_id="owner/dataset",
        repo_type="dataset",
        state="UNRESOLVED",
        backend="huggingface_hub",
    )

    with pytest.raises(ValueError, match="required remote files are missing"):
        resolve_entry(
            dataset_entry,
            info_loader=lambda _repo, _type: (REVISION, REMOTE_FILES),
            required_files=("README.md", "missing.parquet"),
        )


def test_model_resolution_keeps_full_remote_listing() -> None:
    model_entry = AssetLockEntry(
        repo_id="owner/model",
        repo_type="model",
        state="UNRESOLVED",
        backend="transformers",
    )

    resolved = resolve_entry(
        model_entry,
        info_loader=lambda _repo, _type: (REVISION, REMOTE_FILES),
        required_files=("README.md",),
    )

    assert resolved.expected_files == REMOTE_FILES


def test_reconcile_preserves_exact_model_identities_and_resets_changed_dataset() -> None:
    previous = AssetLock.model_validate_json(
        (ROOT / "configs/experiment_stacks/gist_residual_v1.assets.lock.json").read_text(
            encoding="utf-8"
        )
    )
    stack_payload = load_experiment_stack(
        ROOT / "configs/experiment_stacks/gist_residual_v1.yaml"
    ).model_dump(mode="json")
    for asset_id, entry in previous.physical_assets.items():
        stack_payload["physical_assets"][asset_id]["immutable_revision"] = (
            entry.immutable_revision
        )
    stack_payload["physical_assets"]["longtvqa_metadata"].update(
        {"repo_id": "owner/replacement-dataset", "immutable_revision": "b" * 40}
    )

    reconciled = reconcile_lock(ExperimentStack.model_validate(stack_payload), previous)

    for asset_id in (
        "bge_m3",
        "qwen3_8b",
        "qwen3_vl_8b_instruct",
        "siglip2_so400m_patch14_384",
    ):
        assert reconciled.physical_assets[asset_id] == previous.physical_assets[asset_id]
        assert reconciled.physical_assets[asset_id].state is AssetState.VERIFIED
    replacement = reconciled.physical_assets["longtvqa_metadata"]
    assert replacement.repo_id == "owner/replacement-dataset"
    assert replacement.immutable_revision == "b" * 40
    assert replacement.state is AssetState.RESOLVED
    assert replacement.expected_files == ()
    assert replacement.local_snapshot_path is None


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
