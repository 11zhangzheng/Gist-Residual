"""Engineering-evidence-only tests for no-download setup preflight."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from fidmem.assets.cli import operate
from fidmem.assets.resolver import AssetLock, AssetState, load_asset_lock, write_asset_lock
from fidmem.assets.stack import load_experiment_stack


ROOT = Path(__file__).resolve().parents[2]


def test_download_check_never_invokes_downloader(monkeypatch, tmp_path: Path) -> None:
    roots = {
        "FIDMEM_DATA_ROOT": str(tmp_path / "data"),
        "FIDMEM_MODEL_ROOT": str(tmp_path / "models"),
        "FIDMEM_CACHE_ROOT": str(tmp_path / "cache"),
        "FIDMEM_ARTIFACT_ROOT": str(tmp_path / "artifacts"),
    }
    for path in roots.values():
        Path(path).mkdir()
    for name, value in roots.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr("fidmem.assets.cli._hub_version", lambda: "engineering-fixture")
    monkeypatch.setattr(
        "fidmem.assets.cli.check_storage_roots",
        lambda observed: {
            name: {"path": str(path), "free_gb": 100.0}
            for name, path in observed.items()
        },
    )
    invoked = False

    def forbidden_downloader(*_args, **_kwargs):
        nonlocal invoked
        invoked = True
        raise AssertionError("download must not be called by --check")

    result = operate(
        "download",
        stack_path=ROOT / "configs/experiment_stacks/gist_residual_v1.yaml",
        lock_path=ROOT
        / "configs/experiment_stacks/gist_residual_v1.assets.lock.json",
        asset_kind="models",
        check=True,
        dry_run=False,
        resume=False,
        verify_only=False,
        downloader=forbidden_downloader,
    )

    assert result["status"] == "CHECK_PASSED"
    assert result["assets"] == [
        {"asset_id": "bge_m3", "download_invoked": False},
        {"asset_id": "qwen3_8b", "download_invoked": False},
        {"asset_id": "qwen3_vl_8b_instruct", "download_invoked": False},
        {"asset_id": "siglip2_so400m_patch14_384", "download_invoked": False},
    ]
    assert invoked is False


def test_failed_resolution_is_atomically_recorded(monkeypatch, tmp_path: Path) -> None:
    lock_path = tmp_path / "assets.lock.json"
    lock_path.write_bytes(
        (
            ROOT / "configs/experiment_stacks/gist_residual_v1.assets.lock.json"
        ).read_bytes()
    )
    monkeypatch.setattr("fidmem.assets.cli._hub_version", lambda: "engineering-fixture")

    def fail_resolution(_repo: str, _repo_type: str):
        raise RuntimeError("engineering fixture resolution failure")

    with pytest.raises(RuntimeError, match="engineering fixture"):
        operate(
            "resolve",
            stack_path=ROOT / "configs/experiment_stacks/gist_residual_v1.yaml",
            lock_path=lock_path,
            asset_kind="dataset",
            check=False,
            dry_run=False,
            resume=False,
            verify_only=False,
            info_loader=fail_resolution,
        )
    failed = load_asset_lock(lock_path).physical_assets["videomme_v2_metadata"]
    assert failed.state is AssetState.FAILED
    assert failed.failure == "RuntimeError: engineering fixture resolution failure"


def test_reconcile_check_reports_preserved_and_reset_assets_without_writing(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "assets.lock.json"
    lock_path.write_bytes(
        (
            ROOT / "configs/experiment_stacks/gist_residual_v1.assets.lock.json"
        ).read_bytes()
    )
    stack_payload = load_experiment_stack(
        ROOT / "configs/experiment_stacks/gist_residual_v1.yaml"
    ).model_dump(mode="json")
    previous = load_asset_lock(lock_path)
    for asset_id, entry in previous.physical_assets.items():
        stack_payload["physical_assets"][asset_id]["immutable_revision"] = (
            entry.immutable_revision
        )
    replacement = stack_payload["physical_assets"].pop("videomme_v2_metadata")
    replacement.update(
        {"repo_id": "owner/replacement-dataset", "immutable_revision": "b" * 40}
    )
    stack_payload["physical_assets"]["replacement_dataset"] = replacement
    stack_payload["logical_roles"]["source_dataset"] = "replacement_dataset"
    stack_path = tmp_path / "stack.yaml"
    OmegaConf.save(config=OmegaConf.create(stack_payload), f=stack_path)
    before = lock_path.read_bytes()

    checked = operate(
        "reconcile",
        stack_path=stack_path,
        lock_path=lock_path,
        asset_kind="all",
        check=True,
        dry_run=False,
        resume=False,
        verify_only=False,
    )

    assert checked["status"] == "CHECK_PASSED"
    assert checked["preserved_asset_ids"] == [
        "bge_m3",
        "qwen3_8b",
        "qwen3_vl_8b_instruct",
        "siglip2_so400m_patch14_384",
    ]
    assert checked["reset_asset_ids"] == ["replacement_dataset"]
    assert lock_path.read_bytes() == before


def test_reconcile_writes_changed_dataset_with_preserved_verified_models(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "assets.lock.json"
    lock_path.write_bytes(
        (
            ROOT / "configs/experiment_stacks/gist_residual_v1.assets.lock.json"
        ).read_bytes()
    )
    stack_payload = load_experiment_stack(
        ROOT / "configs/experiment_stacks/gist_residual_v1.yaml"
    ).model_dump(mode="json")
    previous = load_asset_lock(lock_path)
    for asset_id, entry in previous.physical_assets.items():
        stack_payload["physical_assets"][asset_id]["immutable_revision"] = (
            entry.immutable_revision
        )
    replacement = stack_payload["physical_assets"].pop("videomme_v2_metadata")
    replacement.update(
        {"repo_id": "owner/replacement-dataset", "immutable_revision": "b" * 40}
    )
    stack_payload["physical_assets"]["replacement_dataset"] = replacement
    stack_payload["logical_roles"]["source_dataset"] = "replacement_dataset"
    stack_path = tmp_path / "stack.yaml"
    OmegaConf.save(config=OmegaConf.create(stack_payload), f=stack_path)

    result = operate(
        "reconcile",
        stack_path=stack_path,
        lock_path=lock_path,
        asset_kind="all",
        check=False,
        dry_run=False,
        resume=False,
        verify_only=False,
    )

    reconciled = load_asset_lock(lock_path)
    assert result["status"] == "COMPLETED"
    assert reconciled.physical_assets["bge_m3"] == previous.physical_assets["bge_m3"]
    assert reconciled.physical_assets["replacement_dataset"].state is AssetState.RESOLVED
    assert reconciled.physical_assets["replacement_dataset"].repo_id == "owner/replacement-dataset"


def test_reconcile_rejects_foreign_stack_lock_without_writing(tmp_path: Path) -> None:
    lock_path = tmp_path / "assets.lock.json"
    previous = load_asset_lock(
        ROOT / "configs/experiment_stacks/gist_residual_v1.assets.lock.json"
    )
    foreign_lock = AssetLock.create(
        stack_id="foreign-stack",
        generated_at=previous.generated_at,
        logical_roles=dict(previous.logical_roles),
        physical_assets=dict(previous.physical_assets),
        huggingface_hub_version=previous.huggingface_hub_version,
    )
    write_asset_lock(lock_path, foreign_lock)
    before = lock_path.read_bytes()

    with pytest.raises(ValueError, match="stack config and asset lock identities differ"):
        operate(
            "reconcile",
            stack_path=ROOT / "configs/experiment_stacks/gist_residual_v1.yaml",
            lock_path=lock_path,
            asset_kind="all",
            check=False,
            dry_run=False,
            resume=False,
            verify_only=False,
        )

    assert lock_path.read_bytes() == before
