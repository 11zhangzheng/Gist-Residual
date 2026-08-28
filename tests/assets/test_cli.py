"""Engineering-evidence-only tests for no-download setup preflight."""

from pathlib import Path

import pytest

from fidmem.assets.cli import operate
from fidmem.assets.resolver import AssetState, load_asset_lock


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

    with pytest.raises(
        ValueError, match="is unresolved|lacks a resolved remote file manifest"
    ):
        operate(
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
    failed = load_asset_lock(lock_path).physical_assets["longtvqa_metadata"]
    assert failed.state is AssetState.FAILED
    assert failed.failure == "RuntimeError: engineering fixture resolution failure"
