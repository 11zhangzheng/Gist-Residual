"""Static engineering validation for setup wrappers on hosts without Bash."""

import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from fidmem.assets import setup as asset_setup
from fidmem.assets.resolver import AssetLock, load_asset_lock, write_asset_lock


ROOT = Path(__file__).resolve().parents[2]


def test_all_setup_wrappers_are_thin_bash_entrypoints() -> None:
    setup = ROOT / "scripts/setup"
    expected = {
        "01_resolve_stack_assets.sh": "fidmem.assets.cli resolve",
        "02_download_models.sh": "fidmem.assets.cli download",
        "03_verify_models.sh": "fidmem.assets.cli verify",
        "04_download_videomme_v2_metadata.sh": "fidmem.assets.cli download",
        "05_verify_videomme_v2_metadata.sh": "fidmem.assets.setup metadata",
        "06_prepare_videomme_v2_videos.sh": "fidmem.assets.setup videos",
        "07_build_videomme_v2_manifests.sh": "fidmem.assets.setup manifests",
        "08_build_authority_draft.sh": "fidmem.assets.setup authority-draft",
    }
    assert {path.name for path in setup.glob("*.sh")} == set(expected)
    for name, command in expected.items():
        text = (setup / name).read_text(encoding="utf-8")
        assert text.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
        assert 'export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"' in text
        assert command in text
        assert '"$@"' in text


def test_video_setup_passes_verified_metadata_subtitle_to_media_preparation(
    monkeypatch, tmp_path: Path
) -> None:
    metadata_root = tmp_path / "verified-metadata"
    metadata_root.mkdir()
    (metadata_root / "subtitle.zip").write_bytes(b"fixture")
    parsed = SimpleNamespace()
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        asset_setup,
        "_metadata_asset_from_lock",
        lambda _path: (parsed, metadata_root),
        raising=False,
    )
    monkeypatch.setattr(asset_setup, "_metadata_from_lock", lambda _path: parsed)
    monkeypatch.setattr(
        asset_setup,
        "storage_roots",
        lambda: {
            "FIDMEM_DATA_ROOT": tmp_path / "data",
            "FIDMEM_MODEL_ROOT": tmp_path / "models",
            "FIDMEM_CACHE_ROOT": tmp_path / "cache",
            "FIDMEM_ARTIFACT_ROOT": tmp_path / "artifacts",
        },
    )
    monkeypatch.setenv("FIDMEM_VIDEOMME_V2_RAW_ROOT", str(tmp_path / "raw"))
    monkeypatch.setenv("FIDMEM_CACHE_ROOT", str(tmp_path / "cache"))

    def capture_prepare(*args, **kwargs):
        observed.update(kwargs)
        return SimpleNamespace(model_dump=lambda **_kwargs: {"status": "CHECKED"})

    monkeypatch.setattr(asset_setup, "prepare_videos", capture_prepare)

    assert asset_setup.main(
        [
            "videos",
            "--check",
            "--project-root",
            str(tmp_path),
            "--lock",
            str(ROOT / "configs/experiment_stacks/gist_residual_v1.assets.lock.json"),
        ]
    ) == 0
    assert observed["subtitle_zip"] == metadata_root / "subtitle.zip"


def test_video_setup_writes_noncheck_result_to_output_directory(
    monkeypatch, tmp_path: Path
) -> None:
    metadata_root = tmp_path / "verified-metadata"
    metadata_root.mkdir()
    parsed = SimpleNamespace()
    monkeypatch.setattr(
        asset_setup,
        "_metadata_asset_from_lock",
        lambda _path: (parsed, metadata_root),
    )
    monkeypatch.setattr(
        asset_setup,
        "storage_roots",
        lambda: {
            "FIDMEM_DATA_ROOT": tmp_path / "data",
            "FIDMEM_MODEL_ROOT": tmp_path / "models",
            "FIDMEM_CACHE_ROOT": tmp_path / "cache",
            "FIDMEM_ARTIFACT_ROOT": tmp_path / "artifacts",
        },
    )
    monkeypatch.setenv("FIDMEM_VIDEOMME_V2_RAW_ROOT", str(tmp_path / "raw"))
    monkeypatch.setenv("FIDMEM_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setattr(
        asset_setup,
        "prepare_videos",
        lambda *args, **kwargs: SimpleNamespace(
            model_dump=lambda **_kwargs: {"status": "PREPARED"}
        ),
    )
    output = tmp_path / "preparation"

    assert asset_setup.main(
        [
            "videos",
            "--resume",
            "--output",
            str(output),
            "--project-root",
            str(tmp_path),
        ]
    ) == 0

    assert json.loads((output / "media_preparation.json").read_text()) == {
        "status": "PREPARED"
    }


def test_metadata_setup_rejects_lock_with_different_frozen_revision(
    tmp_path: Path,
) -> None:
    original = load_asset_lock(
        ROOT / "configs/experiment_stacks/gist_residual_v1.assets.lock.json"
    )
    entries = dict(original.physical_assets)
    source_id = original.logical_roles["source_dataset"]
    entries[source_id] = entries[source_id].model_copy(
        update={"immutable_revision": "a" * 40}
    )
    altered = AssetLock.create(
        stack_id=original.stack_id,
        generated_at=original.generated_at,
        logical_roles=dict(original.logical_roles),
        physical_assets=entries,
        huggingface_hub_version=original.huggingface_hub_version,
    )
    lock_path = tmp_path / "assets.lock.json"
    write_asset_lock(lock_path, altered)

    with pytest.raises(ValueError, match="frozen revision"):
        asset_setup._metadata_asset_from_lock(lock_path)


def test_e01_check_does_not_require_run_directory(
    monkeypatch, tmp_path: Path
) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        asset_setup,
        "storage_roots",
        lambda: {
            "FIDMEM_DATA_ROOT": tmp_path / "data",
            "FIDMEM_MODEL_ROOT": tmp_path / "models",
            "FIDMEM_CACHE_ROOT": tmp_path / "cache",
            "FIDMEM_ARTIFACT_ROOT": tmp_path / "artifacts",
        },
    )
    monkeypatch.setenv("FIDMEM_VIDEOMME_V2_PREPARATION_ROOT", str(tmp_path / "prep"))
    monkeypatch.setenv("FIDMEM_VIDEOMME_V2_HUMAN_AUDIT_RESULT", str(tmp_path / "audit"))
    monkeypatch.delenv("FIDMEM_RUN_DIR", raising=False)

    def capture_e01(*args, **kwargs):
        observed.update(kwargs)
        return {"status": "CHECK_PASSED"}

    monkeypatch.setattr(asset_setup, "prepare_videomme_e01", capture_e01)

    assert asset_setup.main(["e01", "--check", "--project-root", str(tmp_path)]) == 0
    assert observed["output_dir"] is None
