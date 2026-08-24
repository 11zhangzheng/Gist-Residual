from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

import fidmem.router.dagger_workflow as workflow
from fidmem.router.dagger import DAggerConfig, run_dagger

from tests.router.test_dagger_round2_workflow import _run
from tests.router.test_dagger_workflow import SpyTrainer, _always_stop, _contexts


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _rewrite_sealed(path: Path, payload: dict[str, object], hash_field: str) -> None:
    body = {key: value for key, value in payload.items() if key != hash_field}
    body[hash_field] = hashlib.sha256(_canonical_bytes(body)).hexdigest()
    path.write_bytes(_canonical_bytes(body) + b"\n")


def test_failure_before_current_replace_rolls_back_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = DAggerConfig(artifact_root=tmp_path)
    _always_stop.freeze(config.bootstrap_path)
    real_replace = os.replace
    pointer_replaces = 0

    def fail_current_replace(source: str | Path, target: str | Path) -> None:
        nonlocal pointer_replaces
        if Path(target).resolve() == (tmp_path / "current.json").resolve():
            pointer_replaces += 1
            if pointer_replaces == 2:
                raise OSError("injected before current replace")
        real_replace(source, target)

    monkeypatch.setattr(workflow.os, "replace", fail_current_replace)
    contexts = _contexts(1)
    with pytest.raises(OSError, match="before current replace"):
        run_dagger(
            train_contexts=contexts,
            dev_contexts=contexts,
            initial_policy=_always_stop,
            source_policy_checkpoint=config.bootstrap_path,
            trainer=SpyTrainer(),
            config=config,
        )

    pointer = json.loads((tmp_path / "current.json").read_text(encoding="utf-8"))
    assert pointer["round_number"] == 1
    assert (tmp_path / "generations" / "round-0001").is_dir()
    assert not (tmp_path / "generations" / "round-0002").exists()
    assert not tuple((tmp_path / "generations").glob("*.staging"))


def test_failure_after_current_replace_is_committed_with_durability_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_fsync_directory = workflow._fsync_directory

    def fail_root_fsync(path: Path) -> None:
        if path.resolve() == tmp_path.resolve():
            raise OSError("injected post-commit root fsync failure")
        real_fsync_directory(path)

    monkeypatch.setattr(workflow, "_fsync_directory", fail_root_fsync)
    result = _run(tmp_path, SpyTrainer())

    assert len(result.durability_warnings) == 2
    assert all(
        "root fsync failure" in warning for warning in result.durability_warnings
    )
    pointer = json.loads((tmp_path / "current.json").read_text(encoding="utf-8"))
    assert pointer["round_number"] == 2
    assert (tmp_path / "generations" / "round-0001" / "manifest.json").is_file()
    assert (tmp_path / "generations" / "round-0002" / "manifest.json").is_file()

    monkeypatch.setattr(workflow, "_fsync_directory", real_fsync_directory)
    retried = _run(tmp_path, SpyTrainer())
    assert retried.resumed is True
    assert retried.manifests[-1].round_number == 2


def test_replace_that_commits_then_raises_preserves_generation_and_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_replace = os.replace

    def replace_then_raise(source: str | Path, target: str | Path) -> None:
        real_replace(source, target)
        if Path(target).resolve() == (tmp_path / "current.json").resolve():
            raise OSError("injected after current replace")

    monkeypatch.setattr(workflow.os, "replace", replace_then_raise)
    result = _run(tmp_path, SpyTrainer())

    assert len(result.durability_warnings) == 2
    assert all("after current replace" in item for item in result.durability_warnings)
    pointer = json.loads((tmp_path / "current.json").read_text(encoding="utf-8"))
    assert pointer["round_number"] == 2
    assert (tmp_path / "generations" / "round-0001").is_dir()
    assert (tmp_path / "generations" / "round-0002").is_dir()

    monkeypatch.setattr(workflow.os, "replace", real_replace)
    retried = _run(tmp_path, SpyTrainer())
    assert retried.resumed is True


def test_manifest_chain_binds_each_round_to_verified_predecessor(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path, SpyTrainer())

    assert result.manifests[0].previous_generation_id is None
    assert result.manifests[0].previous_generation_manifest_sha256 is None
    assert result.manifests[1].previous_generation_id == result.manifests[0].generation
    assert (
        result.manifests[1].previous_generation_manifest_sha256
        == result.manifests[0].manifest_sha256
    )


def test_resume_rejects_old_deviation_forged_as_new_with_all_hashes_resealed(
    tmp_path: Path,
) -> None:
    _run(tmp_path, SpyTrainer())
    generation = tmp_path / "generations" / "round-0002"
    deviation_path = generation / "deviations.json"
    deviation = json.loads(deviation_path.read_text(encoding="utf-8"))
    old_key = deviation["deviations"][0]["state_key"]
    deviation["new_state_keys"] = [old_key]
    _rewrite_sealed(deviation_path, deviation, "artifact_sha256")

    manifest_path = generation / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["deviation_artifact"]["sha256"] = hashlib.sha256(
        deviation_path.read_bytes()
    ).hexdigest()
    manifest["new_deviation_count"] = 1
    _rewrite_sealed(manifest_path, manifest, "manifest_sha256")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    pointer_path = tmp_path / "current.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["manifest_sha256"] = manifest["manifest_sha256"]
    _rewrite_sealed(pointer_path, pointer, "pointer_sha256")

    with pytest.raises(ValueError, match="new deviation.*predecessor|set difference"):
        _run(tmp_path, SpyTrainer())


def test_resume_rejects_missing_predecessor_generation(tmp_path: Path) -> None:
    _run(tmp_path, SpyTrainer())
    round_one = tmp_path / "generations" / "round-0001"
    removed = tmp_path / "removed-round-0001"
    round_one.rename(removed)

    with pytest.raises(ValueError, match="committed generation is missing"):
        _run(tmp_path, SpyTrainer())


@pytest.mark.parametrize(
    ("bootstrap", "generations", "current"),
    (
        ("generations/bootstrap.pt", "generations", "current.json"),
        ("bootstrap.pt", "generations", "generations/current.json"),
        ("current.json", "generations", "current.json"),
    ),
)
def test_config_rejects_critical_path_overlap(
    tmp_path: Path, bootstrap: str, generations: str, current: str
) -> None:
    with pytest.raises(ValidationError, match="overlap|generations|distinct"):
        DAggerConfig(
            artifact_root=tmp_path,
            bootstrap_checkpoint=bootstrap,
            generations_dir=generations,
            current_pointer=current,
        )


@pytest.mark.parametrize(
    "field",
    (
        "artifact_root",
        "bootstrap_checkpoint",
        "generations_dir",
        "current_pointer",
    ),
)
def test_config_rejects_non_path_types(tmp_path: Path, field: str) -> None:
    values: dict[str, object] = {"artifact_root": tmp_path}
    values[field] = 7
    with pytest.raises(ValidationError, match="strings or Path"):
        DAggerConfig(**values)


def test_config_rejects_existing_file_as_generations_directory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    generations = root / "generations"
    generations.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ValidationError, match="generations_dir.*directory"):
        DAggerConfig(artifact_root=root)


def test_config_rejects_existing_file_directory_and_legacy_paths(
    tmp_path: Path,
) -> None:
    file_root = tmp_path / "artifact-file"
    file_root.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ValidationError, match="artifact_root|directory"):
        DAggerConfig(artifact_root=file_root)

    with pytest.raises(ValidationError, match="seen_keys_path|extra"):
        DAggerConfig(artifact_root=tmp_path / "root", seen_keys_path="seen.json")


def test_config_rejects_symlink_aliases_before_and_after_resolution(
    tmp_path: Path,
) -> None:
    real_root = tmp_path / "real-root"
    real_root.mkdir()
    root_alias = tmp_path / "root-alias"
    internal_target = real_root / "real-generations"
    internal_target.mkdir()
    internal_alias = real_root / "generations-alias"
    try:
        root_alias.symlink_to(real_root, target_is_directory=True)
        internal_alias.symlink_to(internal_target, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlink unavailable: {error}")

    with pytest.raises(ValidationError, match="symlink|alias"):
        DAggerConfig(artifact_root=root_alias)
    with pytest.raises(ValidationError, match="symlink|alias"):
        DAggerConfig(artifact_root=real_root, generations_dir="generations-alias")
