from __future__ import annotations

import subprocess

from fidmem.production.authority import probe_repository, source_tree_sha256


def test_source_tree_identity_excludes_runtime_and_generated_outputs(tmp_path) -> None:
    source = tmp_path / "src" / "fidmem" / "provider.py"
    config = tmp_path / "configs" / "production.yaml"
    source.parent.mkdir(parents=True)
    config.parent.mkdir(parents=True)
    source.write_text("PROVIDER = 'v1'\n", encoding="utf-8")
    config.write_text("model: frozen\n", encoding="utf-8")
    candidates = (
        "src/fidmem/provider.py",
        "configs/production.yaml",
        "PRODUCTION_AUTHORITY.json",
        "artifacts/production/a/report.json",
        ".aris/state.json",
        "refine-logs/EXPERIMENT_TRACKER.md",
        "runtime.log",
        ".pytest_cache/state",
        "src/fidmem/__pycache__/provider.pyc",
        "src/fidmem/.provider.py.tmp",
    )
    baseline = source_tree_sha256(tmp_path, candidates)

    for relative in candidates[2:]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("generated\n", encoding="utf-8")

    assert source_tree_sha256(tmp_path, candidates) == baseline


def test_source_tree_identity_changes_for_source_or_config(tmp_path) -> None:
    source = tmp_path / "src" / "fidmem" / "provider.py"
    config = tmp_path / "configs" / "production.yaml"
    source.parent.mkdir(parents=True)
    config.parent.mkdir(parents=True)
    source.write_text("PROVIDER = 'v1'\n", encoding="utf-8")
    config.write_text("model: frozen\n", encoding="utf-8")
    candidates = ("src/fidmem/provider.py", "configs/production.yaml")
    baseline = source_tree_sha256(tmp_path, candidates)

    source.write_text("PROVIDER = 'v2'\n", encoding="utf-8")
    after_source = source_tree_sha256(tmp_path, candidates)
    config.write_text("model: new-frozen-revision\n", encoding="utf-8")

    assert after_source != baseline
    assert source_tree_sha256(tmp_path, candidates) != after_source


def test_repository_probe_ignores_outputs_but_binds_source(tmp_path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Test User"],
        check=True,
    )
    source = tmp_path / "src" / "fidmem" / "provider.py"
    config = tmp_path / "configs" / "production.yaml"
    source.parent.mkdir(parents=True)
    config.parent.mkdir(parents=True)
    source.write_text("PROVIDER = 'v1'\n", encoding="utf-8")
    config.write_text("model: frozen\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "src", "configs"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-qm", "baseline"], check=True
    )
    baseline = probe_repository(tmp_path).source_tree_sha256

    generated = {
        "PRODUCTION_AUTHORITY.json": "{}\n",
        "artifacts/production/hash/runs/canary/report.json": "{}\n",
        "artifacts/production/hash/cache/envelope.json": "{}\n",
        ".aris/state.json": "{}\n",
        "refine-logs/EXPERIMENT_TRACKER.md": "generated\n",
        "reports/canary.md": "generated\n",
        "logs/runtime.log": "generated\n",
    }
    for relative, content in generated.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    assert probe_repository(tmp_path).source_tree_sha256 == baseline

    source.write_text("PROVIDER = 'v2'\n", encoding="utf-8")
    assert probe_repository(tmp_path).source_tree_sha256 != baseline
