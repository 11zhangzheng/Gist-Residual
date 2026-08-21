from pathlib import Path

import pytest

from fidmem.storage.run_store import RunStore


def test_expired_running_item_is_recovered_but_completed_item_is_not_rerun(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "runs.duckdb", lease_seconds=0)

    assert store.claim("run-1", "event-1", "worker-a")
    assert store.pending("run-1") == ["event-1"]
    assert store.claim("run-1", "event-1", "worker-b")

    output = tmp_path / "event-1.json"
    output.write_text('{"result": "ready"}', encoding="utf-8")
    store.complete("run-1", "event-1", str(output))

    assert store.pending("run-1") == []
    assert not store.claim("run-1", "event-1", "worker-c")


def test_complete_requires_an_existing_output_uri(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.duckdb")
    assert store.claim("run-1", "event-1", "worker-a")

    with pytest.raises(ValueError, match="output_uri"):
        store.complete("run-1", "event-1", str(tmp_path / "missing.json"))
