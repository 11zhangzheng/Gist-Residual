from concurrent.futures import Future, ThreadPoolExecutor
from threading import Barrier
from pathlib import Path

import pytest

from fidmem.storage.run_store import RunStore

def _contend(store: RunStore, item_key: str) -> tuple[list[bool], list[str]]:
    barrier = Barrier(8)

    def claim(worker_id: str) -> tuple[bool, str]:
        barrier.wait()
        return store.claim("run", item_key, worker_id), worker_id

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures: list[Future[tuple[bool, str]]] = [
            pool.submit(claim, f"{item_key}-worker-{index}") for index in range(8)
        ]
        outcomes = [future.result() for future in futures]
    return [won for won, _ in outcomes], [worker for won, worker in outcomes if won]


@pytest.mark.parametrize("starting_status", ["new", "pending", "failed"])
def test_claim_has_one_real_winner_across_repeated_contention(
    tmp_path: Path, starting_status: str
) -> None:
    store = RunStore(tmp_path / f"{starting_status}.duckdb", lease_seconds=0)

    for round_index in range(20):
        item_key = f"{starting_status}-{round_index}"
        expected_attempt = 1
        if starting_status == "pending":
            assert store.claim("run", item_key, "seed")
            assert item_key in store.pending("run")
            expected_attempt = 2
        elif starting_status == "failed":
            assert store.claim("run", item_key, "seed")
            store.fail("run", item_key, "ExpectedFailure", "retry me")
            expected_attempt = 2

        results, winners = _contend(store, item_key)

        assert results.count(True) == 1
        assert len(winners) == 1
        item = store.item("run", item_key)
        assert item is not None
        assert item.status == "running"
        assert item.attempt == expected_attempt
        assert item.worker_id == winners[0]


@pytest.mark.parametrize(
    ("run_id", "item_key", "worker_id"),
    [
        (None, "item", "worker"),
        ("", "item", "worker"),
        ("run", None, "worker"),
        ("run", "", "worker"),
        ("run", "item", None),
        ("run", "item", ""),
    ],
)
def test_claim_rejects_missing_or_empty_identifiers(
    tmp_path: Path,
    run_id: str | None,
    item_key: str | None,
    worker_id: str | None,
) -> None:
    store = RunStore(tmp_path / "identifiers.duckdb")
    with pytest.raises(ValueError, match="non-empty strings"):
        store.claim(run_id, item_key, worker_id)  # type: ignore[arg-type]


def test_claim_propagates_database_connection_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = RunStore(tmp_path / "connect.duckdb")

    def fail_to_connect() -> None:
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(store, "_connect", fail_to_connect)
    with pytest.raises(RuntimeError, match="database unavailable"):
        store.claim("run", "item", "worker")
