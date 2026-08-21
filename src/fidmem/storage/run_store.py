"""DuckDB-backed state for work that can survive worker crashes."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb


class RunStore:
    def __init__(self, database: Path | str, lease_seconds: float = 300) -> None:
        if lease_seconds < 0:
            raise ValueError("lease_seconds must be non-negative")
        self.database = str(database)
        self.lease_seconds = lease_seconds
        with self._connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS run_items (run_id VARCHAR NOT NULL, item_key VARCHAR NOT NULL, status VARCHAR NOT NULL, attempt INTEGER NOT NULL, worker_id VARCHAR, started_at TIMESTAMPTZ, finished_at TIMESTAMPTZ, error_type VARCHAR, error_message VARCHAR, output_uri VARCHAR, PRIMARY KEY (run_id, item_key))")

    def claim(self, run_id: str, item_key: str, worker_id: str) -> bool:
        with self._connect() as connection:
            connection.execute("BEGIN TRANSACTION")
            try:
                connection.execute("INSERT INTO run_items (run_id, item_key, status, attempt) VALUES (?, ?, 'pending', 0) ON CONFLICT (run_id, item_key) DO NOTHING", [run_id, item_key])
                result = connection.execute("UPDATE run_items SET status = 'running', attempt = attempt + 1, worker_id = ?, started_at = ?, finished_at = NULL, error_type = NULL, error_message = NULL WHERE run_id = ? AND item_key = ? AND status IN ('pending', 'failed') RETURNING item_key", [worker_id, _utcnow(), run_id, item_key]).fetchone()
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        return result is not None

    def complete(self, run_id: str, item_key: str, output_uri: str) -> None:
        if not Path(output_uri).exists():
            raise ValueError("output_uri must refer to an existing output")
        with self._connect() as connection:
            result = connection.execute("UPDATE run_items SET status = 'complete', output_uri = ?, finished_at = ? WHERE run_id = ? AND item_key = ? AND status = 'running' RETURNING item_key", [output_uri, _utcnow(), run_id, item_key]).fetchone()
        if result is None:
            raise ValueError("only running items can be completed")

    def fail(self, run_id: str, item_key: str, error_type: str, error_message: str) -> None:
        with self._connect() as connection:
            result = connection.execute("UPDATE run_items SET status = 'failed', error_type = ?, error_message = ?, finished_at = ? WHERE run_id = ? AND item_key = ? AND status = 'running' RETURNING item_key", [error_type, error_message, _utcnow(), run_id, item_key]).fetchone()
        if result is None:
            raise ValueError("only running items can be failed")

    def pending(self, run_id: str) -> list[str]:
        cutoff = _utcnow() - timedelta(seconds=self.lease_seconds)
        with self._connect() as connection:
            connection.execute("UPDATE run_items SET status = 'pending', worker_id = NULL, started_at = NULL WHERE run_id = ? AND status = 'running' AND started_at <= ?", [run_id, cutoff])
            rows = connection.execute("SELECT item_key FROM run_items WHERE run_id = ? AND status = 'pending' ORDER BY item_key", [run_id]).fetchall()
        return [row[0] for row in rows]

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(self.database)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)
