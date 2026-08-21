"""DuckDB-backed state for resumable work."""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
from threading import Lock
import time
from typing import Iterator, Literal
import duckdb

_CLAIM_LOCKS_GUARD = Lock()
_CLAIM_LOCKS: dict[str, Lock] = {}


def _thread_lock(path: Path) -> Lock:
    key = str(path.resolve())
    with _CLAIM_LOCKS_GUARD:
        return _CLAIM_LOCKS.setdefault(key, Lock())


@contextmanager
def _claim_file_lock(path: Path, timeout_seconds: float) -> Iterator[None]:
    """Hold a per-path thread lock and a non-blocking cross-process file lock."""
    if timeout_seconds < 0:
        raise ValueError("timeout_seconds must be non-negative")
    deadline = time.monotonic() + timeout_seconds
    thread_lock = _thread_lock(path)
    if not thread_lock.acquire(timeout=timeout_seconds):
        raise TimeoutError("claim lock timeout")

    handle = None
    file_locked = False
    try:
        handle = open(path, "a+b")
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()

        while True:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                file_locked = True
                break
            except OSError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("claim lock timeout")
                time.sleep(min(0.01, remaining))
        yield
    finally:
        try:
            if handle is not None and file_locked:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            if handle is not None:
                handle.close()
            thread_lock.release()
@dataclass(frozen=True)
class RunItem:
    run_id:str; item_key:str; status:Literal["pending","running","complete","failed"]; attempt:int
    worker_id:str|None; output_uri:str|None; error_type:str|None; error_message:str|None
class RunStore:
    def __init__(self,database:Path|str,lease_seconds:float=300)->None:
        if lease_seconds<0:raise ValueError("lease_seconds must be non-negative")
        self.database=str(database);self.lease_seconds=lease_seconds
        with self._connect() as c:c.execute("CREATE TABLE IF NOT EXISTS run_items (run_id VARCHAR NOT NULL, item_key VARCHAR NOT NULL, status VARCHAR NOT NULL, attempt INTEGER NOT NULL, worker_id VARCHAR, started_at TIMESTAMPTZ, finished_at TIMESTAMPTZ, error_type VARCHAR, error_message VARCHAR, output_uri VARCHAR, PRIMARY KEY (run_id, item_key))")
    def claim(self,run_id:str,item_key:str,worker_id:str)->bool:
        if not all(isinstance(x,str) and x for x in (run_id,item_key,worker_id)): raise ValueError("claim identifiers must be non-empty strings")
        with _claim_file_lock(Path(f"{self.database}.claim.lock"), 10):
            with self._connect() as c:
                c.execute("BEGIN TRANSACTION")
                try:
                    c.execute("INSERT INTO run_items (run_id,item_key,status,attempt) VALUES (?,?,'pending',0) ON CONFLICT (run_id,item_key) DO NOTHING",[run_id,item_key])
                    row=c.execute("UPDATE run_items SET status='running',attempt=attempt+1,worker_id=?,started_at=?,finished_at=NULL,error_type=NULL,error_message=NULL WHERE run_id=? AND item_key=? AND status IN ('pending','failed') RETURNING item_key",[worker_id,_utcnow(),run_id,item_key]).fetchone();c.execute("COMMIT");return row is not None
                except BaseException:c.execute("ROLLBACK");raise
    def complete(self,run_id:str,item_key:str,output_uri:str)->None:
        if not Path(output_uri).exists():raise ValueError("output_uri must refer to an existing output")
        with self._connect() as c:result=c.execute("UPDATE run_items SET status='complete',output_uri=?,finished_at=? WHERE run_id=? AND item_key=? AND status='running' RETURNING item_key",[output_uri,_utcnow(),run_id,item_key]).fetchone()
        if result is None:raise ValueError("only running items can be completed")
    def fail(self,run_id:str,item_key:str,error_type:str,error_message:str)->None:
        with self._connect() as c:result=c.execute("UPDATE run_items SET status='failed',error_type=?,error_message=?,finished_at=? WHERE run_id=? AND item_key=? AND status='running' RETURNING item_key",[error_type,error_message,_utcnow(),run_id,item_key]).fetchone()
        if result is None:raise ValueError("only running items can be failed")
    def pending(self,run_id:str)->list[str]:
        with self._connect() as c:
            c.execute("UPDATE run_items SET status='pending',worker_id=NULL,started_at=NULL WHERE run_id=? AND status='running' AND started_at<=?",[run_id,_utcnow()-timedelta(seconds=self.lease_seconds)])
            return [x[0] for x in c.execute("SELECT item_key FROM run_items WHERE run_id=? AND status='pending' ORDER BY item_key",[run_id]).fetchall()]
    def item(self,run_id:str,item_key:str)->RunItem|None:
        with self._connect() as c:row=c.execute("SELECT run_id,item_key,status,attempt,worker_id,output_uri,error_type,error_message FROM run_items WHERE run_id=? AND item_key=?",[run_id,item_key]).fetchone()
        return None if row is None else RunItem(*row)
    def items(self,run_id:str)->tuple[RunItem,...]:
        with self._connect() as c:rows=c.execute("SELECT run_id,item_key,status,attempt,worker_id,output_uri,error_type,error_message FROM run_items WHERE run_id=? ORDER BY item_key",[run_id]).fetchall()
        return tuple(RunItem(*x) for x in rows)
    def _connect(self)->duckdb.DuckDBPyConnection:return duckdb.connect(self.database)
def _utcnow()->datetime:return datetime.now(timezone.utc)
