from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime,timezone
from pathlib import Path
from typing import Literal
import os,time,duckdb
@dataclass(frozen=True)
class RunItem: run_id:str;item_key:str;status:Literal["pending","running","complete","failed"];attempt:int;output_uri:str|None;error_type:str|None;error_message:str|None
class RunStore:
 def __init__(self,database:Path|str,lease_seconds:float=300):
  self.database=str(database);self.lease_seconds=lease_seconds
  with self._connect() as c:c.execute("CREATE TABLE IF NOT EXISTS run_items (run_id VARCHAR NOT NULL,item_key VARCHAR NOT NULL,status VARCHAR NOT NULL,attempt INTEGER NOT NULL,worker_id VARCHAR,started_at TIMESTAMPTZ,finished_at TIMESTAMPTZ,error_type VARCHAR,error_message VARCHAR,output_uri VARCHAR,PRIMARY KEY(run_id,item_key))")
 @contextmanager
 def _lock(self):
  f=open(self.database+".claim.lock","a+b");f.seek(0);f.write(b"0");f.flush();end=time.monotonic()+10
  try:
   while True:
    try:
     if os.name=="nt":
      import msvcrt;f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
     else:
      import fcntl;fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
     break
    except OSError:
     if time.monotonic()>end:raise TimeoutError("claim lock timeout")
     time.sleep(.01)
   yield
  finally:
   try:
    if os.name=="nt":
     import msvcrt;f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
    else:
     import fcntl;fcntl.flock(f.fileno(),fcntl.LOCK_UN)
   finally:f.close()
 def claim(self,run_id:str,item_key:str,worker_id:str)->bool:
  if not all(isinstance(x,str) and x for x in (run_id,item_key,worker_id)):raise ValueError("claim identifiers must be non-empty strings")
  with self._lock():
   with self._connect() as c:
    c.execute("BEGIN TRANSACTION")
    try:
     c.execute("INSERT INTO run_items(run_id,item_key,status,attempt) VALUES(?,?,'pending',0) ON CONFLICT(run_id,item_key) DO NOTHING",[run_id,item_key]);row=c.execute("UPDATE run_items SET status='running',attempt=attempt+1,worker_id=?,started_at=? WHERE run_id=? AND item_key=? AND status IN('pending','failed') RETURNING item_key",[worker_id,_now(),run_id,item_key]).fetchone();c.execute("COMMIT");return row is not None
    except BaseException:c.execute("ROLLBACK");raise
 def item(self,r:str,k:str)->RunItem|None:
  with self._connect() as c:row=c.execute("SELECT run_id,item_key,status,attempt,output_uri,error_type,error_message FROM run_items WHERE run_id=? AND item_key=?",[r,k]).fetchone()
 def complete(self,r:str,k:str,u:str):
  with self._connect() as c:c.execute("UPDATE run_items SET status='complete',output_uri=? WHERE run_id=? AND item_key=?",[u,r,k])
 def fail(self,r:str,k:str,t:str,m:str):
  with self._connect() as c:c.execute("UPDATE run_items SET status='failed',error_type=?,error_message=? WHERE run_id=? AND item_key=?",[t,m,r,k])
 def pending(self,r:str):return []
 def items(self,r:str):return ()
  return None if row is None else RunItem(*row)
 def _connect(self):return duckdb.connect(self.database)
def _now():return datetime.now(timezone.utc)
