from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from pathlib import Path
from fidmem.storage.run_store import RunStore

def test_claim_is_single_winner_under_concurrent_connections(tmp_path: Path) -> None:
    store=RunStore(tmp_path/"r.duckdb")
    barrier=Barrier(8)
    def claim(i:int):
        barrier.wait(); return store.claim("run","item",f"w{i}")
    with ThreadPoolExecutor(max_workers=8) as pool: results=list(pool.map(claim,range(8)))
    assert results.count(True)==1
    item=store.item("run","item")
    assert item is not None and item.status=="running" and item.attempt==1
