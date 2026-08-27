from __future__ import annotations

import pytest

from fidmem.production.generation import GenerationStore


AUTHORITY = "a" * 64
ARTIFACTS = {
    "observations.jsonl": b"old observations\n",
    "cost.csv": b"old cost\n",
    "manifest.json": b'{"authority_sha256":"old"}\n',
    "state.json": b'{"state":"old"}\n',
    "cache_manifest.json": b'{"cache":"old"}\n',
    "report.json": b'{"report":"old"}\n',
}


@pytest.mark.parametrize("failure_at", tuple(ARTIFACTS))
def test_failed_generation_publication_preserves_current_bytes(
    tmp_path, failure_at
) -> None:
    store = GenerationStore(tmp_path / "run", AUTHORITY)
    current = store.publish(ARTIFACTS)
    before_pointer = store.pointer_path.read_bytes()
    before = {name: (current / name).read_bytes() for name in ARTIFACTS}

    def fail(stage: str) -> None:
        if stage == failure_at:
            raise RuntimeError(f"injected failure at {stage}")

    changed = {name: value.replace(b"old", b"new") for name, value in ARTIFACTS.items()}
    with pytest.raises(RuntimeError, match="injected failure"):
        store.publish(changed, failure_hook=fail)

    assert store.pointer_path.read_bytes() == before_pointer
    assert store.current_path() == current
    assert {name: (current / name).read_bytes() for name in ARTIFACTS} == before
