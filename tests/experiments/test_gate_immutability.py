from pathlib import Path

import pytest

from fidmem.experiments.execution_pack import GateRecord


def _record(run_id: str) -> GateRecord:
    return GateRecord.create(
        gate_id="production_canary",
        experiment_id="E03",
        run_id=run_id,
        status="PASS",
        protocol_version="v1",
        config_sha256="a" * 64,
        result_sha256="b" * 64,
        authority_sha256="c" * 64,
        checks={"all": True},
        thresholds={},
    )


def test_gate_write_is_idempotent_but_never_overwrites_identity(tmp_path: Path) -> None:
    path = tmp_path / "production_canary.json"
    first = _record("run-1")
    first.write(path)
    before = path.read_bytes()
    first.write(path)
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="different identity"):
        _record("run-2").write(path)
    assert path.read_bytes() == before
