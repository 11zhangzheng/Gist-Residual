"""Create hashed gate artifacts from real result files and frozen thresholds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf

from fidmem.experiments.execution_pack import GateRecord
from fidmem.experiments.gates import evaluate_gate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate and record an experiment gate"
    )
    parser.add_argument("--gate", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--protocol-version", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--authority-sha256")
    parser.add_argument("--result", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result_path = Path(args.result)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    thresholds = OmegaConf.to_container(OmegaConf.load(args.thresholds), resolve=True)
    if not isinstance(result, dict) or not isinstance(thresholds, dict):
        raise SystemExit("result and thresholds must be mappings")
    status, checks = evaluate_gate(args.gate, result, thresholds)
    record = GateRecord.create(
        gate_id=args.gate,
        experiment_id=args.experiment,
        run_id=args.run_id,
        status=status,
        protocol_version=args.protocol_version,
        config_sha256=args.config_sha256,
        result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
        authority_sha256=args.authority_sha256,
        checks=checks,
        thresholds=thresholds,
    )
    record.write(args.output)
    print(json.dumps({**record.payload(), "gate_sha256": record.gate_sha256}, indent=2))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
