"""Manual CLI for the paper experiment execution pack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fidmem.experiments.execution_pack import (
    CheckFailure,
    ExperimentRunner,
    load_registry,
    validate_registry,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed manual runner for registered paper experiments."
    )
    parser.add_argument("--experiment", help="stable experiment ID, for example E03")
    parser.add_argument("--registry", default="configs/experiments/registry.yaml")
    parser.add_argument("--config", help="explicit config override")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-root", default="artifacts/experiments")
    parser.add_argument("--gate-root", default="artifacts/experiment-gates")
    parser.add_argument(
        "--gpus", default="", help="comma-separated physical GPU indices"
    )
    parser.add_argument("--run-id", help="immutable run identity override")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--check", action="store_true", help="validate only; never execute"
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--validate-registry", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = Path(args.project_root).resolve()
    registry_path = Path(args.registry)
    if not registry_path.is_absolute():
        registry_path = project_root / registry_path
    registry = load_registry(registry_path)
    if args.list:
        print(
            json.dumps(
                [
                    {
                        "id": item.id,
                        "purpose": item.purpose,
                        "dependencies": list(item.dependencies),
                        "required_gates": list(item.required_gates),
                        "gpu_required": item.gpu_required,
                    }
                    for item in registry.experiments
                ],
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    if args.validate_registry:
        issues = validate_registry(registry, project_root=project_root)
        print(json.dumps({"valid": not issues, "issues": issues}, indent=2))
        return 0 if not issues else 2
    if not args.experiment:
        raise SystemExit(
            "--experiment is required unless --list/--validate-registry is used"
        )
    output_root = Path(args.output_root)
    gate_root = Path(args.gate_root)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    if not gate_root.is_absolute():
        gate_root = project_root / gate_root
    runner = ExperimentRunner(
        registry_path=registry_path,
        project_root=project_root,
        gate_root=gate_root,
        output_root=output_root,
    )
    try:
        preflight = runner.check(
            args.experiment,
            config_path=args.config,
            gpus=args.gpus,
            run_id=args.run_id,
            resume=args.resume,
        )
        if args.check:
            print(json.dumps(preflight, indent=2, ensure_ascii=False))
            return 0
        run_dir = runner.execute_preflighted(
            preflight, run_id=str(preflight["run_id"]), resume=args.resume
        )
        status = json.loads((run_dir / "STATUS.json").read_text(encoding="utf-8"))
        print(json.dumps({"run_dir": str(run_dir), **status}, indent=2))
        return 0 if status["status"] == "COMPLETED" else 2
    except (CheckFailure, OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "FAIL_CLOSED",
                    "experiment_id": args.experiment,
                    "reason": str(exc),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
