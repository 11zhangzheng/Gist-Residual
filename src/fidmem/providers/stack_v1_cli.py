"""E03 entrypoint for a separately supplied Stack v1 model backend implementation."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Any

from fidmem.providers.stack_v1 import (
    ExecutionRequest,
    StackV1Backend,
    _atomic_bytes,
    check_stack_assets,
    execute_batch,
)
from fidmem.experiments.observation_import import import_production_observations
from fidmem.production.cli_support import production_report_result
from fidmem.production.authority import canonical_json_bytes


def _load_requests(path: Path) -> tuple[ExecutionRequest, ...]:
    if not path.is_file():
        raise ValueError(f"provider request manifest is missing: {path}")
    values: list[ExecutionRequest] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                values.append(ExecutionRequest.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(
                    f"invalid provider request at line {line_number}"
                ) from exc
    if not values:
        raise ValueError("provider request manifest is empty")
    return tuple(values)


def _factory(value: str) -> StackV1Backend:
    module_name, separator, attribute = value.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("backend factory must use module.path:callable syntax")
    factory = getattr(importlib.import_module(module_name), attribute)
    return factory()


def _snapshot(path: Path | None) -> dict[str, Any]:
    source = path or (
        Path(os.environ["FIDMEM_CONFIG_SNAPSHOT"])
        if os.environ.get("FIDMEM_CONFIG_SNAPSHOT")
        else None
    )
    if source is None:
        raise ValueError("experiment config snapshot is required")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("experiment config snapshot must be an object")
    return value


def canary_gate_fields(
    validation: dict[str, Any], *, resume_passed: bool
) -> dict[str, bool]:
    return {
        "provenance_passed": bool(validation["production_provenance_valid"]),
        "cost_reconciliation_passed": bool(validation["cost_reconciliation_passed"]),
        "cache_isolation_passed": bool(
            validation["cross_question_cache_isolation_valid"]
        ),
        "resume_passed": resume_passed,
        "identity_consistency_passed": bool(
            validation["provider_model_device_identity_consistent"]
        ),
        "namespace_isolation_passed": bool(validation["production_namespace_isolated"]),
        "manifest_complete": (
            validation["missing_observation_count"] == 0
            and validation["unexpected_observation_question_count"] == 0
        ),
        "collision_free": validation["duplicate_collision_count"] == 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "run"))
    parser.add_argument("--config-snapshot", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = _snapshot(args.config_snapshot)
    inputs = config.get("inputs", {})
    if not isinstance(inputs, dict):
        raise ValueError("E03 inputs must be a mapping")
    stack_path = Path(str(inputs["stack_config"]))
    lock_path = Path(str(inputs["asset_lock"]))
    authority_path = Path(str(config["production_authority"]))
    selection_path = Path(str(inputs["selection_manifest"]))
    _stack, _lock, authority = check_stack_assets(
        stack_path=stack_path,
        lock_path=lock_path,
        authority_path=authority_path,
    )
    assert authority is not None
    requests = _load_requests(Path(str(inputs["provider_request_manifest"])))
    if {request.authority_sha256 for request in requests} != {
        authority.authority_sha256
    }:
        raise ValueError("provider requests do not bind the sealed Authority")
    factory_value = str(inputs.get("provider_backend_factory", ""))
    if "RESEARCH_OWNER_DECISION_REQUIRED" in factory_value or not factory_value:
        raise ValueError(
            "RESEARCH_OWNER_DECISION_REQUIRED: frozen Transformers backend factory"
        )
    backend = _factory(factory_value)
    run_dir = Path(os.environ.get("FIDMEM_RUN_DIR", "."))
    output = run_dir / "provider"
    resume = args.resume or (output / "raw").is_dir()
    result = execute_batch(
        requests,
        backend=backend,
        output_dir=output,
        resume=resume,
        check_only=args.command == "check",
    )
    if args.command == "run":
        results_root = run_dir / "results"
        imported = import_production_observations(
            result["provider_jsonl"],
            results_root,
            authority_path=authority_path,
            resume=(results_root / "CURRENT.json").is_file(),
            run_id=str(os.environ.get("FIDMEM_EXPERIMENT_ID", "E03")),
        )
        resume_probe = execute_batch(
            requests,
            backend=backend,
            output_dir=output,
            resume=True,
            check_only=False,
        )
        resume_import = import_production_observations(
            resume_probe["provider_jsonl"],
            results_root,
            authority_path=authority_path,
            resume=True,
            run_id=str(os.environ.get("FIDMEM_EXPERIMENT_ID", "E03")),
        )
        resume_passed = (
            resume_probe["generated"] == 0
            and resume_probe["resume_hits"] == len(requests)
            and resume_import["cache_misses"] == 0
        )
        validation = production_report_result(
            results_root,
            authority_path=authority_path,
            selection_manifest_path=selection_path,
        )
        validation.update(
            {
                "resume_validation": resume_passed,
                **canary_gate_fields(validation, resume_passed=resume_passed),
            }
        )
        report_path = results_root / "canary_validation.json"
        _atomic_bytes(report_path, canonical_json_bytes(validation))
        result = {
            **result,
            "import": imported,
            "resume_probe": resume_probe,
            "resume_import": resume_import,
            "validation": validation,
        }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
