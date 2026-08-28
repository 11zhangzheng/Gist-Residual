"""Cheap built-in tasks used by the execution pack; no model inference."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import torch

from fidmem.experiments.execution_pack import probe_gpus
from fidmem.production.authority import canonical_json_bytes, probe_repository


def environment_snapshot(project_root: Path) -> dict[str, object]:
    repository = probe_repository(project_root)
    return {
        "schema_version": 1,
        "environment_valid": True,
        "source_identity_recorded": True,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpus": probe_gpus(),
        "repository": repository.model_dump(mode="json"),
        "inference_backend": os.environ.get("FIDMEM_INFERENCE_BACKEND"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cheap execution-pack utility tasks")
    subparsers = parser.add_subparsers(dest="command", required=True)
    environment = subparsers.add_parser(
        "environment", help="record metadata without inference"
    )
    environment.add_argument("--project-root", default=".")
    args = parser.parse_args(argv)
    if args.command == "environment":
        run_value = os.environ.get("FIDMEM_RUN_DIR")
        if not run_value:
            raise RuntimeError("FIDMEM_RUN_DIR is required")
        payload = environment_snapshot(Path(args.project_root).resolve())
        destination = Path(run_value) / "results" / "environment.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(canonical_json_bytes(payload))
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
