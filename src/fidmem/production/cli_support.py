"""Production-only CLI operations kept separate from engineering orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from fidmem.production.authority import (
    AuthorityValidationError,
    seal_authority,
    validate_authority_draft,
)
from fidmem.production.authority_io import load_authority_draft
from fidmem.production.canary import validate_production_run
from fidmem.experiments.observation_import import import_production_observations


def authority_validate_result(
    draft_path: str | Path, *, project_root: str | Path
) -> tuple[int, dict[str, Any]]:
    try:
        draft = load_authority_draft(draft_path)
        report = validate_authority_draft(draft, project_root=project_root)
    except (OSError, ValueError, ValidationError) as exc:
        return 2, {
            "status": "blocked",
            "evidence_class": "engineering",
            "production_ready": False,
            "error_codes": ["authority_draft_invalid"],
            "issues": [{"code": "authority_draft_invalid", "message": str(exc)}],
        }
    result = {
        "status": "ready" if report.production_ready else "blocked",
        "evidence_class": "engineering",
        "production_ready": report.production_ready,
        "error_codes": list(report.error_codes),
        "issues": [item.model_dump(mode="json") for item in report.issues],
        "repository_identity": (
            report.repository_identity.model_dump(mode="json")
            if report.repository_identity
            else None
        ),
        "runtime_identity": (
            report.runtime_identity.model_dump(mode="json")
            if report.runtime_identity
            else None
        ),
    }
    return (0 if report.production_ready else 2), result


def authority_seal_result(
    draft_path: str | Path,
    output_path: str | Path,
    *,
    project_root: str | Path,
) -> tuple[int, dict[str, Any]]:
    try:
        draft = load_authority_draft(draft_path)
        sealed = seal_authority(
            draft, output_path=output_path, project_root=project_root
        )
    except (OSError, ValueError, ValidationError, AuthorityValidationError) as exc:
        return 2, {
            "status": "blocked",
            "evidence_class": "engineering",
            "production_ready": False,
            "reason": str(exc),
            "output_path": str(Path(output_path).resolve()),
        }
    return 0, {
        "status": "sealed",
        "evidence_class": "production",
        "production_ready": True,
        "authority_sha256": sealed.authority_sha256,
        "output_path": str(Path(output_path).resolve()),
    }


def production_import_result(args: Any, root: Path) -> dict[str, Any]:
    imported = import_production_observations(
        args.input_jsonl,
        root,
        authority_path=args.production_authority,
        resume=args.resume,
        run_id=args.run_id,
    )
    return {
        "command": "build-observations",
        "dry_run": False,
        "mode": "production_provider_import",
        "status": "completed",
        "evidence_class": "production",
        "input_jsonl": str(Path(args.input_jsonl).resolve()),
        "resume_validation": (
            True if args.resume and imported["cache_misses"] == 0 else "not_executed"
        ),
        **imported,
    }


def production_report_result(
    root: Path,
    *,
    authority_path: str | Path,
    selection_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    validation = validate_production_run(
        root,
        authority_path=authority_path,
        selection_manifest_path=selection_manifest_path,
    )
    return {
        **validation,
        "accuracy_primary_gate": False,
    }
