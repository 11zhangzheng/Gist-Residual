"""Manual Video-MME-v2/E01/E02 setup commands; never performs model inference."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from fidmem.assets.authority_draft import build_authority_draft, write_authority_draft
from fidmem.assets.videomme_v2 import (
    DATASET_ID,
    FROZEN_REVISION,
    METADATA_FILES,
    prepare_e01 as prepare_videomme_e01,
    prepare_videos,
    verify_metadata,
    verify_raw_videos,
    write_dataset_preparation,
)
from fidmem.assets.resolver import (
    AssetState,
    load_asset_lock,
    storage_roots,
    verify_entry,
)
from fidmem.production.authority import (
    ProductionAuthorityDraft,
    canonical_json_bytes,
    seal_authority,
    validate_authority_draft,
)


STACK_LOCK = Path("configs/experiment_stacks/gist_residual_v1.assets.lock.json")
SPLIT_POLICY = Path("configs/experiment_stacks/videomme_v2_pilot_split_policy.yaml")
PROMPTS = Path("configs/experiment_stacks/gist_residual_v1.prompts.yaml")
OBSERVATION_CONFIGS = Path(
    "configs/experiment_stacks/gist_residual_v1.observation_configs.yaml"
)


def _atomic_payload(path: Path, payload: object) -> None:
    from pydantic import BaseModel

    if isinstance(payload, BaseModel):
        payload = payload.model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(canonical_json_bytes(payload))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _metadata_asset_from_lock(lock_path: Path):
    lock = load_asset_lock(lock_path)
    asset_id = lock.logical_roles["source_dataset"]
    entry = lock.physical_assets[asset_id]
    if entry.repo_id != DATASET_ID:
        raise ValueError("stack source_dataset is not the approved Video-MME-v2 repository")
    if entry.repo_type != "dataset":
        raise ValueError("Video-MME-v2 metadata asset is not a dataset snapshot")
    if entry.immutable_revision != FROZEN_REVISION:
        raise ValueError("Video-MME-v2 metadata asset differs from frozen revision")
    if entry.expected_files != tuple(sorted(METADATA_FILES)):
        raise ValueError("Video-MME-v2 metadata asset file manifest differs")
    if entry.state is not AssetState.VERIFIED:
        raise ValueError("Video-MME-v2 metadata asset is not VERIFIED")
    if entry.local_snapshot_path is None:
        raise ValueError("Video-MME-v2 metadata asset lacks a local snapshot")
    verified = verify_entry(entry)
    if verified.local_snapshot_sha256 != entry.local_snapshot_sha256:
        raise ValueError("Video-MME-v2 metadata snapshot differs from asset lock")
    snapshot_root = Path(entry.local_snapshot_path)
    return (
        verify_metadata(snapshot_root, immutable_revision=FROZEN_REVISION),
        snapshot_root,
    )


def _metadata_from_lock(lock_path: Path):
    return _metadata_asset_from_lock(lock_path)[0]


def _config_snapshot() -> dict[str, Any]:
    value = os.environ.get("FIDMEM_CONFIG_SNAPSHOT")
    if not value:
        raise ValueError("FIDMEM_CONFIG_SNAPSHOT is required")
    payload = json.loads(Path(value).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("experiment config snapshot must be an object")
    return payload


def _resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("metadata", "videos", "manifests", "e01", "authority-draft", "e02"):
        command = subparsers.add_parser(name)
        command.add_argument("--check", action="store_true")
        command.add_argument("--project-root", default=".")
        command.add_argument("--lock", default=str(STACK_LOCK))
        command.add_argument("--output")
        if name == "videos":
            command.add_argument("--resume", action="store_true")
            command.add_argument("--verify-only", action="store_true")
            command.add_argument("--scope", choices=("pilot", "full"), default="pilot")
    args = parser.parse_args(argv)
    root = Path(args.project_root).resolve()
    lock_path = _resolve_project_path(root, args.lock)
    roots = storage_roots()
    if args.command == "metadata":
        parsed, _metadata_root = _metadata_asset_from_lock(lock_path)
        payload = parsed.report.model_dump(mode="json")
        if args.output and not args.check:
            atomic_write_model(Path(args.output), parsed.report)
    elif args.command == "videos":
        parsed, metadata_root = _metadata_asset_from_lock(lock_path)
        raw_value = os.environ.get("FIDMEM_VIDEOMME_V2_RAW_ROOT")
        cache_value = os.environ.get("FIDMEM_CACHE_ROOT")
        if not raw_value or not cache_value:
            raise ValueError("FIDMEM_VIDEOMME_V2_RAW_ROOT and FIDMEM_CACHE_ROOT are required")
        prepared = prepare_videos(
            parsed, Path(raw_value), Path(cache_value), scope=args.scope,
            subtitle_zip=metadata_root / "subtitle.zip",
            check=args.check, resume=args.resume, verify_only=args.verify_only,
        )
        payload = prepared.model_dump(mode="json")
        if args.output and not args.check:
            _atomic_payload(Path(args.output) / "media_preparation.json", payload)
    elif args.command == "manifests":
        parsed, metadata_root = _metadata_asset_from_lock(lock_path)
        raw_value = os.environ.get("FIDMEM_VIDEOMME_V2_RAW_ROOT")
        cache_value = os.environ.get("FIDMEM_CACHE_ROOT")
        if not raw_value or not cache_value:
            raise ValueError("FIDMEM_VIDEOMME_V2_RAW_ROOT and FIDMEM_CACHE_ROOT are required")
        prepared = prepare_videos(
            parsed, Path(raw_value), Path(cache_value), scope="pilot",
            subtitle_zip=metadata_root / "subtitle.zip",
            check=False, resume=False, verify_only=True,
        )
        if prepared.selection is None:
            raise ValueError("pilot preparation lacks subset selection")
        report = verify_raw_videos(prepared.selection.selected_video_ids, Path(raw_value))
        output = (
            Path(args.output)
            if args.output
            else Path(os.environ["FIDMEM_VIDEOMME_V2_PREPARATION_ROOT"])
        )
        payload = write_dataset_preparation(
            parsed,
            report,
            prepared.archive_index,
            prepared.selection,
            output,
            check=args.check,
        )
    elif args.command == "e01":
        preparation = os.environ.get("FIDMEM_VIDEOMME_V2_PREPARATION_ROOT")
        human_value = os.environ.get("FIDMEM_VIDEOMME_V2_HUMAN_AUDIT_RESULT")
        if not preparation or not human_value:
            raise ValueError("FIDMEM_VIDEOMME_V2_PREPARATION_ROOT and FIDMEM_VIDEOMME_V2_HUMAN_AUDIT_RESULT are required")
        output = Path(args.output) if args.output else None
        if output is None and not args.check:
            output = Path(os.environ["FIDMEM_RUN_DIR"]) / "results"
        payload = prepare_videomme_e01(
            Path(preparation), Path(human_value), output_dir=output, check=args.check
        )
    elif args.command == "authority-draft":
        manifests_value = os.environ.get("FIDMEM_E01_RESULTS_ROOT")
        if not manifests_value:
            raise ValueError("FIDMEM_E01_RESULTS_ROOT is required")
        output = (
            Path(args.output)
            if args.output
            else roots["FIDMEM_ARTIFACT_ROOT"]
            / "authority"
            / "ProductionAuthorityDraft.json"
        )
        draft, unresolved = build_authority_draft(
            project_root=root,
            asset_lock_path=lock_path,
            manifests_root=manifests_value,
            split_policy_path=Path(manifests_value) / "split_policy.json",
            prompt_config_path=root / PROMPTS,
            observation_config_path=root / OBSERVATION_CONFIGS,
            evidence_root=roots["FIDMEM_ARTIFACT_ROOT"]
            / "authority"
            / "asset-evidence",
        )
        payload = {
            "status": "CHECK_PASSED" if args.check else "DRAFT_WRITTEN",
            "production_ready": False,
            "runtime_seal": "PENDING",
            "unresolved": list(unresolved),
            "output": str(output),
        }
        if not args.check:
            write_authority_draft(output, draft)
    else:
        snapshot = _config_snapshot()
        draft_path = _resolve_project_path(
            root, str(snapshot["inputs"]["authority_draft"])
        )
        draft = ProductionAuthorityDraft.model_validate_json(
            draft_path.read_text(encoding="utf-8")
        )
        if args.check:
            report = validate_authority_draft(draft, project_root=root)
            if not report.production_ready:
                raise ValueError(
                    "Authority Draft is not sealable: " + ", ".join(report.error_codes)
                )
            payload = report.model_dump(mode="json")
        else:
            run_dir = Path(os.environ["FIDMEM_RUN_DIR"])
            sealed = seal_authority(
                draft,
                output_path=run_dir / "results" / "PRODUCTION_AUTHORITY.json",
                project_root=root,
            )
            payload = sealed.model_dump(mode="json")
            _atomic_payload(
                run_dir / "results" / "authority_gate_result.json",
                {
                    "sealed": True,
                    "authority_valid": True,
                    "runtime_matches": True,
                    "authority_sha256": sealed.authority_sha256,
                },
            )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
