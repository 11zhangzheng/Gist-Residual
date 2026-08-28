"""Manual LongTVQA/E01/E02 setup commands; never performs model inference."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from fidmem.assets.authority_draft import build_authority_draft, write_authority_draft
from fidmem.assets.longtvqa import (
    DATASET_ID,
    atomic_write_model,
    build_human_audit_manifest,
    build_manifests,
    validate_human_audit_result,
    verify_metadata,
    verify_raw_videos,
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
SPLIT_POLICY = Path("configs/experiment_stacks/longtvqa_split_policy.yaml")
PROMPTS = Path("configs/experiment_stacks/gist_residual_v1.prompts.yaml")
OBSERVATION_CONFIGS = Path(
    "configs/experiment_stacks/gist_residual_v1.observation_configs.yaml"
)


def _atomic_payload(path: Path, payload: object) -> None:
    from pydantic import BaseModel

    if isinstance(payload, BaseModel):
        atomic_write_model(path, payload)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(canonical_json_bytes(payload))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _metadata_from_lock(lock_path: Path):
    lock = load_asset_lock(lock_path)
    asset_id = lock.logical_roles["source_dataset"]
    entry = lock.physical_assets[asset_id]
    if entry.repo_id != DATASET_ID:
        raise ValueError("stack source_dataset is not the approved LongTVQA repository")
    if entry.state is not AssetState.VERIFIED:
        raise ValueError("LongTVQA metadata asset is not VERIFIED")
    verified = verify_entry(entry)
    if verified.local_snapshot_sha256 != entry.local_snapshot_sha256:
        raise ValueError("LongTVQA metadata snapshot differs from asset lock")
    return verify_metadata(
        str(entry.local_snapshot_path), immutable_revision=str(entry.immutable_revision)
    )


def prepare_e01(
    *,
    lock_path: Path,
    split_policy_path: Path,
    video_root: Path,
    human_result: Path,
    output_dir: Path | None,
    check: bool,
) -> dict[str, Any]:
    metadata = _metadata_from_lock(lock_path)
    if metadata.report.qa_unconstructible_count:
        raise ValueError(
            "LongTVQA contains QA rows that cannot construct approved actions/options"
        )
    videos = verify_raw_videos(metadata, video_root)
    if videos.status != "PASS":
        raise ValueError("LongTVQA raw-video Source Gate failed")
    audit = build_human_audit_manifest(metadata, seed="longtvqa-human-audit-v1")
    validate_human_audit_result(audit, human_result)
    video_manifest, question_manifest, dataset_manifest, canary, oracle = (
        build_manifests(
            metadata,
            videos,
            split_policy_path=split_policy_path,
        )
    )
    payload = {
        "status": "CHECK_PASSED" if check else "COMPLETED",
        "dataset": DATASET_ID,
        "dataset_revision": metadata.report.immutable_revision,
        "metadata_sha256": metadata.report.metadata_sha256,
        "video_manifest_sha256": video_manifest.manifest_sha256,
        "question_manifest_sha256": question_manifest.manifest_sha256,
        "canary_selection_sha256": canary.selection_sha256,
        "oracle_selection_sha256": oracle.selection_sha256,
        "human_audit_manifest_sha256": audit.manifest_sha256,
        "source_gate": "PASS",
        "manifests_complete": True,
        "video_disjoint": True,
        "hashes_valid": True,
    }
    if not check:
        if output_dir is None:
            raise ValueError("output_dir is required outside --check")
        atomic_write_model(output_dir / "metadata_verification.json", metadata.report)
        atomic_write_model(output_dir / "raw_video_verification.json", videos)
        atomic_write_model(output_dir / "human_audit_manifest.json", audit)
        atomic_write_model(output_dir / "video_manifest.json", video_manifest)
        atomic_write_model(output_dir / "question_manifest.json", question_manifest)
        atomic_write_model(output_dir / "dataset_manifest.json", dataset_manifest)
        atomic_write_model(output_dir / "canary_selection_manifest.json", canary)
        atomic_write_model(output_dir / "oracle_selection_manifest.json", oracle)
        _atomic_payload(output_dir / "source_gate.json", payload)
    return payload


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
    args = parser.parse_args(argv)
    root = Path(args.project_root).resolve()
    lock_path = _resolve_project_path(root, args.lock)
    roots = storage_roots()
    if args.command == "metadata":
        parsed = _metadata_from_lock(lock_path)
        payload = parsed.report.model_dump(mode="json")
        if args.output and not args.check:
            atomic_write_model(Path(args.output), parsed.report)
    elif args.command == "videos":
        parsed = _metadata_from_lock(lock_path)
        video_value = os.environ.get("FIDMEM_LONGTVQA_VIDEO_ROOT")
        if not video_value:
            raise ValueError("FIDMEM_LONGTVQA_VIDEO_ROOT is required")
        report = verify_raw_videos(parsed, video_value)
        audit = build_human_audit_manifest(parsed, seed="longtvqa-human-audit-v1")
        payload = {
            "raw_video": report.model_dump(mode="json"),
            "human_audit": audit.model_dump(mode="json"),
        }
        if args.output and not args.check:
            destination = Path(args.output)
            atomic_write_model(destination / "raw_video_verification.json", report)
            atomic_write_model(destination / "human_audit_manifest.json", audit)
    elif args.command in {"manifests", "e01"}:
        video_value = os.environ.get("FIDMEM_LONGTVQA_VIDEO_ROOT")
        human_value = os.environ.get("FIDMEM_LONGTVQA_HUMAN_AUDIT_RESULT")
        if not video_value or not human_value:
            raise ValueError(
                "FIDMEM_LONGTVQA_VIDEO_ROOT and FIDMEM_LONGTVQA_HUMAN_AUDIT_RESULT are required"
            )
        output = (
            Path(args.output)
            if args.output
            else Path(os.environ["FIDMEM_RUN_DIR"]) / "results"
        )
        payload = prepare_e01(
            lock_path=lock_path,
            split_policy_path=root / SPLIT_POLICY,
            video_root=Path(video_value),
            human_result=Path(human_value),
            output_dir=output,
            check=args.check,
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
            split_policy_path=root / SPLIT_POLICY,
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
