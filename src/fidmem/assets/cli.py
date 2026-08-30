"""Manual CLI for Experiment Stack asset resolve/download/verify/lock steps."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Callable

from fidmem.assets.resolver import (
    AssetLock,
    AssetLockEntry,
    AssetState,
    assert_verified_lock,
    check_storage_roots,
    huggingface_info_loader,
    load_asset_lock,
    reconcile_lock,
    resolve_entry,
    snapshot_download_entry,
    storage_roots,
    utc_now,
    verify_entry,
    write_asset_lock,
)
from fidmem.assets.stack import load_experiment_stack


def _hub_version() -> str:
    try:
        return importlib.metadata.version("huggingface-hub")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("huggingface_hub is not installed") from exc


def _selected(lock: AssetLock, asset_kind: str) -> list[str]:
    return [
        asset_id
        for asset_id, entry in sorted(lock.physical_assets.items())
        if asset_kind == "all"
        or (asset_kind == "models" and entry.repo_type == "model")
        or (asset_kind == "dataset" and entry.repo_type == "dataset")
    ]


def _replace_entries(
    lock: AssetLock,
    entries: dict[str, AssetLockEntry],
    *,
    hub_version: str | None = None,
) -> AssetLock:
    return AssetLock.create(
        stack_id=lock.stack_id,
        generated_at=utc_now(),
        logical_roles=dict(lock.logical_roles),
        physical_assets=entries,
        huggingface_hub_version=hub_version or lock.huggingface_hub_version,
    )


def _failed(entry: AssetLockEntry, exc: Exception) -> AssetLockEntry:
    return entry.model_copy(
        update={
            "state": AssetState.FAILED,
            "failure": f"{type(exc).__name__}: {exc}",
            "verified_at": None,
        }
    )


def _reconciliation_asset_ids(
    stack, lock: AssetLock
) -> tuple[list[str], list[str]]:
    previous_identities = {
        (
            entry.repo_id,
            entry.repo_type,
            entry.immutable_revision,
            entry.backend,
            entry.dtype,
        )
        for entry in lock.physical_assets.values()
    }
    preserved, reset = [], []
    for asset_id, asset in sorted(stack.physical_assets.items()):
        identity = (
            asset.repo_id,
            asset.repo_type,
            asset.immutable_revision,
            asset.backend,
            asset.dtype,
        )
        (preserved if identity in previous_identities else reset).append(asset_id)
    return preserved, reset


def operate(
    action: str,
    *,
    stack_path: Path,
    lock_path: Path,
    asset_kind: str,
    check: bool,
    dry_run: bool,
    resume: bool,
    verify_only: bool,
    info_loader: Callable[
        [str, str], tuple[str, tuple[str, ...]]
    ] = huggingface_info_loader,
    downloader: Callable[..., AssetLockEntry] = snapshot_download_entry,
) -> dict[str, object]:
    stack = load_experiment_stack(stack_path)
    lock = load_asset_lock(lock_path)
    effective_action = "verify" if verify_only else action
    if (
        effective_action != "reconcile"
        and (lock.stack_id != stack.stack_id or lock.logical_roles != stack.logical_roles)
    ):
        raise ValueError("stack config and asset lock identities differ")
    if effective_action == "reconcile":
        reconciled = reconcile_lock(stack, lock)
        preserved_asset_ids, reset_asset_ids = _reconciliation_asset_ids(stack, lock)
        if dry_run:
            return {
                "status": "DRY_RUN",
                "action": effective_action,
                "preserved_asset_ids": preserved_asset_ids,
                "reset_asset_ids": reset_asset_ids,
            }
        if check:
            return {
                "status": "CHECK_PASSED",
                "action": effective_action,
                "preserved_asset_ids": preserved_asset_ids,
                "reset_asset_ids": reset_asset_ids,
            }
        write_asset_lock(lock_path, reconciled)
        return {
            "status": "COMPLETED",
            "action": effective_action,
            "lock_sha256": reconciled.lock_sha256,
            "preserved_asset_ids": preserved_asset_ids,
            "reset_asset_ids": reset_asset_ids,
        }
    selected = _selected(lock, asset_kind)
    if not selected:
        raise ValueError("asset selection is empty")
    roots = storage_roots() if effective_action in {"download", "verify"} else None
    storage_report = check_storage_roots(roots) if roots is not None else {}
    plan = [
        {
            "asset_id": asset_id,
            "repo_id": lock.physical_assets[asset_id].repo_id,
            "revision": lock.physical_assets[asset_id].immutable_revision,
            "state": lock.physical_assets[asset_id].state.value,
            "destination": (
                str(
                    (
                        roots["FIDMEM_DATA_ROOT"]
                        if lock.physical_assets[asset_id].repo_type == "dataset"
                        else roots["FIDMEM_MODEL_ROOT"]
                    )
                    / asset_id
                )
                if roots is not None
                else None
            ),
            "known_file_count": len(lock.physical_assets[asset_id].expected_files),
        }
        for asset_id in selected
    ]
    if dry_run:
        return {"status": "DRY_RUN", "action": effective_action, "assets": plan}
    if check:
        hub_version = _hub_version()
        checked: list[dict[str, object]] = []
        for asset_id in selected:
            entry = lock.physical_assets[asset_id]
            if effective_action == "resolve":
                revision, files = info_loader(entry.repo_id, entry.repo_type)
                resolved = resolve_entry(
                    entry,
                    info_loader=lambda _repo, _type: (revision, files),
                    required_files=stack.physical_assets[asset_id].include_files,
                )
                checked.append(
                    {
                        "asset_id": asset_id,
                        "remote_revision": resolved.immutable_revision,
                        "known_file_count": len(resolved.expected_files),
                    }
                )
            elif effective_action == "download":
                if entry.immutable_revision is None:
                    raise ValueError(f"asset {asset_id} is unresolved")
                if not entry.expected_files:
                    raise ValueError(
                        f"asset {asset_id} lacks a resolved remote file manifest"
                    )
                checked.append({"asset_id": asset_id, "download_invoked": False})
            else:
                verify_entry(entry)
                checked.append({"asset_id": asset_id, "verified_from_disk": True})
        return {
            "status": "CHECK_PASSED",
            "action": effective_action,
            "huggingface_hub_version": hub_version,
            "storage": storage_report,
            "assets": checked,
        }

    entries = dict(lock.physical_assets)
    hub_version = _hub_version()
    if effective_action == "resolve":
        for asset_id in selected:
            try:
                entries[asset_id] = resolve_entry(
                    entries[asset_id],
                    info_loader=info_loader,
                    required_files=stack.physical_assets[asset_id].include_files,
                )
            except Exception as exc:
                entries[asset_id] = _failed(entries[asset_id], exc)
                write_asset_lock(
                    lock_path,
                    _replace_entries(lock, entries, hub_version=hub_version),
                )
                raise
    elif effective_action == "download":
        assert roots is not None
        for asset_id in selected:
            entry = entries[asset_id]
            if resume and entry.state is AssetState.VERIFIED:
                observed = verify_entry(entry)
                if observed.local_snapshot_sha256 != entry.local_snapshot_sha256:
                    raise ValueError(f"asset {asset_id} changed after verification")
                continue
            if entry.immutable_revision is None:
                raise ValueError(f"asset {asset_id} is unresolved")
            if not entry.expected_files:
                raise ValueError(
                    f"asset {asset_id} lacks a resolved remote file manifest"
                )
            destination_root = (
                roots["FIDMEM_DATA_ROOT"]
                if entry.repo_type == "dataset"
                else roots["FIDMEM_MODEL_ROOT"]
            )
            entries[asset_id] = entry.model_copy(
                update={"state": AssetState.DOWNLOADING}
            )
            write_asset_lock(
                lock_path,
                _replace_entries(lock, entries, hub_version=hub_version),
            )
            try:
                entries[asset_id] = downloader(
                    entry,
                    destination=destination_root / asset_id,
                    cache_root=roots["FIDMEM_CACHE_ROOT"],
                )
            except Exception as exc:
                entries[asset_id] = _failed(entry, exc)
                write_asset_lock(
                    lock_path,
                    _replace_entries(lock, entries, hub_version=hub_version),
                )
                raise
    elif effective_action == "verify":
        for asset_id in selected:
            try:
                entries[asset_id] = verify_entry(entries[asset_id])
            except Exception as exc:
                entries[asset_id] = _failed(entries[asset_id], exc)
                write_asset_lock(
                    lock_path,
                    _replace_entries(lock, entries, hub_version=hub_version),
                )
                raise
    elif effective_action == "lock":
        assert_verified_lock(lock, reverify=True)
    else:
        raise ValueError(f"unsupported asset action: {effective_action}")
    updated = _replace_entries(lock, entries, hub_version=hub_version)
    write_asset_lock(lock_path, updated)
    return {
        "status": "COMPLETED",
        "action": effective_action,
        "lock_sha256": updated.lock_sha256,
        "assets": [
            {
                "asset_id": asset_id,
                "state": updated.physical_assets[asset_id].state.value,
            }
            for asset_id in selected
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("resolve", "download", "verify", "lock", "reconcile")
    )
    parser.add_argument(
        "--stack", default="configs/experiment_stacks/gist_residual_v1.yaml"
    )
    parser.add_argument(
        "--lock",
        default="configs/experiment_stacks/gist_residual_v1.assets.lock.json",
    )
    parser.add_argument(
        "--asset-kind", choices=("all", "models", "dataset"), default="all"
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    payload = operate(
        args.action,
        stack_path=Path(args.stack),
        lock_path=Path(args.lock),
        asset_kind=args.asset_kind,
        check=args.check,
        dry_run=args.dry_run,
        resume=args.resume,
        verify_only=args.verify_only,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
