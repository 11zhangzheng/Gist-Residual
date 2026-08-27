"""Reproducible command-line orchestration for the fidelity-memory pipeline.

The CLI deliberately keeps orchestration state in small, inspectable JSON files.
Expensive model code remains in the domain modules; this layer is responsible for
configuration identity, resume semantics, budget checks, and reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fidmem.config import load_config
from fidmem.data.video import probe_video, sample_frames, segment_video
from fidmem.memory.gist import GistBuilder, GistEventInput
from fidmem.storage.cache import ContentAddressedCache
from fidmem.experiments.observation_import import import_observations
from fidmem.production.authority import canonical_json_bytes, load_sealed_authority
from fidmem.production.generation import GenerationStore
from fidmem.production.cli_support import (
    authority_seal_result,
    authority_validate_result,
    production_import_result,
    production_report_result,
)
from fidmem.production.provenance import (
    engineering_run_root,
    production_run_root,
)


COMMANDS = (
    "authority-validate",
    "authority-seal",
    "ingest",
    "build-gist",
    "build-observations",
    "build-oracle",
    "train-router",
    "run-dagger",
    "evaluate",
    "report",
)

UNWIRED_STAGES = frozenset(
    {
        "build-oracle",
        "train-router",
        "run-dagger",
        "evaluate",
    }
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _load_json(path: Path, default: object) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _run_root(args: argparse.Namespace) -> Path:
    artifact_root = Path(args.artifact_root or "artifacts")
    authority_path = getattr(args, "production_authority", None)
    if authority_path:
        authority = load_sealed_authority(authority_path)
        root = production_run_root(
            artifact_root, authority.authority_sha256, args.run_id
        )
    else:
        root = engineering_run_root(artifact_root, args.run_id)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=False, default="configs/base.yaml")
    parser.add_argument("--run-id", default="pilot")
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--production-authority")


def _production_execution_allowed(args: argparse.Namespace) -> bool:
    if args.command == "report":
        return True
    return (
        args.command == "build-observations"
        and bool(getattr(args, "input_jsonl", None))
        and not bool(args.dry_run)
    )



def _config_hash(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _budget_estimate(root: Path, *, observations: int = 0) -> dict[str, float | int]:
    state = _load_json(root / "state.json", {})
    completed = int(state.get("completed_observations", 0))
    expected = max(completed, observations)
    # Pilot coefficients are intentionally conservative and are recorded in the report.
    return {
        "expected_observations": expected,
        "cache_hits": completed,
        "estimated_a800_gpu_hours": round(expected * 0.002, 6),
        "estimated_v100_gpu_hours": round(expected * 0.0005, 6),
    }


def _check_budget(config_path: str, estimate: dict[str, float | int]) -> None:
    config = load_config(config_path)
    if float(estimate["estimated_a800_gpu_hours"]) > config.budget.a800_gpu_hours:
        raise SystemExit("dry-run exceeds configured A800 budget")
    if float(estimate["estimated_v100_gpu_hours"]) > config.budget.v100_gpu_hours:
        raise SystemExit("dry-run exceeds configured V100 budget")


def _production_store(
    root: Path, authority_path: str | Path
) -> GenerationStore:
    authority = load_sealed_authority(authority_path)
    return GenerationStore(root, authority.authority_sha256)


def _generation_artifacts(store: GenerationStore) -> dict[str, bytes]:
    active = store.current_path()
    marker = json.loads((active / "COMMITTED.json").read_text(encoding="utf-8"))
    return {
        str(name): (active / str(name)).read_bytes()
        for name in marker["artifact_sha256"]
    }


def _load_command_state(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    authority_path = getattr(args, "production_authority", None)
    if authority_path and (root / "CURRENT.json").is_file():
        active = _production_store(root, authority_path).current_path()
        return _load_json(active / "state.json", {})
    return _load_json(root / "state.json", {})


def _publish_production_updates(
    root: Path,
    authority_path: str | Path,
    updates: dict[str, bytes],
) -> None:
    store = _production_store(root, authority_path)
    artifacts = _generation_artifacts(store)
    artifacts.update(updates)
    store.publish(artifacts)



def _record_command(
    root: Path, args: argparse.Namespace, result: dict[str, Any]
) -> None:
    state = _load_command_state(root, args)
    recorded_at = _now()
    execution_status = str(
        result.get("status")
        or ("blocked" if result.get("blocked", False) else "completed")
    )
    command_history = list(state.get("command_history", []))
    command_history.append({**result, "recorded_at": recorded_at})
    state.update(
        {
            "run_id": args.run_id,
            "updated_at": recorded_at,
            "config": str(args.config),
            "config_sha256": _config_hash(args.config),
            "last_command": result,
            "command_history": command_history,
            "blocked": bool(result.get("blocked", False)),
            "status": execution_status,
            "reason": result.get("reason"),
            "execution_status": execution_status,
            "evidence_class": result.get("evidence_class", "engineering"),
            "authority_sha256": result.get("authority_sha256"),
        }
    )
    authority_path = getattr(args, "production_authority", None)
    if authority_path:
        _publish_production_updates(
            root, authority_path, {"state.json": canonical_json_bytes(state)}
        )
    else:
        _atomic_json(root / "state.json", state)


def cmd_ingest(args: argparse.Namespace) -> dict[str, Any]:
    root = _run_root(args)
    video = Path(args.video) if args.video else None
    if args.dry_run:
        result = {
            "command": "ingest",
            "video": str(video) if video else None,
            "dry_run": True,
        }
        _record_command(root, args, result)
        print(json.dumps(result, indent=2))
        return result
    if video is None or not video.is_file():
        raise SystemExit("ingest requires --video pointing to an existing file")
    probe = probe_video(video)
    segments = segment_video(video)
    manifest = {
        "video": str(video.resolve()),
        "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "duration_sec": probe.duration_sec,
        "width": probe.width,
        "height": probe.height,
        "fps": probe.fps,
        "segments": [
            {
                "event_id": f"e{index:04d}",
                "start_sec": item.start_sec,
                "end_sec": item.end_sec,
            }
            for index, item in enumerate(segments)
        ],
        "created_at": _now(),
    }
    _atomic_json(root / "ingest.json", manifest)
    result = {
        "command": "ingest",
        "segments": len(segments),
        "manifest": str(root / "ingest.json"),
    }
    _record_command(root, args, result)
    print(json.dumps(result, indent=2))
    return result


class _ByteTokenizer:
    identity = "fidmem.cli.byte-tokenizer.v1"

    def encode(self, text: str) -> tuple[int, ...]:
        return tuple(text.encode("utf-8"))

    def decode(self, tokens: tuple[int, ...]) -> str:
        return bytes(tokens).decode("utf-8", errors="ignore")


def _build_gist(args: argparse.Namespace) -> dict[str, Any]:
    root = _run_root(args)
    ingest = _load_json(root / "ingest.json", None)
    if args.dry_run:
        estimate = _budget_estimate(
            root, observations=len((ingest or {}).get("segments", ()))
        )
        _check_budget(args.config, estimate)
        result = {"command": "build-gist", "dry_run": True, **estimate}
        _record_command(root, args, result)
        print(json.dumps(result, indent=2))
        return result
    if not ingest:
        raise SystemExit("build-gist requires a completed ingest command")
    video = Path(ingest["video"])
    frame_dir = root / "frames"
    cache = ContentAddressedCache(root / "cache" / "gist")
    tokenizer = _ByteTokenizer()
    builder = GistBuilder(
        cache=cache,
        summarizer=lambda text, _budget, _tokenizer: text[:160] or "[no speech]",
        tokenizer=tokenizer,
        text_encoder=lambda text: (float(len(text) + 1), 1.0),
        visual_encoder=lambda _paths, _resolution: (1.0, 0.5),
        model_version="fidmem.cli.gist.v1",
        prompt="cli-deterministic-gist",
    )
    records = []
    duration = float(ingest["duration_sec"])
    for item in ingest["segments"]:
        event_id = item["event_id"]
        start = float(item["start_sec"])
        end = float(item["end_sec"])
        stamps = tuple(start + (end - start) * (i + 1) / 5 for i in range(4))
        stamps = tuple(
            min(max(0.0, value), max(0.0, duration - 0.001)) for value in stamps
        )
        paths = sample_frames(video, stamps, frame_dir / event_id)
        record = builder.build(
            GistEventInput(
                video_id=video.stem,
                event_id=event_id,
                start_sec=start,
                end_sec=end,
                asr_text="[no speech]",
                keyframe_paths=tuple(str(path) for path in paths),
                raw_video_uri=str(video),
                video_hash=ingest["video_sha256"],
                memory_version="cli-v1",
            )
        )
        records.append(record.model_dump(mode="json"))
    _atomic_json(root / "gist.json", {"events": records, "created_at": _now()})
    result = {
        "command": "build-gist",
        "events": len(records),
        "artifact": str(root / "gist.json"),
    }
    _record_command(root, args, result)
    print(json.dumps(result, indent=2))
    return result


def _import_observation_stage(
    args: argparse.Namespace,
    root: Path,
) -> dict[str, Any]:
    if args.production_authority:
        try:
            result = production_import_result(args, root)
        except (OSError, ValueError) as exc:
            result = {
                "command": "build-observations",
                "dry_run": False,
                "mode": "production_provider_import",
                "blocked": True,
                "status": "blocked",
                "evidence_class": "production",
                "authority_sha256": load_sealed_authority(
                    args.production_authority
                ).authority_sha256,
                "reason": str(exc),
                "input_jsonl": str(args.input_jsonl),
            }
        _record_command(root, args, result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result
    try:
        imported = import_observations(
            args.input_jsonl,
            root,
            resume=args.resume,
            run_id=args.run_id,
            config_path=args.config,
        )
    except (OSError, ValueError) as exc:
        result = {
            "command": "build-observations",
            "dry_run": False,
            "mode": "provider_import",
            "blocked": True,
            "status": "blocked",
            "reason": str(exc),
            "input_jsonl": str(args.input_jsonl),
        }
        _record_command(root, args, result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result

    result = {
        "command": "build-observations",
        "dry_run": False,
        "mode": "provider_import",
        "status": "completed",
        "input_jsonl": str(Path(args.input_jsonl).resolve()),
        **imported,
    }
    _record_command(root, args, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def _generic_stage(args: argparse.Namespace, name: str) -> dict[str, Any]:
    root = _run_root(args)
    if (
        name == "build-observations"
        and getattr(args, "input_jsonl", None)
        and not args.dry_run
    ):
        return _import_observation_stage(args, root)
    estimate = _budget_estimate(
        root, observations=int(getattr(args, "observations", 0) or 0)
    )
    if args.dry_run:
        _check_budget(args.config, estimate)
    if name in UNWIRED_STAGES and not args.dry_run:
        result = {
            "command": name,
            "dry_run": False,
            **estimate,
            "blocked": True,
            "status": "blocked",
            "reason": (
                f"{name} is not wired to a real implementation; "
                "refusing a non-dry-run no-op"
            ),
        }
        _record_command(root, args, result)
        print(json.dumps(result, indent=2))
        return result
    state = _load_json(root / "state.json", {})
    if name == "build-observations":
        requested = int(
            args.observations
            or len(_load_json(root / "ingest.json", {}).get("segments", ()))
        )
        completed = int(state.get("completed_observations", 0))
        if not args.dry_run:
            newly_completed = max(0, requested - completed)
            completed = max(completed, requested)
            state["completed_observations"] = completed
            state["observation_cost"] = (
                float(state.get("observation_cost", 0.0)) + newly_completed * 0.002
            )
            _atomic_json(
                root / "observations.json",
                {
                    "count": completed,
                    "resumed": args.resume,
                    "mode": "engineering_smoke",
                },
            )
            _atomic_json(root / "state.json", state)
    result = {"command": name, "dry_run": args.dry_run, **estimate}
    if name == "build-observations":
        result["mode"] = (
            "provider_import_dry_run"
            if getattr(args, "input_jsonl", None)
            else "engineering_smoke"
        )
    _record_command(root, args, result)
    print(json.dumps(result, indent=2))
    return result


def cmd_report(args: argparse.Namespace) -> dict[str, Any]:
    root = _run_root(args)
    state = _load_command_state(root, args)
    if args.production_authority:
        authority = load_sealed_authority(args.production_authority)
        try:
            validation = production_report_result(
                root, authority_path=args.production_authority
            )
            validation["resume_validation"] = any(
                entry.get("resume_validation") is True
                for entry in state.get("command_history", [])
                if entry.get("mode") == "production_provider_import"
            )
        except (OSError, ValueError) as exc:
            result = {
                "command": "report",
                "blocked": True,
                "status": "blocked",
                "evidence_class": "production",
                "authority_sha256": authority.authority_sha256,
                "reason": str(exc),
            }
            _record_command(root, args, result)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return result
        report = {
            "schema_version": 1,
            "run_id": args.run_id,
            "evidence_class": "production",
            "authority_sha256": authority.authority_sha256,
            **validation,
            "command_history": state.get("command_history", []),
            "generated_at": _now(),
        }
        markdown = (
            "# fidmem production validation report\n\n"
            + "\n".join(
                f"- **{key}:** {json.dumps(value, ensure_ascii=False)}"
                for key, value in report.items()
            )
            + "\n"
        )
        _publish_production_updates(
            root,
            args.production_authority,
            {"report.json": canonical_json_bytes(report), "report.md": markdown.encode("utf-8")},
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return report
    artifacts = sorted(path.name for path in root.iterdir() if path.is_file())
    measured_costs = _load_json(root / "summary.json", None)
    costs = (
        measured_costs
        if isinstance(measured_costs, dict)
        else {"observation_cost": state.get("observation_cost", 0.0)}
    )
    report = {
        "schema_version": 1,
        "run_id": args.run_id,
        "config": state.get("config"),
        "config_sha256": state.get("config_sha256"),
        "model_versions": {"cli": "fidmem.cli.v1"},
        "costs": costs,
        "results": state.get("last_command", {}),
        "command_history": state.get("command_history", []),
        "execution_status": state.get("execution_status", "unknown"),
        "failures": state.get("failures", []),
        "incomplete": state.get("incomplete", []),
        "budget_balance": _budget_estimate(root),
        "artifacts": artifacts,
        "generated_at": _now(),
    }
    _atomic_json(root / "report.json", report)
    markdown = (
        "# fidmem run report\n\n"
        + "\n".join(
            f"- **{key}:** {json.dumps(value, ensure_ascii=False)}"
            for key, value in report.items()
        )
        + "\n"
    )
    (root / "report.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def cmd_authority_validate(args: argparse.Namespace) -> dict[str, Any]:
    exit_code, result = authority_validate_result(
        args.draft, project_root=args.project_root
    )
    result["_exit_code"] = exit_code
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def cmd_authority_seal(args: argparse.Namespace) -> dict[str, Any]:
    exit_code, result = authority_seal_result(
        args.draft,
        args.output,
        project_root=args.project_root,
    )
    result["_exit_code"] = exit_code
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fidmem", description="Fidelity-graded long-video memory workflow"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for command in COMMANDS:
        child = sub.add_parser(command)
        if command in {"authority-validate", "authority-seal"}:
            child.add_argument("--draft", required=True)
            child.add_argument("--project-root", default=".")
            if command == "authority-seal":
                child.add_argument("--output", required=True)
            continue
        _common_parser(child)
        if command == "ingest":
            child.add_argument("--video")
        elif command == "build-observations":
            child.add_argument("--observations", type=int, default=0)
            child.add_argument("--input-jsonl")
        elif command in {"train-router", "run-dagger", "evaluate"}:
            child.add_argument("--max-steps", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "production_authority", None):
        try:
            load_sealed_authority(args.production_authority)
        except (OSError, ValueError) as exc:
            print(
                json.dumps(
                    {
                        "status": "blocked",
                        "blocked": True,
                        "evidence_class": "production",
                        "reason": str(exc),
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return 2
        if not _production_execution_allowed(args):
            print(
                json.dumps(
                    {
                        "status": "blocked",
                        "blocked": True,
                        "evidence_class": "production",
                        "reason": (
                            "production Authority is accepted only for a real "
                            "build-observations --input-jsonl execution or report"
                        ),
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return 2
    if args.command == "authority-validate":
        return int(cmd_authority_validate(args)["_exit_code"])
    if args.command == "authority-seal":
        return int(cmd_authority_seal(args)["_exit_code"])
    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "build-gist":
        _build_gist(args)
    elif args.command in {
        "build-observations",
        "build-oracle",
        "train-router",
        "run-dagger",
        "evaluate",
    }:
        result = _generic_stage(args, args.command)
        if result.get("blocked", False):
            return 2
    elif args.command == "report":
        result = cmd_report(args)
        if result.get("blocked", False):
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
