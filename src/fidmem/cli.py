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


COMMANDS = (
    "ingest",
    "build-gist",
    "build-observations",
    "build-oracle",
    "train-router",
    "run-dagger",
    "evaluate",
    "report",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
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
    root = Path(args.artifact_root or "artifacts") / "runs" / args.run_id
    root.mkdir(parents=True, exist_ok=True)
    return root


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=False, default="configs/base.yaml")
    parser.add_argument("--run-id", default="pilot")
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")


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


def _record_command(root: Path, args: argparse.Namespace, result: dict[str, Any]) -> None:
    state = _load_json(root / "state.json", {})
    state.update(
        {
            "run_id": args.run_id,
            "updated_at": _now(),
            "config": str(args.config),
            "config_sha256": _config_hash(args.config),
            "last_command": result,
        }
    )
    _atomic_json(root / "state.json", state)


def cmd_ingest(args: argparse.Namespace) -> dict[str, Any]:
    root = _run_root(args)
    video = Path(args.video) if args.video else None
    if args.dry_run:
        result = {"command": "ingest", "video": str(video) if video else None, "dry_run": True}
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
            {"event_id": f"e{index:04d}", "start_sec": item.start_sec, "end_sec": item.end_sec}
            for index, item in enumerate(segments)
        ],
        "created_at": _now(),
    }
    _atomic_json(root / "ingest.json", manifest)
    result = {"command": "ingest", "segments": len(segments), "manifest": str(root / "ingest.json")}
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
        estimate = _budget_estimate(root, observations=len((ingest or {}).get("segments", ())))
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
        stamps = tuple(min(max(0.0, value), max(0.0, duration - 0.001)) for value in stamps)
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
    result = {"command": "build-gist", "events": len(records), "artifact": str(root / "gist.json")}
    _record_command(root, args, result)
    print(json.dumps(result, indent=2))
    return result


def _generic_stage(args: argparse.Namespace, name: str) -> dict[str, Any]:
    root = _run_root(args)
    estimate = _budget_estimate(root, observations=int(getattr(args, "observations", 0) or 0))
    if args.dry_run:
        _check_budget(args.config, estimate)
    state = _load_json(root / "state.json", {})
    if name == "build-observations":
        requested = int(args.observations or len(_load_json(root / "ingest.json", {}).get("segments", ())))
        completed = int(state.get("completed_observations", 0))
        if not args.dry_run:
            newly_completed = max(0, requested - completed)
            completed = max(completed, requested)
            state["completed_observations"] = completed
            state["observation_cost"] = float(state.get("observation_cost", 0.0)) + newly_completed * 0.002
            _atomic_json(root / "observations.json", {"count": completed, "resumed": args.resume})
            _atomic_json(root / "state.json", state)
    result = {"command": name, "dry_run": args.dry_run, **estimate}
    _record_command(root, args, result)
    print(json.dumps(result, indent=2))
    return result


def cmd_report(args: argparse.Namespace) -> dict[str, Any]:
    root = _run_root(args)
    state = _load_json(root / "state.json", {})
    artifacts = sorted(path.name for path in root.iterdir() if path.is_file())
    report = {
        "schema_version": 1,
        "run_id": args.run_id,
        "config": state.get("config"),
        "config_sha256": state.get("config_sha256"),
        "model_versions": {"cli": "fidmem.cli.v1"},
        "costs": {"observation_cost": state.get("observation_cost", 0.0)},
        "results": state.get("last_command", {}),
        "failures": state.get("failures", []),
        "incomplete": state.get("incomplete", []),
        "budget_balance": _budget_estimate(root),
        "artifacts": artifacts,
        "generated_at": _now(),
    }
    _atomic_json(root / "report.json", report)
    markdown = "# fidmem run report\n\n" + "\n".join(
        f"- **{key}:** {json.dumps(value, ensure_ascii=False)}" for key, value in report.items()
    ) + "\n"
    (root / "report.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fidmem", description="Fidelity-graded long-video memory workflow")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in COMMANDS:
        child = sub.add_parser(command)
        _common_parser(child)
        if command == "ingest":
            child.add_argument("--video")
        elif command == "build-observations":
            child.add_argument("--observations", type=int, default=0)
        elif command in {"train-router", "run-dagger", "evaluate"}:
            child.add_argument("--max-steps", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "build-gist":
        _build_gist(args)
    elif args.command in {"build-observations", "build-oracle", "train-router", "run-dagger", "evaluate"}:
        _generic_stage(args, args.command)
    elif args.command == "report":
        cmd_report(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
