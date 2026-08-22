"""Deterministic, leakage-gated virtual long-video training manifests.

The builder deliberately stores source ranges plus global offsets.  It never
downloads, decodes, or concatenates video: a later consumer can materialize
the virtual timeline without changing the training provenance.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Callable, Iterable, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .leakage import LeakageAuditor, VideoAsset


MANIFEST_VERSION = "longroute-train/v1"


class LongRouteError(RuntimeError):
    """Base exception for a manifest that must not be used for training."""


class LongRouteDataError(LongRouteError):
    """Input cannot meet the non-negotiable synthetic-data contract."""


class LongRouteLeakageError(LongRouteError):
    """A train source overlaps a formal evaluation source."""


class SourceEvent(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    event_id: str
    start_sec: float = Field(ge=0)
    end_sec: float = Field(gt=0)
    label: str
    embedding: tuple[float, ...]
    attributes: dict[str, str | int | float] = {}

    @model_validator(mode="after")
    def range_and_embedding_are_valid(self) -> "SourceEvent":
        if self.end_sec <= self.start_sec:
            raise ValueError("event end_sec must exceed start_sec")
        if not self.label.strip():
            raise ValueError("event label must be non-empty")
        if not self.embedding or not all(math.isfinite(value) for value in self.embedding):
            raise ValueError("event embedding must be finite and non-empty")
        if not any(value != 0 for value in self.embedding):
            raise ValueError("event embedding must not be a zero vector")
        return self


class SourceQuestion(BaseModel):
    model_config = ConfigDict(frozen=True)

    question_id: str
    question: str
    options: tuple[str, ...]
    answer: str
    target_event_id: str
    answer_origin: Literal["source", "free_generated"] = "source"

    @model_validator(mode="after")
    def has_a_permitted_answer(self) -> "SourceQuestion":
        if not self.question.strip() or not self.question_id.strip():
            raise ValueError("question id and question must be non-empty")
        if not self.answer.strip() or self.answer not in self.options:
            raise ValueError("question must provide an answer included in options")
        if self.answer_origin == "free_generated":
            raise ValueError("free-generated answers are not permitted in LongRoute training")
        return self


class SourceVideo(BaseModel):
    model_config = ConfigDict(frozen=True)

    video_id: str
    path: Path
    split: Literal["train"]
    licensed: bool
    events: tuple[SourceEvent, ...]
    questions: tuple[SourceQuestion, ...]

    @model_validator(mode="after")
    def is_usable_train_source(self) -> "SourceVideo":
        if not self.video_id.strip():
            raise ValueError("video_id must be non-empty")
        if not self.licensed:
            raise ValueError("LongRoute source videos must be licensed")
        if not self.events:
            raise ValueError("source video must include events")
        ids = [event.event_id for event in self.events]
        if len(ids) != len(set(ids)):
            raise ValueError("event ids must be unique within a video")
        available = set(ids)
        if any(question.target_event_id not in available for question in self.questions):
            raise ValueError("each question target_event_id must belong to its video")
        return self


class TrainSplitManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    split: Literal["train"]
    videos: tuple[SourceVideo, ...]

    @model_validator(mode="after")
    def video_ids_are_unique(self) -> "TrainSplitManifest":
        ids = [video.video_id for video in self.videos]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("train manifests require non-empty unique video ids")
        return self


class LongRouteConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    output_dir: Path
    audit_size: int = Field(default=100, ge=1)
    min_distractors: int = Field(default=9, ge=9, le=19)
    max_distractors: int = Field(default=19, ge=9, le=19)
    multi_event_ratio: float = Field(default=0.25, ge=0.2, le=0.3)
    dev_fraction: float = Field(default=0.2, gt=0, lt=0.5)

    @model_validator(mode="after")
    def bounds_are_ordered(self) -> "LongRouteConfig":
        if self.min_distractors > self.max_distractors:
            raise ValueError("min_distractors must not exceed max_distractors")
        return self


class VirtualSegment(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    source_video_id: str
    event_id: str
    source_start_sec: float = Field(ge=0)
    source_end_sec: float = Field(gt=0)
    global_start_sec: float = Field(ge=0)
    global_end_sec: float = Field(gt=0)


class LongRouteExample(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    question_id: str
    split: Literal["train", "dev"]
    question: str
    options: tuple[str, ...]
    answer: str
    target_source_video_id: str
    target_event_id: str
    target_position: int = Field(ge=0)
    supporting_event_ids: tuple[str, ...]
    template: Literal["single_event", "before_after", "attribute_comparison", "count"]
    segments: tuple[VirtualSegment, ...]
    duration_sec: float = Field(ge=600, le=3600)
    audit_status: Literal["pending"] = "pending"


class DatasetManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    manifest_version: str
    seed: int
    source_manifest_hashes: dict[str, str]
    builder_config: dict[str, object]
    group_assignment: dict[str, Literal["train", "dev"]]
    split_statistics: dict[str, int]
    multi_event_ratio: float
    leakage_audit_uri: str
    examples: tuple[LongRouteExample, ...]
    asset_sha256s: dict[str, str] = {}
    generation_uri: str = ""

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


ContactSheetProvider = Callable[[LongRouteExample], str | Path]


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False, newline="") as handle:
            temporary = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary:
            Path(temporary).unlink(missing_ok=True)
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _canonical_source_manifest(manifest: TrainSplitManifest) -> TrainSplitManifest:
    videos = tuple(video.model_copy(update={"events": tuple(sorted(video.events, key=lambda event: event.event_id)), "questions": tuple(sorted(video.questions, key=lambda question: question.question_id))}) for video in sorted(manifest.videos, key=lambda video: video.video_id))
    return manifest.model_copy(update={"videos": videos})




def _seeded_index(seed: int, key: str, upper: int) -> int:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % upper


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise LongRouteDataError("embedding dimensions must match for nearest-neighbor distractors")
    if not all(math.isfinite(value) for value in (*left, *right)):
        raise LongRouteDataError("embeddings must be finite")
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(sum(value * value for value in right))
    if denominator == 0:
        raise LongRouteDataError("embeddings must not be zero vectors")
    return sum(a * b for a, b in zip(left, right, strict=True)) / denominator


class LongRouteBuilder:
    """Build a deterministic virtual-concatenation manifest from train-only data."""

    def __init__(
        self,
        train_manifests: Iterable[TrainSplitManifest],
        *,
        eval_assets: Iterable[VideoAsset],
        leakage_auditor: LeakageAuditor,
        config: LongRouteConfig,
        contact_sheet_provider: ContactSheetProvider | None = None,
    ) -> None:
        self.train_manifests = tuple(train_manifests)
        self.eval_assets = tuple(eval_assets)
        self.leakage_auditor = leakage_auditor
        self.config = config
        self.contact_sheet_provider = contact_sheet_provider

    def build(self, seed: int) -> DatasetManifest:
        videos = self._validated_videos()
        root = self.config.output_dir; root.mkdir(parents=True, exist_ok=True)
        hashes = {video.video_id: _sha256(video.path) for video in videos}
        attempt = hashlib.sha256(f"{seed}:{_canonical_json(hashes)}".encode()).hexdigest()[:20]
        payload: dict[str, object] = {"attempt_id": attempt, "seed": seed, "complete": False, "findings": []}
        try:
            assets = tuple(VideoAsset(video.video_id, video.path, (video.events[0].embedding,) * 8) for video in videos)
            report = self.leakage_auditor.audit(assets, self.eval_assets, require_coverage=True)
            payload.update({"parquet_uri": str(report.parquet_path.resolve()), "findings": [item.__dict__ for item in report.findings]})
            if report.findings:
                _atomic_write(root / "leakage-audit.json", _canonical_json(payload))
                raise LongRouteLeakageError("formal evaluation leakage detected; no training manifest was written")
            groups = self._groups(videos, seed); examples = self._examples(videos, groups, seed)
            if len(examples) < self.config.audit_size: raise LongRouteDataError(f"audit requires {self.config.audit_size} examples, found {len(examples)}")
            stage = root / ".staging" / attempt; stage.mkdir(parents=True, exist_ok=False)
            payload["complete"] = True; _atomic_write(stage / "leakage-audit.json", _canonical_json(payload))
            self._write_audit_bundle(examples, seed, stage / "audit")
            if hashes != {video.video_id: _sha256(video.path) for video in videos}: raise LongRouteDataError("source file changed after leakage audit")
            source_hashes = {m.name: hashlib.sha256(_canonical_json(_canonical_source_manifest(m).model_dump(mode="json")).encode()).hexdigest() for m in sorted(self.train_manifests, key=lambda m: m.name)}
            uri = f"generations/{attempt}"
            manifest = DatasetManifest(manifest_version=MANIFEST_VERSION, seed=seed, source_manifest_hashes=source_hashes, builder_config=self.config.model_dump(mode="json", exclude={"output_dir"}), group_assignment=groups, split_statistics={s: sum(e.split == s for e in examples) for s in ("train", "dev")}, multi_event_ratio=sum(e.template != "single_event" for e in examples) / len(examples), leakage_audit_uri=f"{uri}/leakage-audit.json", examples=tuple(examples), asset_sha256s=hashes, generation_uri=uri)
            _atomic_write(stage / "manifest.json", manifest.canonical_json())
            final = root / uri; final.parent.mkdir(parents=True, exist_ok=True); os.replace(stage, final)
            _atomic_write(root / "current-generation.json", _canonical_json({"generation": uri, "seed": seed}))
            _atomic_write(root / "last-attempt.json", _canonical_json({"attempt_id": attempt, "seed": seed, "status": "complete", "audit_uri": manifest.leakage_audit_uri}))
            return manifest
        except (LongRouteError, ValueError, OSError) as error:
            failed = root / f"failed-audit-{attempt}.json"; _atomic_write(failed, _canonical_json(payload | {"error": type(error).__name__, "message": str(error)}))
            _atomic_write(root / "last-attempt.json", _canonical_json({"attempt_id": attempt, "seed": seed, "status": "failed", "error": type(error).__name__, "audit_uri": str(failed)}))
            if isinstance(error, ValueError): raise LongRouteDataError(str(error)) from error
            raise

    def _validated_videos(self) -> tuple[SourceVideo, ...]:
        if not self.train_manifests:
            raise LongRouteDataError("at least one train split manifest is required")
        ids: set[str] = set(); questions: set[str] = set(); names: set[str] = set()
        videos: list[SourceVideo] = []
        for manifest in self.train_manifests:
            if manifest.name in names: raise LongRouteDataError("duplicate manifest identity/name")
            names.add(manifest.name)
            if manifest.split != "train":
                raise LongRouteDataError("LongRoute accepts train split manifests only")
            for video in manifest.videos:
                if video.video_id in ids:
                    raise LongRouteDataError("duplicate video_id across train manifests")
                if not video.path.is_file():
                    raise LongRouteDataError(f"source video path is unavailable: {video.path}")
                for question in video.questions:
                    if question.question_id in questions: raise LongRouteDataError("duplicate global question_id")
                    questions.add(question.question_id)
                ids.add(video.video_id)
                videos.append(video)
        return tuple(sorted(videos, key=lambda item: item.video_id))

    def _groups(self, videos: Sequence[SourceVideo], seed: int) -> dict[str, Literal["train", "dev"]]:
        # Assignment is per video, so all questions/events remain inseparable.
        count = max(1, round(len(videos) * self.config.dev_fraction))
        ranked = sorted(videos, key=lambda item: (hashlib.sha256(f"{seed}:{item.video_id}".encode()).hexdigest(), item.video_id))
        dev_ids = {item.video_id for item in ranked[:count]}
        return {video.video_id: ("dev" if video.video_id in dev_ids else "train") for video in videos}

    def _examples(self, videos: Sequence[SourceVideo], groups: dict[str, Literal["train", "dev"]], seed: int) -> list[LongRouteExample]:
        targets = [(video, question) for video in videos for question in video.questions]
        targets.sort(key=lambda item: (item[1].question_id, item[0].video_id))
        basic: list[LongRouteExample] = []
        for video, question in targets:
            split = groups[video.video_id]
            try:
                basic.append(self._single_example(video, question, split, videos, groups, seed))
            except LongRouteDataError:
                # A tiny group cannot make a legal route; it is omitted rather than leaking across groups.
                continue
        if not basic:
            raise LongRouteDataError("no split group can produce a 10 minutes route with 9-19 distractors")
        lower, upper = math.ceil(0.2 * len(basic)), math.floor(0.3 * len(basic))
        if lower > upper: raise LongRouteDataError("final sample count cannot satisfy the 20-30% multi-event ratio")
        multi_count = min(max(round(len(basic) * self.config.multi_event_ratio), lower), upper)
        ordered_ids = sorted(item.question_id for item in basic)
        random.Random(seed).shuffle(ordered_ids)
        multi_ids = set(ordered_ids[:multi_count])
        output: list[LongRouteExample] = []
        labels = {f"{video.video_id}:{event.event_id}": event.label for video in videos for event in video.events}
        for item in basic:
            if item.question_id not in multi_ids:
                output.append(item)
                continue
            upgraded = self._multi_event(item, labels)
            output.append(upgraded)
        return output

    def _single_example(self, video: SourceVideo, question: SourceQuestion, split: Literal["train", "dev"], videos: Sequence[SourceVideo], groups: dict[str, Literal["train", "dev"]], seed: int) -> LongRouteExample:
        target = next(event for event in video.events if event.event_id == question.target_event_id)
        candidates = [
            (candidate_video, event) for candidate_video in videos if groups[candidate_video.video_id] == split
            for event in candidate_video.events if not (candidate_video.video_id == video.video_id and event.event_id == target.event_id)
        ]
        ranked = sorted(candidates, key=lambda item: (-_cosine(target.embedding, item[1].embedding), item[1].event_id, item[0].video_id))
        chosen: list[tuple[SourceVideo, SourceEvent]] = []
        duration = target.end_sec - target.start_sec
        for candidate in ranked:
            if len(chosen) == self.config.max_distractors:
                break
            length = candidate[1].end_sec - candidate[1].start_sec
            if duration + length > 3600: continue
            chosen.append(candidate); duration += length
            if len(chosen) >= self.config.min_distractors and duration >= 600:
                break
        if len(chosen) < self.config.min_distractors or duration < 600:
            raise LongRouteDataError("cannot reach 10 minutes with no more than 19 distractors")
        position = _seeded_index(seed, question.question_id, len(chosen) + 1)
        entries = chosen[:]
        entries.insert(position, (video, target))
        segments: list[VirtualSegment] = []
        offset = 0.0
        for source, event in entries:
            length = event.end_sec - event.start_sec
            segments.append(VirtualSegment(source_video_id=source.video_id, event_id=event.event_id, source_start_sec=event.start_sec, source_end_sec=event.end_sec, global_start_sec=offset, global_end_sec=offset + length))
            offset += length
        target_id = f"{video.video_id}:{target.event_id}"
        return LongRouteExample(question_id=question.question_id, split=split, question=question.question, options=question.options, answer=question.answer, target_source_video_id=video.video_id, target_event_id=target_id, target_position=position, supporting_event_ids=(target_id,), template="single_event", segments=tuple(segments), duration_sec=offset)

    def _multi_event(self, item: LongRouteExample, labels: dict[str, str]) -> LongRouteExample:
        pos = item.target_position; left_i, right_i = (pos, pos + 1) if pos + 1 < len(item.segments) else (pos - 1, pos)
        left, right = item.segments[left_i], item.segments[right_i]
        left_id, right_id = f"{left.source_video_id}:{left.event_id}", f"{right.source_video_id}:{right.event_id}"
        if right.global_start_sec != left.global_end_sec: raise LongRouteDataError("multi-event supporting segments must be adjacent")
        answer = labels[right_id]
        return item.model_copy(update={"question": f"Which event happens immediately after {labels[left_id]}?", "options": (answer, "None of these"), "answer": answer, "supporting_event_ids": (left_id, right_id), "template": "before_after"})

    def _write_audit_bundle(self, examples: Sequence[LongRouteExample], seed: int, bundle: Path) -> None:
        if self.contact_sheet_provider is None:
            raise LongRouteDataError("a contact_sheet_provider is required for the human audit package")
        selected = sorted(examples, key=lambda item: item.question_id)[: self.config.audit_size]
        records = []
        for example in selected:
            contact = str(self.contact_sheet_provider(example))
            local = Path(contact)
            remote = contact.startswith("https://") or contact.startswith("s3://")
            if not contact or (not remote and (not local.is_file() or not os.access(local, os.R_OK) or local.stat().st_size == 0)):
                raise LongRouteDataError("contact sheet provider returned an empty path")
            records.append({"question_id": example.question_id, "question": example.question, "options": list(example.options), "answer": example.answer, "source_events": [segment.model_dump(mode="json") for segment in example.segments], "global_offsets": [[segment.global_start_sec, segment.global_end_sec] for segment in example.segments], "contact_sheet": contact, "seed": seed})
        _atomic_write(bundle / "samples.json", _canonical_json(records))
        _atomic_write(bundle / "samples.jsonl", "".join(_canonical_json(record) + "\n" for record in records))
        import io
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=["question_id", "valid", "invalid", "reason"], lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({"question_id": record["question_id"], "valid": "", "invalid": "", "reason": ""})
        _atomic_write(bundle / "review.csv", buffer.getvalue())
