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
import random
from pathlib import Path
from typing import Callable, Iterable, Literal, Mapping, Protocol, Sequence
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .leakage import LeakageAuditor, VideoAsset
from .publication import PublicationBackend, PublicationTransaction


MANIFEST_VERSION = "longroute-train/v2"
BUILDER_VERSION = "fidmem-longroute/2"


class LongRouteError(RuntimeError):
    """Base exception for a manifest that must not be used for training."""


class LongRouteDataError(LongRouteError):
    """Input cannot meet the non-negotiable synthetic-data contract."""


class LongRouteLeakageError(LongRouteError):
    """A train source overlaps a formal evaluation source."""


class ContactSheetValidator(Protocol):
    """Validate and canonicalize one generated contact-sheet location."""

    def validate(self, uri: str | Path) -> str:
        ...


RemoteProbe = Callable[[str], None]


def _probe_http_contact_sheet(uri: str) -> None:
    request = Request(uri, method="HEAD")
    with urlopen(request, timeout=10) as response:
        status = getattr(response, "status", 200)
        if status >= 400:
            raise OSError(f"remote contact sheet returned HTTP {status}")
        length = response.headers.get("Content-Length")
        if length is not None and int(length) <= 0:
            raise OSError("remote contact sheet is empty")


class DefaultContactSheetValidator:
    """Real local validation plus explicitly configured remote probes."""

    def __init__(
        self,
        *,
        remote_probes: Mapping[str, RemoteProbe] | None = None,
        remote_schemes: Iterable[str] = (),
    ) -> None:
        probes = {
            scheme.casefold(): probe
            for scheme, probe in (remote_probes or {}).items()
        }
        for scheme in remote_schemes:
            normalized = scheme.casefold()
            if normalized in {"http", "https"}:
                probes.setdefault(normalized, _probe_http_contact_sheet)
            elif normalized not in probes:
                raise ValueError(
                    f"remote scheme {scheme!r} requires an explicit validation probe"
                )
        self._remote_probes = probes

    def validate(self, uri: str | Path) -> str:
        text = str(uri)
        parsed = urlsplit(text)
        is_windows_path = isinstance(uri, Path) or bool(Path(text).drive)
        if parsed.scheme and not is_windows_path:
            probe = self._remote_probes.get(parsed.scheme.casefold())
            if probe is None:
                raise LongRouteDataError(
                    f"contact sheet URI scheme is not configured: {parsed.scheme}"
                )
            try:
                probe(text)
            except (OSError, ValueError) as error:
                raise LongRouteDataError(
                    f"remote contact sheet is not readable: {text}"
                ) from error
            return text

        path = Path(text).resolve()
        try:
            if not path.is_file():
                raise LongRouteDataError(
                    f"contact sheet must be a readable non-empty file: {path}"
                )
            with path.open("rb") as handle:
                if not handle.read(1):
                    raise LongRouteDataError(f"contact sheet is empty: {path}")
        except OSError as error:
            raise LongRouteDataError(
                f"contact sheet is not readable: {path}"
            ) from error
        return str(path)


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
    source_uri: str
    content_sha256: str
    split: Literal["train"]
    licensed: bool
    frame_embeddings: tuple[tuple[float, ...], ...]
    events: tuple[SourceEvent, ...]
    questions: tuple[SourceQuestion, ...]

    @model_validator(mode="after")
    def is_usable_train_source(self) -> "SourceVideo":
        if not self.video_id.strip():
            raise ValueError("video_id must be non-empty")
        if not self.source_uri.strip():
            raise ValueError("source video URI must be non-empty")
        digest = self.content_sha256.casefold()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("source video content_sha256 must be a SHA-256 hex digest")
        if len(self.frame_embeddings) != 8:
            raise ValueError("source video must provide exactly eight frame embeddings")
        width = len(self.frame_embeddings[0]) if self.frame_embeddings else 0
        if (
            width == 0
            or any(len(row) != width for row in self.frame_embeddings)
            or not all(math.isfinite(value) for row in self.frame_embeddings for value in row)
            or not any(value != 0 for row in self.frame_embeddings for value in row)
        ):
            raise ValueError("source video frame embeddings must be finite equal-width non-zero vectors")
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
    dataset_version: str
    source_uri: str
    license: str
    split: Literal["train"]
    videos: tuple[SourceVideo, ...]

    @model_validator(mode="after")
    def video_ids_are_unique(self) -> "TrainSplitManifest":
        if not all(
            value.strip()
            for value in (self.name, self.dataset_version, self.source_uri, self.license)
        ):
            raise ValueError("dataset name/version/source URI/license must be non-empty")
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


class SourceVideoProvenance(BaseModel):
    model_config = ConfigDict(frozen=True)

    video_id: str
    source_uri: str
    content_sha256: str


class SourceManifestProvenance(BaseModel):
    model_config = ConfigDict(frozen=True)

    identity: str
    dataset_name: str
    dataset_version: str
    source_uri: str
    license: str
    canonical_sha256: str
    videos: tuple[SourceVideoProvenance, ...]



class DatasetManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    manifest_version: str
    schema_version: str
    builder_version: str
    seed: int
    source_manifest_hashes: dict[str, str]
    source_manifests: tuple[SourceManifestProvenance, ...]
    builder_config: dict[str, object]
    group_assignment: dict[str, Literal["train", "dev"]]
    split_statistics: dict[str, int]
    multi_event_ratio: float
    leakage_audit_uri: str
    leakage_parquet_uri: str
    examples: tuple[LongRouteExample, ...]
    asset_sha256s: dict[str, str]
    generation_uri: str

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


ContactSheetProvider = Callable[[LongRouteExample], str | Path]


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _canonical_question(question: SourceQuestion) -> SourceQuestion:
    return question.model_copy(update={"options": tuple(sorted(question.options))})


def _canonical_video(video: SourceVideo) -> SourceVideo:
    events = tuple(
        event.model_copy(update={"attributes": dict(sorted(event.attributes.items()))})
        for event in sorted(video.events, key=lambda event: event.event_id)
    )
    questions = tuple(
        _canonical_question(question)
        for question in sorted(video.questions, key=lambda question: question.question_id)
    )
    return video.model_copy(
        update={
            "content_sha256": video.content_sha256.casefold(),
            "events": events,
            "questions": questions,
        }
    )


def _canonical_source_manifest(manifest: TrainSplitManifest) -> TrainSplitManifest:
    videos = tuple(
        _canonical_video(video)
        for video in sorted(manifest.videos, key=lambda video: video.video_id)
    )
    return manifest.model_copy(update={"videos": videos})


def _manifest_identity(manifest: TrainSplitManifest) -> str:
    return f"{manifest.name}@{manifest.dataset_version}|{manifest.source_uri}"


def _source_manifest_hash(manifest: TrainSplitManifest) -> str:
    payload = manifest.model_dump(mode="json")
    for video in payload["videos"]:
        video.pop("path", None)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _source_provenance(
    manifest: TrainSplitManifest,
) -> SourceManifestProvenance:
    canonical_hash = _source_manifest_hash(manifest)
    return SourceManifestProvenance(
        identity=_manifest_identity(manifest),
        dataset_name=manifest.name,
        dataset_version=manifest.dataset_version,
        source_uri=manifest.source_uri,
        license=manifest.license,
        canonical_sha256=canonical_hash,
        videos=tuple(
            SourceVideoProvenance(
                video_id=video.video_id,
                source_uri=video.source_uri,
                content_sha256=video.content_sha256,
            )
            for video in manifest.videos
        ),
    )


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
        publication_backend: PublicationBackend | None = None,
        contact_sheet_validator: ContactSheetValidator | None = None,
    ) -> None:
        self.train_manifests = tuple(train_manifests)
        self.eval_assets = tuple(eval_assets)
        self.leakage_auditor = leakage_auditor
        self.config = config
        self.contact_sheet_provider = contact_sheet_provider
        self.publication_backend = publication_backend or PublicationBackend(
            config.output_dir
        )
        if self.publication_backend.output_root != config.output_dir.resolve():
            raise ValueError("publication backend root must equal config.output_dir")
        self.contact_sheet_validator = (
            contact_sheet_validator or DefaultContactSheetValidator()
        )

    def build(self, seed: int) -> DatasetManifest:
        attempt = hashlib.sha256(
            f"{BUILDER_VERSION}:{seed}:validation".encode("utf-8")
        ).hexdigest()[:20]
        transaction: PublicationTransaction | None = None
        audit_payload: dict[str, object] = {
            "attempt_id": attempt,
            "seed": seed,
            "complete": False,
            "findings": [],
        }
        try:
            manifests, videos, content_hashes = self._validated_sources()
            provenance = tuple(_source_provenance(manifest) for manifest in manifests)
            source_hashes = {
                item.identity: item.canonical_sha256 for item in provenance
            }
            builder_config = self.config.model_dump(
                mode="json", exclude={"output_dir"}
            )
            train_assets = tuple(
                VideoAsset(
                    video.video_id,
                    video.path,
                    video.frame_embeddings,
                )
                for video in videos
            )

            def audit_hash_snapshot() -> dict[str, str]:
                snapshot: dict[str, str] = {}
                for category, assets in (
                    ("train", train_assets),
                    ("eval", self.eval_assets),
                ):
                    for asset in assets:
                        key = f"{category}:{asset.video_id}"
                        if key in snapshot:
                            raise LongRouteDataError(
                                f"duplicate {category} audit video identity: {asset.video_id}"
                            )
                        snapshot[key] = _sha256(asset.path)
                return dict(sorted(snapshot.items()))

            audited_hashes = audit_hash_snapshot()
            attempt = hashlib.sha256(
                _canonical_json(
                    {
                        "builder_version": BUILDER_VERSION,
                        "seed": seed,
                        "source_manifest_hashes": source_hashes,
                        "audited_asset_sha256s": audited_hashes,
                        "builder_config": builder_config,
                    }
                ).encode("utf-8")
            ).hexdigest()[:20]
            audit_payload.update(
                {
                    "attempt_id": attempt,
                    "train_asset_sha256s": content_hashes,
                    "audited_asset_sha256s": audited_hashes,
                }
            )
            report = self.leakage_auditor.audit(
                train_assets,
                self.eval_assets,
                require_coverage=True,
            )
            audit_payload["findings"] = [
                finding.__dict__ for finding in report.findings
            ]
            if report.findings:
                audit_payload["parquet_uri"] = str(report.parquet_path.resolve())
                self.publication_backend.write_root_text(
                    "leakage-audit.json", _canonical_json(audit_payload)
                )
                raise LongRouteLeakageError(
                    "formal evaluation leakage detected; no generation was published"
                )

            groups = self._groups(videos, seed)
            examples = self._examples(videos, groups, seed)
            if len(examples) < self.config.audit_size:
                raise LongRouteDataError(
                    f"audit requires {self.config.audit_size} examples, "
                    f"found {len(examples)}"
                )
            audit_files = self._audit_bundle_payloads(examples, seed)
            transaction = self.publication_backend.begin_generation(attempt)
            uri = transaction.generation_uri
            leakage_audit_uri = f"{uri}/leakage-audit.json"
            leakage_parquet_uri = f"{uri}/leakage-audit.parquet"
            audit_payload.update(
                {
                    "complete": True,
                    "parquet_uri": leakage_parquet_uri,
                }
            )
            manifest = DatasetManifest(
                manifest_version=MANIFEST_VERSION,
                schema_version=MANIFEST_VERSION,
                builder_version=BUILDER_VERSION,
                seed=seed,
                source_manifest_hashes=source_hashes,
                source_manifests=provenance,
                builder_config=builder_config,
                group_assignment=groups,
                split_statistics={
                    split: sum(item.split == split for item in examples)
                    for split in ("train", "dev")
                },
                multi_event_ratio=(
                    sum(item.template != "single_event" for item in examples)
                    / len(examples)
                ),
                leakage_audit_uri=leakage_audit_uri,
                leakage_parquet_uri=leakage_parquet_uri,
                examples=tuple(examples),
                asset_sha256s=content_hashes,
                generation_uri=uri,
            )

            self.publication_backend.write_generation_text(
                transaction,
                "leakage-audit.json",
                _canonical_json(audit_payload),
            )
            parquet_payload = report.parquet_path.read_bytes()
            if not parquet_payload:
                raise LongRouteDataError("leakage audit parquet is empty")
            self.publication_backend.write_generation_bytes(
                transaction,
                "leakage-audit.parquet",
                parquet_payload,
            )
            for relative, file_payload in audit_files.items():
                self.publication_backend.write_generation_text(
                    transaction, relative, file_payload
                )
            self.publication_backend.write_generation_text(
                transaction, "manifest.json", manifest.canonical_json()
            )

            if audit_hash_snapshot() != audited_hashes:
                raise LongRouteDataError(
                    "source or evaluation file changed after leakage audit"
                )
            self.publication_backend.publish_generation(transaction)
            self.publication_backend.publish_current(
                _canonical_json({"generation": uri, "seed": seed})
            )
            self.publication_backend.write_root_text(
                "last-attempt.json",
                _canonical_json(
                    {
                        "attempt_id": attempt,
                        "seed": seed,
                        "status": "complete",
                        "audit_uri": leakage_audit_uri,
                    }
                ),
            )
            return manifest
        except Exception as error:
            self.publication_backend.abort(transaction)
            audit_payload["complete"] = False
            failed_uri = f"failed-audit-{attempt}.json"
            self.publication_backend.write_root_text(
                failed_uri,
                _canonical_json(
                    audit_payload
                    | {
                        "error": type(error).__name__,
                        "message": str(error),
                    }
                ),
            )
            self.publication_backend.write_root_text(
                "last-attempt.json",
                _canonical_json(
                    {
                        "attempt_id": attempt,
                        "seed": seed,
                        "status": "failed",
                        "error": type(error).__name__,
                        "audit_uri": failed_uri,
                    }
                ),
            )
            if isinstance(error, ValueError) and not isinstance(
                error, LongRouteError
            ):
                raise LongRouteDataError(str(error)) from error
            raise



    def _validated_sources(
        self,
    ) -> tuple[
        tuple[TrainSplitManifest, ...],
        tuple[SourceVideo, ...],
        dict[str, str],
    ]:
        if not self.train_manifests:
            raise LongRouteDataError("at least one train split manifest is required")
        manifest_names: set[str] = set()
        manifest_identities: set[str] = set()
        video_ids: set[str] = set()
        video_source_uris: set[str] = set()
        event_ids: set[str] = set()
        question_ids: set[str] = set()
        manifests: list[TrainSplitManifest] = []
        videos: list[SourceVideo] = []
        content_hashes: dict[str, str] = {}

        for raw_manifest in sorted(
            self.train_manifests,
            key=lambda item: (
                item.name,
                item.dataset_version,
                item.source_uri,
            ),
        ):
            name = raw_manifest.name.casefold()
            if name in manifest_names:
                raise LongRouteDataError("duplicate manifest name")
            manifest_names.add(name)
            if raw_manifest.source_uri in manifest_identities:
                raise LongRouteDataError(
                    "duplicate manifest identity/source URI"
                )
            manifest_identities.add(raw_manifest.source_uri)
            if raw_manifest.split != "train":
                raise LongRouteDataError(
                    "LongRoute accepts train split manifests only"
                )
            manifest = _canonical_source_manifest(raw_manifest)
            for video in manifest.videos:
                if video.video_id in video_ids:
                    raise LongRouteDataError(
                        "duplicate global video identity"
                    )
                video_ids.add(video.video_id)
                if video.source_uri in video_source_uris:
                    raise LongRouteDataError(
                        "duplicate video source identity/URI"
                    )
                video_source_uris.add(video.source_uri)
                if not video.path.is_file():
                    raise LongRouteDataError(
                        f"source video path is unavailable: {video.path}"
                    )
                actual_hash = _sha256(video.path)
                if actual_hash != video.content_sha256.casefold():
                    raise LongRouteDataError(
                        f"source video SHA-256 mismatch: {video.video_id}"
                    )
                content_hashes[video.video_id] = actual_hash
                available_events = {event.event_id for event in video.events}
                for event in video.events:
                    if event.event_id in event_ids:
                        raise LongRouteDataError(
                            "duplicate global event identity"
                        )
                    event_ids.add(event.event_id)
                for question in video.questions:
                    if question.question_id in question_ids:
                        raise LongRouteDataError(
                            "duplicate global question identity"
                        )
                    if question.target_event_id not in available_events:
                        raise LongRouteDataError(
                            "question target_event_id must belong to its video"
                        )
                    question_ids.add(question.question_id)
                videos.append(video)
            manifests.append(manifest)

        return (
            tuple(manifests),
            tuple(sorted(videos, key=lambda item: item.video_id)),
            dict(sorted(content_hashes.items())),
        )

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
        labels = {f"{video.video_id}:{event.event_id}": event.label for video in videos for event in video.events}
        eligible: dict[str, LongRouteExample] = {}
        for item in basic:
            upgraded = self._multi_event(item, labels)
            if upgraded is not None:
                eligible[item.question_id] = upgraded
        if len(eligible) < lower:
            raise LongRouteDataError(
                f"multi-event eligible set has {len(eligible)} examples; "
                f"at least {lower} are required"
            )
        desired = min(
            max(round(len(basic) * self.config.multi_event_ratio), lower),
            upper,
        )
        multi_count = min(desired, len(eligible))
        eligible_ids = sorted(eligible)
        random.Random(seed).shuffle(eligible_ids)
        multi_ids = set(eligible_ids[:multi_count])
        return [
            eligible[item.question_id]
            if item.question_id in multi_ids
            else item
            for item in basic
        ]

    def _select_distractors(
        self,
        target: SourceEvent,
        ranked: Sequence[tuple[SourceVideo, SourceEvent]],
    ) -> tuple[tuple[SourceVideo, SourceEvent], ...]:
        target_duration = target.end_sec - target.start_sec
        if target_duration > 3600:
            raise LongRouteDataError("target event exceeds the 60 minute route limit")
        durations = tuple(
            event.end_sec - event.start_sec for _, event in ranked
        )

        def search(
            index: int,
            selected: tuple[tuple[SourceVideo, SourceEvent], ...],
            duration: float,
        ) -> tuple[tuple[SourceVideo, SourceEvent], ...] | None:
            count = len(selected)
            if (
                self.config.min_distractors <= count <= self.config.max_distractors
                and 600 <= duration <= 3600
            ):
                return selected
            remaining_count = len(ranked) - index
            needed = max(0, self.config.min_distractors - count)
            if (
                index == len(ranked)
                or count == self.config.max_distractors
                or remaining_count < needed
            ):
                return None

            remaining_durations = durations[index:]
            if needed:
                minimum_completion = sum(sorted(remaining_durations)[:needed])
                if duration + minimum_completion > 3600:
                    return None
            slots = min(
                self.config.max_distractors - count,
                remaining_count,
            )
            maximum_completion = sum(
                sorted(remaining_durations, reverse=True)[:slots]
            )
            if duration + maximum_completion < 600:
                return None

            candidate_duration = durations[index]
            if duration + candidate_duration <= 3600:
                found = search(
                    index + 1,
                    selected + (ranked[index],),
                    duration + candidate_duration,
                )
                if found is not None:
                    return found
            return search(index + 1, selected, duration)

        chosen = search(0, (), target_duration)
        if chosen is None:
            raise LongRouteDataError(
                "cannot form a 10-60 minute route with 9-19 distractors"
            )
        return chosen



    def _single_example(self, video: SourceVideo, question: SourceQuestion, split: Literal["train", "dev"], videos: Sequence[SourceVideo], groups: dict[str, Literal["train", "dev"]], seed: int) -> LongRouteExample:
        target = next(event for event in video.events if event.event_id == question.target_event_id)
        candidates = [
            (candidate_video, event) for candidate_video in videos if groups[candidate_video.video_id] == split
            for event in candidate_video.events if not (candidate_video.video_id == video.video_id and event.event_id == target.event_id)
        ]
        ranked = sorted(candidates, key=lambda item: (-_cosine(target.embedding, item[1].embedding), item[1].event_id, item[0].video_id))
        chosen = list(self._select_distractors(target, ranked))
        position = _seeded_index(seed, question.question_id, len(chosen) + 1)
        entries = chosen.copy()
        entries.insert(position, (video, target))
        segments: list[VirtualSegment] = []
        offset = 0.0
        for source, event in entries:
            length = event.end_sec - event.start_sec
            segments.append(VirtualSegment(source_video_id=source.video_id, event_id=event.event_id, source_start_sec=event.start_sec, source_end_sec=event.end_sec, global_start_sec=offset, global_end_sec=offset + length))
            offset += length
        target_id = f"{video.video_id}:{target.event_id}"
        return LongRouteExample(question_id=question.question_id, split=split, question=question.question, options=question.options, answer=question.answer, target_source_video_id=video.video_id, target_event_id=target_id, target_position=position, supporting_event_ids=(target_id,), template="single_event", segments=tuple(segments), duration_sec=offset)

    def _multi_event(
        self,
        item: LongRouteExample,
        labels: dict[str, str],
    ) -> LongRouteExample | None:
        position = item.target_position
        candidates: list[tuple[int, int, Literal["after", "before"]]] = []
        if position + 1 < len(item.segments):
            candidates.append((position, position + 1, "after"))
        if position > 0:
            candidates.append((position - 1, position, "before"))

        for left_index, right_index, relation in candidates:
            left = item.segments[left_index]
            right = item.segments[right_index]
            left_id = f"{left.source_video_id}:{left.event_id}"
            right_id = f"{right.source_video_id}:{right.event_id}"
            if right.global_start_sec != left.global_end_sec:
                raise LongRouteDataError(
                    "multi-event supporting segments must be globally adjacent"
                )
            left_label, right_label = labels[left_id], labels[right_id]
            if left_label == right_label:
                continue
            if relation == "after":
                question = f"Which event happens immediately after {left_label}?"
                answer = right_label
            else:
                question = f"Which event happens immediately before {right_label}?"
                answer = left_label
            if answer == "None of these":
                continue
            return item.model_copy(
                update={
                    "question": question,
                    "options": tuple(sorted((answer, "None of these"))),
                    "answer": answer,
                    "supporting_event_ids": (left_id, right_id),
                    "template": "before_after",
                }
            )
        return None

    def _audit_bundle_payloads(
        self,
        examples: Sequence[LongRouteExample],
        seed: int,
    ) -> dict[str, str]:
        if self.contact_sheet_provider is None:
            raise LongRouteDataError(
                "a contact_sheet_provider is required for the human audit package"
            )
        selected = sorted(examples, key=lambda item: item.question_id)[
            : self.config.audit_size
        ]
        if len(selected) != self.config.audit_size:
            raise LongRouteDataError(
                f"audit requires exactly {self.config.audit_size} examples"
            )
        records: list[dict[str, object]] = []
        for example in selected:
            try:
                provided = self.contact_sheet_provider(example)
                contact = self.contact_sheet_validator.validate(provided)
            except LongRouteError:
                raise
            except (OSError, ValueError) as error:
                raise LongRouteDataError(
                    f"contact sheet is not readable for {example.question_id}"
                ) from error
            records.append(
                {
                    "question_id": example.question_id,
                    "question": example.question,
                    "options": list(example.options),
                    "answer": example.answer,
                    "source_events": [
                        segment.model_dump(mode="json")
                        for segment in example.segments
                    ],
                    "global_offsets": [
                        [segment.global_start_sec, segment.global_end_sec]
                        for segment in example.segments
                    ],
                    "contact_sheet": contact,
                    "seed": seed,
                }
            )

        import io

        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(
            buffer,
            fieldnames=["question_id", "valid", "invalid", "reason"],
            lineterminator="\n",
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "question_id": record["question_id"],
                    "valid": "",
                    "invalid": "",
                    "reason": "",
                }
            )
        return {
            "audit/samples.json": _canonical_json(records),
            "audit/samples.jsonl": "".join(
                _canonical_json(record) + "\n" for record in records
            ),
            "audit/review.csv": buffer.getvalue(),
        }
