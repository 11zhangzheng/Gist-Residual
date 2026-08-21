"""Question-independent, auditable Residual memory expansion."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from fidmem.storage.cache import ContentAddressedCache
from fidmem.types import EventRecord


RESIDUAL_FIELDS = (
    "entities", "actions", "attributes", "spatial_relations", "counts",
    "state_changes", "exceptions", "unstructured_details",
)


class ResidualPayload(BaseModel):
    """The fixed public schema; VLM text stays in the audit record."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    entities: tuple[str, ...]
    actions: tuple[str, ...]
    attributes: tuple[str, ...]
    spatial_relations: tuple[str, ...]
    counts: tuple[str, ...]
    state_changes: tuple[str, ...]
    exceptions: tuple[str, ...]
    unstructured_details: tuple[str, ...]

    @field_validator(*RESIDUAL_FIELDS)
    @classmethod
    def details_must_be_strings(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not isinstance(value, str) for value in values):
            raise ValueError("Residual details must be strings")
        return values

    @classmethod
    def empty(cls) -> "ResidualPayload":
        return cls(**{field: () for field in RESIDUAL_FIELDS})


class ResidualAudit(BaseModel):
    """Audit material deliberately excluded from the eight-field payload."""

    model_config = ConfigDict(frozen=True)
    raw_response: str
    repaired_response: str | None = None
    repair_attempted: bool = False
    filtered_exact: int = 0
    filtered_gist_semantic: int = 0
    filtered_residual_semantic: int = 0


class ResidualRecord(BaseModel):
    model_config = ConfigDict(frozen=True)
    event_id: str
    payload: ResidualPayload
    audit: ResidualAudit
    schema_error: str | None = None
    cache_key: str


VLMAdapter = Callable[[tuple[str, ...], str], str]
RepairAdapter = Callable[[str], str]
EmbeddingAdapter = Callable[[str], Sequence[float]]
FrameSampler = Callable[[EventRecord], tuple[str, ...]]


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _propositions(gist: str) -> tuple[str, ...]:
    return tuple(piece.strip() for piece in re.split(r"[;。！？!?\n.]", gist) if piece.strip())


def _source_event_projection(event: EventRecord) -> dict[str, object]:
    """Stable offline event content, excluding online fidelity upgrades."""
    return {
        "video_id": event.video_id,
        "event_id": event.event_id,
        "start_sec": event.start_sec,
        "end_sec": event.end_sec,
        "asr_text": event.asr_text,
        "keyframe_paths": event.keyframe_paths,
        "visual_embedding": event.visual_embedding,
        "text_embedding": event.text_embedding,
        "gist_text": event.gist_text,
        "raw_video_uri": event.raw_video_uri,
        "memory_version": event.memory_version,
    }


def _embedding(values: Sequence[float], *, label: str) -> tuple[float, ...]:
    vector = tuple(float(value) for value in values)
    if not vector:
        raise ValueError(f"{label} embedding must not be empty")
    if not all(math.isfinite(value) for value in vector):
        raise ValueError(f"{label} embedding must contain only finite values")
    if math.sqrt(sum(value * value for value in vector)) == 0.0:
        raise ValueError(f"{label} embedding must not be a zero vector")
    return vector


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions must match")
    return sum(a * b for a, b in zip(left, right, strict=True)) / (
        math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    )


class ResidualGenerator:
    """Expand one event without receiving a question or answer options."""

    def __init__(
        self, *, cache: ContentAddressedCache, vlm: VLMAdapter, embedder: EmbeddingAdapter,
        model_version: str, prompt_template: str, frame_sampler: FrameSampler,
        frame_sampler_version: str, schema_version: str, normalizer_version: str,
        json_repair: RepairAdapter | None = None, embedder_version: str | None = None,
    ) -> None:
        if not all(isinstance(value, str) and value.strip() for value in (
            model_version, prompt_template, frame_sampler_version, schema_version, normalizer_version,
        )):
            raise ValueError("Residual versions and prompt must not be blank")
        self.cache = cache
        self.vlm = vlm
        self.embedder = embedder
        self.model_version = model_version
        self.prompt_template = prompt_template
        self.frame_sampler = frame_sampler
        self.frame_sampler_version = frame_sampler_version
        self.schema_version = schema_version
        self.normalizer_version = normalizer_version
        self.json_repair = json_repair
        self.embedder_version = embedder_version or str(getattr(embedder, "identity", type(embedder).__qualname__))

    def _prompt(self, gist: str) -> str:
        return (
            f"{self.prompt_template}\n\nExisting Gist:\n{gist}\n\n"
            "Return JSON with exactly the required Residual fields. Add only details "
            "not already in the Gist; do not repeat existing propositions. Do not "
            "return direct answers or multiple-choice selections."
        )

    @staticmethod
    def _video_hash(event: EventRecord) -> str:
        return hashlib.sha256(event.raw_video_uri.encode("utf-8")).hexdigest()

    def _cache_key(self, event: EventRecord, gist: str, frames: tuple[str, ...]) -> str:
        prompt = self._prompt(gist)
        return self.cache.key(
            self._video_hash(event), (event.start_sec, event.end_sec), self.model_version, prompt,
            {"namespace": "residual", "source_event": _source_event_projection(event), "gist": gist,
             "frames": frames, "frame_sampler_version": self.frame_sampler_version,
             "schema_version": self.schema_version, "normalizer_version": self.normalizer_version,
             "embedder_version": self.embedder_version},
        )

    def _parse(self, raw_response: str) -> ResidualPayload:
        parsed: Any = json.loads(raw_response)
        if not isinstance(parsed, dict) or set(parsed) != set(RESIDUAL_FIELDS):
            raise ValueError("Residual JSON must contain exactly the eight schema fields")
        return ResidualPayload.model_validate(parsed)

    def _deduplicate(self, payload: ResidualPayload, gist: str) -> tuple[ResidualPayload, tuple[int, int, int]]:
        gist_items = _propositions(gist)
        gist_norms = {_normalize(item) for item in gist_items}
        gist_embeddings = tuple(_embedding(self.embedder(item), label="Gist") for item in gist_items)
        exact = gist_semantic = residual_semantic = 0
        accepted_embeddings: list[tuple[float, ...]] = []
        seen_exact: set[str] = set()
        filtered: dict[str, tuple[str, ...]] = {}
        for field in RESIDUAL_FIELDS:
            kept: list[str] = []
            for item in getattr(payload, field):
                normalized = _normalize(item)
                if not normalized or normalized in gist_norms or normalized in seen_exact:
                    exact += 1
                    continue
                candidate_embedding = _embedding(self.embedder(item), label="Residual")
                if any(_cosine(candidate_embedding, gist_embedding) >= 0.92 for gist_embedding in gist_embeddings):
                    gist_semantic += 1
                    continue
                if any(_cosine(candidate_embedding, prior) >= 0.92 for prior in accepted_embeddings):
                    residual_semantic += 1
                    continue
                seen_exact.add(normalized)
                accepted_embeddings.append(candidate_embedding)
                kept.append(item.strip())
            filtered[field] = tuple(kept)
        return ResidualPayload(**filtered), (exact, gist_semantic, residual_semantic)

    def expand(self, event: EventRecord, gist: str | EventRecord) -> ResidualRecord:
        """Load or create an auditable event-level Residual observation."""
        gist_text = gist.gist_text if isinstance(gist, EventRecord) else gist
        if not isinstance(gist_text, str):
            raise TypeError("gist must be text or an EventRecord")
        frames = tuple(self.frame_sampler(event))
        if not frames:
            raise ValueError("Residual frame sampler returned no frames")
        key = self._cache_key(event, gist_text, frames)
        cached = self.cache.get(key)
        if cached is not None:
            result = ResidualRecord.model_validate(cached)
            if result.event_id != event.event_id:
                raise ValueError("cached Residual identity does not match the requested event")
            return result
        raw_response = self.vlm(frames, self._prompt(gist_text))
        repaired_response: str | None = None
        repair_attempted = False
        try:
            parsed = self._parse(raw_response)
        except (TypeError, ValueError, json.JSONDecodeError) as original_error:
            if self.json_repair is None:
                result = ResidualRecord(event_id=event.event_id, payload=ResidualPayload.empty(),
                    audit=ResidualAudit(raw_response=raw_response), schema_error=str(original_error), cache_key=key)
                self.cache.put(key, result.model_dump(mode="json"))
                return result
            repair_attempted = True
            repaired_response = self.json_repair(raw_response)
            try:
                parsed = self._parse(repaired_response)
            except (TypeError, ValueError, json.JSONDecodeError) as repair_error:
                result = ResidualRecord(event_id=event.event_id, payload=ResidualPayload.empty(),
                    audit=ResidualAudit(raw_response=raw_response, repaired_response=repaired_response, repair_attempted=True),
                    schema_error=str(repair_error), cache_key=key)
                self.cache.put(key, result.model_dump(mode="json"))
                return result
        deduplicated, counts = self._deduplicate(parsed, gist_text)
        result = ResidualRecord(event_id=event.event_id, payload=deduplicated,
            audit=ResidualAudit(raw_response=raw_response, repaired_response=repaired_response,
                repair_attempted=repair_attempted, filtered_exact=counts[0],
                filtered_gist_semantic=counts[1], filtered_residual_semantic=counts[2]), cache_key=key)
        self.cache.put(key, result.model_dump(mode="json"))
        return result
