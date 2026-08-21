"""Low-cost, content-addressed Gist memory construction."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fidmem.storage.cache import ContentAddressedCache
from fidmem.types import EventRecord

SummaryFunction = Callable[[str, int], str]
TextEncoder = Callable[[str], Sequence[float]]
VisualEncoder = Callable[[tuple[str, ...], tuple[int, int]], Sequence[float]]
VLMFunction = Callable[[tuple[str, ...], str, int], str]


class GistEventInput(BaseModel):
    """Cheap event inputs available before semantic memory construction."""

    model_config = ConfigDict(frozen=True)

    video_id: str
    event_id: str
    start_sec: float = Field(ge=0)
    end_sec: float = Field(ge=0)
    asr_text: str | None = None
    keyframe_paths: tuple[str, ...]
    raw_video_uri: str
    video_hash: str
    memory_version: str

    @model_validator(mode="after")
    def end_must_not_precede_start(self) -> "GistEventInput":
        if self.end_sec < self.start_sec:
            raise ValueError("end_sec must be at least start_sec")
        return self


def _sample_evenly(paths: tuple[str, ...], count: int) -> tuple[str, ...]:
    if len(paths) < count:
        raise ValueError(f"Gist construction requires at least four source frames")
    if count == 1:
        return (paths[len(paths) // 2],)
    indices = tuple(round(index * (len(paths) - 1) / (count - 1)) for index in range(count))
    return tuple(paths[index] for index in indices)


def _finite_embedding(values: Sequence[float], name: str) -> tuple[float, ...]:
    embedding = tuple(float(value) for value in values)
    if not embedding:
        raise ValueError(f"{name} must not be empty")
    if not all(math.isfinite(value) for value in embedding):
        raise ValueError(f"{name} must contain only finite values")
    if math.sqrt(sum(value * value for value in embedding)) == 0.0:
        raise ValueError(f"{name} must not be a zero vector")
    return embedding


def _truncate_tokens(text: str, max_tokens: int) -> str:
    tokens = text.split()
    if not tokens:
        raise ValueError("Gist summary must not be empty")
    return " ".join(tokens[:max_tokens])


class GistBuilder:
    """Build the sole full-coverage semantic memory without a main-path VLM."""

    def __init__(
        self,
        *,
        cache: ContentAddressedCache,
        summarizer: SummaryFunction,
        text_encoder: TextEncoder,
        visual_encoder: VisualEncoder,
        model_version: str,
        prompt: str,
        namespace: str = "gist",
        mode: Literal["main", "gist_plus"] = "main",
        vlm: VLMFunction | None = None,
        max_tokens: int = 40,
        frame_count: int = 4,
        visual_resolution: tuple[int, int] = (224, 224),
    ) -> None:
        if mode not in ("main", "gist_plus"):
            raise ValueError("mode must be 'main' or 'gist_plus'")
        if not namespace.strip():
            raise ValueError("cache namespace must not be blank")
        if max_tokens <= 0 or max_tokens > 40:
            raise ValueError("max_tokens must be positive and at most 40")
        if frame_count != 4:
            raise ValueError("the base Gist frame budget is exactly four")
        if any(side <= 0 for side in visual_resolution):
            raise ValueError("visual resolution must be positive")
        if mode == "gist_plus" and vlm is None:
            raise ValueError("gist_plus mode requires a VLM")
        self.cache = cache
        self.summarizer = summarizer
        self.text_encoder = text_encoder
        self.visual_encoder = visual_encoder
        self.model_version = model_version
        self.prompt = prompt
        self.namespace = namespace
        self.mode = mode
        self.vlm = vlm
        self.max_tokens = max_tokens
        self.frame_count = frame_count
        self.visual_resolution = visual_resolution

    def _cache_key(self, event: GistEventInput) -> str:
        return self.cache.key(
            event.video_hash,
            (event.start_sec, event.end_sec),
            self.model_version,
            self.prompt,
            {
                "namespace": self.namespace,
                "mode": self.mode,
                "sampling_config": {
                    "frame_count": self.frame_count,
                    "resolution": self.visual_resolution,
                },
                "asr_text": event.asr_text,
                "keyframe_paths": event.keyframe_paths,
                "raw_video_uri": event.raw_video_uri,
                "memory_version": event.memory_version,
            },
        )

    def build(self, event: GistEventInput) -> EventRecord:
        """Build or load one immutable Gist record."""
        sampled_frames = _sample_evenly(event.keyframe_paths, self.frame_count)
        key = self._cache_key(event)
        cached = self.cache.get(key)
        if cached is not None:
            return EventRecord.model_validate(cached)

        asr_text = (event.asr_text or "").strip() or "[no speech]"
        if asr_text == "[no speech]":
            summary = asr_text
        else:
            summary = self.summarizer(asr_text, self.max_tokens)

        if self.mode == "gist_plus":
            assert self.vlm is not None
            visual_summary = self.vlm(sampled_frames, asr_text, self.max_tokens)
            summary = f"{summary} {visual_summary}"
        gist_text = _truncate_tokens(summary, self.max_tokens)

        record = EventRecord(
            video_id=event.video_id,
            event_id=event.event_id,
            start_sec=event.start_sec,
            end_sec=event.end_sec,
            asr_text=asr_text,
            keyframe_paths=sampled_frames,
            visual_embedding=_finite_embedding(
                self.visual_encoder(sampled_frames, self.visual_resolution),
                "visual_embedding",
            ),
            text_embedding=_finite_embedding(
                self.text_encoder(gist_text), "text_embedding"
            ),
            gist_text=gist_text,
            raw_video_uri=event.raw_video_uri,
            memory_version=(
                f"{event.memory_version}:{self.namespace}:{self.model_version}"
            ),
        )
        self.cache.put(key, record.model_dump(mode="json"))
        return record
