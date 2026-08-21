"""Raw-visual verification with isolated event and question caches."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Callable, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from fidmem.storage.cache import ContentAddressedCache
from fidmem.types import EventRecord

VisualBudget = Literal["low", "high"]
_FRAME_COUNTS: dict[VisualBudget, int] = {"low": 12, "high": 32}


class VisualCostMetadata(BaseModel):
    """Metadata directly matching Task 3's visual cost component."""

    model_config = ConfigDict(frozen=True)
    cost_component: Literal["visual"] = "visual"
    charge_scope: Literal["event_observation", "question_verification"]
    cache_status: Literal["hit", "miss"]
    amortizable: bool
    reused: bool
    input_frames: int = Field(ge=0)
    evidence_frame_count: int = Field(ge=0)


class VisualObservation(BaseModel):
    model_config = ConfigDict(frozen=True)
    event_id: str
    budget: VisualBudget
    frames: tuple[str, ...]
    generic_observation: str
    cache_key: str
    cost_metadata: VisualCostMetadata


class VisualVerification(BaseModel):
    model_config = ConfigDict(frozen=True)
    event_id: str
    budget: VisualBudget
    verification: str
    cache_key: str
    event_cache_key: str
    event_cost_metadata: VisualCostMetadata
    cost_metadata: VisualCostMetadata


class ContextFrontier(BaseModel):
    """Immutable visited radii around an anchor event."""

    model_config = ConfigDict(frozen=True)
    anchor_event_id: str
    left_radius: int = Field(default=0, ge=0)
    right_radius: int = Field(default=0, ge=0)
    exhausted: bool = False


class ContextExpansion(BaseModel):
    model_config = ConfigDict(frozen=True)
    events: tuple[EventRecord, ...]
    frontier: ContextFrontier


EventAdapter = Callable[[tuple[str, ...]], str]
QuestionAdapter = Callable[[str, str, tuple[str, ...]], str]
FrameSampler = Callable[[EventRecord, int], tuple[str, ...]]


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _sample_evenly(paths: Sequence[str], count: int) -> tuple[str, ...]:
    if len(paths) < count:
        raise ValueError(f"visual verification requires at least {count} source frames")
    indices = tuple(round(index * (len(paths) - 1) / (count - 1)) for index in range(count))
    return tuple(paths[index] for index in indices)


def _source_event_projection(event: EventRecord) -> dict[str, object]:
    """Stable offline event content, excluding online Residual state."""
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


class VisualVerifier:
    """Frozen adapter interfaces with event and question cache-key separation."""

    def __init__(
        self, *, cache: ContentAddressedCache, event_adapter: EventAdapter,
        question_adapter: QuestionAdapter, model_version: str, event_prompt: str,
        question_prompt: str, sampler_version: str, frame_sampler: FrameSampler | None = None,
    ) -> None:
        if not all(isinstance(item, str) and item.strip() for item in (
            model_version, event_prompt, question_prompt, sampler_version,
        )):
            raise ValueError("Visual versions and prompts must not be blank")
        self.cache = cache
        self.event_adapter = event_adapter
        self.question_adapter = question_adapter
        self.model_version = model_version
        self.event_prompt = event_prompt
        self.question_prompt = question_prompt
        self.sampler_version = sampler_version
        self.frame_sampler = frame_sampler or (lambda event, count: _sample_evenly(event.keyframe_paths, count))
        self.event_adapter_version = str(getattr(event_adapter, "identity", type(event_adapter).__qualname__))
        self.question_adapter_version = str(getattr(question_adapter, "identity", type(question_adapter).__qualname__))

    @staticmethod
    def _video_hash(event: EventRecord) -> str:
        return hashlib.sha256(event.raw_video_uri.encode("utf-8")).hexdigest()

    @staticmethod
    def _ensure_budget(budget: str) -> VisualBudget:
        if budget not in _FRAME_COUNTS:
            raise ValueError("visual budget must be one of: low, high")
        return budget  # type: ignore[return-value]

    def _event_key(self, event: EventRecord, budget: VisualBudget, frames: tuple[str, ...]) -> str:
        return self.cache.key(
            self._video_hash(event), (event.start_sec, event.end_sec), self.model_version,
            self.event_prompt,
            {"namespace": "visual-event", "source_event": _source_event_projection(event),
             "budget": budget, "frame_count": _FRAME_COUNTS[budget], "frames": frames,
             "sampler_version": self.sampler_version, "adapter_version": self.event_adapter_version},
        )

    def _question_key(self, event: EventRecord, budget: VisualBudget, event_key: str, question: str, options: tuple[str, ...]) -> str:
        normalized_question = _normalize(question)
        normalized_options = tuple(_normalize(option) for option in options)
        options_hash = hashlib.sha256(json.dumps(normalized_options, ensure_ascii=False).encode("utf-8")).hexdigest()
        return self.cache.key(
            self._video_hash(event), (event.start_sec, event.end_sec), self.model_version,
            self.question_prompt,
            {"namespace": "visual-question", "event_id": event.event_id, "budget": budget,
             "event_cache_key": event_key, "normalized_question": normalized_question,
             "options_hash": options_hash, "sampler_version": self.sampler_version,
             "adapter_version": self.question_adapter_version},
        )

    def observe_event(self, event: EventRecord, budget: VisualBudget) -> VisualObservation:
        checked_budget = self._ensure_budget(budget)
        frames = tuple(self.frame_sampler(event, _FRAME_COUNTS[checked_budget]))
        if len(frames) != _FRAME_COUNTS[checked_budget]:
            raise ValueError("visual frame sampler must return the exact requested budget")
        key = self._event_key(event, checked_budget, frames)
        cached = self.cache.get(key)
        if cached is not None:
            observation = VisualObservation.model_validate(cached)
            if observation.event_id != event.event_id:
                raise ValueError("cached visual observation identity does not match event")
            return observation.model_copy(update={"cost_metadata": VisualCostMetadata(
                charge_scope="event_observation", cache_status="hit", amortizable=True,
                reused=True, input_frames=0, evidence_frame_count=len(observation.frames),
            )})
        generic = self.event_adapter(frames)
        observation = VisualObservation(
            event_id=event.event_id, budget=checked_budget, frames=frames,
            generic_observation=str(generic), cache_key=key,
            cost_metadata=VisualCostMetadata(charge_scope="event_observation", cache_status="miss", amortizable=True, reused=False, input_frames=len(frames), evidence_frame_count=len(frames)),
        )
        self.cache.put(key, observation.model_dump(mode="json"))
        return observation

    def verify_question(self, event: EventRecord, question: str, options: tuple[str, ...], budget: VisualBudget) -> VisualVerification:
        if not question.strip():
            raise ValueError("question must not be blank")
        observation = self.observe_event(event, budget)
        key = self._question_key(event, observation.budget, observation.cache_key, question, options)
        cached = self.cache.get(key)
        if cached is not None:
            result = VisualVerification.model_validate(cached)
            if result.event_id != event.event_id:
                raise ValueError("cached visual verification identity does not match event")
            return result.model_copy(update={"cost_metadata": VisualCostMetadata(
                charge_scope="question_verification", cache_status="hit", amortizable=False,
                reused=True, input_frames=0, evidence_frame_count=len(observation.frames),
            ), "event_cost_metadata": observation.cost_metadata})
        verification = self.question_adapter(observation.generic_observation, question, options)
        result = VisualVerification(
            event_id=event.event_id, budget=observation.budget, verification=str(verification),
            cache_key=key, event_cache_key=observation.cache_key,
            event_cost_metadata=observation.cost_metadata,
            cost_metadata=VisualCostMetadata(charge_scope="question_verification", cache_status="miss", amortizable=False, reused=False, input_frames=0, evidence_frame_count=len(observation.frames)),
        )
        self.cache.put(key, result.model_dump(mode="json"))
        return result


def expand_context(events: Sequence[EventRecord], frontier: ContextFrontier) -> ContextExpansion:
    """Expand exactly one unvisited event per available side without generation."""
    ordered = tuple(sorted(events, key=lambda event: (event.start_sec, event.end_sec, event.event_id)))
    ids = tuple(event.event_id for event in ordered)
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate event_id is not valid for a ContextFrontier")
    if frontier.anchor_event_id not in ids:
        raise ValueError("ContextFrontier anchor_event_id is absent from events")
    anchor = ids.index(frontier.anchor_event_id)
    max_left, max_right = anchor, len(ordered) - anchor - 1
    if frontier.left_radius > max_left:
        raise ValueError("left_radius exceeds available context")
    if frontier.right_radius > max_right:
        raise ValueError("right_radius exceeds available context")
    if frontier.exhausted and (frontier.left_radius != max_left or frontier.right_radius != max_right):
        raise ValueError("exhausted ContextFrontier must cover all available context")
    if not frontier.exhausted and frontier.left_radius == max_left and frontier.right_radius == max_right:
        raise ValueError("fully covered ContextFrontier must be exhausted")
    if frontier.exhausted:
        return ContextExpansion(events=(), frontier=frontier)
    new_events: list[EventRecord] = []
    left_radius, right_radius = frontier.left_radius, frontier.right_radius
    if left_radius < max_left:
        left_radius += 1
        new_events.append(ordered[anchor - left_radius])
    if right_radius < max_right:
        right_radius += 1
        new_events.append(ordered[anchor + right_radius])
    next_frontier = ContextFrontier(
        anchor_event_id=frontier.anchor_event_id, left_radius=left_radius,
        right_radius=right_radius, exhausted=left_radius == max_left and right_radius == max_right,
    )
    return ContextExpansion(events=tuple(new_events), frontier=next_frontier)
