"""Deterministic text/visual fusion retrieval over Gist records."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from fidmem.types import EventRecord

QueryEncoder = Callable[[str], Sequence[float]]


@dataclass(frozen=True)
class ScoredEvent:
    """One retrieved event with auditable fused and component scores."""

    event: EventRecord
    score: float
    text_score: float
    visual_score: float


def _as_finite_vector(values: Sequence[float], name: str) -> tuple[float, ...]:
    vector = tuple(float(value) for value in values)
    if not vector:
        raise ValueError(f"{name} must not be empty")
    if not all(math.isfinite(value) for value in vector):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _normalized_cosine(
    query: Sequence[float], candidate: Sequence[float], modality: str
) -> float:
    query_vector = _as_finite_vector(query, f"{modality} query embedding")
    candidate_vector = _as_finite_vector(candidate, f"{modality} event embedding")
    if len(query_vector) != len(candidate_vector):
        raise ValueError(f"{modality} embedding dimension mismatch")
    query_norm = math.sqrt(sum(value * value for value in query_vector))
    candidate_norm = math.sqrt(sum(value * value for value in candidate_vector))
    if query_norm == 0.0 or candidate_norm == 0.0:
        raise ValueError(f"{modality} embedding must not be a zero vector")
    cosine = sum(
        query_value * candidate_value
        for query_value, candidate_value in zip(query_vector, candidate_vector)
    ) / (query_norm * candidate_norm)
    return (max(-1.0, min(1.0, cosine)) + 1.0) / 2.0


class GistIndex:
    """In-memory multimodal index with deterministic ranking semantics."""

    def __init__(
        self,
        events: Sequence[EventRecord],
        *,
        text_query_encoder: QueryEncoder,
        visual_query_encoder: QueryEncoder,
        text_weight: float = 0.6,
        visual_weight: float = 0.4,
    ) -> None:
        if not math.isfinite(text_weight) or not math.isfinite(visual_weight):
            raise ValueError("retrieval weights must be finite")
        if text_weight < 0 or visual_weight < 0 or text_weight + visual_weight <= 0:
            raise ValueError("retrieval weights must be non-negative with a positive sum")
        event_tuple = tuple(events)
        event_ids = tuple(event.event_id for event in event_tuple)
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("event_id values must be unique within a Gist index")
        weight_sum = text_weight + visual_weight
        self.events = event_tuple
        self.text_query_encoder = text_query_encoder
        self.visual_query_encoder = visual_query_encoder
        self.text_weight = text_weight / weight_sum
        self.visual_weight = visual_weight / weight_sum

    def search(self, question: str, k: int) -> tuple[ScoredEvent, ...]:
        """Return at most ``k`` events using normalized cosine fusion."""
        if k <= 0:
            raise ValueError("k must be positive")
        if not question.strip():
            raise ValueError("question must not be blank")
        if not self.events:
            return ()

        text_query = self.text_query_encoder(question)
        visual_query = self.visual_query_encoder(question)
        scored: list[ScoredEvent] = []
        for event in self.events:
            text_score = _normalized_cosine(
                text_query, event.text_embedding, "text"
            )
            visual_score = _normalized_cosine(
                visual_query, event.visual_embedding, "visual"
            )
            score = self.text_weight * text_score + self.visual_weight * visual_score
            scored.append(
                ScoredEvent(
                    event=event,
                    score=score,
                    text_score=text_score,
                    visual_score=visual_score,
                )
            )
        scored.sort(
            key=lambda result: (
                -result.score,
                result.event.start_sec,
                result.event.event_id,
            )
        )
        return tuple(scored[:k])
