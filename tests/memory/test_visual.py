from pathlib import Path

import pytest

from fidmem.memory.visual import ContextFrontier, VisualVerifier, expand_context
from fidmem.storage.cache import ContentAddressedCache
from fidmem.types import EventRecord


class EventAdapter:
    identity = "event-adapter-v1"
    def __init__(self) -> None: self.calls: list[tuple[str, ...]] = []
    def __call__(self, frames: tuple[str, ...]) -> str:
        self.calls.append(frames); return "generic observation"


class QuestionAdapter:
    identity = "question-adapter-v1"
    def __init__(self) -> None: self.calls: list[tuple[str, str, tuple[str, ...]]] = []
    def __call__(self, observation: str, question: str, options: tuple[str, ...]) -> str:
        self.calls.append((observation, question, options)); return f"verified {question}"


def _event(event_id: str, start: int, *, residual: str | None = None) -> EventRecord:
    return EventRecord(video_id="v", event_id=event_id, start_sec=start, end_sec=start + 1,
                       gist_text=f"gist {event_id}", keyframe_paths=tuple(f"{event_id}-{n}.jpg" for n in range(40)),
                       raw_video_uri="video.mp4", memory_version="v1", residual=residual)


def _verifier(tmp_path: Path, event_adapter: EventAdapter, question_adapter: QuestionAdapter) -> VisualVerifier:
    return VisualVerifier(cache=ContentAddressedCache(tmp_path), event_adapter=event_adapter,
                          question_adapter=question_adapter, model_version="visual-v1",
                          event_prompt="describe event", question_prompt="verify answer",
                          sampler_version="uniform-v1")


def test_visual_uses_strict_two_level_cache_and_question_cost_metadata(tmp_path: Path) -> None:
    event_adapter, question_adapter = EventAdapter(), QuestionAdapter()
    verifier = _verifier(tmp_path, event_adapter, question_adapter)
    event = _event("e", 0)
    one = verifier.verify_question(event, "What color?", ("red", "blue"), "low")
    two = verifier.verify_question(event, "How many?", ("one", "two"), "low")

    assert len(event_adapter.calls) == 1
    assert len(question_adapter.calls) == 2
    assert len(event_adapter.calls[0]) == 12
    assert one.event_cache_key == two.event_cache_key
    assert one.cache_key != two.cache_key
    assert one.cost_metadata.amortizable is False
    assert two.cost_metadata.charge_scope == "question_verification"


def test_visual_low_high_are_the_only_exact_frame_budgets(tmp_path: Path) -> None:
    event_adapter, question_adapter = EventAdapter(), QuestionAdapter()
    verifier = _verifier(tmp_path, event_adapter, question_adapter)
    event = _event("e", 0)
    low = verifier.observe_event(event, "low")
    high = verifier.observe_event(event, "high")

    assert len(low.frames) == 12
    assert len(high.frames) == 32
    with pytest.raises(ValueError, match="low.*high"):
        verifier.observe_event(event, "medium")  # type: ignore[arg-type]


def test_visual_observation_hit_is_amortizable_and_question_key_normalizes_options(tmp_path: Path) -> None:
    event_adapter, question_adapter = EventAdapter(), QuestionAdapter()
    verifier = _verifier(tmp_path, event_adapter, question_adapter)
    event = _event("e", 0)
    first = verifier.observe_event(event, "low")
    second = verifier.observe_event(event, "low")
    verifier.verify_question(event, "  What COLOR? ", (" A ",), "low")
    repeated = verifier.verify_question(event, "what color?", ("a",), "low")

    assert first.cost_metadata.amortizable is True
    assert second.cost_metadata.reused is True
    assert len(event_adapter.calls) == 1
    assert len(question_adapter.calls) == 1
    assert repeated.cost_metadata.reused is True


def test_context_frontier_expands_each_side_without_hidden_generation() -> None:
    events = (_event("c", 20), _event("a", 0), _event("b", 10, residual="stored residual"), _event("d", 30))
    first = expand_context(events, ContextFrontier(anchor_event_id="c"))
    second = expand_context(events, first.frontier)
    final = expand_context(events, second.frontier)

    assert tuple(item.event_id for item in first.events) == ("b", "d")
    assert first.events[0].residual == "stored residual"
    assert tuple(item.event_id for item in second.events) == ("a",)
    assert final.events == () and final.frontier.exhausted is True


def test_context_frontier_rejects_invalid_topology_and_radii() -> None:
    events = (_event("a", 0), _event("b", 10))
    with pytest.raises(ValueError, match="anchor"):
        expand_context(events, ContextFrontier(anchor_event_id="missing"))
    with pytest.raises(ValueError, match="left_radius"):
        expand_context(events, ContextFrontier(anchor_event_id="a", left_radius=1))
    with pytest.raises(ValueError, match="duplicate"):
        expand_context((_event("a", 0), _event("a", 1)), ContextFrontier(anchor_event_id="a"))
