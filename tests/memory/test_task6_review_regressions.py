import json
from pathlib import Path

import pytest

from fidmem.memory.residual import ResidualGenerator
from fidmem.memory.visual import ContextFrontier, VisualVerifier, expand_context
from fidmem.storage.cache import ContentAddressedCache
from fidmem.types import EventRecord


def _event(*, residual: str | None = None, event_id: str = "event") -> EventRecord:
    return EventRecord(
        video_id="video", event_id=event_id, start_sec=0, end_sec=2,
        gist_text="blue bottle", keyframe_paths=tuple(f"f{index}" for index in range(40)),
        raw_video_uri="raw.mp4", memory_version="v1", residual=residual,
    )


def _payload(**overrides: list[str]) -> str:
    fields = {
        "entities": [], "actions": [], "attributes": [], "spatial_relations": [],
        "counts": [], "state_changes": [], "exceptions": [], "unstructured_details": [],
    }
    fields.update(overrides)
    return json.dumps(fields)


class CountingVLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    def __call__(self, frames: tuple[str, ...], prompt: str) -> str:
        self.calls += 1
        return self.response


class CountingEventAdapter:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, frames: tuple[str, ...]) -> str:
        self.calls += 1
        return "generic observation"


def test_online_residual_never_changes_residual_or_visual_event_cache_identity(tmp_path: Path) -> None:
    base, upgraded = _event(), _event(residual="online residual")
    vlm = CountingVLM(_payload(attributes=["green label"]))
    residual = ResidualGenerator(
        cache=ContentAddressedCache(tmp_path / "residual"), vlm=vlm,
        embedder=lambda text: (1.0, 0.0) if text == "blue bottle" else (0.0, 1.0),
        model_version="m1", prompt_template="novel details", frame_sampler=lambda event: event.keyframe_paths,
        frame_sampler_version="s1", schema_version="r1", normalizer_version="n1",
    )
    event_adapter = CountingEventAdapter()
    visual = VisualVerifier(
        cache=ContentAddressedCache(tmp_path / "visual"), event_adapter=event_adapter,
        question_adapter=lambda observation, question, options: "answer", model_version="m1",
        event_prompt="event", question_prompt="verify", sampler_version="s1",
    )

    first_residual = residual.expand(base, base.gist_text)
    second_residual = residual.expand(upgraded, upgraded.gist_text)
    first_observation = visual.observe_event(base, "low")
    second_observation = visual.observe_event(upgraded, "low")

    assert first_residual.cache_key == second_residual.cache_key
    assert first_observation.cache_key == second_observation.cache_key
    assert vlm.calls == 1
    assert event_adapter.calls == 1


def test_visual_cost_metadata_records_only_currently_processed_frames(tmp_path: Path) -> None:
    event_adapter = CountingEventAdapter()
    verifier = VisualVerifier(
        cache=ContentAddressedCache(tmp_path), event_adapter=event_adapter,
        question_adapter=lambda observation, question, options: "answer", model_version="m1",
        event_prompt="event", question_prompt="verify", sampler_version="s1",
    )
    event = _event()
    first = verifier.verify_question(event, "first", ("a",), "low")
    second = verifier.verify_question(event, "second", ("b",), "low")
    repeated = verifier.verify_question(event, "first", ("a",), "low")

    assert (first.event_cost_metadata.cache_status, first.event_cost_metadata.input_frames) == ("miss", 12)
    assert (second.event_cost_metadata.cache_status, second.event_cost_metadata.input_frames) == ("hit", 0)
    assert (repeated.event_cost_metadata.cache_status, repeated.event_cost_metadata.input_frames) == ("hit", 0)
    assert [item.cost_metadata.input_frames for item in (first, second, repeated)] == [0, 0, 0]
    assert [item.cost_metadata.evidence_frame_count for item in (first, second, repeated)] == [12, 12, 12]
    assert sum(item.event_cost_metadata.input_frames + item.cost_metadata.input_frames for item in (first, second, repeated)) == 12


def test_context_rejects_covered_frontier_marked_not_exhausted() -> None:
    events = (_event(event_id="a"), _event(event_id="b").model_copy(update={"start_sec": 3, "end_sec": 4}))
    with pytest.raises(ValueError, match="exhausted"):
        expand_context(events, ContextFrontier(anchor_event_id="a", right_radius=1, exhausted=False))


def test_cross_field_normalized_duplicate_is_removed_before_a_second_embedding(tmp_path: Path) -> None:
    calls: list[str] = []
    def embedder(text: str) -> tuple[float, ...]:
        calls.append(text)
        return (1.0, 0.0)
    generator = ResidualGenerator(
        cache=ContentAddressedCache(tmp_path), vlm=CountingVLM(_payload(entities=["green label"], attributes=[" Green  LABEL "])),
        embedder=embedder, model_version="m1", prompt_template="novel details",
        frame_sampler=lambda event: event.keyframe_paths, frame_sampler_version="s1",
        schema_version="r1", normalizer_version="n1",
    )

    result = generator.expand(_event(), "")

    assert result.payload.entities == ("green label",)
    assert result.payload.attributes == ()
    assert result.audit.filtered_exact == 1
    assert calls == ["green label"]
