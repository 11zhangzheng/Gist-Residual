import json
from pathlib import Path

import pytest

from fidmem.memory.residual import ResidualGenerator
from fidmem.memory.visual import ContextFrontier, VisualVerifier, expand_context
from fidmem.storage.cache import ContentAddressedCache
from fidmem.types import EventRecord


def _event(event_id: str = "event") -> EventRecord:
    return EventRecord(
        video_id="video", event_id=event_id, start_sec=0, end_sec=2,
        gist_text="blue bottle", keyframe_paths=tuple(f"f{n}" for n in range(40)),
        raw_video_uri="raw.mp4", memory_version="v1",
    )


def _payload() -> str:
    return json.dumps({
        "entities": [], "actions": [], "attributes": ["green label"], "spatial_relations": [],
        "counts": [], "state_changes": [], "exceptions": [], "unstructured_details": [],
    })


def test_residual_rejects_embedding_dimension_mismatch(tmp_path: Path) -> None:
    generator = ResidualGenerator(
        cache=ContentAddressedCache(tmp_path), vlm=lambda frames, prompt: _payload(),
        embedder=lambda text: (1.0,) if text == "blue bottle" else (1.0, 0.0),
        model_version="m1", prompt_template="novel details", frame_sampler=lambda event: event.keyframe_paths,
        frame_sampler_version="s1", schema_version="r1", normalizer_version="n1",
    )

    with pytest.raises(ValueError, match="dimensions"):
        generator.expand(_event(), "blue bottle")


def test_residual_cache_key_changes_with_model_and_normalizer_versions(tmp_path: Path) -> None:
    cache = ContentAddressedCache(tmp_path)
    calls: list[str] = []
    def vlm(frames: tuple[str, ...], prompt: str) -> str:
        calls.append(prompt)
        return _payload()
    shared = dict(cache=cache, vlm=vlm, embedder=lambda text: (1.0, 0.0) if text == "blue bottle" else (0.0, 1.0),
                  prompt_template="novel details", frame_sampler=lambda event: event.keyframe_paths,
                  frame_sampler_version="s1", schema_version="r1")
    one = ResidualGenerator(**shared, model_version="m1", normalizer_version="n1")
    two = ResidualGenerator(**shared, model_version="m2", normalizer_version="n2")

    assert one.expand(_event(), "blue bottle").cache_key != two.expand(_event(), "blue bottle").cache_key
    assert len(calls) == 2


def test_visual_event_key_never_contains_question_or_options(tmp_path: Path) -> None:
    verifier = VisualVerifier(
        cache=ContentAddressedCache(tmp_path), event_adapter=lambda frames: "generic",
        question_adapter=lambda observation, question, options: "answer", model_version="m1",
        event_prompt="event prompt", question_prompt="question prompt", sampler_version="s1",
    )
    event = _event()
    event_key = verifier.observe_event(event, "low").cache_key
    first = verifier.verify_question(event, "first question", ("one",), "low")
    second = verifier.verify_question(event, "second question", ("two",), "low")

    assert event_key == first.event_cache_key == second.event_cache_key
    assert first.cache_key != second.cache_key


def test_context_rejects_inconsistent_exhausted_frontier() -> None:
    with pytest.raises(ValueError, match="exhausted"):
        expand_context((_event("a"), _event("b").model_copy(update={"start_sec": 3, "end_sec": 4})),
                       ContextFrontier(anchor_event_id="a", exhausted=True))
