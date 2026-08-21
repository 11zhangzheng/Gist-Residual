import json
from pathlib import Path

import pytest

from fidmem.memory.residual import ResidualGenerator
from fidmem.storage.cache import ContentAddressedCache
from fidmem.types import EventRecord


class RecordingVLM:
    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.calls: list[tuple[tuple[str, ...], str]] = []

    def __call__(self, frames: tuple[str, ...], prompt: str) -> str:
        self.calls.append((frames, prompt))
        return self.replies.pop(0)


class RecordingRepair:
    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.calls: list[str] = []

    def __call__(self, response: str) -> str:
        self.calls.append(response)
        return self.replies.pop(0)


class MappingEmbedder:
    identity = "mapping-embedder-v1"

    def __init__(self, values: dict[str, tuple[float, ...]]) -> None:
        self.values = values

    def __call__(self, text: str) -> tuple[float, ...]:
        if text in self.values:
            return self.values[text]
        if text in {"man opens door", "blue bottle"}:
            return (1.0, 0.0)
        return (0.0, 1.0)


def _event(event_id: str = "event-1") -> EventRecord:
    return EventRecord(
        video_id="video-1", event_id=event_id, start_sec=1, end_sec=5,
        gist_text="man opens door; blue bottle", keyframe_paths=("a.jpg", "b.jpg"),
        raw_video_uri="video.mp4", memory_version="memory-v1",
    )


def _payload(**overrides: list[str]) -> str:
    values = {
        "entities": [], "actions": [], "attributes": [], "spatial_relations": [],
        "counts": [], "state_changes": [], "exceptions": [], "unstructured_details": [],
    }
    values.update(overrides)
    return json.dumps(values)


def _generator(tmp_path: Path, *, vlm: RecordingVLM, repair: RecordingRepair | None = None,
               embedder: MappingEmbedder | None = None, cache: ContentAddressedCache | None = None) -> ResidualGenerator:
    return ResidualGenerator(
        cache=cache or ContentAddressedCache(tmp_path), vlm=vlm, json_repair=repair,
        embedder=embedder or MappingEmbedder({}), model_version="residual-v1",
        prompt_template="Extract only novel visual event details.", frame_sampler=lambda event: event.keyframe_paths,
        frame_sampler_version="sample-v1", schema_version="schema-v1", normalizer_version="norm-v1",
    )


def test_residual_has_exact_schema_and_question_independent_prompt(tmp_path: Path) -> None:
    vlm = RecordingVLM([_payload(attributes=["green label"], actions=["lifts bottle"])])
    result = _generator(tmp_path, vlm=vlm, embedder=MappingEmbedder({"lifts bottle": (0.0, -1.0), "green label": (0.0, 1.0)})).expand(
        _event(), "man opens door; blue bottle")

    assert set(result.payload.model_dump()) == {
        "entities", "actions", "attributes", "spatial_relations", "counts", "state_changes", "exceptions", "unstructured_details",
    }
    assert result.payload.attributes == ("green label",)
    assert "man opens door; blue bottle" in vlm.calls[0][1]
    assert "only novel" in vlm.calls[0][1].lower()
    assert "question" not in vlm.calls[0][1].lower()


def test_residual_repairs_once_then_records_schema_error(tmp_path: Path) -> None:
    repair = RecordingRepair([_payload(attributes=["green label"]), "still not json"])
    first_vlm = RecordingVLM(["not json"])
    repaired = _generator(tmp_path / "one", vlm=first_vlm, repair=repair).expand(_event(), _event().gist_text)
    failed_vlm = RecordingVLM(["not json"])
    failed = _generator(tmp_path / "two", vlm=failed_vlm, repair=RecordingRepair(["also broken"])).expand(_event(), _event().gist_text)

    assert repaired.schema_error is None
    assert repaired.payload.attributes == ("green label",)
    assert len(repair.calls) == 1
    assert failed.schema_error is not None
    assert failed.audit.repair_attempted is True
    assert failed.payload.model_dump() == {key: () for key in failed.payload.model_dump()}


def test_residual_deduplicates_exact_and_semantic_gist_and_residual_items(tmp_path: Path) -> None:
    embedder = MappingEmbedder({
        "blue bottle": (1.0, 0.0), "azure bottle": (0.99, 0.01),
        "green label": (0.0, 1.0), "emerald label": (0.0, 0.99),
    })
    result = _generator(
        tmp_path, embedder=embedder,
        vlm=RecordingVLM([_payload(attributes=["Blue  bottle", "azure bottle", "green label", "emerald label"])]),
    ).expand(_event(), "blue bottle")

    assert result.payload.attributes == ("green label",)
    assert result.audit.filtered_exact == 1
    assert result.audit.filtered_gist_semantic == 1
    assert result.audit.filtered_residual_semantic == 1


def test_residual_embedding_validation_keeps_threshold_boundary_precise(tmp_path: Path) -> None:
    result = _generator(
        tmp_path, embedder=MappingEmbedder({"blue bottle": (1.0, 0.0), "edge detail": (0.92, (1 - 0.92**2) ** 0.5)}),
        vlm=RecordingVLM([_payload(attributes=["edge detail"])]),
    ).expand(_event(), "blue bottle")
    assert result.payload.attributes == ()

    bad = _generator(
        tmp_path / "bad", embedder=MappingEmbedder({"blue bottle": (0.0, 0.0), "detail": (1.0, 0.0)}),
        vlm=RecordingVLM([_payload(attributes=["detail"])]),
    )
    with pytest.raises(ValueError, match="zero vector"):
        bad.expand(_event(), "blue bottle")


def test_residual_cache_is_event_identity_and_version_sensitive(tmp_path: Path) -> None:
    cache = ContentAddressedCache(tmp_path)
    vlm = RecordingVLM([_payload(attributes=["detail one"]), _payload(attributes=["detail two"])])
    generator = _generator(tmp_path, vlm=vlm, cache=cache)
    one = generator.expand(_event("one"), _event("one").gist_text)
    again = generator.expand(_event("one"), _event("one").gist_text)
    two = generator.expand(_event("two"), _event("two").gist_text)

    assert one.payload.attributes == ("detail one",)
    assert again == one
    assert two.payload.attributes == ("detail two",)
    assert len(vlm.calls) == 2
