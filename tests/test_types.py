import pytest
from pydantic import ValidationError

from fidmem.types import ActionInstance, ActionType, EventRecord, FidelityLevel, RouterState


def test_event_record_serializes_canonical_gist_fields_and_is_frozen() -> None:
    record = EventRecord(
        video_id="video-1",
        event_id="event-1",
        start_sec=1.5,
        end_sec=4.0,
        asr_text="a person enters",
        keyframe_paths=("f0.jpg", "f1.jpg", "f2.jpg", "f3.jpg"),
        visual_embedding=(1.0, 0.0),
        text_embedding=(0.0, 1.0),
        gist_text="person enters",
        raw_video_uri="video.mp4",
        memory_version="gist-v1",
    )

    assert record.model_dump() == {
        "video_id": "video-1",
        "event_id": "event-1",
        "start_sec": 1.5,
        "end_sec": 4.0,
        "asr_text": "a person enters",
        "keyframe_paths": ("f0.jpg", "f1.jpg", "f2.jpg", "f3.jpg"),
        "visual_embedding": (1.0, 0.0),
        "text_embedding": (0.0, 1.0),
        "gist_text": "person enters",
        "raw_video_uri": "video.mp4",
        "memory_version": "gist-v1",
        "residual": None,
    }
    assert record.start_seconds == 1.5
    assert record.end_seconds == 4.0
    assert record.gist == "person enters"
    with pytest.raises(ValidationError, match="frozen"):
        record.gist_text = "changed"  # type: ignore[misc]


def test_event_record_accepts_legacy_names_but_serializes_canonical_names() -> None:
    record = EventRecord(
        event_id="legacy",
        video_id="video-1",
        start_seconds=0.0,
        end_seconds=2.0,
        gist="legacy gist",
    )

    assert record.start_sec == 0.0
    assert record.end_sec == 2.0
    assert record.gist_text == "legacy gist"
    assert "start_seconds" not in record.model_dump()
    assert "gist" not in record.model_dump()


def test_event_record_rejects_reversed_canonical_time_range() -> None:
    with pytest.raises(ValidationError, match="end_sec"):
        EventRecord(
            event_id="bad",
            video_id="video-1",
            start_sec=2.0,
            end_sec=1.0,
            gist_text="bad range",
        )


def test_router_state_rejects_unknown_candidate_fidelity() -> None:
    state = RouterState(
        question="What color is the bottle?",
        options=("red", "blue"),
        evidence=(),
        action_history=(),
        remaining_budget=1.0,
        candidate_event_ids=("e1",),
        candidate_fidelity_levels={"e1": FidelityLevel.GIST},
        context_frontiers={"e1": (0, 0)},
        cost_preference=0.3,
    )

    assert state.candidate_fidelity_levels["e1"] is FidelityLevel.GIST
    assert ActionInstance(ActionType.EXPAND_RESIDUAL, "e1", None).event_id == "e1"


def test_router_state_rejects_candidate_metadata_for_unknown_event() -> None:
    with pytest.raises(ValidationError, match="candidate_event_ids"):
        RouterState(
            question="What color is the bottle?",
            options=("red", "blue"),
            evidence=(),
            action_history=(),
            remaining_budget=1.0,
            candidate_event_ids=("e1",),
            candidate_fidelity_levels={"e2": FidelityLevel.GIST},
            context_frontiers={"e1": (0, 0)},
            cost_preference=0.3,
        )


def test_router_state_rejects_negative_budget() -> None:
    with pytest.raises(ValidationError, match="remaining_budget"):
        RouterState(
            question="What color is the bottle?",
            options=("red", "blue"),
            evidence=(),
            action_history=(),
            remaining_budget=-0.1,
            candidate_event_ids=(),
            candidate_fidelity_levels={},
            context_frontiers={},
            cost_preference=0.3,
        )
