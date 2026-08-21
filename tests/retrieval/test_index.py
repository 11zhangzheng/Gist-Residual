import pytest

from fidmem.retrieval.index import GistIndex
from fidmem.types import EventRecord


def _record(
    event_id: str,
    start_sec: float,
    text_embedding: tuple[float, ...],
    visual_embedding: tuple[float, ...],
) -> EventRecord:
    return EventRecord(
        video_id="video-1",
        event_id=event_id,
        start_sec=start_sec,
        end_sec=start_sec + 5.0,
        asr_text="speech",
        keyframe_paths=("a.jpg", "b.jpg", "c.jpg", "d.jpg"),
        visual_embedding=visual_embedding,
        text_embedding=text_embedding,
        gist_text=f"gist {event_id}",
        raw_video_uri="video.mp4",
        memory_version="gist-v1",
    )


def _text_query(_: str) -> tuple[float, ...]:
    return (1.0, 0.0)


def _visual_query(_: str) -> tuple[float, ...]:
    return (1.0, 0.0)


def test_search_returns_top_k_with_normalized_component_and_fused_scores() -> None:
    index = GistIndex(
        (
            _record("text-best", 5.0, (1.0, 0.0), (0.0, 1.0)),
            _record("visual-best", 10.0, (0.0, 1.0), (1.0, 0.0)),
            _record("weak-text", 15.0, (-1.0, 0.0), (1.0, 0.0)),
        ),
        text_query_encoder=_text_query,
        visual_query_encoder=_visual_query,
    )

    results = index.search("What happened?", 2)

    assert tuple(result.event.event_id for result in results) == (
        "text-best",
        "visual-best",
    )
    assert results[0].text_score == pytest.approx(1.0)
    assert results[0].visual_score == pytest.approx(0.5)
    assert results[0].score == pytest.approx(0.8)
    assert results[1].text_score == pytest.approx(0.5)
    assert results[1].visual_score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(0.7)


def test_search_ties_are_stable_by_start_then_event_id() -> None:
    records = (
        _record("z", 1.0, (1.0, 0.0), (1.0, 0.0)),
        _record("b", 0.0, (1.0, 0.0), (1.0, 0.0)),
        _record("a", 0.0, (1.0, 0.0), (1.0, 0.0)),
    )
    index = GistIndex(
        records,
        text_query_encoder=_text_query,
        visual_query_encoder=_visual_query,
    )

    first = index.search("question", 3)
    second = index.search("question", 3)

    assert tuple(result.event.event_id for result in first) == ("a", "b", "z")
    assert second == first


def test_empty_index_returns_empty_without_running_encoders() -> None:
    def forbidden(_: str) -> tuple[float, ...]:
        raise AssertionError("an empty index must not encode the query")

    index = GistIndex(
        (), text_query_encoder=forbidden, visual_query_encoder=forbidden
    )

    assert index.search("question", 5) == ()


@pytest.mark.parametrize("k", [0, -1])
def test_search_rejects_non_positive_k(k: int) -> None:
    index = GistIndex(
        (), text_query_encoder=_text_query, visual_query_encoder=_visual_query
    )

    with pytest.raises(ValueError, match="k must be positive"):
        index.search("question", k)


def test_search_rejects_blank_question() -> None:
    index = GistIndex(
        (_record("a", 0.0, (1.0, 0.0), (1.0, 0.0)),),
        text_query_encoder=_text_query,
        visual_query_encoder=_visual_query,
    )

    with pytest.raises(ValueError, match="question must not be blank"):
        index.search("  ", 1)


@pytest.mark.parametrize(
    ("text_embedding", "visual_embedding", "message"),
    [
        ((0.0, 0.0), (1.0, 0.0), "zero vector"),
        ((1.0, 0.0, 0.0), (1.0, 0.0), "dimension"),
    ],
)
def test_search_rejects_invalid_event_embeddings(
    text_embedding: tuple[float, ...],
    visual_embedding: tuple[float, ...],
    message: str,
) -> None:
    index = GistIndex(
        (_record("bad", 0.0, text_embedding, visual_embedding),),
        text_query_encoder=_text_query,
        visual_query_encoder=_visual_query,
    )

    with pytest.raises(ValueError, match=message):
        index.search("question", 1)


def test_search_rejects_zero_query_embedding() -> None:
    index = GistIndex(
        (_record("a", 0.0, (1.0, 0.0), (1.0, 0.0)),),
        text_query_encoder=lambda _: (0.0, 0.0),
        visual_query_encoder=_visual_query,
    )

    with pytest.raises(ValueError, match="zero vector"):
        index.search("question", 1)
