from pathlib import Path

import pytest

from fidmem.memory.gist import GistBuilder, GistEventInput
from fidmem.storage.cache import ContentAddressedCache


class ForbiddenVLM:
    def __call__(
        self, frames: tuple[str, ...], asr_text: str, max_tokens: int
    ) -> str:
        raise AssertionError("the main Gist path must not call a VLM")


class RecordingSummarizer:
    def __init__(self, result: str) -> None:
        self.result = result
        self.calls: list[tuple[str, int]] = []

    def __call__(self, text: str, max_tokens: int) -> str:
        self.calls.append((text, max_tokens))
        return self.result


class RecordingVisualEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], tuple[int, int]]] = []

    def __call__(
        self, frames: tuple[str, ...], resolution: tuple[int, int]
    ) -> tuple[float, ...]:
        self.calls.append((frames, resolution))
        return (0.25, 0.75)


class CountingTextEncoder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, text: str) -> tuple[float, ...]:
        self.calls.append(text)
        return (0.75, 0.25)


class RecordingVLM:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self, frames: tuple[str, ...], asr_text: str, max_tokens: int
    ) -> str:
        self.calls += 1
        return "visual detail"


def _event(*, asr_text: str | None = "a man opens the door") -> GistEventInput:
    return GistEventInput(
        video_id="video-1",
        event_id="event-1",
        start_sec=10.0,
        end_sec=30.0,
        asr_text=asr_text,
        keyframe_paths=tuple(f"frame-{index}.jpg" for index in range(8)),
        raw_video_uri="video.mp4",
        video_hash="abc123",
        memory_version="memory-v1",
    )


def _builder(
    tmp_path: Path,
    *,
    summarizer: RecordingSummarizer | None = None,
    visual_encoder: RecordingVisualEncoder | None = None,
    text_encoder: CountingTextEncoder | None = None,
    vlm: object | None = None,
    mode: str = "main",
    namespace: str = "gist",
) -> tuple[
    GistBuilder, RecordingSummarizer, RecordingVisualEncoder, CountingTextEncoder
]:
    actual_summarizer = summarizer or RecordingSummarizer("man opens door")
    actual_visual_encoder = visual_encoder or RecordingVisualEncoder()
    actual_text_encoder = text_encoder or CountingTextEncoder()
    builder = GistBuilder(
        cache=ContentAddressedCache(tmp_path),
        summarizer=actual_summarizer,
        text_encoder=actual_text_encoder,
        visual_encoder=actual_visual_encoder,
        vlm=vlm,
        model_version="gist-model-v1",
        prompt="summarize the event",
        namespace=namespace,
        mode=mode,
    )
    return builder, actual_summarizer, actual_visual_encoder, actual_text_encoder


def test_main_gist_never_calls_vlm_and_silent_event_keeps_visual_embedding(
    tmp_path: Path,
) -> None:
    builder, summarizer, visual_encoder, text_encoder = _builder(
        tmp_path, vlm=ForbiddenVLM()
    )

    record = builder.build(_event(asr_text=None))

    assert record.gist_text == "[no speech]"
    assert record.asr_text == "[no speech]"
    assert record.visual_embedding == (0.25, 0.75)
    assert record.text_embedding == (0.75, 0.25)
    assert summarizer.calls == []
    assert text_encoder.calls == ["[no speech]"]
    assert len(visual_encoder.calls) == 1


def test_builder_caps_summary_and_encodes_exactly_four_low_resolution_frames(
    tmp_path: Path,
) -> None:
    summarizer = RecordingSummarizer(" ".join(f"t{index}" for index in range(45)))
    visual_encoder = RecordingVisualEncoder()
    builder, _, _, _ = _builder(
        tmp_path, summarizer=summarizer, visual_encoder=visual_encoder
    )

    record = builder.build(_event())

    assert len(record.gist_text.split()) == 40
    assert record.gist_text.endswith("t39")
    assert summarizer.calls == [("a man opens the door", 40)]
    sampled_frames, resolution = visual_encoder.calls[0]
    assert len(sampled_frames) == 4
    assert resolution == (224, 224)
    assert record.keyframe_paths == sampled_frames


def test_builder_reuses_content_addressed_result_without_recomputing(
    tmp_path: Path,
) -> None:
    builder, summarizer, visual_encoder, text_encoder = _builder(tmp_path)

    first = builder.build(_event())
    second = builder.build(_event())

    assert second == first
    assert len(summarizer.calls) == 1
    assert len(visual_encoder.calls) == 1
    assert len(text_encoder.calls) == 1


def test_gist_plus_uses_vlm_and_an_independent_cache_namespace(tmp_path: Path) -> None:
    shared_cache = ContentAddressedCache(tmp_path)
    summarizer = RecordingSummarizer("speech detail")
    visual_encoder = RecordingVisualEncoder()
    text_encoder = CountingTextEncoder()
    vlm = RecordingVLM()
    main = GistBuilder(
        cache=shared_cache,
        summarizer=summarizer,
        text_encoder=text_encoder,
        visual_encoder=visual_encoder,
        vlm=vlm,
        model_version="shared-model-v1",
        prompt="summarize",
        namespace="gist",
        mode="main",
    )
    plus = GistBuilder(
        cache=shared_cache,
        summarizer=summarizer,
        text_encoder=text_encoder,
        visual_encoder=visual_encoder,
        vlm=vlm,
        model_version="shared-model-v1",
        prompt="summarize",
        namespace="gist_plus",
        mode="gist_plus",
    )

    main_record = main.build(_event())
    plus_record = plus.build(_event())

    assert vlm.calls == 1
    assert main_record.gist_text == "speech detail"
    assert plus_record.gist_text == "speech detail visual detail"
    assert len(tuple(tmp_path.glob("*.json"))) == 2


def test_builder_rejects_fewer_than_four_source_frames(tmp_path: Path) -> None:
    builder, _, _, _ = _builder(tmp_path)
    event = _event().model_copy(update={"keyframe_paths": ("a.jpg", "b.jpg", "c.jpg")})

    with pytest.raises(ValueError, match="at least four"):
        builder.build(event)


def test_builder_rejects_summary_budgets_above_hard_cap(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at most 40"):
        GistBuilder(
            cache=ContentAddressedCache(tmp_path),
            summarizer=RecordingSummarizer("summary"),
            text_encoder=CountingTextEncoder(),
            visual_encoder=RecordingVisualEncoder(),
            model_version="gist-model-v1",
            prompt="summarize",
            max_tokens=41,
        )
