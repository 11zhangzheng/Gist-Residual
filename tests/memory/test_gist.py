from pathlib import Path

import pytest

from fidmem.memory.gist import GistBuilder, GistEventInput
from fidmem.storage.cache import ContentAddressedCache

Token = str | int


class WhitespaceTokenizer:
    def __init__(self, identity: str = "whitespace-v1") -> None:
        self.identity = identity

    def encode(self, text: str) -> tuple[Token, ...]:
        return tuple(text.split())

    def decode(self, tokens: tuple[Token, ...]) -> str:
        return " ".join(str(token) for token in tokens)


class CharacterTokenizer:
    identity = "character-v1"

    def encode(self, text: str) -> tuple[Token, ...]:
        return tuple(text)

    def decode(self, tokens: tuple[Token, ...]) -> str:
        return "".join(str(token) for token in tokens)


class ForbiddenVLM:
    def __call__(
        self, frames: tuple[str, ...], asr_text: str, max_tokens: int
    ) -> str:
        raise AssertionError("the main Gist path must not call a VLM")


class RecordingSummarizer:
    def __init__(self, result: str) -> None:
        self.result = result
        self.calls: list[tuple[str, int, str]] = []

    def __call__(
        self,
        text: str,
        max_tokens: int,
        tokenizer: WhitespaceTokenizer | CharacterTokenizer,
    ) -> str:
        self.calls.append((text, max_tokens, tokenizer.identity))
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
    def __init__(self, result: str = "visual detail") -> None:
        self.result = result
        self.calls: list[tuple[tuple[str, ...], str, int]] = []

    def __call__(
        self, frames: tuple[str, ...], asr_text: str, max_tokens: int
    ) -> str:
        self.calls.append((frames, asr_text, max_tokens))
        return self.result


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
    tokenizer: WhitespaceTokenizer | CharacterTokenizer | None = None,
    cache: ContentAddressedCache | None = None,
    vlm: object | None = None,
    mode: str = "main",
    namespace: str = "gist",
    max_tokens: int = 40,
    visual_resolution: tuple[int, int] = (224, 224),
) -> tuple[
    GistBuilder, RecordingSummarizer, RecordingVisualEncoder, CountingTextEncoder
]:
    actual_summarizer = summarizer or RecordingSummarizer("man opens door")
    actual_visual_encoder = visual_encoder or RecordingVisualEncoder()
    actual_text_encoder = text_encoder or CountingTextEncoder()
    builder = GistBuilder(
        cache=cache or ContentAddressedCache(tmp_path),
        summarizer=actual_summarizer,
        tokenizer=tokenizer or WhitespaceTokenizer(),
        text_encoder=actual_text_encoder,
        visual_encoder=actual_visual_encoder,
        vlm=vlm,
        model_version="gist-model-v1",
        prompt="summarize the event",
        namespace=namespace,
        mode=mode,
        max_tokens=max_tokens,
        visual_resolution=visual_resolution,
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


def test_builder_caps_summary_with_the_tokenizer_shared_with_summarizer(
    tmp_path: Path,
) -> None:
    tokenizer = WhitespaceTokenizer("summary-tokenizer-v3")
    summarizer = RecordingSummarizer(" ".join(f"t{index}" for index in range(45)))
    visual_encoder = RecordingVisualEncoder()
    builder, _, _, _ = _builder(
        tmp_path,
        summarizer=summarizer,
        visual_encoder=visual_encoder,
        tokenizer=tokenizer,
    )

    record = builder.build(_event())

    assert len(tokenizer.encode(record.gist_text)) == 40
    assert record.gist_text.endswith("t39")
    assert summarizer.calls == [
        ("a man opens the door", 40, "summary-tokenizer-v3")
    ]
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


def test_cache_key_separates_event_identity_budget_and_tokenizer(tmp_path: Path) -> None:
    cache = ContentAddressedCache(tmp_path)
    summary = " ".join(f"t{index}" for index in range(10))
    event_one = _event()
    event_two = event_one.model_copy(update={"event_id": "event-2"})
    event_three = event_one.model_copy(update={"video_id": "video-2"})
    full, _, _, _ = _builder(
        tmp_path,
        cache=cache,
        summarizer=RecordingSummarizer(summary),
        tokenizer=WhitespaceTokenizer("tokenizer-v1"),
        max_tokens=40,
    )
    short, _, _, _ = _builder(
        tmp_path,
        cache=cache,
        summarizer=RecordingSummarizer(summary),
        tokenizer=WhitespaceTokenizer("tokenizer-v1"),
        max_tokens=5,
    )
    other_tokenizer, _, _, _ = _builder(
        tmp_path,
        cache=cache,
        summarizer=RecordingSummarizer(summary),
        tokenizer=WhitespaceTokenizer("tokenizer-v2"),
        max_tokens=40,
    )

    first = full.build(event_one)
    second = full.build(event_two)
    third = full.build(event_three)
    shortened = short.build(event_one)
    retokenized = other_tokenizer.build(event_one)

    assert (first.video_id, first.event_id) == ("video-1", "event-1")
    assert (second.video_id, second.event_id) == ("video-1", "event-2")
    assert (third.video_id, third.event_id) == ("video-2", "event-1")
    assert len(shortened.gist_text.split()) == 5
    assert retokenized.event_id == "event-1"
    assert len(tuple(tmp_path.glob("*.json"))) == 5


def test_gist_plus_uses_vlm_and_an_independent_cache_namespace(tmp_path: Path) -> None:
    shared_cache = ContentAddressedCache(tmp_path)
    tokenizer = WhitespaceTokenizer()
    summarizer = RecordingSummarizer("speech detail")
    visual_encoder = RecordingVisualEncoder()
    text_encoder = CountingTextEncoder()
    vlm = RecordingVLM()
    main = GistBuilder(
        cache=shared_cache,
        summarizer=summarizer,
        tokenizer=tokenizer,
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
        tokenizer=tokenizer,
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

    assert len(vlm.calls) == 1
    assert main_record.gist_text == "speech detail"
    assert plus_record.gist_text == "visual detail speech detail"
    assert len(tuple(tmp_path.glob("*.json"))) == 2


def test_gist_plus_reserves_visual_tokens_when_speech_fills_main_budget(
    tmp_path: Path,
) -> None:
    tokenizer = WhitespaceTokenizer()
    summarizer = RecordingSummarizer(" ".join(f"speech{n}" for n in range(40)))
    vlm = RecordingVLM("blue bottle behind carton")
    builder, _, _, _ = _builder(
        tmp_path,
        summarizer=summarizer,
        tokenizer=tokenizer,
        vlm=vlm,
        mode="gist_plus",
        namespace="gist_plus",
    )

    record = builder.build(_event())
    tokens = tokenizer.encode(record.gist_text)

    assert len(tokens) <= 40
    assert tokens[:4] == ("blue", "bottle", "behind", "carton")
    assert summarizer.calls == [("a man opens the door", 28, "whitespace-v1")]
    assert vlm.calls[0][2] == 12


def test_builder_uses_subword_tokenizer_instead_of_whitespace_count(
    tmp_path: Path,
) -> None:
    tokenizer = CharacterTokenizer()
    summarizer = RecordingSummarizer("红色瓶子" * 15)
    builder, _, _, _ = _builder(
        tmp_path, summarizer=summarizer, tokenizer=tokenizer
    )

    record = builder.build(_event(asr_text="他拿起红色瓶子"))

    assert len(tokenizer.encode(record.gist_text)) == 40
    assert record.gist_text == ("红色瓶子" * 15)[:40]
    assert summarizer.calls == [("他拿起红色瓶子", 40, "character-v1")]


def test_builder_rejects_fewer_than_four_source_frames(tmp_path: Path) -> None:
    builder, _, _, _ = _builder(tmp_path)
    event = _event().model_copy(update={"keyframe_paths": ("a.jpg", "b.jpg", "c.jpg")})

    with pytest.raises(ValueError, match="at least four"):
        builder.build(event)


def test_builder_rejects_noncanonical_visual_resolution(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly 224x224"):
        _builder(tmp_path, visual_resolution=(112, 112))


def test_builder_rejects_summary_budgets_above_hard_cap(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at most 40"):
        GistBuilder(
            cache=ContentAddressedCache(tmp_path),
            summarizer=RecordingSummarizer("summary"),
            tokenizer=WhitespaceTokenizer(),
            text_encoder=CountingTextEncoder(),
            visual_encoder=RecordingVisualEncoder(),
            model_version="gist-model-v1",
            prompt="summarize",
            max_tokens=41,
        )


def test_builder_requires_a_tokenizer_adapter(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="tokenizer"):
        GistBuilder(  # type: ignore[call-arg]
            cache=ContentAddressedCache(tmp_path),
            summarizer=RecordingSummarizer("summary"),
            text_encoder=CountingTextEncoder(),
            visual_encoder=RecordingVisualEncoder(),
            model_version="gist-model-v1",
            prompt="summarize",
        )
