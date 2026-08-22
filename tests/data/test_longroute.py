from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Callable

import duckdb
import pytest

from fidmem.data.leakage import LeakageAuditor, VideoAsset
from fidmem.data.longroute import (
    DefaultContactSheetValidator,
    LongRouteBuilder,
    LongRouteConfig,
    LongRouteDataError,
    LongRouteLeakageError,
    PublicationBackend,
    SourceEvent,
    SourceQuestion,
    SourceVideo,
    TrainSplitManifest,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(root: Path, *, videos: int = 12, seconds: float = 61.0) -> TrainSplitManifest:
    root.mkdir(parents=True, exist_ok=True)
    source_videos = []
    for index in range(videos):
        path = root / f"video-{index}.bin"
        path.write_bytes(f"video-{index}".encode())
        events = (
            SourceEvent(event_id=f"e-{index}-a", start_sec=0, end_sec=seconds, label=f"action {index}", embedding=(1.0, float(index + 1))),
            SourceEvent(event_id=f"e-{index}-b", start_sec=seconds, end_sec=seconds * 2, label=f"finish {index}", embedding=(1.0, float(index + 1)), attributes={"colour": "red" if index % 2 else "blue"}),
        )
        source_videos.append(
            SourceVideo(
                video_id=f"v-{index}", path=path,
                source_uri=f"https://datasets.example/videos/v-{index}.mp4",
                content_sha256=_sha256(path), split="train", licensed=True,
                frame_embeddings=((1.0, float(index + 1)),) * 8,
                events=events,
                questions=(SourceQuestion(question_id=f"q-{index}", question=f"What happens in {index}?", options=("yes", "no"), answer="yes", target_event_id=events[0].event_id),),
            )
        )
    return TrainSplitManifest(
        name="nextqa", dataset_version="1.0", source_uri="https://datasets.example/nextqa/v1",
        license="CC-BY-4.0", split="train", videos=tuple(source_videos),
    )


def _builder(
    root: Path,
    manifest: TrainSplitManifest | tuple[TrainSplitManifest, ...],
    *,
    audit_size: int = 1,
    publication_backend: PublicationBackend | None = None,
    contact_sheet_validator: object | None = None,
    **config: object,
) -> LongRouteBuilder:
    def sheet(example: object) -> Path:
        path = root / "sheets" / f"{getattr(example, 'question_id')}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"contact-sheet")
        return path

    return LongRouteBuilder(
        manifest if isinstance(manifest, tuple) else (manifest,),
        eval_assets=(),
        leakage_auditor=LeakageAuditor(root / "leakage.parquet"),
        config=LongRouteConfig(output_dir=root / "output", audit_size=audit_size, **config),
        contact_sheet_provider=sheet,
        publication_backend=publication_backend,
        contact_sheet_validator=contact_sheet_validator,
    )


class FaultingPublicationBackend(PublicationBackend):
    """Inject one filesystem failure while preserving the real local backend."""

    def __init__(self, output_root: Path, operation: str, ordinal: int) -> None:
        super().__init__(output_root)
        self.operation = operation
        self.ordinal = ordinal
        self.counts: dict[str, int] = {}

    def _run_operation(self, name: str, action: Callable[[], object]) -> object:
        self.counts[name] = self.counts.get(name, 0) + 1
        if name == self.operation and self.counts[name] == self.ordinal:
            raise OSError(f"injected {name} #{self.ordinal}")
        return super()._run_operation(name, action)

    def _before_pointer_replace(self, *args: object) -> None:
        del args
        self._run_operation(self.POINTER_WRITE, lambda: None)


class RecordingContactSheetValidator:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def validate(self, uri: str | Path) -> str:
        canonical = str(uri)
        self.calls.append(canonical)
        return canonical


def test_same_seed_is_byte_identical_despite_input_order_and_keeps_sources_grouped(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    reversed_manifest = manifest.model_copy(update={"videos": tuple(reversed(manifest.videos))})
    first = _builder(tmp_path / "one", manifest).build(11)
    second = _builder(tmp_path / "two", reversed_manifest).build(11)

    assert first.canonical_json() == second.canonical_json()
    groups = first.group_assignment
    assert all(groups[example.target_source_video_id] == example.split for example in first.examples)
    assert all(groups[segment.source_video_id] == example.split for example in first.examples for segment in example.segments)


def test_different_seed_changes_deterministic_target_positions(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    one = _builder(tmp_path / "one", manifest).build(1)
    two = _builder(tmp_path / "two", manifest).build(2)

    assert [item.target_position for item in one.examples] != [item.target_position for item in two.examples]


def test_eval_duplicate_writes_machine_audit_and_never_writes_manifest(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    copied = tmp_path / "eval.bin"
    copied.write_bytes((tmp_path / "video-0.bin").read_bytes())
    builder = LongRouteBuilder(
        (manifest,),
        eval_assets=(VideoAsset("held-out", copied, ((1.0, 0.0),) * 8),),
        leakage_auditor=LeakageAuditor(tmp_path / "leakage.parquet"),
        config=LongRouteConfig(output_dir=tmp_path / "output", audit_size=1),
        contact_sheet_provider=lambda example: "sheet.jpg",
    )

    with pytest.raises(LongRouteLeakageError):
        builder.build(9)

    audit = json.loads((tmp_path / "output" / "leakage-audit.json").read_text())
    assert audit["complete"] is False
    assert audit["findings"][0]["kind"] == "hash_duplicate"
    assert not (tmp_path / "output" / "longroute-manifest.json").exists()


@pytest.mark.parametrize("distractors", (9, 19))
def test_segment_count_boundaries_and_offsets_are_contiguous(tmp_path: Path, distractors: int) -> None:
    manifest = _manifest(tmp_path, videos=distractors + 1)
    result = _builder(tmp_path / str(distractors), manifest, min_distractors=distractors, max_distractors=distractors).build(3)
    example = result.examples[0]
    assert len(example.segments) == distractors + 1
    assert 600 <= example.duration_sec <= 3600
    assert example.segments[0].global_start_sec == 0
    assert all(left.global_end_sec == right.global_start_sec for left, right in zip(example.segments, example.segments[1:]))
    assert example.segments[-1].global_end_sec == example.duration_sec


def test_fails_instead_of_exceeding_nineteen_distractors_when_ten_minutes_is_impossible(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, videos=20, seconds=10)
    with pytest.raises(LongRouteDataError, match="10 minutes"):
        _builder(tmp_path / "out", manifest).build(3)


def test_nearest_neighbor_order_is_stable_and_multi_event_answers_are_programmatic(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    result = _builder(tmp_path / "out", manifest, multi_event_ratio=0.25).build(8)
    multi = [example for example in result.examples if example.template != "single_event"]
    assert 0.2 <= len(multi) / len(result.examples) <= 0.3
    assert all(len(item.supporting_event_ids) >= 2 for item in multi)
    assert all(item.answer in item.options for item in multi)
    # Equal cosine similarities are resolved by canonical event id, not input order.
    first = result.examples[0]
    non_target = [segment.event_id for segment in first.segments if f"{segment.source_video_id}:{segment.event_id}" != first.target_event_id]
    assert non_target == sorted(non_target)


def test_audit_bundle_has_exact_size_stable_fields_and_insufficient_input_fails(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    result = _builder(tmp_path / "good", manifest, audit_size=3).build(6)
    current = json.loads((tmp_path / "good" / "output" / "current-generation.json").read_text())
    bundle = tmp_path / "good" / "output" / current["generation"] / "audit"
    assert len((bundle / "samples.jsonl").read_text().splitlines()) == 3
    with (bundle / "review.csv").open(newline="", encoding="utf-8") as source:
        assert csv.DictReader(source).fieldnames == ["question_id", "valid", "invalid", "reason"]
    assert all(item.audit_status == "pending" for item in result.examples)
    with pytest.raises(LongRouteDataError, match="audit"):
        _builder(tmp_path / "small", _manifest(tmp_path / "small-src", videos=12), audit_size=11).build(6)


def test_multi_event_supports_only_canonical_ids_present_in_route_and_immediate_order(tmp_path: Path) -> None:
    result = _builder(tmp_path / "out", _manifest(tmp_path), multi_event_ratio=0.25).build(8)
    multi = next(item for item in result.examples if item.template != "single_event")
    route = {f"{segment.source_video_id}:{segment.event_id}": segment for segment in multi.segments}
    assert set(multi.supporting_event_ids) <= set(route)
    assert multi.target_event_id in route
    left, right = (route[event_id] for event_id in multi.supporting_event_ids)
    assert right.global_start_sec == left.global_end_sec
    assert multi.answer in multi.options


def test_missing_centroid_coverage_fails_closed_and_missing_contact_sheet_is_rejected(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    strict = LongRouteBuilder(
        (manifest,), eval_assets=(VideoAsset("eval", tmp_path / "video-1.bin"),),
        leakage_auditor=LeakageAuditor(tmp_path / "audit.parquet"),
        config=LongRouteConfig(output_dir=tmp_path / "strict", audit_size=1),
        contact_sheet_provider=lambda _example: tmp_path / "does-not-exist.jpg",
    )
    with pytest.raises(LongRouteDataError, match="centroid|contact"):
        strict.build(1)


def test_failed_attempt_does_not_replace_current_generation_and_writes_last_attempt(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    output = tmp_path / "out"
    _builder(output, manifest).build(1)
    current = (output / "output" / "current-generation.json").read_text()
    copied = tmp_path / "eval.bin"
    copied.write_bytes((tmp_path / "video-0.bin").read_bytes())
    broken = LongRouteBuilder(
        (manifest,), eval_assets=(VideoAsset("eval", copied, ((1.0, 0.0),) * 8),),
        leakage_auditor=LeakageAuditor(tmp_path / "audit.parquet"),
        config=LongRouteConfig(output_dir=output / "output", audit_size=1),
        contact_sheet_provider=lambda example: (output / "sheets" / f"{example.question_id}.jpg"),
    )
    with pytest.raises(LongRouteLeakageError):
        broken.build(2)
    assert (output / "output" / "current-generation.json").read_text() == current
    assert json.loads((output / "output" / "last-attempt.json").read_text())["status"] == "failed"


def test_review_c5_102_examples_use_exact_integer_multi_event_bounds(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, videos=102, seconds=61)
    result = _builder(tmp_path / "out", manifest, audit_size=1).build(4)
    multi = sum(item.template != "single_event" for item in result.examples)
    assert len(result.examples) == 102
    assert 21 <= multi <= 30
    assert multi >= 21


def test_nested_reordering_does_not_change_canonical_manifest_and_duplicate_question_is_rejected(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    shuffled = manifest.model_copy(update={"videos": tuple(reversed(manifest.videos))})
    assert _builder(tmp_path / "one", manifest).build(5).canonical_json() == _builder(tmp_path / "two", shuffled).build(5).canonical_json()
    original = manifest.videos[0]
    copied_events = tuple(
        event.model_copy(update={"event_id": f"{event.event_id}-copy"})
        for event in original.events
    )
    copied_questions = tuple(
        question.model_copy(
            update={"target_event_id": f"{question.target_event_id}-copy"}
        )
        for question in original.questions
    )
    duplicate_video = original.model_copy(
        update={
            "video_id": "new-video",
            "source_uri": f"{original.source_uri}-copy",
            "events": copied_events,
            "questions": copied_questions,
        }
    )
    duplicate = manifest.model_copy(
        update={"videos": manifest.videos + (duplicate_video,)}
    )
    with pytest.raises(LongRouteDataError, match="question"):
        _builder(tmp_path / "duplicate", duplicate).build(5)

def test_review_c2_toctou_mutation_after_audit_fails_without_repointing_current(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    output = tmp_path / "out"
    _builder(output, manifest).build(1)
    old_pointer = (output / "output" / "current-generation.json").read_text()
    def mutate(example: object) -> Path:
        (tmp_path / "video-0.bin").write_bytes(b"mutated-after-audit")
        path = output / "sheets" / f"{getattr(example, 'question_id')}.jpg"; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"sheet"); return path
    builder = LongRouteBuilder((manifest,), eval_assets=(), leakage_auditor=LeakageAuditor(tmp_path / "audit.parquet"), config=LongRouteConfig(output_dir=output / "output", audit_size=1), contact_sheet_provider=mutate)
    with pytest.raises(LongRouteDataError, match="changed"):
        builder.build(2)
    assert (output / "output" / "current-generation.json").read_text() == old_pointer
    assert json.loads((output / "output" / "last-attempt.json").read_text())["status"] == "failed"


def test_review_c8_contact_sheet_local_and_remote_contract(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    rejected = LongRouteBuilder((manifest,), eval_assets=(), leakage_auditor=LeakageAuditor(tmp_path / "audit.parquet"), config=LongRouteConfig(output_dir=tmp_path / "bad", audit_size=1), contact_sheet_provider=lambda _example: tmp_path / "missing.jpg")
    with pytest.raises(LongRouteDataError, match="contact"):
        rejected.build(1)
    accepted = LongRouteBuilder(
        (manifest,), eval_assets=(), leakage_auditor=LeakageAuditor(tmp_path / "remote.parquet"),
        config=LongRouteConfig(output_dir=tmp_path / "remote", audit_size=1),
        contact_sheet_provider=lambda _example: "https://example.invalid/sheet.jpg",
        contact_sheet_validator=DefaultContactSheetValidator(
            remote_probes={"https": lambda _uri: None}
        ),
    )
    assert accepted.build(2).examples


def _public_generation_names(output_root: Path) -> set[str]:
    generations = output_root / "generations"
    return {entry.name for entry in generations.iterdir()} if generations.is_dir() else set()


@pytest.mark.parametrize(
    ("operation", "ordinal"),
    (
        ("generation_write", 2),
        ("generation_write", 3),
        ("directory_publish", 1),
        ("pointer_write", 1),
    ),
)
def test_review_c3_publication_fault_matrix_is_atomic(
    tmp_path: Path, operation: str, ordinal: int
) -> None:
    manifest = _manifest(tmp_path / "sources")
    root = tmp_path / "case"
    _builder(root, manifest).build(1)
    output = root / "output"
    previous_pointer = (output / "current-generation.json").read_bytes()
    previous_generations = _public_generation_names(output)
    backend = FaultingPublicationBackend(output, operation, ordinal)

    with pytest.raises(OSError, match="injected"):
        _builder(root, manifest, publication_backend=backend).build(2)

    assert (output / "current-generation.json").read_bytes() == previous_pointer
    assert _public_generation_names(output) == previous_generations
    staging = output / ".staging"
    assert not staging.exists() or not any(staging.iterdir())
    attempt = json.loads((output / "last-attempt.json").read_text(encoding="utf-8"))
    assert attempt["status"] == "failed"


def test_review_c3_eval_hit_has_the_same_failed_transaction_semantics(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "sources")
    root = tmp_path / "case"
    _builder(root, manifest).build(1)
    output = root / "output"
    previous_pointer = (output / "current-generation.json").read_bytes()
    previous_generations = _public_generation_names(output)
    copied = tmp_path / "held-out.bin"
    copied.write_bytes(manifest.videos[0].path.read_bytes())
    builder = LongRouteBuilder(
        (manifest,),
        eval_assets=(VideoAsset("held-out", copied, ((0.0, 1.0),) * 8),),
        leakage_auditor=LeakageAuditor(tmp_path / "eval-hit.parquet"),
        config=LongRouteConfig(output_dir=output, audit_size=1),
        contact_sheet_provider=lambda _example: tmp_path / "unused.jpg",
    )

    with pytest.raises(LongRouteLeakageError):
        builder.build(2)

    assert (output / "current-generation.json").read_bytes() == previous_pointer
    assert _public_generation_names(output) == previous_generations
    assert json.loads((output / "last-attempt.json").read_text())["status"] == "failed"
    failed_audit = json.loads((output / "leakage-audit.json").read_text())
    assert failed_audit["findings"][0]["kind"] == "hash_duplicate"



def test_review_c4_skips_3550_second_top_neighbor_for_a_legal_combination(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path / "sources", videos=60, seconds=50.0)
    first = manifest.videos[0]
    target = first.events[0]
    oversized = SourceEvent(
        event_id="e-0-00-oversized",
        start_sec=100.0,
        end_sec=3650.0,
        label="oversized nearest neighbor",
        embedding=target.embedding,
    )
    changed_first = first.model_copy(update={"events": (target, oversized, first.events[1])})
    changed = manifest.model_copy(
        update={"videos": (changed_first, *manifest.videos[1:])}
    )

    result = _builder(tmp_path / "out", changed).build(4)

    example = next(item for item in result.examples if item.question_id == "q-0")
    assert "e-0-00-oversized" not in {segment.event_id for segment in example.segments}
    assert 9 <= len(example.segments) - 1 <= 19
    assert 600 <= example.duration_sec <= 3600


def test_review_c5_single_event_sources_upgrade_via_route_distractors(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path / "sources")
    single_event = manifest.model_copy(
        update={
            "videos": tuple(
                video.model_copy(update={"events": (video.events[0],)})
                for video in manifest.videos
            )
        }
    )

    result = _builder(tmp_path / "out", single_event).build(7)

    multi = next(item for item in result.examples if item.template == "before_after")
    route = {
        f"{segment.source_video_id}:{segment.event_id}": segment
        for segment in multi.segments
    }
    left, right = (route[event_id] for event_id in multi.supporting_event_ids)
    labels = {
        f"{video.video_id}:{event.event_id}": event.label
        for video in single_event.videos
        for event in video.events
    }
    assert left.source_video_id != right.source_video_id
    assert right.global_start_sec == left.global_end_sec
    if "immediately after" in multi.question:
        assert multi.answer == labels[multi.supporting_event_ids[1]]
    else:
        assert "immediately before" in multi.question
        assert multi.answer == labels[multi.supporting_event_ids[0]]


def test_review_c5_fails_when_multi_event_eligible_set_is_too_small(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "sources")
    ambiguous = manifest.model_copy(
        update={
            "videos": tuple(
                video.model_copy(
                    update={
                        "events": (
                            video.events[0].model_copy(update={"label": "same event"}),
                        )
                    }
                )
                for video in manifest.videos
            )
        }
    )

    with pytest.raises(LongRouteDataError, match="eligible"):
        _builder(tmp_path / "out", ambiguous).build(7)



def _two_manifests(
    manifest: TrainSplitManifest,
) -> tuple[TrainSplitManifest, TrainSplitManifest]:
    midpoint = len(manifest.videos) // 2
    left = manifest.model_copy(update={"videos": manifest.videos[:midpoint]})
    right = manifest.model_copy(
        update={
            "name": "activitynet-qa",
            "dataset_version": "2.0",
            "source_uri": "https://datasets.example/activitynet-qa/v2",
            "license": "CC-BY-4.0",
            "videos": manifest.videos[midpoint:],
        }
    )
    return left, right


def test_review_c6_all_nested_input_permutations_are_byte_identical(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "sources")
    enriched_videos = []
    for index, video in enumerate(manifest.videos):
        events = tuple(
            event.model_copy(
                update={"attributes": {"zeta": index, "alpha": event.label}}
            )
            for event in video.events
        )
        questions = (
            video.questions[0].model_copy(update={"options": ("yes", "maybe", "no")}),
            SourceQuestion(
                question_id=f"q-{index}-secondary",
                question=f"What finishes in {index}?",
                options=("third", "finish", "first"),
                answer="finish",
                target_event_id=events[1].event_id,
            ),
        )
        enriched_videos.append(
            video.model_copy(update={"events": events, "questions": questions})
        )
    enriched = manifest.model_copy(update={"videos": tuple(enriched_videos)})
    baseline_sources = _two_manifests(enriched)

    def permute(source: TrainSplitManifest) -> TrainSplitManifest:
        videos = []
        for video in reversed(source.videos):
            events = tuple(
                event.model_copy(
                    update={"attributes": dict(reversed(tuple(event.attributes.items())))}
                )
                for event in reversed(video.events)
            )
            questions = tuple(
                question.model_copy(update={"options": tuple(reversed(question.options))})
                for question in reversed(video.questions)
            )
            videos.append(
                video.model_copy(update={"events": events, "questions": questions})
            )
        return source.model_copy(update={"videos": tuple(videos)})

    permuted_sources = tuple(permute(source) for source in reversed(baseline_sources))
    first = _builder(tmp_path / "one", baseline_sources).build(13)
    second = _builder(tmp_path / "two", permuted_sources).build(13)

    assert first.canonical_json().encode() == second.canonical_json().encode()
    first_file = tmp_path / "one" / "output" / first.generation_uri / "manifest.json"
    second_file = tmp_path / "two" / "output" / second.generation_uri / "manifest.json"
    assert first_file.read_bytes() == second_file.read_bytes()


def test_review_c6_rejects_duplicate_manifest_video_event_and_question_identities(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path / "sources")
    left, right = _two_manifests(manifest)

    with pytest.raises(LongRouteDataError, match="manifest.*name"):
        _builder(
            tmp_path / "duplicate-name",
            (left, right.model_copy(update={"name": left.name})),
        ).build(1)
    with pytest.raises(LongRouteDataError, match="manifest.*identity"):
        _builder(
            tmp_path / "duplicate-source",
            (left, right.model_copy(update={"source_uri": left.source_uri})),
        ).build(1)

    duplicate_video = right.model_copy(
        update={
            "videos": (
                right.videos[0].model_copy(update={"video_id": left.videos[0].video_id}),
                *right.videos[1:],
            )
        }
    )
    with pytest.raises(LongRouteDataError, match="video"):
        _builder(tmp_path / "duplicate-video", (left, duplicate_video)).build(1)

    duplicate_event_id = left.videos[0].events[0].event_id
    duplicate_event_video = right.videos[0].model_copy(
        update={
            "events": (
                right.videos[0].events[0].model_copy(
                    update={"event_id": duplicate_event_id}
                ),
                *right.videos[0].events[1:],
            ),
            "questions": tuple(
                question.model_copy(update={"target_event_id": duplicate_event_id})
                for question in right.videos[0].questions
            ),
        }
    )
    duplicate_event = right.model_copy(
        update={"videos": (duplicate_event_video, *right.videos[1:])}
    )
    with pytest.raises(LongRouteDataError, match="event"):
        _builder(tmp_path / "duplicate-event", (left, duplicate_event)).build(1)

    duplicate_question_video = right.videos[0].model_copy(
        update={
            "questions": (
                right.videos[0].questions[0].model_copy(
                    update={"question_id": left.videos[0].questions[0].question_id}
                ),
            )
        }
    )
    duplicate_question = right.model_copy(
        update={"videos": (duplicate_question_video, *right.videos[1:])}
    )
    with pytest.raises(LongRouteDataError, match="question"):
        _builder(tmp_path / "duplicate-question", (left, duplicate_question)).build(1)


def test_review_c7_published_manifest_has_complete_parseable_provenance(
    tmp_path: Path,
) -> None:
    source = _manifest(tmp_path / "sources")
    result = _builder(tmp_path / "case", source).build(3)
    output = tmp_path / "case" / "output"

    assert result.schema_version
    assert result.builder_version
    assert len(result.source_manifests) == 1
    provenance = result.source_manifests[0]
    assert provenance.dataset_name == source.name
    assert provenance.dataset_version == source.dataset_version
    assert provenance.source_uri == source.source_uri
    assert provenance.license == source.license
    assert result.source_manifest_hashes[provenance.identity] == provenance.canonical_sha256
    assert {
        (video.video_id, video.source_uri, video.content_sha256)
        for video in provenance.videos
    } == {
        (video.video_id, video.source_uri, video.content_sha256)
        for video in source.videos
    }

    manifest_path = output / result.generation_uri / "manifest.json"
    parquet_path = output / result.leakage_parquet_uri
    audit_path = output / result.leakage_audit_uri
    assert manifest_path.is_file()
    assert parquet_path.is_file()
    assert audit_path.is_file()
    assert duckdb.sql(
        "SELECT count(*) FROM read_parquet(?)", params=[str(parquet_path)]
    ).fetchone() == (0,)
    assert json.loads(audit_path.read_text())["parquet_uri"] == result.leakage_parquet_uri


def test_review_c7_rejects_duplicate_video_source_identity_and_hash_mismatch(
    tmp_path: Path,
) -> None:
    source = _manifest(tmp_path / "sources")
    duplicated_uri = source.model_copy(
        update={
            "videos": (
                source.videos[0],
                source.videos[1].model_copy(
                    update={"source_uri": source.videos[0].source_uri}
                ),
                *source.videos[2:],
            )
        }
    )
    with pytest.raises(LongRouteDataError, match="source identity"):
        _builder(tmp_path / "duplicate-uri", duplicated_uri).build(1)

    bad_hash = source.model_copy(
        update={
            "videos": (
                source.videos[0].model_copy(update={"content_sha256": "0" * 64}),
                *source.videos[1:],
            )
        }
    )
    with pytest.raises(LongRouteDataError, match="SHA-256"):
        _builder(tmp_path / "bad-hash", bad_hash).build(1)


def test_review_c8_default_validator_checks_local_and_configured_remote_sheets(
    tmp_path: Path,
) -> None:
    validator = DefaultContactSheetValidator()
    missing = tmp_path / "missing.jpg"
    empty = tmp_path / "empty.jpg"
    empty.write_bytes(b"")
    directory = tmp_path / "directory"
    directory.mkdir()
    valid = tmp_path / "valid.jpg"
    valid.write_bytes(b"sheet")

    for invalid in (missing, empty, directory):
        with pytest.raises(LongRouteDataError, match="contact sheet"):
            validator.validate(invalid)
    assert validator.validate(valid) == str(valid.resolve())

    remote_calls: list[str] = []
    remote = DefaultContactSheetValidator(
        remote_probes={"https": lambda uri: remote_calls.append(uri)}
    )
    uri = "https://sheets.example/audit/q-1.jpg"
    assert remote.validate(uri) == uri
    assert remote_calls == [uri]
    with pytest.raises(LongRouteDataError, match="scheme"):
        remote.validate("ftp://sheets.example/audit/q-1.jpg")


def test_review_c8_permission_failure_is_portable_and_every_sheet_is_validated(
    tmp_path: Path,
) -> None:
    source = _manifest(tmp_path / "sources")
    recording = RecordingContactSheetValidator()
    result = _builder(
        tmp_path / "ok",
        source,
        audit_size=3,
        contact_sheet_validator=recording,
    ).build(5)
    bundle = tmp_path / "ok" / "output" / result.generation_uri / "audit"
    records = json.loads((bundle / "samples.json").read_text())
    assert len(records) == 3
    assert len(recording.calls) == 3
    assert {Path(str(record["contact_sheet"])).name for record in records} == {
        Path(call).name for call in recording.calls
    }

    class PermissionDeniedValidator:
        def validate(self, _uri: str | Path) -> str:
            raise PermissionError("portable unreadable fixture")

    output = tmp_path / "denied"
    with pytest.raises(LongRouteDataError, match="contact sheet.*readable"):
        _builder(
            output,
            source,
            contact_sheet_validator=PermissionDeniedValidator(),
        ).build(6)
    assert json.loads(
        (output / "output" / "last-attempt.json").read_text()
    )["status"] == "failed"



def test_review_c6_nested_event_order_keeps_source_hashes_identical(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    altered = manifest.model_copy(update={"videos": tuple(video.model_copy(update={"events": tuple(reversed(video.events)), "questions": tuple(reversed(video.questions))}) for video in reversed(manifest.videos))})
    assert _builder(tmp_path / "one", manifest).build(1).source_manifest_hashes == _builder(tmp_path / "two", altered).build(1).source_manifest_hashes


@pytest.mark.parametrize("previous_success", (False, True))
def test_review_round2_complete_attempt_write_fails_before_current_commit(
    tmp_path: Path, previous_success: bool
) -> None:
    manifest = _manifest(tmp_path / "sources")
    root = tmp_path / "case"
    output = root / "output"
    if previous_success:
        first = _builder(root, manifest).build(1)
        previous_pointer = (output / "current-generation.json").read_bytes()
        previous_generations = _public_generation_names(output)
        assert (output / first.generation_uri / "manifest.json").is_file()
    else:
        previous_pointer = None
        previous_generations = set()
    backend = FaultingPublicationBackend(output, "root_write", 1)

    with pytest.raises(OSError, match="root_write"):
        _builder(root, manifest, publication_backend=backend).build(2)

    pointer = output / "current-generation.json"
    if previous_pointer is None:
        assert not pointer.exists()
    else:
        assert pointer.read_bytes() == previous_pointer
        generation = json.loads(previous_pointer)["generation"]
        assert (output / generation / "manifest.json").is_file()
    assert _public_generation_names(output) == previous_generations
    assert json.loads((output / "last-attempt.json").read_text())["status"] == "failed"


def test_review_round2_success_attempt_and_current_name_the_same_generation(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path / "sources")
    root = tmp_path / "case"

    result = _builder(root, manifest).build(3)

    output = root / "output"
    pointer = json.loads((output / "current-generation.json").read_text())
    attempt = json.loads((output / "last-attempt.json").read_text())
    assert pointer == {"generation": result.generation_uri, "seed": 3}
    assert attempt["status"] == "complete"
    assert attempt["generation"] == result.generation_uri
    assert (output / pointer["generation"] / "manifest.json").is_file()


class AfterActionPointerBackend(PublicationBackend):
    def _run_operation(self, name: str, action: Callable[[], object]) -> object:
        result = super()._run_operation(name, action)
        if name == self.POINTER_WRITE:
            raise OSError("injected after pointer action")
        return result


class BeforePointerReplaceFailureBackend(PublicationBackend):
    def _before_pointer_replace(self, *_args: object) -> None:
        raise OSError("injected before pointer replace")


class AfterCommitFailureBackend(PublicationBackend):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.abort_calls = 0

    def abort(self, *args: object, **kwargs: object) -> None:
        self.abort_calls += 1
        super().abort(*args, **kwargs)

    def publish_current(self, *args: object, **kwargs: object) -> None:
        super().publish_current(*args, **kwargs)
        raise OSError("injected after committed pointer")


def test_review_round3_pointer_replace_is_not_inside_action_wrapper(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path / "sources")
    root = tmp_path / "case"
    output = root / "output"
    backend = AfterActionPointerBackend(output)

    result = _builder(root, manifest, publication_backend=backend).build(1)

    pointer = json.loads((output / "current-generation.json").read_text())
    assert pointer["generation"] == result.generation_uri
    assert (output / result.generation_uri / "manifest.json").is_file()


def test_review_round3_pre_replace_fault_preserves_previous_current(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path / "sources")
    root = tmp_path / "case"
    output = root / "output"
    first = _builder(root, manifest).build(1)
    previous_pointer = (output / "current-generation.json").read_bytes()
    previous_generations = _public_generation_names(output)
    backend = BeforePointerReplaceFailureBackend(output)

    with pytest.raises(OSError, match="before pointer replace"):
        _builder(root, manifest, publication_backend=backend).build(2)

    assert (output / "current-generation.json").read_bytes() == previous_pointer
    assert _public_generation_names(output) == previous_generations
    assert (output / first.generation_uri / "manifest.json").is_file()
    assert json.loads((output / "last-attempt.json").read_text())["status"] == "failed"


def test_review_round3_committed_transaction_is_never_aborted_by_builder(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path / "sources")
    root = tmp_path / "case"
    output = root / "output"
    backend = AfterCommitFailureBackend(output)

    with pytest.raises(OSError, match="after committed pointer"):
        _builder(root, manifest, publication_backend=backend).build(3)

    pointer = json.loads((output / "current-generation.json").read_text())
    generation = pointer["generation"]
    assert (output / generation / "manifest.json").is_file()
    assert backend.abort_calls == 0
    attempt = json.loads((output / "last-attempt.json").read_text())
    assert attempt["status"] == "complete"
    assert attempt["generation"] == generation
