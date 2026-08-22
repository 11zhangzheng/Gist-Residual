from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from fidmem.data.leakage import LeakageAuditor, VideoAsset
from fidmem.data.longroute import (
    LongRouteBuilder,
    LongRouteConfig,
    LongRouteDataError,
    LongRouteLeakageError,
    SourceEvent,
    SourceQuestion,
    SourceVideo,
    TrainSplitManifest,
)


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
                video_id=f"v-{index}", path=path, split="train", licensed=True,
                events=events,
                questions=(SourceQuestion(question_id=f"q-{index}", question=f"What happens in {index}?", options=("yes", "no"), answer="yes", target_event_id=events[0].event_id),),
            )
        )
    return TrainSplitManifest(name="nextqa-train", split="train", videos=tuple(source_videos))


def _builder(root: Path, manifest: TrainSplitManifest, *, audit_size: int = 1, **config: object) -> LongRouteBuilder:
    return LongRouteBuilder(
        (manifest,),
        eval_assets=(),
        leakage_auditor=LeakageAuditor(root / "leakage.parquet"),
        config=LongRouteConfig(output_dir=root / "output", audit_size=audit_size, **config),
        contact_sheet_provider=lambda example: f"sheets/{example.question_id}.jpg",
    )


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
        eval_assets=(VideoAsset("held-out", copied),),
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
    non_target = [segment.event_id for segment in first.segments if segment.event_id != first.target_event_id]
    assert non_target == sorted(non_target)


def test_audit_bundle_has_exact_size_stable_fields_and_insufficient_input_fails(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    result = _builder(tmp_path / "good", manifest, audit_size=3).build(6)
    bundle = tmp_path / "good" / "output" / "audit"
    assert len((bundle / "samples.jsonl").read_text().splitlines()) == 3
    with (bundle / "review.csv").open(newline="", encoding="utf-8") as source:
        assert csv.DictReader(source).fieldnames == ["question_id", "valid", "invalid", "reason"]
    assert all(item.audit_status == "pending" for item in result.examples)
    with pytest.raises(LongRouteDataError, match="audit"):
        _builder(tmp_path / "small", _manifest(tmp_path / "small-src", videos=12), audit_size=11).build(6)
