from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from fidmem.agent.answerer import FrozenAnswerer
from fidmem.data.longroute import _canonical_source_manifest, _source_provenance
from fidmem.oracle.labels import CostNormalization
from fidmem.router.dataset import (
    FrozenComponentIdentity,
    OracleBCRecord,
    SufficiencyLabelArtifact,
    load_oracle_records,
    materialize_oracle_record,
    seal_sufficiency_label,
    write_oracle_records,
)
from fidmem.router.train_bc import _git_commit
from fidmem.types import ActionInstance, ActionType
from tests.router.test_review_round2 import _labels, _published, _record, _state


def _materializer_kwargs() -> dict[str, object]:
    manifest, example, source_manifest = _published()
    state = _state(example)
    action = ActionInstance(ActionType.EXPAND_RESIDUAL, example.target_event_id, None)
    artifact = seal_sufficiency_label(
        state=state,
        question_id=example.question_id,
        gold_answer=example.answer,
        answerer=FrozenAnswerer(lambda _: example.answer),
        answerer_identity=FrozenComponentIdentity(
            implementation="unit-answerer",
            model_id="answerer",
            revision="a" * 40,
            artifact_sha256="3" * 64,
        ),
    )
    return {
        "observation_snapshot_id": "cache",
        "state": state,
        "action_instances": (action, ActionInstance(ActionType.STOP, None, None)),
        "legal_action_mask": (True, True),
        "preference_labels": _labels(state, action),
        "normalization": CostNormalization(
            constant=10, sample_count=100, source_split="train"
        ),
        "manifest": manifest,
        "example": example,
        "source_manifests": (source_manifest,),
        "sufficiency_artifact": artifact,
    }


def test_every_segment_requires_one_real_owner_asset_event_and_range() -> None:
    kwargs = _materializer_kwargs()
    manifest = kwargs["manifest"]
    example = kwargs["example"]
    source = kwargs["source_manifests"][0]

    with pytest.raises(ValueError, match="canonical source manifests"):
        materialize_oracle_record(**(kwargs | {"source_manifests": ()}))

    duplicate = source.model_copy(
        update={"name": "unit-duplicate", "source_uri": "file:///duplicate.json"}
    )
    duplicate_provenance = _source_provenance(_canonical_source_manifest(duplicate))
    duplicate_manifest = manifest.model_copy(
        update={
            "source_manifests": manifest.source_manifests + (duplicate_provenance,),
            "source_manifest_hashes": manifest.source_manifest_hashes
            | {duplicate_provenance.identity: duplicate_provenance.canonical_sha256},
        }
    )
    with pytest.raises(ValueError, match="exactly one source owner"):
        materialize_oracle_record(
            **(
                kwargs
                | {
                    "manifest": duplicate_manifest,
                    "source_manifests": (source, duplicate),
                }
            )
        )

    missing_asset = manifest.model_copy(update={"asset_sha256s": {}})
    with pytest.raises(ValueError, match="asset hash"):
        materialize_oracle_record(**(kwargs | {"manifest": missing_asset}))

    segment = example.segments[0]
    fake_segment = segment.model_copy(update={"event_id": "forged-event"})
    fake_example = example.model_copy(
        update={
            "segments": (fake_segment,),
            "target_event_id": f"{segment.source_video_id}:forged-event",
            "supporting_event_ids": (f"{segment.source_video_id}:forged-event",),
        }
    )
    fake_manifest = manifest.model_copy(update={"examples": (fake_example,)})
    with pytest.raises(ValueError, match="event is absent"):
        materialize_oracle_record(
            **(kwargs | {"manifest": fake_manifest, "example": fake_example})
        )

    wrong_range_segment = segment.model_copy(update={"source_end_sec": 599})
    wrong_range_example = example.model_copy(
        update={"segments": (wrong_range_segment,)}
    )
    wrong_range_manifest = manifest.model_copy(
        update={"examples": (wrong_range_example,)}
    )
    with pytest.raises(ValueError, match="range"):
        materialize_oracle_record(
            **(
                kwargs
                | {
                    "manifest": wrong_range_manifest,
                    "example": wrong_range_example,
                }
            )
        )


def test_loader_requires_content_addressed_source_authority(
    tmp_path: Path,
) -> None:
    path = tmp_path / "records.jsonl"
    write_oracle_records(path, (_record(),))
    authority_path = Path(f"{path}.authority.json")
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority["source_manifests"] = {}
    authority["authority_sha256"] = hashlib.sha256(
        json.dumps(
            {
                key: value
                for key, value in authority.items()
                if key != "authority_sha256"
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    authority_path.write_text(
        json.dumps(
            authority,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source manifest is absent"):
        load_oracle_records(path)


def test_forged_row_provenance_cannot_become_self_consistent() -> None:
    record = _record()
    payload = record.model_dump(mode="python")
    payload["provenance"] = payload["provenance"] | {
        "source_split": "dev",
        "video_group_id": "forged-video",
        "source_manifest_hash": "f" * 64,
        "asset_sha256": "e" * 64,
        "group_assignment_sha256": "d" * 64,
    }
    payload["video_id"] = "forged-video"
    with pytest.raises(ValueError, match="authoritative Task8 lineage"):
        OracleBCRecord.model_validate(payload)


def test_sufficiency_v1_has_no_caller_label_or_custom_judge() -> None:
    _, example, _ = _published()
    state = _state(example)
    identity = FrozenComponentIdentity(
        implementation="unit-answerer",
        model_id="answerer",
        revision="a" * 40,
        artifact_sha256="3" * 64,
    )
    artifact = seal_sufficiency_label(
        state=state,
        question_id=example.question_id,
        gold_answer=example.answer,
        answerer=FrozenAnswerer(lambda _: "no"),
        answerer_identity=identity,
    )
    assert artifact.label == 0
    assert "label" not in artifact.model_dump(mode="json")

    forged = artifact.model_dump(mode="python") | {"label": 1}
    with pytest.raises(ValueError, match="extra"):
        SufficiencyLabelArtifact.model_validate(forged)
    with pytest.raises(TypeError):
        seal_sufficiency_label(
            state=state,
            question_id=example.question_id,
            gold_answer=example.answer,
            answerer=FrozenAnswerer(lambda _: "yes"),
            answerer_identity=identity,
            judge=lambda _predicted, _gold: True,
        )


def test_sufficiency_self_hash_rejects_answer_tampering() -> None:
    _, example, _ = _published()
    state = _state(example)
    artifact = seal_sufficiency_label(
        state=state,
        question_id=example.question_id,
        gold_answer=example.answer,
        answerer=FrozenAnswerer(lambda _: "no"),
        answerer_identity=FrozenComponentIdentity(
            implementation="unit-answerer",
            model_id="answerer",
            revision="a" * 40,
            artifact_sha256="3" * 64,
        ),
    )
    forged = artifact.model_dump(mode="python") | {"normalized_stop_answer": "yes"}
    with pytest.raises(ValueError, match="self hash"):
        SufficiencyLabelArtifact.model_validate(forged)


def test_git_commit_fails_closed_without_repository_or_build_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(*_args: object, **_kwargs: object) -> object:
        raise OSError("git unavailable")

    monkeypatch.setattr(subprocess, "run", unavailable)
    monkeypatch.delenv("FIDMEM_BUILD_GIT_COMMIT", raising=False)
    with pytest.raises(RuntimeError, match="Git commit identity"):
        _git_commit()
    monkeypatch.setenv("FIDMEM_BUILD_GIT_COMMIT", "unknown")
    with pytest.raises(ValueError, match="40 lowercase"):
        _git_commit()
    monkeypatch.setenv("FIDMEM_BUILD_GIT_COMMIT", "a" * 40)
    assert _git_commit() == "a" * 40
