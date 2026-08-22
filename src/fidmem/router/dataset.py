"""Strict Oracle records and deterministic video-grouped BC datasets.

Rows contain snapshots of already-cached observations. Loading and splitting
these rows is intentionally data-only: it has no provider or VLM callback and
therefore cannot rebuild expensive observations for another random seed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable, Literal, Protocol

import duckdb
import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    model_validator,
)
from torch import Tensor
from torch.utils.data import Dataset

from fidmem.agent.answerer import FrozenAnswerer
from fidmem.data.longroute import DatasetManifest, LongRouteExample
from fidmem.oracle.labels import COST_PREFERENCES, CostNormalization, PreferenceLabel
from fidmem.types import ActionInstance, ActionType, FidelityLevel, RouterState

_ACTION_INDEX = {action: index for index, action in enumerate(ActionType)}
_FIDELITY_INDEX = {level: index for index, level in enumerate(FidelityLevel)}
_NONE_FIDELITY = len(_FIDELITY_INDEX)
_VISUAL_BUDGET_INDEX = {None: 0, "low": 1, "high": 2}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


class TokenizerIdentity(BaseModel):
    """Actual tokenizer implementation plus immutable vocabulary artifacts."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    implementation: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    vocab_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def byte_identity(cls, model_id: str) -> "TokenizerIdentity":
        vocabulary = {"<pad>": 0} | {
            f"byte:{value:02x}": value + 1 for value in range(256)
        }
        vocab_sha256 = hashlib.sha256(
            _canonical_json(vocabulary).encode("utf-8")
        ).hexdigest()
        artifact = {
            "implementation": "fidmem.byte-utf8.v1",
            "revision": "offline-byte-v1",
            "vocab_sha256": vocab_sha256,
            "encoding": "utf-8-bytes-plus-one",
            "padding_id": 0,
        }
        return cls(
            implementation="fidmem.byte-utf8.v1",
            model_id=model_id,
            revision="offline-byte-v1",
            vocab_sha256=vocab_sha256,
            artifact_sha256=hashlib.sha256(
                _canonical_json(artifact).encode("utf-8")
            ).hexdigest(),
        )


class TextTokenizer(Protocol):
    identity: TokenizerIdentity

    def encode(self, text: str, *, maximum: int, label: str) -> list[int]:
        ...


class TestByteTokenizer:
    __test__ = False

    def __init__(self, model_id: str = "byte-test-v1") -> None:
        self.identity = TokenizerIdentity.byte_identity(model_id)

    def encode(self, text: str, *, maximum: int, label: str) -> list[int]:
        encoded = text.encode("utf-8")
        if len(encoded) > maximum:
            raise ValueError(f"{label} exceeds configured maximum of {maximum} tokens")
        return [value + 1 for value in encoded]


class HFTokenizerAdapter:
    """No-truncation adapter identified from the actual local tokenizer."""

    def __init__(self, tokenizer: object) -> None:
        get_vocab = getattr(tokenizer, "get_vocab", None)
        if not callable(get_vocab):
            raise ValueError("pretrained tokenizer does not expose get_vocab")
        vocabulary = get_vocab()
        if (
            not isinstance(vocabulary, Mapping)
            or not vocabulary
            or any(
                not isinstance(token, str)
                or not isinstance(token_id, int)
                or isinstance(token_id, bool)
                or token_id < 0
                for token, token_id in vocabulary.items()
            )
        ):
            raise ValueError("pretrained tokenizer vocabulary is invalid")
        raw_path = getattr(tokenizer, "name_or_path", None)
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("pretrained tokenizer lacks a local artifact path")
        snapshot_path = Path(raw_path).resolve()
        if (
            snapshot_path.parent.name != "snapshots"
            or not snapshot_path.is_dir()
            or len(snapshot_path.name) != 40
            or any(
                character not in "0123456789abcdef" for character in snapshot_path.name
            )
        ):
            raise ValueError(
                "pretrained tokenizer must be loaded from an immutable local snapshot"
            )
        repository_dir = snapshot_path.parent.parent.name
        if not repository_dir.startswith("models--"):
            raise ValueError("pretrained tokenizer snapshot repository is invalid")
        repository_parts = repository_dir.removeprefix("models--").split("--")
        if len(repository_parts) < 2 or any(not part for part in repository_parts):
            raise ValueError("pretrained tokenizer model identity is invalid")
        model_id = "/".join(repository_parts)
        revision = snapshot_path.name
        implementation = f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"
        vocab_sha256 = hashlib.sha256(
            _canonical_json(dict(vocabulary)).encode("utf-8")
        ).hexdigest()
        backend = getattr(tokenizer, "backend_tokenizer", None)
        backend_to_str = getattr(backend, "to_str", None)
        backend_payload = backend_to_str() if callable(backend_to_str) else None
        artifact_payload = {
            "implementation": implementation,
            "model_id": model_id,
            "revision": revision,
            "vocabulary": dict(vocabulary),
            "special_tokens_map": getattr(tokenizer, "special_tokens_map", {}),
            "backend": backend_payload,
        }
        self.identity = TokenizerIdentity(
            implementation=implementation,
            model_id=model_id,
            revision=revision,
            vocab_sha256=vocab_sha256,
            artifact_sha256=hashlib.sha256(
                _canonical_json(artifact_payload).encode("utf-8")
            ).hexdigest(),
        )
        self._tokenizer = tokenizer

    def encode(self, text: str, *, maximum: int, label: str) -> list[int]:
        encode = getattr(self._tokenizer, "encode", None)
        if not callable(encode):
            raise ValueError("pretrained tokenizer does not expose encode")
        token_ids = list(encode(text, add_special_tokens=True, truncation=False))
        if len(token_ids) > maximum:
            raise ValueError(f"{label} exceeds configured maximum of {maximum} tokens")
        if any(
            not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0
            for token_id in token_ids
        ):
            raise ValueError("pretrained tokenizer emitted invalid token ids")
        return token_ids


class LongRouteSourceIdentity(BaseModel):
    """Immutable lineage extracted from a published LongRoute manifest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: Literal["train", "dev"]
    video_group_id: str = Field(min_length=1)
    example_id: str = Field(min_length=1)


class FrozenComponentIdentity(BaseModel):
    """Immutable identity for a component that emitted a supervised label."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    implementation: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _state_sha256(state: RouterState) -> str:
    return hashlib.sha256(
        _canonical_json(state.model_dump(mode="json")).encode("utf-8")
    ).hexdigest()


def _question_sha256(question_id: str, state: RouterState) -> str:
    payload = {
        "question_id": question_id,
        "question": state.question,
        "options": state.options,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


class SufficiencyLabelArtifact(BaseModel):
    """Sealed STOP label with frozen answerer and judge identities."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question_id: str = Field(min_length=1)
    state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    question_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    answerer_identity: FrozenComponentIdentity
    judge_identity: FrozenComponentIdentity
    stop_answer: str = Field(min_length=1)
    gold_answer: str = Field(min_length=1)
    label: Literal[0, 1]


def seal_sufficiency_label(
    *,
    state: RouterState,
    question_id: str,
    gold_answer: str,
    answerer: FrozenAnswerer,
    answerer_identity: FrozenComponentIdentity,
    judge: Callable[[str, str], bool] | None = None,
    judge_identity: FrozenComponentIdentity | None = None,
) -> SufficiencyLabelArtifact:
    """Adapt Task9 STOP evaluation into a self-identifying artifact."""

    if judge is None:

        def judge(predicted, gold):
            return predicted.strip().casefold() == gold.strip().casefold()

        judge_identity = FrozenComponentIdentity(
            implementation="fidmem.exact_match.v1",
            model_id="unicode-casefold-exact-match",
            revision="1",
            artifact_sha256=hashlib.sha256(
                b"strip+unicode-casefold+exact-match/v1"
            ).hexdigest(),
        )
    if judge_identity is None:
        raise ValueError("a custom sufficiency judge requires a frozen identity")
    result = answerer.answer(state.question, state.options, state.evidence)
    return SufficiencyLabelArtifact(
        question_id=question_id,
        state_sha256=_state_sha256(state),
        question_sha256=_question_sha256(question_id, state),
        answerer_identity=answerer_identity,
        judge_identity=judge_identity,
        stop_answer=result.answer,
        gold_answer=gold_answer,
        label=int(judge(result.answer, gold_answer)),
    )


class OracleRecordProvenance(BaseModel):
    """Auditable Task8/Task9 lineage carried by every BC row."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    dataset_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_manifest_canonical_json: str = Field(min_length=2, repr=False)
    longroute_example_canonical_json: str = Field(min_length=2, repr=False)
    longroute_example_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    group_assignment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    asset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_split: Literal["train", "dev"]
    video_group_id: str = Field(min_length=1)
    longroute_example_id: str = Field(min_length=1)
    normalization_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    normalization: CostNormalization
    preference_set_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    preference_values: tuple[float, float, float, float]
    selected_preference: float = Field(ge=0, le=1)
    oracle_utility: float
    optimal_action_tie_count: int = Field(ge=1, strict=True)
    sufficiency_artifact: SufficiencyLabelArtifact

    @model_validator(mode="after")
    def validate_frozen_preferences(self) -> "OracleRecordProvenance":
        if self.preference_values != COST_PREFERENCES:
            raise ValueError("preference_values must match Task9 COST_PREFERENCES")
        if self.selected_preference not in self.preference_values:
            raise ValueError("selected preference is not in the four-value set")
        normalization_hash = hashlib.sha256(
            _canonical_json(self.normalization.model_dump(mode="json")).encode("utf-8")
        ).hexdigest()
        if normalization_hash != self.normalization_manifest_hash:
            raise ValueError("normalization identity does not match canonical payload")
        manifest = DatasetManifest.model_validate_json(
            self.dataset_manifest_canonical_json
        )
        if manifest.canonical_json() != self.dataset_manifest_canonical_json:
            raise ValueError("dataset manifest is not canonical")
        if (
            hashlib.sha256(
                self.dataset_manifest_canonical_json.encode("utf-8")
            ).hexdigest()
            != self.dataset_manifest_hash
        ):
            raise ValueError("dataset manifest hash does not match canonical bytes")
        example = LongRouteExample.model_validate_json(
            self.longroute_example_canonical_json
        )
        if (
            hashlib.sha256(
                self.longroute_example_canonical_json.encode("utf-8")
            ).hexdigest()
            != self.longroute_example_sha256
        ):
            raise ValueError("LongRoute example hash does not match canonical bytes")
        if (
            _canonical_json(example.model_dump(mode="json"))
            != self.longroute_example_canonical_json
        ):
            raise ValueError("LongRoute example is not canonical")
        if example.question_id != self.longroute_example_id:
            raise ValueError("LongRoute example identity mismatch")
        _validate_task8_lineage(manifest, example)
        return self


class OracleBCRecord(BaseModel):
    """One supervised state with a complete candidate-instance action set."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", allow_inf_nan=False, revalidate_instances="always"
    )

    record_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    observation_snapshot_id: str = Field(min_length=1)
    provenance: OracleRecordProvenance
    state: RouterState
    action_instances: tuple[ActionInstance, ...] = Field(min_length=1)
    legal_action_mask: tuple[StrictBool, ...] = Field(min_length=1)
    target_action_index: StrictInt = Field(ge=0)
    sufficiency_target: Literal[0, 1]
    cost_to_go: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_nested_state_and_action_mask(self) -> "OracleBCRecord":
        RouterState.model_validate(self.state.model_dump(mode="python"))
        for action in self.action_instances:
            ActionInstance.model_validate(action.model_dump(mode="python"))
        if len(set(self.action_instances)) != len(self.action_instances):
            raise ValueError("action_instances must be unique")
        if len(self.legal_action_mask) != len(self.action_instances):
            raise ValueError("legal_action_mask must match action_instances")
        if not any(self.legal_action_mask):
            raise ValueError("record must contain at least one legal action")
        if self.target_action_index >= len(self.action_instances):
            raise ValueError("target_action_index is out of range")
        if not self.legal_action_mask[self.target_action_index]:
            raise ValueError("target action must be legal")
        if not math.isfinite(self.cost_to_go):
            raise ValueError("cost_to_go must be finite")
        if self.video_id != self.provenance.video_group_id:
            raise ValueError("video_id must equal provenance video_group_id")
        if self.state.cost_preference != self.provenance.selected_preference:
            raise ValueError("state preference must match provenance")
        example = LongRouteExample.model_validate_json(
            self.provenance.longroute_example_canonical_json
        )
        if self.question_id != example.question_id:
            raise ValueError("question_id must match the LongRoute example")
        if (
            self.state.question != example.question
            or self.state.options != example.options
        ):
            raise ValueError("Router state question must match the LongRoute example")
        artifact = self.provenance.sufficiency_artifact
        if artifact.question_id != self.question_id:
            raise ValueError("sufficiency question identity mismatch")
        if artifact.state_sha256 != _state_sha256(self.state):
            raise ValueError("sufficiency state hash mismatch")
        if artifact.question_sha256 != _question_sha256(self.question_id, self.state):
            raise ValueError("sufficiency question hash mismatch")
        if (
            artifact.gold_answer != example.answer
            or artifact.label != self.sufficiency_target
        ):
            raise ValueError("sufficiency artifact does not match the record label")
        return self


def _validate_task8_lineage(
    manifest: DatasetManifest, example: LongRouteExample
) -> tuple[str, str, str]:
    if manifest.source_manifest_hashes != {
        source.identity: source.canonical_sha256 for source in manifest.source_manifests
    }:
        raise ValueError("source manifest hashes do not match source provenance")
    matching_examples = tuple(
        item for item in manifest.examples if item.question_id == example.question_id
    )
    if len(matching_examples) != 1 or matching_examples[0] != example:
        raise ValueError(
            "LongRoute example is not an exact member of the dataset manifest"
        )
    segment_videos = {segment.source_video_id for segment in example.segments}
    if not segment_videos or any(
        manifest.group_assignment.get(video_id) != example.split
        for video_id in segment_videos
    ):
        raise ValueError("LongRoute group assignment does not match example split")
    if manifest.group_assignment.get(example.target_source_video_id) != example.split:
        raise ValueError("target video group does not match example split")
    canonical_events = {
        f"{segment.source_video_id}:{segment.event_id}" for segment in example.segments
    }
    if example.target_event_id not in canonical_events or not set(
        example.supporting_event_ids
    ).issubset(canonical_events):
        raise ValueError("LongRoute event identities do not match virtual segments")
    owners = tuple(
        source
        for source in manifest.source_manifests
        if any(
            video.video_id == example.target_source_video_id for video in source.videos
        )
    )
    if len(owners) != 1:
        raise ValueError("target video must have exactly one source manifest owner")
    owner = owners[0]
    video = next(
        item for item in owner.videos if item.video_id == example.target_source_video_id
    )
    if manifest.asset_sha256s.get(video.video_id) != video.content_sha256:
        raise ValueError("target video asset hash does not match source provenance")
    return (
        owner.canonical_sha256,
        example.target_source_video_id,
        video.content_sha256,
    )


def materialize_oracle_record(
    *,
    observation_snapshot_id: str,
    state: RouterState,
    action_instances: Sequence[ActionInstance],
    legal_action_mask: Sequence[bool],
    preference_labels: Sequence[PreferenceLabel],
    normalization: CostNormalization,
    manifest: DatasetManifest,
    example: LongRouteExample,
    sufficiency_artifact: SufficiencyLabelArtifact,
) -> OracleBCRecord:
    """Derive one BC row exclusively from verified Task8/Task9 artifacts."""

    labels = tuple(
        PreferenceLabel.model_validate(label.model_dump(mode="python"))
        for label in preference_labels
    )
    if tuple(label.cost_preference for label in labels) != COST_PREFERENCES:
        raise ValueError("preference labels must contain the four Task9 preferences")
    normalization = CostNormalization.model_validate(
        normalization.model_dump(mode="python")
    )
    manifest = DatasetManifest.model_validate(manifest.model_dump(mode="python"))
    example = LongRouteExample.model_validate(example.model_dump(mode="python"))
    source_manifest_hash, video_group_id, asset_sha256 = _validate_task8_lineage(
        manifest, example
    )
    if state.question != example.question or state.options != example.options:
        raise ValueError("Router state question does not match LongRoute example")
    canonical_events = {
        f"{segment.source_video_id}:{segment.event_id}" for segment in example.segments
    }
    local_counts: dict[str, int] = {}
    for segment in example.segments:
        local_counts[segment.event_id] = local_counts.get(segment.event_id, 0) + 1
    canonical_events |= {
        event_id for event_id, count in local_counts.items() if count == 1
    }
    referenced_events = (
        set(state.candidate_event_ids)
        | {item.event_id for item in state.evidence}
        | {
            action.event_id
            for action in (*state.action_history, *action_instances)
            if action.event_id is not None
        }
    )
    if not referenced_events.issubset(canonical_events):
        raise ValueError(
            "Router state/action event ids do not match LongRoute segments"
        )

    selected = next(
        (label for label in labels if label.cost_preference == state.cost_preference),
        None,
    )
    if selected is None or not selected.optimal_paths:
        raise ValueError("selected preference has no Task9 Oracle path")
    for label in labels:
        if not label.optimal_paths:
            raise ValueError("preference label has no optimal Oracle path")
        for path in label.optimal_paths:
            expected_cost = sum(item.step_cost for item in path.transitions)
            if not math.isclose(
                path.total_cost, expected_cost, rel_tol=0, abs_tol=1e-12
            ):
                raise ValueError("Oracle path cost does not match its transitions")
            expected_utility = path.answer_score - label.cost_preference * (
                path.total_cost / normalization.constant
            )
            if not math.isclose(
                path.utility, expected_utility, rel_tol=0, abs_tol=1e-12
            ):
                raise ValueError("Oracle path utility does not match normalization")
        if not math.isclose(
            label.utility,
            label.optimal_paths[0].utility,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("preference utility does not match optimal paths")

    selected_path = sorted(
        selected.optimal_paths,
        key=lambda path: (path.total_cost, path.depth, path.action_signature),
    )[0]
    if not selected_path.transitions or selected_path.transitions[0].state != state:
        raise ValueError("selected Oracle path does not begin at the record state")
    oracle_action = selected_path.transitions[0].action
    optimal_actions = selected.optimal_first_actions
    actions = tuple(action_instances)
    mask = tuple(legal_action_mask)
    if oracle_action not in actions:
        raise ValueError("Oracle action is absent from action_instances")
    target_index = actions.index(oracle_action)
    if target_index >= len(mask) or not mask[target_index]:
        raise ValueError("Oracle action is not legal in the supplied mask")

    normalization_hash = hashlib.sha256(
        _canonical_json(normalization.model_dump(mode="json")).encode("utf-8")
    ).hexdigest()
    preference_payload = tuple(
        {
            "cost_preference": label.cost_preference,
            "utility": label.utility,
            "optimal_first_actions": tuple(
                action.model_dump(mode="json") for action in label.optimal_first_actions
            ),
        }
        for label in labels
    )
    preference_hash = hashlib.sha256(
        _canonical_json(preference_payload).encode("utf-8")
    ).hexdigest()

    if not isinstance(sufficiency_artifact, SufficiencyLabelArtifact):
        raise TypeError("sufficiency_artifact must be a sealed artifact")
    artifact = SufficiencyLabelArtifact.model_validate(
        sufficiency_artifact.model_dump(mode="python")
    )
    if artifact.question_id != example.question_id:
        raise ValueError("sufficiency artifact question identity mismatch")
    if artifact.state_sha256 != _state_sha256(state):
        raise ValueError("sufficiency artifact state hash mismatch")
    if artifact.question_sha256 != _question_sha256(example.question_id, state):
        raise ValueError("sufficiency artifact question hash mismatch")
    if artifact.gold_answer != example.answer:
        raise ValueError("sufficiency artifact gold answer mismatch")

    manifest_json = manifest.canonical_json()
    example_json = _canonical_json(example.model_dump(mode="json"))
    dataset_manifest_hash = hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()
    record_suffix = hashlib.sha256(
        _canonical_json(
            {
                "dataset_manifest_hash": dataset_manifest_hash,
                "state_sha256": _state_sha256(state),
                "observation_snapshot_id": observation_snapshot_id,
            }
        ).encode("utf-8")
    ).hexdigest()[:16]
    record_id = f"{example.question_id}:{record_suffix}"
    cost_to_go = selected_path.total_cost / normalization.constant
    if not math.isfinite(cost_to_go) or cost_to_go < 0:
        raise ValueError("derived cost_to_go must be finite and non-negative")
    return OracleBCRecord(
        record_id=record_id,
        video_id=video_group_id,
        question_id=example.question_id,
        observation_snapshot_id=observation_snapshot_id,
        provenance=OracleRecordProvenance(
            dataset_manifest_hash=dataset_manifest_hash,
            dataset_manifest_canonical_json=manifest_json,
            longroute_example_canonical_json=example_json,
            longroute_example_sha256=hashlib.sha256(
                example_json.encode("utf-8")
            ).hexdigest(),
            group_assignment_sha256=hashlib.sha256(
                _canonical_json(manifest.group_assignment).encode("utf-8")
            ).hexdigest(),
            source_manifest_hash=source_manifest_hash,
            asset_sha256=asset_sha256,
            source_split=example.split,
            video_group_id=video_group_id,
            longroute_example_id=example.question_id,
            normalization_manifest_hash=normalization_hash,
            normalization=normalization,
            preference_set_hash=preference_hash,
            preference_values=COST_PREFERENCES,
            selected_preference=state.cost_preference,
            oracle_utility=selected.utility,
            optimal_action_tie_count=len(optimal_actions),
            sufficiency_artifact=artifact,
        ),
        state=state,
        action_instances=actions,
        legal_action_mask=mask,
        target_action_index=target_index,
        sufficiency_target=artifact.label,
        cost_to_go=cost_to_go,
    )


class SplitManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[2] = 2
    seed: StrictInt
    assignment_source: Literal["upstream_source_split"]
    dataset_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    train_video_ids: tuple[str, ...]
    dev_video_ids: tuple[str, ...]
    test_video_ids: tuple[str, ...]
    train_record_ids: tuple[str, ...]
    dev_record_ids: tuple[str, ...]
    test_record_ids: tuple[str, ...]


class OracleBCDataset(Dataset[OracleBCRecord]):
    """Canonical, immutable view of cached Oracle supervision."""

    def __init__(self, records: Sequence[OracleBCRecord]) -> None:
        validated = tuple(
            OracleBCRecord.model_validate(record.model_dump(mode="python"))
            for record in records
        )
        if not validated:
            raise ValueError("Oracle BC dataset must not be empty")
        ids = tuple(record.record_id for record in validated)
        if len(set(ids)) != len(ids):
            raise ValueError("Oracle BC record ids must be unique")
        self._records = tuple(sorted(validated, key=lambda record: record.record_id))
        video_splits: dict[str, set[str]] = {}
        for record in self._records:
            video_splits.setdefault(record.video_id, set()).add(
                record.provenance.source_split
            )
        if any(len(splits) != 1 for splits in video_splits.values()):
            raise ValueError("one video group cannot cross LongRoute source splits")
        self.normalization_manifest_hashes = tuple(
            sorted(
                {
                    record.provenance.normalization_manifest_hash
                    for record in self._records
                }
            )
        )
        if len(self.normalization_manifest_hashes) != 1:
            raise ValueError(
                "one Oracle BC dataset must use exactly one cost normalization"
            )
        normalization_values = {
            _canonical_json(record.provenance.normalization.model_dump(mode="json"))
            for record in self._records
        }
        if len(normalization_values) != 1:
            raise ValueError("mixed cost normalization constants are forbidden")
        payload = "\n".join(
            _canonical_json(record.model_dump(mode="json")) for record in self._records
        )
        self.identity = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def records(self) -> tuple[OracleBCRecord, ...]:
        return self._records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> OracleBCRecord:
        return self._records[index]

    def subset(self, record_ids: Sequence[str]) -> "OracleBCDataset":
        wanted = set(record_ids)
        selected = tuple(
            record for record in self._records if record.record_id in wanted
        )
        if len(selected) != len(wanted):
            raise ValueError("split manifest contains unknown record ids")
        return OracleBCDataset(selected)


def build_grouped_split(
    records: Sequence[OracleBCRecord],
    *,
    seed: int,
    train_fraction: float = 1.0,
    dev_fraction: float = 0.0,
) -> SplitManifest:
    """Preserve Task8's immutable video-group assignments without remapping."""

    if train_fraction != 1.0 or dev_fraction != 0.0:
        raise ValueError(
            "upstream_source_split requires train_fraction=1 and dev_fraction=0; "
            "a train-pool resplit needs a separately published assignment artifact"
        )
    dataset = OracleBCDataset(records)
    train_videos = tuple(
        sorted(
            {
                record.video_id
                for record in dataset.records
                if record.provenance.source_split == "train"
            }
        )
    )
    dev_videos = tuple(
        sorted(
            {
                record.video_id
                for record in dataset.records
                if record.provenance.source_split == "dev"
            }
        )
    )

    def record_ids(video_ids: tuple[str, ...]) -> tuple[str, ...]:
        selected = set(video_ids)
        return tuple(
            record.record_id
            for record in dataset.records
            if record.video_id in selected
        )

    return SplitManifest(
        seed=seed,
        assignment_source="upstream_source_split",
        dataset_hash=dataset.identity,
        train_video_ids=train_videos,
        dev_video_ids=dev_videos,
        test_video_ids=(),
        train_record_ids=record_ids(train_videos),
        dev_record_ids=record_ids(dev_videos),
        test_record_ids=(),
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("refusing to replace a symlink")
    handle = tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class OracleDatasetAuthority(BaseModel):
    """Dataset-level content-addressed authority stored once beside row data."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    schema_version: Literal[1] = 1
    dataset_manifests: dict[str, str]
    normalizations: dict[str, dict[str, object]]
    record_digests: dict[str, str]
    authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_content_addresses(self) -> "OracleDatasetAuthority":
        for digest, canonical in self.dataset_manifests.items():
            if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != digest:
                raise ValueError("authority dataset manifest hash mismatch")
            manifest = DatasetManifest.model_validate_json(canonical)
            if manifest.canonical_json() != canonical:
                raise ValueError("authority dataset manifest is not canonical")
        for digest, payload in self.normalizations.items():
            if (
                hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
                != digest
            ):
                raise ValueError("authority normalization hash mismatch")
            CostNormalization.model_validate(payload)
        if any(
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in self.record_digests.values()
        ):
            raise ValueError("authority record digest is not SHA-256")
        expected = hashlib.sha256(
            _canonical_json(
                self.model_dump(mode="json", exclude={"authority_sha256"})
            ).encode("utf-8")
        ).hexdigest()
        if expected != self.authority_sha256:
            raise ValueError("authority self hash mismatch")
        return self


def _authority_path(path: Path) -> Path:
    return Path(f"{path}.authority.json")


def _stored_record_payload(record: OracleBCRecord) -> dict[str, object]:
    payload = record.model_dump(mode="json")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("record provenance must be a JSON object")
    provenance.pop("dataset_manifest_canonical_json", None)
    provenance.pop("longroute_example_canonical_json", None)
    return payload


def _stored_record_json(record: OracleBCRecord) -> str:
    return _canonical_json(_stored_record_payload(record))


def _build_dataset_authority(
    records: Sequence[OracleBCRecord],
) -> OracleDatasetAuthority:
    manifests: dict[str, str] = {}
    normalizations: dict[str, dict[str, object]] = {}
    record_digests: dict[str, str] = {}
    for record in records:
        provenance = record.provenance
        canonical = provenance.dataset_manifest_canonical_json
        previous = manifests.setdefault(provenance.dataset_manifest_hash, canonical)
        if previous != canonical:
            raise ValueError("one dataset manifest hash has conflicting payloads")
        normalization = provenance.normalization.model_dump(mode="json")
        previous_normalization = normalizations.setdefault(
            provenance.normalization_manifest_hash, normalization
        )
        if previous_normalization != normalization:
            raise ValueError("one normalization hash has conflicting payloads")
        record_digests[record.record_id] = hashlib.sha256(
            _stored_record_json(record).encode("utf-8")
        ).hexdigest()
    base: dict[str, object] = {
        "schema_version": 1,
        "dataset_manifests": manifests,
        "normalizations": normalizations,
        "record_digests": record_digests,
    }
    return OracleDatasetAuthority(
        **base,
        authority_sha256=hashlib.sha256(
            _canonical_json(base).encode("utf-8")
        ).hexdigest(),
    )


def _inflate_stored_record(
    raw: str, authority: OracleDatasetAuthority
) -> OracleBCRecord:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("Oracle record row is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("Oracle record row must be a JSON object")
    canonical = _canonical_json(payload)
    if raw != canonical:
        raise ValueError("Oracle record row is not canonical JSON")
    record_id = payload.get("record_id")
    if not isinstance(record_id, str):
        raise ValueError("Oracle record id is missing")
    expected_digest = authority.record_digests.get(record_id)
    actual_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if expected_digest != actual_digest:
        raise ValueError("Oracle record digest does not match dataset authority")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Oracle record provenance is missing")
    manifest_hash = provenance.get("dataset_manifest_hash")
    if not isinstance(manifest_hash, str):
        raise ValueError("Oracle record dataset manifest identity is missing")
    manifest_json = authority.dataset_manifests.get(manifest_hash)
    if manifest_json is None:
        raise ValueError("Oracle record manifest is absent from dataset authority")
    normalization_hash = provenance.get("normalization_manifest_hash")
    if not isinstance(normalization_hash, str):
        raise ValueError("Oracle normalization identity is missing")
    normalization_payload = authority.normalizations.get(normalization_hash)
    if normalization_payload is None:
        raise ValueError("Oracle normalization is absent from dataset authority")
    if provenance.get("normalization") != normalization_payload:
        raise ValueError("Oracle normalization does not match dataset authority")
    example_hash = provenance.get("longroute_example_sha256")
    question_id = payload.get("question_id")
    manifest = DatasetManifest.model_validate_json(manifest_json)
    matches = tuple(
        example
        for example in manifest.examples
        if example.question_id == question_id
        and hashlib.sha256(
            _canonical_json(example.model_dump(mode="json")).encode("utf-8")
        ).hexdigest()
        == example_hash
    )
    if len(matches) != 1:
        raise ValueError("Oracle example is absent from dataset authority")
    provenance["dataset_manifest_canonical_json"] = manifest_json
    provenance["longroute_example_canonical_json"] = _canonical_json(
        matches[0].model_dump(mode="json")
    )
    return OracleBCRecord.model_validate(payload)


def write_oracle_records(path: str | Path, records: Sequence[OracleBCRecord]) -> None:
    """Write rows plus one mandatory, content-addressed provenance sidecar."""

    target = Path(path)
    dataset = OracleBCDataset(records)
    authority = _build_dataset_authority(dataset.records)
    record_jsons = tuple(_stored_record_json(record) for record in dataset.records)
    rows = tuple(
        (
            record.record_id,
            record.video_id,
            record.question_id,
            record.provenance.source_split,
            record.provenance.source_manifest_hash,
            record.provenance.normalization_manifest_hash,
            record.provenance.selected_preference,
            record_json,
        )
        for record, record_json in zip(dataset.records, record_jsons, strict=True)
    )
    if target.suffix.casefold() == ".jsonl":
        _atomic_write(target, ("\n".join(record_jsons) + "\n").encode("utf-8"))
        _atomic_write(
            _authority_path(target),
            (_canonical_json(authority.model_dump(mode="json")) + "\n").encode("utf-8"),
        )
        return
    if target.suffix.casefold() != ".parquet":
        raise ValueError("Oracle records path must end in jsonl or parquet")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise ValueError("refusing to replace a symlink")
    descriptor, raw_path = tempfile.mkstemp(dir=target.parent, suffix=".parquet")
    os.close(descriptor)
    temporary = Path(raw_path)
    try:
        connection = duckdb.connect()
        try:
            connection.execute(
                """CREATE TABLE oracle_records(
                    record_id VARCHAR NOT NULL,
                    video_id VARCHAR NOT NULL,
                    question_id VARCHAR NOT NULL,
                    source_split VARCHAR NOT NULL,
                    source_manifest_hash VARCHAR NOT NULL,
                    normalization_manifest_hash VARCHAR NOT NULL,
                    selected_preference DOUBLE NOT NULL,
                    record_json VARCHAR NOT NULL
                )"""
            )
            connection.executemany(
                "INSERT INTO oracle_records VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows
            )
            quoted = str(temporary.resolve()).replace("'", "''")
            connection.execute(f"COPY oracle_records TO '{quoted}' (FORMAT PARQUET)")
        finally:
            connection.close()
        os.replace(temporary, target)
        _atomic_write(
            _authority_path(target),
            (_canonical_json(authority.model_dump(mode="json")) + "\n").encode("utf-8"),
        )
    finally:
        temporary.unlink(missing_ok=True)


def load_oracle_records(path: str | Path) -> tuple[OracleBCRecord, ...]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    suffix = source.suffix.casefold()
    if suffix not in {".jsonl", ".parquet"}:
        raise ValueError("Oracle records path must end in jsonl or parquet")
    authority_path = _authority_path(source)
    if not authority_path.is_file():
        raise ValueError("Oracle dataset authority sidecar is missing")
    try:
        authority = OracleDatasetAuthority.model_validate_json(
            authority_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise ValueError(
            "Oracle dataset authority hash or content is invalid"
        ) from error
    structured_rows: tuple[tuple[object, ...], ...] = ()
    if suffix == ".jsonl":
        raw_rows = tuple(
            line
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    elif suffix == ".parquet":
        connection = duckdb.connect()
        try:
            cursor = connection.execute(
                "SELECT * FROM read_parquet(?)", [str(source.resolve())]
            )
            columns = tuple(item[0] for item in cursor.description)
            expected_columns = (
                "record_id",
                "video_id",
                "question_id",
                "source_split",
                "source_manifest_hash",
                "normalization_manifest_hash",
                "selected_preference",
                "record_json",
            )
            if columns != expected_columns:
                raise ValueError("Oracle parquet structured columns are invalid")
            structured_rows = tuple(cursor.fetchall())
            raw_rows = tuple(row[-1] for row in structured_rows)
        finally:
            connection.close()
    else:
        raise ValueError("Oracle records path must end in jsonl or parquet")
    if not raw_rows or any(not isinstance(row, str) for row in raw_rows):
        raise ValueError("Oracle record source must contain non-empty JSON strings")
    records = tuple(_inflate_stored_record(row, authority) for row in raw_rows)
    if set(authority.record_digests) != {record.record_id for record in records}:
        raise ValueError("dataset authority record set does not match row data")
    used_manifests = {record.provenance.dataset_manifest_hash for record in records}
    used_normalizations = {
        record.provenance.normalization_manifest_hash for record in records
    }
    if used_manifests != set(authority.dataset_manifests):
        raise ValueError("dataset authority manifest set does not match row data")
    if used_normalizations != set(authority.normalizations):
        raise ValueError("dataset authority normalization set does not match row data")
    if suffix == ".parquet":
        for structured, record in zip(structured_rows, records, strict=True):
            expected = (
                record.record_id,
                record.video_id,
                record.question_id,
                record.provenance.source_split,
                record.provenance.source_manifest_hash,
                record.provenance.normalization_manifest_hash,
                record.provenance.selected_preference,
            )
            if structured[:-1] != expected:
                raise ValueError(
                    "structured Parquet provenance does not match record_json"
                )
    return OracleBCDataset(records).records


@dataclass
class RouterBatch:
    question_token_ids: Tensor
    question_token_mask: Tensor
    evidence_token_ids: Tensor
    evidence_token_mask: Tensor
    evidence_item_mask: Tensor
    evidence_fidelity: Tensor
    evidence_numeric: Tensor
    history_token_ids: Tensor
    history_token_mask: Tensor
    history_item_mask: Tensor
    history_action_type: Tensor
    action_token_ids: Tensor
    action_token_mask: Tensor
    legal_action_mask: Tensor
    action_type: Tensor
    action_fidelity: Tensor
    action_visual_budget: Tensor
    action_frontier: Tensor
    action_evidence_affinity: Tensor
    state_numeric: Tensor
    target_action_index: Tensor
    sufficiency_target: Tensor
    cost_to_go_target: Tensor

    def to(self, device: torch.device | str) -> "RouterBatch":
        return RouterBatch(
            **{
                field.name: getattr(self, field.name).to(device)
                for field in fields(self)
            }
        )


def _action_text(action: ActionInstance) -> str:
    return "|".join(
        (action.action_type.value, action.event_id or "", action.visual_budget or "")
    )


def _padded_tokens(
    items: Sequence[Sequence[str]],
    *,
    maximum: int,
    label: str,
    tokenizer: TextTokenizer,
) -> tuple[Tensor, Tensor, Tensor]:
    batch_size = len(items)
    max_items = max(1, max((len(row) for row in items), default=0))
    encoded: list[list[list[int]]] = []
    max_tokens = 1
    for row in items:
        encoded_row = [
            tokenizer.encode(text, maximum=maximum, label=label) for text in row
        ]
        encoded.append(encoded_row)
        for item in encoded_row:
            max_tokens = max(max_tokens, len(item))
    token_ids = torch.zeros((batch_size, max_items, max_tokens), dtype=torch.long)
    token_mask = torch.zeros_like(token_ids, dtype=torch.bool)
    item_mask = torch.zeros((batch_size, max_items), dtype=torch.bool)
    for row_index, row in enumerate(encoded):
        for item_index, item in enumerate(row):
            if item:
                token_ids[row_index, item_index, : len(item)] = torch.tensor(item)
                token_mask[row_index, item_index, : len(item)] = True
            item_mask[row_index, item_index] = True
    return token_ids, token_mask, item_mask


class RouterCollator:
    def __init__(
        self,
        *,
        max_question_tokens: int | None = None,
        max_item_tokens: int | None = None,
        tokenizer: TextTokenizer | None = None,
        max_question_bytes: int | None = None,
        max_item_bytes: int | None = None,
    ) -> None:
        question_limit = max_question_tokens or max_question_bytes
        item_limit = max_item_tokens or max_item_bytes
        if question_limit is None or item_limit is None:
            raise ValueError("token limits are required")
        if question_limit < 1 or item_limit < 1:
            raise ValueError("text token limits must be positive")
        self.max_question_tokens = question_limit
        self.max_item_tokens = item_limit
        self.tokenizer = tokenizer or TestByteTokenizer()

    @classmethod
    def for_test(
        cls, *, max_question_tokens: int, max_item_tokens: int
    ) -> "RouterCollator":
        return cls(
            max_question_tokens=max_question_tokens,
            max_item_tokens=max_item_tokens,
            tokenizer=TestByteTokenizer(),
        )

    def __call__(self, records: Sequence[OracleBCRecord]) -> RouterBatch:
        validated = tuple(
            OracleBCRecord.model_validate(record.model_dump(mode="python"))
            for record in records
        )
        if not validated:
            raise ValueError("cannot collate an empty batch")
        question_rows = [
            (record.state.question + "\n" + "\n".join(record.state.options),)
            for record in validated
        ]
        question_ids, question_mask, _ = _padded_tokens(
            question_rows,
            maximum=self.max_question_tokens,
            label="question",
            tokenizer=self.tokenizer,
        )
        evidence_rows = [
            [item.content for item in record.state.evidence] for record in validated
        ]
        evidence_ids, evidence_token_mask, evidence_item_mask = _padded_tokens(
            evidence_rows,
            maximum=self.max_item_tokens,
            label="evidence item",
            tokenizer=self.tokenizer,
        )
        history_rows = [
            [_action_text(action) for action in record.state.action_history]
            for record in validated
        ]
        history_ids, history_token_mask, history_item_mask = _padded_tokens(
            history_rows,
            maximum=self.max_item_tokens,
            label="history action",
            tokenizer=self.tokenizer,
        )
        action_rows = [
            [_action_text(action) for action in record.action_instances]
            for record in validated
        ]
        action_ids, action_token_mask, action_item_mask = _padded_tokens(
            action_rows,
            maximum=self.max_item_tokens,
            label="action instance",
            tokenizer=self.tokenizer,
        )
        batch_size, max_actions = action_item_mask.shape
        max_evidence = evidence_item_mask.shape[1]

        evidence_fidelity = torch.full(
            (batch_size, max_evidence), _NONE_FIDELITY, dtype=torch.long
        )
        evidence_numeric = torch.zeros(
            (batch_size, max_evidence, 2), dtype=torch.float32
        )
        history_action_type = torch.zeros_like(history_item_mask, dtype=torch.long)
        legal_mask = torch.zeros((batch_size, max_actions), dtype=torch.bool)
        action_type = torch.zeros((batch_size, max_actions), dtype=torch.long)
        action_fidelity = torch.full(
            (batch_size, max_actions), _NONE_FIDELITY, dtype=torch.long
        )
        action_visual_budget = torch.zeros((batch_size, max_actions), dtype=torch.long)
        action_frontier = torch.zeros((batch_size, max_actions, 2), dtype=torch.float32)
        affinity = torch.zeros(
            (batch_size, max_actions, max_evidence), dtype=torch.bool
        )
        state_numeric = torch.zeros((batch_size, 2), dtype=torch.float32)

        for row_index, record in enumerate(validated):
            state = record.state
            state_numeric[row_index] = torch.tensor(
                (state.remaining_budget, state.cost_preference)
            )
            for evidence_index, item in enumerate(state.evidence):
                evidence_fidelity[row_index, evidence_index] = _FIDELITY_INDEX[
                    item.fidelity_level
                ]
                evidence_numeric[row_index, evidence_index] = torch.tensor(
                    (item.score, float(item.acquisition_step))
                )
            for history_index, action in enumerate(state.action_history):
                history_action_type[row_index, history_index] = _ACTION_INDEX[
                    action.action_type
                ]
            legal_mask[row_index, : len(record.legal_action_mask)] = torch.tensor(
                record.legal_action_mask
            )
            for action_index, action in enumerate(record.action_instances):
                action_type[row_index, action_index] = _ACTION_INDEX[action.action_type]
                action_visual_budget[row_index, action_index] = _VISUAL_BUDGET_INDEX[
                    action.visual_budget
                ]
                if (
                    action.event_id is not None
                    and action.event_id in state.candidate_fidelity_levels
                ):
                    action_fidelity[row_index, action_index] = _FIDELITY_INDEX[
                        state.candidate_fidelity_levels[action.event_id]
                    ]
                    action_frontier[row_index, action_index] = torch.tensor(
                        state.context_frontiers[action.event_id], dtype=torch.float32
                    )
                    for evidence_index, item in enumerate(state.evidence):
                        affinity[row_index, action_index, evidence_index] = (
                            item.event_id == action.event_id
                        )
        return RouterBatch(
            question_token_ids=question_ids[:, 0],
            question_token_mask=question_mask[:, 0],
            evidence_token_ids=evidence_ids,
            evidence_token_mask=evidence_token_mask,
            evidence_item_mask=evidence_item_mask,
            evidence_fidelity=evidence_fidelity,
            evidence_numeric=evidence_numeric,
            history_token_ids=history_ids,
            history_token_mask=history_token_mask,
            history_item_mask=history_item_mask,
            history_action_type=history_action_type,
            action_token_ids=action_ids,
            action_token_mask=action_token_mask,
            legal_action_mask=legal_mask,
            action_type=action_type,
            action_fidelity=action_fidelity,
            action_visual_budget=action_visual_budget,
            action_frontier=action_frontier,
            action_evidence_affinity=affinity,
            state_numeric=state_numeric,
            target_action_index=torch.tensor(
                [record.target_action_index for record in validated], dtype=torch.long
            ),
            sufficiency_target=torch.tensor(
                [record.sufficiency_target for record in validated], dtype=torch.float32
            ),
            cost_to_go_target=torch.tensor(
                [record.cost_to_go for record in validated], dtype=torch.float32
            ),
        )
