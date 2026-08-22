"""Cost-preference, STOP-sufficiency, and Oracle pilot audit labels."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, MutableMapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fidmem.agent.answerer import FrozenAnswerer
from fidmem.types import ActionInstance, RouterState

from .search import OraclePath


COST_PREFERENCES = (0.0, 0.1, 0.3, 1.0)


class CostNormalization(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    constant: float = Field(gt=0)
    method: Literal["max_train_total_cost"] = "max_train_total_cost"
    sample_count: int = Field(ge=1)
    source_split: Literal["train"] = "train"


def _normalization_payload(normalization: CostNormalization) -> dict[str, object]:
    return normalization.model_dump(mode="json")


def _write_manifest(
    run_manifest: MutableMapping[str, object] | str | Path,
    normalization: CostNormalization,
) -> None:
    if isinstance(run_manifest, MutableMapping):
        run_manifest["oracle_cost_normalization"] = _normalization_payload(
            normalization
        )
        return
    path = Path(run_manifest)
    existing: dict[str, object] = {}
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("run manifest must contain a JSON object")
        existing = loaded
    existing["oracle_cost_normalization"] = _normalization_payload(normalization)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            existing,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def fit_train_cost_normalization(
    total_costs: Sequence[float],
    *,
    split: str,
    run_manifest: MutableMapping[str, object] | str | Path,
) -> CostNormalization:
    """Fit and persist the only cost scale allowed for every later split."""

    if split != "train":
        raise ValueError("cost normalization may only be fit from the train split")
    costs = tuple(float(cost) for cost in total_costs)
    if not costs:
        raise ValueError("train cost normalization requires at least one sample")
    if any(not math.isfinite(cost) or cost < 0 for cost in costs):
        raise ValueError("train costs must be finite and non-negative")
    constant = max(costs)
    if constant <= 0:
        raise ValueError("train costs must contain a positive value")
    normalization = CostNormalization(
        constant=constant,
        sample_count=len(costs),
    )
    _write_manifest(run_manifest, normalization)
    return normalization


class PreferenceLabel(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    cost_preference: float
    utility: float
    optimal_paths: tuple[OraclePath, ...]

    @property
    def optimal_first_actions(self) -> tuple[ActionInstance, ...]:
        unique: list[ActionInstance] = []
        for path in self.optimal_paths:
            if not path.transitions:
                continue
            action = path.transitions[0].action
            if action not in unique:
                unique.append(action)
        return tuple(unique)


def preference_labels(
    paths: Sequence[OraclePath], normalization: CostNormalization
) -> tuple[PreferenceLabel, ...]:
    """Return all utility-tied optimal paths at four frozen preferences."""

    if not paths:
        raise ValueError("preference labels require at least one path")
    labels: list[PreferenceLabel] = []
    for preference in COST_PREFERENCES:
        scored = tuple(
            (
                path.answer_score
                - preference * (path.total_cost / normalization.constant),
                path,
            )
            for path in paths
        )
        best = max(utility for utility, _ in scored)
        optimal = tuple(
            path.model_copy(update={"utility": utility})
            for utility, path in sorted(
                scored,
                key=lambda item: (
                    -item[0],
                    item[1].total_cost,
                    item[1].depth,
                    item[1].action_signature,
                ),
            )
            if math.isclose(utility, best, rel_tol=0, abs_tol=1e-12)
        )
        labels.append(
            PreferenceLabel(
                cost_preference=preference,
                utility=best,
                optimal_paths=optimal,
            )
        )
    return tuple(labels)


CorrectnessJudge = Callable[[str, str], bool]


def sufficiency_label(
    state: RouterState,
    answerer: FrozenAnswerer,
    *,
    gold_answer: str,
    judge: CorrectnessJudge | None = None,
) -> int:
    """Run the unified frozen STOP Answer and emit a trajectory-agnostic bit."""

    result = answerer.answer(state.question, state.options, state.evidence)
    is_correct = judge or (
        lambda predicted, gold: predicted.strip().casefold() == gold.strip().casefold()
    )
    return int(is_correct(result.answer, gold_answer))


class PilotQuestionTiming(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    question_id: str
    a800_gpu_seconds: float = Field(ge=0)


class PilotTimingAudit(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    question_count: int
    mean_a800_gpu_seconds: float = Field(ge=0)
    p90_a800_gpu_seconds: float = Field(ge=0)
    per_question: tuple[PilotQuestionTiming, ...]


def summarize_pilot_timings(
    timings: Sequence[PilotQuestionTiming],
) -> PilotTimingAudit:
    records = tuple(timings)
    if len(records) != 100:
        raise ValueError("Oracle pilot timing audit requires exactly 100 questions")
    ids = tuple(record.question_id for record in records)
    if len(set(ids)) != len(ids):
        raise ValueError("pilot question ids must be unique")
    ordered = sorted(record.a800_gpu_seconds for record in records)
    return PilotTimingAudit(
        question_count=100,
        mean_a800_gpu_seconds=sum(ordered) / 100,
        p90_a800_gpu_seconds=ordered[89],
        per_question=tuple(sorted(records, key=lambda record: record.question_id)),
    )


class BeamAuditCase(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    question_id: str
    beam_action_signature: tuple[str, ...]
    exhaustive_action_signature: tuple[str, ...]
    beam_cost: float = Field(ge=0)
    exhaustive_cost: float = Field(ge=0)


class BeamSearchAudit(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    case_count: int = Field(ge=1)
    path_hit_rate: float = Field(ge=0, le=1)
    mean_cost_gap: float
    cases: tuple[BeamAuditCase, ...]


def compare_beam_to_exhaustive(
    cases: Sequence[BeamAuditCase],
) -> BeamSearchAudit:
    records = tuple(cases)
    if not records:
        raise ValueError("Beam audit requires an exhaustive-search subset")
    ids = tuple(record.question_id for record in records)
    if len(set(ids)) != len(ids):
        raise ValueError("Beam audit question ids must be unique")
    hits = sum(
        record.beam_action_signature == record.exhaustive_action_signature
        for record in records
    )
    return BeamSearchAudit(
        case_count=len(records),
        path_hit_rate=hits / len(records),
        mean_cost_gap=sum(
            record.beam_cost - record.exhaustive_cost for record in records
        )
        / len(records),
        cases=tuple(sorted(records, key=lambda record: record.question_id)),
    )


class StabilitySample(BaseModel):
    model_config = ConfigDict(frozen=True)

    state_id: str
    answers: tuple[str, str, str]


class AnswerStabilityAudit(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    state_count: int
    repeats_per_state: Literal[3]
    flipped_state_count: int
    answer_flip_rate: float = Field(ge=0, le=1)
    samples: tuple[StabilitySample, ...]


def answer_stability_audit(
    samples: Sequence[StabilitySample],
) -> AnswerStabilityAudit:
    records = tuple(samples)
    if len(records) != 100:
        raise ValueError("answer stability audit requires exactly 100 states")
    ids = tuple(record.state_id for record in records)
    if len(set(ids)) != len(ids):
        raise ValueError("stability state ids must be unique")
    flips = sum(len(set(record.answers)) > 1 for record in records)
    return AnswerStabilityAudit(
        state_count=100,
        repeats_per_state=3,
        flipped_state_count=flips,
        answer_flip_rate=flips / 100,
        samples=tuple(sorted(records, key=lambda record: record.state_id)),
    )


class OraclePilotAudit(BaseModel):
    """Serializable artifact schema for the preregistered Task 9 pilot."""

    model_config = ConfigDict(frozen=True)

    timing: PilotTimingAudit
    beam: BeamSearchAudit
    stability: AnswerStabilityAudit
