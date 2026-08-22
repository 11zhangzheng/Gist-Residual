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
_SUMMARY_REL_TOL = 1e-9
_SUMMARY_ABS_TOL = 1e-12


def _summary_matches(reported: float, expected: float) -> bool:
    return math.isclose(
        reported, expected, rel_tol=_SUMMARY_REL_TOL, abs_tol=_SUMMARY_ABS_TOL
    )


class CostNormalization(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    constant: float = Field(gt=0)
    method: Literal["max_train_total_cost"] = "max_train_total_cost"
    sample_count: int = Field(ge=1)
    source_split: Literal["train"]


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
        source_split="train",
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
    model_config = ConfigDict(
        frozen=True, allow_inf_nan=False, revalidate_instances="always"
    )

    question_count: int = Field(ge=1, strict=True)
    mean_a800_gpu_seconds: float = Field(ge=0)
    p90_a800_gpu_seconds: float = Field(ge=0)
    per_question: tuple[PilotQuestionTiming, ...]

    def validate_consistency(self) -> None:
        if self.question_count != len(self.per_question):
            raise ValueError("question_count does not match per_question")
        if self.question_count != 100:
            raise ValueError("question_count must be exactly 100")
        ids = tuple(record.question_id for record in self.per_question)
        if len(set(ids)) != len(ids):
            raise ValueError("per_question question ids must be unique")
        ordered = sorted(record.a800_gpu_seconds for record in self.per_question)
        expected_mean = sum(ordered) / len(ordered)
        expected_p90 = ordered[math.ceil(0.9 * len(ordered)) - 1]
        if not _summary_matches(self.mean_a800_gpu_seconds, expected_mean):
            raise ValueError("mean_a800_gpu_seconds does not match per_question")
        if not _summary_matches(self.p90_a800_gpu_seconds, expected_p90):
            raise ValueError("p90_a800_gpu_seconds does not match per_question")

    @model_validator(mode="after")
    def summary_must_match_raw_timings(self) -> "PilotTimingAudit":
        self.validate_consistency()
        return self


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
    model_config = ConfigDict(
        frozen=True, allow_inf_nan=False, revalidate_instances="always"
    )

    case_count: int = Field(ge=1, strict=True)
    path_hit_rate: float = Field(ge=0, le=1)
    mean_cost_gap: float
    cases: tuple[BeamAuditCase, ...]

    def validate_consistency(self) -> None:
        if self.case_count != len(self.cases):
            raise ValueError("case_count does not match cases")
        ids = tuple(record.question_id for record in self.cases)
        if len(set(ids)) != len(ids):
            raise ValueError("Beam audit question ids must be unique")
        hits = sum(
            record.beam_action_signature == record.exhaustive_action_signature
            for record in self.cases
        )
        expected_hit_rate = hits / len(self.cases)
        expected_cost_gap = sum(
            record.beam_cost - record.exhaustive_cost for record in self.cases
        ) / len(self.cases)
        if not _summary_matches(self.path_hit_rate, expected_hit_rate):
            raise ValueError("path_hit_rate does not match cases")
        if not _summary_matches(self.mean_cost_gap, expected_cost_gap):
            raise ValueError("mean_cost_gap does not match cases")

    @model_validator(mode="after")
    def summary_must_match_raw_cases(self) -> "BeamSearchAudit":
        self.validate_consistency()
        return self


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
    model_config = ConfigDict(
        frozen=True, allow_inf_nan=False, revalidate_instances="always"
    )

    state_count: int = Field(ge=1, strict=True)
    repeats_per_state: Literal[3]
    flipped_state_count: int = Field(ge=0, strict=True)
    answer_flip_rate: float = Field(ge=0, le=1)
    samples: tuple[StabilitySample, ...]

    def validate_consistency(self) -> None:
        if self.state_count != len(self.samples):
            raise ValueError("state_count does not match samples")
        if self.state_count != 100:
            raise ValueError("state_count must be exactly 100")
        ids = tuple(record.state_id for record in self.samples)
        if len(set(ids)) != len(ids):
            raise ValueError("stability state ids must be unique")
        expected_flips = sum(len(set(record.answers)) > 1 for record in self.samples)
        if self.flipped_state_count != expected_flips:
            raise ValueError("flipped_state_count does not match samples")
        expected_flip_rate = expected_flips / len(self.samples)
        if not _summary_matches(self.answer_flip_rate, expected_flip_rate):
            raise ValueError("answer_flip_rate does not match samples")

    @model_validator(mode="after")
    def summary_must_match_raw_samples(self) -> "AnswerStabilityAudit":
        self.validate_consistency()
        return self


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

    model_config = ConfigDict(frozen=True, revalidate_instances="always")

    timing: PilotTimingAudit
    beam: BeamSearchAudit
    stability: AnswerStabilityAudit

    @model_validator(mode="after")
    def nested_audits_must_remain_consistent(self) -> "OraclePilotAudit":
        self.timing.validate_consistency()
        self.beam.validate_consistency()
        self.stability.validate_consistency()
        return self
