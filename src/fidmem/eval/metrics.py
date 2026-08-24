"""Metrics recomputed exclusively from immutable per-question records."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
import math
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from fidmem.types import ActionInstance

from .error_taxonomy import ErrorSignals, classify_error


class ResourceUsage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False, strict=True)

    total_cost: float = Field(default=0.0, ge=0)
    gpu_seconds: float = Field(default=0.0, ge=0)
    wall_seconds: float = Field(default=0.0, ge=0)
    input_frames: int = Field(default=0, ge=0)
    visual_tokens: int = Field(default=0, ge=0)
    text_tokens: int = Field(default=0, ge=0)
    peak_memory_bytes: int = Field(default=0, ge=0)

    def plus(self, other: "ResourceUsage") -> "ResourceUsage":
        return ResourceUsage(
            total_cost=self.total_cost + other.total_cost,
            gpu_seconds=self.gpu_seconds + other.gpu_seconds,
            wall_seconds=self.wall_seconds + other.wall_seconds,
            input_frames=self.input_frames + other.input_frames,
            visual_tokens=self.visual_tokens + other.visual_tokens,
            text_tokens=self.text_tokens + other.text_tokens,
            peak_memory_bytes=max(self.peak_memory_bytes, other.peak_memory_bytes),
        )


class MetricRecord(Protocol):
    question_id: str
    video_group_id: str
    is_correct: bool
    invalid_reason: str | None
    acquisition_usage: ResourceUsage
    controller_usage: ResourceUsage
    oracle_utility: float
    realized_utility: float
    signals: ErrorSignals
    actions: tuple[ActionInstance, ...]


class RunSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False, strict=True)

    total_questions: int = Field(ge=1)
    valid_questions: int = Field(ge=1)
    invalid_questions: int = Field(ge=0)
    accuracy: float = Field(ge=0, le=1)
    total_cost: float = Field(ge=0)
    mean_total_cost: float = Field(ge=0)
    total_gpu_seconds: float = Field(ge=0)
    mean_gpu_seconds: float = Field(ge=0)
    total_wall_seconds: float = Field(ge=0)
    mean_wall_seconds: float = Field(ge=0)
    total_input_frames: int = Field(ge=0)
    total_visual_tokens: int = Field(ge=0)
    total_text_tokens: int = Field(ge=0)
    peak_memory_bytes: int = Field(ge=0)
    oracle_utility_regret: float = Field(ge=0)
    premature_stop_rate: float = Field(ge=0, le=1)
    unnecessary_expansion_rate: float = Field(ge=0, le=1)
    top_k_recall: float = Field(ge=0, le=1)
    action_distribution: dict[str, float]
    primary_error_distribution: dict[str, int]


class RunPoint(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False, strict=True)

    run_id: str = Field(min_length=1)
    policy_name: str = Field(min_length=1)
    seed: int
    accuracy: float = Field(ge=0, le=1)
    total_cost: float = Field(ge=0)


def _usage(record: MetricRecord) -> ResourceUsage:
    return record.acquisition_usage.plus(record.controller_usage)


def _validated_records(records: Sequence[MetricRecord]) -> tuple[MetricRecord, ...]:
    values = tuple(records)
    if not values:
        raise ValueError("metric records must be non-empty")
    keys = tuple((item.question_id, item.video_group_id) for item in values)
    if len(set(keys)) != len(keys):
        raise ValueError("metric records contain duplicate question/video keys")
    return values


def summarize_results(records: Sequence[MetricRecord]) -> RunSummary:
    """Recompute all run-level values; invalid costs remain in cost totals.

    Accuracy and rates use valid questions only. Resource totals include invalid
    attempts so budget failures cannot make a policy look cheaper.
    """

    values = _validated_records(records)
    valid = tuple(item for item in values if item.invalid_reason is None)
    if not valid:
        raise ValueError("no valid questions are available for metric denominators")
    usages = tuple(_usage(item) for item in values)
    valid_count = len(valid)
    regret_values: list[float] = []
    for item in valid:
        regret = item.oracle_utility - item.realized_utility
        if regret < -1e-12:
            raise ValueError("realized utility cannot exceed oracle utility")
        regret_values.append(max(0.0, regret))
    actions = tuple(action for item in valid for action in item.actions)
    if not actions:
        raise ValueError("action distribution requires at least one valid action")
    action_counts = Counter(action.action_type.value for action in actions)
    classifications = tuple(classify_error(item.signals) for item in valid)
    primary_counts = Counter(
        value.primary.value for value in classifications if value.primary is not None
    )
    total_cost = sum(item.total_cost for item in usages)
    total_gpu = sum(item.gpu_seconds for item in usages)
    total_wall = sum(item.wall_seconds for item in usages)
    return RunSummary(
        total_questions=len(values),
        valid_questions=valid_count,
        invalid_questions=len(values) - valid_count,
        accuracy=sum(item.is_correct for item in valid) / valid_count,
        total_cost=total_cost,
        mean_total_cost=total_cost / len(values),
        total_gpu_seconds=total_gpu,
        mean_gpu_seconds=total_gpu / len(values),
        total_wall_seconds=total_wall,
        mean_wall_seconds=total_wall / len(values),
        total_input_frames=sum(item.input_frames for item in usages),
        total_visual_tokens=sum(item.visual_tokens for item in usages),
        total_text_tokens=sum(item.text_tokens for item in usages),
        peak_memory_bytes=max(item.peak_memory_bytes for item in usages),
        oracle_utility_regret=sum(regret_values) / valid_count,
        premature_stop_rate=(
            sum(item.signals.stopped_with_insufficient_evidence for item in valid)
            / valid_count
        ),
        unnecessary_expansion_rate=(
            sum(item.signals.unnecessary_expansion for item in valid) / valid_count
        ),
        top_k_recall=(
            sum(item.signals.gist_top_k_contains_answer for item in valid)
            / valid_count
        ),
        action_distribution={
            name: count / len(actions) for name, count in sorted(action_counts.items())
        },
        primary_error_distribution=dict(sorted(primary_counts.items())),
    )


def fixed_budget_accuracy(
    records: Sequence[MetricRecord], budget: float
) -> float:
    """Accuracy at a hard budget, with over-budget valid attempts counted wrong."""

    if (
        not isinstance(budget, (int, float))
        or not math.isfinite(budget)
        or budget < 0
    ):
        raise ValueError("fixed budget must be finite and non-negative")
    values = _validated_records(records)
    valid = tuple(item for item in values if item.invalid_reason is None)
    if not valid:
        raise ValueError("no valid questions are available for fixed-budget accuracy")
    return sum(
        item.is_correct and _usage(item).total_cost <= budget for item in valid
    ) / len(valid)


def pareto_frontier(points: Sequence[RunPoint]) -> tuple[RunPoint, ...]:
    """Return stable non-dominated accuracy-max/cost-min run points.

    Exact coordinate ties retain the lexicographically first run identity.
    """

    values = tuple(points)
    if not values:
        raise ValueError("Pareto input must be non-empty")
    ordered = sorted(
        values,
        key=lambda item: (
            item.total_cost,
            -item.accuracy,
            item.run_id,
            item.policy_name,
            item.seed,
        ),
    )
    unique: list[RunPoint] = []
    seen: set[tuple[float, float]] = set()
    for item in ordered:
        coordinate = (item.total_cost, item.accuracy)
        if coordinate not in seen:
            unique.append(item)
            seen.add(coordinate)
    frontier = tuple(
        item
        for item in unique
        if not any(
            other.total_cost <= item.total_cost
            and other.accuracy >= item.accuracy
            and (
                other.total_cost < item.total_cost
                or other.accuracy > item.accuracy
            )
            for other in unique
        )
    )
    if not frontier:
        raise ValueError("Pareto frontier cannot be empty")
    return frontier


def cost_at_accuracy(points: Sequence[RunPoint], threshold: float) -> float | None:
    """Return minimum finite cost reaching threshold, or ``None`` if unreachable."""

    values = tuple(points)
    if not values:
        raise ValueError("Cost@Accuracy input must be non-empty")
    if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError("accuracy threshold must be finite and in [0, 1]")
    eligible = tuple(item.total_cost for item in values if item.accuracy >= threshold)
    return min(eligible) if eligible else None


__all__ = [
    "ResourceUsage",
    "RunPoint",
    "RunSummary",
    "cost_at_accuracy",
    "fixed_budget_accuracy",
    "pareto_frontier",
    "summarize_results",
]
