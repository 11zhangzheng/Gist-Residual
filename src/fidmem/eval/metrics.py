"""Metrics recomputed exclusively from sealed per-question run records."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from .runner import EvaluationRun


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


class ResourceUsage(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        strict=True,
        revalidate_instances="always",
    )

    total_cost: float = Field(default=0.0, ge=0)
    gpu_seconds: float = Field(default=0.0, ge=0)
    wall_seconds: float = Field(default=0.0, ge=0)
    input_frames: float = Field(default=0.0, ge=0)
    visual_tokens: float = Field(default=0.0, ge=0)
    text_tokens: float = Field(default=0.0, ge=0)
    peak_memory_bytes: int = Field(default=0, ge=0)

    def plus(self, other: "ResourceUsage") -> "ResourceUsage":
        left = ResourceUsage.model_validate(self.model_dump(mode="python"))
        right = ResourceUsage.model_validate(other.model_dump(mode="python"))
        return ResourceUsage(
            total_cost=left.total_cost + right.total_cost,
            gpu_seconds=left.gpu_seconds + right.gpu_seconds,
            wall_seconds=left.wall_seconds + right.wall_seconds,
            input_frames=left.input_frames + right.input_frames,
            visual_tokens=left.visual_tokens + right.visual_tokens,
            text_tokens=left.text_tokens + right.text_tokens,
            peak_memory_bytes=max(left.peak_memory_bytes, right.peak_memory_bytes),
        )

    def divided_by(self, divisor: int) -> "ResourceUsage":
        if not isinstance(divisor, int) or isinstance(divisor, bool) or divisor < 1:
            raise ValueError("usage divisor must be a positive integer")
        return ResourceUsage(
            total_cost=self.total_cost / divisor,
            gpu_seconds=self.gpu_seconds / divisor,
            wall_seconds=self.wall_seconds / divisor,
            input_frames=self.input_frames / divisor,
            visual_tokens=self.visual_tokens / divisor,
            text_tokens=self.text_tokens / divisor,
            peak_memory_bytes=self.peak_memory_bytes,
        )


class CostBreakdown(BaseModel):
    """All resource/cost components retained separately and summed on access."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        revalidate_instances="always",
    )

    base_memory: ResourceUsage = ResourceUsage()
    environment: ResourceUsage = ResourceUsage()
    policy_router: ResourceUsage = ResourceUsage()
    prompt_controller: ResourceUsage = ResourceUsage()
    answerer: ResourceUsage = ResourceUsage()

    @property
    def total(self) -> ResourceUsage:
        value = ResourceUsage()
        for component in (
            self.base_memory,
            self.environment,
            self.policy_router,
            self.prompt_controller,
            self.answerer,
        ):
            value = value.plus(component)
        return value


class RunSummary(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        strict=True,
        revalidate_instances="always",
    )

    total_questions: int = Field(ge=1)
    valid_questions: int = Field(ge=0)
    invalid_questions: int = Field(ge=0)
    accuracy: float = Field(ge=0, le=1)
    valid_only_accuracy: float | None = Field(default=None, ge=0, le=1)
    invalid_rate: float = Field(ge=0, le=1)
    total_cost: float = Field(ge=0)
    mean_total_cost: float = Field(ge=0)
    total_gpu_seconds: float = Field(ge=0)
    mean_gpu_seconds: float = Field(ge=0)
    total_wall_seconds: float = Field(ge=0)
    mean_wall_seconds: float = Field(ge=0)
    total_input_frames: float = Field(ge=0)
    total_visual_tokens: float = Field(ge=0)
    total_text_tokens: float = Field(ge=0)
    peak_memory_bytes: int = Field(ge=0)
    oracle_utility_regret: float = Field(ge=0)
    premature_stop_rate: float = Field(ge=0, le=1)
    unnecessary_expansion_rate: float = Field(ge=0, le=1)
    top_k_recall: float = Field(ge=0, le=1)
    action_distribution: dict[str, float]
    primary_error_distribution: dict[str, int]


class RunPoint(BaseModel):
    """Sealed output point; public metric functions never accept point inputs."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        strict=True,
        revalidate_instances="always",
    )

    run_id: str = Field(min_length=1)
    policy_name: str = Field(min_length=1)
    seed: int
    accuracy: float = Field(ge=0, le=1)
    total_cost: float = Field(ge=0)
    run_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    point_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def self_hash_matches(self) -> "RunPoint":
        payload = self.model_dump(mode="json", exclude={"point_sha256"})
        expected = hashlib.sha256(_canonical_json(payload).encode()).hexdigest()
        if self.point_sha256 != expected:
            raise ValueError("run point self hash mismatch")
        return self

    @classmethod
    def from_run(cls, run: "EvaluationRun") -> "RunPoint":
        validated = _validated_run(run)
        summary = _summarize_records(validated.records)
        payload = {
            "run_id": validated.manifest.run_id,
            "policy_name": validated.manifest.policy_name,
            "seed": validated.manifest.seed,
            "accuracy": summary.accuracy,
            "total_cost": summary.total_cost,
            "run_manifest_sha256": validated.manifest.manifest_sha256,
        }
        return cls(
            **payload,
            point_sha256=hashlib.sha256(_canonical_json(payload).encode()).hexdigest(),
        )


def _validated_run(run: object) -> "EvaluationRun":
    from .runner import EvaluationRun

    if type(run) is not EvaluationRun:
        raise TypeError("metrics require an exact validated EvaluationRun")
    return EvaluationRun.model_validate_json(run.model_dump_json())


def _summarize_records(records: tuple[object, ...]) -> RunSummary:
    if not records:
        raise ValueError("metric records must be non-empty")
    keys = tuple((record.question_id, record.video_group_id) for record in records)
    if len(set(keys)) != len(keys):
        raise ValueError("metric records contain duplicate question/video keys")
    total = len(records)
    valid = tuple(record for record in records if record.invalid_reason is None)
    usages = tuple(record.cost_breakdown.total for record in records)
    valid_regret = tuple(record.oracle_utility_regret for record in valid)
    actions = tuple(action for record in records for action in record.actions)
    action_counts = Counter(action.action_type.value for action in actions)
    primary_counts = Counter(
        record.error.primary.value
        for record in valid
        if not record.is_correct and record.error.primary is not None
    )
    total_cost = math.fsum(item.total_cost for item in usages)
    total_gpu = math.fsum(item.gpu_seconds for item in usages)
    total_wall = math.fsum(item.wall_seconds for item in usages)
    return RunSummary(
        total_questions=total,
        valid_questions=len(valid),
        invalid_questions=total - len(valid),
        accuracy=sum(
            record.is_correct and record.invalid_reason is None for record in records
        )
        / total,
        valid_only_accuracy=(
            sum(record.is_correct for record in valid) / len(valid) if valid else None
        ),
        invalid_rate=(total - len(valid)) / total,
        total_cost=total_cost,
        mean_total_cost=total_cost / total,
        total_gpu_seconds=total_gpu,
        mean_gpu_seconds=total_gpu / total,
        total_wall_seconds=total_wall,
        mean_wall_seconds=total_wall / total,
        total_input_frames=sum(item.input_frames for item in usages),
        total_visual_tokens=sum(item.visual_tokens for item in usages),
        total_text_tokens=sum(item.text_tokens for item in usages),
        peak_memory_bytes=max(item.peak_memory_bytes for item in usages),
        oracle_utility_regret=(
            math.fsum(valid_regret) / len(valid_regret) if valid_regret else 0.0
        ),
        premature_stop_rate=sum(
            record.signals.stopped_with_insufficient_evidence
            and record.invalid_reason is None
            for record in records
        )
        / total,
        unnecessary_expansion_rate=sum(
            record.signals.unnecessary_expansion and record.invalid_reason is None
            for record in records
        )
        / total,
        top_k_recall=sum(
            record.signals.gist_top_k_contains_answer and record.invalid_reason is None
            for record in records
        )
        / total,
        action_distribution=(
            {
                name: count / len(actions)
                for name, count in sorted(action_counts.items())
            }
            if actions
            else {}
        ),
        primary_error_distribution=dict(sorted(primary_counts.items())),
    )


def summarize_results(run: object) -> RunSummary:
    validated = _validated_run(run)
    return _summarize_records(validated.records)


def fixed_budget_accuracy(run: object, budget: float) -> float:
    if (
        not isinstance(budget, (int, float))
        or isinstance(budget, bool)
        or not math.isfinite(budget)
        or budget < 0
    ):
        raise ValueError("fixed budget must be finite and non-negative")
    validated = _validated_run(run)
    return sum(
        record.invalid_reason is None
        and record.is_correct
        and record.cost_breakdown.total.total_cost <= budget
        for record in validated.records
    ) / len(validated.records)


def _points(runs: object) -> tuple[RunPoint, ...]:
    try:
        values = tuple(runs)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError("metrics require a sequence of EvaluationRun") from error
    if not values:
        raise ValueError("cross-run metric input must be non-empty")
    validated = tuple(_validated_run(run) for run in values)
    authority = (
        validated[0].manifest.benchmark.manifest_sha256,
        validated[0].manifest.shared,
        validated[0].manifest.cost_preference,
    )
    if any(
        (
            run.manifest.benchmark.manifest_sha256,
            run.manifest.shared,
            run.manifest.cost_preference,
        )
        != authority
        for run in validated[1:]
    ):
        raise ValueError("cross-run metrics require comparable shared authority")
    return tuple(RunPoint.from_run(run) for run in validated)


def pareto_frontier(runs: object) -> tuple[RunPoint, ...]:
    values = _points(runs)
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
    return tuple(
        item
        for item in unique
        if not any(
            other.total_cost <= item.total_cost
            and other.accuracy >= item.accuracy
            and (other.total_cost < item.total_cost or other.accuracy > item.accuracy)
            for other in unique
        )
    )


def cost_at_accuracy(runs: object, threshold: float) -> float | None:
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("accuracy threshold must be finite and in [0, 1]")
    eligible = tuple(
        point.total_cost for point in _points(runs) if point.accuracy >= threshold
    )
    return min(eligible) if eligible else None


__all__ = [
    "CostBreakdown",
    "ResourceUsage",
    "RunPoint",
    "RunSummary",
    "cost_at_accuracy",
    "fixed_budget_accuracy",
    "pareto_frontier",
    "summarize_results",
]
