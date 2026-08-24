"""Identity-bound fair benchmark execution over the Task 7 runner."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fidmem.actions.environment import MemoryEnvironment, OperationMetadata
from fidmem.agent.answerer import FrozenAnswerer
from fidmem.agent.runner import AgentRunner
from fidmem.types import ActionInstance, EvidenceItem, FidelityLevel, RouterState

from .baselines import ControllerCost
from .error_taxonomy import ErrorSignals
from .metrics import ResourceUsage, RunSummary, summarize_results


_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class EvaluationIntegrityError(ValueError):
    pass


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=lambda item: (
            item.model_dump(mode="json")
            if isinstance(item, BaseModel)
            else TypeError(f"unsupported canonical value: {type(item).__name__}")
        ),
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


class BenchmarkQuestionRef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    question_id: str = Field(min_length=1)
    video_group_id: str = Field(min_length=1)
    record_sha256: str = Field(pattern=_SHA256_PATTERN)


class BenchmarkManifest(BaseModel):
    """Formal evaluation lineage. Training records cannot inhabit this schema."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal[1] = 1
    benchmark_id: str = Field(min_length=1)
    benchmark_version: str = Field(min_length=1)
    split: Literal["dev", "validation", "test"]
    provenance_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    group_assignment_sha256: str = Field(pattern=_SHA256_PATTERN)
    leakage_audit_sha256: str = Field(pattern=_SHA256_PATTERN)
    leakage_audit_status: Literal["passed"] = "passed"
    questions: tuple[BenchmarkQuestionRef, ...] = Field(min_length=1)
    manifest_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def identities_and_self_hash_match(self) -> "BenchmarkManifest":
        keys = tuple(
            (item.question_id, item.video_group_id) for item in self.questions
        )
        if len(set(keys)) != len(keys):
            raise ValueError("benchmark contains duplicate question/video keys")
        if len({item.question_id for item in self.questions}) != len(self.questions):
            raise ValueError("benchmark contains duplicate question ids")
        payload = self.model_dump(mode="json", exclude={"manifest_sha256"})
        if _sha256(payload) != self.manifest_sha256:
            raise ValueError("benchmark manifest self hash mismatch")
        return self

    @classmethod
    def create(cls, **values: object) -> "BenchmarkManifest":
        payload = {"schema_version": 1, "leakage_audit_status": "passed", **values}
        payload["manifest_sha256"] = _sha256(payload)
        return cls.model_validate(payload)


class SharedEvaluationIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False, strict=True, revalidate_instances="always")

    environment_sha256: str = Field(pattern=_SHA256_PATTERN)
    answerer_template_sha256: str = Field(pattern=_SHA256_PATTERN)
    answerer_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    cache_graph_sha256: str = Field(pattern=_SHA256_PATTERN)
    cost_table_sha256: str = Field(pattern=_SHA256_PATTERN)
    max_visual_frames: int = Field(ge=0)
    max_evidence_tokens: int = Field(ge=0)
    max_total_cost: float = Field(ge=0)


def frozen_answerer_template_sha256(answerer: FrozenAnswerer) -> str:
    if type(answerer) is not FrozenAnswerer:
        raise TypeError("evaluation requires the exact FrozenAnswerer implementation")
    sentinel = EvidenceItem(
        event_id="sentinel",
        fidelity_level=FidelityLevel.GIST,
        content="sentinel-content",
        score=0.0,
    )
    return hashlib.sha256(
        answerer.render_prompt("sentinel-question", ("A", "B"), (sentinel,)).encode(
            "utf-8"
        )
    ).hexdigest()


@dataclass(frozen=True)
class AnswererBinding:
    answerer: FrozenAnswerer
    template_sha256: str
    config_sha256: str

    @classmethod
    def create(
        cls, answerer: FrozenAnswerer, *, config_sha256: str
    ) -> "AnswererBinding":
        if (
            len(config_sha256) != 64
            or any(character not in "0123456789abcdef" for character in config_sha256)
        ):
            raise ValueError("Answerer config identity must be a SHA-256 digest")
        return cls(
            answerer=answerer,
            template_sha256=frozen_answerer_template_sha256(answerer),
            config_sha256=config_sha256,
        )


def cost_table_sha256(environment: MemoryEnvironment) -> str:
    return _sha256(environment.costs.model_dump(mode="json"))


def environment_sha256(environment: MemoryEnvironment) -> str:
    executor = environment.executor
    executor_identity = (
        f"{getattr(executor, '__module__', type(executor).__module__)}:"
        f"{getattr(executor, '__qualname__', type(executor).__qualname__)}"
    )
    return _sha256(
        {
            "events": tuple(
                event.model_dump(mode="json")
                for event in environment.canonical_events
            ),
            "costs": environment.costs.model_dump(mode="json"),
            "action_semantics_version": environment.action_semantics_version,
            "executor_identity": executor_identity,
        }
    )


def build_shared_identity(
    *,
    environment: MemoryEnvironment,
    answerer: AnswererBinding,
    cache_graph_sha256: str,
    max_visual_frames: int,
    max_evidence_tokens: int,
    max_total_cost: float,
) -> SharedEvaluationIdentity:
    return SharedEvaluationIdentity(
        environment_sha256=environment_sha256(environment),
        answerer_template_sha256=answerer.template_sha256,
        answerer_config_sha256=answerer.config_sha256,
        cache_graph_sha256=cache_graph_sha256,
        cost_table_sha256=cost_table_sha256(environment),
        max_visual_frames=max_visual_frames,
        max_evidence_tokens=max_evidence_tokens,
        max_total_cost=max_total_cost,
    )


class EvaluationQuestion(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
        allow_inf_nan=False,
        strict=True,
    )

    question_id: str = Field(min_length=1)
    video_group_id: str = Field(min_length=1)
    record_sha256: str = Field(pattern=_SHA256_PATTERN)
    initial_state: RouterState
    gold_answer: str = Field(min_length=1)
    environment: MemoryEnvironment
    cache_graph_sha256: str = Field(pattern=_SHA256_PATTERN)
    oracle_utility: float
    signals: ErrorSignals

    @model_validator(mode="after")
    def answer_is_multiple_choice(self) -> "EvaluationQuestion":
        if self.gold_answer not in self.initial_state.options:
            raise ValueError("gold answer must be one of the multiple-choice options")
        return self


class RawQuestionResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False, strict=True, revalidate_instances="always")

    schema_version: Literal[1] = 1
    run_id: str = Field(min_length=1)
    policy_name: str = Field(min_length=1)
    policy_family: Literal["fixed", "adaptive", "learned"]
    policy_identity_sha256: str = Field(pattern=_SHA256_PATTERN)
    seed: int
    question_id: str = Field(min_length=1)
    video_group_id: str = Field(min_length=1)
    record_sha256: str = Field(pattern=_SHA256_PATTERN)
    benchmark_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    split: Literal["dev", "validation", "test"]
    shared: SharedEvaluationIdentity
    cost_preference: float = Field(ge=0, le=1)
    predicted_answer: str = Field(min_length=1)
    gold_answer: str = Field(min_length=1)
    is_correct: bool
    invalid_reason: str | None = None
    acquisition_usage: ResourceUsage
    controller_usage: ResourceUsage
    actions: tuple[ActionInstance, ...] = Field(min_length=1)
    forced_stop: bool = False
    oracle_utility: float
    realized_utility: float
    signals: ErrorSignals

    @model_validator(mode="after")
    def answer_cost_and_invalidity_are_consistent(self) -> "RawQuestionResult":
        if self.is_correct != (self.predicted_answer == self.gold_answer):
            raise ValueError("is_correct must match predicted and gold answers")
        if self.invalid_reason is not None and not self.invalid_reason.strip():
            raise ValueError("invalid reason must be non-empty when present")
        if self.realized_utility > self.oracle_utility + 1e-12:
            raise ValueError("realized utility cannot exceed oracle utility")
        return self

    @property
    def total_usage(self) -> ResourceUsage:
        return self.acquisition_usage.plus(self.controller_usage)


def raw_results_sha256(records: Sequence[RawQuestionResult]) -> str:
    return _sha256(tuple(item.model_dump(mode="json") for item in records))


class RunManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False, strict=True)

    schema_version: Literal[1] = 1
    run_id: str = Field(min_length=1)
    policy_name: str = Field(min_length=1)
    policy_family: Literal["fixed", "adaptive", "learned"]
    policy_identity_sha256: str = Field(pattern=_SHA256_PATTERN)
    seed: int
    benchmark: BenchmarkManifest
    shared: SharedEvaluationIdentity
    cost_preference: float = Field(ge=0, le=1)
    question_keys: tuple[tuple[str, str], ...] = Field(min_length=1)
    raw_results_sha256: str = Field(pattern=_SHA256_PATTERN)
    manifest_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def keys_and_self_hash_match(self) -> "RunManifest":
        if len(set(self.question_keys)) != len(self.question_keys):
            raise ValueError("run contains duplicate question/video keys")
        payload = self.model_dump(mode="json", exclude={"manifest_sha256"})
        if _sha256(payload) != self.manifest_sha256:
            raise ValueError("run manifest self hash mismatch")
        return self

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        policy_name: str,
        policy_family: Literal["fixed", "adaptive", "learned"],
        policy_identity_sha256: str,
        seed: int,
        benchmark: BenchmarkManifest,
        shared: SharedEvaluationIdentity,
        cost_preference: float,
        records: Sequence[RawQuestionResult],
    ) -> "RunManifest":
        values = tuple(
            RawQuestionResult.model_validate(item) for item in records
        )
        if not values:
            raise ValueError("run records must be non-empty")
        keys = tuple((item.question_id, item.video_group_id) for item in values)
        if len(set(keys)) != len(keys):
            raise ValueError("run contains duplicate question/video keys")
        expected_identity = (
            run_id,
            policy_name,
            policy_family,
            policy_identity_sha256,
            seed,
            benchmark.manifest_sha256,
            shared,
            cost_preference,
            benchmark.split,
        )
        benchmark_records = {
            (item.question_id, item.video_group_id): item.record_sha256
            for item in benchmark.questions
        }
        for item in values:
            actual_identity = (
                item.run_id,
                item.policy_name,
                item.policy_family,
                item.policy_identity_sha256,
                item.seed,
                item.benchmark_manifest_sha256,
                item.shared,
                item.cost_preference,
                item.split,
            )
            if actual_identity != expected_identity:
                raise ValueError("raw result identity differs from run identity")
            key = (item.question_id, item.video_group_id)
            if benchmark_records.get(key) != item.record_sha256:
                raise ValueError(
                    "raw result differs from benchmark question provenance"
                )
        payload: dict[str, object] = {
            "schema_version": 1,
            "run_id": run_id,
            "policy_name": policy_name,
            "policy_family": policy_family,
            "policy_identity_sha256": policy_identity_sha256,
            "seed": seed,
            "benchmark": benchmark,
            "shared": shared,
            "cost_preference": cost_preference,
            "question_keys": keys,
            "raw_results_sha256": raw_results_sha256(values),
        }
        payload["manifest_sha256"] = _sha256(payload)
        return cls.model_validate(payload)


class EvaluationRun(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    manifest: RunManifest
    records: tuple[RawQuestionResult, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def records_are_bound_to_manifest(self) -> "EvaluationRun":
        if raw_results_sha256(self.records) != self.manifest.raw_results_sha256:
            raise ValueError("raw results do not match run manifest identity")
        keys = tuple((item.question_id, item.video_group_id) for item in self.records)
        if keys != self.manifest.question_keys:
            raise ValueError("raw result order/keys do not match run manifest")
        expected = (
            self.manifest.run_id,
            self.manifest.policy_name,
            self.manifest.policy_family,
            self.manifest.policy_identity_sha256,
            self.manifest.seed,
            self.manifest.benchmark.manifest_sha256,
            self.manifest.shared,
            self.manifest.cost_preference,
            self.manifest.benchmark.split,
        )
        for item in self.records:
            actual = (
                item.run_id,
                item.policy_name,
                item.policy_family,
                item.policy_identity_sha256,
                item.seed,
                item.benchmark_manifest_sha256,
                item.shared,
                item.cost_preference,
                item.split,
            )
            if actual != expected:
                raise ValueError("raw result identity differs from run manifest")
        return self

    @property
    def summary(self) -> RunSummary:
        return summarize_results(self.records)


class _FairPolicy:
    def __init__(
        self,
        policy: Callable[[RouterState, tuple[ActionInstance, ...]], ActionInstance],
    ) -> None:
        self.policy = policy
        self.controller_usage = ResourceUsage()

    def __call__(
        self, state: RouterState, legal_actions: tuple[ActionInstance, ...]
    ) -> ActionInstance:
        private_state = state.model_copy(deep=True)
        try:
            selected = self.policy(private_state, legal_actions)
        except EvaluationIntegrityError:
            raise
        except Exception as error:
            raise EvaluationIntegrityError("policy failed under the fair action mask") from error
        if not isinstance(selected, ActionInstance) or not any(
            selected is action for action in legal_actions
        ):
            raise EvaluationIntegrityError(
                "policy must return the exact ActionInstance from environment.valid_actions"
            )
        consume = getattr(self.policy, "consume_last_controller_cost", None)
        if callable(consume):
            raw = consume()
            if not isinstance(raw, ControllerCost):
                raise EvaluationIntegrityError("controller cost record is invalid")
            self.controller_usage = self.controller_usage.plus(
                ResourceUsage.model_validate(raw.model_dump(mode="python"))
            )
        return selected


def _metadata_usage(metadata: Sequence[OperationMetadata]) -> ResourceUsage:
    gpu_seconds = 0.0
    wall_seconds = 0.0
    frames = 0
    visual_tokens = 0
    text_tokens = 0
    peak = 0
    for item in metadata:
        frames += item.input_frames
        visual_tokens += item.visual_tokens
        text_tokens += item.text_tokens
        if item.cost_record is not None:
            record = item.cost_record
            gpu_seconds += record.gpu_seconds
            wall_seconds += record.wall_seconds
            peak = max(peak, record.peak_memory_bytes)
            if (
                record.input_frames != item.input_frames
                or record.visual_tokens != item.visual_tokens
                or record.text_tokens != item.text_tokens
            ):
                raise EvaluationIntegrityError(
                    "operation metadata and measured CostRecord disagree"
                )
    return ResourceUsage(
        gpu_seconds=gpu_seconds,
        wall_seconds=wall_seconds,
        input_frames=frames,
        visual_tokens=visual_tokens,
        text_tokens=text_tokens,
        peak_memory_bytes=peak,
    )


def _preflight(
    *,
    questions: tuple[EvaluationQuestion, ...],
    benchmark: BenchmarkManifest,
    answerer: AnswererBinding,
    shared: SharedEvaluationIdentity,
    cost_preference: float,
) -> None:
    if not questions:
        raise EvaluationIntegrityError("evaluation questions must be non-empty")
    keys = tuple((item.question_id, item.video_group_id) for item in questions)
    if len(set(keys)) != len(keys):
        raise EvaluationIntegrityError("duplicate question/video keys")
    refs = {
        (item.question_id, item.video_group_id): item.record_sha256
        for item in benchmark.questions
    }
    if set(keys) != set(refs):
        raise EvaluationIntegrityError("questions do not exactly match benchmark manifest")
    if (
        answerer.template_sha256 != shared.answerer_template_sha256
        or answerer.config_sha256 != shared.answerer_config_sha256
    ):
        raise EvaluationIntegrityError("Answerer identity differs from shared identity")
    if frozen_answerer_template_sha256(answerer.answerer) != answerer.template_sha256:
        raise EvaluationIntegrityError("Answerer template behavior changed")
    for item in questions:
        if refs[(item.question_id, item.video_group_id)] != item.record_sha256:
            raise EvaluationIntegrityError("benchmark record provenance identity mismatch")
        if environment_sha256(item.environment) != shared.environment_sha256:
            raise EvaluationIntegrityError("environment identity differs from shared identity")
        if cost_table_sha256(item.environment) != shared.cost_table_sha256:
            raise EvaluationIntegrityError("cost table identity differs from shared identity")
        if item.cache_graph_sha256 != shared.cache_graph_sha256:
            raise EvaluationIntegrityError("cache graph identity differs from shared identity")
        if item.initial_state.cost_preference != cost_preference:
            raise EvaluationIntegrityError("question cost preference differs from run")


def evaluate_run(
    *,
    run_id: str,
    policy_name: str,
    policy_family: Literal["fixed", "adaptive", "learned"],
    policy_identity_sha256: str,
    policy: Callable[[RouterState, tuple[ActionInstance, ...]], ActionInstance],
    questions: Sequence[EvaluationQuestion],
    benchmark: BenchmarkManifest,
    answerer: AnswererBinding,
    shared: SharedEvaluationIdentity,
    seed: int,
    cost_preference: float,
) -> EvaluationRun:
    """Evaluate one seed/policy using only Task 7 masks, runner, and Answerer."""

    values = tuple(questions)
    _preflight(
        questions=values,
        benchmark=benchmark,
        answerer=answerer,
        shared=shared,
        cost_preference=cost_preference,
    )
    records: list[RawQuestionResult] = []
    for question in values:
        guarded = _FairPolicy(policy)
        result = AgentRunner(
            question.environment, guarded, answerer.answerer
        ).run(
            question.initial_state,
            run_id=f"{run_id}:{question.question_id}:{question.video_group_id}",
        )
        acquisition = ResourceUsage(
            total_cost=sum(item.step_cost for item in result.transitions)
        )
        for transition in result.transitions:
            acquisition = acquisition.plus(
                _metadata_usage(transition.operation_metadata)
            )
        combined = acquisition.plus(guarded.controller_usage)
        evidence_tokens = acquisition.visual_tokens + acquisition.text_tokens
        total_frames = combined.input_frames
        if total_frames > shared.max_visual_frames:
            invalid_reason = "visual_frame_budget_exceeded"
        elif evidence_tokens > shared.max_evidence_tokens:
            invalid_reason = "evidence_token_budget_exceeded"
        elif combined.total_cost > shared.max_total_cost:
            invalid_reason = "total_cost_budget_exceeded"
        else:
            invalid_reason = None
        correct = result.answer.answer == question.gold_answer
        realized_utility = (1.0 if correct else 0.0) - (
            cost_preference * combined.total_cost
        )
        records.append(
            RawQuestionResult(
                run_id=run_id,
                policy_name=policy_name,
                policy_family=policy_family,
                policy_identity_sha256=policy_identity_sha256,
                seed=seed,
                question_id=question.question_id,
                video_group_id=question.video_group_id,
                record_sha256=question.record_sha256,
                benchmark_manifest_sha256=benchmark.manifest_sha256,
                split=benchmark.split,
                shared=shared,
                cost_preference=cost_preference,
                predicted_answer=result.answer.answer,
                gold_answer=question.gold_answer,
                is_correct=correct,
                invalid_reason=invalid_reason,
                acquisition_usage=acquisition,
                controller_usage=guarded.controller_usage,
                actions=tuple(item.action for item in result.transitions),
                forced_stop=result.forced_stop,
                oracle_utility=question.oracle_utility,
                realized_utility=realized_utility,
                signals=question.signals,
            )
        )
    manifest = RunManifest.create(
        run_id=run_id,
        policy_name=policy_name,
        policy_family=policy_family,
        policy_identity_sha256=policy_identity_sha256,
        seed=seed,
        benchmark=benchmark,
        shared=shared,
        cost_preference=cost_preference,
        records=records,
    )
    return EvaluationRun(manifest=manifest, records=tuple(records))


__all__ = [
    "AnswererBinding",
    "BenchmarkManifest",
    "BenchmarkQuestionRef",
    "EvaluationIntegrityError",
    "EvaluationQuestion",
    "EvaluationRun",
    "RawQuestionResult",
    "RunManifest",
    "SharedEvaluationIdentity",
    "build_shared_identity",
    "cost_table_sha256",
    "environment_sha256",
    "evaluate_run",
    "frozen_answerer_template_sha256",
    "raw_results_sha256",
]
