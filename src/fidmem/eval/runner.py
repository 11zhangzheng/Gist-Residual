"""Identity-bound fair benchmark execution over the Task 7 runner."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, model_validator

from fidmem.actions.environment import MemoryEnvironment, OperationMetadata
from fidmem.agent.answerer import (
    FrozenAnswerer,
    callable_identity_sha256,
)
from fidmem.agent.runner import AgentRunner
from fidmem.costs.tracker import CostRecord, CostTracker
from fidmem.oracle.labels import COST_PREFERENCES, CostNormalization
from fidmem.storage.cache import ContentAddressedCache
from fidmem.types import (
    ActionInstance,
    ActionType,
    FidelityLevel,
    RouterState,
)

from .baselines import (
    BCPolicyAdapter,
    ControllerCost,
    DAggerPolicyAdapter,
    FullResidualPolicy,
    GistOnlyPolicy,
    GistResidualPolicy,
    GistVisualPolicy,
    PromptControllerPolicy,
    QuestionOnlyPolicy,
    RulePolicy,
    TextAdaptivePolicy,
    UniformFramesPolicy,
)
from .error_taxonomy import (
    ErrorClassification,
    ErrorSignals,
    classify_error,
)
from .metrics import CostBreakdown, ResourceUsage, RunSummary


_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_FIXED_LONG_POLICIES = (UniformFramesPolicy, FullResidualPolicy)
_BUILTIN_POLICIES = (
    GistOnlyPolicy,
    GistResidualPolicy,
    GistVisualPolicy,
    UniformFramesPolicy,
    FullResidualPolicy,
    RulePolicy,
    PromptControllerPolicy,
    QuestionOnlyPolicy,
    TextAdaptivePolicy,
    BCPolicyAdapter,
    DAggerPolicyAdapter,
)
_EXPECTED_POLICY_FAMILIES = {
    "uniform": "fixed",
    "gist_only": "fixed",
    "gist_residual": "fixed",
    "gist_visual": "fixed",
    "full_residual": "fixed",
    "rule": "adaptive",
    "prompt_vlm": "adaptive",
    "text_adaptive": "adaptive",
    "question_only": "learned",
    "bc": "learned",
    "bc_dagger": "learned",
}


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


def _canonical_source(value: Mapping[str, object] | str) -> str:
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("source manifest must be canonical JSON") from error
    elif isinstance(value, Mapping):
        loaded = dict(value)
    else:
        raise TypeError("source manifest must be a mapping or canonical JSON")
    if not isinstance(loaded, dict):
        raise ValueError("source manifest must contain a JSON object")
    return _canonical_json(loaded)


def _resource_from_record(
    record: CostRecord,
    *,
    total_cost: float = 0.0,
) -> ResourceUsage:
    record.validate_values()
    return ResourceUsage(
        total_cost=total_cost,
        gpu_seconds=record.gpu_seconds,
        wall_seconds=record.wall_seconds,
        input_frames=record.input_frames,
        visual_tokens=record.visual_tokens,
        text_tokens=record.text_tokens,
        peak_memory_bytes=record.peak_memory_bytes,
    )


@dataclass(frozen=True)
class CacheBinding:
    """Actual cache namespace whose content identity is recomputed on access."""

    cache: ContentAddressedCache
    namespace: str

    def __post_init__(self) -> None:
        if type(self.cache) is not ContentAddressedCache:
            raise TypeError("evaluation cache must be ContentAddressedCache")
        if not isinstance(self.namespace, str) or not self.namespace.strip():
            raise ValueError("cache namespace must not be blank")

    @property
    def identity_sha256(self) -> str:
        files: list[dict[str, object]] = []
        for path in sorted(self.cache.root.glob("*.json"), key=lambda item: item.name):
            if not path.is_file() or path.is_symlink():
                raise EvaluationIntegrityError(
                    "cache content must contain regular JSON files"
                )
            raw = path.read_bytes()
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError) as error:
                raise EvaluationIntegrityError(
                    "cache content is not canonical JSON"
                ) from error
            canonical = _canonical_json(payload).encode("utf-8")
            if canonical != raw:
                raise EvaluationIntegrityError(
                    "cache content is not canonically serialized"
                )
            files.append(
                {
                    "key": path.stem,
                    "content_sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
        return _sha256(
            {
                "namespace": self.namespace,
                "files": tuple(files),
            }
        )


class EvaluationBudgets(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        revalidate_instances="always",
    )

    max_visual_frames: int = Field(ge=0)
    max_evidence_tokens: int = Field(ge=0)
    max_total_cost: float = Field(ge=0)


class OracleEvaluationAuthority(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        revalidate_instances="always",
    )

    answer_score: float = Field(ge=0, le=1)
    total_cost: float = Field(ge=0)
    action_types: tuple[ActionType, ...] = Field(min_length=1)
    gold_support_event_ids: tuple[str, ...] = Field(min_length=1)
    required_fidelity: FidelityLevel
    answerer_correct_with_oracle_evidence: bool
    authority_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def identity_and_support_are_valid(self) -> "OracleEvaluationAuthority":
        if len(set(self.gold_support_event_ids)) != len(self.gold_support_event_ids):
            raise ValueError("Oracle support event ids must be unique")
        payload = self.model_dump(mode="json", exclude={"authority_sha256"})
        if _sha256(payload) != self.authority_sha256:
            raise ValueError("Oracle evaluation authority self hash mismatch")
        return self

    @classmethod
    def create(
        cls,
        *,
        answer_score: float,
        total_cost: float,
        action_types: Sequence[ActionType],
        gold_support_event_ids: Sequence[str],
        required_fidelity: FidelityLevel,
        answerer_correct_with_oracle_evidence: bool,
    ) -> "OracleEvaluationAuthority":
        payload = {
            "answer_score": answer_score,
            "total_cost": total_cost,
            "action_types": tuple(action_types),
            "gold_support_event_ids": tuple(gold_support_event_ids),
            "required_fidelity": required_fidelity,
            "answerer_correct_with_oracle_evidence": (
                answerer_correct_with_oracle_evidence
            ),
        }
        return cls(**payload, authority_sha256=_sha256(payload))

    def utility(
        self,
        *,
        cost_preference: float,
        normalization: CostNormalization,
    ) -> float:
        return self.answer_score - cost_preference * (
            self.total_cost / normalization.constant
        )


class BaseMemoryCostAuthority(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        revalidate_instances="always",
    )

    video_group_id: str = Field(min_length=1)
    usage: ResourceUsage
    artifact_name: str = Field(min_length=1)
    authority_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def self_hash_matches(self) -> "BaseMemoryCostAuthority":
        payload = self.model_dump(mode="json", exclude={"authority_sha256"})
        if _sha256(payload) != self.authority_sha256:
            raise ValueError("base memory cost authority self hash mismatch")
        return self

    @classmethod
    def create(
        cls,
        *,
        video_group_id: str,
        usage: ResourceUsage,
        artifact_name: str,
    ) -> "BaseMemoryCostAuthority":
        validated = ResourceUsage.model_validate(usage.model_dump(mode="python"))
        payload = {
            "video_group_id": video_group_id,
            "usage": validated,
            "artifact_name": artifact_name,
        }
        return cls(**payload, authority_sha256=_sha256(payload))


class HardwareAssignment(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        revalidate_instances="always",
    )

    training: Literal["A800"]
    evaluation: Literal["V100"]


def cost_table_sha256(environment: MemoryEnvironment) -> str:
    if type(environment) is not MemoryEnvironment and not isinstance(
        environment, MemoryEnvironment
    ):
        raise TypeError("environment must be MemoryEnvironment")
    return _sha256(environment.costs.model_dump(mode="json"))


def environment_sha256(
    environment: MemoryEnvironment,
    cache: CacheBinding,
) -> str:
    if not isinstance(environment, MemoryEnvironment):
        raise TypeError("environment must be MemoryEnvironment")
    if not isinstance(cache, CacheBinding):
        raise TypeError("environment identity requires actual CacheBinding")
    return _sha256(
        {
            "events": tuple(
                event.model_dump(mode="json") for event in environment.canonical_events
            ),
            "costs": environment.costs.model_dump(mode="json"),
            "action_semantics_version": environment.action_semantics_version,
            "executor_sha256": callable_identity_sha256(environment.executor),
            "cache_identity_sha256": cache.identity_sha256,
        }
    )


class EvaluationQuestion(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
        allow_inf_nan=False,
        strict=True,
        revalidate_instances="always",
    )

    question_id: str = Field(min_length=1)
    video_group_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    split: Literal["dev", "validation", "test"]
    source_manifest_canonical_json: str = Field(min_length=2)
    initial_state: RouterState
    gold_answer: str = Field(min_length=1)
    environment: MemoryEnvironment
    cache: CacheBinding
    budgets: EvaluationBudgets
    oracle: OracleEvaluationAuthority

    @classmethod
    def create(
        cls,
        *,
        question_id: str,
        video_group_id: str,
        video_id: str,
        split: Literal["dev", "validation", "test"],
        source_manifest: Mapping[str, object] | str,
        initial_state: RouterState,
        gold_answer: str,
        environment: MemoryEnvironment,
        cache: CacheBinding,
        budgets: EvaluationBudgets,
        oracle: OracleEvaluationAuthority,
    ) -> "EvaluationQuestion":
        return cls(
            question_id=question_id,
            video_group_id=video_group_id,
            video_id=video_id,
            split=split,
            source_manifest_canonical_json=_canonical_source(source_manifest),
            initial_state=initial_state,
            gold_answer=gold_answer,
            environment=environment,
            cache=cache,
            budgets=budgets,
            oracle=oracle,
        ).validate_authority()

    @property
    def uniform_event_order(self) -> tuple[str, ...]:
        return tuple(event.event_id for event in self.environment.canonical_events)

    def _record_payload(self) -> dict[str, object]:
        return {
            "question_id": self.question_id,
            "video_group_id": self.video_group_id,
            "video_id": self.video_id,
            "split": self.split,
            "source_manifest": json.loads(self.source_manifest_canonical_json),
            "question": self.initial_state.question,
            "options": self.initial_state.options,
            "gold_answer": self.gold_answer,
            "initial_state": self.initial_state.model_dump(mode="json"),
            "canonical_events": tuple(
                event.model_dump(mode="json")
                for event in self.environment.canonical_events
            ),
            "environment_sha256": environment_sha256(self.environment, self.cache),
            "cache_identity_sha256": self.cache.identity_sha256,
            "budgets": self.budgets.model_dump(mode="json"),
            "oracle": self.oracle.model_dump(mode="json"),
        }

    @property
    def record_sha256(self) -> str:
        return _sha256(self._record_payload())

    def validate_authority(self) -> "EvaluationQuestion":
        if type(self.environment) is not MemoryEnvironment and not isinstance(
            self.environment, MemoryEnvironment
        ):
            raise TypeError("question environment must be MemoryEnvironment")
        if not isinstance(self.cache, CacheBinding):
            raise TypeError("question cache must be CacheBinding")
        EvaluationBudgets.model_validate(self.budgets.model_dump(mode="python"))
        OracleEvaluationAuthority.model_validate(self.oracle.model_dump(mode="python"))
        try:
            source = json.loads(self.source_manifest_canonical_json)
        except json.JSONDecodeError as error:
            raise ValueError("source manifest is invalid JSON") from error
        if _canonical_json(source) != self.source_manifest_canonical_json:
            raise ValueError("source manifest is not canonical")
        if self.gold_answer not in self.initial_state.options:
            raise ValueError("gold answer must be one of the multiple-choice options")
        if self.initial_state.remaining_budget != self.budgets.max_total_cost:
            raise ValueError(
                "initial_state remaining budget must exactly match shared budget"
            )
        if not self.initial_state.question.strip():
            raise ValueError("question must not be blank")
        event_ids = set(self.uniform_event_order)
        if not set(self.oracle.gold_support_event_ids).issubset(event_ids):
            raise ValueError("Oracle support events are outside canonical video")
        return self


class BenchmarkQuestionRef(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        revalidate_instances="always",
    )

    question_id: str = Field(min_length=1)
    video_group_id: str = Field(min_length=1)
    record_sha256: str = Field(pattern=_SHA256_PATTERN)
    gold_answer_sha256: str = Field(pattern=_SHA256_PATTERN)


class BenchmarkManifest(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        revalidate_instances="always",
    )

    schema_version: Literal[2] = 2
    benchmark_id: str = Field(min_length=1)
    benchmark_version: str = Field(min_length=1)
    split: Literal["dev", "validation", "test"]
    provenance_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    group_assignment_sha256: str = Field(pattern=_SHA256_PATTERN)
    leakage_audit_canonical_json: str = Field(min_length=2)
    leakage_audit_sha256: str = Field(pattern=_SHA256_PATTERN)
    questions: tuple[BenchmarkQuestionRef, ...] = Field(min_length=1)
    base_memory_costs: tuple[BaseMemoryCostAuthority, ...] = Field(min_length=1)
    normalization: CostNormalization
    normalization_sha256: str = Field(pattern=_SHA256_PATTERN)
    gpu_assignment: HardwareAssignment
    manifest_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def identities_and_self_hash_match(self) -> "BenchmarkManifest":
        keys = tuple((item.question_id, item.video_group_id) for item in self.questions)
        if len(set(keys)) != len(keys):
            raise ValueError("benchmark contains duplicate question/video keys")
        if len({item.question_id for item in self.questions}) != len(self.questions):
            raise ValueError("benchmark contains duplicate question ids")
        source = json.loads(self.leakage_audit_canonical_json)
        if _canonical_json(source) != self.leakage_audit_canonical_json:
            raise ValueError("leakage audit is not canonical")
        if _sha256(source) != self.leakage_audit_sha256:
            raise ValueError("leakage audit identity mismatch")
        normalization = CostNormalization.model_validate(
            self.normalization.model_dump(mode="python")
        )
        if _sha256(normalization.model_dump(mode="json")) != (
            self.normalization_sha256
        ):
            raise ValueError("normalization identity mismatch")
        for authority in self.base_memory_costs:
            BaseMemoryCostAuthority.model_validate(authority.model_dump(mode="python"))
        payload = self.model_dump(mode="json", exclude={"manifest_sha256"})
        if _sha256(payload) != self.manifest_sha256:
            raise ValueError("benchmark manifest self hash mismatch")
        return self

    @classmethod
    def create(
        cls,
        *,
        benchmark_id: str,
        benchmark_version: str,
        questions: Sequence[EvaluationQuestion],
        leakage_audit: Mapping[str, object] | str,
        base_memory_costs: Sequence[BaseMemoryCostAuthority],
        normalization: CostNormalization,
        gpu_assignment: HardwareAssignment,
    ) -> "BenchmarkManifest":
        values = tuple(question.validate_authority() for question in questions)
        if not values:
            raise ValueError("benchmark requires questions")
        split = values[0].split
        if any(question.split != split for question in values):
            raise ValueError("benchmark questions must share one formal split")
        refs = tuple(
            BenchmarkQuestionRef(
                question_id=question.question_id,
                video_group_id=question.video_group_id,
                gold_answer_sha256=_sha256(question.gold_answer),
                record_sha256=question.record_sha256,
            )
            for question in values
        )
        authorities = tuple(
            BaseMemoryCostAuthority.model_validate(item.model_dump(mode="python"))
            for item in base_memory_costs
        )
        groups = {question.video_group_id for question in values}
        if {item.video_group_id for item in authorities} != groups:
            raise ValueError(
                "base memory authorities must exactly cover benchmark groups"
            )
        leakage_json = _canonical_source(leakage_audit)
        normalized = CostNormalization.model_validate(
            normalization.model_dump(mode="python")
        )
        payload: dict[str, object] = {
            "schema_version": 2,
            "benchmark_id": benchmark_id,
            "benchmark_version": benchmark_version,
            "split": split,
            "provenance_sha256": _sha256(
                tuple(
                    (
                        question.question_id,
                        question.video_group_id,
                        question.record_sha256,
                    )
                    for question in values
                )
            ),
            "source_manifest_sha256": _sha256(
                tuple(
                    sorted(
                        question.source_manifest_canonical_json for question in values
                    )
                )
            ),
            "group_assignment_sha256": _sha256(
                tuple(
                    sorted(
                        (question.question_id, question.video_group_id)
                        for question in values
                    )
                )
            ),
            "leakage_audit_canonical_json": leakage_json,
            "leakage_audit_sha256": _sha256(json.loads(leakage_json)),
            "questions": refs,
            "base_memory_costs": authorities,
            "normalization": normalized,
            "normalization_sha256": _sha256(normalized.model_dump(mode="json")),
            "gpu_assignment": HardwareAssignment.model_validate(
                gpu_assignment.model_dump(mode="python")
            ),
        }
        payload["manifest_sha256"] = _sha256(payload)
        return cls.model_validate(payload)


class SharedEvaluationIdentity(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        strict=True,
        revalidate_instances="always",
    )

    environment_set_sha256: str = Field(pattern=_SHA256_PATTERN)
    answerer_identity_sha256: str = Field(pattern=_SHA256_PATTERN)
    cache_graph_sha256: str = Field(pattern=_SHA256_PATTERN)
    cost_table_set_sha256: str = Field(pattern=_SHA256_PATTERN)
    budgets: EvaluationBudgets
    identity_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def self_hash_matches(self) -> "SharedEvaluationIdentity":
        payload = self.model_dump(mode="json", exclude={"identity_sha256"})
        if _sha256(payload) != self.identity_sha256:
            raise ValueError("shared evaluation identity self hash mismatch")
        return self


def build_shared_identity(
    *,
    questions: Sequence[EvaluationQuestion],
    answerer: FrozenAnswerer,
    budgets: EvaluationBudgets,
) -> SharedEvaluationIdentity:
    values = tuple(question.validate_authority() for question in questions)
    if not values:
        raise ValueError("shared identity requires questions")
    validated_budgets = EvaluationBudgets.model_validate(
        budgets.model_dump(mode="python")
    )
    if any(question.budgets != validated_budgets for question in values):
        raise EvaluationIntegrityError(
            "question budgets differ from shared evaluation budgets"
        )
    cache_ids = {question.cache.identity_sha256 for question in values}
    if len(cache_ids) != 1:
        raise EvaluationIntegrityError("questions do not share one actual cache graph")
    try:
        answerer_identity = answerer.identity
    except ValueError as error:
        raise EvaluationIntegrityError(
            "formal evaluation requires an identity-bearing FrozenAnswerer"
        ) from error
    payload = {
        "environment_set_sha256": _sha256(
            tuple(
                sorted(
                    environment_sha256(question.environment, question.cache)
                    for question in values
                )
            )
        ),
        "answerer_identity_sha256": answerer_identity.identity_sha256,
        "cache_graph_sha256": next(iter(cache_ids)),
        "cost_table_set_sha256": _sha256(
            tuple(
                sorted(cost_table_sha256(question.environment) for question in values)
            )
        ),
        "budgets": validated_budgets,
    }
    return SharedEvaluationIdentity(
        **payload,
        identity_sha256=_sha256(payload),
    )


class RawQuestionResult(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        strict=True,
        revalidate_instances="always",
    )

    schema_version: Literal[2] = 2
    run_id: str = Field(min_length=1)
    policy_name: str = Field(min_length=1)
    policy_family: Literal["fixed", "adaptive", "learned"]
    policy_identity_sha256: str = Field(pattern=_SHA256_PATTERN)
    policy_horizon: int = Field(ge=2)
    seed: int
    question_id: str = Field(min_length=1)
    video_group_id: str = Field(min_length=1)
    record_sha256: str = Field(pattern=_SHA256_PATTERN)
    benchmark_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    split: Literal["dev", "validation", "test"]
    shared: SharedEvaluationIdentity
    cost_preference: float
    cost_normalization: CostNormalization
    predicted_answer: str = Field(min_length=1)
    gold_answer: str = Field(min_length=1)
    is_correct: bool
    invalid_reason: str | None = None
    cost_breakdown: CostBreakdown
    cost_breakdown_sha256: str = Field(pattern=_SHA256_PATTERN)
    actions: tuple[ActionInstance, ...] = Field(min_length=1)
    forced_stop: bool = False
    answer_score: float = Field(ge=0, le=1)
    oracle_answer_score: float = Field(ge=0, le=1)
    oracle_total_cost: float = Field(ge=0)
    oracle_utility: float
    realized_utility: float
    oracle_utility_regret: float = Field(ge=0)
    signals: ErrorSignals
    error: ErrorClassification
    result_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def all_raw_relations_match(self) -> "RawQuestionResult":
        if self.cost_preference not in COST_PREFERENCES:
            raise ValueError("cost preference must be a frozen Task 9 value")
        if self.is_correct != (self.predicted_answer == self.gold_answer):
            raise ValueError("is_correct must match predicted and gold answers")
        if self.answer_score != float(self.is_correct):
            raise ValueError("answer score must equal multiple-choice correctness")
        if self.invalid_reason is not None and not self.invalid_reason.strip():
            raise ValueError("invalid reason must be non-empty when present")
        normalization = CostNormalization.model_validate(
            self.cost_normalization.model_dump(mode="python")
        )
        breakdown = CostBreakdown.model_validate(
            self.cost_breakdown.model_dump(mode="python")
        )
        if _sha256(breakdown.model_dump(mode="json")) != (self.cost_breakdown_sha256):
            raise ValueError("cost breakdown identity mismatch")
        expected_realized = self.answer_score - self.cost_preference * (
            breakdown.total.total_cost / normalization.constant
        )
        expected_oracle = self.oracle_answer_score - self.cost_preference * (
            self.oracle_total_cost / normalization.constant
        )
        if not math.isclose(
            self.realized_utility,
            expected_realized,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("realized utility does not match Task 9 formula")
        if not math.isclose(
            self.oracle_utility,
            expected_oracle,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("Oracle utility does not match Task 9 formula")
        expected_regret = expected_oracle - expected_realized
        if expected_regret < -1e-12:
            raise ValueError("realized utility exceeds Oracle authority")
        if not math.isclose(
            self.oracle_utility_regret,
            max(0.0, expected_regret),
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("Oracle utility regret mismatch")
        signals = ErrorSignals.model_validate(self.signals.model_dump(mode="python"))
        expected_error = classify_error(
            signals,
            invalid=self.invalid_reason is not None,
            correct=self.is_correct,
        )
        if self.error != expected_error:
            raise ValueError("error classification does not match raw signals")
        payload = self.model_dump(mode="json", exclude={"result_sha256"})
        if _sha256(payload) != self.result_sha256:
            raise ValueError("raw result self hash mismatch")
        return self

    @property
    def acquisition_usage(self) -> ResourceUsage:
        return self.cost_breakdown.environment

    @property
    def controller_usage(self) -> ResourceUsage:
        return self.cost_breakdown.prompt_controller


def _validated_raw(record: RawQuestionResult) -> RawQuestionResult:
    if type(record) is not RawQuestionResult:
        raise TypeError("raw record must be exact RawQuestionResult")
    return RawQuestionResult.model_validate_json(record.model_dump_json())


def raw_results_sha256(records: Sequence[RawQuestionResult]) -> str:
    return _sha256(
        tuple(_validated_raw(record).model_dump(mode="json") for record in records)
    )


class RunManifest(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        strict=True,
        revalidate_instances="always",
    )

    schema_version: Literal[2] = 2
    run_id: str = Field(min_length=1)
    policy_name: str = Field(min_length=1)
    policy_family: Literal["fixed", "adaptive", "learned"]
    policy_identity_sha256: str = Field(pattern=_SHA256_PATTERN)
    policy_horizon: int = Field(ge=2)
    horizon_category: Literal["router-5", "fixed-full-coverage"]
    seed: int
    benchmark: BenchmarkManifest
    shared: SharedEvaluationIdentity
    cost_preference: float
    gpu_assignment: HardwareAssignment
    question_keys: tuple[tuple[str, str], ...] = Field(min_length=1)
    raw_results_sha256: str = Field(pattern=_SHA256_PATTERN)
    manifest_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def keys_and_self_hash_match(self) -> "RunManifest":
        if self.cost_preference not in COST_PREFERENCES:
            raise ValueError("cost preference must be a frozen Task 9 value")
        if len(set(self.question_keys)) != len(self.question_keys):
            raise ValueError("run contains duplicate question/video keys")
        if self.gpu_assignment != self.benchmark.gpu_assignment:
            raise ValueError("run GPU assignment differs from benchmark")
        if self.horizon_category == "router-5" and self.policy_horizon != 5:
            raise ValueError("adaptive/learned Router horizon must be five")
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
        policy_horizon: int,
        horizon_category: Literal["router-5", "fixed-full-coverage"],
        seed: int,
        benchmark: BenchmarkManifest,
        shared: SharedEvaluationIdentity,
        cost_preference: float,
        records: Sequence[RawQuestionResult],
    ) -> "RunManifest":
        values = tuple(_validated_raw(record) for record in records)
        if not values:
            raise ValueError("run records must be non-empty")
        keys = tuple((item.question_id, item.video_group_id) for item in values)
        expected_keys = tuple(
            (item.question_id, item.video_group_id) for item in benchmark.questions
        )
        if keys != expected_keys:
            raise ValueError(
                "raw keys must exactly equal all benchmark questions in order"
            )
        expected_identity = (
            run_id,
            policy_name,
            policy_family,
            policy_identity_sha256,
            policy_horizon,
            seed,
            benchmark.manifest_sha256,
            shared,
            cost_preference,
            benchmark.split,
        )
        benchmark_records = {
            (item.question_id, item.video_group_id): item
            for item in benchmark.questions
        }
        for item in values:
            actual_identity = (
                item.run_id,
                item.policy_name,
                item.policy_family,
                item.policy_identity_sha256,
                item.policy_horizon,
                item.seed,
                item.benchmark_manifest_sha256,
                item.shared,
                item.cost_preference,
                item.split,
            )
            if actual_identity != expected_identity:
                raise ValueError("raw result identity differs from run identity")
            key = (item.question_id, item.video_group_id)
            authority = benchmark_records.get(key)
            if authority is None or authority.record_sha256 != item.record_sha256:
                raise ValueError("raw result differs from benchmark question provenance")
        validated_benchmark = BenchmarkManifest.model_validate_json(
            benchmark.model_dump_json()
        )
        validated_shared = SharedEvaluationIdentity.model_validate(
            shared.model_dump(mode="python")
        )
        payload: dict[str, object] = {
            "schema_version": 2,
            "run_id": run_id,
            "policy_name": policy_name,
            "policy_family": policy_family,
            "policy_identity_sha256": policy_identity_sha256,
            "policy_horizon": policy_horizon,
            "horizon_category": horizon_category,
            "seed": seed,
            "benchmark": validated_benchmark,
            "shared": validated_shared,
            "cost_preference": cost_preference,
            "gpu_assignment": validated_benchmark.gpu_assignment,
            "question_keys": keys,
            "raw_results_sha256": raw_results_sha256(values),
        }
        payload["manifest_sha256"] = _sha256(payload)
        return cls.model_validate(payload)


class EvaluationRun(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        revalidate_instances="always",
    )

    manifest: RunManifest
    records: tuple[RawQuestionResult, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def records_are_bound_to_manifest(self) -> "EvaluationRun":
        manifest = RunManifest.model_validate_json(self.manifest.model_dump_json())
        records = tuple(_validated_raw(record) for record in self.records)
        if raw_results_sha256(records) != manifest.raw_results_sha256:
            raise ValueError("raw results do not match run manifest identity")
        keys = tuple((item.question_id, item.video_group_id) for item in records)
        if keys != manifest.question_keys:
            raise ValueError("raw result order/keys do not match run manifest")
        benchmark_records = {
            (item.question_id, item.video_group_id): item
            for item in manifest.benchmark.questions
        }
        for record in records:
            authority = benchmark_records[(record.question_id, record.video_group_id)]
            if _sha256(record.gold_answer) != authority.gold_answer_sha256:
                raise ValueError("raw result differs from benchmark question authority")
        return self
    @property
    def summary(self) -> RunSummary:
        from .metrics import summarize_results

        return summarize_results(self)


class _FairPolicy:
    def __init__(self, policy: object) -> None:
        self.policy = policy
        self.policy_usage = ResourceUsage()
        self.controller_usage = ResourceUsage()

    def __call__(
        self,
        state: RouterState,
        legal_actions: tuple[ActionInstance, ...],
    ) -> ActionInstance:
        private_state = RouterState.model_validate(state.model_dump(mode="python"))
        tracker = CostTracker()
        try:
            with tracker.measure(
                "evaluation_policy",
                "miss",
                0,
                0,
                cost_component="online",
            ) as measurement:
                selected = self.policy(private_state, legal_actions)  # type: ignore[operator]
        except EvaluationIntegrityError:
            raise
        except Exception as error:
            raise EvaluationIntegrityError(
                "policy failed under the fair action mask"
            ) from error
        if measurement.record is None:
            raise EvaluationIntegrityError("policy measurement is missing")
        self.policy_usage = self.policy_usage.plus(
            _resource_from_record(measurement.record)
        )
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
            validated = ControllerCost.model_validate(raw.model_dump(mode="python"))
            self.controller_usage = self.controller_usage.plus(
                ResourceUsage.model_validate(validated.model_dump(mode="python"))
            )
        return selected


def _metadata_usage(
    metadata: Sequence[OperationMetadata],
) -> ResourceUsage:
    usage = ResourceUsage()
    for item in metadata:
        if item.cost_record is None:
            component = ResourceUsage(
                input_frames=item.input_frames,
                visual_tokens=item.visual_tokens,
                text_tokens=item.text_tokens,
            )
        else:
            record = item.cost_record
            if (
                record.input_frames != item.input_frames
                or record.visual_tokens != item.visual_tokens
                or record.text_tokens != item.text_tokens
            ):
                raise EvaluationIntegrityError(
                    "operation metadata and measured CostRecord disagree"
                )
            component = _resource_from_record(record)
        usage = usage.plus(component)
    return usage


def _policy_identity_sha256(policy: object) -> str:
    if not isinstance(policy, _BUILTIN_POLICIES):
        raise TypeError(
            "formal evaluation accepts only built-in identity-bearing policies"
        )
    if isinstance(policy, (BCPolicyAdapter, DAggerPolicyAdapter)):
        return _sha256(policy.policy_identity.model_dump(mode="json"))
    return callable_identity_sha256(policy)  # type: ignore[arg-type]


def _policy_family(
    policy: object,
) -> Literal["fixed", "adaptive", "learned"]:
    if isinstance(
        policy,
        (BCPolicyAdapter, DAggerPolicyAdapter, QuestionOnlyPolicy),
    ):
        return "learned"
    if isinstance(
        policy,
        (
            GistOnlyPolicy,
            GistResidualPolicy,
            GistVisualPolicy,
            UniformFramesPolicy,
            FullResidualPolicy,
        ),
    ):
        return "fixed"
    if isinstance(policy, (RulePolicy, PromptControllerPolicy, TextAdaptivePolicy)):
        return "adaptive"
    raise TypeError("formal evaluation requires a built-in policy family")


def _policy_horizon(
    policy: object,
    questions: Sequence[EvaluationQuestion],
) -> tuple[int, Literal["router-5", "fixed-full-coverage"]]:
    if isinstance(policy, _FIXED_LONG_POLICIES):
        event_count = max(len(question.uniform_event_order) for question in questions)
        return (2 * event_count + 2, "fixed-full-coverage")
    return (5, "router-5")


def _bind_fixed_policy(
    policy: object,
    question: EvaluationQuestion,
) -> object:
    if isinstance(policy, UniformFramesPolicy):
        return policy.for_event_order(question.uniform_event_order)
    if isinstance(policy, FullResidualPolicy):
        return policy.for_event_order(question.uniform_event_order)
    return policy


_FIDELITY_RANK = {
    FidelityLevel.GIST: 0,
    FidelityLevel.RESIDUAL: 1,
    FidelityLevel.VISUAL: 2,
}


def _supports_sufficient(
    state: RouterState,
    oracle: OracleEvaluationAuthority,
) -> bool:
    required = _FIDELITY_RANK[oracle.required_fidelity]
    return all(
        event_id in state.candidate_fidelity_levels
        and _FIDELITY_RANK[state.candidate_fidelity_levels[event_id]] >= required
        for event_id in oracle.gold_support_event_ids
    )


def _derive_signals(
    *,
    question: EvaluationQuestion,
    transitions: Sequence[object],
    final_state: RouterState,
) -> ErrorSignals:
    support = set(question.oracle.gold_support_event_ids)
    search_candidates: set[str] = set()
    sufficient_step: int | None = None
    for index, transition in enumerate(transitions):
        if transition.action.action_type is ActionType.SEARCH_GIST:
            search_candidates.update(transition.observation.candidate_event_ids)
        if sufficient_step is None and _supports_sufficient(
            transition.next_state, question.oracle
        ):
            sufficient_step = index
    sufficient = _supports_sufficient(final_state, question.oracle)
    later_expansion = sufficient_step is not None and any(
        transition.action.action_type not in {ActionType.STOP, ActionType.SEARCH_GIST}
        for transition in transitions[sufficient_step + 1 :]
    )
    useful_upgrade = any(
        event_id in final_state.candidate_fidelity_levels
        and _FIDELITY_RANK[final_state.candidate_fidelity_levels[event_id]]
        < _FIDELITY_RANK[question.oracle.required_fidelity]
        for event_id in support
    )
    stopped_insufficient = bool(
        transitions
        and transitions[-1].action.action_type is ActionType.STOP
        and not sufficient
    )
    return ErrorSignals(
        gist_top_k_contains_answer=support.issubset(search_candidates),
        oracle_evidence_sufficient=sufficient,
        answerer_correct_with_oracle_evidence=(
            question.oracle.answerer_correct_with_oracle_evidence
            if sufficient
            else None
        ),
        stopped_with_insufficient_evidence=stopped_insufficient,
        useful_fidelity_upgrade_available=useful_upgrade,
        unnecessary_expansion=later_expansion,
    )


def _base_usage(
    benchmark: BenchmarkManifest,
    question: EvaluationQuestion,
) -> ResourceUsage:
    authority = next(
        (
            item
            for item in benchmark.base_memory_costs
            if item.video_group_id == question.video_group_id
        ),
        None,
    )
    if authority is None:
        raise EvaluationIntegrityError(
            "benchmark lacks base cost authority for video group"
        )
    query_count = sum(
        item.video_group_id == question.video_group_id for item in benchmark.questions
    )
    return authority.usage.divided_by(query_count)


def _preflight(
    *,
    questions: tuple[EvaluationQuestion, ...],
    benchmark: BenchmarkManifest,
    answerer: FrozenAnswerer,
    budgets: EvaluationBudgets,
    cost_preference: float,
) -> SharedEvaluationIdentity:
    if not questions:
        raise EvaluationIntegrityError("evaluation questions must be non-empty")
    if cost_preference not in COST_PREFERENCES:
        raise EvaluationIntegrityError(
            "cost preference must be one of Task 9 frozen values"
        )
    validated_benchmark = BenchmarkManifest.model_validate_json(
        benchmark.model_dump_json()
    )
    values = tuple(question.validate_authority() for question in questions)
    keys = tuple((question.question_id, question.video_group_id) for question in values)
    expected_keys = tuple(
        (question.question_id, question.video_group_id)
        for question in validated_benchmark.questions
    )
    if keys != expected_keys:
        raise EvaluationIntegrityError(
            "questions must exactly match all benchmark questions in order"
        )
    refs = {
        (item.question_id, item.video_group_id): item.record_sha256
        for item in validated_benchmark.questions
    }
    for question in values:
        if question.split != validated_benchmark.split:
            raise EvaluationIntegrityError(
                "question split differs from benchmark split"
            )
        if refs[(question.question_id, question.video_group_id)] != (
            question.record_sha256
        ):
            raise EvaluationIntegrityError(
                "benchmark record provenance identity mismatch"
            )
        if question.initial_state.cost_preference != cost_preference:
            raise EvaluationIntegrityError("question cost preference differs from run")
        if question.budgets != budgets:
            raise EvaluationIntegrityError("question budgets differ from run budgets")
    return build_shared_identity(
        questions=values,
        answerer=answerer,
        budgets=budgets,
    )


def evaluate_run(
    *,
    run_id: str,
    policy_name: str,
    policy_family: Literal["fixed", "adaptive", "learned"],
    policy: object,
    questions: Sequence[EvaluationQuestion],
    benchmark: BenchmarkManifest,
    answerer: FrozenAnswerer,
    budgets: EvaluationBudgets,
    seed: int,
    cost_preference: float,
) -> EvaluationRun:
    apply_evaluation_seed(seed)
    values = tuple(questions)
    shared = _preflight(
        questions=values,
        benchmark=benchmark,
        answerer=answerer,
        budgets=budgets,
        cost_preference=cost_preference,
    )
    policy_identity = _policy_identity_sha256(policy)
    actual_family = _policy_family(policy)
    if policy_family != actual_family:
        raise EvaluationIntegrityError(
            "caller policy family differs from actual policy object"
        )
    horizon, horizon_category = _policy_horizon(policy, values)
    records: list[RawQuestionResult] = []
    for question in values:
        guarded = _FairPolicy(_bind_fixed_policy(policy, question))
        result = AgentRunner(
            question.environment,
            guarded,
            answerer,
            max_transitions=horizon,
        ).run(
            question.initial_state,
            run_id=(f"{run_id}:{question.question_id}:" f"{question.video_group_id}"),
        )
        environment_usage = ResourceUsage(
            total_cost=math.fsum(
                transition.step_cost for transition in result.transitions
            )
        )
        for transition in result.transitions:
            environment_usage = environment_usage.plus(
                _metadata_usage(transition.operation_metadata)
            )
        if not result.answer.usage_authoritative:
            raise EvaluationIntegrityError(
                "formal evaluation requires authoritative Answerer usage"
            )
        answerer_usage = _resource_from_record(
            result.answer.cost_record,
            total_cost=result.answer.total_cost,
        )
        breakdown = CostBreakdown(
            base_memory=_base_usage(benchmark, question),
            environment=environment_usage,
            policy_router=guarded.policy_usage,
            prompt_controller=guarded.controller_usage,
            answerer=answerer_usage,
        )
        online = (
            breakdown.environment.plus(breakdown.policy_router)
            .plus(breakdown.prompt_controller)
            .plus(breakdown.answerer)
        )
        evidence_tokens = (
            breakdown.environment.visual_tokens + breakdown.environment.text_tokens
        )
        if online.input_frames > budgets.max_visual_frames:
            invalid_reason = "visual_frame_budget_exceeded"
        elif evidence_tokens > budgets.max_evidence_tokens:
            invalid_reason = "evidence_token_budget_exceeded"
        elif breakdown.total.total_cost > budgets.max_total_cost:
            invalid_reason = "total_cost_budget_exceeded"
        else:
            invalid_reason = None
        correct = result.answer.answer == question.gold_answer
        answer_score = float(correct)
        normalization = CostNormalization.model_validate(
            benchmark.normalization.model_dump(mode="python")
        )
        realized_utility = answer_score - cost_preference * (
            breakdown.total.total_cost / normalization.constant
        )
        oracle_utility = question.oracle.utility(
            cost_preference=cost_preference,
            normalization=normalization,
        )
        regret = oracle_utility - realized_utility
        if regret < -1e-12:
            raise EvaluationIntegrityError(
                "realized trajectory exceeds frozen Oracle authority"
            )
        signals = _derive_signals(
            question=question,
            transitions=result.transitions,
            final_state=result.final_state,
        )
        error = classify_error(
            signals,
            invalid=invalid_reason is not None,
            correct=correct,
        )
        cost_hash = _sha256(breakdown.model_dump(mode="json"))
        payload: dict[str, object] = {
            "schema_version": 2,
            "run_id": run_id,
            "policy_name": policy_name,
            "policy_family": policy_family,
            "policy_identity_sha256": policy_identity,
            "policy_horizon": horizon,
            "seed": seed,
            "question_id": question.question_id,
            "video_group_id": question.video_group_id,
            "record_sha256": question.record_sha256,
            "benchmark_manifest_sha256": benchmark.manifest_sha256,
            "split": benchmark.split,
            "shared": shared,
            "cost_preference": cost_preference,
            "cost_normalization": normalization,
            "predicted_answer": result.answer.answer,
            "gold_answer": question.gold_answer,
            "is_correct": correct,
            "invalid_reason": invalid_reason,
            "cost_breakdown": breakdown,
            "cost_breakdown_sha256": cost_hash,
            "actions": tuple(transition.action for transition in result.transitions),
            "forced_stop": result.forced_stop,
            "answer_score": answer_score,
            "oracle_answer_score": question.oracle.answer_score,
            "oracle_total_cost": question.oracle.total_cost,
            "oracle_utility": oracle_utility,
            "realized_utility": realized_utility,
            "oracle_utility_regret": max(0.0, regret),
            "signals": signals,
            "error": error,
        }
        payload["result_sha256"] = _sha256(payload)
        records.append(RawQuestionResult.model_validate(payload))
    manifest = RunManifest.create(
        run_id=run_id,
        policy_name=policy_name,
        policy_family=policy_family,
        policy_identity_sha256=policy_identity,
        policy_horizon=horizon,
        horizon_category=horizon_category,
        seed=seed,
        benchmark=benchmark,
        shared=shared,
        cost_preference=cost_preference,
        records=records,
    )
    return EvaluationRun(manifest=manifest, records=tuple(records))


class PolicyConfig(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        revalidate_instances="always",
    )

    family: Literal["fixed", "adaptive", "learned"]


class BudgetSweep(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        revalidate_instances="always",
    )

    total_cost: tuple[float, ...] = Field(min_length=1)
    max_visual_frames: tuple[int, ...] = Field(min_length=1)
    max_evidence_tokens: tuple[int, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def all_budgets_are_nonnegative(self) -> "BudgetSweep":
        if any(value < 0 for value in self.total_cost):
            raise ValueError("total cost budgets must be non-negative")
        if any(value < 0 for value in self.max_visual_frames):
            raise ValueError("visual frame budgets must be non-negative")
        if any(value < 0 for value in self.max_evidence_tokens):
            raise ValueError("evidence token budgets must be non-negative")
        return self


class ReportingConfig(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        revalidate_instances="always",
    )

    preserve_per_seed_raw: bool
    fabricate_confidence_intervals: Literal[False]
    primary_metrics: tuple[str, ...]
    secondary_metrics: tuple[str, ...]


class HardwareConfig(HardwareAssignment):
    a800_gpu_hour_cap: int = Field(ge=0)
    v100_gpu_hour_cap: int = Field(ge=0)


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        revalidate_instances="always",
    )

    policies: dict[str, PolicyConfig]
    budget_sweep: BudgetSweep
    seeds: tuple[int, int, int]
    cost_preferences: tuple[float, float, float, float]
    shared_identities: dict[str, str]
    hardware: HardwareConfig
    reporting: ReportingConfig

    @model_validator(mode="after")
    def frozen_experiment_matrix(self) -> "EvaluationConfig":
        if self.cost_preferences != COST_PREFERENCES:
            raise ValueError("cost_preferences must exactly match Task 9 frozen values")
        if len(set(self.seeds)) != 3:
            raise ValueError("evaluation requires three distinct seeds")
        actual_families = {
            name: policy.family for name, policy in self.policies.items()
        }
        if actual_families != _EXPECTED_POLICY_FAMILIES:
            raise ValueError(
                "evaluation policy matrix must exactly match all frozen baselines"
            )
        if any(not value.strip() for value in self.shared_identities.values()):
            raise ValueError("shared identity references must not be blank")
        return self


class EvaluationCell(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        revalidate_instances="always",
    )

    policy_name: str
    policy_family: Literal["fixed", "adaptive", "learned"]
    seed: int
    budgets: EvaluationBudgets
    cost_preference: float
    shared_identities: dict[str, str]
    hardware: HardwareAssignment


def load_evaluation_config(path: Path | str) -> EvaluationConfig:
    raw = OmegaConf.to_container(OmegaConf.load(Path(path)), resolve=False)
    if not isinstance(raw, dict):
        raise ValueError("evaluation config must contain a mapping")
    payload = dict(raw)
    payload["seeds"] = tuple(payload["seeds"])
    payload["cost_preferences"] = tuple(payload["cost_preferences"])
    sweep = dict(payload["budget_sweep"])
    for key in ("total_cost", "max_visual_frames", "max_evidence_tokens"):
        sweep[key] = tuple(sweep[key])
    payload["budget_sweep"] = sweep
    reporting = dict(payload["reporting"])
    reporting["primary_metrics"] = tuple(reporting["primary_metrics"])
    reporting["secondary_metrics"] = tuple(reporting["secondary_metrics"])
    payload["reporting"] = reporting
    return EvaluationConfig.model_validate(payload)


def evaluation_matrix(config: EvaluationConfig) -> tuple[EvaluationCell, ...]:
    validated = EvaluationConfig.model_validate(config.model_dump(mode="python"))
    cells = []
    for (
        policy_name,
        seed,
        total_cost,
        frames,
        tokens,
        preference,
    ) in itertools.product(
        sorted(validated.policies),
        validated.seeds,
        validated.budget_sweep.total_cost,
        validated.budget_sweep.max_visual_frames,
        validated.budget_sweep.max_evidence_tokens,
        validated.cost_preferences,
    ):
        policy = validated.policies[policy_name]
        cells.append(
            EvaluationCell(
                policy_name=policy_name,
                policy_family=policy.family,
                seed=seed,
                budgets=EvaluationBudgets(
                    max_visual_frames=frames,
                    max_evidence_tokens=tokens,
                    max_total_cost=total_cost,
                ),
                cost_preference=preference,
                shared_identities=dict(validated.shared_identities),
                hardware=HardwareAssignment(
                    training=validated.hardware.training,
                    evaluation=validated.hardware.evaluation,
                ),
            )
        )
    return tuple(cells)


def apply_evaluation_seed(seed: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("evaluation seed must be an integer")
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        np.random.seed(seed)
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


__all__ = [
    "BaseMemoryCostAuthority",
    "BenchmarkManifest",
    "BenchmarkQuestionRef",
    "BudgetSweep",
    "CacheBinding",
    "EvaluationBudgets",
    "EvaluationCell",
    "EvaluationConfig",
    "EvaluationIntegrityError",
    "EvaluationQuestion",
    "EvaluationRun",
    "HardwareAssignment",
    "OracleEvaluationAuthority",
    "RawQuestionResult",
    "RunManifest",
    "SharedEvaluationIdentity",
    "apply_evaluation_seed",
    "build_shared_identity",
    "cost_table_sha256",
    "environment_sha256",
    "evaluate_run",
    "evaluation_matrix",
    "load_evaluation_config",
    "raw_results_sha256",
]
