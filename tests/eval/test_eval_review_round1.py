from __future__ import annotations

import math
from pathlib import Path

import pytest
from pydantic import ValidationError

import fidmem.eval.baselines as baselines
import fidmem.eval.metrics as metrics
import fidmem.eval.runner as evaluation
from fidmem.actions.environment import (
    ActionCostTable,
    ActionObservation,
    MemoryEnvironment,
    OperationMetadata,
)
from fidmem.agent.answerer import AnswererAdapterResult, FrozenAnswerer
from fidmem.costs.tracker import CostRecord
from fidmem.oracle.labels import COST_PREFERENCES, CostNormalization
from fidmem.storage.cache import ContentAddressedCache
from fidmem.types import (
    ActionInstance,
    ActionType,
    EventRecord,
    EvidenceItem,
    FidelityLevel,
    RouterState,
)


def _record(
    operation: str,
    *,
    gpu: float = 0.0,
    wall: float = 0.0,
    frames: int = 0,
    visual_tokens: int = 0,
    text_tokens: int = 0,
    peak: int = 0,
) -> CostRecord:
    return CostRecord(
        operation=operation,
        gpu_seconds=gpu,
        wall_seconds=wall,
        input_frames=frames,
        visual_tokens=visual_tokens,
        text_tokens=text_tokens,
        peak_memory_bytes=peak,
        cache_status="miss",
        device_name="synthetic",
    )


def _events(count: int = 6) -> tuple[EventRecord, ...]:
    return tuple(
        EventRecord(
            video_id="video",
            event_id=f"e{index}",
            start_sec=float(index * 2),
            end_sec=float(index * 2 + 1),
            asr_text=f"asr {index}",
            gist_text=f"gist {index}",
            memory_version="v1",
        )
        for index in range(count)
    )


def _environment(
    *,
    search_ids: tuple[str, ...] = ("e3",),
    search_text: str = "support",
    events: tuple[EventRecord, ...] | None = None,
) -> MemoryEnvironment:
    canonical = events or _events()

    def execute(action: ActionInstance, state: RouterState) -> ActionObservation:
        del state
        if action.action_type is ActionType.SEARCH_GIST:
            evidence = tuple(
                EvidenceItem(
                    event_id=event_id,
                    fidelity_level=FidelityLevel.GIST,
                    content=search_text if event_id == "e3" else f"gist {event_id}",
                    score=1.0 if event_id == "e3" else 0.1,
                )
                for event_id in search_ids
            )
            return ActionObservation(
                action_type=ActionType.SEARCH_GIST,
                target_event_id=None,
                candidate_event_ids=search_ids,
                evidence=evidence,
                operation_metadata=(
                    OperationMetadata(
                        scope="search_gist",
                        cache_status="miss",
                        amortizable=True,
                        text_tokens=2,
                        cost_record=_record("search", wall=0.2, text_tokens=2),
                    ),
                ),
            )
        if action.action_type is ActionType.EXPAND_CONTEXT:
            return ActionObservation()
        if action.action_type is ActionType.EXPAND_RESIDUAL:
            return ActionObservation(
                action_type=ActionType.EXPAND_RESIDUAL,
                target_event_id=action.event_id,
                evidence=(
                    EvidenceItem(
                        event_id=action.event_id or "",
                        fidelity_level=FidelityLevel.RESIDUAL,
                        content=f"residual {action.event_id}",
                        score=0.8,
                    ),
                ),
                operation_metadata=(
                    OperationMetadata(
                        scope="residual",
                        cache_status="miss",
                        amortizable=True,
                        text_tokens=3,
                        cost_record=_record("residual", wall=0.3, text_tokens=3),
                    ),
                ),
            )
        if action.action_type is ActionType.VERIFY_VISUAL:
            frames = 12 if action.visual_budget == "low" else 32
            return ActionObservation(
                action_type=ActionType.VERIFY_VISUAL,
                target_event_id=action.event_id,
                evidence=(
                    EvidenceItem(
                        event_id=action.event_id or "",
                        fidelity_level=FidelityLevel.VISUAL,
                        content=f"visual {action.event_id}",
                        score=1.0,
                        attachments=(f"{action.event_id}.png",),
                    ),
                ),
                operation_metadata=(
                    OperationMetadata(
                        scope="event_observation",
                        cache_status="miss",
                        amortizable=True,
                        input_frames=frames,
                        visual_tokens=4,
                        cost_record=_record(
                            "visual",
                            wall=0.4,
                            frames=frames,
                            visual_tokens=4,
                        ),
                    ),
                    OperationMetadata(
                        scope="question_verification",
                        cache_status="miss",
                        amortizable=False,
                        text_tokens=1,
                        cost_record=_record("verification", wall=0.1, text_tokens=1),
                    ),
                ),
            )
        raise AssertionError(f"unexpected executor action {action.action_type}")

    return MemoryEnvironment(
        events=canonical,
        executor=execute,
        costs=ActionCostTable(
            search_gist=1.0,
            residual=2.0,
            context=0.5,
            visual_low=4.0,
            visual_high=8.0,
            visual_low_question=1.0,
            visual_high_question=2.0,
        ),
    )


def _state(
    *,
    budget: float = 100.0,
    preference: float = 0.1,
) -> RouterState:
    return RouterState(
        question="Which option is supported?",
        options=("A", "B"),
        evidence=(),
        action_history=(),
        remaining_budget=budget,
        candidate_event_ids=(),
        candidate_fidelity_levels={},
        context_frontiers={},
        cost_preference=preference,
    )


def _answerer(answer: str = "A", *, answer_cost: float = 0.5) -> FrozenAnswerer:
    usage = _record(
        "answerer",
        gpu=0.25,
        wall=0.5,
        text_tokens=7,
        peak=1024,
    )

    def adapter(_prompt: str) -> AnswererAdapterResult:
        return AnswererAdapterResult(
            response=answer,
            cost_record=usage,
            total_cost=answer_cost,
        )

    return FrozenAnswerer(
        adapter,
        model_artifact_sha256="a" * 64,
        model_revision="answerer-r1",
        decode_config={"temperature": 0.0, "max_tokens": 8},
    )


def _cache(tmp_path: Path, *, namespace: str = "shared"):
    cache = ContentAddressedCache(tmp_path / namespace)
    key = cache.key("video", (0.0, 1.0), "model", "prompt", {"v": 1})
    cache.put(key, {"value": "cached"})
    return evaluation.CacheBinding(cache=cache, namespace=namespace)


def _question(
    tmp_path: Path,
    *,
    environment: MemoryEnvironment | None = None,
    cache_binding=None,
    budget: float = 100.0,
    preference: float = 0.1,
    question_id: str = "q1",
    video_group_id: str = "group",
    support: tuple[str, ...] = ("e3",),
    required_fidelity: FidelityLevel = FidelityLevel.GIST,
    oracle_actions: tuple[ActionType, ...] = (
        ActionType.SEARCH_GIST,
        ActionType.STOP,
    ),
    oracle_cost: float = 2.0,
    oracle_score: float = 1.0,
):
    env = environment or _environment()
    binding = cache_binding or _cache(tmp_path)
    budgets = evaluation.EvaluationBudgets(
        max_visual_frames=256,
        max_evidence_tokens=256,
        max_total_cost=budget,
    )
    oracle = evaluation.OracleEvaluationAuthority.create(
        answer_score=oracle_score,
        total_cost=oracle_cost,
        action_types=oracle_actions,
        gold_support_event_ids=support,
        required_fidelity=required_fidelity,
        answerer_correct_with_oracle_evidence=oracle_score == 1.0,
    )
    question = evaluation.EvaluationQuestion.create(
        question_id=question_id,
        video_group_id=video_group_id,
        video_id="video",
        split="test",
        source_manifest={
            "dataset": "synthetic",
            "version": "1",
            "video_id": "video",
            "question_id": question_id,
        },
        initial_state=_state(budget=budget, preference=preference),
        gold_answer="A",
        environment=env,
        cache=binding,
        budgets=budgets,
        oracle=oracle,
    )
    return question, budgets


def _benchmark(
    questions,
    *,
    base_cost: float = 4.0,
    normalization: float = 10.0,
):
    groups = {question.video_group_id for question in questions}
    base = tuple(
        evaluation.BaseMemoryCostAuthority.create(
            video_group_id=group,
            usage=metrics.ResourceUsage(
                total_cost=base_cost,
                gpu_seconds=2.0,
                wall_seconds=4.0,
                text_tokens=20,
            ),
            artifact_name="synthetic-base-v1",
        )
        for group in sorted(groups)
    )
    return evaluation.BenchmarkManifest.create(
        benchmark_id="synthetic",
        benchmark_version="1",
        questions=tuple(questions),
        leakage_audit={"status": "passed", "auditor": "synthetic-v1"},
        base_memory_costs=base,
        normalization=CostNormalization(
            constant=normalization,
            sample_count=3,
            source_split="train",
        ),
        gpu_assignment=evaluation.HardwareAssignment(
            training="A800", evaluation="V100"
        ),
    )


def test_runtime_identity_recomputes_closure_cache_budget_and_question_content(
    tmp_path: Path,
) -> None:
    first_cache = _cache(tmp_path, namespace="one")
    second_cache = _cache(tmp_path, namespace="two")
    first = _environment(search_text="first")
    second = _environment(search_text="second")

    assert evaluation.environment_sha256(
        first, first_cache
    ) != evaluation.environment_sha256(second, first_cache)
    assert first_cache.identity_sha256 != second_cache.identity_sha256

    cache = ContentAddressedCache(tmp_path / "mutable")
    binding = evaluation.CacheBinding(cache=cache, namespace="mutable")
    before = binding.identity_sha256
    cache.put("b" * 64, {"changed": True})
    assert binding.identity_sha256 != before

    question, budgets = _question(tmp_path, environment=first)
    changed = question.model_copy(update={"gold_answer": "B"})
    assert question.record_sha256 != changed.record_sha256
    with pytest.raises(ValidationError):
        evaluation.EvaluationQuestion(
            **question.model_dump(mode="python"),
            record_sha256="f" * 64,
        )
    with pytest.raises(ValueError, match="remaining budget"):
        _question(tmp_path, environment=first, budget=99.0)[0].model_copy(
            update={"initial_state": _state(budget=98.0)}
        ).validate_authority()
    assert budgets.max_total_cost == question.initial_state.remaining_budget


def test_restricted_controllers_never_receive_action_or_candidate_existence() -> None:
    state = RouterState(
        question="Q",
        options=("A", "B"),
        evidence=(
            EvidenceItem(
                event_id="secret",
                fidelity_level=FidelityLevel.VISUAL,
                content="SECRET MULTIMODAL",
                score=1.0,
                attachments=("secret.png",),
            ),
            EvidenceItem(
                event_id="public",
                fidelity_level=FidelityLevel.GIST,
                content="public gist",
                score=0.2,
            ),
        ),
        action_history=(ActionInstance(ActionType.SEARCH_GIST, None, None),),
        remaining_budget=10.0,
        candidate_event_ids=("secret", "public"),
        candidate_fidelity_levels={
            "secret": FidelityLevel.VISUAL,
            "public": FidelityLevel.GIST,
        },
        context_frontiers={"secret": (1, 1), "public": (0, 0)},
        cost_preference=0.1,
    )
    stop = ActionInstance(ActionType.STOP, None, None)

    def question_controller(view):
        assert vars(view) == {
            "question": "Q",
            "options": ("A", "B"),
            "remaining_budget": 10.0,
            "cost_preference": 0.1,
        }
        return ActionType.STOP

    def text_controller(view):
        representation = repr(view)
        assert "public gist" in representation
        assert "SECRET MULTIMODAL" not in representation
        assert "secret.png" not in representation
        assert "secret" not in representation
        assert view.action_history == (ActionType.SEARCH_GIST,)
        return ActionType.STOP

    assert baselines.QuestionOnlyPolicy(question_controller)(state, (stop,)) is stop
    assert baselines.TextAdaptivePolicy(text_controller)(state, (stop,)) is stop

    unavailable = baselines.QuestionOnlyPolicy(lambda _view: ActionType.VERIFY_VISUAL)
    assert unavailable(state, (stop,)) is stop


def test_six_event_uniform_and_full_residual_use_only_legal_long_fixed_traces(
    tmp_path: Path,
) -> None:
    environment = _environment()
    question, budgets = _question(
        tmp_path,
        environment=environment,
        oracle_actions=(ActionType.SEARCH_GIST, ActionType.STOP),
        oracle_cost=0.0,
    )
    benchmark = _benchmark((question,))
    answerer = _answerer()

    uniform = evaluation.evaluate_run(
        run_id="uniform",
        policy_name="uniform",
        policy_family="fixed",
        policy=baselines.UniformFramesPolicy(),
        questions=(question,),
        benchmark=benchmark,
        answerer=answerer,
        budgets=budgets,
        seed=1,
        cost_preference=0.1,
    )
    full = evaluation.evaluate_run(
        run_id="full",
        policy_name="full_residual",
        policy_family="fixed",
        policy=baselines.FullResidualPolicy(),
        questions=(question,),
        benchmark=benchmark,
        answerer=answerer,
        budgets=budgets,
        seed=1,
        cost_preference=0.1,
    )

    uniform_actions = uniform.records[0].actions
    verified = tuple(
        action.event_id
        for action in uniform_actions
        if action.action_type is ActionType.VERIFY_VISUAL
    )
    assert verified == tuple(event.event_id for event in environment.canonical_events)
    assert uniform.manifest.policy_horizon > 5
    full_actions = full.records[0].actions
    first_residual = next(
        index
        for index, action in enumerate(full_actions)
        if action.action_type is ActionType.EXPAND_RESIDUAL
    )
    assert all(
        action.action_type is not ActionType.EXPAND_CONTEXT
        for action in full_actions[first_residual:]
    )
    residual_ids = tuple(
        action.event_id
        for action in full_actions
        if action.action_type is ActionType.EXPAND_RESIDUAL
    )
    assert residual_ids == tuple(
        event.event_id for event in environment.canonical_events
    )
    assert full.manifest.policy_horizon > 5


def test_complete_cost_normalized_utility_and_invalid_denominators(
    tmp_path: Path,
) -> None:
    question, budgets = _question(
        tmp_path,
        oracle_cost=5.7,
        oracle_actions=(ActionType.SEARCH_GIST, ActionType.STOP),
    )
    benchmark = _benchmark((question,), base_cost=4.0, normalization=10.0)

    def controller(state, legal):
        selected = next(
            action
            for action in legal
            if action.action_type
            is (
                ActionType.SEARCH_GIST
                if not state.candidate_event_ids
                else ActionType.STOP
            )
        )
        return baselines.PromptControllerDecision(
            action=selected,
            rationale="PRIVATE",
            cost=baselines.ControllerCost(total_cost=0.1),
        )

    run = evaluation.evaluate_run(
        run_id="cost",
        policy_name="prompt_vlm",
        policy_family="adaptive",
        policy=baselines.PromptControllerPolicy(controller),
        questions=(question,),
        benchmark=benchmark,
        answerer=_answerer(answer_cost=0.5),
        budgets=budgets,
        seed=7,
        cost_preference=0.1,
    )
    record = run.records[0]
    assert record.cost_breakdown.base_memory.total_cost == 4.0
    assert record.cost_breakdown.environment.total_cost == 1.0
    assert record.cost_breakdown.prompt_controller.total_cost == 0.2
    assert record.cost_breakdown.answerer.total_cost == 0.5
    assert record.cost_breakdown.total.total_cost == pytest.approx(5.7)
    assert record.oracle_utility_regret == pytest.approx(0.0)

    tiny = evaluation.EvaluationBudgets(
        max_visual_frames=256,
        max_evidence_tokens=256,
        max_total_cost=0.0,
    )
    invalid_question, _ = _question(tmp_path / "invalid", budget=0.0)
    invalid_benchmark = _benchmark((invalid_question,))
    invalid = evaluation.evaluate_run(
        run_id="invalid",
        policy_name="gist_only",
        policy_family="fixed",
        policy=baselines.GistOnlyPolicy(),
        questions=(invalid_question,),
        benchmark=invalid_benchmark,
        answerer=_answerer(),
        budgets=tiny,
        seed=1,
        cost_preference=0.1,
    )
    summary = invalid.summary
    assert summary.accuracy == 0.0
    assert summary.valid_only_accuracy is None
    assert summary.invalid_rate == 1.0
    assert metrics.fixed_budget_accuracy(invalid, 100.0) == 0.0
    assert invalid.records[0].cost_breakdown.total.total_cost > 0.0


def test_exact_coverage_tamper_and_cross_run_metrics_require_validated_runs(
    tmp_path: Path,
) -> None:
    questions = tuple(
        _question(
            tmp_path / f"q{index}",
            question_id=f"q{index}",
            video_group_id=f"group{index}",
        )[0]
        for index in range(4)
    )
    benchmark = _benchmark(questions)

    with pytest.raises(evaluation.EvaluationIntegrityError, match="exactly match"):
        evaluation.evaluate_run(
            run_id="missing",
            policy_name="gist_only",
            policy_family="fixed",
            policy=baselines.GistOnlyPolicy(),
            questions=questions[:1],
            benchmark=benchmark,
            answerer=_answerer(),
            budgets=evaluation.EvaluationBudgets(
                max_visual_frames=256,
                max_evidence_tokens=256,
                max_total_cost=100.0,
            ),
            seed=1,
            cost_preference=0.1,
        )

    single_question, budgets = _question(tmp_path / "single")
    run = evaluation.evaluate_run(
        run_id="sealed",
        policy_name="gist_only",
        policy_family="fixed",
        policy=baselines.GistOnlyPolicy(),
        questions=(single_question,),
        benchmark=_benchmark((single_question,)),
        answerer=_answerer(),
        budgets=budgets,
        seed=1,
        cost_preference=0.1,
    )
    bad_record = run.records[0].model_copy(update={"is_correct": False})
    bad_run = run.model_copy(update={"records": (bad_record,)})
    with pytest.raises((ValidationError, ValueError), match="is_correct|raw results"):
        _ = metrics.summarize_results(bad_run)

    constructed_payload = {
        name: getattr(run.records[0], name)
        for name in type(run.records[0]).model_fields
    }
    constructed_payload["realized_utility"] = math.nan
    constructed = evaluation.RawQuestionResult.model_construct(**constructed_payload)
    bad_constructed = run.model_copy(update={"records": (constructed,)})
    with pytest.raises((ValidationError, ValueError)):
        metrics.summarize_results(bad_constructed)

    with pytest.raises(TypeError, match="EvaluationRun"):
        metrics.pareto_frontier(
            (
                metrics.RunPoint.model_construct(
                    run_id="forged",
                    policy_name="forged",
                    seed=1,
                    accuracy=1.0,
                    total_cost=0.0,
                    run_manifest_sha256="0" * 64,
                    point_sha256="0" * 64,
                ),
            )
        )
    assert metrics.pareto_frontier((run,))[0].run_id == "sealed"
    assert metrics.cost_at_accuracy((run,), 1.0) is not None


def test_taxonomy_is_trajectory_derived_and_correct_primary_is_none(
    tmp_path: Path,
) -> None:
    missed_environment = _environment(search_ids=("e2",))
    missed, budgets = _question(
        tmp_path / "missed",
        environment=missed_environment,
        support=("e3",),
    )
    missed_run = evaluation.evaluate_run(
        run_id="missed",
        policy_name="gist_only",
        policy_family="fixed",
        policy=baselines.GistOnlyPolicy(),
        questions=(missed,),
        benchmark=_benchmark((missed,)),
        answerer=_answerer("B"),
        budgets=budgets,
        seed=1,
        cost_preference=0.1,
    )
    assert missed_run.records[0].error.primary.value == "recall_error"

    correct, correct_budgets = _question(
        tmp_path / "correct",
        required_fidelity=FidelityLevel.GIST,
    )
    correct_run = evaluation.evaluate_run(
        run_id="correct",
        policy_name="gist_residual",
        policy_family="fixed",
        policy=baselines.GistResidualPolicy(),
        questions=(correct,),
        benchmark=_benchmark((correct,)),
        answerer=_answerer("A"),
        budgets=correct_budgets,
        seed=1,
        cost_preference=0.1,
    )
    assert correct_run.records[0].error.primary is None
    assert any(
        cause.value == "over_retrieval"
        for cause in correct_run.records[0].error.secondary
    )
    with pytest.raises(ValidationError):
        evaluation.EvaluationQuestion(
            **correct.model_dump(mode="python"),
            signals={"gist_top_k_contains_answer": False},
        )


def test_adaptive_horizon_stays_five_and_config_is_actually_consumed(
    tmp_path: Path,
) -> None:
    question, budgets = _question(tmp_path)
    never_stop = baselines.TextAdaptivePolicy(
        lambda view: ActionType.SEARCH_GIST
        if not view.action_history
        else ActionType.EXPAND_CONTEXT
    )
    run = evaluation.evaluate_run(
        run_id="learned",
        policy_name="text_adaptive",
        policy_family="adaptive",
        policy=never_stop,
        questions=(question,),
        benchmark=_benchmark((question,)),
        answerer=_answerer(),
        budgets=budgets,
        seed=2026,
        cost_preference=0.1,
    )
    assert run.manifest.policy_horizon == 5
    assert run.records[0].forced_stop is True

    config = evaluation.load_evaluation_config(
        Path("configs/experiment/main_eval.yaml")
    )
    assert config.cost_preferences == COST_PREFERENCES
    cells = evaluation.evaluation_matrix(config)
    assert {cell.policy_name for cell in cells} == set(config.policies)
    assert {cell.seed for cell in cells} == set(config.seeds)
    assert {cell.cost_preference for cell in cells} == set(COST_PREFERENCES)
    assert all(cell.hardware.evaluation == "V100" for cell in cells)

    evaluation.apply_evaluation_seed(2027)
    import random
    import numpy as np
    import torch

    observed = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    evaluation.apply_evaluation_seed(2027)
    repeated = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert observed == repeated
