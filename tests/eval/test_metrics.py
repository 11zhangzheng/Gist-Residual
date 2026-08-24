from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from fidmem.eval.error_taxonomy import (
    ErrorCause,
    ErrorSignals,
    classify_error,
)
from fidmem.eval.metrics import (
    ResourceUsage,
    RunPoint,
    cost_at_accuracy,
    fixed_budget_accuracy,
    pareto_frontier,
    summarize_results,
)
from fidmem.eval.runner import (
    BenchmarkManifest,
    BenchmarkQuestionRef,
    EvaluationRun,
    RawQuestionResult,
    RunManifest,
    SharedEvaluationIdentity,
)
from fidmem.types import ActionInstance, ActionType


def _digest(character: str) -> str:
    return character * 64


def _benchmark(*, split: str = "test") -> BenchmarkManifest:
    return BenchmarkManifest.create(
        benchmark_id="manual",
        benchmark_version="1",
        split=split,
        provenance_sha256=_digest("1"),
        source_manifest_sha256=_digest("2"),
        group_assignment_sha256=_digest("3"),
        leakage_audit_sha256=_digest("4"),
        questions=(
            BenchmarkQuestionRef(question_id="q1", video_group_id="v1", record_sha256=_digest("5")),
            BenchmarkQuestionRef(question_id="q2", video_group_id="v2", record_sha256=_digest("6")),
            BenchmarkQuestionRef(question_id="q3", video_group_id="v3", record_sha256=_digest("7")),
            BenchmarkQuestionRef(question_id="q4", video_group_id="v4", record_sha256=_digest("8")),
        ),
    )


def _shared() -> SharedEvaluationIdentity:
    return SharedEvaluationIdentity(
        environment_sha256=_digest("9"),
        answerer_template_sha256=_digest("a"),
        answerer_config_sha256=_digest("b"),
        cache_graph_sha256=_digest("c"),
        cost_table_sha256=_digest("d"),
        max_visual_frames=100,
        max_evidence_tokens=100,
        max_total_cost=100.0,
    )


def _record(
    index: int,
    *,
    correct: bool,
    cost: float,
    invalid_reason: str | None = None,
    gpu_seconds: float = 0.0,
    frames: int = 0,
    signals: ErrorSignals | None = None,
    oracle_utility: float = 1.0,
    realized_utility: float = 0.0,
    actions: tuple[ActionInstance, ...] | None = None,
) -> RawQuestionResult:
    if actions is None:
        actions = (ActionInstance(ActionType.STOP, None, None),)
    return RawQuestionResult(
        run_id="run",
        policy_name="policy",
        policy_family="fixed",
        policy_identity_sha256=_digest("e"),
        seed=7,
        question_id=f"q{index}",
        video_group_id=f"v{index}",
        record_sha256=_digest(str(index + 4)),
        benchmark_manifest_sha256=_benchmark().manifest_sha256,
        split="test",
        shared=_shared(),
        cost_preference=0.25,
        predicted_answer="A" if correct else "B",
        gold_answer="A",
        is_correct=correct,
        invalid_reason=invalid_reason,
        acquisition_usage=ResourceUsage(
            total_cost=cost,
            gpu_seconds=gpu_seconds,
            input_frames=frames,
        ),
        controller_usage=ResourceUsage(),
        actions=actions,
        oracle_utility=oracle_utility,
        realized_utility=realized_utility,
        signals=signals or ErrorSignals(gist_top_k_contains_answer=True),
    )


def test_raw_models_reject_nan_negative_costs_train_split_and_inconsistent_answers() -> None:
    with pytest.raises(ValidationError):
        ResourceUsage(total_cost=math.nan)
    with pytest.raises(ValidationError):
        ResourceUsage(total_cost="1")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        ResourceUsage(input_frames=-1)
    with pytest.raises(ValidationError):
        _benchmark(split="train")
    with pytest.raises(ValidationError, match="is_correct"):
        _record(1, correct=True, cost=1).model_copy(
            update={"predicted_answer": "B"}
        ).__class__.model_validate(
            _record(1, correct=True, cost=1).model_dump()
            | {"predicted_answer": "B"}
        )


def test_summary_is_recomputed_from_raw_records_with_invalid_separate_and_cost_retained() -> None:
    records = (
        _record(
            1,
            correct=True,
            cost=1,
            gpu_seconds=1,
            frames=1,
            oracle_utility=1,
            realized_utility=0.7,
            actions=(
                ActionInstance(ActionType.SEARCH_GIST, None, None),
                ActionInstance(ActionType.STOP, None, None),
            ),
        ),
        _record(
            2,
            correct=False,
            cost=3,
            gpu_seconds=2,
            frames=2,
            oracle_utility=1,
            realized_utility=0.2,
            signals=ErrorSignals(
                gist_top_k_contains_answer=False,
                stopped_with_insufficient_evidence=True,
                unnecessary_expansion=True,
            ),
            actions=(
                ActionInstance(ActionType.EXPAND_RESIDUAL, "e", None),
                ActionInstance(ActionType.STOP, None, None),
            ),
        ),
        _record(3, correct=True, cost=2, oracle_utility=1, realized_utility=1),
        _record(4, correct=False, cost=4, invalid_reason="visual_frame_budget_exceeded"),
    )
    summary = summarize_results(records)
    assert summary.total_questions == 4
    assert summary.valid_questions == 3
    assert summary.invalid_questions == 1
    assert summary.accuracy == pytest.approx(2 / 3)
    assert summary.total_cost == 10
    assert summary.total_gpu_seconds == 3
    assert summary.total_input_frames == 3
    assert summary.oracle_utility_regret == pytest.approx((0.3 + 0.8 + 0.0) / 3)
    assert summary.premature_stop_rate == pytest.approx(1 / 3)
    assert summary.unnecessary_expansion_rate == pytest.approx(1 / 3)
    assert summary.top_k_recall == pytest.approx(2 / 3)
    assert summary.action_distribution == {
        "EXPAND_RESIDUAL": pytest.approx(0.2),
        "SEARCH_GIST": pytest.approx(0.2),
        "STOP": pytest.approx(0.6),
    }
    assert fixed_budget_accuracy(records, 2.0) == pytest.approx(2 / 3)


def test_pareto_and_cost_at_accuracy_match_hand_calculation_with_stable_ties() -> None:
    points = (
        RunPoint(run_id="a", policy_name="a", seed=1, accuracy=0.6, total_cost=10),
        RunPoint(run_id="b", policy_name="b", seed=1, accuracy=0.7, total_cost=9),
        RunPoint(run_id="c", policy_name="c", seed=1, accuracy=0.8, total_cost=12),
        RunPoint(run_id="d", policy_name="d", seed=1, accuracy=0.8, total_cost=12),
        RunPoint(run_id="e", policy_name="e", seed=1, accuracy=0.5, total_cost=8),
    )
    assert tuple(point.run_id for point in pareto_frontier(points)) == ("e", "b", "c")
    assert cost_at_accuracy(points, 0.7) == 9
    assert cost_at_accuracy(points, 0.9) is None
    with pytest.raises(ValueError, match="non-empty"):
        pareto_frontier(())
    with pytest.raises(ValidationError):
        RunPoint(run_id="nan", policy_name="x", seed=1, accuracy=math.nan, total_cost=1)


def test_error_taxonomy_has_exact_priority_and_secondary_flags_but_skips_invalid() -> None:
    all_signals = ErrorSignals(
        gist_top_k_contains_answer=False,
        oracle_evidence_sufficient=True,
        answerer_correct_with_oracle_evidence=False,
        stopped_with_insufficient_evidence=True,
        useful_fidelity_upgrade_available=True,
        unnecessary_expansion=True,
    )
    classified = classify_error(all_signals)
    assert classified.primary is ErrorCause.RECALL
    assert classified.secondary == (
        ErrorCause.RECALL,
        ErrorCause.ANSWERER,
        ErrorCause.PREMATURE_STOP,
        ErrorCause.INSUFFICIENT_FIDELITY,
        ErrorCause.OVER_RETRIEVAL,
    )

    no_recall = all_signals.model_copy(update={"gist_top_k_contains_answer": True})
    assert classify_error(no_recall).primary is ErrorCause.ANSWERER
    no_answerer = no_recall.model_copy(update={"answerer_correct_with_oracle_evidence": True})
    assert classify_error(no_answerer).primary is ErrorCause.PREMATURE_STOP
    no_stop = no_answerer.model_copy(update={"stopped_with_insufficient_evidence": False})
    assert classify_error(no_stop).primary is ErrorCause.INSUFFICIENT_FIDELITY
    no_upgrade = no_stop.model_copy(update={"useful_fidelity_upgrade_available": False})
    assert classify_error(no_upgrade).primary is ErrorCause.OVER_RETRIEVAL
    assert classify_error(all_signals, invalid=True).primary is None


def test_run_manifest_rejects_duplicate_question_video_keys_and_summary_spoofing() -> None:
    record = _record(1, correct=True, cost=1)
    duplicate = record.model_copy()
    with pytest.raises(ValueError, match="duplicate question/video"):
        RunManifest.create(
            run_id="run",
            policy_name="policy",
            policy_family="fixed",
            policy_identity_sha256=_digest("e"),
            seed=7,
            benchmark=_benchmark(),
            shared=_shared(),
            cost_preference=0.25,
            records=(record, duplicate),
        )

    with pytest.raises(ValueError, match="raw result identity"):
        RunManifest.create(
            run_id="run",
            policy_name="policy",
            policy_family="fixed",
            policy_identity_sha256=_digest("e"),
            seed=7,
            benchmark=_benchmark(),
            shared=_shared(),
            cost_preference=0.25,
            records=(record.model_copy(update={"run_id": "forged"}),),
        )
    with pytest.raises(ValueError, match="benchmark question provenance"):
        RunManifest.create(
            run_id="run", policy_name="policy", policy_family="fixed",
            policy_identity_sha256=_digest("e"), seed=7, benchmark=_benchmark(),
            shared=_shared(), cost_preference=0.25,
            records=(record.model_copy(update={"record_sha256": _digest("f")}),),
        )
    with pytest.raises(ValueError, match="is_correct"):
        RunManifest.create(
            run_id="run",
            policy_name="policy",
            policy_family="fixed",
            policy_identity_sha256=_digest("e"),
            seed=7,
            benchmark=_benchmark(),
            shared=_shared(),
            cost_preference=0.25,
            records=(record.model_copy(update={"is_correct": False}),),
        )
    bad_shared = _shared().model_copy(update={"max_visual_frames": -1})
    with pytest.raises(ValueError, match="max_visual_frames"):
        RunManifest.create(
            run_id="run",
            policy_name="policy",
            policy_family="fixed",
            policy_identity_sha256=_digest("e"),
            seed=7,
            benchmark=_benchmark(),
            shared=bad_shared,
            cost_preference=0.25,
            records=(record.model_copy(update={"shared": bad_shared}),),
        )

    manifest = RunManifest.create(
        run_id="run",
        policy_name="policy",
        policy_family="fixed",
        policy_identity_sha256=_digest("e"),
        seed=7,
        benchmark=_benchmark(),
        shared=_shared(),
        cost_preference=0.25,
        records=(record,),
    )
    run = EvaluationRun(manifest=manifest, records=(record,))
    assert run.summary.accuracy == 1.0
    with pytest.raises(ValidationError, match="summary"):
        EvaluationRun.model_validate(
            run.model_dump(mode="python") | {"summary": {"accuracy": 0.0}}
        )


def test_empty_denominators_and_impossible_oracle_values_fail_closed() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        summarize_results(())
    with pytest.raises(ValueError, match="finite"):
        fixed_budget_accuracy((_record(1, correct=True, cost=1),), math.nan)
    with pytest.raises(ValueError, match="no valid questions"):
        summarize_results((_record(1, correct=False, cost=1, invalid_reason="budget"),))
    with pytest.raises(ValidationError, match="oracle utility"):
        _record(1, correct=True, cost=1, oracle_utility=0.0, realized_utility=0.1)
