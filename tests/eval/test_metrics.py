from __future__ import annotations

import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from fidmem.agent.answerer import AnswererAdapterResult, FrozenAnswerer
from fidmem.eval.baselines import (
    GistOnlyPolicy,
    GistResidualPolicy,
    GistVisualPolicy,
)
from fidmem.eval.error_taxonomy import ErrorCause, ErrorSignals, classify_error
from fidmem.eval.metrics import (
    ResourceUsage,
    RunPoint,
    cost_at_accuracy,
    fixed_budget_accuracy,
    pareto_frontier,
    summarize_results,
)
from fidmem.eval.runner import EvaluationBudgets, evaluate_run

from tests.eval.test_eval_review_round1 import (
    _benchmark,
    _question,
    _record,
)


def _adaptive_answerer() -> FrozenAnswerer:
    usage = _record("answerer", wall=0.5, text_tokens=7)

    def adapter(prompt: str) -> AnswererAdapterResult:
        answer = "A" if "residual e3" in prompt or "visual e3" in prompt else "B"
        return AnswererAdapterResult(
            response=answer,
            cost_record=usage,
            total_cost=0.5,
        )

    return FrozenAnswerer(
        adapter,
        model_artifact_sha256="a" * 64,
        model_revision="metric-answerer-r1",
        decode_config={"temperature": 0.0, "max_tokens": 8},
    )


def _runs(tmp_path: Path):
    question, budgets = _question(tmp_path, oracle_cost=0.0)
    benchmark = _benchmark((question,), base_cost=4.0)
    answerer = _adaptive_answerer()

    def run(run_id: str, policy_name: str, policy):
        return evaluate_run(
            run_id=run_id,
            policy_name=policy_name,
            policy_family="fixed",
            policy=policy,
            questions=(question,),
            benchmark=benchmark,
            answerer=answerer,
            budgets=budgets,
            seed=7,
            cost_preference=0.1,
        )

    return (
        run("gist", "gist_only", GistOnlyPolicy()),
        run("residual", "gist_residual", GistResidualPolicy()),
        run("visual", "gist_visual", GistVisualPolicy()),
    )


def test_strict_numeric_models_and_sealed_run_points_fail_closed() -> None:
    with pytest.raises(ValidationError):
        ResourceUsage(total_cost=math.nan)
    with pytest.raises(ValidationError):
        ResourceUsage(total_cost="1")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        ResourceUsage(input_frames=-1)
    with pytest.raises(ValidationError):
        RunPoint(
            run_id="forged",
            policy_name="forged",
            seed=1,
            accuracy=1.0,
            total_cost=0.0,
            run_manifest_sha256="0" * 64,
            point_sha256="0" * 64,
        )


def test_metrics_recompute_from_validated_runs_and_count_invalid_in_denominator(
    tmp_path: Path,
) -> None:
    gist, residual, _visual = _runs(tmp_path / "valid")
    gist_summary = summarize_results(gist)
    residual_summary = summarize_results(residual)
    assert gist_summary.accuracy == 0.0
    assert gist_summary.valid_only_accuracy == 0.0
    assert residual_summary.accuracy == 1.0
    assert residual_summary.total_cost > gist_summary.total_cost
    assert fixed_budget_accuracy(residual, residual_summary.total_cost) == 1.0
    assert (
        fixed_budget_accuracy(
            residual, math.nextafter(residual_summary.total_cost, 0.0)
        )
        == 0.0
    )

    invalid_question, _ = _question(tmp_path / "invalid", budget=0.0, oracle_cost=0.0)
    tiny = EvaluationBudgets(
        max_visual_frames=256,
        max_evidence_tokens=256,
        max_total_cost=0.0,
    )
    invalid = evaluate_run(
        run_id="invalid",
        policy_name="gist_only",
        policy_family="fixed",
        policy=GistOnlyPolicy(),
        questions=(invalid_question,),
        benchmark=_benchmark((invalid_question,), base_cost=4.0),
        answerer=_adaptive_answerer(),
        budgets=tiny,
        seed=7,
        cost_preference=0.1,
    )
    summary = summarize_results(invalid)
    assert summary.accuracy == 0.0
    assert summary.valid_only_accuracy is None
    assert summary.invalid_rate == 1.0
    assert summary.total_cost > 0.0
    assert fixed_budget_accuracy(invalid, 1000.0) == 0.0


def test_pareto_and_cost_at_accuracy_use_only_comparable_sealed_runs(
    tmp_path: Path,
) -> None:
    gist, residual, visual = _runs(tmp_path / "comparable")
    frontier = pareto_frontier((visual, residual, gist))
    assert tuple(point.run_id for point in frontier) == ("gist", "residual")
    residual_cost = summarize_results(residual).total_cost
    assert cost_at_accuracy((visual, residual, gist), 1.0) == residual_cost
    assert cost_at_accuracy((gist,), 1.0) is None
    with pytest.raises(ValueError, match="non-empty"):
        pareto_frontier(())
    with pytest.raises(TypeError, match="EvaluationRun"):
        pareto_frontier((RunPoint.model_construct(),))

    question = residual.manifest.benchmark.questions[0]
    del question
    other_question, other_budgets = _question(tmp_path / "different")
    different = evaluate_run(
        run_id="different-answerer",
        policy_name="gist_only",
        policy_family="fixed",
        policy=GistOnlyPolicy(),
        questions=(other_question,),
        benchmark=_benchmark((other_question,)),
        answerer=FrozenAnswerer(
            lambda _prompt: AnswererAdapterResult(
                response="A",
                cost_record=_record("answerer"),
                total_cost=0.0,
            ),
            model_artifact_sha256="b" * 64,
            model_revision="different",
            decode_config={"temperature": 0.0},
        ),
        budgets=other_budgets,
        seed=7,
        cost_preference=0.1,
    )
    with pytest.raises(ValueError, match="comparable"):
        pareto_frontier((residual, different))


def test_error_taxonomy_priority_is_exact_and_correct_or_invalid_has_no_primary() -> (
    None
):
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
    no_answerer = no_recall.model_copy(
        update={"answerer_correct_with_oracle_evidence": True}
    )
    assert classify_error(no_answerer).primary is ErrorCause.PREMATURE_STOP
    no_stop = no_answerer.model_copy(
        update={"stopped_with_insufficient_evidence": False}
    )
    assert classify_error(no_stop).primary is ErrorCause.INSUFFICIENT_FIDELITY
    no_upgrade = no_stop.model_copy(update={"useful_fidelity_upgrade_available": False})
    assert classify_error(no_upgrade).primary is ErrorCause.OVER_RETRIEVAL
    assert classify_error(all_signals, invalid=True).primary is None
    assert classify_error(all_signals, correct=True).primary is None
