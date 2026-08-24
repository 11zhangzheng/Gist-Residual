from __future__ import annotations

from pathlib import Path

import fidmem.eval.baselines as baselines
import fidmem.eval.metrics as metrics
import fidmem.eval.runner as evaluation
from fidmem.types import ActionType

from tests.eval.test_eval_review_round1 import (
    _benchmark,
    _cache,
    _answerer,
    _environment,
    _question,
    _state,
)
from tests.eval.test_metrics import _adaptive_answerer


def test_long_fixed_trace_remains_complete_but_invalid_over_frame_budget(
    tmp_path: Path,
) -> None:
    environment = _environment()
    authority, _ = _question(tmp_path / "authority", environment=environment)
    budgets = evaluation.EvaluationBudgets(
        max_visual_frames=10,
        max_evidence_tokens=256,
        max_total_cost=100.0,
    )
    question = evaluation.EvaluationQuestion.create(
        question_id="q-limited",
        video_group_id="group-limited",
        video_id=authority.video_id,
        split="test",
        source_manifest={
            "dataset": "synthetic",
            "version": "1",
            "video_id": "video",
            "question_id": "q-limited",
        },
        initial_state=_state(budget=100.0),
        gold_answer="A",
        environment=environment,
        cache=_cache(tmp_path / "limited"),
        budgets=budgets,
        oracle=authority.oracle,
    )
    run = evaluation.evaluate_run(
        run_id="uniform-over-frames",
        policy_name="uniform",
        policy_family="fixed",
        policy=baselines.UniformFramesPolicy(),
        questions=(question,),
        benchmark=_benchmark((question,)),
        answerer=_answerer(),
        budgets=budgets,
        seed=1,
        cost_preference=0.1,
    )
    record = run.records[0]
    assert record.invalid_reason == "visual_frame_budget_exceeded"
    assert (
        sum(action.action_type is ActionType.VERIFY_VISUAL for action in record.actions)
        == 6
    )
    assert record.cost_breakdown.total.input_frames == 72


def test_base_memory_resources_amortize_by_actual_group_query_count(
    tmp_path: Path,
) -> None:
    first, budgets = _question(
        tmp_path / "first", question_id="q1", video_group_id="shared-group"
    )
    second, _ = _question(
        tmp_path / "second", question_id="q2", video_group_id="shared-group"
    )
    base = evaluation.BaseMemoryCostAuthority.create(
        video_group_id="shared-group",
        usage=metrics.ResourceUsage(total_cost=3.0, text_tokens=5),
        artifact_name="odd-token-base",
    )
    template = _benchmark((first, second))
    benchmark = evaluation.BenchmarkManifest.create(
        benchmark_id="amortized",
        benchmark_version="1",
        questions=(first, second),
        leakage_audit={"status": "passed", "auditor": "synthetic-v1"},
        base_memory_costs=(base,),
        normalization=template.normalization,
        gpu_assignment=template.gpu_assignment,
    )
    run = evaluation.evaluate_run(
        run_id="amortized",
        policy_name="gist_only",
        policy_family="fixed",
        policy=baselines.GistOnlyPolicy(),
        questions=(first, second),
        benchmark=benchmark,
        answerer=_adaptive_answerer(),
        budgets=budgets,
        seed=7,
        cost_preference=0.1,
    )
    assert tuple(
        record.cost_breakdown.base_memory.text_tokens for record in run.records
    ) == (2.5, 2.5)
    assert metrics.summarize_results(run).total_text_tokens == 23.0
