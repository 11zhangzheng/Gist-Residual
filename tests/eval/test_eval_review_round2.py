from __future__ import annotations

import math
from pathlib import Path

import pytest

import fidmem.eval.baselines as baselines
import fidmem.eval.metrics as metrics
import fidmem.eval.runner as evaluation
from fidmem.eval.error_taxonomy import classify_error

from tests.eval.test_eval_review_round1 import _answerer, _benchmark, _question


def _rehash_raw(payload: dict[str, object]) -> evaluation.RawQuestionResult:
    unsigned = dict(payload)
    unsigned.pop("result_sha256", None)
    payload["result_sha256"] = evaluation._sha256(unsigned)
    return evaluation.RawQuestionResult.model_validate(payload)


def test_gold_rehash_tamper_is_rejected_against_live_benchmark_question(
    tmp_path: Path,
) -> None:
    question, budgets = _question(tmp_path)
    benchmark = _benchmark((question,))
    run = evaluation.evaluate_run(
        run_id="wrong",
        policy_name="gist_only",
        policy_family="fixed",
        policy=baselines.GistOnlyPolicy(),
        questions=(question,),
        benchmark=benchmark,
        answerer=_answerer("B"),
        budgets=budgets,
        seed=1,
        cost_preference=0.1,
    )
    assert metrics.summarize_results(run).accuracy == 0.0

    payload = run.records[0].model_dump(mode="python")
    payload.update(
        gold_answer="B",
        is_correct=True,
        answer_score=1.0,
        realized_utility=(
            1.0
            - 0.1
            * run.records[0].cost_breakdown.total.total_cost
            / run.records[0].cost_normalization.constant
        ),
    )
    payload["oracle_utility_regret"] = max(
        0.0,
        float(payload["oracle_utility"])
        - float(payload["realized_utility"]),
    )
    payload["error"] = classify_error(run.records[0].signals, correct=True)
    forged = _rehash_raw(payload)
    manifest = evaluation.RunManifest.create(
        run_id=run.manifest.run_id,
        policy_name=run.manifest.policy_name,
        policy_family=run.manifest.policy_family,
        policy_identity_sha256=run.manifest.policy_identity_sha256,
        policy_horizon=run.manifest.policy_horizon,
        horizon_category=run.manifest.horizon_category,
        seed=run.manifest.seed,
        benchmark=benchmark,
        shared=run.manifest.shared,
        cost_preference=run.manifest.cost_preference,
        records=(forged,),
    )
    forged_run = run.model_copy(
        update={"manifest": manifest, "records": (forged,)}
    )
    with pytest.raises(ValueError, match="benchmark question authority"):
        metrics.summarize_results(forged_run)
