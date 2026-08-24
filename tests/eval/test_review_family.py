from pathlib import Path

import pytest

import fidmem.eval.baselines as baselines
import fidmem.eval.runner as evaluation

from tests.eval.test_eval_review_round1 import _answerer, _benchmark, _question


def test_runner_rejects_caller_forged_policy_family(tmp_path: Path) -> None:
    question, budgets = _question(tmp_path)
    with pytest.raises(evaluation.EvaluationIntegrityError, match="policy family"):
        evaluation.evaluate_run(
            run_id="forged-family",
            policy_name="gist_only",
            policy_family="learned",
            policy=baselines.GistOnlyPolicy(),
            questions=(question,),
            benchmark=_benchmark((question,)),
            answerer=_answerer(),
            budgets=budgets,
            seed=1,
            cost_preference=0.1,
        )
