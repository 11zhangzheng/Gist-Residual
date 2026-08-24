from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

import fidmem.eval.baselines as baselines
import fidmem.eval.runner as evaluation
from fidmem.agent.answerer import AnswererAdapterResult, FrozenAnswerer
from fidmem.types import ActionInstance, ActionType

from tests.eval.test_eval_review_round1 import (
    _answerer,
    _benchmark,
    _cache,
    _environment,
    _question,
    _record,
    _state,
)


class _CopiedActionPolicy(baselines.GistOnlyPolicy):
    def __call__(
        self,
        state,
        legal_actions: tuple[ActionInstance, ...],
    ) -> ActionInstance:
        del state
        return legal_actions[0].model_copy()


def test_runner_rejects_equal_copied_action_from_builtin_subclass(
    tmp_path: Path,
) -> None:
    question, budgets = _question(tmp_path)
    with pytest.raises(
        evaluation.EvaluationIntegrityError, match="exact ActionInstance"
    ):
        evaluation.evaluate_run(
            run_id="copied-action",
            policy_name="malicious-copy",
            policy_family="fixed",
            policy=_CopiedActionPolicy(),
            questions=(question,),
            benchmark=_benchmark((question,)),
            answerer=_answerer(),
            budgets=budgets,
            seed=11,
            cost_preference=0.1,
        )


def test_prompt_rationale_never_reaches_answerer_evidence_or_raw_record(
    tmp_path: Path,
) -> None:
    question, budgets = _question(tmp_path)
    prompts: list[str] = []

    def adapter(prompt: str) -> AnswererAdapterResult:
        prompts.append(prompt)
        return AnswererAdapterResult(
            response="A",
            cost_record=_record("answerer", wall=0.2, text_tokens=5),
            total_cost=0.25,
        )

    answerer = FrozenAnswerer(
        adapter,
        model_artifact_sha256="c" * 64,
        model_revision="private-rationale-test",
        decode_config={"temperature": 0.0},
    )

    def controller(state, legal):
        desired = (
            ActionType.SEARCH_GIST if not state.candidate_event_ids else ActionType.STOP
        )
        selected = next(action for action in legal if action.action_type is desired)
        return baselines.PromptControllerDecision(
            action=selected,
            rationale="SECRET PRIVATE RATIONALE",
            cost=baselines.ControllerCost(total_cost=0.1, text_tokens=3),
        )

    run = evaluation.evaluate_run(
        run_id="private-rationale",
        policy_name="prompt_vlm",
        policy_family="adaptive",
        policy=baselines.PromptControllerPolicy(controller),
        questions=(question,),
        benchmark=_benchmark((question,)),
        answerer=answerer,
        budgets=budgets,
        seed=12,
        cost_preference=0.1,
    )
    assert prompts and "SECRET PRIVATE RATIONALE" not in prompts[0]
    assert "SECRET PRIVATE RATIONALE" not in run.model_dump_json()


@pytest.mark.parametrize(
    ("frames", "tokens", "total_cost", "expected"),
    (
        (256, 10, 100.0, "evidence_token_budget_exceeded"),
        (256, 256, 35.0, "total_cost_budget_exceeded"),
    ),
)
def test_long_uniform_trace_preserves_cost_when_other_budgets_are_exceeded(
    tmp_path: Path,
    frames: int,
    tokens: int,
    total_cost: float,
    expected: str,
) -> None:
    environment = _environment()
    authority, _ = _question(tmp_path / "authority", environment=environment)
    budgets = evaluation.EvaluationBudgets(
        max_visual_frames=frames,
        max_evidence_tokens=tokens,
        max_total_cost=total_cost,
    )
    question = evaluation.EvaluationQuestion.create(
        question_id=f"q-{expected}",
        video_group_id=f"group-{expected}",
        video_id="video",
        split="test",
        source_manifest={"dataset": "synthetic", "case": expected},
        initial_state=_state(budget=total_cost),
        gold_answer="A",
        environment=environment,
        cache=_cache(tmp_path / expected),
        budgets=budgets,
        oracle=authority.oracle,
    )
    run = evaluation.evaluate_run(
        run_id=expected,
        policy_name="uniform",
        policy_family="fixed",
        policy=baselines.UniformFramesPolicy(),
        questions=(question,),
        benchmark=_benchmark((question,)),
        answerer=_answerer(),
        budgets=budgets,
        seed=13,
        cost_preference=0.1,
    )
    record = run.records[0]
    assert record.invalid_reason == expected
    assert len(record.actions) > 5
    assert record.cost_breakdown.total.total_cost > 0


def test_evaluate_run_applies_seed_and_config_rejects_missing_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[int] = []
    monkeypatch.setattr(evaluation, "apply_evaluation_seed", seen.append)
    question, budgets = _question(tmp_path)
    run = evaluation.evaluate_run(
        run_id="seeded",
        policy_name="gist_only",
        policy_family="fixed",
        policy=baselines.GistOnlyPolicy(),
        questions=(question,),
        benchmark=_benchmark((question,)),
        answerer=_answerer(),
        budgets=budgets,
        seed=2028,
        cost_preference=0.1,
    )
    assert run.manifest.seed == 2028
    assert seen == [2028]

    config = evaluation.load_evaluation_config(
        Path("configs/experiment/main_eval.yaml")
    )
    payload = config.model_dump(mode="python")
    payload["policies"].pop("uniform")
    with pytest.raises(ValidationError, match="policy matrix"):
        evaluation.EvaluationConfig.model_validate(payload)


def test_summary_field_spoofing_and_single_event_fixed_horizon_fail_closed(
    tmp_path: Path,
) -> None:
    question, budgets = _question(tmp_path / "summary")
    run = evaluation.evaluate_run(
        run_id="summary",
        policy_name="gist_only",
        policy_family="fixed",
        policy=baselines.GistOnlyPolicy(),
        questions=(question,),
        benchmark=_benchmark((question,)),
        answerer=_answerer(),
        budgets=budgets,
        seed=14,
        cost_preference=0.1,
    )
    with pytest.raises(ValidationError):
        evaluation.EvaluationRun.model_validate(
            run.model_dump(mode="python") | {"summary": {"accuracy": 1.0}}
        )

    one_event_environment = _environment(
        events=_environment().canonical_events[:1],
        search_ids=("e0",),
    )
    one, one_budgets = _question(
        tmp_path / "one-event",
        environment=one_event_environment,
        support=("e0",),
    )
    fixed = evaluation.evaluate_run(
        run_id="one-event-fixed",
        policy_name="uniform",
        policy_family="fixed",
        policy=baselines.UniformFramesPolicy(),
        questions=(one,),
        benchmark=_benchmark((one,)),
        answerer=_answerer(),
        budgets=one_budgets,
        seed=15,
        cost_preference=0.1,
    )
    assert fixed.manifest.policy_horizon == 4
