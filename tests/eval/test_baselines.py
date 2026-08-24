from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from fidmem.actions.environment import (
    ActionCostTable,
    ActionObservation,
    MemoryEnvironment,
    OperationMetadata,
)
from fidmem.agent.answerer import FrozenAnswerer
from fidmem.eval.baselines import (
    BCPolicyAdapter,
    BaselinePolicyError,
    ControllerCost,
    FullResidualPolicy,
    GistOnlyPolicy,
    GistResidualPolicy,
    GistVisualPolicy,
    PromptControllerDecision,
    PromptControllerPolicy,
    QuestionOnlyPolicy,
    RulePolicy,
    TextAdaptivePolicy,
    UniformFramesPolicy,
)
from fidmem.eval.error_taxonomy import ErrorSignals
from fidmem.eval.runner import (
    AnswererBinding,
    BenchmarkManifest,
    BenchmarkQuestionRef,
    EvaluationIntegrityError,
    EvaluationQuestion,
    SharedEvaluationIdentity,
    build_shared_identity,
    evaluate_run,
)
from fidmem.types import (
    ActionInstance,
    ActionType,
    EventRecord,
    EvidenceItem,
    FidelityLevel,
    RouterState,
)


def _digest(character: str) -> str:
    return character * 64


def _state(*, candidates: tuple[str, ...] = (), evidence: tuple[EvidenceItem, ...] = ()) -> RouterState:
    return RouterState(
        question="Which option is supported?",
        options=("A", "B"),
        evidence=evidence,
        action_history=(),
        remaining_budget=100.0,
        candidate_event_ids=candidates,
        candidate_fidelity_levels={event_id: FidelityLevel.GIST for event_id in candidates},
        context_frontiers={event_id: (0, 0) for event_id in candidates},
        cost_preference=0.25,
    )


def _action(
    action_type: ActionType,
    event_id: str | None = None,
    visual_budget: str | None = None,
) -> ActionInstance:
    return ActionInstance(action_type, event_id, visual_budget)  # type: ignore[arg-type]


def test_fixed_and_rule_policies_return_stable_members_of_the_legal_tuple() -> None:
    search = _action(ActionType.SEARCH_GIST)
    stop = _action(ActionType.STOP)
    assert GistOnlyPolicy()(_state(), (search, stop)) is search

    residual_two = _action(ActionType.EXPAND_RESIDUAL, "e2")
    residual_one = _action(ActionType.EXPAND_RESIDUAL, "e1")
    visual_low = _action(ActionType.VERIFY_VISUAL, "e1", "low")
    visual_high = _action(ActionType.VERIFY_VISUAL, "e1", "high")
    context = _action(ActionType.EXPAND_CONTEXT, "e1")
    legal = (residual_two, visual_high, stop, residual_one, context, visual_low)
    candidate_state = _state(candidates=("e1", "e2"))

    assert GistOnlyPolicy()(candidate_state, legal) is stop
    assert GistResidualPolicy()(candidate_state, legal) is residual_one
    assert GistVisualPolicy()(candidate_state, legal) is visual_high
    assert UniformFramesPolicy()(candidate_state, legal) is visual_low
    assert FullResidualPolicy()(candidate_state, legal) is residual_one
    assert RulePolicy(sufficiency_threshold=0.9)(candidate_state, legal) is residual_one


def test_fixed_policy_fails_closed_when_neither_preference_nor_stop_is_legal() -> None:
    residual = _action(ActionType.EXPAND_RESIDUAL, "e1")
    with pytest.raises(BaselinePolicyError, match="STOP is not legal"):
        GistVisualPolicy()(_state(candidates=("e1",)), (residual,))


def test_question_only_and_text_adaptive_controllers_receive_restricted_views() -> None:
    stop = _action(ActionType.STOP)
    visual = EvidenceItem(
        event_id="e1",
        fidelity_level=FidelityLevel.VISUAL,
        content="private visual content",
        score=1.0,
        attachments=("frame.png",),
    )
    gist = EvidenceItem(
        event_id="e1",
        fidelity_level=FidelityLevel.GIST,
        content="public gist text",
        score=0.2,
    )
    state = _state(candidates=("e1",), evidence=(visual, gist))

    def question_controller(view: object, legal: tuple[ActionInstance, ...]) -> ActionInstance:
        assert vars(view) == {
            "question": state.question,
            "options": state.options,
            "remaining_budget": state.remaining_budget,
            "cost_preference": state.cost_preference,
        }
        assert not hasattr(view, "candidate_event_ids")
        return legal[0]

    def text_controller(view: object, legal: tuple[ActionInstance, ...]) -> ActionInstance:
        assert vars(view)["gist_text"] == ("public gist text",)
        assert "private visual content" not in repr(view)
        assert "frame.png" not in repr(view)
        assert not hasattr(view, "candidate_fidelity_levels")
        assert not hasattr(view, "context_frontiers")
        assert not hasattr(view, "evidence")
        return legal[0]

    assert QuestionOnlyPolicy(question_controller)(state, (stop,)) is stop
    assert TextAdaptivePolicy(text_controller)(state, (stop,)) is stop


def test_learning_adapter_rejects_an_equal_copy_instead_of_an_exact_legal_instance() -> None:
    stop = _action(ActionType.STOP)
    copied = stop.model_copy()
    adapter = BCPolicyAdapter(lambda _state, _legal: copied)
    with pytest.raises(BaselinePolicyError, match="exact legal ActionInstance"):
        adapter(_state(), (stop,))


class _RecordingEnvironment(MemoryEnvironment):
    last_valid_actions: tuple[ActionInstance, ...] | None = None

    def valid_actions(self, state: RouterState) -> tuple[ActionInstance, ...]:
        actions = super().valid_actions(state)
        self.last_valid_actions = actions
        return actions


def _environment(*, visual_frames: int = 12, residual_cost: float = 2.0) -> _RecordingEnvironment:
    events = (
        EventRecord(video_id="v", event_id="e1", start_sec=0, end_sec=1),
        EventRecord(video_id="v", event_id="e2", start_sec=2, end_sec=3),
    )

    def executor(action: ActionInstance, state: RouterState) -> ActionObservation:
        if action.action_type is ActionType.SEARCH_GIST:
            return ActionObservation(
                action_type=action.action_type,
                target_event_id=None,
                evidence=(
                    EvidenceItem(event_id="e1", fidelity_level=FidelityLevel.GIST, content="A clue", score=0.8),
                    EvidenceItem(event_id="e2", fidelity_level=FidelityLevel.GIST, content="distractor", score=0.2),
                ),
                candidate_event_ids=("e1", "e2"),
                operation_metadata=(
                    OperationMetadata(scope="search_gist", cache_status="miss", amortizable=True, text_tokens=2),
                ),
            )
        if action.action_type is ActionType.EXPAND_RESIDUAL:
            return ActionObservation(
                action_type=action.action_type,
                target_event_id=action.event_id,
                evidence=(EvidenceItem(event_id=action.event_id or "", fidelity_level=FidelityLevel.RESIDUAL, content="A residual", score=0.9),),
                operation_metadata=(
                    OperationMetadata(scope="residual", cache_status="miss", amortizable=True, text_tokens=2),
                ),
            )
        if action.action_type is ActionType.VERIFY_VISUAL:
            return ActionObservation(
                action_type=action.action_type,
                target_event_id=action.event_id,
                evidence=(EvidenceItem(event_id=action.event_id or "", fidelity_level=FidelityLevel.VISUAL, content="A visual", score=1.0, attachments=("frame.png",)),),
                operation_metadata=(
                    OperationMetadata(scope="event_observation", cache_status="miss", amortizable=True, input_frames=visual_frames, visual_tokens=4),
                    OperationMetadata(scope="question_verification", cache_status="miss", amortizable=False, text_tokens=1),
                ),
            )
        return ActionObservation(action_type=action.action_type, target_event_id=action.event_id)

    return _RecordingEnvironment(
        events=events,
        executor=executor,
        costs=ActionCostTable(residual=residual_cost),
    )


def _benchmark() -> BenchmarkManifest:
    return BenchmarkManifest.create(
        benchmark_id="synthetic-eval",
        benchmark_version="1",
        split="test",
        provenance_sha256=_digest("1"),
        source_manifest_sha256=_digest("2"),
        group_assignment_sha256=_digest("3"),
        leakage_audit_sha256=_digest("4"),
        questions=(BenchmarkQuestionRef(question_id="q1", video_group_id="v", record_sha256=_digest("5")),),
    )


def _question(environment: MemoryEnvironment, *, cache_hash: str = _digest("6")) -> EvaluationQuestion:
    return EvaluationQuestion(
        question_id="q1",
        video_group_id="v",
        record_sha256=_digest("5"),
        initial_state=_state(),
        gold_answer="A",
        environment=environment,
        cache_graph_sha256=cache_hash,
        oracle_utility=1.0,
        signals=ErrorSignals(gist_top_k_contains_answer=True),
    )


def _binding(prompts: list[str], *, config_hash: str = _digest("7")) -> AnswererBinding:
    def adapter(prompt: str) -> str:
        prompts.append(prompt)
        return "A"

    return AnswererBinding.create(FrozenAnswerer(adapter), config_sha256=config_hash)


def test_evaluate_run_passes_the_exact_environment_mask_and_keeps_prompt_rationale_out_of_evidence() -> None:
    environment = _environment()
    prompts: list[str] = []
    binding = _binding(prompts)
    shared = build_shared_identity(
        environment=environment,
        answerer=binding,
        cache_graph_sha256=_digest("6"),
        max_visual_frames=64,
        max_evidence_tokens=16,
        max_total_cost=100,
    )

    def controller(state: RouterState, legal: tuple[ActionInstance, ...]) -> PromptControllerDecision:
        assert legal is environment.last_valid_actions
        selected = next(action for action in legal if action.action_type in ({ActionType.SEARCH_GIST} if not state.candidate_event_ids else {ActionType.STOP}))
        return PromptControllerDecision(
            action=selected,
            rationale="SECRET CONTROLLER RATIONALE",
            cost=ControllerCost(total_cost=0.25, text_tokens=3),
        )

    run = evaluate_run(
        run_id="prompt-seed-1",
        policy_name="prompt-vlm",
        policy_family="adaptive",
        policy_identity_sha256=_digest("8"),
        policy=PromptControllerPolicy(controller),
        questions=(_question(environment),),
        benchmark=_benchmark(),
        answerer=binding,
        shared=shared,
        seed=1,
        cost_preference=0.25,
    )

    record = run.records[0]
    assert record.controller_usage.total_cost == pytest.approx(0.5)
    assert record.controller_usage.text_tokens == 6
    assert record.acquisition_usage.text_tokens == 2
    assert "SECRET CONTROLLER RATIONALE" not in prompts[0]
    assert "SECRET CONTROLLER RATIONALE" not in record.model_dump_json()
    assert run.summary.accuracy == 1.0


def test_evaluate_run_rejects_illegal_or_copied_actions_and_identity_mismatches() -> None:
    environment = _environment()
    prompts: list[str] = []
    binding = _binding(prompts)
    shared = build_shared_identity(
        environment=environment,
        answerer=binding,
        cache_graph_sha256=_digest("6"),
        max_visual_frames=64,
        max_evidence_tokens=16,
        max_total_cost=100,
    )
    common = dict(
        run_id="malicious",
        policy_name="malicious",
        policy_family="adaptive",
        policy_identity_sha256=_digest("8"),
        questions=(_question(environment),),
        benchmark=_benchmark(),
        answerer=binding,
        shared=shared,
        seed=1,
        cost_preference=0.25,
    )

    with pytest.raises(EvaluationIntegrityError, match="exact ActionInstance"):
        evaluate_run(policy=lambda _state, legal: legal[0].model_copy(), **common)

    wrong_binding = _binding([], config_hash=_digest("9"))
    with pytest.raises(EvaluationIntegrityError, match="Answerer identity"):
        evaluate_run(policy=GistOnlyPolicy(), answerer=wrong_binding, **{key: value for key, value in common.items() if key != "answerer"})

    with pytest.raises(EvaluationIntegrityError, match="cache graph identity"):
        evaluate_run(policy=GistOnlyPolicy(), questions=(_question(environment, cache_hash=_digest("a")),), **{key: value for key, value in common.items() if key != "questions"})

    changed_environment = _environment(residual_cost=3.0)
    with pytest.raises(EvaluationIntegrityError, match="environment identity"):
        evaluate_run(policy=GistOnlyPolicy(), questions=(_question(changed_environment),), **{key: value for key, value in common.items() if key != "questions"})


def test_budget_overrun_is_invalid_but_preserves_measured_costs() -> None:
    environment = _environment(visual_frames=32)
    prompts: list[str] = []
    binding = _binding(prompts)
    shared = build_shared_identity(
        environment=environment,
        answerer=binding,
        cache_graph_sha256=_digest("6"),
        max_visual_frames=10,
        max_evidence_tokens=100,
        max_total_cost=100,
    )
    run = evaluate_run(
        run_id="over-budget",
        policy_name="gist-visual",
        policy_family="fixed",
        policy_identity_sha256=_digest("8"),
        policy=GistVisualPolicy(),
        questions=(_question(environment),),
        benchmark=_benchmark(),
        answerer=binding,
        shared=shared,
        seed=1,
        cost_preference=0.25,
    )
    record = run.records[0]
    assert record.invalid_reason == "visual_frame_budget_exceeded"
    assert record.acquisition_usage.input_frames == 64
    with pytest.raises(ValueError, match="no valid questions"):
        _ = run.summary


def test_main_experiment_config_lists_all_families_sweeps_seeds_and_hardware() -> None:
    config = OmegaConf.to_container(
        OmegaConf.load(Path("configs/experiment/main_eval.yaml")), resolve=True
    )
    assert set(config["policies"]) == {
        "uniform",
        "gist_only",
        "gist_residual",
        "gist_visual",
        "full_residual",
        "rule",
        "prompt_vlm",
        "text_adaptive",
        "question_only",
        "bc",
        "bc_dagger",
    }
    assert len(config["seeds"]) == 3
    assert len(config["budget_sweep"]["total_cost"]) >= 3
    assert len(config["cost_preferences"]) >= 3
    assert config["hardware"]["training"] == "A800"
    assert config["hardware"]["evaluation"] == "V100"
    assert set(config["shared_identities"]) >= {
        "answerer_template_sha256",
        "answerer_config_sha256",
        "cache_graph_sha256",
        "cost_table_sha256",
    }
