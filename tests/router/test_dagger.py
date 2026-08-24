from __future__ import annotations

from collections.abc import Callable

from fidmem.actions.environment import (
    ActionCostTable,
    ActionObservation,
    MemoryEnvironment,
    OperationMetadata,
)
from fidmem.oracle.labels import CostNormalization
from fidmem.oracle.search import (
    AnswerEvaluation,
    CachedObservationGraph,
    observation_key,
)
from fidmem.router.dagger import (
    BCPolicy,
    DaggerRoundResult,
    ForbiddenObservationGenerator,
    _should_continue,
    collect_deviations,
    label_best_next_action,
    run_dagger_round,
)
from fidmem.router.model import EncoderIdentity, MemoryRouter, RouterModelConfig
from fidmem.types import (
    ActionInstance,
    ActionType,
    EventRecord,
    EvidenceItem,
    FidelityLevel,
    RouterState,
)

_NORMALIZATION = CostNormalization(constant=10, sample_count=100, source_split="train")


def _action(
    kind: ActionType, event_id: str | None = None, budget: str | None = None
) -> ActionInstance:
    return ActionInstance(kind, event_id, budget)  # type: ignore[arg-type]


def _state() -> RouterState:
    return RouterState(
        question="What color?",
        options=("blue", "red"),
        evidence=(),
        action_history=(),
        remaining_budget=20,
        candidate_event_ids=(),
        candidate_fidelity_levels={},
        context_frontiers={},
        cost_preference=0,
    )


def _environment(executor_calls: list[ActionInstance]) -> MemoryEnvironment:
    def forbidden_executor(
        action: ActionInstance, state: RouterState
    ) -> ActionObservation:
        executor_calls.append(action)
        raise AssertionError("DAgger must not execute providers")

    event = EventRecord(
        video_id="v",
        event_id="e1",
        start_sec=0,
        end_sec=2,
        gist_text="a bottle",
        visual_embedding=(1.0,),
        text_embedding=(1.0,),
        raw_video_uri="v.mp4",
        memory_version="v1",
    )
    return MemoryEnvironment(
        events=(event,),
        executor=forbidden_executor,
        costs=ActionCostTable(
            search_gist=1,
            residual=2,
            context=1,
            visual_low=5,
            visual_high=8,
            visual_low_question=0,
            visual_high_question=0,
        ),
    )


def _toy_graph(
    environment: MemoryEnvironment, initial: RouterState
) -> CachedObservationGraph:
    search = _action(ActionType.SEARCH_GIST)
    search_observation = ActionObservation(
        action_type=ActionType.SEARCH_GIST,
        candidate_event_ids=("e1",),
        evidence=(
            EvidenceItem(
                event_id="e1",
                fidelity_level=FidelityLevel.GIST,
                content="bottle",
                score=1,
            ),
        ),
        operation_metadata=(
            OperationMetadata(
                scope="search_gist", cache_status="miss", amortizable=True
            ),
        ),
    )
    searched = environment.replay(initial, search, search_observation).next_state
    residual = _action(ActionType.EXPAND_RESIDUAL, "e1")
    residual_observation = ActionObservation(
        action_type=ActionType.EXPAND_RESIDUAL,
        target_event_id="e1",
        evidence=(
            EvidenceItem(
                event_id="e1",
                fidelity_level=FidelityLevel.RESIDUAL,
                content="blue",
                score=1,
            ),
        ),
        operation_metadata=(
            OperationMetadata(scope="residual", cache_status="miss", amortizable=True),
        ),
    )
    visual = _action(ActionType.VERIFY_VISUAL, "e1", "low")
    visual_observation = ActionObservation(
        action_type=ActionType.VERIFY_VISUAL,
        target_event_id="e1",
        evidence=(
            EvidenceItem(
                event_id="e1",
                fidelity_level=FidelityLevel.VISUAL,
                content="blue",
                score=1,
                attachments=("frame.jpg",),
            ),
        ),
        operation_metadata=(
            OperationMetadata(
                scope="event_observation",
                cache_status="miss",
                amortizable=True,
                input_frames=12,
            ),
            OperationMetadata(
                scope="question_verification", cache_status="miss", amortizable=False
            ),
        ),
    )
    return CachedObservationGraph(
        {
            observation_key(initial, search): search_observation,
            observation_key(searched, residual): residual_observation,
            observation_key(searched, visual): visual_observation,
        }
    )


def _evaluate(state: RouterState) -> AnswerEvaluation:
    correct = any(
        item.content == "blue" and item.fidelity_level is not FidelityLevel.GIST
        for item in state.evidence
    )
    return AnswerEvaluation(
        answer="blue" if correct else "red",
        answer_score=1.0 if correct else 0.2,
        correct=correct,
    )


def _always_stop() -> Callable[[RouterState, tuple[ActionInstance, ...]], ActionInstance]:
    def policy(
        state: RouterState, legal_actions: tuple[ActionInstance, ...]
    ) -> ActionInstance:
        stop = _action(ActionType.STOP)
        assert stop in legal_actions
        return stop

    return policy


def _tiny_model() -> MemoryRouter:
    identity = EncoderIdentity.test_identity("offline-test")
    return MemoryRouter(
        RouterModelConfig(
            encoder=identity,
            encoder_output_dim=8,
            hidden_dim=12,
            action_type_embedding_dim=4,
            fidelity_embedding_dim=2,
            max_question_tokens=128,
            max_item_tokens=64,
        )
    )


def test_label_best_next_action_selects_cheapest_correct_step_without_provider_io() -> (
    None
):
    calls: list[ActionInstance] = []
    environment = _environment(calls)
    initial = _state()
    graph = _toy_graph(environment, initial)

    best = label_best_next_action(
        initial,
        environment=environment,
        graph=graph,
        evaluator=_evaluate,
        normalization=_NORMALIZATION,
    )

    assert best == _action(ActionType.SEARCH_GIST)
    assert calls == []


def test_collect_deviations_flags_policy_departure_without_provider_io() -> None:
    calls: list[ActionInstance] = []
    environment = _environment(calls)
    initial = _state()
    graph = _toy_graph(environment, initial)

    deviations = collect_deviations(
        (initial,),
        policy=_always_stop(),
        environment=environment,
        graph=graph,
        evaluator=_evaluate,
        normalization=_NORMALIZATION,
    )

    assert len(deviations) == 1
    assert deviations[0].policy_action == _action(ActionType.STOP)
    assert deviations[0].oracle_action == _action(ActionType.SEARCH_GIST)
    assert calls == []


def test_collect_deviations_deduplicates_identical_states() -> None:
    calls: list[ActionInstance] = []
    environment = _environment(calls)
    initial = _state()
    graph = _toy_graph(environment, initial)

    deviations = collect_deviations(
        (initial, initial),
        policy=_always_stop(),
        environment=environment,
        graph=graph,
        evaluator=_evaluate,
        normalization=_NORMALIZATION,
    )

    assert len(deviations) == 1


def test_forbidden_observation_generator_fails_closed() -> None:
    forbidden = ForbiddenObservationGenerator()
    try:
        forbidden(_action(ActionType.SEARCH_GIST), _state())
    except AssertionError:
        return
    raise AssertionError("ForbiddenObservationGenerator did not raise")


def test_round_two_stops_without_utility_or_regret_improvement() -> None:
    previous = DaggerRoundResult(
        round_number=1,
        deviations=(),
        dev_utility=0.5,
        cost_regret=0.3,
        should_continue=True,
    )

    # utility gain 0.002 < 0.005 and regret improvement 0.005 < 2% * 0.3
    assert _should_continue(2, 0.502, 0.295, previous) is False
    # enough utility gain
    assert _should_continue(2, 0.510, 0.295, previous) is True
    # enough relative regret improvement
    assert _should_continue(2, 0.502, 0.290, previous) is True
    # round one always continues; round three never continues
    assert _should_continue(1, 0.5, 0.3, None) is True
    assert _should_continue(3, 0.6, 0.1, previous) is False


def test_run_dagger_round_round_one_always_continues() -> None:
    calls: list[ActionInstance] = []
    environment = _environment(calls)
    initial = _state()
    graph = _toy_graph(environment, initial)

    result = run_dagger_round(
        round_number=1,
        train_states=(initial,),
        dev_states=(initial,),
        policy=_always_stop(),
        environment=environment,
        graph=graph,
        evaluator=_evaluate,
        normalization=_NORMALIZATION,
    )

    assert result.round_number == 1
    assert len(result.deviations) == 1
    assert result.should_continue is True
    assert calls == []


def test_bc_policy_returns_a_legal_action() -> None:
    calls: list[ActionInstance] = []
    environment = _environment(calls)
    initial = _state()
    policy = BCPolicy(_tiny_model())

    legal = environment.valid_actions(initial)
    selected = policy(initial, legal)

    assert selected in legal
