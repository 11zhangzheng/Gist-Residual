from __future__ import annotations

from collections.abc import Callable

from fidmem.actions.environment import (
    ActionCostTable,
    ActionObservation,
    MemoryEnvironment,
    OperationMetadata,
)
from fidmem.oracle.search import (
    AnswerEvaluation,
    CachedObservationGraph,
    beam_search,
    canonical_oracle,
    exhaustive_search,
    observation_key,
)
from fidmem.types import (
    ActionInstance,
    ActionType,
    EventRecord,
    EvidenceItem,
    FidelityLevel,
    RouterState,
)


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
        raise AssertionError("Oracle search must not execute providers")

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


def test_toy_graph_beam_and_exhaustive_find_the_lowest_cost_correct_path_without_provider_io() -> (
    None
):
    calls: list[ActionInstance] = []
    environment = _environment(calls)
    initial = _state()
    graph = _toy_graph(environment, initial)

    beam = beam_search(environment, initial, graph, _evaluate, beam_size=8, max_depth=3)
    exhaustive = exhaustive_search(environment, initial, graph, _evaluate, max_depth=3)
    beam_path = canonical_oracle(beam.paths)
    exact_path = canonical_oracle(exhaustive.paths)

    assert beam_path.total_cost == exact_path.total_cost == 3
    assert tuple(
        transition.action.action_type for transition in beam_path.transitions
    ) == (
        ActionType.SEARCH_GIST,
        ActionType.EXPAND_RESIDUAL,
        ActionType.STOP,
    )
    assert beam_path.action_signature == exact_path.action_signature
    assert calls == []


def test_search_returns_deterministic_pending_atomic_observations_and_never_calls_executor() -> (
    None
):
    calls: list[ActionInstance] = []
    environment = _environment(calls)
    initial = _state()
    search = _action(ActionType.SEARCH_GIST)
    empty_graph = CachedObservationGraph({})

    first = beam_search(environment, initial, empty_graph, _evaluate)
    second = beam_search(environment, initial, empty_graph, _evaluate)

    assert first.pending == second.pending
    assert len(first.pending) == 1
    assert first.pending[0].key == observation_key(initial, search)
    assert first.pending[0].action == search
    assert calls == []


def test_priority_key_is_exactly_negative_utility_cost_depth_and_action_signature() -> (
    None
):
    calls: list[ActionInstance] = []
    environment = _environment(calls)
    initial = _state()
    result = exhaustive_search(
        environment, initial, _toy_graph(environment, initial), _evaluate, max_depth=3
    )
    path = canonical_oracle(result.paths)

    assert path.priority_key == (
        -path.utility,
        path.total_cost,
        path.depth,
        path.action_signature,
    )
