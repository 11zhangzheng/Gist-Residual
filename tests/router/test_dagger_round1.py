from __future__ import annotations

import hashlib

import pytest

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
    CacheArtifactIdentity,
    CachedAnswerEvaluator,
    CachedUtilityGraph,
    ForbiddenObservationGenerator,
    MissingCachedEvaluationError,
    MissingCachedObservationError,
    _oracle_cost,
    _rollout,
    budget_bin,
    evaluation_key,
    label_best_next_action,
)
from fidmem.router.dataset import TestByteTokenizer
from fidmem.router.model import EncoderIdentity, MemoryRouter, RouterModelConfig
from fidmem.types import (
    ActionInstance,
    ActionType,
    EventRecord,
    EvidenceItem,
    FidelityLevel,
    RouterState,
)


def _action(kind: ActionType) -> ActionInstance:
    return ActionInstance(kind, None, None)


def _state(*, preference: float = 0.0, budget: float = 3.0) -> RouterState:
    return RouterState(
        question="What color?",
        options=("blue", "red"),
        evidence=(),
        action_history=(),
        remaining_budget=budget,
        candidate_event_ids=(),
        candidate_fidelity_levels={},
        context_frontiers={},
        cost_preference=preference,
    )


def _identity(label: str) -> CacheArtifactIdentity:
    return CacheArtifactIdentity(
        artifact_sha256=hashlib.sha256(label.encode("utf-8")).hexdigest()
    )


def _environment(calls: list[ActionInstance]) -> MemoryEnvironment:
    class RecordingForbidden(ForbiddenObservationGenerator):
        def __call__(
            self, action: ActionInstance, state: RouterState
        ) -> ActionObservation:
            calls.append(action)
            return super().__call__(action, state)

    event = EventRecord(
        video_id="v",
        event_id="e1",
        start_sec=0,
        end_sec=2,
        gist_text="bottle",
        visual_embedding=(1.0,),
        text_embedding=(1.0,),
        raw_video_uri="v.mp4",
        memory_version="v1",
    )
    return MemoryEnvironment(
        events=(event,),
        executor=RecordingForbidden(),
        costs=ActionCostTable(search_gist=3),
    )


def _cached_graph(
    environment: MemoryEnvironment,
    initial: RouterState,
    *,
    include_observation: bool = True,
    include_initial_evaluation: bool = True,
    include_searched_evaluation: bool = True,
    initial_score: float = 0.2,
) -> tuple[CachedUtilityGraph, RouterState]:
    search = _action(ActionType.SEARCH_GIST)
    observation = ActionObservation(
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
    searched = environment.replay(initial, search, observation).next_state
    observations = CachedObservationGraph(
        {observation_key(initial, search): observation} if include_observation else {}
    )
    evaluations = {}
    if include_initial_evaluation:
        evaluations[evaluation_key(initial)] = AnswerEvaluation(
            answer="red", answer_score=initial_score, correct=False
        )
    if include_searched_evaluation:
        evaluations[evaluation_key(searched)] = AnswerEvaluation(
            answer="blue", answer_score=1.0, correct=True
        )
    evaluator = CachedAnswerEvaluator(
        identity=_identity("answers"), evaluations=evaluations
    )
    return (
        CachedUtilityGraph(
            identity=_identity("utility"),
            observation_identity=_identity("observations"),
            observations=observations,
            evaluator=evaluator,
        ),
        searched,
    )


def test_pending_observation_fails_closed_even_when_stop_path_exists() -> None:
    calls: list[ActionInstance] = []
    environment = _environment(calls)
    initial = _state()
    graph, _ = _cached_graph(environment, initial, include_observation=False)

    with pytest.raises(MissingCachedObservationError, match="pending"):
        label_best_next_action(
            initial,
            environment=environment,
            utility_graph=graph,
            normalization=CostNormalization(
                constant=10, sample_count=1, source_split="train"
            ),
        )
    with pytest.raises(MissingCachedObservationError, match="pending"):
        _oracle_cost(
            initial,
            environment=environment,
            utility_graph=graph,
            normalization=CostNormalization(
                constant=10, sample_count=1, source_split="train"
            ),
        )
    assert calls == []


def test_missing_cached_evaluation_fails_closed_without_answerer_io() -> None:
    calls: list[ActionInstance] = []
    environment = _environment(calls)
    initial = _state()
    graph, _ = _cached_graph(environment, initial, include_searched_evaluation=False)

    with pytest.raises(MissingCachedEvaluationError, match="evaluation"):
        label_best_next_action(
            initial,
            environment=environment,
            utility_graph=graph,
            normalization=CostNormalization(
                constant=10, sample_count=1, source_split="train"
            ),
        )
    assert calls == []


def test_cost_preference_selects_utility_optimal_stop_not_correct_path() -> None:
    calls: list[ActionInstance] = []
    environment = _environment(calls)
    initial = _state(preference=1.0)
    graph, _ = _cached_graph(environment, initial, initial_score=0.9)

    selected = label_best_next_action(
        initial,
        environment=environment,
        utility_graph=graph,
        normalization=CostNormalization(
            constant=10, sample_count=1, source_split="train"
        ),
    )

    # STOP utility is 0.9; the correct SEARCH path utility is 1 - 3/10 = 0.7.
    assert selected == _action(ActionType.STOP)
    assert calls == []


def test_non_stop_rollout_reads_cache_and_replays_with_forbidden_executor() -> None:
    calls: list[ActionInstance] = []
    environment = _environment(calls)
    initial = _state()
    graph, searched = _cached_graph(environment, initial)

    def search_then_stop(
        state: RouterState, legal: tuple[ActionInstance, ...]
    ) -> ActionInstance:
        action = (
            _action(ActionType.SEARCH_GIST)
            if not state.action_history
            else _action(ActionType.STOP)
        )
        assert action in legal
        return action

    with pytest.raises(AssertionError, match="forbidden"):
        environment.step(initial, _action(ActionType.SEARCH_GIST))
    assert calls == [_action(ActionType.SEARCH_GIST)]
    calls.clear()

    transitions = _rollout(
        initial,
        policy=search_then_stop,
        environment=environment,
        utility_graph=graph,
    )

    assert transitions[0].next_state == searched
    assert transitions[-1].terminal is True
    assert calls == []


def _tiny_model(name: str = "offline-test") -> MemoryRouter:
    identity = EncoderIdentity.test_identity(name)
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


def test_bc_policy_rejects_tokenizer_identity_mismatch() -> None:
    with pytest.raises(ValueError, match="tokenizer identity"):
        BCPolicy(_tiny_model("model-a"), tokenizer=TestByteTokenizer("model-b"))


def test_budget_bin_treats_exact_and_near_upper_boundary_as_same_bin() -> None:
    assert budget_bin(20.0, width=1.0) == budget_bin(19.999, width=1.0)
