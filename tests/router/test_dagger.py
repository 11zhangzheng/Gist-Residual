from __future__ import annotations

import hashlib

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
    DaggerRoundResult,
    ForbiddenObservationGenerator,
    _should_continue,
    collect_deviations,
    evaluation_key,
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

_NORMALIZATION = CostNormalization(constant=10, sample_count=1, source_split="train")


def _action(kind: ActionType) -> ActionInstance:
    return ActionInstance(kind, None, None)


def _state() -> RouterState:
    return RouterState(
        question="What color?",
        options=("blue", "red"),
        evidence=(),
        action_history=(),
        remaining_budget=3,
        candidate_event_ids=(),
        candidate_fidelity_levels={},
        context_frontiers={},
        cost_preference=0,
    )


def _identity(label: str) -> CacheArtifactIdentity:
    return CacheArtifactIdentity(
        artifact_sha256=hashlib.sha256(label.encode()).hexdigest()
    )


def _environment() -> MemoryEnvironment:
    return MemoryEnvironment(
        events=(
            EventRecord(
                video_id="v",
                event_id="e1",
                start_sec=0,
                end_sec=2,
                gist_text="bottle",
                visual_embedding=(1.0,),
                text_embedding=(1.0,),
                raw_video_uri="v.mp4",
                memory_version="v1",
            ),
        ),
        executor=ForbiddenObservationGenerator(),
        costs=ActionCostTable(search_gist=3),
    )


def _utility_graph(
    environment: MemoryEnvironment, initial: RouterState
) -> CachedUtilityGraph:
    search = _action(ActionType.SEARCH_GIST)
    observation = ActionObservation(
        action_type=ActionType.SEARCH_GIST,
        candidate_event_ids=("e1",),
        evidence=(
            EvidenceItem(
                event_id="e1",
                fidelity_level=FidelityLevel.GIST,
                content="blue",
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
    return CachedUtilityGraph(
        identity=_identity("utility"),
        observation_identity=_identity("observations"),
        observations=CachedObservationGraph(
            {observation_key(initial, search): observation}
        ),
        evaluator=CachedAnswerEvaluator(
            identity=_identity("evaluations"),
            evaluations={
                evaluation_key(initial): AnswerEvaluation(
                    answer="red", answer_score=0.2, correct=False
                ),
                evaluation_key(searched): AnswerEvaluation(
                    answer="blue", answer_score=1.0, correct=True
                ),
            },
        ),
    )


def _always_stop(
    state: RouterState, legal: tuple[ActionInstance, ...]
) -> ActionInstance:
    stop = _action(ActionType.STOP)
    assert stop in legal
    return stop


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


def test_seen_keys_persist_across_collection_calls() -> None:
    environment = _environment()
    initial = _state()
    graph = _utility_graph(environment, initial)
    seen: set[str] = set()

    first = collect_deviations(
        (initial,),
        policy=_always_stop,
        environment=environment,
        utility_graph=graph,
        normalization=_NORMALIZATION,
        question_ids=("q",),
        seen_keys=seen,
    )
    second = collect_deviations(
        (initial,),
        policy=_always_stop,
        environment=environment,
        utility_graph=graph,
        normalization=_NORMALIZATION,
        question_ids=("q",),
        seen_keys=seen,
    )

    assert len(first) == 1
    assert second == ()
    assert seen == {first[0].state_key}


def test_round_two_stops_without_utility_or_regret_improvement() -> None:
    previous = DaggerRoundResult(
        round_number=1,
        deviations=(),
        dev_utility=0.5,
        cost_regret=0.3,
        should_continue=True,
    )

    assert _should_continue(2, 0.502, 0.295, previous) is False
    assert _should_continue(2, 0.510, 0.295, previous) is True
    assert _should_continue(2, 0.502, 0.290, previous) is True
    assert _should_continue(1, 0.5, 0.3, None) is True
    assert _should_continue(3, 0.6, 0.1, previous) is False


def test_run_dagger_round_round_one_labels_and_continues() -> None:
    environment = _environment()
    initial = _state()
    graph = _utility_graph(environment, initial)

    result = run_dagger_round(
        round_number=1,
        train_states=(initial,),
        dev_states=(initial,),
        policy=_always_stop,
        environment=environment,
        utility_graph=graph,
        normalization=_NORMALIZATION,
        question_ids=("q",),
    )

    assert len(result.deviations) == 1
    assert result.deviations[0].oracle_action == _action(ActionType.SEARCH_GIST)
    assert result.should_continue is True


def test_bc_policy_returns_only_a_legal_action() -> None:
    environment = _environment()
    initial = _state()
    policy = BCPolicy(_tiny_model())

    legal = environment.valid_actions(initial)
    assert policy(initial, legal) in legal
