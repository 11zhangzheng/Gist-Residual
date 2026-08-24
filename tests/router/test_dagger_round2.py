from __future__ import annotations

import pytest

from fidmem.oracle.search import CachedObservationGraph
from fidmem.router.dagger import (
    CachedAnswerEvaluator,
    CachedUtilityGraph,
    MissingCachedObservationError,
    MissingCachedEvaluationError,
    collect_deviations,
)

from tests.router.test_dagger import (
    _NORMALIZATION,
    _always_stop,
    _environment,
    _identity,
    _state,
    _utility_graph,
)


def test_seen_is_committed_only_after_successful_label_and_retry_executes() -> None:
    environment = _environment()
    initial = _state()
    empty = CachedUtilityGraph(
        observations=CachedObservationGraph({}),
        evaluator=CachedAnswerEvaluator(
            evaluator_identity=_identity("empty evaluator"),
            evaluations={},
        ),
    )
    seen: set[str] = set()

    with pytest.raises((MissingCachedObservationError, MissingCachedEvaluationError)):
        collect_deviations(
            (initial,),
            policy=_always_stop,
            environment=environment,
            utility_graph=empty,
            normalization=_NORMALIZATION,
            question_ids=("q",),
            initial_replay_transitions=((),),
            seen_keys=seen,
            budget_bin_width=1.0,
        )

    assert seen == set()
    retry = collect_deviations(
        (initial,),
        policy=_always_stop,
        environment=environment,
        utility_graph=_utility_graph(environment, initial),
        normalization=_NORMALIZATION,
        question_ids=("q",),
        initial_replay_transitions=((),),
        seen_keys=seen,
        budget_bin_width=1.0,
    )
    assert len(retry) == 1
    assert seen == {retry[0].state_key}


def test_observation_graph_content_identity_changes_with_actual_contents() -> None:
    empty = CachedObservationGraph({})
    environment = _environment()
    initial = _state()
    populated = _utility_graph(environment, initial)

    assert len(empty.content_sha256) == 64
    assert empty.content_sha256 != populated.observation_content_sha256
