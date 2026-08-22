"""Deterministic Oracle search over already-cached atomic observations.

Search is deliberately unable to call a memory provider.  It can only ask a
``MemoryEnvironment`` for legal actions and replay observations supplied by a
``CachedObservationGraph``.  Missing observations are returned to the caller
as a generation plan for a separate, resumable pipeline.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, Field

from fidmem.actions.environment import (
    ActionObservation,
    EnvironmentTransition,
    MemoryEnvironment,
)
from fidmem.types import ActionInstance, ActionType, RouterState


class AnswerEvaluation(BaseModel):
    """Frozen Answerer/Judge result used to score one STOP decision."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    answer: str
    answer_score: float = Field(ge=0, le=1)
    correct: bool


AnswerEvaluator = Callable[[RouterState], AnswerEvaluation]
ActionSignature = tuple[str, ...]
PriorityKey = tuple[float, float, int, ActionSignature]


def action_signature(action: ActionInstance) -> str:
    """Return the stable signature used for deterministic path tie-breaking."""

    return "|".join(
        (
            action.action_type.value,
            action.event_id if action.event_id is not None else "-",
            action.visual_budget if action.visual_budget is not None else "-",
        )
    )


def _cache_state_sha256(state: RouterState) -> str:
    # Budget and preference affect legal routing, not the content of an atomic
    # observation. Excluding them makes one cached graph reusable at all four
    # preregistered preferences.
    payload = state.model_dump(
        mode="json",
        exclude={"remaining_budget", "cost_preference"},
    )
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ObservationKey(BaseModel):
    """Content key for an atomic observation at a cached search state."""

    model_config = ConfigDict(frozen=True)

    state_sha256: str
    action_signature: str


def observation_key(state: RouterState, action: ActionInstance) -> ObservationKey:
    return ObservationKey(
        state_sha256=_cache_state_sha256(state),
        action_signature=action_signature(action),
    )


class CachedObservationGraph:
    """Read-only mapping used by Oracle search; it has no provider fallback."""

    def __init__(
        self,
        observations: Mapping[ObservationKey | tuple[str, str], ActionObservation],
    ) -> None:
        normalized: dict[ObservationKey, ActionObservation] = {}
        for raw_key, observation in observations.items():
            key = (
                raw_key
                if isinstance(raw_key, ObservationKey)
                else ObservationKey(
                    state_sha256=raw_key[0], action_signature=raw_key[1]
                )
            )
            if key in normalized:
                raise ValueError(f"duplicate cached observation: {key}")
            normalized[key] = ActionObservation.model_validate(observation)
        self._observations = MappingProxyType(normalized)

    def get(
        self, state: RouterState, action: ActionInstance
    ) -> ActionObservation | None:
        return self._observations.get(observation_key(state, action))

    def __len__(self) -> int:
        return len(self._observations)


class PendingObservation(BaseModel):
    """One missing cache atom that an external generation job may materialize."""

    model_config = ConfigDict(frozen=True)

    key: ObservationKey
    state: RouterState
    action: ActionInstance


class OraclePath(BaseModel):
    """A STOP-terminated path and its frozen Answerer/Judge result."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    transitions: tuple[EnvironmentTransition, ...]
    answer: str
    answer_score: float = Field(ge=0, le=1)
    correct: bool
    total_cost: float = Field(ge=0)
    utility: float

    @property
    def depth(self) -> int:
        return len(self.transitions)

    @property
    def action_signature(self) -> ActionSignature:
        return tuple(action_signature(item.action) for item in self.transitions)

    @property
    def priority_key(self) -> PriorityKey:
        # This tuple is the preregistered search ordering.  Do not add an
        # implicit insertion-order or object-identity tie-breaker.
        return (-self.utility, self.total_cost, self.depth, self.action_signature)


class SearchResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    paths: tuple[OraclePath, ...]
    pending: tuple[PendingObservation, ...]


@dataclass(frozen=True)
class _Node:
    state: RouterState
    transitions: tuple[EnvironmentTransition, ...]
    evaluation: AnswerEvaluation
    total_cost: float
    utility: float

    @property
    def depth(self) -> int:
        return len(self.transitions)

    @property
    def action_signature(self) -> ActionSignature:
        return tuple(action_signature(item.action) for item in self.transitions)

    @property
    def priority_key(self) -> PriorityKey:
        return (-self.utility, self.total_cost, self.depth, self.action_signature)


def _validate_search_inputs(
    *,
    beam_size: int | None,
    max_depth: int,
    cost_preference: float,
    cost_normalizer: float,
) -> None:
    if beam_size is not None and beam_size < 1:
        raise ValueError("beam_size must be at least one")
    if max_depth < 1:
        raise ValueError("max_depth must be at least one")
    if not math.isfinite(cost_preference) or not 0 <= cost_preference <= 1:
        raise ValueError("cost_preference must be finite and between zero and one")
    if not math.isfinite(cost_normalizer) or cost_normalizer <= 0:
        raise ValueError("cost_normalizer must be finite and positive")


def _utility(score: float, cost: float, preference: float, normalizer: float) -> float:
    return score - preference * (cost / normalizer)


def _path(node: _Node) -> OraclePath:
    return OraclePath(
        transitions=node.transitions,
        answer=node.evaluation.answer,
        answer_score=node.evaluation.answer_score,
        correct=node.evaluation.correct,
        total_cost=node.total_cost,
        utility=node.utility,
    )


def _sorted_pending(
    pending: Mapping[ObservationKey, PendingObservation],
) -> tuple[PendingObservation, ...]:
    return tuple(
        pending[key]
        for key in sorted(
            pending,
            key=lambda item: (item.state_sha256, item.action_signature),
        )
    )


def _search(
    environment: MemoryEnvironment,
    initial_state: RouterState,
    graph: CachedObservationGraph,
    evaluator: AnswerEvaluator,
    *,
    beam_size: int | None,
    max_depth: int,
    cost_preference: float | None,
    cost_normalizer: float,
) -> SearchResult:
    preference = (
        initial_state.cost_preference if cost_preference is None else cost_preference
    )
    _validate_search_inputs(
        beam_size=beam_size,
        max_depth=max_depth,
        cost_preference=preference,
        cost_normalizer=cost_normalizer,
    )
    initial_evaluation = evaluator(initial_state)
    initial = _Node(
        state=initial_state,
        transitions=(),
        evaluation=initial_evaluation,
        total_cost=0.0,
        utility=_utility(
            initial_evaluation.answer_score, 0.0, preference, cost_normalizer
        ),
    )
    frontier = [initial]
    completed: list[OraclePath] = []
    pending: dict[ObservationKey, PendingObservation] = {}

    for _ in range(max_depth):
        if not frontier:
            break
        ordered = sorted(frontier, key=lambda node: node.priority_key)
        parents = ordered if beam_size is None else ordered[:beam_size]
        children: list[_Node] = []
        for node in parents:
            for action in environment.valid_actions(node.state):
                if action.action_type is ActionType.STOP:
                    stop_observation = ActionObservation(
                        action_type=ActionType.STOP, target_event_id=None
                    )
                    transition = environment.replay(
                        node.state, action, stop_observation
                    )
                    terminal = _Node(
                        state=transition.next_state,
                        transitions=node.transitions + (transition,),
                        evaluation=node.evaluation,
                        total_cost=node.total_cost,
                        utility=node.utility,
                    )
                    completed.append(_path(terminal))
                    continue

                observation = graph.get(node.state, action)
                if observation is None:
                    key = observation_key(node.state, action)
                    pending.setdefault(
                        key,
                        PendingObservation(key=key, state=node.state, action=action),
                    )
                    continue
                transition = environment.replay(node.state, action, observation)
                total_cost = node.total_cost + transition.step_cost
                evaluation = evaluator(transition.next_state)
                children.append(
                    _Node(
                        state=transition.next_state,
                        transitions=node.transitions + (transition,),
                        evaluation=evaluation,
                        total_cost=total_cost,
                        utility=_utility(
                            evaluation.answer_score,
                            total_cost,
                            preference,
                            cost_normalizer,
                        ),
                    )
                )

        # heapq makes the exact priority tuple explicit and avoids relying on
        # incidental input mapping order. Signatures uniquely identify paths.
        heap: list[tuple[PriorityKey, _Node]] = []
        seen_signatures: set[ActionSignature] = set()
        for child in children:
            if child.action_signature in seen_signatures:
                continue
            seen_signatures.add(child.action_signature)
            heapq.heappush(heap, (child.priority_key, child))
        width = len(heap) if beam_size is None else min(beam_size, len(heap))
        frontier = [heapq.heappop(heap)[1] for _ in range(width)]

    return SearchResult(
        paths=tuple(sorted(completed, key=lambda path: path.priority_key)),
        pending=_sorted_pending(pending),
    )


def beam_search(
    environment: MemoryEnvironment,
    initial_state: RouterState,
    graph: CachedObservationGraph,
    evaluator: AnswerEvaluator,
    *,
    beam_size: int = 8,
    max_depth: int = 5,
    cost_preference: float | None = None,
    cost_normalizer: float = 1.0,
) -> SearchResult:
    """Search the cached graph with the preregistered deterministic beam."""

    return _search(
        environment,
        initial_state,
        graph,
        evaluator,
        beam_size=beam_size,
        max_depth=max_depth,
        cost_preference=cost_preference,
        cost_normalizer=cost_normalizer,
    )


def exhaustive_search(
    environment: MemoryEnvironment,
    initial_state: RouterState,
    graph: CachedObservationGraph,
    evaluator: AnswerEvaluator,
    *,
    max_depth: int = 5,
    cost_preference: float | None = None,
    cost_normalizer: float = 1.0,
) -> SearchResult:
    """Enumerate every cached legal path up to ``max_depth`` for audit."""

    return _search(
        environment,
        initial_state,
        graph,
        evaluator,
        beam_size=None,
        max_depth=max_depth,
        cost_preference=cost_preference,
        cost_normalizer=cost_normalizer,
    )


def canonical_oracle(paths: Sequence[OraclePath]) -> OraclePath:
    """Select minimum-cost correct path, or best-score/cheapest fallback."""

    if not paths:
        raise ValueError("canonical Oracle requires at least one STOP path")
    correct = tuple(path for path in paths if path.correct)
    if correct:
        return min(
            correct,
            key=lambda path: (
                path.total_cost,
                -path.answer_score,
                path.depth,
                path.action_signature,
            ),
        )
    return min(
        paths,
        key=lambda path: (
            -path.answer_score,
            path.total_cost,
            path.depth,
            path.action_signature,
        ),
    )
