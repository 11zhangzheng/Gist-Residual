"""Cached-graph DAgger: relabel policy deviations without any provider/VLM I/O.

DAgger correction operates strictly on the already-cached atomic observation
graph.  Rollout reads observations from :class:`CachedObservationGraph` and
advances state with :meth:`MemoryEnvironment.replay` (a pure validator), never
through :meth:`MemoryEnvironment.step` (which invokes the provider executor).
A missing cached atom fails closed instead of falling back to model I/O.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Protocol

import torch
from pydantic import BaseModel, ConfigDict, Field

from fidmem.actions.environment import ActionObservation, EnvironmentTransition, MemoryEnvironment
from fidmem.agent.runner import RouterPolicy
from fidmem.oracle.labels import CostNormalization
from fidmem.oracle.search import (
    AnswerEvaluator,
    CachedObservationGraph,
    beam_search,
    canonical_oracle,
)
from fidmem.router.dataset import (
    _ACTION_INDEX,
    _FIDELITY_INDEX,
    _NONE_FIDELITY,
    _VISUAL_BUDGET_INDEX,
    RouterBatch,
    TestByteTokenizer,
    TextTokenizer,
    _action_text,
    _padded_tokens,
)
from fidmem.router.model import MemoryRouter
from fidmem.types import ActionInstance, ActionType, RouterState

_MAX_DEPTH = 5
_BEAM_SIZE = 8
_UTILITY_GAIN_THRESHOLD = 0.005
_REGRET_IMPROVEMENT_RATIO = 0.02


class MissingCachedObservationError(RuntimeError):
    """A DAgger rollout needed an atomic observation absent from the cache."""


class InvalidPolicyActionError(ValueError):
    """A policy returned an action outside the environment's legal tuple."""


class ObservationGenerator(Protocol):
    def __call__(self, action: ActionInstance, state: RouterState) -> ActionObservation: ...


class ForbiddenObservationGenerator:
    """Sentinel executor: any invocation proves a forbidden provider/VLM call."""

    def __call__(self, action: ActionInstance, state: RouterState) -> ActionObservation:
        raise AssertionError(
            "observation generation is forbidden during DAgger correction"
        )


def cached_observation_generator(
    graph: CachedObservationGraph,
) -> ObservationGenerator:
    """Return an environment executor that only reads the cached graph."""

    def generate(
        action: ActionInstance, state: RouterState
    ) -> ActionObservation:
        observation = graph.get(state, action)
        if observation is None:
            raise MissingCachedObservationError(
                f"cached observation missing for {action.action_type.value}"
            )
        return observation

    return generate


def encode_router_state(
    state: RouterState,
    action_instances: Sequence[ActionInstance],
    legal_action_mask: Sequence[bool],
    *,
    tokenizer: TextTokenizer,
    max_question_tokens: int,
    max_item_tokens: int,
) -> RouterBatch:
    """Encode one rollout state into a forward-only batch with dummy targets."""

    actions = tuple(action_instances)
    mask = tuple(legal_action_mask)
    if len(actions) != len(mask):
        raise ValueError("legal_action_mask must match action_instances")

    question_rows = [(state.question + "\n" + "\n".join(state.options),)]
    question_ids, question_mask, _ = _padded_tokens(
        question_rows,
        maximum=max_question_tokens,
        label="question",
        tokenizer=tokenizer,
    )
    evidence_rows = [[item.content for item in state.evidence]]
    evidence_ids, evidence_token_mask, evidence_item_mask = _padded_tokens(
        evidence_rows,
        maximum=max_item_tokens,
        label="evidence item",
        tokenizer=tokenizer,
    )
    history_rows = [[_action_text(action) for action in state.action_history]]
    history_ids, history_token_mask, history_item_mask = _padded_tokens(
        history_rows,
        maximum=max_item_tokens,
        label="history action",
        tokenizer=tokenizer,
    )
    action_rows = [[_action_text(action) for action in actions]]
    action_ids, action_token_mask, action_item_mask = _padded_tokens(
        action_rows,
        maximum=max_item_tokens,
        label="action instance",
        tokenizer=tokenizer,
    )

    batch_size, max_actions = action_item_mask.shape
    max_evidence = evidence_item_mask.shape[1]

    evidence_fidelity = torch.full(
        (batch_size, max_evidence), _NONE_FIDELITY, dtype=torch.long
    )
    evidence_numeric = torch.zeros(
        (batch_size, max_evidence, 2), dtype=torch.float32
    )
    history_action_type = torch.zeros_like(history_item_mask, dtype=torch.long)
    legal_mask = torch.zeros((batch_size, max_actions), dtype=torch.bool)
    action_type = torch.zeros((batch_size, max_actions), dtype=torch.long)
    action_fidelity = torch.full(
        (batch_size, max_actions), _NONE_FIDELITY, dtype=torch.long
    )
    action_visual_budget = torch.zeros((batch_size, max_actions), dtype=torch.long)
    action_frontier = torch.zeros((batch_size, max_actions, 2), dtype=torch.float32)
    affinity = torch.zeros(
        (batch_size, max_actions, max_evidence), dtype=torch.bool
    )
    state_numeric = torch.zeros((batch_size, 2), dtype=torch.float32)

    state_numeric[0] = torch.tensor(
        (state.remaining_budget, state.cost_preference)
    )
    for evidence_index, item in enumerate(state.evidence):
        evidence_fidelity[0, evidence_index] = _FIDELITY_INDEX[item.fidelity_level]
        evidence_numeric[0, evidence_index] = torch.tensor(
            (item.score, float(item.acquisition_step))
        )
    for history_index, action in enumerate(state.action_history):
        history_action_type[0, history_index] = _ACTION_INDEX[action.action_type]
    legal_mask[0, : len(mask)] = torch.tensor(mask)
    for action_index, action in enumerate(actions):
        action_type[0, action_index] = _ACTION_INDEX[action.action_type]
        action_visual_budget[0, action_index] = _VISUAL_BUDGET_INDEX[
            action.visual_budget
        ]
        if (
            action.event_id is not None
            and action.event_id in state.candidate_fidelity_levels
        ):
            action_fidelity[0, action_index] = _FIDELITY_INDEX[
                state.candidate_fidelity_levels[action.event_id]
            ]
            action_frontier[0, action_index] = torch.tensor(
                state.context_frontiers[action.event_id], dtype=torch.float32
            )
            for evidence_index, item in enumerate(state.evidence):
                affinity[0, action_index, evidence_index] = (
                    item.event_id == action.event_id
                )

    return RouterBatch(
        question_token_ids=question_ids[:, 0],
        question_token_mask=question_mask[:, 0],
        evidence_token_ids=evidence_ids,
        evidence_token_mask=evidence_token_mask,
        evidence_item_mask=evidence_item_mask,
        evidence_fidelity=evidence_fidelity,
        evidence_numeric=evidence_numeric,
        history_token_ids=history_ids,
        history_token_mask=history_token_mask,
        history_item_mask=history_item_mask,
        history_action_type=history_action_type,
        action_token_ids=action_ids,
        action_token_mask=action_token_mask,
        legal_action_mask=legal_mask,
        action_type=action_type,
        action_fidelity=action_fidelity,
        action_visual_budget=action_visual_budget,
        action_frontier=action_frontier,
        action_evidence_affinity=affinity,
        state_numeric=state_numeric,
        target_action_index=torch.zeros(batch_size, dtype=torch.long),
        sufficiency_target=torch.zeros(batch_size, dtype=torch.float32),
        cost_to_go_target=torch.zeros(batch_size, dtype=torch.float32),
    )


class BCPolicy:
    """Wrap a trained :class:`MemoryRouter` as a greedy :class:`RouterPolicy`."""

    def __init__(
        self,
        model: MemoryRouter,
        *,
        tokenizer: TextTokenizer | None = None,
        max_question_tokens: int | None = None,
        max_item_tokens: int | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer or TestByteTokenizer(
            model_id=model.config.encoder.model_id
        )
        self.max_question_tokens = max_question_tokens or model.config.max_question_tokens
        self.max_item_tokens = max_item_tokens or model.config.max_item_tokens
        self.model.eval()

    def __call__(
        self, state: RouterState, legal_actions: tuple[ActionInstance, ...]
    ) -> ActionInstance:
        if not legal_actions:
            raise InvalidPolicyActionError("no legal actions to score")
        actions = tuple(legal_actions)
        mask = (True,) * len(actions)
        batch = encode_router_state(
            state,
            actions,
            mask,
            tokenizer=self.tokenizer,
            max_question_tokens=self.max_question_tokens,
            max_item_tokens=self.max_item_tokens,
        )
        with torch.no_grad():
            output = self.model(batch)
        best = int(torch.argmax(output.action_logits[0]).item())
        return actions[best]


def _rollout(
    initial: RouterState,
    *,
    policy: RouterPolicy,
    environment: MemoryEnvironment,
    graph: CachedObservationGraph,
    max_steps: int = _MAX_DEPTH,
) -> tuple[EnvironmentTransition, ...]:
    """Rollout a policy over the cached graph using pure replay (no provider)."""

    transitions: list[EnvironmentTransition] = []
    state = initial
    for _ in range(max_steps):
        legal = environment.valid_actions(state)
        if not legal:
            break
        action = policy(state, legal)
        if action not in legal:
            raise InvalidPolicyActionError("policy selected an illegal action")
        if action.action_type is ActionType.STOP:
            observation = ActionObservation(
                action_type=ActionType.STOP, target_event_id=None
            )
        else:
            observation = graph.get(state, action)
            if observation is None:
                raise MissingCachedObservationError(
                    f"cached observation missing for {action.action_type.value}"
                )
        transition = environment.replay(state, action, observation)
        transitions.append(transition)
        state = transition.next_state
        if transition.terminal:
            break
    return tuple(transitions)


def label_best_next_action(
    state: RouterState,
    *,
    environment: MemoryEnvironment,
    graph: CachedObservationGraph,
    evaluator: AnswerEvaluator,
    normalization: CostNormalization,
) -> ActionInstance:
    """Compute the single-step optimal legal action from the cached graph."""

    result = beam_search(
        environment,
        state,
        graph,
        evaluator,
        beam_size=_BEAM_SIZE,
        max_depth=_MAX_DEPTH,
        cost_normalizer=normalization.constant,
    )
    if not result.paths:
        raise MissingCachedObservationError(
            "cached graph yields no complete path to label the state"
        )
    best = canonical_oracle(result.paths)
    if not best.transitions:
        raise MissingCachedObservationError(
            "canonical Oracle path has no leading transition"
        )
    return best.transitions[0].action


def _state_key(state: RouterState) -> str:
    payload = json.dumps(
        state.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Deviation(BaseModel):
    """A rollout state where the policy departed from the Oracle's first move."""

    model_config = ConfigDict(frozen=True)

    state: RouterState
    policy_action: ActionInstance
    oracle_action: ActionInstance


def collect_deviations(
    train_states: Sequence[RouterState],
    *,
    policy: RouterPolicy,
    environment: MemoryEnvironment,
    graph: CachedObservationGraph,
    evaluator: AnswerEvaluator,
    normalization: CostNormalization,
    seen_keys: set[str] | None = None,
) -> tuple[Deviation, ...]:
    """Rollout the policy and return every once-labelled departure state."""

    seen = set(seen_keys or ())
    deviations: list[Deviation] = []
    for initial in train_states:
        transitions = _rollout(
            initial,
            policy=policy,
            environment=environment,
            graph=graph,
        )
        for transition in transitions:
            state = transition.state
            key = _state_key(state)
            if key in seen:
                continue
            seen.add(key)
            oracle_action = label_best_next_action(
                state,
                environment=environment,
                graph=graph,
                evaluator=evaluator,
                normalization=normalization,
            )
            if transition.action != oracle_action:
                deviations.append(
                    Deviation(
                        state=state,
                        policy_action=transition.action,
                        oracle_action=oracle_action,
                    )
                )
    return tuple(deviations)


def _oracle_cost(
    state: RouterState,
    *,
    environment: MemoryEnvironment,
    graph: CachedObservationGraph,
    evaluator: AnswerEvaluator,
    normalization: CostNormalization,
) -> float:
    result = beam_search(
        environment,
        state,
        graph,
        evaluator,
        beam_size=_BEAM_SIZE,
        max_depth=_MAX_DEPTH,
        cost_normalizer=normalization.constant,
    )
    if not result.paths:
        return 0.0
    return canonical_oracle(result.paths).total_cost


def _evaluate_dev(
    dev_states: Sequence[RouterState],
    *,
    policy: RouterPolicy,
    environment: MemoryEnvironment,
    graph: CachedObservationGraph,
    evaluator: AnswerEvaluator,
    normalization: CostNormalization,
) -> tuple[float, float]:
    scores: list[float] = []
    regrets: list[float] = []
    for initial in dev_states:
        transitions = _rollout(
            initial,
            policy=policy,
            environment=environment,
            graph=graph,
        )
        final_state = transitions[-1].next_state if transitions else initial
        scores.append(evaluator(final_state).answer_score)
        policy_cost = sum(transition.step_cost for transition in transitions)
        oracle_cost = _oracle_cost(
            initial,
            environment=environment,
            graph=graph,
            evaluator=evaluator,
            normalization=normalization,
        )
        if oracle_cost > 0:
            regrets.append(max(0.0, policy_cost - oracle_cost) / oracle_cost)
    dev_utility = sum(scores) / len(scores) if scores else 0.0
    cost_regret = sum(regrets) / len(regrets) if regrets else 0.0
    return dev_utility, cost_regret


class DaggerRoundResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    round_number: int = Field(ge=1)
    deviations: tuple[Deviation, ...]
    dev_utility: float
    cost_regret: float = Field(ge=0)
    should_continue: bool


def _should_continue(
    round_number: int,
    dev_utility: float,
    cost_regret: float,
    previous: DaggerRoundResult | None,
) -> bool:
    if round_number >= 3:
        return False
    if round_number == 1:
        return True
    if previous is None:
        return False
    utility_gain = dev_utility - previous.dev_utility
    regret_improvement = previous.cost_regret - cost_regret
    utility_ok = utility_gain >= _UTILITY_GAIN_THRESHOLD
    regret_ok = (
        previous.cost_regret > 0
        and regret_improvement >= _REGRET_IMPROVEMENT_RATIO * previous.cost_regret
    )
    return utility_ok or regret_ok


def run_dagger_round(
    *,
    round_number: int,
    train_states: Sequence[RouterState],
    dev_states: Sequence[RouterState],
    policy: RouterPolicy,
    environment: MemoryEnvironment,
    graph: CachedObservationGraph,
    evaluator: AnswerEvaluator,
    normalization: CostNormalization,
    seen_keys: set[str] | None = None,
    previous: DaggerRoundResult | None = None,
) -> DaggerRoundResult:
    """Run one DAgger round and decide whether another round is warranted."""

    if round_number < 1:
        raise ValueError("round_number must be at least one")
    deviations = collect_deviations(
        train_states,
        policy=policy,
        environment=environment,
        graph=graph,
        evaluator=evaluator,
        normalization=normalization,
        seen_keys=seen_keys,
    )
    dev_utility, cost_regret = _evaluate_dev(
        dev_states,
        policy=policy,
        environment=environment,
        graph=graph,
        evaluator=evaluator,
        normalization=normalization,
    )
    return DaggerRoundResult(
        round_number=round_number,
        deviations=deviations,
        dev_utility=dev_utility,
        cost_regret=cost_regret,
        should_continue=_should_continue(
            round_number, dev_utility, cost_regret, previous
        ),
    )
