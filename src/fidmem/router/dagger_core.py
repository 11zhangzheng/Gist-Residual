"""Pure cached DAgger rollout and utility labelling primitives."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType

import torch
from pydantic import BaseModel, ConfigDict, Field

from fidmem.actions.environment import (
    ActionObservation,
    EnvironmentTransition,
    MemoryEnvironment,
)
from fidmem.agent.runner import RouterPolicy
from fidmem.oracle.labels import CostNormalization
from fidmem.oracle.search import (
    AnswerEvaluation,
    CachedObservationGraph,
    OraclePath,
    action_signature,
    beam_search,
    observation_key,
)
from fidmem.router.dataset import (
    RouterBatch,
    TestByteTokenizer,
    TextTokenizer,
    TokenizerIdentity,
)
from fidmem.router.model import MemoryRouter
from fidmem.types import ActionInstance, ActionType, FidelityLevel, RouterState

_ACTION_INDEX = {action: index for index, action in enumerate(ActionType)}
_FIDELITY_INDEX = {level: index for index, level in enumerate(FidelityLevel)}
_NONE_FIDELITY = len(_FIDELITY_INDEX)
_VISUAL_BUDGET_INDEX = {None: 0, "low": 1, "high": 2}
_MAX_DEPTH = 5
_BEAM_SIZE = 8
_UTILITY_GAIN_THRESHOLD = 0.005
_REGRET_IMPROVEMENT_RATIO = 0.02


class MissingCachedObservationError(RuntimeError):
    """A required cached observation is absent or search remains pending."""


class MissingCachedEvaluationError(RuntimeError):
    """A required frozen Answerer/Judge evaluation is absent."""


class InvalidPolicyActionError(ValueError):
    """A policy returned an action outside the environment's legal tuple."""


class CacheArtifactIdentity(BaseModel):
    """Immutable content identity for one offline cache artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=1, ge=1, strict=True)
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def evaluation_key(state: RouterState) -> str:
    """Return the content key for one cached Answerer/Judge evaluation."""

    return hashlib.sha256(
        _canonical_json(state.model_dump(mode="json")).encode("utf-8")
    ).hexdigest()


class CachedAnswerEvaluator:
    """Read-only frozen evaluations with no Answerer/provider fallback."""

    __slots__ = ("_evaluations", "_identity", "_sealed")

    def __init__(
        self,
        *,
        identity: CacheArtifactIdentity,
        evaluations: Mapping[str, AnswerEvaluation],
    ) -> None:
        object.__setattr__(
            self,
            "_identity",
            CacheArtifactIdentity.model_validate(identity.model_dump(mode="python")),
        )
        normalized: dict[str, AnswerEvaluation] = {}
        for key, evaluation in evaluations.items():
            if (
                not isinstance(key, str)
                or len(key) != 64
                or any(character not in "0123456789abcdef" for character in key)
            ):
                raise ValueError("cached evaluation key must be a SHA-256 digest")
            normalized[key] = AnswerEvaluation.model_validate(
                evaluation.model_dump(mode="python")
            )
        object.__setattr__(self, "_evaluations", MappingProxyType(normalized))
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("CachedAnswerEvaluator is immutable")
        object.__setattr__(self, name, value)

    @property
    def identity(self) -> CacheArtifactIdentity:
        return self._identity

    def get(self, state: RouterState) -> AnswerEvaluation | None:
        return self._evaluations.get(evaluation_key(state))

    def __len__(self) -> int:
        return len(self._evaluations)


class CachedUtilityGraph:
    """Identity-bound observation and evaluation caches used by DAgger."""

    __slots__ = (
        "_evaluator",
        "_identity",
        "_observation_identity",
        "_observations",
        "_sealed",
    )

    def __init__(
        self,
        *,
        identity: CacheArtifactIdentity,
        observation_identity: CacheArtifactIdentity,
        observations: CachedObservationGraph,
        evaluator: CachedAnswerEvaluator,
    ) -> None:
        if not isinstance(observations, CachedObservationGraph):
            raise TypeError("observations must be a CachedObservationGraph")
        if not isinstance(evaluator, CachedAnswerEvaluator):
            raise TypeError("evaluator must be a CachedAnswerEvaluator")
        object.__setattr__(
            self,
            "_identity",
            CacheArtifactIdentity.model_validate(identity.model_dump(mode="python")),
        )
        object.__setattr__(
            self,
            "_observation_identity",
            CacheArtifactIdentity.model_validate(
                observation_identity.model_dump(mode="python")
            ),
        )
        object.__setattr__(self, "_observations", observations)
        object.__setattr__(self, "_evaluator", evaluator)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("CachedUtilityGraph is immutable")
        object.__setattr__(self, name, value)

    @property
    def identity(self) -> CacheArtifactIdentity:
        return self._identity

    @property
    def observation_identity(self) -> CacheArtifactIdentity:
        return self._observation_identity

    @property
    def evaluator_identity(self) -> CacheArtifactIdentity:
        return self._evaluator.identity

    def get(
        self, state: RouterState, action: ActionInstance
    ) -> ActionObservation | None:
        return self._observations.get(state, action)

    def get_evaluation(self, state: RouterState) -> AnswerEvaluation | None:
        return self._evaluator.get(state)


class ForbiddenObservationGenerator:
    """Executor injected into cached DAgger environments; every call fails."""

    def __call__(self, action: ActionInstance, state: RouterState) -> ActionObservation:
        raise AssertionError(
            "observation generation is forbidden during cached DAgger correction"
        )


def _action_text(action: ActionInstance) -> str:
    return "|".join(
        (action.action_type.value, action.event_id or "", action.visual_budget or "")
    )


def _padded_tokens(
    items: Sequence[Sequence[str]],
    *,
    maximum: int,
    label: str,
    tokenizer: TextTokenizer,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = len(items)
    max_items = max(1, max((len(row) for row in items), default=0))
    encoded = [
        [tokenizer.encode(text, maximum=maximum, label=label) for text in row]
        for row in items
    ]
    max_tokens = max(
        1,
        max(
            (len(token_ids) for row in encoded for token_ids in row),
            default=0,
        ),
    )
    token_ids = torch.zeros((batch_size, max_items, max_tokens), dtype=torch.long)
    token_mask = torch.zeros_like(token_ids, dtype=torch.bool)
    item_mask = torch.zeros((batch_size, max_items), dtype=torch.bool)
    for row_index, row in enumerate(encoded):
        for item_index, item in enumerate(row):
            if item:
                token_ids[row_index, item_index, : len(item)] = torch.tensor(item)
                token_mask[row_index, item_index, : len(item)] = True
            item_mask[row_index, item_index] = True
    return token_ids, token_mask, item_mask


def encode_router_state(
    state: RouterState,
    action_instances: Sequence[ActionInstance],
    legal_action_mask: Sequence[bool],
    *,
    tokenizer: TextTokenizer,
    max_question_tokens: int,
    max_item_tokens: int,
) -> RouterBatch:
    """Encode one state without importing Task 10 private helpers."""

    actions = tuple(action_instances)
    mask = tuple(legal_action_mask)
    if not actions or len(actions) != len(mask) or not any(mask):
        raise ValueError("legal action mask must cover at least one action")
    question_ids, question_mask, _ = _padded_tokens(
        ((state.question + "\n" + "\n".join(state.options),),),
        maximum=max_question_tokens,
        label="question",
        tokenizer=tokenizer,
    )
    evidence_ids, evidence_token_mask, evidence_item_mask = _padded_tokens(
        ([item.content for item in state.evidence],),
        maximum=max_item_tokens,
        label="evidence item",
        tokenizer=tokenizer,
    )
    history_ids, history_token_mask, history_item_mask = _padded_tokens(
        ([_action_text(action) for action in state.action_history],),
        maximum=max_item_tokens,
        label="history action",
        tokenizer=tokenizer,
    )
    action_ids, action_token_mask, action_item_mask = _padded_tokens(
        ([_action_text(action) for action in actions],),
        maximum=max_item_tokens,
        label="action instance",
        tokenizer=tokenizer,
    )
    batch_size, max_actions = action_item_mask.shape
    max_evidence = evidence_item_mask.shape[1]
    evidence_fidelity = torch.full(
        (batch_size, max_evidence), _NONE_FIDELITY, dtype=torch.long
    )
    evidence_numeric = torch.zeros((batch_size, max_evidence, 2), dtype=torch.float32)
    history_action_type = torch.zeros_like(history_item_mask, dtype=torch.long)
    legal_mask = torch.zeros((batch_size, max_actions), dtype=torch.bool)
    action_type = torch.zeros((batch_size, max_actions), dtype=torch.long)
    action_fidelity = torch.full(
        (batch_size, max_actions), _NONE_FIDELITY, dtype=torch.long
    )
    action_visual_budget = torch.zeros((batch_size, max_actions), dtype=torch.long)
    action_frontier = torch.zeros((batch_size, max_actions, 2), dtype=torch.float32)
    affinity = torch.zeros((batch_size, max_actions, max_evidence), dtype=torch.bool)
    state_numeric = torch.tensor(
        ((state.remaining_budget, state.cost_preference),), dtype=torch.float32
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
    """Greedy Task 10 policy with exact tokenizer identity and device safety."""

    def __init__(
        self,
        model: MemoryRouter,
        *,
        tokenizer: TextTokenizer | None = None,
        max_question_tokens: int | None = None,
        max_item_tokens: int | None = None,
    ) -> None:
        if tokenizer is None:
            if model.config.encoder.kind == "pretrained":
                raise ValueError(
                    "production BCPolicy requires the actual pinned tokenizer"
                )
            tokenizer = TestByteTokenizer(model.config.encoder.tokenizer.model_id)
        raw_identity = getattr(tokenizer, "identity", None)
        if not isinstance(raw_identity, TokenizerIdentity):
            raise ValueError("tokenizer identity must be a Task 10 TokenizerIdentity")
        identity = TokenizerIdentity.model_validate(
            raw_identity.model_dump(mode="python")
        )
        if identity != model.config.encoder.tokenizer:
            raise ValueError("tokenizer identity does not match encoder config")
        self.model = model
        self.tokenizer = tokenizer
        self.max_question_tokens = (
            max_question_tokens or model.config.max_question_tokens
        )
        self.max_item_tokens = max_item_tokens or model.config.max_item_tokens
        self.model.eval()

    def __call__(
        self, state: RouterState, legal_actions: tuple[ActionInstance, ...]
    ) -> ActionInstance:
        if not legal_actions:
            raise InvalidPolicyActionError("no legal actions to score")
        batch = encode_router_state(
            state,
            legal_actions,
            (True,) * len(legal_actions),
            tokenizer=self.tokenizer,
            max_question_tokens=self.max_question_tokens,
            max_item_tokens=self.max_item_tokens,
        )
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        self.model.eval()
        with torch.no_grad():
            logits = self.model(batch.to(device)).action_logits
            index = int(logits[0].argmax().item())
        if index < 0 or index >= len(legal_actions):
            raise InvalidPolicyActionError("model selected outside the legal mask")
        return legal_actions[index]


def _rollout(
    initial: RouterState,
    *,
    policy: RouterPolicy,
    environment: MemoryEnvironment,
    utility_graph: CachedUtilityGraph | None = None,
    graph: CachedObservationGraph | None = None,
    max_steps: int = _MAX_DEPTH,
) -> tuple[EnvironmentTransition, ...]:
    """Roll out with `get` plus `replay`; the executor is never reachable."""

    if (utility_graph is None) == (graph is None):
        raise ValueError("supply exactly one cached graph")
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
            observation = (
                utility_graph.get(state, action)
                if utility_graph is not None
                else graph.get(state, action)  # type: ignore[union-attr]
            )
            if observation is None:
                raise MissingCachedObservationError(
                    f"cached observation missing for {action_signature(action)}"
                )
        transition = environment.replay(state, action, observation)
        transitions.append(transition)
        state = transition.next_state
        if transition.terminal:
            break
    return tuple(transitions)


def _cached_evaluator(graph: CachedUtilityGraph):
    def evaluate(state: RouterState) -> AnswerEvaluation:
        result = graph.get_evaluation(state)
        if result is None:
            raise MissingCachedEvaluationError(
                f"cached evaluation missing for state {evaluation_key(state)}"
            )
        return result

    return evaluate


def _complete_search_paths(
    result_paths: Sequence[OraclePath], pending: Sequence[object]
) -> tuple[OraclePath, ...]:
    if pending:
        raise MissingCachedObservationError(
            f"cached Oracle search has {len(pending)} pending observations"
        )
    paths = tuple(result_paths)
    if not paths:
        raise MissingCachedObservationError(
            "cached graph yields no complete path to label the state"
        )
    return paths


def _utility_optimal_path(
    paths: Sequence[OraclePath],
    *,
    state: RouterState,
    normalization: CostNormalization,
) -> OraclePath:
    """Use the exact Task 9 utility objective and deterministic signature tie."""

    def key(path: OraclePath) -> tuple[float, float, int, tuple[str, ...]]:
        utility = path.answer_score - state.cost_preference * (
            path.total_cost / normalization.constant
        )
        return (-utility, path.total_cost, path.depth, path.action_signature)

    return min(paths, key=key)


def label_best_next_action(
    state: RouterState,
    *,
    environment: MemoryEnvironment,
    utility_graph: CachedUtilityGraph,
    normalization: CostNormalization,
    beam_size: int = _BEAM_SIZE,
    max_depth: int = _MAX_DEPTH,
) -> ActionInstance:
    """Return the utility-optimal first action only from complete caches."""

    if not isinstance(utility_graph, CachedUtilityGraph):
        raise TypeError("utility_graph must be a CachedUtilityGraph")
    result = beam_search(
        environment,
        state,
        utility_graph,  # read-only get contract matches CachedObservationGraph
        _cached_evaluator(utility_graph),
        beam_size=beam_size,
        max_depth=max_depth,
        cost_normalizer=normalization.constant,
    )
    paths = _complete_search_paths(result.paths, result.pending)
    best = _utility_optimal_path(paths, state=state, normalization=normalization)
    if not best.transitions:
        raise MissingCachedObservationError("utility-optimal path has no transition")
    return best.transitions[0].action


def _oracle_cost(
    state: RouterState,
    *,
    environment: MemoryEnvironment,
    utility_graph: CachedUtilityGraph,
    normalization: CostNormalization,
    beam_size: int = _BEAM_SIZE,
    max_depth: int = _MAX_DEPTH,
) -> float:
    result = beam_search(
        environment,
        state,
        utility_graph,
        _cached_evaluator(utility_graph),
        beam_size=beam_size,
        max_depth=max_depth,
        cost_normalizer=normalization.constant,
    )
    paths = _complete_search_paths(result.paths, result.pending)
    return _utility_optimal_path(
        paths, state=state, normalization=normalization
    ).total_cost


def budget_bin(value: float, *, width: float) -> int:
    """Right-closed budget bin: 19.999 and 20.0 share width-1 bin 20."""

    if not math.isfinite(value) or value < 0:
        raise ValueError("remaining budget must be finite and non-negative")
    if not math.isfinite(width) or width <= 0:
        raise ValueError("budget bin width must be finite and positive")
    return int(math.ceil(value / width))


def _acquired_observation_keys(
    transitions: Sequence[EnvironmentTransition],
) -> tuple[str, ...]:
    keys: list[str] = []
    for transition in transitions:
        if transition.action.action_type is ActionType.STOP:
            continue
        key = observation_key(transition.state, transition.action)
        keys.append(f"{key.state_sha256}|{key.action_signature}")
    return tuple(sorted(set(keys)))


def _state_key(
    state: RouterState,
    *,
    question_id: str = "legacy",
    replayed_transitions: Sequence[EnvironmentTransition] = (),
    budget_bin_width: float = 1.0,
) -> str:
    """Hash only actual replay acquisitions plus routing-relevant state."""

    if not question_id:
        raise ValueError("question_id must be non-empty")
    payload = {
        "question_id": question_id,
        "acquired_observation_keys": _acquired_observation_keys(replayed_transitions),
        "budget_bin": budget_bin(state.remaining_budget, width=budget_bin_width),
        "cost_preference": state.cost_preference,
        "frontier": tuple(
            (event_id, tuple(frontier))
            for event_id, frontier in sorted(state.context_frontiers.items())
        ),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


class Deviation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    state_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    question_id: str = Field(min_length=1)
    state: RouterState
    action_instances: tuple[ActionInstance, ...] = Field(min_length=1)
    legal_action_mask: tuple[bool, ...] = Field(min_length=1)
    policy_action: ActionInstance
    oracle_action: ActionInstance
    acquired_observation_keys: tuple[str, ...]


def collect_deviations(
    train_states: Sequence[RouterState],
    *,
    policy: RouterPolicy,
    environment: MemoryEnvironment,
    utility_graph: CachedUtilityGraph,
    normalization: CostNormalization,
    question_ids: Sequence[str] | None = None,
    seen_keys: set[str] | None = None,
    budget_bin_width: float = 1.0,
    beam_size: int = _BEAM_SIZE,
    max_depth: int = _MAX_DEPTH,
) -> tuple[Deviation, ...]:
    """Collect once-labelled departures and update the caller-owned seen set."""

    ids = tuple(
        question_ids or (f"question-{index}" for index in range(len(train_states)))
    )
    if len(ids) != len(train_states):
        raise ValueError("question_ids must match train_states")
    seen = seen_keys if seen_keys is not None else set()
    deviations: list[Deviation] = []
    for question_id, initial in zip(ids, train_states, strict=True):
        transitions = _rollout(
            initial,
            policy=policy,
            environment=environment,
            utility_graph=utility_graph,
            max_steps=max_depth,
        )
        prefix: list[EnvironmentTransition] = []
        for transition in transitions:
            state = transition.state
            key = _state_key(
                state,
                question_id=question_id,
                replayed_transitions=prefix,
                budget_bin_width=budget_bin_width,
            )
            if key in seen:
                prefix.append(transition)
                continue
            seen.add(key)
            oracle_action = label_best_next_action(
                state,
                environment=environment,
                utility_graph=utility_graph,
                normalization=normalization,
                beam_size=beam_size,
                max_depth=max_depth,
            )
            legal = environment.valid_actions(state)
            if transition.action != oracle_action:
                deviations.append(
                    Deviation(
                        state_key=key,
                        question_id=question_id,
                        state=state,
                        action_instances=legal,
                        legal_action_mask=(True,) * len(legal),
                        policy_action=transition.action,
                        oracle_action=oracle_action,
                        acquired_observation_keys=_acquired_observation_keys(prefix),
                    )
                )
            prefix.append(transition)
    return tuple(deviations)


def _evaluate_dev(
    dev_states: Sequence[RouterState],
    *,
    policy: RouterPolicy,
    environment: MemoryEnvironment,
    utility_graph: CachedUtilityGraph,
    normalization: CostNormalization,
    beam_size: int = _BEAM_SIZE,
    max_depth: int = _MAX_DEPTH,
) -> tuple[float, float]:
    utilities: list[float] = []
    regrets: list[float] = []
    for initial in dev_states:
        transitions = _rollout(
            initial,
            policy=policy,
            environment=environment,
            utility_graph=utility_graph,
            max_steps=max_depth,
        )
        evaluation_state = initial
        if transitions:
            last = transitions[-1]
            evaluation_state = (
                last.state
                if last.action.action_type is ActionType.STOP
                else last.next_state
            )
        evaluation = utility_graph.get_evaluation(evaluation_state)
        if evaluation is None:
            raise MissingCachedEvaluationError(
                f"cached evaluation missing for state {evaluation_key(evaluation_state)}"
            )
        policy_cost = sum(transition.step_cost for transition in transitions)
        utilities.append(
            evaluation.answer_score
            - initial.cost_preference * (policy_cost / normalization.constant)
        )
        oracle_cost = _oracle_cost(
            initial,
            environment=environment,
            utility_graph=utility_graph,
            normalization=normalization,
            beam_size=beam_size,
            max_depth=max_depth,
        )
        if oracle_cost > 0:
            regrets.append(max(0.0, policy_cost - oracle_cost) / oracle_cost)
        elif policy_cost > 0:
            regrets.append(1.0)
        else:
            regrets.append(0.0)
    return (
        sum(utilities) / len(utilities) if utilities else 0.0,
        sum(regrets) / len(regrets) if regrets else 0.0,
    )


class DaggerRoundResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    round_number: int = Field(ge=1)
    deviations: tuple[Deviation, ...]
    dev_utility: float
    cost_regret: float = Field(ge=0)
    should_continue: bool
    seen_keys: tuple[str, ...] = ()


def _should_continue(
    round_number: int,
    dev_utility: float,
    cost_regret: float,
    previous: DaggerRoundResult | None,
    *,
    utility_gain_threshold: float = _UTILITY_GAIN_THRESHOLD,
    regret_improvement_ratio: float = _REGRET_IMPROVEMENT_RATIO,
) -> bool:
    if round_number >= 3:
        return False
    if round_number == 1:
        return True
    if previous is None:
        return False
    utility_gain = dev_utility - previous.dev_utility
    regret_improvement = previous.cost_regret - cost_regret
    return utility_gain >= utility_gain_threshold or (
        previous.cost_regret > 0
        and regret_improvement >= regret_improvement_ratio * previous.cost_regret
    )


def run_dagger_round(
    *,
    round_number: int,
    train_states: Sequence[RouterState],
    dev_states: Sequence[RouterState],
    policy: RouterPolicy,
    environment: MemoryEnvironment,
    utility_graph: CachedUtilityGraph,
    normalization: CostNormalization,
    question_ids: Sequence[str] | None = None,
    seen_keys: set[str] | None = None,
    previous: DaggerRoundResult | None = None,
    budget_bin_width: float = 1.0,
    beam_size: int = _BEAM_SIZE,
    max_depth: int = _MAX_DEPTH,
    utility_gain_threshold: float = _UTILITY_GAIN_THRESHOLD,
    regret_improvement_ratio: float = _REGRET_IMPROVEMENT_RATIO,
) -> DaggerRoundResult:
    if round_number < 1:
        raise ValueError("round_number must be at least one")
    seen = seen_keys if seen_keys is not None else set()
    deviations = collect_deviations(
        train_states,
        policy=policy,
        environment=environment,
        utility_graph=utility_graph,
        normalization=normalization,
        question_ids=question_ids,
        seen_keys=seen,
        budget_bin_width=budget_bin_width,
        beam_size=beam_size,
        max_depth=max_depth,
    )
    dev_utility, cost_regret = _evaluate_dev(
        dev_states,
        policy=policy,
        environment=environment,
        utility_graph=utility_graph,
        normalization=normalization,
        beam_size=beam_size,
        max_depth=max_depth,
    )
    return DaggerRoundResult(
        round_number=round_number,
        deviations=deviations,
        dev_utility=dev_utility,
        cost_regret=cost_regret,
        should_continue=_should_continue(
            round_number,
            dev_utility,
            cost_regret,
            previous,
            utility_gain_threshold=utility_gain_threshold,
            regret_improvement_ratio=regret_improvement_ratio,
        ),
        seen_keys=tuple(sorted(seen)),
    )


__all__ = [
    "BCPolicy",
    "CacheArtifactIdentity",
    "CachedAnswerEvaluator",
    "CachedUtilityGraph",
    "DaggerRoundResult",
    "Deviation",
    "ForbiddenObservationGenerator",
    "InvalidPolicyActionError",
    "MissingCachedEvaluationError",
    "MissingCachedObservationError",
    "_oracle_cost",
    "_rollout",
    "_should_continue",
    "_state_key",
    "budget_bin",
    "collect_deviations",
    "encode_router_state",
    "evaluation_key",
    "label_best_next_action",
    "run_dagger_round",
]
