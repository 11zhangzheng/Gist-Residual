"""Deterministic hard-masked state transitions for memory acquisition."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from fidmem.types import (
    ActionInstance,
    ActionType,
    EventRecord,
    EvidenceItem,
    FidelityLevel,
    RouterState,
)


class TerminalStateError(RuntimeError):
    """Raised when a transition is requested after STOP."""


class IllegalActionError(ValueError):
    """Raised when a policy submits an action outside the hard mask."""


class ObservationValidationError(ValueError):
    """Raised when an executor returns data outside the frozen topology."""


class ActionCostTable(BaseModel):
    """Frozen costs; policy and executor cannot choose their own charge."""

    model_config = ConfigDict(frozen=True)

    search_gist: float = Field(default=1.0, ge=0)
    residual: float = Field(default=2.0, ge=0)
    context: float = Field(default=0.5, ge=0)
    visual_low: float = Field(default=4.0, ge=0)
    visual_high: float = Field(default=8.0, ge=0)
    cache_hit: float = Field(default=0.0, ge=0)


class ActionObservation(BaseModel):
    """Typed atomic output from an injected I/O executor."""

    model_config = ConfigDict(frozen=True)

    evidence: tuple[EvidenceItem, ...] = ()
    candidate_event_ids: tuple[str, ...] = ()
    cache_status: Literal["hit", "miss"] = "miss"
    input_frames: int = Field(default=0, ge=0)


class EnvironmentTransition(BaseModel):
    """One auditable reducer result, including the centrally charged cost."""

    model_config = ConfigDict(frozen=True)

    state: RouterState
    action: ActionInstance
    observation: ActionObservation
    next_state: RouterState
    step_cost: float = Field(ge=0)
    terminal: bool = False


ActionExecutor = Callable[[ActionInstance, RouterState], ActionObservation]


class MemoryEnvironment:
    """Owns topology, legality, cost charging, and pure state reduction."""

    def __init__(
        self,
        *,
        events: Sequence[EventRecord],
        executor: ActionExecutor,
        costs: ActionCostTable | Mapping[str, float] | None = None,
    ) -> None:
        ordered = tuple(sorted(events, key=lambda event: (event.start_sec, event.end_sec, event.event_id)))
        if len({event.event_id for event in ordered}) != len(ordered):
            raise ValueError("event ids must be unique in an environment")
        self._events = ordered
        self._events_by_id = {event.event_id: event for event in ordered}
        self._event_order = tuple(event.event_id for event in ordered)
        self._executor = executor
        self.costs = self._coerce_costs(costs)

    @staticmethod
    def _coerce_costs(costs: ActionCostTable | Mapping[str, float] | None) -> ActionCostTable:
        if costs is None:
            return ActionCostTable()
        if isinstance(costs, ActionCostTable):
            return costs
        return ActionCostTable.model_validate(costs)

    @staticmethod
    def _instance(action_type: ActionType, event_id: str | None = None, visual_budget: str | None = None) -> ActionInstance:
        return ActionInstance(action_type, event_id, visual_budget)  # type: ignore[arg-type]

    @staticmethod
    def _is_terminal(state: RouterState) -> bool:
        return bool(state.action_history and state.action_history[-1].action_type is ActionType.STOP)

    @staticmethod
    def _action_seen(state: RouterState, action: ActionInstance) -> bool:
        return action in state.action_history

    def _cost_for(self, action: ActionInstance, *, cache_status: str | None = None) -> float:
        if action.action_type is ActionType.STOP:
            return 0.0
        if cache_status == "hit":
            return self.costs.cache_hit
        if action.action_type is ActionType.SEARCH_GIST:
            return self.costs.search_gist
        if action.action_type is ActionType.EXPAND_RESIDUAL:
            return self.costs.residual
        if action.action_type is ActionType.EXPAND_CONTEXT:
            return self.costs.context
        if action.visual_budget == "low":
            return self.costs.visual_low
        return self.costs.visual_high

    def _has_unvisited_context(self, event_id: str, frontier: tuple[int, int]) -> bool:
        anchor = self._event_order.index(event_id)
        left, right = frontier
        return left < anchor or right < len(self._event_order) - anchor - 1

    def _context_event_ids(self, event_id: str, frontier: tuple[int, int]) -> tuple[str, ...]:
        anchor = self._event_order.index(event_id)
        left, right = frontier
        result: list[str] = []
        if left < anchor:
            result.append(self._event_order[anchor - left - 1])
        if right < len(self._event_order) - anchor - 1:
            result.append(self._event_order[anchor + right + 1])
        return tuple(result)

    def valid_actions(self, state: RouterState) -> tuple[ActionInstance, ...]:
        """Return a stable, provider-free tuple of every currently legal action."""
        if self._is_terminal(state):
            return ()
        if not state.candidate_event_ids:
            search = self._instance(ActionType.SEARCH_GIST)
            return (self._instance(ActionType.STOP),) if self._action_seen(state, search) else (search,)

        actions: list[ActionInstance] = []
        for event_id in state.candidate_event_ids:
            if event_id not in self._events_by_id:
                raise ObservationValidationError("state contains an event outside the frozen topology")
            residual = self._instance(ActionType.EXPAND_RESIDUAL, event_id)
            context = self._instance(ActionType.EXPAND_CONTEXT, event_id)
            low = self._instance(ActionType.VERIFY_VISUAL, event_id, "low")
            high = self._instance(ActionType.VERIFY_VISUAL, event_id, "high")
            if not self._action_seen(state, residual) and state.remaining_budget >= self._cost_for(residual):
                actions.append(residual)
            if self._has_unvisited_context(event_id, state.context_frontiers[event_id]) and state.remaining_budget >= self._cost_for(context):
                actions.append(context)
            if not self._action_seen(state, low) and state.remaining_budget >= self._cost_for(low):
                actions.append(low)
            if not self._action_seen(state, high) and state.remaining_budget >= self._cost_for(high):
                actions.append(high)
        actions.append(self._instance(ActionType.STOP))
        return tuple(actions)

    def _validate_observation(self, state: RouterState, action: ActionInstance, observation: ActionObservation) -> None:
        if action.action_type is not ActionType.SEARCH_GIST and observation.candidate_event_ids:
            raise ObservationValidationError("only SEARCH_GIST may introduce candidates")
        if any(event_id not in self._events_by_id for event_id in observation.candidate_event_ids):
            raise ObservationValidationError("observation introduced an unknown event id")
        if observation.cache_status == "hit" and observation.input_frames != 0:
            raise ObservationValidationError("cache hits must not report new input frames")
        for item in observation.evidence:
            if item.event_id not in self._events_by_id:
                raise ObservationValidationError("evidence refers to an unknown event id")
            if item.attachments and action.action_type is not ActionType.VERIFY_VISUAL:
                raise ObservationValidationError("only visual verification may attach frame paths")
        if action.action_type is ActionType.STOP and (observation.evidence or observation.candidate_event_ids):
            raise ObservationValidationError("STOP cannot acquire evidence")

    def _reduce(self, state: RouterState, action: ActionInstance, observation: ActionObservation, step_cost: float) -> RouterState:
        candidate_ids = list(state.candidate_event_ids)
        fidelity = dict(state.candidate_fidelity_levels)
        frontiers = dict(state.context_frontiers)
        if action.action_type is ActionType.SEARCH_GIST:
            for event_id in observation.candidate_event_ids:
                if event_id not in fidelity:
                    candidate_ids.append(event_id)
                    fidelity[event_id] = FidelityLevel.GIST
                    frontiers[event_id] = (0, 0)
        elif action.action_type is ActionType.EXPAND_CONTEXT and action.event_id is not None:
            left, right = frontiers[action.event_id]
            anchor = self._event_order.index(action.event_id)
            if left < anchor:
                left += 1
            if right < len(self._event_order) - anchor - 1:
                right += 1
            frontiers[action.event_id] = (left, right)
            for event_id in self._context_event_ids(action.event_id, state.context_frontiers[action.event_id]):
                if event_id not in fidelity:
                    candidate_ids.append(event_id)
                    fidelity[event_id] = FidelityLevel.GIST
                    frontiers[event_id] = (0, 0)
        elif action.event_id is not None and action.action_type is ActionType.EXPAND_RESIDUAL:
            fidelity[action.event_id] = FidelityLevel.RESIDUAL
        elif action.event_id is not None and action.action_type is ActionType.VERIFY_VISUAL:
            fidelity[action.event_id] = FidelityLevel.VISUAL

        acquisition_step = len(state.action_history) + 1
        normalized_evidence = tuple(
            item.model_copy(update={"start_sec": self._events_by_id[item.event_id].start_sec, "acquisition_step": acquisition_step})
            for item in observation.evidence
        )
        return RouterState(
            question=state.question,
            options=state.options,
            evidence=state.evidence + normalized_evidence,
            action_history=state.action_history + (action,),
            remaining_budget=max(0.0, state.remaining_budget - step_cost),
            candidate_event_ids=tuple(candidate_ids),
            candidate_fidelity_levels=fidelity,
            context_frontiers=frontiers,
            cost_preference=state.cost_preference,
        )

    def step(self, state: RouterState, action: ActionInstance) -> EnvironmentTransition:
        """Validate first, then execute one atomic provider call and reduce state."""
        if self._is_terminal(state):
            raise TerminalStateError("cannot step a terminal state")
        legal = self.valid_actions(state)
        if action not in legal:
            raise IllegalActionError("action is not legal in the current state")
        observation = ActionObservation() if action.action_type is ActionType.STOP else self._executor(action, state)
        if not isinstance(observation, ActionObservation):
            raise ObservationValidationError("executor must return an ActionObservation")
        self._validate_observation(state, action, observation)
        step_cost = self._cost_for(action, cache_status=observation.cache_status)
        if step_cost > state.remaining_budget:
            raise IllegalActionError("action exceeds the remaining budget")
        next_state = self._reduce(state, action, observation, step_cost)
        return EnvironmentTransition(
            state=state,
            action=action,
            observation=observation,
            next_state=next_state,
            step_cost=step_cost,
            terminal=action.action_type is ActionType.STOP,
        )
