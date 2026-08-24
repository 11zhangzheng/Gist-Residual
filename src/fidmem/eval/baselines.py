"""Fair baseline policies over the exact hard-masked action tuple."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from fidmem.types import ActionInstance, ActionType, FidelityLevel, RouterState


class BaselinePolicyError(ValueError):
    pass


def _exact(
    candidate: object, legal_actions: tuple[ActionInstance, ...]
) -> ActionInstance:
    if not isinstance(candidate, ActionInstance) or not any(
        candidate is action for action in legal_actions
    ):
        raise BaselinePolicyError("policy must return an exact legal ActionInstance")
    return candidate


def _ordered(
    legal_actions: tuple[ActionInstance, ...],
    action_type: ActionType,
    *,
    visual_budget: Literal["low", "high"] | None = None,
) -> tuple[ActionInstance, ...]:
    matches = tuple(
        action
        for action in legal_actions
        if action.action_type is action_type
        and (visual_budget is None or action.visual_budget == visual_budget)
    )
    return tuple(
        sorted(
            matches,
            key=lambda action: (
                action.event_id or "",
                {None: 0, "low": 1, "high": 2}[action.visual_budget],
                legal_actions.index(action),
            ),
        )
    )


def _first(
    legal_actions: tuple[ActionInstance, ...],
    action_type: ActionType,
    *,
    visual_budget: Literal["low", "high"] | None = None,
) -> ActionInstance | None:
    values = _ordered(legal_actions, action_type, visual_budget=visual_budget)
    return values[0] if values else None


def _stop_or_raise(legal_actions: tuple[ActionInstance, ...]) -> ActionInstance:
    stop = _first(legal_actions, ActionType.STOP)
    if stop is None:
        raise BaselinePolicyError("preferred action is unavailable and STOP is not legal")
    return stop


class GistOnlyPolicy:
    def __call__(
        self, state: RouterState, legal_actions: tuple[ActionInstance, ...]
    ) -> ActionInstance:
        del state
        return _first(legal_actions, ActionType.SEARCH_GIST) or _stop_or_raise(
            legal_actions
        )


class GistResidualPolicy:
    def __call__(
        self, state: RouterState, legal_actions: tuple[ActionInstance, ...]
    ) -> ActionInstance:
        del state
        return (
            _first(legal_actions, ActionType.SEARCH_GIST)
            or _first(legal_actions, ActionType.EXPAND_RESIDUAL)
            or _stop_or_raise(legal_actions)
        )


class GistVisualPolicy:
    def __call__(
        self, state: RouterState, legal_actions: tuple[ActionInstance, ...]
    ) -> ActionInstance:
        del state
        return (
            _first(legal_actions, ActionType.SEARCH_GIST)
            or _first(
                legal_actions, ActionType.VERIFY_VISUAL, visual_budget="high"
            )
            or _stop_or_raise(legal_actions)
        )


class UniformFramesPolicy:
    """Deterministically verify every retrieved candidate at the low frame tier."""

    def __call__(
        self, state: RouterState, legal_actions: tuple[ActionInstance, ...]
    ) -> ActionInstance:
        del state
        return (
            _first(legal_actions, ActionType.SEARCH_GIST)
            or _first(legal_actions, ActionType.VERIFY_VISUAL, visual_budget="low")
            or _stop_or_raise(legal_actions)
        )


class FullResidualPolicy:
    """Expand the full context frontier, then materialize every Residual."""

    def __call__(
        self, state: RouterState, legal_actions: tuple[ActionInstance, ...]
    ) -> ActionInstance:
        del state
        return (
            _first(legal_actions, ActionType.SEARCH_GIST)
            or _first(legal_actions, ActionType.EXPAND_RESIDUAL)
            or _first(legal_actions, ActionType.EXPAND_CONTEXT)
            or _stop_or_raise(legal_actions)
        )


class RulePolicy:
    def __init__(self, *, sufficiency_threshold: float = 0.8) -> None:
        if not 0 <= sufficiency_threshold <= 1:
            raise ValueError("sufficiency threshold must be in [0, 1]")
        self.sufficiency_threshold = sufficiency_threshold

    def __call__(
        self, state: RouterState, legal_actions: tuple[ActionInstance, ...]
    ) -> ActionInstance:
        search = _first(legal_actions, ActionType.SEARCH_GIST)
        if search is not None:
            return search
        if state.evidence and max(item.score for item in state.evidence) >= self.sufficiency_threshold:
            return _stop_or_raise(legal_actions)
        return (
            _first(legal_actions, ActionType.EXPAND_RESIDUAL)
            or _first(legal_actions, ActionType.VERIFY_VISUAL, visual_budget="low")
            or _first(legal_actions, ActionType.EXPAND_CONTEXT)
            or _stop_or_raise(legal_actions)
        )


class ControllerCost(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False, strict=True)

    total_cost: float = Field(default=0.0, ge=0)
    gpu_seconds: float = Field(default=0.0, ge=0)
    wall_seconds: float = Field(default=0.0, ge=0)
    input_frames: int = Field(default=0, ge=0)
    visual_tokens: int = Field(default=0, ge=0)
    text_tokens: int = Field(default=0, ge=0)
    peak_memory_bytes: int = Field(default=0, ge=0)


class PromptControllerDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    action: ActionInstance
    rationale: str = ""
    cost: ControllerCost = ControllerCost()


PromptController = Callable[
    [RouterState, tuple[ActionInstance, ...]], PromptControllerDecision
]


class PromptControllerPolicy:
    """Keep controller prose private and expose only its selected action/cost."""

    def __init__(self, controller: PromptController) -> None:
        self.controller = controller
        self._pending_cost: ControllerCost | None = None

    def __call__(
        self, state: RouterState, legal_actions: tuple[ActionInstance, ...]
    ) -> ActionInstance:
        decision = self.controller(state, legal_actions)
        if not isinstance(decision, PromptControllerDecision):
            raise BaselinePolicyError(
                "prompt controller must return PromptControllerDecision"
            )
        action = _exact(decision.action, legal_actions)
        self._pending_cost = decision.cost
        return action

    def consume_last_controller_cost(self) -> ControllerCost:
        value = self._pending_cost or ControllerCost()
        self._pending_cost = None
        return value


@dataclass(frozen=True)
class QuestionOnlyView:
    question: str
    options: tuple[str, ...]
    remaining_budget: float
    cost_preference: float


@dataclass(frozen=True)
class TextAdaptiveView:
    question: str
    options: tuple[str, ...]
    gist_text: tuple[str, ...]
    action_history: tuple[ActionInstance, ...]
    remaining_budget: float
    cost_preference: float


RestrictedController = Callable[
    [Any, tuple[ActionInstance, ...]], ActionInstance
]


class QuestionOnlyPolicy:
    def __init__(self, controller: RestrictedController) -> None:
        self.controller = controller

    def __call__(
        self, state: RouterState, legal_actions: tuple[ActionInstance, ...]
    ) -> ActionInstance:
        view = QuestionOnlyView(
            question=state.question,
            options=state.options,
            remaining_budget=state.remaining_budget,
            cost_preference=state.cost_preference,
        )
        return _exact(self.controller(view, legal_actions), legal_actions)


class TextAdaptivePolicy:
    """Text-only adaptive baseline with no frontier/fidelity/multimodal view."""

    def __init__(self, controller: RestrictedController) -> None:
        self.controller = controller

    def __call__(
        self, state: RouterState, legal_actions: tuple[ActionInstance, ...]
    ) -> ActionInstance:
        view = TextAdaptiveView(
            question=state.question,
            options=state.options,
            gist_text=tuple(
                item.content
                for item in state.evidence
                if item.fidelity_level is FidelityLevel.GIST
            ),
            action_history=state.action_history,
            remaining_budget=state.remaining_budget,
            cost_preference=state.cost_preference,
        )
        return _exact(self.controller(view, legal_actions), legal_actions)


class BCPolicyAdapter:
    """Exact-action guard for Task 10 ``BCPolicy`` compatible callables."""

    def __init__(self, policy: Callable[[RouterState, tuple[ActionInstance, ...]], ActionInstance]) -> None:
        self.policy = policy

    def __call__(
        self, state: RouterState, legal_actions: tuple[ActionInstance, ...]
    ) -> ActionInstance:
        return _exact(self.policy(state, legal_actions), legal_actions)


class DAggerPolicyAdapter(BCPolicyAdapter):
    """Task 11 adapter that binds the final round manifest to a BC policy."""

    def __init__(
        self,
        policy: Callable[[RouterState, tuple[ActionInstance, ...]], ActionInstance],
        *,
        manifest: object,
    ) -> None:
        from fidmem.router.dagger import DAggerRoundManifest

        if not isinstance(manifest, DAggerRoundManifest):
            raise TypeError("DAgger adapter requires a DAggerRoundManifest")
        if manifest.status != "stopped":
            raise ValueError("DAgger evaluation requires the final stopped manifest")
        super().__init__(policy)
        self.manifest = manifest
        self.policy_identity = manifest.checkpoint_policy_identity


__all__ = [
    "BCPolicyAdapter",
    "BaselinePolicyError",
    "ControllerCost",
    "DAggerPolicyAdapter",
    "FullResidualPolicy",
    "GistOnlyPolicy",
    "GistResidualPolicy",
    "GistVisualPolicy",
    "PromptControllerDecision",
    "PromptControllerPolicy",
    "QuestionOnlyPolicy",
    "QuestionOnlyView",
    "RulePolicy",
    "TextAdaptivePolicy",
    "TextAdaptiveView",
    "UniformFramesPolicy",
]
