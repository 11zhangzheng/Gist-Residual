"""Fair baseline policies over the exact hard-masked action tuple."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
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


def _event_rank(event_id: str | None, event_order: Sequence[str]) -> tuple[int, str]:
    if event_id is None:
        return (-1, "")
    try:
        return (event_order.index(event_id), event_id)
    except ValueError:
        return (len(event_order), event_id)


def _ordered(
    legal_actions: tuple[ActionInstance, ...],
    action_type: ActionType,
    *,
    visual_budget: Literal["low", "high"] | None = None,
    event_order: Sequence[str] = (),
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
                _event_rank(action.event_id, event_order),
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
    event_order: Sequence[str] = (),
) -> ActionInstance | None:
    values = _ordered(
        legal_actions,
        action_type,
        visual_budget=visual_budget,
        event_order=event_order,
    )
    return values[0] if values else None


def _stop_or_raise(legal_actions: tuple[ActionInstance, ...]) -> ActionInstance:
    stop = _first(legal_actions, ActionType.STOP)
    if stop is None:
        raise BaselinePolicyError(
            "preferred action is unavailable and STOP is not legal"
        )
    return stop


def select_legal_action_type(
    predicted: ActionType,
    legal_actions: tuple[ActionInstance, ...],
) -> ActionInstance:
    """Runner-internal deterministic mapping from a blind type to an exact mask item."""

    if not isinstance(predicted, ActionType):
        raise BaselinePolicyError("restricted controller must return ActionType")
    selected = _first(legal_actions, predicted)
    if selected is not None:
        return selected
    return _stop_or_raise(legal_actions)


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
            or _first(legal_actions, ActionType.VERIFY_VISUAL, visual_budget="high")
            or _stop_or_raise(legal_actions)
        )


class UniformFramesPolicy:
    """Discover the full authority order, then low-verify every event in time order."""

    def __init__(self, event_order: Sequence[str] = ()) -> None:
        self.event_order = tuple(event_order)

    def for_event_order(self, event_order: Sequence[str]) -> "UniformFramesPolicy":
        return UniformFramesPolicy(event_order)

    def __call__(
        self, state: RouterState, legal_actions: tuple[ActionInstance, ...]
    ) -> ActionInstance:
        search = _first(legal_actions, ActionType.SEARCH_GIST)
        if search is not None:
            return search
        if self.event_order and not set(self.event_order).issubset(
            state.candidate_event_ids
        ):
            anchor = state.candidate_event_ids[0] if state.candidate_event_ids else None
            context = next(
                (
                    action
                    for action in legal_actions
                    if action.action_type is ActionType.EXPAND_CONTEXT
                    and action.event_id == anchor
                ),
                None,
            ) or _first(
                legal_actions,
                ActionType.EXPAND_CONTEXT,
                event_order=self.event_order,
            )
            return context or _stop_or_raise(legal_actions)
        return _first(
            legal_actions,
            ActionType.VERIFY_VISUAL,
            visual_budget="low",
            event_order=self.event_order,
        ) or _stop_or_raise(legal_actions)


class FullResidualPolicy:
    """Discover every event through CONTEXT, then materialize all Residuals."""

    def __init__(self, event_order: Sequence[str] = ()) -> None:
        self.event_order = tuple(event_order)

    def for_event_order(self, event_order: Sequence[str]) -> "FullResidualPolicy":
        return FullResidualPolicy(event_order)

    def __call__(
        self, state: RouterState, legal_actions: tuple[ActionInstance, ...]
    ) -> ActionInstance:
        search = _first(legal_actions, ActionType.SEARCH_GIST)
        if search is not None:
            return search
        if self.event_order and not set(self.event_order).issubset(
            state.candidate_event_ids
        ):
            anchor = state.candidate_event_ids[0] if state.candidate_event_ids else None
            context = next(
                (
                    action
                    for action in legal_actions
                    if action.action_type is ActionType.EXPAND_CONTEXT
                    and action.event_id == anchor
                ),
                None,
            ) or _first(
                legal_actions,
                ActionType.EXPAND_CONTEXT,
                event_order=self.event_order,
            )
            return context or _stop_or_raise(legal_actions)
        return _first(
            legal_actions,
            ActionType.EXPAND_RESIDUAL,
            event_order=self.event_order,
        ) or _stop_or_raise(legal_actions)


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
        if (
            state.evidence
            and max(item.score for item in state.evidence) >= self.sufficiency_threshold
        ):
            return _stop_or_raise(legal_actions)
        return (
            _first(legal_actions, ActionType.EXPAND_RESIDUAL)
            or _first(legal_actions, ActionType.VERIFY_VISUAL, visual_budget="low")
            or _first(legal_actions, ActionType.EXPAND_CONTEXT)
            or _stop_or_raise(legal_actions)
        )


class ControllerCost(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        strict=True,
        revalidate_instances="always",
    )

    total_cost: float = Field(default=0.0, ge=0)
    gpu_seconds: float = Field(default=0.0, ge=0)
    wall_seconds: float = Field(default=0.0, ge=0)
    input_frames: int = Field(default=0, ge=0)
    visual_tokens: int = Field(default=0, ge=0)
    text_tokens: int = Field(default=0, ge=0)
    peak_memory_bytes: int = Field(default=0, ge=0)


class PromptControllerDecision(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        revalidate_instances="always",
    )

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
        validated = PromptControllerDecision.model_validate(
            decision.model_dump(mode="python")
        )
        self._pending_cost = validated.cost
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
    action_history: tuple[ActionType, ...]
    remaining_budget: float
    cost_preference: float


RestrictedController = Callable[[Any], ActionType]


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
        return select_legal_action_type(self.controller(view), legal_actions)


class TextAdaptivePolicy:
    """Text-only adaptive baseline with no legal/candidate/fidelity/frontier view."""

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
            action_history=tuple(action.action_type for action in state.action_history),
            remaining_budget=state.remaining_budget,
            cost_preference=state.cost_preference,
        )
        return select_legal_action_type(self.controller(view), legal_actions)


class BCPolicyAdapter:
    """Bind an actual Task 10 BCPolicy to its checkpoint content identity."""

    def __init__(self, policy: object, *, checkpoint: str | Path) -> None:
        from fidmem.router.dagger_core import BCPolicy, policy_identity

        if type(policy) is not BCPolicy:
            raise TypeError("BC adapter requires the actual Task 10 BCPolicy")
        self.policy = policy
        self.checkpoint = Path(checkpoint)
        self.policy_identity = policy_identity(policy, self.checkpoint)

    def __call__(
        self, state: RouterState, legal_actions: tuple[ActionInstance, ...]
    ) -> ActionInstance:
        return _exact(self.policy(state, legal_actions), legal_actions)  # type: ignore[operator]


class DAggerPolicyAdapter(BCPolicyAdapter):
    """Bind Task 11's final manifest to the actual Task 10 policy/checkpoint."""

    def __init__(
        self,
        policy: object,
        *,
        checkpoint: str | Path,
        manifest: object,
    ) -> None:
        from fidmem.router.dagger import DAggerRoundManifest

        if not isinstance(manifest, DAggerRoundManifest):
            raise TypeError("DAgger adapter requires a DAggerRoundManifest")
        validated = DAggerRoundManifest.model_validate(
            manifest.model_dump(mode="python")
        )
        if validated.status != "stopped":
            raise ValueError("DAgger evaluation requires the final stopped manifest")
        super().__init__(policy, checkpoint=checkpoint)
        if self.policy_identity != validated.checkpoint_policy_identity:
            raise ValueError("DAgger policy/checkpoint identity differs from manifest")
        if self.policy_identity.checkpoint_sha256 != validated.checkpoint.sha256:
            raise ValueError("DAgger checkpoint artifact differs from manifest")
        self.manifest = validated


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
    "select_legal_action_type",
]
