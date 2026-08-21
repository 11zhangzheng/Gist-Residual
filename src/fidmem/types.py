"""Immutable domain contracts for routing over fidelity-graded memory."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ActionType(str, Enum):
    SEARCH_GIST = "SEARCH_GIST"
    EXPAND_RESIDUAL = "EXPAND_RESIDUAL"
    EXPAND_CONTEXT = "EXPAND_CONTEXT"
    VERIFY_VISUAL = "VERIFY_VISUAL"
    STOP = "STOP"


class FidelityLevel(str, Enum):
    GIST = "GIST"
    RESIDUAL = "RESIDUAL"
    VISUAL = "VISUAL"


class EventRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_id: str
    video_id: str
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)
    gist: str
    residual: str | None = None

    @model_validator(mode="after")
    def end_must_not_precede_start(self) -> "EventRecord":
        if self.end_seconds < self.start_seconds:
            raise ValueError("end_seconds must be at least start_seconds")
        return self


class EvidenceItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_id: str
    fidelity_level: FidelityLevel
    content: str
    score: float


class ActionInstance(BaseModel):
    model_config = ConfigDict(frozen=True)

    action_type: ActionType
    event_id: str | None
    visual_budget: Literal["low", "high"] | None

    def __init__(
        self,
        action_type: ActionType | str,
        event_id: str | None,
        visual_budget: Literal["low", "high"] | None,
        **data: object,
    ) -> None:
        super().__init__(
            action_type=action_type,
            event_id=event_id,
            visual_budget=visual_budget,
            **data,
        )


class RouterState(BaseModel):
    model_config = ConfigDict(frozen=True)

    question: str
    options: tuple[str, ...]
    evidence: tuple[EvidenceItem, ...]
    action_history: tuple[ActionInstance, ...]
    remaining_budget: float = Field(ge=0)
    candidate_event_ids: tuple[str, ...]
    candidate_fidelity_levels: dict[str, FidelityLevel]
    context_frontiers: dict[str, tuple[int, int]]
    cost_preference: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def candidate_metadata_must_match_candidate_ids(self) -> "RouterState":
        candidate_ids = set(self.candidate_event_ids)
        if set(self.candidate_fidelity_levels) != candidate_ids:
            raise ValueError(
                "candidate_event_ids must exactly match candidate_fidelity_levels keys"
            )
        if set(self.context_frontiers) != candidate_ids:
            raise ValueError(
                "candidate_event_ids must exactly match context_frontiers keys"
            )
        return self


class Transition(BaseModel):
    model_config = ConfigDict(frozen=True)

    state: RouterState
    action: ActionInstance
    reward: float
    next_state: RouterState
    terminal: bool = False


class Trajectory(BaseModel):
    model_config = ConfigDict(frozen=True)

    transitions: tuple[Transition, ...]
    total_reward: float
