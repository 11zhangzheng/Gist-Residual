"""Immutable domain contracts for routing over fidelity-graded memory."""

from enum import Enum
from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


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
    """Canonical, immutable event memory record.

    The validation aliases keep Task 1's field names accepted at input while
    all serialization uses the design-spec names.
    """

    model_config = ConfigDict(frozen=True)

    video_id: str
    event_id: str
    start_sec: float = Field(
        default=0.0,
        ge=0,
        validation_alias=AliasChoices("start_sec", "start_seconds"),
    )
    end_sec: float = Field(
        default=0.0,
        ge=0,
        validation_alias=AliasChoices("end_sec", "end_seconds"),
    )
    asr_text: str = ""
    keyframe_paths: tuple[str, ...] = ()
    visual_embedding: tuple[float, ...] = ()
    text_embedding: tuple[float, ...] = ()
    gist_text: str = Field(
        default="",
        validation_alias=AliasChoices("gist_text", "gist"),
    )
    raw_video_uri: str = ""
    memory_version: str = "unknown"
    residual: str | None = None

    @model_validator(mode="after")
    def end_must_not_precede_start(self) -> "EventRecord":
        if self.end_sec < self.start_sec:
            raise ValueError("end_sec must be at least start_sec")
        return self

    @property
    def start_seconds(self) -> float:
        """Read-only compatibility alias for the Task 1 field name."""
        return self.start_sec

    @property
    def end_seconds(self) -> float:
        """Read-only compatibility alias for the Task 1 field name."""
        return self.end_sec

    @property
    def gist(self) -> str:
        """Read-only compatibility alias for the Task 1 field name."""
        return self.gist_text


class EvidenceItem(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    event_id: str
    fidelity_level: FidelityLevel
    content: str
    score: float
    start_sec: float = Field(default=0.0, ge=0)
    acquisition_step: int = Field(default=0, ge=0)
    attachments: tuple[str, ...] = ()

    @model_validator(mode="after")
    def attachments_must_belong_to_visual_evidence(self) -> "EvidenceItem":
        if self.attachments and self.fidelity_level is not FidelityLevel.VISUAL:
            raise ValueError("attachments are only permitted for visual evidence")
        return self



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
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

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
