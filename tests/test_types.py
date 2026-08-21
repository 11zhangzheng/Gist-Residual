import pytest
from pydantic import ValidationError

from fidmem.types import ActionInstance, ActionType, FidelityLevel, RouterState


def test_router_state_rejects_unknown_candidate_fidelity() -> None:
    state = RouterState(
        question="What color is the bottle?",
        options=("red", "blue"),
        evidence=(),
        action_history=(),
        remaining_budget=1.0,
        candidate_event_ids=("e1",),
        candidate_fidelity_levels={"e1": FidelityLevel.GIST},
        context_frontiers={"e1": (0, 0)},
        cost_preference=0.3,
    )

    assert state.candidate_fidelity_levels["e1"] is FidelityLevel.GIST
    assert ActionInstance(ActionType.EXPAND_RESIDUAL, "e1", None).event_id == "e1"


def test_router_state_rejects_candidate_metadata_for_unknown_event() -> None:
    with pytest.raises(ValidationError, match="candidate_event_ids"):
        RouterState(
            question="What color is the bottle?",
            options=("red", "blue"),
            evidence=(),
            action_history=(),
            remaining_budget=1.0,
            candidate_event_ids=("e1",),
            candidate_fidelity_levels={"e2": FidelityLevel.GIST},
            context_frontiers={"e1": (0, 0)},
            cost_preference=0.3,
        )


def test_router_state_rejects_negative_budget() -> None:
    with pytest.raises(ValidationError, match="remaining_budget"):
        RouterState(
            question="What color is the bottle?",
            options=("red", "blue"),
            evidence=(),
            action_history=(),
            remaining_budget=-0.1,
            candidate_event_ids=(),
            candidate_fidelity_levels={},
            context_frontiers={},
            cost_preference=0.3,
        )
