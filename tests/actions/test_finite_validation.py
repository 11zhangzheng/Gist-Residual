from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from fidmem.actions.environment import ActionCostTable, EnvironmentTransition, OperationMetadata
from fidmem.types import EvidenceItem, FidelityLevel, RouterState


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_router_state_rejects_non_finite_budget_and_cost_preference(value: float) -> None:
    base = {
        "question": "q",
        "options": (),
        "evidence": (),
        "action_history": (),
        "remaining_budget": 1.0,
        "candidate_event_ids": (),
        "candidate_fidelity_levels": {},
        "context_frontiers": {},
        "cost_preference": 0.5,
    }
    for field in ("remaining_budget", "cost_preference"):
        with pytest.raises(ValidationError):
            RouterState.model_validate({**base, field: value})


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_evidence_and_action_costs_reject_non_finite_values(value: float) -> None:
    with pytest.raises(ValidationError):
        EvidenceItem(event_id="e", fidelity_level=FidelityLevel.GIST, content="g", score=value)
    for field in ActionCostTable.model_fields:
        with pytest.raises(ValidationError):
            ActionCostTable.model_validate({field: value})


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_environment_transition_rejects_non_finite_step_cost(value: float) -> None:
    with pytest.raises(ValidationError):
        EnvironmentTransition.model_validate(
            {
                "state": {
                    "question": "q",
                    "options": [],
                    "evidence": [],
                    "action_history": [],
                    "remaining_budget": 1,
                    "candidate_event_ids": [],
                    "candidate_fidelity_levels": {},
                    "context_frontiers": {},
                    "cost_preference": 0.5,
                },
                "action": {"action_type": "STOP", "event_id": None, "visual_budget": None},
                "observation": {"action_type": "STOP"},
                "next_state": {
                    "question": "q",
                    "options": [],
                    "evidence": [],
                    "action_history": [
                        {"action_type": "STOP", "event_id": None, "visual_budget": None}
                    ],
                    "remaining_budget": 1,
                    "candidate_event_ids": [],
                    "candidate_fidelity_levels": {},
                    "context_frontiers": {},
                    "cost_preference": 0.5,
                },
                "step_cost": value,
                "terminal": True,
            }
        )


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_operation_metadata_rejects_non_finite_token_and_frame_counts(value: float) -> None:
    base = {"scope": "search_gist", "cache_status": "miss", "amortizable": True}
    for field in ("input_frames", "visual_tokens", "text_tokens"):
        with pytest.raises(ValidationError):
            OperationMetadata.model_validate({**base, field: value})
