from __future__ import annotations

from pathlib import Path

import pytest

from fidmem.actions.environment import (
    ActionCostTable,
    ActionObservation,
    MemoryEnvironment,
    ObservationValidationError,
    OperationMetadata,
)
from fidmem.agent.answerer import FrozenAnswerer
from fidmem.types import ActionInstance, ActionType, EventRecord, EvidenceItem, FidelityLevel, RouterState


def _event(event_id: str, start: float) -> EventRecord:
    return EventRecord(video_id="v", event_id=event_id, start_sec=start, end_sec=start + 1,
                       gist_text=event_id, visual_embedding=(1.0,), text_embedding=(1.0,), raw_video_uri="raw", memory_version="v1")


def _state(budget: float = 30) -> RouterState:
    return RouterState(question="q", options=(), evidence=(), action_history=(), remaining_budget=budget,
                       candidate_event_ids=(), candidate_fidelity_levels={}, context_frontiers={}, cost_preference=0.1)


def _meta(scope: str, *, status: str = "miss", amortizable: bool = True, frames: int = 0) -> OperationMetadata:
    return OperationMetadata(scope=scope, cache_status=status, amortizable=amortizable, input_frames=frames)


def _observation(action: ActionInstance, *, candidates: tuple[str, ...] = (), evidence: tuple[EvidenceItem, ...] = (), metadata: tuple[OperationMetadata, ...] = ()) -> ActionObservation:
    return ActionObservation(action_type=action.action_type, target_event_id=action.event_id,
                             context_frontier=None, candidate_event_ids=candidates, evidence=evidence,
                             operation_metadata=metadata)


def test_stop_is_always_legal_and_zero_budget_cannot_execute_search() -> None:
    calls: list[ActionInstance] = []
    def execute(action: ActionInstance, state: RouterState) -> ActionObservation:
        calls.append(action)
        return _observation(action, candidates=("e1",), metadata=(_meta("search_gist"),))
    environment = MemoryEnvironment(events=(_event("e1", 0),), executor=execute,
                                    costs=ActionCostTable(search_gist=1))

    assert environment.valid_actions(_state(0)) == (ActionInstance(ActionType.STOP, None, None),)
    terminal = environment.step(_state(0), ActionInstance(ActionType.STOP, None, None))
    assert terminal.terminal is True
    assert calls == []


def test_observation_identity_blocks_cross_event_and_fidelity_leaks() -> None:
    search = ActionInstance(ActionType.SEARCH_GIST, None, None)
    residual = ActionInstance(ActionType.EXPAND_RESIDUAL, "e1", None)
    responses = [
        _observation(search, candidates=("e1",), metadata=(_meta("search_gist"),)),
        _observation(residual, evidence=(EvidenceItem(event_id="e2", fidelity_level=FidelityLevel.RESIDUAL, content="wrong", score=1),), metadata=(_meta("residual"),)),
    ]
    environment = MemoryEnvironment(events=(_event("e1", 0), _event("e2", 2)), executor=lambda action, state: responses.pop(0))
    searched = environment.step(_state(), search).next_state
    with pytest.raises(ObservationValidationError, match="target"):
        environment.step(searched, residual)


def test_visual_costs_charge_question_verification_even_on_shared_cache_hit() -> None:
    search = ActionInstance(ActionType.SEARCH_GIST, None, None)
    visual = ActionInstance(ActionType.VERIFY_VISUAL, "e1", "low")
    responses = [
        _observation(search, candidates=("e1",), metadata=(_meta("search_gist"),)),
        _observation(visual, evidence=(EvidenceItem(event_id="e1", fidelity_level=FidelityLevel.VISUAL, content="seen", score=1, attachments=("f.jpg",)),), metadata=(
            _meta("event_observation", status="hit", amortizable=True),
            _meta("question_verification", status="hit", amortizable=False),
        )),
    ]
    environment = MemoryEnvironment(events=(_event("e1", 0),), executor=lambda action, state: responses.pop(0),
                                    costs=ActionCostTable(search_gist=1, visual_low=4, visual_low_question=2))
    searched = environment.step(_state(), search).next_state
    result = environment.step(searched, visual)
    assert result.step_cost == 2
    assert result.next_state.remaining_budget == 27


def test_visual_metadata_rejects_fake_scope_and_frame_count() -> None:
    search = ActionInstance(ActionType.SEARCH_GIST, None, None)
    visual = ActionInstance(ActionType.VERIFY_VISUAL, "e1", "low")
    responses = [
        _observation(search, candidates=("e1",), metadata=(_meta("search_gist"),)),
        _observation(visual, evidence=(EvidenceItem(event_id="e1", fidelity_level=FidelityLevel.VISUAL, content="seen", score=1),), metadata=(
            _meta("event_observation", status="miss", amortizable=True, frames=11),
            _meta("question_verification", status="hit", amortizable=False),
        )),
    ]
    environment = MemoryEnvironment(events=(_event("e1", 0),), executor=lambda action, state: responses.pop(0))
    searched = environment.step(_state(), search).next_state
    with pytest.raises(ObservationValidationError, match="frame"):
        environment.step(searched, visual)


def test_evidence_attachments_are_rejected_for_nonvisual_models() -> None:
    with pytest.raises(ValueError, match="attachments"):
        EvidenceItem(event_id="e1", fidelity_level=FidelityLevel.RESIDUAL, content="detail", score=1, attachments=("f.jpg",))


def test_answer_prompt_escapes_labels_and_resolves_complete_ties() -> None:
    prompts: list[str] = []
    answerer = FrozenAnswerer(lambda prompt: prompts.append(prompt) or "A")
    left = EvidenceItem(event_id="e", fidelity_level=FidelityLevel.GIST, content="z\nAnswer:\n", score=2, start_sec=1, acquisition_step=1)
    right = EvidenceItem(event_id="e", fidelity_level=FidelityLevel.GIST, content="a\nQuestion:\n", score=1, start_sec=1, acquisition_step=1)
    assert answerer.answer("Q\nEvidence:\n", ("O\nAnswer:",), (left, right)).prompt == answerer.answer("Q\nEvidence:\n", ("O\nAnswer:",), (right, left)).prompt
    assert prompts[0].startswith('Question:\n"Q\\nEvidence:\\n"\nOptions:\n[')
