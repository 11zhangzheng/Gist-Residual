from __future__ import annotations

from collections.abc import Callable

import pytest

from fidmem.actions.environment import (
    ActionCostTable,
    ActionObservation,
    IllegalActionError,
    MemoryEnvironment,
    TerminalStateError,
)
from fidmem.types import (
    ActionInstance,
    ActionType,
    EventRecord,
    EvidenceItem,
    FidelityLevel,
    RouterState,
)


def _event(event_id: str, start_sec: float) -> EventRecord:
    return EventRecord(
        video_id="video", event_id=event_id, start_sec=start_sec, end_sec=start_sec + 5,
        gist_text=f"gist {event_id}", visual_embedding=(1.0,), text_embedding=(1.0,),
        raw_video_uri="video.mp4", memory_version="v1",
    )


def _state(*, budget: float = 20.0) -> RouterState:
    return RouterState(
        question="Which bottle?", options=("blue", "red"), evidence=(), action_history=(),
        remaining_budget=budget, candidate_event_ids=(), candidate_fidelity_levels={},
        context_frontiers={}, cost_preference=0.3,
    )


def _action(kind: ActionType, event_id: str | None = None, budget: str | None = None) -> ActionInstance:
    return ActionInstance(kind, event_id, budget)  # type: ignore[arg-type]


def _executor(calls: list[ActionInstance], *, cached_visual: bool = False) -> Callable[[ActionInstance, RouterState], ActionObservation]:
    def execute(action: ActionInstance, state: RouterState) -> ActionObservation:
        calls.append(action)
        if action.action_type is ActionType.SEARCH_GIST:
            return ActionObservation(candidate_event_ids=("e1", "e2"))
        if action.action_type is ActionType.EXPAND_CONTEXT:
            return ActionObservation(evidence=(EvidenceItem(event_id="e2", fidelity_level=FidelityLevel.GIST, content="nearby", score=1),))
        attachments = ("frame-1.jpg",) if action.action_type is ActionType.VERIFY_VISUAL else ()
        fidelity = FidelityLevel.VISUAL if attachments else FidelityLevel.RESIDUAL
        return ActionObservation(
            evidence=(EvidenceItem(event_id=action.event_id or "e1", fidelity_level=fidelity, content=action.action_type.value, score=1, attachments=attachments),),
            cache_status="hit" if cached_visual and action.action_type is ActionType.VERIFY_VISUAL else "miss",
            input_frames=0 if cached_visual and action.action_type is ActionType.VERIFY_VISUAL else len(attachments),
        )
    return execute


def _environment(calls: list[ActionInstance], *, budget: float = 20.0, cached_visual: bool = False) -> MemoryEnvironment:
    return MemoryEnvironment(
        events=(_event("e1", 0), _event("e2", 10)),
        executor=_executor(calls, cached_visual=cached_visual),
        costs=ActionCostTable(search_gist=1, residual=2, context=1, visual_low=4, visual_high=8, cache_hit=0),
    )


def test_hard_mask_tracks_residual_visual_context_and_budget_without_provider_io() -> None:
    calls: list[ActionInstance] = []
    environment = _environment(calls)
    initial = _state()

    assert environment.valid_actions(initial) == (_action(ActionType.SEARCH_GIST),)
    assert calls == []

    after_search = environment.step(initial, _action(ActionType.SEARCH_GIST)).next_state
    legal = environment.valid_actions(after_search)
    assert legal == (
        _action(ActionType.EXPAND_RESIDUAL, "e1"), _action(ActionType.EXPAND_CONTEXT, "e1"),
        _action(ActionType.VERIFY_VISUAL, "e1", "low"), _action(ActionType.VERIFY_VISUAL, "e1", "high"),
        _action(ActionType.EXPAND_RESIDUAL, "e2"), _action(ActionType.EXPAND_CONTEXT, "e2"),
        _action(ActionType.VERIFY_VISUAL, "e2", "low"), _action(ActionType.VERIFY_VISUAL, "e2", "high"),
        _action(ActionType.STOP),
    )
    after_residual = environment.step(after_search, _action(ActionType.EXPAND_RESIDUAL, "e1")).next_state
    assert _action(ActionType.EXPAND_RESIDUAL, "e1") not in environment.valid_actions(after_residual)
    after_low = environment.step(after_residual, _action(ActionType.VERIFY_VISUAL, "e1", "low")).next_state
    assert _action(ActionType.VERIFY_VISUAL, "e1", "low") not in environment.valid_actions(after_low)
    assert _action(ActionType.VERIFY_VISUAL, "e1", "high") in environment.valid_actions(after_low)
    after_context = environment.step(after_low, _action(ActionType.EXPAND_CONTEXT, "e1")).next_state
    assert _action(ActionType.EXPAND_CONTEXT, "e1") not in environment.valid_actions(after_context)
    assert calls == [
        _action(ActionType.SEARCH_GIST), _action(ActionType.EXPAND_RESIDUAL, "e1"),
        _action(ActionType.VERIFY_VISUAL, "e1", "low"), _action(ActionType.EXPAND_CONTEXT, "e1"),
    ]


def test_environment_rejects_illegal_repeat_and_terminal_actions_and_charges_once() -> None:
    calls: list[ActionInstance] = []
    environment = _environment(calls, cached_visual=True)
    initial = _state()
    with pytest.raises(IllegalActionError, match="not legal"):
        environment.step(initial, _action(ActionType.EXPAND_RESIDUAL, "e1"))
    searched = environment.step(initial, _action(ActionType.SEARCH_GIST))
    assert searched.step_cost == 1
    visual = environment.step(searched.next_state, _action(ActionType.VERIFY_VISUAL, "e1", "low"))
    assert visual.step_cost == 0
    assert visual.next_state.remaining_budget == 19
    with pytest.raises(IllegalActionError, match="not legal"):
        environment.step(visual.next_state, _action(ActionType.VERIFY_VISUAL, "e1", "low"))
    stopped = environment.step(visual.next_state, _action(ActionType.STOP))
    assert stopped.terminal is True
    with pytest.raises(TerminalStateError):
        environment.step(stopped.next_state, _action(ActionType.STOP))


def test_budget_mask_and_evidence_reducer_preserve_time_step_and_visual_attachments() -> None:
    calls: list[ActionInstance] = []
    environment = _environment(calls)
    searched = environment.step(_state(budget=3), _action(ActionType.SEARCH_GIST)).next_state
    legal = environment.valid_actions(searched)
    assert _action(ActionType.EXPAND_RESIDUAL, "e1") in legal
    assert _action(ActionType.VERIFY_VISUAL, "e1", "low") not in legal
    residual = environment.step(searched, _action(ActionType.EXPAND_RESIDUAL, "e1"))
    evidence = residual.next_state.evidence[-1]
    assert (evidence.start_sec, evidence.acquisition_step, evidence.attachments) == (0, 2, ())


def test_context_remains_legal_until_all_adjacent_rings_are_visited() -> None:
    calls: list[ActionInstance] = []
    environment = MemoryEnvironment(
        events=(_event("e1", 0), _event("e2", 10), _event("e3", 20)),
        executor=_executor(calls),
        costs=ActionCostTable(search_gist=1, residual=2, context=1, visual_low=4, visual_high=8, cache_hit=0),
    )
    searched = environment.step(_state(), _action(ActionType.SEARCH_GIST)).next_state
    first = environment.step(searched, _action(ActionType.EXPAND_CONTEXT, "e1")).next_state
    assert _action(ActionType.EXPAND_CONTEXT, "e1") in environment.valid_actions(first)
    second = environment.step(first, _action(ActionType.EXPAND_CONTEXT, "e1")).next_state
    assert _action(ActionType.EXPAND_CONTEXT, "e1") not in environment.valid_actions(second)
