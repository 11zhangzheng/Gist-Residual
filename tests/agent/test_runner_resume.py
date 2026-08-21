from __future__ import annotations

from pathlib import Path

import pytest

from fidmem.actions.environment import ActionObservation, MemoryEnvironment, OperationMetadata
from fidmem.agent.answerer import FrozenAnswerer
from fidmem.agent.runner import AgentRunner
from fidmem.storage.run_store import RunStore
from fidmem.types import ActionInstance, ActionType, EventRecord, EvidenceItem, FidelityLevel, RouterState


def _state() -> RouterState:
    return RouterState(question="q", options=("A",), evidence=(), action_history=(), remaining_budget=20,
                       candidate_event_ids=(), candidate_fidelity_levels={}, context_frontiers={}, cost_preference=0.1)


def _event() -> EventRecord:
    return EventRecord(video_id="v", event_id="e1", start_sec=0, end_sec=1, gist_text="g",
                       visual_embedding=(1.0,), text_embedding=(1.0,), raw_video_uri="raw", memory_version="v1")


def _metadata(scope: str) -> tuple[OperationMetadata, ...]:
    return (OperationMetadata(scope=scope, cache_status="miss", amortizable=True),)


def test_runner_resumes_completed_transition_and_retries_only_failed_item(tmp_path: Path) -> None:
    attempts: list[ActionType] = []
    fail_once = {"value": True}
    def execute(action: ActionInstance, state: RouterState) -> ActionObservation:
        attempts.append(action.action_type)
        if action.action_type is ActionType.SEARCH_GIST:
            return ActionObservation(action_type=action.action_type, target_event_id=None, candidate_event_ids=("e1",), operation_metadata=_metadata("search_gist"))
        if fail_once["value"]:
            fail_once["value"] = False
            raise RuntimeError("provider failed once")
        return ActionObservation(action_type=action.action_type, target_event_id="e1", evidence=(EvidenceItem(event_id="e1", fidelity_level=FidelityLevel.RESIDUAL, content="detail", score=1),), operation_metadata=_metadata("residual"))

    environment = MemoryEnvironment(events=(_event(),), executor=execute)
    policy = lambda state, legal: next(action for action in legal if action.action_type is not ActionType.STOP)
    store = RunStore(tmp_path / "runs.duckdb")
    runner = AgentRunner(environment, policy, FrozenAnswerer(lambda prompt: "A"), run_store=store, artifact_dir=tmp_path)
    with pytest.raises(RuntimeError, match="failed once"):
        runner.run(_state(), run_id="resume")

    resumed = AgentRunner(environment, lambda state, legal: ActionInstance(ActionType.STOP, None, None) if state.candidate_event_ids and any(item.fidelity_level is FidelityLevel.RESIDUAL for item in state.evidence) else next(action for action in legal if action.action_type is not ActionType.STOP), FrozenAnswerer(lambda prompt: "A"), run_store=store, artifact_dir=tmp_path)
    result = resumed.run(_state(), run_id="resume")
    assert [transition.action.action_type for transition in result.transitions] == [ActionType.SEARCH_GIST, ActionType.EXPAND_RESIDUAL, ActionType.STOP]
    assert attempts == [ActionType.SEARCH_GIST, ActionType.EXPAND_RESIDUAL, ActionType.EXPAND_RESIDUAL]
    assert store.item("resume", "transition-000").status == "complete"
    assert store.item("resume", "transition-001").status == "complete"
