from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from fidmem.actions.environment import ActionObservation, MemoryEnvironment, OperationMetadata
from fidmem.agent.answerer import FrozenAnswerer
from fidmem.agent.runner import AgentRunner, ResumeValidationError
from fidmem.storage.run_store import RunStore
from fidmem.types import ActionInstance, ActionType, EventRecord, EvidenceItem, FidelityLevel, RouterState


def _state(question: str = "q") -> RouterState:
    return RouterState(
        question=question,
        options=("A",),
        evidence=(),
        action_history=(),
        remaining_budget=20,
        candidate_event_ids=(),
        candidate_fidelity_levels={},
        context_frontiers={},
        cost_preference=0.1,
    )


def _event() -> EventRecord:
    return EventRecord(
        video_id="v",
        event_id="e1",
        start_sec=0,
        end_sec=1,
        gist_text="g",
        visual_embedding=(1.0,),
        text_embedding=(1.0,),
        raw_video_uri="raw",
        memory_version="v1",
    )


def _search_observation() -> ActionObservation:
    return ActionObservation(
        action_type=ActionType.SEARCH_GIST,
        target_event_id=None,
        candidate_event_ids=("e1",),
        operation_metadata=(
            OperationMetadata(scope="search_gist", cache_status="miss", amortizable=True),
        ),
    )


def _runner(
    tmp_path: Path,
    store: RunStore,
    executor: Callable[[ActionInstance, RouterState], ActionObservation],
    policy: Callable[[RouterState, tuple[ActionInstance, ...]], ActionInstance],
    answer: str = "A",
) -> AgentRunner:
    return AgentRunner(
        MemoryEnvironment(events=(_event(),), executor=executor),
        policy,
        FrozenAnswerer(lambda prompt: answer),
        run_store=store,
        artifact_dir=tmp_path,
    )


def _search_then_fail_artifact(tmp_path: Path) -> tuple[RunStore, Path]:
    store = RunStore(tmp_path / "runs.duckdb")
    calls = 0

    def execute(action: ActionInstance, state: RouterState) -> ActionObservation:
        nonlocal calls
        calls += 1
        if action.action_type is ActionType.SEARCH_GIST:
            return _search_observation()
        raise RuntimeError("expected later failure")

    policy = lambda state, legal: next(action for action in legal if action.action_type is not ActionType.STOP)
    with pytest.raises(RuntimeError, match="expected later failure"):
        _runner(tmp_path, store, execute, policy).run(_state(), run_id="tamper")
    assert calls == 2
    item = store.item("tamper", "transition-000")
    assert item is not None and item.status == "complete" and item.output_uri is not None
    failed = store.item("tamper", "transition-001")
    assert failed is not None and failed.status == "failed"
    return store, Path(item.output_uri)


def _mutate_budget(payload: dict[str, object]) -> None:
    payload["next_state"]["remaining_budget"] = 18  # type: ignore[index]


def _mutate_cost(payload: dict[str, object]) -> None:
    payload["step_cost"] = 2


def _mutate_target(payload: dict[str, object]) -> None:
    payload["observation"]["target_event_id"] = "e1"  # type: ignore[index]


def _mutate_scope(payload: dict[str, object]) -> None:
    payload["observation"]["operation_metadata"][0]["scope"] = "residual"  # type: ignore[index]


def _mutate_cache(payload: dict[str, object]) -> None:
    payload["observation"]["operation_metadata"][0]["cache_status"] = "hit"  # type: ignore[index]


def _mutate_nan(payload: dict[str, object]) -> None:
    payload["next_state"]["remaining_budget"] = float("nan")  # type: ignore[index]


def _mutate_inf(payload: dict[str, object]) -> None:
    payload["step_cost"] = float("inf")


@pytest.mark.parametrize(
    "mutate",
    [_mutate_budget, _mutate_cost, _mutate_target, _mutate_scope, _mutate_cache, _mutate_nan, _mutate_inf],
    ids=["remaining-budget", "step-cost", "target", "scope", "cache-status", "nan", "inf"],
)
def test_runner_rejects_each_tampered_transition_and_resumes_original_without_reexecution(
    tmp_path: Path, mutate: Callable[[dict[str, object]], None]
) -> None:
    store, artifact = _search_then_fail_artifact(tmp_path)
    original = artifact.read_bytes()
    payload = json.loads(original)
    mutate(payload)
    artifact.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")
    resume_calls = 0

    def execute(action: ActionInstance, state: RouterState) -> ActionObservation:
        nonlocal resume_calls
        resume_calls += 1
        return ActionObservation(
            action_type=ActionType.EXPAND_RESIDUAL,
            target_event_id="e1",
            evidence=(
                EvidenceItem(
                    event_id="e1",
                    fidelity_level=FidelityLevel.RESIDUAL,
                    content="detail",
                    score=1,
                ),
            ),
            operation_metadata=(
                OperationMetadata(scope="residual", cache_status="miss", amortizable=True),
            ),
        )

    stop_after_residual = lambda state, legal: (
        ActionInstance(ActionType.STOP, None, None)
        if any(item.fidelity_level is FidelityLevel.RESIDUAL for item in state.evidence)
        else next(action for action in legal if action.action_type is ActionType.EXPAND_RESIDUAL)
    )
    runner = _runner(tmp_path, store, execute, stop_after_residual)
    with pytest.raises(ResumeValidationError):
        runner.run(_state(), run_id="tamper")
    assert resume_calls == 0

    artifact.write_bytes(original)
    result = runner.run(_state(), run_id="tamper")
    assert [transition.action.action_type for transition in result.transitions] == [
        ActionType.SEARCH_GIST,
        ActionType.EXPAND_RESIDUAL,
        ActionType.STOP,
    ]
    assert resume_calls == 1


def test_completed_answer_is_bound_to_its_run_and_final_state(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "answers.duckdb")
    policy = lambda state, legal: (
        next(action for action in legal if action.action_type is ActionType.SEARCH_GIST)
        if not state.candidate_event_ids
        else ActionInstance(ActionType.STOP, None, None)
    )
    runner = _runner(tmp_path, store, lambda action, state: _search_observation(), policy)
    runner.run(_state(), run_id="left")
    runner.run(_state(), run_id="right")

    left = store.item("left", "answer")
    right = store.item("right", "answer")
    assert left is not None and left.output_uri is not None
    assert right is not None and right.output_uri is not None
    Path(left.output_uri).write_bytes(Path(right.output_uri).read_bytes())

    with pytest.raises(ResumeValidationError, match="answer artifact"):
        runner.run(_state(), run_id="left")


def test_legacy_unbound_answer_artifact_is_rejected_explicitly(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "legacy.duckdb")
    policy = lambda state, legal: (
        next(action for action in legal if action.action_type is ActionType.SEARCH_GIST)
        if not state.candidate_event_ids
        else ActionInstance(ActionType.STOP, None, None)
    )
    runner = _runner(tmp_path, store, lambda action, state: _search_observation(), policy)
    runner.run(_state(), run_id="legacy")
    answer = store.item("legacy", "answer")
    assert answer is not None and answer.output_uri is not None
    Path(answer.output_uri).write_text('{"answer":"A","prompt":"old"}', encoding="utf-8")

    with pytest.raises(ResumeValidationError, match="answer artifact"):
        runner.run(_state(), run_id="legacy")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gpu_seconds", float("inf")),
        ("gpu_seconds", float("nan")),
        ("wall_seconds", float("inf")),
        ("wall_seconds", float("nan")),
    ],
)
def test_runner_rejects_non_finite_nested_cost_records(
    tmp_path: Path, field: str, value: float
) -> None:
    store, artifact = _search_then_fail_artifact(tmp_path)
    payload = json.loads(artifact.read_bytes())
    cost_record = {
        "operation": "search",
        "gpu_seconds": 0.1,
        "wall_seconds": 0.2,
        "input_frames": 0,
        "visual_tokens": 0,
        "text_tokens": 1,
        "peak_memory_bytes": 0,
        "cache_status": "miss",
        "device_name": "cpu",
    }
    for metadata in (
        payload["observation"]["operation_metadata"][0],
        payload["operation_metadata"][0],
    ):
        metadata["cost_record"] = dict(cost_record)
        metadata["cost_record"][field] = value
    artifact.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")

    runner = _runner(
        tmp_path,
        store,
        lambda action, state: pytest.fail("executor must not run while restoring"),
        lambda state, legal: pytest.fail("policy must not run while restoring"),
    )
    with pytest.raises(ResumeValidationError):
        runner.run(_state(), run_id="tamper")
