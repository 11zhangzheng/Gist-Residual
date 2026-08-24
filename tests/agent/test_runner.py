from __future__ import annotations

from pathlib import Path

import pytest

from fidmem.actions.environment import (
    ActionCostTable,
    ActionObservation,
    MemoryEnvironment,
)
from fidmem.agent.answerer import AnswererResponseError, FrozenAnswerer
from fidmem.agent.runner import AgentRunner, InvalidPolicyActionError
from fidmem.costs.tracker import CostRecord
from fidmem.storage.run_store import RunStore
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
        video_id="video",
        event_id=event_id,
        start_sec=start_sec,
        end_sec=start_sec + 1,
        gist_text=event_id,
        visual_embedding=(1.0,),
        text_embedding=(1.0,),
        raw_video_uri="raw.mp4",
        memory_version="v1",
    )


def _state() -> RouterState:
    return RouterState(
        question="What happened?",
        options=("A", "B"),
        evidence=(),
        action_history=(),
        remaining_budget=20,
        candidate_event_ids=(),
        candidate_fidelity_levels={},
        context_frontiers={},
        cost_preference=0.3,
    )


def _environment(*, fail: bool = False) -> MemoryEnvironment:
    def execute(action: ActionInstance, state: RouterState) -> ActionObservation:
        if fail:
            raise RuntimeError("provider unavailable")
        if action.action_type is ActionType.SEARCH_GIST:
            return ActionObservation(candidate_event_ids=("e1",))
        if action.action_type is ActionType.STOP:
            return ActionObservation()
        fidelity = (
            FidelityLevel.VISUAL
            if action.action_type is ActionType.VERIFY_VISUAL
            else FidelityLevel.RESIDUAL
        )
        return ActionObservation(
            evidence=(
                EvidenceItem(
                    event_id=action.event_id or "e1",
                    fidelity_level=fidelity,
                    content=action.action_type.value,
                    score=1,
                ),
            )
        )

    return MemoryEnvironment(
        events=(_event("e1", 2), _event("e2", 5)),
        executor=execute,
        costs=ActionCostTable(
            search_gist=1,
            residual=1,
            context=1,
            visual_low=1,
            visual_high=2,
            cache_hit=0,
        ),
    )


def test_answerer_prompt_is_strategy_independent_and_evidence_is_stably_sorted() -> (
    None
):
    prompts: list[str] = []
    answerer = FrozenAnswerer(lambda prompt: prompts.append(prompt) or "Answer: blue")
    evidence = (
        EvidenceItem(
            event_id="late",
            fidelity_level=FidelityLevel.GIST,
            content="late",
            score=1,
            start_sec=9,
            acquisition_step=1,
        ),
        EvidenceItem(
            event_id="early",
            fidelity_level=FidelityLevel.VISUAL,
            content="early",
            score=1,
            start_sec=1,
            acquisition_step=2,
            attachments=("frame.jpg",),
        ),
    )
    first = answerer.answer("What color?", ("blue", "red"), evidence)
    second = answerer.answer("What color?", ("blue", "red"), tuple(reversed(evidence)))

    assert first.answer == second.answer == "blue"
    assert prompts[0] == prompts[1]
    assert (
        prompts[0].index("Question:")
        < prompts[0].index("Options:")
        < prompts[0].index("Evidence:")
        < prompts[0].index("Answer:")
    )
    assert prompts[0].index('"event_id":"early"') < prompts[0].index(
        '"event_id":"late"'
    )
    assert '"attachments":["frame.jpg"]' in prompts[0]
    assert "policy" not in prompts[0].lower()


def test_answerer_rejects_empty_adapter_response() -> None:
    with pytest.raises(AnswererResponseError, match="empty"):
        FrozenAnswerer(lambda prompt: "  ").answer("Q", (), ())


def test_runner_records_normal_stop_and_forced_stop_with_at_most_five_transitions(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "runs.duckdb")
    answerer = FrozenAnswerer(lambda prompt: "A")
    normal = AgentRunner(
        _environment(),
        lambda state, legal: next(
            action for action in legal if action.action_type is ActionType.SEARCH_GIST
        )
        if not state.candidate_event_ids
        else ActionInstance(ActionType.STOP, None, None),
        answerer,
        run_store=store,
        artifact_dir=tmp_path,
        worker_id="worker",
    )
    normal_result = normal.run(_state(), run_id="normal")
    assert [
        transition.action.action_type for transition in normal_result.transitions
    ] == [ActionType.SEARCH_GIST, ActionType.STOP]
    assert normal_result.forced_stop is False
    assert store.pending("normal") == []

    forced = AgentRunner(
        _environment(),
        lambda state, legal: next(
            action for action in legal if action.action_type is not ActionType.STOP
        ),
        answerer,
        run_store=store,
        artifact_dir=tmp_path,
        worker_id="worker",
    )
    forced_result = forced.run(_state(), run_id="forced")
    assert len(forced_result.transitions) == 5
    assert forced_result.transitions[-1].action.action_type is ActionType.STOP
    assert forced_result.forced_stop is True


def test_runner_marks_policy_and_provider_failures_as_failed(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.duckdb")
    answerer = FrozenAnswerer(lambda prompt: "A")
    illegal = AgentRunner(
        _environment(),
        lambda state, legal: ActionInstance(ActionType.EXPAND_RESIDUAL, "e1", None),
        answerer,
        run_store=store,
        artifact_dir=tmp_path,
        worker_id="worker",
    )
    with pytest.raises(InvalidPolicyActionError):
        illegal.run(_state(), run_id="illegal")
    failing = AgentRunner(
        _environment(fail=True),
        lambda state, legal: legal[0],
        answerer,
        run_store=store,
        artifact_dir=tmp_path,
        worker_id="worker",
    )
    with pytest.raises(RuntimeError, match="provider unavailable"):
        failing.run(_state(), run_id="provider")
    import duckdb

    with duckdb.connect(str(tmp_path / "runs.duckdb")) as connection:
        statuses = connection.execute(
            "SELECT status FROM run_items WHERE run_id IN ('illegal', 'provider')"
        ).fetchall()
    assert statuses and all(status[0] == "failed" for status in statuses)


def test_answerer_identity_binds_actual_closure_and_frozen_model_config() -> None:
    def adapter_for(value: str):
        return lambda _prompt: value

    common = {
        "model_artifact_sha256": "a" * 64,
        "model_revision": "answerer-rev-7",
        "decode_config": {"temperature": 0.0, "max_tokens": 8},
    }
    first = FrozenAnswerer(adapter_for("A"), **common)
    second = FrozenAnswerer(adapter_for("B"), **common)
    changed_decode = FrozenAnswerer(
        adapter_for("A"),
        **{
            **common,
            "decode_config": {"temperature": 0.1, "max_tokens": 8},
        },
    )

    assert first.identity.adapter_sha256 != second.identity.adapter_sha256
    assert first.identity.identity_sha256 != second.identity.identity_sha256
    assert first.identity.identity_sha256 != changed_decode.identity.identity_sha256
    assert first.identity.model_artifact_sha256 == "a" * 64
    assert first.identity.model_revision == "answerer-rev-7"


def test_answerer_result_carries_authoritative_adapter_cost_and_usage() -> None:
    from fidmem.agent.answerer import AnswererAdapterResult

    measured = CostRecord(
        operation="frozen_answerer",
        gpu_seconds=0.25,
        wall_seconds=0.5,
        input_frames=0,
        visual_tokens=3,
        text_tokens=11,
        peak_memory_bytes=4096,
        cache_status="miss",
        device_name="synthetic",
    )
    answerer = FrozenAnswerer(
        lambda _prompt: AnswererAdapterResult(
            response="Answer: A", cost_record=measured, total_cost=1.75
        ),
        model_artifact_sha256="b" * 64,
        model_revision="answerer-rev-1",
        decode_config={"temperature": 0.0},
    )

    result = answerer.answer("Q", ("A", "B"), ())

    assert result.answer == "A"
    assert result.cost_record == measured
    assert result.total_cost == 1.75


def test_runner_custom_transition_bound_preserves_default_and_extends_fixed_trace() -> (
    None
):
    answerer = FrozenAnswerer(lambda _prompt: "A")
    always_acquire = lambda _state, legal: next(
        action for action in legal if action.action_type is not ActionType.STOP
    )

    default = AgentRunner(_environment(), always_acquire, answerer).run(
        _state(), run_id="default-bound"
    )
    extended = AgentRunner(
        _environment(), always_acquire, answerer, max_transitions=7
    ).run(_state(), run_id="fixed-bound")

    assert len(default.transitions) == 5
    assert default.forced_stop is True
    assert len(extended.transitions) == 7
