"""Bounded execution of a masked memory policy with durable transition logs."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from fidmem.actions.environment import EnvironmentTransition, MemoryEnvironment
from fidmem.agent.answerer import AnswerResult, FrozenAnswerer
from fidmem.storage.run_store import RunStore
from fidmem.types import ActionInstance, ActionType, RouterState


class InvalidPolicyActionError(ValueError):
    """Raised instead of silently repairing an illegal policy proposal."""


class RouterPolicy(Protocol):
    def __call__(self, state: RouterState, legal_actions: tuple[ActionInstance, ...]) -> ActionInstance: ...


class RunResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    transitions: tuple[EnvironmentTransition, ...]
    answer: AnswerResult
    final_state: RouterState
    forced_stop: bool


class AgentRunner:
    """Execute at most five transitions, then answer only from acquired evidence."""

    def __init__(
        self,
        environment: MemoryEnvironment,
        policy: RouterPolicy,
        answerer: FrozenAnswerer,
        *,
        run_store: RunStore | None = None,
        artifact_dir: Path | str | None = None,
        worker_id: str = "agent-runner",
    ) -> None:
        self.environment = environment
        self.policy = policy
        self.answerer = answerer
        self.run_store = run_store
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else None
        self.worker_id = worker_id
        if self.run_store is not None and self.artifact_dir is None:
            raise ValueError("artifact_dir is required when RunStore is enabled")

    def _item_key(self, step: int) -> str:
        return f"transition-{step:03d}"

    def _claim(self, run_id: str, item_key: str) -> None:
        if self.run_store is not None and not self.run_store.claim(run_id, item_key, self.worker_id):
            raise RuntimeError(f"run item cannot be claimed: {run_id}/{item_key}")

    def _complete(self, run_id: str, item_key: str, payload: BaseModel) -> None:
        if self.run_store is None:
            return
        assert self.artifact_dir is not None
        path = self.artifact_dir / run_id / f"{item_key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
        self.run_store.complete(run_id, item_key, str(path))

    def _fail(self, run_id: str, item_key: str, error: BaseException) -> None:
        if self.run_store is not None:
            self.run_store.fail(run_id, item_key, type(error).__name__, str(error))

    def _choose(self, state: RouterState, legal_actions: tuple[ActionInstance, ...]) -> ActionInstance:
        selected = self.policy(state, legal_actions)
        if selected not in legal_actions:
            raise InvalidPolicyActionError("policy selected an action outside the provided legal tuple")
        return selected

    def run(self, initial_state: RouterState, *, run_id: str) -> RunResult:
        state = initial_state
        transitions: list[EnvironmentTransition] = []
        forced_stop = False
        for step in range(5):
            legal_actions = self.environment.valid_actions(state)
            if not legal_actions:
                raise RuntimeError("environment reached a non-terminal state with no legal actions")
            item_key = self._item_key(step)
            self._claim(run_id, item_key)
            try:
                if step == 4:
                    stop = ActionInstance(ActionType.STOP, None, None)
                    if stop not in legal_actions:
                        raise RuntimeError("fifth transition requires a legal STOP action")
                    selected = stop
                    forced_stop = True
                else:
                    selected = self._choose(state, legal_actions)
                transition = self.environment.step(state, selected)
                self._complete(run_id, item_key, transition)
            except BaseException as error:
                self._fail(run_id, item_key, error)
                raise
            transitions.append(transition)
            state = transition.next_state
            if transition.terminal:
                break
        if not transitions or not transitions[-1].terminal:
            raise RuntimeError("runner exceeded its transition bound without STOP")
        answer_key = "answer"
        self._claim(run_id, answer_key)
        try:
            answer = self.answerer.answer(state.question, state.options, state.evidence)
            self._complete(run_id, answer_key, answer)
        except BaseException as error:
            self._fail(run_id, answer_key, error)
            raise
        return RunResult(transitions=tuple(transitions), answer=answer, final_state=state, forced_stop=forced_stop)
