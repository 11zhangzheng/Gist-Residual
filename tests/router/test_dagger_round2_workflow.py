from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from fidmem.actions.environment import ActionCostTable, MemoryEnvironment
from fidmem.oracle.labels import CostNormalization
from fidmem.oracle.search import (
    AnswerEvaluation,
    CachedObservationGraph,
    observation_key,
)
from fidmem.router.dagger import (
    DAggerConfig,
    CachedAnswerEvaluator,
    CachedUtilityGraph,
    DAggerQuestionContext,
    ForbiddenObservationGenerator,
    FrozenActionPolicy,
    _state_key,
    evaluation_key,
    policy_identity,
    run_dagger,
)
from fidmem.router.dagger_workflow import RoundMetrics, _stop_decision
from fidmem.types import ActionInstance, ActionType
from fidmem.router.dataset import OracleBCDataset

from tests.router._fixtures import authoritative_record
from tests.router.test_dagger import _environment, _state, _utility_graph
from tests.router.test_dagger_workflow import (
    SpyTrainer,
    _always_stop,
    _contexts,
    _snapshot,
)


class FailRoundTwoTrainer(SpyTrainer):
    def train(self, **kwargs):
        if kwargs["round_number"] == 2:
            raise RuntimeError("injected round two training failure")
        return super().train(**kwargs)


def _run(tmp_path: Path, trainer: SpyTrainer):
    contexts = _contexts(1)
    config = DAggerConfig(artifact_root=tmp_path)
    _always_stop.freeze(config.bootstrap_path)
    return run_dagger(
        train_contexts=contexts,
        dev_contexts=contexts,
        initial_policy=_always_stop,
        source_policy_checkpoint=config.bootstrap_path,
        trainer=trainer,
        config=config,
    )


def test_round_failure_keeps_prior_current_and_seen_then_retry_resumes(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="round two"):
        _run(tmp_path, FailRoundTwoTrainer())

    pointer = json.loads((tmp_path / "current.json").read_text(encoding="utf-8"))
    assert pointer["round_number"] == 1
    assert (tmp_path / "generations" / "round-0001" / "seen.json").is_file()
    assert not tuple((tmp_path / "generations").glob("*.staging"))
    assert not (tmp_path / "generations" / "round-0002").exists()

    retried = _run(tmp_path, SpyTrainer())
    assert retried.resumed is True
    assert retried.manifests[-1].round_number == 2
    assert retried.manifests[-1].new_deviation_count == 0


def test_environment_and_cache_identities_derive_from_actual_contents() -> None:
    context = _contexts(1)[0]
    changed_environment = MemoryEnvironment(
        events=context.environment.canonical_events,
        executor=ForbiddenObservationGenerator(),
        costs=ActionCostTable(search_gist=2),
    )
    changed_cost = DAggerQuestionContext.from_record(
        record=context.base_record,
        dataset=context.dataset,
        environment=changed_environment,
        snapshot=_snapshot(changed_environment, context.state, "q0"),
    )
    changed_cache = DAggerQuestionContext.from_record(
        record=context.base_record,
        dataset=context.dataset,
        environment=context.environment,
        snapshot=_snapshot(context.environment, context.state, "changed-evaluator"),
    )

    assert changed_cost.environment_identity != context.environment_identity
    assert changed_cost.identity != context.identity
    assert changed_cache.snapshot.identity != context.snapshot.identity
    assert changed_cache.identity != context.identity


class OppositeLoadTrainer(SpyTrainer):
    def load_policy(self, **kwargs):
        del kwargs
        return FrozenActionPolicy(ActionType.SEARCH_GIST)


def test_initial_policy_must_match_trainer_validated_bootstrap(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not match behavior|initial policy"):
        _run(tmp_path, OppositeLoadTrainer())


def test_resume_rejects_changed_environment_cost_identity(tmp_path: Path) -> None:
    contexts = _contexts(1)
    config = DAggerConfig(artifact_root=tmp_path)
    _always_stop.freeze(config.bootstrap_path)
    run_dagger(
        train_contexts=contexts,
        dev_contexts=contexts,
        initial_policy=_always_stop,
        source_policy_checkpoint=config.bootstrap_path,
        trainer=SpyTrainer(),
        config=config,
    )
    original = contexts[0]
    changed_environment = MemoryEnvironment(
        events=original.environment.canonical_events,
        executor=ForbiddenObservationGenerator(),
        costs=ActionCostTable(search_gist=2),
    )
    changed = DAggerQuestionContext.from_record(
        record=original.base_record,
        dataset=original.dataset,
        environment=changed_environment,
        snapshot=_snapshot(changed_environment, original.state, "changed-cost"),
    )

    with pytest.raises(ValueError, match="run identity"):
        run_dagger(
            train_contexts=(changed,),
            dev_contexts=(changed,),
            initial_policy=_always_stop,
            source_policy_checkpoint=config.bootstrap_path,
            trainer=SpyTrainer(),
            config=config,
        )


def test_resume_rejects_changed_environment_event_identity(tmp_path: Path) -> None:
    contexts = _contexts(1)
    config = DAggerConfig(artifact_root=tmp_path)
    _always_stop.freeze(config.bootstrap_path)
    run_dagger(
        train_contexts=contexts,
        dev_contexts=contexts,
        initial_policy=_always_stop,
        source_policy_checkpoint=config.bootstrap_path,
        trainer=SpyTrainer(),
        config=config,
    )
    original = contexts[0]
    changed_events = tuple(
        event.model_copy(update={"gist_text": event.gist_text + " changed"})
        for event in original.environment.canonical_events
    )
    changed_environment = MemoryEnvironment(
        events=changed_events,
        executor=ForbiddenObservationGenerator(),
        costs=original.environment.costs,
    )
    changed = DAggerQuestionContext.from_record(
        record=original.base_record,
        dataset=original.dataset,
        environment=changed_environment,
        snapshot=_snapshot(changed_environment, original.state, "changed-event"),
    )

    with pytest.raises(ValueError, match="run identity"):
        run_dagger(
            train_contexts=(changed,),
            dev_contexts=(changed,),
            initial_policy=_always_stop,
            source_policy_checkpoint=config.bootstrap_path,
            trainer=SpyTrainer(),
            config=config,
        )


def test_resume_rejects_changed_cached_evaluation_contents(tmp_path: Path) -> None:
    contexts = _contexts(1)
    config = DAggerConfig(artifact_root=tmp_path)
    _always_stop.freeze(config.bootstrap_path)
    run_dagger(
        train_contexts=contexts,
        dev_contexts=contexts,
        initial_policy=_always_stop,
        source_policy_checkpoint=config.bootstrap_path,
        trainer=SpyTrainer(),
        config=config,
    )
    original = contexts[0]
    search = ActionInstance(ActionType.SEARCH_GIST, None, None)
    observation = original.snapshot.get(original.state, search)
    assert observation is not None
    searched = original.environment.replay(
        original.state, search, observation
    ).next_state
    changed_snapshot = CachedUtilityGraph(
        observations=CachedObservationGraph(
            {observation_key(original.state, search): observation}
        ),
        evaluator=CachedAnswerEvaluator(
            evaluator_identity=original.snapshot.evaluator_identity,
            evaluations={
                evaluation_key(original.state): AnswerEvaluation(
                    answer="red", answer_score=0.25, correct=False
                ),
                evaluation_key(searched): AnswerEvaluation(
                    answer="blue", answer_score=1.0, correct=True
                ),
            },
        ),
    )
    changed = DAggerQuestionContext.from_record(
        record=original.base_record,
        dataset=original.dataset,
        environment=original.environment,
        snapshot=changed_snapshot,
    )

    with pytest.raises(ValueError, match="run identity"):
        run_dagger(
            train_contexts=(changed,),
            dev_contexts=(changed,),
            initial_policy=_always_stop,
            source_policy_checkpoint=config.bootstrap_path,
            trainer=SpyTrainer(),
            config=config,
        )


def test_policy_checkpoint_identity_rejects_opposite_actual_behavior(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "bootstrap.pt"
    FrozenActionPolicy(ActionType.SEARCH_GIST).freeze(checkpoint)

    with pytest.raises(ValueError, match="does not match behavior"):
        policy_identity(FrozenActionPolicy(ActionType.STOP), checkpoint)


def test_initial_replay_acquisition_changes_state_key() -> None:
    environment = _environment()
    initial = _state()
    graph = _utility_graph(environment, initial)
    search = ActionInstance(ActionType.SEARCH_GIST, None, None)
    observation = graph.get(initial, search)
    assert observation is not None
    transition = environment.replay(initial, search, observation)

    acquired = _state_key(
        transition.next_state,
        question_id="q",
        initial_replay_transitions=(transition,),
        replayed_transitions=(),
        budget_bin_width=1.0,
    )
    omitted = _state_key(
        transition.next_state,
        question_id="q",
        initial_replay_transitions=(),
        replayed_transitions=(),
        budget_bin_width=1.0,
    )
    assert acquired != omitted


def test_context_requires_authoritative_replay_for_preexisting_acquisition() -> None:
    environment = _environment()
    initial = _state()
    graph = _utility_graph(environment, initial)
    search = ActionInstance(ActionType.SEARCH_GIST, None, None)
    observation = graph.get(initial, search)
    assert observation is not None
    transition = environment.replay(initial, search, observation)
    state = transition.next_state
    actions = environment.valid_actions(state)
    record = authoritative_record(
        state=state,
        actions=actions,
        legal_action_mask=(True,) * len(actions),
        target_action_index=len(actions) - 1,
        video_id="video-existing",
        question_id="question-existing",
        sufficiency_target=1,
        cost_to_go=0,
        normalization=CostNormalization(
            constant=10, sample_count=1, source_split="train"
        ),
        observation_snapshot_id="snapshot-existing",
    )
    dataset = OracleBCDataset((record,))

    context = DAggerQuestionContext.from_record(
        record=record,
        dataset=dataset,
        environment=environment,
        snapshot=graph,
        initial_state=initial,
        initial_replay_transitions=(transition,),
    )
    assert context.state == state

    with pytest.raises(ValueError, match="initial replay"):
        DAggerQuestionContext.from_record(
            record=record,
            dataset=dataset,
            environment=environment,
            snapshot=graph,
        )


def test_stop_threshold_boundary_is_decimal_deterministic() -> None:
    config = DAggerConfig(
        artifact_root=Path("unused"),
        utility_gain_threshold=0.005,
        regret_improvement_ratio=0.02,
    )
    previous = RoundMetrics(dev_utility=0.5, cost_regret=1.0)
    exact = RoundMetrics(dev_utility=0.505, cost_regret=1.0)
    below = RoundMetrics(
        dev_utility=math.nextafter(0.505, float("-inf")), cost_regret=1.0
    )

    assert _stop_decision(2, exact, previous, config) == ("completed", "continue")
    assert _stop_decision(2, below, previous, config) == (
        "stopped",
        "threshold_not_met",
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("deviation_count", 999, "derived"),
        (
            "checkpoint",
            {"path": "../outside.pt", "sha256": "0" * 64},
            "canonical relative",
        ),
    ),
)
def test_forged_manifest_is_rejected_even_after_recomputing_self_hash(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    _run(tmp_path, SpyTrainer())
    manifest_path = tmp_path / "generations" / "round-0002" / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload[field] = value
    payload.pop("manifest_sha256")
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    payload["manifest_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    manifest_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        _run(tmp_path, SpyTrainer())
