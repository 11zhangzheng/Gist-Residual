from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from fidmem.actions.environment import (
    ActionCostTable,
    ActionObservation,
    MemoryEnvironment,
    OperationMetadata,
)
from fidmem.oracle.labels import CostNormalization
from fidmem.oracle.search import (
    AnswerEvaluation,
    CachedObservationGraph,
    observation_key,
)
from fidmem.router.dagger import (
    CacheArtifactIdentity,
    CachedAnswerEvaluator,
    CachedUtilityGraph,
    DAggerConfig,
    DAggerQuestionContext,
    ForbiddenObservationGenerator,
    PolicyTrainingResult,
    Task10DaggerProvenance,
    evaluation_key,
    run_dagger,
    select_train_question_subset,
)
from fidmem.router.dataset import OracleBCDataset
from fidmem.types import (
    ActionInstance,
    ActionType,
    EventRecord,
    EvidenceItem,
    FidelityLevel,
    RouterState,
)

from tests.router._fixtures import authoritative_record


def _identity(label: str) -> CacheArtifactIdentity:
    return CacheArtifactIdentity(
        artifact_sha256=hashlib.sha256(label.encode("utf-8")).hexdigest()
    )


def _action(kind: ActionType) -> ActionInstance:
    return ActionInstance(kind, None, None)


def _state(question_id: str) -> RouterState:
    return RouterState(
        question=f"Question {question_id}?",
        options=("blue", "red"),
        evidence=(),
        action_history=(),
        remaining_budget=3,
        candidate_event_ids=(),
        candidate_fidelity_levels={},
        context_frontiers={},
        cost_preference=0,
    )


def _environment() -> MemoryEnvironment:
    return MemoryEnvironment(
        events=(
            EventRecord(
                video_id="source",
                event_id="e1",
                start_sec=0,
                end_sec=2,
                gist_text="bottle",
                visual_embedding=(1.0,),
                text_embedding=(1.0,),
                raw_video_uri="source.mp4",
                memory_version="v1",
            ),
        ),
        executor=ForbiddenObservationGenerator(),
        costs=ActionCostTable(search_gist=3),
    )


def _snapshot(
    environment: MemoryEnvironment, state: RouterState, label: str
) -> CachedUtilityGraph:
    search = _action(ActionType.SEARCH_GIST)
    observation = ActionObservation(
        action_type=ActionType.SEARCH_GIST,
        candidate_event_ids=("e1",),
        evidence=(
            EvidenceItem(
                event_id="e1",
                fidelity_level=FidelityLevel.GIST,
                content="blue",
                score=1,
            ),
        ),
        operation_metadata=(
            OperationMetadata(
                scope="search_gist", cache_status="miss", amortizable=True
            ),
        ),
    )
    searched = environment.replay(state, search, observation).next_state
    return CachedUtilityGraph(
        identity=_identity(f"snapshot:{label}"),
        observation_identity=_identity(f"observations:{label}"),
        observations=CachedObservationGraph(
            {observation_key(state, search): observation}
        ),
        evaluator=CachedAnswerEvaluator(
            identity=_identity(f"evaluations:{label}"),
            evaluations={
                evaluation_key(state): AnswerEvaluation(
                    answer="red", answer_score=0.2, correct=False
                ),
                evaluation_key(searched): AnswerEvaluation(
                    answer="blue", answer_score=1.0, correct=True
                ),
            },
        ),
    )


def _contexts(count: int) -> tuple[DAggerQuestionContext, ...]:
    normalization = CostNormalization(
        constant=10, sample_count=max(1, count), source_split="train"
    )
    actions = (_action(ActionType.SEARCH_GIST), _action(ActionType.STOP))
    records = tuple(
        authoritative_record(
            state=_state(f"q{index}"),
            actions=actions,
            legal_action_mask=(True, True),
            target_action_index=0,
            video_id=f"video-{index}",
            question_id=f"q{index}",
            sufficiency_target=0,
            cost_to_go=0.3,
            normalization=normalization,
            observation_snapshot_id=f"snapshot-{index}",
        )
        for index in range(count)
    )
    dataset = OracleBCDataset(records)
    contexts = []
    for index, record in enumerate(records):
        environment = _environment()
        contexts.append(
            DAggerQuestionContext.from_record(
                record=record,
                dataset=dataset,
                environment=environment,
                snapshot=_snapshot(environment, record.state, f"q{index}"),
            )
        )
    return tuple(contexts)


def _always_stop(
    state: RouterState, legal: tuple[ActionInstance, ...]
) -> ActionInstance:
    stop = _action(ActionType.STOP)
    assert stop in legal
    return stop


class SpyTrainer:
    def __init__(self) -> None:
        self.aggregate_sizes: list[int] = []
        self.new_sizes: list[int] = []
        self.load_calls = 0

    def train(
        self,
        *,
        round_number: int,
        base_dataset: OracleBCDataset,
        deviations: tuple[object, ...],
        new_deviations: tuple[object, ...],
        source_policy_checkpoint: Path,
        output_checkpoint: Path,
    ) -> PolicyTrainingResult:
        self.aggregate_sizes.append(len(deviations))
        self.new_sizes.append(len(new_deviations))
        output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        output_checkpoint.write_bytes(
            f"round={round_number};source={source_policy_checkpoint.name}".encode()
        )
        digest = hashlib.sha256(output_checkpoint.read_bytes()).hexdigest()
        return PolicyTrainingResult(
            policy=_always_stop,
            checkpoint_path=output_checkpoint,
            checkpoint_sha256=digest,
            aggregated_dataset_identity=hashlib.sha256(
                f"{base_dataset.identity}:{len(deviations)}".encode()
            ).hexdigest(),
        )

    def load_policy(
        self,
        *,
        checkpoint: Path,
        base_dataset: OracleBCDataset,
        deviations: tuple[object, ...],
    ):
        self.load_calls += 1
        return _always_stop


def test_fixed_question_subset_is_seeded_canonical_and_worker_independent() -> None:
    contexts = _contexts(8)

    selected = select_train_question_subset(contexts, fraction=0.5, seed=73)
    reordered = select_train_question_subset(
        tuple(reversed(contexts)), fraction=0.5, seed=73
    )

    assert tuple(item.question_id for item in selected) == tuple(
        item.question_id for item in reordered
    )
    assert len(selected) == 4


def test_context_rejects_task10_dataset_identity_impersonation() -> None:
    context = _contexts(1)[0]
    forged = Task10DaggerProvenance(
        dataset_identity="0" * 64,
        base_record_ids=context.task10.base_record_ids,
    )

    with pytest.raises(ValueError, match="Task 10 dataset identity"):
        DAggerQuestionContext.model_validate(
            context.model_dump(mode="python") | {"task10": forged}
        )


def test_multiround_training_persists_seen_keys_manifests_and_resumes(
    tmp_path: Path,
) -> None:
    contexts = _contexts(1)
    initial_checkpoint = tmp_path / "initial.pt"
    initial_checkpoint.write_bytes(b"initial policy")
    config = DAggerConfig(
        artifact_root=tmp_path,
        max_rounds=3,
        beam_size=8,
        max_depth=5,
        utility_gain_threshold=0.005,
        regret_improvement_ratio=0.02,
        train_question_subset_fraction=1.0,
        train_question_subset_seed=37,
        budget_bin_width=1.0,
        seen_keys_path=Path("seen.json"),
        deviation_artifact_path=Path("deviations.json"),
        manifest_dir=Path("manifests"),
        checkpoint_dir=Path("checkpoints"),
    )
    trainer = SpyTrainer()

    result = run_dagger(
        train_contexts=contexts,
        dev_contexts=contexts,
        initial_policy=_always_stop,
        source_policy_checkpoint=initial_checkpoint,
        trainer=trainer,
        config=config,
    )

    assert trainer.aggregate_sizes == [1, 1]
    assert trainer.new_sizes == [1, 0]
    assert len(result.manifests) == 2
    assert result.manifests[-1].status == "stopped"
    assert result.manifests[-1].stop_reason == "threshold_not_met"
    assert (tmp_path / "seen.json").is_file()
    assert len(tuple(tmp_path.glob("deviations-round-*.json"))) == 2
    assert tuple((tmp_path / "manifests").glob("round-*.json"))

    resumed_trainer = SpyTrainer()
    resumed = run_dagger(
        train_contexts=contexts,
        dev_contexts=contexts,
        initial_policy=_always_stop,
        source_policy_checkpoint=initial_checkpoint,
        trainer=resumed_trainer,
        config=config,
    )

    assert resumed.resumed is True
    assert resumed_trainer.aggregate_sizes == []
    assert resumed_trainer.load_calls == 0
    assert resumed.final_checkpoint == result.final_checkpoint


def test_resume_rejects_changed_source_policy_identity(tmp_path: Path) -> None:
    contexts = _contexts(1)
    source = tmp_path / "initial.pt"
    source.write_bytes(b"initial policy")
    config = DAggerConfig(artifact_root=tmp_path)
    run_dagger(
        train_contexts=contexts,
        dev_contexts=contexts,
        initial_policy=_always_stop,
        source_policy_checkpoint=source,
        trainer=SpyTrainer(),
        config=config,
    )
    source.write_bytes(b"forged policy")

    with pytest.raises(ValueError, match="run identity"):
        run_dagger(
            train_contexts=contexts,
            dev_contexts=contexts,
            initial_policy=_always_stop,
            source_policy_checkpoint=source,
            trainer=SpyTrainer(),
            config=config,
        )
