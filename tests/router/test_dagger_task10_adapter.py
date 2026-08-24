from __future__ import annotations

import hashlib
from pathlib import Path

from fidmem.router.dagger import Deviation, Task10PolicyTrainer, load_dagger_file
from fidmem.router.dataset import OracleBCDataset, TestByteTokenizer
from fidmem.router.model import EncoderIdentity, MemoryRouter, RouterModelConfig
from fidmem.router.train_bc import TrainResult

from tests.router._fixtures import authoritative_record
from fidmem.oracle.labels import CostNormalization
from fidmem.types import ActionInstance, ActionType, RouterState


def _action(kind: ActionType) -> ActionInstance:
    return ActionInstance(kind, None, None)


def _state() -> RouterState:
    return RouterState(
        question="Which color?",
        options=("blue", "red"),
        evidence=(),
        action_history=(),
        remaining_budget=3,
        candidate_event_ids=(),
        candidate_fidelity_levels={},
        context_frontiers={},
        cost_preference=0,
    )


def _model() -> MemoryRouter:
    identity = EncoderIdentity.test_identity("dagger-adapter")
    return MemoryRouter(
        RouterModelConfig(
            encoder=identity,
            encoder_output_dim=8,
            hidden_dim=12,
            action_type_embedding_dim=4,
            fidelity_embedding_dim=2,
            max_question_tokens=64,
            max_item_tokens=32,
        )
    )


def test_dagger_config_loads_model_training_and_consumed_runner_fields() -> None:
    model, file_config = load_dagger_file("configs/experiment/dagger.yaml")

    assert model.encoder.model_id == "offline-smoke-encoder-v1"
    assert file_config.training.device == "cpu"
    assert file_config.dagger.max_rounds == 3
    assert file_config.dagger.train_question_subset_seed == 2026
    assert file_config.dagger.budget_bin_width == 1.0


def test_task10_policy_trainer_aggregates_and_calls_public_train_bc(
    tmp_path: Path,
) -> None:
    normalization = CostNormalization(constant=10, sample_count=1, source_split="train")
    record = authoritative_record(
        state=_state(),
        actions=(_action(ActionType.SEARCH_GIST), _action(ActionType.STOP)),
        legal_action_mask=(True, True),
        target_action_index=0,
        video_id="video",
        question_id="question",
        sufficiency_target=0,
        cost_to_go=0.3,
        normalization=normalization,
    )
    dataset = OracleBCDataset((record,))
    model_config, file_config = load_dagger_file("configs/experiment/dagger.yaml")
    training = file_config.training.model_copy(
        update={"artifact_root": tmp_path, "max_steps": 1}
    )
    seen_dataset_sizes: list[int] = []

    def fake_train_bc(model, aggregated, config, *, resume, tokenizer):
        seen_dataset_sizes.append(len(aggregated))
        checkpoint = config.validated_checkpoint_path()
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"task10 public adapter")
        return TrainResult(
            step=1,
            action_accuracy=1.0,
            checkpoint_path=checkpoint,
            config_hash="0" * 64,
            split_manifest=object(),
            seed=config.seed,
            dataset_identity=aggregated.identity,
        )

    trainer = Task10PolicyTrainer(
        model_factory=lambda: (
            _model(),
            TestByteTokenizer("dagger-adapter"),
        ),
        training=training,
        record_materializer=lambda deviation: record.model_copy(
            update={"record_id": f"deviation:{deviation.state_key}"}
        ),
        train_function=fake_train_bc,
    )
    source = tmp_path / "source.pt"
    source.write_bytes(b"source")
    output = tmp_path / "round-1.pt"
    fake_deviation = Deviation(
        state_key=hashlib.sha256(b"d").hexdigest(),
        question_id="question",
        state=record.state,
        action_instances=record.action_instances,
        legal_action_mask=record.legal_action_mask,
        policy_action=_action(ActionType.STOP),
        oracle_action=_action(ActionType.SEARCH_GIST),
        acquired_observation_keys=(),
    )

    result = trainer.train(
        round_number=1,
        base_dataset=dataset,
        deviations=(fake_deviation,),
        new_deviations=(fake_deviation,),
        source_policy_checkpoint=source,
        output_checkpoint=output,
    )

    assert seen_dataset_sizes == [2]
    assert result.checkpoint_path == output
    assert result.checkpoint_sha256 == hashlib.sha256(output.read_bytes()).hexdigest()
