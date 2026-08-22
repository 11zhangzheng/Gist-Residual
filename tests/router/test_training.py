from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from fidmem.router.dataset import (
    OracleBCRecord,
    OracleBCDataset,
    RouterCollator,
    build_grouped_split,
    load_oracle_records,
    write_oracle_records,
)
from fidmem.router.model import MemoryRouter, RouterModelConfig, RouterOutput
from fidmem.router.train_bc import (
    LossWeights,
    TrainConfig,
    behavior_cloning_loss,
    load_checkpoint,
    save_checkpoint,
    train_bc,
)
from fidmem.types import ActionInstance, ActionType, RouterState


def _record(index: int, *, video_id: str | None = None) -> OracleBCRecord:
    target = index % 2
    return OracleBCRecord(
        record_id=f"r{index:03d}",
        video_id=video_id or f"v{index // 2:03d}",
        question_id=f"q{index:03d}",
        observation_snapshot_id="cached-observations-v1",
        state=RouterState(
            question="alpha" if target == 0 else "omega",
            options=("A", "B"),
            evidence=(),
            action_history=(),
            remaining_budget=10,
            candidate_event_ids=(),
            candidate_fidelity_levels={},
            context_frontiers={},
            cost_preference=0.1 * target,
        ),
        action_instances=(
            ActionInstance(ActionType.SEARCH_GIST, None, None),
            ActionInstance(ActionType.STOP, None, None),
        ),
        legal_action_mask=(True, True),
        target_action_index=target,
        sufficiency_target=target,
        cost_to_go=float(1 - target),
    )


def _model_config() -> RouterModelConfig:
    return RouterModelConfig(
        hidden_dim=24,
        token_embedding_dim=16,
        action_type_embedding_dim=8,
        fidelity_embedding_dim=4,
        max_question_bytes=64,
        max_item_bytes=64,
    )


def _train_config(path: Path, *, max_steps: int) -> TrainConfig:
    return TrainConfig(
        seed=17,
        max_steps=max_steps,
        batch_size=32,
        learning_rate=0.02,
        checkpoint_path=path,
        checkpoint_every=max_steps,
        device="cpu",
        loss_weights=LossWeights(action=1.0, sufficiency=0.3, cost_to_go=0.1),
    )


@pytest.mark.parametrize(
    "update",
    (
        {"legal_action_mask": (False, False)},
        {"legal_action_mask": (True,)},
        {"target_action_index": 2},
        {"target_action_index": 1, "legal_action_mask": (True, False)},
        {"cost_to_go": float("nan")},
        {"observation_snapshot_id": ""},
        {"sufficiency_target": True},
    ),
)
def test_oracle_record_rejects_invalid_masks_targets_nonfinite_or_missing_snapshot(
    update: dict[str, object],
) -> None:
    payload = _record(0).model_dump(mode="json")
    payload.update(update)
    with pytest.raises(ValidationError):
        OracleBCRecord.model_validate(payload)


def test_dataset_round_trip_and_grouped_split_never_crosses_video_boundaries(
    tmp_path: Path,
) -> None:
    records = tuple(_record(index) for index in range(20))
    path = tmp_path / "oracle.jsonl"
    write_oracle_records(path, records)

    loaded = load_oracle_records(path)
    first = build_grouped_split(loaded, seed=123, train_fraction=0.7, dev_fraction=0.2)
    second = build_grouped_split(
        tuple(reversed(loaded)), seed=123, train_fraction=0.7, dev_fraction=0.2
    )

    assert loaded == records
    assert first == second
    split_videos = [
        set(first.train_video_ids),
        set(first.dev_video_ids),
        set(first.test_video_ids),
    ]
    assert all(
        split_videos[i].isdisjoint(split_videos[j])
        for i in range(3)
        for j in range(i + 1, 3)
    )
    assert set().union(*split_videos) == {record.video_id for record in records}
    assert first.dataset_hash == OracleBCDataset(records).identity
    assert all(
        record.observation_snapshot_id == "cached-observations-v1" for record in loaded
    )


def test_jsonl_loader_fails_on_extra_fields_and_does_not_accept_pickle(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad.jsonl"
    payload = _record(0).model_dump(mode="json")
    payload["unexpected"] = "not allowed"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_oracle_records(path)

    unsafe = tmp_path / "oracle.pkl"
    unsafe.write_bytes(b"pickle")
    with pytest.raises(ValueError, match="jsonl or parquet"):
        load_oracle_records(unsafe)


def test_behavior_cloning_loss_uses_the_frozen_three_weights() -> None:
    output = RouterOutput(
        action_logits=torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
        sufficiency_logit=torch.tensor([0.2, -0.4]),
        cost_to_go=torch.tensor([1.5, 0.5]),
    )
    targets = {
        "target_action_index": torch.tensor([0, 1]),
        "sufficiency_target": torch.tensor([1.0, 0.0]),
        "cost_to_go_target": torch.tensor([1.0, 1.0]),
    }
    losses = behavior_cloning_loss(output, targets, LossWeights())
    expected = (
        torch.nn.functional.cross_entropy(
            output.action_logits, targets["target_action_index"]
        )
        + 0.3
        * torch.nn.functional.binary_cross_entropy_with_logits(
            output.sufficiency_logit, targets["sufficiency_target"]
        )
        + 0.1
        * torch.nn.functional.smooth_l1_loss(
            output.cost_to_go, targets["cost_to_go_target"]
        )
    )
    assert torch.allclose(losses.total, expected)


def test_32_sample_toy_overfits_to_at_least_95_percent_within_200_steps(
    tmp_path: Path,
) -> None:
    records = tuple(_record(index, video_id=f"v{index:03d}") for index in range(32))
    model = MemoryRouter(_model_config())
    result = train_bc(
        model,
        OracleBCDataset(records),
        _train_config(tmp_path / "overfit.pt", max_steps=200),
    )

    assert result.step <= 200
    assert result.action_accuracy >= 0.95


def test_checkpoint_resume_restores_exact_training_state_and_rejects_identity_mismatch(
    tmp_path: Path,
) -> None:
    records = tuple(_record(index, video_id=f"v{index:03d}") for index in range(32))
    dataset = OracleBCDataset(records)
    model_config = _model_config()

    torch.manual_seed(99)
    uninterrupted = MemoryRouter(model_config)
    uninterrupted_result = train_bc(
        uninterrupted,
        dataset,
        _train_config(tmp_path / "uninterrupted.pt", max_steps=8),
    )

    torch.manual_seed(99)
    first_leg = MemoryRouter(model_config)
    first_result = train_bc(
        first_leg,
        dataset,
        _train_config(tmp_path / "resume.pt", max_steps=3),
    )
    resumed = MemoryRouter(model_config)
    resumed_result = train_bc(
        resumed,
        dataset,
        _train_config(tmp_path / "resume.pt", max_steps=8),
        resume=True,
    )

    assert first_result.step == 3
    assert uninterrupted_result.step == resumed_result.step == 8
    for key, value in uninterrupted.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[key]), key

    mismatched = OracleBCDataset(records + (_record(100, video_id="other"),))
    with pytest.raises(ValueError, match="dataset identity"):
        train_bc(
            MemoryRouter(model_config),
            mismatched,
            _train_config(tmp_path / "resume.pt", max_steps=9),
            resume=True,
        )


def test_checkpoint_manifest_contains_weights_config_split_and_all_rng_states(
    tmp_path: Path,
) -> None:
    records = tuple(_record(index, video_id=f"v{index:03d}") for index in range(4))
    dataset = OracleBCDataset(records)
    model = MemoryRouter(_model_config())
    config = _train_config(tmp_path / "manifest.pt", max_steps=1)
    result = train_bc(model, dataset, config)

    checkpoint = load_checkpoint(
        result.checkpoint_path,
        expected_config_hash=result.config_hash,
        expected_dataset_identity=dataset.identity,
        expected_split_manifest=result.split_manifest,
    )
    assert checkpoint["step"] == 1
    assert checkpoint["git_commit"]
    assert checkpoint["config_hash"] == result.config_hash
    assert checkpoint["split_manifest"] == result.split_manifest.model_dump(mode="json")
    assert checkpoint["loss_weights"] == {
        "action": 1.0,
        "sufficiency": 0.3,
        "cost_to_go": 0.1,
    }
    assert set(checkpoint["rng_state"]) == {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
    }
    assert set(checkpoint) >= {"model", "optimizer", "scheduler", "step"}


def test_training_seed_controls_initial_weights_as_well_as_batch_sampling(
    tmp_path: Path,
) -> None:
    records = tuple(_record(index, video_id=f"v{index:03d}") for index in range(8))
    dataset = OracleBCDataset(records)
    config = _train_config(tmp_path / "seed-a.pt", max_steps=2)

    torch.manual_seed(1)
    first = MemoryRouter(_model_config())
    train_bc(first, dataset, config)

    torch.manual_seed(999)
    second = MemoryRouter(_model_config())
    train_bc(
        second,
        dataset,
        config.model_copy(update={"checkpoint_path": tmp_path / "seed-b.pt"}),
    )

    for key, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[key]), key
