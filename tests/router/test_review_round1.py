from __future__ import annotations

import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pytest
import torch
from fidmem.actions.environment import ActionObservation, EnvironmentTransition
from fidmem.oracle.labels import PreferenceLabel
from fidmem.oracle.search import OraclePath
from fidmem.router.dataset import (
    OracleBCDataset,
    RouterCollator,
    TokenizerIdentity,
    load_oracle_records,
    write_oracle_records,
)
from fidmem.router.model import (
    EncoderIdentity,
    MemoryRouter,
    RouterModelConfig,
    TestTextEncoder,
)
from fidmem.router.train_bc import (
    LossProfile,
    LossWeights,
    ProductionEncoderFactory,
    TrainConfig,
    TrainFileConfig,
    RuntimeIdentity,
    behavior_cloning_loss,
    load_checkpoint,
    run_seed_sweep,
    train_bc,
)
from fidmem.types import (
    ActionInstance,
    ActionType,
    EvidenceItem,
    FidelityLevel,
    RouterState,
)
from omegaconf import OmegaConf
from pydantic import ValidationError
from tests.router._fixtures import authoritative_record


def _state(
    *,
    preference: float = 0.3,
    first_fidelity: FidelityLevel = FidelityLevel.GIST,
    first_frontier: tuple[int, int] = (0, 0),
) -> RouterState:
    return RouterState(
        question="Which candidate contains the matching evidence?",
        options=("left", "right"),
        evidence=(
            EvidenceItem(
                event_id="e-left",
                fidelity_level=FidelityLevel.RESIDUAL,
                content="matching evidence",
                score=1,
            ),
        ),
        action_history=(),
        remaining_budget=10,
        candidate_event_ids=("e-left", "e-right"),
        candidate_fidelity_levels={
            "e-left": first_fidelity,
            "e-right": FidelityLevel.RESIDUAL,
        },
        context_frontiers={"e-left": first_frontier, "e-right": (1, 0)},
        cost_preference=preference,
    )


def _actions(*, reversed_order: bool = False) -> tuple[ActionInstance, ...]:
    pair = (
        ActionInstance(ActionType.EXPAND_RESIDUAL, "e-left", None),
        ActionInstance(ActionType.EXPAND_RESIDUAL, "e-right", None),
    )
    if reversed_order:
        pair = tuple(reversed(pair))
    return pair + (ActionInstance(ActionType.STOP, None, None),)


def _preference_labels(state: RouterState) -> tuple[PreferenceLabel, ...]:
    labels = []
    action = _actions()[0]
    next_state = state.model_copy(update={"action_history": (action,)})
    transition = EnvironmentTransition(
        state=state,
        action=action,
        observation=ActionObservation(
            action_type=action.action_type,
            target_event_id=action.event_id,
        ),
        next_state=next_state,
        step_cost=1,
    )
    for preference in (0.0, 0.1, 0.3, 1.0):
        labels.append(
            PreferenceLabel(
                cost_preference=preference,
                utility=1 - preference / 10,
                optimal_paths=(
                    OraclePath(
                        transitions=(transition,),
                        answer="left",
                        answer_score=1,
                        correct=True,
                        total_cost=1,
                        utility=1 - preference / 10,
                    ),
                ),
            )
        )
    return tuple(labels)


def _record(
    *,
    state: RouterState | None = None,
    actions: tuple[ActionInstance, ...] | None = None,
    video_id: str = "video-group-1",
    question_id: str = "question-1",
    split: str = "train",
):
    current = state or _state()
    candidates = actions or _actions()
    target_index = next(
        index for index, action in enumerate(candidates) if action.event_id == "e-left"
    )
    return authoritative_record(
        state=current,
        actions=candidates,
        legal_action_mask=(True,) * len(candidates),
        target_action_index=target_index,
        video_id=video_id,
        question_id=question_id,
        sufficiency_target=0,
        cost_to_go=0.1,
        split=split,
        observation_snapshot_id="cached-observations-sha256",
    )


def _model() -> MemoryRouter:
    identity = EncoderIdentity.test_identity("offline-test-tokenizer-v1")
    config = RouterModelConfig(
        encoder=identity,
        encoder_output_dim=16,
        hidden_dim=24,
        action_type_embedding_dim=8,
        fidelity_embedding_dim=4,
        max_question_tokens=128,
        max_item_tokens=96,
        production=False,
        enforce_parameter_range=False,
    )
    return MemoryRouter(config, text_encoder=TestTextEncoder(identity, 257, 16))


def _collator() -> RouterCollator:
    return RouterCollator.for_test(max_question_tokens=128, max_item_tokens=96)


def test_materializer_preserves_four_preference_normalization_and_longroute_lineage() -> (
    None
):
    record = _record()

    assert record.provenance.preference_values == (0.0, 0.1, 0.3, 1.0)
    assert record.provenance.selected_preference == 0.3
    assert record.provenance.normalization_manifest_hash
    assert len(record.provenance.dataset_manifest_hash) == 64
    assert len(record.provenance.source_manifest_hash) == 64
    assert record.provenance.source_split == "train"
    assert record.video_id == record.provenance.video_group_id == "video-group-1"
    assert record.action_instances[record.target_action_index].event_id == "e-left"

    payload = record.provenance.model_dump(mode="json")
    payload["normalization_manifest_hash"] = "f" * 64
    with pytest.raises(ValidationError, match="normalization"):
        type(record.provenance).model_validate(payload)


def test_parquet_has_auditable_columns_and_fails_on_structured_provenance_mismatch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "oracle.parquet"
    record = _record()
    write_oracle_records(path, (record,))

    columns = {
        row[0]
        for row in duckdb.sql(
            "DESCRIBE SELECT * FROM read_parquet(?)", params=[str(path)]
        ).fetchall()
    }
    assert {
        "record_id",
        "video_id",
        "question_id",
        "source_split",
        "source_manifest_hash",
        "normalization_manifest_hash",
        "selected_preference",
        "record_json",
    } <= columns
    assert load_oracle_records(path) == (record,)

    duckdb.sql(
        "UPDATE read_parquet(?) SET source_split='dev'", params=[str(path)]
    ) if False else None
    tampered = tmp_path / "tampered.parquet"
    quoted_source = str(path).replace("'", "''")
    quoted_target = str(tampered).replace("'", "''")
    duckdb.sql(
        f"COPY (SELECT * REPLACE ('dev' AS source_split) FROM read_parquet('{quoted_source}')) "
        f"TO '{quoted_target}' (FORMAT PARQUET)"
    )
    shutil.copyfile(
        f"{path}.authority.json",
        f"{tampered}.authority.json",
    )
    with pytest.raises(ValueError, match="structured.*provenance"):
        load_oracle_records(tampered)

    dev = _record(split="dev")
    with pytest.raises(ValueError, match="video.*split"):
        OracleBCDataset((record, dev))


def test_candidate_metadata_enters_all_three_heads_and_padding_is_inert() -> None:
    torch.manual_seed(7)
    model = _model().eval()
    base = _record()
    changed = _record(
        state=_state(
            first_fidelity=FidelityLevel.VISUAL,
            first_frontier=(4, 3),
        )
    )
    collator = _collator()

    with torch.no_grad():
        base_output = model(collator((base,)))
        changed_output = model(collator((changed,)))
    assert not torch.allclose(base_output.action_logits, changed_output.action_logits)
    assert not torch.allclose(
        base_output.sufficiency_logit, changed_output.sufficiency_logit
    )
    assert not torch.allclose(base_output.cost_to_go, changed_output.cost_to_go)

    larger = _record(
        actions=_actions()
        + (ActionInstance(ActionType.VERIFY_VISUAL, "e-right", "high"),)
    )
    with torch.no_grad():
        padded = model(collator((base, larger)))
    assert torch.allclose(base_output.action_logits[0], padded.action_logits[0, :3])
    assert torch.allclose(base_output.sufficiency_logit[0], padded.sufficiency_logit[0])
    assert torch.allclose(base_output.cost_to_go[0], padded.cost_to_go[0])


def _tokenizer_identity(model_id: str, revision: str) -> TokenizerIdentity:
    return TokenizerIdentity(
        implementation="tests.local-tokenizer",
        model_id=model_id,
        revision=revision,
        vocab_sha256="4" * 64,
        artifact_sha256="5" * 64,
    )


def test_production_encoder_identity_is_pinned_and_rejects_test_or_floating_encoder() -> (
    None
):
    with pytest.raises(ValidationError, match="revision"):
        EncoderIdentity(
            kind="pretrained",
            model_id="FacebookAI/roberta-base",
            revision="main",
            tokenizer=_tokenizer_identity("FacebookAI/roberta-base", "main"),
            trust_remote_code=False,
        )
    test_identity = EncoderIdentity.test_identity("offline")
    with pytest.raises(ValidationError, match="production"):
        RouterModelConfig(
            encoder=test_identity,
            encoder_output_dim=16,
            hidden_dim=24,
            production=True,
        )

    pinned = EncoderIdentity(
        kind="pretrained",
        model_id="FacebookAI/roberta-base",
        revision="e2da8e2f811d1448a5b465c236feacd80ffbac7b",
        tokenizer=_tokenizer_identity(
            "FacebookAI/roberta-base",
            "e2da8e2f811d1448a5b465c236feacd80ffbac7b",
        ),
        trust_remote_code=False,
    )
    production = RouterModelConfig(
        encoder=pinned,
        encoder_output_dim=768,
        hidden_dim=384,
        production=True,
        enforce_parameter_range=True,
        min_total_parameters=100_000_000,
        max_total_parameters=150_000_000,
    )
    with pytest.raises(ValueError, match="identity"):
        MemoryRouter(
            production,
            text_encoder=TestTextEncoder(test_identity, 257, 768),
        )


@pytest.mark.parametrize(
    "mutation",
    (
        {"action_logits": torch.tensor([[0.0, float("nan")]])},
        {"target_action_index": torch.tensor([2])},
        {
            "target_action_index": torch.tensor([1]),
            "legal_action_mask": torch.tensor([[True, False]]),
        },
        {"sufficiency_target": torch.tensor([0.5])},
        {"cost_to_go_target": torch.tensor([-1.0])},
        {"target_action_index": torch.tensor([0.0])},
    ),
)
def test_loss_fails_closed_on_bad_outputs_masks_targets_or_dtypes(
    mutation: dict[str, torch.Tensor],
) -> None:
    from fidmem.router.model import RouterOutput

    output = RouterOutput(
        action_logits=torch.tensor([[1.0, 0.0]]),
        sufficiency_logit=torch.tensor([0.0]),
        cost_to_go=torch.tensor([1.0]),
    )
    targets = {
        "target_action_index": torch.tensor([0]),
        "legal_action_mask": torch.tensor([[True, True]]),
        "sufficiency_target": torch.tensor([0.0]),
        "cost_to_go_target": torch.tensor([1.0]),
    }
    output_fields = {"action_logits", "sufficiency_logit", "cost_to_go"}
    output_updates = {
        key: value for key, value in mutation.items() if key in output_fields
    }
    target_updates = {
        key: value for key, value in mutation.items() if key not in output_fields
    }
    output = output._replace(**output_updates)
    targets.update(target_updates)
    with pytest.raises(ValueError):
        behavior_cloning_loss(output, targets, LossProfile.main())


def test_loss_profile_is_audited_and_main_weights_cannot_be_overridden() -> None:
    main = LossProfile.main()
    assert main.name == "main-v1"
    assert main.weights == LossWeights(action=1, sufficiency=0.3, cost_to_go=0.1)
    assert main.frozen
    with pytest.raises(ValidationError, match="main"):
        LossProfile(
            name="main-v1",
            version=1,
            weights=LossWeights(action=1, sufficiency=0.2, cost_to_go=0.1),
            reason="changed",
            frozen=True,
        )
    with pytest.raises(ValidationError, match="reason"):
        LossProfile(
            name="dev-tuned",
            version=1,
            weights=LossWeights(action=1, sufficiency=0.2, cost_to_go=0.1),
            reason="",
            frozen=True,
        )


def test_train_config_contains_artifacts_and_seed_sweep_requires_three_unique_seeds(
    tmp_path: Path,
) -> None:
    profile = LossProfile.main()
    training = TrainConfig(
        seed=1,
        max_steps=1,
        batch_size=2,
        learning_rate=0.01,
        artifact_root=tmp_path,
        checkpoint_path=Path("seed-1/router.pt"),
        checkpoint_every=1,
        device="cpu",
        loss_profile=profile,
    )
    with pytest.raises(ValidationError, match="three unique"):
        TrainFileConfig(
            model_config_path=Path("router-test.yaml"),
            training=training,
            seeds=(1, 1, 2),
        )
    with pytest.raises(ValidationError, match="artifact"):
        training.model_copy(
            update={"checkpoint_path": Path("../outside.pt")}
        ).validated_checkpoint_path()


def test_nonleaking_candidate_relation_toy_overfits_and_seed_sweep_isolated(
    tmp_path: Path,
) -> None:
    records = []
    for index in range(32):
        state = _state()
        actions = _actions(reversed_order=bool(index % 2))
        video_id = f"video-{index:03d}"
        records.append(
            _record(
                state=state,
                actions=actions,
                video_id=video_id,
                question_id=f"q-{index:03d}",
            )
        )
    dataset = OracleBCDataset(tuple(records))
    model = _model()
    training = TrainConfig(
        seed=5,
        max_steps=200,
        batch_size=32,
        learning_rate=0.02,
        artifact_root=tmp_path,
        checkpoint_path=Path("single/router.pt"),
        checkpoint_every=200,
        device="cpu",
        loss_profile=LossProfile.main(),
    )
    result = train_bc(model, dataset, training)
    assert result.action_accuracy >= 0.95

    runs = run_seed_sweep(
        model_factory=_model,
        dataset=dataset,
        training=training.model_copy(update={"max_steps": 1, "checkpoint_every": 1}),
        seeds=(11, 22, 33),
    )
    assert tuple(run.seed for run in runs) == (11, 22, 33)
    assert len({run.checkpoint_path for run in runs}) == 3
    assert all(run.checkpoint_path.is_file() for run in runs)
    assert len({run.dataset_identity for run in runs}) == 1


def test_checkpoint_rejects_tamper_and_records_exact_resume_environment(
    tmp_path: Path,
) -> None:
    dataset = OracleBCDataset((_record(),))
    training = TrainConfig(
        seed=1,
        max_steps=1,
        batch_size=1,
        learning_rate=0.01,
        artifact_root=tmp_path,
        checkpoint_path=Path("run/router.pt"),
        checkpoint_every=1,
        device="cpu",
        loss_profile=LossProfile.main(),
    )
    model = _model()
    result = train_bc(model, dataset, training)
    saved = torch.load(result.checkpoint_path, weights_only=True)
    runtime = RuntimeIdentity.model_validate(saved["runtime_identity"])
    checkpoint = load_checkpoint(
        result.checkpoint_path,
        expected_config_hash=result.config_hash,
        expected_dataset_identity=dataset.identity,
        expected_split_manifest=result.split_manifest,
        expected_encoder_identity=model.config.encoder,
        expected_tokenizer_identity=model.config.encoder.tokenizer,
        expected_loss_profile=LossProfile.main(),
        expected_runtime_identity=runtime,
        expected_git_commit=saved["git_commit"],
        expected_device="cpu",
    )
    assert checkpoint["rng_state"]["source_device"] == "cpu"
    assert checkpoint["rng_state"]["cuda_device_count"] == 0
    assert checkpoint["checkpoint_self_hash"]

    payload = torch.load(result.checkpoint_path, weights_only=True)
    payload["step"] = 99
    torch.save(payload, result.checkpoint_path)
    with pytest.raises(ValueError, match="self hash"):
        load_checkpoint(
            result.checkpoint_path,
            expected_config_hash=result.config_hash,
            expected_dataset_identity=dataset.identity,
            expected_split_manifest=result.split_manifest,
            expected_encoder_identity=model.config.encoder,
            expected_tokenizer_identity=model.config.encoder.tokenizer,
            expected_loss_profile=LossProfile.main(),
            expected_runtime_identity=runtime,
            expected_git_commit=saved["git_commit"],
            expected_device="cpu",
        )


def test_production_yaml_is_pinned_100_to_150m_and_supports_offline_encoder_injection() -> (
    None
):
    raw = OmegaConf.to_container(
        OmegaConf.load("configs/model/router.yaml"), resolve=True
    )
    config = RouterModelConfig.model_validate(raw)
    assert config.production
    assert config.encoder.kind == "pretrained"
    assert config.min_total_parameters == 100_000_000
    assert config.max_total_parameters == 150_000_000

    class SizedOfflineEncoder(torch.nn.Module):
        def __init__(self, identity: EncoderIdentity) -> None:
            super().__init__()
            self.identity = identity
            self.capacity = torch.nn.Parameter(torch.empty(100_000_000, device="meta"))

        def forward(
            self, token_ids: torch.Tensor, token_mask: torch.Tensor
        ) -> torch.Tensor:
            return torch.zeros(
                (*token_ids.shape, 8),
                dtype=torch.float32,
                device=token_ids.device,
            )

    sized_config = config.model_copy(
        update={
            "encoder_output_dim": 8,
            "hidden_dim": 8,
            "action_type_embedding_dim": 2,
            "fidelity_embedding_dim": 2,
        }
    )
    model = MemoryRouter(
        sized_config,
        text_encoder=SizedOfflineEncoder(config.encoder),
    )
    assert 100_000_000 <= model.total_parameter_count <= 150_000_000


def test_production_factory_is_lazy_pinned_and_local_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    identity = EncoderIdentity(
        kind="pretrained",
        model_id="org/backbone",
        revision="1" * 40,
        tokenizer=_tokenizer_identity("org/tokenizer", "2" * 40),
        trust_remote_code=False,
    )
    calls: list[tuple[str, str, dict[str, object]]] = []
    snapshot = tmp_path / "models--org--tokenizer" / "snapshots" / ("2" * 40)
    snapshot.mkdir(parents=True)

    class SnapshotLoader:
        @staticmethod
        def resolve(
            model_id: str,
            **kwargs: object,
        ) -> str:
            calls.append(("snapshot", model_id, kwargs))
            return str(snapshot)

    class TokenizerLoader:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> object:
            calls.append(("tokenizer", model_id, kwargs))
            return object()

    class ModelLoader:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> torch.nn.Module:
            calls.append(("model", model_id, kwargs))
            return torch.nn.Identity()

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=SnapshotLoader.resolve),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModel=ModelLoader, AutoTokenizer=TokenizerLoader),
    )
    encoder, tokenizer = ProductionEncoderFactory.load(identity)

    assert encoder.identity == identity
    assert tokenizer is not None
    assert calls == [
        (
            "snapshot",
            "org/tokenizer",
            {
                "revision": "2" * 40,
                "local_files_only": True,
            },
        ),
        (
            "tokenizer",
            str(snapshot.resolve()),
            {
                "local_files_only": True,
                "trust_remote_code": False,
            },
        ),
        (
            "model",
            "org/backbone",
            {
                "local_files_only": True,
                "revision": "1" * 40,
                "trust_remote_code": False,
            },
        ),
    ]
