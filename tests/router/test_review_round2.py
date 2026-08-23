from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import torch

from fidmem.actions.environment import ActionObservation, EnvironmentTransition
from fidmem.agent.answerer import FrozenAnswerer
from fidmem.data.longroute import (
    BUILDER_VERSION,
    MANIFEST_VERSION,
    DatasetManifest,
    LongRouteExample,
    SourceEvent,
    SourceQuestion,
    SourceVideo,
    TrainSplitManifest,
    VirtualSegment,
    _canonical_source_manifest,
    _source_provenance,
)
from fidmem.oracle.labels import COST_PREFERENCES, CostNormalization, PreferenceLabel
from fidmem.oracle.search import OraclePath
from fidmem.router.dataset import (
    FrozenComponentIdentity,
    HFTokenizerAdapter,
    OracleBCDataset,
    TestByteTokenizer,
    TokenizerIdentity,
    build_grouped_split,
    load_oracle_records,
    materialize_oracle_record,
    seal_sufficiency_label,
    write_oracle_records,
)
from fidmem.router.model import EncoderIdentity, MemoryRouter, RouterModelConfig
from fidmem.router.train_bc import (
    LossProfile,
    RuntimeIdentity,
    TrainConfig,
    current_runtime_identity,
    load_checkpoint,
    train_bc,
)
from fidmem.types import ActionInstance, ActionType, FidelityLevel, RouterState


def _published(
    *,
    question_id: str = "q-1",
    video_id: str = "video-1",
    split: str = "train",
) -> tuple[DatasetManifest, LongRouteExample, TrainSplitManifest]:
    event_id = f"{video_id}:event-1"
    example = LongRouteExample(
        question_id=question_id,
        split=split,
        question="Which candidate is supported?",
        options=("yes", "no"),
        answer="yes",
        target_source_video_id=video_id,
        target_event_id=event_id,
        target_position=0,
        supporting_event_ids=(event_id,),
        template="single_event",
        segments=(
            VirtualSegment(
                source_video_id=video_id,
                event_id="event-1",
                source_start_sec=0,
                source_end_sec=600,
                global_start_sec=0,
                global_end_sec=600,
            ),
        ),
        duration_sec=600,
    )
    asset_hash = "2" * 64
    source_manifest = TrainSplitManifest(
        name="unit",
        dataset_version="v1",
        source_uri="file:///source.json",
        license="test",
        split="train",
        videos=(
            SourceVideo(
                video_id=video_id,
                path=Path(f"{video_id}.mp4"),
                source_uri=f"file:///{video_id}.mp4",
                content_sha256=asset_hash,
                split="train",
                licensed=True,
                frame_embeddings=tuple((1.0, float(index + 1)) for index in range(8)),
                events=(
                    SourceEvent(
                        event_id="event-1",
                        start_sec=0,
                        end_sec=600,
                        label="event one",
                        embedding=(1.0, 0.5),
                    ),
                ),
                questions=(
                    SourceQuestion(
                        question_id=question_id,
                        question=example.question,
                        options=example.options,
                        answer=example.answer,
                        target_event_id="event-1",
                    ),
                ),
            ),
        ),
    )
    source = _source_provenance(_canonical_source_manifest(source_manifest))
    source_hash = source.canonical_sha256

    manifest = DatasetManifest(
        manifest_version=MANIFEST_VERSION,
        schema_version=MANIFEST_VERSION,
        builder_version=BUILDER_VERSION,
        seed=7,
        source_manifest_hashes={source.identity: source_hash},
        source_manifests=(source,),
        builder_config={"dev_fraction": 0.2},
        group_assignment={video_id: split},
        split_statistics={"train": int(split == "train"), "dev": int(split == "dev")},
        multi_event_ratio=0,
        leakage_audit_uri="generation/leakage.json",
        leakage_parquet_uri="generation/leakage.parquet",
        examples=(example,),
        asset_sha256s={video_id: asset_hash},
        generation_uri="generation",
    )
    return manifest, example, source_manifest


def _state(example: LongRouteExample) -> RouterState:
    return RouterState(
        question=example.question,
        options=example.options,
        evidence=(),
        action_history=(),
        remaining_budget=10,
        candidate_event_ids=(example.target_event_id,),
        candidate_fidelity_levels={example.target_event_id: FidelityLevel.GIST},
        context_frontiers={example.target_event_id: (0, 0)},
        cost_preference=0.3,
    )


def _labels(
    state: RouterState,
    action: ActionInstance,
    *,
    reported_total_cost: float = 2,
    normalization_constant: float = 10,
) -> tuple[PreferenceLabel, ...]:
    next_state = state.model_copy(update={"action_history": (action,)})
    transition = EnvironmentTransition(
        state=state,
        action=action,
        observation=ActionObservation(
            action_type=action.action_type,
            target_event_id=action.event_id,
        ),
        next_state=next_state,
        step_cost=2,
    )
    labels = []
    for preference in COST_PREFERENCES:
        utility = 1 - preference * (reported_total_cost / normalization_constant)
        labels.append(
            PreferenceLabel(
                cost_preference=preference,
                utility=utility,
                optimal_paths=(
                    OraclePath(
                        transitions=(transition,),
                        answer="yes",
                        answer_score=1,
                        correct=True,
                        total_cost=reported_total_cost,
                        utility=utility,
                    ),
                ),
            )
        )
    return tuple(labels)


def _record(
    *,
    split: str = "train",
    question_id: str = "q-1",
    video_id: str = "video-1",
    normalization: CostNormalization | None = None,
):
    manifest, example, source_manifest = _published(
        question_id=question_id,
        video_id=video_id,
        split=split,
    )
    state = _state(example)
    action = ActionInstance(ActionType.EXPAND_RESIDUAL, example.target_event_id, None)
    sufficiency = seal_sufficiency_label(
        state=state,
        question_id=example.question_id,
        gold_answer=example.answer,
        answerer=FrozenAnswerer(lambda _: "yes"),
        answerer_identity=FrozenComponentIdentity(
            implementation="unit-answerer",
            model_id="answerer",
            revision="a" * 40,
            artifact_sha256="3" * 64,
        ),
    )
    resolved_normalization = normalization or CostNormalization(
        constant=10, sample_count=100, source_split="train"
    )
    return materialize_oracle_record(
        observation_snapshot_id="cached-observations",
        state=state,
        action_instances=(action, ActionInstance(ActionType.STOP, None, None)),
        legal_action_mask=(True, True),
        preference_labels=_labels(
            state,
            action,
            normalization_constant=resolved_normalization.constant,
        ),
        normalization=resolved_normalization,
        manifest=manifest,
        example=example,
        source_manifests=(source_manifest,),
        sufficiency_artifact=sufficiency,
    )


def test_materializer_derives_canonical_lineage_cost_and_sufficiency(
    tmp_path: Path,
) -> None:
    record = _record()
    manifest, _, _ = _published()
    expected_manifest_hash = hashlib.sha256(
        manifest.canonical_json().encode("utf-8")
    ).hexdigest()
    expected_normalization_hash = hashlib.sha256(
        json.dumps(
            record.provenance.normalization.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

    assert record.provenance.dataset_manifest_hash == expected_manifest_hash
    assert record.provenance.normalization_manifest_hash == expected_normalization_hash
    assert record.cost_to_go == 0.2
    assert record.sufficiency_target == 1
    assert record.question_id == "q-1"
    assert record.video_id == "video-1"

    path = tmp_path / "records.parquet"
    write_oracle_records(path, (record,))
    assert load_oracle_records(path) == (record,)


def test_materializer_rejects_forged_task8_path_cost_and_bare_sufficiency() -> None:
    manifest, example, source_manifest = _published()
    state = _state(example)
    action = ActionInstance(ActionType.EXPAND_RESIDUAL, example.target_event_id, None)
    normalization = CostNormalization(
        constant=10, sample_count=100, source_split="train"
    )
    artifact = seal_sufficiency_label(
        state=state,
        question_id=example.question_id,
        gold_answer=example.answer,
        answerer=FrozenAnswerer(lambda _: "yes"),
        answerer_identity=FrozenComponentIdentity(
            implementation="unit-answerer",
            model_id="answerer",
            revision="a" * 40,
            artifact_sha256="3" * 64,
        ),
    )
    kwargs = dict(
        observation_snapshot_id="cache",
        state=state,
        action_instances=(action, ActionInstance(ActionType.STOP, None, None)),
        legal_action_mask=(True, True),
        preference_labels=_labels(state, action),
        normalization=normalization,
        manifest=manifest,
        example=example,
        source_manifests=(source_manifest,),
        sufficiency_artifact=artifact,
    )

    forged_hash = manifest.model_copy(
        update={"source_manifest_hashes": {"source-1": "f" * 64}}
    )
    with pytest.raises(ValueError, match="source manifest"):
        materialize_oracle_record(**(kwargs | {"manifest": forged_hash}))

    forged_group = manifest.model_copy(update={"group_assignment": {"video-1": "dev"}})
    with pytest.raises(ValueError, match="group"):
        materialize_oracle_record(**(kwargs | {"manifest": forged_group}))

    with pytest.raises(ValueError, match="path.*cost"):
        materialize_oracle_record(
            **(
                kwargs
                | {
                    "preference_labels": _labels(
                        state, action, reported_total_cost=999999
                    )
                }
            )
        )

    with pytest.raises((TypeError, ValueError), match="sufficiency"):
        materialize_oracle_record(**(kwargs | {"sufficiency_artifact": 1}))


def test_dataset_rejects_mixed_or_forged_normalization_identity() -> None:
    first = _record()
    second = _record(
        question_id="q-2",
        video_id="video-2",
        normalization=CostNormalization(
            constant=20, sample_count=100, source_split="train"
        ),
    )
    with pytest.raises(ValueError, match="normalization"):
        OracleBCDataset((first, second))

    forged = first.model_copy(
        update={
            "provenance": first.provenance.model_copy(
                update={"normalization_manifest_hash": "f" * 64}
            )
        }
    )
    with pytest.raises(ValueError, match="normalization"):
        OracleBCDataset((forged,))


def test_grouped_split_never_maps_upstream_dev_into_train() -> None:
    train = _record(split="train", question_id="q-train", video_id="video-train")
    dev = _record(split="dev", question_id="q-dev", video_id="video-dev")

    assignment = build_grouped_split(
        (train, dev),
        seed=99,
        train_fraction=1,
        dev_fraction=0,
    )

    assert assignment.assignment_source == "upstream_source_split"
    assert assignment.train_record_ids == (train.record_id,)
    assert assignment.dev_record_ids == (dev.record_id,)
    assert dev.record_id not in assignment.train_record_ids


def test_dataset_authority_sidecar_is_required_content_addressed_and_tamper_evident(
    tmp_path: Path,
) -> None:
    record = _record()
    path = tmp_path / "records.parquet"
    write_oracle_records(path, (record,))
    authority = Path(f"{path}.authority.json")
    assert authority.is_file()

    payload = json.loads(authority.read_text(encoding="utf-8"))
    assert payload["dataset_manifests"][record.provenance.dataset_manifest_hash]
    assert set(payload["record_digests"]) == {record.record_id}

    authority.unlink()
    with pytest.raises(ValueError, match="authority.*missing"):
        load_oracle_records(path)

    write_oracle_records(path, (record,))
    payload = json.loads(authority.read_text(encoding="utf-8"))
    payload["record_digests"][record.record_id] = "f" * 64
    authority.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="authority.*hash|record.*digest"):
        load_oracle_records(path)


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _tiny_model(name: str = "offline-test") -> MemoryRouter:
    identity = EncoderIdentity.test_identity(name)
    return MemoryRouter(
        RouterModelConfig(
            encoder=identity,
            encoder_output_dim=8,
            hidden_dim=12,
            action_type_embedding_dim=4,
            fidelity_embedding_dim=2,
            max_question_tokens=128,
            max_item_tokens=64,
        )
    )


def _tiny_training(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        seed=7,
        max_steps=1,
        batch_size=1,
        learning_rate=0.01,
        artifact_root=tmp_path,
        checkpoint_path=Path("run/router.pt"),
        checkpoint_every=1,
        device="cpu",
        loss_profile=LossProfile.main(),
    )


def test_train_rejects_tokenizer_implementation_or_model_impersonation(
    tmp_path: Path,
) -> None:
    model = _tiny_model("offline-test")
    wrong_model = TestByteTokenizer(model_id="offline-byte")
    with pytest.raises(ValueError, match="tokenizer identity"):
        train_bc(
            model,
            OracleBCDataset((_record(),)),
            _tiny_training(tmp_path),
            tokenizer=wrong_model,
        )

    forged = TokenizerIdentity(
        implementation="offline-test",
        model_id="offline-test",
        revision="offline-test-v1",
        vocab_sha256=wrong_model.identity.vocab_sha256,
        artifact_sha256=wrong_model.identity.artifact_sha256,
    )
    wrong_model.identity = forged
    with pytest.raises(ValueError, match="tokenizer identity"):
        train_bc(
            _tiny_model("offline-test"),
            OracleBCDataset((_record(),)),
            _tiny_training(tmp_path),
            tokenizer=wrong_model,
        )


def test_checkpoint_binds_git_tokenizer_and_deterministic_runtime(
    tmp_path: Path,
) -> None:
    model = _tiny_model()
    dataset = OracleBCDataset((_record(),))
    training = _tiny_training(tmp_path)
    result = train_bc(model, dataset, training)
    payload = torch.load(result.checkpoint_path, weights_only=True)

    runtime = RuntimeIdentity.model_validate(payload["runtime_identity"])
    assert runtime == current_runtime_identity("cpu")
    assert payload["tokenizer_identity"] == (
        model.config.encoder.tokenizer.model_dump(mode="json")
    )
    assert runtime.deterministic_algorithms
    assert runtime.cudnn_deterministic
    assert not runtime.cudnn_benchmark

    for invalid_commit in ("0" * 40, "unknown", "A" * 40):
        with pytest.raises(ValueError, match="git commit|40 lowercase"):
            load_checkpoint(
                result.checkpoint_path,
                expected_config_hash=result.config_hash,
                expected_dataset_identity=dataset.identity,
                expected_split_manifest=result.split_manifest,
                expected_encoder_identity=model.config.encoder,
                expected_tokenizer_identity=model.config.encoder.tokenizer,
                expected_loss_profile=training.loss_profile,
                expected_runtime_identity=runtime,
                expected_git_commit=invalid_commit,
                expected_device="cpu",
            )

    mismatched_runtime = runtime.model_copy(
        update={"torch_version": runtime.torch_version + "-different"}
    )
    with pytest.raises(ValueError, match="runtime"):
        load_checkpoint(
            result.checkpoint_path,
            expected_config_hash=result.config_hash,
            expected_dataset_identity=dataset.identity,
            expected_split_manifest=result.split_manifest,
            expected_encoder_identity=model.config.encoder,
            expected_tokenizer_identity=model.config.encoder.tokenizer,
            expected_loss_profile=training.loss_profile,
            expected_runtime_identity=mismatched_runtime,
            expected_git_commit=_git_head(),
            expected_device="cpu",
        )


def test_hf_tokenizer_identity_is_derived_from_actual_local_snapshot(
    tmp_path: Path,
) -> None:
    revision = "a" * 40
    snapshot = tmp_path / "models--org--actual-tokenizer" / "snapshots" / revision
    snapshot.mkdir(parents=True)

    class Backend:
        @staticmethod
        def to_str() -> str:
            return '{"actual":"backend"}'

    class LocalTokenizer:
        name_or_path = str(snapshot)
        init_kwargs = {"_commit_hash": "f" * 40, "name_or_path": "forged/repo"}
        special_tokens_map = {"pad_token": "<pad>"}
        backend_tokenizer = Backend()

        @staticmethod
        def get_vocab() -> dict[str, int]:
            return {"<pad>": 0, "actual": 1}

        @staticmethod
        def encode(
            text: str,
            *,
            add_special_tokens: bool,
            truncation: bool,
        ) -> list[int]:
            del text, add_special_tokens, truncation
            return [1]

    adapter = HFTokenizerAdapter(LocalTokenizer())
    assert adapter.identity.model_id == "org/actual-tokenizer"
    assert adapter.identity.revision == revision
    assert adapter.identity.revision != LocalTokenizer.init_kwargs["_commit_hash"]
    assert adapter.identity.implementation.endswith(".LocalTokenizer")
    assert adapter.identity.vocab_sha256 != "0" * 64
    assert adapter.identity.artifact_sha256 != "0" * 64
