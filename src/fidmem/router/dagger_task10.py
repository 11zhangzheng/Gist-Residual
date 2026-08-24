"""Production Task 10 adapter and strict Task 11 experiment loader."""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Callable
from pathlib import Path

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field

from fidmem.agent.runner import RouterPolicy

from .dagger_core import BCPolicy, Deviation
from .dagger_workflow import DAggerConfig, PolicyTrainingResult
from .dataset import (
    HFTokenizerAdapter,
    OracleBCDataset,
    OracleBCRecord,
    TestByteTokenizer,
    TextTokenizer,
    TokenizerIdentity,
    build_grouped_split,
)
from .model import MemoryRouter, ProductionEncoderFactory, RouterModelConfig
from .train_bc import (
    TrainConfig,
    TrainResult,
    canonical_config_hash,
    configure_deterministic_runtime,
    load_checkpoint,
    train_bc,
)

ModelFactory = Callable[[], MemoryRouter | tuple[MemoryRouter, TextTokenizer]]
RecordMaterializer = Callable[[Deviation], OracleBCRecord]
TrainFunction = Callable[..., TrainResult]


class DAggerFileConfig(BaseModel):
    """Every experiment field is consumed by loading, training, or the runner."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        protected_namespaces=(),
    )

    model_config_path: Path
    model_overrides: dict[str, object] = Field(default_factory=dict)
    training: TrainConfig
    dagger: DAggerConfig


def load_dagger_file(
    path: str | Path,
) -> tuple[RouterModelConfig, DAggerFileConfig]:
    source = Path(path)
    raw = OmegaConf.to_container(OmegaConf.load(source), resolve=True)
    file_config = DAggerFileConfig.model_validate(raw)
    model_path = file_config.model_config_path
    if not model_path.is_absolute():
        model_path = (source.parent / model_path).resolve()
    model_raw = OmegaConf.to_container(OmegaConf.load(model_path), resolve=True)
    if not isinstance(model_raw, dict):
        raise ValueError("router model config must be a mapping")
    model_raw.update(file_config.model_overrides)
    model_config = RouterModelConfig.model_validate(model_raw)
    if file_config.training.device not in {"cpu", "cuda"}:
        raise ValueError("Task 10 training device is invalid")
    return model_config, file_config


def _split_factory_result(
    built: MemoryRouter | tuple[MemoryRouter, TextTokenizer],
) -> tuple[MemoryRouter, TextTokenizer]:
    if isinstance(built, tuple):
        model, tokenizer = built
    else:
        model = built
        if model.config.encoder.kind == "pretrained":
            raise ValueError("pretrained DAgger trainer requires its pinned tokenizer")
        tokenizer = TestByteTokenizer(model.config.encoder.tokenizer.model_id)
    identity = getattr(tokenizer, "identity", None)
    if not isinstance(identity, TokenizerIdentity):
        raise ValueError("DAgger trainer tokenizer lacks Task 10 identity")
    if identity != model.config.encoder.tokenizer:
        raise ValueError("DAgger trainer tokenizer identity does not match model")
    return model, tokenizer


class Task10PolicyTrainer:
    """Aggregate BC plus all deviations and call Task 10's public train_bc."""

    def __init__(
        self,
        *,
        model_factory: ModelFactory,
        training: TrainConfig,
        record_materializer: RecordMaterializer,
        train_function: TrainFunction = train_bc,
    ) -> None:
        self._model_factory = model_factory
        self._training = TrainConfig.model_validate(training.model_dump(mode="python"))
        self._record_materializer = record_materializer
        self._train_function = train_function

    def _aggregate(
        self,
        base_dataset: OracleBCDataset,
        deviations: tuple[Deviation, ...],
    ) -> OracleBCDataset:
        if not all(isinstance(deviation, Deviation) for deviation in deviations):
            raise TypeError("Task 10 adapter requires validated DAgger deviations")
        records = tuple(base_dataset.records) + tuple(
            self._record_materializer(deviation) for deviation in deviations
        )
        return OracleBCDataset(records)

    def _round_config(self, output_checkpoint: Path) -> TrainConfig:
        return TrainConfig.model_validate(
            self._training.model_copy(
                update={"checkpoint_path": output_checkpoint}
            ).model_dump(mode="python")
        )

    def train(
        self,
        *,
        round_number: int,
        base_dataset: OracleBCDataset,
        deviations: tuple[Deviation, ...],
        new_deviations: tuple[Deviation, ...],
        source_policy_checkpoint: Path,
        output_checkpoint: Path,
    ) -> PolicyTrainingResult:
        if round_number < 1:
            raise ValueError("DAgger round number must be positive")
        if not source_policy_checkpoint.is_file():
            raise ValueError("DAgger source policy checkpoint is missing")
        if output_checkpoint.exists():
            raise ValueError("refusing to overwrite an unmanifested round checkpoint")
        if any(
            item.state_key not in {entry.state_key for entry in deviations}
            for item in new_deviations
        ):
            raise ValueError("new deviations must be included in the aggregate")
        aggregated = self._aggregate(base_dataset, deviations)
        model, tokenizer = _split_factory_result(self._model_factory())
        config = self._round_config(output_checkpoint)
        result = self._train_function(
            model,
            aggregated,
            config,
            resume=False,
            tokenizer=tokenizer,
        )
        checkpoint = Path(result.checkpoint_path).resolve()
        expected = output_checkpoint.resolve()
        if (
            checkpoint != expected
            or not checkpoint.is_file()
            or checkpoint.is_symlink()
        ):
            raise ValueError("Task 10 trainer did not freeze the requested checkpoint")
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if result.dataset_identity != aggregated.identity:
            raise ValueError("Task 10 trainer returned the wrong dataset identity")
        return PolicyTrainingResult(
            policy=BCPolicy(model, tokenizer=tokenizer),
            checkpoint_path=checkpoint,
            checkpoint_sha256=digest,
            aggregated_dataset_identity=aggregated.identity,
        )

    def load_policy(
        self,
        *,
        checkpoint: Path,
        base_dataset: OracleBCDataset,
        deviations: tuple[Deviation, ...],
    ) -> RouterPolicy:
        """Restore a manifested Task 10 checkpoint using its public validator."""

        aggregated = self._aggregate(base_dataset, deviations)
        model, tokenizer = _split_factory_result(self._model_factory())
        config = self._round_config(checkpoint)
        runtime = configure_deterministic_runtime(config.device)
        split = build_grouped_split(
            aggregated.records,
            seed=config.split_seed,
            train_fraction=config.train_fraction,
            dev_fraction=config.dev_fraction,
        )
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        payload = load_checkpoint(
            checkpoint,
            expected_config_hash=canonical_config_hash(model, config),
            expected_dataset_identity=aggregated.identity,
            expected_split_manifest=split,
            expected_encoder_identity=model.config.encoder,
            expected_tokenizer_identity=tokenizer.identity,
            expected_loss_profile=config.loss_profile,
            expected_runtime_identity=runtime,
            expected_git_commit=commit,
            expected_device=config.device,
        )
        model.load_state_dict(payload["model"], strict=True)  # type: ignore[arg-type]
        return BCPolicy(model, tokenizer=tokenizer)


def build_task10_policy_trainer(
    *,
    model_config: RouterModelConfig,
    file_config: DAggerFileConfig,
    record_materializer: RecordMaterializer,
) -> Task10PolicyTrainer:
    """Consume model overrides/training config and build the production adapter."""

    def model_factory() -> MemoryRouter | tuple[MemoryRouter, TextTokenizer]:
        if model_config.encoder.kind == "pretrained":
            encoder, raw_tokenizer = ProductionEncoderFactory.load(model_config.encoder)
            tokenizer = HFTokenizerAdapter(raw_tokenizer)
            return MemoryRouter(model_config, text_encoder=encoder), tokenizer
        return MemoryRouter(model_config)

    return Task10PolicyTrainer(
        model_factory=model_factory,
        training=file_config.training,
        record_materializer=record_materializer,
    )


__all__ = [
    "DAggerFileConfig",
    "Task10PolicyTrainer",
    "build_task10_policy_trainer",
    "load_dagger_file",
]
