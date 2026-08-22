"""Behavior-cloning training and exact, CPU-compatible checkpoint resume."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, NamedTuple

import numpy as np
import torch
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor, nn

from fidmem.oracle.labels import COST_PREFERENCES, CostNormalization
from fidmem.types import (
    ActionInstance,
    ActionType,
    EvidenceItem,
    FidelityLevel,
    RouterState,
)

from .dataset import (
    HFTokenizerAdapter,
    OracleBCDataset,
    OracleBCRecord,
    OracleRecordProvenance,
    RouterBatch,
    RouterCollator,
    SplitManifest,
    TextTokenizer,
    build_grouped_split,
    load_oracle_records,
)
from .model import (
    EncoderIdentity,
    MemoryRouter,
    ProductionEncoderFactory,
    RouterModelConfig,
    RouterOutput,
)


class LossWeights(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    action: float = Field(default=1.0, gt=0)
    sufficiency: float = Field(default=0.3, ge=0)
    cost_to_go: float = Field(default=0.1, ge=0)


class LossProfile(BaseModel):
    """Versioned and auditable loss policy."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    name: str = Field(min_length=1)
    version: int = Field(ge=1)
    weights: LossWeights
    reason: str
    frozen: bool = True

    @classmethod
    def main(cls) -> "LossProfile":
        return cls(
            name="main-v1",
            version=1,
            weights=LossWeights(),
            reason="pre-registered Task 10 objective",
            frozen=True,
        )

    @model_validator(mode="after")
    def validate_audit_contract(self) -> "LossProfile":
        if self.name == "main-v1":
            if self.version != 1 or self.weights != LossWeights() or not self.frozen:
                raise ValueError("main-v1 loss profile is immutable")
        elif not self.reason.strip():
            raise ValueError("non-main loss profile requires a recorded reason")
        return self


class TrainConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    seed: int = 13
    max_steps: int = Field(default=20, ge=1)
    batch_size: int = Field(default=16, ge=1)
    learning_rate: float = Field(default=3e-4, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    scheduler_gamma: float = Field(default=1.0, gt=0)
    artifact_root: Path = Path("artifacts")
    checkpoint_path: Path
    checkpoint_every: int = Field(default=20, ge=1)
    device: Literal["cpu", "cuda"] = "cpu"
    split_seed: int = 2026
    train_fraction: float = Field(default=1.0, ge=0, le=1)
    dev_fraction: float = Field(default=0.0, ge=0, le=1)
    loss_profile: LossProfile = Field(default_factory=LossProfile.main)

    @model_validator(mode="after")
    def split_is_valid(self) -> "TrainConfig":
        if self.train_fraction <= 0:
            raise ValueError("train_fraction must be positive")
        if self.train_fraction + self.dev_fraction > 1:
            raise ValueError("train and dev fractions must not exceed one")
        return self

    @model_validator(mode="after")
    def checkpoint_is_contained(self) -> "TrainConfig":
        self._resolved_checkpoint_path()
        return self

    @property
    def loss_weights(self) -> LossWeights:
        return self.loss_profile.weights

    def _resolved_checkpoint_path(self) -> Path:
        root = self.artifact_root.resolve()
        candidate = self.checkpoint_path
        target = (
            candidate.resolve()
            if candidate.is_absolute()
            else (root / candidate).resolve()
        )
        if not target.is_relative_to(root) or target == root:
            raise ValueError("checkpoint path must stay within artifact root")
        if target.suffix.casefold() not in {".pt", ".pth"}:
            raise ValueError("checkpoint path must end in .pt or .pth")
        return target

    def validated_checkpoint_path(self) -> Path:
        validated = TrainConfig.model_validate(self.model_dump(mode="python"))
        return validated._resolved_checkpoint_path()


class LossBreakdown(NamedTuple):
    total: Tensor
    action: Tensor
    sufficiency: Tensor
    cost_to_go: Tensor


@dataclass(frozen=True)
class TrainResult:
    step: int
    action_accuracy: float
    checkpoint_path: Path
    config_hash: str
    split_manifest: SplitManifest
    seed: int
    dataset_identity: str


def behavior_cloning_loss(
    output: RouterOutput,
    targets: RouterBatch | Mapping[str, Tensor],
    profile: LossProfile,
) -> LossBreakdown:
    validated_profile = LossProfile.model_validate(profile.model_dump(mode="python"))
    if isinstance(targets, RouterBatch):
        action_target = targets.target_action_index
        legal_mask = targets.legal_action_mask
        sufficiency_target = targets.sufficiency_target
        cost_target = targets.cost_to_go_target
    else:
        required = {
            "target_action_index",
            "legal_action_mask",
            "sufficiency_target",
            "cost_to_go_target",
        }
        if set(targets) != required:
            raise ValueError("loss targets do not match the strict schema")
        action_target = targets["target_action_index"]
        legal_mask = targets["legal_action_mask"]
        sufficiency_target = targets["sufficiency_target"]
        cost_target = targets["cost_to_go_target"]

    logits = output.action_logits
    batch_size = logits.shape[0] if logits.ndim == 2 else -1
    if batch_size < 1 or not torch.isfinite(logits).all():
        raise ValueError("action logits must be a finite non-empty matrix")
    if legal_mask.dtype != torch.bool or legal_mask.shape != logits.shape:
        raise ValueError("legal action mask must be boolean and match logits")
    if not legal_mask.any(dim=1).all():
        raise ValueError("each loss row must have at least one legal action")
    if action_target.dtype != torch.long or action_target.shape != (batch_size,):
        raise ValueError("action targets must be an int64 vector")
    if (
        (action_target < 0).any()
        or (action_target >= logits.shape[1]).any()
        or not legal_mask.gather(1, action_target.unsqueeze(1)).all()
    ):
        raise ValueError("action target must select a legal candidate")

    for name, value in (
        ("sufficiency logit", output.sufficiency_logit),
        ("cost-to-go output", output.cost_to_go),
        ("sufficiency target", sufficiency_target),
        ("cost-to-go target", cost_target),
    ):
        if value.shape != (batch_size,) or not torch.isfinite(value).all():
            raise ValueError(f"{name} must be a finite batch vector")
    if not torch.logical_or(sufficiency_target == 0, sufficiency_target == 1).all():
        raise ValueError("sufficiency target must be binary")
    if (cost_target < 0).any():
        raise ValueError("cost-to-go target must be non-negative")
    devices = {
        value.device
        for value in (
            logits,
            output.sufficiency_logit,
            output.cost_to_go,
            action_target,
            legal_mask,
            sufficiency_target,
            cost_target,
        )
    }
    if len(devices) != 1:
        raise ValueError("loss outputs and targets must share one device")

    weights = validated_profile.weights
    action = torch.nn.functional.cross_entropy(logits, action_target)
    sufficiency = torch.nn.functional.binary_cross_entropy_with_logits(
        output.sufficiency_logit, sufficiency_target.to(output.sufficiency_logit.dtype)
    )
    cost = torch.nn.functional.smooth_l1_loss(
        output.cost_to_go, cost_target.to(output.cost_to_go.dtype)
    )
    total = (
        weights.action * action
        + weights.sufficiency * sufficiency
        + weights.cost_to_go * cost
    )
    if not torch.isfinite(total):
        raise ValueError("weighted behavior-cloning loss is non-finite")
    return LossBreakdown(total, action, sufficiency, cost)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _config_payload(model: MemoryRouter, config: TrainConfig) -> dict[str, object]:
    training = config.model_dump(mode="json")
    # Run length and local storage location may change on continuation; device
    # remains identity-bearing because exact CPU/CUDA replay is not interchangeable.
    for field in ("max_steps", "artifact_root", "checkpoint_path", "checkpoint_every"):
        training.pop(field)
    return {"model": model.config.model_dump(mode="json"), "training": training}


def canonical_config_hash(model: MemoryRouter, config: TrainConfig) -> str:
    payload = _canonical_json(_config_payload(model, config)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _hash_semantic_value(hasher: "hashlib._Hash", value: object) -> None:
    if isinstance(value, Tensor):
        tensor = value.detach().cpu().contiguous()
        hasher.update(b"T")
        hasher.update(str(tensor.dtype).encode("ascii"))
        hasher.update(_canonical_json(list(tensor.shape)).encode("ascii"))
        hasher.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, Mapping):
        hasher.update(b"M")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _hash_semantic_value(hasher, key)
            _hash_semantic_value(hasher, value[key])
    elif isinstance(value, tuple):
        hasher.update(b"U")
        for item in value:
            _hash_semantic_value(hasher, item)
    elif isinstance(value, list):
        hasher.update(b"L")
        for item in value:
            _hash_semantic_value(hasher, item)
    elif value is None:
        hasher.update(b"N")
    elif isinstance(value, bool):
        hasher.update(b"B1" if value else b"B0")
    elif isinstance(value, int):
        hasher.update(b"I" + str(value).encode("ascii"))
    elif isinstance(value, float):
        hasher.update(b"F" + value.hex().encode("ascii"))
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        hasher.update(b"S" + str(len(encoded)).encode("ascii") + b":" + encoded)
    elif isinstance(value, bytes):
        hasher.update(b"Y" + str(len(value)).encode("ascii") + b":" + value)
    else:
        raise ValueError(f"checkpoint contains unsupported value type {type(value)!r}")


def _checkpoint_self_hash(payload: Mapping[str, object]) -> str:
    hasher = hashlib.sha256()
    _hash_semantic_value(
        hasher,
        {key: value for key, value in payload.items() if key != "checkpoint_self_hash"},
    )
    return hasher.hexdigest()


def _capture_rng(source_device: Literal["cpu", "cuda"]) -> dict[str, object]:
    numpy_state = np.random.get_state()
    cuda_count = torch.cuda.device_count() if source_device == "cuda" else 0
    if source_device == "cuda" and (not torch.cuda.is_available() or cuda_count < 1):
        raise RuntimeError("CUDA RNG capture requested without CUDA devices")
    return {
        "source_device": source_device,
        "cuda_device_count": cuda_count,
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].astype(np.int64, copy=True)),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if source_device == "cuda" else []
        ),
    }


def _validate_rng_state(
    state: Mapping[str, object], expected_device: Literal["cpu", "cuda"]
) -> None:
    required = {
        "source_device",
        "cuda_device_count",
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
    }
    if set(state) != required or state["source_device"] != expected_device:
        raise ValueError("checkpoint RNG state or source device is incomplete")
    numpy_state = state["numpy"]
    if (
        not isinstance(numpy_state, Mapping)
        or set(numpy_state)
        != {"bit_generator", "keys", "position", "has_gauss", "cached_gaussian"}
        or not isinstance(numpy_state["keys"], Tensor)
    ):
        raise ValueError("checkpoint NumPy RNG state is invalid")
    cpu_state = state["torch_cpu"]
    if not isinstance(cpu_state, Tensor) or cpu_state.dtype != torch.uint8:
        raise ValueError("checkpoint Torch CPU RNG state is invalid")
    cuda_count = state["cuda_device_count"]
    cuda_states = state["torch_cuda"]
    if (
        not isinstance(cuda_count, int)
        or isinstance(cuda_count, bool)
        or not isinstance(cuda_states, list)
    ):
        raise ValueError("checkpoint CUDA RNG metadata is invalid")
    if expected_device == "cpu":
        if cuda_count != 0 or cuda_states:
            raise ValueError("CPU checkpoint must not carry CUDA RNG state")
    else:
        if (
            not torch.cuda.is_available()
            or cuda_count != torch.cuda.device_count()
            or len(cuda_states) != cuda_count
            or any(
                not isinstance(item, Tensor) or item.dtype != torch.uint8
                for item in cuda_states
            )
        ):
            raise ValueError(
                "CUDA checkpoint does not match the current device topology"
            )


def _restore_rng(
    state: Mapping[str, object], expected_device: Literal["cpu", "cuda"]
) -> None:
    _validate_rng_state(state, expected_device)
    try:
        random.setstate(state["python"])  # type: ignore[arg-type]
        numpy_state = state["numpy"]
        assert isinstance(numpy_state, Mapping)
        keys = numpy_state["keys"]
        assert isinstance(keys, Tensor)
        np.random.set_state(
            (
                str(numpy_state["bit_generator"]),
                keys.cpu().numpy().astype(np.uint32, copy=True),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
        cpu_state = state["torch_cpu"]
        assert isinstance(cpu_state, Tensor)
        torch.set_rng_state(cpu_state.cpu())
        if expected_device == "cuda":
            cuda_states = state["torch_cuda"]
            assert isinstance(cuda_states, list)
            torch.cuda.set_rng_state_all([item.cpu() for item in cuda_states])
    except (AssertionError, TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("checkpoint RNG state could not be restored exactly") from exc


def _durable_replace(temporary: Path, target: Path) -> None:
    if os.name == "nt":
        import ctypes

        move_file = ctypes.WinDLL("kernel32", use_last_error=True).MoveFileExW
        move_file.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint)
        move_file.restype = ctypes.c_int
        replace_existing = 0x1
        write_through = 0x8
        if not move_file(str(temporary), str(target), replace_existing | write_through):
            error = ctypes.get_last_error()
            raise OSError(error, "durable checkpoint replace failed", str(target))
    else:
        os.replace(temporary, target)
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(target.parent, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def save_checkpoint(path: str | Path, payload: Mapping[str, object]) -> Path:
    target = Path(path)
    if target.suffix.casefold() not in {".pt", ".pth"}:
        raise ValueError("checkpoint path must end in .pt or .pth")
    if payload.get("checkpoint_self_hash") != _checkpoint_self_hash(payload):
        raise ValueError("checkpoint self hash is absent or invalid")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise ValueError("refusing to replace checkpoint symlink")
    handle = tempfile.NamedTemporaryFile(
        dir=target.parent, prefix=f".{target.name}.", delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        _durable_replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def load_checkpoint(
    path: str | Path,
    *,
    expected_config_hash: str,
    expected_dataset_identity: str,
    expected_split_manifest: SplitManifest,
    expected_encoder_identity: EncoderIdentity,
    expected_loss_profile: LossProfile,
    expected_device: Literal["cpu", "cuda"],
) -> dict[str, object]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("checkpoint must be a regular local file")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("checkpoint could not be safely deserialized") from exc
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a mapping")
    required = {
        "schema_version",
        "model",
        "optimizer",
        "scheduler",
        "step",
        "rng_state",
        "git_commit",
        "config_hash",
        "dataset_identity",
        "split_manifest",
        "encoder_identity",
        "loss_profile",
        "device",
        "canonical_config",
        "checkpoint_self_hash",
    }
    if set(payload) != required or payload["schema_version"] != 2:
        raise ValueError("checkpoint schema is invalid")
    self_hash = payload["checkpoint_self_hash"]
    if (
        not isinstance(self_hash, str)
        or len(self_hash) != 64
        or self_hash != _checkpoint_self_hash(payload)
    ):
        raise ValueError("checkpoint self hash does not match payload")
    canonical_config = payload["canonical_config"]
    if not isinstance(canonical_config, Mapping):
        raise ValueError("checkpoint canonical config is invalid")
    internal_hash = hashlib.sha256(
        _canonical_json(canonical_config).encode("utf-8")
    ).hexdigest()
    if internal_hash != payload["config_hash"]:
        raise ValueError("checkpoint canonical config hash does not match payload")
    if payload["config_hash"] != expected_config_hash:
        raise ValueError("checkpoint config identity does not match")
    if payload["dataset_identity"] != expected_dataset_identity:
        raise ValueError("checkpoint dataset identity does not match")
    if payload["split_manifest"] != expected_split_manifest.model_dump(mode="json"):
        raise ValueError("checkpoint split manifest does not match")
    expected_encoder = EncoderIdentity.model_validate(
        expected_encoder_identity.model_dump(mode="python")
    )
    if payload["encoder_identity"] != expected_encoder.model_dump(mode="json"):
        raise ValueError("checkpoint encoder identity does not match")
    expected_profile = LossProfile.model_validate(
        expected_loss_profile.model_dump(mode="python")
    )
    if payload["loss_profile"] != expected_profile.model_dump(mode="json"):
        raise ValueError("checkpoint loss profile does not match")
    if payload["device"] != expected_device:
        raise ValueError("checkpoint device does not match")
    if not isinstance(payload["git_commit"], str) or not payload["git_commit"]:
        raise ValueError("checkpoint git commit is invalid")
    if (
        not isinstance(payload["step"], int)
        or isinstance(payload["step"], bool)
        or payload["step"] < 0
    ):
        raise ValueError("checkpoint step is invalid")
    rng_state = payload["rng_state"]
    if not isinstance(rng_state, Mapping):
        raise ValueError("checkpoint RNG state is invalid")
    _validate_rng_state(rng_state, expected_device)
    return payload


def _optimizer_to_device(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, Tensor):
                state[key] = value.to(device)


def _checkpoint_payload(
    *,
    model: MemoryRouter,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    config: TrainConfig,
    config_hash: str,
    dataset: OracleBCDataset,
    split_manifest: SplitManifest,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 2,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "rng_state": _capture_rng(config.device),
        "git_commit": _git_commit(),
        "config_hash": config_hash,
        "dataset_identity": dataset.identity,
        "split_manifest": split_manifest.model_dump(mode="json"),
        "encoder_identity": model.config.encoder.model_dump(mode="json"),
        "loss_profile": config.loss_profile.model_dump(mode="json"),
        "device": config.device,
        "canonical_config": _config_payload(model, config),
    }
    payload["checkpoint_self_hash"] = _checkpoint_self_hash(payload)
    return payload


def _select_rows(batch: RouterBatch, indices: Tensor) -> RouterBatch:
    return RouterBatch(
        **{
            field: getattr(batch, field).index_select(0, indices)
            for field in batch.__dataclass_fields__
        }
    )


def _reset_trainable_module(module: nn.Module) -> None:
    if isinstance(module, nn.MultiheadAttention):
        module._reset_parameters()
        return
    reset = getattr(module, "reset_parameters", None)
    if callable(reset):
        reset()


def train_bc(
    model: MemoryRouter,
    dataset: OracleBCDataset,
    config: TrainConfig,
    *,
    resume: bool = False,
    tokenizer: TextTokenizer | None = None,
) -> TrainResult:
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if model.config.encoder.kind == "pretrained" and tokenizer is None:
        raise ValueError("pretrained Router training requires its pinned tokenizer")
    device = torch.device(config.device)
    checkpoint_path = config.validated_checkpoint_path()
    split_manifest = build_grouped_split(
        dataset.records,
        seed=config.split_seed,
        train_fraction=config.train_fraction,
        dev_fraction=config.dev_fraction,
    )
    if not split_manifest.train_record_ids:
        raise ValueError("grouped split produced no training rows")
    train_dataset = dataset.subset(split_manifest.train_record_ids)
    collator = RouterCollator(
        max_question_tokens=model.config.max_question_tokens,
        max_item_tokens=model.config.max_item_tokens,
        tokenizer=tokenizer,
    )
    full_batch = collator(train_dataset.records)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=config.scheduler_gamma
    )
    config_hash = canonical_config_hash(model, config)
    start_step = 0
    if resume:
        checkpoint = load_checkpoint(
            checkpoint_path,
            expected_config_hash=config_hash,
            expected_dataset_identity=dataset.identity,
            expected_split_manifest=split_manifest,
            expected_encoder_identity=model.config.encoder,
            expected_loss_profile=config.loss_profile,
            expected_device=config.device,
        )
        model.load_state_dict(checkpoint["model"], strict=True)  # type: ignore[arg-type]
        optimizer.load_state_dict(checkpoint["optimizer"])  # type: ignore[arg-type]
        scheduler.load_state_dict(checkpoint["scheduler"])  # type: ignore[arg-type]
        _optimizer_to_device(optimizer, device)
        start_step = int(checkpoint["step"])
        _restore_rng(checkpoint["rng_state"], config.device)  # type: ignore[arg-type]
    else:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if config.device == "cuda":
            torch.cuda.manual_seed_all(config.seed)
        if model.config.encoder.kind == "test":
            model.apply(_reset_trainable_module)
        else:
            for name, child in model.named_children():
                if name != "text_encoder":
                    child.apply(_reset_trainable_module)
    if start_step > config.max_steps:
        raise ValueError("checkpoint step exceeds requested max_steps")
    model.train()
    for step in range(start_step + 1, config.max_steps + 1):
        indices = torch.randint(0, len(train_dataset), (config.batch_size,))
        batch = _select_rows(full_batch, indices).to(device)
        optimizer.zero_grad(set_to_none=True)
        losses = behavior_cloning_loss(model(batch), batch, config.loss_profile)
        losses.total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()
        if step % config.checkpoint_every == 0 or step == config.max_steps:
            save_checkpoint(
                checkpoint_path,
                _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    config=config,
                    config_hash=config_hash,
                    dataset=dataset,
                    split_manifest=split_manifest,
                ),
            )
    model.eval()
    with torch.no_grad():
        evaluation = model(full_batch.to(device)).action_logits.argmax(dim=-1).cpu()
    accuracy = float((evaluation == full_batch.target_action_index).float().mean())
    return TrainResult(
        step=config.max_steps,
        action_accuracy=accuracy,
        checkpoint_path=checkpoint_path,
        config_hash=config_hash,
        split_manifest=split_manifest,
        seed=config.seed,
        dataset_identity=dataset.identity,
    )


def run_seed_sweep(
    *,
    model_factory: Callable[[], MemoryRouter | tuple[MemoryRouter, TextTokenizer]],
    dataset: OracleBCDataset,
    training: TrainConfig,
    seeds: tuple[int, int, int],
    resume: bool = False,
) -> tuple[TrainResult, TrainResult, TrainResult]:
    if (
        len(seeds) != 3
        or len(set(seeds)) != 3
        or any(isinstance(seed, bool) for seed in seeds)
    ):
        raise ValueError("seed sweep requires exactly three unique integer seeds")
    dataset_identity = dataset.identity
    results: list[TrainResult] = []
    for seed in seeds:
        built = model_factory()
        if isinstance(built, tuple):
            model, tokenizer = built
        else:
            model, tokenizer = built, None
        seed_config = training.model_copy(
            update={
                "seed": seed,
                "checkpoint_path": Path(f"seed-{seed}") / training.checkpoint_path.name,
            }
        )
        result = train_bc(
            model,
            dataset,
            seed_config,
            resume=resume,
            tokenizer=tokenizer,
        )
        if result.dataset_identity != dataset_identity:
            raise RuntimeError("seed sweep changed the cached dataset identity")
        results.append(result)
    return tuple(results)  # type: ignore[return-value]


class TrainFileConfig(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        protected_namespaces=(),
    )

    model_config_path: Path
    model_overrides: dict[str, object] = Field(default_factory=dict)
    training: TrainConfig
    dataset_path: Path | None = None
    smoke_dataset_size: int = Field(default=32, ge=2)
    seeds: tuple[int, int, int] = (13, 37, 73)

    @model_validator(mode="after")
    def seeds_are_unique(self) -> "TrainFileConfig":
        if len(set(self.seeds)) != 3 or any(
            isinstance(seed, bool) for seed in self.seeds
        ):
            raise ValueError("seed sweep requires three unique integer seeds")
        return self


def _smoke_dataset(size: int) -> OracleBCDataset:
    records = []
    normalization = CostNormalization(
        constant=10.0,
        sample_count=size,
        source_split="train",
    )
    for index in range(size):
        video_id = f"smoke-v{index:04d}"
        target_event = "event-a" if index % 2 else "event-b"
        candidates = ("event-a", "event-b")
        actions = tuple(
            ActionInstance(ActionType.EXPAND_RESIDUAL, event_id, None)
            for event_id in (candidates if index % 3 else tuple(reversed(candidates)))
        ) + (ActionInstance(ActionType.STOP, None, None),)
        target_action = ActionInstance(ActionType.EXPAND_RESIDUAL, target_event, None)
        provenance = OracleRecordProvenance(
            dataset_manifest_hash="a" * 64,
            source_manifest_hash="b" * 64,
            source_split="train",
            video_group_id=video_id,
            longroute_example_id=f"smoke-example-{index:04d}",
            normalization_manifest_hash="c" * 64,
            normalization=normalization,
            preference_set_hash="d" * 64,
            preference_values=COST_PREFERENCES,
            selected_preference=0.3,
            oracle_utility=1.0,
            optimal_action_tie_count=1,
        )
        records.append(
            OracleBCRecord(
                record_id=f"smoke-r{index:04d}",
                video_id=video_id,
                question_id=f"smoke-q{index:04d}",
                observation_snapshot_id="synthetic-cached-observation-v1",
                provenance=provenance,
                state=RouterState(
                    question="Which candidate matches the acquired evidence?",
                    options=("candidate A", "candidate B"),
                    evidence=(
                        EvidenceItem(
                            event_id=target_event,
                            fidelity_level=FidelityLevel.GIST,
                            content="the acquired evidence",
                            score=1.0,
                        ),
                    ),
                    action_history=(),
                    remaining_budget=10,
                    candidate_event_ids=candidates,
                    candidate_fidelity_levels={
                        event_id: FidelityLevel.GIST for event_id in candidates
                    },
                    context_frontiers={event_id: (0, 0) for event_id in candidates},
                    cost_preference=0.3,
                ),
                action_instances=actions,
                legal_action_mask=(True, True, True),
                target_action_index=actions.index(target_action),
                sufficiency_target=0,
                cost_to_go=1.0,
            )
        )
    return OracleBCDataset(records)


def load_train_file(path: str | Path) -> tuple[RouterModelConfig, TrainFileConfig]:
    source = Path(path)
    raw = OmegaConf.to_container(OmegaConf.load(source), resolve=True)
    file_config = TrainFileConfig.model_validate(raw)
    model_path = file_config.model_config_path
    if not model_path.is_absolute():
        model_path = (source.parent / model_path).resolve()
    model_raw = OmegaConf.to_container(OmegaConf.load(model_path), resolve=True)
    if not isinstance(model_raw, dict):
        raise ValueError("router model config must be a mapping")
    model_raw.update(file_config.model_overrides)
    return RouterModelConfig.model_validate(model_raw), file_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the memory router with behavior cloning"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args(argv)
    model_config, file_config = load_train_file(arguments.config)
    training = file_config.training
    if arguments.max_steps is not None:
        training = training.model_copy(update={"max_steps": arguments.max_steps})
    records = (
        load_oracle_records(file_config.dataset_path)
        if file_config.dataset_path is not None
        else _smoke_dataset(file_config.smoke_dataset_size).records
    )
    dataset = OracleBCDataset(records)

    def model_factory() -> MemoryRouter | tuple[MemoryRouter, TextTokenizer]:
        if model_config.encoder.kind == "pretrained":
            encoder, raw_tokenizer = ProductionEncoderFactory.load(model_config.encoder)
            tokenizer = HFTokenizerAdapter(model_config.encoder, raw_tokenizer)
            return MemoryRouter(model_config, text_encoder=encoder), tokenizer
        return MemoryRouter(model_config)

    results = run_seed_sweep(
        model_factory=model_factory,
        dataset=dataset,
        training=training,
        seeds=file_config.seeds,
        resume=arguments.resume,
    )
    print(
        _canonical_json(
            {
                "runs": [
                    {
                        "seed": result.seed,
                        "step": result.step,
                        "action_accuracy": result.action_accuracy,
                        "checkpoint_path": str(result.checkpoint_path),
                        "config_hash": result.config_hash,
                        "dataset_hash": result.split_manifest.dataset_hash,
                    }
                    for result in results
                ]
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
