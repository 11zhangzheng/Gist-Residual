"""Behavior-cloning training and exact, CPU-compatible checkpoint resume."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, NamedTuple

import numpy as np
import torch
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor, nn

from fidmem.types import ActionInstance, ActionType, RouterState

from .dataset import (
    OracleBCDataset,
    OracleBCRecord,
    RouterBatch,
    RouterCollator,
    SplitManifest,
    build_grouped_split,
    load_oracle_records,
)
from .model import MemoryRouter, RouterModelConfig, RouterOutput


class LossWeights(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    action: float = Field(default=1.0, gt=0)
    sufficiency: float = Field(default=0.3, ge=0)
    cost_to_go: float = Field(default=0.1, ge=0)


class TrainConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    seed: int = 13
    max_steps: int = Field(default=20, ge=1)
    batch_size: int = Field(default=16, ge=1)
    learning_rate: float = Field(default=3e-4, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    scheduler_gamma: float = Field(default=1.0, gt=0)
    checkpoint_path: Path
    checkpoint_every: int = Field(default=20, ge=1)
    device: Literal["cpu", "cuda"] = "cpu"
    split_seed: int = 2026
    train_fraction: float = Field(default=1.0, ge=0, le=1)
    dev_fraction: float = Field(default=0.0, ge=0, le=1)
    loss_weights: LossWeights = Field(default_factory=LossWeights)

    @model_validator(mode="after")
    def split_is_valid(self) -> "TrainConfig":
        if self.train_fraction <= 0:
            raise ValueError("train_fraction must be positive")
        if self.train_fraction + self.dev_fraction > 1:
            raise ValueError("train and dev fractions must not exceed one")
        return self


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


def behavior_cloning_loss(
    output: RouterOutput,
    targets: RouterBatch | Mapping[str, Tensor],
    weights: LossWeights,
) -> LossBreakdown:
    if isinstance(targets, RouterBatch):
        action_target = targets.target_action_index
        sufficiency_target = targets.sufficiency_target
        cost_target = targets.cost_to_go_target
    else:
        action_target = targets["target_action_index"]
        sufficiency_target = targets["sufficiency_target"]
        cost_target = targets["cost_to_go_target"]
    action = torch.nn.functional.cross_entropy(output.action_logits, action_target)
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
    return LossBreakdown(total, action, sufficiency, cost)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _config_payload(model: MemoryRouter, config: TrainConfig) -> dict[str, object]:
    training = config.model_dump(mode="json")
    # These run-control fields may change when a stopped run is continued on
    # CPU or written to another local path; none changes an optimizer update.
    for field in ("max_steps", "checkpoint_path", "checkpoint_every", "device"):
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


def _capture_rng() -> dict[str, object]:
    numpy_state = np.random.get_state()
    return {
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
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }


def _restore_rng(state: Mapping[str, object]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(state) != required:
        raise ValueError("checkpoint RNG state is incomplete")
    random.setstate(state["python"])  # type: ignore[arg-type]
    numpy_state = state["numpy"]
    if not isinstance(numpy_state, Mapping) or not isinstance(
        numpy_state.get("keys"), Tensor
    ):
        raise ValueError("checkpoint NumPy RNG state is invalid")
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            numpy_state["keys"].cpu().numpy().astype(np.uint32, copy=True),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    cpu_state = state["torch_cpu"]
    if not isinstance(cpu_state, Tensor):
        raise ValueError("checkpoint Torch CPU RNG state is invalid")
    torch.set_rng_state(cpu_state.cpu())
    cuda_states = state["torch_cuda"]
    if torch.cuda.is_available() and isinstance(cuda_states, list) and cuda_states:
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_states])


def save_checkpoint(path: str | Path, payload: Mapping[str, object]) -> Path:
    target = Path(path)
    if target.suffix.casefold() not in {".pt", ".pth"}:
        raise ValueError("checkpoint path must end in .pt or .pth")
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
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def load_checkpoint(
    path: str | Path,
    *,
    expected_config_hash: str,
    expected_dataset_identity: str,
    expected_split_manifest: SplitManifest,
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
        "loss_weights",
        "canonical_config",
    }
    if set(payload) != required or payload["schema_version"] != 1:
        raise ValueError("checkpoint schema is invalid")
    if payload["config_hash"] != expected_config_hash:
        raise ValueError("checkpoint config identity does not match")
    if payload["dataset_identity"] != expected_dataset_identity:
        raise ValueError("checkpoint dataset identity does not match")
    if payload["split_manifest"] != expected_split_manifest.model_dump(mode="json"):
        raise ValueError("checkpoint split manifest does not match")
    if not isinstance(payload["step"], int) or payload["step"] < 0:
        raise ValueError("checkpoint step is invalid")
    if not isinstance(payload["rng_state"], Mapping):
        raise ValueError("checkpoint RNG state is invalid")
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
    return {
        "schema_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "rng_state": _capture_rng(),
        "git_commit": _git_commit(),
        "config_hash": config_hash,
        "dataset_identity": dataset.identity,
        "split_manifest": split_manifest.model_dump(mode="json"),
        "loss_weights": config.loss_weights.model_dump(mode="json"),
        "canonical_config": _config_payload(model, config),
    }


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
) -> TrainResult:
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(config.device)
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
        max_question_bytes=model.config.max_question_bytes,
        max_item_bytes=model.config.max_item_bytes,
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
            config.checkpoint_path,
            expected_config_hash=config_hash,
            expected_dataset_identity=dataset.identity,
            expected_split_manifest=split_manifest,
        )
        model.load_state_dict(checkpoint["model"], strict=True)  # type: ignore[arg-type]
        optimizer.load_state_dict(checkpoint["optimizer"])  # type: ignore[arg-type]
        scheduler.load_state_dict(checkpoint["scheduler"])  # type: ignore[arg-type]
        _optimizer_to_device(optimizer, device)
        start_step = int(checkpoint["step"])
        _restore_rng(checkpoint["rng_state"])  # type: ignore[arg-type]
    else:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        model.apply(_reset_trainable_module)
    if start_step > config.max_steps:
        raise ValueError("checkpoint step exceeds requested max_steps")
    model.train()
    for step in range(start_step + 1, config.max_steps + 1):
        indices = torch.randint(0, len(train_dataset), (config.batch_size,))
        batch = _select_rows(full_batch, indices).to(device)
        optimizer.zero_grad(set_to_none=True)
        losses = behavior_cloning_loss(model(batch), batch, config.loss_weights)
        if not torch.isfinite(losses.total):
            raise FloatingPointError("behavior cloning loss became non-finite")
        losses.total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()
        if step % config.checkpoint_every == 0 or step == config.max_steps:
            save_checkpoint(
                config.checkpoint_path,
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
        checkpoint_path=config.checkpoint_path,
        config_hash=config_hash,
        split_manifest=split_manifest,
    )


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


def _smoke_dataset(size: int) -> OracleBCDataset:
    records = []
    for index in range(size):
        target = index % 2
        records.append(
            OracleBCRecord(
                record_id=f"smoke-r{index:04d}",
                video_id=f"smoke-v{index:04d}",
                question_id=f"smoke-q{index:04d}",
                observation_snapshot_id="synthetic-cached-observation-v1",
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
    result = train_bc(
        MemoryRouter(model_config),
        OracleBCDataset(records),
        training,
        resume=arguments.resume,
    )
    print(
        _canonical_json(
            {
                "step": result.step,
                "action_accuracy": result.action_accuracy,
                "checkpoint_path": str(result.checkpoint_path),
                "config_hash": result.config_hash,
                "dataset_hash": result.split_manifest.dataset_hash,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
