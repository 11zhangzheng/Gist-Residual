"""Validated application configuration loaded from OmegaConf YAML files."""

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, model_validator


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    top_k: int = Field(ge=1)


class OracleConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_depth: int = Field(ge=1)
    beam_size: int = Field(ge=1)


class VisualConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    low_frames: int = Field(ge=1)
    high_frames: int = Field(ge=1)

    @model_validator(mode="after")
    def high_frames_must_cover_low_frames(self) -> "VisualConfig":
        if self.high_frames < self.low_frames:
            raise ValueError("high_frames must be at least low_frames")
        return self


class BudgetConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    a800_gpu_hours: float = Field(ge=0)
    v100_gpu_hours: float = Field(ge=0)


class AppConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    retrieval: RetrievalConfig
    oracle: OracleConfig
    visual: VisualConfig
    budget: BudgetConfig


def load_config(path: str | Path) -> AppConfig:
    """Load an OmegaConf YAML file into the immutable application contract."""
    raw_config = OmegaConf.load(Path(path))
    raw_data: Any = OmegaConf.to_container(raw_config, resolve=True)
    return AppConfig.model_validate(raw_data)
