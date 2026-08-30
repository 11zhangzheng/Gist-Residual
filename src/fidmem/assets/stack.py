"""Versioned, provider-neutral Experiment Stack configuration."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal, Self

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_FORBIDDEN_REVISIONS = {"main", "master", "latest"}


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PhysicalAsset(_FrozenModel):
    repo_id: str = Field(min_length=1)
    repo_type: Literal["model", "dataset"]
    immutable_revision: str | None = None
    backend: str = Field(min_length=1)
    dtype: str | None = None
    include_files: tuple[str, ...] = ()

    @model_validator(mode="after")
    def revision_is_immutable(self) -> Self:
        if self.immutable_revision is None:
            return self
        revision = self.immutable_revision.strip().lower()
        if revision in _FORBIDDEN_REVISIONS or not _COMMIT_RE.fullmatch(revision):
            raise ValueError("immutable_revision must be a full lowercase commit SHA")
        return self

    @field_validator("include_files")
    @classmethod
    def include_files_are_safe(cls, include_files: tuple[str, ...]) -> tuple[str, ...]:
        include_files = tuple(sorted(set(include_files)))
        for name in include_files:
            posix_path = PurePosixPath(name)
            windows_path = PureWindowsPath(name)
            if (
                not name.strip()
                or "\\" in name
                or posix_path.is_absolute()
                or bool(windows_path.drive)
                or bool(windows_path.root)
                or any(part in {"", ".", ".."} for part in name.split("/"))
            ):
                raise ValueError("include_files must contain safe relative file names")
        return include_files


class BenchmarkIdentity(_FrozenModel):
    repo_id: str = Field(min_length=1)
    status: Literal["DEFERRED", "CANDIDATE_ASSETS_UNVERIFIED"]


class ExperimentStack(_FrozenModel):
    schema_version: Literal[1] = 1
    stack_id: Literal["gist-residual-v1"]
    status: Literal["CANDIDATE_ASSETS_UNVERIFIED"]
    backend: str = Field(min_length=1)
    dtype: Literal["bfloat16"]
    physical_assets: dict[str, PhysicalAsset] = Field(min_length=1)
    logical_roles: dict[str, str] = Field(min_length=1)
    target_benchmarks: dict[str, BenchmarkIdentity] = Field(default_factory=dict)

    @model_validator(mode="after")
    def role_mappings_are_complete_and_deduplicated(self) -> Self:
        expected = {
            "source_dataset",
            "gist_text_encoder",
            "gist_visual_encoder",
            "embedding_model",
            "residual_model",
            "visual_model",
            "answerer",
        }
        if set(self.logical_roles) != expected:
            raise ValueError(
                "logical_roles must exactly match the approved Experiment Stack v1"
            )
        unknown = sorted(set(self.logical_roles.values()) - set(self.physical_assets))
        if unknown:
            raise ValueError(
                f"logical roles reference unknown physical assets: {unknown}"
            )
        if (
            self.logical_roles["gist_text_encoder"]
            != self.logical_roles["embedding_model"]
        ):
            raise ValueError(
                "Gist text and embedding roles must share one physical snapshot"
            )
        if self.logical_roles["residual_model"] != self.logical_roles["visual_model"]:
            raise ValueError(
                "Residual and Visual roles must share one physical snapshot"
            )
        return self


def load_experiment_stack(path: str | Path) -> ExperimentStack:
    raw = OmegaConf.to_container(OmegaConf.load(Path(path)), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("Experiment Stack config must be a mapping")
    return ExperimentStack.model_validate(raw)
