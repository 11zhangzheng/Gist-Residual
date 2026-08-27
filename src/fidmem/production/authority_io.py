"""YAML/JSON loading for deliberately unsealed Authority drafts."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from fidmem.production.authority import ProductionAuthorityDraft


def load_authority_draft(path: str | Path) -> ProductionAuthorityDraft:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"Authority draft does not exist: {source}")
    payload = OmegaConf.to_container(OmegaConf.load(source), resolve=True)
    if not isinstance(payload, dict):
        raise ValueError("Authority draft must be a mapping")
    return ProductionAuthorityDraft.model_validate(payload)
