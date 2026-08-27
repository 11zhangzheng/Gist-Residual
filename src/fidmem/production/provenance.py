"""Authority-bound production provenance and cache gates."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fidmem.production.authority import canonical_sha256
from fidmem.storage.cache import ContentAddressedCache

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
EvidenceClass = Literal["engineering", "production"]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ProductionContext(_FrozenModel):
    """Classify an artifact without allowing ambiguous Authority provenance."""

    evidence_class: EvidenceClass
    authority_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def evidence_and_authority_are_consistent(self) -> Self:
        if self.evidence_class == "production" and self.authority_sha256 is None:
            raise ValueError("production evidence requires authority_sha256")
        if self.evidence_class == "engineering" and self.authority_sha256 is not None:
            raise ValueError("engineering evidence must not carry authority_sha256")
        return self


class AuthorityBoundCacheEnvelope(_FrozenModel):
    schema_version: Literal[1] = 1
    evidence_class: EvidenceClass
    authority_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    payload: Any
    payload_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def envelope_is_consistent(self) -> Self:
        ProductionContext(
            evidence_class=self.evidence_class,
            authority_sha256=self.authority_sha256,
        )
        if canonical_sha256(self.payload) != self.payload_sha256:
            raise ValueError("cache payload hash mismatch")
        return self


def _validated_run_id(run_id: str) -> str:
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be one non-empty path component")
    return run_id


def production_run_root(
    artifact_root: str | Path, authority_sha256: str, run_id: str
) -> Path:
    context = ProductionContext(
        evidence_class="production", authority_sha256=authority_sha256
    )
    return (
        Path(artifact_root)
        / "production"
        / context.authority_sha256
        / "runs"
        / _validated_run_id(run_id)
    )


def engineering_run_root(artifact_root: str | Path, run_id: str) -> Path:
    return Path(artifact_root) / "development" / "runs" / _validated_run_id(run_id)


def require_single_authority(
    expected_authority_sha256: str, observed_authority_sha256s: list[str | None]
) -> None:
    ProductionContext(
        evidence_class="production", authority_sha256=expected_authority_sha256
    )
    if any(value is None for value in observed_authority_sha256s):
        raise ValueError("production artifact has missing authority_sha256")
    if any(value != expected_authority_sha256 for value in observed_authority_sha256s):
        raise ValueError("production artifact authority mismatch")


class AuthorityBoundCache:
    """Store cache values in self-verifying, evidence-classified envelopes."""

    def __init__(self, cache: ContentAddressedCache) -> None:
        self.cache = cache

    def put_bound(
        self,
        key: str,
        payload: Any,
        *,
        evidence_class: EvidenceClass = "production",
        authority_sha256: str | None = None,
    ) -> None:
        envelope = AuthorityBoundCacheEnvelope(
            evidence_class=evidence_class,
            authority_sha256=authority_sha256,
            payload=payload,
            payload_sha256=canonical_sha256(payload),
        )
        existing = self.cache.get(key)
        if existing is not None:
            current = AuthorityBoundCacheEnvelope.model_validate(existing)
            if (
                current.evidence_class != envelope.evidence_class
                or current.authority_sha256 != envelope.authority_sha256
            ):
                raise ValueError(
                    "refusing to overwrite cache from a different Authority"
                )
        self.cache.put(key, envelope.model_dump(mode="json"))

    def get_bound(
        self,
        key: str,
        *,
        evidence_class: EvidenceClass = "production",
        expected_authority_sha256: str | None = None,
    ) -> Any | None:
        expected = ProductionContext(
            evidence_class=evidence_class,
            authority_sha256=expected_authority_sha256,
        )
        raw = self.cache.get(key)
        if raw is None:
            return None
        envelope = AuthorityBoundCacheEnvelope.model_validate(raw)
        if (
            envelope.evidence_class != expected.evidence_class
            or envelope.authority_sha256 != expected.authority_sha256
        ):
            raise ValueError("cache authority mismatch")
        return envelope.payload
