from pathlib import Path

import pytest

from fidmem.production.provenance import (
    AuthorityBoundCache,
    ProductionContext,
    engineering_run_root,
    production_run_root,
    require_single_authority,
)
from fidmem.storage.cache import ContentAddressedCache


AUTHORITY_A = "a" * 64
AUTHORITY_B = "b" * 64


def test_production_and_development_namespaces_are_disjoint(tmp_path: Path) -> None:
    production = production_run_root(tmp_path, AUTHORITY_A, "r001")
    engineering = engineering_run_root(tmp_path, "r001")

    assert production == tmp_path / "production" / AUTHORITY_A / "runs" / "r001"
    assert engineering == tmp_path / "development" / "runs" / "r001"
    assert production != engineering


def test_production_context_requires_authority_and_engineering_rejects_it() -> None:
    assert (
        ProductionContext(
            evidence_class="production", authority_sha256=AUTHORITY_A
        ).authority_sha256
        == AUTHORITY_A
    )

    with pytest.raises(ValueError, match="requires authority_sha256"):
        ProductionContext(evidence_class="production")
    with pytest.raises(ValueError, match="must not carry authority_sha256"):
        ProductionContext(evidence_class="engineering", authority_sha256=AUTHORITY_A)


def test_authority_bound_cache_rejects_cross_authority_reuse(tmp_path: Path) -> None:
    cache = AuthorityBoundCache(ContentAddressedCache(tmp_path / "cache"))
    cache.put_bound("observation", {"text": "real"}, authority_sha256=AUTHORITY_A)

    assert cache.get_bound("observation", expected_authority_sha256=AUTHORITY_A) == {
        "text": "real"
    }
    with pytest.raises(ValueError, match="authority mismatch"):
        cache.get_bound("observation", expected_authority_sha256=AUTHORITY_B)


def test_authority_bound_cache_detects_payload_tampering(tmp_path: Path) -> None:
    raw_cache = ContentAddressedCache(tmp_path / "cache")
    cache = AuthorityBoundCache(raw_cache)
    cache.put_bound("observation", {"text": "real"}, authority_sha256=AUTHORITY_A)
    envelope = raw_cache.get("observation")
    envelope["payload"]["text"] = "tampered"
    raw_cache.put("observation", envelope)

    with pytest.raises(ValueError, match="payload hash mismatch"):
        cache.get_bound("observation", expected_authority_sha256=AUTHORITY_A)


def test_single_authority_gate_rejects_missing_or_mixed_hashes() -> None:
    assert require_single_authority(AUTHORITY_A, [AUTHORITY_A, AUTHORITY_A]) is None
    with pytest.raises(ValueError, match="missing authority"):
        require_single_authority(AUTHORITY_A, [AUTHORITY_A, None])
    with pytest.raises(ValueError, match="authority mismatch"):
        require_single_authority(AUTHORITY_A, [AUTHORITY_A, AUTHORITY_B])
