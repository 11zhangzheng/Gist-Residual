from __future__ import annotations

import pytest
from pydantic import ValidationError

from fidmem.production.authority import (
    ProductionAuthorityDraft,
    PromptIdentity,
    canonical_json_bytes,
    canonical_sha256,
)


def test_draft_can_be_incomplete_but_never_production_ready() -> None:
    draft = ProductionAuthorityDraft()

    assert draft.lifecycle == "draft"
    assert draft.production_ready is False


def test_prompt_content_is_bound_to_its_hash() -> None:
    prompt = PromptIdentity(
        name="residual",
        version="1",
        content="Extract novel event details.",
        sha256="0" * 64,
    )

    with pytest.raises(ValueError, match="prompt sha256"):
        prompt.verify_content_hash()


def test_prompt_with_hand_checked_hash_is_valid() -> None:
    prompt = PromptIdentity(
        name="residual",
        version="1",
        content="Extract novel event details.",
        sha256="7ac60c4931d5fadf13d0d61d1f03e0c97d18927416c77da2032c0c52cdf21208",
    )

    assert prompt.verify_content_hash() is prompt
    assert canonical_sha256({"content": prompt.content}) == (
        "d5b46c5d1f2cddc55d360dda2aa440c9b641ad8c3dc070be3c8b24c1d8b9e19e"
    )


def test_malformed_sha_is_rejected() -> None:
    with pytest.raises(ValidationError):
        PromptIdentity(name="p", version="1", content="x", sha256="bad")


def test_canonical_json_bytes_has_one_strict_utf8_representation() -> None:
    first = {"unicode": "汉字", "nested": {"z": [1, True, None], "a": 2}}
    second = {"nested": {"a": 2, "z": [1, True, None]}, "unicode": "汉字"}
    expected = (
        b'{"nested":{"a":2,"z":[1,true,null]},' b'"unicode":"\xe6\xb1\x89\xe5\xad\x97"}'
    )

    assert canonical_json_bytes(first) == expected
    assert canonical_json_bytes(second) == expected
    assert not expected.startswith(b"\xef\xbb\xbf")
    assert not expected.endswith(b"\n")


def test_canonical_json_rejects_nan() -> None:
    with pytest.raises(ValueError, match="JSON"):
        canonical_json_bytes({"value": float("nan")})
