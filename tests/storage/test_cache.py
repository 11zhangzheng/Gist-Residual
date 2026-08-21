from pathlib import Path

from fidmem.storage.cache import CacheKey, ContentAddressedCache


def test_prompt_change_invalidates_cache(tmp_path: Path) -> None:
    cache = ContentAddressedCache(tmp_path)

    a = cache.key("vhash", (0.0, 30.0), "model-v1", "prompt-a", {"frames": 12})
    b = cache.key("vhash", (0.0, 30.0), "model-v1", "prompt-b", {"frames": 12})

    assert a != b
    cache.put(a, {"value": 3})
    assert cache.get(a) == {"value": 3}


def test_cache_key_is_stable_for_equivalent_mapping_order() -> None:
    first = CacheKey.build(
        "vhash", (0.0, 30.0), "model-v1", "prompt-a", {"frames": 12, "stride": 2}
    )
    second = CacheKey.build(
        "vhash", (0.0, 30.0), "model-v1", "prompt-a", {"stride": 2, "frames": 12}
    )

    assert first == second
