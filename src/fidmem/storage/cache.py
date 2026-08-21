"""Content-addressed JSON cache with atomic replacement writes."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


class CacheKey:
    @staticmethod
    def build(video_hash: str, time_range: tuple[float, float], model: str, prompt: str, payload: Any) -> str:
        serialized = json.dumps({"video_hash": video_hash, "time_range": time_range, "model": model, "prompt": prompt, "payload": payload}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class ContentAddressedCache:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def key(self, video_hash: str, time_range: tuple[float, float], model: str, prompt: str, payload: Any) -> str:
        return CacheKey.build(video_hash, time_range, model, prompt, payload)

    def get(self, key: str) -> Any | None:
        path = self.root / f"{key}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def put(self, key: str, payload: Any) -> None:
        destination = self.root / f"{key}.json"
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.root, prefix=f".{key}.", suffix=".tmp", delete=False) as file:
                temporary_name = file.name
                file.write(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_name, destination)
        except BaseException:
            if temporary_name:
                Path(temporary_name).unlink(missing_ok=True)
            raise
