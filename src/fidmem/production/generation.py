"""Immutable production generations published through one atomic pointer."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path

from fidmem.production.authority import canonical_json_bytes, canonical_sha256


FailureHook = Callable[[str], None]


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


class GenerationStore:
    """Publish complete immutable artifact sets and atomically switch CURRENT."""

    def __init__(self, run_root: str | Path, authority_sha256: str) -> None:
        self.run_root = Path(run_root)
        self.authority_sha256 = authority_sha256
        self.generations_root = self.run_root / "generations"
        self.pointer_path = self.run_root / "CURRENT.json"

    def _marker(self, artifacts: Mapping[str, bytes]) -> dict[str, object]:
        return {
            "schema_version": 1,
            "evidence_class": "production",
            "authority_sha256": self.authority_sha256,
            "artifact_sha256": {
                name: _sha256_bytes(content)
                for name, content in sorted(artifacts.items())
            },
        }

    def _generation_id(self, marker: Mapping[str, object]) -> str:
        return canonical_sha256(marker)

    def _validate_generation(self, path: Path) -> dict[str, object]:
        marker_path = path / "COMMITTED.json"
        if not marker_path.is_file():
            raise ValueError("production generation lacks COMMITTED.json")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("authority_sha256") != self.authority_sha256:
            raise ValueError("production generation Authority mismatch")
        hashes = marker.get("artifact_sha256")
        if not isinstance(hashes, dict):
            raise ValueError("production generation artifact hashes are invalid")
        for name, expected in hashes.items():
            artifact = path / str(name)
            if (
                not artifact.is_file()
                or _sha256_bytes(artifact.read_bytes()) != expected
            ):
                raise ValueError("production generation artifact hash mismatch")
        return marker

    def current_path(self) -> Path:
        if not self.pointer_path.is_file():
            raise ValueError("production run has no CURRENT generation")
        pointer = json.loads(self.pointer_path.read_text(encoding="utf-8"))
        if pointer.get("authority_sha256") != self.authority_sha256:
            raise ValueError("existing run uses a different Authority")
        generation_id = pointer.get("generation_id")
        if not isinstance(generation_id, str) or not generation_id:
            raise ValueError("CURRENT pointer generation id is invalid")
        generation_sha256 = pointer.get("generation_sha256")
        if not isinstance(generation_sha256, str) or len(generation_sha256) != 64:
            raise ValueError("CURRENT pointer generation SHA-256 is invalid")
        generation = self.generations_root / generation_id
        marker = self._validate_generation(generation)
        if (
            self._generation_id(marker) != generation_sha256
            or generation_sha256[:24] != generation_id
        ):
            raise ValueError("CURRENT generation identity mismatch")
        return generation

    def publish(
        self,
        artifacts: Mapping[str, bytes],
        *,
        failure_hook: FailureHook | None = None,
    ) -> Path:
        if not artifacts:
            raise ValueError("production generation must contain artifacts")
        marker = self._marker(artifacts)
        generation_sha256 = self._generation_id(marker)
        generation_id = generation_sha256[:24]
        destination = self.generations_root / generation_id
        self.generations_root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=self.generations_root))
        try:
            for name, content in sorted(artifacts.items()):
                if Path(name).name != name:
                    raise ValueError(
                        "generation artifact name must be one path component"
                    )
                path = staging / name
                with path.open("wb") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                if failure_hook is not None:
                    failure_hook(name)
            marker_bytes = canonical_json_bytes(marker)
            with (staging / "COMMITTED.json").open("wb") as stream:
                stream.write(marker_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            self._validate_generation(staging)
            if destination.exists():
                self._validate_generation(destination)
            else:
                os.replace(staging, destination)

            pointer = {
                "schema_version": 1,
                "evidence_class": "production",
                "authority_sha256": self.authority_sha256,
                "generation_id": generation_id,
                "generation_sha256": generation_sha256,
            }
            handle, temporary_name = tempfile.mkstemp(
                prefix=".CURRENT.", suffix=".tmp", dir=self.run_root
            )
            try:
                with os.fdopen(handle, "wb") as stream:
                    stream.write(canonical_json_bytes(pointer))
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary_name, self.pointer_path)
            finally:
                Path(temporary_name).unlink(missing_ok=True)
            return destination
        finally:
            if staging.exists():
                shutil.rmtree(staging)
