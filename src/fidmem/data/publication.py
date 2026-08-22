"""Atomic publication primitives for immutable LongRoute generations."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Callable, TypeVar


T = TypeVar("T")
_GENERATION_ID = re.compile(r"^[0-9a-f]{20}$")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
        raise


@dataclass
class PublicationTransaction:
    """One private staging tree and its immutable public destination."""

    generation_id: str
    generation_uri: str
    staging_dir: Path
    final_dir: Path
    published: bool = False


class PublicationBackend:
    """Local atomic backend used by LongRouteBuilder in production.

    Subclasses may wrap _run_operation to add metrics, tracing, or
    deterministic fault injection without changing publication semantics.
    """

    GENERATION_WRITE = "generation_write"
    DIRECTORY_PUBLISH = "directory_publish"
    POINTER_WRITE = "pointer_write"

    def __init__(self, output_root: str | Path) -> None:
        self.output_root = Path(output_root).resolve()

    def _run_operation(self, name: str, action: Callable[[], T]) -> T:
        del name
        return action()

    @staticmethod
    def _resolve_under(base: Path, relative: str | Path) -> Path:
        candidate = Path(relative)
        if candidate.is_absolute():
            raise ValueError("publication paths must be relative to their root")
        resolved = (base / candidate).resolve()
        try:
            resolved.relative_to(base.resolve())
        except ValueError as error:
            raise ValueError("publication path escapes its configured root") from error
        return resolved

    def begin_generation(self, generation_id: str) -> PublicationTransaction:
        if _GENERATION_ID.fullmatch(generation_id) is None:
            raise ValueError("generation_id must be exactly twenty lowercase hex characters")
        self.output_root.mkdir(parents=True, exist_ok=True)
        staging_root = self._resolve_under(self.output_root, ".staging")
        staging_root.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f"{generation_id}-", dir=staging_root)
        ).resolve()
        final = self._resolve_under(
            self.output_root, Path("generations") / generation_id
        )
        if final.exists():
            shutil.rmtree(staging)
            raise FileExistsError(f"immutable generation already exists: {generation_id}")
        return PublicationTransaction(
            generation_id=generation_id,
            generation_uri=f"generations/{generation_id}",
            staging_dir=staging,
            final_dir=final,
        )

    def write_generation_bytes(
        self,
        transaction: PublicationTransaction,
        relative: str | Path,
        payload: bytes,
    ) -> Path:
        target = self._resolve_under(transaction.staging_dir, relative)

        def write() -> Path:
            _atomic_write_bytes(target, payload)
            return target

        return self._run_operation(self.GENERATION_WRITE, write)

    def write_generation_text(
        self,
        transaction: PublicationTransaction,
        relative: str | Path,
        payload: str,
    ) -> Path:
        return self.write_generation_bytes(
            transaction, relative, payload.encode("utf-8")
        )

    def publish_generation(self, transaction: PublicationTransaction) -> None:
        transaction.final_dir.parent.mkdir(parents=True, exist_ok=True)

        def publish() -> None:
            os.replace(transaction.staging_dir, transaction.final_dir)

        self._run_operation(self.DIRECTORY_PUBLISH, publish)
        transaction.published = True

    def publish_current(self, payload: str) -> None:
        pointer = self._resolve_under(self.output_root, "current-generation.json")
        self._run_operation(
            self.POINTER_WRITE,
            lambda: _atomic_write_bytes(pointer, payload.encode("utf-8")),
        )

    def write_root_text(self, relative: str | Path, payload: str) -> Path:
        target = self._resolve_under(self.output_root, relative)
        _atomic_write_bytes(target, payload.encode("utf-8"))
        return target

    def abort(self, transaction: PublicationTransaction | None) -> None:
        if transaction is None:
            return
        if transaction.staging_dir.is_dir():
            shutil.rmtree(transaction.staging_dir)
        if transaction.published and transaction.final_dir.is_dir():
            shutil.rmtree(transaction.final_dir)
            transaction.published = False