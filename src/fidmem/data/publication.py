"""Atomic publication primitives for immutable LongRoute generations."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import shutil
import secrets
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


@dataclass(frozen=True)
class PublicationTransaction:
    """Opaque handle for one backend-owned publication transaction."""

    transaction_id: str
    generation_id: str
    generation_uri: str
    staging_dir: Path
    final_dir: Path
    _capability: object = field(repr=False, compare=False)


@dataclass
class _TransactionRecord:
    transaction: PublicationTransaction
    transaction_id: str
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
    ROOT_WRITE = "root_write"

    def __init__(self, output_root: str | Path) -> None:
        self.output_root = Path(output_root).resolve()
        self._capability = object()
        self._transactions: dict[str, _TransactionRecord] = {}

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

    def _validated_transaction(
        self, transaction: PublicationTransaction
    ) -> _TransactionRecord:
        if type(transaction) is not PublicationTransaction:
            raise ValueError("untrusted publication transaction")
        try:
            transaction_id = transaction.transaction_id
            record = self._transactions.get(transaction_id)
            staging = Path(transaction.staging_dir).resolve()
            final = Path(transaction.final_dir).resolve()
            capability_matches = transaction._capability is self._capability
        except (AttributeError, OSError, TypeError) as error:
            raise ValueError(
                "untrusted or mutated publication transaction"
            ) from error
        if (
            record is None
            or record.transaction is not transaction
            or not capability_matches
            or record.transaction_id != transaction_id
            or transaction.generation_id != record.generation_id
            or transaction.generation_uri != record.generation_uri
            or staging != record.staging_dir
            or final != record.final_dir
        ):
            raise ValueError("untrusted or mutated publication transaction")

        staging_root = self._resolve_under(self.output_root, ".staging")
        expected_final = self._resolve_under(
            self.output_root, Path("generations") / record.generation_id
        )
        if (
            record.staging_dir.resolve() != record.staging_dir
            or record.staging_dir.parent != staging_root
            or record.final_dir != expected_final
            or record.generation_uri != f"generations/{record.generation_id}"
        ):
            raise ValueError("publication transaction registry is invalid")
        return record

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
        transaction_id = secrets.token_hex(16)
        while transaction_id in self._transactions:
            transaction_id = secrets.token_hex(16)
        generation_uri = f"generations/{generation_id}"
        transaction = PublicationTransaction(
            transaction_id=transaction_id,
            generation_id=generation_id,
            generation_uri=generation_uri,
            staging_dir=staging,
            final_dir=final,
            _capability=self._capability,
        )
        self._transactions[transaction_id] = _TransactionRecord(
            transaction=transaction,
            transaction_id=transaction_id,
            generation_id=generation_id,
            generation_uri=generation_uri,
            staging_dir=staging,
            final_dir=final,
        )
        return transaction

    def write_generation_bytes(
        self,
        transaction: PublicationTransaction,
        relative: str | Path,
        payload: bytes,
    ) -> Path:
        record = self._validated_transaction(transaction)
        if record.published or not record.staging_dir.is_dir():
            raise ValueError("publication transaction is not writable")
        target = self._resolve_under(record.staging_dir, relative)

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
        record = self._validated_transaction(transaction)
        if (
            record.published
            or not record.staging_dir.is_dir()
            or record.final_dir.exists()
        ):
            raise ValueError("publication transaction cannot be published")
        record.final_dir.parent.mkdir(parents=True, exist_ok=True)

        def publish() -> None:
            os.replace(record.staging_dir, record.final_dir)

        self._run_operation(self.DIRECTORY_PUBLISH, publish)
        record.published = True

    def publish_current(
        self,
        payload: str,
        *,
        transaction: PublicationTransaction,
    ) -> None:
        record = self._validated_transaction(transaction)
        if not record.published or not record.final_dir.is_dir():
            raise ValueError(
                "publication transaction generation is not ready to commit"
            )
        pointer = self._resolve_under(self.output_root, "current-generation.json")
        self._run_operation(
            self.POINTER_WRITE,
            lambda: _atomic_write_bytes(pointer, payload.encode("utf-8")),
        )
        self._transactions.pop(record.transaction_id, None)

    def write_root_text(self, relative: str | Path, payload: str) -> Path:
        target = self._resolve_under(self.output_root, relative)

        def write() -> Path:
            _atomic_write_bytes(target, payload.encode("utf-8"))
            return target

        return self._run_operation(self.ROOT_WRITE, write)

    def abort(self, transaction: PublicationTransaction | None) -> None:
        if transaction is None:
            return
        record = self._validated_transaction(transaction)
        if record.staging_dir.is_dir():
            shutil.rmtree(record.staging_dir)
        if record.published and record.final_dir.is_dir():
            shutil.rmtree(record.final_dir)
        self._transactions.pop(record.transaction_id, None)
