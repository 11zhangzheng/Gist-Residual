from dataclasses import fields
from pathlib import Path
from typing import Callable

import pytest

from fidmem.data.publication import PublicationBackend, PublicationTransaction


class RecordingPublicationBackend(PublicationBackend):
    def __init__(self, output_root: Path) -> None:
        super().__init__(output_root)
        self.operations: list[str] = []

    def _run_operation(self, name: str, action: Callable[[], object]) -> object:
        self.operations.append(name)
        return super()._run_operation(name, action)


def _forged_transaction(
    staging_dir: Path, final_dir: Path
) -> PublicationTransaction:
    transaction = object.__new__(PublicationTransaction)
    values: dict[str, object] = {
        "transaction_id": "forged-transaction",
        "generation_id": "a" * 20,
        "generation_uri": f"generations/{'a' * 20}",
        "staging_dir": staging_dir,
        "final_dir": final_dir,
        "published": False,
        "_capability": object(),
    }
    for field in fields(PublicationTransaction):
        object.__setattr__(transaction, field.name, values.get(field.name, object()))
    return transaction


def _invoke(
    backend: PublicationBackend,
    transaction: PublicationTransaction,
    operation: str,
) -> None:
    if operation == "write":
        backend.write_generation_text(transaction, "payload.txt", "unsafe")
    elif operation == "publish":
        backend.publish_generation(transaction)
    elif operation == "abort":
        backend.abort(transaction)
    else:
        backend.publish_current(transaction, seed=1)


@pytest.mark.parametrize("operation", ("write", "publish", "abort", "current"))
def test_forged_transactions_cannot_write_publish_or_abort(
    tmp_path: Path, operation: str
) -> None:
    backend = PublicationBackend(tmp_path / "output")
    staging = tmp_path / "outside-staging"
    staging.mkdir()
    sentinel = staging / "sentinel.txt"
    sentinel.write_text("keep")
    final = tmp_path / "outside-final"
    forged = _forged_transaction(staging, final)

    with pytest.raises(ValueError, match="transaction"):
        _invoke(backend, forged, operation)

    assert sentinel.read_text() == "keep"
    assert not (staging / "payload.txt").exists()
    assert not final.exists()


@pytest.mark.parametrize("operation", ("write", "publish", "abort", "current"))
def test_cross_backend_transactions_are_rejected_without_touching_owner_state(
    tmp_path: Path, operation: str
) -> None:
    owner = PublicationBackend(tmp_path / "owner")
    attacker = PublicationBackend(tmp_path / "attacker")
    transaction = owner.begin_generation("b" * 20)
    sentinel = transaction.staging_dir / "sentinel.txt"
    sentinel.write_text("keep")
    try:
        with pytest.raises(ValueError, match="transaction"):
            _invoke(attacker, transaction, operation)

        assert transaction.staging_dir.is_dir()
        assert sentinel.read_text() == "keep"
        assert not (transaction.staging_dir / "payload.txt").exists()
        assert not transaction.final_dir.exists()
    finally:
        owner.abort(transaction)


@pytest.mark.parametrize("location", ("inside", "outside"))
@pytest.mark.parametrize(
    ("field_name", "operation"),
    (
        ("staging_dir", "write"),
        ("staging_dir", "abort"),
        ("staging_dir", "current"),
        ("final_dir", "publish"),
        ("final_dir", "abort"),
        ("final_dir", "current"),
    ),
)
def test_mutated_transaction_paths_are_rejected_against_registry_values(
    tmp_path: Path,
    location: str,
    field_name: str,
    operation: str,
) -> None:
    backend = PublicationBackend(tmp_path / "output")
    transaction = backend.begin_generation("c" * 20)
    original = getattr(transaction, field_name)
    if operation == "current" or (
        operation == "abort" and field_name == "final_dir"
    ):
        backend.publish_generation(transaction)
    base = backend.output_root if location == "inside" else tmp_path / "outside"
    base.mkdir(parents=True, exist_ok=True)
    tampered = base / f"tampered-{field_name}-{operation}"
    sentinel = tampered / "sentinel.txt"
    if operation != "publish":
        tampered.mkdir()
        sentinel.write_text("keep")
    object.__setattr__(transaction, field_name, tampered)
    try:
        with pytest.raises(ValueError, match="transaction"):
            _invoke(backend, transaction, operation)

        if operation == "write":
            assert sentinel.read_text() == "keep"
            assert not (tampered / "payload.txt").exists()
        elif operation == "publish":
            assert not tampered.exists()
        elif operation == "current":
            assert not (backend.output_root / "current-generation.json").exists()
        else:
            assert sentinel.read_text() == "keep"
    finally:
        object.__setattr__(transaction, field_name, original)
        backend.abort(transaction)


def test_current_pointer_requires_a_published_owned_transaction(
    tmp_path: Path,
) -> None:
    backend = PublicationBackend(tmp_path / "output")

    with pytest.raises(ValueError, match="transaction"):
        backend.publish_current("not-a-transaction", seed=1)  # type: ignore[arg-type]

    assert not (backend.output_root / "current-generation.json").exists()


def test_current_pointer_rejects_a_missing_published_generation(
    tmp_path: Path,
) -> None:
    backend = PublicationBackend(tmp_path / "output")
    transaction = backend.begin_generation("d" * 20)
    backend.publish_generation(transaction)
    transaction.final_dir.rmdir()

    with pytest.raises(ValueError, match="transaction|generation"):
        backend.publish_current(transaction, seed=1)

    assert not (backend.output_root / "current-generation.json").exists()
    backend.abort(transaction)


def test_current_pointer_is_derived_from_the_owned_transaction_registry(
    tmp_path: Path,
) -> None:
    backend = PublicationBackend(tmp_path / "output")
    transaction = backend.begin_generation("e" * 20)
    backend.publish_generation(transaction)

    backend.publish_current(transaction, seed=17)

    assert (backend.output_root / "current-generation.json").read_text() == (
        '{"generation":"generations/' + "e" * 20 + '","seed":17}'
    )


def test_current_pointer_rejects_a_caller_supplied_wrong_generation_payload(
    tmp_path: Path,
) -> None:
    backend = PublicationBackend(tmp_path / "output")
    transaction = backend.begin_generation("f" * 20)
    backend.publish_generation(transaction)

    with pytest.raises(TypeError):
        backend.publish_current(
            '{"generation":"../../wrong"}', transaction=transaction
        )

    assert not (backend.output_root / "current-generation.json").exists()


@pytest.mark.parametrize(
    "relative",
    (
        "current-generation.json",
        f"generations/{'1' * 20}/manifest.json",
        f".staging/{'2' * 20}/payload.json",
        "nested/last-attempt.json",
    ),
)
def test_root_text_rejects_reserved_or_nested_paths_before_io(
    tmp_path: Path, relative: str
) -> None:
    backend = RecordingPublicationBackend(tmp_path / "output")
    target = backend.output_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("keep")

    with pytest.raises(ValueError, match="telemetry|root"):
        backend.write_root_text(relative, "overwrite")

    assert backend.operations == []
    assert target.read_text() == "keep"


def test_root_text_allows_explicit_last_attempt_telemetry(
    tmp_path: Path,
) -> None:
    backend = RecordingPublicationBackend(tmp_path / "output")

    path = backend.write_root_text("last-attempt.json", '{"status":"failed"}')

    assert backend.operations == [backend.ROOT_WRITE]
    assert path == backend.output_root / "last-attempt.json"
