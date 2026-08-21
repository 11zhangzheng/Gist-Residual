from __future__ import annotations

import msvcrt
from pathlib import Path

import pytest

from fidmem.storage import run_store


class _FakeHandle:
    def __init__(self, *, close_error: OSError | None = None) -> None:
        self.close_error = close_error
        self.closed = False

    def seek(self, offset: int, whence: int = 0) -> int:
        return 0

    def tell(self) -> int:
        return 1

    def fileno(self) -> int:
        return 1

    def close(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


def _inject_handle(
    monkeypatch: pytest.MonkeyPatch,
    handle: _FakeHandle,
    *,
    unlock_error: OSError | None,
    calls: list[str],
) -> None:
    monkeypatch.setattr(run_store, "open", lambda path, mode: handle, raising=False)

    def locking(fd: int, mode: int, count: int) -> None:
        if mode == msvcrt.LK_UNLCK:
            calls.append("unlock")
            if unlock_error is not None:
                raise unlock_error
        else:
            calls.append("lock")

    monkeypatch.setattr(msvcrt, "locking", locking)


def test_body_error_wins_over_unlock_error_and_close_is_still_attempted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    handle = _FakeHandle()
    _inject_handle(monkeypatch, handle, unlock_error=OSError("unlock failed"), calls=calls)

    with pytest.raises(ValueError, match="body failed"):
        with run_store._claim_file_lock(tmp_path / "body-unlock.lock", 1):
            raise ValueError("body failed")

    assert calls == ["lock", "unlock"]
    assert handle.closed is True


def test_unlock_error_propagates_after_successful_body_and_close_is_attempted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    handle = _FakeHandle()
    _inject_handle(monkeypatch, handle, unlock_error=OSError("unlock failed"), calls=calls)

    with pytest.raises(OSError, match="unlock failed"):
        with run_store._claim_file_lock(tmp_path / "unlock.lock", 1):
            pass

    assert calls == ["lock", "unlock"]
    assert handle.closed is True


def test_body_error_wins_over_close_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    handle = _FakeHandle(close_error=OSError("close failed"))
    _inject_handle(monkeypatch, handle, unlock_error=None, calls=calls)

    with pytest.raises(ValueError, match="body failed"):
        with run_store._claim_file_lock(tmp_path / "body-close.lock", 1):
            raise ValueError("body failed")

    assert calls == ["lock", "unlock"]
    assert handle.closed is True


def test_close_error_propagates_after_successful_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    handle = _FakeHandle(close_error=OSError("close failed"))
    _inject_handle(monkeypatch, handle, unlock_error=None, calls=calls)

    with pytest.raises(OSError, match="close failed"):
        with run_store._claim_file_lock(tmp_path / "close.lock", 1):
            pass

    assert calls == ["lock", "unlock"]
    assert handle.closed is True


def test_body_error_wins_when_unlock_and_close_both_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    handle = _FakeHandle(close_error=OSError("close failed"))
    _inject_handle(monkeypatch, handle, unlock_error=OSError("unlock failed"), calls=calls)

    with pytest.raises(ValueError, match="body failed"):
        with run_store._claim_file_lock(tmp_path / "body-both.lock", 1):
            raise ValueError("body failed")

    assert calls == ["lock", "unlock"]
    assert handle.closed is True


def test_first_cleanup_error_wins_when_unlock_and_close_both_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    handle = _FakeHandle(close_error=OSError("close failed"))
    _inject_handle(monkeypatch, handle, unlock_error=OSError("unlock failed"), calls=calls)

    with pytest.raises(OSError, match="unlock failed"):
        with run_store._claim_file_lock(tmp_path / "both.lock", 1):
            pass

    assert calls == ["lock", "unlock"]
    assert handle.closed is True
