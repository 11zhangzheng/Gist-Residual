from __future__ import annotations

from pathlib import Path

import pytest

from fidmem.storage import run_store


class _FakeHandle:
    def __init__(
        self,
        calls: list[str],
        *,
        close_error: BaseException | None = None,
    ) -> None:
        self.calls = calls
        self.close_error = close_error

    def seek(self, offset: int, whence: int = 0) -> int:
        return 0

    def tell(self) -> int:
        return 1

    def close(self) -> None:
        self.calls.append("close")
        if self.close_error is not None:
            raise self.close_error


class _FakeThreadLock:
    def __init__(
        self,
        calls: list[str],
        *,
        release_error: BaseException | None = None,
    ) -> None:
        self.calls = calls
        self.release_error = release_error

    def acquire(self, *, timeout: float) -> bool:
        self.calls.append("acquire")
        return True

    def release(self) -> None:
        self.calls.append("release")
        if self.release_error is not None:
            raise self.release_error


def _inject_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[str],
    *,
    unlock_error: BaseException | None = None,
    close_error: BaseException | None = None,
    release_error: BaseException | None = None,
) -> None:
    handle = _FakeHandle(calls, close_error=close_error)
    thread_lock = _FakeThreadLock(calls, release_error=release_error)
    monkeypatch.setattr(run_store, "open", lambda path, mode: handle, raising=False)
    monkeypatch.setattr(run_store, "_thread_lock", lambda path: thread_lock)
    monkeypatch.setattr(run_store, "_try_lock_handle", lambda current: calls.append("lock"))

    def unlock(current: object) -> None:
        calls.append("unlock")
        if unlock_error is not None:
            raise unlock_error

    monkeypatch.setattr(run_store, "_unlock_handle", unlock)


@pytest.mark.parametrize(
    "body_error",
    [
        ValueError("body failed"),
        RuntimeError("body failed"),
        KeyboardInterrupt("body failed"),
        SystemExit("body failed"),
    ],
)
def test_body_error_wins_over_all_cleanup_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    body_error: BaseException,
) -> None:
    calls: list[str] = []
    _inject_cleanup(
        monkeypatch,
        calls,
        unlock_error=OSError("unlock failed"),
        close_error=OSError("close failed"),
        release_error=OSError("release failed"),
    )

    with pytest.raises(type(body_error)) as caught:
        with run_store._claim_file_lock(tmp_path / "body.lock", 1):
            raise body_error

    assert caught.value is body_error
    assert calls == ["acquire", "lock", "unlock", "close", "release"]


def test_first_cleanup_error_wins_after_successful_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    unlock_error = OSError("unlock failed")
    _inject_cleanup(
        monkeypatch,
        calls,
        unlock_error=unlock_error,
        close_error=OSError("close failed"),
        release_error=OSError("release failed"),
    )

    with pytest.raises(OSError) as caught:
        with run_store._claim_file_lock(tmp_path / "all-cleanup.lock", 1):
            pass

    assert caught.value is unlock_error
    assert calls == ["acquire", "lock", "unlock", "close", "release"]


def test_close_error_wins_when_unlock_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    close_error = OSError("close failed")
    _inject_cleanup(
        monkeypatch,
        calls,
        close_error=close_error,
        release_error=OSError("release failed"),
    )

    with pytest.raises(OSError) as caught:
        with run_store._claim_file_lock(tmp_path / "close.lock", 1):
            pass

    assert caught.value is close_error
    assert calls == ["acquire", "lock", "unlock", "close", "release"]


def test_release_error_propagates_when_it_is_the_only_cleanup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    release_error = OSError("release failed")
    _inject_cleanup(monkeypatch, calls, release_error=release_error)

    with pytest.raises(OSError) as caught:
        with run_store._claim_file_lock(tmp_path / "release.lock", 1):
            pass

    assert caught.value is release_error
    assert calls == ["acquire", "lock", "unlock", "close", "release"]
