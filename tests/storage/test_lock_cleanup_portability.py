from __future__ import annotations

import ast
import builtins
from pathlib import Path
import runpy

import pytest


_TARGET = Path(__file__).with_name("test_lock_cleanup.py")


def test_cleanup_tests_have_no_top_level_msvcrt_import() -> None:
    tree = ast.parse(_TARGET.read_text(encoding="utf-8"))
    top_level_imports = {
        alias.name
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "msvcrt" not in top_level_imports


def test_cleanup_test_module_executes_when_msvcrt_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "msvcrt":
            raise ImportError("simulated POSIX collection")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    namespace = runpy.run_path(str(_TARGET))
    assert "test_body_error_wins_over_all_cleanup_failures" in namespace
