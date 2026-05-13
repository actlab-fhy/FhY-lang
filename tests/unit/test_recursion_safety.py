"""Tests recursion-depth handling in the CLI.

Behavioral "validate_ast handles a deep AST" test deliberately not
included. ``sys.setrecursionlimit`` only gates Python's
interpreter-level check; it cannot enlarge the OS thread stack. On
Linux/macOS the default 8 MB stack is roomy enough that a 5000-deep
expression chain fits within RECURSION_LIMIT; on Windows the default
is ~1 MB, and the same chain hits a hard SIGSEGV before Python's
RecursionError check ever fires (xdist subprocess workers inherit the
default). So the practical depth ceiling is OS-dependent in a way the
``RECURSION_LIMIT`` constant alone cannot describe.

The two tests below verify what is portably true: the constant is set
to a sensible value, and the CLI applies it before any compilation
work begins. Anything more would be testing CPython's stack
allocator, not fhy_lang.
"""

import sys

import pytest

from fhy_lang.cli import RECURSION_LIMIT, compile_fhy_source


def test_recursion_limit_constant_is_above_python_default():
    """Test the documented recursion limit comfortably exceeds the Python default."""
    assert RECURSION_LIMIT >= 10_000


def test_compile_fhy_source_raises_recursion_limit(monkeypatch):
    """Test the CLI entry point raises sys.recursionlimit before any compilation."""
    recorded_limits: list[int] = []
    monkeypatch.setattr(sys, "setrecursionlimit", recorded_limits.append)

    with pytest.raises(SystemExit):
        compile_fhy_source(main_file=None)

    assert any(value >= RECURSION_LIMIT for value in recorded_limits), (
        f"expected sys.setrecursionlimit to be called with >= {RECURSION_LIMIT}, "
        f"got {recorded_limits!r}"
    )
