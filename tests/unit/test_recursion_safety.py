"""Tests recursion-depth handling in the CLI and AST validators."""

import sys

import pytest
from fhy_core import (
    CoreDataType,
    Identifier,
    NumericalType,
    PrimitiveDataType,
    TypeQualifier,
)

from fhy_lang import validate_ast
from fhy_lang.ast import (
    Argument,
    BinaryExpression,
    BinaryOperation,
    ExpressionStatement,
    IdentifierExpression,
    IntLiteral,
    Module,
    Procedure,
    QualifiedType,
)
from fhy_lang.cli import RECURSION_LIMIT, compile_fhy_source

# Programmatically chosen so the AST is well above the default ~1000
# Python recursion limit but well within the bumped CLI limit. We do
# not also test "huge depth raises RecursionError" because Python's
# setrecursionlimit only gates the interpreter's check; the OS thread
# stack can run out first (default ~8 MB on Linux, smaller on Windows),
# and the result is a SIGSEGV rather than a clean RecursionError. That
# behavior is a property of CPython, not of fhy_lang.
_DEEP_BUT_SAFE_DEPTH = 5_000


def _int32() -> NumericalType:
    return NumericalType(PrimitiveDataType(CoreDataType.INT32))


def _make_left_leaning_addition_chain(depth: int) -> BinaryExpression:
    """Build ``((((1 + 1) + 1) + 1) ... + 1)`` with ``depth`` operations."""
    expression: BinaryExpression | IntLiteral = IntLiteral(value=1)
    for _ in range(depth):
        expression = BinaryExpression(
            operation=BinaryOperation.ADDITION,
            left=expression,
            right=IntLiteral(value=1),
        )
    return expression  # type: ignore[return-value]


def _make_module_assigning_expression_to_b(expression: BinaryExpression) -> Module:
    b = Identifier("b")
    procedure = Procedure(
        name=Identifier("main"),
        args=(
            Argument(
                name=b,
                qualified_type=QualifiedType(
                    base_type=_int32(),
                    type_qualifier=TypeQualifier.OUTPUT,
                ),
            ),
        ),
        body=(
            ExpressionStatement(
                left=IdentifierExpression(identifier=b),
                right=expression,
            ),
        ),
    )
    return Module(statements=(procedure,))


def test_recursion_limit_constant_is_high_enough_for_deep_asts():
    """Test the documented recursion limit covers ASTs of at least 5000 nodes.

    Empirically each AST node costs ~3-4 stack frames during pass walks,
    so a 5000-deep chain needs ~20_000 frames. RECURSION_LIMIT must
    exceed that with margin.
    """
    assert RECURSION_LIMIT >= 30_000


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


def test_validate_ast_handles_deep_chain_with_raised_limit():
    """Test validate_ast survives a 5000-deep AST chain when the limit is raised."""
    original_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(RECURSION_LIMIT)
    try:
        deep_expression = _make_left_leaning_addition_chain(_DEEP_BUT_SAFE_DEPTH)
        program_ast = _make_module_assigning_expression_to_b(deep_expression)

        output_ast, _ = validate_ast(program_ast, perform_optimizations=False)

        assert output_ast is not None
    finally:
        sys.setrecursionlimit(original_limit)
