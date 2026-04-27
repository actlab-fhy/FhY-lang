"""Tests for the FhY expression side-effect analysis."""

from fhy_core import (
    Identifier,
)

from fhy_lang.ast import (
    BinaryExpression,
    BinaryOperation,
    FunctionExpression,
    IdentifierExpression,
    IntLiteral,
)
from fhy_lang.ast.passes import ExpressionSideEffectAnalysis
from fhy_lang.builtins import (
    BUILTIN_REDUCTION_FUNCTION_IDENTIFIERS,
)


def _make_int_literal(value: int) -> IntLiteral:
    return IntLiteral(value=value)


def _make_identifier_expr(name: Identifier) -> IdentifierExpression:
    return IdentifierExpression(identifier=name)


def test_plain_identifier_has_no_side_effects():
    """Test that a bare `IdentifierExpression` is flagged as side-effect free."""
    analysis = ExpressionSideEffectAnalysis()

    assert analysis.run(_make_identifier_expr(Identifier("x"))) is False


def test_literal_has_no_side_effects():
    """Test that a bare literal is flagged as side-effect free."""
    analysis = ExpressionSideEffectAnalysis()

    assert analysis.run(_make_int_literal(42)) is False


def test_binary_of_pure_operands_has_no_side_effects():
    """Test that `x + 1` is flagged as side-effect free."""
    expression = BinaryExpression(
        operation=BinaryOperation.ADDITION,
        left=_make_identifier_expr(Identifier("x")),
        right=_make_int_literal(1),
    )

    assert ExpressionSideEffectAnalysis().run(expression) is False


def test_non_reduction_call_has_side_effects():
    """Test that a user-defined operation call is flagged as side-effecting.

    The analysis is conservative: any :class:`FunctionExpression` whose
    callee is not a built-in reduction is flagged, even though FhY
    operations are semantically pure.

    """
    foo = Identifier("foo")
    call = FunctionExpression(
        function=_make_identifier_expr(foo),
        args=(_make_identifier_expr(Identifier("x")),),
    )

    assert ExpressionSideEffectAnalysis().run(call) is True


def test_builtin_reduction_call_has_no_side_effects():
    """Test that a built-in reduction (e.g. `sum`) is flagged as pure."""
    reduction_identifier = next(iter(BUILTIN_REDUCTION_FUNCTION_IDENTIFIERS.values()))
    call = FunctionExpression(
        function=_make_identifier_expr(reduction_identifier),
        args=(_make_identifier_expr(Identifier("x")),),
    )

    assert ExpressionSideEffectAnalysis().run(call) is False


def test_nested_non_reduction_call_is_detected():
    """Test that a call hidden inside a subtree is still flagged."""
    foo = Identifier("foo")
    nested_call = FunctionExpression(
        function=_make_identifier_expr(foo),
        args=(_make_identifier_expr(Identifier("x")),),
    )
    expression = BinaryExpression(
        operation=BinaryOperation.ADDITION,
        left=_make_identifier_expr(Identifier("x")),
        right=nested_call,
    )

    assert ExpressionSideEffectAnalysis().run(expression) is True
