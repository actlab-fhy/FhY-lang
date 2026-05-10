"""Tests basic AST properties and methods."""

from fhy_core import Identifier

from fhy_lang.ast import (
    BinaryExpression,
    BinaryOperation,
    IdentifierExpression,
    IntLiteral,
    TupleExpression,
    UnaryExpression,
    UnaryOperation,
)


def test_int_literal_equivalent_when_value_matches():
    """Test int literals with the same value are structurally equivalent."""
    a = IntLiteral(value=7)
    b = IntLiteral(value=7)

    assert a.is_structurally_equivalent(b)
    assert b.is_structurally_equivalent(a)


def test_int_literal_inequivalent_when_value_differs():
    """Test int literals with different values are not structurally equivalent."""
    a = IntLiteral(value=7)
    b = IntLiteral(value=8)

    assert not a.is_structurally_equivalent(b)


def test_identifier_expression_equivalent_when_identifier_matches():
    """Test identifier expressions sharing an identifier are equivalent."""
    ident = Identifier("x")
    a = IdentifierExpression(identifier=ident)
    b = IdentifierExpression(identifier=ident)

    assert a.is_structurally_equivalent(b)


def test_unary_expression_equivalent_when_operation_and_operand_match():
    """Test unary expressions with matching operation and operand are equivalent."""
    a = UnaryExpression(
        operation=UnaryOperation.NEGATION,
        expression=IntLiteral(value=1),
    )
    b = UnaryExpression(
        operation=UnaryOperation.NEGATION,
        expression=IntLiteral(value=1),
    )

    assert a.is_structurally_equivalent(b)


def test_binary_expression_equivalent_when_operation_and_operands_match():
    """Test binary expressions with matching operation and operands are equivalent."""
    a = BinaryExpression(
        operation=BinaryOperation.ADDITION,
        left=IntLiteral(value=1),
        right=IntLiteral(value=2),
    )
    b = BinaryExpression(
        operation=BinaryOperation.ADDITION,
        left=IntLiteral(value=1),
        right=IntLiteral(value=2),
    )

    assert a.is_structurally_equivalent(b)


def test_binary_expression_inequivalent_when_operation_differs():
    """Test binary expressions with different operators are not equivalent."""
    a = BinaryExpression(
        operation=BinaryOperation.ADDITION,
        left=IntLiteral(value=1),
        right=IntLiteral(value=2),
    )
    b = BinaryExpression(
        operation=BinaryOperation.SUBTRACTION,
        left=IntLiteral(value=1),
        right=IntLiteral(value=2),
    )

    assert not a.is_structurally_equivalent(b)


def test_tuple_expression_equivalent_when_elements_match():
    """Test tuple expressions with matching elements are equivalent."""
    a = TupleExpression(
        expressions=(IntLiteral(value=1), IntLiteral(value=2)),
    )
    b = TupleExpression(
        expressions=(IntLiteral(value=1), IntLiteral(value=2)),
    )

    assert a.is_structurally_equivalent(b)
