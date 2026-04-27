"""Tests basic AST properties and methods."""

import pytest
from fhy_core import Identifier, Provenance

from fhy_lang.ast import (
    BinaryExpression,
    BinaryOperation,
    IdentifierExpression,
    IntLiteral,
    TupleExpression,
    UnaryExpression,
    UnaryOperation,
)


@pytest.fixture
def other_provenance() -> Provenance:
    """Return a provenance distinct from ``Provenance.unknown()``."""
    return Provenance.unknown().add_note("different")


def test_int_literal_equivalent_across_provenance(other_provenance):
    """Test int literals with same value but different provenance are equivalent."""
    a = IntLiteral(value=7)
    b = IntLiteral(value=7, provenance=other_provenance)
    assert a.provenance != b.provenance
    assert a.is_structurally_equivalent(b)
    assert b.is_structurally_equivalent(a)


def test_int_literal_inequivalent_when_value_differs():
    """Test int literals with different values are not structurally equivalent."""
    a = IntLiteral(value=7)
    b = IntLiteral(value=8)
    assert not a.is_structurally_equivalent(b)


def test_identifier_expression_equivalent_across_provenance(other_provenance):
    """Test identifier expressions with shared identifier are equivalent."""
    ident = Identifier("x")
    a = IdentifierExpression(identifier=ident)
    b = IdentifierExpression(identifier=ident, provenance=other_provenance)
    assert a.provenance != b.provenance
    assert a.is_structurally_equivalent(b)


def test_unary_expression_equivalent_across_provenance(other_provenance):
    """Test unary expression equivalence ignores provenance on node and operand."""
    operand_a = IntLiteral(value=1)
    operand_b = IntLiteral(value=1, provenance=other_provenance)
    a = UnaryExpression(
        operation=UnaryOperation.NEGATION,
        expression=operand_a,
    )
    b = UnaryExpression(
        operation=UnaryOperation.NEGATION,
        expression=operand_b,
        provenance=other_provenance,
    )
    assert a.is_structurally_equivalent(b)


def test_binary_expression_equivalent_across_provenance(other_provenance):
    """Test binary expression equivalence ignores provenance on node and operands."""

    def make(p_outer: Provenance, p_inner: Provenance) -> BinaryExpression:
        return BinaryExpression(
            operation=BinaryOperation.ADDITION,
            left=IntLiteral(value=1, provenance=p_inner),
            right=IntLiteral(value=2, provenance=p_inner),
            provenance=p_outer,
        )

    a = make(Provenance.unknown(), Provenance.unknown())
    b = make(other_provenance, other_provenance)
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


def test_tuple_expression_equivalent_across_provenance(other_provenance):
    """Test tuple expression equivalence ignores provenance on node and elements."""
    a = TupleExpression(
        expressions=(
            IntLiteral(value=1),
            IntLiteral(value=2),
        ),
    )
    b = TupleExpression(
        expressions=(
            IntLiteral(value=1, provenance=other_provenance),
            IntLiteral(value=2, provenance=other_provenance),
        ),
        provenance=other_provenance,
    )
    assert a.is_structurally_equivalent(b)
