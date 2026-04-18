"""PyTest unit test fixtures and utilities."""

from collections.abc import Callable

import pytest
from fhy.lang.ast import (
    BinaryExpression,
    BinaryOperation,
    IdentifierExpression,
    Module,
    Node,
)
from fhy.lang.converter.from_fhy_source import from_fhy_source as fhy_source
from fhy_core import (
    CoreDataType,
    Identifier,
    NumericalType,
    PrimitiveDataType,
    Provenance,
)


@pytest.fixture
def construct_ast() -> Callable[[str], Node]:
    """Construct an abstract syntax tree (AST) from a raw text file source."""

    def _inner(source: str) -> Node:
        return fhy_source(source, provenance=Provenance.unknown())

    return _inner


@pytest.fixture
def int32() -> NumericalType:
    return NumericalType(PrimitiveDataType(CoreDataType.INT32))


@pytest.fixture
def x_plus_y_expression_ast() -> tuple[BinaryExpression, Identifier, Identifier]:
    """x + y expression AST."""
    x = Identifier("x")
    y = Identifier("y")
    expression = BinaryExpression(
        left=IdentifierExpression(identifier=x, provenance=Provenance.unknown()),
        right=IdentifierExpression(identifier=y, provenance=Provenance.unknown()),
        operation=BinaryOperation.ADDITION,
        provenance=Provenance.unknown(),
    )
    return expression, x, y


@pytest.fixture
def empty_module_ast() -> Module:
    return Module(
        provenance=Provenance.unknown(),
    )
