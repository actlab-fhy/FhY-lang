"""PyTest unit test fixtures and utilities."""

from collections.abc import Callable

import pytest
from fhy_core import (
    CoreDataType,
    Identifier,
    IndexType,
    NumericalType,
    PrimitiveDataType,
    Provenance,
    TypeQualifier,
)
from fhy_core import (
    IdentifierExpression as CoreIdentifierExpression,
)
from fhy_core import (
    LiteralExpression as CoreLiteralExpression,
)
from fhy_lang.lang.ast import (
    Argument,
    ArrayAccessExpression,
    BinaryExpression,
    BinaryOperation,
    DeclarationStatement,
    ExpressionStatement,
    ForAllStatement,
    IdentifierExpression,
    Module,
    Node,
    Procedure,
    QualifiedType,
)
from fhy_lang.lang.converter.from_fhy_source import from_fhy_source as fhy_source


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


@pytest.fixture
def forall_vector_sum_module_ast(
    int32,
) -> tuple[Module, Identifier, Identifier, Identifier, Identifier, Identifier]:
    """
    proc main(input int32[N] a, output int32[N] b) {
        temp index[1:N] i;
        temp int32 acc;
        acc = a[i];
        forall (i) {
            acc = acc + a[i];
        }
        b[i] = acc;
    }
    """
    a, b, i, acc, N = (
        Identifier("a"),
        Identifier("b"),
        Identifier("i"),
        Identifier("acc"),
        Identifier("N"),
    )
    index_declaration_ast = DeclarationStatement(
        variable_name=i,
        variable_type=QualifiedType(
            base_type=IndexType(
                lower_bound=CoreLiteralExpression(1),
                upper_bound=CoreIdentifierExpression(N),
                stride=None,
            ),
            type_qualifier=TypeQualifier.TEMP,
            provenance=Provenance.unknown(),
        ),
        provenance=Provenance.unknown(),
    )
    acc_declaration_ast = DeclarationStatement(
        variable_name=acc,
        variable_type=QualifiedType(
            base_type=int32,
            type_qualifier=TypeQualifier.TEMP,
            provenance=Provenance.unknown(),
        ),
        provenance=Provenance.unknown(),
    )
    acc_init_ast = ExpressionStatement(
        left=IdentifierExpression(identifier=acc, provenance=Provenance.unknown()),
        right=ArrayAccessExpression(
            array_expression=IdentifierExpression(
                identifier=a, provenance=Provenance.unknown()
            ),
            indices=(
                IdentifierExpression(identifier=i, provenance=Provenance.unknown()),
            ),
            provenance=Provenance.unknown(),
        ),
        provenance=Provenance.unknown(),
    )
    acc_update_ast = ExpressionStatement(
        left=IdentifierExpression(identifier=acc, provenance=Provenance.unknown()),
        right=BinaryExpression(
            left=IdentifierExpression(identifier=acc, provenance=Provenance.unknown()),
            right=IdentifierExpression(identifier=a, provenance=Provenance.unknown()),
            operation=BinaryOperation.ADDITION,
            provenance=Provenance.unknown(),
        ),
        provenance=Provenance.unknown(),
    )
    b_assign_ast = ExpressionStatement(
        left=ArrayAccessExpression(
            array_expression=IdentifierExpression(
                identifier=b, provenance=Provenance.unknown()
            ),
            indices=(
                IdentifierExpression(identifier=i, provenance=Provenance.unknown()),
            ),
            provenance=Provenance.unknown(),
        ),
        right=IdentifierExpression(identifier=acc, provenance=Provenance.unknown()),
        provenance=Provenance.unknown(),
    )
    forall_ast = ForAllStatement(
        index=index_declaration_ast,
        body=(acc_update_ast,),
        provenance=Provenance.unknown(),
    )
    procedure_ast = Procedure(
        name=Identifier("main"),
        args=(
            Argument(
                name=a,
                qualified_type=QualifiedType(
                    base_type=NumericalType(
                        PrimitiveDataType(CoreDataType.INT32),
                        shape=(CoreIdentifierExpression(N),),
                    ),
                    type_qualifier=TypeQualifier.INPUT,
                    provenance=Provenance.unknown(),
                ),
                provenance=Provenance.unknown(),
            ),
            Argument(
                name=b,
                qualified_type=QualifiedType(
                    base_type=int32,
                    type_qualifier=TypeQualifier.OUTPUT,
                    provenance=Provenance.unknown(),
                ),
                provenance=Provenance.unknown(),
            ),
        ),
        body=(
            index_declaration_ast,
            acc_declaration_ast,
            acc_init_ast,
            forall_ast,
            b_assign_ast,
        ),
        provenance=Provenance.unknown(),
    )
    ast = Module(statements=(procedure_ast,), provenance=Provenance.unknown())
    return ast, a, b, i, acc, N
