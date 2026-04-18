"""Tests the FhY AST to core expression converter."""

import pytest
from fhy.lang import (
    ASTBinaryExpression,
    ASTBinaryOperation,
    ASTComplexLiteral,
    ASTExpression,
    ASTFloatLiteral,
    ASTIdentifierExpression,
    ASTIntLiteral,
    ASTUnaryExpression,
    ASTUnaryOperation,
    convert_ast_expression_to_core_expression,
)
from fhy_core import BinaryExpression as CoreBinaryExpression
from fhy_core import BinaryOperation as CoreBinaryOperation
from fhy_core import Expression as CoreExpression
from fhy_core import Identifier, Provenance
from fhy_core import IdentifierExpression as CoreIdentifierExpression
from fhy_core import LiteralExpression as CoreLiteralExpression
from fhy_core import UnaryExpression as CoreUnaryExpression
from fhy_core import UnaryOperation as CoreUnaryOperation


def test_convert_identifier_expression():
    """Test FhY AST identifier expression conversion works as expected."""
    a = Identifier("A")
    ast_expression = ASTIdentifierExpression(
        identifier=a, provenance=Provenance.unknown()
    )
    core_expression = convert_ast_expression_to_core_expression(ast_expression)
    expected_core_expression = CoreIdentifierExpression(a)
    assert core_expression.is_structurally_equivalent(expected_core_expression)


@pytest.mark.parametrize(
    ["ast_expression", "core_expression"],
    [
        (
            ASTIntLiteral(value=1, provenance=Provenance.unknown()),
            CoreLiteralExpression(1),
        ),
        (
            ASTFloatLiteral(value=1.0, provenance=Provenance.unknown()),
            CoreLiteralExpression(1.0),
        ),
        (
            ASTUnaryExpression(
                operation=ASTUnaryOperation.NEGATION,
                expression=ASTIntLiteral(value=1, provenance=Provenance.unknown()),
                provenance=Provenance.unknown(),
            ),
            CoreUnaryExpression(CoreUnaryOperation.NEGATE, CoreLiteralExpression(1)),
        ),
        (
            ASTBinaryExpression(
                operation=ASTBinaryOperation.ADDITION,
                left=ASTIntLiteral(value=1, provenance=Provenance.unknown()),
                right=ASTIntLiteral(value=2, provenance=Provenance.unknown()),
                provenance=Provenance.unknown(),
            ),
            CoreBinaryExpression(
                CoreBinaryOperation.ADD,
                CoreLiteralExpression(1),
                CoreLiteralExpression(2),
            ),
        ),
        (
            ASTBinaryExpression(
                operation=ASTBinaryOperation.SUBTRACTION,
                left=ASTBinaryExpression(
                    operation=ASTBinaryOperation.ADDITION,
                    left=ASTIntLiteral(value=1, provenance=Provenance.unknown()),
                    right=ASTIntLiteral(value=2, provenance=Provenance.unknown()),
                    provenance=Provenance.unknown(),
                ),
                right=ASTIntLiteral(value=3, provenance=Provenance.unknown()),
                provenance=Provenance.unknown(),
            ),
            CoreBinaryExpression(
                CoreBinaryOperation.SUBTRACT,
                CoreBinaryExpression(
                    CoreBinaryOperation.ADD,
                    CoreLiteralExpression(1),
                    CoreLiteralExpression(2),
                ),
                CoreLiteralExpression(3),
            ),
        ),
    ],
)
def test_ast_to_core_expression(
    ast_expression: ASTExpression, core_expression: CoreExpression
):
    """Test FhY AST to core expression conversion works as expected."""
    assert convert_ast_expression_to_core_expression(
        ast_expression
    ).is_structurally_equivalent(core_expression)


@pytest.mark.parametrize(
    ["ast_operation", "core_operation"],
    [
        (ASTUnaryOperation.NEGATION, CoreUnaryOperation.NEGATE),
        (ASTUnaryOperation.LOGICAL_NOT, CoreUnaryOperation.LOGICAL_NOT),
    ],
)
def test_unary_operation_lowering(
    ast_operation: ASTUnaryOperation, core_operation: CoreUnaryOperation
):
    """Test FhY AST unary operation lowering works as expected."""
    ast_expression = ASTUnaryExpression(
        operation=ast_operation,
        expression=ASTIntLiteral(value=1, provenance=Provenance.unknown()),
        provenance=Provenance.unknown(),
    )
    core_expression = convert_ast_expression_to_core_expression(ast_expression)
    expected_core_expression = CoreUnaryExpression(
        core_operation, CoreLiteralExpression(1)
    )
    assert core_expression.is_structurally_equivalent(expected_core_expression)


@pytest.mark.parametrize(
    ["ast_operation", "core_operation"],
    [
        (ASTBinaryOperation.ADDITION, CoreBinaryOperation.ADD),
        (ASTBinaryOperation.SUBTRACTION, CoreBinaryOperation.SUBTRACT),
        (ASTBinaryOperation.MULTIPLICATION, CoreBinaryOperation.MULTIPLY),
        (ASTBinaryOperation.DIVISION, CoreBinaryOperation.DIVIDE),
        (ASTBinaryOperation.FLOORDIV, CoreBinaryOperation.FLOOR_DIVIDE),
        (ASTBinaryOperation.MODULO, CoreBinaryOperation.MODULO),
        (ASTBinaryOperation.POWER, CoreBinaryOperation.POWER),
        (ASTBinaryOperation.EQUAL_TO, CoreBinaryOperation.EQUAL),
        (ASTBinaryOperation.NOT_EQUAL_TO, CoreBinaryOperation.NOT_EQUAL),
        (ASTBinaryOperation.LESS_THAN, CoreBinaryOperation.LESS),
        (ASTBinaryOperation.LESS_THAN_OR_EQUAL, CoreBinaryOperation.LESS_EQUAL),
        (ASTBinaryOperation.GREATER_THAN, CoreBinaryOperation.GREATER),
    ],
)
def test_binary_operation_lowering(
    ast_operation: ASTBinaryOperation, core_operation: CoreBinaryOperation
):
    """Test FhY AST binary operation lowering works as expected."""
    ast_expression = ASTBinaryExpression(
        operation=ast_operation,
        left=ASTIntLiteral(value=1, provenance=Provenance.unknown()),
        right=ASTIntLiteral(value=2, provenance=Provenance.unknown()),
        provenance=Provenance.unknown(),
    )
    core_expression = convert_ast_expression_to_core_expression(ast_expression)
    expected_core_expression = CoreBinaryExpression(
        core_operation, CoreLiteralExpression(1), CoreLiteralExpression(2)
    )
    assert core_expression.is_structurally_equivalent(expected_core_expression)


def test_fails_with_complex_literal():
    """Test FhY AST to core expression conversion fails with a complex literal."""
    ast_expression = ASTComplexLiteral(value=1.0j, provenance=Provenance.unknown())
    with pytest.raises(NotImplementedError):
        convert_ast_expression_to_core_expression(ast_expression)
