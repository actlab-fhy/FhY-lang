"""Converter from AST expressions to core expressions."""

__all__ = ["convert_ast_expression_to_core_expression"]

from typing import ClassVar

from fhy_core import BinaryExpression as CoreBinaryExpression
from fhy_core import BinaryOperation as CoreBinaryOperation
from fhy_core import Expression as CoreExpression
from fhy_core import IdentifierExpression as CoreIdentifierExpression
from fhy_core import LiteralExpression as CoreLiteralExpression
from fhy_core import UnaryExpression as CoreUnaryExpression
from fhy_core import UnaryOperation as CoreUnaryOperation
from fhy_core import VisitablePass, register_pass
from frozendict import frozendict

from fhy_lang.lang.ast.node import BinaryExpression as ASTBinaryExpression
from fhy_lang.lang.ast.node import BinaryOperation as ASTBinaryOperation
from fhy_lang.lang.ast.node import ComplexLiteral as ASTComplexLiteralExpression
from fhy_lang.lang.ast.node import Expression as ASTExpression
from fhy_lang.lang.ast.node import FloatLiteral as ASTFloatLiteralExpression
from fhy_lang.lang.ast.node import IdentifierExpression as ASTIdentifierExpression
from fhy_lang.lang.ast.node import IntLiteral as ASTIntLiteralExpression
from fhy_lang.lang.ast.node import UnaryExpression as ASTUnaryExpression
from fhy_lang.lang.ast.node import UnaryOperation as ASTUnaryOperation

_AST_TO_CORE_UNARY_OPERATIONS: frozendict[ASTUnaryOperation, CoreUnaryOperation] = (
    frozendict(
        {
            ASTUnaryOperation.NEGATION: CoreUnaryOperation.NEGATE,
            ASTUnaryOperation.LOGICAL_NOT: CoreUnaryOperation.LOGICAL_NOT,
        }
    )
)

_AST_TO_CORE_BINARY_OPERATIONS: frozendict[ASTBinaryOperation, CoreBinaryOperation] = (
    frozendict(
        {
            ASTBinaryOperation.ADDITION: CoreBinaryOperation.ADD,
            ASTBinaryOperation.SUBTRACTION: CoreBinaryOperation.SUBTRACT,
            ASTBinaryOperation.MULTIPLICATION: CoreBinaryOperation.MULTIPLY,
            ASTBinaryOperation.DIVISION: CoreBinaryOperation.DIVIDE,
            ASTBinaryOperation.FLOORDIV: CoreBinaryOperation.FLOOR_DIVIDE,
            ASTBinaryOperation.MODULO: CoreBinaryOperation.MODULO,
            ASTBinaryOperation.POWER: CoreBinaryOperation.POWER,
            ASTBinaryOperation.EQUAL_TO: CoreBinaryOperation.EQUAL,
            ASTBinaryOperation.NOT_EQUAL_TO: CoreBinaryOperation.NOT_EQUAL,
            ASTBinaryOperation.LESS_THAN: CoreBinaryOperation.LESS,
            ASTBinaryOperation.LESS_THAN_OR_EQUAL: CoreBinaryOperation.LESS_EQUAL,
            ASTBinaryOperation.GREATER_THAN: CoreBinaryOperation.GREATER,
            ASTBinaryOperation.GREATER_THAN_OR_EQUAL: CoreBinaryOperation.GREATER_EQUAL,
            ASTBinaryOperation.LOGICAL_AND: CoreBinaryOperation.LOGICAL_AND,
            ASTBinaryOperation.LOGICAL_OR: CoreBinaryOperation.LOGICAL_OR,
        }
    )
)


@register_pass(
    "fhy_ast_to_core_expression_converter",
    "Converts FhY AST expressions to FhY Core expressions.",
)
class ASTToCoreExpressionConverter(VisitablePass[ASTExpression, CoreExpression]):
    """Convert AST expressions to core expressions."""

    _ast_to_core_unary_operations: ClassVar[
        frozendict[ASTUnaryOperation, CoreUnaryOperation]
    ] = _AST_TO_CORE_UNARY_OPERATIONS
    _ast_to_core_binary_operations: ClassVar[
        frozendict[ASTBinaryOperation, CoreBinaryOperation]
    ] = _AST_TO_CORE_BINARY_OPERATIONS

    def get_noop_output(self, ir: ASTExpression) -> CoreExpression:
        raise RuntimeError("This pass does not support a noop output.")

    def visit_unary_expression(self, node: ASTUnaryExpression) -> CoreUnaryExpression:
        return CoreUnaryExpression(
            self._ast_to_core_unary_operations[node.operation],
            self.visit(node.expression),
        )

    def visit_binary_expression(
        self, node: ASTBinaryExpression
    ) -> CoreBinaryExpression:
        return CoreBinaryExpression(
            self._ast_to_core_binary_operations[node.operation],
            self.visit(node.left),
            self.visit(node.right),
        )

    def visit_identifier_expression(
        self, node: ASTIdentifierExpression
    ) -> CoreIdentifierExpression:
        return CoreIdentifierExpression(node.identifier)

    def visit_int_literal(self, node: ASTIntLiteralExpression) -> CoreLiteralExpression:
        return CoreLiteralExpression(node.value)

    def visit_float_literal(
        self, node: ASTFloatLiteralExpression
    ) -> CoreLiteralExpression:
        return CoreLiteralExpression(node.value)

    def visit_complex_literal(
        self, node: ASTComplexLiteralExpression
    ) -> CoreLiteralExpression:
        raise NotImplementedError(
            "Complex literals are not yet supported for core expressions."
        )


def convert_ast_expression_to_core_expression(
    ast_expression: ASTExpression,
) -> CoreExpression:
    """Convert an AST expression to a core expression."""
    return ASTToCoreExpressionConverter().visit(ast_expression)
