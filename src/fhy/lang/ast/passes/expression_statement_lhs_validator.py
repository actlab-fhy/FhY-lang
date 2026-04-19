"""Validate the left-hand side of expression statements in the AST."""

__all__ = [
    "validate_expression_statement_lhs",
]

from fhy_core import (
    AnalysisVisitablePass,
    DiagnosticLevel,
    register_pass,
)

from fhy.lang.ast.error import FhYStructuralError
from fhy.lang.ast.node import (
    ArrayAccessExpression,
    ExpressionStatement,
    FunctionExpression,
    IdentifierExpression,
    Module,
    Node,
)


@register_pass(
    "fhy_ast_expression_statement_lhs_validator",
    "Validates the left-hand side of expression statements in the AST.",
)
class _ExpressionStatementLHSValidator(AnalysisVisitablePass[Node]):
    def visit_expression_statement(self, node: ExpressionStatement) -> None:
        if node.left is None:
            if not isinstance(node.right, FunctionExpression):
                location = ""
                if node.provenance is not None and node.provenance.span is not None:
                    location = f"{node.provenance.span}: "
                self.report(
                    DiagnosticLevel.WARNING,
                    f"{location}Expression statement without a left-hand side "
                    "has no effect unless its right-hand side is a function "
                    f"call; got {type(node.right).__name__}.",
                )
            return
        elif isinstance(node.left, IdentifierExpression):
            return
        elif isinstance(node.left, ArrayAccessExpression):
            if not isinstance(node.left.array_expression, IdentifierExpression):
                raise FhYStructuralError(
                    "The array expression of an array access expression on the "
                    "left-hand side of an expression statement must be an "
                    "identifier expression; got "
                    f"{type(node.left.array_expression).__name__}.",
                    node.left.provenance,
                )
            return
        else:
            raise FhYStructuralError(
                "The left-hand side of an expression statement must be an "
                "identifier expression or an array access expression; got "
                f"{type(node.left).__name__}.",
                node.left.provenance,
            )


def validate_expression_statement_lhs(module: Module) -> None:
    """Validate the left-hand side of expression statements in the AST.

    Args:
        module: The module to validate.

    Raises:
        FhYStructuralError: If an expression statement LHS validation
            error is detected.

    """
    validator = _ExpressionStatementLHSValidator()
    validator(module)
