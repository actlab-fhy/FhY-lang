"""Validate the left-hand side of expression statements in the AST."""

__all__ = [
    "ExpressionStatementLHSValidator",
]

from fhy_core import (
    AnalysisVisitablePass,
    DiagnosticLevel,
    register_pass,
)

from fhy.lang.ast.node import (
    ArrayAccessExpression,
    ExpressionStatement,
    FunctionExpression,
    IdentifierExpression,
    Node,
)

from .utils import format_diagnostic_message


@register_pass(
    "fhy_ast_expression_statement_lhs_validator",
    "Validates the left-hand side of expression statements in the AST.",
)
class ExpressionStatementLHSValidator(AnalysisVisitablePass[Node]):
    """Validate the shape of expression-statement left-hand sides.

    Emits an ERROR diagnostic when the LHS is an expression form that
    cannot receive an assignment (e.g., a binary expression, a non-
    identifier-backed array access). Emits a WARNING when a bare
    expression statement has no side-effecting right-hand side.

    """

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
                self.report(
                    DiagnosticLevel.ERROR,
                    format_diagnostic_message(
                        "structural error",
                        "The array expression of an array access expression "
                        "on the left-hand side of an expression statement "
                        "must be an identifier expression; got "
                        f"{type(node.left.array_expression).__name__}.",
                        node.left.provenance,
                    ),
                )
            return
        else:
            self.report(
                DiagnosticLevel.ERROR,
                format_diagnostic_message(
                    "structural error",
                    "The left-hand side of an expression statement must be "
                    "an identifier expression or an array access expression; "
                    f"got {type(node.left).__name__}.",
                    node.left.provenance,
                ),
            )
