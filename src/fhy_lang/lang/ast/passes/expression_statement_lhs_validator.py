"""Validate the left-hand side of expression statements in the AST."""

__all__ = [
    "ExpressionStatementLHSValidator",
]

from fhy_core import (
    AnalysisVisitablePass,
    DiagnosticLevel,
    register_pass,
)

from fhy_lang.lang.ast.node import (
    ArrayAccessExpression,
    ExpressionStatement,
    FunctionExpression,
    IdentifierExpression,
    Node,
)

from .utils import format_diagnostic_message, format_location_prefix


@register_pass(
    "fhy_ast_expression_statement_lhs_validator",
    "Validates the left-hand side of expression statements in the AST.",
)
class ExpressionStatementLHSValidator(AnalysisVisitablePass[Node]):
    """Validate the shape of expression-statement left-hand sides.

    Emits an ERROR diagnostic when the LHS is an expression form that
    cannot receive an assignment (e.g., a binary expression, a
    non-identifier-backed array access). Emits a WARNING when a bare
    expression statement has no side-effecting right-hand side.

    """

    def visit_expression_statement(self, node: ExpressionStatement) -> None:
        if node.left is None:
            self._check_bare_expression_statement(node)
        elif isinstance(node.left, IdentifierExpression):
            return
        elif isinstance(node.left, ArrayAccessExpression):
            self._check_array_access_lhs(node)
        else:
            self._report_unsupported_lhs(node)

    def _check_bare_expression_statement(self, node: ExpressionStatement) -> None:
        if isinstance(node.right, FunctionExpression):
            return
        location = format_location_prefix(node.provenance)
        self.report(
            DiagnosticLevel.WARNING,
            f"{location}Expression statement without a left-hand side has no "
            "effect unless its right-hand side is a function call; got "
            f"{type(node.right).__name__}.",
        )

    def _check_array_access_lhs(self, node: ExpressionStatement) -> None:
        assert isinstance(node.left, ArrayAccessExpression)
        if isinstance(node.left.array_expression, IdentifierExpression):
            return
        self.report(
            DiagnosticLevel.ERROR,
            format_diagnostic_message(
                "structural error",
                "The array expression of an array access expression on the "
                "left-hand side of an expression statement must be an "
                "identifier expression; got "
                f"{type(node.left.array_expression).__name__}.",
                node.left.provenance,
            ),
        )

    def _report_unsupported_lhs(self, node: ExpressionStatement) -> None:
        assert node.left is not None
        self.report(
            DiagnosticLevel.ERROR,
            format_diagnostic_message(
                "structural error",
                "The left-hand side of an expression statement must be an "
                "identifier expression or an array access expression; got "
                f"{type(node.left).__name__}.",
                node.left.provenance,
            ),
        )
