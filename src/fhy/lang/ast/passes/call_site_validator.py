"""Validate the function call sites in the AST."""

__all__ = [
    "CallSiteValidator",
]

from fhy_core import (
    DiagnosticLevel,
    FunctionKeyword,
    FunctionSymbolTableFrame,
    ImportSymbolTableFrame,
    SymbolTable,
    SymbolTableError,
    register_pass,
)

from fhy.lang.ast.node import (
    ExpressionStatement,
    FunctionExpression,
    IdentifierExpression,
)
from fhy.lang.builtins import BUILTIN_REDUCTION_FUNCTION_IDENTIFIERS

from .analysis_pass_with_symbol_table import AnalysisPassWithSymbolTable
from .utils import format_diagnostic_message


@register_pass(
    "fhy_ast_call_site_validator",
    "Validates the function call sites in the AST.",
)
class CallSiteValidator(AnalysisPassWithSymbolTable):
    """Validate structural and resolution constraints on call sites.

    Emits ERROR diagnostics for:
      - A non-identifier function name expression.
      - A function name that does not resolve to a function frame.
      - A non-reduction call that carries indices.
      - A user-function call with the wrong argument count.
      - A procedure appearing in a value position.
      - A procedure call with a left-hand side.

    Emits a WARNING when an operation is called as a bare expression
    statement (its return value is discarded).

    """

    _statement_context_calls: set[int]

    def __init__(self, symbol_table: SymbolTable) -> None:
        super().__init__(symbol_table)
        self._statement_context_calls = set()

    def before_visit_expression_statement(self, node: ExpressionStatement) -> None:
        if isinstance(node.right, FunctionExpression):
            self._statement_context_calls.add(id(node.right))

    def visit_function_expression(self, node: FunctionExpression) -> None:
        if not isinstance(node.function, IdentifierExpression):
            self.report(
                DiagnosticLevel.ERROR,
                format_diagnostic_message(
                    "structural error",
                    "The expression passed as the function name of a call "
                    "must be an identifier expression; got "
                    f"{type(node.function).__name__}.",
                    node.function.provenance,
                ),
            )
            return
        identifier = node.function.identifier
        try:
            frame = self.get_frame_from_namespace(self.current_namespace, identifier)
        except SymbolTableError:
            self.report(
                DiagnosticLevel.ERROR,
                format_diagnostic_message(
                    "semantic error",
                    f"{identifier.name_hint!r} is not defined as a function.",
                    node.function.provenance,
                ),
            )
            return
        if not isinstance(frame, FunctionSymbolTableFrame | ImportSymbolTableFrame):
            self.report(
                DiagnosticLevel.ERROR,
                format_diagnostic_message(
                    "semantic error",
                    f"{identifier.name_hint!r} is not defined as a function.",
                    node.function.provenance,
                ),
            )
            return
        is_reduction = (
            isinstance(frame, ImportSymbolTableFrame)
            and frame.name in BUILTIN_REDUCTION_FUNCTION_IDENTIFIERS.values()
        )
        if not is_reduction and len(node.indices) > 0:
            self.report(
                DiagnosticLevel.ERROR,
                format_diagnostic_message(
                    "structural error",
                    f"Non-reduction function {identifier.name_hint!r} cannot "
                    "be called with indices.",
                    node.provenance,
                ),
            )
        if isinstance(frame, FunctionSymbolTableFrame) and len(node.args) != len(
            frame.signature
        ):
            self.report(
                DiagnosticLevel.ERROR,
                format_diagnostic_message(
                    "structural error",
                    f"Function {identifier.name_hint!r} expects "
                    f"{len(frame.signature)} argument(s); got "
                    f"{len(node.args)}.",
                    node.provenance,
                ),
            )
        if (
            isinstance(frame, FunctionSymbolTableFrame)
            and frame.keyword == FunctionKeyword.PROCEDURE
            and id(node) not in self._statement_context_calls
        ):
            self.report(
                DiagnosticLevel.ERROR,
                format_diagnostic_message(
                    "structural error",
                    f"Procedure {identifier.name_hint!r} can only be called "
                    "as a bare expression statement; it cannot appear in a "
                    "value position (e.g., inside a binary, ternary, array-"
                    "access, or other expression).",
                    node.provenance,
                ),
            )

    def visit_expression_statement(self, node: ExpressionStatement) -> None:
        if not isinstance(node.right, FunctionExpression):
            return
        if not isinstance(node.right.function, IdentifierExpression):
            return
        identifier = node.right.function.identifier
        try:
            frame = self.get_frame_from_namespace(self.current_namespace, identifier)
        except SymbolTableError:
            return
        if (
            isinstance(frame, FunctionSymbolTableFrame)
            and frame.keyword == FunctionKeyword.PROCEDURE
            and node.left is not None
        ):
            self.report(
                DiagnosticLevel.ERROR,
                format_diagnostic_message(
                    "structural error",
                    f"Procedure {identifier.name_hint!r} cannot be called "
                    "with a left-hand side expression.",
                    node.provenance,
                ),
            )
        if (
            isinstance(frame, FunctionSymbolTableFrame)
            and frame.keyword == FunctionKeyword.OPERATION
            and node.left is None
        ):
            location = ""
            if node.provenance is not None and node.provenance.span is not None:
                location = f"{node.provenance.span}: "
            self.report(
                DiagnosticLevel.WARNING,
                f"{location}Operation {identifier.name_hint!r} is called as "
                "a bare statement; its return value is discarded.",
            )
