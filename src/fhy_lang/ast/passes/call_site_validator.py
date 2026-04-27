"""Validate the function call sites in the AST."""

__all__ = [
    "CallSiteValidator",
]

import logging

from fhy_core import (
    DiagnosticLevel,
    FunctionKeyword,
    FunctionSymbolTableFrame,
    ImportSymbolTableFrame,
    SymbolTable,
    SymbolTableError,
    SymbolTableFrame,
    get_logger,
    register_pass,
)

from fhy_lang.ast.node import (
    ExpressionStatement,
    FunctionExpression,
    IdentifierExpression,
)
from fhy_lang.builtins import BUILTIN_REDUCTION_FUNCTION_IDENTIFIERS

from .analysis_pass_with_symbol_table import AnalysisPassWithSymbolTable
from .utils import format_diagnostic_message, format_location_prefix

_logger: logging.Logger = get_logger(__name__)


def _is_reduction_frame(frame: SymbolTableFrame) -> bool:
    return (
        isinstance(frame, ImportSymbolTableFrame)
        and frame.name in BUILTIN_REDUCTION_FUNCTION_IDENTIFIERS.values()
    )


def _is_procedure_frame(frame: SymbolTableFrame) -> bool:
    return (
        isinstance(frame, FunctionSymbolTableFrame)
        and frame.keyword == FunctionKeyword.PROCEDURE
    )


def _is_operation_frame(frame: SymbolTableFrame) -> bool:
    return (
        isinstance(frame, FunctionSymbolTableFrame)
        and frame.keyword == FunctionKeyword.OPERATION
    )


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

    _statement_context_call_ids: set[int]

    def __init__(self, symbol_table: SymbolTable) -> None:
        super().__init__(symbol_table)
        self._statement_context_call_ids = set()

    def before_visit_expression_statement(self, node: ExpressionStatement) -> None:
        if isinstance(node.right, FunctionExpression):
            self._statement_context_call_ids.add(id(node.right))

    def visit_function_expression(self, node: FunctionExpression) -> None:
        if not isinstance(node.function, IdentifierExpression):
            self._report_non_identifier_callee(node)
            return
        identifier = node.function.identifier
        try:
            frame = self.get_frame_from_namespace(self.current_namespace, identifier)
        except SymbolTableError:
            _logger.debug(
                "Call site: unresolved callee %s in namespace %s.",
                identifier.name_hint,
                self.current_namespace,
            )
            self._report_unresolved_callee(node, identifier.name_hint)
            return
        _logger.debug(
            "Call site: %s resolved to %s.",
            identifier.name_hint,
            type(frame).__name__,
        )
        if not isinstance(frame, FunctionSymbolTableFrame | ImportSymbolTableFrame):
            self._report_unresolved_callee(node, identifier.name_hint)
            return
        if not _is_reduction_frame(frame) and len(node.indices) > 0:
            self._report_non_reduction_indices(node, identifier.name_hint)
        if isinstance(frame, FunctionSymbolTableFrame):
            self._check_function_argument_count(node, identifier.name_hint, frame)
            self._check_procedure_value_position(node, identifier.name_hint, frame)

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
        if _is_procedure_frame(frame) and node.left is not None:
            self._report_procedure_with_left_hand_side(node, identifier.name_hint)
        if _is_operation_frame(frame) and node.left is None:
            self._warn_operation_return_discarded(node, identifier.name_hint)

    def _check_function_argument_count(
        self,
        node: FunctionExpression,
        name_hint: str,
        frame: FunctionSymbolTableFrame,
    ) -> None:
        if len(node.args) == len(frame.signature):
            return
        self.report(
            DiagnosticLevel.ERROR,
            format_diagnostic_message(
                "structural error",
                f"Function {name_hint!r} expects {len(frame.signature)} "
                f"argument(s); got {len(node.args)}.",
                node.provenance,
            ),
        )

    def _check_procedure_value_position(
        self,
        node: FunctionExpression,
        name_hint: str,
        frame: FunctionSymbolTableFrame,
    ) -> None:
        if frame.keyword != FunctionKeyword.PROCEDURE:
            return
        if id(node) in self._statement_context_call_ids:
            return
        self.report(
            DiagnosticLevel.ERROR,
            format_diagnostic_message(
                "structural error",
                f"Procedure {name_hint!r} can only be called as a bare "
                "expression statement; it cannot appear in a value position "
                "(e.g., inside a binary, ternary, array-access, or other "
                "expression).",
                node.provenance,
            ),
        )

    def _report_non_identifier_callee(self, node: FunctionExpression) -> None:
        self.report(
            DiagnosticLevel.ERROR,
            format_diagnostic_message(
                "structural error",
                "The expression passed as the function name of a call must "
                "be an identifier expression; got "
                f"{type(node.function).__name__}.",
                node.function.provenance,
            ),
        )

    def _report_unresolved_callee(
        self, node: FunctionExpression, name_hint: str
    ) -> None:
        self.report(
            DiagnosticLevel.ERROR,
            format_diagnostic_message(
                "semantic error",
                f"{name_hint!r} is not defined as a function.",
                node.function.provenance,
            ),
        )

    def _report_non_reduction_indices(
        self, node: FunctionExpression, name_hint: str
    ) -> None:
        self.report(
            DiagnosticLevel.ERROR,
            format_diagnostic_message(
                "structural error",
                f"Non-reduction function {name_hint!r} cannot be called with indices.",
                node.provenance,
            ),
        )

    def _report_procedure_with_left_hand_side(
        self, node: ExpressionStatement, name_hint: str
    ) -> None:
        self.report(
            DiagnosticLevel.ERROR,
            format_diagnostic_message(
                "structural error",
                f"Procedure {name_hint!r} cannot be called with a left-hand "
                "side expression.",
                node.provenance,
            ),
        )

    def _warn_operation_return_discarded(
        self, node: ExpressionStatement, name_hint: str
    ) -> None:
        location = format_location_prefix(node.provenance)
        self.report(
            DiagnosticLevel.WARNING,
            f"{location}Operation {name_hint!r} is called as a bare "
            "statement; its return value is discarded.",
        )
