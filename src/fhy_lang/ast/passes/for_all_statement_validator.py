"""Validate the for-all statements in the AST."""

__all__ = [
    "ForAllStatementValidator",
]

import logging

from fhy_core import (
    DiagnosticLevel,
    IndexType,
    SymbolTableError,
    VariableSymbolTableFrame,
    get_logger,
    register_pass,
)

from fhy_lang.ast.node import (
    ForAllStatement,
    IdentifierExpression,
)

from .analysis_pass_with_symbol_table import AnalysisPassWithSymbolTable
from .utils import format_diagnostic_message

_logger: logging.Logger = get_logger(__name__)


@register_pass(
    "fhy_ast_for_all_statement_validator",
    "Validates the for-all statements in the AST.",
)
class ForAllStatementValidator(AnalysisPassWithSymbolTable):
    """Validate for-all statement indices.

    The index expression must be an :class:`IdentifierExpression` whose
    identifier resolves in the symbol table to an index-typed variable.

    """

    def visit_for_all_statement(self, node: ForAllStatement) -> None:
        _logger.debug("ForAll check: index=%s.", type(node.index).__name__)
        if not isinstance(node.index, IdentifierExpression):
            self._report_non_identifier_index(node)
            return
        identifier = node.index.identifier
        try:
            frame = self.get_frame_from_namespace(self.current_namespace, identifier)
        except SymbolTableError:
            # Leave unresolved-identifier reporting to a later validator; do
            # not emit a duplicate here.
            return
        if not isinstance(frame, VariableSymbolTableFrame) or not isinstance(
            frame.type, IndexType
        ):
            self._report_non_index_typed_identifier(node, identifier.name_hint)

    def _report_non_identifier_index(self, node: ForAllStatement) -> None:
        self.report(
            DiagnosticLevel.ERROR,
            format_diagnostic_message(
                "structural error",
                "The index expression of a for-all statement must be an "
                f"identifier expression; got {type(node.index).__name__}.",
                node.index.provenance,
            ),
        )

    def _report_non_index_typed_identifier(
        self, node: ForAllStatement, name_hint: str
    ) -> None:
        self.report(
            DiagnosticLevel.ERROR,
            format_diagnostic_message(
                "type error",
                f"The identifier {name_hint!r} used as the index of a "
                "for-all statement must refer to an index variable.",
                node.index.provenance,
            ),
        )
