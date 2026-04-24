"""Validate the reductions in the AST."""

__all__ = [
    "ReductionValidator",
]

import logging

from fhy_core import (
    DiagnosticLevel,
    Identifier,
    IndexType,
    SymbolTableError,
    SymbolTableFrame,
    VariableSymbolTableFrame,
    get_logger,
    register_pass,
)

from fhy_lang.lang.ast.node import (
    Expression,
    FunctionExpression,
    IdentifierExpression,
)

from .analysis_pass_with_symbol_table import AnalysisPassWithSymbolTable
from .identifier_collector import collect_identifiers
from .utils import format_diagnostic_message

_logger: logging.Logger = get_logger(__name__)


def _is_index_variable_frame(frame: SymbolTableFrame) -> bool:
    return isinstance(frame, VariableSymbolTableFrame) and isinstance(
        frame.type, IndexType
    )


@register_pass(
    "fhy_ast_reduction_validator",
    "Validates the reductions in the AST.",
)
class ReductionValidator(AnalysisPassWithSymbolTable):
    """Validate reduction call sites.

    Enforces the shape constraints on reduction calls: exactly one argument,
    identifier-only indices, index identifiers that resolve to index
    variables, index uniqueness, and that every declared index is actually
    used inside the reduction argument.

    """

    def visit_function_expression(self, node: FunctionExpression) -> None:
        _logger.debug(
            "Reduction check: %d index/indices, %d arg(s).",
            len(node.indices),
            len(node.args),
        )
        if len(node.indices) > 0 and len(node.args) != 1:
            self._report_incorrect_argument_count(node)
        seen_indices = self._collect_and_validate_indices(node)
        if not seen_indices:
            return
        self._check_all_indices_are_used(node, seen_indices)

    def _collect_and_validate_indices(
        self, node: FunctionExpression
    ) -> set[Identifier]:
        seen_indices: set[Identifier] = set()
        for index in node.indices:
            identifier = self._validate_single_index(index, seen_indices)
            if identifier is not None:
                seen_indices.add(identifier)
        return seen_indices

    def _validate_single_index(
        self, index: Expression, seen_indices: set[Identifier]
    ) -> Identifier | None:
        if not isinstance(index, IdentifierExpression):
            self._report_non_identifier_index(index)
            return None
        identifier = index.identifier
        try:
            frame = self.get_frame_from_namespace(self.current_namespace, identifier)
        except SymbolTableError:
            return None
        if not _is_index_variable_frame(frame):
            self._report_non_index_variable(index, identifier.name_hint)
            return None
        if identifier in seen_indices:
            self._report_duplicate_index(index, identifier.name_hint)
            return None
        return identifier

    def _check_all_indices_are_used(
        self, node: FunctionExpression, seen_indices: set[Identifier]
    ) -> None:
        used_identifiers: set[Identifier] = set()
        for argument in node.args:
            used_identifiers.update(collect_identifiers(argument))
        for identifier in seen_indices:
            if identifier in used_identifiers:
                continue
            self._report_unused_index(node, identifier.name_hint)

    def _report_incorrect_argument_count(self, node: FunctionExpression) -> None:
        self.report(
            DiagnosticLevel.ERROR,
            format_diagnostic_message(
                "structural error",
                "A reduction must be passed exactly one argument; got "
                f"{len(node.args)}.",
                node.provenance,
            ),
        )

    def _report_non_identifier_index(self, index: Expression) -> None:
        self.report(
            DiagnosticLevel.ERROR,
            format_diagnostic_message(
                "structural error",
                "Each index passed to a reduction must be an identifier "
                f"expression; got {type(index).__name__}.",
                index.provenance,
            ),
        )

    def _report_non_index_variable(self, index: Expression, name_hint: str) -> None:
        self.report(
            DiagnosticLevel.ERROR,
            format_diagnostic_message(
                "type error",
                f"The identifier {name_hint!r} passed as an index to a "
                "reduction must refer to an index variable.",
                index.provenance,
            ),
        )

    def _report_duplicate_index(self, index: Expression, name_hint: str) -> None:
        self.report(
            DiagnosticLevel.ERROR,
            format_diagnostic_message(
                "semantic error",
                f"The identifier {name_hint!r} is passed more than once as "
                "an index to a reduction; reduction indices must be distinct.",
                index.provenance,
            ),
        )

    def _report_unused_index(self, node: FunctionExpression, name_hint: str) -> None:
        self.report(
            DiagnosticLevel.ERROR,
            format_diagnostic_message(
                "semantic error",
                f"The identifier {name_hint!r} is passed as an index to a "
                "reduction but is not used within the reduction.",
                node.provenance,
            ),
        )
