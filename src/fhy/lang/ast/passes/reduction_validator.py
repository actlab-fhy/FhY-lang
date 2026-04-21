"""Validate the reductions in the AST."""

__all__ = [
    "ReductionValidator",
]

from fhy_core import (
    DiagnosticLevel,
    Identifier,
    IndexType,
    SymbolTableError,
    VariableSymbolTableFrame,
    register_pass,
)

from fhy.lang.ast.node import (
    FunctionExpression,
    IdentifierExpression,
)

from .analysis_pass_with_symbol_table import AnalysisPassWithSymbolTable
from .identifier_collector import collect_identifiers
from .utils import format_diagnostic_message


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

    def visit_function_expression(self, node: FunctionExpression) -> None:  # noqa: C901
        if len(node.indices) > 0 and len(node.args) != 1:
            self.report(
                DiagnosticLevel.ERROR,
                format_diagnostic_message(
                    "structural error",
                    "A reduction must be passed exactly one argument; got "
                    f"{len(node.args)}.",
                    node.provenance,
                ),
            )
        seen_indices: set[Identifier] = set()
        for index in node.indices:
            if not isinstance(index, IdentifierExpression):
                self.report(
                    DiagnosticLevel.ERROR,
                    format_diagnostic_message(
                        "structural error",
                        "Each index passed to a reduction must be an "
                        f"identifier expression; got {type(index).__name__}.",
                        index.provenance,
                    ),
                )
                continue
            identifier = index.identifier
            try:
                frame = self.get_frame_from_namespace(
                    self.current_namespace, identifier
                )
            except SymbolTableError:
                continue
            if not isinstance(frame, VariableSymbolTableFrame) or not isinstance(
                frame.type, IndexType
            ):
                self.report(
                    DiagnosticLevel.ERROR,
                    format_diagnostic_message(
                        "type error",
                        f'The identifier "{identifier.name_hint!r}" passed '
                        "as an index to a reduction must refer to an index "
                        "variable.",
                        index.provenance,
                    ),
                )
                continue
            if identifier in seen_indices:
                self.report(
                    DiagnosticLevel.ERROR,
                    format_diagnostic_message(
                        "semantic error",
                        f'The identifier "{identifier.name_hint!r}" is '
                        "passed more than once as an index to a reduction; "
                        "reduction indices must be distinct.",
                        index.provenance,
                    ),
                )
                continue
            seen_indices.add(identifier)

        if not seen_indices:
            return
        used_identifiers: set[Identifier] = set()
        for arg in node.args:
            used_identifiers.update(collect_identifiers(arg))
        for identifier in seen_indices:
            if identifier not in used_identifiers:
                self.report(
                    DiagnosticLevel.ERROR,
                    format_diagnostic_message(
                        "semantic error",
                        f'The identifier "{identifier.name_hint!r}" is '
                        "passed as an index to a reduction but is not used "
                        "within the reduction.",
                        node.provenance,
                    ),
                )
