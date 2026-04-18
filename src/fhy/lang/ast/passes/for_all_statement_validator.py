"""Validate the for-all statements in the AST."""

__all__ = [
    "validate_for_all_statements",
]

from fhy_core import (
    IndexType,
    SymbolTable,
    VariableSymbolTableFrame,
    register_pass,
)

from fhy.lang.ast.error import FhYStructuralError, FhYTypeError
from fhy.lang.ast.node import (
    ForAllStatement,
    IdentifierExpression,
    Module,
)

from .analysis_pass_with_symbol_table import AnalysisPassWithSymbolTable


@register_pass(
    "fhy_ast_for_all_statement_validator",
    "Validates the for-all statements in the AST.",
)
class _ForAllStatementValidator(AnalysisPassWithSymbolTable):
    def visit_for_all_statement(self, node: ForAllStatement) -> None:
        if not isinstance(node.index, IdentifierExpression):
            raise FhYStructuralError(
                "The index expression of a for-all statement must be an "
                f"identifier expression; got {type(node.index).__name__}."
            )
        identifier = node.index.identifier
        frame = self.get_frame_from_namespace(self.current_namespace, identifier)
        if not isinstance(frame, VariableSymbolTableFrame) or not isinstance(
            frame.type, IndexType
        ):
            raise FhYTypeError(
                f'The identifier "{identifier.name_hint!r}" used as the index of '
                "a for-all statement must refer to an index variable."
            )


def validate_for_all_statements(module: Module, symbol_table: SymbolTable) -> None:
    """Validate the for-all statements in the AST.

    Args:
        module: The module to validate.
        symbol_table: The symbol table to use.

    Raises:
        FhYStructuralError: If a for-all statement's index expression is not an
            identifier expression.
        FhYTypeError: If a forall statement's index identifier does not
            refer to an index variable via the symbol table.

    """
    validator = _ForAllStatementValidator(symbol_table)
    validator(module)
