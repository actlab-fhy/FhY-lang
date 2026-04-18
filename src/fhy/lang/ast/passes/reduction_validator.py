"""Validate the reductions in the AST."""

__all__ = [
    "validate_reductions",
]

from fhy_core import (
    Identifier,
    IndexType,
    SymbolTable,
    VariableSymbolTableFrame,
    register_pass,
)

from fhy.lang.ast.error import FhYSemanticsError, FhYStructuralError, FhYTypeError
from fhy.lang.ast.node import (
    FunctionExpression,
    IdentifierExpression,
    Module,
)

from .analysis_pass_with_symbol_table import AnalysisPassWithSymbolTable
from .identifier_collector import collect_identifiers


@register_pass(
    "fhy_ast_reduction_validator",
    "Validates the reductions in the AST.",
)
class _ReductionValidator(AnalysisPassWithSymbolTable):
    def visit_function_expression(self, node: FunctionExpression) -> None:
        if len(node.indices) > 0 and len(node.args) != 1:
            raise FhYStructuralError(
                "A reduction must be passed exactly one argument; got "
                f"{len(node.args)}."
            )
        seen_indices = set()
        for index in node.indices:
            if not isinstance(index, IdentifierExpression):
                raise FhYStructuralError(
                    "Each index passed to a reduction must be an identifier "
                    f"expression; got {type(index).__name__}."
                )
            identifier = index.identifier
            frame = self.get_frame_from_namespace(self.current_namespace, identifier)
            if not isinstance(frame, VariableSymbolTableFrame) or not isinstance(
                frame.type, IndexType
            ):
                raise FhYTypeError(
                    f'The identifier "{identifier.name_hint!r}" passed as an '
                    "index to a reduction must refer to an index variable."
                )
            if identifier in seen_indices:
                raise FhYSemanticsError(
                    f'The identifier "{identifier.name_hint!r}" is passed more '
                    "than once as an index to a reduction; reduction indices "
                    "must be distinct."
                )
            seen_indices.add(identifier)

        if not seen_indices:
            return
        used_identifiers: set[Identifier] = set()
        for arg in node.args:
            used_identifiers.update(collect_identifiers(arg))
        for identifier in seen_indices:
            if identifier not in used_identifiers:
                raise FhYSemanticsError(
                    f'The identifier "{identifier.name_hint!r}" is passed as '
                    "an index to a reduction but is not used within the "
                    "reduction."
                )


def validate_reductions(module: Module, symbol_table: SymbolTable) -> None:
    """Validate the reductions in the AST.

    Args:
        module: The module to validate.
        symbol_table: The symbol table to use.

    Raises:
        FhYStructuralError: If an expression passed as a reduction index is not
            an identifier expression, or if a reduction is not passed exactly
            one argument.
        FhYTypeError: If a reduction index identifier does not refer to an
            index variable via the symbol table.
        FhYSemanticsError: If the reduction indices are not distinct, or if a
            reduction index is not used within the reduction argument.

    """
    validator = _ReductionValidator(symbol_table)
    validator(module)
