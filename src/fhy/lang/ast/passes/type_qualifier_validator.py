"""Validate the type qualifiers of the AST."""

__all__ = [
    "FhYTypeQualifierValidatorError",
    "validate_type_qualifiers",
]

from fhy_core import AnalysisVisitablePass, SymbolTable, register_error, register_pass

from fhy.lang.ast.error import FhYTypeError
from fhy.lang.ast.node import Module, Node


@register_error
class FhYTypeQualifierValidatorError(FhYTypeError):
    """Raised when a type qualifier validation error is detected."""


@register_pass(
    "fhy_ast_type_qualifier_validator",
    "Validates the type qualifiers of the AST.",
)
class _TypeQualifierValidator(AnalysisVisitablePass[Node]):
    _symbol_table: SymbolTable

    def __init__(self, symbol_table: SymbolTable) -> None:
        super().__init__()
        self._symbol_table = symbol_table


def validate_type_qualifiers(module: Module, symbol_table: SymbolTable) -> None:
    """Validate the type qualifiers of the AST.

    Args:
        module: The module to validate.
        symbol_table: The symbol table to use.

    Raises:
        FhYTypeQualifierValidatorError: If a type qualifier use error is detected.

    """
    validator = _TypeQualifierValidator(symbol_table)
    validator(module)
