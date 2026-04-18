"""Validate the type qualifiers of the AST."""

__all__ = [
    "FhYTypeQualifierValidatorError",
    "validate_type_qualifiers",
]

from fhy_core import (
    Identifier,
    SymbolTable,
    TypeQualifier,
    VariableSymbolTableFrame,
    register_error,
    register_pass,
)

from fhy.lang.ast.error import FhYTypeError
from fhy.lang.ast.node import (
    Argument,
    ArrayAccessExpression,
    DeclarationStatement,
    Expression,
    ExpressionStatement,
    IdentifierExpression,
    Module,
    Operation,
)

from .analysis_pass_with_symbol_table import AnalysisPassWithSymbolTable


@register_error
class FhYTypeQualifierValidatorError(FhYTypeError):
    """Raised when a type qualifier validation error is detected."""


@register_pass(
    "fhy_ast_type_qualifier_validator",
    "Validates the type qualifiers of the AST.",
)
class _TypeQualifierValidator(AnalysisPassWithSymbolTable):
    def visit_argument(self, node: Argument) -> None:
        qualifier = node.qualified_type.type_qualifier
        if qualifier == TypeQualifier.TEMP:
            raise FhYTypeQualifierValidatorError(
                f"Argument {node.name.name_hint!r} cannot have type qualifier "
                f"{TypeQualifier.TEMP.value!r}; TEMP variables may only be "
                "defined in declaration statements.",
                node.provenance,
            )

    def visit_declaration_statement(self, node: DeclarationStatement) -> None:
        qualifier = node.variable_type.type_qualifier
        if qualifier == TypeQualifier.INPUT:
            raise FhYTypeQualifierValidatorError(
                f"Declaration of {node.variable_name.name_hint!r} cannot have "
                f"type qualifier {TypeQualifier.INPUT.value!r}; INPUT variables "
                "may only be defined in argument lists.",
                node.provenance,
            )
        if qualifier == TypeQualifier.OUTPUT:
            raise FhYTypeQualifierValidatorError(
                f"Declaration of {node.variable_name.name_hint!r} cannot have "
                f"type qualifier {TypeQualifier.OUTPUT.value!r}; OUTPUT "
                "variables may only be defined in argument lists or return "
                "types.",
                node.provenance,
            )

    def visit_operation(self, node: Operation) -> None:
        qualifier = node.return_type.type_qualifier
        if qualifier != TypeQualifier.OUTPUT:
            raise FhYTypeQualifierValidatorError(
                f"Return type of operation {node.name.name_hint!r} must have "
                f"type qualifier {TypeQualifier.OUTPUT.value!r}; got "
                f"{qualifier.value!r}.",
                node.return_type.provenance,
            )

    def visit_expression_statement(self, node: ExpressionStatement) -> None:
        if node.left is None:
            return
        target = self._get_assignment_target(node.left)
        if target is None:
            return
        frame = self.get_frame_from_namespace(self.current_namespace, target)
        if not isinstance(frame, VariableSymbolTableFrame):
            return
        qualifier = frame.type_qualifier
        if qualifier == TypeQualifier.INPUT:
            raise FhYTypeQualifierValidatorError(
                f"Cannot assign to {target.name_hint!r}; INPUT variables are "
                "read-only.",
                node.provenance,
            )
        if qualifier == TypeQualifier.PARAM:
            raise FhYTypeQualifierValidatorError(
                f"Cannot assign to {target.name_hint!r}; PARAM variables are "
                "compile-time constants.",
                node.provenance,
            )

    def _get_assignment_target(self, expression: Expression) -> Identifier | None:
        if isinstance(expression, IdentifierExpression):
            return expression.identifier
        elif isinstance(expression, ArrayAccessExpression):
            return self._get_assignment_target(expression.array_expression)
        else:
            return None


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
