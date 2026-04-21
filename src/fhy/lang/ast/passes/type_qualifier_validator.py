"""Validate the type qualifiers of the AST."""

__all__ = [
    "TypeQualifierValidator",
]

from fhy_core import (
    DiagnosticLevel,
    Identifier,
    SymbolTableError,
    TypeQualifier,
    VariableSymbolTableFrame,
    register_pass,
)

from fhy.lang.ast.node import (
    Argument,
    ArrayAccessExpression,
    DeclarationStatement,
    Expression,
    ExpressionStatement,
    IdentifierExpression,
    Operation,
)

from .analysis_pass_with_symbol_table import AnalysisPassWithSymbolTable
from .utils import format_diagnostic_message

_KIND = "type qualifier error"


@register_pass(
    "fhy_ast_type_qualifier_validator",
    "Validates the type qualifiers of the AST.",
)
class TypeQualifierValidator(AnalysisPassWithSymbolTable):
    """Validate FhY type-qualifier placement rules.

    - INPUTs appear only in argument lists (read-only at use sites).
    - TEMPs appear only in declaration statements.
    - OUTPUTs appear only in argument lists or return types.
    - PARAMs are compile-time constants (cannot be assigned).
    - Operation return types must be qualified OUTPUT.

    """

    def visit_argument(self, node: Argument) -> None:
        qualifier = node.qualified_type.type_qualifier
        if qualifier == TypeQualifier.TEMP:
            self.report(
                DiagnosticLevel.ERROR,
                format_diagnostic_message(
                    _KIND,
                    f"Argument {node.name.name_hint!r} cannot have type "
                    f"qualifier {TypeQualifier.TEMP.value!r}; TEMP variables "
                    "may only be defined in declaration statements.",
                    node.provenance,
                ),
            )

    def visit_declaration_statement(self, node: DeclarationStatement) -> None:
        qualifier = node.variable_type.type_qualifier
        if qualifier == TypeQualifier.INPUT:
            self.report(
                DiagnosticLevel.ERROR,
                format_diagnostic_message(
                    _KIND,
                    f"Declaration of {node.variable_name.name_hint!r} cannot "
                    f"have type qualifier {TypeQualifier.INPUT.value!r}; "
                    "INPUT variables may only be defined in argument lists.",
                    node.provenance,
                ),
            )
        if qualifier == TypeQualifier.OUTPUT:
            self.report(
                DiagnosticLevel.ERROR,
                format_diagnostic_message(
                    _KIND,
                    f"Declaration of {node.variable_name.name_hint!r} cannot "
                    f"have type qualifier {TypeQualifier.OUTPUT.value!r}; "
                    "OUTPUT variables may only be defined in argument lists "
                    "or return types.",
                    node.provenance,
                ),
            )

    def visit_operation(self, node: Operation) -> None:
        qualifier = node.return_type.type_qualifier
        if qualifier != TypeQualifier.OUTPUT:
            self.report(
                DiagnosticLevel.ERROR,
                format_diagnostic_message(
                    _KIND,
                    f"Return type of operation {node.name.name_hint!r} must "
                    f"have type qualifier {TypeQualifier.OUTPUT.value!r}; "
                    f"got {qualifier.value!r}.",
                    node.return_type.provenance,
                ),
            )

    def visit_expression_statement(self, node: ExpressionStatement) -> None:
        if node.left is None:
            return
        target = self._get_assignment_target(node.left)
        if target is None:
            return
        try:
            frame = self.get_frame_from_namespace(self.current_namespace, target)
        except SymbolTableError:
            return
        if not isinstance(frame, VariableSymbolTableFrame):
            return
        qualifier = frame.type_qualifier
        if qualifier == TypeQualifier.INPUT:
            self.report(
                DiagnosticLevel.ERROR,
                format_diagnostic_message(
                    _KIND,
                    f"Cannot assign to {target.name_hint!r}; INPUT variables "
                    "are read-only.",
                    node.provenance,
                ),
            )
        if qualifier == TypeQualifier.PARAM:
            self.report(
                DiagnosticLevel.ERROR,
                format_diagnostic_message(
                    _KIND,
                    f"Cannot assign to {target.name_hint!r}; PARAM variables "
                    "are compile-time constants.",
                    node.provenance,
                ),
            )

    def _get_assignment_target(self, expression: Expression) -> Identifier | None:
        if isinstance(expression, IdentifierExpression):
            return expression.identifier
        elif isinstance(expression, ArrayAccessExpression):
            return self._get_assignment_target(expression.array_expression)
        else:
            return None
