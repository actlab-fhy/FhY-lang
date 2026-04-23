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

_TYPE_QUALIFIER_ERROR_KIND = "type qualifier error"


def _get_assignment_target_identifier(
    expression: Expression,
) -> Identifier | None:
    if isinstance(expression, IdentifierExpression):
        return expression.identifier
    elif isinstance(expression, ArrayAccessExpression):
        return _get_assignment_target_identifier(expression.array_expression)
    else:
        return None


@register_pass(
    "fhy_ast_type_qualifier_validator",
    "Validates the type qualifiers of the AST.",
)
class TypeQualifierValidator(AnalysisPassWithSymbolTable):
    """Validate FhY type-qualifier placement rules.

    - ``INPUT`` appears only in argument lists (read-only at use sites).
    - ``TEMP`` appears only in declaration statements.
    - ``OUTPUT`` appears only in argument lists or return types.
    - ``PARAM`` are compile-time constants (cannot be assigned).
    - Operation return types must be qualified ``OUTPUT``.

    """

    def visit_argument(self, node: Argument) -> None:
        qualifier = node.qualified_type.type_qualifier
        if qualifier != TypeQualifier.TEMP:
            return
        self.report(
            DiagnosticLevel.ERROR,
            format_diagnostic_message(
                _TYPE_QUALIFIER_ERROR_KIND,
                f"Argument {node.name.name_hint!r} cannot have type qualifier "
                f"{TypeQualifier.TEMP.value!r}; TEMP variables may only be "
                "defined in declaration statements.",
                node.provenance,
            ),
        )

    def visit_declaration_statement(self, node: DeclarationStatement) -> None:
        qualifier = node.variable_type.type_qualifier
        if qualifier == TypeQualifier.INPUT:
            self._report_invalid_declaration_qualifier(
                node,
                TypeQualifier.INPUT,
                "INPUT variables may only be defined in argument lists.",
            )
        elif qualifier == TypeQualifier.OUTPUT:
            self._report_invalid_declaration_qualifier(
                node,
                TypeQualifier.OUTPUT,
                "OUTPUT variables may only be defined in argument lists or "
                "return types.",
            )

    def visit_operation(self, node: Operation) -> None:
        qualifier = node.return_type.type_qualifier
        if qualifier == TypeQualifier.OUTPUT:
            return
        self.report(
            DiagnosticLevel.ERROR,
            format_diagnostic_message(
                _TYPE_QUALIFIER_ERROR_KIND,
                f"Return type of operation {node.name.name_hint!r} must have "
                f"type qualifier {TypeQualifier.OUTPUT.value!r}; got "
                f"{qualifier.value!r}.",
                node.return_type.provenance,
            ),
        )

    def visit_expression_statement(self, node: ExpressionStatement) -> None:
        if node.left is None:
            return
        target = _get_assignment_target_identifier(node.left)
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
            self._report_assignment_to_read_only(
                node, target, "INPUT variables are read-only."
            )
        elif qualifier == TypeQualifier.PARAM:
            self._report_assignment_to_read_only(
                node, target, "PARAM variables are compile-time constants."
            )

    def _report_invalid_declaration_qualifier(
        self,
        node: DeclarationStatement,
        qualifier: TypeQualifier,
        reason: str,
    ) -> None:
        self.report(
            DiagnosticLevel.ERROR,
            format_diagnostic_message(
                _TYPE_QUALIFIER_ERROR_KIND,
                f"Declaration of {node.variable_name.name_hint!r} cannot have "
                f"type qualifier {qualifier.value!r}; {reason}",
                node.provenance,
            ),
        )

    def _report_assignment_to_read_only(
        self,
        node: ExpressionStatement,
        target: Identifier,
        reason: str,
    ) -> None:
        self.report(
            DiagnosticLevel.ERROR,
            format_diagnostic_message(
                _TYPE_QUALIFIER_ERROR_KIND,
                f"Cannot assign to {target.name_hint!r}; {reason}",
                node.provenance,
            ),
        )
