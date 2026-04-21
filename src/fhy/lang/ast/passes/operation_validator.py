"""Validate the structural constraints on operations in the AST."""

__all__ = [
    "OperationValidator",
]

from fhy_core import (
    AnalysisVisitablePass,
    DiagnosticLevel,
    NumericalType,
    Type,
    TypeQualifier,
    register_pass,
)

from fhy.lang.ast.node import (
    Argument,
    ForAllStatement,
    Node,
    Operation,
    ReturnStatement,
    SelectionStatement,
    Statement,
)

from .utils import format_diagnostic_message


def _contains_return_statement(statements: tuple[Statement, ...]) -> bool:
    for statement in statements:
        if isinstance(statement, ReturnStatement):
            return True
        elif isinstance(statement, ForAllStatement) and _contains_return_statement(
            statement.body
        ):
            return True
        elif isinstance(statement, SelectionStatement) and (
            _contains_return_statement(statement.true_body)
            or _contains_return_statement(statement.false_body)
        ):
            return True
    return False


def _is_scalar_numerical(type_: Type) -> bool:
    return isinstance(type_, NumericalType) and type_.is_scalar()


@register_pass(
    "fhy_ast_operation_validator",
    "Validates the structural constraints on operations in the AST.",
)
class OperationValidator(AnalysisVisitablePass[Node]):
    """Validate operation argument and return-type shape constraints.

    Every argument and the return type must be a scalar numerical type
    (shape-free primitive), the return-type qualifier must be OUTPUT, and
    the body must contain at least one return statement (somewhere on any
    path).

    """

    def visit_operation(self, node: Operation) -> None:
        for argument in node.args:
            self._check_scalar_argument(node, argument)

        return_type = node.return_type
        if not _is_scalar_numerical(return_type.base_type):
            self.report(
                DiagnosticLevel.ERROR,
                format_diagnostic_message(
                    "type error",
                    f"Operation {node.name.name_hint!r} must have a scalar "
                    f"return type; got {return_type.base_type}.",
                    return_type.provenance,
                ),
            )
        if return_type.type_qualifier != TypeQualifier.OUTPUT:
            self.report(
                DiagnosticLevel.ERROR,
                format_diagnostic_message(
                    "type error",
                    f"Operation {node.name.name_hint!r} must have an OUTPUT "
                    "return type qualifier; got "
                    f"{return_type.type_qualifier.value!r}.",
                    return_type.provenance,
                ),
            )

        if not _contains_return_statement(node.body):
            self.report(
                DiagnosticLevel.ERROR,
                format_diagnostic_message(
                    "structural error",
                    f"Operation {node.name.name_hint!r} must contain a "
                    "return statement.",
                    node.provenance,
                ),
            )

    def _check_scalar_argument(self, node: Operation, argument: Argument) -> None:
        if not _is_scalar_numerical(argument.qualified_type.base_type):
            self.report(
                DiagnosticLevel.ERROR,
                format_diagnostic_message(
                    "type error",
                    f"Argument {argument.name.name_hint!r} of operation "
                    f"{node.name.name_hint!r} must have a scalar numerical "
                    f"type; got {argument.qualified_type.base_type}.",
                    argument.provenance,
                ),
            )
