"""Validate the structural constraints on operations in the AST."""

__all__ = [
    "validate_operations",
]

from fhy_core import (
    AnalysisVisitablePass,
    NumericalType,
    TypeQualifier,
    register_pass,
)

from fhy.lang.ast.error import FhYStructuralError, FhYTypeError
from fhy.lang.ast.node import (
    Argument,
    ForAllStatement,
    Module,
    Node,
    Operation,
    ReturnStatement,
    SelectionStatement,
    Statement,
)


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


def _is_scalar_numerical(type_) -> bool:
    return isinstance(type_, NumericalType) and type_.is_scalar()


@register_pass(
    "fhy_ast_operation_validator",
    "Validates the structural constraints on operations in the AST.",
)
class _OperationValidator(AnalysisVisitablePass[Node]):
    def visit_operation(self, node: Operation) -> None:
        for argument in node.args:
            self._check_scalar_argument(node, argument)

        return_type = node.return_type
        if not _is_scalar_numerical(return_type.base_type):
            raise FhYTypeError(
                f"Operation {node.name.name_hint!r} must have a scalar "
                f"return type; got {return_type.base_type}.",
                return_type.provenance,
            )
        if return_type.type_qualifier != TypeQualifier.OUTPUT:
            raise FhYTypeError(
                f"Operation {node.name.name_hint!r} must have an OUTPUT "
                "return type qualifier; got "
                f"{return_type.type_qualifier.value!r}.",
                return_type.provenance,
            )

        if not _contains_return_statement(node.body):
            raise FhYStructuralError(
                f"Operation {node.name.name_hint!r} must contain a return "
                "statement.",
                node.provenance,
            )

    def _check_scalar_argument(self, node: Operation, argument: Argument) -> None:
        if not _is_scalar_numerical(argument.qualified_type.base_type):
            raise FhYTypeError(
                f"Argument {argument.name.name_hint!r} of operation "
                f"{node.name.name_hint!r} must have a scalar numerical type; "
                f"got {argument.qualified_type.base_type}.",
                argument.provenance,
            )


def validate_operations(module: Module) -> None:
    """Validate the structural constraints on operations in the AST.

    Args:
        module: The module to validate.

    Raises:
        FhYTypeError: If an operation's argument type is not a scalar
            numerical type, or if its return type is not a scalar OUTPUT.
        FhYStructuralError: If an operation's body does not contain any
            return statement.

    """
    validator = _OperationValidator()
    validator(module)
