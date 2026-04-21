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
    Node,
    Operation,
)

from .utils import format_diagnostic_message


def _is_scalar_numerical(type_: Type) -> bool:
    return isinstance(type_, NumericalType) and type_.is_scalar()


@register_pass(
    "fhy_ast_operation_validator",
    "Validates the structural constraints on operations in the AST.",
)
class OperationValidator(AnalysisVisitablePass[Node]):
    """Validate operation argument and return-type shape constraints.

    Every argument and the return type must be a scalar numerical type
    (shape-free primitive), and the return-type qualifier must be OUTPUT.

    Return-statement placement (operations must return on every path,
    procedures must not return at all) is enforced separately by
    :class:`~fhy.lang.ast.passes.return_validator.ReturnValidator`.

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
