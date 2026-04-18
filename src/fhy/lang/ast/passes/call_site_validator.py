"""Validate the function call sites in the AST."""

__all__ = [
    "validate_call_sites",
]

from fhy_core import (
    FunctionKeyword,
    FunctionSymbolTableFrame,
    ImportSymbolTableFrame,
    SymbolTable,
    register_pass,
)

from fhy.lang.ast.error import FhYSemanticsError, FhYStructuralError
from fhy.lang.ast.node import (
    ExpressionStatement,
    FunctionExpression,
    IdentifierExpression,
    Module,
)
from fhy.lang.builtins import BUILTIN_REDUCTION_FUNCTION_IDENTIFIERS

from .analysis_pass_with_symbol_table import AnalysisPassWithSymbolTable


@register_pass(
    "fhy_ast_call_site_validator",
    "Validates the function call sites in the AST.",
)
class _CallSiteValidator(AnalysisPassWithSymbolTable):
    def visit_function_expression(self, node: FunctionExpression) -> None:
        if not isinstance(node.function, IdentifierExpression):
            raise FhYStructuralError(
                "The expression passed as the function name of a call must be "
                f"an identifier expression; got {type(node.function).__name__}."
            )
        identifier = node.function.identifier
        frame = self.get_frame_from_namespace(self.current_namespace, identifier)
        if not isinstance(frame, FunctionSymbolTableFrame | ImportSymbolTableFrame):
            raise FhYSemanticsError(
                f"{identifier.name_hint!r} is not defined as a function."
            )
        is_reduction = (
            isinstance(frame, ImportSymbolTableFrame)
            and frame.name in BUILTIN_REDUCTION_FUNCTION_IDENTIFIERS.values()
        )
        if not is_reduction and len(node.indices) > 0:
            raise FhYStructuralError(
                f"Non-reduction function {identifier.name_hint!r} cannot be "
                "called with indices."
            )
        if isinstance(frame, FunctionSymbolTableFrame) and len(node.args) != len(
            frame.signature
        ):
            raise FhYStructuralError(
                f"Function {identifier.name_hint!r} expects "
                f"{len(frame.signature)} argument(s); got {len(node.args)}."
            )

    def visit_expression_statement(self, node: ExpressionStatement) -> None:
        if not isinstance(node.right, FunctionExpression):
            return
        if not isinstance(node.right.function, IdentifierExpression):
            return
        identifier = node.right.function.identifier
        frame = self.get_frame_from_namespace(self.current_namespace, identifier)
        if (
            isinstance(frame, FunctionSymbolTableFrame)
            and frame.keyword == FunctionKeyword.PROCEDURE
            and node.left is not None
        ):
            raise FhYStructuralError(
                f"Procedure {identifier.name_hint!r} cannot be called with a "
                "left-hand side expression."
            )


def validate_call_sites(module: Module, symbol_table: SymbolTable) -> None:
    """Validate the function call sites in the AST.

    Args:
        module: The module to validate.
        symbol_table: The symbol table to use.

    Raises:
        FhYStructuralError: If the function name is not an identifier
            expression, if a non-reduction function is called with indices, if
            a user-defined function is called with the wrong number of
            arguments, or if a procedure is called with a left-hand side
            expression.
        FhYSemanticsError: If the function name does not resolve to a function
            via the symbol table.

    """
    validator = _CallSiteValidator(symbol_table)
    validator(module)
