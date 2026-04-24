"""Validate tuple access expressions in the AST."""

__all__ = [
    "TupleAccessValidator",
]

import logging

from fhy_core import (
    DiagnosticLevel,
    SymbolTableError,
    SymbolTableFrame,
    TupleType,
    VariableSymbolTableFrame,
    get_logger,
    register_pass,
)

from fhy_lang.lang.ast.node import (
    Expression,
    IdentifierExpression,
    IntLiteral,
    TupleAccessExpression,
    TupleExpression,
)

from .analysis_pass_with_symbol_table import AnalysisPassWithSymbolTable
from .utils import format_diagnostic_message

_logger: logging.Logger = get_logger(__name__)


def _get_tuple_type_from_frame(frame: SymbolTableFrame) -> TupleType | None:
    if isinstance(frame, VariableSymbolTableFrame) and isinstance(
        frame.type, TupleType
    ):
        return frame.type
    return None


def _get_nested_tuple_type(
    outer_tuple_type: TupleType, element_index: int
) -> TupleType | None:
    types = outer_tuple_type.types
    if not 0 <= element_index < len(types):
        return None
    element_type = types[element_index]
    if isinstance(element_type, TupleType):
        return element_type
    return None


@register_pass(
    "fhy_ast_tuple_access_validator",
    "Validates that tuple access expressions use an in-range element index.",
)
class TupleAccessValidator(AnalysisPassWithSymbolTable):
    """Validate tuple access expressions against the tuple's arity.

    For every :class:`TupleAccessExpression`, the pass:

    - Determines the tuple's arity statically when possible -- directly from
      a :class:`TupleExpression` literal, from an :class:`IdentifierExpression`
      that resolves to a :class:`VariableSymbolTableFrame` with a
      :class:`TupleType`, or recursively through a nested tuple access whose
      element type is itself a :class:`TupleType`.
    - Emits an ERROR diagnostic when the ``element_index`` literal is negative
      or falls outside ``[0, arity)``.

    When the tuple's arity cannot be determined statically (e.g. the base
    expression is a function call whose return type is not a tuple in the
    symbol table, or a nested access reaches through a literal whose inner
    element types are not declared) the visitor stays silent so it does not
    duplicate diagnostics already surfaced by :class:`TypeChecker`.

    """

    def visit_tuple_access_expression(self, node: TupleAccessExpression) -> None:
        arity = self._resolve_tuple_arity(node.tuple_expression)
        _logger.debug(
            "TupleAccess check: index=%d, resolved_arity=%s.",
            node.element_index.value,
            arity,
        )
        if arity is None:
            return
        self._check_element_index_in_range(node, arity)

    def _resolve_tuple_arity(self, expression: Expression) -> int | None:
        if isinstance(expression, TupleExpression):
            return len(expression.expressions)
        tuple_type = self._resolve_tuple_type(expression)
        if tuple_type is None:
            return None
        return len(tuple_type.types)

    def _resolve_tuple_type(self, expression: Expression) -> TupleType | None:
        if isinstance(expression, IdentifierExpression):
            return self._resolve_identifier_tuple_type(expression)
        elif isinstance(expression, TupleAccessExpression):
            return self._resolve_nested_tuple_type(expression)
        else:
            # :class:`TupleExpression` is intentionally excluded: a tuple
            # literal's element types are not recorded on the AST node, so
            # we cannot reliably build a :class:`TupleType` that would let
            # nested access chains resolve their inner arities.
            return None

    def _resolve_identifier_tuple_type(
        self, expression: IdentifierExpression
    ) -> TupleType | None:
        try:
            frame = self.get_frame_from_namespace(
                self.current_namespace, expression.identifier
            )
        except SymbolTableError:
            return None
        return _get_tuple_type_from_frame(frame)

    def _resolve_nested_tuple_type(
        self, expression: TupleAccessExpression
    ) -> TupleType | None:
        outer_tuple_type = self._resolve_tuple_type(expression.tuple_expression)
        if outer_tuple_type is None:
            return None
        return _get_nested_tuple_type(outer_tuple_type, expression.element_index.value)

    def _check_element_index_in_range(
        self, node: TupleAccessExpression, arity: int
    ) -> None:
        if 0 <= node.element_index.value < arity:
            return
        self._report_element_index_out_of_range(node, node.element_index, arity)

    def _report_element_index_out_of_range(
        self,
        node: TupleAccessExpression,
        element_index: IntLiteral,
        arity: int,
    ) -> None:
        self.report(
            DiagnosticLevel.ERROR,
            format_diagnostic_message(
                "semantic error",
                f"Tuple access element index {element_index.value} is out of "
                f"range for a tuple of arity {arity}; valid indices are "
                f"[0, {arity}).",
                node.provenance,
            ),
        )
