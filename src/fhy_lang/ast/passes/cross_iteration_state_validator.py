"""Validate that indexed assignments do not read cross-iteration state.

An indexed assignment of the form ``A[i1, ..., ik] = ... A[e1, ..., ek] ...``
is only well-formed if every right-hand-side read of the same array
``A`` uses index expressions structurally equal to the left-hand-side
indices. Reads of a different array are unconstrained by this rule.

The check is purely structural: it does not attempt to prove
equivalences like ``i + 0 == i``. If two index expressions differ in
their AST shape they are treated as different, even if a value-level
constant-folding pass could collapse them. This conservatism keeps
the rule cheap to enforce without a full semantic equivalence checker.

"""

__all__ = [
    "CrossIterationStateValidator",
]

import logging
from collections.abc import Sequence

from fhy_core import (
    AnalysisVisitablePass,
    DiagnosticLevel,
    Identifier,
    get_logger,
    register_pass,
)

from fhy_lang.ast.node import (
    ArrayAccessExpression,
    Expression,
    ExpressionStatement,
    IdentifierExpression,
    Node,
)

from .utils import format_diagnostic_message

_logger: logging.Logger = get_logger(__name__)


def _are_index_tuples_structurally_equal(
    left_indices: Sequence[Expression], right_indices: Sequence[Expression]
) -> bool:
    if len(left_indices) != len(right_indices):
        return False
    return all(
        left.is_structurally_equivalent(right)
        for left, right in zip(left_indices, right_indices)
    )


def _collect_array_accesses(expression: Expression) -> list[ArrayAccessExpression]:
    """Return every ArrayAccessExpression reachable from ``expression``."""
    accesses: list[ArrayAccessExpression] = []
    pending: list[Node] = [expression]
    while pending:
        node = pending.pop()
        if isinstance(node, ArrayAccessExpression):
            accesses.append(node)
        # Descend into all logical children via the standard Visitable
        # contract; new node kinds are picked up automatically.
        children = node.get_visit_children()
        pending.extend(child for child in children if isinstance(child, Node))
    return accesses


def _get_array_identifier(access: ArrayAccessExpression) -> Identifier | None:
    """Return the underlying :class:`Identifier` of a simple array access.

    Returns ``None`` for chained or non-identifier array bases (those
    are structural errors that other passes report).
    """
    base = access.array_expression
    if isinstance(base, IdentifierExpression):
        return base.identifier
    return None


@register_pass(
    "fhy_ast_cross_iteration_state_validator",
    "Validates that indexed assignments do not read cross-iteration state.",
)
class CrossIterationStateValidator(AnalysisVisitablePass[Node]):
    """Reject indexed assignments whose RHS reads the LHS array at a different index.

    For each :class:`ExpressionStatement` whose ``left`` is an
    :class:`ArrayAccessExpression` ``A[i1, ..., ik]``:

    * Walk the ``right`` expression tree, collecting every
      :class:`ArrayAccessExpression`.
    * For each, if it indexes the same identifier ``A`` (the array
      being written), require that its index tuple be structurally
      equivalent to ``(i1, ..., ik)``.
    * Reads of a different array are unconstrained.

    The walk descends into nested expressions and into the bodies of
    :class:`ForAllStatement` constructs; the rule is structural and
    applies regardless of ordering.

    """

    def visit_expression_statement(self, node: ExpressionStatement) -> None:
        lhs = node.left
        if not isinstance(lhs, ArrayAccessExpression):
            return
        target_identifier = _get_array_identifier(lhs)
        if target_identifier is None:
            return
        for access in _collect_array_accesses(node.right):
            if _get_array_identifier(access) != target_identifier:
                continue
            if _are_index_tuples_structurally_equal(access.indices, lhs.indices):
                continue
            self._report_cross_iteration_read(node, target_identifier)

    def _report_cross_iteration_read(
        self, node: ExpressionStatement, target_identifier: Identifier
    ) -> None:
        _logger.debug(
            "Cross-iteration read on %s detected in expression statement.",
            target_identifier,
        )
        self.report(
            DiagnosticLevel.ERROR,
            format_diagnostic_message(
                "iteration error",
                f"cross-iteration state is not allowed: array "
                f"{target_identifier.name_hint!r} is read with index "
                f"expression(s) that differ from the left-hand-side "
                "index expression(s) for this iteration.",
                node.provenance,
            ),
        )
