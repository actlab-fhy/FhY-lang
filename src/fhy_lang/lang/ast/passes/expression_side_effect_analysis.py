"""Expression side-effect analysis for the FhY AST.

Conservatively flags whether an expression subtree contains a call that
should be treated as potentially observable, so the optimization passes
(dead code elimination, algebraic simplification) can decide whether a
subtree is safe to discard.

Rules:
    - A call to a built-in reduction (``sum``, ``max``, ...) is considered
      pure: these are closed-form mathematical reductions with no
      observable behavior beyond their return value.
    - Any other :class:`FunctionExpression` is conservatively flagged as
      possibly side-effecting. The call-site validator rejects procedure
      calls in value positions, so the non-reduction calls we see are
      operation calls. Operations are pure in FhY semantics, but the
      optimizer still preserves them because their evaluation may be
      observable to the user (debugging, benchmarking) and absorbing them
      is not a correctness-preserving rewrite on its own.

"""

__all__ = [
    "ExpressionSideEffectAnalysis",
]

import logging

from fhy_core import (
    Analysis,
    AnalysisVisitablePass,
    get_logger,
    register_pass,
)

from fhy_lang.lang.ast.node import (
    Expression,
    FunctionExpression,
    IdentifierExpression,
    Node,
)
from fhy_lang.lang.builtins import (
    BUILTIN_REDUCTION_FUNCTION_IDENTIFIERS,
)

_logger: logging.Logger = get_logger(__name__)


@register_pass(
    "fhy_ast_non_reduction_call_finder",
    "Walks an expression subtree, flagging any non-reduction function call.",
)
class _NonReductionCallFinder(AnalysisVisitablePass[Node]):
    """Walker that flags any non-reduction function call in a subtree."""

    _found: bool

    def __init__(self) -> None:
        super().__init__()
        self._found = False

    @property
    def found(self) -> bool:
        return self._found

    def visit_function_expression(self, node: FunctionExpression) -> None:
        if (
            isinstance(node.function, IdentifierExpression)
            and node.function.identifier
            in BUILTIN_REDUCTION_FUNCTION_IDENTIFIERS.values()
        ):
            return
        self._found = True


class ExpressionSideEffectAnalysis(Analysis[Expression, bool]):
    """Analysis that returns whether an expression may have observable side effects.

    Returns ``True`` when the expression subtree contains any
    :class:`FunctionExpression` whose callee is not a built-in reduction.

    """

    def run(self, ir: Expression) -> bool:
        finder = _NonReductionCallFinder()
        finder(ir)
        _logger.debug(
            "Side-effect analysis: %s -> may_have_side_effects=%s.",
            type(ir).__name__,
            finder.found,
        )
        return finder.found
