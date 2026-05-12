"""Algebraic simplification optimization over the FhY AST.

Applies identity- and absorbing-element rewrites that are independent of
the specific numeric width or precision of the operands. Complements the
constant folding pass: that pass collapses fully-literal subtrees, this
pass handles cases where at least one operand is non-literal but the
other is a structural identity / absorbing literal.

Rewrites are conservative: any rewrite that discards an operand subtree
(e.g. ``x * 0``) only fires when that subtree has no observable function
calls, mirroring the side-effect guard used by dead code elimination.

"""

__all__ = [
    "AlgebraicSimplificationPass",
]

import logging

from fhy_core import (
    Provenance,
    get_logger,
    register_pass,
)

from fhy_lang.ast.node import (
    BinaryExpression,
    BinaryOperation,
    ComplexLiteral,
    Expression,
    FloatLiteral,
    IntLiteral,
    Node,
    UnaryExpression,
    UnaryOperation,
)

from .expression_side_effect_analysis import ExpressionSideEffectAnalysis
from .transformer import Transformer

_logger: logging.Logger = get_logger(__name__)


def _is_literal_zero(expression: Expression) -> bool:
    """Return True if ``expression`` is a compile-time literal zero.

    Matches the same shape-rule used by the constant-safety validator:
    recognizes ``IntLiteral(0)``, ``FloatLiteral(0.0)``,
    ``ComplexLiteral(0+0j)``, and any chain of unary negations of those.

    """
    if isinstance(expression, IntLiteral):
        return expression.value == 0
    elif isinstance(expression, FloatLiteral):
        return expression.value == 0.0
    elif isinstance(expression, ComplexLiteral):
        return expression.value == 0
    elif (
        isinstance(expression, UnaryExpression)
        and expression.operation == UnaryOperation.NEGATION
    ):
        return _is_literal_zero(expression.expression)
    else:
        return False


def _is_literal_one(expression: Expression) -> bool:
    """Return True if ``expression`` is a compile-time literal one."""
    if isinstance(expression, IntLiteral):
        return expression.value == 1
    elif isinstance(expression, FloatLiteral):
        return expression.value == 1.0
    elif isinstance(expression, ComplexLiteral):
        return expression.value == 1
    else:
        return False


def _make_int_zero(provenance: Provenance) -> IntLiteral:
    return IntLiteral(value=0, provenance=provenance)


@register_pass(
    "fhy_ast_algebraic_simplification",
    "Applies algebraic identity and absorbing-element rewrites to the AST.",
)
class AlgebraicSimplificationPass(Transformer):
    """Algebraic-identity rewriter for the FhY AST."""

    _simplified_count: int

    def __init__(self) -> None:
        super().__init__()
        self._simplified_count = 0

    def run_pass(self, ir: Node) -> Node:
        self._simplified_count = 0
        return super().run_pass(ir)

    def did_change(self, input_ir: Node, output: Node) -> bool:
        _ = (input_ir, output)
        return self._simplified_count > 0

    def visit_unary_expression(  # type: ignore[override]
        self, node: UnaryExpression
    ) -> Expression:
        # Base ``Transformer.visit_unary_expression`` narrows to
        # ``UnaryExpression``; peeling a double negation legitimately
        # widens the return type to any ``Expression``. The dispatcher in
        # ``Transformer.visit_expression`` consumes this as ``Expression``.
        inner = self.visit_expression(node.expression)
        simplified = _try_simplify_unary(node.operation, inner)
        if simplified is not None:
            self._simplified_count += 1
            _logger.debug(
                "Simplified double unary %s (self-inverse).",
                node.operation.value,
            )
            return simplified
        return UnaryExpression(
            operation=node.operation,
            expression=inner,
            provenance=node.provenance,
        )

    def visit_binary_expression(self, node: BinaryExpression) -> Expression:
        left = self.visit_expression(node.left)
        right = self.visit_expression(node.right)
        simplified = self._try_simplify_binary(
            node.operation, left, right, node.provenance
        )
        if simplified is not None:
            self._simplified_count += 1
            _logger.debug(
                "Simplified binary %s via identity/absorbing-element rewrite.",
                node.operation.value,
            )
            return simplified
        return BinaryExpression(
            operation=node.operation,
            left=left,
            right=right,
            provenance=node.provenance,
        )

    def _try_simplify_binary(  # noqa: C901, PLR0912
        self,
        operation: BinaryOperation,
        left: Expression,
        right: Expression,
        provenance: Provenance,
    ) -> Expression | None:
        if operation == BinaryOperation.ADDITION:
            if _is_literal_zero(right):
                return left
            elif _is_literal_zero(left):
                return right
            else:
                return None

        if operation == BinaryOperation.SUBTRACTION:
            if _is_literal_zero(right):
                return left
            else:
                return None

        if operation == BinaryOperation.MULTIPLICATION:
            if _is_literal_one(right):
                return left
            elif _is_literal_one(left):
                return right
            elif _is_literal_zero(right) and not self._may_have_side_effects(left):
                return right
            elif _is_literal_zero(left) and not self._may_have_side_effects(right):
                return left
            else:
                return None

        if operation == BinaryOperation.DIVISION:
            # ``x / 0`` is left to the constant-safety validator; never
            # rewrite by pattern alone.
            if _is_literal_zero(right):
                return None
            elif _is_literal_one(right):
                return left
            else:
                return None

        if operation == BinaryOperation.FLOORDIV:
            if _is_literal_zero(right):
                return None
            elif _is_literal_one(right):
                return left
            else:
                return None

        if operation == BinaryOperation.MODULO:
            if _is_literal_zero(right):
                return None
            elif _is_literal_one(right) and not self._may_have_side_effects(left):
                return _make_int_zero(provenance)
            else:
                return None

        if operation == BinaryOperation.POWER:
            if _is_literal_one(right):
                return left
            elif _is_literal_zero(right) and not self._may_have_side_effects(left):
                # ``x ** 0 == 1`` for every numeric ``x`` that FhY supports
                # (we exclude the ``0 ** 0`` corner by requiring the literal
                # zero shape on the right operand only).
                return IntLiteral(value=1, provenance=provenance)
            else:
                return None

        return None

    def _may_have_side_effects(self, expression: Expression) -> bool:
        return self.get_analysis(ExpressionSideEffectAnalysis, expression)


def _try_simplify_unary(
    operation: UnaryOperation, operand: Expression
) -> Expression | None:
    # Double negation and double bitwise-not are self-inverses on
    # arithmetic / integer values respectively.
    if not isinstance(operand, UnaryExpression):
        return None
    if operation == operand.operation and operation in {
        UnaryOperation.NEGATION,
        UnaryOperation.BITWISE_NOT,
    }:
        return operand.expression
    return None
