"""Validate constant-safety properties of the FhY AST.

Currently flags division, floor-division, and modulo operations whose
right-hand operand reduces to a compile-time literal zero. The check is
purely structural — it does not constant-fold arbitrary expressions, only
recognizes literal zero, negated literal zero, and nested negations of
literal zero. More precise reasoning belongs to a future constant-folding
optimization pass.

"""

__all__ = [
    "validate_constant_safety",
]

from fhy_core import (
    AnalysisVisitablePass,
    register_pass,
)

from fhy.lang.ast.error import FhYSemanticsError
from fhy.lang.ast.node import (
    BinaryExpression,
    BinaryOperation,
    ComplexLiteral,
    Expression,
    FloatLiteral,
    IntLiteral,
    Module,
    Node,
    UnaryExpression,
    UnaryOperation,
)

_ZERO_DIVISOR_OPERATIONS: frozenset[BinaryOperation] = frozenset(
    {
        BinaryOperation.DIVISION,
        BinaryOperation.FLOORDIV,
        BinaryOperation.MODULO,
    }
)


def _is_literal_zero(expression: Expression) -> bool:
    """Return True if `expression` is a compile-time literal zero.

    Recognizes `IntLiteral(0)`, `FloatLiteral(0.0)`, `ComplexLiteral(0+0j)`,
    and any chain of unary negations of the above (so `-0`, `--0`, etc. all
    qualify).

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


@register_pass(
    "fhy_ast_constant_safety_validator",
    "Flags division or modulo by a compile-time literal zero.",
)
class _ConstantSafetyValidator(AnalysisVisitablePass[Node]):
    def visit_binary_expression(self, node: BinaryExpression) -> None:
        if node.operation not in _ZERO_DIVISOR_OPERATIONS:
            return
        if not _is_literal_zero(node.right):
            return
        raise FhYSemanticsError(
            f"Right-hand side of a {node.operation.value!r} operation is a "
            "compile-time literal zero.",
            node.provenance,
        )


def validate_constant_safety(module: Module) -> None:
    """Validate constant-safety properties of the FhY AST.

    Args:
        module: The module to validate.

    Raises:
        FhYSemanticsError: When division, floor-division, or modulo has a
            compile-time literal zero on the right-hand side.

    """
    validator = _ConstantSafetyValidator()
    validator(module)
