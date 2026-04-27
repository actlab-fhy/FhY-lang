"""Constant folding optimization over the FhY AST.

Folds two kinds of constants:

1. AST-level literals: a :class:`UnaryExpression`, :class:`BinaryExpression`,
   or :class:`TernaryExpression` whose operands are fully literal is
   evaluated at compile time and replaced by a single :class:`IntLiteral`,
   :class:`FloatLiteral`, or :class:`ComplexLiteral`.

2. Core expressions embedded in types: shape expressions on
   :class:`NumericalType` and bound / stride expressions on
   :class:`IndexType` are simplified via
   :func:`fhy_core.simplify_expression`.

Unsafe folds (e.g., division or modulo by zero) are intentionally left in
place so the constant-safety validator can diagnose them.

"""

__all__ = [
    "ConstantFoldingPass",
]

import logging
from typing import TypeGuard

from fhy_core import (
    Expression as CoreExpression,
)
from fhy_core import (
    IndexType,
    NumericalType,
    Provenance,
    TupleType,
    Type,
    get_logger,
    register_pass,
    simplify_expression,
)

from fhy_lang.ast.node import (
    BinaryExpression,
    BinaryOperation,
    ComplexLiteral,
    Expression,
    FloatLiteral,
    IntLiteral,
    Node,
    QualifiedType,
    TernaryExpression,
    UnaryExpression,
    UnaryOperation,
)

from .transformer import Transformer

_logger: logging.Logger = get_logger(__name__)

_AstLiteral = IntLiteral | FloatLiteral | ComplexLiteral
_RealLiteral = IntLiteral | FloatLiteral

_INTEGER_BITWISE_BINARY_OPERATIONS: frozenset[BinaryOperation] = frozenset(
    {
        BinaryOperation.LEFT_SHIFT,
        BinaryOperation.RIGHT_SHIFT,
        BinaryOperation.BITWISE_AND,
        BinaryOperation.BITWISE_OR,
        BinaryOperation.BITWISE_XOR,
    }
)
_REAL_ONLY_BINARY_OPERATIONS: frozenset[BinaryOperation] = frozenset(
    {
        BinaryOperation.FLOORDIV,
        BinaryOperation.MODULO,
        BinaryOperation.LESS_THAN,
        BinaryOperation.LESS_THAN_OR_EQUAL,
        BinaryOperation.GREATER_THAN,
        BinaryOperation.GREATER_THAN_OR_EQUAL,
    }
)

_NUMERIC_FOLDABLE_EXCEPTIONS = (TypeError, ValueError, ZeroDivisionError, OverflowError)


def _is_literal_expression(expression: Expression) -> TypeGuard[_AstLiteral]:
    return isinstance(expression, IntLiteral | FloatLiteral | ComplexLiteral)


def _build_literal_from_value(
    value: int | float | complex,  # noqa: PYI041
    provenance: Provenance,
) -> _AstLiteral | None:
    # `bool` is a subclass of `int`; normalize to a plain `IntLiteral`.
    if isinstance(value, bool):
        return IntLiteral(value=int(value), provenance=provenance)
    elif isinstance(value, int):
        return IntLiteral(value=value, provenance=provenance)
    elif isinstance(value, float):
        return FloatLiteral(value=value, provenance=provenance)
    elif isinstance(value, complex):
        return ComplexLiteral(value=value, provenance=provenance)
    else:
        return None


def _fold_integer_bitwise_operation(
    operation: BinaryOperation,
    left_value: int,
    right_value: int,
    provenance: Provenance,
) -> IntLiteral | None:
    if operation == BinaryOperation.LEFT_SHIFT:
        return IntLiteral(value=left_value << right_value, provenance=provenance)
    elif operation == BinaryOperation.RIGHT_SHIFT:
        return IntLiteral(value=left_value >> right_value, provenance=provenance)
    elif operation == BinaryOperation.BITWISE_AND:
        return IntLiteral(value=left_value & right_value, provenance=provenance)
    elif operation == BinaryOperation.BITWISE_OR:
        return IntLiteral(value=left_value | right_value, provenance=provenance)
    elif operation == BinaryOperation.BITWISE_XOR:
        return IntLiteral(value=left_value ^ right_value, provenance=provenance)
    else:
        return None


def _fold_real_only_binary_operation(
    operation: BinaryOperation,
    left: _RealLiteral,
    right: _RealLiteral,
    provenance: Provenance,
) -> Expression | None:
    left_value = left.value
    right_value = right.value
    if operation == BinaryOperation.FLOORDIV:
        if right_value == 0:
            return None
        return _build_literal_from_value(left_value // right_value, provenance)
    elif operation == BinaryOperation.MODULO:
        if right_value == 0:
            return None
        return _build_literal_from_value(left_value % right_value, provenance)
    elif operation == BinaryOperation.LESS_THAN:
        return IntLiteral(value=int(left_value < right_value), provenance=provenance)
    elif operation == BinaryOperation.LESS_THAN_OR_EQUAL:
        return IntLiteral(value=int(left_value <= right_value), provenance=provenance)
    elif operation == BinaryOperation.GREATER_THAN:
        return IntLiteral(value=int(left_value > right_value), provenance=provenance)
    elif operation == BinaryOperation.GREATER_THAN_OR_EQUAL:
        return IntLiteral(value=int(left_value >= right_value), provenance=provenance)
    else:
        return None


def _fold_arithmetic_binary_operation(
    operation: BinaryOperation,
    left: _AstLiteral,
    right: _AstLiteral,
    provenance: Provenance,
) -> Expression | None:
    if operation == BinaryOperation.ADDITION:
        return _build_literal_from_value(left.value + right.value, provenance)
    elif operation == BinaryOperation.SUBTRACTION:
        return _build_literal_from_value(left.value - right.value, provenance)
    elif operation == BinaryOperation.MULTIPLICATION:
        return _build_literal_from_value(left.value * right.value, provenance)
    elif operation == BinaryOperation.DIVISION:
        if right.value == 0:
            # Diagnosed by the constant-safety validator; leave unfolded.
            return None
        return _build_literal_from_value(left.value / right.value, provenance)
    elif operation == BinaryOperation.POWER:
        return _build_literal_from_value(left.value**right.value, provenance)
    else:
        return None


def _fold_equality_binary_operation(
    operation: BinaryOperation,
    left: _AstLiteral,
    right: _AstLiteral,
    provenance: Provenance,
) -> IntLiteral | None:
    if operation == BinaryOperation.EQUAL_TO:
        return IntLiteral(value=int(left.value == right.value), provenance=provenance)
    elif operation == BinaryOperation.NOT_EQUAL_TO:
        return IntLiteral(value=int(left.value != right.value), provenance=provenance)
    else:
        return None


def _fold_logical_binary_operation(
    operation: BinaryOperation,
    left: _AstLiteral,
    right: _AstLiteral,
    provenance: Provenance,
) -> IntLiteral | None:
    if operation == BinaryOperation.LOGICAL_AND:
        folded = bool(left.value) and bool(right.value)
        return IntLiteral(value=int(folded), provenance=provenance)
    elif operation == BinaryOperation.LOGICAL_OR:
        folded = bool(left.value) or bool(right.value)
        return IntLiteral(value=int(folded), provenance=provenance)
    else:
        return None


def _fold_unary_operation(
    operation: UnaryOperation,
    operand: _AstLiteral,
    provenance: Provenance,
) -> Expression | None:
    try:
        if operation == UnaryOperation.NEGATION:
            return _build_literal_from_value(-operand.value, provenance)
        elif operation == UnaryOperation.BITWISE_NOT:
            # Bitwise-not is only defined for integers; leave the rest alone.
            if not isinstance(operand, IntLiteral):
                return None
            return IntLiteral(value=~operand.value, provenance=provenance)
        elif operation == UnaryOperation.LOGICAL_NOT:
            return IntLiteral(
                value=0 if bool(operand.value) else 1, provenance=provenance
            )
        else:
            return None
    except _NUMERIC_FOLDABLE_EXCEPTIONS:
        return None


def _fold_binary_operation(
    operation: BinaryOperation,
    left: _AstLiteral,
    right: _AstLiteral,
    provenance: Provenance,
) -> Expression | None:
    try:
        if operation in _INTEGER_BITWISE_BINARY_OPERATIONS:
            if not isinstance(left, IntLiteral) or not isinstance(right, IntLiteral):
                return None
            return _fold_integer_bitwise_operation(
                operation, left.value, right.value, provenance
            )
        elif operation in _REAL_ONLY_BINARY_OPERATIONS:
            if isinstance(left, ComplexLiteral) or isinstance(right, ComplexLiteral):
                return None
            return _fold_real_only_binary_operation(operation, left, right, provenance)
        arithmetic = _fold_arithmetic_binary_operation(
            operation, left, right, provenance
        )
        if arithmetic is not None:
            return arithmetic
        equality = _fold_equality_binary_operation(operation, left, right, provenance)
        if equality is not None:
            return equality
        return _fold_logical_binary_operation(operation, left, right, provenance)
    except _NUMERIC_FOLDABLE_EXCEPTIONS:
        return None


def _core_expressions_equal(
    left: CoreExpression | None, right: CoreExpression | None
) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    try:
        return left.is_structurally_equivalent(right)
    except Exception:
        return repr(left) == repr(right)


def _core_expression_tuples_equal(
    left: tuple[CoreExpression, ...], right: tuple[CoreExpression, ...]
) -> bool:
    if len(left) != len(right):
        return False
    return all(
        _core_expressions_equal(left_item, right_item)
        for left_item, right_item in zip(left, right)
    )


def _try_evaluate_condition_as_boolean(condition: _AstLiteral) -> bool | None:
    try:
        return bool(condition.value)
    except _NUMERIC_FOLDABLE_EXCEPTIONS:
        return None


@register_pass(
    "fhy_ast_constant_folding",
    "Folds constant-literal subexpressions and simplifies core expressions "
    "embedded in types.",
)
class ConstantFoldingPass(Transformer):
    """Constant-folding rewriter for the FhY AST."""

    _folded_count: int

    def __init__(self) -> None:
        super().__init__()
        self._folded_count = 0

    def run_pass(self, ir: Node) -> Node:
        self._folded_count = 0
        _logger.info("Starting constant folding pass...")
        result = super().run_pass(ir)
        _logger.info(
            "Constant folding complete: %d expression(s) folded.",
            self._folded_count,
        )
        return result

    def did_change(self, input_ir: Node, output: Node) -> bool:
        _ = (input_ir, output)
        return self._folded_count > 0

    def visit_unary_expression(  # type: ignore[override]
        self, node: UnaryExpression
    ) -> Expression:
        # Base `Transformer.visit_unary_expression` narrows to
        # `UnaryExpression`; folding legitimately widens the return type to
        # any `Expression` (a literal is returned when the operand is fully
        # constant). The visitor dispatcher in `Transformer.visit_expression`
        # treats the result as `Expression`, so the widening is safe.
        inner = self.visit_expression(node.expression)
        if _is_literal_expression(inner):
            folded = _fold_unary_operation(node.operation, inner, node.provenance)
            if folded is not None:
                self._folded_count += 1
                _logger.debug(
                    "Folded unary %s(%r) -> %r.",
                    node.operation.value,
                    inner.value,
                    getattr(folded, "value", folded),
                )
                return folded
        return UnaryExpression(
            operation=node.operation,
            expression=inner,
            provenance=node.provenance,
        )

    def visit_binary_expression(self, node: BinaryExpression) -> Expression:
        left = self.visit_expression(node.left)
        right = self.visit_expression(node.right)
        if _is_literal_expression(left) and _is_literal_expression(right):
            folded = _fold_binary_operation(
                node.operation, left, right, node.provenance
            )
            if folded is not None:
                self._folded_count += 1
                _logger.debug(
                    "Folded binary %r %s %r -> %r.",
                    left.value,
                    node.operation.value,
                    right.value,
                    getattr(folded, "value", folded),
                )
                return folded
        return BinaryExpression(
            operation=node.operation,
            left=left,
            right=right,
            provenance=node.provenance,
        )

    def visit_ternary_expression(self, node: TernaryExpression) -> Expression:
        condition = self.visit_expression(node.condition)
        true_branch = self.visit_expression(node.true)
        false_branch = self.visit_expression(node.false)
        if _is_literal_expression(condition):
            is_truthy = _try_evaluate_condition_as_boolean(condition)
            if is_truthy is True:
                self._folded_count += 1
                _logger.debug("Folded ternary with truthy literal condition.")
                return true_branch
            if is_truthy is False:
                self._folded_count += 1
                _logger.debug("Folded ternary with falsy literal condition.")
                return false_branch
        return TernaryExpression(
            condition=condition,
            true=true_branch,
            false=false_branch,
            provenance=node.provenance,
        )

    def visit_qualified_type(self, node: QualifiedType) -> QualifiedType:
        new_base_type = self._simplify_type(node.base_type)
        if new_base_type is node.base_type:
            return node
        return QualifiedType(
            base_type=new_base_type,
            type_qualifier=node.type_qualifier,
            provenance=node.provenance,
        )

    def _simplify_type(self, type_node: Type) -> Type:
        if isinstance(type_node, NumericalType):
            return self._simplify_numerical_type(type_node)
        elif isinstance(type_node, IndexType):
            return self._simplify_index_type(type_node)
        elif isinstance(type_node, TupleType):
            return self._simplify_tuple_type(type_node)
        else:
            return type_node

    def _simplify_numerical_type(self, type_node: NumericalType) -> NumericalType:
        original_shape = tuple(type_node.shape)
        simplified_shape = tuple(simplify_expression(dim) for dim in original_shape)
        if _core_expression_tuples_equal(original_shape, simplified_shape):
            return type_node
        self._folded_count += 1
        return NumericalType(type_node.data_type, shape=simplified_shape)

    def _simplify_index_type(self, type_node: IndexType) -> IndexType:
        new_lower = simplify_expression(type_node.lower_bound)
        new_upper = simplify_expression(type_node.upper_bound)
        new_stride = simplify_expression(type_node.stride)
        lower_unchanged = _core_expressions_equal(type_node.lower_bound, new_lower)
        upper_unchanged = _core_expressions_equal(type_node.upper_bound, new_upper)
        stride_unchanged = _core_expressions_equal(type_node.stride, new_stride)
        if lower_unchanged and upper_unchanged and stride_unchanged:
            return type_node
        self._folded_count += 1
        return IndexType(new_lower, new_upper, new_stride)

    def _simplify_tuple_type(self, type_node: TupleType) -> TupleType:
        new_types = [self._simplify_type(t) for t in type_node.types]
        all_unchanged = all(
            new_type is original_type
            for new_type, original_type in zip(new_types, type_node.types)
        )
        if all_unchanged:
            return type_node
        return TupleType(new_types)
