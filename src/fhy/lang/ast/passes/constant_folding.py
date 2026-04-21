"""Constant folding optimization over the FhY AST.

Folds two kinds of constants:

1. AST-level literals: a `UnaryExpression`, `BinaryExpression`, or
   `TernaryExpression` whose operands are fully literal is evaluated at
   compile time and replaced by a single `IntLiteral` / `FloatLiteral` /
   `ComplexLiteral`.

2. Core expressions embedded in types: shape expressions on
   `NumericalType` and bound / stride expressions on `IndexType` are
   simplified via :func:`fhy_core.simplify_expression`.

Unsafe folds (e.g., division / modulo by zero) are intentionally left in
place so the constant-safety validator can diagnose them.

"""

__all__ = [
    "ConstantFoldingPass",
]

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
    register_pass,
    simplify_expression,
)

from fhy.lang.ast.node import (
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


def _is_literal(expression: Expression) -> TypeGuard[_AstLiteral]:
    return isinstance(expression, IntLiteral | FloatLiteral | ComplexLiteral)


def _literal_from_value(
    value: int | float | complex,  # noqa: PYI041
    provenance: Provenance,
) -> _AstLiteral | None:
    # `bool` is a subclass of `int`; normalize to a plain IntLiteral.
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
        return _literal_from_value(left_value // right_value, provenance)
    elif operation == BinaryOperation.MODULO:
        if right_value == 0:
            return None
        return _literal_from_value(left_value % right_value, provenance)
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
        return super().run_pass(ir)

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
        if _is_literal(inner):
            folded = self._try_fold_unary(node.operation, inner, node.provenance)
            if folded is not None:
                self._folded_count += 1
                return folded
        return UnaryExpression(
            operation=node.operation,
            expression=inner,
            provenance=node.provenance,
        )

    def visit_binary_expression(self, node: BinaryExpression) -> Expression:
        left = self.visit_expression(node.left)
        right = self.visit_expression(node.right)
        if _is_literal(left) and _is_literal(right):
            folded = self._try_fold_binary(node.operation, left, right, node.provenance)
            if folded is not None:
                self._folded_count += 1
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
        if _is_literal(condition):
            try:
                is_truthy = bool(condition.value)
            except (TypeError, ValueError):
                is_truthy = None
            if is_truthy is True:
                self._folded_count += 1
                return true_branch
            elif is_truthy is False:
                self._folded_count += 1
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
        if self._core_expressions_all_equal(original_shape, simplified_shape):
            return type_node
        self._folded_count += 1
        return NumericalType(type_node.data_type, shape=simplified_shape)

    def _simplify_index_type(self, type_node: IndexType) -> IndexType:
        new_lower = simplify_expression(type_node.lower_bound)
        new_upper = simplify_expression(type_node.upper_bound)
        new_stride = (
            simplify_expression(type_node.stride)
            if type_node.stride is not None
            else None
        )
        lower_unchanged = self._core_expression_equal(type_node.lower_bound, new_lower)
        upper_unchanged = self._core_expression_equal(type_node.upper_bound, new_upper)
        stride_unchanged = self._core_expression_equal(type_node.stride, new_stride)
        if lower_unchanged and upper_unchanged and stride_unchanged:
            return type_node
        self._folded_count += 1
        return IndexType(new_lower, new_upper, stride=new_stride)

    def _simplify_tuple_type(self, type_node: TupleType) -> TupleType:
        new_types = [self._simplify_type(t) for t in type_node.types]
        if all(
            new_type is original_type
            for new_type, original_type in zip(new_types, type_node.types)
        ):
            return type_node
        return TupleType(new_types)

    @staticmethod
    def _core_expression_equal(
        a: CoreExpression | None, b: CoreExpression | None
    ) -> bool:
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        try:
            return a.is_structurally_equivalent(b)
        except Exception:
            return repr(a) == repr(b)

    @classmethod
    def _core_expressions_all_equal(
        cls,
        left_expressions: tuple[CoreExpression, ...],
        right_expressions: tuple[CoreExpression, ...],
    ) -> bool:
        if len(left_expressions) != len(right_expressions):
            return False
        return all(
            cls._core_expression_equal(left, right)
            for left, right in zip(left_expressions, right_expressions)
        )

    @staticmethod
    def _try_fold_unary(
        operation: UnaryOperation,
        operand: _AstLiteral,
        provenance: Provenance,
    ) -> Expression | None:
        try:
            if operation == UnaryOperation.NEGATION:
                return _literal_from_value(-operand.value, provenance)
            elif operation == UnaryOperation.BITWISE_NOT:
                if isinstance(operand, IntLiteral):
                    return IntLiteral(value=~operand.value, provenance=provenance)
                # Bitwise-not is only defined for integers; leave others alone.
                return None
            elif operation == UnaryOperation.LOGICAL_NOT:
                return IntLiteral(
                    value=0 if bool(operand.value) else 1, provenance=provenance
                )
        except (TypeError, ValueError):
            return None
        return None

    @staticmethod
    def _try_fold_binary(  # noqa: C901, PLR0912
        operation: BinaryOperation,
        left: _AstLiteral,
        right: _AstLiteral,
        provenance: Provenance,
    ) -> Expression | None:
        try:
            # Arithmetic operations defined on every numeric type, including
            # `complex`.
            if operation == BinaryOperation.ADDITION:
                return _literal_from_value(left.value + right.value, provenance)
            elif operation == BinaryOperation.SUBTRACTION:
                return _literal_from_value(left.value - right.value, provenance)
            elif operation == BinaryOperation.MULTIPLICATION:
                return _literal_from_value(left.value * right.value, provenance)
            elif operation == BinaryOperation.DIVISION:
                if right.value == 0:
                    return None  # diagnosed by the constant-safety validator
                return _literal_from_value(left.value / right.value, provenance)
            elif operation == BinaryOperation.POWER:
                return _literal_from_value(left.value**right.value, provenance)

            # Equality works on every numeric type.
            elif operation == BinaryOperation.EQUAL_TO:
                return IntLiteral(
                    value=int(left.value == right.value),
                    provenance=provenance,
                )
            elif operation == BinaryOperation.NOT_EQUAL_TO:
                return IntLiteral(
                    value=int(left.value != right.value),
                    provenance=provenance,
                )

            # Logical operations only inspect truthiness; safe for every
            # numeric type.
            elif operation == BinaryOperation.LOGICAL_AND:
                return IntLiteral(
                    value=1 if bool(left.value) and bool(right.value) else 0,
                    provenance=provenance,
                )
            elif operation == BinaryOperation.LOGICAL_OR:
                return IntLiteral(
                    value=1 if bool(left.value) or bool(right.value) else 0,
                    provenance=provenance,
                )

            # Bitwise and shift operations only apply to integers.
            elif operation in _INTEGER_BITWISE_BINARY_OPERATIONS:
                if not isinstance(left, IntLiteral) or not isinstance(
                    right, IntLiteral
                ):
                    return None
                return _fold_integer_bitwise_operation(
                    operation, left.value, right.value, provenance
                )

            # Floor-division, modulo, and ordered comparisons require real
            # (non-complex) operands.
            elif operation in _REAL_ONLY_BINARY_OPERATIONS:
                if isinstance(left, ComplexLiteral) or isinstance(
                    right, ComplexLiteral
                ):
                    return None
                return _fold_real_only_binary_operation(
                    operation, left, right, provenance
                )
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            return None
        return None
