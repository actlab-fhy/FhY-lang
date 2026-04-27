"""Tests for the FhY algebraic simplification optimization pass."""

from fhy_core import (
    Identifier,
    Provenance,
    TypeQualifier,
)

from fhy_lang.lang.ast import (
    BinaryExpression,
    BinaryOperation,
    ExpressionStatement,
    FunctionExpression,
    IdentifierExpression,
    IntLiteral,
    Procedure,
    UnaryExpression,
    UnaryOperation,
)
from fhy_lang.lang.ast.passes import AlgebraicSimplificationPass

from .utils import (
    make_argument,
    make_module_with_statement,
    make_procedure,
)


def _make_int_literal(value: int) -> IntLiteral:
    return IntLiteral(value=value, provenance=Provenance.unknown())


def _make_identifier_expr(name: Identifier) -> IdentifierExpression:
    return IdentifierExpression(identifier=name, provenance=Provenance.unknown())


def _make_binary_expr(
    left: IdentifierExpression
    | BinaryExpression
    | IntLiteral
    | UnaryExpression
    | FunctionExpression,
    right: IdentifierExpression
    | BinaryExpression
    | IntLiteral
    | UnaryExpression
    | FunctionExpression,
    operation: BinaryOperation = BinaryOperation.ADDITION,
) -> BinaryExpression:
    return BinaryExpression(
        operation=operation,
        left=left,
        right=right,
        provenance=Provenance.unknown(),
    )


def _make_unary_expr(operation: UnaryOperation, inner) -> UnaryExpression:
    return UnaryExpression(
        operation=operation,
        expression=inner,
        provenance=Provenance.unknown(),
    )


def _wrap_as_main_rhs(expression, int32) -> Procedure:
    """Build `proc main(input int32 x, output int32 b) { b = expression; }`."""
    x = Identifier("x")
    b = Identifier("b")
    return make_procedure(
        name=Identifier("main"),
        args=(
            make_argument(x, TypeQualifier.INPUT, int32),
            make_argument(b, TypeQualifier.OUTPUT, int32),
        ),
        body=(
            ExpressionStatement(
                left=_make_identifier_expr(b),
                right=expression,
                provenance=Provenance.unknown(),
            ),
        ),
    )


def _get_rhs(module):
    return module.statements[0].body[0].right


def test_simplifies_addition_of_zero_from_the_right(int32):
    """Test that `x + 0` folds to `x`."""
    x = Identifier("x")
    procedure_ast = _wrap_as_main_rhs(
        _make_binary_expr(_make_identifier_expr(x), _make_int_literal(0)), int32
    )
    program_ast = make_module_with_statement(procedure_ast)

    optimized = AlgebraicSimplificationPass()(program_ast)

    rhs = _get_rhs(optimized)
    assert isinstance(rhs, IdentifierExpression)
    assert rhs.identifier == x


def test_simplifies_addition_of_zero_from_the_left(int32):
    """Test that `0 + x` folds to `x`."""
    x = Identifier("x")
    procedure_ast = _wrap_as_main_rhs(
        _make_binary_expr(_make_int_literal(0), _make_identifier_expr(x)), int32
    )
    program_ast = make_module_with_statement(procedure_ast)

    optimized = AlgebraicSimplificationPass()(program_ast)

    rhs = _get_rhs(optimized)
    assert isinstance(rhs, IdentifierExpression)
    assert rhs.identifier == x


def test_simplifies_subtraction_of_zero(int32):
    """Test that `x - 0` folds to `x`."""
    x = Identifier("x")
    procedure_ast = _wrap_as_main_rhs(
        _make_binary_expr(
            _make_identifier_expr(x), _make_int_literal(0), BinaryOperation.SUBTRACTION
        ),
        int32,
    )
    program_ast = make_module_with_statement(procedure_ast)

    optimized = AlgebraicSimplificationPass()(program_ast)

    rhs = _get_rhs(optimized)
    assert isinstance(rhs, IdentifierExpression)
    assert rhs.identifier == x


def test_simplifies_multiplication_by_one(int32):
    """Test that `x * 1` and `1 * x` fold to `x`."""
    x = Identifier("x")
    right_one_procedure_ast = _wrap_as_main_rhs(
        _make_binary_expr(
            _make_identifier_expr(x),
            _make_int_literal(1),
            BinaryOperation.MULTIPLICATION,
        ),
        int32,
    )
    left_one_procedure_ast = _wrap_as_main_rhs(
        _make_binary_expr(
            _make_int_literal(1),
            _make_identifier_expr(x),
            BinaryOperation.MULTIPLICATION,
        ),
        int32,
    )
    right_module = make_module_with_statement(right_one_procedure_ast)
    left_module = make_module_with_statement(left_one_procedure_ast)

    right_rhs = _get_rhs(AlgebraicSimplificationPass()(right_module))
    left_rhs = _get_rhs(AlgebraicSimplificationPass()(left_module))

    assert isinstance(right_rhs, IdentifierExpression)
    assert right_rhs.identifier == x
    assert isinstance(left_rhs, IdentifierExpression)
    assert left_rhs.identifier == x


def test_simplifies_multiplication_by_zero_when_side_effect_free(int32):
    """Test that `x * 0` folds to `0` when `x` is side-effect free."""
    x = Identifier("x")
    procedure_ast = _wrap_as_main_rhs(
        _make_binary_expr(
            _make_identifier_expr(x),
            _make_int_literal(0),
            BinaryOperation.MULTIPLICATION,
        ),
        int32,
    )
    program_ast = make_module_with_statement(procedure_ast)

    optimized = AlgebraicSimplificationPass()(program_ast)

    rhs = _get_rhs(optimized)
    assert isinstance(rhs, IntLiteral)
    assert rhs.value == 0


def test_does_not_simplify_multiplication_by_zero_when_operand_has_call(int32):
    """Test that `foo(x) * 0` is left alone to preserve the call."""
    x = Identifier("x")
    foo = Identifier("foo")
    call = FunctionExpression(
        function=_make_identifier_expr(foo),
        args=(_make_identifier_expr(x),),
        provenance=Provenance.unknown(),
    )
    procedure_ast = _wrap_as_main_rhs(
        _make_binary_expr(call, _make_int_literal(0), BinaryOperation.MULTIPLICATION),
        int32,
    )
    program_ast = make_module_with_statement(procedure_ast)

    optimized = AlgebraicSimplificationPass()(program_ast)

    rhs = _get_rhs(optimized)
    assert isinstance(rhs, BinaryExpression)
    assert rhs.operation == BinaryOperation.MULTIPLICATION


def test_simplifies_division_by_one(int32):
    """Test that `x / 1` folds to `x`."""
    x = Identifier("x")
    procedure_ast = _wrap_as_main_rhs(
        _make_binary_expr(
            _make_identifier_expr(x), _make_int_literal(1), BinaryOperation.DIVISION
        ),
        int32,
    )
    program_ast = make_module_with_statement(procedure_ast)

    optimized = AlgebraicSimplificationPass()(program_ast)

    rhs = _get_rhs(optimized)
    assert isinstance(rhs, IdentifierExpression)
    assert rhs.identifier == x


def test_does_not_simplify_division_by_zero(int32):
    """Test that `x / 0` is NOT touched (left to the constant-safety validator)."""
    x = Identifier("x")
    procedure_ast = _wrap_as_main_rhs(
        _make_binary_expr(
            _make_identifier_expr(x), _make_int_literal(0), BinaryOperation.DIVISION
        ),
        int32,
    )
    program_ast = make_module_with_statement(procedure_ast)

    optimized = AlgebraicSimplificationPass()(program_ast)

    rhs = _get_rhs(optimized)
    assert isinstance(rhs, BinaryExpression)
    assert rhs.operation == BinaryOperation.DIVISION


def test_simplifies_power_of_one(int32):
    """Test that `x ** 1` folds to `x`."""
    x = Identifier("x")
    procedure_ast = _wrap_as_main_rhs(
        _make_binary_expr(
            _make_identifier_expr(x), _make_int_literal(1), BinaryOperation.POWER
        ),
        int32,
    )
    program_ast = make_module_with_statement(procedure_ast)

    optimized = AlgebraicSimplificationPass()(program_ast)

    rhs = _get_rhs(optimized)
    assert isinstance(rhs, IdentifierExpression)
    assert rhs.identifier == x


def test_simplifies_power_of_zero_when_side_effect_free(int32):
    """Test that `x ** 0` folds to `IntLiteral(1)` when `x` has no side effects."""
    x = Identifier("x")
    procedure_ast = _wrap_as_main_rhs(
        _make_binary_expr(
            _make_identifier_expr(x), _make_int_literal(0), BinaryOperation.POWER
        ),
        int32,
    )
    program_ast = make_module_with_statement(procedure_ast)

    optimized = AlgebraicSimplificationPass()(program_ast)

    rhs = _get_rhs(optimized)
    assert isinstance(rhs, IntLiteral)
    assert rhs.value == 1


def test_simplifies_modulo_by_one_when_side_effect_free(int32):
    """Test that `x % 1` folds to `IntLiteral(0)` when `x` has no side effects."""
    x = Identifier("x")
    procedure_ast = _wrap_as_main_rhs(
        _make_binary_expr(
            _make_identifier_expr(x), _make_int_literal(1), BinaryOperation.MODULO
        ),
        int32,
    )
    program_ast = make_module_with_statement(procedure_ast)

    optimized = AlgebraicSimplificationPass()(program_ast)

    rhs = _get_rhs(optimized)
    assert isinstance(rhs, IntLiteral)
    assert rhs.value == 0


def test_simplifies_double_negation(int32):
    """Test that `-(-x)` folds to `x`."""
    x = Identifier("x")
    procedure_ast = _wrap_as_main_rhs(
        _make_unary_expr(
            UnaryOperation.NEGATION,
            _make_unary_expr(UnaryOperation.NEGATION, _make_identifier_expr(x)),
        ),
        int32,
    )
    program_ast = make_module_with_statement(procedure_ast)

    optimized = AlgebraicSimplificationPass()(program_ast)

    rhs = _get_rhs(optimized)
    assert isinstance(rhs, IdentifierExpression)
    assert rhs.identifier == x


def test_simplifies_double_bitwise_not(int32):
    """Test that `~~x` folds to `x`."""
    x = Identifier("x")
    procedure_ast = _wrap_as_main_rhs(
        _make_unary_expr(
            UnaryOperation.BITWISE_NOT,
            _make_unary_expr(UnaryOperation.BITWISE_NOT, _make_identifier_expr(x)),
        ),
        int32,
    )
    program_ast = make_module_with_statement(procedure_ast)

    optimized = AlgebraicSimplificationPass()(program_ast)

    rhs = _get_rhs(optimized)
    assert isinstance(rhs, IdentifierExpression)
    assert rhs.identifier == x


def test_does_not_simplify_non_matching_pattern(int32):
    """Test that `x + 1` is left alone (no identity applies)."""
    x = Identifier("x")
    procedure_ast = _wrap_as_main_rhs(
        _make_binary_expr(_make_identifier_expr(x), _make_int_literal(1)), int32
    )
    program_ast = make_module_with_statement(procedure_ast)

    optimized = AlgebraicSimplificationPass()(program_ast)

    rhs = _get_rhs(optimized)
    assert isinstance(rhs, BinaryExpression)
    assert rhs.operation == BinaryOperation.ADDITION


def test_simplifies_nested_expression(int32):
    """Test that `(x + 0) * 1` folds to `x` in a single pass."""
    x = Identifier("x")
    procedure_ast = _wrap_as_main_rhs(
        _make_binary_expr(
            _make_binary_expr(_make_identifier_expr(x), _make_int_literal(0)),
            _make_int_literal(1),
            BinaryOperation.MULTIPLICATION,
        ),
        int32,
    )
    program_ast = make_module_with_statement(procedure_ast)

    optimized = AlgebraicSimplificationPass()(program_ast)

    rhs = _get_rhs(optimized)
    assert isinstance(rhs, IdentifierExpression)
    assert rhs.identifier == x


def test_did_change_reports_simplification_activity(int32):
    """Test it reports `did_change=True` when it rewrites, `False` otherwise."""
    x = Identifier("x")
    simplified_procedure_ast = _wrap_as_main_rhs(
        _make_binary_expr(_make_identifier_expr(x), _make_int_literal(0)), int32
    )
    simplified_module = make_module_with_statement(simplified_procedure_ast)

    untouched_procedure_ast = _wrap_as_main_rhs(
        _make_binary_expr(_make_identifier_expr(x), _make_int_literal(7)), int32
    )
    untouched_module = make_module_with_statement(untouched_procedure_ast)

    simplifier = AlgebraicSimplificationPass()
    assert simplifier.execute(simplified_module).changed is True

    untouched_simplifier = AlgebraicSimplificationPass()
    assert untouched_simplifier.execute(untouched_module).changed is False
