"""Tests for the FhY constant folding optimization pass."""

from fhy_core import (
    Identifier,
    Provenance,
    TypeQualifier,
)

from fhy_lang.ast import (
    BinaryExpression,
    BinaryOperation,
    ComplexLiteral,
    ExpressionStatement,
    FloatLiteral,
    IdentifierExpression,
    IntLiteral,
    Procedure,
    TernaryExpression,
    UnaryExpression,
    UnaryOperation,
)
from fhy_lang.ast.passes import ConstantFoldingPass

from .utils import (
    make_argument,
    make_module_with_statement,
    make_procedure,
)


def _int(value: int) -> IntLiteral:
    return IntLiteral(value=value, provenance=Provenance.unknown())


def _float(value: float) -> FloatLiteral:
    return FloatLiteral(value=value, provenance=Provenance.unknown())


def _complex(value: complex) -> ComplexLiteral:
    return ComplexLiteral(value=value, provenance=Provenance.unknown())


def _binary(
    left, right, operation: BinaryOperation = BinaryOperation.ADDITION
) -> BinaryExpression:
    return BinaryExpression(
        operation=operation,
        left=left,
        right=right,
        provenance=Provenance.unknown(),
    )


def _unary(operation: UnaryOperation, inner) -> UnaryExpression:
    return UnaryExpression(
        operation=operation,
        expression=inner,
        provenance=Provenance.unknown(),
    )


def _wrap_as_main_rhs(expression, int32) -> Procedure:
    """Build `proc main(output int32 b) { b = expression; }`."""
    b = Identifier("b")
    return make_procedure(
        name=Identifier("main"),
        args=(make_argument(b, TypeQualifier.OUTPUT, int32),),
        body=(
            ExpressionStatement(
                left=IdentifierExpression(
                    identifier=b, provenance=Provenance.unknown()
                ),
                right=expression,
                provenance=Provenance.unknown(),
            ),
        ),
    )


def _get_rhs(module) -> object:
    return module.statements[0].body[0].right


def test_folds_int_addition(int32):
    """Test that `2 + 3` folds to `IntLiteral(5)`."""
    procedure_ast = _wrap_as_main_rhs(_binary(_int(2), _int(3)), int32)
    program_ast = make_module_with_statement(procedure_ast)

    optimized = ConstantFoldingPass()(program_ast)

    rhs = _get_rhs(optimized)
    assert isinstance(rhs, IntLiteral)
    assert rhs.value == 5


def test_folds_mixed_int_and_float(int32):
    """Test that `2 * 3.0` folds to `FloatLiteral(6.0)` (Python-style promotion)."""
    procedure_ast = _wrap_as_main_rhs(
        _binary(_int(2), _float(3.0), BinaryOperation.MULTIPLICATION), int32
    )
    program_ast = make_module_with_statement(procedure_ast)

    optimized = ConstantFoldingPass()(program_ast)

    rhs = _get_rhs(optimized)
    assert isinstance(rhs, FloatLiteral)
    assert rhs.value == 6.0


def test_folds_nested_expression(int32):
    """Test that `(1 + 2) * 3` folds to `IntLiteral(9)` in a single pass."""
    procedure_ast = _wrap_as_main_rhs(
        _binary(
            _binary(_int(1), _int(2)),
            _int(3),
            BinaryOperation.MULTIPLICATION,
        ),
        int32,
    )
    program_ast = make_module_with_statement(procedure_ast)

    optimized = ConstantFoldingPass()(program_ast)

    rhs = _get_rhs(optimized)
    assert isinstance(rhs, IntLiteral)
    assert rhs.value == 9


def test_folds_comparison_to_int_zero_or_one(int32):
    """Test that relational comparisons fold to `IntLiteral(0)` / `IntLiteral(1)`."""
    greater_than_procedure_ast = _wrap_as_main_rhs(
        _binary(_int(3), _int(2), BinaryOperation.GREATER_THAN), int32
    )
    less_than_procedure_ast = _wrap_as_main_rhs(
        _binary(_int(1), _int(2), BinaryOperation.LESS_THAN), int32
    )
    greater_than_module = make_module_with_statement(greater_than_procedure_ast)
    less_than_module = make_module_with_statement(less_than_procedure_ast)

    greater_than_result = _get_rhs(ConstantFoldingPass()(greater_than_module))
    less_than_result = _get_rhs(ConstantFoldingPass()(less_than_module))

    assert isinstance(greater_than_result, IntLiteral)
    assert greater_than_result.value == 1
    assert isinstance(less_than_result, IntLiteral)
    assert less_than_result.value == 1  # 1 < 2 is True -> 1


def test_folds_unary_negation(int32):
    """Test that `-(7)` folds to `IntLiteral(-7)`."""
    procedure_ast = _wrap_as_main_rhs(_unary(UnaryOperation.NEGATION, _int(7)), int32)
    program_ast = make_module_with_statement(procedure_ast)

    optimized = ConstantFoldingPass()(program_ast)

    rhs = _get_rhs(optimized)
    assert isinstance(rhs, IntLiteral)
    assert rhs.value == -7


def test_folds_logical_not(int32):
    """Test that `!(0)` folds to `IntLiteral(1)` and `!(5)` to `IntLiteral(0)`."""
    true_procedure_ast = _wrap_as_main_rhs(
        _unary(UnaryOperation.LOGICAL_NOT, _int(0)), int32
    )
    false_procedure_ast = _wrap_as_main_rhs(
        _unary(UnaryOperation.LOGICAL_NOT, _int(5)), int32
    )
    true_module = make_module_with_statement(true_procedure_ast)
    false_module = make_module_with_statement(false_procedure_ast)

    true_result = _get_rhs(ConstantFoldingPass()(true_module))
    false_result = _get_rhs(ConstantFoldingPass()(false_module))

    assert isinstance(true_result, IntLiteral)
    assert true_result.value == 1
    assert isinstance(false_result, IntLiteral)
    assert false_result.value == 0


def test_ternary_with_literal_condition_collapses(int32):
    """Test that a ternary with a literal true condition collapses to the
    true branch."""
    a, c = Identifier("a"), Identifier("c")
    ternary = TernaryExpression(
        condition=_int(1),  # literal truthy
        true=IdentifierExpression(identifier=a, provenance=Provenance.unknown()),
        false=IdentifierExpression(identifier=c, provenance=Provenance.unknown()),
        provenance=Provenance.unknown(),
    )
    procedure_ast = make_procedure(
        name=Identifier("main"),
        args=(
            make_argument(a, TypeQualifier.INPUT, int32),
            make_argument(c, TypeQualifier.INPUT, int32),
            make_argument(Identifier("b"), TypeQualifier.OUTPUT, int32),
        ),
        body=(
            ExpressionStatement(
                left=IdentifierExpression(
                    identifier=Identifier("b"), provenance=Provenance.unknown()
                ),
                right=ternary,
                provenance=Provenance.unknown(),
            ),
        ),
    )
    program_ast = make_module_with_statement(procedure_ast)

    optimized = ConstantFoldingPass()(program_ast)

    rhs = _get_rhs(optimized)
    assert isinstance(rhs, IdentifierExpression)
    assert rhs.identifier == a


def test_does_not_fold_expression_with_identifier_operand(int32):
    """Test that `x + 2` is not folded (x is not a literal)."""
    x = Identifier("x")
    procedure_ast = make_procedure(
        name=Identifier("main"),
        args=(
            make_argument(x, TypeQualifier.INPUT, int32),
            make_argument(Identifier("b"), TypeQualifier.OUTPUT, int32),
        ),
        body=(
            ExpressionStatement(
                left=IdentifierExpression(
                    identifier=Identifier("b"), provenance=Provenance.unknown()
                ),
                right=_binary(
                    IdentifierExpression(identifier=x, provenance=Provenance.unknown()),
                    _int(2),
                ),
                provenance=Provenance.unknown(),
            ),
        ),
    )
    program_ast = make_module_with_statement(procedure_ast)

    optimized = ConstantFoldingPass()(program_ast)

    rhs = _get_rhs(optimized)
    assert isinstance(rhs, BinaryExpression)


def test_does_not_fold_division_by_literal_zero(int32):
    """Test that division by a literal zero is left for the
    constant-safety validator (not folded into a literal)."""
    procedure_ast = _wrap_as_main_rhs(
        _binary(_int(4), _int(0), BinaryOperation.DIVISION), int32
    )
    program_ast = make_module_with_statement(procedure_ast)

    optimized = ConstantFoldingPass()(program_ast)

    rhs = _get_rhs(optimized)
    assert isinstance(rhs, BinaryExpression)
    assert rhs.operation == BinaryOperation.DIVISION


def test_folds_bitwise_operations_on_int_literals(int32):
    """Test that bitwise operations over integer literals fold."""
    cases = [
        (_binary(_int(0b1100), _int(0b1010), BinaryOperation.BITWISE_AND), 0b1000),
        (_binary(_int(0b1100), _int(0b1010), BinaryOperation.BITWISE_OR), 0b1110),
        (_binary(_int(0b1100), _int(0b1010), BinaryOperation.BITWISE_XOR), 0b0110),
        (_binary(_int(1), _int(3), BinaryOperation.LEFT_SHIFT), 1 << 3),
        (_binary(_int(32), _int(2), BinaryOperation.RIGHT_SHIFT), 32 >> 2),
    ]
    for expression, expected_value in cases:
        procedure_ast = _wrap_as_main_rhs(expression, int32)
        program_ast = make_module_with_statement(procedure_ast)

        optimized = ConstantFoldingPass()(program_ast)

        rhs = _get_rhs(optimized)
        assert isinstance(rhs, IntLiteral)
        assert rhs.value == expected_value


def test_folds_complex_literal_arithmetic(int32):
    """Test that complex literal arithmetic folds."""
    procedure_ast = _wrap_as_main_rhs(
        _binary(_complex(1 + 2j), _complex(3 + 4j), BinaryOperation.ADDITION),
        int32,
    )
    program_ast = make_module_with_statement(procedure_ast)

    optimized = ConstantFoldingPass()(program_ast)

    rhs = _get_rhs(optimized)
    assert isinstance(rhs, ComplexLiteral)
    assert rhs.value == complex(4, 6)


def test_did_change_reports_folding_activity(int32):
    """The pass reports `did_change=True` when it folds and `False` when it doesn't."""
    fold_procedure_ast = _wrap_as_main_rhs(_binary(_int(2), _int(3)), int32)
    fold_module = make_module_with_statement(fold_procedure_ast)

    x = Identifier("x")
    no_fold_procedure_ast = make_procedure(
        name=Identifier("main"),
        args=(
            make_argument(x, TypeQualifier.INPUT, int32),
            make_argument(Identifier("b"), TypeQualifier.OUTPUT, int32),
        ),
        body=(
            ExpressionStatement(
                left=IdentifierExpression(
                    identifier=Identifier("b"), provenance=Provenance.unknown()
                ),
                right=IdentifierExpression(
                    identifier=x, provenance=Provenance.unknown()
                ),
                provenance=Provenance.unknown(),
            ),
        ),
    )
    no_fold_module = make_module_with_statement(no_fold_procedure_ast)

    folding_pass = ConstantFoldingPass()
    folding_result = folding_pass.execute(fold_module)
    assert folding_result.changed is True

    no_folding_pass = ConstantFoldingPass()
    no_folding_result = no_folding_pass.execute(no_fold_module)
    assert no_folding_result.changed is False
