"""Tests for the FhY constant-safety validator."""

import pytest
from fhy_core import (
    Identifier,
    Provenance,
    TypeQualifier,
    ValidationFailedError,
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
    UnaryExpression,
    UnaryOperation,
)
from fhy_lang.ast.passes import ConstantSafetyValidator

from .utils import (
    make_argument,
    make_module_with_statement,
    make_procedure,
    run_validator,
)


def _make_binary_assignment(
    target: Identifier,
    source: Identifier,
    operation: BinaryOperation,
    divisor_expression,
) -> ExpressionStatement:
    return ExpressionStatement(
        left=IdentifierExpression(identifier=target, provenance=Provenance.unknown()),
        right=BinaryExpression(
            operation=operation,
            left=IdentifierExpression(
                identifier=source, provenance=Provenance.unknown()
            ),
            right=divisor_expression,
            provenance=Provenance.unknown(),
        ),
        provenance=Provenance.unknown(),
    )


def _build_main_with_division_by(
    divisor_expression, operation: BinaryOperation, int32
) -> Procedure:
    a, b = Identifier("a"), Identifier("b")
    return make_procedure(
        name=Identifier("main"),
        args=(
            make_argument(a, TypeQualifier.INPUT, int32),
            make_argument(b, TypeQualifier.OUTPUT, int32),
        ),
        body=(_make_binary_assignment(b, a, operation, divisor_expression),),
    )


@pytest.mark.parametrize(
    "operation",
    [
        BinaryOperation.DIVISION,
        BinaryOperation.FLOORDIV,
        BinaryOperation.MODULO,
    ],
)
def test_zero_divisor_int_literal_raises(int32, operation):
    """Test that `a / 0`, `a // 0`, and `a % 0` are flagged."""
    procedure_ast = _build_main_with_division_by(
        IntLiteral(value=0, provenance=Provenance.unknown()),
        operation,
        int32,
    )
    program_ast = make_module_with_statement(procedure_ast)

    with pytest.raises(ValidationFailedError, match="semantic error"):
        run_validator(ConstantSafetyValidator(), program_ast)


def test_zero_divisor_float_literal_raises(int32):
    """Test that `a / 0.0` is flagged."""
    procedure_ast = _build_main_with_division_by(
        FloatLiteral(value=0.0, provenance=Provenance.unknown()),
        BinaryOperation.DIVISION,
        int32,
    )
    program_ast = make_module_with_statement(procedure_ast)

    with pytest.raises(ValidationFailedError, match="semantic error"):
        run_validator(ConstantSafetyValidator(), program_ast)


def test_zero_divisor_complex_literal_raises(int32):
    """Test that `a / (0+0j)` is flagged."""
    procedure_ast = _build_main_with_division_by(
        ComplexLiteral(value=complex(0, 0), provenance=Provenance.unknown()),
        BinaryOperation.DIVISION,
        int32,
    )
    program_ast = make_module_with_statement(procedure_ast)

    with pytest.raises(ValidationFailedError, match="semantic error"):
        run_validator(ConstantSafetyValidator(), program_ast)


def test_negated_zero_divisor_raises(int32):
    """Test that `-0` (a unary negation of zero) is also flagged."""
    negated_zero = UnaryExpression(
        operation=UnaryOperation.NEGATION,
        expression=IntLiteral(value=0, provenance=Provenance.unknown()),
        provenance=Provenance.unknown(),
    )
    procedure_ast = _build_main_with_division_by(
        negated_zero,
        BinaryOperation.DIVISION,
        int32,
    )
    program_ast = make_module_with_statement(procedure_ast)

    with pytest.raises(ValidationFailedError, match="semantic error"):
        run_validator(ConstantSafetyValidator(), program_ast)


def test_doubly_negated_zero_divisor_raises(int32):
    """Test that `--0` is still recognized as a literal zero."""
    inner = UnaryExpression(
        operation=UnaryOperation.NEGATION,
        expression=IntLiteral(value=0, provenance=Provenance.unknown()),
        provenance=Provenance.unknown(),
    )
    outer = UnaryExpression(
        operation=UnaryOperation.NEGATION,
        expression=inner,
        provenance=Provenance.unknown(),
    )
    procedure_ast = _build_main_with_division_by(outer, BinaryOperation.DIVISION, int32)
    program_ast = make_module_with_statement(procedure_ast)

    with pytest.raises(ValidationFailedError, match="semantic error"):
        run_validator(ConstantSafetyValidator(), program_ast)


def test_nonzero_literal_divisor_does_not_raise(int32):
    """Test that `a / 1` passes validation."""
    procedure_ast = _build_main_with_division_by(
        IntLiteral(value=1, provenance=Provenance.unknown()),
        BinaryOperation.DIVISION,
        int32,
    )
    program_ast = make_module_with_statement(procedure_ast)

    run_validator(ConstantSafetyValidator(), program_ast)


def test_identifier_divisor_does_not_raise(int32):
    """Test that a division by an identifier (non-literal) passes validation."""
    a, b, c = Identifier("a"), Identifier("b"), Identifier("c")
    procedure_ast = make_procedure(
        name=Identifier("main"),
        args=(
            make_argument(a, TypeQualifier.INPUT, int32),
            make_argument(c, TypeQualifier.INPUT, int32),
            make_argument(b, TypeQualifier.OUTPUT, int32),
        ),
        body=(
            _make_binary_assignment(
                b,
                a,
                BinaryOperation.DIVISION,
                IdentifierExpression(identifier=c, provenance=Provenance.unknown()),
            ),
        ),
    )
    program_ast = make_module_with_statement(procedure_ast)

    run_validator(ConstantSafetyValidator(), program_ast)


def test_zero_on_left_hand_side_does_not_raise(int32):
    """Test that `0 / a` (zero on the LHS) is not flagged."""
    a, b = Identifier("a"), Identifier("b")
    procedure_ast = make_procedure(
        name=Identifier("main"),
        args=(
            make_argument(a, TypeQualifier.INPUT, int32),
            make_argument(b, TypeQualifier.OUTPUT, int32),
        ),
        body=(
            ExpressionStatement(
                left=IdentifierExpression(
                    identifier=b, provenance=Provenance.unknown()
                ),
                right=BinaryExpression(
                    operation=BinaryOperation.DIVISION,
                    left=IntLiteral(value=0, provenance=Provenance.unknown()),
                    right=IdentifierExpression(
                        identifier=a, provenance=Provenance.unknown()
                    ),
                    provenance=Provenance.unknown(),
                ),
                provenance=Provenance.unknown(),
            ),
        ),
    )
    program_ast = make_module_with_statement(procedure_ast)

    run_validator(ConstantSafetyValidator(), program_ast)


def test_non_division_operation_with_zero_right_does_not_raise(int32):
    """Test that `a + 0`, `a * 0`, etc. are not flagged by this pass."""
    for operation in (
        BinaryOperation.ADDITION,
        BinaryOperation.SUBTRACTION,
        BinaryOperation.MULTIPLICATION,
    ):
        procedure_ast = _build_main_with_division_by(
            IntLiteral(value=0, provenance=Provenance.unknown()),
            operation,
            int32,
        )
        program_ast = make_module_with_statement(procedure_ast)
        run_validator(ConstantSafetyValidator(), program_ast)
