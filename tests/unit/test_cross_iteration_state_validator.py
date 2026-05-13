"""Tests the cross-iteration-state validator AST pass."""

import pytest
from fhy_core import (
    CoreDataType,
    Identifier,
    IndexType,
    LiteralExpression,
    NumericalType,
    PrimitiveDataType,
    TypeQualifier,
    ValidationFailedError,
)
from fhy_core import (
    IdentifierExpression as CoreIdentifierExpression,
)

from fhy_lang.ast import (
    Argument,
    ArrayAccessExpression,
    BinaryExpression,
    BinaryOperation,
    DeclarationStatement,
    ExpressionStatement,
    ForAllStatement,
    IdentifierExpression,
    IntLiteral,
    Module,
    Procedure,
    QualifiedType,
)
from fhy_lang.ast.passes import CrossIterationStateValidator

from .utils import run_validator


def _int32_vec(dim_id: Identifier) -> NumericalType:
    return NumericalType(
        PrimitiveDataType(CoreDataType.INT32),
        shape=(CoreIdentifierExpression(dim_id),),
    )


def _int32_mat(d1: Identifier, d2: Identifier) -> NumericalType:
    return NumericalType(
        PrimitiveDataType(CoreDataType.INT32),
        shape=(CoreIdentifierExpression(d1), CoreIdentifierExpression(d2)),
    )


def _index_1_to(upper: Identifier) -> IndexType:
    return IndexType(
        lower_bound=LiteralExpression(1),
        upper_bound=CoreIdentifierExpression(upper),
        stride=None,
    )


def _qt(t, q):
    return QualifiedType(base_type=t, type_qualifier=q)


def _make_proc_with_body(arg_specs, body) -> Module:
    main = Identifier("main")
    return Module(
        statements=(
            Procedure(
                name=main,
                templates=(),
                args=tuple(
                    Argument(name=name, qualified_type=_qt(t, q))
                    for name, q, t in arg_specs
                ),
                body=tuple(body),
            ),
        ),
    )


def test_self_read_with_matching_index_passes():
    """Test ``A[i] = A[i] * 2`` is allowed (no cross-iteration)."""
    a, i, n = Identifier("A"), Identifier("i"), Identifier("N")
    statements = (
        DeclarationStatement(
            variable_name=i,
            variable_type=_qt(_index_1_to(n), TypeQualifier.TEMP),
        ),
        ExpressionStatement(
            left=ArrayAccessExpression(
                array_expression=IdentifierExpression(identifier=a),
                indices=(IdentifierExpression(identifier=i),),
            ),
            right=BinaryExpression(
                operation=BinaryOperation.MULTIPLICATION,
                left=ArrayAccessExpression(
                    array_expression=IdentifierExpression(identifier=a),
                    indices=(IdentifierExpression(identifier=i),),
                ),
                right=IntLiteral(value=2),
            ),
        ),
    )
    module = _make_proc_with_body(
        [(a, TypeQualifier.OUTPUT, _int32_vec(n))], statements
    )

    run_validator(CrossIterationStateValidator(), module)


def test_self_read_with_offset_index_fails():
    """Test ``A[i] = A[i - 1] + 1`` is rejected."""
    a, i, n = Identifier("A"), Identifier("i"), Identifier("N")
    statements = (
        DeclarationStatement(
            variable_name=i,
            variable_type=_qt(_index_1_to(n), TypeQualifier.TEMP),
        ),
        ExpressionStatement(
            left=ArrayAccessExpression(
                array_expression=IdentifierExpression(identifier=a),
                indices=(IdentifierExpression(identifier=i),),
            ),
            right=BinaryExpression(
                operation=BinaryOperation.ADDITION,
                left=ArrayAccessExpression(
                    array_expression=IdentifierExpression(identifier=a),
                    indices=(
                        BinaryExpression(
                            operation=BinaryOperation.SUBTRACTION,
                            left=IdentifierExpression(identifier=i),
                            right=IntLiteral(value=1),
                        ),
                    ),
                ),
                right=IntLiteral(value=1),
            ),
        ),
    )
    module = _make_proc_with_body(
        [(a, TypeQualifier.OUTPUT, _int32_vec(n))], statements
    )

    with pytest.raises(ValidationFailedError, match="cross-iteration"):
        run_validator(CrossIterationStateValidator(), module)


def test_self_read_with_constant_index_fails():
    """Test ``A[i] = A[0] + 1`` is rejected."""
    a, i, n = Identifier("A"), Identifier("i"), Identifier("N")
    statements = (
        DeclarationStatement(
            variable_name=i,
            variable_type=_qt(_index_1_to(n), TypeQualifier.TEMP),
        ),
        ExpressionStatement(
            left=ArrayAccessExpression(
                array_expression=IdentifierExpression(identifier=a),
                indices=(IdentifierExpression(identifier=i),),
            ),
            right=BinaryExpression(
                operation=BinaryOperation.ADDITION,
                left=ArrayAccessExpression(
                    array_expression=IdentifierExpression(identifier=a),
                    indices=(IntLiteral(value=0),),
                ),
                right=IntLiteral(value=1),
            ),
        ),
    )
    module = _make_proc_with_body(
        [(a, TypeQualifier.OUTPUT, _int32_vec(n))], statements
    )

    with pytest.raises(ValidationFailedError, match="cross-iteration"):
        run_validator(CrossIterationStateValidator(), module)


def test_self_read_with_mismatched_multi_dim_index_fails():
    """Test ``A[i, j] = A[j, i]`` (transposition) is rejected."""
    a = Identifier("A")
    i, j = Identifier("i"), Identifier("j")
    n, m = Identifier("N"), Identifier("M")
    statements = (
        DeclarationStatement(
            variable_name=i, variable_type=_qt(_index_1_to(n), TypeQualifier.TEMP)
        ),
        DeclarationStatement(
            variable_name=j, variable_type=_qt(_index_1_to(m), TypeQualifier.TEMP)
        ),
        ExpressionStatement(
            left=ArrayAccessExpression(
                array_expression=IdentifierExpression(identifier=a),
                indices=(
                    IdentifierExpression(identifier=i),
                    IdentifierExpression(identifier=j),
                ),
            ),
            right=ArrayAccessExpression(
                array_expression=IdentifierExpression(identifier=a),
                indices=(
                    IdentifierExpression(identifier=j),
                    IdentifierExpression(identifier=i),
                ),
            ),
        ),
    )
    module = _make_proc_with_body(
        [(a, TypeQualifier.OUTPUT, _int32_mat(n, m))], statements
    )

    with pytest.raises(ValidationFailedError, match="cross-iteration"):
        run_validator(CrossIterationStateValidator(), module)


def test_self_read_with_matching_multi_dim_index_passes():
    """Test ``A[i, j] = A[i, j] + 1`` is allowed (matching indices)."""
    a = Identifier("A")
    i, j = Identifier("i"), Identifier("j")
    n, m = Identifier("N"), Identifier("M")
    statements = (
        DeclarationStatement(
            variable_name=i, variable_type=_qt(_index_1_to(n), TypeQualifier.TEMP)
        ),
        DeclarationStatement(
            variable_name=j, variable_type=_qt(_index_1_to(m), TypeQualifier.TEMP)
        ),
        ExpressionStatement(
            left=ArrayAccessExpression(
                array_expression=IdentifierExpression(identifier=a),
                indices=(
                    IdentifierExpression(identifier=i),
                    IdentifierExpression(identifier=j),
                ),
            ),
            right=BinaryExpression(
                operation=BinaryOperation.ADDITION,
                left=ArrayAccessExpression(
                    array_expression=IdentifierExpression(identifier=a),
                    indices=(
                        IdentifierExpression(identifier=i),
                        IdentifierExpression(identifier=j),
                    ),
                ),
                right=IntLiteral(value=1),
            ),
        ),
    )
    module = _make_proc_with_body(
        [(a, TypeQualifier.OUTPUT, _int32_mat(n, m))], statements
    )

    run_validator(CrossIterationStateValidator(), module)


def test_other_array_read_with_offset_index_passes():
    """Test ``A[i] = B[i - 1] + 1`` is allowed (different array on RHS)."""
    a, b = Identifier("A"), Identifier("B")
    i, n = Identifier("i"), Identifier("N")
    statements = (
        DeclarationStatement(
            variable_name=i, variable_type=_qt(_index_1_to(n), TypeQualifier.TEMP)
        ),
        ExpressionStatement(
            left=ArrayAccessExpression(
                array_expression=IdentifierExpression(identifier=a),
                indices=(IdentifierExpression(identifier=i),),
            ),
            right=BinaryExpression(
                operation=BinaryOperation.ADDITION,
                left=ArrayAccessExpression(
                    array_expression=IdentifierExpression(identifier=b),
                    indices=(
                        BinaryExpression(
                            operation=BinaryOperation.SUBTRACTION,
                            left=IdentifierExpression(identifier=i),
                            right=IntLiteral(value=1),
                        ),
                    ),
                ),
                right=IntLiteral(value=1),
            ),
        ),
    )
    module = _make_proc_with_body(
        [
            (b, TypeQualifier.INPUT, _int32_vec(n)),
            (a, TypeQualifier.OUTPUT, _int32_vec(n)),
        ],
        statements,
    )

    run_validator(CrossIterationStateValidator(), module)


def test_inside_forall_does_not_inhibit_check():
    """Test the rule applies inside ``forall`` too.

    ``forall (i) { A[i] = A[i - 1] + 1; }`` is rejected -- the rule is
    structural, not semantic about ordered iteration.
    """
    a, i, n = Identifier("A"), Identifier("i"), Identifier("N")
    inner = ExpressionStatement(
        left=ArrayAccessExpression(
            array_expression=IdentifierExpression(identifier=a),
            indices=(IdentifierExpression(identifier=i),),
        ),
        right=BinaryExpression(
            operation=BinaryOperation.ADDITION,
            left=ArrayAccessExpression(
                array_expression=IdentifierExpression(identifier=a),
                indices=(
                    BinaryExpression(
                        operation=BinaryOperation.SUBTRACTION,
                        left=IdentifierExpression(identifier=i),
                        right=IntLiteral(value=1),
                    ),
                ),
            ),
            right=IntLiteral(value=1),
        ),
    )
    statements = (
        DeclarationStatement(
            variable_name=i, variable_type=_qt(_index_1_to(n), TypeQualifier.TEMP)
        ),
        ForAllStatement(
            index=IdentifierExpression(identifier=i),
            body=(inner,),
        ),
    )
    module = _make_proc_with_body(
        [(a, TypeQualifier.OUTPUT, _int32_vec(n))], statements
    )

    with pytest.raises(ValidationFailedError, match="cross-iteration"):
        run_validator(CrossIterationStateValidator(), module)


def test_non_indexed_assignment_does_not_trigger_check():
    """Test scalar assignment ``y = expr`` is unaffected by this validator."""
    y, x = Identifier("y"), Identifier("x")
    int32_scalar = NumericalType(PrimitiveDataType(CoreDataType.INT32))
    statements = (
        ExpressionStatement(
            left=IdentifierExpression(identifier=y),
            right=IdentifierExpression(identifier=x),
        ),
    )
    module = _make_proc_with_body(
        [
            (x, TypeQualifier.INPUT, int32_scalar),
            (y, TypeQualifier.OUTPUT, int32_scalar),
        ],
        statements,
    )

    run_validator(CrossIterationStateValidator(), module)
