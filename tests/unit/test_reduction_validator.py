"""Tests the reduction validator AST pass."""

import pytest
from fhy_core import (
    Identifier,
    IndexType,
    LiteralExpression,
    NumericalType,
    TypeQualifier,
    ValidationFailedError,
)

from fhy_lang.ast import (
    Argument,
    ArrayAccessExpression,
    BinaryExpression,
    BinaryOperation,
    DeclarationStatement,
    ExpressionStatement,
    FunctionExpression,
    IdentifierExpression,
    IntLiteral,
    Module,
    Procedure,
    QualifiedType,
)
from fhy_lang.ast.passes import (
    ReductionValidator,
    build_symbol_table,
)

from .utils import run_validator


@pytest.fixture
def dummy_index_type() -> IndexType:
    return IndexType(
        lower_bound=LiteralExpression(1),
        upper_bound=LiteralExpression(10),
        stride=None,
    )


def test_empty_module(empty_module_ast):
    """Test validation of an empty module."""
    symbol_table = build_symbol_table(empty_module_ast)

    run_validator(ReductionValidator(symbol_table), empty_module_ast)


def test_valid_reduction_with_index_variable(
    int32: NumericalType, dummy_index_type: IndexType
):
    """Test a reduction call with an index variable."""
    main = Identifier("main")
    sum_ = Identifier("sum")
    a, t, k = Identifier("a"), Identifier("t"), Identifier("k")
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(
                    Argument(
                        name=a,
                        qualified_type=QualifiedType(
                            base_type=int32,
                            type_qualifier=TypeQualifier.INPUT,
                        ),
                    ),
                ),
                body=(
                    DeclarationStatement(
                        variable_name=k,
                        variable_type=QualifiedType(
                            base_type=dummy_index_type,
                            type_qualifier=TypeQualifier.TEMP,
                        ),
                    ),
                    DeclarationStatement(
                        variable_name=t,
                        variable_type=QualifiedType(
                            base_type=int32,
                            type_qualifier=TypeQualifier.TEMP,
                        ),
                    ),
                    ExpressionStatement(
                        left=IdentifierExpression(identifier=t),
                        right=FunctionExpression(
                            function=IdentifierExpression(
                                identifier=sum_,
                            ),
                            indices=(
                                IdentifierExpression(
                                    identifier=k,
                                ),
                            ),
                            args=(
                                ArrayAccessExpression(
                                    array_expression=IdentifierExpression(
                                        identifier=a,
                                    ),
                                    indices=(
                                        IdentifierExpression(
                                            identifier=k,
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    symbol_table = build_symbol_table(program_ast)

    run_validator(ReductionValidator(symbol_table), program_ast)


def test_valid_function_expression_without_indices(int32: NumericalType):
    """Test a non-reduction function call is not rejected."""
    main = Identifier("main")
    foo = Identifier("foo")
    a, t = Identifier("a"), Identifier("t")
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(
                    Argument(
                        name=a,
                        qualified_type=QualifiedType(
                            base_type=int32,
                            type_qualifier=TypeQualifier.INPUT,
                        ),
                    ),
                ),
                body=(
                    DeclarationStatement(
                        variable_name=t,
                        variable_type=QualifiedType(
                            base_type=int32,
                            type_qualifier=TypeQualifier.TEMP,
                        ),
                    ),
                    ExpressionStatement(
                        left=IdentifierExpression(identifier=t),
                        right=FunctionExpression(
                            function=IdentifierExpression(
                                identifier=foo,
                            ),
                            indices=(),
                            args=(
                                IdentifierExpression(
                                    identifier=a,
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    symbol_table = build_symbol_table(program_ast)

    run_validator(ReductionValidator(symbol_table), program_ast)


def test_fails_with_non_identifier_reduction_index(
    int32: NumericalType, dummy_index_type: IndexType
):
    """Test failure when a reduction index is not an identifier expression."""
    main = Identifier("main")
    sum_ = Identifier("sum")
    a, t = Identifier("a"), Identifier("t")
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(
                    Argument(
                        name=a,
                        qualified_type=QualifiedType(
                            base_type=int32,
                            type_qualifier=TypeQualifier.INPUT,
                        ),
                    ),
                ),
                body=(
                    DeclarationStatement(
                        variable_name=t,
                        variable_type=QualifiedType(
                            base_type=int32,
                            type_qualifier=TypeQualifier.TEMP,
                        ),
                    ),
                    ExpressionStatement(
                        left=IdentifierExpression(identifier=t),
                        right=FunctionExpression(
                            function=IdentifierExpression(
                                identifier=sum_,
                            ),
                            indices=(
                                BinaryExpression(
                                    operation=BinaryOperation.ADDITION,
                                    left=IntLiteral(
                                        value=0,
                                    ),
                                    right=IntLiteral(
                                        value=1,
                                    ),
                                ),
                            ),
                            args=(
                                IdentifierExpression(
                                    identifier=a,
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(ValidationFailedError, match="structural error"):
        run_validator(ReductionValidator(symbol_table), program_ast)


def test_fails_with_non_index_variable_reduction_index(int32: NumericalType):
    """Test failure when a reduction index identifier is not an index variable."""
    main = Identifier("main")
    sum_ = Identifier("sum")
    a, t = Identifier("a"), Identifier("t")
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(
                    Argument(
                        name=a,
                        qualified_type=QualifiedType(
                            base_type=int32,
                            type_qualifier=TypeQualifier.INPUT,
                        ),
                    ),
                ),
                body=(
                    DeclarationStatement(
                        variable_name=t,
                        variable_type=QualifiedType(
                            base_type=int32,
                            type_qualifier=TypeQualifier.TEMP,
                        ),
                    ),
                    ExpressionStatement(
                        left=IdentifierExpression(identifier=t),
                        right=FunctionExpression(
                            function=IdentifierExpression(
                                identifier=sum_,
                            ),
                            indices=(
                                IdentifierExpression(
                                    identifier=t,
                                ),
                            ),
                            args=(
                                IdentifierExpression(
                                    identifier=a,
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(ValidationFailedError, match="type error"):
        run_validator(ReductionValidator(symbol_table), program_ast)


def test_fails_with_non_distinct_reduction_indices(
    int32: NumericalType, dummy_index_type: IndexType
):
    """Test failure when reduction indices repeat."""
    main = Identifier("main")
    sum_ = Identifier("sum")
    a, t, k = Identifier("a"), Identifier("t"), Identifier("k")
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(
                    Argument(
                        name=a,
                        qualified_type=QualifiedType(
                            base_type=int32,
                            type_qualifier=TypeQualifier.INPUT,
                        ),
                    ),
                ),
                body=(
                    DeclarationStatement(
                        variable_name=k,
                        variable_type=QualifiedType(
                            base_type=dummy_index_type,
                            type_qualifier=TypeQualifier.TEMP,
                        ),
                    ),
                    DeclarationStatement(
                        variable_name=t,
                        variable_type=QualifiedType(
                            base_type=int32,
                            type_qualifier=TypeQualifier.TEMP,
                        ),
                    ),
                    ExpressionStatement(
                        left=IdentifierExpression(identifier=t),
                        right=FunctionExpression(
                            function=IdentifierExpression(
                                identifier=sum_,
                            ),
                            indices=(
                                IdentifierExpression(
                                    identifier=k,
                                ),
                                IdentifierExpression(
                                    identifier=k,
                                ),
                            ),
                            args=(
                                IdentifierExpression(
                                    identifier=a,
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(ValidationFailedError, match="semantic error"):
        run_validator(ReductionValidator(symbol_table), program_ast)


def test_fails_with_unused_reduction_index(
    int32: NumericalType, dummy_index_type: IndexType
):
    """Test failure when a reduction index is not used in the reduction args."""
    main = Identifier("main")
    sum_ = Identifier("sum")
    a, t, k = Identifier("a"), Identifier("t"), Identifier("k")
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(
                    Argument(
                        name=a,
                        qualified_type=QualifiedType(
                            base_type=int32,
                            type_qualifier=TypeQualifier.INPUT,
                        ),
                    ),
                ),
                body=(
                    DeclarationStatement(
                        variable_name=k,
                        variable_type=QualifiedType(
                            base_type=dummy_index_type,
                            type_qualifier=TypeQualifier.TEMP,
                        ),
                    ),
                    DeclarationStatement(
                        variable_name=t,
                        variable_type=QualifiedType(
                            base_type=int32,
                            type_qualifier=TypeQualifier.TEMP,
                        ),
                    ),
                    ExpressionStatement(
                        left=IdentifierExpression(identifier=t),
                        right=FunctionExpression(
                            function=IdentifierExpression(
                                identifier=sum_,
                            ),
                            indices=(
                                IdentifierExpression(
                                    identifier=k,
                                ),
                            ),
                            args=(
                                IdentifierExpression(
                                    identifier=a,
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(ValidationFailedError, match="semantic error"):
        run_validator(ReductionValidator(symbol_table), program_ast)


def test_fails_with_reduction_zero_args(
    int32: NumericalType, dummy_index_type: IndexType
):
    """Test failure when a reduction has no arguments."""
    main = Identifier("main")
    sum_ = Identifier("sum")
    t, k = Identifier("t"), Identifier("k")
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(),
                body=(
                    DeclarationStatement(
                        variable_name=k,
                        variable_type=QualifiedType(
                            base_type=dummy_index_type,
                            type_qualifier=TypeQualifier.TEMP,
                        ),
                    ),
                    DeclarationStatement(
                        variable_name=t,
                        variable_type=QualifiedType(
                            base_type=int32,
                            type_qualifier=TypeQualifier.TEMP,
                        ),
                    ),
                    ExpressionStatement(
                        left=IdentifierExpression(identifier=t),
                        right=FunctionExpression(
                            function=IdentifierExpression(
                                identifier=sum_,
                            ),
                            indices=(
                                IdentifierExpression(
                                    identifier=k,
                                ),
                            ),
                            args=(),
                        ),
                    ),
                ),
            ),
        ),
    )
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(ValidationFailedError, match="structural error"):
        run_validator(ReductionValidator(symbol_table), program_ast)


def test_fails_with_reduction_multiple_args(
    int32: NumericalType, dummy_index_type: IndexType
):
    """Test failure when a reduction has more than one argument."""
    main = Identifier("main")
    sum_ = Identifier("sum")
    a, b, t, k = (
        Identifier("a"),
        Identifier("b"),
        Identifier("t"),
        Identifier("k"),
    )
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(
                    Argument(
                        name=a,
                        qualified_type=QualifiedType(
                            base_type=int32,
                            type_qualifier=TypeQualifier.INPUT,
                        ),
                    ),
                    Argument(
                        name=b,
                        qualified_type=QualifiedType(
                            base_type=int32,
                            type_qualifier=TypeQualifier.INPUT,
                        ),
                    ),
                ),
                body=(
                    DeclarationStatement(
                        variable_name=k,
                        variable_type=QualifiedType(
                            base_type=dummy_index_type,
                            type_qualifier=TypeQualifier.TEMP,
                        ),
                    ),
                    DeclarationStatement(
                        variable_name=t,
                        variable_type=QualifiedType(
                            base_type=int32,
                            type_qualifier=TypeQualifier.TEMP,
                        ),
                    ),
                    ExpressionStatement(
                        left=IdentifierExpression(identifier=t),
                        right=FunctionExpression(
                            function=IdentifierExpression(
                                identifier=sum_,
                            ),
                            indices=(
                                IdentifierExpression(
                                    identifier=k,
                                ),
                            ),
                            args=(
                                ArrayAccessExpression(
                                    array_expression=IdentifierExpression(
                                        identifier=a,
                                    ),
                                    indices=(
                                        IdentifierExpression(
                                            identifier=k,
                                        ),
                                    ),
                                ),
                                ArrayAccessExpression(
                                    array_expression=IdentifierExpression(
                                        identifier=b,
                                    ),
                                    indices=(
                                        IdentifierExpression(
                                            identifier=k,
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(ValidationFailedError, match="structural error"):
        run_validator(ReductionValidator(symbol_table), program_ast)
