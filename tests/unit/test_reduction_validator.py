"""Tests the reduction validator AST pass."""

import pytest
from fhy.lang.ast import (
    Argument,
    ArrayAccessExpression,
    BinaryExpression,
    BinaryOperation,
    DeclarationStatement,
    ExpressionStatement,
    FhYSemanticsError,
    FhYStructuralError,
    FhYTypeError,
    FunctionExpression,
    IdentifierExpression,
    IntLiteral,
    Module,
    Procedure,
    QualifiedType,
)
from fhy.lang.ast.passes import (
    build_symbol_table,
    validate_reductions,
)
from fhy_core import (
    Identifier,
    IndexType,
    LiteralExpression,
    NumericalType,
    PassExecutionError,
    Provenance,
    TypeQualifier,
)


@pytest.fixture
def dummy_index_type() -> IndexType:
    return IndexType(
        lower_bound=LiteralExpression(1),
        upper_bound=LiteralExpression(10),
        stride=None,
    )


def test_empty_program():
    """Tests validation of an empty program."""
    program_ast = Module(provenance=Provenance.unknown())
    symbol_table = build_symbol_table(program_ast)

    validate_reductions(program_ast, symbol_table)


def test_valid_reduction_with_index_variable(
    int32: NumericalType, dummy_index_type: IndexType
):
    """Tests a reduction call with an index variable."""
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
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                ),
                body=(
                    DeclarationStatement(
                        variable_name=k,
                        variable_type=QualifiedType(
                            base_type=dummy_index_type,
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    DeclarationStatement(
                        variable_name=t,
                        variable_type=QualifiedType(
                            base_type=int32,
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    ExpressionStatement(
                        left=IdentifierExpression(
                            identifier=t, provenance=Provenance.unknown()
                        ),
                        right=FunctionExpression(
                            function=IdentifierExpression(
                                identifier=sum_,
                                provenance=Provenance.unknown(),
                            ),
                            indices=(
                                IdentifierExpression(
                                    identifier=k,
                                    provenance=Provenance.unknown(),
                                ),
                            ),
                            args=(
                                ArrayAccessExpression(
                                    array_expression=IdentifierExpression(
                                        identifier=a,
                                        provenance=Provenance.unknown(),
                                    ),
                                    indices=(
                                        IdentifierExpression(
                                            identifier=k,
                                            provenance=Provenance.unknown(),
                                        ),
                                    ),
                                    provenance=Provenance.unknown(),
                                ),
                            ),
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                ),
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )
    symbol_table = build_symbol_table(program_ast)

    validate_reductions(program_ast, symbol_table)


def test_valid_function_expression_without_indices(int32: NumericalType):
    """Tests a non-reduction function call is not rejected."""
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
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                ),
                body=(
                    DeclarationStatement(
                        variable_name=t,
                        variable_type=QualifiedType(
                            base_type=int32,
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    ExpressionStatement(
                        left=IdentifierExpression(
                            identifier=t, provenance=Provenance.unknown()
                        ),
                        right=FunctionExpression(
                            function=IdentifierExpression(
                                identifier=foo,
                                provenance=Provenance.unknown(),
                            ),
                            indices=(),
                            args=(
                                IdentifierExpression(
                                    identifier=a,
                                    provenance=Provenance.unknown(),
                                ),
                            ),
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                ),
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )
    symbol_table = build_symbol_table(program_ast)

    validate_reductions(program_ast, symbol_table)


def test_fails_with_non_identifier_reduction_index(
    int32: NumericalType, dummy_index_type: IndexType
):
    """Tests failure when a reduction index is not an identifier expression."""
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
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                ),
                body=(
                    DeclarationStatement(
                        variable_name=t,
                        variable_type=QualifiedType(
                            base_type=int32,
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    ExpressionStatement(
                        left=IdentifierExpression(
                            identifier=t, provenance=Provenance.unknown()
                        ),
                        right=FunctionExpression(
                            function=IdentifierExpression(
                                identifier=sum_,
                                provenance=Provenance.unknown(),
                            ),
                            indices=(
                                BinaryExpression(
                                    operation=BinaryOperation.ADDITION,
                                    left=IntLiteral(
                                        value=0,
                                        provenance=Provenance.unknown(),
                                    ),
                                    right=IntLiteral(
                                        value=1,
                                        provenance=Provenance.unknown(),
                                    ),
                                    provenance=Provenance.unknown(),
                                ),
                            ),
                            args=(
                                IdentifierExpression(
                                    identifier=a,
                                    provenance=Provenance.unknown(),
                                ),
                            ),
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                ),
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(PassExecutionError, match=FhYStructuralError.__name__):
        validate_reductions(program_ast, symbol_table)


def test_fails_with_non_index_variable_reduction_index(int32: NumericalType):
    """Tests failure when a reduction index identifier is not an index variable."""
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
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                ),
                body=(
                    DeclarationStatement(
                        variable_name=t,
                        variable_type=QualifiedType(
                            base_type=int32,
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    ExpressionStatement(
                        left=IdentifierExpression(
                            identifier=t, provenance=Provenance.unknown()
                        ),
                        right=FunctionExpression(
                            function=IdentifierExpression(
                                identifier=sum_,
                                provenance=Provenance.unknown(),
                            ),
                            indices=(
                                IdentifierExpression(
                                    identifier=t,
                                    provenance=Provenance.unknown(),
                                ),
                            ),
                            args=(
                                IdentifierExpression(
                                    identifier=a,
                                    provenance=Provenance.unknown(),
                                ),
                            ),
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                ),
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(PassExecutionError, match=FhYTypeError.__name__):
        validate_reductions(program_ast, symbol_table)


def test_fails_with_non_distinct_reduction_indices(
    int32: NumericalType, dummy_index_type: IndexType
):
    """Tests failure when reduction indices repeat."""
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
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                ),
                body=(
                    DeclarationStatement(
                        variable_name=k,
                        variable_type=QualifiedType(
                            base_type=dummy_index_type,
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    DeclarationStatement(
                        variable_name=t,
                        variable_type=QualifiedType(
                            base_type=int32,
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    ExpressionStatement(
                        left=IdentifierExpression(
                            identifier=t, provenance=Provenance.unknown()
                        ),
                        right=FunctionExpression(
                            function=IdentifierExpression(
                                identifier=sum_,
                                provenance=Provenance.unknown(),
                            ),
                            indices=(
                                IdentifierExpression(
                                    identifier=k,
                                    provenance=Provenance.unknown(),
                                ),
                                IdentifierExpression(
                                    identifier=k,
                                    provenance=Provenance.unknown(),
                                ),
                            ),
                            args=(
                                IdentifierExpression(
                                    identifier=a,
                                    provenance=Provenance.unknown(),
                                ),
                            ),
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                ),
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(PassExecutionError, match=FhYSemanticsError.__name__):
        validate_reductions(program_ast, symbol_table)


def test_fails_with_unused_reduction_index(
    int32: NumericalType, dummy_index_type: IndexType
):
    """Tests failure when a reduction index is not used in the reduction args."""
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
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                ),
                body=(
                    DeclarationStatement(
                        variable_name=k,
                        variable_type=QualifiedType(
                            base_type=dummy_index_type,
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    DeclarationStatement(
                        variable_name=t,
                        variable_type=QualifiedType(
                            base_type=int32,
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    ExpressionStatement(
                        left=IdentifierExpression(
                            identifier=t, provenance=Provenance.unknown()
                        ),
                        right=FunctionExpression(
                            function=IdentifierExpression(
                                identifier=sum_,
                                provenance=Provenance.unknown(),
                            ),
                            indices=(
                                IdentifierExpression(
                                    identifier=k,
                                    provenance=Provenance.unknown(),
                                ),
                            ),
                            args=(
                                IdentifierExpression(
                                    identifier=a,
                                    provenance=Provenance.unknown(),
                                ),
                            ),
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                ),
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(PassExecutionError, match=FhYSemanticsError.__name__):
        validate_reductions(program_ast, symbol_table)


def test_fails_with_reduction_zero_args(
    int32: NumericalType, dummy_index_type: IndexType
):
    """Tests failure when a reduction has no arguments."""
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
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    DeclarationStatement(
                        variable_name=t,
                        variable_type=QualifiedType(
                            base_type=int32,
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    ExpressionStatement(
                        left=IdentifierExpression(
                            identifier=t, provenance=Provenance.unknown()
                        ),
                        right=FunctionExpression(
                            function=IdentifierExpression(
                                identifier=sum_,
                                provenance=Provenance.unknown(),
                            ),
                            indices=(
                                IdentifierExpression(
                                    identifier=k,
                                    provenance=Provenance.unknown(),
                                ),
                            ),
                            args=(),
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                ),
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(PassExecutionError, match=FhYStructuralError.__name__):
        validate_reductions(program_ast, symbol_table)


def test_fails_with_reduction_multiple_args(
    int32: NumericalType, dummy_index_type: IndexType
):
    """Tests failure when a reduction has more than one argument."""
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
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    Argument(
                        name=b,
                        qualified_type=QualifiedType(
                            base_type=int32,
                            type_qualifier=TypeQualifier.INPUT,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                ),
                body=(
                    DeclarationStatement(
                        variable_name=k,
                        variable_type=QualifiedType(
                            base_type=dummy_index_type,
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    DeclarationStatement(
                        variable_name=t,
                        variable_type=QualifiedType(
                            base_type=int32,
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    ExpressionStatement(
                        left=IdentifierExpression(
                            identifier=t, provenance=Provenance.unknown()
                        ),
                        right=FunctionExpression(
                            function=IdentifierExpression(
                                identifier=sum_,
                                provenance=Provenance.unknown(),
                            ),
                            indices=(
                                IdentifierExpression(
                                    identifier=k,
                                    provenance=Provenance.unknown(),
                                ),
                            ),
                            args=(
                                ArrayAccessExpression(
                                    array_expression=IdentifierExpression(
                                        identifier=a,
                                        provenance=Provenance.unknown(),
                                    ),
                                    indices=(
                                        IdentifierExpression(
                                            identifier=k,
                                            provenance=Provenance.unknown(),
                                        ),
                                    ),
                                    provenance=Provenance.unknown(),
                                ),
                                ArrayAccessExpression(
                                    array_expression=IdentifierExpression(
                                        identifier=b,
                                        provenance=Provenance.unknown(),
                                    ),
                                    indices=(
                                        IdentifierExpression(
                                            identifier=k,
                                            provenance=Provenance.unknown(),
                                        ),
                                    ),
                                    provenance=Provenance.unknown(),
                                ),
                            ),
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                ),
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(PassExecutionError, match=FhYStructuralError.__name__):
        validate_reductions(program_ast, symbol_table)
