"""Tests the type checker AST pass."""

import pytest
from fhy.lang.ast import (
    Argument,
    ArrayAccessExpression,
    BinaryExpression,
    BinaryOperation,
    DeclarationStatement,
    ExpressionStatement,
    FhYTypeError,
    ForAllStatement,
    FunctionExpression,
    IdentifierExpression,
    IntLiteral,
    Module,
    Operation,
    Procedure,
    QualifiedType,
    ReturnStatement,
    TernaryExpression,
)
from fhy.lang.ast.passes import build_symbol_table, validate_types
from fhy.lang.builtins import BUILTIN_REDUCTION_FUNCTION_IDENTIFIERS
from fhy_core import (
    CoreDataType,
    Identifier,
    IndexType,
    LiteralExpression,
    NumericalType,
    PassExecutionError,
    PrimitiveDataType,
    Provenance,
    TypeQualifier,
)
from fhy_core import (
    IdentifierExpression as CoreIdentifierExpression,
)


def _make_int32_type(shape=()) -> NumericalType:
    return NumericalType(PrimitiveDataType(CoreDataType.INT32), shape=shape)


def _make_float32_type(shape=()) -> NumericalType:
    return NumericalType(PrimitiveDataType(CoreDataType.FLOAT32), shape=shape)


def test_empty_module(empty_module_ast):
    """Test validation of an empty program."""
    symbol_table = build_symbol_table(empty_module_ast)

    validate_types(empty_module_ast, symbol_table)


def test_valid_matmul_like_reduction():
    """Test a matmul-style reduction: C[i,j] = sum[k](A[i,k] * B[k,j])."""
    main = Identifier("matmul")
    m, n, p = Identifier("m"), Identifier("n"), Identifier("p")
    a_, b_, c_ = Identifier("A"), Identifier("B"), Identifier("C")
    i, j, k = Identifier("i"), Identifier("j"), Identifier("k")
    sum_ = BUILTIN_REDUCTION_FUNCTION_IDENTIFIERS["sum"]

    def qt(t, q):
        return QualifiedType(
            base_type=t, type_qualifier=q, provenance=Provenance.unknown()
        )

    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(
                    Argument(
                        name=a_,
                        qualified_type=qt(
                            _make_int32_type(
                                shape=(
                                    CoreIdentifierExpression(m),
                                    CoreIdentifierExpression(n),
                                )
                            ),
                            TypeQualifier.INPUT,
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    Argument(
                        name=b_,
                        qualified_type=qt(
                            _make_int32_type(
                                shape=(
                                    CoreIdentifierExpression(n),
                                    CoreIdentifierExpression(p),
                                )
                            ),
                            TypeQualifier.INPUT,
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    Argument(
                        name=c_,
                        qualified_type=qt(
                            _make_int32_type(
                                shape=(
                                    CoreIdentifierExpression(m),
                                    CoreIdentifierExpression(p),
                                )
                            ),
                            TypeQualifier.OUTPUT,
                        ),
                        provenance=Provenance.unknown(),
                    ),
                ),
                body=(
                    DeclarationStatement(
                        variable_name=i,
                        variable_type=qt(
                            IndexType(
                                lower_bound=LiteralExpression(1),
                                upper_bound=CoreIdentifierExpression(m),
                                stride=None,
                            ),
                            TypeQualifier.TEMP,
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    DeclarationStatement(
                        variable_name=j,
                        variable_type=qt(
                            IndexType(
                                lower_bound=LiteralExpression(1),
                                upper_bound=CoreIdentifierExpression(p),
                                stride=None,
                            ),
                            TypeQualifier.TEMP,
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    DeclarationStatement(
                        variable_name=k,
                        variable_type=qt(
                            IndexType(
                                lower_bound=LiteralExpression(1),
                                upper_bound=CoreIdentifierExpression(n),
                                stride=None,
                            ),
                            TypeQualifier.TEMP,
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    ExpressionStatement(
                        left=ArrayAccessExpression(
                            array_expression=IdentifierExpression(
                                identifier=c_, provenance=Provenance.unknown()
                            ),
                            indices=(
                                IdentifierExpression(
                                    identifier=i,
                                    provenance=Provenance.unknown(),
                                ),
                                IdentifierExpression(
                                    identifier=j,
                                    provenance=Provenance.unknown(),
                                ),
                            ),
                            provenance=Provenance.unknown(),
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
                                BinaryExpression(
                                    operation=BinaryOperation.MULTIPLICATION,
                                    left=ArrayAccessExpression(
                                        array_expression=IdentifierExpression(
                                            identifier=a_,
                                            provenance=Provenance.unknown(),
                                        ),
                                        indices=(
                                            IdentifierExpression(
                                                identifier=i,
                                                provenance=Provenance.unknown(),
                                            ),
                                            IdentifierExpression(
                                                identifier=k,
                                                provenance=Provenance.unknown(),
                                            ),
                                        ),
                                        provenance=Provenance.unknown(),
                                    ),
                                    right=ArrayAccessExpression(
                                        array_expression=IdentifierExpression(
                                            identifier=b_,
                                            provenance=Provenance.unknown(),
                                        ),
                                        indices=(
                                            IdentifierExpression(
                                                identifier=k,
                                                provenance=Provenance.unknown(),
                                            ),
                                            IdentifierExpression(
                                                identifier=j,
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
                        provenance=Provenance.unknown(),
                    ),
                ),
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )
    symbol_table = build_symbol_table(program_ast)

    validate_types(program_ast, symbol_table)


def test_valid_forall_binds_index():
    """Test that a forall-bound index is removed from the free-index set."""
    main = Identifier("main")
    examples, n_ = Identifier("examples"), Identifier("n")
    x_, big_x = Identifier("x"), Identifier("X")
    e, i = Identifier("e"), Identifier("i")

    def qt(t, q):
        return QualifiedType(
            base_type=t, type_qualifier=q, provenance=Provenance.unknown()
        )

    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(
                    Argument(
                        name=big_x,
                        qualified_type=qt(
                            _make_float32_type(
                                shape=(
                                    CoreIdentifierExpression(examples),
                                    CoreIdentifierExpression(n_),
                                )
                            ),
                            TypeQualifier.INPUT,
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    Argument(
                        name=x_,
                        qualified_type=qt(
                            _make_float32_type(shape=(CoreIdentifierExpression(n_),)),
                            TypeQualifier.OUTPUT,
                        ),
                        provenance=Provenance.unknown(),
                    ),
                ),
                body=(
                    DeclarationStatement(
                        variable_name=e,
                        variable_type=qt(
                            IndexType(
                                lower_bound=LiteralExpression(1),
                                upper_bound=CoreIdentifierExpression(examples),
                                stride=None,
                            ),
                            TypeQualifier.TEMP,
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    DeclarationStatement(
                        variable_name=i,
                        variable_type=qt(
                            IndexType(
                                lower_bound=LiteralExpression(1),
                                upper_bound=CoreIdentifierExpression(n_),
                                stride=None,
                            ),
                            TypeQualifier.TEMP,
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    ForAllStatement(
                        index=IdentifierExpression(
                            identifier=e, provenance=Provenance.unknown()
                        ),
                        body=(
                            ExpressionStatement(
                                left=ArrayAccessExpression(
                                    array_expression=IdentifierExpression(
                                        identifier=x_,
                                        provenance=Provenance.unknown(),
                                    ),
                                    indices=(
                                        IdentifierExpression(
                                            identifier=i,
                                            provenance=Provenance.unknown(),
                                        ),
                                    ),
                                    provenance=Provenance.unknown(),
                                ),
                                right=ArrayAccessExpression(
                                    array_expression=IdentifierExpression(
                                        identifier=big_x,
                                        provenance=Provenance.unknown(),
                                    ),
                                    indices=(
                                        IdentifierExpression(
                                            identifier=e,
                                            provenance=Provenance.unknown(),
                                        ),
                                        IdentifierExpression(
                                            identifier=i,
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
            ),
        ),
        provenance=Provenance.unknown(),
    )
    symbol_table = build_symbol_table(program_ast)

    validate_types(program_ast, symbol_table)


def test_valid_operation_return_type():
    """Test that an operation's return statement matches its return type."""
    op = Identifier("op")
    a = Identifier("a")
    program_ast = Module(
        statements=(
            Operation(
                name=op,
                args=(
                    Argument(
                        name=a,
                        qualified_type=QualifiedType(
                            base_type=_make_int32_type(),
                            type_qualifier=TypeQualifier.INPUT,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                ),
                body=(
                    ReturnStatement(
                        expression=IdentifierExpression(
                            identifier=a, provenance=Provenance.unknown()
                        ),
                        provenance=Provenance.unknown(),
                    ),
                ),
                return_type=QualifiedType(
                    base_type=_make_int32_type(),
                    type_qualifier=TypeQualifier.OUTPUT,
                    provenance=Provenance.unknown(),
                ),
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )
    symbol_table = build_symbol_table(program_ast)

    validate_types(program_ast, symbol_table)


def test_valid_operation_call_assignment():
    """Test assigning an operation call's result to a variable with the same type."""
    m_ = Identifier("m")
    op = Identifier("op")
    main = Identifier("main")
    x_, y_ = Identifier("x"), Identifier("y")
    program_ast = Module(
        statements=(
            Operation(
                name=op,
                args=(
                    Argument(
                        name=x_,
                        qualified_type=QualifiedType(
                            base_type=_make_float32_type(
                                shape=(CoreIdentifierExpression(m_),)
                            ),
                            type_qualifier=TypeQualifier.INPUT,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                ),
                body=(),
                return_type=QualifiedType(
                    base_type=_make_float32_type(shape=(CoreIdentifierExpression(m_),)),
                    type_qualifier=TypeQualifier.OUTPUT,
                    provenance=Provenance.unknown(),
                ),
                provenance=Provenance.unknown(),
            ),
            Procedure(
                name=main,
                args=(
                    Argument(
                        name=x_,
                        qualified_type=QualifiedType(
                            base_type=_make_float32_type(
                                shape=(CoreIdentifierExpression(m_),)
                            ),
                            type_qualifier=TypeQualifier.INPUT,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    Argument(
                        name=y_,
                        qualified_type=QualifiedType(
                            base_type=_make_float32_type(
                                shape=(CoreIdentifierExpression(m_),)
                            ),
                            type_qualifier=TypeQualifier.OUTPUT,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                ),
                body=(
                    ExpressionStatement(
                        left=IdentifierExpression(
                            identifier=y_, provenance=Provenance.unknown()
                        ),
                        right=FunctionExpression(
                            function=IdentifierExpression(
                                identifier=op,
                                provenance=Provenance.unknown(),
                            ),
                            args=(
                                IdentifierExpression(
                                    identifier=x_,
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

    validate_types(program_ast, symbol_table)


def test_valid_operation_call_with_indexed_argument():
    """Test that an operation can be called with an indexed argument."""
    main = Identifier("main")
    op = Identifier("op")
    a, b, i, N = Identifier("A"), Identifier("B"), Identifier("i"), Identifier("N")
    operation_ast = Operation(
        name=op,
        args=(
            Argument(
                name=a,
                qualified_type=QualifiedType(
                    base_type=_make_int32_type(),
                    type_qualifier=TypeQualifier.INPUT,
                    provenance=Provenance.unknown(),
                ),
                provenance=Provenance.unknown(),
            ),
        ),
        body=(
            ReturnStatement(
                expression=IdentifierExpression(
                    identifier=a,
                    provenance=Provenance.unknown(),
                ),
                provenance=Provenance.unknown(),
            ),
        ),
        return_type=QualifiedType(
            base_type=_make_int32_type(),
            type_qualifier=TypeQualifier.OUTPUT,
            provenance=Provenance.unknown(),
        ),
        provenance=Provenance.unknown(),
    )
    program_ast = Module(
        statements=(
            operation_ast,
            Procedure(
                name=main,
                args=(
                    Argument(
                        name=a,
                        qualified_type=QualifiedType(
                            base_type=_make_int32_type(
                                shape=(CoreIdentifierExpression(N),)
                            ),
                            type_qualifier=TypeQualifier.INPUT,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    Argument(
                        name=b,
                        qualified_type=QualifiedType(
                            base_type=_make_int32_type(
                                shape=(CoreIdentifierExpression(N),)
                            ),
                            type_qualifier=TypeQualifier.OUTPUT,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                ),
                body=(
                    ExpressionStatement(
                        left=IdentifierExpression(
                            identifier=b, provenance=Provenance.unknown()
                        ),
                        right=FunctionExpression(
                            function=IdentifierExpression(
                                identifier=op,
                                provenance=Provenance.unknown(),
                            ),
                            args=(
                                ArrayAccessExpression(
                                    array_expression=IdentifierExpression(
                                        identifier=a,
                                        provenance=Provenance.unknown(),
                                    ),
                                    indices=(
                                        IdentifierExpression(
                                            identifier=i,
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

    validate_types(program_ast, symbol_table)


def test_fails_on_element_type_mismatch():
    """Test failure when the LHS and RHS element types differ."""
    main = Identifier("main")
    a, b = Identifier("a"), Identifier("b")
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(
                    Argument(
                        name=a,
                        qualified_type=QualifiedType(
                            base_type=_make_int32_type(),
                            type_qualifier=TypeQualifier.INPUT,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    Argument(
                        name=b,
                        qualified_type=QualifiedType(
                            base_type=_make_float32_type(),
                            type_qualifier=TypeQualifier.OUTPUT,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                ),
                body=(
                    ExpressionStatement(
                        left=IdentifierExpression(
                            identifier=b, provenance=Provenance.unknown()
                        ),
                        right=IdentifierExpression(
                            identifier=a, provenance=Provenance.unknown()
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
        validate_types(program_ast, symbol_table)


def test_fails_on_free_index_mismatch():
    """Test failure when the LHS and RHS free-index sets differ."""
    main = Identifier("main")
    m_, n_ = Identifier("m"), Identifier("n")
    x_, y_ = Identifier("x"), Identifier("y")
    i, j = Identifier("i"), Identifier("j")
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(
                    Argument(
                        name=x_,
                        qualified_type=QualifiedType(
                            base_type=_make_int32_type(
                                shape=(
                                    CoreIdentifierExpression(m_),
                                    CoreIdentifierExpression(n_),
                                )
                            ),
                            type_qualifier=TypeQualifier.INPUT,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    Argument(
                        name=y_,
                        qualified_type=QualifiedType(
                            base_type=_make_int32_type(
                                shape=(CoreIdentifierExpression(m_),)
                            ),
                            type_qualifier=TypeQualifier.OUTPUT,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                ),
                body=(
                    DeclarationStatement(
                        variable_name=i,
                        variable_type=QualifiedType(
                            base_type=IndexType(
                                lower_bound=LiteralExpression(1),
                                upper_bound=CoreIdentifierExpression(m_),
                                stride=None,
                            ),
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    DeclarationStatement(
                        variable_name=j,
                        variable_type=QualifiedType(
                            base_type=IndexType(
                                lower_bound=LiteralExpression(1),
                                upper_bound=CoreIdentifierExpression(n_),
                                stride=None,
                            ),
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    ExpressionStatement(
                        left=ArrayAccessExpression(
                            array_expression=IdentifierExpression(
                                identifier=y_, provenance=Provenance.unknown()
                            ),
                            indices=(
                                IdentifierExpression(
                                    identifier=i,
                                    provenance=Provenance.unknown(),
                                ),
                            ),
                            provenance=Provenance.unknown(),
                        ),
                        right=ArrayAccessExpression(
                            array_expression=IdentifierExpression(
                                identifier=x_, provenance=Provenance.unknown()
                            ),
                            indices=(
                                IdentifierExpression(
                                    identifier=i,
                                    provenance=Provenance.unknown(),
                                ),
                                IdentifierExpression(
                                    identifier=j,
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
        validate_types(program_ast, symbol_table)


def test_fails_on_return_type_mismatch():
    """Test failure when a return expression does not match the declared return type."""
    op = Identifier("op")
    a = Identifier("a")
    program_ast = Module(
        statements=(
            Operation(
                name=op,
                args=(
                    Argument(
                        name=a,
                        qualified_type=QualifiedType(
                            base_type=_make_int32_type(),
                            type_qualifier=TypeQualifier.INPUT,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                ),
                body=(
                    ReturnStatement(
                        expression=IdentifierExpression(
                            identifier=a, provenance=Provenance.unknown()
                        ),
                        provenance=Provenance.unknown(),
                    ),
                ),
                return_type=QualifiedType(
                    base_type=_make_float32_type(),
                    type_qualifier=TypeQualifier.OUTPUT,
                    provenance=Provenance.unknown(),
                ),
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(PassExecutionError, match=FhYTypeError.__name__):
        validate_types(program_ast, symbol_table)


def test_fails_on_declaration_initializer_mismatch():
    """Test failure when an initializer does not match the declared type."""
    main = Identifier("main")
    t = Identifier("t")
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(),
                body=(
                    DeclarationStatement(
                        variable_name=t,
                        variable_type=QualifiedType(
                            base_type=_make_float32_type(),
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        expression=IntLiteral(value=1, provenance=Provenance.unknown()),
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
        validate_types(program_ast, symbol_table)


def test_fails_on_unreduced_index():
    """Test failure when a reduction does not remove the free index from the RHS."""
    main = Identifier("main")
    m_ = Identifier("m")
    a, b = Identifier("a"), Identifier("b")
    k = Identifier("k")
    sum_ = BUILTIN_REDUCTION_FUNCTION_IDENTIFIERS["sum"]
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(
                    Argument(
                        name=a,
                        qualified_type=QualifiedType(
                            base_type=_make_int32_type(
                                shape=(CoreIdentifierExpression(m_),)
                            ),
                            type_qualifier=TypeQualifier.INPUT,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    Argument(
                        name=b,
                        qualified_type=QualifiedType(
                            base_type=_make_int32_type(),
                            type_qualifier=TypeQualifier.OUTPUT,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                ),
                body=(
                    DeclarationStatement(
                        variable_name=k,
                        variable_type=QualifiedType(
                            base_type=IndexType(
                                lower_bound=LiteralExpression(1),
                                upper_bound=CoreIdentifierExpression(m_),
                                stride=None,
                            ),
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    ExpressionStatement(
                        left=IdentifierExpression(
                            identifier=b, provenance=Provenance.unknown()
                        ),
                        # Reduction over no indices leaves `k` free on the RHS,
                        # but the LHS is a scalar with no free indices.
                        right=FunctionExpression(
                            function=IdentifierExpression(
                                identifier=sum_,
                                provenance=Provenance.unknown(),
                            ),
                            indices=(),
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

    with pytest.raises(PassExecutionError, match=FhYTypeError.__name__):
        validate_types(program_ast, symbol_table)


def test_valid_data_type_promotion_in_binary_op():
    """Test that a binary op promotes operand data types to the LHS data type."""
    op = Identifier("add")
    m_ = Identifier("m")
    a, b = Identifier("a"), Identifier("b")

    def qt(t, q):
        return QualifiedType(
            base_type=t, type_qualifier=q, provenance=Provenance.unknown()
        )

    int16_vec = NumericalType(
        PrimitiveDataType(CoreDataType.INT16),
        shape=(CoreIdentifierExpression(m_),),
    )
    int32_vec = NumericalType(
        PrimitiveDataType(CoreDataType.INT32),
        shape=(CoreIdentifierExpression(m_),),
    )
    program_ast = Module(
        statements=(
            Operation(
                name=op,
                args=(
                    Argument(
                        name=a,
                        qualified_type=qt(int32_vec, TypeQualifier.INPUT),
                        provenance=Provenance.unknown(),
                    ),
                    Argument(
                        name=b,
                        qualified_type=qt(int16_vec, TypeQualifier.INPUT),
                        provenance=Provenance.unknown(),
                    ),
                ),
                body=(
                    ReturnStatement(
                        expression=BinaryExpression(
                            operation=BinaryOperation.ADDITION,
                            left=IdentifierExpression(
                                identifier=a, provenance=Provenance.unknown()
                            ),
                            right=IdentifierExpression(
                                identifier=b, provenance=Provenance.unknown()
                            ),
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                ),
                return_type=qt(int32_vec, TypeQualifier.OUTPUT),
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )
    symbol_table = build_symbol_table(program_ast)

    validate_types(program_ast, symbol_table)


def test_valid_narrower_source_promotes_to_wider_lhs():
    """Test that a narrower-width source can be assigned to a wider-width LHS."""
    main = Identifier("main")
    m_ = Identifier("m")
    a, b = Identifier("a"), Identifier("b")

    def qt(t, q):
        return QualifiedType(
            base_type=t, type_qualifier=q, provenance=Provenance.unknown()
        )

    uint8_vec = NumericalType(
        PrimitiveDataType(CoreDataType.UINT8),
        shape=(CoreIdentifierExpression(m_),),
    )
    uint16_vec = NumericalType(
        PrimitiveDataType(CoreDataType.UINT16),
        shape=(CoreIdentifierExpression(m_),),
    )
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(
                    Argument(
                        name=a,
                        qualified_type=qt(uint8_vec, TypeQualifier.INPUT),
                        provenance=Provenance.unknown(),
                    ),
                    Argument(
                        name=b,
                        qualified_type=qt(uint16_vec, TypeQualifier.OUTPUT),
                        provenance=Provenance.unknown(),
                    ),
                ),
                body=(
                    ExpressionStatement(
                        left=IdentifierExpression(
                            identifier=b, provenance=Provenance.unknown()
                        ),
                        right=IdentifierExpression(
                            identifier=a, provenance=Provenance.unknown()
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

    validate_types(program_ast, symbol_table)


def test_fails_on_wider_source_to_narrower_lhs():
    """Test that a wider source cannot be assigned to a narrower LHS."""
    main = Identifier("main")
    m_ = Identifier("m")
    a, b = Identifier("a"), Identifier("b")

    def qt(t, q):
        return QualifiedType(
            base_type=t, type_qualifier=q, provenance=Provenance.unknown()
        )

    uint8_vec = NumericalType(
        PrimitiveDataType(CoreDataType.UINT8),
        shape=(CoreIdentifierExpression(m_),),
    )
    uint16_vec = NumericalType(
        PrimitiveDataType(CoreDataType.UINT16),
        shape=(CoreIdentifierExpression(m_),),
    )
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(
                    Argument(
                        name=a,
                        qualified_type=qt(uint16_vec, TypeQualifier.INPUT),
                        provenance=Provenance.unknown(),
                    ),
                    Argument(
                        name=b,
                        qualified_type=qt(uint8_vec, TypeQualifier.OUTPUT),
                        provenance=Provenance.unknown(),
                    ),
                ),
                body=(
                    ExpressionStatement(
                        left=IdentifierExpression(
                            identifier=b, provenance=Provenance.unknown()
                        ),
                        right=IdentifierExpression(
                            identifier=a, provenance=Provenance.unknown()
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
        validate_types(program_ast, symbol_table)


def test_valid_ternary_expression_promotes_branches():
    """Test a ternary expression whose branches promote to the LHS type."""
    op = Identifier("pick")
    a, b, c, d = (
        Identifier("a"),
        Identifier("b"),
        Identifier("c"),
        Identifier("d"),
    )

    def qt(t, q):
        return QualifiedType(
            base_type=t, type_qualifier=q, provenance=Provenance.unknown()
        )

    int16_scalar = NumericalType(PrimitiveDataType(CoreDataType.INT16))
    int32_scalar = NumericalType(PrimitiveDataType(CoreDataType.INT32))
    program_ast = Module(
        statements=(
            Operation(
                name=op,
                args=(
                    Argument(
                        name=a,
                        qualified_type=qt(int32_scalar, TypeQualifier.INPUT),
                        provenance=Provenance.unknown(),
                    ),
                    Argument(
                        name=b,
                        qualified_type=qt(int32_scalar, TypeQualifier.INPUT),
                        provenance=Provenance.unknown(),
                    ),
                    Argument(
                        name=c,
                        qualified_type=qt(int32_scalar, TypeQualifier.INPUT),
                        provenance=Provenance.unknown(),
                    ),
                    Argument(
                        name=d,
                        qualified_type=qt(int16_scalar, TypeQualifier.INPUT),
                        provenance=Provenance.unknown(),
                    ),
                ),
                body=(
                    ReturnStatement(
                        expression=TernaryExpression(
                            condition=BinaryExpression(
                                operation=BinaryOperation.LESS_THAN,
                                left=IdentifierExpression(
                                    identifier=a,
                                    provenance=Provenance.unknown(),
                                ),
                                right=IdentifierExpression(
                                    identifier=b,
                                    provenance=Provenance.unknown(),
                                ),
                                provenance=Provenance.unknown(),
                            ),
                            true=IdentifierExpression(
                                identifier=c, provenance=Provenance.unknown()
                            ),
                            false=IdentifierExpression(
                                identifier=d, provenance=Provenance.unknown()
                            ),
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                ),
                return_type=qt(int32_scalar, TypeQualifier.OUTPUT),
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )
    symbol_table = build_symbol_table(program_ast)

    validate_types(program_ast, symbol_table)


def test_valid_ternary_expression_unions_free_indices():
    """Test that a ternary expression's free indices union across branches."""
    main = Identifier("pick")
    m_ = Identifier("m")
    a, b, c, y_ = (
        Identifier("a"),
        Identifier("b"),
        Identifier("c"),
        Identifier("y"),
    )
    i = Identifier("i")

    def qt(t, q):
        return QualifiedType(
            base_type=t, type_qualifier=q, provenance=Provenance.unknown()
        )

    int32_vec = NumericalType(
        PrimitiveDataType(CoreDataType.INT32),
        shape=(CoreIdentifierExpression(m_),),
    )
    int32_scalar = NumericalType(PrimitiveDataType(CoreDataType.INT32))
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(
                    Argument(
                        name=a,
                        qualified_type=qt(int32_scalar, TypeQualifier.INPUT),
                        provenance=Provenance.unknown(),
                    ),
                    Argument(
                        name=b,
                        qualified_type=qt(int32_vec, TypeQualifier.INPUT),
                        provenance=Provenance.unknown(),
                    ),
                    Argument(
                        name=c,
                        qualified_type=qt(int32_vec, TypeQualifier.INPUT),
                        provenance=Provenance.unknown(),
                    ),
                    Argument(
                        name=y_,
                        qualified_type=qt(int32_vec, TypeQualifier.OUTPUT),
                        provenance=Provenance.unknown(),
                    ),
                ),
                body=(
                    DeclarationStatement(
                        variable_name=i,
                        variable_type=qt(
                            IndexType(
                                lower_bound=LiteralExpression(1),
                                upper_bound=CoreIdentifierExpression(m_),
                                stride=None,
                            ),
                            TypeQualifier.TEMP,
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    ExpressionStatement(
                        left=ArrayAccessExpression(
                            array_expression=IdentifierExpression(
                                identifier=y_,
                                provenance=Provenance.unknown(),
                            ),
                            indices=(
                                IdentifierExpression(
                                    identifier=i,
                                    provenance=Provenance.unknown(),
                                ),
                            ),
                            provenance=Provenance.unknown(),
                        ),
                        right=TernaryExpression(
                            condition=IdentifierExpression(
                                identifier=a, provenance=Provenance.unknown()
                            ),
                            true=ArrayAccessExpression(
                                array_expression=IdentifierExpression(
                                    identifier=b,
                                    provenance=Provenance.unknown(),
                                ),
                                indices=(
                                    IdentifierExpression(
                                        identifier=i,
                                        provenance=Provenance.unknown(),
                                    ),
                                ),
                                provenance=Provenance.unknown(),
                            ),
                            false=ArrayAccessExpression(
                                array_expression=IdentifierExpression(
                                    identifier=c,
                                    provenance=Provenance.unknown(),
                                ),
                                indices=(
                                    IdentifierExpression(
                                        identifier=i,
                                        provenance=Provenance.unknown(),
                                    ),
                                ),
                                provenance=Provenance.unknown(),
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

    validate_types(program_ast, symbol_table)


def test_fails_on_ternary_branch_shape_mismatch():
    """Test failure when ternary branches have different shapes."""
    op = Identifier("pick")
    m_ = Identifier("m")
    a, b, c = Identifier("a"), Identifier("b"), Identifier("c")

    def qt(t, q):
        return QualifiedType(
            base_type=t, type_qualifier=q, provenance=Provenance.unknown()
        )

    int32_scalar = NumericalType(PrimitiveDataType(CoreDataType.INT32))
    int32_vec = NumericalType(
        PrimitiveDataType(CoreDataType.INT32),
        shape=(CoreIdentifierExpression(m_),),
    )
    program_ast = Module(
        statements=(
            Operation(
                name=op,
                args=(
                    Argument(
                        name=a,
                        qualified_type=qt(int32_scalar, TypeQualifier.INPUT),
                        provenance=Provenance.unknown(),
                    ),
                    Argument(
                        name=b,
                        qualified_type=qt(int32_scalar, TypeQualifier.INPUT),
                        provenance=Provenance.unknown(),
                    ),
                    Argument(
                        name=c,
                        qualified_type=qt(int32_vec, TypeQualifier.INPUT),
                        provenance=Provenance.unknown(),
                    ),
                ),
                body=(
                    ReturnStatement(
                        expression=TernaryExpression(
                            condition=BinaryExpression(
                                operation=BinaryOperation.LESS_THAN,
                                left=IdentifierExpression(
                                    identifier=a,
                                    provenance=Provenance.unknown(),
                                ),
                                right=IdentifierExpression(
                                    identifier=b,
                                    provenance=Provenance.unknown(),
                                ),
                                provenance=Provenance.unknown(),
                            ),
                            true=IdentifierExpression(
                                identifier=a, provenance=Provenance.unknown()
                            ),
                            false=IdentifierExpression(
                                identifier=c, provenance=Provenance.unknown()
                            ),
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                ),
                return_type=qt(int32_scalar, TypeQualifier.OUTPUT),
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(PassExecutionError, match=FhYTypeError.__name__):
        validate_types(program_ast, symbol_table)
