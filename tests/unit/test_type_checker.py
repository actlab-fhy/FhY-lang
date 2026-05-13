"""Tests the type checker AST pass."""

import pytest
from fhy_core import (
    BinaryExpression as CoreBinaryExpression,
)
from fhy_core import (
    BinaryOperation as CoreBinaryOperation,
)
from fhy_core import (
    CoreDataType,
    Identifier,
    IndexType,
    LiteralExpression,
    NumericalType,
    PrimitiveDataType,
    Type,
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
    ComplexLiteral,
    DeclarationStatement,
    ExpressionStatement,
    FloatLiteral,
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
    UnaryExpression,
    UnaryOperation,
)
from fhy_lang.ast.passes import TypeChecker, build_symbol_table
from fhy_lang.builtins import BUILTIN_REDUCTION_FUNCTION_IDENTIFIERS

from .utils import run_validator


def _make_int32_type(shape=()) -> NumericalType:
    return NumericalType(PrimitiveDataType(CoreDataType.INT32), shape=shape)


def _make_float32_type(shape=()) -> NumericalType:
    return NumericalType(PrimitiveDataType(CoreDataType.FLOAT32), shape=shape)


def test_empty_module(empty_module_ast):
    """Test validation of an empty program."""
    symbol_table = build_symbol_table(empty_module_ast)

    run_validator(TypeChecker(symbol_table), empty_module_ast)


def test_valid_matmul_like_reduction():
    """Test a matmul-style reduction: C[i,j] = sum[k](A[i,k] * B[k,j])."""
    main = Identifier("matmul")
    m, n, p = Identifier("m"), Identifier("n"), Identifier("p")
    a_, b_, c_ = Identifier("A"), Identifier("B"), Identifier("C")
    i, j, k = Identifier("i"), Identifier("j"), Identifier("k")
    sum_ = BUILTIN_REDUCTION_FUNCTION_IDENTIFIERS["sum"]

    def qt(t, q):
        return QualifiedType(base_type=t, type_qualifier=q)

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
                    ),
                    ExpressionStatement(
                        left=ArrayAccessExpression(
                            array_expression=IdentifierExpression(identifier=c_),
                            indices=(
                                IdentifierExpression(
                                    identifier=i,
                                ),
                                IdentifierExpression(
                                    identifier=j,
                                ),
                            ),
                        ),
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
                                BinaryExpression(
                                    operation=BinaryOperation.MULTIPLICATION,
                                    left=ArrayAccessExpression(
                                        array_expression=IdentifierExpression(
                                            identifier=a_,
                                        ),
                                        indices=(
                                            IdentifierExpression(
                                                identifier=i,
                                            ),
                                            IdentifierExpression(
                                                identifier=k,
                                            ),
                                        ),
                                    ),
                                    right=ArrayAccessExpression(
                                        array_expression=IdentifierExpression(
                                            identifier=b_,
                                        ),
                                        indices=(
                                            IdentifierExpression(
                                                identifier=k,
                                            ),
                                            IdentifierExpression(
                                                identifier=j,
                                            ),
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

    run_validator(TypeChecker(symbol_table), program_ast)


def test_valid_forall_binds_index():
    """Test that a forall-bound index is removed from the free-index set."""
    main = Identifier("main")
    examples, n_ = Identifier("examples"), Identifier("n")
    x_, big_x = Identifier("x"), Identifier("X")
    e, i = Identifier("e"), Identifier("i")

    def qt(t, q):
        return QualifiedType(base_type=t, type_qualifier=q)

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
                    ),
                    Argument(
                        name=x_,
                        qualified_type=qt(
                            _make_float32_type(shape=(CoreIdentifierExpression(n_),)),
                            TypeQualifier.OUTPUT,
                        ),
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
                    ),
                    ForAllStatement(
                        index=IdentifierExpression(identifier=e),
                        body=(
                            ExpressionStatement(
                                left=ArrayAccessExpression(
                                    array_expression=IdentifierExpression(
                                        identifier=x_,
                                    ),
                                    indices=(
                                        IdentifierExpression(
                                            identifier=i,
                                        ),
                                    ),
                                ),
                                right=ArrayAccessExpression(
                                    array_expression=IdentifierExpression(
                                        identifier=big_x,
                                    ),
                                    indices=(
                                        IdentifierExpression(
                                            identifier=e,
                                        ),
                                        IdentifierExpression(
                                            identifier=i,
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

    run_validator(TypeChecker(symbol_table), program_ast)


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
                        ),
                    ),
                ),
                body=(
                    ReturnStatement(
                        expression=IdentifierExpression(identifier=a),
                    ),
                ),
                return_type=QualifiedType(
                    base_type=_make_int32_type(),
                    type_qualifier=TypeQualifier.OUTPUT,
                ),
            ),
        ),
    )
    symbol_table = build_symbol_table(program_ast)

    run_validator(TypeChecker(symbol_table), program_ast)


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
                        ),
                    ),
                ),
                body=(),
                return_type=QualifiedType(
                    base_type=_make_float32_type(shape=(CoreIdentifierExpression(m_),)),
                    type_qualifier=TypeQualifier.OUTPUT,
                ),
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
                        ),
                    ),
                    Argument(
                        name=y_,
                        qualified_type=QualifiedType(
                            base_type=_make_float32_type(
                                shape=(CoreIdentifierExpression(m_),)
                            ),
                            type_qualifier=TypeQualifier.OUTPUT,
                        ),
                    ),
                ),
                body=(
                    ExpressionStatement(
                        left=IdentifierExpression(identifier=y_),
                        right=FunctionExpression(
                            function=IdentifierExpression(
                                identifier=op,
                            ),
                            args=(
                                IdentifierExpression(
                                    identifier=x_,
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    symbol_table = build_symbol_table(program_ast)

    run_validator(TypeChecker(symbol_table), program_ast)


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
                ),
            ),
        ),
        body=(
            ReturnStatement(
                expression=IdentifierExpression(
                    identifier=a,
                ),
            ),
        ),
        return_type=QualifiedType(
            base_type=_make_int32_type(),
            type_qualifier=TypeQualifier.OUTPUT,
        ),
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
                        ),
                    ),
                    Argument(
                        name=b,
                        qualified_type=QualifiedType(
                            base_type=_make_int32_type(
                                shape=(CoreIdentifierExpression(N),)
                            ),
                            type_qualifier=TypeQualifier.OUTPUT,
                        ),
                    ),
                ),
                body=(
                    ExpressionStatement(
                        left=IdentifierExpression(identifier=b),
                        right=FunctionExpression(
                            function=IdentifierExpression(
                                identifier=op,
                            ),
                            args=(
                                ArrayAccessExpression(
                                    array_expression=IdentifierExpression(
                                        identifier=a,
                                    ),
                                    indices=(
                                        IdentifierExpression(
                                            identifier=i,
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

    run_validator(TypeChecker(symbol_table), program_ast)


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
                        ),
                    ),
                    Argument(
                        name=b,
                        qualified_type=QualifiedType(
                            base_type=_make_float32_type(),
                            type_qualifier=TypeQualifier.OUTPUT,
                        ),
                    ),
                ),
                body=(
                    ExpressionStatement(
                        left=IdentifierExpression(identifier=b),
                        right=IdentifierExpression(identifier=a),
                    ),
                ),
            ),
        ),
    )
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(ValidationFailedError, match="type error"):
        run_validator(TypeChecker(symbol_table), program_ast)


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
                        ),
                    ),
                    Argument(
                        name=y_,
                        qualified_type=QualifiedType(
                            base_type=_make_int32_type(
                                shape=(CoreIdentifierExpression(m_),)
                            ),
                            type_qualifier=TypeQualifier.OUTPUT,
                        ),
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
                        ),
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
                        ),
                    ),
                    ExpressionStatement(
                        left=ArrayAccessExpression(
                            array_expression=IdentifierExpression(identifier=y_),
                            indices=(
                                IdentifierExpression(
                                    identifier=i,
                                ),
                            ),
                        ),
                        right=ArrayAccessExpression(
                            array_expression=IdentifierExpression(identifier=x_),
                            indices=(
                                IdentifierExpression(
                                    identifier=i,
                                ),
                                IdentifierExpression(
                                    identifier=j,
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
        run_validator(TypeChecker(symbol_table), program_ast)


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
                        ),
                    ),
                ),
                body=(
                    ReturnStatement(
                        expression=IdentifierExpression(identifier=a),
                    ),
                ),
                return_type=QualifiedType(
                    base_type=_make_float32_type(),
                    type_qualifier=TypeQualifier.OUTPUT,
                ),
            ),
        ),
    )
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(ValidationFailedError, match="type error"):
        run_validator(TypeChecker(symbol_table), program_ast)


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
                        ),
                        expression=IntLiteral(value=1),
                    ),
                ),
            ),
        ),
    )
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(ValidationFailedError, match="type error"):
        run_validator(TypeChecker(symbol_table), program_ast)


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
                        ),
                    ),
                    Argument(
                        name=b,
                        qualified_type=QualifiedType(
                            base_type=_make_int32_type(),
                            type_qualifier=TypeQualifier.OUTPUT,
                        ),
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
                        ),
                    ),
                    ExpressionStatement(
                        left=IdentifierExpression(identifier=b),
                        # Reduction over no indices leaves `k` free on the RHS,
                        # but the LHS is a scalar with no free indices.
                        right=FunctionExpression(
                            function=IdentifierExpression(
                                identifier=sum_,
                            ),
                            indices=(),
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

    with pytest.raises(ValidationFailedError, match="type error"):
        run_validator(TypeChecker(symbol_table), program_ast)


def test_valid_data_type_promotion_in_binary_op():
    """Test that a binary op promotes operand data types to the LHS data type."""
    op = Identifier("add")
    m_ = Identifier("m")
    a, b = Identifier("a"), Identifier("b")

    def qt(t, q):
        return QualifiedType(base_type=t, type_qualifier=q)

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
                    ),
                    Argument(
                        name=b,
                        qualified_type=qt(int16_vec, TypeQualifier.INPUT),
                    ),
                ),
                body=(
                    ReturnStatement(
                        expression=BinaryExpression(
                            operation=BinaryOperation.ADDITION,
                            left=IdentifierExpression(identifier=a),
                            right=IdentifierExpression(identifier=b),
                        ),
                    ),
                ),
                return_type=qt(int32_vec, TypeQualifier.OUTPUT),
            ),
        ),
    )
    symbol_table = build_symbol_table(program_ast)

    run_validator(TypeChecker(symbol_table), program_ast)


def test_valid_narrower_source_promotes_to_wider_lhs():
    """Test that a narrower-width source can be assigned to a wider-width LHS."""
    main = Identifier("main")
    m_ = Identifier("m")
    a, b = Identifier("a"), Identifier("b")

    def qt(t, q):
        return QualifiedType(base_type=t, type_qualifier=q)

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
                    ),
                    Argument(
                        name=b,
                        qualified_type=qt(uint16_vec, TypeQualifier.OUTPUT),
                    ),
                ),
                body=(
                    ExpressionStatement(
                        left=IdentifierExpression(identifier=b),
                        right=IdentifierExpression(identifier=a),
                    ),
                ),
            ),
        ),
    )
    symbol_table = build_symbol_table(program_ast)

    run_validator(TypeChecker(symbol_table), program_ast)


def test_fails_on_wider_source_to_narrower_lhs():
    """Test that a wider source cannot be assigned to a narrower LHS."""
    main = Identifier("main")
    m_ = Identifier("m")
    a, b = Identifier("a"), Identifier("b")

    def qt(t, q):
        return QualifiedType(base_type=t, type_qualifier=q)

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
                    ),
                    Argument(
                        name=b,
                        qualified_type=qt(uint8_vec, TypeQualifier.OUTPUT),
                    ),
                ),
                body=(
                    ExpressionStatement(
                        left=IdentifierExpression(identifier=b),
                        right=IdentifierExpression(identifier=a),
                    ),
                ),
            ),
        ),
    )
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(ValidationFailedError, match="type error"):
        run_validator(TypeChecker(symbol_table), program_ast)


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
        return QualifiedType(base_type=t, type_qualifier=q)

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
                    ),
                    Argument(
                        name=b,
                        qualified_type=qt(int32_scalar, TypeQualifier.INPUT),
                    ),
                    Argument(
                        name=c,
                        qualified_type=qt(int32_scalar, TypeQualifier.INPUT),
                    ),
                    Argument(
                        name=d,
                        qualified_type=qt(int16_scalar, TypeQualifier.INPUT),
                    ),
                ),
                body=(
                    ReturnStatement(
                        expression=TernaryExpression(
                            condition=BinaryExpression(
                                operation=BinaryOperation.LESS_THAN,
                                left=IdentifierExpression(
                                    identifier=a,
                                ),
                                right=IdentifierExpression(
                                    identifier=b,
                                ),
                            ),
                            true=IdentifierExpression(identifier=c),
                            false=IdentifierExpression(identifier=d),
                        ),
                    ),
                ),
                return_type=qt(int32_scalar, TypeQualifier.OUTPUT),
            ),
        ),
    )
    symbol_table = build_symbol_table(program_ast)

    run_validator(TypeChecker(symbol_table), program_ast)


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
        return QualifiedType(base_type=t, type_qualifier=q)

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
                    ),
                    Argument(
                        name=b,
                        qualified_type=qt(int32_vec, TypeQualifier.INPUT),
                    ),
                    Argument(
                        name=c,
                        qualified_type=qt(int32_vec, TypeQualifier.INPUT),
                    ),
                    Argument(
                        name=y_,
                        qualified_type=qt(int32_vec, TypeQualifier.OUTPUT),
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
                    ),
                    ExpressionStatement(
                        left=ArrayAccessExpression(
                            array_expression=IdentifierExpression(
                                identifier=y_,
                            ),
                            indices=(
                                IdentifierExpression(
                                    identifier=i,
                                ),
                            ),
                        ),
                        right=TernaryExpression(
                            condition=IdentifierExpression(identifier=a),
                            true=ArrayAccessExpression(
                                array_expression=IdentifierExpression(
                                    identifier=b,
                                ),
                                indices=(
                                    IdentifierExpression(
                                        identifier=i,
                                    ),
                                ),
                            ),
                            false=ArrayAccessExpression(
                                array_expression=IdentifierExpression(
                                    identifier=c,
                                ),
                                indices=(
                                    IdentifierExpression(
                                        identifier=i,
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

    run_validator(TypeChecker(symbol_table), program_ast)


def test_fails_on_ternary_branch_shape_mismatch():
    """Test failure when ternary branches have different shapes."""
    op = Identifier("pick")
    m_ = Identifier("m")
    a, b, c = Identifier("a"), Identifier("b"), Identifier("c")

    def qt(t, q):
        return QualifiedType(base_type=t, type_qualifier=q)

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
                    ),
                    Argument(
                        name=b,
                        qualified_type=qt(int32_scalar, TypeQualifier.INPUT),
                    ),
                    Argument(
                        name=c,
                        qualified_type=qt(int32_vec, TypeQualifier.INPUT),
                    ),
                ),
                body=(
                    ReturnStatement(
                        expression=TernaryExpression(
                            condition=BinaryExpression(
                                operation=BinaryOperation.LESS_THAN,
                                left=IdentifierExpression(
                                    identifier=a,
                                ),
                                right=IdentifierExpression(
                                    identifier=b,
                                ),
                            ),
                            true=IdentifierExpression(identifier=a),
                            false=IdentifierExpression(identifier=c),
                        ),
                    ),
                ),
                return_type=qt(int32_scalar, TypeQualifier.OUTPUT),
            ),
        ),
    )
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(ValidationFailedError, match="type error"):
        run_validator(TypeChecker(symbol_table), program_ast)


def test_fails_on_int_literal_overflowing_declared_type():
    """Test that `uint8 t = 300;` fails because literal 300 doesn't fit in uint8."""
    op = Identifier("f")
    a, t = Identifier("a"), Identifier("t")

    def qt(t_, q):
        return QualifiedType(base_type=t_, type_qualifier=q)

    uint8_scalar = NumericalType(PrimitiveDataType(CoreDataType.UINT8))
    program_ast = Module(
        statements=(
            Operation(
                name=op,
                args=(
                    Argument(
                        name=a,
                        qualified_type=qt(uint8_scalar, TypeQualifier.INPUT),
                    ),
                ),
                body=(
                    DeclarationStatement(
                        variable_name=t,
                        variable_type=qt(uint8_scalar, TypeQualifier.TEMP),
                        expression=IntLiteral(value=300),
                    ),
                    ReturnStatement(
                        expression=IdentifierExpression(identifier=t),
                    ),
                ),
                return_type=qt(uint8_scalar, TypeQualifier.OUTPUT),
            ),
        ),
    )
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(ValidationFailedError, match="type error"):
        run_validator(TypeChecker(symbol_table), program_ast)


def test_valid_int_literal_fits_declared_type_via_promotion():
    """Test `int32 t = 5;` validates: literal 5 fits uint8 and promotes to int32."""
    op = Identifier("f")
    a, t = Identifier("a"), Identifier("t")

    def qt(t_, q):
        return QualifiedType(base_type=t_, type_qualifier=q)

    int32_scalar = NumericalType(PrimitiveDataType(CoreDataType.INT32))
    program_ast = Module(
        statements=(
            Operation(
                name=op,
                args=(
                    Argument(
                        name=a,
                        qualified_type=qt(int32_scalar, TypeQualifier.INPUT),
                    ),
                ),
                body=(
                    DeclarationStatement(
                        variable_name=t,
                        variable_type=qt(int32_scalar, TypeQualifier.TEMP),
                        expression=IntLiteral(value=5),
                    ),
                    ReturnStatement(
                        expression=IdentifierExpression(identifier=t),
                    ),
                ),
                return_type=qt(int32_scalar, TypeQualifier.OUTPUT),
            ),
        ),
    )
    symbol_table = build_symbol_table(program_ast)

    run_validator(TypeChecker(symbol_table), program_ast)


def test_fails_on_call_site_argument_data_type_mismatch():
    """Test that calling `f(x)` where `f` expects int32 but `x` is float32 fails."""
    f, main = Identifier("f"), Identifier("main")
    x = Identifier("x")
    a, t = Identifier("a"), Identifier("t")

    def qt(t_, q):
        return QualifiedType(base_type=t_, type_qualifier=q)

    int32_scalar = NumericalType(PrimitiveDataType(CoreDataType.INT32))
    float32_scalar = NumericalType(PrimitiveDataType(CoreDataType.FLOAT32))

    callee = Operation(
        name=f,
        args=(
            Argument(
                name=x,
                qualified_type=qt(int32_scalar, TypeQualifier.INPUT),
            ),
        ),
        body=(
            ReturnStatement(
                expression=IdentifierExpression(identifier=x),
            ),
        ),
        return_type=qt(int32_scalar, TypeQualifier.OUTPUT),
    )
    caller = Operation(
        name=main,
        args=(
            Argument(
                name=a,
                qualified_type=qt(float32_scalar, TypeQualifier.INPUT),
            ),
        ),
        body=(
            DeclarationStatement(
                variable_name=t,
                variable_type=qt(int32_scalar, TypeQualifier.TEMP),
            ),
            ExpressionStatement(
                left=IdentifierExpression(identifier=t),
                right=FunctionExpression(
                    function=IdentifierExpression(identifier=f),
                    args=(IdentifierExpression(identifier=a),),
                ),
            ),
            ReturnStatement(
                expression=IdentifierExpression(identifier=t),
            ),
        ),
        return_type=qt(int32_scalar, TypeQualifier.OUTPUT),
    )
    program_ast = Module(statements=(callee, caller))
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(ValidationFailedError, match="type error"):
        run_validator(TypeChecker(symbol_table), program_ast)


def test_fails_on_call_site_literal_shape_mismatch():
    """Test that calling `f(x)` where `f` expects int32[3] but `x` is int32[5] fails."""
    f, main = Identifier("f"), Identifier("main")
    x = Identifier("x")
    a, t = Identifier("a"), Identifier("t")

    def qt(t_, q):
        return QualifiedType(base_type=t_, type_qualifier=q)

    int32_scalar = NumericalType(PrimitiveDataType(CoreDataType.INT32))
    int32_vec3 = NumericalType(
        PrimitiveDataType(CoreDataType.INT32), shape=(LiteralExpression(3),)
    )
    int32_vec5 = NumericalType(
        PrimitiveDataType(CoreDataType.INT32), shape=(LiteralExpression(5),)
    )

    callee = Operation(
        name=f,
        args=(
            Argument(
                name=x,
                qualified_type=qt(int32_vec3, TypeQualifier.INPUT),
            ),
        ),
        body=(
            ReturnStatement(
                expression=IntLiteral(value=0),
            ),
        ),
        return_type=qt(int32_scalar, TypeQualifier.OUTPUT),
    )
    caller = Operation(
        name=main,
        args=(
            Argument(
                name=a,
                qualified_type=qt(int32_vec5, TypeQualifier.INPUT),
            ),
        ),
        body=(
            DeclarationStatement(
                variable_name=t,
                variable_type=qt(int32_scalar, TypeQualifier.TEMP),
            ),
            ExpressionStatement(
                left=IdentifierExpression(identifier=t),
                right=FunctionExpression(
                    function=IdentifierExpression(identifier=f),
                    args=(IdentifierExpression(identifier=a),),
                ),
            ),
            ReturnStatement(
                expression=IdentifierExpression(identifier=t),
            ),
        ),
        return_type=qt(int32_scalar, TypeQualifier.OUTPUT),
    )
    program_ast = Module(statements=(callee, caller))
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(ValidationFailedError, match="type error"):
        run_validator(TypeChecker(symbol_table), program_ast)


def test_valid_call_site_literal_shape_match_with_promotion():
    """Test that calling `f(x)` with int32[3] where `f` expects int64[3] validates."""
    f, main = Identifier("f"), Identifier("main")
    x = Identifier("x")
    a, t = Identifier("a"), Identifier("t")

    def qt(t_, q):
        return QualifiedType(base_type=t_, type_qualifier=q)

    int32_scalar = NumericalType(PrimitiveDataType(CoreDataType.INT32))
    int32_vec3 = NumericalType(
        PrimitiveDataType(CoreDataType.INT32), shape=(LiteralExpression(3),)
    )
    int64_scalar = NumericalType(PrimitiveDataType(CoreDataType.INT64))
    int64_vec3 = NumericalType(
        PrimitiveDataType(CoreDataType.INT64), shape=(LiteralExpression(3),)
    )

    callee = Operation(
        name=f,
        args=(
            Argument(
                name=x,
                qualified_type=qt(int64_vec3, TypeQualifier.INPUT),
            ),
        ),
        body=(
            ReturnStatement(
                expression=IntLiteral(value=0),
            ),
        ),
        return_type=qt(int64_scalar, TypeQualifier.OUTPUT),
    )
    caller = Operation(
        name=main,
        args=(
            Argument(
                name=a,
                qualified_type=qt(int32_vec3, TypeQualifier.INPUT),
            ),
        ),
        body=(
            DeclarationStatement(
                variable_name=t,
                variable_type=qt(int64_scalar, TypeQualifier.TEMP),
            ),
            ExpressionStatement(
                left=IdentifierExpression(identifier=t),
                right=FunctionExpression(
                    function=IdentifierExpression(identifier=f),
                    args=(IdentifierExpression(identifier=a),),
                ),
            ),
            ReturnStatement(
                expression=IdentifierExpression(identifier=t),
            ),
        ),
        return_type=qt(int64_scalar, TypeQualifier.OUTPUT),
    )
    program_ast = Module(statements=(callee, caller))
    symbol_table = build_symbol_table(program_ast)

    # Should not raise: arg data type int32 promotes to the declared
    # int64 parameter type; shapes match on literal 3 == 3.
    run_validator(TypeChecker(symbol_table), program_ast)


def _make_int32_returning_op(
    input_name: Identifier, input_type: Type, body: tuple
) -> Module:
    """Build a Module with a single int32-returning Operation around ``body``.

    The operation is named ``op`` and takes a single INPUT argument
    ``input_name`` of type ``input_type``. The ``body`` is the operation
    body and is responsible for ending with a ``ReturnStatement``.
    """
    op = Identifier("op")
    int32_scalar = NumericalType(PrimitiveDataType(CoreDataType.INT32))
    return Module(
        statements=(
            Operation(
                name=op,
                args=(
                    Argument(
                        name=input_name,
                        qualified_type=QualifiedType(
                            base_type=input_type,
                            type_qualifier=TypeQualifier.INPUT,
                        ),
                    ),
                ),
                body=body,
                return_type=QualifiedType(
                    base_type=int32_scalar, type_qualifier=TypeQualifier.OUTPUT
                ),
            ),
        ),
    )


def test_fails_on_ternary_condition_with_shape():
    """Test ternary fails when condition is a shaped (non-scalar) value."""
    a = Identifier("a")
    int32_vec3 = NumericalType(
        PrimitiveDataType(CoreDataType.INT32), shape=(LiteralExpression(3),)
    )
    body = (
        ReturnStatement(
            expression=TernaryExpression(
                condition=IdentifierExpression(identifier=a),
                true=IntLiteral(value=1),
                false=IntLiteral(value=0),
            ),
        ),
    )

    program_ast = _make_int32_returning_op(a, int32_vec3, body)
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(
        ValidationFailedError, match="Ternary condition must be a scalar"
    ):
        run_validator(TypeChecker(symbol_table), program_ast)


def test_fails_on_ternary_condition_with_float():
    """Test ternary fails when condition is a float (non-integer) scalar."""
    a = Identifier("a")
    float32_scalar = _make_float32_type()
    body = (
        ReturnStatement(
            expression=TernaryExpression(
                condition=IdentifierExpression(identifier=a),
                true=IntLiteral(value=1),
                false=IntLiteral(value=0),
            ),
        ),
    )

    program_ast = _make_int32_returning_op(a, float32_scalar, body)
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(
        ValidationFailedError, match="Ternary condition must be an integer-typed scalar"
    ):
        run_validator(TypeChecker(symbol_table), program_ast)


def test_fails_on_unary_bitwise_not_on_float():
    """Test unary `~` fails on a float operand."""
    a = Identifier("a")
    float32_scalar = _make_float32_type()
    body = (
        ReturnStatement(
            expression=UnaryExpression(
                operation=UnaryOperation.BITWISE_NOT,
                expression=IdentifierExpression(identifier=a),
            ),
        ),
    )

    program_ast = _make_int32_returning_op(a, float32_scalar, body)
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(
        ValidationFailedError, match="Unary '~' requires an integer operand"
    ):
        run_validator(TypeChecker(symbol_table), program_ast)


def test_fails_on_unary_logical_not_on_shaped_operand():
    """Test unary `!` fails on a shaped (non-scalar) operand."""
    a = Identifier("a")
    int32_vec3 = NumericalType(
        PrimitiveDataType(CoreDataType.INT32), shape=(LiteralExpression(3),)
    )
    body = (
        ReturnStatement(
            expression=UnaryExpression(
                operation=UnaryOperation.LOGICAL_NOT,
                expression=IdentifierExpression(identifier=a),
            ),
        ),
    )

    program_ast = _make_int32_returning_op(a, int32_vec3, body)
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(
        ValidationFailedError, match="Unary '!' requires a scalar operand"
    ):
        run_validator(TypeChecker(symbol_table), program_ast)


def test_fails_on_binary_with_shape_mismatch():
    """Test binary expression fails when operand shapes are not equivalent."""
    op = Identifier("op")
    a, b = Identifier("a"), Identifier("b")
    int32_vec3 = NumericalType(
        PrimitiveDataType(CoreDataType.INT32), shape=(LiteralExpression(3),)
    )
    int32_vec5 = NumericalType(
        PrimitiveDataType(CoreDataType.INT32), shape=(LiteralExpression(5),)
    )

    def qt(t_, q):
        return QualifiedType(base_type=t_, type_qualifier=q)

    program_ast = Module(
        statements=(
            Operation(
                name=op,
                args=(
                    Argument(
                        name=a, qualified_type=qt(int32_vec3, TypeQualifier.INPUT)
                    ),
                    Argument(
                        name=b, qualified_type=qt(int32_vec5, TypeQualifier.INPUT)
                    ),
                ),
                body=(
                    ReturnStatement(
                        expression=BinaryExpression(
                            operation=BinaryOperation.ADDITION,
                            left=IdentifierExpression(identifier=a),
                            right=IdentifierExpression(identifier=b),
                        ),
                    ),
                ),
                return_type=qt(int32_vec3, TypeQualifier.OUTPUT),
            ),
        ),
    )
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(
        ValidationFailedError,
        match="Binary expression operand shapes are not structurally equivalent",
    ):
        run_validator(TypeChecker(symbol_table), program_ast)


def test_fails_on_array_access_with_wrong_index_arity():
    """Test array access fails when index count differs from array dimensions."""
    op = Identifier("op")
    a, i, j = Identifier("a"), Identifier("i"), Identifier("j")
    int32_scalar = NumericalType(PrimitiveDataType(CoreDataType.INT32))
    int32_vec3 = NumericalType(
        PrimitiveDataType(CoreDataType.INT32), shape=(LiteralExpression(3),)
    )

    def qt(t_, q):
        return QualifiedType(base_type=t_, type_qualifier=q)

    index_type = IndexType(
        lower_bound=LiteralExpression(1),
        upper_bound=LiteralExpression(3),
        stride=None,
    )
    program_ast = Module(
        statements=(
            Operation(
                name=op,
                args=(
                    Argument(
                        name=a, qualified_type=qt(int32_vec3, TypeQualifier.INPUT)
                    ),
                ),
                body=(
                    DeclarationStatement(
                        variable_name=i,
                        variable_type=qt(index_type, TypeQualifier.TEMP),
                    ),
                    DeclarationStatement(
                        variable_name=j,
                        variable_type=qt(index_type, TypeQualifier.TEMP),
                    ),
                    ReturnStatement(
                        expression=ArrayAccessExpression(
                            array_expression=IdentifierExpression(identifier=a),
                            indices=(
                                IdentifierExpression(identifier=i),
                                IdentifierExpression(identifier=j),
                            ),
                        ),
                    ),
                ),
                return_type=qt(int32_scalar, TypeQualifier.OUTPUT),
            ),
        ),
    )
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(ValidationFailedError, match="indices but the array has"):
        run_validator(TypeChecker(symbol_table), program_ast)


def test_fails_on_reduction_with_multiple_arguments():
    """Test that a reduction call must receive exactly one argument."""
    op = Identifier("op")
    a, k = Identifier("a"), Identifier("k")
    sum_ = BUILTIN_REDUCTION_FUNCTION_IDENTIFIERS["sum"]
    int32_scalar = NumericalType(PrimitiveDataType(CoreDataType.INT32))
    int32_vec3 = NumericalType(
        PrimitiveDataType(CoreDataType.INT32), shape=(LiteralExpression(3),)
    )

    def qt(t_, q):
        return QualifiedType(base_type=t_, type_qualifier=q)

    index_type = IndexType(
        lower_bound=LiteralExpression(1),
        upper_bound=LiteralExpression(3),
        stride=None,
    )
    program_ast = Module(
        statements=(
            Operation(
                name=op,
                args=(
                    Argument(
                        name=a, qualified_type=qt(int32_vec3, TypeQualifier.INPUT)
                    ),
                ),
                body=(
                    DeclarationStatement(
                        variable_name=k,
                        variable_type=qt(index_type, TypeQualifier.TEMP),
                    ),
                    ReturnStatement(
                        expression=FunctionExpression(
                            function=IdentifierExpression(identifier=sum_),
                            indices=(IdentifierExpression(identifier=k),),
                            args=(
                                ArrayAccessExpression(
                                    array_expression=IdentifierExpression(identifier=a),
                                    indices=(IdentifierExpression(identifier=k),),
                                ),
                                IntLiteral(value=0),
                            ),
                        ),
                    ),
                ),
                return_type=qt(int32_scalar, TypeQualifier.OUTPUT),
            ),
        ),
    )
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(
        ValidationFailedError, match="must be passed exactly one argument"
    ):
        run_validator(TypeChecker(symbol_table), program_ast)


def test_fails_on_calling_procedure_as_expression():
    """Test that calling a Procedure (no return value) as an expression fails."""
    p, main = Identifier("p"), Identifier("main")
    a, t = Identifier("a"), Identifier("t")
    int32_scalar = NumericalType(PrimitiveDataType(CoreDataType.INT32))

    def qt(t_, q):
        return QualifiedType(base_type=t_, type_qualifier=q)

    callee = Procedure(
        name=p,
        args=(Argument(name=a, qualified_type=qt(int32_scalar, TypeQualifier.INPUT)),),
        body=(),
    )
    caller = Operation(
        name=main,
        args=(Argument(name=a, qualified_type=qt(int32_scalar, TypeQualifier.INPUT)),),
        body=(
            DeclarationStatement(
                variable_name=t,
                variable_type=qt(int32_scalar, TypeQualifier.TEMP),
                expression=FunctionExpression(
                    function=IdentifierExpression(identifier=p),
                    args=(IdentifierExpression(identifier=a),),
                ),
            ),
            ReturnStatement(expression=IdentifierExpression(identifier=t)),
        ),
        return_type=qt(int32_scalar, TypeQualifier.OUTPUT),
    )
    program_ast = Module(statements=(callee, caller))
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(ValidationFailedError, match="cannot be used as an expression"):
        run_validator(TypeChecker(symbol_table), program_ast)


def test_fails_on_int_literal_overflowing_all_integer_types():
    """Test that an integer literal too large for any supported type fails."""
    op = Identifier("op")
    a, t = Identifier("a"), Identifier("t")
    int32_scalar = NumericalType(PrimitiveDataType(CoreDataType.INT32))

    def qt(t_, q):
        return QualifiedType(base_type=t_, type_qualifier=q)

    # 2**128 exceeds every supported integer width.
    program_ast = Module(
        statements=(
            Operation(
                name=op,
                args=(
                    Argument(
                        name=a, qualified_type=qt(int32_scalar, TypeQualifier.INPUT)
                    ),
                ),
                body=(
                    DeclarationStatement(
                        variable_name=t,
                        variable_type=qt(int32_scalar, TypeQualifier.TEMP),
                        expression=IntLiteral(value=2**128),
                    ),
                    ReturnStatement(expression=IdentifierExpression(identifier=t)),
                ),
                return_type=qt(int32_scalar, TypeQualifier.OUTPUT),
            ),
        ),
    )
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(
        ValidationFailedError, match="does not fit in any supported integer type"
    ):
        run_validator(TypeChecker(symbol_table), program_ast)


def test_fails_on_complex_literal_in_expression():
    """Test that a ComplexLiteral is rejected by the type checker."""
    op = Identifier("op")
    a, t = Identifier("a"), Identifier("t")
    int32_scalar = NumericalType(PrimitiveDataType(CoreDataType.INT32))

    def qt(t_, q):
        return QualifiedType(base_type=t_, type_qualifier=q)

    program_ast = Module(
        statements=(
            Operation(
                name=op,
                args=(
                    Argument(
                        name=a, qualified_type=qt(int32_scalar, TypeQualifier.INPUT)
                    ),
                ),
                body=(
                    DeclarationStatement(
                        variable_name=t,
                        variable_type=qt(int32_scalar, TypeQualifier.TEMP),
                        expression=ComplexLiteral(value=complex(1, 2)),
                    ),
                    ReturnStatement(expression=IdentifierExpression(identifier=t)),
                ),
                return_type=qt(int32_scalar, TypeQualifier.OUTPUT),
            ),
        ),
    )
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(
        ValidationFailedError, match="Complex literals are not yet supported"
    ):
        run_validator(TypeChecker(symbol_table), program_ast)


def test_valid_negative_int_literal_resolves_to_signed_type():
    """Test that a negative IntLiteral resolves through the signed integer chain."""
    op = Identifier("op")
    a, t = Identifier("a"), Identifier("t")
    int32_scalar = NumericalType(PrimitiveDataType(CoreDataType.INT32))

    def qt(t_, q):
        return QualifiedType(base_type=t_, type_qualifier=q)

    program_ast = Module(
        statements=(
            Operation(
                name=op,
                args=(
                    Argument(
                        name=a, qualified_type=qt(int32_scalar, TypeQualifier.INPUT)
                    ),
                ),
                body=(
                    DeclarationStatement(
                        variable_name=t,
                        variable_type=qt(int32_scalar, TypeQualifier.TEMP),
                        expression=IntLiteral(value=-1),
                    ),
                    ReturnStatement(expression=IdentifierExpression(identifier=t)),
                ),
                return_type=qt(int32_scalar, TypeQualifier.OUTPUT),
            ),
        ),
    )
    symbol_table = build_symbol_table(program_ast)

    # -1 picks the INT weak target, then resolves to a signed concrete type
    # that promotes into int32.
    run_validator(TypeChecker(symbol_table), program_ast)


def test_fails_on_bare_procedure_call_with_mismatched_argument_type():
    """Test bare-statement procedure call validates argument types."""
    p, main = Identifier("p"), Identifier("main")
    a, b = Identifier("a"), Identifier("b")
    int32_scalar = NumericalType(PrimitiveDataType(CoreDataType.INT32))
    float32_scalar = NumericalType(PrimitiveDataType(CoreDataType.FLOAT32))

    def qt(t_, q):
        return QualifiedType(base_type=t_, type_qualifier=q)

    callee = Procedure(
        name=p,
        args=(Argument(name=a, qualified_type=qt(int32_scalar, TypeQualifier.INPUT)),),
        body=(),
    )
    caller = Procedure(
        name=main,
        args=(
            Argument(name=b, qualified_type=qt(float32_scalar, TypeQualifier.INPUT)),
        ),
        body=(
            ExpressionStatement(
                left=None,
                right=FunctionExpression(
                    function=IdentifierExpression(identifier=p),
                    args=(IdentifierExpression(identifier=b),),
                ),
            ),
        ),
    )
    program_ast = Module(statements=(callee, caller))
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(ValidationFailedError, match="cannot pass"):
        run_validator(TypeChecker(symbol_table), program_ast)


def test_valid_float_literal_in_declaration():
    """Test that a FloatLiteral expression infers to float and matches a float LHS."""
    op = Identifier("op")
    a, t = Identifier("a"), Identifier("t")
    float32_scalar = _make_float32_type()

    def qt(t_, q):
        return QualifiedType(base_type=t_, type_qualifier=q)

    program_ast = Module(
        statements=(
            Operation(
                name=op,
                args=(
                    Argument(
                        name=a, qualified_type=qt(float32_scalar, TypeQualifier.INPUT)
                    ),
                ),
                body=(
                    DeclarationStatement(
                        variable_name=t,
                        variable_type=qt(float32_scalar, TypeQualifier.TEMP),
                        expression=FloatLiteral(value=3.14),
                    ),
                    ReturnStatement(expression=IdentifierExpression(identifier=t)),
                ),
                return_type=qt(float32_scalar, TypeQualifier.OUTPUT),
            ),
        ),
    )
    symbol_table = build_symbol_table(program_ast)

    run_validator(TypeChecker(symbol_table), program_ast)


def test_fails_on_ternary_condition_with_index_type():
    """Test ternary fails when condition resolves to a non-numerical (index) type."""
    op = Identifier("op")
    a, i = Identifier("a"), Identifier("i")
    int32_scalar = NumericalType(PrimitiveDataType(CoreDataType.INT32))

    def qt(t_, q):
        return QualifiedType(base_type=t_, type_qualifier=q)

    index_type = IndexType(
        lower_bound=LiteralExpression(1),
        upper_bound=LiteralExpression(3),
        stride=None,
    )
    program_ast = Module(
        statements=(
            Operation(
                name=op,
                args=(
                    Argument(
                        name=a, qualified_type=qt(int32_scalar, TypeQualifier.INPUT)
                    ),
                ),
                body=(
                    DeclarationStatement(
                        variable_name=i,
                        variable_type=qt(index_type, TypeQualifier.TEMP),
                    ),
                    ReturnStatement(
                        expression=TernaryExpression(
                            condition=IdentifierExpression(identifier=i),
                            true=IntLiteral(value=1),
                            false=IntLiteral(value=0),
                        ),
                    ),
                ),
                return_type=qt(int32_scalar, TypeQualifier.OUTPUT),
            ),
        ),
    )
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(
        ValidationFailedError, match="Ternary condition must be a numerical scalar"
    ):
        run_validator(TypeChecker(symbol_table), program_ast)


def test_fails_on_array_access_on_non_numerical_base():
    """Test that indexing a non-numerical (index-typed) identifier fails."""
    op = Identifier("op")
    a, i = Identifier("a"), Identifier("i")
    int32_scalar = NumericalType(PrimitiveDataType(CoreDataType.INT32))

    def qt(t_, q):
        return QualifiedType(base_type=t_, type_qualifier=q)

    index_type = IndexType(
        lower_bound=LiteralExpression(1),
        upper_bound=LiteralExpression(3),
        stride=None,
    )
    program_ast = Module(
        statements=(
            Operation(
                name=op,
                args=(
                    Argument(
                        name=a, qualified_type=qt(int32_scalar, TypeQualifier.INPUT)
                    ),
                ),
                body=(
                    DeclarationStatement(
                        variable_name=i,
                        variable_type=qt(index_type, TypeQualifier.TEMP),
                    ),
                    ReturnStatement(
                        expression=ArrayAccessExpression(
                            array_expression=IdentifierExpression(identifier=i),
                            indices=(IdentifierExpression(identifier=i),),
                        ),
                    ),
                ),
                return_type=qt(int32_scalar, TypeQualifier.OUTPUT),
            ),
        ),
    )
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(
        ValidationFailedError, match="Array access requires a numerical type"
    ):
        run_validator(TypeChecker(symbol_table), program_ast)


def test_fails_on_binary_with_promotion_failure():
    """Test binary expression fails when operand data types cannot be promoted."""
    op = Identifier("op")
    a, b = Identifier("a"), Identifier("b")
    int32_scalar = NumericalType(PrimitiveDataType(CoreDataType.INT32))
    complex_scalar = NumericalType(PrimitiveDataType(CoreDataType.COMPLEX64))
    int64_scalar = NumericalType(PrimitiveDataType(CoreDataType.INT64))

    def qt(t_, q):
        return QualifiedType(base_type=t_, type_qualifier=q)

    # Mixing int64 and complex64 has no common promotion target.
    program_ast = Module(
        statements=(
            Operation(
                name=op,
                args=(
                    Argument(
                        name=a, qualified_type=qt(int64_scalar, TypeQualifier.INPUT)
                    ),
                    Argument(
                        name=b, qualified_type=qt(complex_scalar, TypeQualifier.INPUT)
                    ),
                ),
                body=(
                    ReturnStatement(
                        expression=BinaryExpression(
                            operation=BinaryOperation.ADDITION,
                            left=IdentifierExpression(identifier=a),
                            right=IdentifierExpression(identifier=b),
                        ),
                    ),
                ),
                return_type=qt(int32_scalar, TypeQualifier.OUTPUT),
            ),
        ),
    )
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(ValidationFailedError, match="Cannot promote"):
        run_validator(TypeChecker(symbol_table), program_ast)


def test_valid_all_literal_tensor_shape():
    """Test a procedure with an all-literal tensor shape `int32[1, 32, 32, 16]`
    validates."""
    main = Identifier("main")
    a = Identifier("a")
    int32_4d_literal = NumericalType(
        PrimitiveDataType(CoreDataType.INT32),
        shape=(
            LiteralExpression(1),
            LiteralExpression(32),
            LiteralExpression(32),
            LiteralExpression(16),
        ),
    )
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(
                    Argument(
                        name=a,
                        qualified_type=QualifiedType(
                            base_type=int32_4d_literal,
                            type_qualifier=TypeQualifier.INPUT,
                        ),
                    ),
                ),
                body=(),
            ),
        ),
    )
    symbol_table = build_symbol_table(program_ast)

    run_validator(TypeChecker(symbol_table), program_ast)


def test_valid_index_range_with_param_stride():
    """Test that an index range whose stride references a `param` validates
    ."""
    main = Identifier("main")
    k_param, i_index = Identifier("K"), Identifier("i")
    n_dim = Identifier("N")
    uint32_scalar = NumericalType(PrimitiveDataType(CoreDataType.UINT32))
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(
                    Argument(
                        name=k_param,
                        qualified_type=QualifiedType(
                            base_type=uint32_scalar,
                            type_qualifier=TypeQualifier.PARAM,
                        ),
                    ),
                ),
                body=(
                    DeclarationStatement(
                        variable_name=i_index,
                        variable_type=QualifiedType(
                            base_type=IndexType(
                                lower_bound=LiteralExpression(1),
                                upper_bound=CoreIdentifierExpression(n_dim),
                                stride=CoreIdentifierExpression(k_param),
                            ),
                            type_qualifier=TypeQualifier.TEMP,
                        ),
                    ),
                ),
            ),
        ),
    )
    symbol_table = build_symbol_table(program_ast)

    run_validator(TypeChecker(symbol_table), program_ast)


def test_valid_single_placeholder_dim_solved():
    """Test a call site solves a single-placeholder formal dim."""
    f, main = Identifier("f"), Identifier("main")
    x, a = Identifier("x"), Identifier("a")
    n_formal = Identifier("N")
    int32_scalar = NumericalType(PrimitiveDataType(CoreDataType.INT32))
    int32_vec_n = NumericalType(
        PrimitiveDataType(CoreDataType.INT32),
        shape=(CoreIdentifierExpression(n_formal),),
    )
    int32_vec5 = NumericalType(
        PrimitiveDataType(CoreDataType.INT32), shape=(LiteralExpression(5),)
    )

    def qt(t_, q):
        return QualifiedType(base_type=t_, type_qualifier=q)

    callee = Operation(
        name=f,
        args=(Argument(name=x, qualified_type=qt(int32_vec_n, TypeQualifier.INPUT)),),
        body=(ReturnStatement(expression=IntLiteral(value=0)),),
        return_type=qt(int32_scalar, TypeQualifier.OUTPUT),
    )
    caller = Procedure(
        name=main,
        args=(Argument(name=a, qualified_type=qt(int32_vec5, TypeQualifier.INPUT)),),
        body=(
            ExpressionStatement(
                left=None,
                right=FunctionExpression(
                    function=IdentifierExpression(identifier=f),
                    args=(IdentifierExpression(identifier=a),),
                ),
            ),
        ),
    )
    program_ast = Module(statements=(callee, caller))
    symbol_table = build_symbol_table(program_ast)

    run_validator(TypeChecker(symbol_table), program_ast)


def test_valid_arithmetic_placeholder_dim_solved():
    """Test a call site solves a formal dim of the form `N + 4`."""
    f, main = Identifier("f"), Identifier("main")
    x, a = Identifier("x"), Identifier("a")
    n_formal = Identifier("N")
    int32_scalar = NumericalType(PrimitiveDataType(CoreDataType.INT32))
    # Formal shape: N + 4
    formal_dim = CoreBinaryExpression(
        operation=CoreBinaryOperation.ADD,
        left=CoreIdentifierExpression(n_formal),
        right=LiteralExpression(4),
    )
    int32_vec_n_plus_4 = NumericalType(
        PrimitiveDataType(CoreDataType.INT32), shape=(formal_dim,)
    )
    int32_vec8 = NumericalType(
        PrimitiveDataType(CoreDataType.INT32), shape=(LiteralExpression(8),)
    )

    def qt(t_, q):
        return QualifiedType(base_type=t_, type_qualifier=q)

    callee = Operation(
        name=f,
        args=(
            Argument(
                name=x, qualified_type=qt(int32_vec_n_plus_4, TypeQualifier.INPUT)
            ),
        ),
        body=(ReturnStatement(expression=IntLiteral(value=0)),),
        return_type=qt(int32_scalar, TypeQualifier.OUTPUT),
    )
    caller = Procedure(
        name=main,
        args=(Argument(name=a, qualified_type=qt(int32_vec8, TypeQualifier.INPUT)),),
        body=(
            ExpressionStatement(
                left=None,
                right=FunctionExpression(
                    function=IdentifierExpression(identifier=f),
                    args=(IdentifierExpression(identifier=a),),
                ),
            ),
        ),
    )
    program_ast = Module(statements=(callee, caller))
    symbol_table = build_symbol_table(program_ast)

    # N solves to 4 (because 8 == N + 4); the call type-checks.
    run_validator(TypeChecker(symbol_table), program_ast)


def test_fails_on_non_reducible_shape_dimension():
    """Test that a tensor shape mentioning a non-compile-time-reducible
    identifier is rejected."""
    main = Identifier("main")
    x_var, arr = Identifier("x"), Identifier("arr")
    int32_scalar = NumericalType(PrimitiveDataType(CoreDataType.INT32))
    # `temp int32[x]` where x is a non-param temp variable.
    int32_vec_x = NumericalType(
        PrimitiveDataType(CoreDataType.INT32),
        shape=(CoreIdentifierExpression(x_var),),
    )

    def qt(t_, q):
        return QualifiedType(base_type=t_, type_qualifier=q)

    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(),
                body=(
                    DeclarationStatement(
                        variable_name=x_var,
                        variable_type=qt(int32_scalar, TypeQualifier.TEMP),
                        expression=IntLiteral(value=5),
                    ),
                    DeclarationStatement(
                        variable_name=arr,
                        variable_type=qt(int32_vec_x, TypeQualifier.TEMP),
                    ),
                ),
            ),
        ),
    )
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(ValidationFailedError):
        run_validator(TypeChecker(symbol_table), program_ast)


def test_fails_on_non_reducible_index_bound():
    """Test that an index type whose bound references a non-compile-time-
    reducible identifier is rejected."""
    main = Identifier("main")
    x_var, i_index = Identifier("x"), Identifier("i")
    n_dim = Identifier("N")
    int32_scalar = NumericalType(PrimitiveDataType(CoreDataType.INT32))

    def qt(t_, q):
        return QualifiedType(base_type=t_, type_qualifier=q)

    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(),
                body=(
                    DeclarationStatement(
                        variable_name=x_var,
                        variable_type=qt(int32_scalar, TypeQualifier.TEMP),
                        expression=IntLiteral(value=2),
                    ),
                    DeclarationStatement(
                        variable_name=i_index,
                        variable_type=qt(
                            IndexType(
                                lower_bound=CoreIdentifierExpression(x_var),
                                upper_bound=CoreIdentifierExpression(n_dim),
                                stride=None,
                            ),
                            TypeQualifier.TEMP,
                        ),
                    ),
                ),
            ),
        ),
    )
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(ValidationFailedError):
        run_validator(TypeChecker(symbol_table), program_ast)


def test_fails_on_placeholder_reassigned_in_same_call():
    """Test that a call site rejects an actual whose dims would force the
    same formal placeholder to take two different values."""
    f, main = Identifier("f"), Identifier("main")
    x, a = Identifier("x"), Identifier("a")
    n_formal = Identifier("N")
    int32_scalar = NumericalType(PrimitiveDataType(CoreDataType.INT32))
    int32_n_by_n = NumericalType(
        PrimitiveDataType(CoreDataType.INT32),
        shape=(
            CoreIdentifierExpression(n_formal),
            CoreIdentifierExpression(n_formal),
        ),
    )
    int32_3_by_5 = NumericalType(
        PrimitiveDataType(CoreDataType.INT32),
        shape=(LiteralExpression(3), LiteralExpression(5)),
    )

    def qt(t_, q):
        return QualifiedType(base_type=t_, type_qualifier=q)

    callee = Operation(
        name=f,
        args=(Argument(name=x, qualified_type=qt(int32_n_by_n, TypeQualifier.INPUT)),),
        body=(ReturnStatement(expression=IntLiteral(value=0)),),
        return_type=qt(int32_scalar, TypeQualifier.OUTPUT),
    )
    caller = Procedure(
        name=main,
        args=(Argument(name=a, qualified_type=qt(int32_3_by_5, TypeQualifier.INPUT)),),
        body=(
            ExpressionStatement(
                left=None,
                right=FunctionExpression(
                    function=IdentifierExpression(identifier=f),
                    args=(IdentifierExpression(identifier=a),),
                ),
            ),
        ),
    )
    program_ast = Module(statements=(callee, caller))
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(ValidationFailedError):
        run_validator(TypeChecker(symbol_table), program_ast)


def test_fails_on_two_placeholders_in_one_dim_unsolvable():
    """Test a formal dim of `N + M` with only one actual dim is unsolvable
    ."""
    f, main = Identifier("f"), Identifier("main")
    x, a = Identifier("x"), Identifier("a")
    n_formal, m_formal = Identifier("N"), Identifier("M")
    int32_scalar = NumericalType(PrimitiveDataType(CoreDataType.INT32))
    formal_dim = CoreBinaryExpression(
        operation=CoreBinaryOperation.ADD,
        left=CoreIdentifierExpression(n_formal),
        right=CoreIdentifierExpression(m_formal),
    )
    int32_vec_n_plus_m = NumericalType(
        PrimitiveDataType(CoreDataType.INT32), shape=(formal_dim,)
    )
    int32_vec10 = NumericalType(
        PrimitiveDataType(CoreDataType.INT32), shape=(LiteralExpression(10),)
    )

    def qt(t_, q):
        return QualifiedType(base_type=t_, type_qualifier=q)

    callee = Operation(
        name=f,
        args=(
            Argument(
                name=x, qualified_type=qt(int32_vec_n_plus_m, TypeQualifier.INPUT)
            ),
        ),
        body=(ReturnStatement(expression=IntLiteral(value=0)),),
        return_type=qt(int32_scalar, TypeQualifier.OUTPUT),
    )
    caller = Procedure(
        name=main,
        args=(Argument(name=a, qualified_type=qt(int32_vec10, TypeQualifier.INPUT)),),
        body=(
            ExpressionStatement(
                left=None,
                right=FunctionExpression(
                    function=IdentifierExpression(identifier=f),
                    args=(IdentifierExpression(identifier=a),),
                ),
            ),
        ),
    )
    program_ast = Module(statements=(callee, caller))
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(ValidationFailedError):
        run_validator(TypeChecker(symbol_table), program_ast)


def test_fails_on_index_only_on_rhs_without_consumer():
    """Test rejection of indexed assignment whose RHS uses an index variable
     that is not on the LHS and not consumed by any reduction or forall
    ."""
    main = Identifier("main")
    out_, in_ = Identifier("out"), Identifier("in")
    i_, j_ = Identifier("i"), Identifier("j")
    n_dim, m_dim = Identifier("N"), Identifier("M")

    def qt(t_, q):
        return QualifiedType(base_type=t_, type_qualifier=q)

    int32_vec_n = NumericalType(
        PrimitiveDataType(CoreDataType.INT32),
        shape=(CoreIdentifierExpression(n_dim),),
    )
    int32_mat_n_m = NumericalType(
        PrimitiveDataType(CoreDataType.INT32),
        shape=(
            CoreIdentifierExpression(n_dim),
            CoreIdentifierExpression(m_dim),
        ),
    )

    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(
                    Argument(
                        name=in_,
                        qualified_type=qt(int32_mat_n_m, TypeQualifier.INPUT),
                    ),
                    Argument(
                        name=out_,
                        qualified_type=qt(int32_vec_n, TypeQualifier.OUTPUT),
                    ),
                ),
                body=(
                    DeclarationStatement(
                        variable_name=i_,
                        variable_type=qt(
                            IndexType(
                                lower_bound=LiteralExpression(1),
                                upper_bound=CoreIdentifierExpression(n_dim),
                                stride=None,
                            ),
                            TypeQualifier.TEMP,
                        ),
                    ),
                    DeclarationStatement(
                        variable_name=j_,
                        variable_type=qt(
                            IndexType(
                                lower_bound=LiteralExpression(1),
                                upper_bound=CoreIdentifierExpression(m_dim),
                                stride=None,
                            ),
                            TypeQualifier.TEMP,
                        ),
                    ),
                    # out[i] = in[i, j]  --  j is free on RHS, not on LHS,
                    # and not consumed by any enclosing reduction or forall.
                    ExpressionStatement(
                        left=ArrayAccessExpression(
                            array_expression=IdentifierExpression(identifier=out_),
                            indices=(IdentifierExpression(identifier=i_),),
                        ),
                        right=ArrayAccessExpression(
                            array_expression=IdentifierExpression(identifier=in_),
                            indices=(
                                IdentifierExpression(identifier=i_),
                                IdentifierExpression(identifier=j_),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(ValidationFailedError):
        run_validator(TypeChecker(symbol_table), program_ast)
