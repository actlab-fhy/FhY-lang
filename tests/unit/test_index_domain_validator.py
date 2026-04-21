"""Tests the index domain validator AST pass."""

import pytest
from fhy.lang.ast import (
    Argument,
    ArrayAccessExpression,
    DeclarationStatement,
    ExpressionStatement,
    FloatLiteral,
    IdentifierExpression,
    IntLiteral,
    Module,
    Procedure,
    QualifiedType,
    TupleAccessExpression,
)
from fhy.lang.ast.passes import (
    IndexDomainValidator,
    build_symbol_table,
)
from fhy_core import (
    CoreDataType,
    Identifier,
    IndexType,
    LiteralExpression,
    NumericalType,
    PrimitiveDataType,
    Provenance,
    TypeQualifier,
    ValidationFailedError,
)
from fhy_core import (
    IdentifierExpression as CoreIdentifierExpression,
)

from .utils import run_validator


def _make_int32_vector(size_identifier: Identifier) -> NumericalType:
    return NumericalType(
        PrimitiveDataType(CoreDataType.INT32),
        shape=(CoreIdentifierExpression(size_identifier),),
    )


def test_empty_module(empty_module_ast):
    """Test validation of an empty module."""
    symbol_table = build_symbol_table(empty_module_ast)

    run_validator(IndexDomainValidator(symbol_table), empty_module_ast)


def test_valid_in_bounds_symbolic_access():
    """Test an array access whose symbolic index range is in bounds."""
    main = Identifier("main")
    m = Identifier("m")
    a, i = Identifier("a"), Identifier("i")
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(
                    Argument(
                        name=a,
                        qualified_type=QualifiedType(
                            base_type=_make_int32_vector(m),
                            type_qualifier=TypeQualifier.INPUT,
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
                                upper_bound=CoreIdentifierExpression(m),
                                stride=None,
                            ),
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    ExpressionStatement(
                        left=None,
                        right=ArrayAccessExpression(
                            array_expression=IdentifierExpression(
                                identifier=a, provenance=Provenance.unknown()
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
                ),
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )
    symbol_table = build_symbol_table(program_ast)

    run_validator(IndexDomainValidator(symbol_table), program_ast)


def test_valid_in_bounds_constant_access():
    """Test an array access with constant bounds that are in range."""
    main = Identifier("main")
    a, i = Identifier("a"), Identifier("i")
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(
                    Argument(
                        name=a,
                        qualified_type=QualifiedType(
                            base_type=NumericalType(
                                PrimitiveDataType(CoreDataType.INT32),
                                shape=(LiteralExpression(10),),
                            ),
                            type_qualifier=TypeQualifier.INPUT,
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
                                lower_bound=LiteralExpression(2),
                                upper_bound=LiteralExpression(5),
                                stride=None,
                            ),
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    ExpressionStatement(
                        left=None,
                        right=ArrayAccessExpression(
                            array_expression=IdentifierExpression(
                                identifier=a, provenance=Provenance.unknown()
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
                ),
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )
    symbol_table = build_symbol_table(program_ast)

    run_validator(IndexDomainValidator(symbol_table), program_ast)


def test_fails_with_constant_out_of_bounds_lower():
    """Test failure when a constant index range's lower bound is below 1."""
    main = Identifier("main")
    a, i = Identifier("a"), Identifier("i")
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(
                    Argument(
                        name=a,
                        qualified_type=QualifiedType(
                            base_type=NumericalType(
                                PrimitiveDataType(CoreDataType.INT32),
                                shape=(LiteralExpression(10),),
                            ),
                            type_qualifier=TypeQualifier.INPUT,
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
                                lower_bound=LiteralExpression(0),
                                upper_bound=LiteralExpression(5),
                                stride=None,
                            ),
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    ExpressionStatement(
                        left=None,
                        right=ArrayAccessExpression(
                            array_expression=IdentifierExpression(
                                identifier=a, provenance=Provenance.unknown()
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
                ),
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(ValidationFailedError, match="semantic error"):
        run_validator(IndexDomainValidator(symbol_table), program_ast)


def test_fails_with_constant_out_of_bounds_upper():
    """Test failure when a constant index range's upper bound exceeds size."""
    main = Identifier("main")
    a, i = Identifier("a"), Identifier("i")
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(
                    Argument(
                        name=a,
                        qualified_type=QualifiedType(
                            base_type=NumericalType(
                                PrimitiveDataType(CoreDataType.INT32),
                                shape=(LiteralExpression(10),),
                            ),
                            type_qualifier=TypeQualifier.INPUT,
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
                                upper_bound=LiteralExpression(20),
                                stride=None,
                            ),
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    ExpressionStatement(
                        left=None,
                        right=ArrayAccessExpression(
                            array_expression=IdentifierExpression(
                                identifier=a, provenance=Provenance.unknown()
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
                ),
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(ValidationFailedError, match="semantic error"):
        run_validator(IndexDomainValidator(symbol_table), program_ast)


def test_fails_with_symbolic_mismatched_dimension():
    """Test failure when the symbolic index upper bound doesn't match the array size."""
    main = Identifier("main")
    m, n = Identifier("m"), Identifier("n")
    a, i = Identifier("a"), Identifier("i")
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(
                    Argument(
                        name=a,
                        qualified_type=QualifiedType(
                            base_type=_make_int32_vector(m),
                            type_qualifier=TypeQualifier.INPUT,
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
                                upper_bound=CoreIdentifierExpression(n),
                                stride=None,
                            ),
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    ExpressionStatement(
                        left=None,
                        right=ArrayAccessExpression(
                            array_expression=IdentifierExpression(
                                identifier=a, provenance=Provenance.unknown()
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
                ),
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(ValidationFailedError, match="semantic error"):
        run_validator(IndexDomainValidator(symbol_table), program_ast)


def test_valid_scalar_literal_access():
    """Test an array access with a scalar unsigned-integer PARAM literal."""
    main = Identifier("main")
    a = Identifier("a")
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(
                    Argument(
                        name=a,
                        qualified_type=QualifiedType(
                            base_type=NumericalType(
                                PrimitiveDataType(CoreDataType.INT32),
                                shape=(LiteralExpression(10),),
                            ),
                            type_qualifier=TypeQualifier.INPUT,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                ),
                body=(
                    ExpressionStatement(
                        left=None,
                        right=ArrayAccessExpression(
                            array_expression=IdentifierExpression(
                                identifier=a, provenance=Provenance.unknown()
                            ),
                            indices=(
                                IntLiteral(
                                    value=5,
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

    run_validator(IndexDomainValidator(symbol_table), program_ast)


def test_valid_scalar_param_identifier_access():
    """Test an array access with a scalar unsigned-integer PARAM identifier."""
    main = Identifier("main")
    m = Identifier("m")
    a, p = Identifier("a"), Identifier("p")
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(
                    Argument(
                        name=a,
                        qualified_type=QualifiedType(
                            base_type=_make_int32_vector(m),
                            type_qualifier=TypeQualifier.INPUT,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    Argument(
                        name=p,
                        qualified_type=QualifiedType(
                            base_type=NumericalType(
                                PrimitiveDataType(CoreDataType.UINT32),
                            ),
                            type_qualifier=TypeQualifier.PARAM,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                ),
                body=(
                    ExpressionStatement(
                        left=None,
                        right=ArrayAccessExpression(
                            array_expression=IdentifierExpression(
                                identifier=a, provenance=Provenance.unknown()
                            ),
                            indices=(
                                IdentifierExpression(
                                    identifier=p,
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

    # p is unconstrained, so z3 can find violations (e.g., p = 0 or p > m).
    with pytest.raises(ValidationFailedError, match="semantic error"):
        run_validator(IndexDomainValidator(symbol_table), program_ast)


def test_fails_with_scalar_literal_out_of_bounds():
    """Test failure when a scalar literal index is out of bounds."""
    main = Identifier("main")
    a = Identifier("a")
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(
                    Argument(
                        name=a,
                        qualified_type=QualifiedType(
                            base_type=NumericalType(
                                PrimitiveDataType(CoreDataType.INT32),
                                shape=(LiteralExpression(10),),
                            ),
                            type_qualifier=TypeQualifier.INPUT,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                ),
                body=(
                    ExpressionStatement(
                        left=None,
                        right=ArrayAccessExpression(
                            array_expression=IdentifierExpression(
                                identifier=a, provenance=Provenance.unknown()
                            ),
                            indices=(
                                IntLiteral(
                                    value=0,
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

    with pytest.raises(ValidationFailedError, match="semantic error"):
        run_validator(IndexDomainValidator(symbol_table), program_ast)


def test_fails_with_float_literal_index():
    """Test failure when the index is a non-unsigned-integer value."""
    main = Identifier("main")
    a = Identifier("a")
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(
                    Argument(
                        name=a,
                        qualified_type=QualifiedType(
                            base_type=NumericalType(
                                PrimitiveDataType(CoreDataType.INT32),
                                shape=(LiteralExpression(10),),
                            ),
                            type_qualifier=TypeQualifier.INPUT,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                ),
                body=(
                    ExpressionStatement(
                        left=None,
                        right=ArrayAccessExpression(
                            array_expression=IdentifierExpression(
                                identifier=a, provenance=Provenance.unknown()
                            ),
                            indices=(
                                FloatLiteral(
                                    value=1.5,
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

    with pytest.raises(ValidationFailedError, match="type error"):
        run_validator(IndexDomainValidator(symbol_table), program_ast)


def test_fails_with_non_param_scalar_index(int32: NumericalType):
    """Test failure when a scalar integer index is not a PARAM."""
    main = Identifier("main")
    a, t = Identifier("a"), Identifier("t")
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(
                    Argument(
                        name=a,
                        qualified_type=QualifiedType(
                            base_type=NumericalType(
                                PrimitiveDataType(CoreDataType.INT32),
                                shape=(LiteralExpression(10),),
                            ),
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
                            base_type=NumericalType(
                                PrimitiveDataType(CoreDataType.UINT32),
                            ),
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    ExpressionStatement(
                        left=None,
                        right=ArrayAccessExpression(
                            array_expression=IdentifierExpression(
                                identifier=a, provenance=Provenance.unknown()
                            ),
                            indices=(
                                IdentifierExpression(
                                    identifier=t,
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

    with pytest.raises(ValidationFailedError, match="type error"):
        run_validator(IndexDomainValidator(symbol_table), program_ast)


def test_fails_with_non_identifier_array_expression(int32: NumericalType):
    """Test failure when the array expression is not an identifier."""
    main = Identifier("main")
    m = Identifier("m")
    a, i = Identifier("a"), Identifier("i")
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(
                    Argument(
                        name=a,
                        qualified_type=QualifiedType(
                            base_type=_make_int32_vector(m),
                            type_qualifier=TypeQualifier.INPUT,
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
                                upper_bound=CoreIdentifierExpression(m),
                                stride=None,
                            ),
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    ExpressionStatement(
                        left=None,
                        right=ArrayAccessExpression(
                            array_expression=ArrayAccessExpression(
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
                ),
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(ValidationFailedError, match="structural error"):
        run_validator(IndexDomainValidator(symbol_table), program_ast)


def test_fails_when_array_variable_is_not_numerical():
    """Test failure when the array variable's type is not a NumericalType."""
    main = Identifier("main")
    i, j = Identifier("i"), Identifier("j")
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(),
                body=(
                    DeclarationStatement(
                        variable_name=i,
                        variable_type=QualifiedType(
                            base_type=IndexType(
                                lower_bound=LiteralExpression(1),
                                upper_bound=LiteralExpression(10),
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
                                upper_bound=LiteralExpression(10),
                                stride=None,
                            ),
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    ExpressionStatement(
                        left=None,
                        right=ArrayAccessExpression(
                            array_expression=IdentifierExpression(
                                identifier=i, provenance=Provenance.unknown()
                            ),
                            indices=(
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

    with pytest.raises(ValidationFailedError, match="type error"):
        run_validator(IndexDomainValidator(symbol_table), program_ast)


def test_fails_with_indices_shape_mismatch():
    """Test failure when the number of indices does not match the dimensions."""
    main = Identifier("main")
    m = Identifier("m")
    a, i = Identifier("a"), Identifier("i")
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(
                    Argument(
                        name=a,
                        qualified_type=QualifiedType(
                            base_type=_make_int32_vector(m),
                            type_qualifier=TypeQualifier.INPUT,
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
                                upper_bound=CoreIdentifierExpression(m),
                                stride=None,
                            ),
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    ExpressionStatement(
                        left=None,
                        right=ArrayAccessExpression(
                            array_expression=IdentifierExpression(
                                identifier=a, provenance=Provenance.unknown()
                            ),
                            indices=(
                                IdentifierExpression(
                                    identifier=i,
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
    )
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(ValidationFailedError, match="structural error"):
        run_validator(IndexDomainValidator(symbol_table), program_ast)


def test_fails_with_tuple_access_index():
    """Test failure when an array-access index is a tuple access expression."""
    main = Identifier("main")
    a, t = Identifier("a"), Identifier("t")
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(
                    Argument(
                        name=a,
                        qualified_type=QualifiedType(
                            base_type=NumericalType(
                                PrimitiveDataType(CoreDataType.INT32),
                                shape=(LiteralExpression(10),),
                            ),
                            type_qualifier=TypeQualifier.INPUT,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                ),
                body=(
                    ExpressionStatement(
                        left=None,
                        right=ArrayAccessExpression(
                            array_expression=IdentifierExpression(
                                identifier=a, provenance=Provenance.unknown()
                            ),
                            indices=(
                                TupleAccessExpression(
                                    tuple_expression=IdentifierExpression(
                                        identifier=t,
                                        provenance=Provenance.unknown(),
                                    ),
                                    element_index=IntLiteral(
                                        value=0,
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

    with pytest.raises(ValidationFailedError, match="type error"):
        run_validator(IndexDomainValidator(symbol_table), program_ast)
