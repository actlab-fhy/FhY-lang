"""Tests the tuple access validator AST pass."""

import pytest
from fhy_core import (
    CoreDataType,
    Identifier,
    NumericalType,
    PrimitiveDataType,
    Provenance,
    TupleType,
    TypeQualifier,
    ValidationFailedError,
)

from fhy_lang.lang.ast import (
    DeclarationStatement,
    ExpressionStatement,
    IdentifierExpression,
    IntLiteral,
    Module,
    Procedure,
    QualifiedType,
    TupleAccessExpression,
    TupleExpression,
)
from fhy_lang.lang.ast.passes import (
    TupleAccessValidator,
    build_symbol_table,
)

from .utils import run_validator


@pytest.fixture
def int32_float_pair_type() -> TupleType:
    return TupleType(
        [
            NumericalType(PrimitiveDataType(CoreDataType.INT32)),
            NumericalType(PrimitiveDataType(CoreDataType.FLOAT)),
        ]
    )


def test_empty_module(empty_module_ast):
    """Test validation of an empty module."""
    symbol_table = build_symbol_table(empty_module_ast)

    run_validator(TupleAccessValidator(symbol_table), empty_module_ast)


def test_valid_tuple_access_on_identifier(int32_float_pair_type: TupleType):
    """Test a tuple access with an in-range element index on an identifier."""
    main = Identifier("main")
    t, u = Identifier("t"), Identifier("u")
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(),
                body=(
                    DeclarationStatement(
                        variable_name=t,
                        variable_type=QualifiedType(
                            base_type=int32_float_pair_type,
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    DeclarationStatement(
                        variable_name=u,
                        variable_type=QualifiedType(
                            base_type=int32_float_pair_type,
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    ExpressionStatement(
                        left=IdentifierExpression(
                            identifier=u, provenance=Provenance.unknown()
                        ),
                        right=TupleAccessExpression(
                            tuple_expression=IdentifierExpression(
                                identifier=t, provenance=Provenance.unknown()
                            ),
                            element_index=IntLiteral(
                                value=1, provenance=Provenance.unknown()
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

    run_validator(TupleAccessValidator(symbol_table), program_ast)


def test_fails_with_element_index_equal_to_arity(
    int32_float_pair_type: TupleType,
):
    """Test failure when the tuple access element index equals the arity."""
    main = Identifier("main")
    t, u = Identifier("t"), Identifier("u")
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(),
                body=(
                    DeclarationStatement(
                        variable_name=t,
                        variable_type=QualifiedType(
                            base_type=int32_float_pair_type,
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    DeclarationStatement(
                        variable_name=u,
                        variable_type=QualifiedType(
                            base_type=int32_float_pair_type,
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    ExpressionStatement(
                        left=IdentifierExpression(
                            identifier=u, provenance=Provenance.unknown()
                        ),
                        right=TupleAccessExpression(
                            tuple_expression=IdentifierExpression(
                                identifier=t, provenance=Provenance.unknown()
                            ),
                            element_index=IntLiteral(
                                value=2, provenance=Provenance.unknown()
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
        run_validator(TupleAccessValidator(symbol_table), program_ast)


def test_fails_with_negative_element_index(int32_float_pair_type: TupleType):
    """Test failure when the tuple access element index is negative."""
    main = Identifier("main")
    t, u = Identifier("t"), Identifier("u")
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(),
                body=(
                    DeclarationStatement(
                        variable_name=t,
                        variable_type=QualifiedType(
                            base_type=int32_float_pair_type,
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    DeclarationStatement(
                        variable_name=u,
                        variable_type=QualifiedType(
                            base_type=int32_float_pair_type,
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    ExpressionStatement(
                        left=IdentifierExpression(
                            identifier=u, provenance=Provenance.unknown()
                        ),
                        right=TupleAccessExpression(
                            tuple_expression=IdentifierExpression(
                                identifier=t, provenance=Provenance.unknown()
                            ),
                            element_index=IntLiteral(
                                value=-1, provenance=Provenance.unknown()
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
        run_validator(TupleAccessValidator(symbol_table), program_ast)


def test_valid_tuple_access_on_tuple_literal(int32: NumericalType):
    """Test a tuple access with an in-range element index on a tuple literal."""
    main = Identifier("main")
    u = Identifier("u")
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(),
                body=(
                    DeclarationStatement(
                        variable_name=u,
                        variable_type=QualifiedType(
                            base_type=int32,
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    ExpressionStatement(
                        left=IdentifierExpression(
                            identifier=u, provenance=Provenance.unknown()
                        ),
                        right=TupleAccessExpression(
                            tuple_expression=TupleExpression(
                                expressions=(
                                    IntLiteral(
                                        value=1, provenance=Provenance.unknown()
                                    ),
                                    IntLiteral(
                                        value=2, provenance=Provenance.unknown()
                                    ),
                                    IntLiteral(
                                        value=3, provenance=Provenance.unknown()
                                    ),
                                ),
                                provenance=Provenance.unknown(),
                            ),
                            element_index=IntLiteral(
                                value=2, provenance=Provenance.unknown()
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

    run_validator(TupleAccessValidator(symbol_table), program_ast)


def test_fails_with_tuple_literal_access_out_of_range(int32: NumericalType):
    """Test failure when a tuple literal is accessed beyond its arity."""
    main = Identifier("main")
    u = Identifier("u")
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(),
                body=(
                    DeclarationStatement(
                        variable_name=u,
                        variable_type=QualifiedType(
                            base_type=int32,
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    ExpressionStatement(
                        left=IdentifierExpression(
                            identifier=u, provenance=Provenance.unknown()
                        ),
                        right=TupleAccessExpression(
                            tuple_expression=TupleExpression(
                                expressions=(
                                    IntLiteral(
                                        value=1, provenance=Provenance.unknown()
                                    ),
                                    IntLiteral(
                                        value=2, provenance=Provenance.unknown()
                                    ),
                                ),
                                provenance=Provenance.unknown(),
                            ),
                            element_index=IntLiteral(
                                value=5, provenance=Provenance.unknown()
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
        run_validator(TupleAccessValidator(symbol_table), program_ast)


def test_valid_nested_tuple_access(int32: NumericalType):
    """Test nested tuple access where each element index is in range."""
    main = Identifier("main")
    t, u = Identifier("t"), Identifier("u")
    inner_pair_type = TupleType([int32, int32])
    outer_tuple_type = TupleType([int32, inner_pair_type])
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(),
                body=(
                    DeclarationStatement(
                        variable_name=t,
                        variable_type=QualifiedType(
                            base_type=outer_tuple_type,
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    DeclarationStatement(
                        variable_name=u,
                        variable_type=QualifiedType(
                            base_type=inner_pair_type,
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    ExpressionStatement(
                        left=IdentifierExpression(
                            identifier=u, provenance=Provenance.unknown()
                        ),
                        right=TupleAccessExpression(
                            tuple_expression=TupleAccessExpression(
                                tuple_expression=IdentifierExpression(
                                    identifier=t, provenance=Provenance.unknown()
                                ),
                                element_index=IntLiteral(
                                    value=1, provenance=Provenance.unknown()
                                ),
                                provenance=Provenance.unknown(),
                            ),
                            element_index=IntLiteral(
                                value=1, provenance=Provenance.unknown()
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

    run_validator(TupleAccessValidator(symbol_table), program_ast)


def test_fails_with_nested_tuple_access_out_of_range(int32: NumericalType):
    """Test failure when a nested tuple access element index is out of range."""
    main = Identifier("main")
    t, u = Identifier("t"), Identifier("u")
    inner_pair_type = TupleType([int32, int32])
    outer_tuple_type = TupleType([int32, inner_pair_type])
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(),
                body=(
                    DeclarationStatement(
                        variable_name=t,
                        variable_type=QualifiedType(
                            base_type=outer_tuple_type,
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    DeclarationStatement(
                        variable_name=u,
                        variable_type=QualifiedType(
                            base_type=inner_pair_type,
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    ExpressionStatement(
                        left=IdentifierExpression(
                            identifier=u, provenance=Provenance.unknown()
                        ),
                        right=TupleAccessExpression(
                            tuple_expression=TupleAccessExpression(
                                tuple_expression=IdentifierExpression(
                                    identifier=t, provenance=Provenance.unknown()
                                ),
                                element_index=IntLiteral(
                                    value=1, provenance=Provenance.unknown()
                                ),
                                provenance=Provenance.unknown(),
                            ),
                            element_index=IntLiteral(
                                value=5, provenance=Provenance.unknown()
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
        run_validator(TupleAccessValidator(symbol_table), program_ast)


def test_tuple_access_on_non_tuple_identifier_is_silent(int32: NumericalType):
    """Test that a tuple access on a non-tuple identifier is silently ignored.

    Type mismatches between a tuple access and its base expression are
    surfaced by the type checker, not by this validator.

    """
    main = Identifier("main")
    s, u = Identifier("s"), Identifier("u")
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(),
                body=(
                    DeclarationStatement(
                        variable_name=s,
                        variable_type=QualifiedType(
                            base_type=int32,
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    DeclarationStatement(
                        variable_name=u,
                        variable_type=QualifiedType(
                            base_type=int32,
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    ExpressionStatement(
                        left=IdentifierExpression(
                            identifier=u, provenance=Provenance.unknown()
                        ),
                        right=TupleAccessExpression(
                            tuple_expression=IdentifierExpression(
                                identifier=s, provenance=Provenance.unknown()
                            ),
                            element_index=IntLiteral(
                                value=0, provenance=Provenance.unknown()
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

    run_validator(TupleAccessValidator(symbol_table), program_ast)
