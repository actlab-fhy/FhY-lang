"""Tests the type qualifier validator AST pass."""

import pytest
from fhy_core import (
    Identifier,
    NumericalType,
    Provenance,
    TypeQualifier,
    ValidationFailedError,
)
from fhy_lang.lang.ast import (
    Argument,
    ArrayAccessExpression,
    DeclarationStatement,
    ExpressionStatement,
    IdentifierExpression,
    IntLiteral,
    Module,
    Operation,
    Procedure,
    QualifiedType,
)
from fhy_lang.lang.ast.passes import TypeQualifierValidator, build_symbol_table

from .utils import run_validator


def test_empty_module(empty_module_ast):
    """Test validation of an empty module."""
    symbol_table = build_symbol_table(empty_module_ast)

    run_validator(TypeQualifierValidator(symbol_table), empty_module_ast)


def test_valid_procedure(int32: NumericalType):
    """Test a procedure with valid qualifiers on args and declarations."""
    main = Identifier("main")
    a, b, t = Identifier("a"), Identifier("b"), Identifier("t")
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
                            type_qualifier=TypeQualifier.OUTPUT,
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
                        right=IdentifierExpression(
                            identifier=a, provenance=Provenance.unknown()
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    ExpressionStatement(
                        left=IdentifierExpression(
                            identifier=b, provenance=Provenance.unknown()
                        ),
                        right=IdentifierExpression(
                            identifier=t, provenance=Provenance.unknown()
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

    run_validator(TypeQualifierValidator(symbol_table), program_ast)


def test_valid_operation_with_output_return_type(int32: NumericalType):
    """Test an operation with an OUTPUT return type."""
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
                            base_type=int32,
                            type_qualifier=TypeQualifier.INPUT,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                ),
                body=(),
                return_type=QualifiedType(
                    base_type=int32,
                    type_qualifier=TypeQualifier.OUTPUT,
                    provenance=Provenance.unknown(),
                ),
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )
    symbol_table = build_symbol_table(program_ast)

    run_validator(TypeQualifierValidator(symbol_table), program_ast)


def test_valid_param_argument(int32: NumericalType):
    """Test that PARAM is allowed on arguments."""
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
                            base_type=int32,
                            type_qualifier=TypeQualifier.PARAM,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                ),
                body=(),
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )
    symbol_table = build_symbol_table(program_ast)

    run_validator(TypeQualifierValidator(symbol_table), program_ast)


def test_fails_with_temp_argument(int32: NumericalType):
    """Test failure when an argument is qualified TEMP."""
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
                            base_type=int32,
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                ),
                body=(),
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(ValidationFailedError, match="type qualifier error"):
        run_validator(TypeQualifierValidator(symbol_table), program_ast)


def test_fails_with_input_declaration(int32: NumericalType):
    """Test failure when a declaration is qualified INPUT."""
    main = Identifier("main")
    a = Identifier("a")
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(),
                body=(
                    DeclarationStatement(
                        variable_name=a,
                        variable_type=QualifiedType(
                            base_type=int32,
                            type_qualifier=TypeQualifier.INPUT,
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

    with pytest.raises(ValidationFailedError, match="type qualifier error"):
        run_validator(TypeQualifierValidator(symbol_table), program_ast)


def test_fails_with_output_declaration(int32: NumericalType):
    """Test failure when a declaration is qualified OUTPUT."""
    main = Identifier("main")
    a = Identifier("a")
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(),
                body=(
                    DeclarationStatement(
                        variable_name=a,
                        variable_type=QualifiedType(
                            base_type=int32,
                            type_qualifier=TypeQualifier.OUTPUT,
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

    with pytest.raises(ValidationFailedError, match="type qualifier error"):
        run_validator(TypeQualifierValidator(symbol_table), program_ast)


def test_fails_with_input_return_type(int32: NumericalType):
    """Test failure when an operation return type is INPUT."""
    op = Identifier("op")
    program_ast = Module(
        statements=(
            Operation(
                name=op,
                args=(),
                body=(),
                return_type=QualifiedType(
                    base_type=int32,
                    type_qualifier=TypeQualifier.INPUT,
                    provenance=Provenance.unknown(),
                ),
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(ValidationFailedError, match="type qualifier error"):
        run_validator(TypeQualifierValidator(symbol_table), program_ast)


def test_fails_with_temp_return_type(int32: NumericalType):
    """Test failure when an operation return type is TEMP."""
    op = Identifier("op")
    program_ast = Module(
        statements=(
            Operation(
                name=op,
                args=(),
                body=(),
                return_type=QualifiedType(
                    base_type=int32,
                    type_qualifier=TypeQualifier.TEMP,
                    provenance=Provenance.unknown(),
                ),
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(ValidationFailedError, match="type qualifier error"):
        run_validator(TypeQualifierValidator(symbol_table), program_ast)


def test_fails_with_assignment_to_input(int32: NumericalType):
    """Test failure when assigning to an INPUT variable."""
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
                            base_type=int32,
                            type_qualifier=TypeQualifier.INPUT,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                ),
                body=(
                    ExpressionStatement(
                        left=IdentifierExpression(
                            identifier=a, provenance=Provenance.unknown()
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

    with pytest.raises(ValidationFailedError, match="type qualifier error"):
        run_validator(TypeQualifierValidator(symbol_table), program_ast)


def test_fails_with_assignment_to_param(int32: NumericalType):
    """Test failure when assigning to a PARAM variable."""
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
                            base_type=int32,
                            type_qualifier=TypeQualifier.PARAM,
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
                            identifier=a, provenance=Provenance.unknown()
                        ),
                        right=IdentifierExpression(
                            identifier=t, provenance=Provenance.unknown()
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

    with pytest.raises(ValidationFailedError, match="type qualifier error"):
        run_validator(TypeQualifierValidator(symbol_table), program_ast)


def test_fails_with_array_access_assignment_to_input(int32: NumericalType):
    """Test failure when assigning to an element of an INPUT variable."""
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
                        left=ArrayAccessExpression(
                            array_expression=IdentifierExpression(
                                identifier=a, provenance=Provenance.unknown()
                            ),
                            indices=(
                                IntLiteral(value=0, provenance=Provenance.unknown()),
                            ),
                            provenance=Provenance.unknown(),
                        ),
                        right=IdentifierExpression(
                            identifier=t, provenance=Provenance.unknown()
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

    with pytest.raises(ValidationFailedError, match="type qualifier error"):
        run_validator(TypeQualifierValidator(symbol_table), program_ast)


def test_valid_array_access_assignment_to_output(int32: NumericalType):
    """Test that assigning to an element of an OUTPUT variable is allowed."""
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
                            base_type=int32,
                            type_qualifier=TypeQualifier.OUTPUT,
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
                        left=ArrayAccessExpression(
                            array_expression=IdentifierExpression(
                                identifier=a, provenance=Provenance.unknown()
                            ),
                            indices=(
                                IntLiteral(value=0, provenance=Provenance.unknown()),
                            ),
                            provenance=Provenance.unknown(),
                        ),
                        right=IdentifierExpression(
                            identifier=t, provenance=Provenance.unknown()
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

    run_validator(TypeQualifierValidator(symbol_table), program_ast)
