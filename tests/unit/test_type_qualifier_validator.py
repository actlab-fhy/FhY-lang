"""Tests the type qualifier validator AST pass."""

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
    IdentifierExpression,
    IntLiteral,
    Module,
    Operation,
    Procedure,
    QualifiedType,
)
from fhy_lang.ast.passes import TypeQualifierValidator, build_symbol_table

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
                        ),
                    ),
                    Argument(
                        name=b,
                        qualified_type=QualifiedType(
                            base_type=int32,
                            type_qualifier=TypeQualifier.OUTPUT,
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
                        right=IdentifierExpression(identifier=a),
                    ),
                    ExpressionStatement(
                        left=IdentifierExpression(identifier=b),
                        right=IdentifierExpression(identifier=t),
                    ),
                ),
            ),
        ),
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
                        ),
                    ),
                ),
                body=(),
                return_type=QualifiedType(
                    base_type=int32,
                    type_qualifier=TypeQualifier.OUTPUT,
                ),
            ),
        ),
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
                        ),
                    ),
                ),
                body=(),
            ),
        ),
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
                        ),
                    ),
                ),
                body=(),
            ),
        ),
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
                        ),
                    ),
                ),
            ),
        ),
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
                        ),
                    ),
                ),
            ),
        ),
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
                ),
            ),
        ),
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
                ),
            ),
        ),
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
                        ),
                    ),
                ),
                body=(
                    ExpressionStatement(
                        left=IdentifierExpression(identifier=a),
                        right=IdentifierExpression(identifier=a),
                    ),
                ),
            ),
        ),
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
                        left=IdentifierExpression(identifier=a),
                        right=IdentifierExpression(identifier=t),
                    ),
                ),
            ),
        ),
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
                        left=ArrayAccessExpression(
                            array_expression=IdentifierExpression(identifier=a),
                            indices=(IntLiteral(value=0),),
                        ),
                        right=IdentifierExpression(identifier=t),
                    ),
                ),
            ),
        ),
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
                        left=ArrayAccessExpression(
                            array_expression=IdentifierExpression(identifier=a),
                            indices=(IntLiteral(value=0),),
                        ),
                        right=IdentifierExpression(identifier=t),
                    ),
                ),
            ),
        ),
    )
    symbol_table = build_symbol_table(program_ast)

    run_validator(TypeQualifierValidator(symbol_table), program_ast)


def test_fails_on_indexed_assignment_to_input_array():
    """Test failure when an indexed assignment targets an ``input`` array."""
    main = Identifier("main")
    n = Identifier("N")
    a, i = Identifier("a"), Identifier("i")
    int32_vector_n = NumericalType(
        PrimitiveDataType(CoreDataType.INT32),
        shape=(CoreIdentifierExpression(n),),
    )
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(
                    Argument(
                        name=a,
                        qualified_type=QualifiedType(
                            base_type=int32_vector_n,
                            type_qualifier=TypeQualifier.INPUT,
                        ),
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
                        ),
                    ),
                    ExpressionStatement(
                        left=ArrayAccessExpression(
                            array_expression=IdentifierExpression(identifier=a),
                            indices=(IdentifierExpression(identifier=i),),
                        ),
                        right=IntLiteral(value=0),
                    ),
                ),
            ),
        ),
    )
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(ValidationFailedError, match="type qualifier error"):
        run_validator(TypeQualifierValidator(symbol_table), program_ast)


# === HARDENING TESTS ===


def test_passes_on_param_decl_with_literal_initializer():
    """Test ``param int32 y = 5`` is allowed (literal is constant)."""
    main = Identifier("main")
    y = Identifier("y")
    int32_scalar = NumericalType(PrimitiveDataType(CoreDataType.INT32))
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(),
                body=(
                    DeclarationStatement(
                        variable_name=y,
                        variable_type=QualifiedType(
                            base_type=int32_scalar,
                            type_qualifier=TypeQualifier.PARAM,
                        ),
                        expression=IntLiteral(value=5),
                    ),
                ),
            ),
        ),
    )
    symbol_table = build_symbol_table(program_ast)

    run_validator(TypeQualifierValidator(symbol_table), program_ast)


def test_passes_on_param_decl_with_param_referenced_initializer():
    """Test a ``param`` initialized from another ``param`` is allowed."""
    main = Identifier("main")
    k_param, y = Identifier("K"), Identifier("y")
    int32_scalar = NumericalType(PrimitiveDataType(CoreDataType.INT32))
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(
                    Argument(
                        name=k_param,
                        qualified_type=QualifiedType(
                            base_type=int32_scalar,
                            type_qualifier=TypeQualifier.PARAM,
                        ),
                    ),
                ),
                body=(
                    DeclarationStatement(
                        variable_name=y,
                        variable_type=QualifiedType(
                            base_type=int32_scalar,
                            type_qualifier=TypeQualifier.PARAM,
                        ),
                        expression=IdentifierExpression(identifier=k_param),
                    ),
                ),
            ),
        ),
    )
    symbol_table = build_symbol_table(program_ast)

    run_validator(TypeQualifierValidator(symbol_table), program_ast)


def test_fails_on_param_decl_with_non_constant_initializer():
    """Test ``param int32 y = x + 1`` (where ``x`` is a TEMP) is rejected
    ."""
    main = Identifier("main")
    x_temp, y_param = Identifier("x"), Identifier("y")
    int32_scalar = NumericalType(PrimitiveDataType(CoreDataType.INT32))
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(),
                body=(
                    DeclarationStatement(
                        variable_name=x_temp,
                        variable_type=QualifiedType(
                            base_type=int32_scalar,
                            type_qualifier=TypeQualifier.TEMP,
                        ),
                        expression=IntLiteral(value=5),
                    ),
                    DeclarationStatement(
                        variable_name=y_param,
                        variable_type=QualifiedType(
                            base_type=int32_scalar,
                            type_qualifier=TypeQualifier.PARAM,
                        ),
                        expression=BinaryExpression(
                            operation=BinaryOperation.ADDITION,
                            left=IdentifierExpression(identifier=x_temp),
                            right=IntLiteral(value=1),
                        ),
                    ),
                ),
            ),
        ),
    )
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(ValidationFailedError, match="compile-time-reducible"):
        run_validator(TypeQualifierValidator(symbol_table), program_ast)


def test_fails_on_param_decl_with_input_referenced_initializer():
    """Test ``param int32 y = a`` (where ``a`` is INPUT) is rejected."""
    main = Identifier("main")
    a_input, y_param = Identifier("a"), Identifier("y")
    int32_scalar = NumericalType(PrimitiveDataType(CoreDataType.INT32))
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(
                    Argument(
                        name=a_input,
                        qualified_type=QualifiedType(
                            base_type=int32_scalar,
                            type_qualifier=TypeQualifier.INPUT,
                        ),
                    ),
                ),
                body=(
                    DeclarationStatement(
                        variable_name=y_param,
                        variable_type=QualifiedType(
                            base_type=int32_scalar,
                            type_qualifier=TypeQualifier.PARAM,
                        ),
                        expression=IdentifierExpression(identifier=a_input),
                    ),
                ),
            ),
        ),
    )
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(ValidationFailedError, match="compile-time-reducible"):
        run_validator(TypeQualifierValidator(symbol_table), program_ast)
