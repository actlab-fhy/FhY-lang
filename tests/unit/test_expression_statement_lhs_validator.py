"""Tests the expression statement LHS validator AST pass."""

import pytest
from fhy.lang.ast import (
    Argument,
    ArrayAccessExpression,
    BinaryExpression,
    BinaryOperation,
    DeclarationStatement,
    ExpressionStatement,
    FhYStructuralError,
    IdentifierExpression,
    IntLiteral,
    Module,
    Procedure,
    QualifiedType,
)
from fhy.lang.ast.passes import (
    validate_expression_statement_lhs,
)
from fhy_core import (
    Identifier,
    NumericalType,
    PassExecutionError,
    Provenance,
    TypeQualifier,
)


def test_empty_module(empty_module_ast):
    """Test validation of an empty module."""
    validate_expression_statement_lhs(empty_module_ast)


def test_valid_identifier_lhs(int32: NumericalType):
    """Test a valid identifier expression on the LHS."""
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
                        left=IdentifierExpression(
                            identifier=t, provenance=Provenance.unknown()
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

    validate_expression_statement_lhs(program_ast)


def test_valid_array_access_lhs(int32: NumericalType):
    """Test a valid array access expression on the LHS."""
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

    validate_expression_statement_lhs(program_ast)


def test_valid_no_lhs(int32: NumericalType):
    """Test an expression statement without a LHS (e.g. a procedure call)."""
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
                        left=None,
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

    validate_expression_statement_lhs(program_ast)


def test_fails_with_binary_expression_lhs(int32: NumericalType):
    """Test failure when the LHS is a binary expression."""
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
                            base_type=int32,
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    ExpressionStatement(
                        left=BinaryExpression(
                            operation=BinaryOperation.ADDITION,
                            left=IdentifierExpression(
                                identifier=t, provenance=Provenance.unknown()
                            ),
                            right=IntLiteral(value=1, provenance=Provenance.unknown()),
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

    with pytest.raises(
        PassExecutionError,
        match=FhYStructuralError.__name__,
    ):
        validate_expression_statement_lhs(program_ast)


def test_fails_with_literal_lhs(int32: NumericalType):
    """Test failure when the LHS is a literal."""
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
                            base_type=int32,
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    ExpressionStatement(
                        left=IntLiteral(value=0, provenance=Provenance.unknown()),
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

    with pytest.raises(
        PassExecutionError,
        match=FhYStructuralError.__name__,
    ):
        validate_expression_statement_lhs(program_ast)


def test_fails_with_array_access_on_non_identifier(int32: NumericalType):
    """Test failure when the array expression of an LHS array access is not
    an identifier expression.
    """
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
                            base_type=int32,
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    ExpressionStatement(
                        left=ArrayAccessExpression(
                            array_expression=BinaryExpression(
                                operation=BinaryOperation.ADDITION,
                                left=IdentifierExpression(
                                    identifier=t,
                                    provenance=Provenance.unknown(),
                                ),
                                right=IntLiteral(
                                    value=1, provenance=Provenance.unknown()
                                ),
                                provenance=Provenance.unknown(),
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

    with pytest.raises(
        PassExecutionError,
        match=FhYStructuralError.__name__,
    ):
        validate_expression_statement_lhs(program_ast)


def test_fails_with_nested_array_access_lhs(int32: NumericalType):
    """Test failure when the LHS is a nested array access expression."""
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
                            base_type=int32,
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    ExpressionStatement(
                        left=ArrayAccessExpression(
                            array_expression=ArrayAccessExpression(
                                array_expression=IdentifierExpression(
                                    identifier=t,
                                    provenance=Provenance.unknown(),
                                ),
                                indices=(
                                    IntLiteral(
                                        value=0,
                                        provenance=Provenance.unknown(),
                                    ),
                                ),
                                provenance=Provenance.unknown(),
                            ),
                            indices=(
                                IntLiteral(value=1, provenance=Provenance.unknown()),
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

    with pytest.raises(
        PassExecutionError,
        match=FhYStructuralError.__name__,
    ):
        validate_expression_statement_lhs(program_ast)
