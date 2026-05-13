"""Tests the expression statement LHS validator AST pass."""

import pytest
from fhy_core import (
    DiagnosticLevel,
    Identifier,
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
    TupleAccessExpression,
)
from fhy_lang.ast.passes import (
    ExpressionStatementLHSValidator,
)

from .utils import run_validator


def test_empty_module(empty_module_ast):
    """Test validation of an empty module."""
    run_validator(ExpressionStatementLHSValidator(), empty_module_ast)


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
                ),
            ),
        ),
    )

    run_validator(ExpressionStatementLHSValidator(), program_ast)


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

    run_validator(ExpressionStatementLHSValidator(), program_ast)


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
                        ),
                    ),
                ),
                body=(
                    ExpressionStatement(
                        left=None,
                        right=IdentifierExpression(identifier=a),
                    ),
                ),
            ),
        ),
    )

    run_validator(ExpressionStatementLHSValidator(), program_ast)


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
                        ),
                    ),
                    ExpressionStatement(
                        left=BinaryExpression(
                            operation=BinaryOperation.ADDITION,
                            left=IdentifierExpression(identifier=t),
                            right=IntLiteral(value=1),
                        ),
                        right=IdentifierExpression(identifier=t),
                    ),
                ),
            ),
        ),
    )

    with pytest.raises(
        ValidationFailedError,
        match="structural error",
    ):
        run_validator(ExpressionStatementLHSValidator(), program_ast)


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
                        ),
                    ),
                    ExpressionStatement(
                        left=IntLiteral(value=0),
                        right=IdentifierExpression(identifier=t),
                    ),
                ),
            ),
        ),
    )

    with pytest.raises(
        ValidationFailedError,
        match="structural error",
    ):
        run_validator(ExpressionStatementLHSValidator(), program_ast)


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
                        ),
                    ),
                    ExpressionStatement(
                        left=ArrayAccessExpression(
                            array_expression=BinaryExpression(
                                operation=BinaryOperation.ADDITION,
                                left=IdentifierExpression(
                                    identifier=t,
                                ),
                                right=IntLiteral(value=1),
                            ),
                            indices=(IntLiteral(value=0),),
                        ),
                        right=IdentifierExpression(identifier=t),
                    ),
                ),
            ),
        ),
    )

    with pytest.raises(
        ValidationFailedError,
        match="structural error",
    ):
        run_validator(ExpressionStatementLHSValidator(), program_ast)


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
                        ),
                    ),
                    ExpressionStatement(
                        left=ArrayAccessExpression(
                            array_expression=ArrayAccessExpression(
                                array_expression=IdentifierExpression(
                                    identifier=t,
                                ),
                                indices=(
                                    IntLiteral(
                                        value=0,
                                    ),
                                ),
                            ),
                            indices=(IntLiteral(value=1),),
                        ),
                        right=IdentifierExpression(identifier=t),
                    ),
                ),
            ),
        ),
    )

    with pytest.raises(
        ValidationFailedError,
        match="structural error",
    ):
        run_validator(ExpressionStatementLHSValidator(), program_ast)


def test_warns_on_bare_value_expression_statement(int32: NumericalType):
    """Test that an expression statement with no LHS and a non-call RHS warns."""
    main = Identifier("main")
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(),
                body=(
                    ExpressionStatement(
                        left=None,
                        right=IntLiteral(value=1),
                    ),
                ),
            ),
        ),
    )
    validator = ExpressionStatementLHSValidator()
    result = validator.execute(program_ast)
    warnings = [d for d in result.diagnostics if d.level == DiagnosticLevel.WARNING]
    assert len(warnings) == 1
    assert "IntLiteral" in warnings[0].message_text


def test_does_not_warn_on_function_call_statement(int32: NumericalType):
    """Test that a bare function-call expression statement does not warn."""
    main = Identifier("main")
    other = Identifier("other")
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(),
                body=(
                    ExpressionStatement(
                        left=None,
                        right=FunctionExpression(
                            function=IdentifierExpression(
                                identifier=other,
                            ),
                            args=(),
                        ),
                    ),
                ),
            ),
        ),
    )
    validator = ExpressionStatementLHSValidator()
    result = validator.execute(program_ast)
    warnings = [d for d in result.diagnostics if d.level == DiagnosticLevel.WARNING]
    assert warnings == []


def test_fails_with_function_call_lhs(int32: NumericalType):
    """Test failure when the LHS is a function-call expression."""
    main = Identifier("main")
    f = Identifier("f")
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(),
                body=(
                    ExpressionStatement(
                        left=FunctionExpression(
                            function=IdentifierExpression(identifier=f),
                            args=(),
                        ),
                        right=IntLiteral(value=0),
                    ),
                ),
            ),
        ),
    )

    with pytest.raises(
        ValidationFailedError,
        match="structural error",
    ):
        run_validator(ExpressionStatementLHSValidator(), program_ast)


def test_fails_with_tuple_access_lhs(int32: NumericalType):
    """Test failure when the LHS is a tuple-access expression."""
    main = Identifier("main")
    t = Identifier("t")
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(),
                body=(
                    ExpressionStatement(
                        left=TupleAccessExpression(
                            tuple_expression=IdentifierExpression(identifier=t),
                            element_index=IntLiteral(value=0),
                        ),
                        right=IntLiteral(value=0),
                    ),
                ),
            ),
        ),
    )

    with pytest.raises(
        ValidationFailedError,
        match="structural error",
    ):
        run_validator(ExpressionStatementLHSValidator(), program_ast)


def test_fails_with_chained_array_access_lhs(int32: NumericalType):
    """Test failure when the LHS is a chained array access ``a[i][j]``."""
    main = Identifier("main")
    a, i, j = Identifier("a"), Identifier("i"), Identifier("j")
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(),
                body=(
                    ExpressionStatement(
                        left=ArrayAccessExpression(
                            array_expression=ArrayAccessExpression(
                                array_expression=IdentifierExpression(identifier=a),
                                indices=(IdentifierExpression(identifier=i),),
                            ),
                            indices=(IdentifierExpression(identifier=j),),
                        ),
                        right=IntLiteral(value=0),
                    ),
                ),
            ),
        ),
    )

    with pytest.raises(
        ValidationFailedError,
        match="structural error",
    ):
        run_validator(ExpressionStatementLHSValidator(), program_ast)


def test_warns_on_bare_binary_expression_statement(int32: NumericalType):
    """Test that a bare binary expression statement emits a warning diagnostic."""
    main = Identifier("main")
    x, y = Identifier("x"), Identifier("y")
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(),
                body=(
                    ExpressionStatement(
                        left=None,
                        right=BinaryExpression(
                            operation=BinaryOperation.ADDITION,
                            left=IdentifierExpression(identifier=x),
                            right=IdentifierExpression(identifier=y),
                        ),
                    ),
                ),
            ),
        ),
    )
    validator = ExpressionStatementLHSValidator()
    result = validator.execute(program_ast)
    warnings = [d for d in result.diagnostics if d.level == DiagnosticLevel.WARNING]
    assert len(warnings) == 1
    assert "BinaryExpression" in warnings[0].message_text
