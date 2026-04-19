"""Tests the call-site validator AST pass."""

from collections.abc import Sequence

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
    FunctionExpression,
    IdentifierExpression,
    IntLiteral,
    Module,
    Operation,
    Procedure,
    QualifiedType,
    Statement,
)
from fhy.lang.ast.passes import (
    build_symbol_table,
    validate_call_sites,
)
from fhy.lang.builtins import BUILTIN_REDUCTION_FUNCTION_IDENTIFIERS
from fhy_core import (
    Identifier,
    IndexType,
    LiteralExpression,
    NumericalType,
    PassExecutionError,
    Provenance,
    Type,
    TypeQualifier,
)
from fhy_core import (
    IdentifierExpression as CoreIdentifierExpression,
)


def _make_module_ast(statements: Sequence[Statement]) -> Module:
    return Module(
        statements=statements,
        provenance=Provenance.unknown(),
    )


def _make_procedure_ast(
    name: Identifier, args: Sequence[Argument], body: Sequence[Statement]
) -> Module:
    return Procedure(
        name=name,
        templates=(),
        args=args,
        body=body,
        provenance=Provenance.unknown(),
    )


def _make_operation_ast(
    name: Identifier,
    args: Sequence[Argument],
    body: Sequence[Statement],
    return_type: Type,
) -> Module:
    return Operation(
        name=name,
        templates=(),
        args=args,
        body=body,
        return_type=QualifiedType(
            base_type=return_type,
            type_qualifier=TypeQualifier.OUTPUT,
            provenance=Provenance.unknown(),
        ),
        provenance=Provenance.unknown(),
    )


def test_empty_module(empty_module_ast):
    """Test validation of an empty module."""
    symbol_table = build_symbol_table(empty_module_ast)

    validate_call_sites(empty_module_ast, symbol_table)


def test_valid_operation_call_with_lhs(int32: NumericalType):
    """Test that an operation can be called with a left-hand side."""
    main = Identifier("main")
    op = Identifier("op")
    a, t = Identifier("a"), Identifier("t")
    module_statements = (
        _make_operation_ast(
            op,
            (
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
            (),
            int32,
        ),
        _make_procedure_ast(
            main,
            (
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
            (
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
                            identifier=op,
                            provenance=Provenance.unknown(),
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
        ),
    )
    program_ast = _make_module_ast(module_statements)
    symbol_table = build_symbol_table(program_ast)

    validate_call_sites(program_ast, symbol_table)


def test_valid_procedure_call_without_lhs(int32: NumericalType):
    """Test that a procedure can be called without a left-hand side."""
    main = Identifier("main")
    other = Identifier("other")
    a = Identifier("a")

    module_statements = (
        _make_procedure_ast(
            other,
            (
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
            (),
        ),
        _make_procedure_ast(
            main,
            (
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
            (
                ExpressionStatement(
                    left=None,
                    right=FunctionExpression(
                        function=IdentifierExpression(
                            identifier=other,
                            provenance=Provenance.unknown(),
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
        ),
    )
    program_ast = _make_module_ast(module_statements)
    symbol_table = build_symbol_table(program_ast)

    validate_call_sites(program_ast, symbol_table)


def test_valid_reduction_call_with_indices(int32: NumericalType):
    """Test that a reduction builtin can be called with indices."""
    main = Identifier("main")
    sum_ = BUILTIN_REDUCTION_FUNCTION_IDENTIFIERS["sum"]
    a, t, k = Identifier("a"), Identifier("t"), Identifier("k")
    m = Identifier("m")
    module_statements = (
        _make_procedure_ast(
            main,
            (
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
            (
                DeclarationStatement(
                    variable_name=k,
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
        ),
    )
    program_ast = _make_module_ast(module_statements)
    symbol_table = build_symbol_table(program_ast)

    validate_call_sites(program_ast, symbol_table)


def test_fails_with_non_identifier_function_expression(int32: NumericalType):
    """Test failure when the function name is not an identifier expression."""
    main = Identifier("main")
    a = Identifier("a")
    module_statements = (
        _make_procedure_ast(
            main,
            (
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
            (
                ExpressionStatement(
                    left=None,
                    right=FunctionExpression(
                        function=BinaryExpression(
                            operation=BinaryOperation.ADDITION,
                            left=IdentifierExpression(
                                identifier=a,
                                provenance=Provenance.unknown(),
                            ),
                            right=IntLiteral(
                                value=1,
                                provenance=Provenance.unknown(),
                            ),
                            provenance=Provenance.unknown(),
                        ),
                        args=(),
                        provenance=Provenance.unknown(),
                    ),
                    provenance=Provenance.unknown(),
                ),
            ),
        ),
    )
    program_ast = _make_module_ast(module_statements)
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(PassExecutionError, match=FhYStructuralError.__name__):
        validate_call_sites(program_ast, symbol_table)


def test_fails_with_non_function_identifier(int32: NumericalType):
    """Test failure when the function name identifier is not a function."""
    main = Identifier("main")
    a = Identifier("a")
    module_statements = (
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
                    right=FunctionExpression(
                        function=IdentifierExpression(
                            identifier=a,
                            provenance=Provenance.unknown(),
                        ),
                        args=(),
                        provenance=Provenance.unknown(),
                    ),
                    provenance=Provenance.unknown(),
                ),
            ),
            provenance=Provenance.unknown(),
        ),
    )
    program_ast = _make_module_ast(module_statements)
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(PassExecutionError, match=FhYSemanticsError.__name__):
        validate_call_sites(program_ast, symbol_table)


def test_fails_with_non_reduction_call_having_indices(int32: NumericalType):
    """Test failure when a non-reduction function is called with indices."""
    main = Identifier("main")
    other = Identifier("other")
    a, k = Identifier("a"), Identifier("k")
    module_statements = (
        _make_procedure_ast(
            other,
            (
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
        ),
        _make_procedure_ast(
            main,
            (
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
            (
                DeclarationStatement(
                    variable_name=k,
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
                    right=FunctionExpression(
                        function=IdentifierExpression(
                            identifier=other,
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
        ),
    )
    program_ast = _make_module_ast(module_statements)
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(PassExecutionError, match=FhYStructuralError.__name__):
        validate_call_sites(program_ast, symbol_table)


def test_fails_with_procedure_call_having_lhs(int32: NumericalType):
    """Test failure when a procedure is called with a left-hand side."""
    main = Identifier("main")
    other = Identifier("other")
    a, t = Identifier("a"), Identifier("t")
    module_statements = (
        _make_procedure_ast(
            other,
            (
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
        ),
        _make_procedure_ast(
            main,
            (
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
            (
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
                            identifier=other,
                            provenance=Provenance.unknown(),
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
        ),
    )
    program_ast = _make_module_ast(module_statements)
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(PassExecutionError, match=FhYStructuralError.__name__):
        validate_call_sites(program_ast, symbol_table)


def test_fails_with_wrong_number_of_arguments(int32: NumericalType):
    """Test failure when a function is called with the wrong number of arguments."""
    main = Identifier("main")
    other = Identifier("other")
    a, b = Identifier("a"), Identifier("b")
    module_statements = (
        _make_procedure_ast(
            other,
            (
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
            (),
        ),
        _make_procedure_ast(
            main,
            (
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
            (
                ExpressionStatement(
                    left=None,
                    right=FunctionExpression(
                        function=IdentifierExpression(
                            identifier=other,
                            provenance=Provenance.unknown(),
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
        ),
    )
    program_ast = _make_module_ast(module_statements)
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(PassExecutionError, match=FhYStructuralError.__name__):
        validate_call_sites(program_ast, symbol_table)


def test_fails_with_procedure_called_in_value_position(int32: NumericalType):
    """Test failure when a procedure is called inside a value expression."""
    main = Identifier("main")
    other = Identifier("other")
    a, t = Identifier("a"), Identifier("t")
    program_ast = Module(
        statements=(
            Procedure(
                name=other,
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
                provenance=Provenance.unknown(),
            ),
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
                        right=BinaryExpression(
                            operation=BinaryOperation.ADDITION,
                            left=FunctionExpression(
                                function=IdentifierExpression(
                                    identifier=other,
                                    provenance=Provenance.unknown(),
                                ),
                                args=(
                                    IdentifierExpression(
                                        identifier=a,
                                        provenance=Provenance.unknown(),
                                    ),
                                ),
                                provenance=Provenance.unknown(),
                            ),
                            right=IntLiteral(value=1, provenance=Provenance.unknown()),
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
        validate_call_sites(program_ast, symbol_table)


def test_warns_on_operation_called_as_bare_statement(int32: NumericalType):
    """Test that calling an operation as a bare statement emits a warning."""
    from fhy.lang.ast.passes.call_site_validator import _CallSiteValidator
    from fhy_core import DiagnosticLevel

    main = Identifier("main")
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
                        right=FunctionExpression(
                            function=IdentifierExpression(
                                identifier=op,
                                provenance=Provenance.unknown(),
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
    validator = _CallSiteValidator(symbol_table)
    result = validator.execute(program_ast)
    warnings = [d for d in result.diagnostics if d.level == DiagnosticLevel.WARNING]
    assert len(warnings) == 1
    assert "op" in warnings[0].message_text
