"""Tests the recursion validator AST pass."""

import pytest
from fhy_core import (
    Identifier,
    NumericalType,
    TypeQualifier,
    ValidationFailedError,
)

from fhy_lang.ast import (
    Argument,
    ExpressionStatement,
    FunctionExpression,
    IdentifierExpression,
    Module,
    Operation,
    Procedure,
    QualifiedType,
    ReturnStatement,
    Statement,
)
from fhy_lang.ast.passes import RecursionValidator

from .utils import run_validator


def _make_scalar_argument(
    name: Identifier, qualifier: TypeQualifier, base_type: NumericalType
) -> Argument:
    return Argument(
        name=name,
        qualified_type=QualifiedType(
            base_type=base_type,
            type_qualifier=qualifier,
        ),
    )


def _make_procedure(name: Identifier, body: tuple[Statement, ...]) -> Procedure:
    return Procedure(
        name=name,
        args=(),
        body=body,
    )


def _make_bare_call(callee: Identifier) -> ExpressionStatement:
    return ExpressionStatement(
        left=None,
        right=FunctionExpression(
            function=IdentifierExpression(identifier=callee),
            template_types=(),
            indices=(),
            args=(),
        ),
    )


def test_empty_module(empty_module_ast):
    """Test validation of an empty module."""
    run_validator(RecursionValidator(), empty_module_ast)


def test_valid_non_recursive_procedure():
    """Test a procedure that does not call any other function."""
    main = Identifier("main")
    program_ast = Module(
        statements=(_make_procedure(main, body=()),),
    )

    run_validator(RecursionValidator(), program_ast)


def test_valid_chain_of_calls_without_cycle():
    """Test a chain ``a -> b -> c`` of procedure calls with no cycle."""
    a, b, c = Identifier("a"), Identifier("b"), Identifier("c")
    program_ast = Module(
        statements=(
            _make_procedure(a, body=(_make_bare_call(b),)),
            _make_procedure(b, body=(_make_bare_call(c),)),
            _make_procedure(c, body=()),
        ),
    )

    run_validator(RecursionValidator(), program_ast)


def test_fails_with_direct_self_recursion():
    """Test failure when a procedure directly calls itself."""
    main = Identifier("main")
    program_ast = Module(
        statements=(_make_procedure(main, body=(_make_bare_call(main),)),),
    )

    with pytest.raises(ValidationFailedError, match="semantic error"):
        run_validator(RecursionValidator(), program_ast)


def test_fails_with_mutual_recursion_between_two_procedures():
    """Test failure when two procedures call each other."""
    a, b = Identifier("a"), Identifier("b")
    program_ast = Module(
        statements=(
            _make_procedure(a, body=(_make_bare_call(b),)),
            _make_procedure(b, body=(_make_bare_call(a),)),
        ),
    )

    with pytest.raises(ValidationFailedError, match="semantic error"):
        run_validator(RecursionValidator(), program_ast)


def test_fails_with_mutual_recursion_through_three_procedures():
    """Test failure when three procedures form a ``a -> b -> c -> a`` cycle."""
    a, b, c = Identifier("a"), Identifier("b"), Identifier("c")
    program_ast = Module(
        statements=(
            _make_procedure(a, body=(_make_bare_call(b),)),
            _make_procedure(b, body=(_make_bare_call(c),)),
            _make_procedure(c, body=(_make_bare_call(a),)),
        ),
    )

    with pytest.raises(ValidationFailedError, match="semantic error"):
        run_validator(RecursionValidator(), program_ast)


def test_fails_with_self_recursive_operation(int32: NumericalType):
    """Test failure when an operation directly calls itself."""
    op = Identifier("op")
    x = Identifier("x")
    program_ast = Module(
        statements=(
            Operation(
                name=op,
                args=(_make_scalar_argument(x, TypeQualifier.INPUT, int32),),
                return_type=QualifiedType(
                    base_type=int32,
                    type_qualifier=TypeQualifier.OUTPUT,
                ),
                body=(
                    ReturnStatement(
                        expression=FunctionExpression(
                            function=IdentifierExpression(identifier=op),
                            template_types=(),
                            indices=(),
                            args=(IdentifierExpression(identifier=x),),
                        ),
                    ),
                ),
            ),
        ),
    )

    with pytest.raises(ValidationFailedError, match="semantic error"):
        run_validator(RecursionValidator(), program_ast)


def test_call_to_unknown_identifier_is_not_recursion():
    """Test that calls to identifiers not declared as functions are ignored.

    Unresolved call targets (builtins, reductions, undefined names) cannot
    close a recursion cycle within the module, so the validator must not
    flag them. Other validators -- the call-site validator in particular --
    own the diagnostic when an unknown identifier is called.

    """
    main = Identifier("main")
    unknown = Identifier("unknown")
    program_ast = Module(
        statements=(_make_procedure(main, body=(_make_bare_call(unknown),)),),
    )

    run_validator(RecursionValidator(), program_ast)
