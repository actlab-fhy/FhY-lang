"""Tests for the FhY unreachable code elimination pass."""

from fhy_core import (
    Identifier,
    Provenance,
    TypeQualifier,
)

from fhy_lang.ast import (
    ExpressionStatement,
    IdentifierExpression,
    Module,
    Operation,
    QualifiedType,
    ReturnStatement,
    SelectionStatement,
    Statement,
)
from fhy_lang.ast.passes import UnreachableCodeEliminationPass

from .utils import (
    make_argument,
    make_identifier_assignment,
    make_module_with_statement,
    make_procedure,
)


def _make_identifier_expr(name: Identifier) -> IdentifierExpression:
    return IdentifierExpression(identifier=name, provenance=Provenance.unknown())


def _make_operation(
    name: Identifier,
    args: tuple,
    body: tuple[Statement, ...],
    return_type_base,
) -> Operation:
    return Operation(
        name=name,
        args=args,
        body=body,
        return_type=QualifiedType(
            base_type=return_type_base,
            type_qualifier=TypeQualifier.OUTPUT,
            provenance=Provenance.unknown(),
        ),
        provenance=Provenance.unknown(),
    )


def _get_first_function_body(module: Module) -> tuple[Statement, ...]:
    return module.statements[0].body  # type: ignore[return-value,union-attr]


def test_removes_statement_following_return(int32):
    """Test that statements after a return are dropped."""
    a, r = Identifier("a"), Identifier("r")
    unreachable_assignment_ast = make_identifier_assignment(r, a)
    operation_ast = _make_operation(
        name=Identifier("f"),
        args=(make_argument(a, TypeQualifier.INPUT, int32),),
        body=(
            make_identifier_assignment(r, a),
            ReturnStatement(
                expression=_make_identifier_expr(r), provenance=Provenance.unknown()
            ),
            unreachable_assignment_ast,
        ),
        return_type_base=int32,
    )
    program_ast = make_module_with_statement(operation_ast)

    optimized = UnreachableCodeEliminationPass()(program_ast)

    body = _get_first_function_body(optimized)
    assert len(body) == 2
    assert isinstance(body[0], ExpressionStatement)
    assert isinstance(body[1], ReturnStatement)


def test_removes_multiple_trailing_statements(int32):
    """Test that every statement after a return is dropped."""
    a, r = Identifier("a"), Identifier("r")
    operation_ast = _make_operation(
        name=Identifier("f"),
        args=(make_argument(a, TypeQualifier.INPUT, int32),),
        body=(
            ReturnStatement(
                expression=_make_identifier_expr(a), provenance=Provenance.unknown()
            ),
            make_identifier_assignment(r, a),
            make_identifier_assignment(r, a),
            make_identifier_assignment(r, a),
        ),
        return_type_base=int32,
    )
    program_ast = make_module_with_statement(operation_ast)

    optimized = UnreachableCodeEliminationPass()(program_ast)

    body = _get_first_function_body(optimized)
    assert len(body) == 1
    assert isinstance(body[0], ReturnStatement)


def test_removes_unreachable_tail_only_inside_returning_branch(int32):
    """Test that a trailing stmt inside a branch that returns is dropped,
    but statements after the selection are preserved when the other
    branch falls through."""
    a, c, r = Identifier("a"), Identifier("c"), Identifier("r")
    true_unreachable = make_identifier_assignment(r, a)
    selection = SelectionStatement(
        condition=_make_identifier_expr(c),
        true_body=(
            ReturnStatement(
                expression=_make_identifier_expr(a), provenance=Provenance.unknown()
            ),
            true_unreachable,  # unreachable — after the return
        ),
        false_body=(make_identifier_assignment(r, a),),
        provenance=Provenance.unknown(),
    )
    trailing = ReturnStatement(
        expression=_make_identifier_expr(r), provenance=Provenance.unknown()
    )
    operation_ast = _make_operation(
        name=Identifier("f"),
        args=(
            make_argument(a, TypeQualifier.INPUT, int32),
            make_argument(c, TypeQualifier.INPUT, int32),
        ),
        body=(selection, trailing),
        return_type_base=int32,
    )
    program_ast = make_module_with_statement(operation_ast)

    optimized = UnreachableCodeEliminationPass()(program_ast)

    body = _get_first_function_body(optimized)
    # The selection and the outer trailing return both survive.
    assert len(body) == 2
    assert isinstance(body[0], SelectionStatement)
    assert isinstance(body[1], ReturnStatement)
    # Inside the (transformed) selection: true branch has only the return.
    new_selection = body[0]
    assert len(new_selection.true_body) == 1
    assert isinstance(new_selection.true_body[0], ReturnStatement)
    assert len(new_selection.false_body) == 1


def test_removes_statements_after_selection_when_both_branches_return(int32):
    """Test that statements after a both-branch-returning selection are dropped."""
    a, c = Identifier("a"), Identifier("c")
    selection = SelectionStatement(
        condition=_make_identifier_expr(c),
        true_body=(
            ReturnStatement(
                expression=_make_identifier_expr(a), provenance=Provenance.unknown()
            ),
        ),
        false_body=(
            ReturnStatement(
                expression=_make_identifier_expr(a), provenance=Provenance.unknown()
            ),
        ),
        provenance=Provenance.unknown(),
    )
    operation_ast = _make_operation(
        name=Identifier("f"),
        args=(
            make_argument(a, TypeQualifier.INPUT, int32),
            make_argument(c, TypeQualifier.INPUT, int32),
        ),
        body=(
            selection,
            # unreachable — both branches of the selection return
            make_identifier_assignment(Identifier("z"), a),
        ),
        return_type_base=int32,
    )
    program_ast = make_module_with_statement(operation_ast)

    optimized = UnreachableCodeEliminationPass()(program_ast)

    body = _get_first_function_body(optimized)
    assert len(body) == 1
    assert isinstance(body[0], SelectionStatement)


def test_preserves_fully_reachable_body(int32):
    """Test that a body with no unreachable code is left alone."""
    a, b = Identifier("a"), Identifier("b")
    procedure_ast = make_procedure(
        name=Identifier("main"),
        args=(
            make_argument(a, TypeQualifier.INPUT, int32),
            make_argument(b, TypeQualifier.OUTPUT, int32),
        ),
        body=(
            make_identifier_assignment(b, a),
            make_identifier_assignment(b, a),
        ),
    )
    program_ast = make_module_with_statement(procedure_ast)

    optimized = UnreachableCodeEliminationPass()(program_ast)

    body = _get_first_function_body(optimized)
    assert len(body) == 2


def test_did_change_reports_removal_activity(int32):
    """Test it reports `did_change=True` when it removes, `False` otherwise."""
    a = Identifier("a")
    unreachable_body = (
        ReturnStatement(
            expression=_make_identifier_expr(a), provenance=Provenance.unknown()
        ),
        make_identifier_assignment(a, a),
    )
    with_unreachable = make_module_with_statement(
        _make_operation(
            name=Identifier("f"),
            args=(make_argument(a, TypeQualifier.INPUT, int32),),
            body=unreachable_body,
            return_type_base=int32,
        )
    )
    fully_reachable_body = (
        ReturnStatement(
            expression=_make_identifier_expr(a), provenance=Provenance.unknown()
        ),
    )
    fully_reachable = make_module_with_statement(
        _make_operation(
            name=Identifier("f"),
            args=(make_argument(a, TypeQualifier.INPUT, int32),),
            body=fully_reachable_body,
            return_type_base=int32,
        )
    )

    removing_pass = UnreachableCodeEliminationPass()
    assert removing_pass.execute(with_unreachable).changed is True

    idle_pass = UnreachableCodeEliminationPass()
    assert idle_pass.execute(fully_reachable).changed is False
