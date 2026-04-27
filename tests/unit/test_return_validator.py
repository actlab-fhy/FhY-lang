"""Tests the return validator AST pass.

Covers both rules enforced by :class:`ReturnValidator`:

- Every control-flow path through an :class:`Operation` body must reach a
  :class:`ReturnStatement`.
- A :class:`Procedure` body must not contain any :class:`ReturnStatement`.

"""

import pytest
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

from fhy_lang.lang.ast import (
    Argument,
    DeclarationStatement,
    ForAllStatement,
    IdentifierExpression,
    IntLiteral,
    Module,
    Operation,
    Procedure,
    QualifiedType,
    ReturnStatement,
    SelectionStatement,
)
from fhy_lang.lang.ast.passes import ReturnValidator

from .utils import run_validator


def _qt(type_, qualifier: TypeQualifier) -> QualifiedType:
    return QualifiedType(
        base_type=type_,
        type_qualifier=qualifier,
        provenance=Provenance.unknown(),
    )


def _scalar_int32() -> NumericalType:
    return NumericalType(PrimitiveDataType(CoreDataType.INT32))


def _make_operation(name: Identifier, body: tuple) -> Operation:
    return Operation(
        name=name,
        args=(
            Argument(
                name=Identifier("x"),
                qualified_type=_qt(_scalar_int32(), TypeQualifier.INPUT),
                provenance=Provenance.unknown(),
            ),
        ),
        body=body,
        return_type=_qt(_scalar_int32(), TypeQualifier.OUTPUT),
        provenance=Provenance.unknown(),
    )


def _make_procedure(name: Identifier, body: tuple) -> Procedure:
    return Procedure(
        name=name,
        args=(
            Argument(
                name=Identifier("x"),
                qualified_type=_qt(_scalar_int32(), TypeQualifier.INPUT),
                provenance=Provenance.unknown(),
            ),
        ),
        body=body,
        provenance=Provenance.unknown(),
    )


def _return_literal() -> ReturnStatement:
    return ReturnStatement(
        expression=IntLiteral(value=0, provenance=Provenance.unknown()),
        provenance=Provenance.unknown(),
    )


# =============================================================================
# Operation: every path must return
# =============================================================================


def test_empty_module_passes():
    """Test that an empty module trivially validates."""
    program_ast = Module(provenance=Provenance.unknown())

    run_validator(ReturnValidator(), program_ast)


def test_operation_with_single_return_passes():
    """Test that an operation whose body ends with a return validates."""
    program_ast = Module(
        statements=(_make_operation(Identifier("op"), (_return_literal(),)),),
        provenance=Provenance.unknown(),
    )

    run_validator(ReturnValidator(), program_ast)


def test_operation_with_empty_body_fails():
    """Test that an operation with an empty body does not return on any path."""
    program_ast = Module(
        statements=(_make_operation(Identifier("op"), ()),),
        provenance=Provenance.unknown(),
    )

    with pytest.raises(ValidationFailedError, match="structural error"):
        run_validator(ReturnValidator(), program_ast)


def test_operation_with_no_return_fails():
    """Test that an operation whose body does not contain a return fails."""
    body = (
        ForAllStatement(
            index=IdentifierExpression(
                identifier=Identifier("i"), provenance=Provenance.unknown()
            ),
            body=(),
            provenance=Provenance.unknown(),
        ),
    )
    program_ast = Module(
        statements=(_make_operation(Identifier("op"), body),),
        provenance=Provenance.unknown(),
    )

    with pytest.raises(ValidationFailedError, match="structural error"):
        run_validator(ReturnValidator(), program_ast)


def test_operation_return_in_only_one_branch_fails():
    """Test that an operation that only returns in one branch of a selection fails.

    This exercises the "every path" part of the check: an operation whose
    only return lives inside the true-branch of a selection leaves the
    false-branch as a fall-through path that reaches the function exit
    without having returned.
    """
    selection = SelectionStatement(
        condition=IdentifierExpression(
            identifier=Identifier("x"), provenance=Provenance.unknown()
        ),
        true_body=(_return_literal(),),
        false_body=(),
        provenance=Provenance.unknown(),
    )
    program_ast = Module(
        statements=(_make_operation(Identifier("op"), (selection,)),),
        provenance=Provenance.unknown(),
    )

    with pytest.raises(ValidationFailedError, match="structural error"):
        run_validator(ReturnValidator(), program_ast)


def test_operation_return_in_both_branches_passes():
    """Test that an operation returning in both branches of a selection passes."""
    selection = SelectionStatement(
        condition=IdentifierExpression(
            identifier=Identifier("x"), provenance=Provenance.unknown()
        ),
        true_body=(_return_literal(),),
        false_body=(_return_literal(),),
        provenance=Provenance.unknown(),
    )
    program_ast = Module(
        statements=(_make_operation(Identifier("op"), (selection,)),),
        provenance=Provenance.unknown(),
    )

    run_validator(ReturnValidator(), program_ast)


def test_operation_return_inside_forall_body_passes():
    """Test that an operation whose forall body always returns passes.

    A `forall` body that itself always returns terminates the function on
    every path (the forall body runs at least once under FhY semantics),
    so the operation as a whole returns on every path.
    """
    forall = ForAllStatement(
        index=IdentifierExpression(
            identifier=Identifier("i"), provenance=Provenance.unknown()
        ),
        body=(_return_literal(),),
        provenance=Provenance.unknown(),
    )
    program_ast = Module(
        statements=(_make_operation(Identifier("op"), (forall,)),),
        provenance=Provenance.unknown(),
    )

    run_validator(ReturnValidator(), program_ast)


# =============================================================================
# Procedure: no return allowed
# =============================================================================


def test_procedure_without_return_passes():
    """Test that a procedure with no return validates."""
    program_ast = Module(
        statements=(_make_procedure(Identifier("main"), ()),),
        provenance=Provenance.unknown(),
    )

    run_validator(ReturnValidator(), program_ast)


def test_procedure_with_top_level_return_fails():
    """Test that a top-level return in a procedure fails."""
    program_ast = Module(
        statements=(_make_procedure(Identifier("main"), (_return_literal(),)),),
        provenance=Provenance.unknown(),
    )

    with pytest.raises(ValidationFailedError, match="structural error"):
        run_validator(ReturnValidator(), program_ast)


def test_procedure_with_return_inside_forall_fails():
    """Test that a return nested inside a for-all body in a procedure fails."""
    forall = ForAllStatement(
        index=IdentifierExpression(
            identifier=Identifier("i"), provenance=Provenance.unknown()
        ),
        body=(_return_literal(),),
        provenance=Provenance.unknown(),
    )
    program_ast = Module(
        statements=(_make_procedure(Identifier("main"), (forall,)),),
        provenance=Provenance.unknown(),
    )

    with pytest.raises(ValidationFailedError, match="structural error"):
        run_validator(ReturnValidator(), program_ast)


def test_procedure_with_return_inside_selection_fails():
    """Test that a return nested inside a selection branch in a procedure fails."""
    selection = SelectionStatement(
        condition=IdentifierExpression(
            identifier=Identifier("x"), provenance=Provenance.unknown()
        ),
        true_body=(_return_literal(),),
        false_body=(),
        provenance=Provenance.unknown(),
    )
    program_ast = Module(
        statements=(_make_procedure(Identifier("main"), (selection,)),),
        provenance=Provenance.unknown(),
    )

    with pytest.raises(ValidationFailedError, match="structural error"):
        run_validator(ReturnValidator(), program_ast)


def test_procedure_with_multiple_reachable_returns_reports_each():
    """Test that each reachable offending return produces its own diagnostic.

    The validator walks the CFG, so every return emitted into the CFG
    gets its own error. Placing returns in both branches of a selection
    makes them both reachable (and both emitted) so this test verifies
    that distinct diagnostics are produced, not deduplicated.
    """
    selection = SelectionStatement(
        condition=IdentifierExpression(
            identifier=Identifier("x"), provenance=Provenance.unknown()
        ),
        true_body=(_return_literal(),),
        false_body=(_return_literal(),),
        provenance=Provenance.unknown(),
    )
    program_ast = Module(
        statements=(_make_procedure(Identifier("main"), (selection,)),),
        provenance=Provenance.unknown(),
    )

    with pytest.raises(ValidationFailedError) as exc_info:
        run_validator(ReturnValidator(), program_ast)

    errors = exc_info.value.report.errors()
    assert len(errors) == 2
    assert all("structural error" in d.message_text for d in errors)


# =============================================================================
# Forall-semantic reminder: an index-decl + forall + return in body still
# requires the forall body to always return; a non-returning forall body
# leaves a fall-through.
# =============================================================================


def test_operation_forall_body_without_return_fails():
    """Test that a forall body that doesn't return leaves a fall-through."""
    i = Identifier("i")

    decl = DeclarationStatement(
        variable_name=i,
        variable_type=_qt(
            IndexType(
                lower_bound=LiteralExpression(1),
                upper_bound=LiteralExpression(10),
                stride=None,
            ),
            TypeQualifier.TEMP,
        ),
        provenance=Provenance.unknown(),
    )
    forall = ForAllStatement(
        index=IdentifierExpression(identifier=i, provenance=Provenance.unknown()),
        body=(),
        provenance=Provenance.unknown(),
    )
    program_ast = Module(
        statements=(_make_operation(Identifier("op"), (decl, forall)),),
        provenance=Provenance.unknown(),
    )

    with pytest.raises(ValidationFailedError, match="structural error"):
        run_validator(ReturnValidator(), program_ast)
