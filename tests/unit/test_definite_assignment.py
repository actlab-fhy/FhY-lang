"""Tests for the FhY definite-assignment analysis and validator."""

import pytest
from fhy.lang import FhYSemanticsError
from fhy.lang.ast import (
    ExpressionStatement,
    FunctionExpression,
    IdentifierExpression,
    Module,
    Procedure,
)
from fhy.lang.ast.cfg import build_cfg
from fhy.lang.ast.passes import (
    DefiniteAssignmentAnalysis,
    build_symbol_table,
    validate_definite_assignment,
)
from fhy.lang.ast.passes.definite_assignment import _compute_definite_assignment
from fhy_core import (
    Identifier,
    Provenance,
    TypeQualifier,
)

from .utils import (
    make_argument,
    make_identifier_assignment,
    make_module_with_statement,
    make_procedure,
    make_uninitialized_temp_declaration,
)


def _build_main_procedure_with_body(body, int32) -> Procedure:
    """Build `proc main(input int32 a, output int32 b) { body }`."""
    a, b = Identifier("a"), Identifier("b")
    return make_procedure(
        name=Identifier("main"),
        args=(
            make_argument(a, TypeQualifier.INPUT, int32),
            make_argument(b, TypeQualifier.OUTPUT, int32),
        ),
        body=body,
    )


def test_input_arguments_are_definitely_assigned_at_entry_exit(int32):
    """Test  the input arguments are definitely assigned at the entry and exit nodes."""
    a, b = Identifier("a"), Identifier("b")
    procedure_ast = make_procedure(
        name=Identifier("main"),
        args=(
            make_argument(a, TypeQualifier.INPUT, int32),
            make_argument(b, TypeQualifier.OUTPUT, int32),
        ),
        body=(make_identifier_assignment(b, a),),
    )
    program_ast = make_module_with_statement(procedure_ast)
    symbol_table = build_symbol_table(program_ast)
    cfg = build_cfg(procedure_ast)

    result = _compute_definite_assignment(procedure_ast, cfg, symbol_table)

    # INPUT `a` is definitely assigned from entry; OUTPUT `b` only becomes
    # assigned after the single `b = a` statement.
    assert a in result.live_out[cfg.entry.id]
    assert b not in result.live_out[cfg.entry.id]
    assert b in result.live_in[cfg.exit.id]


def test_output_not_assigned_on_all_paths_raises(int32):
    """Test an OUTPUT is not definitely assigned on all paths raises an error."""
    procedure_ast = _build_main_procedure_with_body(
        body=(make_uninitialized_temp_declaration(Identifier("x"), int32),),
        int32=int32,
    )
    program_ast = make_module_with_statement(procedure_ast)
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(FhYSemanticsError):
        validate_definite_assignment(program_ast, symbol_table)


def test_output_assigned_on_all_paths_validates(int32):
    """Test an OUTPUT argument is definitely assigned on all paths validates."""
    a, b = Identifier("a"), Identifier("b")
    procedure_ast = make_procedure(
        name=Identifier("main"),
        args=(
            make_argument(a, TypeQualifier.INPUT, int32),
            make_argument(b, TypeQualifier.OUTPUT, int32),
        ),
        body=(make_identifier_assignment(b, a),),
    )
    program_ast = make_module_with_statement(procedure_ast)
    symbol_table = build_symbol_table(program_ast)

    # Should not raise.
    validate_definite_assignment(program_ast, symbol_table)


def test_use_before_def_on_temp_raises(int32):
    """Test that a use before def on a TEMP raises an error."""
    a, b, x = Identifier("a"), Identifier("b"), Identifier("x")
    procedure_ast = make_procedure(
        name=Identifier("main"),
        args=(
            make_argument(a, TypeQualifier.INPUT, int32),
            make_argument(b, TypeQualifier.OUTPUT, int32),
        ),
        body=(
            make_uninitialized_temp_declaration(x, int32),
            # `x` is read before being written.
            make_identifier_assignment(b, x),
        ),
    )
    program_ast = make_module_with_statement(procedure_ast)
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(FhYSemanticsError):
        validate_definite_assignment(program_ast, symbol_table)


def test_temp_used_after_being_assigned_validates(int32):
    """Test that a TEMP used after being assigned validates."""
    a, b, x = Identifier("a"), Identifier("b"), Identifier("x")
    procedure_ast = make_procedure(
        name=Identifier("main"),
        args=(
            make_argument(a, TypeQualifier.INPUT, int32),
            make_argument(b, TypeQualifier.OUTPUT, int32),
        ),
        body=(
            make_uninitialized_temp_declaration(x, int32),
            make_identifier_assignment(x, a),
            make_identifier_assignment(b, x),
        ),
    )
    program_ast = make_module_with_statement(procedure_ast)
    symbol_table = build_symbol_table(program_ast)

    validate_definite_assignment(program_ast, symbol_table)


def test_analysis_result_contains_all_functions(int32):
    """Test that the analysis result contains all functions."""
    a, b = Identifier("a"), Identifier("b")
    foo = make_procedure(
        name=Identifier("foo"),
        args=(
            make_argument(a, TypeQualifier.INPUT, int32),
            make_argument(b, TypeQualifier.OUTPUT, int32),
        ),
        body=(make_identifier_assignment(b, a),),
    )
    bar = make_procedure(
        name=Identifier("bar"),
        args=(
            make_argument(a, TypeQualifier.INPUT, int32),
            make_argument(b, TypeQualifier.OUTPUT, int32),
        ),
        body=(make_identifier_assignment(b, a),),
    )
    program_ast = Module(statements=(foo, bar), provenance=Provenance.unknown())

    result = DefiniteAssignmentAnalysis().run(program_ast)

    assert foo.name in result.by_function
    assert bar.name in result.by_function


def test_procedure_call_with_output_arg_assigns_through(int32):
    """Test that a bare procedure call with an OUTPUT argument is treated as writing
    to that argument in the caller's scope."""
    a, b = Identifier("a"), Identifier("b")
    x, y = Identifier("x"), Identifier("y")
    callee = make_procedure(
        name=Identifier("write_y"),
        args=(
            make_argument(x, TypeQualifier.INPUT, int32),
            make_argument(y, TypeQualifier.OUTPUT, int32),
        ),
        body=(make_identifier_assignment(y, x),),
    )
    caller = make_procedure(
        name=Identifier("main"),
        args=(
            make_argument(a, TypeQualifier.INPUT, int32),
            make_argument(b, TypeQualifier.OUTPUT, int32),
        ),
        body=(
            # `write_y(a, b);` — bare procedure call writes to its OUTPUT
            # argument `b`, so `b` becomes definitely assigned afterwards.
            ExpressionStatement(
                left=None,
                right=FunctionExpression(
                    function=IdentifierExpression(
                        identifier=callee.name, provenance=Provenance.unknown()
                    ),
                    args=(
                        IdentifierExpression(
                            identifier=a, provenance=Provenance.unknown()
                        ),
                        IdentifierExpression(
                            identifier=b, provenance=Provenance.unknown()
                        ),
                    ),
                    provenance=Provenance.unknown(),
                ),
                provenance=Provenance.unknown(),
            ),
        ),
    )
    program_ast = Module(statements=(callee, caller), provenance=Provenance.unknown())
    symbol_table = build_symbol_table(program_ast)

    # Should not raise — b is written by the procedure call.
    validate_definite_assignment(program_ast, symbol_table)


def test_computed_sets_include_intermediate_assignments(int32):
    """Test that the computed sets include intermediate assignments."""
    a, b, x = Identifier("a"), Identifier("b"), Identifier("x")
    procedure_ast = make_procedure(
        name=Identifier("main"),
        args=(
            make_argument(a, TypeQualifier.INPUT, int32),
            make_argument(b, TypeQualifier.OUTPUT, int32),
        ),
        body=(
            make_uninitialized_temp_declaration(x, int32),
            make_identifier_assignment(x, a),
            make_identifier_assignment(b, x),
        ),
    )
    program_ast = make_module_with_statement(procedure_ast)
    symbol_table = build_symbol_table(program_ast)
    cfg = build_cfg(procedure_ast)
    result = _compute_definite_assignment(procedure_ast, cfg, symbol_table)

    statement_nodes = [node for node in cfg.nodes if node.kind.value == "statement"]
    # After the `x = a` statement (second statement node), x is in live_out.
    x_assign_node = statement_nodes[1]
    assert x in result.live_out[x_assign_node.id]
    # After the final `b = x`, both x and b are definitely assigned.
    b_assign_node = statement_nodes[2]
    assert {a, b, x}.issubset(result.live_out[b_assign_node.id])
