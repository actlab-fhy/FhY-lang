"""Tests for the FhY definite-assignment analysis and validator."""

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
from fhy_core import (
    IdentifierExpression as CoreIdentifierExpression,
)

from fhy_lang.ast import (
    Argument,
    ArrayAccessExpression,
    DeclarationStatement,
    ExpressionStatement,
    FunctionExpression,
    IdentifierExpression,
    IntLiteral,
    Module,
    Procedure,
    QualifiedType,
    SelectionStatement,
)
from fhy_lang.ast.passes import (
    DefiniteAssignmentValidator,
    build_symbol_table,
)
from fhy_lang.ast.passes.definite_assignment import (
    _ScalarDefiniteAssignmentAnalysis,
)

from .utils import (
    make_argument,
    make_identifier_assignment,
    make_module_with_statement,
    make_procedure,
    make_uninitialized_temp_declaration,
    run_validator,
)


def _make_int32_array(shape: tuple) -> NumericalType:
    return NumericalType(PrimitiveDataType(CoreDataType.INT32), shape=shape)


def _argument(name: Identifier, qualifier: TypeQualifier, base_type) -> Argument:
    return Argument(
        name=name,
        qualified_type=QualifiedType(
            base_type=base_type,
            type_qualifier=qualifier,
            provenance=Provenance.unknown(),
        ),
        provenance=Provenance.unknown(),
    )


def _index_declaration(
    name: Identifier, lower_bound, upper_bound
) -> DeclarationStatement:
    return DeclarationStatement(
        variable_name=name,
        variable_type=QualifiedType(
            base_type=IndexType(
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                stride=None,
            ),
            type_qualifier=TypeQualifier.TEMP,
            provenance=Provenance.unknown(),
        ),
        provenance=Provenance.unknown(),
    )


def _array_write(
    array_name: Identifier,
    index_names: tuple[Identifier, ...],
    source: Identifier,
) -> ExpressionStatement:
    return ExpressionStatement(
        left=ArrayAccessExpression(
            array_expression=IdentifierExpression(
                identifier=array_name, provenance=Provenance.unknown()
            ),
            indices=tuple(
                IdentifierExpression(identifier=name, provenance=Provenance.unknown())
                for name in index_names
            ),
            provenance=Provenance.unknown(),
        ),
        right=IdentifierExpression(identifier=source, provenance=Provenance.unknown()),
        provenance=Provenance.unknown(),
    )


def _array_write_with_literal_indices(
    array_name: Identifier,
    index_values: tuple[int, ...],
    source: Identifier,
) -> ExpressionStatement:
    return ExpressionStatement(
        left=ArrayAccessExpression(
            array_expression=IdentifierExpression(
                identifier=array_name, provenance=Provenance.unknown()
            ),
            indices=tuple(
                IntLiteral(value=value, provenance=Provenance.unknown())
                for value in index_values
            ),
            provenance=Provenance.unknown(),
        ),
        right=IdentifierExpression(identifier=source, provenance=Provenance.unknown()),
        provenance=Provenance.unknown(),
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

    result = _ScalarDefiniteAssignmentAnalysis(procedure_ast, symbol_table).run()

    # INPUT `a` is definitely assigned from entry; OUTPUT `b` only becomes
    # assigned after the single `b = a` statement.
    assert a in result.definitely_assigned_out[result.cfg.entry.id]
    assert b not in result.definitely_assigned_out[result.cfg.entry.id]
    assert b in result.definitely_assigned_in[result.cfg.exit.id]


def test_output_not_assigned_on_all_paths_raises(int32):
    """Test an OUTPUT is not definitely assigned on all paths raises an error."""
    procedure_ast = _build_main_procedure_with_body(
        body=(make_uninitialized_temp_declaration(Identifier("x"), int32),),
        int32=int32,
    )
    program_ast = make_module_with_statement(procedure_ast)
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(ValidationFailedError, match="semantic error"):
        run_validator(DefiniteAssignmentValidator(symbol_table), program_ast)


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
    run_validator(DefiniteAssignmentValidator(symbol_table), program_ast)


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

    with pytest.raises(ValidationFailedError, match="semantic error"):
        run_validator(DefiniteAssignmentValidator(symbol_table), program_ast)


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

    run_validator(DefiniteAssignmentValidator(symbol_table), program_ast)


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
    run_validator(DefiniteAssignmentValidator(symbol_table), program_ast)


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
    result = _ScalarDefiniteAssignmentAnalysis(procedure_ast, symbol_table).run()

    statement_nodes = [
        node for node in result.cfg.nodes if node.kind.value == "statement"
    ]
    # After the `x = a` statement (second statement node), x is in
    # definitely_assigned_out.
    x_assign_node = statement_nodes[1]
    assert x in result.definitely_assigned_out[x_assign_node.id]
    # After the final `b = x`, both x and b are definitely assigned.
    b_assign_node = statement_nodes[2]
    assert {a, b, x}.issubset(result.definitely_assigned_out[b_assign_node.id])


def test_output_array_fully_written_via_index_validates():
    """Test that a matmul-style ``C[i, j] = a`` write with indices spanning
    the shape fully covers the array."""
    a, c, i, j, m, p = (
        Identifier("a"),
        Identifier("C"),
        Identifier("i"),
        Identifier("j"),
        Identifier("m"),
        Identifier("p"),
    )
    one = LiteralExpression(1)
    shape = (CoreIdentifierExpression(m), CoreIdentifierExpression(p))
    procedure_ast = Procedure(
        name=Identifier("matmul_like"),
        args=(
            _argument(
                a,
                TypeQualifier.INPUT,
                NumericalType(PrimitiveDataType(CoreDataType.INT32)),
            ),
            _argument(c, TypeQualifier.OUTPUT, _make_int32_array(shape)),
        ),
        body=(
            _index_declaration(i, one, CoreIdentifierExpression(m)),
            _index_declaration(j, one, CoreIdentifierExpression(p)),
            _array_write(c, (i, j), a),
        ),
        provenance=Provenance.unknown(),
    )
    program_ast = make_module_with_statement(procedure_ast)
    symbol_table = build_symbol_table(program_ast)

    run_validator(DefiniteAssignmentValidator(symbol_table), program_ast)


def test_output_array_only_one_element_written_raises():
    """Test that a single-element write ``C[1] = a`` fails coverage for an
    N-element array."""
    a, c, n = Identifier("a"), Identifier("C"), Identifier("n")
    shape = (CoreIdentifierExpression(n),)
    procedure_ast = Procedure(
        name=Identifier("write_first"),
        args=(
            _argument(
                a,
                TypeQualifier.INPUT,
                NumericalType(PrimitiveDataType(CoreDataType.INT32)),
            ),
            _argument(c, TypeQualifier.OUTPUT, _make_int32_array(shape)),
        ),
        body=(_array_write_with_literal_indices(c, (1,), a),),
        provenance=Provenance.unknown(),
    )
    program_ast = make_module_with_statement(procedure_ast)
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(ValidationFailedError, match="not fully written on every"):
        run_validator(DefiniteAssignmentValidator(symbol_table), program_ast)


def test_output_array_with_no_writes_raises():
    """Test that an OUTPUT array left entirely unwritten is flagged by the
    coverage check."""
    a, c, n = Identifier("a"), Identifier("C"), Identifier("n")
    shape = (CoreIdentifierExpression(n),)
    procedure_ast = Procedure(
        name=Identifier("no_writes"),
        args=(
            _argument(
                a,
                TypeQualifier.INPUT,
                NumericalType(PrimitiveDataType(CoreDataType.INT32)),
            ),
            _argument(c, TypeQualifier.OUTPUT, _make_int32_array(shape)),
        ),
        body=(),
        provenance=Provenance.unknown(),
    )
    program_ast = make_module_with_statement(procedure_ast)
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(ValidationFailedError, match="not fully written on every"):
        run_validator(DefiniteAssignmentValidator(symbol_table), program_ast)


def test_output_array_sub_range_written_raises():
    """Test that a write over an index sub-range fails coverage for the
    full array shape."""
    a, c, i = Identifier("a"), Identifier("C"), Identifier("i")
    shape = (LiteralExpression(10),)
    procedure_ast = Procedure(
        name=Identifier("partial"),
        args=(
            _argument(
                a,
                TypeQualifier.INPUT,
                NumericalType(PrimitiveDataType(CoreDataType.INT32)),
            ),
            _argument(c, TypeQualifier.OUTPUT, _make_int32_array(shape)),
        ),
        body=(
            _index_declaration(i, LiteralExpression(1), LiteralExpression(5)),
            _array_write(c, (i,), a),
        ),
        provenance=Provenance.unknown(),
    )
    program_ast = make_module_with_statement(procedure_ast)
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(ValidationFailedError, match="not fully written on every"):
        run_validator(DefiniteAssignmentValidator(symbol_table), program_ast)


def test_output_array_writes_in_both_selection_branches_validates():
    """Test that writes on both branches of a selection covering the same
    range validates."""
    a, c, i = Identifier("a"), Identifier("C"), Identifier("i")
    shape = (LiteralExpression(10),)
    procedure_ast = Procedure(
        name=Identifier("guarded_full"),
        args=(
            _argument(
                a,
                TypeQualifier.INPUT,
                NumericalType(PrimitiveDataType(CoreDataType.INT32)),
            ),
            _argument(c, TypeQualifier.OUTPUT, _make_int32_array(shape)),
        ),
        body=(
            _index_declaration(i, LiteralExpression(1), LiteralExpression(10)),
            SelectionStatement(
                condition=IdentifierExpression(
                    identifier=a, provenance=Provenance.unknown()
                ),
                true_body=(_array_write(c, (i,), a),),
                false_body=(_array_write(c, (i,), a),),
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )
    program_ast = make_module_with_statement(procedure_ast)
    symbol_table = build_symbol_table(program_ast)

    run_validator(DefiniteAssignmentValidator(symbol_table), program_ast)


def test_output_array_write_in_only_one_selection_branch_raises():
    """Test that a write on only one branch of a selection fails the
    must-coverage check."""
    a, c, i = Identifier("a"), Identifier("C"), Identifier("i")
    shape = (LiteralExpression(10),)
    procedure_ast = Procedure(
        name=Identifier("guarded_partial"),
        args=(
            _argument(
                a,
                TypeQualifier.INPUT,
                NumericalType(PrimitiveDataType(CoreDataType.INT32)),
            ),
            _argument(c, TypeQualifier.OUTPUT, _make_int32_array(shape)),
        ),
        body=(
            _index_declaration(i, LiteralExpression(1), LiteralExpression(10)),
            SelectionStatement(
                condition=IdentifierExpression(
                    identifier=a, provenance=Provenance.unknown()
                ),
                true_body=(_array_write(c, (i,), a),),
                false_body=(),
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )
    program_ast = make_module_with_statement(procedure_ast)
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(ValidationFailedError, match="not fully written on every"):
        run_validator(DefiniteAssignmentValidator(symbol_table), program_ast)


def test_output_array_written_by_procedure_call_through_identifier_validates():
    """Test that a bare call binding the array identifier to an OUTPUT
    argument covers the array fully."""
    a, c, n = Identifier("a"), Identifier("C"), Identifier("n")
    x, y = Identifier("x"), Identifier("y")
    shape = (CoreIdentifierExpression(n),)
    callee_i = Identifier("i")
    callee = Procedure(
        name=Identifier("fill_all"),
        args=(
            _argument(
                x,
                TypeQualifier.INPUT,
                NumericalType(PrimitiveDataType(CoreDataType.INT32)),
            ),
            _argument(y, TypeQualifier.OUTPUT, _make_int32_array(shape)),
        ),
        body=(
            _index_declaration(
                callee_i, LiteralExpression(1), CoreIdentifierExpression(n)
            ),
            _array_write(y, (callee_i,), x),
        ),
        provenance=Provenance.unknown(),
    )
    caller_body_i = Identifier("i")
    caller = Procedure(
        name=Identifier("main"),
        args=(
            _argument(
                a,
                TypeQualifier.INPUT,
                NumericalType(PrimitiveDataType(CoreDataType.INT32)),
            ),
            _argument(c, TypeQualifier.OUTPUT, _make_int32_array(shape)),
        ),
        body=(
            _index_declaration(
                caller_body_i,
                LiteralExpression(1),
                CoreIdentifierExpression(n),
            ),
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
                            identifier=c, provenance=Provenance.unknown()
                        ),
                    ),
                    provenance=Provenance.unknown(),
                ),
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )
    program_ast = Module(statements=(callee, caller), provenance=Provenance.unknown())
    symbol_table = build_symbol_table(program_ast)

    run_validator(DefiniteAssignmentValidator(symbol_table), program_ast)
