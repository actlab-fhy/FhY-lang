"""Tests the dead code elimination pass."""

from fhy_core import (
    FixpointGroupRecord,
    FixpointPassGroup,
    Identifier,
    PassManager,
    Provenance,
    TypeQualifier,
)
from fhy_lang.lang.ast import (
    BinaryExpression,
    BinaryOperation,
    DeclarationStatement,
    ExpressionStatement,
    FunctionExpression,
    IdentifierExpression,
    Module,
    Operation,
    Procedure,
    QualifiedType,
)
from fhy_lang.lang.ast.passes import (
    DeadCodeEliminationPass,
    build_symbol_table,
)

from .utils import (
    make_argument,
    make_identifier_assignment,
    make_initialized_temp_declaration,
    make_module_with_statement,
    make_procedure,
    make_uninitialized_temp_declaration,
)


def _run_dce(ast: Module) -> Module:
    symbol_table = build_symbol_table(ast)
    dce_pass = DeadCodeEliminationPass(symbol_table)
    return dce_pass(ast)


def _get_main_procedure(ast: Module) -> Procedure:
    for statement in ast.statements:
        if isinstance(statement, Procedure):
            return statement
    raise AssertionError("no procedure in module")


def _build_main_procedure(args, body) -> Procedure:
    """Build a `Procedure` named `main` with the given args and body."""
    return make_procedure(name=Identifier("main"), args=args, body=body)


def test_dce_removes_dead_assignment_to_temp(int32):
    """Test an assignment to a TEMP that is overwritten before use is removed."""
    a, b, x = Identifier("a"), Identifier("b"), Identifier("x")
    procedure_ast = _build_main_procedure(
        args=(
            make_argument(a, TypeQualifier.INPUT, int32),
            make_argument(b, TypeQualifier.OUTPUT, int32),
        ),
        body=(
            make_uninitialized_temp_declaration(x, int32),
            make_identifier_assignment(x, a),
            make_identifier_assignment(x, a),
            make_identifier_assignment(b, x),
        ),
    )
    program_ast = make_module_with_statement(procedure_ast)

    optimized_ast = _run_dce(program_ast)

    optimized_procedure = _get_main_procedure(optimized_ast)
    assignment_count = sum(
        1 for s in optimized_procedure.body if isinstance(s, ExpressionStatement)
    )
    assert assignment_count == 2
    assert any(isinstance(s, DeclarationStatement) for s in optimized_procedure.body)


def test_dce_keeps_live_assignment(construct_ast):
    """Test an assignment whose value is used downstream is preserved."""
    source = """
    proc main(input int32 a, output int32 b) {
        b = a;
    }
    """
    program_ast = construct_ast(source)

    optimized_ast = _run_dce(program_ast)

    optimized_procedure = _get_main_procedure(optimized_ast)
    assert len(optimized_procedure.body) == 1
    assert isinstance(optimized_procedure.body[0], ExpressionStatement)


def test_dce_does_not_remove_writes_to_output(int32):
    """Test DCE must never remove writes to OUTPUT arguments."""
    a, b = Identifier("a"), Identifier("b")
    procedure_ast = _build_main_procedure(
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

    optimized_ast = _run_dce(program_ast)

    optimized_procedure = _get_main_procedure(optimized_ast)
    assignments = [
        s for s in optimized_procedure.body if isinstance(s, ExpressionStatement)
    ]
    assert len(assignments) == 2


def test_dce_removes_dead_initialized_declaration(int32):
    """Test a TEMP declaration-with-initializer whose value is never used is removed."""
    a, b, unused = Identifier("a"), Identifier("b"), Identifier("unused")
    procedure_ast = _build_main_procedure(
        args=(
            make_argument(a, TypeQualifier.INPUT, int32),
            make_argument(b, TypeQualifier.OUTPUT, int32),
        ),
        body=(
            make_initialized_temp_declaration(unused, int32, a),
            make_identifier_assignment(b, a),
        ),
    )
    program_ast = make_module_with_statement(procedure_ast)

    optimized_ast = _run_dce(program_ast)

    optimized_procedure = _get_main_procedure(optimized_ast)
    assert not any(
        isinstance(s, DeclarationStatement) for s in optimized_procedure.body
    )


def test_dce_removes_dead_uninitialized_declaration(int32):
    """Test a TEMP declaration without an initializer whose variable is never
    referenced is removed."""
    a, b, unused = Identifier("a"), Identifier("b"), Identifier("unused")
    procedure_ast = _build_main_procedure(
        args=(
            make_argument(a, TypeQualifier.INPUT, int32),
            make_argument(b, TypeQualifier.OUTPUT, int32),
        ),
        body=(
            make_uninitialized_temp_declaration(unused, int32),
            make_identifier_assignment(b, a),
        ),
    )
    program_ast = make_module_with_statement(procedure_ast)

    optimized_ast = _run_dce(program_ast)

    optimized_procedure = _get_main_procedure(optimized_ast)
    assert not any(
        isinstance(s, DeclarationStatement) for s in optimized_procedure.body
    )


def test_dce_keeps_uninitialized_declaration_when_used(int32):
    """Test a TEMP declaration without an initializer is kept when the
    variable is later read."""
    a, b, x = Identifier("a"), Identifier("b"), Identifier("x")
    procedure_ast = _build_main_procedure(
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

    optimized_ast = _run_dce(program_ast)

    optimized_procedure = _get_main_procedure(optimized_ast)
    assert any(isinstance(s, DeclarationStatement) for s in optimized_procedure.body)


def test_dce_preserves_function_call_with_side_effects(int32):
    """Test DCE must not remove assignments whose RHS contains a function call."""
    add, x, y, result = (
        Identifier("add"),
        Identifier("x"),
        Identifier("y"),
        Identifier("result"),
    )
    add_operation_ast = Operation(
        name=add,
        templates=(),
        args=(
            make_argument(x, TypeQualifier.INPUT, int32),
            make_argument(y, TypeQualifier.INPUT, int32),
        ),
        body=(
            make_initialized_temp_declaration(result, int32, x),
            ExpressionStatement(
                left=IdentifierExpression(
                    identifier=result, provenance=Provenance.unknown()
                ),
                right=BinaryExpression(
                    operation=BinaryOperation.ADDITION,
                    left=IdentifierExpression(
                        identifier=x, provenance=Provenance.unknown()
                    ),
                    right=IdentifierExpression(
                        identifier=y, provenance=Provenance.unknown()
                    ),
                    provenance=Provenance.unknown(),
                ),
                provenance=Provenance.unknown(),
            ),
        ),
        return_type=QualifiedType(
            base_type=int32,
            type_qualifier=TypeQualifier.OUTPUT,
            provenance=Provenance.unknown(),
        ),
        provenance=Provenance.unknown(),
    )
    a, b, t = Identifier("a"), Identifier("b"), Identifier("t")
    main_procedure_ast = _build_main_procedure(
        args=(
            make_argument(a, TypeQualifier.INPUT, int32),
            make_argument(b, TypeQualifier.OUTPUT, int32),
        ),
        body=(
            make_uninitialized_temp_declaration(t, int32),
            ExpressionStatement(
                left=IdentifierExpression(
                    identifier=t, provenance=Provenance.unknown()
                ),
                right=FunctionExpression(
                    function=IdentifierExpression(
                        identifier=add, provenance=Provenance.unknown()
                    ),
                    args=(
                        IdentifierExpression(
                            identifier=a, provenance=Provenance.unknown()
                        ),
                        IdentifierExpression(
                            identifier=a, provenance=Provenance.unknown()
                        ),
                    ),
                    provenance=Provenance.unknown(),
                ),
                provenance=Provenance.unknown(),
            ),
            make_identifier_assignment(t, a),
            make_identifier_assignment(b, t),
        ),
    )
    program_ast = Module(
        statements=(add_operation_ast, main_procedure_ast),
        provenance=Provenance.unknown(),
    )

    optimized_ast = _run_dce(program_ast)

    optimized_procedure = _get_main_procedure(optimized_ast)
    expression_statements = [
        s for s in optimized_procedure.body if isinstance(s, ExpressionStatement)
    ]
    assert len(expression_statements) == 3


def test_dce_is_idempotent_in_fixpoint_group(int32):
    """Test the fixpoint group converges in a single additional no-op iteration."""
    a, b, x = Identifier("a"), Identifier("b"), Identifier("x")
    procedure_ast = _build_main_procedure(
        args=(
            make_argument(a, TypeQualifier.INPUT, int32),
            make_argument(b, TypeQualifier.OUTPUT, int32),
        ),
        body=(
            make_uninitialized_temp_declaration(x, int32),
            make_identifier_assignment(x, a),
            make_identifier_assignment(x, a),
            make_identifier_assignment(x, a),
            make_identifier_assignment(b, x),
        ),
    )
    program_ast = make_module_with_statement(procedure_ast)

    symbol_table = build_symbol_table(program_ast)
    pass_manager = PassManager[Module](Identifier("test_dce_fixpoint"))
    fixpoint_group = FixpointPassGroup[Module](name=Identifier("dce_group"))
    fixpoint_group.add_pass(DeadCodeEliminationPass(symbol_table))
    pass_manager.add_fixpoint_group(fixpoint_group)

    result = pass_manager.run(program_ast)

    assert len(result.records) == 1
    fixpoint_record = result.records[0]
    assert isinstance(fixpoint_record, FixpointGroupRecord)
    assert fixpoint_record.converged is True
    assert fixpoint_record.iterations >= 2
    assert fixpoint_record.iteration_records[0].changed is True
    assert fixpoint_record.iteration_records[-1].changed is False

    optimized_procedure = _get_main_procedure(result.output)
    expression_statements = [
        s for s in optimized_procedure.body if isinstance(s, ExpressionStatement)
    ]
    assert len(expression_statements) == 2
