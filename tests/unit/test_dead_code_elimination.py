"""Tests the dead code elimination pass."""

from fhy.lang.ast import (
    Argument,
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
from fhy.lang.ast.passes import (
    DeadCodeEliminationPass,
    build_symbol_table,
)
from fhy_core import (
    AnalysisManager,
    FixpointGroupRecord,
    FixpointPassGroup,
    Identifier,
    PassManager,
    Provenance,
    TypeQualifier,
)


def _run_dce(ast: Module) -> Module:
    symbol_table = build_symbol_table(ast)
    manager = AnalysisManager[Module]()
    pass_ = DeadCodeEliminationPass(manager, symbol_table)
    return pass_(ast)


def _get_main_procedure(ast: Module) -> Procedure:
    for statement in ast.statements:
        if isinstance(statement, Procedure):
            return statement
    raise AssertionError("no procedure in module")


def test_dce_removes_dead_assignment_to_temp(int32):
    """Test an assignment to a TEMP that is overwritten before use is removed."""
    a, b, x = Identifier("a"), Identifier("b"), Identifier("x")
    procedure_ast = Procedure(
        name=Identifier("main"),
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
            Argument(
                name=b,
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
                variable_name=x,
                variable_type=QualifiedType(
                    base_type=int32,
                    type_qualifier=TypeQualifier.TEMP,
                    provenance=Provenance.unknown(),
                ),
                provenance=Provenance.unknown(),
            ),
            ExpressionStatement(
                left=IdentifierExpression(
                    identifier=x, provenance=Provenance.unknown()
                ),
                right=IdentifierExpression(
                    identifier=a, provenance=Provenance.unknown()
                ),
                provenance=Provenance.unknown(),
            ),
            ExpressionStatement(
                left=IdentifierExpression(
                    identifier=x, provenance=Provenance.unknown()
                ),
                right=IdentifierExpression(
                    identifier=a, provenance=Provenance.unknown()
                ),
                provenance=Provenance.unknown(),
            ),
            ExpressionStatement(
                left=IdentifierExpression(
                    identifier=b, provenance=Provenance.unknown()
                ),
                right=IdentifierExpression(
                    identifier=x, provenance=Provenance.unknown()
                ),
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )
    program_ast = Module(statements=(procedure_ast,), provenance=Provenance.unknown())

    optimized = _run_dce(program_ast)

    procedure_ast = _get_main_procedure(optimized)
    statements = procedure_ast.body
    assignment_count = sum(1 for s in statements if isinstance(s, ExpressionStatement))
    assert assignment_count == 2
    assert any(isinstance(s, DeclarationStatement) for s in statements)


def test_dce_keeps_live_assignment(construct_ast):
    """Test an assignment whose value is used downstream is preserved."""
    source = """
    proc main(input int32 a, output int32 b) {
        b = a;
    }
    """
    ast = construct_ast(source)
    optimized = _run_dce(ast)

    procedure = _get_main_procedure(optimized)
    assert len(procedure.body) == 1
    assert isinstance(procedure.body[0], ExpressionStatement)


def test_dce_does_not_remove_writes_to_output(int32):
    """Test DCE must never remove writes to OUTPUT arguments."""
    a, b = Identifier("a"), Identifier("b")
    procedure_ast = Procedure(
        name=Identifier("main"),
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
            Argument(
                name=b,
                qualified_type=QualifiedType(
                    base_type=int32,
                    type_qualifier=TypeQualifier.OUTPUT,
                    provenance=Provenance.unknown(),
                ),
                provenance=Provenance.unknown(),
            ),
        ),
        body=(
            ExpressionStatement(
                left=IdentifierExpression(
                    identifier=b, provenance=Provenance.unknown()
                ),
                right=IdentifierExpression(
                    identifier=a, provenance=Provenance.unknown()
                ),
                provenance=Provenance.unknown(),
            ),
            ExpressionStatement(
                left=IdentifierExpression(
                    identifier=b, provenance=Provenance.unknown()
                ),
                right=IdentifierExpression(
                    identifier=a, provenance=Provenance.unknown()
                ),
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )
    program_ast = Module(statements=(procedure_ast,), provenance=Provenance.unknown())

    optimized = _run_dce(program_ast)

    procedure_ast = _get_main_procedure(optimized)
    assignments = [s for s in procedure_ast.body if isinstance(s, ExpressionStatement)]
    assert len(assignments) == 2


def test_dce_removes_dead_initialized_declaration(int32):
    """Test a TEMP declaration-with-initializer whose value is never used is removed."""
    a, b, unused = Identifier("a"), Identifier("b"), Identifier("unused")
    procedure_ast = Procedure(
        name=Identifier("main"),
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
            Argument(
                name=b,
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
                variable_name=unused,
                variable_type=QualifiedType(
                    base_type=int32,
                    type_qualifier=TypeQualifier.TEMP,
                    provenance=Provenance.unknown(),
                ),
                expression=IdentifierExpression(
                    identifier=a, provenance=Provenance.unknown()
                ),
                provenance=Provenance.unknown(),
            ),
            ExpressionStatement(
                left=IdentifierExpression(
                    identifier=b, provenance=Provenance.unknown()
                ),
                right=IdentifierExpression(
                    identifier=a, provenance=Provenance.unknown()
                ),
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )
    program_ast = Module(statements=(procedure_ast,), provenance=Provenance.unknown())

    optimized = _run_dce(program_ast)

    procedure_ast = _get_main_procedure(optimized)
    assert not any(isinstance(s, DeclarationStatement) for s in procedure_ast.body)


def test_dce_preserves_function_call_with_side_effects(int32):
    """Test DCE must not remove assignments whose RHS contains a function call."""
    source = """
    op add(input int32 x, input int32 y) -> output int32 {
        temp int32 result;
        result = x + y;
        return result;
    }

    proc main(input int32 a, output int32 b) {
        temp int32 t;
        t = add(a, a);
        t = a;
        b = t;
    }
    """
    add, x, y, result = (
        Identifier("add"),
        Identifier("x"),
        Identifier("y"),
        Identifier("result"),
    )
    operation_ast = Operation(
        name=add,
        templates=(),
        args=(
            Argument(
                name=x,
                qualified_type=QualifiedType(
                    base_type=int32,
                    type_qualifier=TypeQualifier.INPUT,
                    provenance=Provenance.unknown(),
                ),
                provenance=Provenance.unknown(),
            ),
            Argument(
                name=y,
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
                variable_name=result,
                variable_type=QualifiedType(
                    base_type=int32,
                    type_qualifier=TypeQualifier.TEMP,
                    provenance=Provenance.unknown(),
                ),
                expression=IdentifierExpression(
                    identifier=x, provenance=Provenance.unknown()
                ),
                provenance=Provenance.unknown(),
            ),
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
    procedure_ast = Procedure(
        name=Identifier("main"),
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
            Argument(
                name=b,
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
                left=IdentifierExpression(
                    identifier=t, provenance=Provenance.unknown()
                ),
                right=FunctionExpression(
                    function=IdentifierExpression(
                        identifier=add, provenance=Provenance.unknown()
                    ),
                    template_types=(),
                    indices=(),
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
            ExpressionStatement(
                left=IdentifierExpression(
                    identifier=t, provenance=Provenance.unknown()
                ),
                right=IdentifierExpression(
                    identifier=a, provenance=Provenance.unknown()
                ),
                provenance=Provenance.unknown(),
            ),
            ExpressionStatement(
                left=IdentifierExpression(
                    identifier=b, provenance=Provenance.unknown()
                ),
                right=IdentifierExpression(
                    identifier=t, provenance=Provenance.unknown()
                ),
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )
    program_ast = Module(
        statements=(
            operation_ast,
            procedure_ast,
        ),
        provenance=Provenance.unknown(),
    )

    optimized = _run_dce(program_ast)

    procedure_ast = _get_main_procedure(optimized)
    expression_statements = [
        s for s in procedure_ast.body if isinstance(s, ExpressionStatement)
    ]
    assert len(expression_statements) == 3


def test_dce_is_idempotent_in_fixpoint_group(int32):
    """Test the fixpoint group converges in a single additional no-op iteration."""
    x, a, b = Identifier("x"), Identifier("a"), Identifier("b")
    procedure_ast = Procedure(
        name=Identifier("main"),
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
            Argument(
                name=b,
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
                variable_name=x,
                variable_type=QualifiedType(
                    base_type=int32,
                    type_qualifier=TypeQualifier.TEMP,
                    provenance=Provenance.unknown(),
                ),
                provenance=Provenance.unknown(),
            ),
            ExpressionStatement(
                left=IdentifierExpression(
                    identifier=x, provenance=Provenance.unknown()
                ),
                right=IdentifierExpression(
                    identifier=a, provenance=Provenance.unknown()
                ),
                provenance=Provenance.unknown(),
            ),
            ExpressionStatement(
                left=IdentifierExpression(
                    identifier=x, provenance=Provenance.unknown()
                ),
                right=IdentifierExpression(
                    identifier=a, provenance=Provenance.unknown()
                ),
                provenance=Provenance.unknown(),
            ),
            ExpressionStatement(
                left=IdentifierExpression(
                    identifier=x, provenance=Provenance.unknown()
                ),
                right=IdentifierExpression(
                    identifier=a, provenance=Provenance.unknown()
                ),
                provenance=Provenance.unknown(),
            ),
            ExpressionStatement(
                left=IdentifierExpression(
                    identifier=b, provenance=Provenance.unknown()
                ),
                right=IdentifierExpression(
                    identifier=x, provenance=Provenance.unknown()
                ),
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )
    program_ast = Module(statements=(procedure_ast,), provenance=Provenance.unknown())

    symbol_table = build_symbol_table(program_ast)
    manager = PassManager[Module](Identifier("test_dce_fixpoint"))
    fixpoint_group = FixpointPassGroup[Module](
        name=Identifier("dce_group"),
    )
    fixpoint_group.add_pass(
        DeadCodeEliminationPass(manager.analysis_manager, symbol_table)
    )
    manager.add_fixpoint_group(fixpoint_group)

    result = manager.run(program_ast)

    assert len(result.records) == 1
    record = result.records[0]
    assert isinstance(record, FixpointGroupRecord)
    assert record.converged is True
    assert record.iterations >= 2
    assert record.iteration_records[0].changed is True
    assert record.iteration_records[-1].changed is False
    optimized_procedure = _get_main_procedure(result.output)
    expression_statements = [
        s for s in optimized_procedure.body if isinstance(s, ExpressionStatement)
    ]
    assert len(expression_statements) == 2
