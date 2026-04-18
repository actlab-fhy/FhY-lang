"""Tests the liveness analysis pass."""

from fhy.lang.ast import (
    Argument,
    DeclarationStatement,
    ExpressionStatement,
    IdentifierExpression,
    Module,
    Procedure,
    QualifiedType,
    SelectionStatement,
)
from fhy.lang.ast.passes import (
    LivenessAnalysis,
    LivenessResult,
)
from fhy_core import (
    Identifier,
    Provenance,
    TypeQualifier,
)


def _run_analysis(ast: Module) -> LivenessResult:
    return LivenessAnalysis().run(ast)


def test_liveness_result_default_is_empty_sets():
    """Test a statement with no recorded liveness returns empty sets."""
    procedure_ast = Procedure(
        name=Identifier("main"),
        args=(),
        body=(),
        provenance=Provenance.unknown(),
    )
    program_ast = Module(statements=(procedure_ast,), provenance=Provenance.unknown())

    result = _run_analysis(program_ast)

    assert result.get_live_input_identifiers(procedure_ast) == frozenset()
    assert result.get_live_output_identifiers(procedure_ast) == frozenset()


def test_dead_assignment_is_not_live_out(int32):
    """Test an assignment to a TEMP whose value is overwritten is not live out."""
    a, b, x = Identifier("a"), Identifier("b"), Identifier("x")
    declaration_ast = DeclarationStatement(
        variable_name=x,
        variable_type=QualifiedType(
            base_type=int32,
            type_qualifier=TypeQualifier.TEMP,
            provenance=Provenance.unknown(),
        ),
        provenance=Provenance.unknown(),
    )
    assignment_ast_0 = ExpressionStatement(
        left=IdentifierExpression(identifier=x, provenance=Provenance.unknown()),
        right=IdentifierExpression(identifier=a, provenance=Provenance.unknown()),
        provenance=Provenance.unknown(),
    )
    assignment_ast_1 = ExpressionStatement(
        left=IdentifierExpression(identifier=x, provenance=Provenance.unknown()),
        right=IdentifierExpression(identifier=a, provenance=Provenance.unknown()),
        provenance=Provenance.unknown(),
    )
    assignment_ast_2 = ExpressionStatement(
        left=IdentifierExpression(identifier=b, provenance=Provenance.unknown()),
        right=IdentifierExpression(identifier=x, provenance=Provenance.unknown()),
        provenance=Provenance.unknown(),
    )
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
        body=(declaration_ast, assignment_ast_0, assignment_ast_1, assignment_ast_2),
        provenance=Provenance.unknown(),
    )
    program_ast = Module(statements=(procedure_ast,), provenance=Provenance.unknown())

    result = _run_analysis(program_ast)

    assert x not in result.get_live_output_identifiers(assignment_ast_0)
    assert x in result.get_live_output_identifiers(assignment_ast_1)
    assert result.get_live_output_identifiers(assignment_ast_2) == frozenset({b})
    assert a in result.get_live_input_identifiers(assignment_ast_0)


def test_selection_statement_unions_branch_liveness(int32):
    """Test selection statement live_in should union condition and branch uses."""
    a, b, c, y = Identifier("a"), Identifier("b"), Identifier("c"), Identifier("y")

    def _expr(name: Identifier):
        return IdentifierExpression(identifier=name, provenance=Provenance.unknown())

    def _arg(name: Identifier, qualifier: TypeQualifier) -> Argument:
        return Argument(
            name=name,
            qualified_type=QualifiedType(
                base_type=int32,
                type_qualifier=qualifier,
                provenance=Provenance.unknown(),
            ),
            provenance=Provenance.unknown(),
        )

    true_stmt = ExpressionStatement(
        left=_expr(y), right=_expr(a), provenance=Provenance.unknown()
    )
    false_stmt = ExpressionStatement(
        left=_expr(y), right=_expr(b), provenance=Provenance.unknown()
    )
    selection = SelectionStatement(
        condition=_expr(c),
        true_body=(true_stmt,),
        false_body=(false_stmt,),
        provenance=Provenance.unknown(),
    )

    procedure = Procedure(
        name=Identifier("main"),
        args=(
            _arg(a, TypeQualifier.INPUT),
            _arg(b, TypeQualifier.INPUT),
            _arg(c, TypeQualifier.INPUT),
            _arg(y, TypeQualifier.OUTPUT),
        ),
        body=(selection,),
        provenance=Provenance.unknown(),
    )
    ast = Module(statements=(procedure,), provenance=Provenance.unknown())

    result = _run_analysis(ast)

    live_in_selection = result.get_live_input_identifiers(selection)
    assert {a, b, c}.issubset(live_in_selection)
    assert result.get_live_input_identifiers(true_stmt) == frozenset({a})
    assert result.get_live_input_identifiers(false_stmt) == frozenset({b})


def test_live_out_of_for_all_body_includes_loop_carried_uses(
    forall_vector_sum_module_ast,
):
    """Test a ForAll body must see loop-carried values in live_out."""

    program_ast, a, b, i, acc, N = forall_vector_sum_module_ast

    result = _run_analysis(program_ast)

    procedure = program_ast.statements[0]
    forall = procedure.body[3]
    body_assign = forall.body[0]

    assert acc in result.get_live_output_identifiers(body_assign)
    assert acc in result.get_live_input_identifiers(body_assign)
