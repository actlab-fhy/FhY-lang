"""Tests the liveness analysis pass."""

from fhy_core import (
    Identifier,
    Provenance,
    TypeQualifier,
)
from fhy_lang.lang.ast import (
    IdentifierExpression,
    Module,
    SelectionStatement,
)
from fhy_lang.lang.ast.passes import (
    LivenessAnalysis,
    LivenessResult,
)

from .utils import (
    make_argument,
    make_identifier_assignment,
    make_module_with_statement,
    make_procedure,
    make_uninitialized_temp_declaration,
)


def _run_liveness_analysis(ast: Module) -> LivenessResult:
    return LivenessAnalysis().run(ast)


def test_liveness_result_default_is_empty_sets():
    """Test a statement with no recorded liveness returns empty sets."""
    procedure_ast = make_procedure(name=Identifier("main"), args=(), body=())
    program_ast = make_module_with_statement(procedure_ast)

    result = _run_liveness_analysis(program_ast)

    assert result.get_live_input_identifiers(procedure_ast) == frozenset()
    assert result.get_live_output_identifiers(procedure_ast) == frozenset()


def test_dead_assignment_is_not_live_out(int32):
    """Test an assignment to a TEMP whose value is overwritten is not live out."""
    a, b, x = Identifier("a"), Identifier("b"), Identifier("x")
    declaration_ast = make_uninitialized_temp_declaration(x, int32)
    first_x_assignment_ast = make_identifier_assignment(x, a)
    second_x_assignment_ast = make_identifier_assignment(x, a)
    b_assignment_ast = make_identifier_assignment(b, x)
    procedure_ast = make_procedure(
        name=Identifier("main"),
        args=(
            make_argument(a, TypeQualifier.INPUT, int32),
            make_argument(b, TypeQualifier.OUTPUT, int32),
        ),
        body=(
            declaration_ast,
            first_x_assignment_ast,
            second_x_assignment_ast,
            b_assignment_ast,
        ),
    )
    program_ast = make_module_with_statement(procedure_ast)

    result = _run_liveness_analysis(program_ast)

    assert x not in result.get_live_output_identifiers(first_x_assignment_ast)
    assert x in result.get_live_output_identifiers(second_x_assignment_ast)
    assert result.get_live_output_identifiers(b_assignment_ast) == frozenset({b})
    assert a in result.get_live_input_identifiers(first_x_assignment_ast)


def test_selection_statement_unions_branch_liveness(int32):
    """Test selection statement live_in should union condition and branch uses."""
    a, b, c, y = Identifier("a"), Identifier("b"), Identifier("c"), Identifier("y")
    true_branch_assignment_ast = make_identifier_assignment(y, a)
    false_branch_assignment_ast = make_identifier_assignment(y, b)
    selection_ast = SelectionStatement(
        condition=IdentifierExpression(identifier=c, provenance=Provenance.unknown()),
        true_body=(true_branch_assignment_ast,),
        false_body=(false_branch_assignment_ast,),
        provenance=Provenance.unknown(),
    )
    procedure_ast = make_procedure(
        name=Identifier("main"),
        args=(
            make_argument(a, TypeQualifier.INPUT, int32),
            make_argument(b, TypeQualifier.INPUT, int32),
            make_argument(c, TypeQualifier.INPUT, int32),
            make_argument(y, TypeQualifier.OUTPUT, int32),
        ),
        body=(selection_ast,),
    )
    program_ast = make_module_with_statement(procedure_ast)

    result = _run_liveness_analysis(program_ast)

    live_in_of_selection = result.get_live_input_identifiers(selection_ast)
    assert {a, b, c}.issubset(live_in_of_selection)
    assert result.get_live_input_identifiers(true_branch_assignment_ast) == frozenset(
        {a}
    )
    assert result.get_live_input_identifiers(false_branch_assignment_ast) == frozenset(
        {b}
    )


def test_live_out_of_for_all_body_includes_loop_carried_uses(
    forall_vector_sum_module_ast,
):
    """Test a ForAll body must see loop-carried values in live_out."""
    program_ast, a, b, i, acc, N = forall_vector_sum_module_ast

    result = _run_liveness_analysis(program_ast)

    procedure_ast = program_ast.statements[0]
    forall_ast = procedure_ast.body[3]
    body_assignment_ast = forall_ast.body[0]

    assert acc in result.get_live_output_identifiers(body_assignment_ast)
    assert acc in result.get_live_input_identifiers(body_assignment_ast)
