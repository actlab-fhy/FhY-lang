"""Tests for the FhY control flow graph construction."""

from fhy_core import (
    Identifier,
    IndexType,
    Provenance,
    TypeQualifier,
    parse_expression,
)

from fhy_lang.lang.ast import (
    DeclarationStatement,
    ForAllStatement,
    IdentifierExpression,
    Operation,
    QualifiedType,
    ReturnStatement,
    SelectionStatement,
)
from fhy_lang.lang.ast.cfg import (
    CFGEdgeKind,
    CFGNodeKind,
    build_cfg,
)

from .utils import (
    make_argument,
    make_identifier_assignment,
    make_procedure,
    make_uninitialized_temp_declaration,
)


def test_entry_and_exit_are_always_present():
    """Test that the CFG has an ENTRY and EXIT node."""
    procedure_ast = make_procedure(name=Identifier("main"), args=(), body=())

    cfg = build_cfg(procedure_ast)

    assert cfg.entry.kind == CFGNodeKind.ENTRY
    assert cfg.exit.kind == CFGNodeKind.EXIT
    # An empty body connects ENTRY directly to EXIT.
    assert cfg.get_successors(cfg.entry) == (cfg.exit,)


def test_straight_line_body_has_sequential_edges(int32):
    """Test that the CFG has sequential edges for a straight-line body."""
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

    cfg = build_cfg(procedure_ast)

    # ENTRY -> decl -> assign(x,a) -> assign(b,x) -> EXIT
    statement_nodes = [n for n in cfg.nodes if n.kind == CFGNodeKind.STATEMENT]
    assert len(statement_nodes) == 3
    assert cfg.get_successors(cfg.entry) == (statement_nodes[0],)
    assert cfg.get_successors(statement_nodes[0]) == (statement_nodes[1],)
    assert cfg.get_successors(statement_nodes[1]) == (statement_nodes[2],)
    assert cfg.get_successors(statement_nodes[2]) == (cfg.exit,)
    # All edges in straight-line code are UNCONDITIONAL.
    for node in cfg.nodes:
        for _, kind in cfg.get_outgoing_edges(node):
            assert kind == CFGEdgeKind.UNCONDITIONAL


def test_for_all_statement_has_loop_back_and_loop_exit_edges(int32):
    """Test that the CFG has loop back and loop exit edges for a forall statement."""
    i, a, b = Identifier("i"), Identifier("a"), Identifier("b")
    index_declaration_ast = DeclarationStatement(
        variable_name=i,
        variable_type=QualifiedType(
            base_type=IndexType(
                parse_expression("1"), parse_expression("n"), stride=None
            ),
            type_qualifier=TypeQualifier.TEMP,
            provenance=Provenance.unknown(),
        ),
        provenance=Provenance.unknown(),
    )
    forall_ast = ForAllStatement(
        index=IdentifierExpression(identifier=i, provenance=Provenance.unknown()),
        body=(make_identifier_assignment(b, a),),
        provenance=Provenance.unknown(),
    )
    procedure_ast = make_procedure(
        name=Identifier("main"),
        args=(
            make_argument(a, TypeQualifier.INPUT, int32),
            make_argument(b, TypeQualifier.OUTPUT, int32),
        ),
        body=(index_declaration_ast, forall_ast),
    )

    cfg = build_cfg(procedure_ast)

    forall_nodes = [
        n
        for n in cfg.nodes
        if n.kind == CFGNodeKind.STATEMENT and isinstance(n.statement, ForAllStatement)
    ]
    assert len(forall_nodes) == 1
    header = forall_nodes[0]

    # The forall header branches to the body via LOOP_BODY.
    outgoing_kinds = {kind for _, kind in cfg.get_outgoing_edges(header)}
    assert CFGEdgeKind.LOOP_BODY in outgoing_kinds

    # The body's tail closes the loop with a LOOP_BACK edge to the header
    # and leaves the loop with a LOOP_EXIT edge. FhY forall is assumed to
    # execute at least once, so the LOOP_EXIT edge originates at the body
    # tail rather than the header — this keeps the body's must-definitions
    # flowing to whatever follows the loop.
    incoming_kinds = {kind for _, kind in cfg.get_incoming_edges(header)}
    assert CFGEdgeKind.LOOP_BACK in incoming_kinds

    exit_preds = cfg.get_incoming_edges(cfg.exit)
    assert any(kind == CFGEdgeKind.LOOP_EXIT for _, kind in exit_preds)


def test_return_statement_terminates_control_flow(int32):
    """Test that the CFG has a single exit edge for a return statement."""
    a, r = Identifier("a"), Identifier("r")
    operation_ast = Operation(
        name=Identifier("f"),
        args=(make_argument(a, TypeQualifier.INPUT, int32),),
        body=(
            make_uninitialized_temp_declaration(r, int32),
            make_identifier_assignment(r, a),
            ReturnStatement(
                expression=IdentifierExpression(
                    identifier=r, provenance=Provenance.unknown()
                ),
                provenance=Provenance.unknown(),
            ),
            # An unreachable tail; the builder must skip it.
            make_identifier_assignment(r, a),
        ),
        return_type=QualifiedType(
            base_type=int32,
            type_qualifier=TypeQualifier.OUTPUT,
            provenance=Provenance.unknown(),
        ),
        provenance=Provenance.unknown(),
    )

    cfg = build_cfg(operation_ast)

    return_nodes = [
        n
        for n in cfg.nodes
        if n.kind == CFGNodeKind.STATEMENT and isinstance(n.statement, ReturnStatement)
    ]
    assert len(return_nodes) == 1
    return_node = return_nodes[0]

    # Return has exactly one successor: the EXIT node.
    assert cfg.get_successors(return_node) == (cfg.exit,)

    # The unreachable statement following `return` should not be in the CFG.
    statement_nodes = [n for n in cfg.nodes if n.kind == CFGNodeKind.STATEMENT]
    # decl, assign, return — 3 statement nodes; the 4th (unreachable) is skipped.
    assert len(statement_nodes) == 3


def test_selection_with_empty_branches_preserves_both_edge_kinds(int32):
    """Test that the CFG preserves both TRUE and FALSE edges
    for a selection statement with empty branches."""
    a, c = Identifier("a"), Identifier("c")
    operation_ast = Operation(
        name=Identifier("f"),
        args=(
            make_argument(a, TypeQualifier.INPUT, int32),
            make_argument(c, TypeQualifier.INPUT, int32),
        ),
        return_type=QualifiedType(
            base_type=int32,
            type_qualifier=TypeQualifier.OUTPUT,
            provenance=Provenance.unknown(),
        ),
        body=(
            SelectionStatement(
                condition=IdentifierExpression(
                    identifier=c, provenance=Provenance.unknown()
                ),
                true_body=(),
                false_body=(),
                provenance=Provenance.unknown(),
            ),
            ReturnStatement(
                expression=IdentifierExpression(
                    identifier=a, provenance=Provenance.unknown()
                ),
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )

    cfg = build_cfg(operation_ast)

    selection_nodes = [
        n
        for n in cfg.nodes
        if n.kind == CFGNodeKind.STATEMENT
        and isinstance(n.statement, SelectionStatement)
    ]
    assert len(selection_nodes) == 1
    condition = selection_nodes[0]

    merge_nodes = [n for n in cfg.nodes if n.kind == CFGNodeKind.MERGE]
    assert len(merge_nodes) == 1
    merge = merge_nodes[0]

    outgoing = cfg.get_outgoing_edges(condition)
    edge_kinds = tuple(kind for target, kind in outgoing if target is merge)
    # Both TRUE and FALSE edges should be preserved as parallel edges.
    assert CFGEdgeKind.TRUE in edge_kinds
    assert CFGEdgeKind.FALSE in edge_kinds


def test_node_ids_are_unique_and_registered(int32):
    """Test that the CFG has unique node IDs and registers the ENTRY and EXIT nodes."""
    a, b = Identifier("a"), Identifier("b")
    procedure_ast = make_procedure(
        name=Identifier("main"),
        args=(
            make_argument(a, TypeQualifier.INPUT, int32),
            make_argument(b, TypeQualifier.OUTPUT, int32),
        ),
        body=(make_identifier_assignment(b, a),),
    )

    cfg = build_cfg(procedure_ast)

    ids = [n.id for n in cfg.nodes]
    assert len(set(ids)) == len(ids)
    assert cfg.entry in cfg.nodes
    assert cfg.exit in cfg.nodes
