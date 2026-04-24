"""Construct a control flow graph from a FhY function."""

__all__ = [
    "build_cfg",
]

from collections.abc import Sequence

import networkx as nx

from fhy_lang.lang.ast.node import (
    ForAllStatement,
    Operation,
    Procedure,
    ReturnStatement,
    SelectionStatement,
    Statement,
)

from .graph import CFGEdgeKind, CFGNode, CFGNodeKind, ControlFlowGraph

_FunctionDefinition = Procedure | Operation

# A pending predecessor: (node, edge_kind_from_it). The next node added to the
# CFG draws an edge of `edge_kind_from_it` from `node`. `None` means control
# flow cannot reach here (e.g., after a `ReturnStatement`).
_Tail = tuple[CFGNode, CFGEdgeKind] | None


class _CFGBuilder:
    """Recursive-descent CFG builder for a FhY function body."""

    _function: _FunctionDefinition
    _next_id: int
    _graph: "nx.MultiDiGraph[int]"
    _entry: CFGNode
    _exit: CFGNode

    def __init__(self, function: _FunctionDefinition) -> None:
        self._function = function
        self._next_id = 0
        self._graph = nx.MultiDiGraph()
        self._entry = self._new_node(CFGNodeKind.ENTRY)
        self._exit = self._new_node(CFGNodeKind.EXIT)

    def build(self) -> ControlFlowGraph:
        tail: _Tail = (self._entry, CFGEdgeKind.UNCONDITIONAL)
        tail = self._build_block(self._function.body, tail)
        if tail is not None:
            pred, edge_kind = tail
            self._add_edge(pred, self._exit, edge_kind)
        return ControlFlowGraph(
            function_name=self._function.name,
            graph=self._graph,
            entry=self._entry,
            exit=self._exit,
        )

    def _new_node(
        self, kind: CFGNodeKind, statement: Statement | None = None
    ) -> CFGNode:
        node = CFGNode(id=self._next_id, kind=kind, statement=statement)
        self._graph.add_node(node.id, node=node)
        self._next_id += 1
        return node

    def _add_edge(self, source: CFGNode, target: CFGNode, kind: CFGEdgeKind) -> None:
        self._graph.add_edge(source.id, target.id, kind=kind)

    def _connect(self, tail: _Tail, target: CFGNode) -> None:
        if tail is None:
            return
        pred, edge_kind = tail
        self._add_edge(pred, target, edge_kind)

    def _build_block(self, body: Sequence[Statement], tail: _Tail) -> _Tail:
        current = tail
        for statement in body:
            if current is None:
                return None
            current = self._build_statement(statement, current)
        return current

    def _build_statement(self, statement: Statement, tail: _Tail) -> _Tail:
        if tail is None:
            return None

        if isinstance(statement, ReturnStatement):
            node = self._new_node(CFGNodeKind.STATEMENT, statement)
            self._connect(tail, node)
            self._add_edge(node, self._exit, CFGEdgeKind.UNCONDITIONAL)
            return None

        if isinstance(statement, SelectionStatement):
            condition = self._new_node(CFGNodeKind.STATEMENT, statement)
            self._connect(tail, condition)

            true_tail = self._build_block(
                statement.true_body,
                (condition, CFGEdgeKind.TRUE),
            )
            false_tail = self._build_block(
                statement.false_body,
                (condition, CFGEdgeKind.FALSE),
            )

            if true_tail is None and false_tail is None:
                return None

            merge = self._new_node(CFGNodeKind.MERGE)
            self._connect(true_tail, merge)
            self._connect(false_tail, merge)
            return (merge, CFGEdgeKind.UNCONDITIONAL)

        if isinstance(statement, ForAllStatement):
            header = self._new_node(CFGNodeKind.STATEMENT, statement)
            self._connect(tail, header)

            body_tail = self._build_block(
                statement.body,
                (header, CFGEdgeKind.LOOP_BODY),
            )
            if body_tail is None:
                # Body always returns; nothing after the loop is reachable.
                return None
            # The back-edge from the body tail to the header is always
            # labeled LOOP_BACK, irrespective of the pending edge kind the
            # body tail was going to emit into its (non-existent) successor.
            back_pred, _ = body_tail
            self._add_edge(back_pred, header, CFGEdgeKind.LOOP_BACK)
            # FhY forall is bounded and assumed to execute at least once, so
            # the LOOP_EXIT edge leaving the loop originates at the body
            # tail rather than the header. This preserves the body's
            # must-definitions on the path out of the loop — a header-sourced
            # LOOP_EXIT would model a zero-trip path that FhY's semantics
            # forbid and would render definite-assignment unsound for
            # writes inside the body.
            return (back_pred, CFGEdgeKind.LOOP_EXIT)

        node = self._new_node(CFGNodeKind.STATEMENT, statement)
        self._connect(tail, node)
        return (node, CFGEdgeKind.UNCONDITIONAL)


def build_cfg(function: _FunctionDefinition) -> ControlFlowGraph:
    """Build a control flow graph for a FhY function.

    Args:
        function: The `Procedure` or `Operation` whose body to analyze.

    Returns:
        The constructed `ControlFlowGraph` with synthetic `ENTRY` / `EXIT`
        nodes wrapping `function.body`'s statements.

    """
    return _CFGBuilder(function).build()
