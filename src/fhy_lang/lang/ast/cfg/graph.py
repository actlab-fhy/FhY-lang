"""Control flow graph data structure for FhY functions.

The graph is backed by a `networkx.MultiDiGraph` held privately on
`ControlFlowGraph`. Immutability is enforced by encapsulation: the public
surface exposes only read-only accessors, and the backing graph is built
solely by :func:`build_cfg` during construction.

"""

__all__ = [
    "CFGEdgeKind",
    "CFGNode",
    "CFGNodeKind",
    "ControlFlowGraph",
]

from collections.abc import Iterator
from dataclasses import dataclass
from typing import cast

import networkx as nx
from fhy_core import Identifier, StrEnum

from fhy_lang.lang.ast.node import Statement


class CFGNodeKind(StrEnum):
    """Kinds of nodes that can appear in a FhY control flow graph."""

    ENTRY = "entry"
    EXIT = "exit"
    STATEMENT = "statement"
    MERGE = "merge"


class CFGEdgeKind(StrEnum):
    """Kinds of edges between CFG nodes."""

    UNCONDITIONAL = "unconditional"
    TRUE = "true"
    FALSE = "false"
    LOOP_BODY = "loop_body"
    LOOP_EXIT = "loop_exit"
    LOOP_BACK = "loop_back"


_NODE_ATTRIBUTE = "node"
_EDGE_KIND_ATTRIBUTE = "kind"


@dataclass(frozen=True)
class CFGNode:
    """A node in a FhY control flow graph.

    `STATEMENT` nodes wrap an AST `Statement`. A `SelectionStatement` or
    `ForAllStatement` becomes a single statement node representing its
    condition / loop header; each statement within its body is emitted as
    its own node. `MERGE` nodes are synthetic join points introduced after
    selection branches. `ENTRY` / `EXIT` mark the synthetic function entry
    and exit.

    """

    id: int
    kind: CFGNodeKind
    statement: Statement | None = None

    def __post_init__(self) -> None:
        if self.kind == CFGNodeKind.STATEMENT and self.statement is None:
            raise ValueError("CFGNode of kind STATEMENT must carry an AST statement.")
        if self.kind != CFGNodeKind.STATEMENT and self.statement is not None:
            raise ValueError(
                f"CFGNode of kind {self.kind.value} must not carry an AST statement."
            )


class ControlFlowGraph:
    """Control flow graph for a single FhY function.

    Constructed exclusively by :func:`build_cfg`. The underlying
    `networkx.MultiDiGraph` is private: the class exposes only read-only
    accessors, so callers cannot mutate the graph after construction.

    """

    _function_name: Identifier
    _graph: "nx.MultiDiGraph[int]"
    _entry: CFGNode
    _exit: CFGNode

    def __init__(
        self,
        *,
        function_name: Identifier,
        graph: "nx.MultiDiGraph[int]",
        entry: CFGNode,
        exit: CFGNode,
    ) -> None:
        self._function_name = function_name
        self._graph = graph
        self._entry = entry
        self._exit = exit

    @property
    def function_name(self) -> Identifier:
        return self._function_name

    @property
    def entry(self) -> CFGNode:
        return self._entry

    @property
    def exit(self) -> CFGNode:
        return self._exit

    @property
    def nodes(self) -> tuple[CFGNode, ...]:
        """Nodes in construction order."""
        return tuple(
            self._graph.nodes[node_id][_NODE_ATTRIBUTE] for node_id in self._graph.nodes
        )

    def iter_nodes(self) -> Iterator[CFGNode]:
        """Iterate over all nodes in construction order."""
        for node_id in self._graph.nodes:
            yield self._graph.nodes[node_id][_NODE_ATTRIBUTE]

    def get_successors(self, node: CFGNode) -> tuple[CFGNode, ...]:
        """Return the successor nodes of `node` (unique, insertion-ordered)."""
        return tuple(
            self._graph.nodes[successor_id][_NODE_ATTRIBUTE]
            for successor_id in self._graph.successors(node.id)
        )

    def get_predecessors(self, node: CFGNode) -> tuple[CFGNode, ...]:
        """Return the predecessor nodes of `node` (unique, insertion-ordered)."""
        return tuple(
            self._graph.nodes[predecessor_id][_NODE_ATTRIBUTE]
            for predecessor_id in self._graph.predecessors(node.id)
        )

    def get_outgoing_edges(
        self, node: CFGNode
    ) -> tuple[tuple[CFGNode, CFGEdgeKind], ...]:
        """Return outgoing edges `(target, kind)` for `node`.

        Parallel edges between the same pair of nodes (e.g. a selection
        whose TRUE and FALSE branches are both empty) are each returned
        separately.

        """
        edges = cast(
            "Iterator[tuple[int, int, CFGEdgeKind]]",
            self._graph.out_edges(node.id, data=_EDGE_KIND_ATTRIBUTE),
        )
        return tuple(
            (self._graph.nodes[target_id][_NODE_ATTRIBUTE], kind)
            for _, target_id, kind in edges
        )

    def get_incoming_edges(
        self, node: CFGNode
    ) -> tuple[tuple[CFGNode, CFGEdgeKind], ...]:
        """Return incoming edges `(source, kind)` for `node`."""
        edges = cast(
            "Iterator[tuple[int, int, CFGEdgeKind]]",
            self._graph.in_edges(node.id, data=_EDGE_KIND_ATTRIBUTE),
        )
        return tuple(
            (self._graph.nodes[source_id][_NODE_ATTRIBUTE], kind)
            for source_id, _, kind in edges
        )
