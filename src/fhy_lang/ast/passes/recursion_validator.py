"""Validate that no function in the AST is recursive.

FhY does not support recursive functions. A function is recursive when it
calls itself directly, or when it is part of a mutually-recursive cycle
reachable by following :class:`FunctionExpression` call edges through the
module's :class:`Procedure` and :class:`Operation` bodies.

"""

from __future__ import annotations

__all__ = [
    "RecursionValidator",
]

import logging

import networkx as nx
from fhy_core import (
    AnalysisVisitablePass,
    CompilerPass,
    DiagnosticLevel,
    Identifier,
    get_logger,
    register_pass,
)

from fhy_lang.ast.node import (
    FunctionExpression,
    IdentifierExpression,
    Module,
    Node,
    Operation,
    Procedure,
)

from .utils import format_diagnostic_message

_logger: logging.Logger = get_logger(__name__)

_FunctionDefinition = Procedure | Operation


def _get_function_definitions(module: Module) -> dict[Identifier, _FunctionDefinition]:
    function_definitions: dict[Identifier, _FunctionDefinition] = {}
    for statement in module.statements:
        if isinstance(statement, Procedure | Operation):
            function_definitions[statement.name] = statement
    return function_definitions


class _FunctionCallCollector(AnalysisVisitablePass[Node]):
    """Collect identifier-backed function calls rooted in a function body."""

    _called_identifiers: set[Identifier]

    def __init__(self) -> None:
        super().__init__()
        self._called_identifiers = set()

    @property
    def called_identifiers(self) -> frozenset[Identifier]:
        return frozenset(self._called_identifiers)

    def visit_function_expression(self, node: FunctionExpression) -> None:
        if isinstance(node.function, IdentifierExpression):
            self._called_identifiers.add(node.function.identifier)


def _collect_called_identifiers(
    function: _FunctionDefinition,
) -> frozenset[Identifier]:
    collector = _FunctionCallCollector()
    for statement in function.body:
        collector(statement)
    return collector.called_identifiers


def _build_call_graph(
    function_definitions: dict[Identifier, _FunctionDefinition],
) -> nx.DiGraph[Identifier]:
    """Build a directed call graph restricted to user-defined functions.

    Edges pointing to identifiers outside ``function_definitions`` (builtins,
    reductions, unresolved callees) are dropped because they cannot close a
    recursive cycle within the module.

    """
    graph: nx.DiGraph[Identifier] = nx.DiGraph()
    for caller_name in function_definitions:
        graph.add_node(caller_name)
    for caller_name, caller in function_definitions.items():
        for callee_name in _collect_called_identifiers(caller):
            if callee_name in function_definitions:
                graph.add_edge(caller_name, callee_name)
    return graph


def _find_recursive_components(
    graph: nx.DiGraph[Identifier],
) -> list[frozenset[Identifier]]:
    """Return the strongly-connected components that represent recursion.

    A component represents recursion when it has more than one member
    (mutual recursion) or when it has a single member with a self-loop
    (direct recursion).

    """
    recursive_components: list[frozenset[Identifier]] = []
    for component in nx.strongly_connected_components(graph):
        if len(component) > 1:
            recursive_components.append(frozenset(component))
        else:
            (only_member,) = component
            if graph.has_edge(only_member, only_member):
                recursive_components.append(frozenset(component))
    return recursive_components


def _format_cycle_description(
    graph: nx.DiGraph[Identifier], component: frozenset[Identifier]
) -> str:
    """Format a user-facing cycle description for a recursive component."""
    subgraph = graph.subgraph(component)
    try:
        cycle_edges = nx.find_cycle(subgraph)
    except nx.NetworkXNoCycle:
        # Every recursive component is required to contain a cycle, so this
        # branch is defensive: if `find_cycle` somehow fails we fall back to
        # a sorted enumeration of the component's members.
        sorted_members = sorted(component, key=lambda identifier: identifier.name_hint)
        return " -> ".join(member.name_hint for member in sorted_members)
    path = [edge[0] for edge in cycle_edges] + [cycle_edges[-1][1]]
    return " -> ".join(identifier.name_hint for identifier in path)


@register_pass(
    "fhy_ast_recursion_validator",
    "Validates that no function in the AST is recursive.",
)
class RecursionValidator(CompilerPass[Module, None]):
    """Flag every function that participates in a recursive call cycle.

    The validator builds a directed call graph over the module's
    :class:`Procedure` and :class:`Operation` definitions and reports each
    strongly-connected component that represents recursion. One ERROR
    diagnostic is emitted per recursive function so every offender surfaces
    in a single pass.

    """

    def get_noop_output(self, ir: Module) -> None:
        _ = ir

    def run_pass(self, ir: Module) -> None:
        function_definitions = _get_function_definitions(ir)
        if not function_definitions:
            _logger.debug("No user-defined functions; skipping recursion check.")
            return
        call_graph = _build_call_graph(function_definitions)
        _logger.debug(
            "Built call graph: %d node(s), %d edge(s).",
            call_graph.number_of_nodes(),
            call_graph.number_of_edges(),
        )
        recursive_components = _find_recursive_components(call_graph)
        for component in recursive_components:
            self._report_recursive_component(
                call_graph, function_definitions, component
            )
        _logger.info(
            "Recursion validation complete: %d recursive component(s) found.",
            len(recursive_components),
        )

    def _report_recursive_component(
        self,
        call_graph: nx.DiGraph[Identifier],
        function_definitions: dict[Identifier, _FunctionDefinition],
        component: frozenset[Identifier],
    ) -> None:
        cycle_description = _format_cycle_description(call_graph, component)
        for function_name in component:
            function = function_definitions[function_name]
            self._report_recursive_function(function, cycle_description)

    def _report_recursive_function(
        self, function: _FunctionDefinition, cycle_description: str
    ) -> None:
        self.report(
            DiagnosticLevel.ERROR,
            format_diagnostic_message(
                "semantic error",
                f"Function {function.name.name_hint!r} is recursive; "
                f"recursion is not allowed (cycle: {cycle_description}).",
                function.provenance,
            ),
        )
