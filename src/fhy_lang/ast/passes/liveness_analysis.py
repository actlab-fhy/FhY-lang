"""Liveness analysis over the FhY AST."""

__all__ = [
    "LivenessAnalysis",
    "LivenessResult",
]

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

from fhy_core import (
    Analysis,
    AnalysisVisitablePass,
    FrozenMixin,
    Identifier,
    TypeQualifier,
    get_logger,
    register_pass,
)
from frozendict import frozendict

from fhy_lang.ast.node import (
    DeclarationStatement,
    Expression,
    ExpressionStatement,
    ForAllStatement,
    IdentifierExpression,
    Module,
    Node,
    Operation,
    Procedure,
    ReturnStatement,
    SelectionStatement,
    Statement,
)

from .identifier_collector import collect_identifiers

_logger: logging.Logger = get_logger(__name__)


@dataclass(frozen=True)
class LivenessResult(FrozenMixin):
    """Per-statement liveness information for a FhY AST module.

    Sets are keyed by the Python `id()` of each statement node so results are
    tied to the analyzed immutable AST instance.

    """

    live_in: frozendict[int, frozenset[Identifier]] = field(default_factory=frozendict)
    live_out: frozendict[int, frozenset[Identifier]] = field(default_factory=frozendict)

    def get_live_input_identifiers(self, statement: Statement) -> frozenset[Identifier]:
        """Return the set of identifiers live on entry to the statement."""
        return self.live_in.get(id(statement), frozenset())

    def get_live_output_identifiers(
        self, statement: Statement
    ) -> frozenset[Identifier]:
        """Return the set of identifiers live on exit from the statement."""
        return self.live_out.get(id(statement), frozenset())


def _collect_uses(expression: Expression | None) -> frozenset[Identifier]:
    if expression is None:
        return frozenset()
    return collect_identifiers(expression)


def _get_simple_assignment_target(expression: Expression) -> Identifier | None:
    """Return the identifier killed by a simple assignment LHS, if any.

    A plain identifier target fully defines the variable. Array-access targets
    only update selected elements, so the variable remains live and is not
    returned here.

    """
    if isinstance(expression, IdentifierExpression):
        return expression.identifier
    return None


def _get_expression_statement_live_input(
    statement: ExpressionStatement, live_out: frozenset[Identifier]
) -> frozenset[Identifier]:
    uses = _collect_uses(statement.right)
    defs: frozenset[Identifier] = frozenset()
    if statement.left is not None:
        target = _get_simple_assignment_target(statement.left)
        if target is not None:
            defs = frozenset({target})
        else:
            uses = uses | _collect_uses(statement.left)
    return (live_out - defs) | uses


def _get_selection_statement_live_input(
    statement: SelectionStatement,
    live_out: frozenset[Identifier],
    live_in_map: dict[int, frozenset[Identifier]],
    live_out_map: dict[int, frozenset[Identifier]],
) -> frozenset[Identifier]:
    condition_uses = _collect_uses(statement.condition)
    true_in = _analyze_block(statement.true_body, live_out, live_in_map, live_out_map)
    false_in = _analyze_block(statement.false_body, live_out, live_in_map, live_out_map)
    return condition_uses | true_in | false_in


def _get_for_all_statement_live_input(
    statement: ForAllStatement,
    live_out: frozenset[Identifier],
    live_in_map: dict[int, frozenset[Identifier]],
    live_out_map: dict[int, frozenset[Identifier]],
) -> frozenset[Identifier]:
    index_uses = _collect_uses(statement.index)
    body_in: frozenset[Identifier] = frozenset()
    while True:
        body_out = live_out | body_in
        new_body_in = _analyze_block(
            statement.body, body_out, live_in_map, live_out_map
        )
        if new_body_in == body_in:
            break
        body_in = new_body_in
    return index_uses | body_in


def _get_statement_live_input_identifiers(
    statement: Statement,
    live_out: frozenset[Identifier],
    live_in_map: dict[int, frozenset[Identifier]],
    live_out_map: dict[int, frozenset[Identifier]],
) -> frozenset[Identifier]:
    if isinstance(statement, DeclarationStatement):
        uses = _collect_uses(statement.expression)
        defs = frozenset({statement.variable_name})
        return (live_out - defs) | uses
    elif isinstance(statement, ExpressionStatement):
        return _get_expression_statement_live_input(statement, live_out)
    elif isinstance(statement, ReturnStatement):
        return _collect_uses(statement.expression)
    elif isinstance(statement, SelectionStatement):
        return _get_selection_statement_live_input(
            statement, live_out, live_in_map, live_out_map
        )
    elif isinstance(statement, ForAllStatement):
        return _get_for_all_statement_live_input(
            statement, live_out, live_in_map, live_out_map
        )
    else:
        return live_out


def _analyze_block(
    statements: Sequence[Statement],
    exit_live: frozenset[Identifier],
    live_in_map: dict[int, frozenset[Identifier]],
    live_out_map: dict[int, frozenset[Identifier]],
) -> frozenset[Identifier]:
    current_out = exit_live
    for statement in reversed(list(statements)):
        live_out_map[id(statement)] = current_out
        stmt_in = _get_statement_live_input_identifiers(
            statement, current_out, live_in_map, live_out_map
        )
        live_in_map[id(statement)] = stmt_in
        current_out = stmt_in
    return current_out


def _get_procedure_exit_live_output_identifiers(
    procedure: Procedure,
) -> frozenset[Identifier]:
    return frozenset(
        arg.name
        for arg in procedure.args
        if arg.qualified_type.type_qualifier == TypeQualifier.OUTPUT
    )


@register_pass(
    "fhy_ast_liveness_analysis",
    "Computes per-statement live-in/live-out sets for the FhY AST.",
)
class _LivenessAnalysisPass(AnalysisVisitablePass[Node]):
    """Walk a FhY module and compute liveness for each function body."""

    _live_in: dict[int, frozenset[Identifier]]
    _live_out: dict[int, frozenset[Identifier]]

    def __init__(self) -> None:
        super().__init__()
        self._live_in = {}
        self._live_out = {}

    @property
    def result(self) -> LivenessResult:
        return LivenessResult(
            live_in=frozendict(self._live_in),
            live_out=frozendict(self._live_out),
        )

    def visit_procedure(self, node: Procedure) -> None:
        _logger.debug("Liveness: analyzing procedure %s.", node.name)
        _analyze_block(
            node.body,
            _get_procedure_exit_live_output_identifiers(node),
            self._live_in,
            self._live_out,
        )

    def visit_operation(self, node: Operation) -> None:
        _logger.debug("Liveness: analyzing operation %s.", node.name)
        _analyze_block(node.body, frozenset(), self._live_in, self._live_out)

    def get_visit_children(self, node: Node) -> Sequence[Node]:
        if isinstance(node, Procedure | Operation):
            return ()
        return super().get_visit_children(node)


class LivenessAnalysis(Analysis[Module, LivenessResult]):
    """Cached liveness analysis for a FhY AST module."""

    def run(self, ir: Module) -> LivenessResult:
        _logger.info("Starting liveness analysis...")
        walker = _LivenessAnalysisPass()
        walker(ir)
        result = walker.result
        _logger.info(
            "Liveness analysis complete: %d statement(s) analyzed.",
            len(result.live_in),
        )
        return result
