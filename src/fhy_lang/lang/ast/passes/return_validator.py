"""Validate return-statement placement in operations and procedures.

Two rules, symmetric across FhY's two function kinds:

- An :class:`Operation` returns a value. Every control-flow path through
  its body must reach a :class:`ReturnStatement`. This is stronger than
  "the body contains a return somewhere" — an operation whose ``return``
  lives only inside one branch of a selection (or whose return is
  unreachable after an earlier fall-through) is ill-formed.

- A :class:`Procedure` does not return a value. Its body must therefore
  contain no reachable :class:`ReturnStatement`.

Both checks consult the function's CFG instead of walking the AST
directly. That keeps the validator future-proof: whenever the CFG
builder learns about a new control-flow construct (e.g. break / early
exit) both "every path reaches a return" and "no return reachable" stay
correct without changes here.

"""

__all__ = [
    "ReturnValidator",
]

import logging

from fhy_core import (
    AnalysisVisitablePass,
    DiagnosticLevel,
    get_logger,
    register_pass,
)

from fhy_lang.lang.ast.cfg import CFGNodeKind, build_cfg
from fhy_lang.lang.ast.node import (
    Node,
    Operation,
    Procedure,
    ReturnStatement,
)

from .utils import format_diagnostic_message

_logger: logging.Logger = get_logger(__name__)


@register_pass(
    "fhy_ast_return_validator",
    "Validates return-statement placement in operations and procedures.",
)
class ReturnValidator(AnalysisVisitablePass[Node]):
    """Validate return-statement placement for operations and procedures.

    Operations: every predecessor of the CFG exit must be a ``STATEMENT``
    node whose statement is a :class:`ReturnStatement`. Fall-through
    paths, empty bodies, and branches that miss a return are all caught
    because the CFG builder emits a non-return predecessor of the exit
    in exactly those cases.

    Procedures: every CFG node of kind ``STATEMENT`` whose statement is
    a :class:`ReturnStatement` is reported as an error, with the return
    statement's own provenance. This relies on the CFG: unreachable
    returns (e.g. dead code after an earlier return) are not emitted
    into the CFG and therefore not reported here — they're the
    responsibility of a separate unreachable-statement check.

    """

    def visit_operation(self, node: Operation) -> None:
        cfg = build_cfg(node)
        predecessors = cfg.get_predecessors(cfg.exit)
        all_predecessors_are_returns = all(
            predecessor.kind == CFGNodeKind.STATEMENT
            and isinstance(predecessor.statement, ReturnStatement)
            for predecessor in predecessors
        )
        _logger.debug(
            "Return check: operation %s has %d exit predecessor(s); " "all-return=%s.",
            node.name,
            len(predecessors),
            all_predecessors_are_returns,
        )
        if all_predecessors_are_returns:
            return
        self.report(
            DiagnosticLevel.ERROR,
            format_diagnostic_message(
                "structural error",
                f"Operation {node.name.name_hint!r} does not return on every "
                "control-flow path; every path through its body must reach a "
                "return statement.",
                node.provenance,
            ),
        )

    def visit_procedure(self, node: Procedure) -> None:
        cfg = build_cfg(node)
        _logger.debug("Return check: scanning procedure %s for returns.", node.name)
        for cfg_node in cfg.iter_nodes():
            if cfg_node.kind != CFGNodeKind.STATEMENT:
                continue
            if not isinstance(cfg_node.statement, ReturnStatement):
                continue
            self.report(
                DiagnosticLevel.ERROR,
                format_diagnostic_message(
                    "structural error",
                    f"Procedure {node.name.name_hint!r} contains a return "
                    "statement; procedures must not return a value.",
                    cfg_node.statement.provenance,
                ),
            )
