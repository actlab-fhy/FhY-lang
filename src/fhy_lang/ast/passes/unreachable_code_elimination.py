"""Unreachable code elimination over the FhY AST.

Removes statements that the control flow graph proves cannot be reached.
Concretely this covers statements following a :class:`ReturnStatement`
in a function body, including trailing statements inside a branch of a
``SelectionStatement`` whose both branches unconditionally return, and
statements following a ``forall`` whose body unconditionally returns.

The CFG builder already stops emitting nodes once control flow cannot
fall through (see :mod:`fhy_lang.ast.cfg.builder`), so a statement whose
identity is not referenced by any CFG node is necessarily unreachable.
This pass collects those ids up-front, then walks the AST via the
:class:`Transformer` and drops every statement that is not in the
reachable set.

"""

__all__ = [
    "UnreachableCodeEliminationPass",
]

import logging

from fhy_core import get_logger, register_pass

from fhy_lang.ast.cfg import build_cfg
from fhy_lang.ast.node import (
    Module,
    Node,
    Operation,
    Procedure,
    Statement,
)

from .transformer import Statements, Transformer

_logger: logging.Logger = get_logger(__name__)


def _collect_reachable_statement_ids(module: Module) -> frozenset[int]:
    """Return ``id()`` of every Statement referenced by any function CFG.

    Top-level module statements (``Procedure``, ``Operation``, ``Import``,
    ``Native``) are always considered reachable; they are not part of any
    function body and therefore do not appear as nodes in any CFG.

    """
    reachable: set[int] = set()
    for top_level_statement in module.statements:
        reachable.add(id(top_level_statement))
        if isinstance(top_level_statement, Procedure | Operation):
            cfg = build_cfg(top_level_statement)
            for cfg_node in cfg.nodes:
                if cfg_node.statement is not None:
                    reachable.add(id(cfg_node.statement))
    return frozenset(reachable)


@register_pass(
    "fhy_ast_unreachable_code_elimination",
    "Removes AST statements that the CFG proves to be unreachable.",
)
class UnreachableCodeEliminationPass(Transformer):
    """Drop statements that cannot be reached in any execution path."""

    _reachable_statement_ids: frozenset[int]
    _removed_count: int

    def __init__(self) -> None:
        super().__init__()
        self._reachable_statement_ids = frozenset()
        self._removed_count = 0

    def run_pass(self, ir: Node) -> Node:
        if not isinstance(ir, Module):
            raise TypeError(
                f"{type(self).__name__} requires a Module input; "
                f"got {type(ir).__name__}."
            )
        _logger.info("Starting unreachable code elimination pass...")
        self._reachable_statement_ids = _collect_reachable_statement_ids(ir)
        self._removed_count = 0
        _logger.debug(
            "Reachable statement set contains %d statement(s).",
            len(self._reachable_statement_ids),
        )
        result = super().run_pass(ir)
        _logger.info(
            "Unreachable code elimination complete: %d statement(s) removed.",
            self._removed_count,
        )
        return result

    def did_change(self, input_ir: Node, output: Node) -> bool:
        _ = (input_ir, output)
        return self._removed_count > 0

    def visit_statement(self, node: Statement) -> Statements:
        if id(node) not in self._reachable_statement_ids:
            self._removed_count += 1
            _logger.debug("Removed unreachable %s statement.", type(node).__name__)
            return []
        return super().visit_statement(node)
