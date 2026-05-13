"""Validate the entry-point requirement: every module declares a ``proc main``.

Every FhY program must declare exactly one ``Procedure`` named ``main``.
The check is name-based on ``Identifier.name_hint`` (not identity), so
any procedure whose source name is ``main`` counts, regardless of which
``Identifier`` object the parser produced. The pass runs early in the
structural pipeline so a missing or duplicated entry point fails
compilation before downstream passes are attempted.

"""

__all__ = [
    "MainProcedureValidator",
]

import logging

from fhy_core import (
    AnalysisVisitablePass,
    DiagnosticLevel,
    get_logger,
    register_pass,
)

from fhy_lang.ast.node import Module, Node, Procedure

from .utils import format_diagnostic_message

_logger: logging.Logger = get_logger(__name__)

_MAIN_PROCEDURE_NAME = "main"
_MAX_MAIN_PROCEDURES = 1


@register_pass(
    "fhy_ast_main_procedure_validator",
    "Validates that the module declares exactly one 'proc main' entry point.",
)
class MainProcedureValidator(AnalysisVisitablePass[Node]):
    """Validate that the module declares exactly one ``proc main``.

    Counts the number of :class:`Procedure` nodes in
    ``module.statements`` whose ``name.name_hint`` equals ``"main"``.
    Reports an ``ERROR`` if the count is ``0`` (no entry point) or
    ``>= 2`` (ambiguous entry point).

    ``op`` declarations named ``main`` do not count: the language
    reference reserves the entry-point role for a :class:`Procedure`.

    """

    def visit_module(self, node: Module) -> None:
        mains = [
            statement
            for statement in node.statements
            if isinstance(statement, Procedure)
            and statement.name.name_hint == _MAIN_PROCEDURE_NAME
        ]
        _logger.debug(
            "Main-procedure check: module %s has %d 'proc main' declaration(s).",
            node.name,
            len(mains),
        )
        if len(mains) == 0:
            self.report(
                DiagnosticLevel.ERROR,
                format_diagnostic_message(
                    "module structure error",
                    "every program must declare a 'proc main' as its entry point.",
                    node.provenance,
                ),
            )
        elif len(mains) > _MAX_MAIN_PROCEDURES:
            self.report(
                DiagnosticLevel.ERROR,
                format_diagnostic_message(
                    "module structure error",
                    f"a program may declare at most one 'proc main'; found "
                    f"{len(mains)}.",
                    node.provenance,
                ),
            )
