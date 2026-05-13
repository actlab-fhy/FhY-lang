"""Validate the kinds of statement that may appear at module scope.

A FhY module may only contain procedures, operations, and ``param``
declarations of scalar numerical types with compile-time-constant
initializers. Anything else (non-``param`` declarations, ``param``
declarations of tensor or tuple type, ``param`` declarations without
an initializer, ``forall`` statements, assignments, bare expressions,
``return`` statements) is a module-structure error.

The pass runs first in the structural validation pipeline so the
downstream passes can assume the module's top-level invariants hold.
Each violation is reported with its own diagnostic so multiple
issues can be surfaced in one compilation.

"""

__all__ = [
    "ModuleScopeValidator",
]

import logging

from fhy_core import (
    AnalysisVisitablePass,
    DiagnosticLevel,
    NumericalType,
    TypeQualifier,
    get_logger,
    register_pass,
)

from fhy_lang.ast.node import (
    DeclarationStatement,
    ExpressionStatement,
    ForAllStatement,
    Module,
    Node,
    Operation,
    Procedure,
    ReturnStatement,
    Statement,
)
from fhy_lang.types import TupleType

from .utils import format_diagnostic_message

_logger: logging.Logger = get_logger(__name__)


@register_pass(
    "fhy_ast_module_scope_validator",
    "Validates which statements are allowed at module scope.",
)
class ModuleScopeValidator(AnalysisVisitablePass[Node]):
    """Validate that only allowed kinds of statement appear at module scope.

    Allowed:
        - :class:`Procedure`
        - :class:`Operation`
        - :class:`DeclarationStatement` with ``PARAM`` qualifier, a scalar
          numerical base type, and a non-``None`` initializer expression.

    Anything else is reported as a module-structure error.

    """

    def visit_module(self, node: Module) -> None:
        _logger.debug(
            "Module-scope check: walking %d top-level statement(s).",
            len(node.statements),
        )
        for statement in node.statements:
            self._check_statement(statement)

    def _check_statement(self, statement: Statement) -> None:
        if isinstance(statement, (Procedure, Operation)):
            return
        elif isinstance(statement, DeclarationStatement):
            self._check_declaration(statement)
            return
        elif isinstance(statement, ForAllStatement):
            self._report(
                "'forall' is not allowed at module scope.",
                statement,
            )
            return
        elif isinstance(statement, ExpressionStatement):
            if statement.left is None:
                self._report(
                    "bare expression statements are not allowed at module scope.",
                    statement,
                )
            else:
                self._report(
                    "assignment statements are not allowed at module scope.",
                    statement,
                )
            return
        elif isinstance(statement, ReturnStatement):
            self._report(
                "'return' is not allowed at module scope.",
                statement,
            )
            return
        else:
            # Defensive guard: any new Statement subclass added to the AST
            # without a corresponding rule here is rejected at module scope
            # by default. Future work that introduces a new top-level
            # construct must either allow-list it here or update this
            # validator's contract.
            self._report(
                f"statement of kind {type(statement).__name__!r} is not allowed "
                "at module scope.",
                statement,
            )

    def _check_declaration(self, statement: DeclarationStatement) -> None:
        qualifier = statement.variable_type.type_qualifier
        if qualifier != TypeQualifier.PARAM:
            self._report(
                "only 'param' declarations are allowed at module scope; "
                f"found type qualifier {qualifier.value!r}.",
                statement,
            )
            return
        base_type = statement.variable_type.base_type
        if isinstance(base_type, TupleType):
            self._report(
                "'param' declaration at module scope must be a scalar "
                "numerical type; got tuple type.",
                statement,
            )
            return
        if isinstance(base_type, NumericalType) and not base_type.is_scalar():
            self._report(
                "'param' declaration at module scope must be a scalar; got "
                f"tensor with shape {base_type.shape}.",
                statement,
            )
            return
        if statement.expression is None:
            self._report(
                "'param' declaration at module scope must have a "
                "compile-time-constant initializer.",
                statement,
            )

    def _report(self, message: str, statement: Statement) -> None:
        self.report(
            DiagnosticLevel.ERROR,
            format_diagnostic_message(
                "module structure error", message, statement.provenance
            ),
        )
