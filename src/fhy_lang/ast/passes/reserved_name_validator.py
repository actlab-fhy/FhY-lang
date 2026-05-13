"""Validate that user-defined names do not collide with reserved names.

A user-defined identifier (function name, argument name, declaration
variable name) must not have the same ``name_hint`` as one of the
reserved data-type names (``int``, ``int32``, ..., ``complex128``),
the reserved reduction names (``sum``, ``prod``, ``min``, ``max``),
or the built-in function names (``exp``, ...).

The grammar does not promote those names to keywords, so they can
otherwise be matched by the ``IDENTIFIER`` rule and slip into the AST.
This pass is the explicit enforcement.

Only declaration sites (``Procedure.name``, ``Operation.name``,
``Argument.name``, ``DeclarationStatement.variable_name``) are
checked. Use sites that resolve to the built-ins are unaffected:
``sum[k](...)`` is fine because ``sum`` here is the built-in
reduction, not a user-defined name.

"""

__all__ = [
    "ReservedNameValidator",
]

import logging

from fhy_core import (
    AnalysisVisitablePass,
    DiagnosticLevel,
    Identifier,
    get_logger,
    register_pass,
)

from fhy_lang.ast.node import (
    Argument,
    DeclarationStatement,
    Node,
    Operation,
    Procedure,
)
from fhy_lang.builtins import (
    BUILTIN_FUNCTION_IDENTIFIERS,
    BUILTIN_REDUCTION_FUNCTION_IDENTIFIERS,
    BUILTIN_TYPE_IDENTIFIERS,
)

from .utils import format_diagnostic_message

_logger: logging.Logger = get_logger(__name__)


def _build_reservation_kinds() -> dict[str, str]:
    """Map each reserved name to the kind of reservation it represents."""
    kinds: dict[str, str] = {}
    for type_name in BUILTIN_TYPE_IDENTIFIERS:
        kinds[type_name] = "data-type keyword"
    for reduction_name in BUILTIN_REDUCTION_FUNCTION_IDENTIFIERS:
        kinds[reduction_name] = "built-in reduction"
    for function_name in BUILTIN_FUNCTION_IDENTIFIERS:
        kinds[function_name] = "built-in function"
    return kinds


_RESERVATION_KINDS: dict[str, str] = _build_reservation_kinds()


@register_pass(
    "fhy_ast_reserved_name_validator",
    "Validates that user-defined identifiers do not shadow reserved names.",
)
class ReservedNameValidator(AnalysisVisitablePass[Node]):
    """Reject user-defined identifiers whose name collides with a reserved name.

    Visits every declaration-site identifier in the module:

    - :class:`Procedure` and :class:`Operation` names.
    - :class:`Argument` names (function-argument lists).
    - :class:`DeclarationStatement` variable names (function bodies and
      module-scope ``param`` declarations).

    For each, the identifier's ``name_hint`` is looked up in the
    union of reserved data-type, reduction, and built-in-function
    names. Collisions are reported as ``ERROR`` diagnostics naming the
    offending identifier and the reservation source.

    """

    def visit_procedure(self, node: Procedure) -> None:
        self._check_identifier(node.name, "procedure", node)

    def visit_operation(self, node: Operation) -> None:
        self._check_identifier(node.name, "operation", node)

    def visit_argument(self, node: Argument) -> None:
        self._check_identifier(node.name, "function argument", node)

    def visit_declaration_statement(self, node: DeclarationStatement) -> None:
        self._check_identifier(node.variable_name, "declaration", node)

    def _check_identifier(
        self, identifier: Identifier, declaration_kind: str, node: Node
    ) -> None:
        reservation_kind = _RESERVATION_KINDS.get(identifier.name_hint)
        if reservation_kind is None:
            return
        _logger.debug(
            "Reserved-name collision: %s %s clashes with reserved %s.",
            declaration_kind,
            identifier.name_hint,
            reservation_kind,
        )
        self.report(
            DiagnosticLevel.ERROR,
            format_diagnostic_message(
                "name collision",
                f"{declaration_kind} name {identifier.name_hint!r} is "
                f"reserved as a {reservation_kind} and may not be used as a "
                "user-defined identifier.",
                node.provenance,
            ),
        )
