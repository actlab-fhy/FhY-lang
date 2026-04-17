"""Identifier replacement transformer."""

from fhy_core import Identifier, register_pass

from fhy.lang.ast.node import Node
from fhy.lang.ast.transformer import Transformer


@register_pass("fhy_ast_identifier_replacer", "Replaces identifiers in the FhY AST.")
class IdentifierReplacer(Transformer):
    """Replace identifiers.

    Args:
        identifier_map: mapping describing
            identifiers to change from and to.

    """

    _identifier_map: dict[Identifier, Identifier]

    def __init__(self, identifier_map: dict[Identifier, Identifier]):
        super().__init__()
        self._identifier_map = identifier_map

    def visit_identifier(self, identifier: Identifier) -> Identifier:
        return self._identifier_map.get(identifier, identifier)


def replace_identifiers(
    node: Node, identifier_map: dict[Identifier, Identifier]
) -> Node:
    """Replace identifiers within AST.

    Args:
        node: AST node.
        identifier_map: mapping describing identifiers to change from and to.

    Returns:
        Node with identifiers replaced as prescribed by mapping.

    """
    return IdentifierReplacer(identifier_map)(node)
