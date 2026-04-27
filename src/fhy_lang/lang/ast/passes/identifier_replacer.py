"""Identifier replacement transformer."""

__all__ = ["replace_identifiers"]

from fhy_core import (
    Identifier,
    IndexType,
    NumericalType,
    TemplateDataType,
    TupleType,
    Type,
    register_pass,
)
from fhy_core import replace_identifiers as replace_core_identifiers

from fhy_lang.lang.ast.node import Node

from .transformer import Transformer


@register_pass("fhy_ast_identifier_replacer", "Replaces identifiers in the FhY AST.")
class _IdentifierReplacer(Transformer):
    """Replace identifiers according to a given mapping."""

    _identifier_map: dict[Identifier, Identifier]

    def __init__(self, identifier_map: dict[Identifier, Identifier]) -> None:
        super().__init__()
        self._identifier_map = identifier_map

    def visit_numerical_type(self, node: NumericalType) -> Type:
        return NumericalType(
            self.visit_data_type(node.data_type),
            shape=[
                replace_core_identifiers(dimension, self._identifier_map)
                for dimension in node.shape
            ],
        )

    def visit_index_type(self, node: IndexType) -> Type:
        return IndexType(
            replace_core_identifiers(node.lower_bound, self._identifier_map),
            replace_core_identifiers(node.upper_bound, self._identifier_map),
            replace_core_identifiers(node.stride, self._identifier_map),
        )

    def visit_tuple_type(self, node: TupleType) -> Type:
        return TupleType([self.visit_type(inner_type) for inner_type in node.types])

    def visit_template_data_type(
        self, template_data_type: TemplateDataType
    ) -> TemplateDataType:
        return TemplateDataType(
            self.visit_identifier(template_data_type.data_type),
            template_data_type.widths,
        )

    def visit_identifier(self, identifier: Identifier) -> Identifier:
        return self._identifier_map.get(identifier, identifier)


def replace_identifiers(
    node: Node, identifier_map: dict[Identifier, Identifier]
) -> Node:
    """Replace identifiers within the AST according to the given mapping.

    Args:
        node: AST node.
        identifier_map: Mapping describing the identifiers to replace.

    Returns:
        A node with identifiers replaced as prescribed by the mapping.

    """
    return _IdentifierReplacer(identifier_map)(node)
