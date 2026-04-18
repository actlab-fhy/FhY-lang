"""Identifier replacement transformer."""

from fhy_core import (
    Identifier,
    IndexType,
    NumericalType,
    TemplateDataType,
    TupleType,
    register_pass,
)
from fhy_core import replace_identifiers as replace_core_identifiers

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

    def visit_numerical_type(self, numerical_type: NumericalType) -> NumericalType:
        return NumericalType(
            self.visit_data_type(numerical_type.data_type),
            shape=[
                replace_core_identifiers(dim, self._identifier_map)
                for dim in numerical_type.shape
            ],
        )

    Transformer.visit_type.register(NumericalType)(visit_numerical_type)  # type: ignore[attr-defined]

    def visit_index_type(self, index_type: IndexType) -> IndexType:
        new_stride = (
            replace_core_identifiers(index_type.stride, self._identifier_map)
            if index_type.stride is not None
            else None
        )
        return IndexType(
            replace_core_identifiers(index_type.lower_bound, self._identifier_map),
            replace_core_identifiers(index_type.upper_bound, self._identifier_map),
            stride=new_stride,
        )

    Transformer.visit_type.register(IndexType)(visit_index_type)  # type: ignore[attr-defined]

    def visit_tuple_type(self, tuple_type: TupleType) -> TupleType:
        return TupleType([self.visit_type(type) for type in tuple_type.types])

    Transformer.visit_type.register(TupleType)(visit_tuple_type)  # type: ignore[attr-defined]

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
    """Replace identifiers within AST.

    Args:
        node: AST node.
        identifier_map: mapping describing identifiers to change from and to.

    Returns:
        Node with identifiers replaced as prescribed by mapping.

    """
    return IdentifierReplacer(identifier_map)(node)
