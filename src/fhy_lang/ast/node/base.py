"""Base abstract AST node.

``NodeData``, ``is_valid_node_data``, and ``deserialize_node_provenance``
are public at the module level for sibling ``node`` modules to compose
into payload validators and deserializers. They are not re-exported from
``fhy_lang.ast``.
"""

__all__ = [
    "Node",
    "NodeData",
    "deserialize_node_provenance",
    "is_valid_node_data",
]

from abc import ABC
from dataclasses import dataclass, field
from typing import TypedDict, TypeGuard

from fhy_core import (
    FrozenMixin,
    HasProvenanceMixin,
    Provenance,
    SerializedDict,
    StructuralEquivalenceMixin,
    VisitableMixin,
    WrappedFamilySerializable,
    is_serialized_dict,
)


class NodeData(TypedDict):
    """Data for a node."""

    provenance: SerializedDict


def is_valid_node_data(data: SerializedDict) -> TypeGuard[NodeData]:
    """Return True if the data is a valid node data."""
    return "provenance" in data and is_serialized_dict(data["provenance"])


def deserialize_node_provenance(data: NodeData) -> Provenance:
    """Deserialize node provenance from validated node data."""
    provenance_data = data["provenance"]
    return Provenance.deserialize_from_dict(provenance_data)


@dataclass(frozen=True, kw_only=True)
class Node(
    WrappedFamilySerializable,
    VisitableMixin,
    FrozenMixin,
    HasProvenanceMixin,
    StructuralEquivalenceMixin,
    ABC,
):
    """A node in the FhY AST.

    Concrete subclasses register a unique ``type_id`` via
    ``@register_serializable`` and supply ``serialize_data_to_dict``,
    ``deserialize_data_from_dict``, ``get_visit_children``, and an override
    of ``is_structurally_equivalent``.

    Structural equivalence is id-based for ``Identifier``-typed fields:
    two nodes are equivalent only when every Identifier-bearing field is
    the same identifier (same id, per ``Identifier.__eq__``). Two
    ``Identifier("x")`` instances constructed independently are not
    equivalent. This contract recurses through nested nodes: a ``Module``
    whose statements contain independently-constructed identifiers is
    not structurally equivalent to one with the same name hints but
    separate ids.

    ``deserialize_data_from_dict`` raises:

    - ``DeserializationDictStructureError`` when the payload is missing
      required keys or has wrong key types.
    - ``DeserializationValueError`` when a key is structurally valid but
      its value is out-of-domain (e.g., an unknown enum value).

    ``deserialize_from_dict`` (the wrapped-family entry point on the
    family base classes ``Node``, ``Statement``, ``Expression``,
    ``Function``) additionally raises ``SerializationError`` from
    ``fhy_core`` when the wrapped envelope is malformed or when
    ``__type__`` resolves to a class outside the expected family.
    """

    provenance: Provenance = field(default_factory=Provenance.unknown)

    def get_provenance(self) -> Provenance:
        return self.provenance

    def is_structurally_equivalent(self, other: object) -> bool:
        return isinstance(other, Node)

    def serialize_data_to_dict(self) -> SerializedDict:
        return {"provenance": self.provenance.serialize_to_dict()}
