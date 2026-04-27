"""Base abstract AST node."""

__all__ = [
    "Node",
    "NodeData",
    "deserialize_node_provenance",
    "is_valid_node_data",
]

from abc import ABC
from dataclasses import dataclass
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
    """A node in the FhY AST."""

    provenance: Provenance

    def get_provenance(self) -> Provenance:
        return self.provenance

    def is_structurally_equivalent(self, other: object) -> bool:
        return isinstance(other, Node)

    def serialize_data_to_dict(self) -> SerializedDict:
        return {"provenance": (self.provenance.serialize_to_dict())}
