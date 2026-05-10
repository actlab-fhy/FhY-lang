"""``QualifiedType`` AST node — pairs a ``Type`` with a ``TypeQualifier``.

Carries the ``HasTypeMixin[Type]`` contract so callers can recover the
underlying type without unwrapping the qualifier. Deserialization rejects
unknown ``TypeQualifier`` values with ``DeserializationValueError``.
"""

from dataclasses import dataclass
from typing import TypeGuard

from fhy_core import (
    DeserializationDictStructureError,
    DeserializationValueError,
    HasTypeMixin,
    SerializedDict,
    Type,
    TypeQualifier,
    is_serialized_dict,
    register_serializable,
)

from .base import Node, NodeData, deserialize_node_provenance, is_valid_node_data


class _QualifiedTypeData(NodeData):
    base_type: SerializedDict
    type_qualifier: str


def _is_valid_qualified_type_data(
    data: SerializedDict,
) -> TypeGuard[_QualifiedTypeData]:
    return (
        "base_type" in data
        and is_serialized_dict(data["base_type"])
        and "type_qualifier" in data
        and isinstance(data["type_qualifier"], str)
        and is_valid_node_data(data)
    )


@register_serializable(type_id="fhy_ast_qualified_type")
@dataclass(frozen=True, kw_only=True)
class QualifiedType(Node, HasTypeMixin[Type]):
    """FhY qualified type AST node."""

    base_type: Type
    type_qualifier: TypeQualifier

    def get_type(self) -> Type:
        return self.base_type

    def is_structurally_equivalent(self, other: object) -> bool:
        return (
            isinstance(other, QualifiedType)
            and super().is_structurally_equivalent(other)
            and self.base_type.is_structurally_equivalent(other.base_type)
            and self.type_qualifier == other.type_qualifier
        )

    def serialize_data_to_dict(self) -> SerializedDict:
        data = super().serialize_data_to_dict()
        data["base_type"] = self.base_type.serialize_to_dict()
        data["type_qualifier"] = self.type_qualifier.value
        return data

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "QualifiedType":
        if not _is_valid_qualified_type_data(data):
            raise DeserializationDictStructureError(
                cls, _QualifiedTypeData.__annotations__, data
            )
        try:
            type_qualifier = TypeQualifier(data["type_qualifier"])
        except ValueError as exc:
            raise DeserializationValueError(
                cls,
                "type_qualifier",
                "a valid type qualifier",
                data["type_qualifier"],
            ) from exc
        return cls(
            base_type=Type.deserialize_from_dict(data["base_type"]),
            type_qualifier=type_qualifier,
            provenance=deserialize_node_provenance(data),
        )
