# Copyright (c) 2024 FhY Developers
# Christopher Priebe <cpriebe@ucsd.edu>
# Jason C Del Rio <j3delrio@ucsd.edu>
# Hadi S Esmaeilzadeh <hadi@ucsd.edu>
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are
# permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of
# conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list
# of conditions and the following disclaimer in the documentation and/or other materials
# provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be
# used to endorse or promote products derived from this software without specific prior
# written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
# SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.

"""Qualified type AST node."""

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
class QualifiedType(Node, HasTypeMixin):
    """FhY qualified type AST node."""

    base_type: Type
    type_qualifier: TypeQualifier

    @property
    def type(self) -> Type:
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
        type_qualifier = data["type_qualifier"]
        if type_qualifier not in TypeQualifier._value2member_map_:
            raise DeserializationValueError(
                cls,
                "type_qualifier",
                "a valid type qualifier",
                type_qualifier,
            )
        return cls(
            base_type=Type.deserialize_from_dict(data["base_type"]),
            type_qualifier=TypeQualifier(type_qualifier),
            provenance=deserialize_node_provenance(data),
        )
