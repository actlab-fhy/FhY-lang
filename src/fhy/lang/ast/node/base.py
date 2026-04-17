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
        return isinstance(other, Node) and self.provenance == other.provenance

    def serialize_data_to_dict(self) -> SerializedDict:
        return {"provenance": (self.provenance.serialize_to_dict())}
