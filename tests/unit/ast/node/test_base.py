"""Tests for ``fhy_lang.ast.node.base``.

Pins the ``Node`` family contract: abstractness, frozen-dataclass behavior,
and provenance round-trip via the simplest concrete ``Node`` subclass
(``Argument``).
"""

import pytest
from fhy_core import (
    CoreDataType,
    DeserializationDictStructureError,
    Identifier,
    NumericalType,
    PrimitiveDataType,
    Provenance,
    TypeQualifier,
)
from fhy_core.serialization import SerializationError

from fhy_lang.ast.node import Argument, Node, QualifiedType


def _build_argument(provenance: Provenance | None = None) -> Argument:
    qualified_type = QualifiedType(
        base_type=NumericalType(PrimitiveDataType(CoreDataType.INT32)),
        type_qualifier=TypeQualifier.INPUT,
    )
    if provenance is None:
        return Argument(name=Identifier("x"), qualified_type=qualified_type)
    return Argument(
        name=Identifier("x"),
        qualified_type=qualified_type,
        provenance=provenance,
    )


def test_node_is_abstract_and_cannot_be_instantiated():
    """Test ``Node`` raises ``TypeError`` when instantiated directly."""
    with pytest.raises(TypeError):
        Node()  # type: ignore[abstract]


def test_argument_default_provenance_is_unknown():
    """Test default provenance on a concrete ``Node`` is ``UnknownProvenance``."""
    argument = _build_argument()

    assert argument.get_provenance() == Provenance.unknown()


def test_argument_round_trips_through_serialization_with_default_provenance():
    """Test serialize/deserialize is identity for a default-provenance node."""
    argument = _build_argument()

    restored = Argument.deserialize_data_from_dict(argument.serialize_data_to_dict())

    assert argument.is_structurally_equivalent(restored)
    assert restored.get_provenance() == Provenance.unknown()


def test_argument_round_trips_through_serialization_with_named_provenance():
    """Test non-default provenance survives a serialize/deserialize round-trip."""
    provenance = Provenance.fuse(Provenance.unknown(), metadata="audit-test")
    argument = _build_argument(provenance=provenance)

    restored = Argument.deserialize_data_from_dict(argument.serialize_data_to_dict())

    assert restored.get_provenance() == provenance


def test_argument_round_trips_through_wrapped_envelope():
    """Test ``Node.deserialize_from_dict`` reconstructs from a wrapped payload."""
    argument = _build_argument()
    wrapped = argument.serialize_to_dict()

    restored = Node.deserialize_from_dict(wrapped)

    assert isinstance(restored, Argument)
    assert argument.is_structurally_equivalent(restored)


def test_argument_deserialize_data_rejects_payload_missing_provenance():
    """Test missing ``provenance`` key raises ``DeserializationDictStructureError``."""
    payload = _build_argument().serialize_data_to_dict()
    payload.pop("provenance")

    with pytest.raises(DeserializationDictStructureError):
        Argument.deserialize_data_from_dict(payload)


def test_argument_deserialize_data_rejects_non_dict_provenance():
    """Test non-dict ``provenance`` raises ``DeserializationDictStructureError``."""
    payload = _build_argument().serialize_data_to_dict()
    payload["provenance"] = "not-a-dict"

    with pytest.raises(DeserializationDictStructureError):
        Argument.deserialize_data_from_dict(payload)


def test_argument_is_frozen_against_attribute_assignment():
    """Test ``Argument`` is frozen — assigning to a field raises."""
    argument = _build_argument()

    with pytest.raises(Exception):  # FrozenInstanceError or FrozenMutationError
        argument.name = Identifier("y")  # type: ignore[misc]


def test_node_deserialize_from_dict_rejects_unwrapped_payload():
    """Test ``Node.deserialize_from_dict`` requires a wrapped envelope."""
    bare = _build_argument().serialize_data_to_dict()

    with pytest.raises(SerializationError):
        Node.deserialize_from_dict(bare)
