"""Tests for ``fhy_lang.ast.node.qualified_type``.

Pins ``QualifiedType`` construction, equivalence, visitor/type contracts,
and serialization round-trip including unknown-qualifier rejection.
"""

import pytest
from fhy_core import (
    DeserializationDictStructureError,
    DeserializationValueError,
    NumericalType,
    TypeQualifier,
)

from fhy_lang.ast.node import QualifiedType


def test_qualified_type_get_type_returns_base_type(int32: NumericalType):
    """Test ``get_type`` returns the ``base_type`` field."""
    qualified_type = QualifiedType(base_type=int32, type_qualifier=TypeQualifier.INPUT)

    assert qualified_type.get_type() is int32


def test_qualified_type_is_equivalent_when_base_and_qualifier_match(
    input_int32: QualifiedType,
):
    """Test equivalent base type + same qualifier are equivalent."""
    other = QualifiedType(
        base_type=input_int32.base_type, type_qualifier=input_int32.type_qualifier
    )

    assert input_int32.is_structurally_equivalent(other)


def test_qualified_type_is_inequivalent_when_qualifier_differs(
    input_int32: QualifiedType, output_int32: QualifiedType
):
    """Test differing qualifier breaks equivalence."""
    assert not input_int32.is_structurally_equivalent(output_int32)


def test_qualified_type_is_inequivalent_when_base_type_differs(
    int32: NumericalType, float32: NumericalType
):
    """Test differing base type breaks equivalence."""
    a = QualifiedType(base_type=int32, type_qualifier=TypeQualifier.INPUT)
    b = QualifiedType(base_type=float32, type_qualifier=TypeQualifier.INPUT)

    assert not a.is_structurally_equivalent(b)


def test_qualified_type_is_inequivalent_with_non_qualified_type(
    input_int32: QualifiedType,
):
    """Test ``is_structurally_equivalent`` rejects non-``QualifiedType`` inputs."""
    assert not input_int32.is_structurally_equivalent("not-a-type")


def test_qualified_type_deserialize_data_rejects_unknown_qualifier(
    input_int32: QualifiedType,
):
    """Test ``DeserializationValueError`` on an unknown ``type_qualifier``."""
    payload = input_int32.serialize_data_to_dict()
    payload["type_qualifier"] = "not-a-real-qualifier"

    with pytest.raises(DeserializationValueError):
        QualifiedType.deserialize_data_from_dict(payload)


def test_qualified_type_deserialize_data_rejects_payload_missing_base_type(
    input_int32: QualifiedType,
):
    """Test ``DeserializationDictStructureError`` on missing ``base_type``."""
    payload = input_int32.serialize_data_to_dict()
    payload.pop("base_type")

    with pytest.raises(DeserializationDictStructureError):
        QualifiedType.deserialize_data_from_dict(payload)


def test_qualified_type_deserialize_data_rejects_payload_missing_qualifier(
    input_int32: QualifiedType,
):
    """Test ``DeserializationDictStructureError`` on missing ``type_qualifier``."""
    payload = input_int32.serialize_data_to_dict()
    payload.pop("type_qualifier")

    with pytest.raises(DeserializationDictStructureError):
        QualifiedType.deserialize_data_from_dict(payload)


def test_qualified_type_deserialize_data_rejects_non_string_qualifier(
    input_int32: QualifiedType,
):
    """Test ``DeserializationDictStructureError`` on non-string qualifier."""
    payload = input_int32.serialize_data_to_dict()
    payload["type_qualifier"] = 0

    with pytest.raises(DeserializationDictStructureError):
        QualifiedType.deserialize_data_from_dict(payload)


@pytest.mark.parametrize("qualifier", list(TypeQualifier))
def test_qualified_type_round_trips_for_every_qualifier(
    int32: NumericalType, qualifier: TypeQualifier
):
    """Test every ``TypeQualifier`` value survives a wrapped round-trip."""
    qualified_type = QualifiedType(base_type=int32, type_qualifier=qualifier)

    restored = QualifiedType.deserialize_from_dict(qualified_type.serialize_to_dict())

    assert isinstance(restored, QualifiedType)
    assert restored.type_qualifier == qualifier
    assert qualified_type.is_structurally_equivalent(restored)
