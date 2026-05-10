"""Tests for ``fhy_lang.ast.node.qualified_type``.

Pins ``QualifiedType`` construction, equivalence, visitor/type contracts,
and serialization round-trip including unknown-qualifier rejection.
"""

import pytest
from fhy_core import (
    CoreDataType,
    DeserializationDictStructureError,
    DeserializationValueError,
    NumericalType,
    PrimitiveDataType,
    TypeQualifier,
)

from fhy_lang.ast.node import QualifiedType


def _int32() -> NumericalType:
    return NumericalType(PrimitiveDataType(CoreDataType.INT32))


def _float32() -> NumericalType:
    return NumericalType(PrimitiveDataType(CoreDataType.FLOAT32))


def _input_int32() -> QualifiedType:
    return QualifiedType(base_type=_int32(), type_qualifier=TypeQualifier.INPUT)


def test_qualified_type_get_type_returns_base_type():
    """Test ``get_type`` returns the ``base_type`` field."""
    base = _int32()
    qualified_type = QualifiedType(base_type=base, type_qualifier=TypeQualifier.INPUT)

    assert qualified_type.get_type() is base


def test_qualified_type_is_equivalent_when_base_and_qualifier_match():
    """Test equivalent base type + same qualifier → equivalent."""
    a = _input_int32()
    b = _input_int32()

    assert a.is_structurally_equivalent(b)


def test_qualified_type_is_inequivalent_when_qualifier_differs():
    """Test differing qualifier breaks equivalence."""
    a = QualifiedType(base_type=_int32(), type_qualifier=TypeQualifier.INPUT)
    b = QualifiedType(base_type=_int32(), type_qualifier=TypeQualifier.OUTPUT)

    assert not a.is_structurally_equivalent(b)


def test_qualified_type_is_inequivalent_when_base_type_differs():
    """Test differing base type breaks equivalence."""
    a = QualifiedType(base_type=_int32(), type_qualifier=TypeQualifier.INPUT)
    b = QualifiedType(base_type=_float32(), type_qualifier=TypeQualifier.INPUT)

    assert not a.is_structurally_equivalent(b)


def test_qualified_type_is_inequivalent_with_non_qualified_type():
    """Test ``is_structurally_equivalent`` rejects non-``QualifiedType`` inputs."""
    assert not _input_int32().is_structurally_equivalent("not-a-type")


def test_qualified_type_round_trips_through_serialization():
    """Test a wrapped serialize/deserialize round-trip preserves equivalence."""
    qualified_type = _input_int32()

    restored = QualifiedType.deserialize_from_dict(qualified_type.serialize_to_dict())

    assert qualified_type.is_structurally_equivalent(restored)


def test_qualified_type_deserialize_data_rejects_unknown_qualifier():
    """Test ``DeserializationValueError`` on an unknown ``type_qualifier``."""
    payload = _input_int32().serialize_data_to_dict()
    payload["type_qualifier"] = "not-a-real-qualifier"

    with pytest.raises(DeserializationValueError):
        QualifiedType.deserialize_data_from_dict(payload)


def test_qualified_type_deserialize_data_rejects_payload_missing_base_type():
    """Test ``DeserializationDictStructureError`` on missing ``base_type``."""
    payload = _input_int32().serialize_data_to_dict()
    payload.pop("base_type")

    with pytest.raises(DeserializationDictStructureError):
        QualifiedType.deserialize_data_from_dict(payload)


def test_qualified_type_deserialize_data_rejects_payload_missing_qualifier():
    """Test ``DeserializationDictStructureError`` on missing ``type_qualifier``."""
    payload = _input_int32().serialize_data_to_dict()
    payload.pop("type_qualifier")

    with pytest.raises(DeserializationDictStructureError):
        QualifiedType.deserialize_data_from_dict(payload)


def test_qualified_type_deserialize_data_rejects_non_string_qualifier():
    """Test ``DeserializationDictStructureError`` on non-string qualifier."""
    payload = _input_int32().serialize_data_to_dict()
    payload["type_qualifier"] = 0

    with pytest.raises(DeserializationDictStructureError):
        QualifiedType.deserialize_data_from_dict(payload)
