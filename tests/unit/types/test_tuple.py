"""Tests for the FhY-lang ``TupleType`` and its dispatcher handlers."""

import pytest
from fhy_core import (
    CoreDataType,
    DeserializationDictStructureError,
    Frozen,
    FrozenMutationError,
    Identifier,
    IndexType,
    LiteralExpression,
    NumericalType,
    PrimitiveDataType,
    TemplateDataType,
    Type,
    TypeUnificationEnvironment,
    VerificationError,
    bind_template,
    is_structurally_equivalent,
    substitute_template,
    unify,
)

from fhy_lang import TupleType

# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def int32_type() -> NumericalType:
    return NumericalType(PrimitiveDataType(CoreDataType.INT32))


@pytest.fixture
def float32_type() -> NumericalType:
    return NumericalType(PrimitiveDataType(CoreDataType.FLOAT32))


@pytest.fixture
def index_zero_to_ten_type() -> IndexType:
    return IndexType(LiteralExpression(0), LiteralExpression(10), LiteralExpression(1))


@pytest.fixture
def template_identifier() -> Identifier:
    return Identifier("T")


# ===========================================================================
# Construction and accessors
# ===========================================================================


def test_construction_freezes_instance(
    int32_type: NumericalType, float32_type: NumericalType
) -> None:
    """Test ``TupleType`` instances are deeply frozen on construction."""
    tuple_type = TupleType([int32_type, float32_type])

    assert isinstance(tuple_type, Frozen)
    assert tuple_type.is_frozen
    with pytest.raises(FrozenMutationError):
        tuple_type._freeze_probe = "mutation"


def test_types_property_returns_list_of_elements(
    int32_type: NumericalType, float32_type: NumericalType
) -> None:
    """Test the ``types`` property returns a list mirroring construction order."""
    tuple_type = TupleType([int32_type, float32_type])

    assert tuple_type.types == [int32_type, float32_type]
    assert isinstance(tuple_type.types, list)


def test_empty_tuple_type_is_constructible_and_self_equivalent() -> None:
    """Test ``TupleType([])`` constructs and is structurally equivalent to itself."""
    left = TupleType([])
    right = TupleType([])

    assert left.types == []
    assert left.is_structurally_equivalent(right)
    assert is_structurally_equivalent(left, right)


# ===========================================================================
# Structural equivalence (method)
# ===========================================================================


def test_structural_equivalence_method_true_for_equal_tuples(
    int32_type: NumericalType, float32_type: NumericalType
) -> None:
    """Test the method returns ``True`` for equal-arity, equal-element tuples."""
    left = TupleType([int32_type, float32_type])
    right = TupleType([int32_type, float32_type])

    assert left.is_structurally_equivalent(right)


def test_structural_equivalence_false_for_swapped_element_order(
    int32_type: NumericalType, float32_type: NumericalType
) -> None:
    """Test element order is significant for structural equivalence."""
    left = TupleType([int32_type, float32_type])
    right = TupleType([float32_type, int32_type])

    assert not left.is_structurally_equivalent(right)


def test_structural_equivalence_false_for_non_tuple(
    int32_type: NumericalType,
) -> None:
    """Test a ``TupleType`` is not structurally equivalent to a non-tuple ``Type``."""
    tuple_type = TupleType([int32_type])

    assert not tuple_type.is_structurally_equivalent(int32_type)


# ===========================================================================
# Stringification
# ===========================================================================


def test_str_uses_str_of_inner_types_not_repr(
    int32_type: NumericalType, float32_type: NumericalType
) -> None:
    """Test ``__str__`` renders inner types via ``str``, matching sibling types."""
    tuple_type = TupleType([int32_type, float32_type])

    rendered = str(tuple_type)

    assert rendered.startswith("(")
    assert rendered.endswith(")")
    assert "int32" in rendered
    assert "float32" in rendered
    assert "NumericalType" not in rendered
    assert "PrimitiveDataType" not in rendered


# ===========================================================================
# Serialization round-trip
# ===========================================================================


def test_serialization_round_trip_preserves_structural_equivalence(
    int32_type: NumericalType, float32_type: NumericalType
) -> None:
    """Test a serialized tuple round-trips to a structurally equivalent tuple."""
    original = TupleType([int32_type, float32_type])

    restored = Type.deserialize_from_dict(original.serialize_to_dict())

    assert isinstance(restored, TupleType)
    assert original.is_structurally_equivalent(restored)


def test_serialization_round_trip_handles_nested_tuples(
    int32_type: NumericalType,
    float32_type: NumericalType,
    index_zero_to_ten_type: IndexType,
) -> None:
    """Test a tuple containing a nested tuple round-trips through serialization."""
    original = TupleType(
        [int32_type, TupleType([float32_type, index_zero_to_ten_type])]
    )

    restored = Type.deserialize_from_dict(original.serialize_to_dict())

    assert isinstance(restored, TupleType)
    assert original.is_structurally_equivalent(restored)


def test_deserialization_rejects_payload_missing_types_key() -> None:
    """Test deserialization raises when the data section omits ``types``."""
    payload = {"__type__": "tuple_type", "__data__": {}}

    with pytest.raises(DeserializationDictStructureError):
        Type.deserialize_from_dict(payload)


def test_deserialization_rejects_payload_with_non_list_types() -> None:
    """Test deserialization raises when ``types`` is not a list."""
    payload = {"__type__": "tuple_type", "__data__": {"types": "not-a-list"}}

    with pytest.raises(DeserializationDictStructureError):
        Type.deserialize_from_dict(payload)


# ===========================================================================
# is_structurally_equivalent dispatcher
# ===========================================================================


def test_dispatcher_structural_equivalence_matches_method(
    int32_type: NumericalType, float32_type: NumericalType
) -> None:
    """Test the dispatcher returns the same answer as the bound method."""
    left = TupleType([int32_type, float32_type])
    right = TupleType([int32_type, float32_type])

    assert is_structurally_equivalent(left, right) == left.is_structurally_equivalent(
        right
    )


def test_dispatcher_structural_equivalence_false_for_arity_mismatch(
    int32_type: NumericalType, float32_type: NumericalType
) -> None:
    """Test the dispatcher returns ``False`` when tuple arities differ."""
    left = TupleType([int32_type, float32_type])
    right = TupleType([int32_type])

    assert not is_structurally_equivalent(left, right)


def test_dispatcher_structural_equivalence_false_for_non_tuple_right(
    int32_type: NumericalType,
) -> None:
    """Test the dispatcher returns ``False`` when the right side is not a tuple."""
    tuple_type = TupleType([int32_type])

    assert not is_structurally_equivalent(tuple_type, int32_type)


# ===========================================================================
# bind_template / substitute_template round-trip
# ===========================================================================


def test_bind_then_substitute_reconstructs_actual(
    int32_type: NumericalType,
    float32_type: NumericalType,
    template_identifier: Identifier,
) -> None:
    """Test bind-then-substitute over a templated pattern reconstructs the actual."""
    pattern = TupleType(
        [NumericalType(TemplateDataType(template_identifier)), float32_type]
    )
    actual = TupleType([int32_type, float32_type])

    environment = bind_template(pattern, actual, TypeUnificationEnvironment.empty())
    substituted = substitute_template(pattern, environment)

    assert substituted.is_structurally_equivalent(actual)


def test_bind_template_arity_mismatch_raises(
    int32_type: NumericalType, float32_type: NumericalType
) -> None:
    """Test ``bind_template`` raises ``VerificationError`` on tuple arity mismatch."""
    pattern = TupleType([int32_type, float32_type])
    actual = TupleType([int32_type])

    with pytest.raises(VerificationError, match="arity mismatch"):
        bind_template(pattern, actual, TypeUnificationEnvironment.empty())


def test_bind_template_non_tuple_actual_raises(int32_type: NumericalType) -> None:
    """Test ``bind_template`` raises ``VerificationError`` for a non-tuple actual."""
    pattern = TupleType([int32_type])

    with pytest.raises(VerificationError, match="TupleType"):
        bind_template(pattern, int32_type, TypeUnificationEnvironment.empty())


# ===========================================================================
# unify
# ===========================================================================


def test_unify_equal_tuples_leaves_environment_unchanged(
    int32_type: NumericalType, float32_type: NumericalType
) -> None:
    """Test unifying two equal tuples returns the input environment unchanged."""
    left = TupleType([int32_type, float32_type])
    right = TupleType([int32_type, float32_type])
    initial_environment = TypeUnificationEnvironment.empty()

    unified, environment = unify(left, right, initial_environment)

    assert unified.is_structurally_equivalent(left)
    assert environment.is_structurally_equivalent(initial_environment)


def test_unify_arity_mismatch_raises(
    int32_type: NumericalType, float32_type: NumericalType
) -> None:
    """Test ``unify`` raises ``VerificationError`` on tuple arity mismatch."""
    left = TupleType([int32_type, float32_type])
    right = TupleType([int32_type])

    with pytest.raises(VerificationError, match="arity mismatch"):
        unify(left, right, TypeUnificationEnvironment.empty())


def test_unify_non_tuple_actual_raises(int32_type: NumericalType) -> None:
    """Test ``unify`` raises ``VerificationError`` for a non-tuple actual."""
    left = TupleType([int32_type])

    with pytest.raises(VerificationError, match="TupleType"):
        unify(left, int32_type, TypeUnificationEnvironment.empty())


def test_unify_carries_template_bindings_from_elements(
    int32_type: NumericalType,
    float32_type: NumericalType,
    template_identifier: Identifier,
) -> None:
    """Test ``unify`` accumulates template bindings discovered inside elements."""
    expected = TupleType(
        [NumericalType(TemplateDataType(template_identifier)), float32_type]
    )
    actual = TupleType([int32_type, float32_type])

    _, environment = unify(expected, actual, TypeUnificationEnvironment.empty())

    bound = environment.get_data_type_binding(template_identifier)
    assert bound is not None
    assert is_structurally_equivalent(bound, PrimitiveDataType(CoreDataType.INT32))


# ===========================================================================
# Conflict detection
# ===========================================================================


def test_bind_template_conflicting_template_in_multiple_slots_raises(
    int32_type: NumericalType,
    float32_type: NumericalType,
    template_identifier: Identifier,
) -> None:
    """Test conflicting bindings for one template across slots raise."""
    pattern = TupleType(
        [
            NumericalType(TemplateDataType(template_identifier)),
            NumericalType(TemplateDataType(template_identifier)),
        ]
    )
    actual = TupleType([int32_type, float32_type])

    with pytest.raises(VerificationError, match="Conflicting"):
        bind_template(pattern, actual, TypeUnificationEnvironment.empty())
