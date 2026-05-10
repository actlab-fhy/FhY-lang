"""Tests for the FhY-lang ``TupleType`` and its dispatcher handlers.

The ``TupleType`` class itself is owned by FhY-lang, but its handlers register
against the dispatchers exposed from ``fhy_core``. The cross-layer test here
deliberately calls ``fhy_core`` dispatchers directly to confirm the extension
mechanism works without modifying ``fhy_core``.
"""

import pytest
from fhy_core import (
    CoreDataType,
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


def _int32() -> NumericalType:
    return NumericalType(PrimitiveDataType(CoreDataType.INT32))


def _float32() -> NumericalType:
    return NumericalType(PrimitiveDataType(CoreDataType.FLOAT32))


def _index_zero_to_ten() -> IndexType:
    return IndexType(LiteralExpression(0), LiteralExpression(10), LiteralExpression(1))


# ---------------------------------------------------------------------------
# Construction and accessors
# ---------------------------------------------------------------------------


def test_construction_freezes_instance() -> None:
    tuple_type = TupleType([_int32(), _float32()])
    assert isinstance(tuple_type, Frozen)
    assert tuple_type.is_frozen
    with pytest.raises(FrozenMutationError):
        tuple_type._freeze_probe = "mutation"


def test_types_property_returns_list_of_elements() -> None:
    int_type = _int32()
    float_type = _float32()
    tuple_type = TupleType([int_type, float_type])
    assert tuple_type.types == [int_type, float_type]
    assert isinstance(tuple_type.types, list)


def test_structural_equivalence_method_true_for_equal_tuples() -> None:
    left = TupleType([_int32(), _float32()])
    right = TupleType([_int32(), _float32()])
    assert left.is_structurally_equivalent(right)


def test_structural_equivalence_false_for_swapped_element_order() -> None:
    left = TupleType([_int32(), _float32()])
    right = TupleType([_float32(), _int32()])
    assert not left.is_structurally_equivalent(right)


def test_structural_equivalence_false_for_non_tuple() -> None:
    tuple_type = TupleType([_int32()])
    assert not tuple_type.is_structurally_equivalent(_int32())


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


def test_serialization_round_trip_preserves_structural_equivalence() -> None:
    original = TupleType([_int32(), _float32()])
    restored = Type.deserialize_from_dict(original.serialize_to_dict())
    assert isinstance(restored, TupleType)
    assert original.is_structurally_equivalent(restored)


def test_serialization_round_trip_handles_nested_tuples() -> None:
    original = TupleType(
        [
            _int32(),
            TupleType([_float32(), _index_zero_to_ten()]),
        ]
    )
    restored = Type.deserialize_from_dict(original.serialize_to_dict())
    assert isinstance(restored, TupleType)
    assert original.is_structurally_equivalent(restored)


# ---------------------------------------------------------------------------
# is_structurally_equivalent dispatcher
# ---------------------------------------------------------------------------


def test_dispatcher_structural_equivalence_matches_method() -> None:
    left = TupleType([_int32(), _float32()])
    right = TupleType([_int32(), _float32()])
    assert is_structurally_equivalent(left, right) == left.is_structurally_equivalent(
        right
    )


def test_dispatcher_structural_equivalence_false_for_arity_mismatch() -> None:
    left = TupleType([_int32(), _float32()])
    right = TupleType([_int32()])
    assert not is_structurally_equivalent(left, right)


def test_dispatcher_structural_equivalence_false_for_non_tuple_right() -> None:
    tuple_type = TupleType([_int32()])
    assert not is_structurally_equivalent(tuple_type, _int32())


# ---------------------------------------------------------------------------
# bind_template / substitute_template round-trip
# ---------------------------------------------------------------------------


def test_bind_then_substitute_reconstructs_actual() -> None:
    template_identifier = Identifier("T")
    pattern = TupleType(
        [
            NumericalType(TemplateDataType(template_identifier)),
            _float32(),
        ]
    )
    actual = TupleType([_int32(), _float32()])
    environment = bind_template(pattern, actual, TypeUnificationEnvironment.empty())
    substituted = substitute_template(pattern, environment)
    assert substituted.is_structurally_equivalent(actual)


def test_bind_template_arity_mismatch_raises() -> None:
    pattern = TupleType([_int32(), _float32()])
    actual = TupleType([_int32()])
    with pytest.raises(VerificationError, match="arity mismatch"):
        bind_template(pattern, actual, TypeUnificationEnvironment.empty())


def test_bind_template_non_tuple_actual_raises() -> None:
    pattern = TupleType([_int32()])
    with pytest.raises(VerificationError, match="TupleType"):
        bind_template(pattern, _int32(), TypeUnificationEnvironment.empty())


# ---------------------------------------------------------------------------
# unify
# ---------------------------------------------------------------------------


def test_unify_equal_tuples_leaves_environment_unchanged() -> None:
    left = TupleType([_int32(), _float32()])
    right = TupleType([_int32(), _float32()])
    initial_environment = TypeUnificationEnvironment.empty()
    unified, environment = unify(left, right, initial_environment)
    assert unified.is_structurally_equivalent(left)
    assert environment.is_structurally_equivalent(initial_environment)


def test_unify_arity_mismatch_raises() -> None:
    left = TupleType([_int32(), _float32()])
    right = TupleType([_int32()])
    with pytest.raises(VerificationError, match="arity mismatch"):
        unify(left, right, TypeUnificationEnvironment.empty())


def test_unify_non_tuple_actual_raises() -> None:
    left = TupleType([_int32()])
    with pytest.raises(VerificationError, match="TupleType"):
        unify(left, _int32(), TypeUnificationEnvironment.empty())


def test_unify_carries_template_bindings_from_elements() -> None:
    template_identifier = Identifier("T")
    expected = TupleType(
        [
            NumericalType(TemplateDataType(template_identifier)),
            _float32(),
        ]
    )
    actual = TupleType([_int32(), _float32()])
    _, environment = unify(expected, actual, TypeUnificationEnvironment.empty())
    bound = environment.get_data_type_binding(template_identifier)
    assert bound is not None
    assert is_structurally_equivalent(bound, PrimitiveDataType(CoreDataType.INT32))


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------


def test_bind_template_conflicting_template_in_multiple_slots_raises() -> None:
    template_identifier = Identifier("T")
    pattern = TupleType(
        [
            NumericalType(TemplateDataType(template_identifier)),
            NumericalType(TemplateDataType(template_identifier)),
        ]
    )
    actual = TupleType([_int32(), _float32()])
    with pytest.raises(VerificationError, match="Conflicting"):
        bind_template(pattern, actual, TypeUnificationEnvironment.empty())


# ---------------------------------------------------------------------------
# Cross-layer extension
# ---------------------------------------------------------------------------


def test_dispatchers_called_directly_from_fhy_core() -> None:
    """Exercise the four dispatchers from ``fhy_core`` against ``TupleType``.

    Importing the dispatchers from ``fhy_core`` (not from FhY-lang) and
    feeding them ``TupleType`` values demonstrates that the registration
    mechanism crosses the package boundary without any modification to
    ``fhy_core``.
    """
    template_identifier = Identifier("T")
    pattern = TupleType(
        [
            NumericalType(TemplateDataType(template_identifier)),
            _float32(),
        ]
    )
    actual = TupleType([_int32(), _float32()])

    assert is_structurally_equivalent(actual, TupleType([_int32(), _float32()]))

    environment = bind_template(pattern, actual, TypeUnificationEnvironment.empty())
    substituted = substitute_template(pattern, environment)
    assert is_structurally_equivalent(substituted, actual)

    unified, _ = unify(actual, actual, TypeUnificationEnvironment.empty())
    assert is_structurally_equivalent(unified, actual)
