"""Heterogeneous fixed-arity product type for the algorithm IR.

Defines :class:`TupleType` and registers the four dispatcher handlers that
plug it into ``fhy_core``'s open type-system dispatchers. Importing this
module installs the handlers as a side effect of class registration.
"""

__all__ = ["TupleType"]

from collections.abc import Sequence
from typing import TypedDict, TypeGuard

from fhy_core import (
    DeserializationDictStructureError,
    SerializedDict,
    Type,
    TypeUnificationEnvironment,
    VerificationError,
    bind_template,
    format_comma_separated_list,
    is_serialized_dict,
    is_structurally_equivalent,
    register_serializable,
    substitute_template,
    unify,
)


class _TupleTypeData(TypedDict):
    types: list[SerializedDict]


def _is_valid_tuple_type_data(data: SerializedDict) -> TypeGuard[_TupleTypeData]:
    return (
        "types" in data
        and isinstance(data["types"], list)
        and all(is_serialized_dict(ty_dict) for ty_dict in data["types"])
    )


@register_serializable(type_id="tuple_type")
class TupleType(Type):
    """Heterogeneous fixed-arity product type.

    Models FhY constructs where a single operation produces multiple typed
    results bundled together.
    """

    _types: tuple[Type, ...]

    def __init__(self, types: Sequence[Type]) -> None:
        super().__init__()
        self._types = tuple(types)
        self.freeze(deep=True)

    @property
    def types(self) -> list[Type]:
        return list(self._types)

    def serialize_data_to_dict(self) -> SerializedDict:
        return {"types": [ty.serialize_to_dict() for ty in self._types]}

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "TupleType":
        if not _is_valid_tuple_type_data(data):
            raise DeserializationDictStructureError(
                cls, _TupleTypeData.__annotations__, data
            )
        return cls([Type.deserialize_from_dict(ty_dict) for ty_dict in data["types"]])

    def __str__(self) -> str:
        return f"({format_comma_separated_list(self._types, str_func=str)})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._types!r})"


@is_structurally_equivalent.register
def _(left: TupleType, right: object) -> bool:
    if not isinstance(right, TupleType):
        return False
    if len(left.types) != len(right.types):
        return False
    return all(
        is_structurally_equivalent(left_element, right_element)
        for left_element, right_element in zip(left.types, right.types, strict=True)
    )


@bind_template.register
def _(
    pattern: TupleType,
    actual: Type,
    environment: TypeUnificationEnvironment,
) -> TypeUnificationEnvironment:
    if not isinstance(actual, TupleType):
        raise VerificationError(
            f"Cannot bind TupleType pattern against {type(actual).__name__}."
        )
    if len(pattern.types) != len(actual.types):
        raise VerificationError(
            f"Tuple arity mismatch: pattern has {len(pattern.types)} "
            f"elements, actual has {len(actual.types)}."
        )
    next_environment = environment
    for pattern_element, actual_element in zip(
        pattern.types, actual.types, strict=True
    ):
        next_environment = bind_template(
            pattern_element, actual_element, next_environment
        )
    return next_environment


@substitute_template.register
def _(type_: TupleType, environment: TypeUnificationEnvironment) -> Type:
    return TupleType(
        [substitute_template(element, environment) for element in type_.types]
    )


@unify.register
def _(
    expected: TupleType,
    actual: Type,
    environment: TypeUnificationEnvironment,
) -> tuple[Type, TypeUnificationEnvironment]:
    if not isinstance(actual, TupleType):
        raise VerificationError(f"Cannot unify TupleType with {type(actual).__name__}.")
    if len(expected.types) != len(actual.types):
        raise VerificationError(
            f"Tuple arity mismatch during unification: "
            f"{len(expected.types)} vs {len(actual.types)}."
        )
    next_environment = environment
    unified_elements: list[Type] = []
    for expected_element, actual_element in zip(
        expected.types, actual.types, strict=True
    ):
        unified_element, next_environment = unify(
            expected_element, actual_element, next_environment
        )
        unified_elements.append(unified_element)
    return TupleType(unified_elements), next_environment
