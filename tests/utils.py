"""Common utilities (not fixtures) used in unit and integration tests."""

from typing import Any

from fhy_core import Identifier


def list_to_types(xs: list[Any]) -> list[type]:
    """Convert a list of objects into a list of object types.

    Args:
        xs: List of objects.

    Returns:
        List of object types.

    """
    return [type(x) for x in xs]


def assert_type(obj: Any, expected_type: type, what_it_is: str) -> None:
    """Assert that an object is of a specific type.

    Args:
        obj: Object to check.
        expected_type: Expected type.
        what_it_is: Description of the object.

    """
    assert isinstance(obj, expected_type), (
        f'Expected {what_it_is} to be "{expected_type.__name__}", '
        f'but got "{type(obj)}".'
    )


def assert_sequence_type(obj: Any, expected_type: type, what_it_is: str) -> None:
    """Assert that all elements in a sequence are of a specific type.

    Args:
        obj: Sequence to check.
        expected_type: Expected type.
        what_it_is: Description of the sequence.

    """
    assert all(isinstance(x, expected_type) for x in obj), (
        f'Expected all {what_it_is} to be "{expected_type.__name__}", '
        f'got "{list_to_types(obj)}".'
    )


def assert_name(
    name: Identifier,
    expected_name: Identifier | str,
    expected_id: int | None = None,
    what_it_is: str = "name",
) -> None:
    """Assert that an identifier has the expected name and ID.

    Args:
        name: Identifier to check.
        expected_name: Expected name.
        expected_id: Expected ID.
        what_it_is: Description of the identifier.

    """
    if isinstance(expected_name, Identifier):
        if expected_id is not None:
            raise ValueError(
                'Cannot specify expected_name as an "Identifier" and then '
                "provide an expected_id."
            )
        assert name == expected_name, (
            f'Expected {what_it_is} to be "{expected_name}", got "{name}".'
        )
    else:
        assert name.name_hint == expected_name, (
            f'Expected {what_it_is} to be "{expected_name}", got "{name.name_hint}".'
        )
        if expected_id is not None:
            assert name.id == expected_id, (
                f'Expected {what_it_is} to have ID "{expected_id}", got "{name.id}".'
            )
