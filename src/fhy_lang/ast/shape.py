"""Shape narrowing for ``NumericalType`` consumers.

``fhy_core.NumericalType.shape`` is ``list[Expression | EllipsisType]`` so
that wildcard ``...`` dimensions can flow through the type system. The FhY
language frontend and its passes do not yet handle wildcard shapes, so they
narrow the shape to ``list[Expression]`` at the boundary using
:func:`narrow_shape`. When wildcard support lands in the passes, the call
sites can drop the narrowing and accept the broader type directly.
"""

__all__ = ["narrow_shape"]

from collections.abc import Sequence
from types import EllipsisType

from fhy_core import Expression


def narrow_shape(
    shape: Sequence[Expression | EllipsisType],
) -> list[Expression]:
    """Return ``shape`` as a concrete list of expressions.

    Args:
        shape: A ``NumericalType.shape`` sequence, possibly containing
            wildcard ``Ellipsis`` dimensions.

    Returns:
        A list with the same elements, typed as :class:`Expression`.

    Raises:
        NotImplementedError: When any dimension is ``Ellipsis``. Wildcard
            shapes are not yet supported by FhY-lang's frontend or passes.
    """
    narrowed: list[Expression] = []
    for index, dimension in enumerate(shape):
        if dimension is Ellipsis:
            raise NotImplementedError(
                f"Wildcard shape dimension at position {index} is not yet "
                "supported by FhY-lang."
            )
        narrowed.append(dimension)
    return narrowed
