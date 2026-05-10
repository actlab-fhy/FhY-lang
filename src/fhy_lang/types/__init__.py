"""Algorithm-IR type vocabulary owned by FhY-lang.

The types in this package are the algorithm-IR concepts that ``fhy_core``
deliberately does not ship: they are eliminated by the algorithm-to-capability
lowering pass and never appear in capability or hardware IRs. ``fhy_core``
exposes open dispatchers (``is_structurally_equivalent``, ``bind_template``,
``substitute_template``, ``unify``) so layer-specific types can register
handlers from their owning package without modifying ``fhy_core``.

Importing this package imports its submodules for their handler-registration
side effects.
"""

__all__ = ["TupleType"]

from .tuple import TupleType
