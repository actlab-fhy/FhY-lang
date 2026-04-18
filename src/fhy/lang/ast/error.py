"""FhY AST errors."""

__all__ = ["FhYSemanticsError", "FhYTypeError", "FhYStructuralError"]

from fhy_core import register_error


@register_error
class FhYStructuralError(Exception):
    """Raised when a structural error in a FhY program is detected."""


@register_error
class FhYSemanticsError(Exception):
    """Raised when a semantic error in a FhY program is detected."""


@register_error
class FhYTypeError(TypeError):
    """Raised when a type error in a FhY program is detected."""
