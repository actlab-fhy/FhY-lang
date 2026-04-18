"""FhY AST errors."""

__all__ = ["FhYSemanticsError"]

from fhy_core import register_error


@register_error
class FhYSemanticsError(Exception):
    """Raised when a semantic error in a FhY program is detected."""
