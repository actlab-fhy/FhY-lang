"""FhY language errors."""

__all__ = ["FhYSyntaxError"]

from fhy_core import register_error


@register_error
class FhYSyntaxError(SyntaxError):
    """Raised when a syntax error in a FhY program is detected."""
