"""FhY language errors."""

__all__ = ["FhYInternalError", "FhYSyntaxError"]

from fhy_core import register_error


@register_error
class FhYSyntaxError(SyntaxError):
    """Raised when a syntax error in a FhY program is detected."""


@register_error
class FhYInternalError(RuntimeError):
    """Raised when the converter reaches a state believed unreachable.

    These branches exist as defensive guards behind ANTLR grammar guarantees.
    Hitting one indicates a converter or grammar bug, not user error.
    """
