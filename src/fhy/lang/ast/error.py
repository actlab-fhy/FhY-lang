"""FhY AST errors."""

__all__ = ["FhYSemanticsError", "FhYTypeError", "FhYStructuralError"]

from fhy_core import Provenance, register_error


def _format_location(provenance: Provenance | None) -> str | None:
    if provenance is None:
        return None
    elif provenance.span is not None:
        return str(provenance.span)
    elif provenance.origins:
        return str(provenance.origins[0])
    else:
        return None


def _format_message(message: str, provenance: Provenance | None) -> str:
    location = _format_location(provenance)
    if location is None:
        return message
    return f"{location}: {message}"


class FhYValidationError(Exception):
    """Raised when a validation error in a FhY program is detected."""

    def __init__(self, message: str, provenance: Provenance | None = None) -> None:
        self.provenance = provenance
        super().__init__(_format_message(message, provenance))


@register_error
class FhYStructuralError(FhYValidationError):
    """Raised when a structural error in a FhY program is detected."""


@register_error
class FhYSemanticsError(FhYValidationError):
    """Raised when a semantic error in a FhY program is detected."""


@register_error
class FhYTypeError(FhYValidationError):
    """Raised when a type error in a FhY program is detected."""
