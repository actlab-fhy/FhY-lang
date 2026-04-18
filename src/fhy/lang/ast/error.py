"""FhY AST errors."""

__all__ = ["FhYSemanticsError", "FhYTypeError", "FhYStructuralError"]

from fhy_core import Provenance, register_error


def _format_location(provenance: Provenance | None) -> str | None:
    if provenance is None:
        return None
    if provenance.span is not None:
        return str(provenance.span)
    if provenance.origins:
        return str(provenance.origins[0])
    return None


def _format_message(message: str, provenance: Provenance | None) -> str:
    location = _format_location(provenance)
    if location is None:
        return message
    return f"{location}: {message}"


@register_error
class FhYStructuralError(Exception):
    """Raised when a structural error in a FhY program is detected."""

    provenance: Provenance | None

    def __init__(self, message: str, provenance: Provenance | None = None) -> None:
        self.provenance = provenance
        super().__init__(_format_message(message, provenance))


@register_error
class FhYSemanticsError(Exception):
    """Raised when a semantic error in a FhY program is detected."""

    provenance: Provenance | None

    def __init__(self, message: str, provenance: Provenance | None = None) -> None:
        self.provenance = provenance
        super().__init__(_format_message(message, provenance))


@register_error
class FhYTypeError(TypeError):
    """Raised when a type error in a FhY program is detected."""

    provenance: Provenance | None

    def __init__(self, message: str, provenance: Provenance | None = None) -> None:
        self.provenance = provenance
        super().__init__(_format_message(message, provenance))
