"""FhY errors."""

__all__ = ["FhYSyntaxError", "FhYASTBuildError", "FhYSemanticsError"]

from fhy_core import register_error


@register_error
class FhYSyntaxError(SyntaxError):
    """Raised when a syntax error in a FhY program is detected."""


@register_error
class FhYASTBuildError(Exception):
    """Raised when an error occurs during the construction of the AST."""


@register_error
class FhYSemanticsError(Exception):
    """Raised when a semantic error in a FhY program is detected."""
