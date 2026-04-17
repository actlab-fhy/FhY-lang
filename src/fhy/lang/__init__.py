"""FhY language package."""

__all__ = ["replace_identifiers", "from_fhy_source", "FhYSyntaxError"]

from .ast import replace_identifiers
from .converter import FhYSyntaxError, from_fhy_source
