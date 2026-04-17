"""FhY language package."""

__all__ = ["replace_identifiers", "from_fhy_source"]

from .ast import replace_identifiers
from .converter import from_fhy_source
