"""Built-in identifiers for FhY data types."""

from fhy_core import CoreDataType, Identifier
from frozendict import frozendict

BUILTIN_TYPE_IDENTIFIERS: frozendict[str, Identifier] = frozendict(
    {t.value: Identifier(t.value) for t in CoreDataType}
)
