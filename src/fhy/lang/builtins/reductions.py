"""Built-in identifiers for FhY reductions."""

from fhy_core import Identifier
from frozendict import frozendict

BUILTIN_REDUCTION_FUNCTION_IDENTIFIERS: frozendict[str, Identifier] = frozendict(
    {
        "sum": Identifier("sum"),
        "prod": Identifier("prod"),
        "min": Identifier("min"),
        "max": Identifier("max"),
    }
)
