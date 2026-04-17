"""Built-in identifiers for FhY reductions."""

from fhy_core import Identifier

BUILTIN_REDUCTION_FUNCTION_IDENTIFIERS: dict[str, Identifier] = {
    "sum": Identifier("sum"),
    "prod": Identifier("prod"),
    "min": Identifier("min"),
    "max": Identifier("max"),
}
