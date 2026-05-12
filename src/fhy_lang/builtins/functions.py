"""Built-in identifiers for built-in functions in FhY."""

from fhy_core import Identifier
from frozendict import frozendict

BUILTIN_FUNCTION_IDENTIFIERS: frozendict[str, Identifier] = frozendict(
    {
        "exp": Identifier("exp")  # TODO: Make math library function?
    }
)
