"""Built-in identifiers for built-in functions in FhY."""

import logging

from fhy_core import Identifier, get_logger
from frozendict import frozendict

_logger: logging.Logger = get_logger(__name__)

BUILTIN_FUNCTION_IDENTIFIERS: frozendict[str, Identifier] = frozendict(
    {
        "exp": Identifier("exp")  # TODO: Make math library function?
    }
)

_logger.debug(
    "Registered FhY built-in functions: %d [%s].",
    len(BUILTIN_FUNCTION_IDENTIFIERS),
    ", ".join(sorted(BUILTIN_FUNCTION_IDENTIFIERS)),
)
