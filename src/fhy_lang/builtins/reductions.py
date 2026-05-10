"""Built-in identifiers for FhY reductions."""

import logging

from fhy_core import Identifier, get_logger
from frozendict import frozendict

_logger: logging.Logger = get_logger(__name__)

BUILTIN_REDUCTION_FUNCTION_IDENTIFIERS: frozendict[str, Identifier] = frozendict(
    {
        "sum": Identifier("sum"),
        "prod": Identifier("prod"),
        "min": Identifier("min"),
        "max": Identifier("max"),
    }
)

_logger.debug(
    "Registered FhY built-in reductions: %d [%s].",
    len(BUILTIN_REDUCTION_FUNCTION_IDENTIFIERS),
    ", ".join(sorted(BUILTIN_REDUCTION_FUNCTION_IDENTIFIERS)),
)
