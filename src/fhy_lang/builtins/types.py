"""Built-in identifiers for FhY data types."""

import logging

from fhy_core import CoreDataType, Identifier, get_logger
from frozendict import frozendict

_logger: logging.Logger = get_logger(__name__)

BUILTIN_TYPE_IDENTIFIERS: frozendict[str, Identifier] = frozendict(
    {t.value: Identifier(t.value) for t in CoreDataType}
)

_logger.debug(
    "Registered FhY built-in types: %d [%s].",
    len(BUILTIN_TYPE_IDENTIFIERS),
    ", ".join(sorted(BUILTIN_TYPE_IDENTIFIERS)),
)
