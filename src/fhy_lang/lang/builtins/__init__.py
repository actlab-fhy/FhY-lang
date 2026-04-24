"""Built-in identifiers for FhY."""

import logging

from fhy_core import get_logger

from .functions import BUILTIN_FUNCTION_IDENTIFIERS
from .namespace import BUILTINS_NAMESPACE_NAME
from .reductions import BUILTIN_REDUCTION_FUNCTION_IDENTIFIERS

__all__ = [
    "BUILTINS_NAMESPACE_NAME",
    "BUILTIN_FUNCTION_IDENTIFIERS",
    "BUILTIN_LANG_IDENTIFIERS",
    "BUILTIN_REDUCTION_FUNCTION_IDENTIFIERS",
]

_logger: logging.Logger = get_logger(__name__)

BUILTIN_LANG_IDENTIFIERS = {
    **BUILTIN_FUNCTION_IDENTIFIERS,
    **BUILTIN_REDUCTION_FUNCTION_IDENTIFIERS,
}

_logger.debug(
    "Registered FhY builtins: %d function(s) [%s], %d reduction(s) [%s].",
    len(BUILTIN_FUNCTION_IDENTIFIERS),
    ", ".join(sorted(BUILTIN_FUNCTION_IDENTIFIERS)),
    len(BUILTIN_REDUCTION_FUNCTION_IDENTIFIERS),
    ", ".join(sorted(BUILTIN_REDUCTION_FUNCTION_IDENTIFIERS)),
)
