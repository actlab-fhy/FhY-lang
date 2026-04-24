"""Option definitions for compilation of FhY source code."""

import logging
from dataclasses import dataclass, field

from fhy_core import get_logger

_logger: logging.Logger = get_logger(__name__)


@dataclass(frozen=True, kw_only=True)
class CompilationOptions:
    """Supported FhY compilation options."""

    verbose: bool = field(default=False)
    perform_optimizations: bool = field(default=False)

    def __post_init__(self) -> None:
        _logger.info(
            "Compilation options: verbose=%s, perform_optimizations=%s",
            self.verbose,
            self.perform_optimizations,
        )
