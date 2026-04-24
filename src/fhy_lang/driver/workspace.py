"""FhY project workspace."""

__all__ = ["Workspace"]

import logging
from pathlib import Path

from fhy_core import get_logger

_logger: logging.Logger = get_logger(__name__)


class Workspace:
    """Workspace describing FhY project."""

    _source_file: Path

    def __init__(self, source_file: Path):
        self._source_file = source_file
        _logger.info("Workspace initialized with source file: %s", source_file)

    @property
    def source_file(self) -> Path:
        """The FhY source file."""
        return self._source_file
