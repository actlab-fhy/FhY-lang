"""FhY project workspace."""

__all__ = ["Workspace"]

from pathlib import Path


class Workspace:
    """Workspace describing FhY project."""

    _source_file: Path

    def __init__(self, source_file: Path):
        self._source_file = source_file

    @property
    def source_file(self) -> Path:
        """The FhY source file."""
        return self._source_file
