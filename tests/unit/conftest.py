"""PyTest unit test fixtures and utilities."""

from collections.abc import Callable

import pytest
from fhy.lang.ast import (
    Node,
)
from fhy.lang.converter.from_fhy_source import from_fhy_source as fhy_source
from fhy_core import (
    Provenance,
)


@pytest.fixture
def construct_ast() -> Callable[[str], Node]:
    """Construct an abstract syntax tree (AST) from a raw text file source."""

    def _inner(source: str) -> Node:
        return fhy_source(source, provenance=Provenance.unknown())

    return _inner
