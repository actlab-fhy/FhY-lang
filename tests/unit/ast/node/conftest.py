"""Shared fixtures for AST node tests.

The ``int32`` fixture is provided by ``tests/unit/conftest.py``; the
fixtures here build on it for the qualified-type variants used across
the AST node test files.
"""

import pytest
from fhy_core import (
    CoreDataType,
    NumericalType,
    PrimitiveDataType,
    TypeQualifier,
)

from fhy_lang.ast.node import QualifiedType


@pytest.fixture
def float32() -> NumericalType:
    """Float32 numerical type."""
    return NumericalType(PrimitiveDataType(CoreDataType.FLOAT32))


@pytest.fixture
def input_int32(int32: NumericalType) -> QualifiedType:
    """Int32 qualified type with the ``INPUT`` qualifier."""
    return QualifiedType(base_type=int32, type_qualifier=TypeQualifier.INPUT)


@pytest.fixture
def output_int32(int32: NumericalType) -> QualifiedType:
    """Int32 qualified type with the ``OUTPUT`` qualifier."""
    return QualifiedType(base_type=int32, type_qualifier=TypeQualifier.OUTPUT)
