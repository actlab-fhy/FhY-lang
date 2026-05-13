"""Tests the main-procedure validator AST pass."""

import pytest
from fhy_core import (
    CoreDataType,
    Identifier,
    NumericalType,
    PrimitiveDataType,
    TypeQualifier,
    ValidationFailedError,
)

from fhy_lang.ast import (
    Module,
    Operation,
    Procedure,
    QualifiedType,
    ReturnStatement,
)
from fhy_lang.ast.node import IntLiteral
from fhy_lang.ast.passes import MainProcedureValidator

from .utils import run_validator


def _int32_scalar() -> NumericalType:
    return NumericalType(PrimitiveDataType(CoreDataType.INT32))


def _make_proc(name: Identifier) -> Procedure:
    return Procedure(name=name, templates=(), args=(), body=())


def _make_op(name: Identifier) -> Operation:
    return Operation(
        name=name,
        templates=(),
        args=(),
        body=(ReturnStatement(expression=IntLiteral(value=0)),),
        return_type=QualifiedType(
            base_type=_int32_scalar(), type_qualifier=TypeQualifier.OUTPUT
        ),
    )


def test_module_with_one_proc_main_passes():
    """Test a module with exactly one ``proc main`` passes validation."""
    module = Module(statements=(_make_proc(Identifier("main")),))

    run_validator(MainProcedureValidator(), module)


def test_module_with_no_main_fails():
    """Test a module with no ``proc main`` is rejected."""
    module = Module(statements=(_make_proc(Identifier("helper")),))

    with pytest.raises(ValidationFailedError, match="must declare a 'proc main'"):
        run_validator(MainProcedureValidator(), module)


def test_empty_module_fails():
    """Test an empty module is rejected as missing ``proc main``."""
    module = Module(statements=())

    with pytest.raises(ValidationFailedError, match="must declare a 'proc main'"):
        run_validator(MainProcedureValidator(), module)


def test_module_with_two_mains_fails():
    """Test a module with two ``proc main`` declarations is rejected."""
    module = Module(
        statements=(
            _make_proc(Identifier("main")),
            _make_proc(Identifier("main")),
        ),
    )

    with pytest.raises(ValidationFailedError, match="at most one 'proc main'"):
        run_validator(MainProcedureValidator(), module)


def test_module_with_op_named_main_fails():
    """Test that an ``op`` named ``main`` does not satisfy the entry-point
    requirement."""
    module = Module(statements=(_make_op(Identifier("main")),))

    with pytest.raises(ValidationFailedError, match="must declare a 'proc main'"):
        run_validator(MainProcedureValidator(), module)


def test_module_with_main_proc_and_helper_proc_passes():
    """Test a module containing a ``proc main`` plus other procedures passes."""
    module = Module(
        statements=(
            _make_proc(Identifier("helper")),
            _make_proc(Identifier("main")),
        ),
    )

    run_validator(MainProcedureValidator(), module)


def test_module_with_main_proc_and_helper_op_passes():
    """Test a module containing a ``proc main`` plus an ``op`` passes."""
    module = Module(
        statements=(
            _make_op(Identifier("helper")),
            _make_proc(Identifier("main")),
        ),
    )

    run_validator(MainProcedureValidator(), module)
