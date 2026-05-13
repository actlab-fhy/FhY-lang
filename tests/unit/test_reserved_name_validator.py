"""Tests the reserved-name validator AST pass."""

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
    Argument,
    DeclarationStatement,
    IntLiteral,
    Module,
    Operation,
    Procedure,
    QualifiedType,
    ReturnStatement,
)
from fhy_lang.ast.passes import ReservedNameValidator
from fhy_lang.builtins import (
    BUILTIN_FUNCTION_IDENTIFIERS,
    BUILTIN_REDUCTION_FUNCTION_IDENTIFIERS,
    BUILTIN_TYPE_IDENTIFIERS,
)

from .utils import run_validator

_RESERVED_TYPE_NAMES = sorted(BUILTIN_TYPE_IDENTIFIERS.keys())
_RESERVED_REDUCTION_NAMES = sorted(BUILTIN_REDUCTION_FUNCTION_IDENTIFIERS.keys())
_RESERVED_BUILTIN_FUNCTION_NAMES = sorted(BUILTIN_FUNCTION_IDENTIFIERS.keys())


def _int32_scalar() -> NumericalType:
    return NumericalType(PrimitiveDataType(CoreDataType.INT32))


def _make_module_with_proc_named(name: str) -> Module:
    return Module(
        statements=(Procedure(name=Identifier(name), templates=(), args=(), body=()),),
    )


def _make_module_with_op_named(name: str) -> Module:
    return Module(
        statements=(
            Operation(
                name=Identifier(name),
                templates=(),
                args=(),
                body=(ReturnStatement(expression=IntLiteral(value=0)),),
                return_type=QualifiedType(
                    base_type=_int32_scalar(), type_qualifier=TypeQualifier.OUTPUT
                ),
            ),
        ),
    )


def _make_module_with_argument_named(name: str) -> Module:
    return Module(
        statements=(
            Procedure(
                name=Identifier("main"),
                templates=(),
                args=(
                    Argument(
                        name=Identifier(name),
                        qualified_type=QualifiedType(
                            base_type=_int32_scalar(),
                            type_qualifier=TypeQualifier.INPUT,
                        ),
                    ),
                ),
                body=(),
            ),
        ),
    )


def _make_module_with_declaration_named(name: str) -> Module:
    return Module(
        statements=(
            Procedure(
                name=Identifier("main"),
                templates=(),
                args=(),
                body=(
                    DeclarationStatement(
                        variable_name=Identifier(name),
                        variable_type=QualifiedType(
                            base_type=_int32_scalar(),
                            type_qualifier=TypeQualifier.TEMP,
                        ),
                        expression=IntLiteral(value=0),
                    ),
                ),
            ),
        ),
    )


def test_user_identifiers_with_non_reserved_names_pass():
    """Test a module with ordinary user-defined names passes."""
    module = Module(
        statements=(
            Procedure(name=Identifier("main"), templates=(), args=(), body=()),
        ),
    )

    run_validator(ReservedNameValidator(), module)


@pytest.mark.parametrize("name", _RESERVED_TYPE_NAMES)
def test_proc_name_collides_with_type_keyword_fails(name):
    """Test a ``proc`` named after a reserved type keyword is rejected."""
    module = _make_module_with_proc_named(name)

    with pytest.raises(ValidationFailedError, match="reserved"):
        run_validator(ReservedNameValidator(), module)


@pytest.mark.parametrize("name", _RESERVED_TYPE_NAMES)
def test_op_name_collides_with_type_keyword_fails(name):
    """Test an ``op`` named after a reserved type keyword is rejected."""
    module = _make_module_with_op_named(name)

    with pytest.raises(ValidationFailedError, match="reserved"):
        run_validator(ReservedNameValidator(), module)


@pytest.mark.parametrize("name", _RESERVED_REDUCTION_NAMES)
def test_proc_name_collides_with_reduction_name_fails(name):
    """Test a ``proc`` named after a reserved reduction is rejected."""
    module = _make_module_with_proc_named(name)

    with pytest.raises(ValidationFailedError, match="reserved"):
        run_validator(ReservedNameValidator(), module)


@pytest.mark.parametrize("name", _RESERVED_REDUCTION_NAMES)
def test_op_name_collides_with_reduction_name_fails(name):
    """Test an ``op`` named after a reserved reduction is rejected."""
    module = _make_module_with_op_named(name)

    with pytest.raises(ValidationFailedError, match="reserved"):
        run_validator(ReservedNameValidator(), module)


@pytest.mark.parametrize("name", _RESERVED_BUILTIN_FUNCTION_NAMES)
def test_proc_name_collides_with_builtin_function_fails(name):
    """Test a ``proc`` shadowing a built-in function name is rejected."""
    module = _make_module_with_proc_named(name)

    with pytest.raises(ValidationFailedError, match="reserved"):
        run_validator(ReservedNameValidator(), module)


@pytest.mark.parametrize("name", _RESERVED_TYPE_NAMES + _RESERVED_REDUCTION_NAMES)
def test_argument_name_collides_with_reserved_name_fails(name):
    """Test that argument names cannot use reserved names."""
    module = _make_module_with_argument_named(name)

    with pytest.raises(ValidationFailedError, match="reserved"):
        run_validator(ReservedNameValidator(), module)


@pytest.mark.parametrize("name", _RESERVED_TYPE_NAMES + _RESERVED_REDUCTION_NAMES)
def test_declaration_variable_name_collides_with_reserved_name_fails(name):
    """Test declaration names cannot use reserved names."""
    module = _make_module_with_declaration_named(name)

    with pytest.raises(ValidationFailedError, match="reserved"):
        run_validator(ReservedNameValidator(), module)


def test_passes_with_names_that_only_resemble_reserved_names():
    """Test names that are prefixes/suffixes of reserved names are accepted."""
    # 'int32_helper', 'sums', 'expr' all *contain* reserved names as
    # substrings but are not themselves reserved.
    for name in ("int32_helper", "sums", "expr", "minimum"):
        module = _make_module_with_proc_named(name)
        run_validator(ReservedNameValidator(), module)
