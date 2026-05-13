"""Tests the module-scope validator AST pass."""

import pytest
from fhy_core import (
    CoreDataType,
    Identifier,
    LiteralExpression,
    NumericalType,
    PrimitiveDataType,
    TypeQualifier,
    ValidationFailedError,
)

from fhy_lang import TupleType
from fhy_lang.ast import (
    BinaryExpression,
    BinaryOperation,
    DeclarationStatement,
    ExpressionStatement,
    ForAllStatement,
    IdentifierExpression,
    IntLiteral,
    Module,
    Operation,
    Procedure,
    QualifiedType,
    ReturnStatement,
)
from fhy_lang.ast.passes import ModuleScopeValidator

from .utils import run_validator


def _int32_scalar() -> NumericalType:
    return NumericalType(PrimitiveDataType(CoreDataType.INT32))


def _int32_tensor() -> NumericalType:
    return NumericalType(
        PrimitiveDataType(CoreDataType.INT32), shape=(LiteralExpression(4),)
    )


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


def _make_param_scalar(
    name: Identifier, initializer_value: int
) -> DeclarationStatement:
    return DeclarationStatement(
        variable_name=name,
        variable_type=QualifiedType(
            base_type=_int32_scalar(), type_qualifier=TypeQualifier.PARAM
        ),
        expression=IntLiteral(value=initializer_value),
    )


def test_module_with_only_main_passes():
    """Test a module containing only a single ``proc main`` passes."""
    module = Module(statements=(_make_proc(Identifier("main")),))

    run_validator(ModuleScopeValidator(), module)


def test_module_with_param_op_main_passes():
    """Test a module with a module-scope ``param``, an ``op``, and a
    ``proc main`` passes."""
    module = Module(
        statements=(
            _make_param_scalar(Identifier("K"), 4),
            _make_op(Identifier("sigmoid")),
            _make_proc(Identifier("main")),
        ),
    )

    run_validator(ModuleScopeValidator(), module)


def test_fails_on_non_param_declaration_at_module_scope():
    """Test rejection of a non-``param`` declaration at module scope."""
    module = Module(
        statements=(
            DeclarationStatement(
                variable_name=Identifier("x"),
                variable_type=QualifiedType(
                    base_type=_int32_scalar(), type_qualifier=TypeQualifier.TEMP
                ),
                expression=IntLiteral(value=0),
            ),
        ),
    )

    with pytest.raises(ValidationFailedError, match="only 'param' declarations"):
        run_validator(ModuleScopeValidator(), module)


def test_fails_on_param_of_tensor_type_at_module_scope():
    """Test rejection of a ``param`` of tensor type at module scope."""
    module = Module(
        statements=(
            DeclarationStatement(
                variable_name=Identifier("X"),
                variable_type=QualifiedType(
                    base_type=_int32_tensor(), type_qualifier=TypeQualifier.PARAM
                ),
                expression=IntLiteral(value=0),
            ),
        ),
    )

    with pytest.raises(ValidationFailedError, match="must be a scalar"):
        run_validator(ModuleScopeValidator(), module)


def test_fails_on_param_of_tuple_type_at_module_scope():
    """Test rejection of a ``param`` of tuple type at module scope."""
    module = Module(
        statements=(
            DeclarationStatement(
                variable_name=Identifier("T"),
                variable_type=QualifiedType(
                    base_type=TupleType(types=[_int32_scalar(), _int32_scalar()]),
                    type_qualifier=TypeQualifier.PARAM,
                ),
                expression=IntLiteral(value=0),
            ),
        ),
    )

    with pytest.raises(ValidationFailedError, match="scalar numerical type"):
        run_validator(ModuleScopeValidator(), module)


def test_fails_on_param_without_initializer_at_module_scope():
    """Test rejection of a ``param`` declaration without initializer."""
    module = Module(
        statements=(
            DeclarationStatement(
                variable_name=Identifier("K"),
                variable_type=QualifiedType(
                    base_type=_int32_scalar(), type_qualifier=TypeQualifier.PARAM
                ),
            ),
        ),
    )

    with pytest.raises(
        ValidationFailedError, match="compile-time-constant initializer"
    ):
        run_validator(ModuleScopeValidator(), module)


def test_fails_on_assignment_at_module_scope():
    """Test rejection of an assignment statement at module scope."""
    module = Module(
        statements=(
            ExpressionStatement(
                left=IdentifierExpression(identifier=Identifier("x")),
                right=IntLiteral(value=0),
            ),
        ),
    )

    with pytest.raises(ValidationFailedError, match="not allowed at module scope"):
        run_validator(ModuleScopeValidator(), module)


def test_fails_on_forall_at_module_scope():
    """Test rejection of a ``forall`` statement at module scope."""
    module = Module(
        statements=(
            ForAllStatement(
                index=IdentifierExpression(identifier=Identifier("i")),
                body=(),
            ),
        ),
    )

    with pytest.raises(ValidationFailedError, match="'forall' is not allowed"):
        run_validator(ModuleScopeValidator(), module)


def test_fails_on_bare_expression_at_module_scope():
    """Test rejection of a bare expression statement at module scope."""
    module = Module(
        statements=(
            ExpressionStatement(
                left=None,
                right=BinaryExpression(
                    operation=BinaryOperation.ADDITION,
                    left=IntLiteral(value=1),
                    right=IntLiteral(value=1),
                ),
            ),
        ),
    )

    with pytest.raises(ValidationFailedError, match="not allowed at module scope"):
        run_validator(ModuleScopeValidator(), module)


def test_passes_with_multiple_module_scope_params_and_functions():
    """Test multiple ``param`` scalars + multiple functions all pass."""
    module = Module(
        statements=(
            _make_param_scalar(Identifier("M"), 4),
            _make_param_scalar(Identifier("N"), 8),
            _make_op(Identifier("foo")),
            _make_op(Identifier("bar")),
            _make_proc(Identifier("main")),
        ),
    )

    run_validator(ModuleScopeValidator(), module)
