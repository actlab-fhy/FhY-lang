"""Tests the operation validator AST pass."""

import pytest
from fhy.lang.ast import (
    Argument,
    FhYStructuralError,
    FhYTypeError,
    IdentifierExpression,
    IntLiteral,
    Module,
    Operation,
    Procedure,
    QualifiedType,
    ReturnStatement,
)
from fhy.lang.ast.passes import validate_operations
from fhy_core import (
    CoreDataType,
    Identifier,
    NumericalType,
    PassExecutionError,
    PrimitiveDataType,
    Provenance,
    TypeQualifier,
)
from fhy_core import (
    IdentifierExpression as CoreIdentifierExpression,
)


def _qt(type_, qualifier: TypeQualifier) -> QualifiedType:
    return QualifiedType(
        base_type=type_,
        type_qualifier=qualifier,
        provenance=Provenance.unknown(),
    )


def _scalar_int32() -> NumericalType:
    return NumericalType(PrimitiveDataType(CoreDataType.INT32))


def test_empty_program():
    """Test validation of an empty program."""
    program_ast = Module(provenance=Provenance.unknown())

    validate_operations(program_ast)


def test_valid_scalar_operation():
    """Test a valid scalar operation passes."""
    op = Identifier("op")
    a = Identifier("a")
    program_ast = Module(
        statements=(
            Operation(
                name=op,
                args=(
                    Argument(
                        name=a,
                        qualified_type=_qt(_scalar_int32(), TypeQualifier.INPUT),
                        provenance=Provenance.unknown(),
                    ),
                ),
                body=(
                    ReturnStatement(
                        expression=IdentifierExpression(
                            identifier=a, provenance=Provenance.unknown()
                        ),
                        provenance=Provenance.unknown(),
                    ),
                ),
                return_type=_qt(_scalar_int32(), TypeQualifier.OUTPUT),
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )

    validate_operations(program_ast)


def test_fails_with_non_scalar_argument():
    """Test failure when an operation takes a non-scalar argument."""
    op = Identifier("op")
    m = Identifier("m")
    a = Identifier("a")
    vector_int32 = NumericalType(
        PrimitiveDataType(CoreDataType.INT32),
        shape=(CoreIdentifierExpression(m),),
    )
    program_ast = Module(
        statements=(
            Operation(
                name=op,
                args=(
                    Argument(
                        name=a,
                        qualified_type=_qt(vector_int32, TypeQualifier.INPUT),
                        provenance=Provenance.unknown(),
                    ),
                ),
                body=(
                    ReturnStatement(
                        expression=IntLiteral(value=0, provenance=Provenance.unknown()),
                        provenance=Provenance.unknown(),
                    ),
                ),
                return_type=_qt(_scalar_int32(), TypeQualifier.OUTPUT),
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )

    with pytest.raises(PassExecutionError, match=FhYTypeError.__name__):
        validate_operations(program_ast)


def test_fails_with_non_scalar_return_type():
    """Test failure when the operation return type is not a scalar."""
    op = Identifier("op")
    m = Identifier("m")
    a = Identifier("a")
    vector_int32 = NumericalType(
        PrimitiveDataType(CoreDataType.INT32),
        shape=(CoreIdentifierExpression(m),),
    )
    program_ast = Module(
        statements=(
            Operation(
                name=op,
                args=(
                    Argument(
                        name=a,
                        qualified_type=_qt(_scalar_int32(), TypeQualifier.INPUT),
                        provenance=Provenance.unknown(),
                    ),
                ),
                body=(
                    ReturnStatement(
                        expression=IdentifierExpression(
                            identifier=a, provenance=Provenance.unknown()
                        ),
                        provenance=Provenance.unknown(),
                    ),
                ),
                return_type=_qt(vector_int32, TypeQualifier.OUTPUT),
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )

    with pytest.raises(PassExecutionError, match=FhYTypeError.__name__):
        validate_operations(program_ast)


def test_fails_with_non_output_return_qualifier():
    """Test failure when the operation's return type qualifier is not OUTPUT."""
    op = Identifier("op")
    a = Identifier("a")
    program_ast = Module(
        statements=(
            Operation(
                name=op,
                args=(
                    Argument(
                        name=a,
                        qualified_type=_qt(_scalar_int32(), TypeQualifier.INPUT),
                        provenance=Provenance.unknown(),
                    ),
                ),
                body=(
                    ReturnStatement(
                        expression=IdentifierExpression(
                            identifier=a, provenance=Provenance.unknown()
                        ),
                        provenance=Provenance.unknown(),
                    ),
                ),
                return_type=_qt(_scalar_int32(), TypeQualifier.INPUT),
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )

    with pytest.raises(PassExecutionError, match=FhYTypeError.__name__):
        validate_operations(program_ast)


def test_fails_with_missing_return_statement():
    """Test failure when an operation body has no return statement."""
    op = Identifier("op")
    a = Identifier("a")
    program_ast = Module(
        statements=(
            Operation(
                name=op,
                args=(
                    Argument(
                        name=a,
                        qualified_type=_qt(_scalar_int32(), TypeQualifier.INPUT),
                        provenance=Provenance.unknown(),
                    ),
                ),
                body=(),
                return_type=_qt(_scalar_int32(), TypeQualifier.OUTPUT),
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )

    with pytest.raises(PassExecutionError, match=FhYStructuralError.__name__):
        validate_operations(program_ast)


def test_procedure_is_not_validated_as_operation():
    """Test that a procedure (with no return, tensor args) is not flagged."""
    main = Identifier("main")
    m = Identifier("m")
    a = Identifier("a")
    vector_int32 = NumericalType(
        PrimitiveDataType(CoreDataType.INT32),
        shape=(CoreIdentifierExpression(m),),
    )
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(
                    Argument(
                        name=a,
                        qualified_type=_qt(vector_int32, TypeQualifier.INPUT),
                        provenance=Provenance.unknown(),
                    ),
                ),
                body=(),
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )

    validate_operations(program_ast)
