"""Tests the for-all statement validator AST pass."""

import pytest
from fhy_core import (
    Identifier,
    IndexType,
    LiteralExpression,
    NumericalType,
    Provenance,
    TypeQualifier,
    ValidationFailedError,
)
from fhy_lang.lang.ast import (
    DeclarationStatement,
    ForAllStatement,
    IdentifierExpression,
    IntLiteral,
    Module,
    Procedure,
    QualifiedType,
)
from fhy_lang.lang.ast.passes import (
    ForAllStatementValidator,
    build_symbol_table,
)

from .utils import run_validator


@pytest.fixture
def dummy_index_type() -> IndexType:
    return IndexType(
        lower_bound=LiteralExpression(1),
        upper_bound=LiteralExpression(10),
        stride=None,
    )


def test_empty_module(empty_module_ast):
    """Test validation of an empty module."""
    symbol_table = build_symbol_table(empty_module_ast)

    run_validator(ForAllStatementValidator(symbol_table), empty_module_ast)


def test_valid_for_all_with_index_variable(dummy_index_type: IndexType):
    """Test a for-all statement with an index variable as its index."""
    main = Identifier("main")
    i = Identifier("i")
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(),
                body=(
                    DeclarationStatement(
                        variable_name=i,
                        variable_type=QualifiedType(
                            base_type=dummy_index_type,
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    ForAllStatement(
                        index=IdentifierExpression(
                            identifier=i, provenance=Provenance.unknown()
                        ),
                        body=(),
                        provenance=Provenance.unknown(),
                    ),
                ),
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )
    symbol_table = build_symbol_table(program_ast)

    run_validator(ForAllStatementValidator(symbol_table), program_ast)


def test_fails_with_non_identifier_index(int32: NumericalType):
    """Test failure when the for-all index expression is not an identifier."""
    main = Identifier("main")
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(),
                body=(
                    ForAllStatement(
                        index=IntLiteral(value=0, provenance=Provenance.unknown()),
                        body=(),
                        provenance=Provenance.unknown(),
                    ),
                ),
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(ValidationFailedError, match="structural error"):
        run_validator(ForAllStatementValidator(symbol_table), program_ast)


def test_fails_with_non_index_variable(int32: NumericalType):
    """Test failure when the for-all index identifier is not an index variable."""
    main = Identifier("main")
    t = Identifier("t")
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(),
                body=(
                    DeclarationStatement(
                        variable_name=t,
                        variable_type=QualifiedType(
                            base_type=int32,
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    ForAllStatement(
                        index=IdentifierExpression(
                            identifier=t, provenance=Provenance.unknown()
                        ),
                        body=(),
                        provenance=Provenance.unknown(),
                    ),
                ),
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )
    symbol_table = build_symbol_table(program_ast)

    with pytest.raises(ValidationFailedError, match="type error"):
        run_validator(ForAllStatementValidator(symbol_table), program_ast)


def test_valid_nested_for_all_statements(dummy_index_type: IndexType):
    """Test nested for-all statements with distinct index variables."""
    main = Identifier("main")
    i, j = Identifier("i"), Identifier("j")
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                args=(),
                body=(
                    DeclarationStatement(
                        variable_name=i,
                        variable_type=QualifiedType(
                            base_type=dummy_index_type,
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    DeclarationStatement(
                        variable_name=j,
                        variable_type=QualifiedType(
                            base_type=dummy_index_type,
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    ForAllStatement(
                        name=Identifier("forall_i"),
                        index=IdentifierExpression(
                            identifier=i, provenance=Provenance.unknown()
                        ),
                        body=(
                            ForAllStatement(
                                name=Identifier("forall_j"),
                                index=IdentifierExpression(
                                    identifier=j,
                                    provenance=Provenance.unknown(),
                                ),
                                body=(),
                                provenance=Provenance.unknown(),
                            ),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                ),
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )
    symbol_table = build_symbol_table(program_ast)

    run_validator(ForAllStatementValidator(symbol_table), program_ast)
