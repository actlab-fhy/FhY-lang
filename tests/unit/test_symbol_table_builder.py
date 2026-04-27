"""Tests the symbol table builder AST pass."""

import pytest
from fhy_core import (
    CoreDataType,
    FunctionKeyword,
    FunctionSymbolTableFrame,
    Identifier,
    NumericalType,
    PassExecutionError,
    PrimitiveDataType,
    Provenance,
    TypeQualifier,
    VariableSymbolTableFrame,
)

from fhy_lang.lang.ast import (
    Argument,
    DeclarationStatement,
    Module,
    Procedure,
    QualifiedType,
)
from fhy_lang.lang.ast.passes import build_symbol_table
from fhy_lang.lang.ast.passes.symbol_table_builder import FhYSymbolTableBuilderError


def test_empty_module(empty_module_ast):
    """Test an empty module."""
    symbol_table = build_symbol_table(empty_module_ast)

    assert symbol_table.get_number_of_namespaces() == 2
    module_namespace = symbol_table.get_namespace(empty_module_ast.name)
    assert len(module_namespace) == 0


def test_empty_procedure():
    """Test empty procedure body containing procedure name in symbol table."""
    main = Identifier("main")
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                templates=(),
                args=(),
                body=(),
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )

    symbol_table = build_symbol_table(program_ast)

    assert symbol_table.get_number_of_namespaces() == 3
    module_namespace_symbol_table = symbol_table.get_namespace(program_ast.name)
    assert len(module_namespace_symbol_table) == 1
    assert main in module_namespace_symbol_table
    assert symbol_table.is_symbol_defined_in_namespace(program_ast.name, main)
    frame = symbol_table.get_frame_from_namespace(program_ast.name, main)
    assert isinstance(frame, FunctionSymbolTableFrame)
    assert frame.name == main
    assert frame.keyword == FunctionKeyword.PROCEDURE
    assert frame.signature == ()
    main_namespace_symbol_table = symbol_table.get_namespace(main)
    assert len(main_namespace_symbol_table) == 0


def test_procedure_with_arguments():
    """Test procedure with arguments."""
    main = Identifier("main")
    a, b = Identifier("a"), Identifier("b")
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                templates=(),
                args=(
                    Argument(
                        name=a,
                        qualified_type=QualifiedType(
                            base_type=NumericalType(
                                PrimitiveDataType(CoreDataType.INT32)
                            ),
                            type_qualifier=TypeQualifier.INPUT,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                    Argument(
                        name=b,
                        qualified_type=QualifiedType(
                            base_type=NumericalType(
                                PrimitiveDataType(CoreDataType.INT64)
                            ),
                            type_qualifier=TypeQualifier.INPUT,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                ),
                body=(),
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )

    symbol_table = build_symbol_table(program_ast)

    assert symbol_table.get_number_of_namespaces() == 3
    module_namespace = symbol_table.get_namespace(program_ast.name)
    assert len(module_namespace) == 1
    assert main in module_namespace
    assert symbol_table.is_symbol_defined_in_namespace(program_ast.name, main)
    frame = symbol_table.get_frame_from_namespace(program_ast.name, main)
    assert isinstance(frame, FunctionSymbolTableFrame)
    assert frame.name == main
    assert frame.keyword == FunctionKeyword.PROCEDURE
    assert len(frame.signature) == 2
    assert frame.signature[0][0] == TypeQualifier.INPUT
    assert frame.signature[0][1].is_structurally_equivalent(
        NumericalType(PrimitiveDataType(CoreDataType.INT32))
    )
    assert frame.signature[1][0] == TypeQualifier.INPUT
    assert frame.signature[1][1].is_structurally_equivalent(
        NumericalType(PrimitiveDataType(CoreDataType.INT64))
    )
    main_namespace = symbol_table.get_namespace(main)
    assert len(main_namespace) == 2
    assert a in main_namespace and symbol_table.is_symbol_defined_in_namespace(main, a)
    assert b in main_namespace and symbol_table.is_symbol_defined_in_namespace(main, b)
    a_frame = symbol_table.get_frame_from_namespace(main, a)
    assert isinstance(a_frame, VariableSymbolTableFrame)
    assert a_frame.name == a
    assert a_frame.type.is_structurally_equivalent(
        NumericalType(PrimitiveDataType(CoreDataType.INT32))
    )
    assert a_frame.type_qualifier == TypeQualifier.INPUT
    b_frame = symbol_table.get_frame_from_namespace(main, b)
    assert isinstance(b_frame, VariableSymbolTableFrame)
    assert b_frame.name == b
    assert b_frame.type.is_structurally_equivalent(
        NumericalType(PrimitiveDataType(CoreDataType.INT64))
    )
    assert b_frame.type_qualifier == TypeQualifier.INPUT


def test_procedure_with_declaration_statement():
    """Test procedure with declaration statement."""
    main = Identifier("main")
    a = Identifier("a")
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                templates=(),
                args=(),
                body=(
                    DeclarationStatement(
                        variable_name=a,
                        variable_type=QualifiedType(
                            base_type=NumericalType(
                                PrimitiveDataType(CoreDataType.INT32)
                            ),
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
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

    main_namespace = symbol_table.get_namespace(main)
    assert len(main_namespace) == 1
    assert a in main_namespace and symbol_table.is_symbol_defined_in_namespace(main, a)
    a_frame = symbol_table.get_frame_from_namespace(main, a)
    assert isinstance(a_frame, VariableSymbolTableFrame)
    assert a_frame.name == a
    assert a_frame.type.is_structurally_equivalent(
        NumericalType(PrimitiveDataType(CoreDataType.INT32))
    )
    assert a_frame.type_qualifier == TypeQualifier.TEMP


def test_fails_with_already_defined_variable():
    """Test failure with already defined variable."""
    main = Identifier("main")
    a = Identifier("a")
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                templates=(),
                args=(
                    Argument(
                        name=a,
                        qualified_type=QualifiedType(
                            base_type=NumericalType(
                                PrimitiveDataType(CoreDataType.INT32)
                            ),
                            type_qualifier=TypeQualifier.INPUT,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                ),
                body=(
                    DeclarationStatement(
                        variable_name=a,
                        variable_type=QualifiedType(
                            base_type=NumericalType(
                                PrimitiveDataType(CoreDataType.INT32)
                            ),
                            type_qualifier=TypeQualifier.TEMP,
                            provenance=Provenance.unknown(),
                        ),
                        provenance=Provenance.unknown(),
                    ),
                ),
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )
    with pytest.raises(PassExecutionError, match=FhYSymbolTableBuilderError.__name__):
        build_symbol_table(program_ast)


def test_fails_with_already_defined_procedure():
    """Test failure with already defined procedure."""
    main = Identifier("main")
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                templates=(),
                args=(),
                body=(),
                provenance=Provenance.unknown(),
            ),
            Procedure(
                name=main,
                templates=(),
                args=(),
                body=(),
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )
    with pytest.raises(PassExecutionError, match=FhYSymbolTableBuilderError.__name__):
        build_symbol_table(program_ast)
