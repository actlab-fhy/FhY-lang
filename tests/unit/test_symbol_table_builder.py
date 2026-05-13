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
    TypeQualifier,
    VariableSymbolTableFrame,
)
from fhy_core import (
    IdentifierExpression as CoreIdentifierExpression,
)

from fhy_lang.ast import (
    Argument,
    DeclarationStatement,
    ForAllStatement,
    IdentifierExpression,
    IntLiteral,
    Module,
    Procedure,
    QualifiedType,
)
from fhy_lang.ast.passes import build_symbol_table
from fhy_lang.ast.passes.symbol_table_builder import FhYSymbolTableBuilderError


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
            ),
        ),
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
                        ),
                    ),
                    Argument(
                        name=b,
                        qualified_type=QualifiedType(
                            base_type=NumericalType(
                                PrimitiveDataType(CoreDataType.INT64)
                            ),
                            type_qualifier=TypeQualifier.INPUT,
                        ),
                    ),
                ),
                body=(),
            ),
        ),
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
                        ),
                    ),
                ),
            ),
        ),
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
                        ),
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
                        ),
                    ),
                ),
            ),
        ),
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
            ),
            Procedure(
                name=main,
                templates=(),
                args=(),
                body=(),
            ),
        ),
    )
    with pytest.raises(PassExecutionError, match=FhYSymbolTableBuilderError.__name__):
        build_symbol_table(program_ast)


def test_shape_identifier_resolves_to_existing_module_param(int32):
    """Test a shape identifier used in an argument type resolves to an
    existing module-level ``param`` instead of being implicitly redeclared
    in the function namespace."""
    main = Identifier("main")
    a = Identifier("a")
    n = Identifier("N")
    program_ast = Module(
        statements=(
            DeclarationStatement(
                variable_name=n,
                variable_type=QualifiedType(
                    base_type=NumericalType(PrimitiveDataType(CoreDataType.UINT32)),
                    type_qualifier=TypeQualifier.PARAM,
                ),
                expression=IntLiteral(value=8),
            ),
            Procedure(
                name=main,
                templates=(),
                args=(
                    Argument(
                        name=a,
                        qualified_type=QualifiedType(
                            base_type=NumericalType(
                                PrimitiveDataType(CoreDataType.INT32),
                                shape=(CoreIdentifierExpression(n),),
                            ),
                            type_qualifier=TypeQualifier.INPUT,
                        ),
                    ),
                ),
                body=(),
            ),
        ),
    )

    symbol_table = build_symbol_table(program_ast)

    module_namespace = symbol_table.get_namespace(program_ast.name)
    assert n in module_namespace, (
        "Module-level 'param' N should be registered in the module namespace."
    )
    assert main in module_namespace
    # N must not be implicitly re-declared inside the procedure namespace.
    main_namespace = symbol_table.get_namespace(main)
    assert n not in main_namespace, (
        "Module-level 'param' N should be reused; the builder should not "
        "introduce a second implicit symbol for it inside the procedure."
    )


def test_forall_introduces_new_namespace_scope(int32):
    """Test that a ``forall`` body lives in its own namespace, distinct
    from the enclosing procedure's namespace."""
    main = Identifier("main")
    i = Identifier("i")
    y = Identifier("y")
    forall_name = Identifier("forall_body")
    forall = ForAllStatement(
        name=forall_name,
        index=IdentifierExpression(identifier=i),
        body=(
            DeclarationStatement(
                variable_name=y,
                variable_type=QualifiedType(
                    base_type=int32,
                    type_qualifier=TypeQualifier.TEMP,
                ),
            ),
        ),
    )
    program_ast = Module(
        statements=(
            Procedure(
                name=main,
                templates=(),
                args=(),
                body=(forall,),
            ),
        ),
    )

    symbol_table = build_symbol_table(program_ast)

    main_namespace = symbol_table.get_namespace(main)
    forall_namespace = symbol_table.get_namespace(forall_name)
    assert y in forall_namespace, "'y' should live in the forall's namespace."
    assert y not in main_namespace, (
        "'y' should NOT leak into the enclosing procedure's namespace."
    )
