"""Unit Test Symbol Table Builder Module."""

from fhy.lang.ast import Argument, Module, Procedure, QualifiedType
from fhy.lang.ast.passes import build_symbol_table
from fhy_core import (
    CoreDataType,
    FunctionKeyword,
    FunctionSymbolTableFrame,
    Identifier,
    NumericalType,
    PrimitiveDataType,
    Provenance,
    TypeQualifier,
    VariableSymbolTableFrame,
)


def test_empty_program():
    """Tests an empty program."""
    program_ast = Module(provenance=Provenance.unknown())

    symbol_table = build_symbol_table(program_ast)

    assert symbol_table.get_number_of_namespaces() == 2
    module_namespace = symbol_table.get_namespace(program_ast.name)
    assert len(module_namespace) == 0


def test_empty_procedure():
    """Tests empty procedure body containing procedure name in symbol table."""
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


# @pytest.mark.skip()
# def test_procedure_with_declaration_statement(construct_ast):
#     """Test procedure body variables are in symbol table procedure namespace."""
#     source_file_content = "proc main(input int32[A, B] a) {temp int32[A] b;}"
#     _ast = construct_ast(source_file_content)

#     symbol_table = build_symbol_table(_ast)

#     assert len(symbol_table) == 2

#     module_namespace_symbol_table = next(iter(symbol_table.values()))
#     assert len(module_namespace_symbol_table) == 1
#     assert "main" in _get_symbol_table_string_keys(module_namespace_symbol_table)

#     main_namespace_symbol_table = list(symbol_table.values())[1]
#     assert len(main_namespace_symbol_table) == 4

#     for char in "abAB":
#         assert char in _get_symbol_table_string_keys(
#             main_namespace_symbol_table
#         ), f'Expected Variable in Symbol table: "{char}"'


# @pytest.mark.skip()
# def test_fails_with_undefined_shape_variable(construct_ast):
#     """Tests an error is raised with an undeclared shape variable, 'C'."""
#     source_file_content = "proc main(input int32[A, B] a) {temp int32[C] b;}"
#     _ast = construct_ast(source_file_content)

#     with pytest.raises(error.FhYSemanticsError):
#         build_symbol_table(_ast)


# @pytest.mark.skip()
# def test_fails_with_already_defined_variable(construct_ast):
#     """Tests redefinition of a variable raises an error."""
#     source_file_content = "proc main(input int32[A, B] a) {temp int32[A] a;}"
#     _ast = construct_ast(source_file_content)

#     with pytest.raises(error.FhYSemanticsError):
#         build_symbol_table(_ast)


# @pytest.mark.skip()
# def test_fails_with_already_defined_procedure(construct_ast):
#     """Tests that redefining a procedure name raises an error."""
#     source_file_content = "proc main() {} proc main() {}"
#     _ast = construct_ast(source_file_content)

#     with pytest.raises(error.FhYSemanticsError):
#         build_symbol_table(_ast)


# @pytest.mark.skip()
# def test_import_variable(construct_ast):
#     """Test import and usage of a variable.

#     Variable should be present at both the module and procedure level (namespace).

#     """
#     source_file_content = (
#         "import constants.pi; proc main() {temp int32 a = constants.pi;}"
#     )
#     _ast = construct_ast(source_file_content)

#     symbol_table = build_symbol_table(_ast)

#     assert len(symbol_table) == 2

#     module_namespace_symbol_table = next(iter(symbol_table.values()))
#     assert len(module_namespace_symbol_table) == 2
#     assert "main" in _get_symbol_table_string_keys(module_namespace_symbol_table)
#     assert "constants.pi" in _get_symbol_table_string_keys(
#         module_namespace_symbol_table
#     )

#     main_namespace_symbol_table = list(symbol_table.values())[1]
#     assert len(main_namespace_symbol_table) == 1
#     assert "a" in _get_symbol_table_string_keys(main_namespace_symbol_table)


# def test_fails_with_already_defined_import(construct_ast):
#     """Tests that reimporting the same variable raises an error."""
#     source_file_content = "import constants.pi; import constants.pi;"
#     _ast = construct_ast(source_file_content)

#     with pytest.raises(error.FhYSemanticsError):
#         build_symbol_table(_ast)
