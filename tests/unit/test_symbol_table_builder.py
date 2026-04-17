"""Unit Test Symbol Table Builder Module."""

from fhy.lang.ast import Module, Procedure
from fhy.lang.ast.passes import build_symbol_table
from fhy_core import Identifier, Provenance


def test_empty_program():
    """Tests an empty program."""
    program_ast = Module(provenance=Provenance.unknown())
    module_name = program_ast.name

    symbol_table = build_symbol_table(program_ast)

    error_message: str = "Expected 2 namespaces, got "
    error_message += f"{symbol_table.get_number_of_namespaces()}."
    assert symbol_table.get_number_of_namespaces() == 2, error_message
    module_namespace = symbol_table.get_namespace(module_name)
    error_message = "Expected 0 symbols in the module, "
    error_message += f"got {len(module_namespace)}."
    assert len(module_namespace) == 0, error_message


def test_empty_procedure():
    """Tests empty procedure body containing procedure name in symbol table."""
    function_name = Identifier("main")
    program_ast = Module(
        statements=(
            Procedure(
                name=function_name,
                templates=[],
                args=[],
                body=[],
                provenance=Provenance.unknown(),
            ),
        ),
        provenance=Provenance.unknown(),
    )
    module_name = program_ast.name

    symbol_table = build_symbol_table(program_ast)

    error_message: str = "Expected 3 namespaces, "
    error_message += f"got {symbol_table.get_number_of_namespaces()}."
    assert symbol_table.get_number_of_namespaces() == 3, error_message
    module_namespace_symbol_table = symbol_table.get_namespace(module_name)
    error_message: str = f"Expected 1 symbol, got {len(module_namespace_symbol_table)}."
    assert len(module_namespace_symbol_table) == 1, error_message
    main_namespace_symbol_table = symbol_table.get_namespace(function_name)
    assert len(main_namespace_symbol_table) == 0


# @pytest.mark.skip()
# def test_procedure_with_arguments(construct_ast):
#     """Test Empty Procedure with Arguments.

#     names are in the symbol table, but within procedure namespace.

#     """
#     source_file_content = "proc main(input int32[A, B] a, input int32[A, C] b) {}"
#     _ast = construct_ast(source_file_content)

#     symbol_table = build_symbol_table(_ast)

#     assert len(symbol_table) == 2

#     module_namespace_symbol_table = next(iter(symbol_table.values()))
#     assert len(module_namespace_symbol_table) == 1
#     assert "main" in _get_symbol_table_string_keys(module_namespace_symbol_table)

#     main_namespace_symbol_table = list(symbol_table.values())[1]
#     assert len(main_namespace_symbol_table) == 5
#     for char in "abABC":
#         assert char in _get_symbol_table_string_keys(
#             main_namespace_symbol_table
#         ), f'Expected Variable in Symbol table: "{char}"'


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
