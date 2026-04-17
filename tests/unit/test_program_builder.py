"""Test Driver Program Builder and Peripherals."""

import os
from pathlib import Path

import pytest
from fhy.driver import utils
from fhy.driver.ast_program_builder.builder import _ASTProgramBuilder
from fhy.driver.ast_program_builder.module_tree import ModuleTree
from fhy.driver.ast_program_builder.source_file_ast import SourceFileAST
from fhy.driver.compilation_options import CompilationOptions
from fhy.driver.utils import get_imported_symbol_module_components_and_name
from fhy.driver.workspace import Workspace
from fhy.error import FhYImportError
from fhy.ir.program import Program as IRProgram
from fhy.lang.ast.passes import collect_identifiers, collect_imported_identifiers
from fhy_core import SymbolTable


@pytest.fixture
def data_dir() -> str:
    path = os.path.join(__file__, os.pardir, "data")

    return os.path.abspath(path)


def _workspace(_path: str) -> Workspace:
    return Workspace(root=Path(_path))


@pytest.fixture
def circular_dir(data_dir) -> Workspace:
    return _workspace(os.path.join(data_dir, "circular_import", "a.fhy"))


@pytest.fixture
def unidirectional_import(data_dir) -> Workspace:
    return _workspace(os.path.join(data_dir, "unidirectional_import", "a.fhy"))


@pytest.fixture
def config() -> CompilationOptions:
    return CompilationOptions()


@pytest.mark.parametrize(["symbol", "expected"], [("a.b.c", (["a", "b"], "c"))])
def test_separation(symbol: str, expected: tuple[list[str], str]):
    result_a, result_b = utils.get_imported_symbol_module_components_and_name(symbol)

    assert result_a == expected[0], "Unexpected Import Components"
    assert result_b == expected[1], "Unexpected Import Name"


def test_program_instantiation():
    """Simple ir.Program Instantiation Check."""
    program = IRProgram()
    assert hasattr(
        program, "_components"
    ), 'Expected "_components" Attribute to ir.Program.'
    assert isinstance(
        program._components, dict
    ), "Expected _component Attribute to be a dictionary."

    assert hasattr(
        program, "_symbol_table"
    ), "Expected _symbol_table Attribute to ir.Program."
    assert isinstance(
        program._symbol_table, SymbolTable
    ), "Expected _component Attribute to be an ir.SymbolTable."


def test_builder_file_asts(unidirectional_import, config):
    program = _ASTProgramBuilder(unidirectional_import, config)
    ast_files = program._build_source_file_asts()

    assert isinstance(ast_files, list), "Expected to return a list"
    assert len(ast_files) == 2, f"Expected to detect 2 Files: {len(ast_files)}"
    assert all(
        isinstance(i, SourceFileAST) for i in ast_files
    ), "Expected SourceFileAST Objects"

    assert ast_files[0].path.name == "a.fhy"
    assert ast_files[1].path.name == "b.fhy"


def test_get_filepath_names(unidirectional_import, config):
    program = _ASTProgramBuilder(unidirectional_import, config)
    ast_files = program._build_source_file_asts()
    paths = {i.path for i in ast_files}

    result = {program._get_module_name_from_source_file_path(j) for j in paths}
    assert result == {
        "unidirectional_import.a",
        "unidirectional_import.b",
    }, "Expected Source Path names to match."


def test_get_path_from_symbol(unidirectional_import, config):
    symbol = "unidirectional_import.a.A"
    program = _ASTProgramBuilder(unidirectional_import, config)
    result = program._get_source_file_path_from_imported_symbol(symbol)

    assert result == unidirectional_import.main


def test_builder_module_tree(unidirectional_import, config):
    program = _ASTProgramBuilder(unidirectional_import, config)
    ast_files = program._build_source_file_asts()
    paths = {i.path for i in ast_files}

    result = program._build_module_tree(paths)
    assert isinstance(result, ModuleTree), "Expected to return ModuleTree Object"
    assert len(result.children) == 1, "Expected One Child"

    # Confirm Source Directory
    src_name = {i.name for i in result.children}
    assert src_name == {"root.unidirectional_import"}, "Unexpected Source Name"
    src_module_name = {i.module_name for i in result.children}
    assert src_module_name == {"unidirectional_import"}, "Unexpected Source Module Name"

    # Confirm Modules (children) of Source Directory
    src_dir_tree = next(iter(result.children))
    assert isinstance(src_dir_tree, ModuleTree), "Expected to return ModuleTree Object"

    assert len(src_dir_tree.children) == 2, "Expected Two Children"
    src_dir_names = {i.name for i in src_dir_tree.children}
    assert src_dir_names == {
        "root.unidirectional_import.a",
        "root.unidirectional_import.b",
    }, "Unexpected Source Name"
    src_dir_module_names = {i.module_name for i in src_dir_tree.children}
    assert src_dir_module_names == {"a", "b"}, "Unexpected Source Module Names"

    for child in src_dir_tree.children:
        assert isinstance(
            child, ModuleTree
        ), "Expected to Children to be ModuleTree Objects"
        assert len(child.children) == 0, "Expected children to be end leaf nodes."


def test_get_correct_module_by_name(unidirectional_import, config):
    program = _ASTProgramBuilder(unidirectional_import, config)
    ast_files = program._build_source_file_asts()
    paths = {i.path for i in ast_files}
    tree = program._build_module_tree(paths)

    ids = collect_imported_identifiers(ast_files[0].ast)
    assert len(ids) == 1, "Expected one ID."
    name = next(iter(ids)).name_hint

    result = program._get_module_by_name(tree, name)
    assert isinstance(result, ModuleTree), "Expected Module Tree"
    assert result.name != f"root.{name}", "Expected Different Names."
    route, symbol = get_imported_symbol_module_components_and_name(name)

    assert f"{result.name}.{symbol}" == f"root.{name}", "Expected Same Names"
    assert result.name == ".".join(("root", *route)), "Expected Same Names"


def test_identifier_validation(unidirectional_import, config):
    """Confirm that Identifiers are Correctly Replaced."""
    program = _ASTProgramBuilder(unidirectional_import, config)
    ast_files: list[SourceFileAST] = program._build_source_file_asts()
    paths: set[Path] = {i.path for i in ast_files}
    tree: ModuleTree = program._build_module_tree(paths)

    result: list[SourceFileAST] = program._resolve_imports(ast_files, tree)

    # Use sets to show differences in asts
    previous_ids = set()
    for j in ast_files:
        previous_ids.update(collect_identifiers(j.ast))
    current_ids = set()
    for k in result:
        current_ids.update(collect_identifiers(k.ast))

    assert len(previous_ids) > len(current_ids), "Expected IDs to be Replaced."
    difference = previous_ids.difference(current_ids)
    assert len(difference) == 1, "Expected One Identifier to be Replaced."

    identifier = next(iter(difference))
    assert (
        identifier.name_hint == "unidirectional_import.b.B"
    ), "UnExpected Identifier Replaced."


def test_identifier_validation_circular_import(circular_dir, config):
    """Raise ImportError when Encountering Circular Import."""
    program = _ASTProgramBuilder(circular_dir, config)
    ast_files = program._build_source_file_asts()
    paths = {i.path for i in ast_files}
    tree = program._build_module_tree(paths)

    with pytest.raises(FhYImportError):
        program._resolve_imports(ast_files, tree)


def test_program_build(unidirectional_import, config):
    builder = _ASTProgramBuilder(unidirectional_import, config)
    program = builder.build()

    assert isinstance(program, IRProgram), "Expected an Program to be built."
