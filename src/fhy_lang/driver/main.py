"""Main compilation driver."""

import logging

from fhy_core import SymbolTable, get_logger

from fhy_lang.lang import ASTModule, validate_ast

from .ast_program_builder import build_ast_program
from .compilation_options import CompilationOptions
from .workspace import Workspace

_logger: logging.Logger = get_logger(__name__)


def compile_fhy(
    workspace: Workspace, options: CompilationOptions
) -> tuple[ASTModule, SymbolTable]:
    """Compile a FhY program.

    Args:
        workspace: The workspace containing the FhY  program.
        options: The compilation options.

    Returns:
        The compiled FhY program.

    """
    _logger.info("Compiling the FhY program...")
    ast_program = build_ast_program(workspace, options)
    ast_program, symbol_table = validate_ast(
        ast_program, perform_optimizations=options.perform_optimizations
    )
    _logger.info("FhY program compilation finished successfully.")
    return ast_program, symbol_table
