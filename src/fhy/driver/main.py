"""Main compilation driver."""

import logging

from fhy_core import get_logger

from fhy.lang.ast import Module

from .ast_program_builder import build_ast_program
from .compilation_options import CompilationOptions
from .workspace import Workspace

_logger: logging.Logger = get_logger(__name__)


def compile_fhy(workspace: Workspace, options: CompilationOptions) -> Module:
    """Compile a FhY program.

    Args:
        workspace: The workspace containing the FhY  program.
        options: The compilation options.

    Returns:
        The compiled FhY program.

    """
    _logger.info("Compiling the FhY program...")
    ast_program = build_ast_program(workspace, options)
    _logger.info("FhY program compilation finished successfully.")
    return ast_program
