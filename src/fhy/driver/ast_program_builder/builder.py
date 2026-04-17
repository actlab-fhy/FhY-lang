"""FhY program builder module."""

import logging

from fhy_core import (
    Position,
    Provenance,
    Span,
)

from fhy.lang.ast import Module
from fhy.lang.converter import from_fhy_source

from ..compilation_options import CompilationOptions
from ..workspace import Workspace

_logger: logging.Logger = logging.getLogger(__name__)


class _ASTProgramBuilder:
    _workspace: Workspace
    _options: CompilationOptions

    def __init__(
        self,
        workspace: Workspace,
        options: CompilationOptions,
    ):
        self._workspace = workspace
        self._options = options

    def build(self) -> Module:
        with open(self._workspace.source_file) as f:
            source_text = f.read()
        span = Span(
            file_path=self._workspace.source_file,
            start_position=Position(line=1, column=1),
            end_position=Position(
                line=len(source_text.splitlines()),
                column=len(source_text.splitlines()[-1]),
            ),
        )
        return from_fhy_source(source_text, Provenance(span=span))


def build_ast_program(workspace: Workspace, options: CompilationOptions) -> Module:
    """Build an AST Module.

    Args:
        workspace: The workspace containing the FhY program.
        options: The compilation options.

    Returns:
        The compiled FhY program.

    """
    builder = _ASTProgramBuilder(workspace, options)

    _logger.info("Building the FhY AST module...")
    module = builder.build()
    _logger.info("FhY AST module built successfully.")
    return module
