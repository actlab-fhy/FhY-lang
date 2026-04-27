"""FhY program builder module."""

import logging

from fhy_core import (
    Position,
    Provenance,
    Span,
    get_logger,
)

from fhy_lang import ASTModule, from_fhy_source

from ..compilation_options import CompilationOptions
from ..workspace import Workspace

_logger: logging.Logger = get_logger(__name__)


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

    def build(self) -> ASTModule:
        with open(self._workspace.source_file) as f:
            source_text = f.read()
        lines = source_text.splitlines()
        _logger.debug(
            "Read source file %s: %d bytes, %d lines",
            self._workspace.source_file,
            len(source_text),
            len(lines),
        )
        if len(lines) == 0:
            _logger.warning("Source file %s is empty.", self._workspace.source_file)
            span = Span(file_path=self._workspace.source_file)
        else:
            span = Span(
                file_path=self._workspace.source_file,
                start_position=Position(line=1, column=1),
                end_position=Position(
                    line=len(lines),
                    column=len(lines[-1]),
                ),
            )
        return from_fhy_source(source_text, Provenance(span=span))


def build_ast_program(workspace: Workspace, options: CompilationOptions) -> ASTModule:
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
