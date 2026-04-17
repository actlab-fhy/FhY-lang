# Copyright (c) 2024 FhY Developers
# Christopher Priebe <cpriebe@ucsd.edu>
# Jason C Del Rio <j3delrio@ucsd.edu>
# Hadi S Esmaeilzadeh <hadi@ucsd.edu>
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are
# permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of
# conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list
# of conditions and the following disclaimer in the documentation and/or other materials
# provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be
# used to endorse or promote products derived from this software without specific prior
# written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
# SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.

"""FhY builder module.

Orchestrate the build of a FhY source into an ir.Program.

"""

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
