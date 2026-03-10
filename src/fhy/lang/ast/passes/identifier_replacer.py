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

"""Identifier replacement transformer."""

from copy import copy
from typing import cast

from fhy_core import Identifier

from fhy.lang.ast.alias import ASTStructure
from fhy.lang.ast.visitor import Transformer


class IdentifierReplacer(Transformer):
    """Replace identifiers.

    Args:
        identifier_map (Dict[ir.Identifier, ir.Identifier]): mapping describing
            identifiers to change from and to.

    """

    _identifier_map: dict[Identifier, Identifier]

    def __init__(self, identifier_map: dict[Identifier, Identifier]):
        super().__init__()
        self._identifier_map = identifier_map

    def visit_identifier(self, identifier: Identifier) -> Identifier:
        return copy(self._identifier_map.get(identifier, identifier))


def replace_identifiers(
    node: ASTStructure, identifier_map: dict[Identifier, Identifier]
) -> ASTStructure:
    """Replace identifiers within AST.

    Args:
        node: AST structure.
        identifier_map (Dict[ir.Identifier, ir.Identifier]): mapping describing
            identifiers to change from and to.

    Returns:
        Node with identifiers replaced as prescribed by mapping.

    """
    return cast(ASTStructure, IdentifierReplacer(identifier_map)(node))
