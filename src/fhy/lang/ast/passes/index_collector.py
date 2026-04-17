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

"""Index collection passes."""

from collections.abc import Callable

from fhy_core import Identifier, register_pass
from fhy_core.pass_infrastructure import AnalysisVisitablePass

from fhy.lang.ast.node import core
from fhy.lang.ast.node.expression import FunctionExpression, IdentifierExpression


@register_pass(
    "fhy_ast_index_collector", "Collects all the indices used in an AST expression."
)
class IndexCollector(AnalysisVisitablePass[core.Expression]):
    """Collect all the indices used in an AST expression."""

    _is_identifier_index: Callable[[Identifier], bool]
    _indices: set[Identifier]

    def __init__(self, is_identifier_index_func: Callable[[Identifier], bool]) -> None:
        super().__init__()
        self._indices = set()
        self._is_identifier_index = is_identifier_index_func

    @property
    def indices(self) -> frozenset[Identifier]:
        return frozenset(self._indices)

    def visit_identifier_expression(self, node: IdentifierExpression) -> None:
        if self._is_identifier_index(node.identifier):
            self._indices.add(node.identifier)


def collect_indices(
    node: core.Expression,
    is_identifier_index: Callable[[Identifier], bool],
) -> frozenset[Identifier]:
    """Collect all the indices used in an AST expression.

    Args:
        node: The AST expression node to collect indices from.
        is_identifier_index: A function that determines if an identifier is an index.

    Returns:
        The set of indices used in the AST expression.

    """
    index_collector = IndexCollector(is_identifier_index)
    index_collector(node)
    return index_collector.indices


# NOTE: if FhY supports indices in reduction's parameters that are not just the
#       identifier itself, this pass must be modified
@register_pass(
    "fhy_ast_reduced_index_collector",
    "Collects all the indices used in an AST expression that are reduced.",
)
class ReducedIndexCollector(AnalysisVisitablePass[core.Expression]):
    """Collect all the indices used in an AST expression that are reduced."""

    _is_identifier_index: Callable[[Identifier], bool]
    _reduced_indices: set[Identifier]

    def __init__(self, is_identifier_index_func: Callable[[Identifier], bool]) -> None:
        super().__init__()
        self._reduced_indices = set()
        self._is_identifier_index = is_identifier_index_func

    @property
    def reduced_indices(self) -> frozenset[Identifier]:
        return frozenset(self._reduced_indices)

    def visit_function_expression(self, node: FunctionExpression) -> None:
        for index in node.indices:
            if not isinstance(index, IdentifierExpression):
                raise RuntimeError(f"Index {index} is not an identifier expression.")
            if self._is_identifier_index(index.identifier):
                self._reduced_indices.add(index.identifier)


def collect_reduced_indices(
    node: core.Expression,
    is_identifier_index: Callable[[Identifier], bool],
) -> frozenset[Identifier]:
    """Collect all the indices used in an AST expression that are reduced.

    Args:
        node: The AST expression node to collect indices from.
        is_identifier_index: A function that
            determines if an identifier is an index.

    Returns:
        The set of indices used in the AST expression that are reduced.

    """
    reduced_index_collector = ReducedIndexCollector(is_identifier_index)
    reduced_index_collector(node)
    return reduced_index_collector.reduced_indices
