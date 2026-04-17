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

"""Simple visitor AST Pass to collect symbol identifiers."""

from functools import singledispatchmethod

from fhy_core import (
    DataType,
    Identifier,
    IndexType,
    NumericalType,
    PrimitiveDataType,
    TemplateDataType,
    TupleType,
    Type,
    register_pass,
)
from fhy_core import collect_identifiers as collect_core_identifiers
from fhy_core.pass_infrastructure import (
    AnalysisVisitablePass,  # get from top-level; also, fix other places
)

from fhy.lang.ast.node import (
    Argument,
    DeclarationStatement,
    FunctionExpression,
    IdentifierExpression,
    Module,
    Node,
    Operation,
    Procedure,
    QualifiedType,
)


@register_pass(
    "fhy_ast_identifier_collector",
    "Collects all identifiers in the FhY AST for any given node.",
)
class IdentifierCollector(AnalysisVisitablePass[Node]):
    """Collect all identifiers in the AST for any given node."""

    _identifiers: set[Identifier]

    def __init__(self) -> None:
        super().__init__()
        self._identifiers = set()

    @property
    def identifiers(self) -> frozenset[Identifier]:
        return frozenset(self._identifiers)

    def visit_module(self, module: Module) -> None:
        self._identifiers.add(module.name)

    def visit_procedure(self, procedure: Procedure) -> None:
        self._identifiers.add(procedure.name)
        for template in procedure.templates:
            self._collect_identifiers_from_data_type(template)

    def visit_operation(self, operation: Operation) -> None:
        self._identifiers.add(operation.name)
        for template in operation.templates:
            self._collect_identifiers_from_data_type(template)

    def visit_argument(self, argument: Argument) -> None:
        self._identifiers.add(argument.name)

    def visit_declaration_statement(
        self, declaration_statement: DeclarationStatement
    ) -> None:
        self._identifiers.add(declaration_statement.variable_name)

    def visit_function_expression(
        self, function_expression: FunctionExpression
    ) -> None:
        if isinstance(function_expression.function, IdentifierExpression):
            self._identifiers.add(function_expression.function.identifier)
        for template in function_expression.template_types:
            self._collect_identifiers_from_data_type(template)

    def visit_identifier_expression(
        self, identifier_expression: IdentifierExpression
    ) -> None:
        self._identifiers.add(identifier_expression.identifier)

    def visit_qualified_type(self, qualified_type: QualifiedType) -> None:
        self._collect_identifiers_from_type(qualified_type.base_type)

    @singledispatchmethod
    def _collect_identifiers_from_type(self, type: Type) -> None:
        raise NotImplementedError(
            f"Collecting identifiers from type {type} is not supported."
        )

    @_collect_identifiers_from_type.register(NumericalType)
    def _(self, numerical_type: NumericalType) -> None:
        for dim in numerical_type.shape:
            self._identifiers.update(collect_core_identifiers(dim))

    @_collect_identifiers_from_type.register(TupleType)
    def _(self, tuple_type: TupleType) -> None:
        for type in tuple_type.types:
            self._collect_identifiers_from_type(type)

    @_collect_identifiers_from_type.register(IndexType)
    def _(self, index_type: IndexType) -> None:
        self._identifiers.update(collect_core_identifiers(index_type.lower_bound))
        self._identifiers.update(collect_core_identifiers(index_type.upper_bound))
        if index_type.stride is not None:
            self._identifiers.update(collect_core_identifiers(index_type.stride))

    @singledispatchmethod
    def _collect_identifiers_from_data_type(self, data_type: DataType) -> None:
        raise NotImplementedError(
            f"Collecting identifiers from data type {data_type} is not supported."
        )

    @_collect_identifiers_from_data_type.register(TemplateDataType)
    def _(self, template_data_type: TemplateDataType) -> None:
        self._identifiers.add(template_data_type.data_type)

    @_collect_identifiers_from_data_type.register(PrimitiveDataType)
    def _(self, primitive_data_type: PrimitiveDataType) -> None:
        return


def collect_identifiers(node: Node) -> frozenset[Identifier]:
    """Return a set of identifier objects from a given AST node object."""
    collector = IdentifierCollector()
    collector(node)
    return collector.identifiers
