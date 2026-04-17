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

"""Transformer pattern for the FhY AST nodes."""

__all__ = ["Transformer"]

from collections.abc import Sequence
from functools import singledispatchmethod
from typing import TypeVar

from fhy_core import (
    DataType,
    IndexType,
    NumericalType,
    PrimitiveDataType,
    TemplateDataType,
    TupleType,
    Type,
    VisitablePass,
)

from fhy.lang.ast.alias import ASTStructure

from .node import (
    Argument,
    ArrayAccessExpression,
    BinaryExpression,
    ComplexLiteral,
    DeclarationStatement,
    Expression,
    ExpressionStatement,
    FloatLiteral,
    ForAllStatement,
    FunctionExpression,
    IdentifierExpression,
    Import,
    IntLiteral,
    Module,
    Operation,
    Procedure,
    QualifiedType,
    ReturnStatement,
    SelectionStatement,
    Statement,
    TernaryExpression,
    TupleAccessExpression,
    TupleExpression,
    UnaryExpression,
)

Statements = Statement | list[Statement]
_T = TypeVar("_T")


class Transformer(VisitablePass[ASTStructure, ASTStructure]):
    """AST node transformer."""

    def get_noop_output(self, ir: ASTStructure) -> ASTStructure:
        return ir

    def visit_sequence(
        self, nodes: Sequence[_T], is_length_same: bool = True
    ) -> tuple[_T, ...]:
        """Visit a list of nodes or structures.

        Args:
            nodes: Nodes or structures to visit.
            is_length_same: Whether the length of the transformed nodes or
                structures should be the same as the input nodes or structures.

        Returns:
            Transformed nodes or structures.

        """
        if is_length_same:
            return tuple(self.visit(node) for node in nodes)
        else:
            new_nodes: list[_T] = []
            for node in nodes:
                new_node = self.visit(node)
                # TODO: Implement returning "None" to remove the element.
                # if new_node is None:
                #     continue
                if isinstance(new_node, Sequence):
                    new_nodes.extend(new_node)
                else:
                    new_nodes.append(new_node)
            return tuple(new_nodes)

    def visit_module(self, node: Module) -> Module:
        """Transform a module node.

        Args:
            node: Module node to transform.

        """
        new_statements = self.visit_sequence(node.statements, is_length_same=False)

        return Module(
            name=node.name, statements=new_statements, provenance=node.provenance
        )

    def visit_import(self, node: Import) -> Import:
        """Transform an import node.

        Args:
            node: Import node to transform.

        """
        return node

    def visit_operation(self, node: Operation) -> Operation:
        """Transform an operation node.

        Args:
            node: Operation node to transform.

        """
        new_templates = self.visit_sequence(node.templates)
        new_args = self.visit_sequence(node.args)
        new_return_type: QualifiedType = self.visit_qualified_type(node.return_type)
        new_body = self.visit_sequence(node.body, is_length_same=False)

        return Operation(
            name=node.name,
            templates=new_templates,
            args=new_args,
            return_type=new_return_type,
            body=new_body,
        )

    def visit_procedure(self, node: Procedure) -> Procedure:
        """Transform a Procedure node.

        Args:
            node: Procedure node to transform.

        """
        new_templates = self.visit_sequence(node.templates)
        new_args = self.visit_sequence(node.args)
        new_body = self.visit_sequence(node.body, is_length_same=False)

        return Procedure(
            name=node.name, templates=new_templates, args=new_args, body=new_body
        )

    def visit_argument(self, node: Argument) -> Argument:
        """Transform an argument node.

        Args:
            node: Argument node to transform.

        """
        return Argument(
            qualified_type=self.visit_qualified_type(node.qualified_type),
            name=node.name,
        )

    def visit_declaration_statement(self, node: DeclarationStatement) -> Statements:
        """Transform a declaration statement node.

        Args:
            node: Declaration statement node to transform.

        """
        return DeclarationStatement(
            variable_name=node.variable_name,
            variable_type=self.visit_qualified_type(node.variable_type),
            expression=self.visit_expression(node.expression),
        )

    def visit_expression_statement(self, node: ExpressionStatement) -> Statements:
        return ExpressionStatement(
            left=self.visit_expression(node.left),
            right=self.visit_expression(node.right),
        )

    def visit_selection_statement(self, node: SelectionStatement) -> Statements:
        """Transform a selection statement node.

        Args:
            node: Selection statement node to transform.

        """
        return SelectionStatement(
            condition=self.visit_expression(node.condition),
            true_body=self.visit_sequence(node.true_body, is_length_same=False),
            false_body=self.visit_sequence(node.false_body, is_length_same=False),
        )

    def visit_for_all_statement(self, node: ForAllStatement) -> Statements:
        """Transform an iteration statement node.

        Args:
            node: For-all statement node to transform.

        """
        return ForAllStatement(
            index=self.visit_expression(node.index),
            body=self.visit_sequence(node.body, is_length_same=False),
        )

    def visit_return_statement(self, node: ReturnStatement) -> Statements:
        """Transform a return statement node.

        Args:
            node: Return statement node to transform.

        """
        return ReturnStatement(expression=self.visit_expression(node.expression))

    def visit_unary_expression(self, node: UnaryExpression) -> UnaryExpression:
        return UnaryExpression(
            operation=node.operation, expression=self.visit_expression(node.expression)
        )

    def visit_binary_expression(self, node: BinaryExpression) -> Expression:
        """Transform a binary expression node.

        Args:
            node: Binary expression node to transform.

        """
        return BinaryExpression(
            operation=node.operation,
            left=self.visit_expression(node.left),
            right=self.visit_expression(node.right),
        )

    def visit_ternary_expression(self, node: TernaryExpression) -> Expression:
        """Transform a ternary expression node.

        Args:
            node: Ternary expression node to transform.

        """
        return TernaryExpression(
            condition=self.visit_expression(node.condition),
            true=self.visit_expression(node.true),
            false=self.visit_expression(node.false),
        )

    def visit_function_expression(self, node: FunctionExpression) -> Expression:
        """Transform a function expression node.

        Args:
            node: Function expression node to transform.

        """
        return FunctionExpression(
            function=self.visit_expression(node.function),
            template_types=self.visit_sequence(node.template_types),
            indices=self.visit_sequence(node.indices),
            args=self.visit_sequence(node.args),
        )

    def visit_array_access_expression(self, node: ArrayAccessExpression) -> Expression:
        """Transform an array access expression node.

        Args:
            node: Array access expression node to transform.

        """
        return ArrayAccessExpression(
            array_expression=self.visit_expression(node.array_expression),
            indices=self.visit_sequence(node.indices),
        )

    def visit_tuple_expression(self, node: TupleExpression) -> Expression:
        """Transform a tuple expression node.

        Args:
            node: Tuple expression node to transform.

        """
        return TupleExpression(expressions=self.visit_sequence(node.expressions))

    def visit_tuple_access_expression(self, node: TupleAccessExpression) -> Expression:
        """Transform a tuple access expression node.

        Args:
            node: Tuple access expression node to transform.

        """
        return TupleAccessExpression(
            tuple_expression=self.visit_expression(node.tuple_expression),
            element_index=self.visit_int_literal(node.element_index),
        )

    def visit_identifier_expression(self, node: IdentifierExpression) -> Expression:
        """Transform an identifier expression node.

        Args:
            node: Identifier expression node to transform.

        """
        return IdentifierExpression(identifier=self.visit_identifier(node.identifier))

    def visit_int_literal(self, node: IntLiteral) -> IntLiteral:
        """Transform an int literal node.

        Args:
            node: Int literal node to transform.

        """
        return node

    def visit_float_literal(self, node: FloatLiteral) -> FloatLiteral:
        """Transform a float literal node.

        Args:
            node: Float literal node to transform.

        """
        return node

    def visit_complex_literal(self, node: ComplexLiteral) -> ComplexLiteral:
        """Transform a complex literal node.

        Args:
            node: Complex literal node to transform.

        """
        return node

    def visit_qualified_type(self, node: QualifiedType) -> QualifiedType:
        """Transform a qualified type node.

        Args:
            node: Qualified type node to transform.

        """
        return QualifiedType(
            base_type=self.visit_type(node.base_type),
            type_qualifier=self.visit_type_qualifier(node.type_qualifier),
        )

    @singledispatchmethod
    def visit_type(self, node: Type) -> Type:
        """Transform a type.

        Args:
            node: Type to transform.

        """
        raise NotImplementedError(f'Type "{type(node)}" is not supported.')

    @visit_type.register(NumericalType)
    def _(self, numerical_type: NumericalType) -> NumericalType:
        return numerical_type

    @visit_type.register(TupleType)
    def _(self, tuple_type: TupleType) -> TupleType:
        return tuple_type

    @visit_type.register(IndexType)
    def _(self, index_type: IndexType) -> IndexType:
        return index_type

    @singledispatchmethod
    def visit_data_type(self, data_type: DataType) -> DataType:
        """Transform a data type.

        Args:
            data_type: Data type to transform.

        """
        raise NotImplementedError(f'Data type "{type(data_type)}" is not supported.')

    @visit_data_type.register(PrimitiveDataType)
    def _(self, primitive_data_type: PrimitiveDataType) -> PrimitiveDataType:
        return primitive_data_type

    @visit_data_type.register(TemplateDataType)
    def _(self, template_data_type: TemplateDataType) -> TemplateDataType:
        return template_data_type
