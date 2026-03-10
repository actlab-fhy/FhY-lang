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

"""Visitor patterns to visit the FhY AST nodes."""

from abc import ABC
from collections.abc import Callable, Iterable, Sequence
from copy import copy
from typing import Any, ClassVar, TypeVar

from fhy_core import (
    BinaryExpression as CoreBinaryExpression,
)
from fhy_core import (
    CoreDataType,
    DataType,
    Identifier,
    IndexType,
    NumericalType,
    PrimitiveDataType,
    TemplateDataType,
    TupleType,
    Type,
    TypeQualifier,
)
from fhy_core import (
    Expression as CoreExpression,
)
from fhy_core import (
    IdentifierExpression as CoreIdentifierExpression,
)
from fhy_core import (
    LiteralExpression as CoreLiteralExpression,
)
from fhy_core import (
    UnaryExpression as CoreUnaryExpression,
)
from frozendict import frozendict

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
from .span import Source, Span


class BasePass(ABC):
    """Abstract visitor pattern AST nodes and structures."""

    _VISITOR_METHODS: ClassVar[frozendict[type[ASTStructure], str]] = frozendict(
        {
            Module: "visit_module",
            Import: "visit_import",
            Operation: "visit_operation",
            Procedure: "visit_procedure",
            Argument: "visit_argument",
            DeclarationStatement: "visit_declaration_statement",
            ExpressionStatement: "visit_expression_statement",
            SelectionStatement: "visit_selection_statement",
            ForAllStatement: "visit_for_all_statement",
            ReturnStatement: "visit_return_statement",
            UnaryExpression: "visit_unary_expression",
            BinaryExpression: "visit_binary_expression",
            TernaryExpression: "visit_ternary_expression",
            FunctionExpression: "visit_function_expression",
            ArrayAccessExpression: "visit_array_access_expression",
            TupleExpression: "visit_tuple_expression",
            TupleAccessExpression: "visit_tuple_access_expression",
            IdentifierExpression: "visit_identifier_expression",
            IntLiteral: "visit_int_literal",
            FloatLiteral: "visit_float_literal",
            ComplexLiteral: "visit_complex_literal",
            QualifiedType: "visit_qualified_type",
            CoreBinaryExpression: "visit_core_binary_expression",
            CoreUnaryExpression: "visit_core_unary_expression",
            CoreLiteralExpression: "visit_core_literal_expression",
            CoreIdentifierExpression: "visit_core_identifier_expression",
            NumericalType: "visit_numerical_type",
            IndexType: "visit_index_type",
            TupleType: "visit_tuple_type",
            PrimitiveDataType: "visit_primitive_data_type",
            TemplateDataType: "visit_template_data_type",
            TypeQualifier: "visit_type_qualifier",
            CoreDataType: "visit_core_data_type",
            Identifier: "visit_identifier",
            Source: "visit_source",
            Span: "visit_span",
        }
    )

    def __call__(self, node: Any) -> Any:
        return self.visit(node)

    def visit(self, node: Any) -> Any:
        """Visit a node or structure based on its type.

        Args:
            node: AST node or structure to visit.

        Returns:
            Result of visiting the node.

        """
        method_name: str | None = self._VISITOR_METHODS.get(type(node), None)
        if method_name is None:
            return self.default(node)
        else:
            method: Callable[[ASTStructure], Any] = getattr(self, method_name)
            return method(node)

    def default(self, node: Any) -> Any:
        """Visit a node that is not supported.

        Args:
            node: AST node or structure to visit.

        Returns:
            Result of visiting the node.

        Raises:
            NotImplementedError: If the node is not supported.

        """
        raise NotImplementedError(
            f'AST node/structure "{type(node)}" is not supported.'
        )


def _check_ast_node_type(
    node: Any, ast_node_type: type, ast_node_type_name: str, pass_name: str
) -> None:
    if not isinstance(node, ast_node_type):
        raise RuntimeError(
            f'{pass_name} expects AST node/structure "{ast_node_type_name}", '
            f"but got {type(node)}."
        )


class Visitor(BasePass):
    """AST node visitor."""

    def visit(self, node: ASTStructure | Sequence[ASTStructure]) -> None:
        if isinstance(node, list):
            self.visit_sequence(node)
        else:
            super().visit(node)

    def visit_module(self, node: Module) -> None:
        """Visit a module node.

        Args:
            node: Module node to visit.

        """
        self.visit(node.name)
        self.visit(node.statements)

    def visit_import(self, node: Import) -> None:
        """Visit an import node.

        Args:
            node: Import node to visit.

        """
        self.visit(node.name)

    def visit_operation(self, node: Operation) -> None:
        """Visit an operation node.

        Args:
            node: Operation node to visit.

        """
        self.visit(node.name)
        self.visit(node.templates)
        self.visit(node.args)
        self.visit(node.return_type)
        self.visit(node.body)

    def visit_procedure(self, node: Procedure) -> None:
        """Visit a procedure node.

        Args:
            node: Procedure node to visit.

        """
        self.visit(node.name)
        self.visit(node.templates)
        self.visit(node.args)
        self.visit(node.body)

    def visit_argument(self, node: Argument) -> None:
        """Visit an argument node.

        Args:
            node: Argument node to visit.

        """
        self.visit(node.qualified_type)
        if node.name is not None:
            self.visit(node.name)

    def visit_declaration_statement(self, node: DeclarationStatement) -> None:
        """Visit a declaration statement node.

        Args:
            node: Declaration statement node to visit.

        """
        self.visit(node.variable_name)
        self.visit(node.variable_type)
        if node.expression is not None:
            self.visit(node.expression)

    def visit_expression_statement(self, node: ExpressionStatement) -> None:
        """Visit an expression statement node.

        Args:
            node: Expression statement node to visit.

        """
        if node.left is not None:
            self.visit(node.left)
        self.visit(node.right)

    def visit_selection_statement(self, node: SelectionStatement) -> None:
        """Visit a selection statement node.

        Args:
            node: Selection statement node to visit.

        """
        self.visit(node.condition)
        self.visit(node.true_body)
        self.visit(node.false_body)

    def visit_for_all_statement(self, node: ForAllStatement) -> None:
        """Visit a for-all statement node.

        Args:
            node: For-all statement node to visit.

        """
        self.visit(node.index)
        self.visit(node.body)

    def visit_return_statement(self, node: ReturnStatement) -> None:
        """Visit a return statement node.

        Args:
            node: Return statement node to visit.

        """
        self.visit(node.expression)

    def visit_unary_expression(self, node: UnaryExpression) -> None:
        """Visit a unary expression node.

        Args:
            node: Unary expression node to visit.

        """
        self.visit(node.expression)

    def visit_binary_expression(self, node: BinaryExpression) -> None:
        """Visit a binary expression node.

        Args:
            node: Binary expression node to visit.

        """
        self.visit(node.left)
        self.visit(node.right)

    def visit_ternary_expression(self, node: TernaryExpression) -> None:
        """Visit a ternary expression node.

        Args:
            node: Ternary expression node to visit.

        """
        self.visit(node.condition)
        self.visit(node.true)
        self.visit(node.false)

    def visit_function_expression(self, node: FunctionExpression) -> None:
        """Visit a function expression node.

        Args:
            node: Function expression node to visit.

        """
        self.visit(node.function)
        self.visit(node.template_types)
        self.visit(node.indices)
        self.visit(node.args)

    def visit_array_access_expression(self, node: ArrayAccessExpression) -> None:
        """Visit an array access expression node.

        Args:
            node: Array access expression node to visit.

        """
        self.visit(node.array_expression)
        self.visit(node.indices)

    def visit_tuple_expression(self, node: TupleExpression) -> None:
        """Visit a tuple expression node.

        Args:
            node: Tuple expression node to visit.

        """
        self.visit(node.expressions)

    def visit_tuple_access_expression(self, node: TupleAccessExpression) -> None:
        """Visit a tuple access expression node.

        Args:
            node: Tuple access expression node to visit.

        """
        self.visit(node.tuple_expression)
        self.visit(node.element_index)

    def visit_identifier_expression(self, node: IdentifierExpression) -> None:
        """Visit an identifier expression node.

        Args:
            node: Identifier expression node to visit.

        """
        self.visit(node.identifier)

    def visit_int_literal(self, node: IntLiteral) -> None:
        """Visit an integer literal node.

        Args:
            node: Integer literal node to visit.

        """

    def visit_float_literal(self, node: FloatLiteral) -> None:
        """Visit a float literal node.

        Args:
            node: Float literal node to visit.

        """

    def visit_complex_literal(self, node: ComplexLiteral) -> None:
        """Visit a complex literal node.

        Args:
            node: Complex literal node to visit.

        """

    def visit_qualified_type(self, node: QualifiedType) -> None:
        """Visit a qualified type node.

        Args:
            node: Qualified type node to visit.

        """
        self.visit(node.base_type)
        self.visit(node.type_qualifier)

    def visit_core_binary_expression(self, node: CoreBinaryExpression) -> None:
        """Visit a core binary expression node.

        Args:
            node: Core binary expression node to visit.

        """
        self.visit(node.left)
        self.visit(node.right)

    def visit_core_unary_expression(self, node: CoreUnaryExpression) -> None:
        """Visit a core unary expression node.

        Args:
            node: Core unary expression node to visit.

        """
        self.visit(node.operand)

    def visit_core_identifier_expression(self, node: CoreIdentifierExpression) -> None:
        """Visit a core identifier expression node.

        Args:
            node: Core identifier expression node to visit.

        """
        self.visit(node.identifier)

    def visit_core_literal_expression(self, node: CoreLiteralExpression) -> None:
        """Visit a core literal expression node.

        Args:
            node: Core literal expression node to visit.

        """

    def visit_numerical_type(self, numerical_type: NumericalType) -> None:
        """Visit a numerical type.

        Args:
            numerical_type: Numerical type to visit.

        """
        self.visit(numerical_type.data_type)
        self.visit(numerical_type.shape)

    def visit_index_type(self, index_type: IndexType) -> None:
        """Visit an index type.

        Args:
            index_type: Index type to visit.

        """
        self.visit(index_type.lower_bound)
        self.visit(index_type.upper_bound)
        if index_type.stride is not None:
            self.visit(index_type.stride)

    def visit_tuple_type(self, tuple_type: TupleType) -> None:
        """Visit a tuple type.

        Args:
            tuple_type: Tuple type to visit.

        """
        self.visit(tuple_type.types)

    def visit_primitive_data_type(self, node: PrimitiveDataType) -> None:
        """Visit a primitive data type.

        Args:
            node: Primitive data type to visit.

        """
        self.visit(node.core_data_type)

    def visit_template_data_type(self, node: TemplateDataType) -> None:
        """Visit a template data type.

        Args:
            node: Template data type to visit.

        """
        self.visit(node.data_type)

    def visit_type_qualifier(self, type_qualifier: TypeQualifier) -> None:
        """Visit a type qualifier.

        Args:
            type_qualifier: Type qualifier to visit.

        """

    def visit_core_data_type(self, core_data_type: CoreDataType) -> None:
        """Visit a core data type.

        Args:
            core_data_type: Core data type to visit.

        """

    def visit_identifier(self, identifier: Identifier) -> None:
        """Visit an identifier.

        Args:
            identifier: Identifier to visit.

        """

    def visit_sequence(self, nodes: Iterable[ASTStructure]) -> None:
        """Visit a list of nodes or structures.

        Args:
            nodes: Nodes or structures to visit.

        """
        for node in nodes:
            self.visit(node)

    def visit_span(self, span: Span | None) -> None:
        """Visit a span.

        Args:
            span: Span to visit.

        """
        if span is not None and span.source is not None:
            self.visit(span.source)

    def visit_source(self, source: Source) -> None:
        """Visit a source.

        Args:
            source: Source to visit.

        """


class ExpressionVisitor(Visitor):
    """Visitor for expression nodes."""

    def __call__(self, node: Expression) -> Any:
        _check_ast_node_type(node, Expression, "expression", self.__class__.__name__)
        return super().__call__(node)


Statements = Statement | list[Statement]
_T = TypeVar("_T")


class Transformer(BasePass):
    """AST node transformer."""

    def visit_list(self, nodes: list[_T], is_length_same: bool = True) -> list[_T]:
        """Visit a list of nodes or structures.

        Args:
            nodes: Nodes or structures to visit.
            is_length_same (bool): Whether the length of the transformed nodes or
                structures should be the same as the input nodes or structures.

        Returns:
            Transformed nodes or structures.

        """
        if is_length_same:
            return [self.visit(node) for node in nodes]
        else:
            new_nodes: list[_T] = []
            for node in nodes:
                new_node = self.visit(node)
                # TODO: Implement returning "None" to remove the element.
                # if new_node is None:
                #     continue
                if isinstance(new_node, list):
                    new_nodes.extend(new_node)
                else:
                    new_nodes.append(new_node)
            return new_nodes

    def visit_module(self, node: Module) -> Module:
        """Transform a module node.

        Args:
            node: Module node to transform.

        """
        span: Span | None = self.visit_span(node.span)
        new_name: Identifier = self.visit_identifier(node.name)
        new_statements: list[Statement] = self.visit_list(
            node.statements, is_length_same=False
        )

        return Module(span=span, name=new_name, statements=new_statements)

    def visit_statement(self, node: Statement) -> Statements:
        """Transform a statement.

        Args:
            node: Statement to transform.

        """
        if isinstance(node, Import):
            return self.visit_import(node)
        elif isinstance(node, Operation):
            return self.visit_operation(node)
        elif isinstance(node, Procedure):
            return self.visit_procedure(node)
        elif isinstance(node, DeclarationStatement):
            return self.visit_declaration_statement(node)
        elif isinstance(node, ExpressionStatement):
            return self.visit_expression_statement(node)
        elif isinstance(node, SelectionStatement):
            return self.visit_selection_statement(node)
        elif isinstance(node, ForAllStatement):
            return self.visit_for_all_statement(node)
        elif isinstance(node, ReturnStatement):
            return self.visit_return_statement(node)
        else:
            raise NotImplementedError(f'Node "{type(node)}" is not supported.')

    def visit_import(self, node: Import) -> Import:
        """Transform an import node.

        Args:
            node: Import node to transform.

        """
        new_name: Identifier = self.visit_identifier(node.name)
        span: Span | None = self.visit_span(node.span)

        return Import(span=span, name=new_name)

    def visit_operation(self, node: Operation) -> Operation:
        """Transform an operation node.

        Args:
            node: Operation node to transform.

        """
        span: Span | None = self.visit_span(node.span)
        new_name: Identifier = self.visit_identifier(node.name)
        new_templates: list[TemplateDataType] = self.visit_list(node.templates)
        new_args: list[Argument] = self.visit_arguments(node.args)
        new_return_type: QualifiedType = self.visit_qualified_type(node.return_type)
        new_body: list[Statement] = self.visit_list(node.body, is_length_same=False)

        return Operation(
            span=span,
            name=new_name,
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
        span: Span | None = self.visit_span(node.span)
        new_name: Identifier = self.visit_identifier(node.name)
        new_templates: list[TemplateDataType] = self.visit_list(node.templates)
        new_args: list[Argument] = self.visit_arguments(node.args)
        new_body: list[Statement] = self.visit_list(node.body, is_length_same=False)

        return Procedure(
            span=span,
            name=new_name,
            templates=new_templates,
            args=new_args,
            body=new_body,
        )

    def visit_arguments(self, nodes: list[Argument]) -> list[Argument]:
        """Transform a list of argument nodes.

        Args:
            nodes: List of argument nodes to transform.

        """
        return self.visit_list(nodes)

    def visit_argument(self, node: Argument) -> Argument:
        """Transform an argument node.

        Args:
            node: Argument node to transform.

        """
        span: Span | None = self.visit_span(node.span)
        new_qualified_type: QualifiedType = self.visit_qualified_type(
            node.qualified_type
        )
        new_name: Identifier | None
        if node.name is not None:
            new_name = self.visit_identifier(node.name)
        else:
            new_name = None

        return Argument(span=span, qualified_type=new_qualified_type, name=new_name)

    def visit_declaration_statement(self, node: DeclarationStatement) -> Statements:
        """Transform a declaration statement node.

        Args:
            node: Declaration statement node to transform.

        """
        span: Span | None = self.visit_span(node.span)
        new_variable_name: Identifier = self.visit_identifier(node.variable_name)
        new_variable_type: QualifiedType = self.visit_qualified_type(node.variable_type)
        new_expression: Expression | None

        if node.expression is not None:
            new_expression = self.visit_expression(node.expression)
        else:
            new_expression = None

        return DeclarationStatement(
            span=span,
            variable_name=new_variable_name,
            variable_type=new_variable_type,
            expression=new_expression,
        )

    def visit_expression_statement(self, node: ExpressionStatement) -> Statements:
        """Transform an expression statement node.

        Args:
            node: Expression statement node to transform.

        """
        span: Span | None = self.visit_span(node.span)
        new_left: Expression | None
        if node.left is not None:
            new_left = self.visit_expression(node.left)
        else:
            new_left = None
        new_right = self.visit_expression(node.right)

        return ExpressionStatement(span=span, left=new_left, right=new_right)

    def visit_selection_statement(self, node: SelectionStatement) -> Statements:
        """Transform a selection statement node.

        Args:
            node: Selection statement node to transform.

        """
        span: Span | None = self.visit_span(node.span)
        new_condition: Expression = self.visit_expression(node.condition)
        new_true_body: list[Statement] = self.visit_list(
            node.true_body, is_length_same=False
        )
        new_false_body: list[Statement] = self.visit_list(
            node.false_body, is_length_same=False
        )

        return SelectionStatement(
            span=span,
            condition=new_condition,
            true_body=new_true_body,
            false_body=new_false_body,
        )

    def visit_for_all_statement(self, node: ForAllStatement) -> Statements:
        """Transform an iteration statement node.

        Args:
            node: For-all statement node to transform.

        """
        span: Span | None = self.visit_span(node.span)
        new_index: Expression = self.visit_expression(node.index)
        new_body: list[Statement] = self.visit_list(node.body, is_length_same=False)

        return ForAllStatement(span=span, index=new_index, body=new_body)

    def visit_return_statement(self, node: ReturnStatement) -> Statements:
        """Transform a return statement node.

        Args:
            node: Return statement node to transform.

        """
        span: Span | None = self.visit_span(node.span)
        new_expression: Expression = self.visit_expression(node.expression)

        return ReturnStatement(span=span, expression=new_expression)

    # ruff: noqa: C901
    def visit_expression(self, node: Expression) -> Expression:
        if isinstance(node, UnaryExpression):
            return self.visit_unary_expression(node)
        elif isinstance(node, BinaryExpression):
            return self.visit_binary_expression(node)
        elif isinstance(node, TernaryExpression):
            return self.visit_ternary_expression(node)
        elif isinstance(node, FunctionExpression):
            return self.visit_function_expression(node)
        elif isinstance(node, ArrayAccessExpression):
            return self.visit_array_access_expression(node)
        elif isinstance(node, TupleExpression):
            return self.visit_tuple_expression(node)
        elif isinstance(node, TupleAccessExpression):
            return self.visit_tuple_access_expression(node)
        elif isinstance(node, IdentifierExpression):
            return self.visit_identifier_expression(node)
        elif isinstance(node, IntLiteral):
            return self.visit_int_literal(node)
        elif isinstance(node, FloatLiteral):
            return self.visit_float_literal(node)
        else:
            raise NotImplementedError(f'Node "{type(node)}" is not supported.')

    def visit_unary_expression(self, node: UnaryExpression) -> Expression:
        """Transform a unary expression node.

        Args:
            node: Unary expression node to transform.

        """
        span: Span | None = self.visit_span(node.span)
        new_expression: Expression = self.visit_expression(node.expression)

        return UnaryExpression(
            span=span, operation=node.operation, expression=new_expression
        )

    def visit_binary_expression(self, node: BinaryExpression) -> Expression:
        """Transform a binary expression node.

        Args:
            node: Binary expression node to transform.

        """
        span: Span | None = self.visit_span(node.span)
        new_left: Expression = self.visit_expression(node.left)
        new_right: Expression = self.visit_expression(node.right)

        return BinaryExpression(
            span=span, operation=node.operation, left=new_left, right=new_right
        )

    def visit_ternary_expression(self, node: TernaryExpression) -> Expression:
        """Transform a ternary expression node.

        Args:
            node: Ternary expression node to transform.

        """
        span: Span | None = self.visit_span(node.span)
        new_condition: Expression = self.visit_expression(node.condition)
        new_true: Expression = self.visit_expression(node.true)
        new_false: Expression = self.visit_expression(node.false)

        return TernaryExpression(
            span=span, condition=new_condition, true=new_true, false=new_false
        )

    def visit_function_expression(self, node: FunctionExpression) -> Expression:
        """Transform a function expression node.

        Args:
            node: Function expression node to transform.

        """
        span: Span | None = self.visit_span(node.span)
        new_function: Expression = self.visit_expression(node.function)
        new_template_types: list[DataType] = self.visit_list(node.template_types)
        new_indices: list[Expression] = self.visit_list(node.indices)
        new_args: list[Expression] = self.visit_list(node.args)

        return FunctionExpression(
            span=span,
            function=new_function,
            template_types=new_template_types,
            indices=list(new_indices),
            args=list(new_args),
        )

    def visit_array_access_expression(self, node: ArrayAccessExpression) -> Expression:
        """Transform an array access expression node.

        Args:
            node: Array access expression node to transform.

        """
        span: Span | None = self.visit_span(node.span)
        new_array_expresssion: Expression = self.visit_expression(node.array_expression)
        new_indices: list[Expression] = self.visit_list(node.indices)

        return ArrayAccessExpression(
            span=span, array_expression=new_array_expresssion, indices=new_indices
        )

    def visit_tuple_expression(self, node: TupleExpression) -> Expression:
        """Transform a tuple expression node.

        Args:
            node: Tuple expression node to transform.

        """
        span: Span | None = self.visit_span(node.span)
        new_expressions: list[Expression] = self.visit_list(node.expressions)

        return TupleExpression(span=span, expressions=list(new_expressions))

    def visit_tuple_access_expression(self, node: TupleAccessExpression) -> Expression:
        """Transform a tuple access expression node.

        Args:
            node: Tuple access expression node to transform.

        """
        span: Span | None = self.visit_span(node.span)
        new_tuple_expression: Expression = self.visit_expression(node.tuple_expression)
        new_element_index: IntLiteral = self.visit_int_literal(node.element_index)

        return TupleAccessExpression(
            span=span,
            tuple_expression=new_tuple_expression,
            element_index=new_element_index,
        )

    def visit_identifier_expression(self, node: IdentifierExpression) -> Expression:
        """Transform an identifier expression node.

        Args:
            node: Identifier expression node to transform.

        """
        span: Span | None = self.visit_span(node.span)
        new_identifier: Identifier = self.visit_identifier(node.identifier)

        return IdentifierExpression(span=span, identifier=new_identifier)

    def visit_int_literal(self, node: IntLiteral) -> IntLiteral:
        """Transform an int literal node.

        Args:
            node: Int literal node to transform.

        """
        return copy(node)

    def visit_float_literal(self, node: FloatLiteral) -> FloatLiteral:
        """Transform a float literal node.

        Args:
            node: Float literal node to transform.

        """
        return copy(node)

    def visit_complex_literal(self, node: ComplexLiteral) -> ComplexLiteral:
        """Transform a complex literal node.

        Args:
            node: Complex literal node to transform.

        """
        return copy(node)

    def visit_qualified_type(self, node: QualifiedType) -> QualifiedType:
        """Transform a qualified type node.

        Args:
            node: Qualified type node to transform.

        """
        span: Span | None = self.visit_span(node.span)
        new_base_type: Type = self.visit(node.base_type)
        new_type_qualifier: TypeQualifier = self.visit_type_qualifier(
            node.type_qualifier
        )

        return QualifiedType(
            span=span, base_type=new_base_type, type_qualifier=new_type_qualifier
        )

    def visit_types(self, nodes: list[Type]) -> list[Type]:
        """Transform a list of types.

        Args:
            nodes: List of types to transform.

        """
        return [self.visit_type(node) for node in nodes]

    def visit_type(self, node: Type) -> Type:
        """Transform a type.

        Args:
            node: Type to transform.

        """
        if isinstance(node, NumericalType):
            return self.visit_numerical_type(node)
        elif isinstance(node, IndexType):
            return self.visit_index_type(node)
        elif isinstance(node, TupleType):
            return self.visit_tuple_type(node)
        else:
            raise NotImplementedError(f'Type "{type(node)}" is not supported.')

    def visit_core_expression(self, node: CoreExpression) -> CoreExpression:
        """Transform a core expression.

        Args:
            node: Core expression to transform.

        """
        if isinstance(node, CoreBinaryExpression):
            return self.visit_core_binary_expression(node)
        elif isinstance(node, CoreUnaryExpression):
            return self.visit_core_unary_expression(node)
        elif isinstance(node, CoreLiteralExpression):
            return self.visit_core_literal_expression(node)
        elif isinstance(node, CoreIdentifierExpression):
            return self.visit_core_identifier_expression(node)
        else:
            raise NotImplementedError(
                f'Core expression "{type(node)}" is not supported.'
            )

    def visit_core_binary_expression(
        self, node: CoreBinaryExpression
    ) -> CoreExpression:
        """Transform a core binary expression.

        Args:
            node: Core binary expression to transform.

        """
        new_left: CoreExpression = self.visit(node.left)
        new_right: CoreExpression = self.visit(node.right)

        return CoreBinaryExpression(
            operation=node.operation, left=new_left, right=new_right
        )

    def visit_core_unary_expression(self, node: CoreUnaryExpression) -> CoreExpression:
        """Transform a core unary expression.

        Args:
            node: Core unary expression to transform.

        """
        new_operand: CoreExpression = self.visit(node.operand)

        return CoreUnaryExpression(operation=node.operation, operand=new_operand)

    def visit_core_identifier_expression(
        self, node: CoreIdentifierExpression
    ) -> CoreExpression:
        """Transform a core identifier expression.

        Args:
            node: Core identifier expression to transform.

        """
        new_identifier: Identifier = self.visit(node.identifier)

        return CoreIdentifierExpression(identifier=new_identifier)

    def visit_core_literal_expression(
        self, node: CoreLiteralExpression
    ) -> CoreLiteralExpression:
        """Transform a core literal expression.

        Args:
            node: Core literal expression to transform.

        """
        return CoreLiteralExpression(node.value)

    def visit_numerical_type(self, numerical_type: NumericalType) -> NumericalType:
        """Transform a numerical type.

        Args:
            numerical_type: Numerical type to transform.

        """
        new_data_type = self.visit_data_type(numerical_type.data_type)
        new_shape = [self.visit_core_expression(j) for j in numerical_type.shape]

        return NumericalType(data_type=new_data_type, shape=new_shape)

    def visit_tuple_type(self, tuple_type: TupleType) -> TupleType:
        """Transform a tuple type.

        Args:
            tuple_type: Tuple type node to transform.

        """
        new_types = self.visit_types(tuple_type.types)

        return TupleType(types=new_types)

    def visit_data_type(self, data_type: DataType) -> DataType:
        """Transform a data type.

        Args:
            data_type: Data type to transform.

        """
        if isinstance(data_type, PrimitiveDataType):
            return self.visit_primitive_data_type(data_type)
        elif isinstance(data_type, TemplateDataType):
            return self.visit_template_data_type(data_type)
        else:
            raise NotImplementedError(
                f'Data type "{type(data_type)}" is not supported.'
            )

    def visit_index_type(self, index_type: IndexType) -> IndexType:
        """Transform an index type.

        Args:
            index_type: Index type to transform.

        """
        new_lower_bound = self.visit_core_expression(index_type.lower_bound)
        new_upper_bound = self.visit_core_expression(index_type.upper_bound)
        new_stride: CoreExpression | None
        if index_type.stride is not None:
            new_stride = self.visit_core_expression(index_type.stride)
        else:
            new_stride = None

        return IndexType(
            lower_bound=new_lower_bound,
            upper_bound=new_upper_bound,
            stride=new_stride,
        )

    def visit_primitive_data_type(
        self, primitive_data_type: PrimitiveDataType
    ) -> PrimitiveDataType:
        """Transform a primitive data type.

        Args:
            primitive_data_type: Primitive data type to transform.

        """
        new_core_data_type: CoreDataType = self.visit_core_data_type(
            primitive_data_type.core_data_type
        )
        return PrimitiveDataType(core_data_type=new_core_data_type)

    def visit_core_data_type(self, core_data_type: CoreDataType) -> CoreDataType:
        """Transform a core data type.

        Args:
            core_data_type: Core data type to transform.

        """
        return copy(core_data_type)

    def visit_template_data_type(self, node: TemplateDataType) -> TemplateDataType:
        """Transform a template data type.

        Args:
            node: Template data type to transform.

        """
        new_template_data_type = self.visit(node.data_type)
        return TemplateDataType(data_type=new_template_data_type)

    def visit_type_qualifier(self, type_qualifier: TypeQualifier) -> TypeQualifier:
        """Transform a type qualifier.

        Args:
            type_qualifier: Type qualifier to transform.

        """
        return copy(type_qualifier)

    def visit_identifier(self, identifier: Identifier) -> Identifier:
        """Transform an identifier.

        Args:
            identifier: Identifier to transform.

        """
        return copy(identifier)

    def visit_span(self, span: Span | None) -> Span | None:
        """Transform a span.

        Args:
            span: Span to transform.

        """
        if span is None:
            return None
        else:
            new_source: Source | None = self.visit_source(span.source)

            return Span(
                source=new_source,
                start_line=span.line.start,
                end_line=span.line.stop,
                start_column=span.column.start,
                end_column=span.column.stop,
            )

    def visit_source(self, source: Source | None) -> Source | None:
        """Transform a source.

        Args:
            source: Source to transform.

        """
        return source or copy(source)
