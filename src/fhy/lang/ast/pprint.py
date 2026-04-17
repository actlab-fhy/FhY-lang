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

"""Pretty print serialization of AST nodes into FhY language."""

__all__ = ["pformat_ast"]

from collections.abc import Sequence
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
    VisitablePass,
    pformat_expression,
    register_pass,
)

from fhy.lang import ast
from fhy.lang.ast.node import Node


@register_pass(
    "fhy_ast_pretty_formatter",
    "Formats a FhY AST node back into pseudo-FhY language source.",
)
class ASTPrettyFormatter(VisitablePass[Node, str]):
    """Formats an AST node back into pseudo-FhY language source.

    The output is not guaranteed to be a valid FhY program.

    """

    _show_id: bool
    _indent_char: str
    _current_indent: int

    def __init__(self, indent_char: str, show_id: bool) -> None:
        super().__init__()
        self._show_id = show_id
        self._indent_char = indent_char
        self._current_indent = 0

    @property
    def _indentation(self) -> str:
        """Current indentations."""
        return self._indent_char * self._current_indent

    def get_noop_output(self, ir: Node) -> str:
        raise RuntimeError("This pass does not support a noop output.")

    def _increment_indent(self) -> None:
        self._current_indent += 1

    def _decrement_indent(self) -> None:
        if self._current_indent <= 0:
            raise RuntimeError("Indent cannot be negative")
        self._current_indent -= 1

    def _format_statements(self, statements: list[str]) -> str:
        return "\n".join(f"{self._indentation}{line}" for line in statements)

    def visit_module(self, module: ast.Module) -> str:
        return "\n".join(self.visit(statement) for statement in module.statements)

    def visit_import(self, node: ast.Import) -> str:
        return "import " + self._pformat_identifier(node.name) + ";"

    def visit_operation(self, operation: ast.Operation) -> str:
        self._increment_indent()
        pprinted_statements = self._format_statements(
            [self.visit(statement) for statement in operation.body]
        )
        self._decrement_indent()
        name = self._pformat_identifier(operation.name)
        templates = ", ".join(
            self._pformat_data_type(arg) for arg in operation.templates
        )
        args = ", ".join(self.visit(arg) for arg in operation.args)
        ret = self.visit(operation.return_type)

        return (
            f"op {name}<{templates}>({args}) -> {ret} "
            + "{\n"
            + pprinted_statements
            + "\n}"
        )

    def visit_procedure(self, procedure: ast.Procedure) -> str:
        self._increment_indent()
        pprinted_statements = self._format_statements(
            [self.visit(statement) for statement in procedure.body]
        )
        self._decrement_indent()
        name = self._pformat_identifier(procedure.name)
        templates = ", ".join(
            self._pformat_data_type(template) for template in procedure.templates
        )
        args = ", ".join(self.visit(arg) for arg in procedure.args)

        return (
            f"proc {name}<{templates}>({args}) " + "{\n" + pprinted_statements + "\n}"
        )

    def visit_argument(self, argument: ast.Argument) -> str:
        return (
            f"{self.visit(argument.qualified_type)} "
            f"{self._pformat_identifier(argument.name)}"
        )

    def visit_declaration_statement(
        self, declaration_statement: ast.DeclarationStatement
    ) -> str:
        left_type = self.visit(declaration_statement.variable_type)
        left_name = self._pformat_identifier(declaration_statement.variable_name)
        left = f"{left_type} {left_name}"
        if declaration_statement.expression is not None:
            right = f" = {self.visit(declaration_statement.expression)};"
        else:
            right = ";"

        return left + right

    def visit_expression_statement(
        self, expression_statement: ast.ExpressionStatement
    ) -> str:
        if expression_statement.left is not None:
            left = f"{self.visit(expression_statement.left)} = "
        else:
            left = ""

        return left + self.visit(expression_statement.right) + ";"

    def visit_selection_statement(
        self, selection_statement: ast.SelectionStatement
    ) -> str:
        self._increment_indent()
        true_body = self._format_statements(
            [self.visit(statement) for statement in selection_statement.true_body]
        )
        false_body = self._format_statements(
            [self.visit(statement) for statement in selection_statement.false_body]
        )
        self._decrement_indent()
        condition = self.visit(selection_statement.condition)

        text = f"if {condition} " + "{\n" + true_body + f"\n{self._indentation}" + "}"

        if len(false_body) > 0:
            text += " else {\n" + false_body + f"\n{self._indentation}" + "}"

        return text

    def visit_for_all_statement(self, for_all_statement: ast.ForAllStatement) -> str:
        self._increment_indent()
        pprinted_body = self._format_statements(
            [self.visit(statement) for statement in for_all_statement.body]
        )
        self._decrement_indent()
        index = self.visit(for_all_statement.index)

        return (
            f"forall ({index}) "
            + "{\n"
            + pprinted_body
            + f"\n{self._indentation}"
            + "}"
        )

    def visit_return_statement(self, return_statement: ast.ReturnStatement) -> str:
        return f"return {self.visit(return_statement.expression)};"

    def visit_unary_expression(self, unary_expression: ast.UnaryExpression) -> str:
        return (
            f"{unary_expression.operation.value}"
            f"({self.visit(unary_expression.expression)})"
        )

    def visit_binary_expression(self, binary_expression: ast.BinaryExpression) -> str:
        left = self.visit(binary_expression.left)
        right = self.visit(binary_expression.right)

        return f"({left} {binary_expression.operation.value} {right})"

    def visit_ternary_expression(
        self, ternary_expression: ast.TernaryExpression
    ) -> str:
        condition = self.visit(ternary_expression.condition)
        _true = self.visit(ternary_expression.true)
        _false = self.visit(ternary_expression.false)

        return f"({condition} ? {_true} : {_false})"

    def visit_function_expression(
        self, function_expression: ast.FunctionExpression
    ) -> str:
        template_types = ", ".join(
            self._pformat_data_type(template_type)
            for template_type in function_expression.template_types
        )
        indices = ", ".join(self.visit(index) for index in function_expression.indices)
        args = ", ".join(self.visit(arg) for arg in function_expression.args)
        func = self.visit(function_expression.function)

        return f"{func}<{template_types}>[{indices}]({args})"

    def _build_base_tuple(self, nodes: Sequence[Node]) -> str:
        a: str = "( " + ", ".join([self.visit(i) for i in nodes])
        a += ", )" if len(nodes) == 1 else " )"

        return a

    def visit_tuple_expression(self, node: ast.TupleExpression) -> str:
        return self._build_base_tuple(node.expressions)

    def visit_tuple_access_expression(self, node: ast.TupleAccessExpression) -> str:
        _tuple: str = self.visit(node.tuple_expression)
        element: str = self.visit_int_literal(node.element_index)

        return f"{_tuple}.{element}"

    def visit_array_access_expression(
        self, array_access_expression: ast.ArrayAccessExpression
    ) -> str:
        index = ", ".join(
            self.visit(index) for index in array_access_expression.indices
        )
        return f"{self.visit(array_access_expression.array_expression)}[{index}]"

    def visit_identifier_expression(
        self, identifier_expression: ast.IdentifierExpression
    ) -> str:
        return self._pformat_identifier(identifier_expression.identifier)

    def visit_int_literal(self, int_literal: ast.IntLiteral) -> str:
        return str(int_literal.value)

    def visit_float_literal(self, float_literal: ast.FloatLiteral) -> str:
        return str(float_literal.value)

    def visit_complex_literal(self, complex_literal: ast.ComplexLiteral) -> str:
        return str(complex_literal.value)

    def visit_qualified_type(self, qualified_type: ast.QualifiedType) -> str:
        return (
            f"{qualified_type.type_qualifier.value} "
            f"{self._pformat_type(qualified_type.base_type)}"
        )

    @singledispatchmethod
    def _pformat_type(self, type: Type) -> str:
        raise NotImplementedError(f"Pretty formatting type {type} not implemented")

    @_pformat_type.register(NumericalType)
    def _(self, numerical_type: NumericalType) -> str:
        if len(numerical_type.shape) == 0:
            shape = ""
        else:
            shape = ", ".join(
                pformat_expression(dim, show_id=self._show_id)
                for dim in numerical_type.shape
            )
            shape = f"[{shape}]"

        return f"{self._pformat_data_type(numerical_type.data_type)}{shape}"

    @_pformat_type.register(IndexType)
    def _(self, index_type: IndexType) -> str:
        index_range = (
            f"{pformat_expression(index_type.lower_bound, show_id=self._show_id)}:"
        )
        index_range += (
            f"{pformat_expression(index_type.upper_bound, show_id=self._show_id)}:"
        )

        if index_type.stride is not None:
            index_range += (
                f"{pformat_expression(index_type.stride, show_id=self._show_id)}"
            )
        else:
            index_range += "1"

        return f"index[{index_range}]"

    @_pformat_type.register(TupleType)
    def _(self, tuple_type: TupleType) -> str:
        return (
            "tuple ( "
            + ", ".join(
                self._pformat_type(inner_type) for inner_type in tuple_type._types
            )
            + " )"
        )

    @singledispatchmethod
    def _pformat_data_type(self, data_type: DataType) -> str:
        raise NotImplementedError(
            f"Pretty formatting data type {data_type} not implemented"
        )

    @_pformat_data_type.register(PrimitiveDataType)
    def _(self, primitive_data_type: PrimitiveDataType) -> str:
        return str(primitive_data_type.core_data_type.value)

    @_pformat_data_type.register(TemplateDataType)
    def _(self, template_data_type: TemplateDataType) -> str:
        return self._pformat_identifier(template_data_type.data_type)

    def _pformat_identifier(self, identifier: Identifier) -> str:
        if self._show_id:
            return f"({identifier.name_hint}::{identifier.id})"
        else:
            return identifier.name_hint


def pformat_ast(ast: ast.Node, indent_char: str = "  ", show_id: bool = False) -> str:
    """Returns FhY text from a given an AST node.

    Args:
        ast (ASTNode): a valid FhY AST node
        indent_char (str): character(s) used to indent the output text
        show_id (bool): Include assigned ID in output text if true

    Raises:
        RuntimeError: When class is improperly used and the indent becomes negative.

    """
    pformatter = ASTPrettyFormatter(indent_char, show_id)
    return pformatter(ast)
