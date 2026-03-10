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

"""Conversion (serialization) of the FhY AST to and from JSON format."""

from collections.abc import Sequence

from fhy.lang.ast.alias import ASTStructure

# from collections.abc import Callable, Sequence
# from typing import Any, NoReturn, TypeAlias

# from fhy_core import (
#     BinaryExpression as CoreBinaryExpression,
# )
# from fhy_core import (
#     BinaryOperation as CoreBinaryOperation,
# )
# from fhy_core import (
#     CoreDataType,
#     DataType,
#     Identifier,
#     IndexType,
#     NumericalType,
#     PrimitiveDataType,
#     TemplateDataType,
#     TupleType,
#     Type,
#     TypeQualifier,
# )
# from fhy_core import (
#     Expression as CoreExpression,
# )
# from fhy_core import (
#     IdentifierExpression as CoreIdentifierExpression,
# )
# from fhy_core import (
#     LiteralExpression as CoreLiteralExpression,
# )
# from fhy_core import (
#     UnaryExpression as CoreUnaryExpression,
# )
# from fhy_core import (
#     UnaryOperation as CoreUnaryOperation,
# )

# from fhy.lang.ast import node as ast_node
# from fhy.lang.ast.alias import ASTStructure
# from fhy.lang.ast.span import Source, Span
# from fhy.lang.ast.visitor import BasePass

# AlmostJSONAttributes: TypeAlias = dict[str, ASTStructure | str]


# def convert(value: object) -> AlmostJSONAttributes | list[AlmostJSONAttributes]:
#     """Recursively convert objects into dictionary records."""
#     if isinstance(value, AlmostJson):
#         return value.data()

#     elif isinstance(value, list):
#         return [convert(child) for child in value]

#     else:
#         return value


# class AlmostJson:
#     """Data structure format for JSON serialization preparations."""

#     cls_name: str
#     attributes: AlmostJSONAttributes

#     def __init__(self, cls_name: str, attributes: AlmostJSONAttributes) -> None:
#         self.cls_name = cls_name
#         self.attributes = attributes

#     def data(self) -> AlmostJSONAttributes:
#         """Return class attributes into a dictionary record, recursively."""
#         attributes = {
#             k: val
#             for k, v in self.attributes.items()
#             if (val := convert(v)) is not None
#         }

#         return dict(cls_name=self.cls_name, attributes=attributes)

#     def __eq__(self, value: object) -> bool:
#         return (
#             isinstance(value, AlmostJson)
#             and self.cls_name == value.cls_name
#             and self.attributes == value.attributes
#         )


# JSONObject = AlmostJson | Sequence[AlmostJson]


# def _get_node_cls_name(node: ASTStructure) -> str:
#     return node.__class__.__qualname__


# class ASTtoJSON(BasePass):
#     """Convert an AST node into a json object.

#     Args:
#         include_span (bool): Include span in output if true.

#     """

#     include_span: bool

#     def __init__(self, include_span: bool = True):
#         self.include_span = include_span

#     def default(self, node):
#         if isinstance(node, list):
#             return self.visit_sequence(node)

#         return super().default(node)

#     def visit_module(self, node: ast_node.Module) -> AlmostJson:
#         statements: list[AlmostJson] = self.visit_sequence(node.statements)
#         obj = AlmostJson(
#             cls_name=_get_node_cls_name(node),
#             attributes=dict(
#                 span=self.visit_span(node.span),
#                 statements=statements,
#             ),
#         )

#         return obj

#     def visit_operation(self, node: ast_node.Operation) -> AlmostJson:
#         templates: list[AlmostJson] = self.visit_sequence(node.templates)
#         args: list[AlmostJson] = self.visit_sequence(node.args)
#         ret_type: AlmostJson = self.visit(node.return_type)
#         body: list[AlmostJson] = self.visit_sequence(node.body)
#         name: AlmostJson = self.visit_identifier(node.name)
#         obj = AlmostJson(
#             cls_name=_get_node_cls_name(node),
#             attributes=dict(
#                 span=self.visit_span(node.span),
#                 templates=templates,
#                 name=name,
#                 args=args,
#                 return_type=ret_type,
#                 body=body,
#             ),
#         )

#         return obj

#     def visit_procedure(self, node: ast_node.Procedure) -> AlmostJson:
#         templates: list[AlmostJson] = self.visit_sequence(node.templates)
#         args: list[AlmostJson] = self.visit_sequence(node.args)
#         body: list[AlmostJson] = self.visit_sequence(node.body)
#         name: AlmostJson = self.visit_identifier(node.name)
#         obj = AlmostJson(
#             cls_name=_get_node_cls_name(node),
#             attributes=dict(
#                 span=self.visit_span(node.span),
#                 templates=templates,
#                 name=name,
#                 args=args,
#                 body=body,
#             ),
#         )

#         return obj

#     def visit_import(self, node: ast_node.Import) -> AlmostJson:
#         identifier: AlmostJson = self.visit_identifier(node.name)
#         return AlmostJson(
#             cls_name=_get_node_cls_name(node),
#             attributes=dict(span=self.visit_span(node.span), name=identifier),
#         )

#     def visit_argument(self, node: ast_node.Argument) -> AlmostJson:
#         qtype: AlmostJson = self.visit_qualified_type(node.qualified_type)
#         name: AlmostJson | None = (
#             self.visit_identifier(node.name) if node.name is not None else None
#         )
#         obj = AlmostJson(
#             cls_name=_get_node_cls_name(node),
#             attributes=dict(
#                 span=self.visit_span(node.span), qualified_type=qtype, name=name
#             ),
#         )

#         return obj

#     def visit_declaration_statement(
#         self, node: ast_node.DeclarationStatement
#     ) -> AlmostJson:
#         varname: AlmostJson = self.visit_identifier(node.variable_name)
#         vartype: AlmostJson = self.visit_qualified_type(node.variable_type)
#         express: AlmostJson | None = (
#             self.visit(node.expression) if node.expression is not None else None
#         )
#         obj = AlmostJson(
#             cls_name=_get_node_cls_name(node),
#             attributes=dict(
#                 span=self.visit_span(node.span),
#                 variable_name=varname,
#                 variable_type=vartype,
#                 expression=express,
#             ),
#         )

#         return obj

#     def visit_expression_statement(
#         self, node: ast_node.ExpressionStatement
#     ) -> AlmostJson:
#         left: AlmostJson | None = (
#             self.visit(node.left) if node.left is not None else None
#         )
#         right: AlmostJson = self.visit(node.right)
#         obj = AlmostJson(
#             cls_name=_get_node_cls_name(node),
#             attributes=dict(
#                 span=self.visit_span(node.span),
#                 left=left,
#                 right=right,
#             ),
#         )

#         return obj

#     def visit_selection_statement(
#         self, node: ast_node.SelectionStatement
#     ) -> AlmostJson:
#         condition: AlmostJson = self.visit(node.condition)
#         tbody: list[AlmostJson] = self.visit_sequence(node.true_body)
#         fbody: list[AlmostJson] = self.visit_sequence(node.false_body)
#         obj = AlmostJson(
#             cls_name=_get_node_cls_name(node),
#             attributes=dict(
#                 span=self.visit_span(node.span),
#                 condition=condition,
#                 true_body=tbody,
#                 false_body=fbody,
#             ),
#         )

#         return obj

#     def visit_for_all_statement(self, node: ast_node.ForAllStatement) -> AlmostJson:
#         index: AlmostJson = self.visit(node.index)
#         body: list[AlmostJson] = self.visit_sequence(node.body)
#         obj = AlmostJson(
#             cls_name=_get_node_cls_name(node),
#             attributes=dict(
#                 span=self.visit_span(node.span),
#                 index=index,
#                 body=body,
#             ),
#         )

#         return obj

#     def visit_return_statement(self, node: ast_node.ReturnStatement) -> AlmostJson:
#         express: AlmostJson = self.visit(node.expression)
#         obj = AlmostJson(
#             cls_name=_get_node_cls_name(node),
#             attributes=dict(
#                 span=self.visit_span(node.span),
#                 expression=express,
#             ),
#         )

#         return obj

#     def visit_unary_expression(self, node: ast_node.UnaryExpression) -> AlmostJson:
#         express: AlmostJson = self.visit(node.expression)
#         obj = AlmostJson(
#             cls_name=_get_node_cls_name(node),
#             attributes=dict(
#                 span=self.visit_span(node.span),
#                 expression=express,
#                 operation=node.operation.value,
#             ),
#         )

#         return obj

#     def visit_binary_expression(self, node: ast_node.BinaryExpression) -> AlmostJson:
#         left: AlmostJson = self.visit(node.left)
#         right: AlmostJson = self.visit(node.right)
#         obj = AlmostJson(
#             cls_name=_get_node_cls_name(node),
#             attributes=dict(
#                 span=self.visit_span(node.span),
#                 left=left,
#                 operation=node.operation.value,
#                 right=right,
#             ),
#         )

#         return obj

#     def visit_ternary_expression(self,
#   node: ast_node.TernaryExpression) -> AlmostJson:
#         condition: AlmostJson = self.visit(node.condition)
#         true: AlmostJson = self.visit(node.true)
#         false: AlmostJson = self.visit(node.false)
#         obj = AlmostJson(
#             cls_name=_get_node_cls_name(node),
#             attributes=dict(
#                 span=self.visit_span(node.span),
#                 condition=condition,
#                 true=true,
#                 false=false,
#             ),
#         )

#         return obj

#     def visit_function_expression(
#         self, node: ast_node.FunctionExpression
#     ) -> AlmostJson:
#         function: AlmostJson = self.visit(node.function)
#         template: list[AlmostJson] = self.visit_sequence(node.template_types)
#         index: list[AlmostJson] = self.visit_sequence(node.indices)
#         args: list[AlmostJson] = self.visit_sequence(node.args)
#         obj = AlmostJson(
#             cls_name=_get_node_cls_name(node),
#             attributes=dict(
#                 span=self.visit_span(node.span),
#                 function=function,
#                 template_types=template,
#                 indices=index,
#                 args=args,
#             ),
#         )

#         return obj

#     def visit_array_access_expression(
#         self, node: ast_node.ArrayAccessExpression
#     ) -> AlmostJson:
#         array: AlmostJson = self.visit(node.array_expression)
#         index: list[AlmostJson] = self.visit_sequence(node.indices)
#         obj = AlmostJson(
#             cls_name=_get_node_cls_name(node),
#             attributes=dict(
#                 span=self.visit_span(node.span),
#                 array_expression=array,
#                 indices=index,
#             ),
#         )

#         return obj

#     def visit_tuple_expression(self, node: ast_node.TupleExpression) -> AlmostJson:
#         express: list[AlmostJson] = self.visit_sequence(node.expressions)
#         obj = AlmostJson(
#             cls_name=_get_node_cls_name(node),
#             attributes=dict(
#                 span=self.visit_span(node.span),
#                 expressions=express,
#             ),
#         )

#         return obj

#     def visit_tuple_access_expression(
#         self, node: ast_node.TupleAccessExpression
#     ) -> AlmostJson:
#         _tuple: AlmostJson = self.visit(node.tuple_expression)
#         element: AlmostJson = self.visit_int_literal(node.element_index)
#         obj = AlmostJson(
#             cls_name=_get_node_cls_name(node),
#             attributes=dict(
#                 span=self.visit_span(node.span),
#                 tuple_expression=_tuple,
#                 element_index=element,
#             ),
#         )

#         return obj

#     def visit_identifier_expression(
#         self, node: ast_node.IdentifierExpression
#     ) -> AlmostJson:
#         identifier: AlmostJson = self.visit(node.identifier)
#         obj = AlmostJson(
#             cls_name=_get_node_cls_name(node),
#             attributes=dict(span=self.visit_span(node.span), identifier=identifier),
#         )

#         return obj

#     def visit_int_literal(self, node: ast_node.IntLiteral) -> AlmostJson:
#         return AlmostJson(
#             cls_name=_get_node_cls_name(node),
#             attributes=dict(span=self.visit_span(node.span), value=node.value),
#         )

#     def visit_float_literal(self, node: ast_node.FloatLiteral) -> AlmostJson:
#         return AlmostJson(
#             cls_name=_get_node_cls_name(node),
#             attributes=dict(span=self.visit_span(node.span), value=node.value),
#         )

#     def visit_complex_literal(self, node: ast_node.ComplexLiteral) -> AlmostJson:
#         result = dict(real=node.value.real, imag=node.value.imag)

#         return AlmostJson(
#             cls_name=_get_node_cls_name(node),
#             attributes=dict(span=self.visit_span(node.span), value=result),
#         )

#     def visit_primitive_data_type(self, node: PrimitiveDataType) -> AlmostJson:
#         return AlmostJson(
#             cls_name=_get_node_cls_name(node),
#             attributes=dict(core_data_type=node.core_data_type.value),
#         )

#     def visit_template_data_type(self, node: TemplateDataType) -> AlmostJson:
#         return AlmostJson(
#             cls_name=_get_node_cls_name(node),
#             attributes=dict(data_type=self.visit(node.template_type)),
#         )

#     def visit_qualified_type(self, node: ast_node.QualifiedType) -> AlmostJson:
#         base: AlmostJson = self.visit(node.base_type)

#         return AlmostJson(
#             cls_name=_get_node_cls_name(node),
#             attributes=dict(
#                 span=self.visit_span(node.span),
#                 base_type=base,
#                 type_qualifier=node.type_qualifier.value,
#             ),
#         )

#     def visit_numerical_type(self, numerical_type: NumericalType) -> AlmostJson:
#         dtype: AlmostJson = self.visit(numerical_type.data_type)
#         shape: list[AlmostJson] = [
#             self.visit_core_expression(e) for e in numerical_type.shape
#         ]

#         return AlmostJson(
#             cls_name=_get_node_cls_name(numerical_type),
#             attributes=dict(data_type=dtype, shape=shape),
#         )

#     def visit_index_type(self, index_type: IndexType) -> AlmostJson:
#         lower: AlmostJson = self.visit_core_expression(index_type.lower_bound)
#         upper: AlmostJson = self.visit_core_expression(index_type.upper_bound)
#         if index_type.stride is not None:
#             stride: AlmostJson = self.visit_core_expression(index_type.stride)
#         else:
#             stride = self.visit_core_expression(CoreLiteralExpression(1))

#         return AlmostJson(
#             cls_name=_get_node_cls_name(index_type),
#             attributes=dict(lower_bound=lower, upper_bound=upper, stride=stride),
#         )

#     def visit_tuple_type(self, tuple_type: TupleType) -> AlmostJson:
#         types: list[AlmostJson] = self.visit_sequence(tuple_type._types)

#         return AlmostJson(
#             cls_name=_get_node_cls_name(tuple_type),
#             attributes=dict(types=types),
#         )

#     def visit_core_expression(self, expression: CoreExpression) -> AlmostJson:
#         if isinstance(expression, CoreBinaryExpression):
#             return self.visit_core_binary_expression(expression)
#         elif isinstance(expression, CoreUnaryExpression):
#             return self.visit_core_unary_expression(expression)
#         elif isinstance(expression, CoreLiteralExpression):
#             return self.visit_core_literal_expression(expression)
#         elif isinstance(expression, CoreIdentifierExpression):
#             return self.visit_core_Identifier_expression(expression)
#         else:
#             raise ValueError("Invalid core expression.")

#     def visit_core_binary_expression(
#         self, expression: CoreBinaryExpression
#     ) -> AlmostJson:
#         left: AlmostJson = self.visit_core_expression(expression.left)
#         right: AlmostJson = self.visit_core_expression(expression.right)

#         return AlmostJson(
#             cls_name="CoreBinaryExpression",
#             attributes=dict(
#                 left=left, operation=expression.operation.value, right=right
#             ),
#         )

#     def visit_core_unary_expression(
#         self, expression: CoreUnaryExpression
#     ) -> AlmostJson:
#         operand: AlmostJson = self.visit_core_expression(expression.operand)

#         return AlmostJson(
#             cls_name="CoreUnaryExpression",
#             attributes=dict(operand=operand, operation=expression.operation.value),
#         )

#     def visit_core_literal_expression(
#         self, expression: CoreLiteralExpression
#     ) -> AlmostJson:
#         return AlmostJson(
#             cls_name="CoreLiteralExpression",
#             attributes=dict(value=expression.value),
#         )

#     def visit_core_Identifier_expression(
#         self, expression: CoreIdentifierExpression
#     ) -> AlmostJson:
#         identifier = self.visit_identifier(expression.identifier)
#         return AlmostJson(
#             cls_name="CoreIdentifierExpression",
#             attributes=dict(name=identifier),
#         )

#     def visit_identifier(self, identifier: Identifier) -> AlmostJson:
#         return AlmostJson(
#             cls_name=_get_node_cls_name(identifier),
#             attributes=dict(name_hint=identifier.name_hint, _id=identifier.id),
#         )

#     def visit_sequence(self, nodes: Sequence[ASTStructure]) -> list[AlmostJson]:
#         return [self.visit(node) for node in nodes]

#     def visit_span(self, span: Span | None) -> AlmostJson | None:
#         if span is None or not self.include_span:
#             return None

#         if span.source is not None:
#             source: AlmostJson | None = self.visit_source(span.source)
#         else:
#             source = span.source

#         return AlmostJson(
#             cls_name=_get_node_cls_name(span),
#             attributes=dict(
#                 start_line=span.line.start,
#                 end_line=span.line.stop,
#                 start_column=span.column.start,
#                 end_column=span.column.stop,
#                 source=source,
#             ),
#         )

#     def visit_source(self, source: Source) -> AlmostJson:
#         return AlmostJson(
#             cls_name=_get_node_cls_name(source),
#             attributes=dict(namespace=str(source.namespace)),
#         )


# def _raise_invalid_object_type(
#     obj: ASTStructure, expected: type[ASTStructure], name: str
# ) -> NoReturn:
#     raise ValueError(
#         f"Invalid object type for {name}. Expected {expected.__name__} "
#         f"but got {type(obj).__name__}"
#     )


# def _raise_invalid_sequence_type(
#     sequence: Sequence[ASTStructure], expected: type[ASTStructure], name: str
# ) -> NoReturn:
#     raise ValueError(
#         f"Invalid sequence type. Expected {expected.__name__} but got "
#         f" {type(sequence[0]).__name__} for {name}"
#     )


# class JSONtoAST(BasePass):
#     """Converts a JSON object into AST nodes."""

#     def visit(self, node: JSONObject | None) -> Any:
#         if isinstance(node, list):
#             return self.visit_sequence(node)

#         elif not isinstance(node, AlmostJson):
#             return self.default(node)

#         name = f"visit_{node.cls_name}"
#         method: Callable[[JSONObject], ASTStructure]
#         method = getattr(self, name, self.default)

#         return method(node)

#     def visit_module(self, node: AlmostJson | None) -> ast_node.Module:
#         if node is None:
#             raise ValueError("Invalid Module")

#         values = node.attributes
#         span = self._get_span_from_attrs(values)
#         statements = self._get_statements_from_attrs(values, "statements")

#         return ast_node.Module(span=span, statements=statements)

#     def visit_operation(self, node: AlmostJson | None) -> ast_node.Operation:
#         if node is None:
#             raise ValueError("Invalid Operation")

#         values = node.attributes

#         span = self._get_span_from_attrs(values)
#         templates: list[TemplateDataType] = self.visit_sequence(
# values.get("templates"))
#         args: list[ast_node.Argument] = self.visit_sequence(values.get("args"))
#         body = self._get_statements_from_attrs(values, "body")
#         name = self._get_identifier_from_attrs(values, "name")
#         ret_type: ast_node.QualifiedType = self.visit_qualified_type(
#             values.get("return_type")
#         )

#         return ast_node.Operation(
#             span=span,
#             name=name,
#             templates=templates,
#             args=args,
#             body=body,
#             return_type=ret_type,
#         )

#     def visit_procedure(self, node: AlmostJson | None) -> ast_node.Procedure:
#         if node is None:
#             raise ValueError("Invalid Procedure")

#         values = node.attributes
#         span = self._get_span_from_attrs(values)
#         templates: list[TemplateDataType] = self.visit_sequence(
# values.get("templates"))
#         args: list[ast_node.Argument] = self.visit_sequence(values.get("args"))
#         body = self._get_statements_from_attrs(values, "body")
#         name = self._get_identifier_from_attrs(values, "name")

#         return ast_node.Procedure(
#             span=span, name=name, templates=templates, args=args, body=body
#         )

#     def visit_import(self, node: AlmostJson | None) -> ast_node.Import:
#         if node is None:
#             raise ValueError("Invalid Import statement")

#         values = node.attributes
#         span = self._get_span_from_attrs(values)
#         name = self._get_identifier_from_attrs(values, "name")

#         return ast_node.Import(span=span, name=name)

#     def visit_argument(self, node: AlmostJson | None) -> ast_node.Argument:
#         if node is None:
#             raise ValueError("Invalid Argument")

#         values = node.attributes
#         span = self._get_span_from_attrs(values)
#         qtype: ast_node.QualifiedType = self.visit_qualified_type(
#             values.get("qualified_type")
#         )
#         name: Identifier = self.visit_identifier(values.get("name"))

#         return ast_node.Argument(span=span, name=name, qualified_type=qtype)

#     def _get_statements_from_attrs(
#         self, values: AlmostJSONAttributes, key_name: str
#     ) -> list[ast_node.Statement]:
#         potential_statements = values.get(key_name)
#         if not isinstance(potential_statements, list):
#             _raise_invalid_object_type(potential_statements, list, "statements")
#         if any(
#             not isinstance(stmt, ast_node.Statement) for stmt in potential_statements
#         ):
#             _raise_invalid_sequence_type(
#                 potential_statements, ast_node.Statement, "statements"
#             )
#         return potential_statements

#     def visit_declaration_statement(
#         self, node: AlmostJson | None
#     ) -> ast_node.DeclarationStatement:
#         if node is None:
#             raise ValueError("Invalid DeclarationStatement")

#         values = node.attributes
#         span = self._get_span_from_attrs(values)
#         varname = self.visit_identifier(values.get("variable_name"))
#         vartype: ast_node.QualifiedType = self.visit_qualified_type(
#             values.get("variable_type")
#         )
#         if (_expression := values.get("expression")) is not None:
#             values["expression"] = self.visit(_expression)
#         expression = self._get_expression_from_attrs(values, "expression")

#         return ast_node.DeclarationStatement(
#             span=span,
#             variable_name=varname,
#             variable_type=vartype,
#             expression=expression,
#         )

#     def visit_expression_statement(
#         self, node: AlmostJson | None
#     ) -> ast_node.ExpressionStatement:
#         if node is None:
#             raise ValueError("Invalid ExpressionStatement")

#         values = node.attributes
#         span = self._get_span_from_attrs(values)
#         if (_left := values.get("left")) is not None:
#             values["left"] = self.visit(_left)
#         left = self._get_expression_from_attrs(values, "left")
#         right = self._get_expression_from_attrs(values, "right")

#         return ast_node.ExpressionStatement(span=span, left=left, right=right)

#     def visit_selection_statement(
#         self, node: AlmostJson | None
#     ) -> ast_node.SelectionStatement:
#         if node is None:
#             raise ValueError("Invalid SelectionStatement")

#         values = node.attributes
#         span = self._get_span_from_attrs(values)
#         condition = self._get_expression_from_attrs(values, "condition")
#         tbody = self._get_statements_from_attrs(values, "true_body")
#         fbody = self._get_statements_from_attrs(values, "false_body")

#         return ast_node.SelectionStatement(
#             span=span, condition=condition, true_body=tbody, false_body=fbody
#         )

#     def visit_for_all_statement(
#         self, node: AlmostJson | None
#     ) -> ast_node.ForAllStatement:
#         if node is None:
#             raise ValueError("Invalid ForAllStatement")

#         values = node.attributes
#         span = self._get_span_from_attrs(values)
#         index = self._get_expression_from_attrs(values, "index")
#         body = self._get_statements_from_attrs(values, "body")

#         return ast_node.ForAllStatement(span=span, index=index, body=body)

#     def visit_return_statement(
#         self, node: AlmostJson | None
#     ) -> ast_node.ReturnStatement:
#         if node is None:
#             raise ValueError("Invalid ReturnStatement")

#         values = node.attributes
#         span = self._get_span_from_attrs(values)
#         expression = self._get_expression_from_attrs(values, "expression")

#         return ast_node.ReturnStatement(span=span, expression=expression)

#     def _get_expression_sequence_from_attrs(
#         self, values: AlmostJSONAttributes, key_name: str
#     ) -> list[ast_node.Expression]:
#         potential_expressions = values.get(key_name)
#         if not isinstance(potential_expressions, list):
#             _raise_invalid_object_type(potential_expressions, list, "expressions")
#         if any(
#             not isinstance(expr, ast_node.Expression) for expr in \
# potential_expressions
#         ):
#             _raise_invalid_sequence_type(
#                 potential_expressions, ast_node.Expression, "expressions"
#             )
#         return potential_expressions

#     def _get_expression_from_attrs(
#         self, values: AlmostJSONAttributes, key_name: str
#     ) -> ast_node.Expression:
#         potential_expression = values.get(key_name)
#         if not isinstance(potential_expression, AlmostJson):
#             _raise_invalid_object_type(potential_expression, AlmostJson, "expression")
#         return self.visit(potential_expression)

#     def visit_unary_expression(
#         self, node: AlmostJson | None
#     ) -> ast_node.UnaryExpression:
#         if node is None:
#             raise ValueError("Invalid UnaryExpression")

#         values = node.attributes
#         span = self._get_span_from_attrs(values)
#         expression = self._get_expression_from_attrs(values, "expression")
#         operator: ast_node.UnaryOperation = ast_node.UnaryOperation(
#             str(values.get("operation"))
#         )

#         return ast_node.UnaryExpression(
#             span=span, operation=operator, expression=expression
#         )

#     def visit_binary_expression(
#         self, node: AlmostJson | None
#     ) -> ast_node.BinaryExpression:
#         if node is None:
#             raise ValueError("Invalid BinaryExpression")

#         values = node.attributes
#         span = self._get_span_from_attrs(values)
#         left = self._get_expression_from_attrs(values, "left")
#         right = self._get_expression_from_attrs(values, "right")
#         operator = ast_node.BinaryOperation(str(values.get("operation")))

#         return ast_node.BinaryExpression(
#             span=span, left=left, right=right, operation=operator
#         )

#     def visit_ternary_expression(
#         self, node: AlmostJson | None
#     ) -> ast_node.TernaryExpression:
#         if node is None:
#             raise ValueError("Invalid TernaryExpression")

#         values = node.attributes
#         span = self._get_span_from_attrs(values)
#         condition = self._get_expression_from_attrs(values, "condition")
#         true = self._get_expression_from_attrs(values, "true")
#         false = self._get_expression_from_attrs(values, "false")

#         return ast_node.TernaryExpression(
#             span=span, condition=condition, true=true, false=false
#         )

#     def visit_function_expression(
#         self, node: AlmostJson | None
#     ) -> ast_node.FunctionExpression:
#         if node is None:
#             raise ValueError("Invalid FunctionExpression")

#         values = node.attributes
#         span = self._get_span_from_attrs(values)
#         function = self._get_expression_from_attrs(values, "function")
#         template: list[DataType] = self.visit_sequence(values.get("template_types"))
#         index = self._get_expression_sequence_from_attrs(values, "indices")
#         args = self._get_expression_sequence_from_attrs(values, "args")

#         return ast_node.FunctionExpression(
#             span=span,
#             function=function,
#             template_types=template,
#             indices=index,
#             args=args,
#         )

#     def visit_array_access_expression(
#         self, node: AlmostJson | None
#     ) -> ast_node.ArrayAccessExpression:
#         if node is None:
#             raise ValueError("Invalid ArrayAccessExpression")

#         values = node.attributes
#         span = self._get_span_from_attrs(values)
#         array = self._get_expression_from_attrs(values, "array_expression")
#         index = self._get_expression_sequence_from_attrs(values, "indices")

#         return ast_node.ArrayAccessExpression(
#             span=span,
#             array_expression=array,
#             indices=index,
#         )

#     def visit_tuple_expression(
#         self, node: AlmostJson | None
#     ) -> ast_node.TupleExpression:
#         if node is None:
#             raise ValueError("Invalid TupleExpression")

#         values = node.attributes
#         span = self._get_span_from_attrs(values)
#         expressions = self._get_expression_sequence_from_attrs(values, "expressions")

#         return ast_node.TupleExpression(span=span, expressions=expressions)

#     def visit_tuple_access_expression(
#         self, node: AlmostJson | None
#     ) -> ast_node.TupleAccessExpression:
#         if node is None:
#             raise ValueError("Invalid TupleAccessExpression")

#         values = node.attributes
#         span = self._get_span_from_attrs(values)
#         _tuple = self.visit_tuple_expression(values.get("tuple_expression"))
#         element: ast_node.IntLiteral = self.visit_int_literal(
#             values.get("element_index")
#         )

#         return ast_node.TupleAccessExpression(
#             span=span, tuple_expression=_tuple, element_index=element
#         )

#     def visit_identifier_expression(
#         self, node: AlmostJson | None
#     ) -> ast_node.IdentifierExpression:
#         if node is None:
#             raise ValueError("Invalid IdentifierExpression")

#         values = node.attributes
#         identifier = self._get_identifier_from_attrs(values, "identifier")
#         span = self._get_span_from_attrs(values)

#         return ast_node.IdentifierExpression(span=span, identifier=identifier)

#     def visit_int_literal(self, node: AlmostJson | None) -> ast_node.IntLiteral:
#         if node is None:
#             raise ValueError("Invalid IntLiteral")

#         values = node.attributes
#         span = self._get_span_from_attrs(values)
#         if (value := values.get("value")) is None:
#             raise ValueError("Invalid IntLiteral Value")

#         return ast_node.IntLiteral(span=span, value=value)

#     def visit_float_literal(self, node: AlmostJson | None) -> ast_node.FloatLiteral:
#         if node is None:
#             raise ValueError("Invalid FloatLiteral")

#         values = node.attributes
#         span = self._get_span_from_attrs(values)
#         if (value := values.get("value")) is None:
#             raise ValueError("Invalid FloatLiteral Value")

#         return ast_node.FloatLiteral(span=span, value=value)

#     def visit_complex_literal(self,
# node: AlmostJson | None) -> ast_node.ComplexLiteral:
#         if node is None:
#             raise ValueError("Invalid ComplexLiteral")

#         values = node.attributes
#         span = self._get_span_from_attrs(values)
#         if (value := values.get("value")) is None:
#             raise ValueError("Invalid ComplexLiteral Value")

#         # Combine real and imaginary parts to construct the complex number
#         result: complex = complex(real=value.get("real"), imag=value.get("imag"))

#         return ast_node.ComplexLiteral(span=span, value=result)

#     def visit_primitive_data_type(self, node: AlmostJson | None) -> PrimitiveDataType:
#         if node is None:
#             raise ValueError("Invalid DataType")

#         core = CoreDataType(str(node.attributes.get("core_data_type")))

#         return PrimitiveDataType(core_data_type=core)

#     def visit_template_data_type(self, node: AlmostJson | None) -> TemplateDataType:
#         if node is None:
#             raise ValueError("Invalid DataType")

#         template = self.visit(node.attributes.get("data_type"))

#         return TemplateDataType(data_type=template)

#     def visit_qualified_type(self,
# node: AlmostJson | None) -> ast_node.QualifiedType:
#         if node is None:
#             raise ValueError("Invalid QualifiedType")

#         values = node.attributes
#         base: Type = self.visit(values.get("base_type"))
#         span = self._get_span_from_attrs(values)
#         qtype = TypeQualifier(str(values.get("type_qualifier")))

#         return ast_node.QualifiedType(
#             span=span,
#             base_type=base,
#             type_qualifier=qtype,
#         )

#     def visit_numerical_type(self,
# numerical_type: AlmostJson | None) -> NumericalType:
#         if numerical_type is None:
#             raise ValueError("Invalid numerical_type")

#         values = numerical_type.attributes
#         dtype: type.DataType = self.visit(values.get("data_type"))
#         shape: list[CoreExpression] = [
#             self.visit_core_expression(e) for e in values.get("shape")
#         ]

#         return NumericalType(data_type=dtype, shape=shape)

#     def visit_index_type(self, index_type: AlmostJson | None) -> IndexType:
#         if index_type is None:
#             raise ValueError("No Index Type Provided")

#         values = index_type.attributes
#         lower: CoreExpression = self.visit_core_expression(
# values.get("lower_bound"))
#         upper: CoreExpression = self.visit_core_expression(
# values.get("upper_bound"))

#         if (_stride := values.get("stride")) is not None:
#             stride = self.visit_core_expression(_stride)
#         else:
#             stride = self.visit_core_expression(
#                 AlmostJson(
#                     cls_name="CoreLiteralExpression",
#                     attributes=dict(
#                         value=1,
#                     ),
#                 )
#             )

#         return IndexType(
#             lower_bound=lower,
#             upper_bound=upper,
#             stride=stride,
#         )

#     def _get_core_expression_from_attrs(
#         self, values: AlmostJSONAttributes, key_name: str
#     ) -> CoreExpression:
#         potential_expression = values.get(key_name)
#         if not isinstance(potential_expression, AlmostJson):
#             _raise_invalid_object_type(potential_expression, AlmostJson, "expression")
#         return self.visit_core_expression(potential_expression)

#     def visit_core_expression(self, expression: AlmostJson | None) -> CoreExpression:
#         if expression is None:
#             raise ValueError("No Core Expression Provided")

#         name = expression.cls_name
#         if name == "CoreBinaryExpression":
#             return self.visit_core_binary_expression(expression)
#         elif name == "CoreUnaryExpression":
#             return self.visit_core_unary_expression(expression)
#         elif name == "CoreLiteralExpression":
#             return self.visit_core_literal_expression(expression)
#         elif name == "CoreIdentifierExpression":
#             return self.visit_core_identifier_expression(expression)
#         else:
#             raise ValueError("Invalid Core Expression.")

#     def visit_core_binary_expression(
#         self, expression: AlmostJson | None
#     ) -> CoreBinaryExpression:
#         if expression is None:
#             raise ValueError("No Core Binary Expression Provided")

#         values = expression.attributes
#         left = self._get_core_expression_from_attrs(values, "left")
#         right = self._get_core_expression_from_attrs(values, "right")

#         return CoreBinaryExpression(
#             left=left,
#             operation=CoreBinaryOperation(str(values.get("operation"))),
#             right=right,
#         )

#     def visit_core_unary_expression(
#         self, expression: AlmostJson | None
#     ) -> CoreUnaryExpression:
#         if expression is None:
#             raise ValueError("No Core Unary Expression Provided")

#         values = expression.attributes
#         operand = self._get_core_expression_from_attrs(values, "operand")

#         return CoreUnaryExpression(
#             operand=operand,
#             operation=CoreUnaryOperation(str(values.get("operation"))),
#         )

#     def visit_core_literal_expression(
#         self, expression: AlmostJson | None
#     ) -> CoreLiteralExpression:
#         if expression is None:
#             raise ValueError("No Core Literal Expression Provided")

#         values = expression.attributes
#         value = values.get("value")

#         return CoreLiteralExpression(value=value)

#     def visit_core_identifier_expression(
#         self, expression: AlmostJson | None
#     ) -> CoreIdentifierExpression:
#         if expression is None:
#             raise ValueError("No Core Identifier Expression Provided")

#         values = expression.attributes
#         identifier = self._get_identifier_from_attrs(values, "identifier")

#         return CoreIdentifierExpression(identifier=identifier)

#     def visit_tuple_type(self, tuple_type: AlmostJson | None) -> TupleType:
#         if tuple_type is None:
#             raise ValueError("No Tuple Type Provided.")

#         values = tuple_type.attributes
#         if (v := values.get("types")) is None:
#             raise ValueError("Invalid Tuple Type. No Type definitions of Elements.")

#         types: list[Type] = self.visit_sequence(v)

#         return TupleType(types=types)

#     def _get_identifier_from_attrs(
#         self, values: AlmostJSONAttributes, key_name: str
#     ) -> Identifier:
#         potential_identifier = values.get(key_name)
#         if not isinstance(potential_identifier, AlmostJson):
#             _raise_invalid_object_type(potential_identifier, AlmostJson, "identifier")
#         return self.visit_identifier(potential_identifier)

#     def visit_identifier(self, identifier: AlmostJson | None) -> Identifier:
#         if identifier is None:
#             raise ValueError("No Identifier Provided")

#         if (hint := identifier.attributes.get("name_hint")) is None:
#             raise ValueError("Invalid Identifier Name")

#         identity = Identifier(name_hint=hint)

#         if (value := identifier.attributes.get("_id")) is None:
#             raise ValueError("Invalid ID.")

#         # NOTE: We are Hacking the Identifier Class, which automatically
#         #       assigns an ID.
#         identity._id = value

#         return identity

#     def visit_sequence(self, nodes: list[AlmostJson] | None) -> list[ASTStructure]:
#         if nodes is None:
#             return []

#         return [self.visit(node) for node in nodes]

#     def _get_span_from_attrs(self, attrs: dict) -> Span | None:
#         potential_span = attrs.get("span")
#         if not isinstance(potential_span, Span | None):
#             _raise_invalid_object_type(potential_span, Span, "span")
#         return self.visit_span(potential_span)

#     def visit_span(self, span: AlmostJson | None) -> Span | None:
#         if span is None:
#             return None

#         values = span.attributes

#         source: Source | None = None
#         if (_source := values.get("source")) is not None:
#             source = self.visit_source(_source)
#         maybe = dict(source=source)

#         return Span(
#             start_column=values.get("start_column") or 0,
#             end_column=values.get("end_column") or 0,
#             start_line=values.get("start_line") or 0,
#             end_line=values.get("end_line") or 0,
#             **{k: v for k, v in maybe.items() if v is not None},
#         )

#     def visit_source(self, source: AlmostJson) -> Source:
#         return Source(namespace=source.attributes.get("namespace") or "_null")


def dump(
    node: ASTStructure | Sequence[ASTStructure],
    indent: int | str | None = "  ",
    include_span: bool = True,
) -> str:
    """Serialize an AST node to json string, with a given indent.

    Args:
        node (ASTStructure | List[ASTStructure]): FhY AST-like node(s).
        indent (optional, str | int): indentation of json output for human readability.
            If an integer is provided, it indicates the number of spaces used.
        include_span (bool): If true, include span in json output.

    Returns:
        (str): json serialized node(s)

    """
    # to_json = ASTtoJSON(include_span)
    # obj: AlmostJson | list[AlmostJson] = to_json.visit(node)
    # data = [j.data() for j in obj] if isinstance(obj, list) else obj.data()

    # return json.dumps(data, indent=indent)
    return "{}"


# def to_almost_json(obj: Any):
#     """Json object hook to convert to AlmostJson object if matches criteria."""
#     if isinstance(obj, dict) and set(obj.keys()) == {"cls_name", "attributes"}:
#         return AlmostJson(
#             cls_name=obj["cls_name"],
#             attributes=obj["attributes"],
#         )

#     return obj


def load(json_string: str, **kwargs: object) -> ASTStructure | Sequence[ASTStructure]:
    """Loads a Json string to construct an ASTNode."""
    # node_almost: AlmostJson = json.loads(
    #     json_string, object_hook=to_almost_json, **kwargs
    # )

    # return JSONtoAST().visit(node_almost)
    return []
