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

"""Converter from AST expressions to core expressions."""

from typing import ClassVar, NoReturn, cast

from fhy_core import BinaryExpression as CoreBinaryExpression
from fhy_core import BinaryOperation as CoreBinaryOperation
from fhy_core import Expression as CoreExpression
from fhy_core import IdentifierExpression as CoreIdentifierExpression
from fhy_core import LiteralExpression as CoreLiteralExpression
from fhy_core import UnaryExpression as CoreUnaryExpression
from fhy_core import UnaryOperation as CoreUnaryOperation
from frozendict import frozendict

from fhy.lang.ast.node import ASTNode
from fhy.lang.ast.node import BinaryExpression as ASTBinaryExpression
from fhy.lang.ast.node import BinaryOperation as ASTBinaryOperation
from fhy.lang.ast.node import ComplexLiteral as ASTComplexLiteralExpression
from fhy.lang.ast.node import Expression as ASTExpression
from fhy.lang.ast.node import FloatLiteral as ASTFloatLiteralExpression
from fhy.lang.ast.node import IdentifierExpression as ASTIdentifierExpression
from fhy.lang.ast.node import IntLiteral as ASTIntLiteralExpression
from fhy.lang.ast.node import UnaryExpression as ASTUnaryExpression
from fhy.lang.ast.node import UnaryOperation as ASTUnaryOperation
from fhy.lang.ast.visitor import BasePass


class ASTToCoreExpressionConverter(BasePass):
    """Convert AST expressions to core expressions."""

    _AST_TO_CORE_UNARY_OPERATIONS: ClassVar[
        frozendict[ASTUnaryOperation, CoreUnaryOperation]
    ] = frozendict(
        {
            ASTUnaryOperation.NEGATION: CoreUnaryOperation.NEGATE,
            ASTUnaryOperation.LOGICAL_NOT: CoreUnaryOperation.LOGICAL_NOT,
        }
    )
    _AST_TO_CORE_BINARY_OPERATIONS: ClassVar[
        frozendict[ASTBinaryOperation, CoreBinaryOperation]
    ] = frozendict(
        {
            ASTBinaryOperation.ADDITION: CoreBinaryOperation.ADD,
            ASTBinaryOperation.SUBTRACTION: CoreBinaryOperation.SUBTRACT,
            ASTBinaryOperation.MULTIPLICATION: CoreBinaryOperation.MULTIPLY,
            ASTBinaryOperation.DIVISION: CoreBinaryOperation.DIVIDE,
            ASTBinaryOperation.FLOORDIV: CoreBinaryOperation.FLOOR_DIVIDE,
            ASTBinaryOperation.MODULO: CoreBinaryOperation.MODULO,
            ASTBinaryOperation.POWER: CoreBinaryOperation.POWER,
            ASTBinaryOperation.EQUAL_TO: CoreBinaryOperation.EQUAL,
            ASTBinaryOperation.NOT_EQUAL_TO: CoreBinaryOperation.NOT_EQUAL,
            ASTBinaryOperation.LESS_THAN: CoreBinaryOperation.LESS,
            ASTBinaryOperation.LESS_THAN_OR_EQUAL: CoreBinaryOperation.LESS_EQUAL,
            ASTBinaryOperation.GREATER_THAN: CoreBinaryOperation.GREATER,
            ASTBinaryOperation.GREATER_THAN_OR_EQUAL: CoreBinaryOperation.GREATER_EQUAL,
            ASTBinaryOperation.LOGICAL_AND: CoreBinaryOperation.LOGICAL_AND,
            ASTBinaryOperation.LOGICAL_OR: CoreBinaryOperation.LOGICAL_OR,
        }
    )

    def default(self, node: ASTNode) -> NoReturn:
        raise RuntimeError(
            f"Core expressions do not support {node.__class__.__name__} AST nodes."
        )

    def visit_unary_expression(self, node: ASTUnaryExpression) -> CoreUnaryExpression:
        return CoreUnaryExpression(
            self._AST_TO_CORE_UNARY_OPERATIONS[node.operation],
            self.visit(node.expression),
        )

    def visit_binary_expression(
        self, node: ASTBinaryExpression
    ) -> CoreBinaryExpression:
        return CoreBinaryExpression(
            self._AST_TO_CORE_BINARY_OPERATIONS[node.operation],
            self.visit(node.left),
            self.visit(node.right),
        )

    def visit_identifier_expression(
        self, node: ASTIdentifierExpression
    ) -> CoreIdentifierExpression:
        return CoreIdentifierExpression(node.identifier)

    def visit_int_literal(self, node: ASTIntLiteralExpression) -> CoreLiteralExpression:
        return CoreLiteralExpression(node.value)

    def visit_float_literal(
        self, node: ASTFloatLiteralExpression
    ) -> CoreLiteralExpression:
        return CoreLiteralExpression(node.value)

    def visit_complex_literal(
        self, node: ASTComplexLiteralExpression
    ) -> CoreLiteralExpression:
        raise NotImplementedError(
            "Complex literals are not yet supported for core expressions."
        )


def convert_ast_expression_to_core_expression(
    ast_expression: ASTExpression,
) -> CoreExpression:
    """Convert an AST expression to a core expression."""
    return cast(CoreExpression, ASTToCoreExpressionConverter().visit(ast_expression))
