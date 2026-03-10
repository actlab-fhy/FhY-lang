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

"""FhY AST Node Definitions, broken down into several grammar construct categories.

1. Base and Core Nodes (Abstract)
2. Expressions
3. Components
4. Statements
5. Qualified Type Node

We also have visitor and listener patterns for FhY ASTNodes.

"""

from .node import (
    Argument,
    ArrayAccessExpression,
    ASTNode,
    BinaryExpression,
    BinaryOperation,
    ComplexLiteral,
    DeclarationStatement,
    Expression,
    ExpressionStatement,
    FloatLiteral,
    ForAllStatement,
    Function,
    FunctionExpression,
    IdentifierExpression,
    Import,
    IntLiteral,
    Literal,
    Module,
    Native,
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
    UnaryOperation,
)
from .passes import (
    collect_identifiers,
    collect_imported_identifiers,
    replace_identifiers,
)
from .span import Source, Span
from .visitor import BasePass, Transformer, Visitor

__all__ = [
    "ASTNode",
    "Module",
    "Function",
    "Statement",
    "Expression",
    "UnaryOperation",
    "BinaryOperation",
    "UnaryExpression",
    "BinaryExpression",
    "TernaryExpression",
    "TupleExpression",
    "TupleAccessExpression",
    "FunctionExpression",
    "ArrayAccessExpression",
    "IdentifierExpression",
    "Literal",
    "IntLiteral",
    "FloatLiteral",
    "ComplexLiteral",
    "QualifiedType",
    "Import",
    "Argument",
    "Procedure",
    "Operation",
    "DeclarationStatement",
    "ExpressionStatement",
    "SelectionStatement",
    "ForAllStatement",
    "ReturnStatement",
    "Native",
    "collect_identifiers",
    "collect_imported_identifiers",
    "replace_identifiers",
    "Source",
    "Span",
    "Visitor",
    "BasePass",
    "Transformer",
]
