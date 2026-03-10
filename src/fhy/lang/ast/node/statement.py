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

"""Statement nodes for the statements in the FhY language.

Statement ASTNodes:
    DeclarationStatement: Declares a Variable, with or without assignment
    ExpressionStatement:
    ForAllStatement: An Iteration statement evaluating an expression over a body

"""

from dataclasses import dataclass, field

from fhy_core import Identifier, TemplateDataType

from .base import ASTNode
from .core import Expression, Function, Statement
from .qualified_type import QualifiedType


# TODO: Support Import Alias (e.g. import x as y)
#       alias (Optional[str]): reference name (in present namespace)
#       Define how we handle identifier with different name.
@dataclass(frozen=True, kw_only=True)
class Import(Statement):
    """Import statement node.

    Args:
        name (Identifier): Name of imported object.

    Attributes:
        name (Identifier): Name of imported object

    """

    name: Identifier

    def get_visit_attrs(self) -> list[str]:
        attrs = super().get_visit_attrs()
        attrs.extend(["name"])

        return attrs


@dataclass(frozen=True, kw_only=True)
class Argument(ASTNode):
    """Function argument node.

    Args:
        name (Identifier): Variable name of the argument.
        qualified_type (QualifiedType): Type of the argument.

    Attributes:
        name (Identifier): Variable name of the argument.
        qualified_type (QualifiedType): Type of the argument.

    """

    name: Identifier
    qualified_type: QualifiedType

    def get_visit_attrs(self) -> list[str]:
        attrs = super().get_visit_attrs()
        attrs.extend(["name", "qualified_type"])

        return attrs


@dataclass(frozen=True, kw_only=True)
class Procedure(Function):
    """FhY procedure AST node.

    Args:
        templates (List[TemplateDataType]): Template data types.
        args (List[Argument], optional): Arguments of the procedure.
        body (List[Statement], optional): Body of the procedure.

    Attributes:
        templates (List[TemplateDataType]): Template data types.
        args (List[Argument]): Arguments of the procedure.
        body (List[Statement]): Body of the procedure.

    """

    templates: list[TemplateDataType] = field(default_factory=list)
    args: list[Argument] = field(default_factory=list)
    body: list[Statement] = field(default_factory=list)

    def get_visit_attrs(self) -> list[str]:
        attrs: list[str] = super().get_visit_attrs()
        attrs.extend(["templates", "args", "body"])

        return attrs


@dataclass(frozen=True, kw_only=True)
class Operation(Function):
    """FhY operation AST node.

    Args:
        templates (List[TemplateDataType]): Template data types.
        args (List[Argument]): Arguments of the operation.
        body (List[Statement]): Body of the operation.
        return_type (QualifiedType): Return type of the operation.

    Attributes:
        templates (List[TemplateDataType]): Template data types.
        args (List[Argument]): Arguments of the operation.
        body (List[Statement]): Body of the operation.
        return_type (QualifiedType): Return type of the operation.

    """

    templates: list[TemplateDataType] = field(default_factory=list)
    args: list[Argument] = field(default_factory=list)
    body: list[Statement] = field(default_factory=list)
    return_type: QualifiedType

    def get_visit_attrs(self) -> list[str]:
        attrs = super().get_visit_attrs()
        attrs.extend(["templates", "args", "body", "return_type"])

        return attrs


@dataclass(frozen=True, kw_only=True)
class Native(Function):
    """FhY native AST node.

    Args:
        args (List[Argument]): Arguments of the native function.

    Attributes:
        args (List[Argument]): Arguments of the native function.

    """

    args: list[Argument] = field(default_factory=list)

    def get_visit_attrs(self) -> list[str]:
        attrs = super().get_visit_attrs()
        attrs.extend(["args"])

        return attrs


@dataclass(frozen=True, kw_only=True)
class DeclarationStatement(Statement):
    """Declaration statement AST node.

    Args:
        variable_name (Identifier): Name of the declared variable.
        variable_type (QualifiedType): Type of the declared variable.
        expression (Expression, optional): Expression to assign to the variable.

    Attributes:
        variable_name (Identifier): Name of the declared variable.
        variable_type (QualifiedType): Type of the declared variable.
        expression (Expression, optional): Expression to assign to the variable.

    """

    variable_name: Identifier
    variable_type: QualifiedType
    expression: Expression | None = field(default=None)

    def get_visit_attrs(self) -> list[str]:
        attrs = super().get_visit_attrs()
        attrs.extend(["variable_name", "variable_type", "expression"])

        return attrs


@dataclass(frozen=True, kw_only=True)
class ExpressionStatement(Statement):
    """Expression statement AST node.

    Args:
        left (Expression, optional): Expression assigned to.
        right (Expression): Expression to be evaluated.

    Attributes:
        left (Expression, optional): Expression assigned to.
        right (Expression): Expression to be evaluated.

    """

    left: Expression | None = field(default=None)
    right: Expression

    def get_visit_attrs(self) -> list[str]:
        attrs = super().get_visit_attrs()
        attrs.extend(["left", "right"])

        return attrs


@dataclass(frozen=True, kw_only=True)
class ForAllStatement(Statement):
    """ForAll statement AST node.

    Args:
        index (Expression): Loop index to iterate through.
        body (List[Statement]): Body of the ForAll statement.

    Attributes:
        index (Expression): Loop index to iterate through.
        body (List[Statement]): Body of the ForAll statement.

    """

    index: Expression
    body: list[Statement] = field(default_factory=list)

    def get_visit_attrs(self) -> list[str]:
        attrs = super().get_visit_attrs()
        attrs.extend(["index", "body"])

        return attrs


@dataclass(frozen=True, kw_only=True)
class SelectionStatement(Statement):
    """Selection statement AST node.

    Args:
        condition (Expression): Condition to evaluate.
        true_body (List[Statement]): Statements to evaluate if true.
        false_body (List[Statement]): Statements to evaluate if false.

    Attributes:
        condition (Expression): Condition to evaluate.
        true_body (List[Statement]): Statements to evaluate if true.
        false_body (List[Statement]): Statements to evaluate if false.

    """

    condition: Expression
    true_body: list[Statement] = field(default_factory=list)
    false_body: list[Statement] = field(default_factory=list)

    def get_visit_attrs(self) -> list[str]:
        attrs = super().get_visit_attrs()
        attrs.extend(["condition", "true_body", "false_body"])

        return attrs


@dataclass(frozen=True, kw_only=True)
class ReturnStatement(Statement):
    """Return statement AST node.

    Args:
        expression (Expression): Expression to return.

    Attributes:
        expression (Expression): Expression to be evaluated and returned.

    """

    expression: Expression

    def get_visit_attrs(self) -> list[str]:
        attrs = super().get_visit_attrs()
        attrs.extend(["expression"])

        return attrs
