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

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TypeGuard

from fhy_core import (
    DeserializationDictStructureError,
    Identifier,
    SerializedDict,
    TemplateDataType,
    Visitable,
    is_serialized_dict,
    register_serializable,
)

from .base import Node, NodeData, deserialize_node_provenance, is_valid_node_data
from .core import (
    Expression,
    Function,
    FunctionData,
    Statement,
    StatementData,
    is_valid_function_data,
    is_valid_statement_data,
)
from .qualified_type import QualifiedType


# TODO: Support Import Alias (e.g. import x as y)
#       alias (Optional[str]): reference name (in present namespace)
#       Define how we handle identifier with different name.
class _ImportData(StatementData):
    name: SerializedDict


def _is_valid_import_data(data: SerializedDict) -> TypeGuard[_ImportData]:
    return (
        "name" in data
        and is_serialized_dict(data["name"])
        and is_valid_statement_data(data)
    )


@register_serializable(type_id="fhy_ast_import")
@dataclass(frozen=True, kw_only=True)
class Import(Statement):
    """Import statement node.

    Args:
        name (Identifier): Name of imported object.

    Attributes:
        name (Identifier): Name of imported object

    """

    name: Identifier

    def get_visit_children(self) -> Sequence[Visitable]:
        return ()

    def is_structurally_equivalent(self, other: object) -> bool:
        return (
            isinstance(other, Import)
            and super().is_structurally_equivalent(other)
            and self.name == other.name
        )

    def serialize_data_to_dict(self) -> SerializedDict:
        data = super().serialize_data_to_dict()
        data["name"] = self.name.serialize_to_dict()
        return data

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "Import":
        if not _is_valid_import_data(data):
            raise DeserializationDictStructureError(
                cls, _ImportData.__annotations__, data
            )
        return cls(
            name=Identifier.deserialize_from_dict(data["name"]),
            provenance=deserialize_node_provenance(data),
        )


class _ArgumentData(NodeData):
    name: SerializedDict
    qualified_type: SerializedDict


def _is_valid_argument_data(data: SerializedDict) -> TypeGuard[_ArgumentData]:
    return (
        "name" in data
        and is_serialized_dict(data["name"])
        and "qualified_type" in data
        and is_serialized_dict(data["qualified_type"])
        and is_valid_node_data(data)
    )


@register_serializable(type_id="fhy_ast_argument")
@dataclass(frozen=True, kw_only=True)
class Argument(Node):
    """Function argument node."""

    name: Identifier
    qualified_type: QualifiedType

    def get_visit_children(self) -> Sequence[Visitable]:
        return (self.qualified_type,)

    def is_structurally_equivalent(self, other: object) -> bool:
        return (
            isinstance(other, Argument)
            and super().is_structurally_equivalent(other)
            and self.name == other.name
            and self.qualified_type.is_structurally_equivalent(other.qualified_type)
        )

    def serialize_data_to_dict(self) -> SerializedDict:
        data = super().serialize_data_to_dict()
        data["name"] = self.name.serialize_to_dict()
        data["qualified_type"] = self.qualified_type.serialize_to_dict()
        return data

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "Argument":
        if not _is_valid_argument_data(data):
            raise DeserializationDictStructureError(
                cls, _ArgumentData.__annotations__, data
            )
        return cls(
            name=Identifier.deserialize_from_dict(data["name"]),
            qualified_type=QualifiedType.deserialize_from_dict(data["qualified_type"]),
            provenance=deserialize_node_provenance(data),
        )


class _ProcedureData(FunctionData):
    templates: list[SerializedDict]
    args: list[SerializedDict]
    body: list[SerializedDict]


def _is_valid_procedure_data(data: SerializedDict) -> TypeGuard[_ProcedureData]:
    return (
        "templates" in data
        and isinstance(data["templates"], list)
        and all(is_serialized_dict(template) for template in data["templates"])
        and "args" in data
        and isinstance(data["args"], list)
        and all(is_serialized_dict(argument) for argument in data["args"])
        and "body" in data
        and isinstance(data["body"], list)
        and all(is_serialized_dict(statement) for statement in data["body"])
        and is_valid_function_data(data)
    )


@register_serializable(type_id="fhy_ast_procedure")
@dataclass(frozen=True, kw_only=True)
class Procedure(Function):
    """FhY procedure AST node."""

    templates: tuple[TemplateDataType, ...] = field(default_factory=tuple)
    args: tuple[Argument, ...] = field(default_factory=tuple)
    body: tuple[Statement, ...] = field(default_factory=tuple)

    def get_visit_children(self) -> Sequence[Visitable]:
        return (*self.args, *self.body)

    def is_structurally_equivalent(self, other: object) -> bool:
        return (
            isinstance(other, Procedure)
            and super().is_structurally_equivalent(other)
            and len(self.templates) == len(other.templates)
            and all(
                template.is_structurally_equivalent(other_template)
                for template, other_template in zip(self.templates, other.templates)
            )
            and len(self.args) == len(other.args)
            and all(
                arg.is_structurally_equivalent(other_arg)
                for arg, other_arg in zip(self.args, other.args)
            )
            and len(self.body) == len(other.body)
            and all(
                statement.is_structurally_equivalent(other_statement)
                for statement, other_statement in zip(self.body, other.body)
            )
        )

    def serialize_data_to_dict(self) -> SerializedDict:
        data = super().serialize_data_to_dict()
        data["templates"] = [
            template.serialize_to_dict() for template in self.templates
        ]
        data["args"] = [arg.serialize_to_dict() for arg in self.args]
        data["body"] = [statement.serialize_to_dict() for statement in self.body]
        return data

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "Procedure":
        if not _is_valid_procedure_data(data):
            raise DeserializationDictStructureError(
                cls, _ProcedureData.__annotations__, data
            )
        return cls(
            name=Identifier.deserialize_from_dict(data["name"]),
            templates=[
                TemplateDataType.deserialize_from_dict(template)
                for template in data["templates"]
            ],
            args=[
                Argument.deserialize_from_dict(argument) for argument in data["args"]
            ],
            body=[
                Statement.deserialize_from_dict(statement) for statement in data["body"]
            ],
            provenance=deserialize_node_provenance(data),
        )


class _OperationData(FunctionData):
    templates: list[SerializedDict]
    args: list[SerializedDict]
    body: list[SerializedDict]
    return_type: SerializedDict


def _is_valid_operation_data(data: SerializedDict) -> TypeGuard[_OperationData]:
    return (
        "templates" in data
        and isinstance(data["templates"], list)
        and all(is_serialized_dict(template) for template in data["templates"])
        and "args" in data
        and isinstance(data["args"], list)
        and all(is_serialized_dict(argument) for argument in data["args"])
        and "body" in data
        and isinstance(data["body"], list)
        and all(is_serialized_dict(statement) for statement in data["body"])
        and "return_type" in data
        and is_serialized_dict(data["return_type"])
        and is_valid_function_data(data)
    )


@register_serializable(type_id="fhy_ast_operation")
@dataclass(frozen=True, kw_only=True)
class Operation(Function):
    """FhY operation AST node."""

    templates: tuple[TemplateDataType, ...] = field(default_factory=tuple)
    args: tuple[Argument, ...] = field(default_factory=tuple)
    body: tuple[Statement, ...] = field(default_factory=tuple)
    return_type: QualifiedType

    def get_visit_children(self) -> Sequence[Visitable]:
        return (*self.args, *self.body, self.return_type)

    def is_structurally_equivalent(self, other: object) -> bool:
        return (
            isinstance(other, Operation)
            and super().is_structurally_equivalent(other)
            and len(self.templates) == len(other.templates)
            and all(
                template.is_structurally_equivalent(other_template)
                for template, other_template in zip(self.templates, other.templates)
            )
            and len(self.args) == len(other.args)
            and all(
                arg.is_structurally_equivalent(other_arg)
                for arg, other_arg in zip(self.args, other.args)
            )
            and len(self.body) == len(other.body)
            and all(
                statement.is_structurally_equivalent(other_statement)
                for statement, other_statement in zip(self.body, other.body)
            )
            and self.return_type.is_structurally_equivalent(other.return_type)
        )

    def serialize_data_to_dict(self) -> SerializedDict:
        data = super().serialize_data_to_dict()
        data["templates"] = [
            template.serialize_to_dict() for template in self.templates
        ]
        data["args"] = [arg.serialize_to_dict() for arg in self.args]
        data["body"] = [statement.serialize_to_dict() for statement in self.body]
        data["return_type"] = self.return_type.serialize_to_dict()
        return data

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "Operation":
        if not _is_valid_operation_data(data):
            raise DeserializationDictStructureError(
                cls, _OperationData.__annotations__, data
            )
        return cls(
            name=Identifier.deserialize_from_dict(data["name"]),
            templates=[
                TemplateDataType.deserialize_from_dict(template)
                for template in data["templates"]
            ],
            args=[
                Argument.deserialize_from_dict(argument) for argument in data["args"]
            ],
            body=[
                Statement.deserialize_from_dict(statement) for statement in data["body"]
            ],
            return_type=QualifiedType.deserialize_from_dict(data["return_type"]),
            provenance=deserialize_node_provenance(data),
        )


class _NativeData(FunctionData):
    args: list[SerializedDict]


def _is_valid_native_data(data: SerializedDict) -> TypeGuard[_NativeData]:
    return (
        "args" in data
        and isinstance(data["args"], list)
        and all(is_serialized_dict(argument) for argument in data["args"])
        and is_valid_function_data(data)
    )


@register_serializable(type_id="fhy_ast_native")
@dataclass(frozen=True, kw_only=True)
class Native(Function):
    """FhY native AST node."""

    args: tuple[Argument, ...] = field(default_factory=tuple)

    def get_visit_children(self) -> Sequence[Visitable]:
        return self.args

    def is_structurally_equivalent(self, other: object) -> bool:
        return (
            isinstance(other, Native)
            and super().is_structurally_equivalent(other)
            and len(self.args) == len(other.args)
            and all(
                arg.is_structurally_equivalent(other_arg)
                for arg, other_arg in zip(self.args, other.args)
            )
        )

    def serialize_data_to_dict(self) -> SerializedDict:
        data = super().serialize_data_to_dict()
        data["args"] = [arg.serialize_to_dict() for arg in self.args]
        return data

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "Native":
        if not _is_valid_native_data(data):
            raise DeserializationDictStructureError(
                cls, _NativeData.__annotations__, data
            )
        return cls(
            name=Identifier.deserialize_from_dict(data["name"]),
            args=[
                Argument.deserialize_from_dict(argument) for argument in data["args"]
            ],
            provenance=deserialize_node_provenance(data),
        )


class _DeclarationStatementData(StatementData):
    variable_name: SerializedDict
    variable_type: SerializedDict
    expression: SerializedDict | None


def _is_valid_declaration_statement_data(
    data: SerializedDict,
) -> TypeGuard[_DeclarationStatementData]:
    return (
        "variable_name" in data
        and is_serialized_dict(data["variable_name"])
        and "variable_type" in data
        and is_serialized_dict(data["variable_type"])
        and "expression" in data
        and (is_serialized_dict(data["expression"]) or data["expression"] is None)
        and is_valid_statement_data(data)
    )


@register_serializable(type_id="fhy_ast_declaration_statement")
@dataclass(frozen=True, kw_only=True)
class DeclarationStatement(Statement):
    """FhY declaration statement AST node."""

    variable_name: Identifier
    variable_type: QualifiedType
    expression: Expression | None = field(default=None)

    def get_visit_children(self) -> Sequence[Visitable]:
        if self.expression is None:
            return (self.variable_type,)
        else:
            return (self.variable_type, self.expression)

    def is_structurally_equivalent(self, other: object) -> bool:
        return (
            isinstance(other, DeclarationStatement)
            and super().is_structurally_equivalent(other)
            and self.variable_name == other.variable_name
            and self.variable_type.is_structurally_equivalent(other.variable_type)
            and (
                (self.expression is None and other.expression is None)
                or (
                    self.expression is not None
                    and other.expression is not None
                    and self.expression.is_structurally_equivalent(other.expression)
                )
            )
        )

    def serialize_data_to_dict(self) -> SerializedDict:
        data = super().serialize_data_to_dict()
        data["variable_name"] = self.variable_name.serialize_to_dict()
        data["variable_type"] = self.variable_type.serialize_to_dict()
        data["expression"] = (
            self.expression.serialize_to_dict() if self.expression is not None else None
        )
        return data

    @classmethod
    def deserialize_data_from_dict(
        cls,
        data: SerializedDict,
    ) -> "DeclarationStatement":
        if not _is_valid_declaration_statement_data(data):
            raise DeserializationDictStructureError(
                cls, _DeclarationStatementData.__annotations__, data
            )
        expression = data["expression"]
        return cls(
            variable_name=Identifier.deserialize_from_dict(data["variable_name"]),
            variable_type=QualifiedType.deserialize_from_dict(data["variable_type"]),
            expression=(
                Expression.deserialize_from_dict(expression)
                if expression is not None
                else None
            ),
            provenance=deserialize_node_provenance(data),
        )


class _ExpressionStatementData(StatementData):
    left: SerializedDict | None
    right: SerializedDict


def _is_valid_expression_statement_data(
    data: SerializedDict,
) -> TypeGuard[_ExpressionStatementData]:
    return (
        "left" in data
        and (is_serialized_dict(data["left"]) or data["left"] is None)
        and "right" in data
        and is_serialized_dict(data["right"])
        and is_valid_statement_data(data)
    )


@register_serializable(type_id="fhy_ast_expression_statement")
@dataclass(frozen=True, kw_only=True)
class ExpressionStatement(Statement):
    """FhY expression statement AST node."""

    left: Expression | None = field(default=None)
    right: Expression

    def get_visit_children(self) -> Sequence[Visitable]:
        if self.left is None:
            return (self.right,)
        else:
            return (self.left, self.right)

    def is_structurally_equivalent(self, other: object) -> bool:
        return (
            isinstance(other, ExpressionStatement)
            and super().is_structurally_equivalent(other)
            and (
                (self.left is None and other.left is None)
                or (
                    self.left is not None
                    and other.left is not None
                    and self.left.is_structurally_equivalent(other.left)
                )
            )
            and self.right.is_structurally_equivalent(other.right)
        )

    def serialize_data_to_dict(self) -> SerializedDict:
        data = super().serialize_data_to_dict()
        data["left"] = self.left.serialize_to_dict() if self.left is not None else None
        data["right"] = self.right.serialize_to_dict()
        return data

    @classmethod
    def deserialize_data_from_dict(
        cls,
        data: SerializedDict,
    ) -> "ExpressionStatement":
        if not _is_valid_expression_statement_data(data):
            raise DeserializationDictStructureError(
                cls, _ExpressionStatementData.__annotations__, data
            )
        left = data["left"]
        return cls(
            left=Expression.deserialize_from_dict(left) if left is not None else None,
            right=Expression.deserialize_from_dict(data["right"]),
            provenance=deserialize_node_provenance(data),
        )


class _ForAllStatementData(StatementData):
    index: SerializedDict
    body: list[SerializedDict]


def _is_valid_forall_statement_data(
    data: SerializedDict,
) -> TypeGuard[_ForAllStatementData]:
    return (
        "index" in data
        and is_serialized_dict(data["index"])
        and "body" in data
        and isinstance(data["body"], list)
        and all(is_serialized_dict(statement) for statement in data["body"])
        and is_valid_statement_data(data)
    )


@register_serializable(type_id="fhy_ast_forall_statement")
@dataclass(frozen=True, kw_only=True)
class ForAllStatement(Statement):
    """FhY for-all statement AST node."""

    index: Expression
    body: tuple[Statement, ...] = field(default_factory=tuple)

    def get_visit_children(self) -> Sequence[Visitable]:
        return (self.index, *self.body)

    def is_structurally_equivalent(self, other: object) -> bool:
        return (
            isinstance(other, ForAllStatement)
            and super().is_structurally_equivalent(other)
            and self.index.is_structurally_equivalent(other.index)
            and len(self.body) == len(other.body)
            and all(
                statement.is_structurally_equivalent(other_statement)
                for statement, other_statement in zip(self.body, other.body)
            )
        )

    def serialize_data_to_dict(self) -> SerializedDict:
        data = super().serialize_data_to_dict()
        data["index"] = self.index.serialize_to_dict()
        data["body"] = [statement.serialize_to_dict() for statement in self.body]
        return data

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "ForAllStatement":
        if not _is_valid_forall_statement_data(data):
            raise DeserializationDictStructureError(
                cls, _ForAllStatementData.__annotations__, data
            )
        return cls(
            index=Expression.deserialize_from_dict(data["index"]),
            body=[
                Statement.deserialize_from_dict(statement) for statement in data["body"]
            ],
            provenance=deserialize_node_provenance(data),
        )


class _SelectionStatementData(StatementData):
    condition: SerializedDict
    true_body: list[SerializedDict]
    false_body: list[SerializedDict]


def _is_valid_selection_statement_data(
    data: SerializedDict,
) -> TypeGuard[_SelectionStatementData]:
    return (
        "condition" in data
        and is_serialized_dict(data["condition"])
        and "true_body" in data
        and isinstance(data["true_body"], list)
        and all(is_serialized_dict(statement) for statement in data["true_body"])
        and "false_body" in data
        and isinstance(data["false_body"], list)
        and all(is_serialized_dict(statement) for statement in data["false_body"])
        and is_valid_statement_data(data)
    )


@register_serializable(type_id="fhy_ast_selection_statement")
@dataclass(frozen=True, kw_only=True)
class SelectionStatement(Statement):
    """FhY selection statement AST node."""

    condition: Expression
    true_body: tuple[Statement, ...] = field(default_factory=tuple)
    false_body: tuple[Statement, ...] = field(default_factory=tuple)

    def get_visit_children(self) -> Sequence[Visitable]:
        return (
            self.condition,
            *self.true_body,
            *self.false_body,
        )

    def is_structurally_equivalent(self, other: object) -> bool:
        return (
            isinstance(other, SelectionStatement)
            and super().is_structurally_equivalent(other)
            and self.condition.is_structurally_equivalent(other.condition)
            and len(self.true_body) == len(other.true_body)
            and all(
                statement.is_structurally_equivalent(other_statement)
                for statement, other_statement in zip(self.true_body, other.true_body)
            )
            and len(self.false_body) == len(other.false_body)
            and all(
                statement.is_structurally_equivalent(other_statement)
                for statement, other_statement in zip(self.false_body, other.false_body)
            )
        )

    def serialize_data_to_dict(self) -> SerializedDict:
        data = super().serialize_data_to_dict()
        data["condition"] = self.condition.serialize_to_dict()
        data["true_body"] = [
            statement.serialize_to_dict() for statement in self.true_body
        ]
        data["false_body"] = [
            statement.serialize_to_dict() for statement in self.false_body
        ]
        return data

    @classmethod
    def deserialize_data_from_dict(
        cls,
        data: SerializedDict,
    ) -> "SelectionStatement":
        if not _is_valid_selection_statement_data(data):
            raise DeserializationDictStructureError(
                cls, _SelectionStatementData.__annotations__, data
            )
        return cls(
            condition=Expression.deserialize_from_dict(data["condition"]),
            true_body=[
                Statement.deserialize_from_dict(statement)
                for statement in data["true_body"]
            ],
            false_body=[
                Statement.deserialize_from_dict(statement)
                for statement in data["false_body"]
            ],
            provenance=deserialize_node_provenance(data),
        )


class _ReturnStatementData(StatementData):
    expression: SerializedDict


def _is_valid_return_statement_data(
    data: SerializedDict,
) -> TypeGuard[_ReturnStatementData]:
    return (
        "expression" in data
        and is_serialized_dict(data["expression"])
        and is_valid_statement_data(data)
    )


@register_serializable(type_id="fhy_ast_return_statement")
@dataclass(frozen=True, kw_only=True)
class ReturnStatement(Statement):
    """FhY return statement AST node."""

    expression: Expression

    def get_visit_children(self) -> Sequence[Visitable]:
        return (self.expression,)

    def is_structurally_equivalent(self, other: object) -> bool:
        return (
            isinstance(other, ReturnStatement)
            and super().is_structurally_equivalent(other)
            and self.expression.is_structurally_equivalent(other.expression)
        )

    def serialize_data_to_dict(self) -> SerializedDict:
        data = super().serialize_data_to_dict()
        data["expression"] = self.expression.serialize_to_dict()
        return data

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "ReturnStatement":
        if not _is_valid_return_statement_data(data):
            raise DeserializationDictStructureError(
                cls, _ReturnStatementData.__annotations__, data
            )
        return cls(
            expression=Expression.deserialize_from_dict(data["expression"]),
            provenance=deserialize_node_provenance(data),
        )
