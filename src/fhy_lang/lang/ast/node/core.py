"""Core AST nodes for FhY language constructs."""

__all__ = [
    "Expression",
    "ExpressionData",
    "Function",
    "FunctionData",
    "Module",
    "Statement",
    "StatementData",
    "is_valid_expression_data",
    "is_valid_function_data",
    "is_valid_statement_data",
]

from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TypeGuard

from fhy_core import (
    DeserializationDictStructureError,
    HasIdentifierMixin,
    Identifier,
    SerializedDict,
    Visitable,
    is_serialized_dict,
    register_serializable,
)

from .base import (
    Node,
    NodeData,
    deserialize_node_provenance,
    is_valid_node_data,
)


class _ModuleData(NodeData):
    """Data for a module node."""

    name: SerializedDict
    statements: list[SerializedDict]


def _is_valid_module_data(data: SerializedDict) -> TypeGuard[_ModuleData]:
    return (
        "name" in data
        and is_serialized_dict(data["name"])
        and "statements" in data
        and isinstance(data["statements"], list)
        and all(is_serialized_dict(statement) for statement in data["statements"])
        and is_valid_node_data(data)
    )


@register_serializable(type_id="fhy_ast_module")
@dataclass(frozen=True, kw_only=True)
class Module(Node, HasIdentifierMixin):
    """FhY module AST node."""

    name: Identifier = field(default=Identifier("module"))
    statements: tuple["Statement", ...] = field(default_factory=tuple)

    def get_identifier(self) -> Identifier:
        return self.name

    def get_visit_children(self) -> Sequence[Visitable]:
        return self.statements

    def is_structurally_equivalent(self, other: object) -> bool:
        return (
            isinstance(other, Module)
            and super().is_structurally_equivalent(other)
            and self.name == other.name
            and len(self.statements) == len(other.statements)
            and all(
                statement.is_structurally_equivalent(other_statement)
                for statement, other_statement in zip(self.statements, other.statements)
            )
        )

    def serialize_data_to_dict(self) -> SerializedDict:
        data = super().serialize_data_to_dict()
        data["name"] = self.name.serialize_to_dict()
        data["statements"] = [
            statement.serialize_to_dict() for statement in self.statements
        ]
        return data

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "Module":
        if not _is_valid_module_data(data):
            raise DeserializationDictStructureError(
                cls, _ModuleData.__annotations__, data
            )
        name = Identifier.deserialize_from_dict(data["name"])
        statements = tuple(
            Statement.deserialize_from_dict(statement)
            for statement in data["statements"]
        )
        return cls(
            name=name,
            statements=statements,
            provenance=deserialize_node_provenance(data),
        )


class StatementData(NodeData):
    """Data for a statement node."""


def is_valid_statement_data(data: SerializedDict) -> TypeGuard[StatementData]:
    """Return True if the data is a valid statement data."""
    return is_valid_node_data(data)


class Statement(Node, ABC):
    """Abstract statement AST node."""


class FunctionData(NodeData):
    """Data for a function node."""

    name: SerializedDict


def is_valid_function_data(data: SerializedDict) -> TypeGuard[FunctionData]:
    """Return True if the data is a valid function data."""
    return (
        "name" in data and is_serialized_dict(data["name"]) and is_valid_node_data(data)
    )


@dataclass(frozen=True, kw_only=True)
class Function(Statement, HasIdentifierMixin, ABC):
    """Abstract FhY function node.

    Used as a base for the function nodes such as procedures and operations.

    """

    name: Identifier

    def get_identifier(self) -> Identifier:
        return self.name

    def is_structurally_equivalent(self, other: object) -> bool:
        return (
            isinstance(other, Function)
            and super().is_structurally_equivalent(other)
            and self.name == other.name
        )

    def serialize_data_to_dict(self) -> SerializedDict:
        data = super().serialize_data_to_dict()
        data["name"] = self.name.serialize_to_dict()
        return data


class ExpressionData(NodeData):
    """Data for an expression node."""


def is_valid_expression_data(data: SerializedDict) -> TypeGuard[ExpressionData]:
    """Return True if the data is a valid expression data."""
    return is_valid_node_data(data)


class Expression(Node, ABC):
    """Abstract expression AST node."""
