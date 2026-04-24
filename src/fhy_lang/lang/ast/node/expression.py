"""Expression nodes for the expressions in the FhY language."""

from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TypeGuard

from fhy_core import (
    DataType,
    DeserializationDictStructureError,
    DeserializationValueError,
    HasOperandsMixin,
    Identifier,
    SerializedDict,
    StrEnum,
    Visitable,
    is_serialized_dict,
    register_serializable,
)

from .base import deserialize_node_provenance, is_valid_node_data
from .core import Expression, ExpressionData, is_valid_expression_data


class UnaryOperation(StrEnum):
    """FhY language unary operators.

    Arithmetic:
        Negation

    Logical:
        Logical Not

    Bitwise:
        Bitwise Not

    """

    NEGATION = "-"
    BITWISE_NOT = "~"
    LOGICAL_NOT = "!"


class _UnaryExpressionData(ExpressionData):
    operation: str
    expression: SerializedDict


def _is_valid_unary_expression_data(
    data: SerializedDict,
) -> TypeGuard[_UnaryExpressionData]:
    return (
        "operation" in data
        and isinstance(data["operation"], str)
        and "expression" in data
        and is_serialized_dict(data["expression"])
        and is_valid_expression_data(data)
    )


@register_serializable(type_id="fhy_ast_unary_expression")
@dataclass(frozen=True, kw_only=True)
class UnaryExpression(Expression, HasOperandsMixin[Expression]):
    """FhY unary expression AST node."""

    operation: UnaryOperation
    expression: Expression

    def get_operands(self) -> tuple[Expression, ...]:
        return (self.expression,)

    def get_visit_children(self) -> Sequence[Visitable]:
        return (self.expression,)

    def is_structurally_equivalent(self, other: object) -> bool:
        return (
            isinstance(other, UnaryExpression)
            and super().is_structurally_equivalent(other)
            and self.operation == other.operation
            and self.expression.is_structurally_equivalent(other.expression)
        )

    def serialize_data_to_dict(self) -> SerializedDict:
        data = super().serialize_data_to_dict()
        data["operation"] = self.operation.value
        data["expression"] = self.expression.serialize_to_dict()
        return data

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "UnaryExpression":
        if not _is_valid_unary_expression_data(data):
            raise DeserializationDictStructureError(
                cls, _UnaryExpressionData.__annotations__, data
            )
        operation = data["operation"]
        if operation not in UnaryOperation._value2member_map_:
            raise DeserializationValueError(
                cls, "operation", "a valid unary operation", operation
            )
        return cls(
            operation=UnaryOperation(operation),
            expression=Expression.deserialize_from_dict(data["expression"]),
            provenance=deserialize_node_provenance(data),
        )


class BinaryOperation(StrEnum):
    """FhY language binary operators.

    Arithmetic:
        Addition
        Subtraction
        Multiplication
        Division
        Floor Division
        Modulo
        Power

    Logical:
        And
        Or

    Relational:
        Equality
        Inequality
        Less Than
        Less Than or Equal To
        Greater Than
        Greater Than or Equal To

    Bitwise:
        And
        Or
        Xor
        Left Shift
        Right Shift

    """

    MULTIPLICATION = "*"
    DIVISION = "/"
    ADDITION = "+"
    SUBTRACTION = "-"
    LEFT_SHIFT = "<<"
    RIGHT_SHIFT = ">>"
    LESS_THAN = "<"
    LESS_THAN_OR_EQUAL = "<="
    GREATER_THAN = ">"
    GREATER_THAN_OR_EQUAL = ">="
    EQUAL_TO = "=="
    NOT_EQUAL_TO = "!="
    BITWISE_AND = "&"
    BITWISE_XOR = "^"
    BITWISE_OR = "|"
    LOGICAL_AND = "&&"
    LOGICAL_OR = "||"
    MODULO = "%"
    POWER = "**"
    FLOORDIV = "//"


class _BinaryExpressionData(ExpressionData):
    operation: str
    left: SerializedDict
    right: SerializedDict


def _is_valid_binary_expression_data(
    data: SerializedDict,
) -> TypeGuard[_BinaryExpressionData]:
    return (
        "operation" in data
        and isinstance(data["operation"], str)
        and "left" in data
        and is_serialized_dict(data["left"])
        and "right" in data
        and is_serialized_dict(data["right"])
        and is_valid_node_data(data)
    )


@register_serializable(type_id="fhy_ast_binary_expression")
@dataclass(frozen=True, kw_only=True)
class BinaryExpression(Expression, HasOperandsMixin[Expression]):
    """FhY binary expression AST node."""

    operation: BinaryOperation
    left: Expression
    right: Expression

    def get_operands(self) -> tuple[Expression, Expression]:
        return (self.left, self.right)

    def get_visit_children(self) -> Sequence[Visitable]:
        return (self.left, self.right)

    def is_structurally_equivalent(self, other: object) -> bool:
        return (
            isinstance(other, BinaryExpression)
            and super().is_structurally_equivalent(other)
            and self.operation == other.operation
            and self.left.is_structurally_equivalent(other.left)
            and self.right.is_structurally_equivalent(other.right)
        )

    def serialize_data_to_dict(self) -> SerializedDict:
        data = super().serialize_data_to_dict()
        data["operation"] = self.operation.value
        data["left"] = self.left.serialize_to_dict()
        data["right"] = self.right.serialize_to_dict()
        return data

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "BinaryExpression":
        if not _is_valid_binary_expression_data(data):
            raise DeserializationDictStructureError(
                cls, _BinaryExpressionData.__annotations__, data
            )
        operation = data["operation"]
        if operation not in BinaryOperation._value2member_map_:
            raise DeserializationValueError(
                cls, "operation", "a valid binary operation", operation
            )
        return cls(
            operation=BinaryOperation(operation),
            left=Expression.deserialize_from_dict(data["left"]),
            right=Expression.deserialize_from_dict(data["right"]),
            provenance=deserialize_node_provenance(data),
        )


class _TernaryExpressionData(ExpressionData):
    condition: SerializedDict
    true: SerializedDict
    false: SerializedDict


def _is_valid_ternary_expression_data(
    data: SerializedDict,
) -> TypeGuard[_TernaryExpressionData]:
    return (
        "condition" in data
        and is_serialized_dict(data["condition"])
        and "true" in data
        and is_serialized_dict(data["true"])
        and "false" in data
        and is_serialized_dict(data["false"])
        and is_valid_expression_data(data)
    )


@register_serializable(type_id="fhy_ast_ternary_expression")
@dataclass(frozen=True, kw_only=True)
class TernaryExpression(Expression):
    """FhY ternary expression AST node."""

    condition: Expression
    true: Expression
    false: Expression

    def get_visit_children(self) -> Sequence[Visitable]:
        return (self.condition, self.true, self.false)

    def is_structurally_equivalent(self, other: object) -> bool:
        return (
            isinstance(other, TernaryExpression)
            and super().is_structurally_equivalent(other)
            and self.condition.is_structurally_equivalent(other.condition)
            and self.true.is_structurally_equivalent(other.true)
            and self.false.is_structurally_equivalent(other.false)
        )

    def serialize_data_to_dict(self) -> SerializedDict:
        data = super().serialize_data_to_dict()
        data["condition"] = self.condition.serialize_to_dict()
        data["true"] = self.true.serialize_to_dict()
        data["false"] = self.false.serialize_to_dict()
        return data

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "TernaryExpression":
        if not _is_valid_ternary_expression_data(data):
            raise DeserializationDictStructureError(
                cls, _TernaryExpressionData.__annotations__, data
            )
        return cls(
            condition=Expression.deserialize_from_dict(data["condition"]),
            true=Expression.deserialize_from_dict(data["true"]),
            false=Expression.deserialize_from_dict(data["false"]),
            provenance=deserialize_node_provenance(data),
        )


class _TupleAccessExpressionData(ExpressionData):
    """Data for a tuple-access expression node."""

    tuple_expression: SerializedDict
    element_index: SerializedDict


def _is_valid_tuple_access_expression_data(
    data: SerializedDict,
) -> TypeGuard[_TupleAccessExpressionData]:
    return (
        "tuple_expression" in data
        and is_serialized_dict(data["tuple_expression"])
        and "element_index" in data
        and is_serialized_dict(data["element_index"])
        and is_valid_expression_data(data)
    )


@register_serializable(type_id="fhy_ast_tuple_access_expression")
@dataclass(frozen=True, kw_only=True)
class TupleAccessExpression(Expression, HasOperandsMixin[Expression]):
    """FhY tuple access expression AST node."""

    tuple_expression: Expression
    element_index: "IntLiteral"

    def get_operands(self) -> tuple[Expression, Expression]:
        return (
            self.tuple_expression,
            self.element_index,
        )

    def get_visit_children(self) -> Sequence[Visitable]:
        return (self.tuple_expression, self.element_index)

    def is_structurally_equivalent(self, other: object) -> bool:
        return (
            isinstance(other, TupleAccessExpression)
            and super().is_structurally_equivalent(other)
            and self.tuple_expression.is_structurally_equivalent(other.tuple_expression)
            and self.element_index.is_structurally_equivalent(other.element_index)
        )

    def serialize_data_to_dict(self) -> SerializedDict:
        data = super().serialize_data_to_dict()
        data["tuple_expression"] = self.tuple_expression.serialize_to_dict()
        data["element_index"] = (
            self.element_index.serialize_to_dict()
            if self.element_index is not None
            else None
        )
        return data

    @classmethod
    def deserialize_data_from_dict(
        cls,
        data: SerializedDict,
    ) -> "TupleAccessExpression":
        if not _is_valid_tuple_access_expression_data(data):
            raise DeserializationDictStructureError(
                cls, _TupleAccessExpressionData.__annotations__, data
            )
        return cls(
            tuple_expression=Expression.deserialize_from_dict(data["tuple_expression"]),
            element_index=IntLiteral.deserialize_from_dict(data["element_index"]),
            provenance=deserialize_node_provenance(data),
        )


class _FunctionExpressionData(ExpressionData):
    function: SerializedDict
    template_types: list[SerializedDict]
    indices: list[SerializedDict]
    args: list[SerializedDict]


def _is_valid_function_expression_data(
    data: SerializedDict,
) -> TypeGuard[_FunctionExpressionData]:
    return (
        "function" in data
        and is_serialized_dict(data["function"])
        and "template_types" in data
        and isinstance(data["template_types"], list)
        and all(is_serialized_dict(type_data) for type_data in data["template_types"])
        and "indices" in data
        and isinstance(data["indices"], list)
        and all(is_serialized_dict(index_data) for index_data in data["indices"])
        and "args" in data
        and isinstance(data["args"], list)
        and all(is_serialized_dict(arg_data) for arg_data in data["args"])
        and is_valid_expression_data(data)
    )


@register_serializable(type_id="fhy_ast_function_expression")
@dataclass(frozen=True, kw_only=True)
class FunctionExpression(Expression, HasOperandsMixin[Expression]):
    """FhY function call expression AST node."""

    function: Expression
    template_types: tuple[DataType, ...] = field(default_factory=tuple)
    indices: tuple[Expression, ...] = field(default_factory=tuple)
    args: tuple[Expression, ...] = field(default_factory=tuple)

    def get_operands(self) -> tuple[Expression, ...]:
        return self.args

    def get_visit_children(self) -> Sequence[Visitable]:
        return (self.function, *self.indices, *self.args)

    def is_structurally_equivalent(self, other: object) -> bool:
        return (
            isinstance(other, FunctionExpression)
            and super().is_structurally_equivalent(other)
            and self.function.is_structurally_equivalent(other.function)
            and len(self.template_types) == len(other.template_types)
            and all(
                template_type.is_structurally_equivalent(other_template_type)
                for template_type, other_template_type in zip(
                    self.template_types, other.template_types
                )
            )
            and len(self.indices) == len(other.indices)
            and all(
                index.is_structurally_equivalent(other_index)
                for index, other_index in zip(self.indices, other.indices)
            )
            and len(self.args) == len(other.args)
            and all(
                arg.is_structurally_equivalent(other_arg)
                for arg, other_arg in zip(self.args, other.args)
            )
        )

    def serialize_data_to_dict(self) -> SerializedDict:
        data = super().serialize_data_to_dict()
        data["function"] = self.function.serialize_to_dict()
        data["template_types"] = [
            template_type.serialize_to_dict() for template_type in self.template_types
        ]
        data["indices"] = [index.serialize_to_dict() for index in self.indices]
        data["args"] = [arg.serialize_to_dict() for arg in self.args]
        return data

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "FunctionExpression":
        if not _is_valid_function_expression_data(data):
            raise DeserializationDictStructureError(
                cls, _FunctionExpressionData.__annotations__, data
            )
        return cls(
            function=Expression.deserialize_from_dict(data["function"]),
            template_types=tuple(
                DataType.deserialize_from_dict(template_type)
                for template_type in data["template_types"]
            ),
            indices=tuple(
                Expression.deserialize_from_dict(index) for index in data["indices"]
            ),
            args=tuple(Expression.deserialize_from_dict(arg) for arg in data["args"]),
            provenance=deserialize_node_provenance(data),
        )


class _ArrayAccessExpressionData(ExpressionData):
    """Data for an array-access expression node."""

    array_expression: SerializedDict
    indices: list[SerializedDict]


def _is_valid_array_access_expression_data(
    data: SerializedDict,
) -> TypeGuard[_ArrayAccessExpressionData]:
    return (
        "array_expression" in data
        and is_serialized_dict(data["array_expression"])
        and "indices" in data
        and isinstance(data["indices"], list)
        and all(is_serialized_dict(index_data) for index_data in data["indices"])
        and is_valid_expression_data(data)
    )


@register_serializable(type_id="fhy_ast_array_access_expression")
@dataclass(frozen=True, kw_only=True)
class ArrayAccessExpression(Expression, HasOperandsMixin[Expression]):
    """FhY array access expression AST node."""

    array_expression: Expression
    indices: tuple[Expression, ...] = field(default_factory=tuple)

    def get_operands(self) -> tuple[Expression, ...]:
        return (self.array_expression, *self.indices)

    def get_visit_children(self) -> Sequence[Visitable]:
        return (self.array_expression, *self.indices)

    def is_structurally_equivalent(self, other: object) -> bool:
        return (
            isinstance(other, ArrayAccessExpression)
            and super().is_structurally_equivalent(other)
            and self.array_expression.is_structurally_equivalent(other.array_expression)
            and len(self.indices) == len(other.indices)
            and all(
                index.is_structurally_equivalent(other_index)
                for index, other_index in zip(self.indices, other.indices)
            )
        )

    def serialize_data_to_dict(self) -> SerializedDict:
        data = super().serialize_data_to_dict()
        data["array_expression"] = self.array_expression.serialize_to_dict()
        data["indices"] = tuple(index.serialize_to_dict() for index in self.indices)
        return data

    @classmethod
    def deserialize_data_from_dict(
        cls,
        data: SerializedDict,
    ) -> "ArrayAccessExpression":
        if not _is_valid_array_access_expression_data(data):
            raise DeserializationDictStructureError(
                cls, _ArrayAccessExpressionData.__annotations__, data
            )
        return cls(
            array_expression=Expression.deserialize_from_dict(data["array_expression"]),
            indices=tuple(
                Expression.deserialize_from_dict(index) for index in data["indices"]
            ),
            provenance=deserialize_node_provenance(data),
        )


class _TupleExpressionData(ExpressionData):
    expressions: tuple[SerializedDict, ...]


def _is_valid_tuple_expression_data(
    data: SerializedDict,
) -> TypeGuard[_TupleExpressionData]:
    return (
        "expressions" in data
        and isinstance(data["expressions"], list)
        and all(is_serialized_dict(expression) for expression in data["expressions"])
        and is_valid_expression_data(data)
    )


@register_serializable(type_id="fhy_ast_tuple_expression")
@dataclass(frozen=True, kw_only=True)
class TupleExpression(Expression, HasOperandsMixin[Expression]):
    """FhY tuple expression AST node."""

    expressions: tuple[Expression, ...] = field(default_factory=tuple)

    def get_operands(self) -> tuple[Expression, ...]:
        return self.expressions

    def get_visit_children(self) -> Sequence[Visitable]:
        return self.expressions

    def is_structurally_equivalent(self, other: object) -> bool:
        return (
            isinstance(other, TupleExpression)
            and super().is_structurally_equivalent(other)
            and len(self.expressions) == len(other.expressions)
            and all(
                expression.is_structurally_equivalent(other_expression)
                for expression, other_expression in zip(
                    self.expressions, other.expressions
                )
            )
        )

    def serialize_data_to_dict(self) -> SerializedDict:
        data = super().serialize_data_to_dict()
        data["expressions"] = [
            expression.serialize_to_dict() for expression in self.expressions
        ]
        return data

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "TupleExpression":
        if not _is_valid_tuple_expression_data(data):
            raise DeserializationDictStructureError(
                cls, _TupleExpressionData.__annotations__, data
            )
        return cls(
            expressions=tuple(
                Expression.deserialize_from_dict(expression)
                for expression in data["expressions"]
            ),
            provenance=deserialize_node_provenance(data),
        )


class _IdentifierExpressionData(ExpressionData):
    identifier: SerializedDict


def _is_valid_identifier_expression_data(
    data: SerializedDict,
) -> TypeGuard[_IdentifierExpressionData]:
    return (
        "identifier" in data
        and is_serialized_dict(data["identifier"])
        and is_valid_expression_data(data)
    )


@register_serializable(type_id="fhy_ast_identifier_expression")
@dataclass(frozen=True, kw_only=True)
class IdentifierExpression(Expression):
    """FhY identifier expression AST node."""

    identifier: Identifier

    def is_structurally_equivalent(self, other: object) -> bool:
        return (
            isinstance(other, IdentifierExpression)
            and super().is_structurally_equivalent(other)
            and self.identifier == other.identifier
        )

    def serialize_data_to_dict(self) -> SerializedDict:
        data = super().serialize_data_to_dict()
        data["identifier"] = self.identifier.serialize_to_dict()
        return data

    @classmethod
    def deserialize_data_from_dict(
        cls,
        data: SerializedDict,
    ) -> "IdentifierExpression":
        if not _is_valid_identifier_expression_data(data):
            raise DeserializationDictStructureError(
                cls, _IdentifierExpressionData.__annotations__, data
            )
        return cls(
            identifier=Identifier.deserialize_from_dict(data["identifier"]),
            provenance=deserialize_node_provenance(data),
        )


class _LiteralData(ExpressionData): ...


def _is_valid_literal_data(data: SerializedDict) -> TypeGuard[_LiteralData]:
    return is_valid_expression_data(data)


@dataclass(frozen=True, kw_only=True)
class Literal(Expression, ABC):
    """Abstract expression node to define concrete values."""


class _IntLiteralData(_LiteralData):
    value: int


def _is_valid_int_literal_data(data: SerializedDict) -> TypeGuard[_IntLiteralData]:
    return (
        "value" in data
        and isinstance(data["value"], int)
        and _is_valid_literal_data(data)
    )


@register_serializable(type_id="fhy_ast_int_literal")
@dataclass(frozen=True, kw_only=True)
class IntLiteral(Literal):
    """FhY integer literal AST node."""

    value: int

    def is_structurally_equivalent(self, other: object) -> bool:
        return (
            isinstance(other, IntLiteral)
            and super().is_structurally_equivalent(other)
            and self.value == other.value
        )

    def serialize_data_to_dict(self) -> SerializedDict:
        data = super().serialize_data_to_dict()
        data["value"] = self.value
        return data

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "IntLiteral":
        if not _is_valid_int_literal_data(data):
            raise DeserializationDictStructureError(
                cls, _IntLiteralData.__annotations__, data
            )
        return cls(value=data["value"], provenance=deserialize_node_provenance(data))


class _FloatLiteralData(_LiteralData):
    value: float


def _is_valid_float_literal_data(data: SerializedDict) -> TypeGuard[_FloatLiteralData]:
    return (
        "value" in data
        and isinstance(data["value"], float)
        and _is_valid_literal_data(data)
    )


@register_serializable(type_id="fhy_ast_float_literal")
@dataclass(frozen=True, kw_only=True)
class FloatLiteral(Literal):
    """FhY floating point literal AST node."""

    value: float

    def is_structurally_equivalent(self, other: object) -> bool:
        return (
            isinstance(other, FloatLiteral)
            and super().is_structurally_equivalent(other)
            and self.value == other.value
        )

    def serialize_data_to_dict(self) -> SerializedDict:
        data = super().serialize_data_to_dict()
        data["value"] = self.value
        return data

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "FloatLiteral":
        if not _is_valid_float_literal_data(data):
            raise DeserializationDictStructureError(
                cls, _FloatLiteralData.__annotations__, data
            )
        return cls(value=data["value"], provenance=deserialize_node_provenance(data))


class _ComplexLiteralData(_LiteralData):
    real: float
    imag: float


def _is_valid_complex_literal_data(
    data: SerializedDict,
) -> TypeGuard[_ComplexLiteralData]:
    return (
        "real" in data
        and isinstance(data["real"], float)
        and "imag" in data
        and isinstance(data["imag"], float)
        and _is_valid_literal_data(data)
    )


@register_serializable(type_id="fhy_ast_complex_literal")
@dataclass(frozen=True, kw_only=True)
class ComplexLiteral(Literal):
    """FhY complex literal AST node."""

    value: complex

    def is_structurally_equivalent(self, other: object) -> bool:
        return (
            isinstance(other, ComplexLiteral)
            and super().is_structurally_equivalent(other)
            and self.value == other.value
        )

    def serialize_data_to_dict(self) -> SerializedDict:
        data = super().serialize_data_to_dict()
        data["real"] = self.value.real
        data["imag"] = self.value.imag
        return data

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "ComplexLiteral":
        if not _is_valid_complex_literal_data(data):
            raise DeserializationDictStructureError(
                cls, _ComplexLiteralData.__annotations__, data
            )
        return cls(
            value=complex(data["real"], data["imag"]),
            provenance=deserialize_node_provenance(data),
        )
