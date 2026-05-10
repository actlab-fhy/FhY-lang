"""Tests for ``fhy_lang.ast.node.expression``.

Covers every concrete expression node and literal: construction,
structural equivalence (id-based), visitor/operands contracts (including
the documented exclusion of templates from the AST walk),
serialization round-trips, and deserialization error paths.
"""

import pytest
from fhy_core import (
    DataType,
    DeserializationDictStructureError,
    DeserializationValueError,
    Identifier,
    TemplateDataType,
)

from fhy_lang.ast.node import (
    ArrayAccessExpression,
    BinaryExpression,
    BinaryOperation,
    ComplexLiteral,
    Expression,
    FloatLiteral,
    FunctionExpression,
    IdentifierExpression,
    IntLiteral,
    Literal,
    TernaryExpression,
    TupleAccessExpression,
    TupleExpression,
    UnaryExpression,
    UnaryOperation,
)
from fhy_lang.ast.node import expression as expression_module

# ===========================================================================
# Helpers
# ===========================================================================


def _make_int_literal(value: int) -> IntLiteral:
    """Build an ``IntLiteral`` node with the given value."""
    return IntLiteral(value=value)


def _make_identifier_expression(name: str = "x") -> IdentifierExpression:
    """Build an ``IdentifierExpression`` whose identifier has the given name."""
    return IdentifierExpression(identifier=Identifier(name))


# ===========================================================================
# Abstracts
# ===========================================================================


def test_literal_is_abstract():
    """Test ``Literal`` cannot be instantiated."""
    with pytest.raises(TypeError):
        Literal()  # type: ignore[abstract]


# ===========================================================================
# IntLiteral  (redistributed from test_ast.py)
# ===========================================================================


def test_int_literal_equivalent_when_value_matches():
    """Test int literals with the same value are structurally equivalent."""
    a = IntLiteral(value=7)
    b = IntLiteral(value=7)

    assert a.is_structurally_equivalent(b)
    assert b.is_structurally_equivalent(a)


def test_int_literal_inequivalent_when_value_differs():
    """Test int literals with different values are not structurally equivalent."""
    a = IntLiteral(value=7)
    b = IntLiteral(value=8)

    assert not a.is_structurally_equivalent(b)


def test_int_literal_inequivalent_with_float_literal_of_same_value():
    """Test ``IntLiteral(1)`` is not equivalent to ``FloatLiteral(1.0)``."""
    assert not IntLiteral(value=1).is_structurally_equivalent(FloatLiteral(value=1.0))


def test_int_literal_round_trips_through_serialization():
    """Test ``IntLiteral`` survives a wrapped serialize/deserialize round-trip."""
    literal = IntLiteral(value=42)

    restored = Expression.deserialize_from_dict(literal.serialize_to_dict())

    assert isinstance(restored, IntLiteral)
    assert restored.value == 42


def test_int_literal_deserialize_data_rejects_missing_value():
    """Test ``IntLiteral.deserialize_data_from_dict`` rejects missing ``value``."""
    payload = IntLiteral(value=1).serialize_data_to_dict()
    payload.pop("value")

    with pytest.raises(DeserializationDictStructureError):
        IntLiteral.deserialize_data_from_dict(payload)


def test_int_literal_deserialize_data_rejects_non_int_value():
    """Test a non-int ``value`` raises ``DeserializationDictStructureError``."""
    payload = IntLiteral(value=1).serialize_data_to_dict()
    payload["value"] = "not-an-int"

    with pytest.raises(DeserializationDictStructureError):
        IntLiteral.deserialize_data_from_dict(payload)


# ===========================================================================
# FloatLiteral  (reject NaN/infinity at construction; deserialize coerces int to float)
# ===========================================================================


def test_float_literal_equivalent_when_value_matches():
    """Test float literals with the same value are equivalent."""
    assert FloatLiteral(value=2.5).is_structurally_equivalent(FloatLiteral(value=2.5))


def test_float_literal_inequivalent_when_value_differs():
    """Test float literals with different values are inequivalent."""
    assert not FloatLiteral(value=2.5).is_structurally_equivalent(
        FloatLiteral(value=3.5)
    )


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), -float("inf")],
    ids=["nan", "+inf", "-inf"],
)
def test_float_literal_construction_rejects_non_finite_values(value: float):
    """Test ``FloatLiteral`` rejects NaN and infinities at construction."""
    with pytest.raises(ValueError):
        FloatLiteral(value=value)


def test_float_literal_round_trips_through_serialization():
    """Test ``FloatLiteral`` survives a wrapped round-trip."""
    literal = FloatLiteral(value=3.14)

    restored = Expression.deserialize_from_dict(literal.serialize_to_dict())

    assert isinstance(restored, FloatLiteral)
    assert restored.value == 3.14


def test_float_literal_deserialize_data_accepts_integer_value():
    """Test deserialization coerces an integer ``value`` to ``float``."""
    payload = FloatLiteral(value=1.0).serialize_data_to_dict()
    payload["value"] = 2

    restored = FloatLiteral.deserialize_data_from_dict(payload)

    assert isinstance(restored.value, float)
    assert restored.value == 2.0


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), -float("inf")], ids=["nan", "+inf", "-inf"]
)
def test_float_literal_deserialize_data_rejects_non_finite_value_with_value_error(
    value: float,
):
    """Test non-finite payload values surface as ``DeserializationValueError``."""
    payload = FloatLiteral(value=1.0).serialize_data_to_dict()
    payload["value"] = value

    with pytest.raises(DeserializationValueError):
        FloatLiteral.deserialize_data_from_dict(payload)


def test_float_literal_deserialize_data_rejects_overflowing_int_with_value_error():
    """Test integer payloads too large for ``float`` raise the value error."""
    payload = FloatLiteral(value=1.0).serialize_data_to_dict()
    payload["value"] = 10**400

    with pytest.raises(DeserializationValueError):
        FloatLiteral.deserialize_data_from_dict(payload)


def test_float_literal_deserialize_data_rejects_bool_value():
    """Test ``bool`` ``value`` raises ``DeserializationDictStructureError``."""
    payload = FloatLiteral(value=1.0).serialize_data_to_dict()
    payload["value"] = True

    with pytest.raises(DeserializationDictStructureError):
        FloatLiteral.deserialize_data_from_dict(payload)


def test_float_literal_deserialize_data_rejects_string_value():
    """Test ``str`` ``value`` raises ``DeserializationDictStructureError``."""
    payload = FloatLiteral(value=1.0).serialize_data_to_dict()
    payload["value"] = "not-a-number"

    with pytest.raises(DeserializationDictStructureError):
        FloatLiteral.deserialize_data_from_dict(payload)


def test_float_literal_deserialize_data_rejects_missing_value():
    """Test missing ``value`` raises ``DeserializationDictStructureError``."""
    payload = FloatLiteral(value=1.0).serialize_data_to_dict()
    payload.pop("value")

    with pytest.raises(DeserializationDictStructureError):
        FloatLiteral.deserialize_data_from_dict(payload)


# ===========================================================================
# ComplexLiteral
# ===========================================================================


def test_complex_literal_equivalent_when_value_matches():
    """Test complex literals with the same value are equivalent."""
    assert ComplexLiteral(value=complex(1.0, 2.0)).is_structurally_equivalent(
        ComplexLiteral(value=complex(1.0, 2.0))
    )


def test_complex_literal_inequivalent_when_real_differs():
    """Test differing real parts break equivalence."""
    assert not ComplexLiteral(value=complex(1.0, 2.0)).is_structurally_equivalent(
        ComplexLiteral(value=complex(3.0, 2.0))
    )


def test_complex_literal_inequivalent_when_imag_differs():
    """Test differing imag parts break equivalence."""
    assert not ComplexLiteral(value=complex(1.0, 2.0)).is_structurally_equivalent(
        ComplexLiteral(value=complex(1.0, 4.0))
    )


@pytest.mark.parametrize(
    "real, imag",
    [
        (float("nan"), 0.0),
        (0.0, float("nan")),
        (float("inf"), 0.0),
        (0.0, -float("inf")),
        (float("nan"), float("nan")),
    ],
    ids=["real-nan", "imag-nan", "real-+inf", "imag--inf", "both-nan"],
)
def test_complex_literal_construction_rejects_non_finite_components(
    real: float, imag: float
):
    """Test ``ComplexLiteral`` rejects NaN or infinity in either component."""
    with pytest.raises(ValueError):
        ComplexLiteral(value=complex(real, imag))


def test_complex_literal_round_trips_through_serialization():
    """Test ``ComplexLiteral`` survives a wrapped round-trip."""
    literal = ComplexLiteral(value=complex(2.0, -3.0))

    restored = Expression.deserialize_from_dict(literal.serialize_to_dict())

    assert isinstance(restored, ComplexLiteral)
    assert restored.value == complex(2.0, -3.0)


def test_complex_literal_deserialize_data_accepts_integer_real_and_imag():
    """Test deserialization coerces integer ``real``/``imag`` to ``float``."""
    payload = ComplexLiteral(value=complex(1.0, 2.0)).serialize_data_to_dict()
    payload["real"] = 3
    payload["imag"] = 4

    restored = ComplexLiteral.deserialize_data_from_dict(payload)

    assert restored.value == complex(3.0, 4.0)
    assert isinstance(restored.value.real, float)
    assert isinstance(restored.value.imag, float)


@pytest.mark.parametrize(
    "real, imag",
    [
        (float("nan"), 0.0),
        (0.0, float("inf")),
        (-float("inf"), 0.0),
    ],
    ids=["real-nan", "imag-+inf", "real--inf"],
)
def test_complex_literal_deserialize_data_rejects_non_finite_with_value_error(
    real: float, imag: float
):
    """Test non-finite real/imag payloads surface as ``DeserializationValueError``."""
    payload = ComplexLiteral(value=complex(1.0, 2.0)).serialize_data_to_dict()
    payload["real"] = real
    payload["imag"] = imag

    with pytest.raises(DeserializationValueError):
        ComplexLiteral.deserialize_data_from_dict(payload)


@pytest.mark.parametrize("field", ["real", "imag"])
def test_complex_literal_deserialize_data_rejects_bool_in_either_component(field: str):
    """Test ``bool`` real/imag raises ``DeserializationDictStructureError``."""
    payload = ComplexLiteral(value=complex(1.0, 2.0)).serialize_data_to_dict()
    payload[field] = True

    with pytest.raises(DeserializationDictStructureError):
        ComplexLiteral.deserialize_data_from_dict(payload)


@pytest.mark.parametrize("field", ["real", "imag"])
def test_complex_literal_deserialize_data_rejects_missing_component(field: str):
    """Test missing ``real``/``imag`` raises ``DeserializationDictStructureError``."""
    payload = ComplexLiteral(value=complex(1.0, 2.0)).serialize_data_to_dict()
    payload.pop(field)

    with pytest.raises(DeserializationDictStructureError):
        ComplexLiteral.deserialize_data_from_dict(payload)


# ===========================================================================
# IdentifierExpression
# ===========================================================================


def test_identifier_expression_equivalent_when_identifier_is_same():
    """Test identifier expressions sharing one ``Identifier`` are equivalent."""
    ident = Identifier("x")

    assert IdentifierExpression(identifier=ident).is_structurally_equivalent(
        IdentifierExpression(identifier=ident)
    )


def test_identifier_expression_inequivalent_when_identifiers_are_independent():
    """Test independently-constructed ``Identifier("x")`` instances are inequivalent."""
    a = IdentifierExpression(identifier=Identifier("x"))
    b = IdentifierExpression(identifier=Identifier("x"))

    assert not a.is_structurally_equivalent(b)


def test_identifier_expression_inequivalent_when_names_differ():
    """Test different name hints with different ids are inequivalent (trivial)."""
    a = IdentifierExpression(identifier=Identifier("x"))
    b = IdentifierExpression(identifier=Identifier("y"))

    assert not a.is_structurally_equivalent(b)


def test_identifier_expression_round_trips_through_serialization():
    """Test ``IdentifierExpression`` survives serialization round-trip."""
    ident = Identifier("x")
    expression = IdentifierExpression(identifier=ident)

    restored = Expression.deserialize_from_dict(expression.serialize_to_dict())

    assert isinstance(restored, IdentifierExpression)
    assert restored.identifier == ident  # same id


def test_identifier_expression_deserialize_rejects_missing_identifier():
    """Test missing ``identifier`` raises ``DeserializationDictStructureError``."""
    payload = IdentifierExpression(identifier=Identifier("x")).serialize_data_to_dict()
    payload.pop("identifier")

    with pytest.raises(DeserializationDictStructureError):
        IdentifierExpression.deserialize_data_from_dict(payload)


# ===========================================================================
# UnaryExpression
# ===========================================================================


def test_unary_expression_equivalent_when_operation_and_operand_match():
    """Test unary expressions with matching operation and operand are equivalent."""
    a = UnaryExpression(
        operation=UnaryOperation.NEGATION, expression=_make_int_literal(1)
    )
    b = UnaryExpression(
        operation=UnaryOperation.NEGATION, expression=_make_int_literal(1)
    )

    assert a.is_structurally_equivalent(b)


def test_unary_expression_inequivalent_when_operation_differs():
    """Test differing unary operations break equivalence."""
    a = UnaryExpression(
        operation=UnaryOperation.NEGATION, expression=_make_int_literal(1)
    )
    b = UnaryExpression(
        operation=UnaryOperation.LOGICAL_NOT, expression=_make_int_literal(1)
    )

    assert not a.is_structurally_equivalent(b)


def test_unary_expression_inequivalent_when_operand_differs():
    """Test differing operand breaks equivalence."""
    a = UnaryExpression(
        operation=UnaryOperation.NEGATION, expression=_make_int_literal(1)
    )
    b = UnaryExpression(
        operation=UnaryOperation.NEGATION, expression=_make_int_literal(2)
    )

    assert not a.is_structurally_equivalent(b)


def test_unary_expression_get_operands_returns_inner_expression():
    """Test ``UnaryExpression.get_operands`` returns a 1-tuple of the inner."""
    inner = _make_int_literal(1)
    unary = UnaryExpression(operation=UnaryOperation.NEGATION, expression=inner)

    assert unary.get_operands() == (inner,)


def test_unary_expression_get_visit_children_returns_inner_expression():
    """Test ``UnaryExpression.get_visit_children`` walks only the inner expression."""
    inner = _make_int_literal(1)
    unary = UnaryExpression(operation=UnaryOperation.NEGATION, expression=inner)

    assert tuple(unary.get_visit_children()) == (inner,)


def test_unary_expression_round_trips_through_serialization():
    """Test ``UnaryExpression`` survives serialization round-trip."""
    expression = UnaryExpression(
        operation=UnaryOperation.NEGATION, expression=_make_int_literal(1)
    )

    restored = Expression.deserialize_from_dict(expression.serialize_to_dict())

    assert isinstance(restored, UnaryExpression)
    assert restored.operation is UnaryOperation.NEGATION


def test_unary_expression_deserialize_rejects_unknown_operation():
    """Test an unknown operation string raises ``DeserializationValueError``."""
    payload = UnaryExpression(
        operation=UnaryOperation.NEGATION, expression=_make_int_literal(1)
    ).serialize_data_to_dict()
    payload["operation"] = "??"

    with pytest.raises(DeserializationValueError):
        UnaryExpression.deserialize_data_from_dict(payload)


def test_unary_expression_deserialize_rejects_missing_operation():
    """Test missing ``operation`` raises ``DeserializationDictStructureError``."""
    payload = UnaryExpression(
        operation=UnaryOperation.NEGATION, expression=_make_int_literal(1)
    ).serialize_data_to_dict()
    payload.pop("operation")

    with pytest.raises(DeserializationDictStructureError):
        UnaryExpression.deserialize_data_from_dict(payload)


# ===========================================================================
# BinaryExpression
# ===========================================================================


def test_binary_expression_equivalent_when_operation_and_operands_match():
    """Test binary expressions with matching operation and operands are equivalent."""
    a = BinaryExpression(
        operation=BinaryOperation.ADDITION,
        left=_make_int_literal(1),
        right=_make_int_literal(2),
    )
    b = BinaryExpression(
        operation=BinaryOperation.ADDITION,
        left=_make_int_literal(1),
        right=_make_int_literal(2),
    )

    assert a.is_structurally_equivalent(b)


def test_binary_expression_inequivalent_when_operation_differs():
    """Test differing binary operations break equivalence."""
    a = BinaryExpression(
        operation=BinaryOperation.ADDITION,
        left=_make_int_literal(1),
        right=_make_int_literal(2),
    )
    b = BinaryExpression(
        operation=BinaryOperation.SUBTRACTION,
        left=_make_int_literal(1),
        right=_make_int_literal(2),
    )

    assert not a.is_structurally_equivalent(b)


def test_binary_expression_inequivalent_when_left_differs():
    """Test differing left operand breaks equivalence."""
    a = BinaryExpression(
        operation=BinaryOperation.ADDITION,
        left=_make_int_literal(1),
        right=_make_int_literal(2),
    )
    b = BinaryExpression(
        operation=BinaryOperation.ADDITION,
        left=_make_int_literal(99),
        right=_make_int_literal(2),
    )

    assert not a.is_structurally_equivalent(b)


def test_binary_expression_inequivalent_when_right_differs():
    """Test differing right operand breaks equivalence."""
    a = BinaryExpression(
        operation=BinaryOperation.ADDITION,
        left=_make_int_literal(1),
        right=_make_int_literal(2),
    )
    b = BinaryExpression(
        operation=BinaryOperation.ADDITION,
        left=_make_int_literal(1),
        right=_make_int_literal(99),
    )

    assert not a.is_structurally_equivalent(b)


def test_binary_expression_get_operands_returns_left_and_right():
    """Test ``BinaryExpression.get_operands`` returns ``(left, right)``."""
    left, right = _make_int_literal(1), _make_int_literal(2)
    binary = BinaryExpression(
        operation=BinaryOperation.ADDITION, left=left, right=right
    )

    assert binary.get_operands() == (left, right)


def test_binary_expression_get_visit_children_returns_left_and_right():
    """Test ``BinaryExpression.get_visit_children`` walks left then right."""
    left, right = _make_int_literal(1), _make_int_literal(2)
    binary = BinaryExpression(
        operation=BinaryOperation.ADDITION, left=left, right=right
    )

    assert tuple(binary.get_visit_children()) == (left, right)


def test_binary_expression_round_trips_through_serialization():
    """Test ``BinaryExpression`` survives serialization round-trip."""
    expression = BinaryExpression(
        operation=BinaryOperation.MULTIPLICATION,
        left=_make_int_literal(3),
        right=_make_int_literal(4),
    )

    restored = Expression.deserialize_from_dict(expression.serialize_to_dict())

    assert isinstance(restored, BinaryExpression)
    assert restored.operation is BinaryOperation.MULTIPLICATION


def test_binary_expression_deserialize_rejects_unknown_operation():
    """Test unknown operation raises ``DeserializationValueError``."""
    payload = BinaryExpression(
        operation=BinaryOperation.ADDITION,
        left=_make_int_literal(1),
        right=_make_int_literal(2),
    ).serialize_data_to_dict()
    payload["operation"] = "<<>>"

    with pytest.raises(DeserializationValueError):
        BinaryExpression.deserialize_data_from_dict(payload)


def test_binary_expression_validator_delegates_to_expression_data_check(monkeypatch):
    """Test the binary expression validator calls ``is_valid_expression_data``."""
    payload = BinaryExpression(
        operation=BinaryOperation.ADDITION,
        left=_make_int_literal(1),
        right=_make_int_literal(2),
    ).serialize_data_to_dict()

    monkeypatch.setattr(
        expression_module, "is_valid_expression_data", lambda data: False
    )

    with pytest.raises(DeserializationDictStructureError):
        BinaryExpression.deserialize_data_from_dict(payload)


# ===========================================================================
# TernaryExpression
# ===========================================================================


def test_ternary_expression_equivalent_when_all_branches_match():
    """Test ternary expressions with matching condition/true/false are equivalent."""
    a = TernaryExpression(
        condition=_make_int_literal(0),
        true=_make_int_literal(1),
        false=_make_int_literal(2),
    )
    b = TernaryExpression(
        condition=_make_int_literal(0),
        true=_make_int_literal(1),
        false=_make_int_literal(2),
    )

    assert a.is_structurally_equivalent(b)


@pytest.mark.parametrize("field", ["condition", "true", "false"])
def test_ternary_expression_inequivalent_when_one_branch_differs(field: str):
    """Test differing condition / true / false breaks equivalence."""
    base = {
        "condition": _make_int_literal(0),
        "true": _make_int_literal(1),
        "false": _make_int_literal(2),
    }
    other = dict(base)
    other[field] = _make_int_literal(99)

    assert not TernaryExpression(**base).is_structurally_equivalent(
        TernaryExpression(**other)
    )


def test_ternary_expression_get_visit_children_returns_three_branches_in_order():
    """Test ``TernaryExpression.get_visit_children`` walks condition, true, false."""
    condition, true, false = (
        _make_int_literal(0),
        _make_int_literal(1),
        _make_int_literal(2),
    )
    ternary = TernaryExpression(condition=condition, true=true, false=false)

    assert tuple(ternary.get_visit_children()) == (condition, true, false)


def test_ternary_expression_round_trips_through_serialization():
    """Test ``TernaryExpression`` survives a wrapped round-trip."""
    expression = TernaryExpression(
        condition=_make_int_literal(0),
        true=_make_int_literal(1),
        false=_make_int_literal(2),
    )

    restored = Expression.deserialize_from_dict(expression.serialize_to_dict())

    assert isinstance(restored, TernaryExpression)


# ===========================================================================
# TupleExpression  (redistributed from test_ast.py)
# ===========================================================================


def test_tuple_expression_equivalent_when_elements_match():
    """Test tuple expressions with matching elements are equivalent."""
    a = TupleExpression(expressions=(_make_int_literal(1), _make_int_literal(2)))
    b = TupleExpression(expressions=(_make_int_literal(1), _make_int_literal(2)))

    assert a.is_structurally_equivalent(b)


def test_tuple_expression_inequivalent_when_lengths_differ():
    """Test differing tuple lengths break equivalence."""
    a = TupleExpression(expressions=(_make_int_literal(1), _make_int_literal(2)))
    b = TupleExpression(expressions=(_make_int_literal(1),))

    assert not a.is_structurally_equivalent(b)


def test_tuple_expression_inequivalent_when_an_element_differs():
    """Test a single mismatched element breaks equivalence."""
    a = TupleExpression(expressions=(_make_int_literal(1), _make_int_literal(2)))
    b = TupleExpression(expressions=(_make_int_literal(1), _make_int_literal(99)))

    assert not a.is_structurally_equivalent(b)


def test_tuple_expression_with_no_elements_is_equivalent_to_self():
    """Test an empty ``TupleExpression`` is equivalent to another empty one."""
    a = TupleExpression()
    b = TupleExpression()

    assert a.is_structurally_equivalent(b)


def test_tuple_expression_get_operands_returns_inner_tuple():
    """Test ``TupleExpression.get_operands`` returns the elements tuple."""
    elements = (_make_int_literal(1), _make_int_literal(2))
    tuple_expr = TupleExpression(expressions=elements)

    assert tuple_expr.get_operands() == elements


def test_tuple_expression_get_visit_children_returns_expressions():
    """Test ``TupleExpression.get_visit_children`` returns the elements tuple."""
    elements = (_make_int_literal(1), _make_int_literal(2))
    tuple_expr = TupleExpression(expressions=elements)

    assert tuple(tuple_expr.get_visit_children()) == elements


def test_tuple_expression_round_trips_through_serialization():
    """Test ``TupleExpression`` survives a wrapped round-trip."""
    expression = TupleExpression(
        expressions=(_make_int_literal(1), _make_int_literal(2))
    )

    restored = Expression.deserialize_from_dict(expression.serialize_to_dict())

    assert isinstance(restored, TupleExpression)
    assert len(restored.expressions) == 2


# ===========================================================================
# TupleAccessExpression
# ===========================================================================


def test_tuple_access_expression_equivalent_when_tuple_and_index_match():
    """Test tuple-access expressions with matching parts are equivalent."""
    a = TupleAccessExpression(
        tuple_expression=_make_identifier_expression("t"),
        element_index=_make_int_literal(0),
    )
    b = TupleAccessExpression(
        tuple_expression=_make_identifier_expression("t"),
        element_index=_make_int_literal(0),
    )

    # Identifiers in tuple_expression are independent, so the nodes are inequivalent.
    assert not a.is_structurally_equivalent(b)


def test_tuple_access_expression_equivalent_when_sharing_inner_identifier():
    """Test a shared inner ``IdentifierExpression`` is equivalent across nodes."""
    inner = _make_identifier_expression("t")
    a = TupleAccessExpression(
        tuple_expression=inner, element_index=_make_int_literal(0)
    )
    b = TupleAccessExpression(
        tuple_expression=inner, element_index=_make_int_literal(0)
    )

    assert a.is_structurally_equivalent(b)


def test_tuple_access_expression_inequivalent_when_index_differs():
    """Test differing ``element_index`` breaks equivalence."""
    inner = _make_identifier_expression("t")
    a = TupleAccessExpression(
        tuple_expression=inner, element_index=_make_int_literal(0)
    )
    b = TupleAccessExpression(
        tuple_expression=inner, element_index=_make_int_literal(1)
    )

    assert not a.is_structurally_equivalent(b)


def test_tuple_access_expression_get_operands_returns_tuple_and_index():
    """Test ``get_operands`` returns ``(tuple_expression, element_index)``."""
    inner = _make_identifier_expression("t")
    index = _make_int_literal(2)

    tuple_access = TupleAccessExpression(tuple_expression=inner, element_index=index)

    assert tuple_access.get_operands() == (inner, index)


def test_tuple_access_expression_get_visit_children_returns_tuple_then_index():
    """Test ``get_visit_children`` walks ``tuple_expression`` then ``element_index``."""
    inner = _make_identifier_expression("t")
    index = _make_int_literal(2)

    tuple_access = TupleAccessExpression(tuple_expression=inner, element_index=index)

    assert tuple(tuple_access.get_visit_children()) == (inner, index)


def test_tuple_access_expression_round_trips_through_serialization():
    """Test ``TupleAccessExpression`` survives a round-trip."""
    expression = TupleAccessExpression(
        tuple_expression=_make_identifier_expression("t"),
        element_index=_make_int_literal(3),
    )

    restored = Expression.deserialize_from_dict(expression.serialize_to_dict())

    assert isinstance(restored, TupleAccessExpression)


# ===========================================================================
# ArrayAccessExpression
# ===========================================================================


def test_array_access_expression_equivalent_when_array_and_indices_match():
    """Test array accesses sharing the array expression and indices are equivalent."""
    array = _make_identifier_expression("a")
    a = ArrayAccessExpression(
        array_expression=array, indices=(_make_int_literal(0), _make_int_literal(1))
    )
    b = ArrayAccessExpression(
        array_expression=array, indices=(_make_int_literal(0), _make_int_literal(1))
    )

    assert a.is_structurally_equivalent(b)


def test_array_access_expression_inequivalent_when_index_count_differs():
    """Test differing index counts break equivalence."""
    array = _make_identifier_expression("a")
    a = ArrayAccessExpression(array_expression=array, indices=(_make_int_literal(0),))
    b = ArrayAccessExpression(
        array_expression=array, indices=(_make_int_literal(0), _make_int_literal(1))
    )

    assert not a.is_structurally_equivalent(b)


def test_array_access_expression_get_operands_returns_array_then_indices():
    """Test ``get_operands`` returns ``(array_expression, *indices)``."""
    array = _make_identifier_expression("a")
    i0, i1 = _make_int_literal(0), _make_int_literal(1)

    access = ArrayAccessExpression(array_expression=array, indices=(i0, i1))

    assert access.get_operands() == (array, i0, i1)


def test_array_access_expression_get_visit_children_returns_array_then_indices():
    """Test ``get_visit_children`` walks ``array_expression`` then ``indices``."""
    array = _make_identifier_expression("a")
    i0, i1 = _make_int_literal(0), _make_int_literal(1)

    access = ArrayAccessExpression(array_expression=array, indices=(i0, i1))

    assert tuple(access.get_visit_children()) == (array, i0, i1)


def test_array_access_expression_serializes_indices_as_list():
    """Test ``serialize_data_to_dict`` emits ``indices`` as a list."""
    access = ArrayAccessExpression(
        array_expression=_make_identifier_expression("a"),
        indices=(_make_int_literal(0),),
    )

    data = access.serialize_data_to_dict()

    assert isinstance(data["indices"], list)


def test_array_access_expression_round_trips_through_serialization():
    """Test ``ArrayAccessExpression`` survives a wrapped round-trip."""
    access = ArrayAccessExpression(
        array_expression=_make_identifier_expression("a"),
        indices=(_make_int_literal(0), _make_int_literal(1)),
    )

    restored = Expression.deserialize_from_dict(access.serialize_to_dict())

    assert isinstance(restored, ArrayAccessExpression)
    assert len(restored.indices) == 2


# ===========================================================================
# FunctionExpression  (templates excluded from visitor walk)
# ===========================================================================


def test_function_expression_equivalent_when_all_components_match():
    """Test function expressions with matching function/indices/args are equivalent."""
    fn = _make_identifier_expression("f")
    a = FunctionExpression(
        function=fn, indices=(_make_int_literal(0),), args=(_make_int_literal(1),)
    )
    b = FunctionExpression(
        function=fn, indices=(_make_int_literal(0),), args=(_make_int_literal(1),)
    )

    assert a.is_structurally_equivalent(b)


def test_function_expression_get_operands_returns_only_args():
    """Test ``FunctionExpression.get_operands`` returns only the call args tuple."""
    arg_x = _make_int_literal(1)
    arg_y = _make_int_literal(2)
    function_expression = FunctionExpression(
        function=_make_identifier_expression("f"),
        indices=(_make_int_literal(0),),
        args=(arg_x, arg_y),
    )

    assert function_expression.get_operands() == (arg_x, arg_y)


def test_function_expression_inequivalent_when_args_count_differs():
    """Test differing argument count breaks equivalence."""
    fn = _make_identifier_expression("f")
    a = FunctionExpression(function=fn, args=(_make_int_literal(1),))
    b = FunctionExpression(
        function=fn, args=(_make_int_literal(1), _make_int_literal(2))
    )

    assert not a.is_structurally_equivalent(b)


def test_function_expression_get_visit_children_excludes_template_types():
    """Test ``get_visit_children`` does not include ``template_types``."""
    template: DataType = TemplateDataType(Identifier("T"))
    fn = _make_identifier_expression("f")
    function_expression = FunctionExpression(
        function=fn,
        template_types=(template,),
        indices=(_make_int_literal(0),),
        args=(_make_int_literal(1),),
    )

    children = tuple(function_expression.get_visit_children())

    assert template not in children


def test_function_expression_round_trips_through_serialization():
    """Test ``FunctionExpression`` survives a wrapped round-trip."""
    fn = _make_identifier_expression("f")
    function_expression = FunctionExpression(function=fn, args=(_make_int_literal(1),))

    restored = Expression.deserialize_from_dict(function_expression.serialize_to_dict())

    assert isinstance(restored, FunctionExpression)


# ===========================================================================
# Cross-type rejection: every concrete expression rejects an unrelated node
# from ``is_structurally_equivalent``.
# ===========================================================================


def test_float_literal_is_inequivalent_with_non_float_literal_node():
    """Test ``FloatLiteral`` rejects a non-``FloatLiteral`` node."""
    assert not FloatLiteral(value=1.0).is_structurally_equivalent(IntLiteral(value=1))


def test_complex_literal_is_inequivalent_with_non_complex_literal_node():
    """Test ``ComplexLiteral`` rejects a non-``ComplexLiteral`` node."""
    assert not ComplexLiteral(value=complex(1.0, 0.0)).is_structurally_equivalent(
        FloatLiteral(value=1.0)
    )


def test_identifier_expression_is_inequivalent_with_non_identifier_expression_node():
    """Test ``IdentifierExpression`` rejects an unrelated node."""
    assert not IdentifierExpression(
        identifier=Identifier("x")
    ).is_structurally_equivalent(IntLiteral(value=0))


def test_unary_expression_is_inequivalent_with_non_unary_node():
    """Test ``UnaryExpression`` rejects an unrelated node."""
    unary = UnaryExpression(
        operation=UnaryOperation.NEGATION, expression=_make_int_literal(1)
    )

    assert not unary.is_structurally_equivalent(_make_int_literal(1))


def test_binary_expression_is_inequivalent_with_non_binary_node():
    """Test ``BinaryExpression`` rejects an unrelated node."""
    binary = BinaryExpression(
        operation=BinaryOperation.ADDITION,
        left=_make_int_literal(1),
        right=_make_int_literal(2),
    )

    assert not binary.is_structurally_equivalent(_make_int_literal(1))


def test_ternary_expression_is_inequivalent_with_non_ternary_node():
    """Test ``TernaryExpression`` rejects an unrelated node."""
    ternary = TernaryExpression(
        condition=_make_int_literal(0),
        true=_make_int_literal(1),
        false=_make_int_literal(2),
    )

    assert not ternary.is_structurally_equivalent(_make_int_literal(1))


def test_tuple_expression_is_inequivalent_with_non_tuple_expression_node():
    """Test ``TupleExpression`` rejects an unrelated node."""
    assert not TupleExpression().is_structurally_equivalent(_make_int_literal(0))


def test_tuple_access_expression_is_inequivalent_with_non_tuple_access_node():
    """Test ``TupleAccessExpression`` rejects an unrelated node."""
    tuple_access = TupleAccessExpression(
        tuple_expression=_make_identifier_expression("t"),
        element_index=_make_int_literal(0),
    )

    assert not tuple_access.is_structurally_equivalent(_make_int_literal(0))


def test_array_access_expression_is_inequivalent_with_non_array_access_node():
    """Test ``ArrayAccessExpression`` rejects an unrelated node."""
    array_access = ArrayAccessExpression(
        array_expression=_make_identifier_expression("a")
    )

    assert not array_access.is_structurally_equivalent(_make_int_literal(0))


def test_function_expression_is_inequivalent_with_non_function_expression_node():
    """Test ``FunctionExpression`` rejects an unrelated node."""
    function_expression = FunctionExpression(function=_make_identifier_expression("f"))

    assert not function_expression.is_structurally_equivalent(_make_int_literal(0))


# ===========================================================================
# Per-class field-change inequivalence cases for nodes that weren't yet
# pinned field-by-field.
# ===========================================================================


def test_tuple_access_expression_inequivalent_when_tuple_expression_differs():
    """Test differing ``tuple_expression`` breaks equivalence (same index)."""
    index = _make_int_literal(0)
    a = TupleAccessExpression(
        tuple_expression=_make_identifier_expression("t"), element_index=index
    )
    b = TupleAccessExpression(
        tuple_expression=_make_identifier_expression("u"), element_index=index
    )

    assert not a.is_structurally_equivalent(b)


def test_array_access_expression_inequivalent_when_array_differs():
    """Test differing ``array_expression`` breaks equivalence (same indices)."""
    indices = (_make_int_literal(0),)
    a = ArrayAccessExpression(
        array_expression=_make_identifier_expression("a"), indices=indices
    )
    b = ArrayAccessExpression(
        array_expression=_make_identifier_expression("b"), indices=indices
    )

    assert not a.is_structurally_equivalent(b)


def test_array_access_expression_inequivalent_when_index_contents_differ():
    """Test differing index contents break equivalence (same count, same array)."""
    array = _make_identifier_expression("a")
    a = ArrayAccessExpression(array_expression=array, indices=(_make_int_literal(0),))
    b = ArrayAccessExpression(array_expression=array, indices=(_make_int_literal(1),))

    assert not a.is_structurally_equivalent(b)


def test_function_expression_inequivalent_when_function_differs():
    """Test differing function callee breaks equivalence."""
    a = FunctionExpression(function=_make_identifier_expression("f"))
    b = FunctionExpression(function=_make_identifier_expression("g"))

    assert not a.is_structurally_equivalent(b)


def test_function_expression_inequivalent_when_template_count_differs():
    """Test differing template-type count breaks equivalence."""
    fn = _make_identifier_expression("f")
    a = FunctionExpression(
        function=fn, template_types=(TemplateDataType(Identifier("T")),)
    )
    b = FunctionExpression(function=fn, template_types=())

    assert not a.is_structurally_equivalent(b)


def test_function_expression_inequivalent_when_template_contents_differ():
    """Test differing template-type contents break equivalence."""
    fn = _make_identifier_expression("f")
    a = FunctionExpression(
        function=fn, template_types=(TemplateDataType(Identifier("T")),)
    )
    b = FunctionExpression(
        function=fn, template_types=(TemplateDataType(Identifier("U")),)
    )

    assert not a.is_structurally_equivalent(b)


def test_function_expression_inequivalent_when_index_count_differs():
    """Test differing index count breaks equivalence."""
    fn = _make_identifier_expression("f")
    a = FunctionExpression(function=fn, indices=(_make_int_literal(0),))
    b = FunctionExpression(function=fn, indices=())

    assert not a.is_structurally_equivalent(b)


def test_function_expression_inequivalent_when_index_contents_differ():
    """Test differing index contents break equivalence (same count)."""
    fn = _make_identifier_expression("f")
    a = FunctionExpression(function=fn, indices=(_make_int_literal(0),))
    b = FunctionExpression(function=fn, indices=(_make_int_literal(1),))

    assert not a.is_structurally_equivalent(b)


def test_function_expression_inequivalent_when_arg_contents_differ():
    """Test differing argument contents break equivalence (same count)."""
    fn = _make_identifier_expression("f")
    a = FunctionExpression(function=fn, args=(_make_int_literal(0),))
    b = FunctionExpression(function=fn, args=(_make_int_literal(1),))

    assert not a.is_structurally_equivalent(b)


# ===========================================================================
# Deserialization rejects an empty dict for every concrete expression
# (these are simple structure-error pins, distinct from the missing-key tests).
# ===========================================================================


@pytest.mark.parametrize(
    "node_class",
    [
        IntLiteral,
        FloatLiteral,
        ComplexLiteral,
        IdentifierExpression,
        UnaryExpression,
        BinaryExpression,
        TernaryExpression,
        TupleExpression,
        TupleAccessExpression,
        ArrayAccessExpression,
        FunctionExpression,
    ],
)
def test_expression_deserialize_data_from_dict_rejects_empty_payload(
    node_class: type,
):
    """Test concrete expression classes reject an empty deserialization payload."""
    with pytest.raises(DeserializationDictStructureError):
        node_class.deserialize_data_from_dict({})


# ===========================================================================
# Edge-case literal value boundaries
# ===========================================================================


def test_int_literal_accepts_bool_value_and_preserves_type():
    """Test ``IntLiteral`` accepts a ``bool`` value and round-trips it."""
    literal = IntLiteral(value=True)

    restored = Expression.deserialize_from_dict(literal.serialize_to_dict())

    assert isinstance(restored, IntLiteral)
    assert restored.value is True


@pytest.mark.parametrize("value", [0, -1, 2**100, -(2**63), 2**63 - 1])
def test_int_literal_round_trips_extreme_values(value: int):
    """Test ``IntLiteral`` round-trips boundary, large, and negative integers."""
    literal = IntLiteral(value=value)

    restored = Expression.deserialize_from_dict(literal.serialize_to_dict())

    assert isinstance(restored, IntLiteral)
    assert restored.value == value


def test_negative_zero_float_is_not_equivalent_to_int_zero():
    """Test ``FloatLiteral(-0.0)`` is cross-class distinct from ``IntLiteral(0)``."""
    int_zero = IntLiteral(value=0)
    neg_zero = FloatLiteral(value=-0.0)

    assert not int_zero.is_structurally_equivalent(neg_zero)
    assert not neg_zero.is_structurally_equivalent(int_zero)


def test_complex_literal_treats_signed_zero_imag_as_equivalent():
    """Test ``ComplexLiteral`` equivalence follows Python's complex equality."""
    a = ComplexLiteral(value=complex(1.0, 0.0))
    b = ComplexLiteral(value=complex(1.0, -0.0))

    assert a.is_structurally_equivalent(b)


# ===========================================================================
# Empty-container round-trips
# ===========================================================================


def test_empty_tuple_expression_round_trip_preserves_emptiness():
    """Test a ``TupleExpression`` with no elements round-trips intact."""
    tuple_expression = TupleExpression()

    restored = Expression.deserialize_from_dict(tuple_expression.serialize_to_dict())

    assert isinstance(restored, TupleExpression)
    assert restored.expressions == ()


def test_empty_function_expression_round_trip_preserves_emptiness():
    """Test ``FunctionExpression`` with no template/indices/args round-trips intact."""
    function_expression = FunctionExpression(function=_make_identifier_expression("f"))

    restored = Expression.deserialize_from_dict(function_expression.serialize_to_dict())

    assert isinstance(restored, FunctionExpression)
    assert restored.template_types == ()
    assert restored.indices == ()
    assert restored.args == ()


def test_empty_array_access_expression_round_trip_preserves_emptiness():
    """Test an ``ArrayAccessExpression`` with no indices round-trips intact."""
    access = ArrayAccessExpression(array_expression=_make_identifier_expression("a"))

    restored = Expression.deserialize_from_dict(access.serialize_to_dict())

    assert isinstance(restored, ArrayAccessExpression)
    assert restored.indices == ()


# ===========================================================================
# Nesting and composition
# ===========================================================================


def test_deeply_nested_binary_expression_round_trips():
    """Test a five-level nested ``BinaryExpression`` round-trips intact."""
    node: Expression = _make_int_literal(1)
    for op, value in [
        (BinaryOperation.ADDITION, 2),
        (BinaryOperation.MULTIPLICATION, 3),
        (BinaryOperation.SUBTRACTION, 4),
        (BinaryOperation.POWER, 5),
        (BinaryOperation.FLOORDIV, 6),
    ]:
        node = BinaryExpression(operation=op, left=node, right=_make_int_literal(value))

    restored = Expression.deserialize_from_dict(node.serialize_to_dict())

    assert node.is_structurally_equivalent(restored)


def test_function_expression_callee_can_be_another_function_expression():
    """Test ``FunctionExpression`` accepts a ``FunctionExpression`` as its callee."""
    inner = FunctionExpression(function=_make_identifier_expression("f"))
    outer = FunctionExpression(function=inner, args=(_make_int_literal(1),))

    restored = Expression.deserialize_from_dict(outer.serialize_to_dict())

    assert isinstance(restored, FunctionExpression)
    assert isinstance(restored.function, FunctionExpression)


def test_tuple_access_chained_on_array_access_round_trips():
    """Test ``TupleAccessExpression`` over ``ArrayAccessExpression`` round-trips."""
    array = ArrayAccessExpression(
        array_expression=_make_identifier_expression("a"),
        indices=(_make_int_literal(0),),
    )
    access = TupleAccessExpression(
        tuple_expression=array, element_index=_make_int_literal(0)
    )

    restored = Expression.deserialize_from_dict(access.serialize_to_dict())

    assert isinstance(restored, TupleAccessExpression)
    assert isinstance(restored.tuple_expression, ArrayAccessExpression)


# ===========================================================================
# Visit-children ordering with multiple elements per slot
# ===========================================================================


def test_function_expression_visit_children_orders_function_indices_then_args_multi():
    """Test ``FunctionExpression`` walks function, then all indices, then all args."""
    fn = _make_identifier_expression("f")
    idx_a = _make_int_literal(0)
    idx_b = _make_int_literal(1)
    arg_x = _make_int_literal(10)
    arg_y = _make_int_literal(20)
    function_expression = FunctionExpression(
        function=fn, indices=(idx_a, idx_b), args=(arg_x, arg_y)
    )

    assert tuple(function_expression.get_visit_children()) == (
        fn,
        idx_a,
        idx_b,
        arg_x,
        arg_y,
    )


# ===========================================================================
# Exhaustive enum round-trips
# ===========================================================================


@pytest.mark.parametrize("operation", list(UnaryOperation))
def test_unary_expression_round_trips_for_every_operation(operation: UnaryOperation):
    """Test every ``UnaryOperation`` value survives a wrapped round-trip."""
    expression = UnaryExpression(operation=operation, expression=_make_int_literal(1))

    restored = Expression.deserialize_from_dict(expression.serialize_to_dict())

    assert isinstance(restored, UnaryExpression)
    assert restored.operation is operation


@pytest.mark.parametrize("operation", list(BinaryOperation))
def test_binary_expression_round_trips_for_every_operation(
    operation: BinaryOperation,
):
    """Test every ``BinaryOperation`` value survives a wrapped round-trip."""
    expression = BinaryExpression(
        operation=operation,
        left=_make_int_literal(1),
        right=_make_int_literal(2),
    )

    restored = Expression.deserialize_from_dict(expression.serialize_to_dict())

    assert isinstance(restored, BinaryExpression)
    assert restored.operation is operation
