"""Tests for ``fhy_lang.ast.node.base``.

Pins the ``Node`` family contract: abstractness, frozen-dataclass behavior,
and provenance round-trip via the simplest concrete ``Node`` subclass
(``Argument``). Also pins cross-cutting ``Node`` contracts that hold across
every concrete subclass (self-equivalence, symmetry, non-``Node`` rejection,
frozen attribute assignment).
"""

import dataclasses
from dataclasses import FrozenInstanceError

import pytest
from fhy_core import (
    CoreDataType,
    DeserializationDictStructureError,
    FrozenMutationError,
    Identifier,
    NumericalType,
    PrimitiveDataType,
    Provenance,
    TypeQualifier,
)
from fhy_core.provenance import NamedProvenance
from fhy_core.serialization import SerializationError

from fhy_lang.ast.node import (
    Argument,
    ArrayAccessExpression,
    BinaryExpression,
    BinaryOperation,
    ComplexLiteral,
    DeclarationStatement,
    ExpressionStatement,
    FloatLiteral,
    ForAllStatement,
    FunctionExpression,
    IdentifierExpression,
    Import,
    IntLiteral,
    Module,
    Native,
    Node,
    Operation,
    Procedure,
    QualifiedType,
    ReturnStatement,
    SelectionStatement,
    TernaryExpression,
    TupleAccessExpression,
    TupleExpression,
    UnaryExpression,
    UnaryOperation,
)


def _named_provenance(name: str) -> Provenance:
    """Return a ``NamedProvenance`` distinct from ``Provenance.unknown()``."""
    return NamedProvenance(child=Provenance.unknown(), name=name)


def _build_argument(provenance: Provenance | None = None) -> Argument:
    qualified_type = QualifiedType(
        base_type=NumericalType(PrimitiveDataType(CoreDataType.INT32)),
        type_qualifier=TypeQualifier.INPUT,
    )
    if provenance is None:
        return Argument(name=Identifier("x"), qualified_type=qualified_type)
    return Argument(
        name=Identifier("x"),
        qualified_type=qualified_type,
        provenance=provenance,
    )


def test_node_is_abstract_and_cannot_be_instantiated():
    """Test ``Node`` raises ``TypeError`` when instantiated directly."""
    with pytest.raises(TypeError):
        Node()  # type: ignore[abstract]


def test_argument_default_provenance_is_unknown():
    """Test default provenance on a concrete ``Node`` is ``UnknownProvenance``."""
    argument = _build_argument()

    assert argument.get_provenance() == Provenance.unknown()


def test_argument_round_trips_through_serialization_with_default_provenance():
    """Test serialize/deserialize is identity for a default-provenance node."""
    argument = _build_argument()

    restored = Argument.deserialize_data_from_dict(argument.serialize_data_to_dict())

    assert argument.is_structurally_equivalent(restored)
    assert restored.get_provenance() == Provenance.unknown()


def test_argument_round_trips_through_serialization_with_named_provenance():
    """Test non-default provenance survives a serialize/deserialize round-trip."""
    provenance = Provenance.fuse(Provenance.unknown(), metadata="audit-test")
    argument = _build_argument(provenance=provenance)

    restored = Argument.deserialize_data_from_dict(argument.serialize_data_to_dict())

    assert restored.get_provenance() == provenance


def test_argument_round_trips_through_wrapped_envelope():
    """Test ``Node.deserialize_from_dict`` reconstructs from a wrapped payload."""
    argument = _build_argument()
    wrapped = argument.serialize_to_dict()

    restored = Node.deserialize_from_dict(wrapped)

    assert isinstance(restored, Argument)
    assert argument.is_structurally_equivalent(restored)


def test_argument_deserialize_data_rejects_payload_missing_provenance():
    """Test missing ``provenance`` key raises ``DeserializationDictStructureError``."""
    payload = _build_argument().serialize_data_to_dict()
    payload.pop("provenance")

    with pytest.raises(DeserializationDictStructureError):
        Argument.deserialize_data_from_dict(payload)


def test_argument_deserialize_data_rejects_non_dict_provenance():
    """Test non-dict ``provenance`` raises ``DeserializationDictStructureError``."""
    payload = _build_argument().serialize_data_to_dict()
    payload["provenance"] = "not-a-dict"

    with pytest.raises(DeserializationDictStructureError):
        Argument.deserialize_data_from_dict(payload)


def test_argument_is_frozen_against_attribute_assignment():
    """Test assigning to a field on a frozen ``Argument`` raises."""
    argument = _build_argument()

    with pytest.raises((FrozenInstanceError, FrozenMutationError)):
        argument.name = Identifier("y")  # type: ignore[misc]


def test_node_deserialize_from_dict_rejects_unwrapped_payload():
    """Test ``Node.deserialize_from_dict`` requires a wrapped envelope."""
    bare = _build_argument().serialize_data_to_dict()

    with pytest.raises(SerializationError):
        Node.deserialize_from_dict(bare)


# ===========================================================================
# Provenance: constructor preservation, serialization payload, equivalence,
# and per-subnode round-trip
# ===========================================================================


def test_node_get_provenance_returns_constructor_value():
    """Test ``get_provenance`` returns exactly the provenance passed at construction."""
    provenance = _named_provenance("constructor-value")
    argument = _build_argument(provenance=provenance)

    assert argument.get_provenance() == provenance


def test_node_serialize_data_to_dict_includes_provenance_key():
    """Test the serialized payload always carries a ``provenance`` entry."""
    payload = _build_argument().serialize_data_to_dict()

    assert "provenance" in payload
    assert payload["provenance"] == Provenance.unknown().serialize_to_dict()


def test_node_equivalence_ignores_provenance():
    """Test structural equivalence holds across nodes that differ only in provenance."""
    name = Identifier("x")
    qualified_type = QualifiedType(
        base_type=NumericalType(PrimitiveDataType(CoreDataType.INT32)),
        type_qualifier=TypeQualifier.INPUT,
    )
    a = Argument(name=name, qualified_type=qualified_type)
    b = Argument(
        name=name,
        qualified_type=qualified_type,
        provenance=_named_provenance("other"),
    )

    assert a.get_provenance() != b.get_provenance()
    assert a.is_structurally_equivalent(b)


def test_distinct_provenance_per_subnode_survives_round_trip():
    """Test each sub-node retains its own provenance through a wrapped round-trip."""
    outer_prov = _named_provenance("outer")
    left_prov = _named_provenance("left")
    right_prov = _named_provenance("right")
    node = BinaryExpression(
        operation=BinaryOperation.ADDITION,
        left=IntLiteral(value=1, provenance=left_prov),
        right=IntLiteral(value=2, provenance=right_prov),
        provenance=outer_prov,
    )

    restored = Node.deserialize_from_dict(node.serialize_to_dict())

    assert isinstance(restored, BinaryExpression)
    assert restored.get_provenance() == outer_prov
    assert restored.left.get_provenance() == left_prov
    assert restored.right.get_provenance() == right_prov


# ===========================================================================
# Cross-cutting Node contracts: self-equivalence, non-Node rejection, symmetry,
# frozen attribute assignment across every concrete subclass
# ===========================================================================


def _representative_nodes() -> list[Node]:
    qualified_type = QualifiedType(
        base_type=NumericalType(PrimitiveDataType(CoreDataType.INT32)),
        type_qualifier=TypeQualifier.INPUT,
    )
    return [
        IntLiteral(value=1),
        FloatLiteral(value=1.0),
        ComplexLiteral(value=complex(1, 0)),
        IdentifierExpression(identifier=Identifier("x")),
        UnaryExpression(
            operation=UnaryOperation.NEGATION, expression=IntLiteral(value=1)
        ),
        BinaryExpression(
            operation=BinaryOperation.ADDITION,
            left=IntLiteral(value=1),
            right=IntLiteral(value=2),
        ),
        TernaryExpression(
            condition=IdentifierExpression(identifier=Identifier("c")),
            true=IntLiteral(value=1),
            false=IntLiteral(value=0),
        ),
        TupleExpression(expressions=(IntLiteral(value=1),)),
        TupleAccessExpression(
            tuple_expression=IdentifierExpression(identifier=Identifier("t")),
            element_index=IntLiteral(value=0),
        ),
        ArrayAccessExpression(
            array_expression=IdentifierExpression(identifier=Identifier("a"))
        ),
        FunctionExpression(function=IdentifierExpression(identifier=Identifier("f"))),
        Module(),
        Import(name=Identifier("foo")),
        Argument(name=Identifier("x"), qualified_type=qualified_type),
        Procedure(name=Identifier("p")),
        Operation(name=Identifier("op"), return_type=qualified_type),
        Native(name=Identifier("n")),
        DeclarationStatement(
            variable_name=Identifier("x"), variable_type=qualified_type
        ),
        ExpressionStatement(right=IntLiteral(value=1)),
        ForAllStatement(index=IdentifierExpression(identifier=Identifier("i"))),
        SelectionStatement(condition=IdentifierExpression(identifier=Identifier("c"))),
        ReturnStatement(expression=IntLiteral(value=1)),
        QualifiedType(
            base_type=NumericalType(PrimitiveDataType(CoreDataType.INT32)),
            type_qualifier=TypeQualifier.INPUT,
        ),
    ]


@pytest.mark.parametrize("node", _representative_nodes())
def test_node_is_structurally_equivalent_to_itself(node: Node):
    """Test every concrete AST node is structurally equivalent to itself."""
    assert node.is_structurally_equivalent(node)


@pytest.mark.parametrize("non_node", [None, "string", 42, 3.14, [], {}, object()])
def test_node_equivalence_rejects_non_node_objects(non_node: object):
    """Test ``is_structurally_equivalent`` returns False for any non-``Node``."""
    assert not IntLiteral(value=1).is_structurally_equivalent(non_node)


_SYMMETRY_QT = QualifiedType(
    base_type=NumericalType(PrimitiveDataType(CoreDataType.INT32)),
    type_qualifier=TypeQualifier.INPUT,
)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (IntLiteral(value=1), IntLiteral(value=2)),
        (IntLiteral(value=1), FloatLiteral(value=1.0)),
        (
            BinaryExpression(
                operation=BinaryOperation.ADDITION,
                left=IntLiteral(value=1),
                right=IntLiteral(value=2),
            ),
            UnaryExpression(
                operation=UnaryOperation.NEGATION, expression=IntLiteral(value=1)
            ),
        ),
        (
            Procedure(name=Identifier("p")),
            Operation(name=Identifier("p"), return_type=_SYMMETRY_QT),
        ),
        (Procedure(name=Identifier("p")), Native(name=Identifier("p"))),
        (Import(name=Identifier("a")), Import(name=Identifier("b"))),
        (Module(name=Identifier("a")), Module(name=Identifier("b"))),
        (
            ReturnStatement(expression=IntLiteral(value=1)),
            ExpressionStatement(right=IntLiteral(value=1)),
        ),
    ],
)
def test_equivalence_is_symmetric(a: Node, b: Node):
    """Test ``a.is_structurally_equivalent(b)`` matches the reverse direction."""
    assert a.is_structurally_equivalent(b) == b.is_structurally_equivalent(a)


@pytest.mark.parametrize(
    "node",
    [
        IntLiteral(value=1),
        FloatLiteral(value=1.0),
        BinaryExpression(
            operation=BinaryOperation.ADDITION,
            left=IntLiteral(value=1),
            right=IntLiteral(value=2),
        ),
        Module(),
        Import(name=Identifier("foo")),
        ReturnStatement(expression=IntLiteral(value=1)),
        QualifiedType(
            base_type=NumericalType(PrimitiveDataType(CoreDataType.INT32)),
            type_qualifier=TypeQualifier.INPUT,
        ),
    ],
)
def test_concrete_node_rejects_attribute_assignment(node: Node):
    """Test attribute assignment on a frozen concrete ``Node`` raises."""
    target = dataclasses.fields(node)[0].name
    with pytest.raises((FrozenInstanceError, FrozenMutationError)):
        setattr(node, target, getattr(node, target))
