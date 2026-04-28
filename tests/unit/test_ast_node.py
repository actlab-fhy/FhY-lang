"""Tests for AST node construction, accessors, equivalence, and serialization."""

import dataclasses
import math

import pytest
from fhy_core import (
    CoreDataType,
    DeserializationDictStructureError,
    DeserializationValueError,
    Identifier,
    Note,
    NumericalType,
    PrimitiveDataType,
    Provenance,
    TemplateDataType,
    TypeQualifier,
)
from fhy_core.serialization import SerializationError, UnknownTypeIdError

from fhy_lang.ast import (
    Argument,
    ArrayAccessExpression,
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
    Node,
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

# =============================================================================
# Shared helpers and fixtures
# =============================================================================


@pytest.fixture
def other_provenance() -> Provenance:
    """Return a provenance distinct from ``Provenance.unknown()``."""
    return Provenance.unknown().add_note(Note("different"))


@pytest.fixture
def int32_type() -> NumericalType:
    """Return a reusable ``int32`` numerical type."""
    return NumericalType(PrimitiveDataType(CoreDataType.INT32))


@pytest.fixture
def int32_qualified_type(int32_type: NumericalType) -> QualifiedType:
    """Return a reusable ``input int32`` qualified type."""
    return QualifiedType(base_type=int32_type, type_qualifier=TypeQualifier.INPUT)


def _roundtrip(node: Node) -> Node:
    return Node.deserialize_from_dict(node.serialize_to_dict())


# =============================================================================
# Node accessors and serialization
# =============================================================================


def test_node_get_provenance_returns_constructor_value(
    other_provenance: Provenance,
) -> None:
    """Test get_provenance returns the provenance supplied at construction."""
    node = IntLiteral(value=1, provenance=other_provenance)
    assert node.get_provenance() == other_provenance


def test_node_serialize_includes_provenance() -> None:
    """Test serialize_to_dict embeds the node provenance under 'provenance'."""
    node = IntLiteral(value=1)
    payload = node.serialize_to_dict()
    assert "provenance" in payload["__data__"]
    assert payload["__data__"]["provenance"] == Provenance.unknown().serialize_to_dict()


def test_node_roundtrip_preserves_provenance(other_provenance: Provenance) -> None:
    """Test deserialization restores a node's provenance from its serialized form."""
    node = IntLiteral(value=3, provenance=other_provenance)
    restored = _roundtrip(node)
    assert restored.get_provenance() == other_provenance


def test_node_equivalence_ignores_provenance(
    other_provenance: Provenance,
) -> None:
    """Test structural equivalence holds across nodes with differing provenance."""
    a = IntLiteral(value=5)
    b = IntLiteral(value=5, provenance=other_provenance)
    assert a.provenance != b.provenance
    assert a.is_structurally_equivalent(b)


# =============================================================================
# Module
# =============================================================================


def test_module_default_name_is_module_identifier() -> None:
    """Test a Module built without arguments has the default name 'module'."""
    module = Module()
    assert module.get_identifier().name_hint == "module"
    assert module.get_visit_children() == ()


def test_module_get_identifier_returns_constructor_name() -> None:
    """Test Module.get_identifier returns the identifier passed at construction."""
    name = Identifier("mymod")
    module = Module(name=name)
    assert module.get_identifier() is name


def test_module_get_visit_children_returns_statements() -> None:
    """Test Module.get_visit_children returns the module statement tuple."""
    statement = Import(name=Identifier("a"))
    module = Module(statements=(statement,))
    assert module.get_visit_children() == (statement,)


def test_module_roundtrip_preserves_structure() -> None:
    """Test a populated Module survives a serialize/deserialize round-trip."""
    statement = ReturnStatement(expression=IntLiteral(value=7))
    module = Module(name=Identifier("m"), statements=(statement,))
    restored = _roundtrip(module)
    assert isinstance(restored, Module)
    assert module.is_structurally_equivalent(restored)


def test_module_equivalence_rejects_non_module() -> None:
    """Test Module equivalence returns False against an unrelated node type."""
    module = Module()
    assert not module.is_structurally_equivalent(IntLiteral(value=0))


def test_module_equivalence_rejects_different_name() -> None:
    """Test Module equivalence returns False when the module names differ."""
    a = Module(name=Identifier("a"))
    b = Module(name=Identifier("b"))
    assert not a.is_structurally_equivalent(b)


def test_module_equivalence_rejects_different_statement_count() -> None:
    """Test Module equivalence returns False when statement counts differ."""
    a = Module(name=Identifier("m"))
    b = Module(
        name=Identifier("m"),
        statements=(Import(name=Identifier("x")),),
    )
    assert not a.is_structurally_equivalent(b)


def test_module_equivalence_rejects_different_statement_content() -> None:
    """Test Module equivalence returns False when statement contents differ."""
    a = Module(statements=(Import(name=Identifier("x")),))
    b = Module(statements=(Import(name=Identifier("y")),))
    assert not a.is_structurally_equivalent(b)


def test_module_deserialize_rejects_empty_dict() -> None:
    """Test Module.deserialize_data_from_dict rejects a dict missing required keys."""
    with pytest.raises(DeserializationDictStructureError):
        Module.deserialize_data_from_dict({})


# =============================================================================
# QualifiedType
# =============================================================================


def test_qualified_type_get_type_returns_base_type(int32_type: NumericalType) -> None:
    """Test QualifiedType.get_type returns the base type."""
    qt = QualifiedType(base_type=int32_type, type_qualifier=TypeQualifier.INPUT)
    assert qt.get_type() is int32_type


def test_qualified_type_roundtrip_preserves_structure(
    int32_type: NumericalType,
) -> None:
    """Test QualifiedType survives a serialize/deserialize round-trip."""
    qt = QualifiedType(base_type=int32_type, type_qualifier=TypeQualifier.OUTPUT)
    restored = _roundtrip(qt)
    assert isinstance(restored, QualifiedType)
    assert qt.is_structurally_equivalent(restored)


def test_qualified_type_equivalence_rejects_non_qualified_type() -> None:
    """Test QualifiedType equivalence returns False against an unrelated node."""
    qt = QualifiedType(
        base_type=NumericalType(PrimitiveDataType(CoreDataType.INT32)),
        type_qualifier=TypeQualifier.INPUT,
    )
    assert not qt.is_structurally_equivalent(IntLiteral(value=0))


def test_qualified_type_equivalence_rejects_different_base_type() -> None:
    """Test QualifiedType equivalence returns False when base types differ."""
    a = QualifiedType(
        base_type=NumericalType(PrimitiveDataType(CoreDataType.INT32)),
        type_qualifier=TypeQualifier.INPUT,
    )
    b = QualifiedType(
        base_type=NumericalType(PrimitiveDataType(CoreDataType.FLOAT32)),
        type_qualifier=TypeQualifier.INPUT,
    )
    assert not a.is_structurally_equivalent(b)


def test_qualified_type_equivalence_rejects_different_qualifier(
    int32_type: NumericalType,
) -> None:
    """Test QualifiedType equivalence returns False when qualifiers differ."""
    a = QualifiedType(base_type=int32_type, type_qualifier=TypeQualifier.INPUT)
    b = QualifiedType(base_type=int32_type, type_qualifier=TypeQualifier.OUTPUT)
    assert not a.is_structurally_equivalent(b)


def test_qualified_type_deserialize_rejects_empty_dict() -> None:
    """Test QualifiedType deserialization rejects a dict missing required keys."""
    with pytest.raises(DeserializationDictStructureError):
        QualifiedType.deserialize_data_from_dict({})


def test_qualified_type_deserialize_rejects_invalid_qualifier(
    int32_type: NumericalType,
) -> None:
    """Test QualifiedType deserialization rejects an unknown type qualifier value."""
    bad = {
        "provenance": Provenance.unknown().serialize_to_dict(),
        "base_type": int32_type.serialize_to_dict(),
        "type_qualifier": "not_a_real_qualifier",
    }
    with pytest.raises(DeserializationValueError):
        QualifiedType.deserialize_data_from_dict(bad)


# =============================================================================
# Import
# =============================================================================


def test_import_get_visit_children_is_empty() -> None:
    """Test Import has no visitable children."""
    node = Import(name=Identifier("foo"))
    assert node.get_visit_children() == ()


def test_import_roundtrip_preserves_structure() -> None:
    """Test Import survives a serialize/deserialize round-trip."""
    node = Import(name=Identifier("foo"))
    restored = _roundtrip(node)
    assert isinstance(restored, Import)
    assert node.is_structurally_equivalent(restored)


def test_import_equivalence_rejects_non_import() -> None:
    """Test Import equivalence returns False against an unrelated node."""
    node = Import(name=Identifier("foo"))
    assert not node.is_structurally_equivalent(IntLiteral(value=0))


def test_import_equivalence_rejects_different_name() -> None:
    """Test Import equivalence returns False when names differ."""
    a = Import(name=Identifier("foo"))
    b = Import(name=Identifier("bar"))
    assert not a.is_structurally_equivalent(b)


def test_import_deserialize_rejects_empty_dict() -> None:
    """Test Import deserialization rejects a dict missing required keys."""
    with pytest.raises(DeserializationDictStructureError):
        Import.deserialize_data_from_dict({})


# =============================================================================
# Argument
# =============================================================================


def test_argument_get_visit_children_returns_qualified_type(
    int32_qualified_type: QualifiedType,
) -> None:
    """Test Argument.get_visit_children returns its qualified type."""
    arg = Argument(name=Identifier("x"), qualified_type=int32_qualified_type)
    assert arg.get_visit_children() == (int32_qualified_type,)


def test_argument_roundtrip_preserves_structure(
    int32_qualified_type: QualifiedType,
) -> None:
    """Test Argument survives a serialize/deserialize round-trip."""
    arg = Argument(name=Identifier("x"), qualified_type=int32_qualified_type)
    restored = _roundtrip(arg)
    assert isinstance(restored, Argument)
    assert arg.is_structurally_equivalent(restored)


def test_argument_equivalence_rejects_non_argument(
    int32_qualified_type: QualifiedType,
) -> None:
    """Test Argument equivalence returns False against an unrelated node."""
    arg = Argument(name=Identifier("x"), qualified_type=int32_qualified_type)
    assert not arg.is_structurally_equivalent(IntLiteral(value=0))


def test_argument_equivalence_rejects_different_name(
    int32_qualified_type: QualifiedType,
) -> None:
    """Test Argument equivalence returns False when names differ."""
    a = Argument(name=Identifier("x"), qualified_type=int32_qualified_type)
    b = Argument(name=Identifier("y"), qualified_type=int32_qualified_type)
    assert not a.is_structurally_equivalent(b)


def test_argument_equivalence_rejects_different_qualified_type(
    int32_qualified_type: QualifiedType, int32_type: NumericalType
) -> None:
    """Test Argument equivalence returns False when qualified types differ."""
    other = QualifiedType(base_type=int32_type, type_qualifier=TypeQualifier.OUTPUT)
    a = Argument(name=Identifier("x"), qualified_type=int32_qualified_type)
    b = Argument(name=Identifier("x"), qualified_type=other)
    assert not a.is_structurally_equivalent(b)


def test_argument_deserialize_rejects_empty_dict() -> None:
    """Test Argument deserialization rejects a dict missing required keys."""
    with pytest.raises(DeserializationDictStructureError):
        Argument.deserialize_data_from_dict({})


# =============================================================================
# Procedure
# =============================================================================


def _make_procedure(
    *,
    name: Identifier | None = None,
    args: tuple[Argument, ...] = (),
    body: tuple[Statement, ...] = (),
    templates: tuple[TemplateDataType, ...] = (),
) -> Procedure:
    """Build a Procedure with sensible defaults for testing."""
    return Procedure(
        name=name if name is not None else Identifier("p"),
        templates=templates,
        args=args,
        body=body,
    )


def test_procedure_get_identifier_returns_function_name() -> None:
    """Test Procedure.get_identifier returns the function name."""
    name = Identifier("p")
    proc = _make_procedure(name=name)
    assert proc.get_identifier() is name


def test_procedure_get_visit_children_orders_args_then_body(
    int32_qualified_type: QualifiedType,
) -> None:
    """Test Procedure.get_visit_children yields args followed by body statements."""
    arg = Argument(name=Identifier("x"), qualified_type=int32_qualified_type)
    stmt = ReturnStatement(expression=IntLiteral(value=0))
    proc = _make_procedure(args=(arg,), body=(stmt,))
    assert proc.get_visit_children() == (arg, stmt)


def test_procedure_roundtrip_preserves_structure(
    int32_qualified_type: QualifiedType,
) -> None:
    """Test Procedure survives a serialize/deserialize round-trip."""
    arg = Argument(name=Identifier("x"), qualified_type=int32_qualified_type)
    stmt = ReturnStatement(expression=IntLiteral(value=0))
    template = TemplateDataType(Identifier("T"))
    proc = _make_procedure(args=(arg,), body=(stmt,), templates=(template,))
    restored = _roundtrip(proc)
    assert isinstance(restored, Procedure)
    assert proc.is_structurally_equivalent(restored)


def test_procedure_equivalence_rejects_non_procedure() -> None:
    """Test Procedure equivalence returns False against an unrelated node."""
    proc = _make_procedure()
    assert not proc.is_structurally_equivalent(IntLiteral(value=0))


def test_procedure_equivalence_rejects_different_name() -> None:
    """Test Procedure equivalence returns False when function names differ."""
    a = _make_procedure(name=Identifier("p"))
    b = _make_procedure(name=Identifier("q"))
    assert not a.is_structurally_equivalent(b)


def test_procedure_equivalence_rejects_different_template_count() -> None:
    """Test Procedure equivalence returns False when template counts differ."""
    a = _make_procedure(templates=(TemplateDataType(Identifier("T")),))
    b = _make_procedure()
    assert not a.is_structurally_equivalent(b)


def test_procedure_equivalence_rejects_different_template_content() -> None:
    """Test Procedure equivalence returns False when template contents differ."""
    a = _make_procedure(templates=(TemplateDataType(Identifier("T")),))
    b = _make_procedure(templates=(TemplateDataType(Identifier("U")),))
    assert not a.is_structurally_equivalent(b)


def test_procedure_equivalence_rejects_different_arg_count(
    int32_qualified_type: QualifiedType,
) -> None:
    """Test Procedure equivalence returns False when argument counts differ."""
    arg = Argument(name=Identifier("x"), qualified_type=int32_qualified_type)
    a = _make_procedure(args=(arg,))
    b = _make_procedure()
    assert not a.is_structurally_equivalent(b)


def test_procedure_equivalence_rejects_different_arg_content(
    int32_qualified_type: QualifiedType,
) -> None:
    """Test Procedure equivalence returns False when argument contents differ."""
    arg_a = Argument(name=Identifier("x"), qualified_type=int32_qualified_type)
    arg_b = Argument(name=Identifier("y"), qualified_type=int32_qualified_type)
    a = _make_procedure(args=(arg_a,))
    b = _make_procedure(args=(arg_b,))
    assert not a.is_structurally_equivalent(b)


def test_procedure_equivalence_rejects_different_body_count() -> None:
    """Test Procedure equivalence returns False when body lengths differ."""
    a = _make_procedure(body=(ReturnStatement(expression=IntLiteral(value=0)),))
    b = _make_procedure()
    assert not a.is_structurally_equivalent(b)


def test_procedure_equivalence_rejects_different_body_content() -> None:
    """Test Procedure equivalence returns False when body contents differ."""
    a = _make_procedure(body=(ReturnStatement(expression=IntLiteral(value=0)),))
    b = _make_procedure(body=(ReturnStatement(expression=IntLiteral(value=1)),))
    assert not a.is_structurally_equivalent(b)


def test_procedure_deserialize_rejects_empty_dict() -> None:
    """Test Procedure deserialization rejects a dict missing required keys."""
    with pytest.raises(DeserializationDictStructureError):
        Procedure.deserialize_data_from_dict({})


# =============================================================================
# Operation
# =============================================================================


def _make_operation(
    *,
    name: Identifier | None = None,
    args: tuple[Argument, ...] = (),
    body: tuple[Statement, ...] = (),
    templates: tuple[TemplateDataType, ...] = (),
    return_type: QualifiedType,
) -> Operation:
    """Build an Operation with sensible defaults for testing."""
    return Operation(
        name=name if name is not None else Identifier("op"),
        templates=templates,
        args=args,
        body=body,
        return_type=return_type,
    )


def test_operation_get_visit_children_orders_args_body_return(
    int32_qualified_type: QualifiedType,
) -> None:
    """Test Operation.get_visit_children yields args, body, and return type in order."""
    arg = Argument(name=Identifier("x"), qualified_type=int32_qualified_type)
    stmt = ReturnStatement(expression=IntLiteral(value=0))
    op = _make_operation(args=(arg,), body=(stmt,), return_type=int32_qualified_type)
    assert op.get_visit_children() == (arg, stmt, int32_qualified_type)


def test_operation_roundtrip_preserves_structure(
    int32_qualified_type: QualifiedType,
) -> None:
    """Test Operation survives a serialize/deserialize round-trip."""
    arg = Argument(name=Identifier("x"), qualified_type=int32_qualified_type)
    stmt = ReturnStatement(expression=IntLiteral(value=0))
    template = TemplateDataType(Identifier("T"))
    op = _make_operation(
        args=(arg,),
        body=(stmt,),
        templates=(template,),
        return_type=int32_qualified_type,
    )
    restored = _roundtrip(op)
    assert isinstance(restored, Operation)
    assert op.is_structurally_equivalent(restored)


def test_operation_equivalence_rejects_non_operation(
    int32_qualified_type: QualifiedType,
) -> None:
    """Test Operation equivalence returns False against an unrelated node."""
    op = _make_operation(return_type=int32_qualified_type)
    assert not op.is_structurally_equivalent(IntLiteral(value=0))


def test_operation_equivalence_rejects_different_name(
    int32_qualified_type: QualifiedType,
) -> None:
    """Test Operation equivalence returns False when function names differ."""
    a = _make_operation(name=Identifier("p"), return_type=int32_qualified_type)
    b = _make_operation(name=Identifier("q"), return_type=int32_qualified_type)
    assert not a.is_structurally_equivalent(b)


def test_operation_equivalence_rejects_different_template_count(
    int32_qualified_type: QualifiedType,
) -> None:
    """Test Operation equivalence returns False when template counts differ."""
    a = _make_operation(
        templates=(TemplateDataType(Identifier("T")),),
        return_type=int32_qualified_type,
    )
    b = _make_operation(return_type=int32_qualified_type)
    assert not a.is_structurally_equivalent(b)


def test_operation_equivalence_rejects_different_template_content(
    int32_qualified_type: QualifiedType,
) -> None:
    """Test Operation equivalence returns False when template contents differ."""
    a = _make_operation(
        templates=(TemplateDataType(Identifier("T")),),
        return_type=int32_qualified_type,
    )
    b = _make_operation(
        templates=(TemplateDataType(Identifier("U")),),
        return_type=int32_qualified_type,
    )
    assert not a.is_structurally_equivalent(b)


def test_operation_equivalence_rejects_different_arg_count(
    int32_qualified_type: QualifiedType,
) -> None:
    """Test Operation equivalence returns False when argument counts differ."""
    arg = Argument(name=Identifier("x"), qualified_type=int32_qualified_type)
    a = _make_operation(args=(arg,), return_type=int32_qualified_type)
    b = _make_operation(return_type=int32_qualified_type)
    assert not a.is_structurally_equivalent(b)


def test_operation_equivalence_rejects_different_arg_content(
    int32_qualified_type: QualifiedType,
) -> None:
    """Test Operation equivalence returns False when argument contents differ."""
    a = _make_operation(
        args=(Argument(name=Identifier("x"), qualified_type=int32_qualified_type),),
        return_type=int32_qualified_type,
    )
    b = _make_operation(
        args=(Argument(name=Identifier("y"), qualified_type=int32_qualified_type),),
        return_type=int32_qualified_type,
    )
    assert not a.is_structurally_equivalent(b)


def test_operation_equivalence_rejects_different_body_count(
    int32_qualified_type: QualifiedType,
) -> None:
    """Test Operation equivalence returns False when body lengths differ."""
    a = _make_operation(
        body=(ReturnStatement(expression=IntLiteral(value=0)),),
        return_type=int32_qualified_type,
    )
    b = _make_operation(return_type=int32_qualified_type)
    assert not a.is_structurally_equivalent(b)


def test_operation_equivalence_rejects_different_body_content(
    int32_qualified_type: QualifiedType,
) -> None:
    """Test Operation equivalence returns False when body contents differ."""
    a = _make_operation(
        body=(ReturnStatement(expression=IntLiteral(value=0)),),
        return_type=int32_qualified_type,
    )
    b = _make_operation(
        body=(ReturnStatement(expression=IntLiteral(value=1)),),
        return_type=int32_qualified_type,
    )
    assert not a.is_structurally_equivalent(b)


def test_operation_equivalence_rejects_different_return_type(
    int32_qualified_type: QualifiedType, int32_type: NumericalType
) -> None:
    """Test Operation equivalence returns False when return types differ."""
    other = QualifiedType(base_type=int32_type, type_qualifier=TypeQualifier.STATE)
    a = _make_operation(return_type=int32_qualified_type)
    b = _make_operation(return_type=other)
    assert not a.is_structurally_equivalent(b)


def test_operation_deserialize_rejects_empty_dict() -> None:
    """Test Operation deserialization rejects a dict missing required keys."""
    with pytest.raises(DeserializationDictStructureError):
        Operation.deserialize_data_from_dict({})


# =============================================================================
# Native
# =============================================================================


def test_native_get_visit_children_returns_args(
    int32_qualified_type: QualifiedType,
) -> None:
    """Test Native.get_visit_children returns its argument tuple."""
    arg = Argument(name=Identifier("x"), qualified_type=int32_qualified_type)
    native = Native(name=Identifier("n"), args=(arg,))
    assert native.get_visit_children() == (arg,)


def test_native_roundtrip_preserves_structure(
    int32_qualified_type: QualifiedType,
) -> None:
    """Test Native survives a serialize/deserialize round-trip."""
    arg = Argument(name=Identifier("x"), qualified_type=int32_qualified_type)
    native = Native(name=Identifier("n"), args=(arg,))
    restored = _roundtrip(native)
    assert isinstance(restored, Native)
    assert native.is_structurally_equivalent(restored)


def test_native_equivalence_rejects_non_native() -> None:
    """Test Native equivalence returns False against an unrelated node."""
    native = Native(name=Identifier("n"))
    assert not native.is_structurally_equivalent(IntLiteral(value=0))


def test_native_equivalence_rejects_different_name() -> None:
    """Test Native equivalence returns False when function names differ."""
    a = Native(name=Identifier("n"))
    b = Native(name=Identifier("m"))
    assert not a.is_structurally_equivalent(b)


def test_native_equivalence_rejects_different_arg_count(
    int32_qualified_type: QualifiedType,
) -> None:
    """Test Native equivalence returns False when argument counts differ."""
    arg = Argument(name=Identifier("x"), qualified_type=int32_qualified_type)
    a = Native(name=Identifier("n"), args=(arg,))
    b = Native(name=Identifier("n"))
    assert not a.is_structurally_equivalent(b)


def test_native_equivalence_rejects_different_arg_content(
    int32_qualified_type: QualifiedType,
) -> None:
    """Test Native equivalence returns False when argument contents differ."""
    a = Native(
        name=Identifier("n"),
        args=(Argument(name=Identifier("x"), qualified_type=int32_qualified_type),),
    )
    b = Native(
        name=Identifier("n"),
        args=(Argument(name=Identifier("y"), qualified_type=int32_qualified_type),),
    )
    assert not a.is_structurally_equivalent(b)


def test_native_deserialize_rejects_empty_dict() -> None:
    """Test Native deserialization rejects a dict missing required keys."""
    with pytest.raises(DeserializationDictStructureError):
        Native.deserialize_data_from_dict({})


# =============================================================================
# DeclarationStatement
# =============================================================================


def test_declaration_get_visit_children_omits_expression_when_none(
    int32_qualified_type: QualifiedType,
) -> None:
    """Test DeclarationStatement omits expression from children when it is None."""
    decl = DeclarationStatement(
        variable_name=Identifier("x"),
        variable_type=int32_qualified_type,
    )
    assert decl.get_visit_children() == (int32_qualified_type,)


def test_declaration_get_visit_children_includes_expression_when_set(
    int32_qualified_type: QualifiedType,
) -> None:
    """Test DeclarationStatement includes expression in children when supplied."""
    expr = IntLiteral(value=1)
    decl = DeclarationStatement(
        variable_name=Identifier("x"),
        variable_type=int32_qualified_type,
        expression=expr,
    )
    assert decl.get_visit_children() == (int32_qualified_type, expr)


def test_declaration_roundtrip_with_expression(
    int32_qualified_type: QualifiedType,
) -> None:
    """Test DeclarationStatement with an expression survives a round-trip."""
    decl = DeclarationStatement(
        variable_name=Identifier("x"),
        variable_type=int32_qualified_type,
        expression=IntLiteral(value=2),
    )
    restored = _roundtrip(decl)
    assert isinstance(restored, DeclarationStatement)
    assert decl.is_structurally_equivalent(restored)


def test_declaration_roundtrip_without_expression(
    int32_qualified_type: QualifiedType,
) -> None:
    """Test DeclarationStatement without an expression survives a round-trip."""
    decl = DeclarationStatement(
        variable_name=Identifier("x"),
        variable_type=int32_qualified_type,
    )
    restored = _roundtrip(decl)
    assert isinstance(restored, DeclarationStatement)
    assert decl.is_structurally_equivalent(restored)
    assert restored.expression is None


def test_declaration_equivalence_rejects_non_declaration(
    int32_qualified_type: QualifiedType,
) -> None:
    """Test DeclarationStatement equivalence returns False against an unrelated node."""
    decl = DeclarationStatement(
        variable_name=Identifier("x"),
        variable_type=int32_qualified_type,
    )
    assert not decl.is_structurally_equivalent(IntLiteral(value=0))


def test_declaration_equivalence_rejects_different_name(
    int32_qualified_type: QualifiedType,
) -> None:
    """Test DeclarationStatement equivalence returns False when names differ."""
    a = DeclarationStatement(
        variable_name=Identifier("x"),
        variable_type=int32_qualified_type,
    )
    b = DeclarationStatement(
        variable_name=Identifier("y"),
        variable_type=int32_qualified_type,
    )
    assert not a.is_structurally_equivalent(b)


def test_declaration_equivalence_rejects_different_type(
    int32_qualified_type: QualifiedType, int32_type: NumericalType
) -> None:
    """Test DeclarationStatement equivalence returns False when types differ."""
    other = QualifiedType(base_type=int32_type, type_qualifier=TypeQualifier.OUTPUT)
    a = DeclarationStatement(
        variable_name=Identifier("x"),
        variable_type=int32_qualified_type,
    )
    b = DeclarationStatement(
        variable_name=Identifier("x"),
        variable_type=other,
    )
    assert not a.is_structurally_equivalent(b)


def test_declaration_equivalence_rejects_expression_presence_mismatch(
    int32_qualified_type: QualifiedType,
) -> None:
    """Test DeclarationStatement equivalence rejects None vs supplied expression."""
    a = DeclarationStatement(
        variable_name=Identifier("x"),
        variable_type=int32_qualified_type,
    )
    b = DeclarationStatement(
        variable_name=Identifier("x"),
        variable_type=int32_qualified_type,
        expression=IntLiteral(value=1),
    )
    assert not a.is_structurally_equivalent(b)
    assert not b.is_structurally_equivalent(a)


def test_declaration_equivalence_rejects_different_expression(
    int32_qualified_type: QualifiedType,
) -> None:
    """Test DeclarationStatement equivalence returns False when expressions differ."""
    a = DeclarationStatement(
        variable_name=Identifier("x"),
        variable_type=int32_qualified_type,
        expression=IntLiteral(value=1),
    )
    b = DeclarationStatement(
        variable_name=Identifier("x"),
        variable_type=int32_qualified_type,
        expression=IntLiteral(value=2),
    )
    assert not a.is_structurally_equivalent(b)


def test_declaration_deserialize_rejects_empty_dict() -> None:
    """Test DeclarationStatement deserialization rejects a dict missing keys."""
    with pytest.raises(DeserializationDictStructureError):
        DeclarationStatement.deserialize_data_from_dict({})


# =============================================================================
# ExpressionStatement
# =============================================================================


def test_expression_statement_get_visit_children_omits_left_when_none() -> None:
    """Test ExpressionStatement omits left from children when it is None."""
    right = IntLiteral(value=1)
    stmt = ExpressionStatement(right=right)
    assert stmt.get_visit_children() == (right,)


def test_expression_statement_get_visit_children_includes_left_when_set() -> None:
    """Test ExpressionStatement yields left then right when left is supplied."""
    left = IdentifierExpression(identifier=Identifier("x"))
    right = IntLiteral(value=1)
    stmt = ExpressionStatement(left=left, right=right)
    assert stmt.get_visit_children() == (left, right)


def test_expression_statement_roundtrip_with_left() -> None:
    """Test ExpressionStatement with a left expression survives a round-trip."""
    stmt = ExpressionStatement(
        left=IdentifierExpression(identifier=Identifier("x")),
        right=IntLiteral(value=1),
    )
    restored = _roundtrip(stmt)
    assert isinstance(restored, ExpressionStatement)
    assert stmt.is_structurally_equivalent(restored)


def test_expression_statement_roundtrip_without_left() -> None:
    """Test ExpressionStatement without a left expression survives a round-trip."""
    stmt = ExpressionStatement(right=IntLiteral(value=1))
    restored = _roundtrip(stmt)
    assert isinstance(restored, ExpressionStatement)
    assert stmt.is_structurally_equivalent(restored)
    assert restored.left is None


def test_expression_statement_equivalence_rejects_non_statement() -> None:
    """Test ExpressionStatement equivalence returns False against an unrelated node."""
    stmt = ExpressionStatement(right=IntLiteral(value=1))
    assert not stmt.is_structurally_equivalent(IntLiteral(value=1))


def test_expression_statement_equivalence_rejects_left_presence_mismatch() -> None:
    """Test ExpressionStatement equivalence rejects None vs supplied left."""
    a = ExpressionStatement(right=IntLiteral(value=1))
    b = ExpressionStatement(
        left=IdentifierExpression(identifier=Identifier("x")),
        right=IntLiteral(value=1),
    )
    assert not a.is_structurally_equivalent(b)
    assert not b.is_structurally_equivalent(a)


def test_expression_statement_equivalence_rejects_different_left() -> None:
    """Test ExpressionStatement equivalence rejects different left expressions."""
    a = ExpressionStatement(
        left=IdentifierExpression(identifier=Identifier("x")),
        right=IntLiteral(value=1),
    )
    b = ExpressionStatement(
        left=IdentifierExpression(identifier=Identifier("y")),
        right=IntLiteral(value=1),
    )
    assert not a.is_structurally_equivalent(b)


def test_expression_statement_equivalence_rejects_different_right() -> None:
    """Test ExpressionStatement equivalence rejects different right expressions."""
    a = ExpressionStatement(right=IntLiteral(value=1))
    b = ExpressionStatement(right=IntLiteral(value=2))
    assert not a.is_structurally_equivalent(b)


def test_expression_statement_deserialize_rejects_empty_dict() -> None:
    """Test ExpressionStatement deserialization rejects a dict missing keys."""
    with pytest.raises(DeserializationDictStructureError):
        ExpressionStatement.deserialize_data_from_dict({})


# =============================================================================
# ForAllStatement
# =============================================================================


def test_for_all_default_name_is_forall_identifier() -> None:
    """Test ForAllStatement built without a name has the default 'forall'."""
    node = ForAllStatement(index=IdentifierExpression(identifier=Identifier("i")))
    assert node.get_identifier().name_hint == "forall"


def test_for_all_get_identifier_returns_constructor_name() -> None:
    """Test ForAllStatement.get_identifier returns the name passed at construction."""
    name = Identifier("loop")
    node = ForAllStatement(
        name=name, index=IdentifierExpression(identifier=Identifier("i"))
    )
    assert node.get_identifier() is name


def test_for_all_get_visit_children_orders_index_then_body() -> None:
    """Test ForAllStatement.get_visit_children yields the index then body statements."""
    index = IdentifierExpression(identifier=Identifier("i"))
    body_stmt = ReturnStatement(expression=IntLiteral(value=0))
    node = ForAllStatement(index=index, body=(body_stmt,))
    assert node.get_visit_children() == (index, body_stmt)


def test_for_all_roundtrip_preserves_structure() -> None:
    """Test ForAllStatement survives a serialize/deserialize round-trip."""
    index = IdentifierExpression(identifier=Identifier("i"))
    body_stmt = ReturnStatement(expression=IntLiteral(value=0))
    node = ForAllStatement(name=Identifier("loop"), index=index, body=(body_stmt,))
    restored = _roundtrip(node)
    assert isinstance(restored, ForAllStatement)
    assert node.is_structurally_equivalent(restored)


def test_for_all_equivalence_rejects_non_for_all() -> None:
    """Test ForAllStatement equivalence returns False against an unrelated node."""
    node = ForAllStatement(index=IdentifierExpression(identifier=Identifier("i")))
    assert not node.is_structurally_equivalent(IntLiteral(value=0))


def test_for_all_equivalence_rejects_different_name() -> None:
    """Test ForAllStatement equivalence returns False when names differ."""
    index = IdentifierExpression(identifier=Identifier("i"))
    a = ForAllStatement(name=Identifier("loop_a"), index=index)
    b = ForAllStatement(name=Identifier("loop_b"), index=index)
    assert not a.is_structurally_equivalent(b)


def test_for_all_equivalence_rejects_different_index() -> None:
    """Test ForAllStatement equivalence returns False when indices differ."""
    a = ForAllStatement(index=IdentifierExpression(identifier=Identifier("i")))
    b = ForAllStatement(index=IdentifierExpression(identifier=Identifier("j")))
    assert not a.is_structurally_equivalent(b)


def test_for_all_equivalence_rejects_different_body_count() -> None:
    """Test ForAllStatement equivalence returns False when body lengths differ."""
    index = IdentifierExpression(identifier=Identifier("i"))
    a = ForAllStatement(index=index)
    b = ForAllStatement(
        index=index, body=(ReturnStatement(expression=IntLiteral(value=0)),)
    )
    assert not a.is_structurally_equivalent(b)


def test_for_all_equivalence_rejects_different_body_content() -> None:
    """Test ForAllStatement equivalence returns False when body contents differ."""
    index = IdentifierExpression(identifier=Identifier("i"))
    a = ForAllStatement(
        index=index, body=(ReturnStatement(expression=IntLiteral(value=0)),)
    )
    b = ForAllStatement(
        index=index, body=(ReturnStatement(expression=IntLiteral(value=1)),)
    )
    assert not a.is_structurally_equivalent(b)


def test_for_all_deserialize_rejects_empty_dict() -> None:
    """Test ForAllStatement deserialization rejects a dict missing required keys."""
    with pytest.raises(DeserializationDictStructureError):
        ForAllStatement.deserialize_data_from_dict({})


# =============================================================================
# SelectionStatement
# =============================================================================


def test_selection_get_visit_children_orders_condition_true_false() -> None:
    """Test SelectionStatement orders condition, then true body, then false body."""
    cond = IdentifierExpression(identifier=Identifier("c"))
    t_stmt = ReturnStatement(expression=IntLiteral(value=1))
    f_stmt = ReturnStatement(expression=IntLiteral(value=0))
    node = SelectionStatement(condition=cond, true_body=(t_stmt,), false_body=(f_stmt,))
    assert node.get_visit_children() == (cond, t_stmt, f_stmt)


def test_selection_roundtrip_preserves_structure() -> None:
    """Test SelectionStatement survives a serialize/deserialize round-trip."""
    node = SelectionStatement(
        condition=IdentifierExpression(identifier=Identifier("c")),
        true_body=(ReturnStatement(expression=IntLiteral(value=1)),),
        false_body=(ReturnStatement(expression=IntLiteral(value=0)),),
    )
    restored = _roundtrip(node)
    assert isinstance(restored, SelectionStatement)
    assert node.is_structurally_equivalent(restored)


def test_selection_equivalence_rejects_non_selection() -> None:
    """Test SelectionStatement equivalence returns False against an unrelated node."""
    node = SelectionStatement(
        condition=IdentifierExpression(identifier=Identifier("c"))
    )
    assert not node.is_structurally_equivalent(IntLiteral(value=0))


def test_selection_equivalence_rejects_different_condition() -> None:
    """Test SelectionStatement equivalence returns False when conditions differ."""
    a = SelectionStatement(condition=IdentifierExpression(identifier=Identifier("c")))
    b = SelectionStatement(condition=IdentifierExpression(identifier=Identifier("d")))
    assert not a.is_structurally_equivalent(b)


def test_selection_equivalence_rejects_different_true_body_count() -> None:
    """Test SelectionStatement equivalence returns False when true bodies differ in length."""  # noqa: E501
    cond = IdentifierExpression(identifier=Identifier("c"))
    a = SelectionStatement(condition=cond)
    b = SelectionStatement(
        condition=cond,
        true_body=(ReturnStatement(expression=IntLiteral(value=1)),),
    )
    assert not a.is_structurally_equivalent(b)


def test_selection_equivalence_rejects_different_true_body_content() -> None:
    """Test SelectionStatement equivalence returns False when true body contents differ."""  # noqa: E501
    cond = IdentifierExpression(identifier=Identifier("c"))
    a = SelectionStatement(
        condition=cond,
        true_body=(ReturnStatement(expression=IntLiteral(value=1)),),
    )
    b = SelectionStatement(
        condition=cond,
        true_body=(ReturnStatement(expression=IntLiteral(value=2)),),
    )
    assert not a.is_structurally_equivalent(b)


def test_selection_equivalence_rejects_different_false_body_count() -> None:
    """Test SelectionStatement equivalence returns False when false bodies differ in length."""  # noqa: E501
    cond = IdentifierExpression(identifier=Identifier("c"))
    a = SelectionStatement(condition=cond)
    b = SelectionStatement(
        condition=cond,
        false_body=(ReturnStatement(expression=IntLiteral(value=0)),),
    )
    assert not a.is_structurally_equivalent(b)


def test_selection_equivalence_rejects_different_false_body_content() -> None:
    """Test SelectionStatement equivalence returns False when false body contents differ."""  # noqa: E501
    cond = IdentifierExpression(identifier=Identifier("c"))
    a = SelectionStatement(
        condition=cond,
        false_body=(ReturnStatement(expression=IntLiteral(value=0)),),
    )
    b = SelectionStatement(
        condition=cond,
        false_body=(ReturnStatement(expression=IntLiteral(value=1)),),
    )
    assert not a.is_structurally_equivalent(b)


def test_selection_deserialize_rejects_empty_dict() -> None:
    """Test SelectionStatement deserialization rejects a dict missing required keys."""
    with pytest.raises(DeserializationDictStructureError):
        SelectionStatement.deserialize_data_from_dict({})


# =============================================================================
# ReturnStatement
# =============================================================================


def test_return_get_visit_children_returns_expression() -> None:
    """Test ReturnStatement.get_visit_children yields the wrapped expression."""
    expr = IntLiteral(value=1)
    node = ReturnStatement(expression=expr)
    assert node.get_visit_children() == (expr,)


def test_return_roundtrip_preserves_structure() -> None:
    """Test ReturnStatement survives a serialize/deserialize round-trip."""
    node = ReturnStatement(expression=IntLiteral(value=1))
    restored = _roundtrip(node)
    assert isinstance(restored, ReturnStatement)
    assert node.is_structurally_equivalent(restored)


def test_return_equivalence_rejects_non_return() -> None:
    """Test ReturnStatement equivalence returns False against an unrelated node."""
    node = ReturnStatement(expression=IntLiteral(value=1))
    assert not node.is_structurally_equivalent(IntLiteral(value=1))


def test_return_equivalence_rejects_different_expression() -> None:
    """Test ReturnStatement equivalence returns False when expressions differ."""
    a = ReturnStatement(expression=IntLiteral(value=1))
    b = ReturnStatement(expression=IntLiteral(value=2))
    assert not a.is_structurally_equivalent(b)


def test_return_deserialize_rejects_empty_dict() -> None:
    """Test ReturnStatement deserialization rejects a dict missing required keys."""
    with pytest.raises(DeserializationDictStructureError):
        ReturnStatement.deserialize_data_from_dict({})


# =============================================================================
# UnaryExpression
# =============================================================================


def test_unary_get_operands_returns_single_expression() -> None:
    """Test UnaryExpression.get_operands returns a one-tuple with the operand."""
    inner = IntLiteral(value=1)
    node = UnaryExpression(operation=UnaryOperation.NEGATION, expression=inner)
    assert node.get_operands() == (inner,)
    assert node.get_visit_children() == (inner,)


def test_unary_roundtrip_preserves_structure() -> None:
    """Test UnaryExpression survives a serialize/deserialize round-trip."""
    node = UnaryExpression(
        operation=UnaryOperation.LOGICAL_NOT, expression=IntLiteral(value=1)
    )
    restored = _roundtrip(node)
    assert isinstance(restored, UnaryExpression)
    assert node.is_structurally_equivalent(restored)


def test_unary_equivalence_rejects_non_unary() -> None:
    """Test UnaryExpression equivalence returns False against an unrelated node."""
    node = UnaryExpression(
        operation=UnaryOperation.NEGATION, expression=IntLiteral(value=1)
    )
    assert not node.is_structurally_equivalent(IntLiteral(value=1))


def test_unary_equivalence_rejects_different_operation() -> None:
    """Test UnaryExpression equivalence returns False when operations differ."""
    a = UnaryExpression(
        operation=UnaryOperation.NEGATION, expression=IntLiteral(value=1)
    )
    b = UnaryExpression(
        operation=UnaryOperation.BITWISE_NOT, expression=IntLiteral(value=1)
    )
    assert not a.is_structurally_equivalent(b)


def test_unary_equivalence_rejects_different_expression() -> None:
    """Test UnaryExpression equivalence returns False when operands differ."""
    a = UnaryExpression(
        operation=UnaryOperation.NEGATION, expression=IntLiteral(value=1)
    )
    b = UnaryExpression(
        operation=UnaryOperation.NEGATION, expression=IntLiteral(value=2)
    )
    assert not a.is_structurally_equivalent(b)


def test_unary_deserialize_rejects_empty_dict() -> None:
    """Test UnaryExpression deserialization rejects a dict missing required keys."""
    with pytest.raises(DeserializationDictStructureError):
        UnaryExpression.deserialize_data_from_dict({})


def test_unary_deserialize_rejects_unknown_operation() -> None:
    """Test UnaryExpression deserialization rejects an unknown operation value."""
    bad = {
        "provenance": Provenance.unknown().serialize_to_dict(),
        "operation": "@@@",
        "expression": IntLiteral(value=1).serialize_to_dict(),
    }
    with pytest.raises(DeserializationValueError):
        UnaryExpression.deserialize_data_from_dict(bad)


# =============================================================================
# BinaryExpression
# =============================================================================


def test_binary_get_operands_returns_left_and_right() -> None:
    """Test BinaryExpression.get_operands returns the (left, right) operand pair."""
    left = IntLiteral(value=1)
    right = IntLiteral(value=2)
    node = BinaryExpression(operation=BinaryOperation.ADDITION, left=left, right=right)
    assert node.get_operands() == (left, right)
    assert node.get_visit_children() == (left, right)


def test_binary_roundtrip_preserves_structure() -> None:
    """Test BinaryExpression survives a serialize/deserialize round-trip."""
    node = BinaryExpression(
        operation=BinaryOperation.MULTIPLICATION,
        left=IntLiteral(value=1),
        right=IntLiteral(value=2),
    )
    restored = _roundtrip(node)
    assert isinstance(restored, BinaryExpression)
    assert node.is_structurally_equivalent(restored)


def test_binary_equivalence_rejects_non_binary() -> None:
    """Test BinaryExpression equivalence returns False against an unrelated node."""
    node = BinaryExpression(
        operation=BinaryOperation.ADDITION,
        left=IntLiteral(value=1),
        right=IntLiteral(value=2),
    )
    assert not node.is_structurally_equivalent(IntLiteral(value=1))


def test_binary_equivalence_rejects_different_left() -> None:
    """Test BinaryExpression equivalence returns False when left operands differ."""
    a = BinaryExpression(
        operation=BinaryOperation.ADDITION,
        left=IntLiteral(value=1),
        right=IntLiteral(value=2),
    )
    b = BinaryExpression(
        operation=BinaryOperation.ADDITION,
        left=IntLiteral(value=99),
        right=IntLiteral(value=2),
    )
    assert not a.is_structurally_equivalent(b)


def test_binary_equivalence_rejects_different_right() -> None:
    """Test BinaryExpression equivalence returns False when right operands differ."""
    a = BinaryExpression(
        operation=BinaryOperation.ADDITION,
        left=IntLiteral(value=1),
        right=IntLiteral(value=2),
    )
    b = BinaryExpression(
        operation=BinaryOperation.ADDITION,
        left=IntLiteral(value=1),
        right=IntLiteral(value=99),
    )
    assert not a.is_structurally_equivalent(b)


def test_binary_deserialize_rejects_empty_dict() -> None:
    """Test BinaryExpression deserialization rejects a dict missing required keys."""
    with pytest.raises(DeserializationDictStructureError):
        BinaryExpression.deserialize_data_from_dict({})


def test_binary_deserialize_rejects_unknown_operation() -> None:
    """Test BinaryExpression deserialization rejects an unknown operation value."""
    bad = {
        "provenance": Provenance.unknown().serialize_to_dict(),
        "operation": "@@@",
        "left": IntLiteral(value=1).serialize_to_dict(),
        "right": IntLiteral(value=2).serialize_to_dict(),
    }
    with pytest.raises(DeserializationValueError):
        BinaryExpression.deserialize_data_from_dict(bad)


# =============================================================================
# TernaryExpression
# =============================================================================


def test_ternary_get_visit_children_orders_condition_true_false() -> None:
    """Test TernaryExpression yields children in (condition, true, false) order."""
    cond = IdentifierExpression(identifier=Identifier("c"))
    true_e = IntLiteral(value=1)
    false_e = IntLiteral(value=0)
    node = TernaryExpression(condition=cond, true=true_e, false=false_e)
    assert node.get_visit_children() == (cond, true_e, false_e)


def test_ternary_roundtrip_preserves_structure() -> None:
    """Test TernaryExpression survives a serialize/deserialize round-trip."""
    node = TernaryExpression(
        condition=IdentifierExpression(identifier=Identifier("c")),
        true=IntLiteral(value=1),
        false=IntLiteral(value=0),
    )
    restored = _roundtrip(node)
    assert isinstance(restored, TernaryExpression)
    assert node.is_structurally_equivalent(restored)


def test_ternary_equivalence_rejects_non_ternary() -> None:
    """Test TernaryExpression equivalence returns False against an unrelated node."""
    node = TernaryExpression(
        condition=IdentifierExpression(identifier=Identifier("c")),
        true=IntLiteral(value=1),
        false=IntLiteral(value=0),
    )
    assert not node.is_structurally_equivalent(IntLiteral(value=1))


@pytest.mark.parametrize(
    "field_name",
    ["condition", "true", "false"],
)
def test_ternary_equivalence_rejects_field_change(field_name: str) -> None:
    """Test TernaryExpression equivalence returns False when one field changes."""
    base = {
        "condition": IdentifierExpression(identifier=Identifier("c")),
        "true": IntLiteral(value=1),
        "false": IntLiteral(value=0),
    }
    changed = dict(base)
    if field_name == "condition":
        changed[field_name] = IdentifierExpression(identifier=Identifier("d"))
    else:
        changed[field_name] = IntLiteral(value=999)
    a = TernaryExpression(**base)
    b = TernaryExpression(**changed)
    assert not a.is_structurally_equivalent(b)


def test_ternary_deserialize_rejects_empty_dict() -> None:
    """Test TernaryExpression deserialization rejects a dict missing required keys."""
    with pytest.raises(DeserializationDictStructureError):
        TernaryExpression.deserialize_data_from_dict({})


# =============================================================================
# TupleAccessExpression
# =============================================================================


def test_tuple_access_get_operands_returns_tuple_and_index() -> None:
    """Test TupleAccessExpression.get_operands returns the tuple expr and index."""
    tup = IdentifierExpression(identifier=Identifier("t"))
    idx = IntLiteral(value=0)
    node = TupleAccessExpression(tuple_expression=tup, element_index=idx)
    assert node.get_operands() == (tup, idx)
    assert node.get_visit_children() == (tup, idx)


def test_tuple_access_roundtrip_preserves_structure() -> None:
    """Test TupleAccessExpression survives a serialize/deserialize round-trip."""
    node = TupleAccessExpression(
        tuple_expression=IdentifierExpression(identifier=Identifier("t")),
        element_index=IntLiteral(value=2),
    )
    restored = _roundtrip(node)
    assert isinstance(restored, TupleAccessExpression)
    assert node.is_structurally_equivalent(restored)


def test_tuple_access_equivalence_rejects_non_tuple_access() -> None:
    """Test TupleAccessExpression equivalence rejects an unrelated node type."""
    node = TupleAccessExpression(
        tuple_expression=IdentifierExpression(identifier=Identifier("t")),
        element_index=IntLiteral(value=0),
    )
    assert not node.is_structurally_equivalent(IntLiteral(value=0))


def test_tuple_access_equivalence_rejects_different_tuple_expression() -> None:
    """Test TupleAccessExpression equivalence returns False when tuple expressions differ."""  # noqa: E501
    a = TupleAccessExpression(
        tuple_expression=IdentifierExpression(identifier=Identifier("t")),
        element_index=IntLiteral(value=0),
    )
    b = TupleAccessExpression(
        tuple_expression=IdentifierExpression(identifier=Identifier("u")),
        element_index=IntLiteral(value=0),
    )
    assert not a.is_structurally_equivalent(b)


def test_tuple_access_equivalence_rejects_different_index() -> None:
    """Test TupleAccessExpression equivalence returns False when indices differ."""
    a = TupleAccessExpression(
        tuple_expression=IdentifierExpression(identifier=Identifier("t")),
        element_index=IntLiteral(value=0),
    )
    b = TupleAccessExpression(
        tuple_expression=IdentifierExpression(identifier=Identifier("t")),
        element_index=IntLiteral(value=1),
    )
    assert not a.is_structurally_equivalent(b)


def test_tuple_access_deserialize_rejects_empty_dict() -> None:
    """Test TupleAccessExpression deserialization rejects a dict missing required keys."""  # noqa: E501
    with pytest.raises(DeserializationDictStructureError):
        TupleAccessExpression.deserialize_data_from_dict({})


# =============================================================================
# FunctionExpression
# =============================================================================


def test_function_expression_get_visit_children_orders_function_indices_args() -> None:
    """Test FunctionExpression yields function, indices, then args in order."""
    fn = IdentifierExpression(identifier=Identifier("f"))
    idx = IntLiteral(value=0)
    arg = IntLiteral(value=1)
    node = FunctionExpression(function=fn, indices=(idx,), args=(arg,))
    assert node.get_visit_children() == (fn, idx, arg)
    assert node.get_operands() == (arg,)


def test_function_expression_roundtrip_preserves_structure() -> None:
    """Test FunctionExpression survives a serialize/deserialize round-trip."""
    node = FunctionExpression(
        function=IdentifierExpression(identifier=Identifier("f")),
        template_types=(PrimitiveDataType(CoreDataType.INT32),),
        indices=(IntLiteral(value=0),),
        args=(IntLiteral(value=1),),
    )
    restored = _roundtrip(node)
    assert isinstance(restored, FunctionExpression)
    assert node.is_structurally_equivalent(restored)


def test_function_expression_equivalence_rejects_non_function_expression() -> None:
    """Test FunctionExpression equivalence returns False against an unrelated node."""
    node = FunctionExpression(function=IdentifierExpression(identifier=Identifier("f")))
    assert not node.is_structurally_equivalent(IntLiteral(value=0))


def test_function_expression_equivalence_rejects_different_function() -> None:
    """Test FunctionExpression equivalence returns False when callees differ."""
    a = FunctionExpression(function=IdentifierExpression(identifier=Identifier("f")))
    b = FunctionExpression(function=IdentifierExpression(identifier=Identifier("g")))
    assert not a.is_structurally_equivalent(b)


def test_function_expression_equivalence_rejects_different_template_count() -> None:
    """Test FunctionExpression equivalence returns False when template counts differ."""
    fn = IdentifierExpression(identifier=Identifier("f"))
    a = FunctionExpression(
        function=fn,
        template_types=(PrimitiveDataType(CoreDataType.INT32),),
    )
    b = FunctionExpression(function=fn)
    assert not a.is_structurally_equivalent(b)


def test_function_expression_equivalence_rejects_different_template_content() -> None:
    """Test FunctionExpression equivalence returns False when template contents differ."""  # noqa: E501
    fn = IdentifierExpression(identifier=Identifier("f"))
    a = FunctionExpression(
        function=fn,
        template_types=(PrimitiveDataType(CoreDataType.INT32),),
    )
    b = FunctionExpression(
        function=fn,
        template_types=(PrimitiveDataType(CoreDataType.FLOAT32),),
    )
    assert not a.is_structurally_equivalent(b)


def test_function_expression_equivalence_rejects_different_index_count() -> None:
    """Test FunctionExpression equivalence returns False when index counts differ."""
    fn = IdentifierExpression(identifier=Identifier("f"))
    a = FunctionExpression(function=fn, indices=(IntLiteral(value=0),))
    b = FunctionExpression(function=fn)
    assert not a.is_structurally_equivalent(b)


def test_function_expression_equivalence_rejects_different_index_content() -> None:
    """Test FunctionExpression equivalence returns False when index contents differ."""
    fn = IdentifierExpression(identifier=Identifier("f"))
    a = FunctionExpression(function=fn, indices=(IntLiteral(value=0),))
    b = FunctionExpression(function=fn, indices=(IntLiteral(value=1),))
    assert not a.is_structurally_equivalent(b)


def test_function_expression_equivalence_rejects_different_arg_count() -> None:
    """Test FunctionExpression equivalence returns False when argument counts differ."""
    fn = IdentifierExpression(identifier=Identifier("f"))
    a = FunctionExpression(function=fn, args=(IntLiteral(value=0),))
    b = FunctionExpression(function=fn)
    assert not a.is_structurally_equivalent(b)


def test_function_expression_equivalence_rejects_different_arg_content() -> None:
    """Test FunctionExpression equivalence returns False when argument contents differ."""  # noqa: E501
    fn = IdentifierExpression(identifier=Identifier("f"))
    a = FunctionExpression(function=fn, args=(IntLiteral(value=0),))
    b = FunctionExpression(function=fn, args=(IntLiteral(value=1),))
    assert not a.is_structurally_equivalent(b)


def test_function_expression_deserialize_rejects_empty_dict() -> None:
    """Test FunctionExpression deserialization rejects a dict missing required keys."""
    with pytest.raises(DeserializationDictStructureError):
        FunctionExpression.deserialize_data_from_dict({})


# =============================================================================
# ArrayAccessExpression
# =============================================================================


def test_array_access_get_operands_includes_array_and_indices() -> None:
    """Test ArrayAccessExpression.get_operands returns the array followed by indices."""
    arr = IdentifierExpression(identifier=Identifier("a"))
    idx = IntLiteral(value=0)
    node = ArrayAccessExpression(array_expression=arr, indices=(idx,))
    assert node.get_operands() == (arr, idx)
    assert node.get_visit_children() == (arr, idx)


def test_array_access_roundtrip_preserves_structure() -> None:
    """Test ArrayAccessExpression survives a serialize/deserialize round-trip."""
    node = ArrayAccessExpression(
        array_expression=IdentifierExpression(identifier=Identifier("a")),
        indices=(IntLiteral(value=0), IntLiteral(value=1)),
    )
    restored = _roundtrip(node)
    assert isinstance(restored, ArrayAccessExpression)
    assert node.is_structurally_equivalent(restored)


def test_array_access_equivalence_rejects_non_array_access() -> None:
    """Test ArrayAccessExpression equivalence rejects an unrelated node type."""
    node = ArrayAccessExpression(
        array_expression=IdentifierExpression(identifier=Identifier("a"))
    )
    assert not node.is_structurally_equivalent(IntLiteral(value=0))


def test_array_access_equivalence_rejects_different_array() -> None:
    """Test ArrayAccessExpression equivalence returns False when arrays differ."""
    a = ArrayAccessExpression(
        array_expression=IdentifierExpression(identifier=Identifier("a"))
    )
    b = ArrayAccessExpression(
        array_expression=IdentifierExpression(identifier=Identifier("b"))
    )
    assert not a.is_structurally_equivalent(b)


def test_array_access_equivalence_rejects_different_index_count() -> None:
    """Test ArrayAccessExpression equivalence returns False when index counts differ."""
    arr = IdentifierExpression(identifier=Identifier("a"))
    a = ArrayAccessExpression(array_expression=arr, indices=(IntLiteral(value=0),))
    b = ArrayAccessExpression(array_expression=arr)
    assert not a.is_structurally_equivalent(b)


def test_array_access_equivalence_rejects_different_index_content() -> None:
    """Test ArrayAccessExpression equivalence returns False when index contents differ."""  # noqa: E501
    arr = IdentifierExpression(identifier=Identifier("a"))
    a = ArrayAccessExpression(array_expression=arr, indices=(IntLiteral(value=0),))
    b = ArrayAccessExpression(array_expression=arr, indices=(IntLiteral(value=1),))
    assert not a.is_structurally_equivalent(b)


def test_array_access_deserialize_rejects_empty_dict() -> None:
    """Test ArrayAccessExpression deserialization rejects a dict missing required keys."""  # noqa: E501
    with pytest.raises(DeserializationDictStructureError):
        ArrayAccessExpression.deserialize_data_from_dict({})


# =============================================================================
# TupleExpression
# =============================================================================


def test_tuple_expression_get_operands_returns_expressions() -> None:
    """Test TupleExpression.get_operands returns the contained expressions."""
    a = IntLiteral(value=1)
    b = IntLiteral(value=2)
    node = TupleExpression(expressions=(a, b))
    assert node.get_operands() == (a, b)
    assert node.get_visit_children() == (a, b)


def test_tuple_expression_roundtrip_preserves_structure() -> None:
    """Test TupleExpression survives a serialize/deserialize round-trip."""
    node = TupleExpression(expressions=(IntLiteral(value=1), IntLiteral(value=2)))
    restored = _roundtrip(node)
    assert isinstance(restored, TupleExpression)
    assert node.is_structurally_equivalent(restored)


def test_tuple_expression_equivalence_rejects_non_tuple_expression() -> None:
    """Test TupleExpression equivalence returns False against an unrelated node."""
    node = TupleExpression()
    assert not node.is_structurally_equivalent(IntLiteral(value=0))


def test_tuple_expression_equivalence_rejects_different_count() -> None:
    """Test TupleExpression equivalence returns False when element counts differ."""
    a = TupleExpression(expressions=(IntLiteral(value=1),))
    b = TupleExpression()
    assert not a.is_structurally_equivalent(b)


def test_tuple_expression_equivalence_rejects_different_content() -> None:
    """Test TupleExpression equivalence returns False when element contents differ."""
    a = TupleExpression(expressions=(IntLiteral(value=1),))
    b = TupleExpression(expressions=(IntLiteral(value=2),))
    assert not a.is_structurally_equivalent(b)


def test_tuple_expression_deserialize_rejects_empty_dict() -> None:
    """Test TupleExpression deserialization rejects a dict missing required keys."""
    with pytest.raises(DeserializationDictStructureError):
        TupleExpression.deserialize_data_from_dict({})


# =============================================================================
# IdentifierExpression
# =============================================================================


def test_identifier_expression_roundtrip_preserves_structure() -> None:
    """Test IdentifierExpression survives a serialize/deserialize round-trip."""
    node = IdentifierExpression(identifier=Identifier("x"))
    restored = _roundtrip(node)
    assert isinstance(restored, IdentifierExpression)
    assert node.is_structurally_equivalent(restored)


def test_identifier_expression_equivalence_rejects_non_identifier_expression() -> None:
    """Test IdentifierExpression equivalence returns False against an unrelated node."""
    node = IdentifierExpression(identifier=Identifier("x"))
    assert not node.is_structurally_equivalent(IntLiteral(value=0))


def test_identifier_expression_equivalence_rejects_different_identifier() -> None:
    """Test IdentifierExpression equivalence returns False when identifiers differ."""
    a = IdentifierExpression(identifier=Identifier("x"))
    b = IdentifierExpression(identifier=Identifier("y"))
    assert not a.is_structurally_equivalent(b)


def test_identifier_expression_deserialize_rejects_empty_dict() -> None:
    """Test IdentifierExpression deserialization rejects a dict missing required keys."""  # noqa: E501
    with pytest.raises(DeserializationDictStructureError):
        IdentifierExpression.deserialize_data_from_dict({})


# =============================================================================
# IntLiteral
# =============================================================================


def test_int_literal_roundtrip_preserves_value() -> None:
    """Test IntLiteral survives a serialize/deserialize round-trip."""
    node = IntLiteral(value=42)
    restored = _roundtrip(node)
    assert isinstance(restored, IntLiteral)
    assert restored.value == 42
    assert node.is_structurally_equivalent(restored)


def test_int_literal_equivalence_rejects_non_int_literal() -> None:
    """Test IntLiteral equivalence returns False against an unrelated node."""
    node = IntLiteral(value=1)
    assert not node.is_structurally_equivalent(FloatLiteral(value=1.0))


def test_int_literal_deserialize_rejects_empty_dict() -> None:
    """Test IntLiteral deserialization rejects a dict missing required keys."""
    with pytest.raises(DeserializationDictStructureError):
        IntLiteral.deserialize_data_from_dict({})


# =============================================================================
# FloatLiteral
# =============================================================================


def test_float_literal_roundtrip_preserves_value() -> None:
    """Test FloatLiteral survives a serialize/deserialize round-trip."""
    node = FloatLiteral(value=2.5)
    restored = _roundtrip(node)
    assert isinstance(restored, FloatLiteral)
    assert restored.value == 2.5
    assert node.is_structurally_equivalent(restored)


def test_float_literal_equivalence_rejects_non_float_literal() -> None:
    """Test FloatLiteral equivalence returns False against an unrelated node."""
    node = FloatLiteral(value=1.0)
    assert not node.is_structurally_equivalent(IntLiteral(value=1))


def test_float_literal_equivalence_rejects_different_value() -> None:
    """Test FloatLiteral equivalence returns False when values differ."""
    a = FloatLiteral(value=1.0)
    b = FloatLiteral(value=2.0)
    assert not a.is_structurally_equivalent(b)


def test_float_literal_deserialize_rejects_empty_dict() -> None:
    """Test FloatLiteral deserialization rejects a dict missing required keys."""
    with pytest.raises(DeserializationDictStructureError):
        FloatLiteral.deserialize_data_from_dict({})


# =============================================================================
# ComplexLiteral
# =============================================================================


def test_complex_literal_roundtrip_preserves_value() -> None:
    """Test ComplexLiteral survives a serialize/deserialize round-trip."""
    node = ComplexLiteral(value=complex(1.5, -2.5))
    restored = _roundtrip(node)
    assert isinstance(restored, ComplexLiteral)
    assert restored.value == complex(1.5, -2.5)
    assert node.is_structurally_equivalent(restored)


def test_complex_literal_equivalence_rejects_non_complex_literal() -> None:
    """Test ComplexLiteral equivalence returns False against an unrelated node."""
    node = ComplexLiteral(value=complex(1.0, 0.0))
    assert not node.is_structurally_equivalent(FloatLiteral(value=1.0))


def test_complex_literal_equivalence_rejects_different_value() -> None:
    """Test ComplexLiteral equivalence returns False when values differ."""
    a = ComplexLiteral(value=complex(1.0, 0.0))
    b = ComplexLiteral(value=complex(0.0, 1.0))
    assert not a.is_structurally_equivalent(b)


def test_complex_literal_deserialize_rejects_empty_dict() -> None:
    """Test ComplexLiteral deserialization rejects a dict missing required keys."""
    with pytest.raises(DeserializationDictStructureError):
        ComplexLiteral.deserialize_data_from_dict({})


# =============================================================================
# Cross-class deserialization through the abstract dispatchers
# =============================================================================


def test_expression_abstract_deserialize_dispatches_to_concrete_class() -> None:
    """Test Expression.deserialize_from_dict reconstructs the concrete subclass."""
    node = IntLiteral(value=5)
    restored = Expression.deserialize_from_dict(node.serialize_to_dict())
    assert isinstance(restored, IntLiteral)
    assert restored.is_structurally_equivalent(node)


def test_statement_abstract_deserialize_dispatches_to_concrete_class() -> None:
    """Test Statement.deserialize_from_dict reconstructs the concrete subclass."""
    node = ReturnStatement(expression=IntLiteral(value=1))
    restored = Statement.deserialize_from_dict(node.serialize_to_dict())
    assert isinstance(restored, ReturnStatement)
    assert restored.is_structurally_equivalent(node)


# =============================================================================
# Edge cases — equivalence semantics
# =============================================================================


@pytest.fixture
def representative_nodes() -> list[Node]:
    """Provide one instance of every concrete AST node class."""
    qt = QualifiedType(
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
        Argument(name=Identifier("x"), qualified_type=qt),
        Procedure(name=Identifier("p")),
        Operation(name=Identifier("op"), return_type=qt),
        Native(name=Identifier("n")),
        DeclarationStatement(variable_name=Identifier("x"), variable_type=qt),
        ExpressionStatement(right=IntLiteral(value=1)),
        ForAllStatement(index=IdentifierExpression(identifier=Identifier("i"))),
        SelectionStatement(condition=IdentifierExpression(identifier=Identifier("c"))),
        ReturnStatement(expression=IntLiteral(value=1)),
        QualifiedType(
            base_type=NumericalType(PrimitiveDataType(CoreDataType.INT32)),
            type_qualifier=TypeQualifier.INPUT,
        ),
    ]


def test_node_is_structurally_equivalent_to_itself(
    representative_nodes: list[Node],
) -> None:
    """Test every concrete AST node is structurally equivalent to itself."""
    for node in representative_nodes:
        assert node.is_structurally_equivalent(node), (
            f"{type(node).__name__} is not self-equivalent"
        )


@pytest.mark.parametrize("non_node", [None, "string", 42, 3.14, [], {}, object()])
def test_node_equivalence_rejects_non_node_objects(
    non_node: object,
) -> None:
    """Test structural equivalence returns False for any non-Node argument."""
    node = IntLiteral(value=1)
    assert not node.is_structurally_equivalent(non_node)


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
        (
            Procedure(name=Identifier("p")),
            Native(name=Identifier("p")),
        ),
        (Import(name=Identifier("a")), Import(name=Identifier("b"))),
        (Module(name=Identifier("a")), Module(name=Identifier("b"))),
        (
            ReturnStatement(expression=IntLiteral(value=1)),
            ExpressionStatement(right=IntLiteral(value=1)),
        ),
    ],
)
def test_equivalence_is_symmetric(a: Node, b: Node) -> None:
    """Test ``a.is_structurally_equivalent(b)`` agrees with the reverse direction."""
    assert a.is_structurally_equivalent(b) == b.is_structurally_equivalent(a)


def test_two_default_for_all_statements_are_not_structurally_equivalent() -> None:
    """Test default-built ForAllStatements differ because the name factory is fresh."""
    index = IdentifierExpression(identifier=Identifier("i"))
    a = ForAllStatement(index=index)
    b = ForAllStatement(index=index)
    assert a.name != b.name
    assert not a.is_structurally_equivalent(b)


def test_two_default_modules_are_structurally_equivalent() -> None:
    """Test default-built Modules share the class-level default name and match."""
    a = Module()
    b = Module()
    assert a.name is b.name
    assert a.is_structurally_equivalent(b)


# =============================================================================
# Edge cases — literal value boundaries
# =============================================================================


def test_float_literal_nan_value_is_preserved_but_not_self_equivalent() -> None:
    """Test FloatLiteral round-trips a NaN value but loses self-equivalence."""
    node = FloatLiteral(value=float("nan"))
    restored = _roundtrip(node)
    assert isinstance(restored, FloatLiteral)
    assert math.isnan(restored.value)
    # NaN != NaN by IEEE-754, and the equivalence implementation uses ==.
    assert not node.is_structurally_equivalent(restored)


def test_float_literal_infinity_round_trip_preserves_sign() -> None:
    """Test FloatLiteral round-trip preserves +inf and -inf as distinct values."""
    pos = FloatLiteral(value=float("inf"))
    neg = FloatLiteral(value=float("-inf"))
    assert pos.is_structurally_equivalent(_roundtrip(pos))
    assert neg.is_structurally_equivalent(_roundtrip(neg))
    assert not pos.is_structurally_equivalent(neg)


def test_negative_zero_float_is_not_equivalent_to_int_zero() -> None:
    """Test FloatLiteral(-0.0) is cross-class distinct from IntLiteral(0)."""
    int_zero = IntLiteral(value=0)
    neg_zero = FloatLiteral(value=-0.0)
    assert not int_zero.is_structurally_equivalent(neg_zero)
    assert not neg_zero.is_structurally_equivalent(int_zero)


def test_complex_literal_nan_value_is_preserved_but_not_self_equivalent() -> None:
    """Test ComplexLiteral round-trips a NaN component but loses self-equivalence."""
    node = ComplexLiteral(value=complex(float("nan"), 0.0))
    restored = _roundtrip(node)
    assert isinstance(restored, ComplexLiteral)
    assert math.isnan(restored.value.real)
    assert not node.is_structurally_equivalent(restored)


def test_complex_literal_treats_signed_zero_imag_as_equivalent() -> None:
    """Test ComplexLiteral equivalence follows Python's complex equality semantics."""
    a = ComplexLiteral(value=complex(1.0, 0.0))
    b = ComplexLiteral(value=complex(1.0, -0.0))
    assert a.is_structurally_equivalent(b)


def test_int_literal_accepts_bool_value_and_preserves_type() -> None:
    """Test IntLiteral accepts a bool value and round-trips it without coercing."""
    node = IntLiteral(value=True)
    restored = _roundtrip(node)
    assert isinstance(restored, IntLiteral)
    assert restored.value is True
    # IntLiteral(True) and IntLiteral(1) compare equivalent because True == 1.
    assert node.is_structurally_equivalent(IntLiteral(value=1))


@pytest.mark.parametrize("value", [0, -1, 2**100, -(2**63), 2**63 - 1])
def test_int_literal_round_trips_extreme_values(value: int) -> None:
    """Test IntLiteral round-trips boundary, large, and negative integer values."""
    node = IntLiteral(value=value)
    restored = _roundtrip(node)
    assert isinstance(restored, IntLiteral)
    assert restored.value == value
    assert node.is_structurally_equivalent(restored)


# =============================================================================
# Edge cases — empty containers and asymmetric branches
# =============================================================================


def _make_qualified_type() -> QualifiedType:
    """Return a fresh ``input int32`` qualified type."""
    return QualifiedType(
        base_type=NumericalType(PrimitiveDataType(CoreDataType.INT32)),
        type_qualifier=TypeQualifier.INPUT,
    )


def test_empty_module_round_trip_preserves_emptiness() -> None:
    """Test a Module with no statements round-trips with an empty statement tuple."""
    node = Module()
    restored = _roundtrip(node)
    assert isinstance(restored, Module)
    assert restored.statements == ()
    assert node.is_structurally_equivalent(restored)


def test_empty_procedure_round_trip_preserves_emptiness() -> None:
    """Test a Procedure with no templates, args, or body round-trips intact."""
    node = Procedure(name=Identifier("p"))
    restored = _roundtrip(node)
    assert isinstance(restored, Procedure)
    assert restored.templates == () and restored.args == () and restored.body == ()
    assert node.is_structurally_equivalent(restored)


def test_empty_operation_round_trip_preserves_emptiness() -> None:
    """Test an Operation with no templates, args, or body round-trips intact."""
    node = Operation(name=Identifier("op"), return_type=_make_qualified_type())
    restored = _roundtrip(node)
    assert isinstance(restored, Operation)
    assert restored.templates == () and restored.args == () and restored.body == ()
    assert node.is_structurally_equivalent(restored)


def test_empty_native_round_trip_preserves_emptiness() -> None:
    """Test a Native with no args round-trips intact."""
    node = Native(name=Identifier("n"))
    restored = _roundtrip(node)
    assert isinstance(restored, Native)
    assert restored.args == ()
    assert node.is_structurally_equivalent(restored)


def test_empty_for_all_round_trip_preserves_emptiness() -> None:
    """Test a ForAllStatement with no body round-trips intact."""
    node = ForAllStatement(index=IdentifierExpression(identifier=Identifier("i")))
    restored = _roundtrip(node)
    assert isinstance(restored, ForAllStatement)
    assert restored.body == ()
    assert node.is_structurally_equivalent(restored)


def test_empty_selection_round_trip_preserves_emptiness() -> None:
    """Test a SelectionStatement with both bodies empty round-trips intact."""
    node = SelectionStatement(
        condition=IdentifierExpression(identifier=Identifier("c"))
    )
    restored = _roundtrip(node)
    assert isinstance(restored, SelectionStatement)
    assert restored.true_body == () and restored.false_body == ()
    assert node.is_structurally_equivalent(restored)


def test_empty_tuple_expression_round_trip_preserves_emptiness() -> None:
    """Test a TupleExpression with no elements round-trips intact."""
    node = TupleExpression()
    restored = _roundtrip(node)
    assert isinstance(restored, TupleExpression)
    assert restored.expressions == ()
    assert node.is_structurally_equivalent(restored)


def test_empty_function_expression_round_trip_preserves_emptiness() -> None:
    """Test a FunctionExpression with no template_types/indices/args round-trips intact."""  # noqa: E501
    node = FunctionExpression(function=IdentifierExpression(identifier=Identifier("f")))
    restored = _roundtrip(node)
    assert isinstance(restored, FunctionExpression)
    assert (
        restored.template_types == () and restored.indices == () and restored.args == ()
    )
    assert node.is_structurally_equivalent(restored)


def test_empty_array_access_round_trip_preserves_emptiness() -> None:
    """Test an ArrayAccessExpression with no indices round-trips intact."""
    node = ArrayAccessExpression(
        array_expression=IdentifierExpression(identifier=Identifier("a"))
    )
    restored = _roundtrip(node)
    assert isinstance(restored, ArrayAccessExpression)
    assert restored.indices == ()
    assert node.is_structurally_equivalent(restored)


def test_selection_statement_with_only_true_branch_round_trips() -> None:
    """Test SelectionStatement with a populated true branch and empty false round-trips."""  # noqa: E501
    node = SelectionStatement(
        condition=IdentifierExpression(identifier=Identifier("c")),
        true_body=(ReturnStatement(expression=IntLiteral(value=1)),),
    )
    restored = _roundtrip(node)
    assert isinstance(restored, SelectionStatement)
    assert restored.false_body == ()
    assert node.is_structurally_equivalent(restored)


def test_selection_statement_with_only_false_branch_round_trips() -> None:
    """Test SelectionStatement with an empty true branch and populated false round-trips."""  # noqa: E501
    node = SelectionStatement(
        condition=IdentifierExpression(identifier=Identifier("c")),
        false_body=(ReturnStatement(expression=IntLiteral(value=0)),),
    )
    restored = _roundtrip(node)
    assert isinstance(restored, SelectionStatement)
    assert restored.true_body == ()
    assert node.is_structurally_equivalent(restored)


# =============================================================================
# Edge cases — nesting and composition
# =============================================================================


def test_deeply_nested_binary_expression_round_trips() -> None:
    """Test a five-level nested BinaryExpression round-trips and stays equivalent."""
    # ((((1 + 2) * 3) - 4) ** 5) // 6
    node: Expression = IntLiteral(value=1)
    for op, value in [
        (BinaryOperation.ADDITION, 2),
        (BinaryOperation.MULTIPLICATION, 3),
        (BinaryOperation.SUBTRACTION, 4),
        (BinaryOperation.POWER, 5),
        (BinaryOperation.FLOORDIV, 6),
    ]:
        node = BinaryExpression(operation=op, left=node, right=IntLiteral(value=value))
    restored = _roundtrip(node)
    assert node.is_structurally_equivalent(restored)


def test_function_expression_callable_is_an_expression_not_only_identifier() -> None:
    """Test FunctionExpression accepts another FunctionExpression as its callee."""
    inner = FunctionExpression(
        function=IdentifierExpression(identifier=Identifier("f"))
    )
    outer = FunctionExpression(function=inner, args=(IntLiteral(value=1),))
    restored = _roundtrip(outer)
    assert outer.is_structurally_equivalent(restored)


def test_tuple_access_chained_on_array_access_round_trips() -> None:
    """Test TupleAccessExpression on top of ArrayAccessExpression round-trips."""
    array = ArrayAccessExpression(
        array_expression=IdentifierExpression(identifier=Identifier("a")),
        indices=(IntLiteral(value=0),),
    )
    node = TupleAccessExpression(
        tuple_expression=array, element_index=IntLiteral(value=0)
    )
    restored = _roundtrip(node)
    assert node.is_structurally_equivalent(restored)


def test_nested_for_all_statement_round_trips() -> None:
    """Test a ForAllStatement nested inside another ForAllStatement round-trips."""
    inner = ForAllStatement(
        index=IdentifierExpression(identifier=Identifier("j")),
        body=(ReturnStatement(expression=IntLiteral(value=1)),),
    )
    outer = ForAllStatement(
        index=IdentifierExpression(identifier=Identifier("i")),
        body=(inner,),
    )
    restored = _roundtrip(outer)
    assert outer.is_structurally_equivalent(restored)


# =============================================================================
# Edge cases — full grammar integration
# =============================================================================


def test_module_containing_every_statement_kind_round_trips() -> None:
    """Test a Module that includes every Statement subclass round-trips intact."""
    qt = _make_qualified_type()
    decl = DeclarationStatement(variable_name=Identifier("x"), variable_type=qt)
    expr_stmt = ExpressionStatement(
        left=IdentifierExpression(identifier=Identifier("x")),
        right=IntLiteral(value=1),
    )
    forall = ForAllStatement(
        index=IdentifierExpression(identifier=Identifier("i")),
        body=(ReturnStatement(expression=IntLiteral(value=0)),),
    )
    selection = SelectionStatement(
        condition=IdentifierExpression(identifier=Identifier("c")),
        true_body=(ReturnStatement(expression=IntLiteral(value=1)),),
        false_body=(ReturnStatement(expression=IntLiteral(value=0)),),
    )
    ret = ReturnStatement(expression=IntLiteral(value=0))
    procedure = Procedure(
        name=Identifier("proc"),
        args=(Argument(name=Identifier("a"), qualified_type=qt),),
        body=(decl, expr_stmt, forall, selection, ret),
    )
    operation = Operation(name=Identifier("op"), return_type=qt)
    native = Native(name=Identifier("nat"))
    import_stmt = Import(name=Identifier("lib"))
    module = Module(
        name=Identifier("m"),
        statements=(import_stmt, procedure, operation, native),
    )
    restored = _roundtrip(module)
    assert module.is_structurally_equivalent(restored)


# =============================================================================
# Edge cases — frozen dataclass and abstract class enforcement
# =============================================================================


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
def test_frozen_node_rejects_attribute_assignment(node: Node) -> None:
    """Test attribute assignment on a frozen AST node raises FrozenInstanceError."""
    field_names = [f.name for f in dataclasses.fields(node)]
    target = field_names[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(node, target, getattr(node, target))


def test_abstract_node_cannot_be_instantiated() -> None:
    """Test the abstract Node base class cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Node()  # type: ignore[abstract]


def test_abstract_statement_cannot_be_instantiated() -> None:
    """Test the abstract Statement base class cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Statement()  # type: ignore[abstract]


def test_abstract_expression_cannot_be_instantiated() -> None:
    """Test the abstract Expression base class cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Expression()  # type: ignore[abstract]


def test_abstract_function_cannot_be_instantiated() -> None:
    """Test the abstract Function base class cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Function(name=Identifier("f"))  # type: ignore[abstract]


def test_abstract_literal_cannot_be_instantiated() -> None:
    """Test the abstract Literal base class cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Literal()  # type: ignore[abstract]


# =============================================================================
# Edge cases — top-level dispatch and family-mismatch errors
# =============================================================================


def test_top_level_deserialize_rejects_non_wrapped_dict() -> None:
    """Test Node.deserialize_from_dict rejects a dict missing the wrapper keys."""
    with pytest.raises(SerializationError):
        Node.deserialize_from_dict({})


def test_top_level_deserialize_rejects_unknown_type_id() -> None:
    """Test Node.deserialize_from_dict rejects an unregistered ``__type__`` value."""
    with pytest.raises(UnknownTypeIdError):
        Node.deserialize_from_dict(
            {"__type__": "fhy_ast_does_not_exist", "__data__": {}}
        )


def test_abstract_dispatch_rejects_wrong_family() -> None:
    """Test Statement.deserialize_from_dict rejects an Expression-family payload."""
    payload = IntLiteral(value=1).serialize_to_dict()
    with pytest.raises(SerializationError):
        Statement.deserialize_from_dict(payload)


def test_concrete_class_deserialize_rejects_wrong_family() -> None:
    """Test IntLiteral.deserialize_from_dict rejects a Module-family payload."""
    payload = Module().serialize_to_dict()
    with pytest.raises(SerializationError):
        IntLiteral.deserialize_from_dict(payload)


# =============================================================================
# Edge cases — enum exhaustion
# =============================================================================


@pytest.mark.parametrize("operation", list(UnaryOperation))
def test_unary_expression_round_trip_for_every_operation(
    operation: UnaryOperation,
) -> None:
    """Test every UnaryOperation enum value survives a round-trip."""
    node = UnaryExpression(operation=operation, expression=IntLiteral(value=1))
    restored = _roundtrip(node)
    assert isinstance(restored, UnaryExpression)
    assert restored.operation == operation
    assert node.is_structurally_equivalent(restored)


@pytest.mark.parametrize("operation", list(BinaryOperation))
def test_binary_expression_round_trip_for_every_operation(
    operation: BinaryOperation,
) -> None:
    """Test every BinaryOperation enum value survives a round-trip."""
    node = BinaryExpression(
        operation=operation,
        left=IntLiteral(value=1),
        right=IntLiteral(value=2),
    )
    restored = _roundtrip(node)
    assert isinstance(restored, BinaryExpression)
    assert restored.operation == operation
    assert node.is_structurally_equivalent(restored)


@pytest.mark.parametrize("qualifier", list(TypeQualifier))
def test_qualified_type_round_trip_for_every_qualifier(
    qualifier: TypeQualifier,
) -> None:
    """Test every TypeQualifier enum value survives a round-trip on QualifiedType."""
    node = QualifiedType(
        base_type=NumericalType(PrimitiveDataType(CoreDataType.INT32)),
        type_qualifier=qualifier,
    )
    restored = _roundtrip(node)
    assert isinstance(restored, QualifiedType)
    assert restored.type_qualifier == qualifier
    assert node.is_structurally_equivalent(restored)


# =============================================================================
# Edge cases — provenance propagation and visit-order ordering
# =============================================================================


def test_distinct_provenance_per_subnode_survives_round_trip() -> None:
    """Test each sub-node retains its own provenance through serialize/deserialize."""
    outer_prov = Provenance.unknown().add_note(Note("outer"))
    left_prov = Provenance.unknown().add_note(Note("left"))
    right_prov = Provenance.unknown().add_note(Note("right"))
    node = BinaryExpression(
        operation=BinaryOperation.ADDITION,
        left=IntLiteral(value=1, provenance=left_prov),
        right=IntLiteral(value=2, provenance=right_prov),
        provenance=outer_prov,
    )
    restored = _roundtrip(node)
    assert isinstance(restored, BinaryExpression)
    assert restored.get_provenance() == outer_prov
    assert restored.left.get_provenance() == left_prov
    assert restored.right.get_provenance() == right_prov


def test_procedure_visit_children_orders_multiple_args_then_multiple_body() -> None:
    """Test Procedure preserves the relative order of multiple args and body items."""
    qt = _make_qualified_type()
    arg_a = Argument(name=Identifier("a"), qualified_type=qt)
    arg_b = Argument(name=Identifier("b"), qualified_type=qt)
    stmt_x = ReturnStatement(expression=IntLiteral(value=0))
    stmt_y = ReturnStatement(expression=IntLiteral(value=1))
    proc = Procedure(name=Identifier("p"), args=(arg_a, arg_b), body=(stmt_x, stmt_y))
    assert proc.get_visit_children() == (arg_a, arg_b, stmt_x, stmt_y)


def test_operation_visit_children_orders_args_body_then_return_type() -> None:
    """Test Operation places multiple args first, body next, return type last."""
    qt = _make_qualified_type()
    arg_a = Argument(name=Identifier("a"), qualified_type=qt)
    arg_b = Argument(name=Identifier("b"), qualified_type=qt)
    stmt_x = ReturnStatement(expression=IntLiteral(value=0))
    stmt_y = ReturnStatement(expression=IntLiteral(value=1))
    op = Operation(
        name=Identifier("op"),
        args=(arg_a, arg_b),
        body=(stmt_x, stmt_y),
        return_type=qt,
    )
    assert op.get_visit_children() == (arg_a, arg_b, stmt_x, stmt_y, qt)


def test_function_expression_visit_children_orders_function_indices_then_args() -> None:
    """Test FunctionExpression visit order is function, then all indices, then all args."""  # noqa: E501
    fn = IdentifierExpression(identifier=Identifier("f"))
    idx_a = IntLiteral(value=0)
    idx_b = IntLiteral(value=1)
    arg_x = IntLiteral(value=10)
    arg_y = IntLiteral(value=20)
    node = FunctionExpression(function=fn, indices=(idx_a, idx_b), args=(arg_x, arg_y))
    assert node.get_visit_children() == (fn, idx_a, idx_b, arg_x, arg_y)


def test_selection_visit_children_orders_condition_true_body_then_false_body() -> None:
    """Test SelectionStatement visit order is condition, true body items, false body items."""  # noqa: E501
    cond = IdentifierExpression(identifier=Identifier("c"))
    t_a = ReturnStatement(expression=IntLiteral(value=1))
    t_b = ReturnStatement(expression=IntLiteral(value=2))
    f_a = ReturnStatement(expression=IntLiteral(value=3))
    f_b = ReturnStatement(expression=IntLiteral(value=4))
    node = SelectionStatement(
        condition=cond, true_body=(t_a, t_b), false_body=(f_a, f_b)
    )
    assert node.get_visit_children() == (cond, t_a, t_b, f_a, f_b)
