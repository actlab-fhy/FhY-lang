"""Tests for ``fhy_lang.ast.node.core``.

Pins ``Module`` semantics (default-name id-freshness, equivalence,
serialization, error contract) and confirms the abstract families
``Statement``, ``Expression``, ``Function`` cannot be instantiated.
"""

import pytest
from fhy_core import (
    DeserializationDictStructureError,
    Identifier,
)
from fhy_core.serialization import SerializationError

from fhy_lang.ast.node import (
    Expression,
    Function,
    Import,
    IntLiteral,
    Module,
    ReturnStatement,
    Statement,
)


def test_statement_is_abstract():
    """Test ``Statement`` cannot be instantiated."""
    with pytest.raises(TypeError):
        Statement()  # type: ignore[abstract]


def test_expression_is_abstract():
    """Test ``Expression`` cannot be instantiated."""
    with pytest.raises(TypeError):
        Expression()  # type: ignore[abstract]


def test_function_is_abstract():
    """Test ``Function`` cannot be instantiated even with required name field."""
    with pytest.raises(TypeError):
        Function(name=Identifier("f"))  # type: ignore[abstract]


# ===========================================================================
# Module: default-name id-freshness
# ===========================================================================


def test_module_default_name_has_module_hint():
    """Test the default ``Module.name`` has hint ``"module"``."""
    assert Module().name.name_hint == "module"


def test_module_default_names_have_distinct_ids_across_instances():
    """Test default-named ``Module``s get fresh ``Identifier`` ids per instance."""
    first = Module()
    second = Module()

    assert first.name.id != second.name.id


def test_module_default_names_are_inequivalent_across_instances():
    """Test two default-named ``Module``s are not structurally equivalent."""
    assert not Module().is_structurally_equivalent(Module())


# ===========================================================================
# Module: structural equivalence (id-based contract)
# ===========================================================================


def test_module_is_inequivalent_when_inner_identifiers_have_independent_ids():
    """Test modules with independently-built inner identifiers are inequivalent."""
    name = Identifier("m")
    a = Module(name=name, statements=(Import(name=Identifier("foo")),))
    b = Module(name=name, statements=(Import(name=Identifier("foo")),))

    assert not a.is_structurally_equivalent(b)


def test_module_is_structurally_equivalent_when_name_and_imports_share_ids():
    """Test modules sharing a name id and statement-identifier ids are equivalent."""
    name = Identifier("m")
    foo = Identifier("foo")
    a = Module(name=name, statements=(Import(name=foo),))
    b = Module(name=name, statements=(Import(name=foo),))

    assert a.is_structurally_equivalent(b)


def test_module_is_inequivalent_when_name_ids_differ():
    """Test independently-constructed names with the same hint break equivalence."""
    a = Module(name=Identifier("m"))
    b = Module(name=Identifier("m"))

    assert not a.is_structurally_equivalent(b)


def test_module_is_inequivalent_when_statement_count_differs():
    """Test differing statement counts break equivalence."""
    name = Identifier("m")
    foo = Identifier("foo")
    a = Module(name=name, statements=(Import(name=foo),))
    b = Module(name=name, statements=())

    assert not a.is_structurally_equivalent(b)


def test_module_is_inequivalent_when_a_statement_differs():
    """Test a single inequivalent statement breaks Module equivalence."""
    name = Identifier("m")
    a = Module(name=name, statements=(Import(name=Identifier("foo")),))
    b = Module(name=name, statements=(Import(name=Identifier("bar")),))

    assert not a.is_structurally_equivalent(b)


def test_module_is_inequivalent_with_non_module():
    """Test ``Module.is_structurally_equivalent`` rejects non-``Module`` inputs."""
    assert not Module().is_structurally_equivalent("not-a-module")
    assert not Module().is_structurally_equivalent(Import(name=Identifier("x")))


# ===========================================================================
# Module: visitor and identifier contracts
# ===========================================================================


def test_module_get_visit_children_returns_statements_in_order():
    """Test ``Module.get_visit_children`` returns the statements tuple in order."""
    name = Identifier("m")
    a = Import(name=Identifier("a"))
    b = Import(name=Identifier("b"))
    module = Module(name=name, statements=(a, b))

    assert tuple(module.get_visit_children()) == (a, b)


def test_module_get_identifier_returns_name():
    """Test ``Module.get_identifier`` returns the ``name`` field."""
    name = Identifier("m")

    assert Module(name=name).get_identifier() is name


# ===========================================================================
# Module: serialization round-trip and error paths
# ===========================================================================


def test_module_round_trips_through_serialization():
    """Test ``Module`` survives a wrapped serialize/deserialize round-trip."""
    name = Identifier("m")
    foo = Identifier("foo")
    module = Module(name=name, statements=(Import(name=foo),))

    restored = Module.deserialize_from_dict(module.serialize_to_dict())

    assert isinstance(restored, Module)
    assert restored.name == name
    assert restored.statements[0].name == foo


def test_module_deserialize_data_rejects_payload_missing_name():
    """Test ``DeserializationDictStructureError`` on a payload missing ``name``."""
    payload = Module().serialize_data_to_dict()
    payload.pop("name")

    with pytest.raises(DeserializationDictStructureError):
        Module.deserialize_data_from_dict(payload)


def test_module_deserialize_data_rejects_payload_missing_statements():
    """Test missing ``statements`` raises ``DeserializationDictStructureError``."""
    payload = Module().serialize_data_to_dict()
    payload.pop("statements")

    with pytest.raises(DeserializationDictStructureError):
        Module.deserialize_data_from_dict(payload)


def test_module_deserialize_data_rejects_non_list_statements():
    """Test ``DeserializationDictStructureError`` when ``statements`` is not a list."""
    payload = Module().serialize_data_to_dict()
    payload["statements"] = "not-a-list"

    with pytest.raises(DeserializationDictStructureError):
        Module.deserialize_data_from_dict(payload)


def test_module_deserialize_data_rejects_non_dict_statement_entry():
    """Test ``DeserializationDictStructureError`` on a non-dict statement entry."""
    payload = Module().serialize_data_to_dict()
    payload["statements"] = ["not-a-dict"]

    with pytest.raises(DeserializationDictStructureError):
        Module.deserialize_data_from_dict(payload)


def test_module_payload_through_statement_family_raises_serialization_error():
    """Test ``Statement.deserialize_from_dict`` rejects a ``Module`` payload."""
    wrapped = Module().serialize_to_dict()

    with pytest.raises(SerializationError):
        Statement.deserialize_from_dict(wrapped)


def test_expression_payload_through_statement_family_raises_serialization_error():
    """Test ``Statement.deserialize_from_dict`` rejects an ``Expression`` payload."""
    wrapped = IntLiteral(value=1).serialize_to_dict()

    with pytest.raises(SerializationError):
        Statement.deserialize_from_dict(wrapped)


def test_statement_payload_through_expression_family_raises_serialization_error():
    """Test feeding a ``Statement`` payload to ``Expression`` family fails."""
    wrapped = ReturnStatement(expression=IntLiteral(value=1)).serialize_to_dict()

    with pytest.raises(SerializationError):
        Expression.deserialize_from_dict(wrapped)


def test_unknown_type_id_in_wrapped_payload_raises_serialization_error():
    """Test an unknown ``__type__`` id in a wrapped payload raises."""
    wrapped = Module().serialize_to_dict()
    wrapped["__type__"] = "definitely-not-a-registered-type-id"

    with pytest.raises(SerializationError):
        Module.deserialize_from_dict(wrapped)


def test_module_with_default_statements_is_empty():
    """Test ``Module()`` has an empty ``statements`` tuple by default."""
    assert Module().statements == ()
