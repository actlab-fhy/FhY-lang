"""Tests for ``fhy_lang.ast.node.statement``.

Covers every concrete statement node: construction, structural equivalence
(id-based), visitor contracts (including the documented exclusion of
templates from the AST walk), serialization round-trips, and
deserialization error paths.
"""

import pytest
from fhy_core import (
    CoreDataType,
    DeserializationDictStructureError,
    Identifier,
    NumericalType,
    PrimitiveDataType,
    TemplateDataType,
    TypeQualifier,
)

from fhy_lang.ast.node import (
    Argument,
    DeclarationStatement,
    ExpressionStatement,
    ForAllStatement,
    IdentifierExpression,
    Import,
    IntLiteral,
    Native,
    Operation,
    Procedure,
    QualifiedType,
    ReturnStatement,
    SelectionStatement,
    Statement,
)

# ===========================================================================
# Helpers
# ===========================================================================


def _int32() -> NumericalType:
    return NumericalType(PrimitiveDataType(CoreDataType.INT32))


def _input_int32() -> QualifiedType:
    return QualifiedType(base_type=_int32(), type_qualifier=TypeQualifier.INPUT)


def _output_int32() -> QualifiedType:
    return QualifiedType(base_type=_int32(), type_qualifier=TypeQualifier.OUTPUT)


def _identifier_expression(name: str = "x") -> IdentifierExpression:
    return IdentifierExpression(identifier=Identifier(name))


def _argument(name: str = "x") -> Argument:
    return Argument(name=Identifier(name), qualified_type=_input_int32())


# ===========================================================================
# Argument
# ===========================================================================


def test_argument_equivalent_when_name_id_and_type_match():
    """Test arguments sharing a name id and equivalent type are equivalent."""
    name = Identifier("a")
    a = Argument(name=name, qualified_type=_input_int32())
    b = Argument(name=name, qualified_type=_input_int32())

    assert a.is_structurally_equivalent(b)


def test_argument_inequivalent_when_name_ids_differ():
    """Test independently-constructed names break equivalence."""
    a = Argument(name=Identifier("a"), qualified_type=_input_int32())
    b = Argument(name=Identifier("a"), qualified_type=_input_int32())

    assert not a.is_structurally_equivalent(b)


def test_argument_inequivalent_when_qualified_type_differs():
    """Test differing qualified type breaks equivalence."""
    name = Identifier("a")
    a = Argument(name=name, qualified_type=_input_int32())
    b = Argument(name=name, qualified_type=_output_int32())

    assert not a.is_structurally_equivalent(b)


def test_argument_get_visit_children_returns_qualified_type():
    """Test ``Argument.get_visit_children`` walks only the qualified type."""
    qualified = _input_int32()
    argument = Argument(name=Identifier("a"), qualified_type=qualified)

    assert tuple(argument.get_visit_children()) == (qualified,)


def test_argument_round_trips_through_serialization():
    """Test ``Argument`` survives a wrapped round-trip."""
    argument = _argument("a")

    restored = Argument.deserialize_from_dict(argument.serialize_to_dict())

    assert isinstance(restored, Argument)


# ===========================================================================
# Import
# ===========================================================================


def test_import_equivalent_when_name_ids_match():
    """Test imports sharing a name id are equivalent."""
    name = Identifier("foo")

    assert Import(name=name).is_structurally_equivalent(Import(name=name))


def test_import_inequivalent_when_name_ids_differ():
    """Test independently-constructed names break equivalence."""
    assert not Import(name=Identifier("foo")).is_structurally_equivalent(
        Import(name=Identifier("foo"))
    )


def test_import_get_visit_children_is_empty():
    """Test ``Import.get_visit_children`` returns an empty sequence."""
    assert tuple(Import(name=Identifier("foo")).get_visit_children()) == ()


def test_import_round_trips_through_serialization():
    """Test ``Import`` survives a wrapped round-trip."""
    name = Identifier("foo")
    statement = Import(name=name)

    restored = Statement.deserialize_from_dict(statement.serialize_to_dict())

    assert isinstance(restored, Import)
    assert restored.name == name


def test_import_deserialize_data_rejects_missing_name():
    """Test missing ``name`` raises ``DeserializationDictStructureError``."""
    payload = Import(name=Identifier("foo")).serialize_data_to_dict()
    payload.pop("name")

    with pytest.raises(DeserializationDictStructureError):
        Import.deserialize_data_from_dict(payload)


# ===========================================================================
# DeclarationStatement
# ===========================================================================


def test_declaration_statement_equivalent_with_no_expression():
    """Test declarations without expressions are equivalent when name + type match."""
    name = Identifier("x")
    a = DeclarationStatement(variable_name=name, variable_type=_input_int32())
    b = DeclarationStatement(variable_name=name, variable_type=_input_int32())

    assert a.is_structurally_equivalent(b)


def test_declaration_statement_equivalent_with_matching_expression():
    """Test declarations with matching expressions are equivalent."""
    name = Identifier("x")
    a = DeclarationStatement(
        variable_name=name, variable_type=_input_int32(), expression=IntLiteral(value=1)
    )
    b = DeclarationStatement(
        variable_name=name, variable_type=_input_int32(), expression=IntLiteral(value=1)
    )

    assert a.is_structurally_equivalent(b)


def test_declaration_statement_inequivalent_when_one_has_expression():
    """Test ``None`` vs. set ``expression`` breaks equivalence."""
    name = Identifier("x")
    a = DeclarationStatement(variable_name=name, variable_type=_input_int32())
    b = DeclarationStatement(
        variable_name=name, variable_type=_input_int32(), expression=IntLiteral(value=1)
    )

    assert not a.is_structurally_equivalent(b)


def test_declaration_statement_inequivalent_when_expressions_differ():
    """Test differing expressions break equivalence."""
    name = Identifier("x")
    a = DeclarationStatement(
        variable_name=name, variable_type=_input_int32(), expression=IntLiteral(value=1)
    )
    b = DeclarationStatement(
        variable_name=name, variable_type=_input_int32(), expression=IntLiteral(value=2)
    )

    assert not a.is_structurally_equivalent(b)


def test_declaration_statement_get_visit_children_omits_none_expression():
    """Test the visit walk skips a ``None`` expression and yields only the type."""
    qualified = _input_int32()
    declaration = DeclarationStatement(
        variable_name=Identifier("x"), variable_type=qualified
    )

    assert tuple(declaration.get_visit_children()) == (qualified,)


def test_declaration_statement_get_visit_children_includes_expression_when_present():
    """Test the visit walk yields type and expression when both are set."""
    qualified = _input_int32()
    expression = IntLiteral(value=1)
    declaration = DeclarationStatement(
        variable_name=Identifier("x"),
        variable_type=qualified,
        expression=expression,
    )

    assert tuple(declaration.get_visit_children()) == (qualified, expression)


def test_declaration_statement_round_trips_through_serialization():
    """Test ``DeclarationStatement`` survives a wrapped round-trip."""
    statement = DeclarationStatement(
        variable_name=Identifier("x"),
        variable_type=_input_int32(),
        expression=IntLiteral(value=1),
    )

    restored = Statement.deserialize_from_dict(statement.serialize_to_dict())

    assert isinstance(restored, DeclarationStatement)
    assert restored.expression is not None


def test_declaration_statement_round_trips_when_expression_is_none():
    """Test a ``DeclarationStatement`` with no expression round-trips."""
    statement = DeclarationStatement(
        variable_name=Identifier("x"),
        variable_type=_input_int32(),
    )

    restored = Statement.deserialize_from_dict(statement.serialize_to_dict())

    assert isinstance(restored, DeclarationStatement)
    assert restored.expression is None


# ===========================================================================
# ExpressionStatement
# ===========================================================================


def test_expression_statement_equivalent_with_matching_left_and_right():
    """Test expression statements with matching left and right are equivalent."""
    a = ExpressionStatement(left=IntLiteral(value=1), right=IntLiteral(value=2))
    b = ExpressionStatement(left=IntLiteral(value=1), right=IntLiteral(value=2))

    assert a.is_structurally_equivalent(b)


def test_expression_statement_equivalent_when_both_have_no_left():
    """Test rhs-only expression statements are equivalent when right matches."""
    a = ExpressionStatement(right=IntLiteral(value=1))
    b = ExpressionStatement(right=IntLiteral(value=1))

    assert a.is_structurally_equivalent(b)


def test_expression_statement_inequivalent_when_one_has_left():
    """Test ``None`` left vs. set left breaks equivalence."""
    a = ExpressionStatement(right=IntLiteral(value=1))
    b = ExpressionStatement(left=IntLiteral(value=0), right=IntLiteral(value=1))

    assert not a.is_structurally_equivalent(b)


def test_expression_statement_get_visit_children_omits_none_left():
    """Test the visit walk skips a ``None`` left and yields only right."""
    right = IntLiteral(value=1)
    statement = ExpressionStatement(right=right)

    assert tuple(statement.get_visit_children()) == (right,)


def test_expression_statement_get_visit_children_includes_left_when_present():
    """Test the visit walk yields left and right when both are set."""
    left = IntLiteral(value=0)
    right = IntLiteral(value=1)
    statement = ExpressionStatement(left=left, right=right)

    assert tuple(statement.get_visit_children()) == (left, right)


def test_expression_statement_round_trips_through_serialization():
    """Test ``ExpressionStatement`` survives a wrapped round-trip."""
    statement = ExpressionStatement(left=IntLiteral(value=0), right=IntLiteral(value=1))

    restored = Statement.deserialize_from_dict(statement.serialize_to_dict())

    assert isinstance(restored, ExpressionStatement)


# ===========================================================================
# ReturnStatement
# ===========================================================================


def test_return_statement_equivalent_when_expressions_match():
    """Test return statements with matching expressions are equivalent."""
    a = ReturnStatement(expression=IntLiteral(value=1))
    b = ReturnStatement(expression=IntLiteral(value=1))

    assert a.is_structurally_equivalent(b)


def test_return_statement_inequivalent_when_expressions_differ():
    """Test differing return expressions break equivalence."""
    a = ReturnStatement(expression=IntLiteral(value=1))
    b = ReturnStatement(expression=IntLiteral(value=2))

    assert not a.is_structurally_equivalent(b)


def test_return_statement_get_visit_children_returns_expression():
    """Test the visit walk yields the return expression."""
    expression = IntLiteral(value=1)
    statement = ReturnStatement(expression=expression)

    assert tuple(statement.get_visit_children()) == (expression,)


def test_return_statement_round_trips_through_serialization():
    """Test ``ReturnStatement`` survives a wrapped round-trip."""
    statement = ReturnStatement(expression=IntLiteral(value=1))

    restored = Statement.deserialize_from_dict(statement.serialize_to_dict())

    assert isinstance(restored, ReturnStatement)


# ===========================================================================
# ForAllStatement  (default-name id-freshness, hint = "forall")
# ===========================================================================


def test_forall_statement_default_name_has_forall_hint():
    """Test the default ``ForAllStatement.name`` has hint ``"forall"``."""
    forall = ForAllStatement(index=_identifier_expression("i"))

    assert forall.name.name_hint == "forall"


def test_forall_statement_default_names_have_distinct_ids_across_instances():
    """Test default-named ``ForAllStatement``s get fresh ids per instance."""
    a = ForAllStatement(index=_identifier_expression("i"))
    b = ForAllStatement(index=_identifier_expression("i"))

    assert a.name.id != b.name.id


def test_forall_statement_equivalent_when_name_id_and_index_share_identity():
    """Test forall statements sharing name + index identifiers are equivalent."""
    name = Identifier("loop")
    index = _identifier_expression("i")
    a = ForAllStatement(name=name, index=index)
    b = ForAllStatement(name=name, index=index)

    assert a.is_structurally_equivalent(b)


def test_forall_statement_inequivalent_when_name_ids_differ():
    """Test independently-constructed names break equivalence."""
    a = ForAllStatement(name=Identifier("loop"), index=_identifier_expression("i"))
    b = ForAllStatement(name=Identifier("loop"), index=_identifier_expression("i"))

    assert not a.is_structurally_equivalent(b)


def test_forall_statement_inequivalent_when_body_count_differs():
    """Test differing body counts break equivalence."""
    name = Identifier("loop")
    index = _identifier_expression("i")
    return_stmt = ReturnStatement(expression=IntLiteral(value=1))
    a = ForAllStatement(name=name, index=index)
    b = ForAllStatement(name=name, index=index, body=(return_stmt,))

    assert not a.is_structurally_equivalent(b)


def test_forall_statement_get_visit_children_includes_index_then_body():
    """Test the visit walk yields ``index`` followed by ``body`` statements."""
    index = _identifier_expression("i")
    return_stmt = ReturnStatement(expression=IntLiteral(value=1))
    forall = ForAllStatement(index=index, body=(return_stmt,))

    assert tuple(forall.get_visit_children()) == (index, return_stmt)


def test_forall_statement_round_trips_through_serialization():
    """Test ``ForAllStatement`` survives a wrapped round-trip."""
    statement = ForAllStatement(
        name=Identifier("loop"),
        index=_identifier_expression("i"),
        body=(ReturnStatement(expression=IntLiteral(value=1)),),
    )

    restored = Statement.deserialize_from_dict(statement.serialize_to_dict())

    assert isinstance(restored, ForAllStatement)


# ===========================================================================
# SelectionStatement
# ===========================================================================


def test_selection_statement_equivalent_when_all_components_match():
    """Test selection statements with matching condition and bodies are equivalent."""
    return_stmt = ReturnStatement(expression=IntLiteral(value=1))
    a = SelectionStatement(
        condition=IntLiteral(value=0),
        true_body=(return_stmt,),
        false_body=(return_stmt,),
    )
    b = SelectionStatement(
        condition=IntLiteral(value=0),
        true_body=(return_stmt,),
        false_body=(return_stmt,),
    )

    assert a.is_structurally_equivalent(b)


def test_selection_statement_inequivalent_when_true_body_differs():
    """Test differing ``true_body`` breaks equivalence."""
    a = SelectionStatement(condition=IntLiteral(value=0))
    b = SelectionStatement(
        condition=IntLiteral(value=0),
        true_body=(ReturnStatement(expression=IntLiteral(value=1)),),
    )

    assert not a.is_structurally_equivalent(b)


def test_selection_statement_get_visit_children_walks_condition_then_bodies():
    """Test the visit walk yields condition, true_body, false_body in order."""
    condition = IntLiteral(value=0)
    t = ReturnStatement(expression=IntLiteral(value=1))
    f = ReturnStatement(expression=IntLiteral(value=2))
    selection = SelectionStatement(condition=condition, true_body=(t,), false_body=(f,))

    assert tuple(selection.get_visit_children()) == (condition, t, f)


def test_selection_statement_round_trips_through_serialization():
    """Test ``SelectionStatement`` survives a wrapped round-trip."""
    statement = SelectionStatement(condition=IntLiteral(value=0))

    restored = Statement.deserialize_from_dict(statement.serialize_to_dict())

    assert isinstance(restored, SelectionStatement)


# ===========================================================================
# Procedure  (templates excluded from visitor walk)
# ===========================================================================


def test_procedure_equivalent_when_all_components_match():
    """Test procedures with matching name id, args, and body are equivalent."""
    name = Identifier("p")
    arg = _argument("a")
    body = ReturnStatement(expression=IntLiteral(value=1))
    a = Procedure(name=name, args=(arg,), body=(body,))
    b = Procedure(name=name, args=(arg,), body=(body,))

    assert a.is_structurally_equivalent(b)


def test_procedure_inequivalent_when_name_ids_differ():
    """Test independently-constructed procedure names break equivalence."""
    a = Procedure(name=Identifier("p"))
    b = Procedure(name=Identifier("p"))

    assert not a.is_structurally_equivalent(b)


def test_procedure_get_visit_children_excludes_templates():
    """Test ``Procedure.get_visit_children`` skips templates."""
    template = TemplateDataType(Identifier("T"))
    arg = _argument("a")
    body_stmt = ReturnStatement(expression=IntLiteral(value=1))

    procedure = Procedure(
        name=Identifier("p"),
        templates=(template,),
        args=(arg,),
        body=(body_stmt,),
    )

    children = tuple(procedure.get_visit_children())

    assert template not in children
    assert children == (arg, body_stmt)


def test_procedure_round_trips_through_serialization():
    """Test ``Procedure`` survives a wrapped round-trip."""
    procedure = Procedure(
        name=Identifier("p"),
        args=(_argument("a"),),
        body=(ReturnStatement(expression=IntLiteral(value=1)),),
    )

    restored = Statement.deserialize_from_dict(procedure.serialize_to_dict())

    assert isinstance(restored, Procedure)


@pytest.mark.parametrize("missing_key", ["name", "templates", "args", "body"])
def test_procedure_deserialize_data_rejects_payload_missing_required_key(
    missing_key: str,
):
    """Test missing required keys raise ``DeserializationDictStructureError``."""
    payload = Procedure(name=Identifier("p")).serialize_data_to_dict()
    payload.pop(missing_key)

    with pytest.raises(DeserializationDictStructureError):
        Procedure.deserialize_data_from_dict(payload)


# ===========================================================================
# Operation  (templates excluded from visitor walk)
# ===========================================================================


def test_operation_equivalent_when_all_components_match():
    """Test operations matching on name, args, body, return type are equivalent."""
    name = Identifier("op")
    arg = _argument("a")
    body = ReturnStatement(expression=IntLiteral(value=1))
    return_type = _output_int32()
    a = Operation(name=name, args=(arg,), body=(body,), return_type=return_type)
    b = Operation(name=name, args=(arg,), body=(body,), return_type=return_type)

    assert a.is_structurally_equivalent(b)


def test_operation_inequivalent_when_return_type_differs():
    """Test differing ``return_type`` breaks equivalence."""
    name = Identifier("op")
    a = Operation(name=name, return_type=_output_int32())
    b = Operation(name=name, return_type=_input_int32())

    assert not a.is_structurally_equivalent(b)


def test_operation_get_visit_children_excludes_templates_includes_return_type():
    """Test ``Operation.get_visit_children`` skips templates; ends with return_type."""
    template = TemplateDataType(Identifier("T"))
    arg = _argument("a")
    body_stmt = ReturnStatement(expression=IntLiteral(value=1))
    return_type = _output_int32()

    operation = Operation(
        name=Identifier("op"),
        templates=(template,),
        args=(arg,),
        body=(body_stmt,),
        return_type=return_type,
    )

    children = tuple(operation.get_visit_children())

    assert template not in children
    assert children == (arg, body_stmt, return_type)


def test_operation_round_trips_through_serialization():
    """Test ``Operation`` survives a wrapped round-trip."""
    operation = Operation(
        name=Identifier("op"),
        return_type=_output_int32(),
    )

    restored = Statement.deserialize_from_dict(operation.serialize_to_dict())

    assert isinstance(restored, Operation)


def test_operation_deserialize_data_rejects_missing_return_type():
    """Test missing ``return_type`` raises ``DeserializationDictStructureError``."""
    payload = Operation(
        name=Identifier("op"), return_type=_output_int32()
    ).serialize_data_to_dict()
    payload.pop("return_type")

    with pytest.raises(DeserializationDictStructureError):
        Operation.deserialize_data_from_dict(payload)


# ===========================================================================
# Native
# ===========================================================================


def test_native_equivalent_when_name_id_and_args_match():
    """Test natives with matching name id and args are equivalent."""
    name = Identifier("n")
    arg = _argument("a")

    assert Native(name=name, args=(arg,)).is_structurally_equivalent(
        Native(name=name, args=(arg,))
    )


def test_native_inequivalent_when_args_count_differs():
    """Test differing argument count breaks equivalence."""
    name = Identifier("n")
    a = Native(name=name, args=())
    b = Native(name=name, args=(_argument("a"),))

    assert not a.is_structurally_equivalent(b)


def test_native_get_visit_children_returns_args():
    """Test ``Native.get_visit_children`` walks only its args."""
    arg = _argument("a")
    native = Native(name=Identifier("n"), args=(arg,))

    assert tuple(native.get_visit_children()) == (arg,)


def test_native_round_trips_through_serialization():
    """Test ``Native`` survives a wrapped round-trip."""
    native = Native(name=Identifier("n"), args=(_argument("a"),))

    restored = Statement.deserialize_from_dict(native.serialize_to_dict())

    assert isinstance(restored, Native)
