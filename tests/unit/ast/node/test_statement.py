"""Tests for ``fhy_lang.ast.node.statement``.

Covers every concrete statement node: construction, structural equivalence
(id-based), visitor contracts (including the documented exclusion of
templates from the AST walk), serialization round-trips, and
deserialization error paths.
"""

import pytest
from fhy_core import (
    DeserializationDictStructureError,
    Identifier,
    TemplateDataType,
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


def _make_identifier_expression(name: str = "x") -> IdentifierExpression:
    """Build an ``IdentifierExpression`` whose identifier has the given name."""
    return IdentifierExpression(identifier=Identifier(name))


def _make_argument(name: str, qualified_type: QualifiedType) -> Argument:
    """Build an ``Argument`` with the given name and qualified type."""
    return Argument(name=Identifier(name), qualified_type=qualified_type)


# ===========================================================================
# Argument
# ===========================================================================


def test_argument_equivalent_when_name_id_and_type_match(input_int32: QualifiedType):
    """Test arguments sharing a name id and equivalent type are equivalent."""
    name = Identifier("a")
    a = Argument(name=name, qualified_type=input_int32)
    b = Argument(name=name, qualified_type=input_int32)

    assert a.is_structurally_equivalent(b)


def test_argument_inequivalent_when_name_ids_differ(input_int32: QualifiedType):
    """Test independently-constructed names break equivalence."""
    a = Argument(name=Identifier("a"), qualified_type=input_int32)
    b = Argument(name=Identifier("a"), qualified_type=input_int32)

    assert not a.is_structurally_equivalent(b)


def test_argument_inequivalent_when_qualified_type_differs(
    input_int32: QualifiedType, output_int32: QualifiedType
):
    """Test differing qualified type breaks equivalence."""
    name = Identifier("a")
    a = Argument(name=name, qualified_type=input_int32)
    b = Argument(name=name, qualified_type=output_int32)

    assert not a.is_structurally_equivalent(b)


def test_argument_get_visit_children_returns_qualified_type(
    input_int32: QualifiedType,
):
    """Test ``Argument.get_visit_children`` walks only the qualified type."""
    argument = Argument(name=Identifier("a"), qualified_type=input_int32)

    assert tuple(argument.get_visit_children()) == (input_int32,)


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


def test_declaration_statement_equivalent_with_no_expression(
    input_int32: QualifiedType,
):
    """Test declarations without expressions are equivalent when name + type match."""
    name = Identifier("x")
    a = DeclarationStatement(variable_name=name, variable_type=input_int32)
    b = DeclarationStatement(variable_name=name, variable_type=input_int32)

    assert a.is_structurally_equivalent(b)


def test_declaration_statement_equivalent_with_matching_expression(
    input_int32: QualifiedType,
):
    """Test declarations with matching expressions are equivalent."""
    name = Identifier("x")
    a = DeclarationStatement(
        variable_name=name, variable_type=input_int32, expression=IntLiteral(value=1)
    )
    b = DeclarationStatement(
        variable_name=name, variable_type=input_int32, expression=IntLiteral(value=1)
    )

    assert a.is_structurally_equivalent(b)


def test_declaration_statement_inequivalent_when_one_has_expression(
    input_int32: QualifiedType,
):
    """Test ``None`` vs. set ``expression`` breaks equivalence."""
    name = Identifier("x")
    a = DeclarationStatement(variable_name=name, variable_type=input_int32)
    b = DeclarationStatement(
        variable_name=name, variable_type=input_int32, expression=IntLiteral(value=1)
    )

    assert not a.is_structurally_equivalent(b)


def test_declaration_statement_inequivalent_when_expressions_differ(
    input_int32: QualifiedType,
):
    """Test differing expressions break equivalence."""
    name = Identifier("x")
    a = DeclarationStatement(
        variable_name=name, variable_type=input_int32, expression=IntLiteral(value=1)
    )
    b = DeclarationStatement(
        variable_name=name, variable_type=input_int32, expression=IntLiteral(value=2)
    )

    assert not a.is_structurally_equivalent(b)


def test_declaration_statement_get_visit_children_omits_none_expression(
    input_int32: QualifiedType,
):
    """Test the visit walk skips a ``None`` expression and yields only the type."""
    declaration = DeclarationStatement(
        variable_name=Identifier("x"), variable_type=input_int32
    )

    assert tuple(declaration.get_visit_children()) == (input_int32,)


def test_declaration_statement_get_visit_children_includes_expression_when_present(
    input_int32: QualifiedType,
):
    """Test the visit walk yields type and expression when both are set."""
    expression = IntLiteral(value=1)
    declaration = DeclarationStatement(
        variable_name=Identifier("x"),
        variable_type=input_int32,
        expression=expression,
    )

    assert tuple(declaration.get_visit_children()) == (input_int32, expression)


def test_declaration_statement_round_trips_through_serialization(
    input_int32: QualifiedType,
):
    """Test ``DeclarationStatement`` survives a wrapped round-trip."""
    statement = DeclarationStatement(
        variable_name=Identifier("x"),
        variable_type=input_int32,
        expression=IntLiteral(value=1),
    )

    restored = Statement.deserialize_from_dict(statement.serialize_to_dict())

    assert isinstance(restored, DeclarationStatement)
    assert restored.expression is not None


def test_declaration_statement_round_trips_when_expression_is_none(
    input_int32: QualifiedType,
):
    """Test a ``DeclarationStatement`` with no expression round-trips."""
    statement = DeclarationStatement(
        variable_name=Identifier("x"),
        variable_type=input_int32,
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
    forall = ForAllStatement(index=_make_identifier_expression("i"))

    assert forall.name.name_hint == "forall"


def test_forall_statement_default_names_have_distinct_ids_across_instances():
    """Test default-named ``ForAllStatement``s get fresh ids per instance."""
    a = ForAllStatement(index=_make_identifier_expression("i"))
    b = ForAllStatement(index=_make_identifier_expression("i"))

    assert a.name.id != b.name.id


def test_forall_statement_equivalent_when_name_id_and_index_share_identity():
    """Test forall statements sharing name + index identifiers are equivalent."""
    name = Identifier("loop")
    index = _make_identifier_expression("i")
    a = ForAllStatement(name=name, index=index)
    b = ForAllStatement(name=name, index=index)

    assert a.is_structurally_equivalent(b)


def test_forall_statement_inequivalent_when_name_ids_differ():
    """Test independently-constructed names break equivalence."""
    a = ForAllStatement(name=Identifier("loop"), index=_make_identifier_expression("i"))
    b = ForAllStatement(name=Identifier("loop"), index=_make_identifier_expression("i"))

    assert not a.is_structurally_equivalent(b)


def test_forall_statement_inequivalent_when_body_count_differs():
    """Test differing body counts break equivalence."""
    name = Identifier("loop")
    index = _make_identifier_expression("i")
    return_stmt = ReturnStatement(expression=IntLiteral(value=1))
    a = ForAllStatement(name=name, index=index)
    b = ForAllStatement(name=name, index=index, body=(return_stmt,))

    assert not a.is_structurally_equivalent(b)


def test_forall_statement_get_visit_children_includes_index_then_body():
    """Test the visit walk yields ``index`` followed by ``body`` statements."""
    index = _make_identifier_expression("i")
    return_stmt = ReturnStatement(expression=IntLiteral(value=1))
    forall = ForAllStatement(index=index, body=(return_stmt,))

    assert tuple(forall.get_visit_children()) == (index, return_stmt)


def test_forall_statement_round_trips_through_serialization():
    """Test ``ForAllStatement`` survives a wrapped round-trip."""
    statement = ForAllStatement(
        name=Identifier("loop"),
        index=_make_identifier_expression("i"),
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


def test_procedure_equivalent_when_all_components_match(input_int32: QualifiedType):
    """Test procedures with matching name id, args, and body are equivalent."""
    name = Identifier("p")
    arg = _make_argument("a", input_int32)
    body = ReturnStatement(expression=IntLiteral(value=1))
    a = Procedure(name=name, args=(arg,), body=(body,))
    b = Procedure(name=name, args=(arg,), body=(body,))

    assert a.is_structurally_equivalent(b)


def test_procedure_inequivalent_when_name_ids_differ():
    """Test independently-constructed procedure names break equivalence."""
    a = Procedure(name=Identifier("p"))
    b = Procedure(name=Identifier("p"))

    assert not a.is_structurally_equivalent(b)


def test_procedure_get_visit_children_excludes_templates(input_int32: QualifiedType):
    """Test ``Procedure.get_visit_children`` skips templates."""
    template = TemplateDataType(Identifier("T"))
    arg = _make_argument("a", input_int32)
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


def test_procedure_round_trips_through_serialization(input_int32: QualifiedType):
    """Test ``Procedure`` survives a wrapped round-trip."""
    procedure = Procedure(
        name=Identifier("p"),
        args=(_make_argument("a", input_int32),),
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


def test_operation_equivalent_when_all_components_match(
    input_int32: QualifiedType, output_int32: QualifiedType
):
    """Test operations matching on name, args, body, return type are equivalent."""
    name = Identifier("op")
    arg = _make_argument("a", input_int32)
    body = ReturnStatement(expression=IntLiteral(value=1))
    a = Operation(name=name, args=(arg,), body=(body,), return_type=output_int32)
    b = Operation(name=name, args=(arg,), body=(body,), return_type=output_int32)

    assert a.is_structurally_equivalent(b)


def test_operation_inequivalent_when_return_type_differs(
    input_int32: QualifiedType, output_int32: QualifiedType
):
    """Test differing ``return_type`` breaks equivalence."""
    name = Identifier("op")
    a = Operation(name=name, return_type=output_int32)
    b = Operation(name=name, return_type=input_int32)

    assert not a.is_structurally_equivalent(b)


def test_operation_get_visit_children_excludes_templates_includes_return_type(
    input_int32: QualifiedType, output_int32: QualifiedType
):
    """Test ``Operation.get_visit_children`` skips templates; ends with return_type."""
    template = TemplateDataType(Identifier("T"))
    arg = _make_argument("a", input_int32)
    body_stmt = ReturnStatement(expression=IntLiteral(value=1))

    operation = Operation(
        name=Identifier("op"),
        templates=(template,),
        args=(arg,),
        body=(body_stmt,),
        return_type=output_int32,
    )

    children = tuple(operation.get_visit_children())

    assert template not in children
    assert children == (arg, body_stmt, output_int32)


def test_operation_round_trips_through_serialization(output_int32: QualifiedType):
    """Test ``Operation`` survives a wrapped round-trip."""
    operation = Operation(name=Identifier("op"), return_type=output_int32)

    restored = Statement.deserialize_from_dict(operation.serialize_to_dict())

    assert isinstance(restored, Operation)


def test_operation_deserialize_data_rejects_missing_return_type(
    output_int32: QualifiedType,
):
    """Test missing ``return_type`` raises ``DeserializationDictStructureError``."""
    payload = Operation(
        name=Identifier("op"), return_type=output_int32
    ).serialize_data_to_dict()
    payload.pop("return_type")

    with pytest.raises(DeserializationDictStructureError):
        Operation.deserialize_data_from_dict(payload)


# ===========================================================================
# Native
# ===========================================================================


def test_native_equivalent_when_name_id_and_args_match(input_int32: QualifiedType):
    """Test natives with matching name id and args are equivalent."""
    name = Identifier("n")
    arg = _make_argument("a", input_int32)

    assert Native(name=name, args=(arg,)).is_structurally_equivalent(
        Native(name=name, args=(arg,))
    )


def test_native_inequivalent_when_args_count_differs(input_int32: QualifiedType):
    """Test differing argument count breaks equivalence."""
    name = Identifier("n")
    a = Native(name=name, args=())
    b = Native(name=name, args=(_make_argument("a", input_int32),))

    assert not a.is_structurally_equivalent(b)


def test_native_get_visit_children_returns_args(input_int32: QualifiedType):
    """Test ``Native.get_visit_children`` walks only its args."""
    arg = _make_argument("a", input_int32)
    native = Native(name=Identifier("n"), args=(arg,))

    assert tuple(native.get_visit_children()) == (arg,)


def test_native_round_trips_through_serialization(input_int32: QualifiedType):
    """Test ``Native`` survives a wrapped round-trip."""
    native = Native(name=Identifier("n"), args=(_make_argument("a", input_int32),))

    restored = Statement.deserialize_from_dict(native.serialize_to_dict())

    assert isinstance(restored, Native)


# ===========================================================================
# Cross-type rejection: every concrete statement rejects an unrelated node
# from ``is_structurally_equivalent``. Pinned per-class because each subtype
# implements its own equivalence check.
# ===========================================================================


def test_argument_is_inequivalent_with_non_argument_node(input_int32: QualifiedType):
    """Test ``Argument.is_structurally_equivalent`` rejects an unrelated node."""
    argument = Argument(name=Identifier("x"), qualified_type=input_int32)

    assert not argument.is_structurally_equivalent(IntLiteral(value=0))


def test_import_is_inequivalent_with_non_import_node():
    """Test ``Import.is_structurally_equivalent`` rejects an unrelated node."""
    assert not Import(name=Identifier("foo")).is_structurally_equivalent(
        IntLiteral(value=0)
    )


def test_declaration_statement_is_inequivalent_with_non_declaration_node(
    input_int32: QualifiedType,
):
    """Test ``DeclarationStatement`` equivalence rejects an unrelated node."""
    declaration = DeclarationStatement(
        variable_name=Identifier("x"), variable_type=input_int32
    )

    assert not declaration.is_structurally_equivalent(IntLiteral(value=0))


def test_expression_statement_is_inequivalent_with_non_statement_node():
    """Test ``ExpressionStatement`` equivalence rejects an unrelated node."""
    statement = ExpressionStatement(right=IntLiteral(value=1))

    assert not statement.is_structurally_equivalent(IntLiteral(value=1))


def test_return_statement_is_inequivalent_with_non_return_node():
    """Test ``ReturnStatement`` equivalence rejects an unrelated node."""
    statement = ReturnStatement(expression=IntLiteral(value=1))

    assert not statement.is_structurally_equivalent(IntLiteral(value=1))


def test_forall_statement_is_inequivalent_with_non_forall_node():
    """Test ``ForAllStatement`` equivalence rejects an unrelated node."""
    forall = ForAllStatement(index=_make_identifier_expression("i"))

    assert not forall.is_structurally_equivalent(IntLiteral(value=0))


def test_selection_statement_is_inequivalent_with_non_selection_node():
    """Test ``SelectionStatement`` equivalence rejects an unrelated node."""
    selection = SelectionStatement(condition=_make_identifier_expression("c"))

    assert not selection.is_structurally_equivalent(IntLiteral(value=0))


def test_procedure_is_inequivalent_with_non_procedure_node():
    """Test ``Procedure.is_structurally_equivalent`` rejects an unrelated node."""
    assert not Procedure(name=Identifier("p")).is_structurally_equivalent(
        IntLiteral(value=0)
    )


def test_operation_is_inequivalent_with_non_operation_node(
    output_int32: QualifiedType,
):
    """Test ``Operation.is_structurally_equivalent`` rejects an unrelated node."""
    operation = Operation(name=Identifier("op"), return_type=output_int32)

    assert not operation.is_structurally_equivalent(IntLiteral(value=0))


def test_native_is_inequivalent_with_non_native_node():
    """Test ``Native.is_structurally_equivalent`` rejects an unrelated node."""
    assert not Native(name=Identifier("n")).is_structurally_equivalent(
        IntLiteral(value=0)
    )


# ===========================================================================
# Per-class field-change inequivalence cases that pin each contributing field
# beyond the "name id" check already in the file.
# ===========================================================================


def test_declaration_statement_inequivalent_when_variable_names_differ(
    input_int32: QualifiedType,
):
    """Test independently-constructed variable names break equivalence."""
    a = DeclarationStatement(variable_name=Identifier("x"), variable_type=input_int32)
    b = DeclarationStatement(variable_name=Identifier("y"), variable_type=input_int32)

    assert not a.is_structurally_equivalent(b)


def test_declaration_statement_inequivalent_when_variable_types_differ(
    input_int32: QualifiedType, output_int32: QualifiedType
):
    """Test differing variable types break equivalence."""
    name = Identifier("x")
    a = DeclarationStatement(variable_name=name, variable_type=input_int32)
    b = DeclarationStatement(variable_name=name, variable_type=output_int32)

    assert not a.is_structurally_equivalent(b)


def test_expression_statement_inequivalent_when_lefts_differ():
    """Test differing left operands break equivalence."""
    a = ExpressionStatement(left=IntLiteral(value=0), right=IntLiteral(value=1))
    b = ExpressionStatement(left=IntLiteral(value=2), right=IntLiteral(value=1))

    assert not a.is_structurally_equivalent(b)


def test_expression_statement_inequivalent_when_rights_differ():
    """Test differing right operands break equivalence."""
    a = ExpressionStatement(right=IntLiteral(value=1))
    b = ExpressionStatement(right=IntLiteral(value=2))

    assert not a.is_structurally_equivalent(b)


def test_forall_statement_inequivalent_when_index_differs():
    """Test differing forall index expressions break equivalence."""
    name = Identifier("loop")
    a = ForAllStatement(name=name, index=_make_identifier_expression("i"))
    b = ForAllStatement(name=name, index=_make_identifier_expression("j"))

    assert not a.is_structurally_equivalent(b)


def test_forall_statement_inequivalent_when_body_content_differs():
    """Test differing forall body contents break equivalence (same count)."""
    name = Identifier("loop")
    index = _make_identifier_expression("i")
    a = ForAllStatement(
        name=name,
        index=index,
        body=(ReturnStatement(expression=IntLiteral(value=1)),),
    )
    b = ForAllStatement(
        name=name,
        index=index,
        body=(ReturnStatement(expression=IntLiteral(value=2)),),
    )

    assert not a.is_structurally_equivalent(b)


def test_selection_statement_inequivalent_when_condition_differs():
    """Test differing selection conditions break equivalence."""
    a = SelectionStatement(condition=IntLiteral(value=0))
    b = SelectionStatement(condition=IntLiteral(value=1))

    assert not a.is_structurally_equivalent(b)


def test_selection_statement_inequivalent_when_true_body_content_differs():
    """Test differing true-body contents break equivalence (same count)."""
    condition = IntLiteral(value=0)
    a = SelectionStatement(
        condition=condition,
        true_body=(ReturnStatement(expression=IntLiteral(value=1)),),
    )
    b = SelectionStatement(
        condition=condition,
        true_body=(ReturnStatement(expression=IntLiteral(value=2)),),
    )

    assert not a.is_structurally_equivalent(b)


def test_selection_statement_inequivalent_when_false_body_count_differs():
    """Test differing false-body counts break equivalence."""
    condition = IntLiteral(value=0)
    a = SelectionStatement(condition=condition)
    b = SelectionStatement(
        condition=condition,
        false_body=(ReturnStatement(expression=IntLiteral(value=0)),),
    )

    assert not a.is_structurally_equivalent(b)


def test_selection_statement_inequivalent_when_false_body_content_differs():
    """Test differing false-body contents break equivalence (same count)."""
    condition = IntLiteral(value=0)
    a = SelectionStatement(
        condition=condition,
        false_body=(ReturnStatement(expression=IntLiteral(value=0)),),
    )
    b = SelectionStatement(
        condition=condition,
        false_body=(ReturnStatement(expression=IntLiteral(value=1)),),
    )

    assert not a.is_structurally_equivalent(b)


def test_procedure_inequivalent_when_template_count_differs():
    """Test differing template counts break Procedure equivalence."""
    name = Identifier("p")
    a = Procedure(name=name, templates=(TemplateDataType(Identifier("T")),))
    b = Procedure(name=name, templates=())

    assert not a.is_structurally_equivalent(b)


def test_procedure_inequivalent_when_template_contents_differ():
    """Test differing template contents break Procedure equivalence."""
    name = Identifier("p")
    a = Procedure(name=name, templates=(TemplateDataType(Identifier("T")),))
    b = Procedure(name=name, templates=(TemplateDataType(Identifier("U")),))

    assert not a.is_structurally_equivalent(b)


def test_procedure_inequivalent_when_arg_count_differs(input_int32: QualifiedType):
    """Test differing argument counts break Procedure equivalence."""
    name = Identifier("p")
    arg = _make_argument("a", input_int32)
    a = Procedure(name=name, args=(arg,))
    b = Procedure(name=name, args=())

    assert not a.is_structurally_equivalent(b)


def test_procedure_inequivalent_when_arg_contents_differ(input_int32: QualifiedType):
    """Test differing argument contents break Procedure equivalence."""
    name = Identifier("p")
    a = Procedure(name=name, args=(_make_argument("a", input_int32),))
    b = Procedure(name=name, args=(_make_argument("b", input_int32),))

    assert not a.is_structurally_equivalent(b)


def test_procedure_inequivalent_when_body_count_differs():
    """Test differing body counts break Procedure equivalence."""
    name = Identifier("p")
    a = Procedure(name=name, body=(ReturnStatement(expression=IntLiteral(value=0)),))
    b = Procedure(name=name, body=())

    assert not a.is_structurally_equivalent(b)


def test_procedure_inequivalent_when_body_contents_differ():
    """Test differing body contents break Procedure equivalence."""
    name = Identifier("p")
    a = Procedure(name=name, body=(ReturnStatement(expression=IntLiteral(value=0)),))
    b = Procedure(name=name, body=(ReturnStatement(expression=IntLiteral(value=1)),))

    assert not a.is_structurally_equivalent(b)


def test_operation_inequivalent_when_name_ids_differ(output_int32: QualifiedType):
    """Test independently-constructed operation names break equivalence."""
    a = Operation(name=Identifier("op"), return_type=output_int32)
    b = Operation(name=Identifier("op"), return_type=output_int32)

    assert not a.is_structurally_equivalent(b)


def test_operation_inequivalent_when_template_count_differs(
    output_int32: QualifiedType,
):
    """Test differing template counts break Operation equivalence."""
    name = Identifier("op")
    a = Operation(
        name=name,
        templates=(TemplateDataType(Identifier("T")),),
        return_type=output_int32,
    )
    b = Operation(name=name, templates=(), return_type=output_int32)

    assert not a.is_structurally_equivalent(b)


def test_operation_inequivalent_when_template_contents_differ(
    output_int32: QualifiedType,
):
    """Test differing template contents break Operation equivalence."""
    name = Identifier("op")
    a = Operation(
        name=name,
        templates=(TemplateDataType(Identifier("T")),),
        return_type=output_int32,
    )
    b = Operation(
        name=name,
        templates=(TemplateDataType(Identifier("U")),),
        return_type=output_int32,
    )

    assert not a.is_structurally_equivalent(b)


def test_operation_inequivalent_when_arg_count_differs(
    input_int32: QualifiedType, output_int32: QualifiedType
):
    """Test differing argument counts break Operation equivalence."""
    name = Identifier("op")
    arg = _make_argument("a", input_int32)
    a = Operation(name=name, args=(arg,), return_type=output_int32)
    b = Operation(name=name, args=(), return_type=output_int32)

    assert not a.is_structurally_equivalent(b)


def test_operation_inequivalent_when_arg_contents_differ(
    input_int32: QualifiedType, output_int32: QualifiedType
):
    """Test differing argument contents break Operation equivalence."""
    name = Identifier("op")
    a = Operation(
        name=name,
        args=(_make_argument("a", input_int32),),
        return_type=output_int32,
    )
    b = Operation(
        name=name,
        args=(_make_argument("b", input_int32),),
        return_type=output_int32,
    )

    assert not a.is_structurally_equivalent(b)


def test_operation_inequivalent_when_body_count_differs(output_int32: QualifiedType):
    """Test differing body counts break Operation equivalence."""
    name = Identifier("op")
    a = Operation(
        name=name,
        body=(ReturnStatement(expression=IntLiteral(value=0)),),
        return_type=output_int32,
    )
    b = Operation(name=name, body=(), return_type=output_int32)

    assert not a.is_structurally_equivalent(b)


def test_operation_inequivalent_when_body_contents_differ(
    output_int32: QualifiedType,
):
    """Test differing body contents break Operation equivalence."""
    name = Identifier("op")
    a = Operation(
        name=name,
        body=(ReturnStatement(expression=IntLiteral(value=0)),),
        return_type=output_int32,
    )
    b = Operation(
        name=name,
        body=(ReturnStatement(expression=IntLiteral(value=1)),),
        return_type=output_int32,
    )

    assert not a.is_structurally_equivalent(b)


def test_native_inequivalent_when_name_ids_differ():
    """Test independently-constructed native names break equivalence."""
    a = Native(name=Identifier("n"))
    b = Native(name=Identifier("n"))

    assert not a.is_structurally_equivalent(b)


def test_native_inequivalent_when_arg_contents_differ(input_int32: QualifiedType):
    """Test differing argument contents break Native equivalence (same count)."""
    name = Identifier("n")
    a = Native(name=name, args=(_make_argument("a", input_int32),))
    b = Native(name=name, args=(_make_argument("b", input_int32),))

    assert not a.is_structurally_equivalent(b)


# ===========================================================================
# Identifier accessor tests for Function-like statements
# ===========================================================================


def test_procedure_get_identifier_returns_name():
    """Test ``Procedure.get_identifier`` returns the constructor name."""
    name = Identifier("p")
    procedure = Procedure(name=name)

    assert procedure.get_identifier() is name


def test_forall_statement_get_identifier_returns_constructor_name():
    """Test ``ForAllStatement.get_identifier`` returns the constructor name."""
    name = Identifier("loop")
    forall = ForAllStatement(name=name, index=_make_identifier_expression("i"))

    assert forall.get_identifier() is name


# ===========================================================================
# Deserialization rejects an empty dict for every concrete statement that
# is not already covered by a missing-key test above.
# ===========================================================================


@pytest.mark.parametrize(
    "node_class",
    [
        DeclarationStatement,
        ExpressionStatement,
        ForAllStatement,
        SelectionStatement,
        ReturnStatement,
        Native,
        Operation,
    ],
)
def test_statement_deserialize_data_from_dict_rejects_empty_payload(
    node_class: type,
):
    """Test concrete statement classes reject an empty deserialization payload."""
    with pytest.raises(DeserializationDictStructureError):
        node_class.deserialize_data_from_dict({})


# ===========================================================================
# Empty-container round-trips and partial-branch Selection round-trips
# ===========================================================================


def test_empty_procedure_round_trip_preserves_emptiness():
    """Test a ``Procedure`` with no templates/args/body round-trips intact."""
    procedure = Procedure(name=Identifier("p"))

    restored = Statement.deserialize_from_dict(procedure.serialize_to_dict())

    assert isinstance(restored, Procedure)
    assert restored.templates == ()
    assert restored.args == ()
    assert restored.body == ()


def test_empty_operation_round_trip_preserves_emptiness(output_int32: QualifiedType):
    """Test an ``Operation`` with no templates/args/body round-trips intact."""
    operation = Operation(name=Identifier("op"), return_type=output_int32)

    restored = Statement.deserialize_from_dict(operation.serialize_to_dict())

    assert isinstance(restored, Operation)
    assert restored.templates == ()
    assert restored.args == ()
    assert restored.body == ()


def test_empty_native_round_trip_preserves_emptiness():
    """Test a ``Native`` with no args round-trips intact."""
    native = Native(name=Identifier("n"))

    restored = Statement.deserialize_from_dict(native.serialize_to_dict())

    assert isinstance(restored, Native)
    assert restored.args == ()


def test_empty_forall_statement_round_trip_preserves_emptiness():
    """Test a ``ForAllStatement`` with no body round-trips intact."""
    forall = ForAllStatement(index=_make_identifier_expression("i"))

    restored = Statement.deserialize_from_dict(forall.serialize_to_dict())

    assert isinstance(restored, ForAllStatement)
    assert restored.body == ()


def test_empty_selection_statement_round_trip_preserves_emptiness():
    """Test a ``SelectionStatement`` with empty bodies round-trips intact."""
    selection = SelectionStatement(condition=_make_identifier_expression("c"))

    restored = Statement.deserialize_from_dict(selection.serialize_to_dict())

    assert isinstance(restored, SelectionStatement)
    assert restored.true_body == ()
    assert restored.false_body == ()


def test_selection_statement_with_only_true_branch_round_trips():
    """Test a ``SelectionStatement`` with a populated true branch round-trips."""
    selection = SelectionStatement(
        condition=_make_identifier_expression("c"),
        true_body=(ReturnStatement(expression=IntLiteral(value=1)),),
    )

    restored = Statement.deserialize_from_dict(selection.serialize_to_dict())

    assert isinstance(restored, SelectionStatement)
    assert restored.false_body == ()
    assert len(restored.true_body) == 1


def test_selection_statement_with_only_false_branch_round_trips():
    """Test a ``SelectionStatement`` with a populated false branch round-trips."""
    selection = SelectionStatement(
        condition=_make_identifier_expression("c"),
        false_body=(ReturnStatement(expression=IntLiteral(value=0)),),
    )

    restored = Statement.deserialize_from_dict(selection.serialize_to_dict())

    assert isinstance(restored, SelectionStatement)
    assert restored.true_body == ()
    assert len(restored.false_body) == 1


def test_expression_statement_round_trips_when_left_is_none():
    """Test an ``ExpressionStatement`` with no left round-trips with ``left`` None."""
    statement = ExpressionStatement(right=IntLiteral(value=1))

    restored = Statement.deserialize_from_dict(statement.serialize_to_dict())

    assert isinstance(restored, ExpressionStatement)
    assert restored.left is None


# ===========================================================================
# Visit-children ordering with multiple elements per slot
# ===========================================================================


def test_procedure_visit_children_orders_multiple_args_then_multiple_body(
    input_int32: QualifiedType,
):
    """Test ``Procedure`` preserves relative order across many args/body items."""
    arg_a = _make_argument("a", input_int32)
    arg_b = _make_argument("b", input_int32)
    stmt_x = ReturnStatement(expression=IntLiteral(value=0))
    stmt_y = ReturnStatement(expression=IntLiteral(value=1))
    procedure = Procedure(
        name=Identifier("p"),
        args=(arg_a, arg_b),
        body=(stmt_x, stmt_y),
    )

    assert tuple(procedure.get_visit_children()) == (arg_a, arg_b, stmt_x, stmt_y)


def test_operation_visit_children_orders_args_body_then_return_type(
    input_int32: QualifiedType, output_int32: QualifiedType
):
    """Test ``Operation`` orders multiple args, then body items, then return type."""
    arg_a = _make_argument("a", input_int32)
    arg_b = _make_argument("b", input_int32)
    stmt_x = ReturnStatement(expression=IntLiteral(value=0))
    stmt_y = ReturnStatement(expression=IntLiteral(value=1))
    operation = Operation(
        name=Identifier("op"),
        args=(arg_a, arg_b),
        body=(stmt_x, stmt_y),
        return_type=output_int32,
    )

    assert tuple(operation.get_visit_children()) == (
        arg_a,
        arg_b,
        stmt_x,
        stmt_y,
        output_int32,
    )


def test_selection_visit_children_orders_condition_true_then_false():
    """Test ``SelectionStatement`` visit order is condition, true, then false bodies."""
    condition = _make_identifier_expression("c")
    t_a = ReturnStatement(expression=IntLiteral(value=1))
    t_b = ReturnStatement(expression=IntLiteral(value=2))
    f_a = ReturnStatement(expression=IntLiteral(value=3))
    f_b = ReturnStatement(expression=IntLiteral(value=4))
    selection = SelectionStatement(
        condition=condition, true_body=(t_a, t_b), false_body=(f_a, f_b)
    )

    assert tuple(selection.get_visit_children()) == (condition, t_a, t_b, f_a, f_b)


# ===========================================================================
# Nested ForAll round-trip
# ===========================================================================


def test_nested_forall_statement_round_trips():
    """Test a ``ForAllStatement`` nested inside another round-trips intact."""
    inner = ForAllStatement(
        index=_make_identifier_expression("j"),
        body=(ReturnStatement(expression=IntLiteral(value=1)),),
    )
    outer = ForAllStatement(
        index=_make_identifier_expression("i"),
        body=(inner,),
    )

    restored = Statement.deserialize_from_dict(outer.serialize_to_dict())

    assert isinstance(restored, ForAllStatement)
    assert isinstance(restored.body[0], ForAllStatement)
