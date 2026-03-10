"""Pytest Unit Test Fixtures and Utilities.

Fixtures present in this file are innately available to all modules present within this
directory. This is not true of Subdirectories, which will need it's own conftest.py
file.

"""

from collections.abc import Callable, Generator
from typing import TypeVar

import pytest
from fhy.lang.ast import (
    Argument,
    ArrayAccessExpression,
    ASTNode,
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
from fhy.lang.ast.span import Source, Span
from fhy.lang.converter.from_fhy_source import from_fhy_source as fhy_source
from fhy_core import (
    CoreDataType,
    Identifier,
    IndexType,
    NumericalType,
    PrimitiveDataType,
    TemplateDataType,
    TupleType,
    TypeQualifier,
    get_logger,
)
from fhy_core import (
    Expression as CoreExpression,
)
from fhy_core import (
    IdentifierExpression as CoreIdentifierExpression,
)
from fhy_core import (
    LiteralExpression as CoreLiteralExpression,
)

logger = get_logger(__name__)
logger.setLevel(10)
TLiteral = TypeVar("TLiteral", IntLiteral, FloatLiteral, ComplexLiteral)
T = TypeVar("T")
fixture_node_names: list[str] = []


def add_fixture_node(f: T) -> T:
    """Simple wrapper to collect statically constructed ast node fixtures.

    This is used for ease of testing where we can instead grab the `fixture_node_names`
    variable from this module.

    """
    fixture_node_names.append(f.__name__)

    return f


@pytest.fixture
def construct_ast() -> Callable[[str], ASTNode]:
    """Construct an Abstract Syntax Tree (AST) from a raw text file source."""

    def _inner(source: str) -> ASTNode:
        return fhy_source(source, log=logger)

    return _inner


@pytest.fixture
def construct_id() -> Generator[Callable[[str], tuple[dict, Span]], None, None]:
    def inner(name: str) -> tuple[dict, Span]:
        """Build an Identifier from a Name Hint."""
        _id = Identifier(name)
        obj = dict(cls_name="Identifier", attributes=dict(name_hint=name, _id=_id._id))

        return obj, _id

    yield inner


@pytest.fixture
def core_literal_expression() -> (
    Callable[[int | float | complex], tuple[dict, TLiteral]]
):
    def _build(value: int | float | complex):  # noqa: PYI041
        _literal: TLiteral
        result = dict(value=value)
        _literal = CoreLiteralExpression(value=value)

        name = _literal.__class__.__qualname__
        obj = dict(cls_name=name, attributes=dict(value=result))
        return obj, _literal

    return _build


@add_fixture_node
@pytest.fixture
def span_node() -> tuple[dict, Span]:
    a, b, c, d, e = 0, 10, 0, 10, "test"
    obj = dict(
        cls_name="Span",
        attributes=dict(
            start_line=a,
            end_line=b,
            start_column=c,
            end_column=d,
            source=dict(cls_name="Source", attributes=dict(namespace=e)),
        ),
    )
    sp = Span(a, b, c, d, Source(e))

    return obj, sp


@pytest.fixture
def literals(span_node) -> Callable[[int | float | complex], tuple[dict, TLiteral]]:
    def _build(value: int | float | complex):  # noqa: PYI041
        span_obj, span_cls = span_node
        _literal: TLiteral
        if isinstance(value, complex):
            result = dict(real=value.real, imag=value.imag)
            _literal = ComplexLiteral(span=span_cls, value=value)
        elif isinstance(value, float):
            result = value
            _literal = FloatLiteral(span=span_cls, value=value)
        elif isinstance(value, int):
            result = value
            _literal = IntLiteral(span=span_cls, value=value)
        else:
            raise TypeError(f"Invalid Value Type: {value}")

        name = _literal.get_key_name()
        obj = dict(cls_name=name, attributes=dict(span=span_obj, value=result))
        return obj, _literal

    return _build


@pytest.fixture
def build_numerical_type() -> (
    Generator[
        Callable[
            [CoreDataType, list[dict], list[CoreExpression]],
            tuple[dict, NumericalType],
        ],
        None,
        None,
    ]
):
    def inner(
        core: CoreDataType,
        shape_objs: list[dict],
        shape_cls: list[CoreExpression],
    ) -> tuple[dict, NumericalType]:
        """Builds a Numerical Type obj and node."""
        obj = dict(
            cls_name="NumericalType",
            attributes=dict(
                data_type=dict(
                    cls_name="PrimitiveDataType",
                    attributes=dict(core_data_type=core.value),
                ),
                shape=shape_objs,
            ),
        )

        numerical = NumericalType(
            data_type=PrimitiveDataType(core_data_type=core), shape=shape_cls
        )

        return obj, numerical

    yield inner


@add_fixture_node
@pytest.fixture
def index_type(core_literal_expression, construct_id) -> tuple[dict, IndexType]:
    text = "index[1:k:1]"
    one_obj, one_cls = core_literal_expression(1)
    upper_obj, upper_cls = construct_id("k")

    obj = dict(
        cls_name="IndexType",
        attributes=dict(
            lower_bound=one_obj,
            upper_bound=upper_obj,
            stride=one_obj,
        ),
    )

    index = IndexType(
        lower_bound=one_cls,
        upper_bound=CoreIdentifierExpression(upper_cls),
        stride=one_cls,
    )

    return obj, index


@add_fixture_node
@pytest.fixture
def tuple_type(construct_id, build_numerical_type) -> tuple[dict, TupleType]:
    text = "tuple ( int32[m], )"
    shape_1_obj, shape_1_id = construct_id("m")
    num_obj, numerical = build_numerical_type(
        CoreDataType.INT32, [shape_1_obj], [CoreIdentifierExpression(shape_1_id)]
    )

    obj = dict(cls_name="TupleType", attributes=dict(types=[num_obj]))

    tup = TupleType(types=[numerical])

    return obj, tup


@add_fixture_node
@pytest.fixture
def qualified(
    span_node, construct_id, build_numerical_type
) -> tuple[dict, QualifiedType]:
    text = "input int32[m, n]"
    span_obj, span_cls = span_node
    shape_1_obj, shape_1_id = construct_id("m")
    shape_2_obj, shape_2_id = construct_id("n")
    num_obj, num_cls = build_numerical_type(
        CoreDataType.INT32,
        [shape_1_obj, shape_2_obj],
        [CoreIdentifierExpression(shape_1_id), CoreIdentifierExpression(shape_2_id)],
    )

    obj = dict(
        cls_name="QualifiedType",
        attributes=dict(
            span=span_obj,
            base_type=num_obj,
            type_qualifier="input",
        ),
    )

    qualified_type = QualifiedType(
        span=span_cls,
        base_type=num_cls,
        type_qualifier=TypeQualifier.INPUT,
    )

    return obj, qualified_type


@add_fixture_node
@pytest.fixture
def arg1(span_node, qualified, construct_id) -> tuple[dict, Argument]:
    text: str = "input int32[m, n] rupaul"
    span_obj, span_cls = span_node
    qtype_obj, qtype_cls = qualified
    arg_id_obj, arg_id = construct_id("rupaul")

    obj = dict(
        cls_name="Argument",
        attributes=dict(
            span=span_obj,
            name=arg_id_obj,
            qualified_type=qtype_obj,
        ),
    )

    arg = Argument(
        span=span_cls,
        name=arg_id,
        qualified_type=qtype_cls,
    )

    return obj, arg


# EXPRESSIONS
@add_fixture_node
@pytest.fixture
def unary(span_node, literals) -> tuple[dict, UnaryExpression]:
    text: str = "-(5)"
    span_obj, span_cls = span_node
    negative = UnaryOperation.NEGATION
    literal_obj, literal_cls = literals(5)

    obj = dict(
        cls_name="UnaryExpression",
        attributes=dict(
            span=span_obj,
            operation=negative.value,
            expression=literal_obj,
        ),
    )

    _unary = UnaryExpression(span=span_cls, operation=negative, expression=literal_cls)

    return obj, _unary


@add_fixture_node
@pytest.fixture
def binary(span_node, literals) -> tuple[dict, BinaryExpression]:
    text: str = "(5 + 10.0)"
    span_obj, span_cls = span_node
    addition = BinaryOperation.ADDITION
    lit_obj_left, lit_cls_left = literals(5)
    lit_obj_right, lit_cls_right = literals(10.0)

    obj = dict(
        cls_name="BinaryExpression",
        attributes=dict(
            span=span_obj,
            operation=addition.value,
            left=lit_obj_left,
            right=lit_obj_right,
        ),
    )

    bin_express = BinaryExpression(
        span=span_cls, operation=addition, left=lit_cls_left, right=lit_cls_right
    )

    return obj, bin_express


@add_fixture_node
@pytest.fixture
def ternary(span_node, literals, binary, unary) -> tuple[dict, TernaryExpression]:
    text: str = "(1 ? (5 + 10.0) : -(5))"
    span_obj, span_cls = span_node
    cond_obj, cond_cls = literals(1)
    binary_obj, binary_cls = binary
    unary_obj, unary_cls = unary

    obj = dict(
        cls_name="TernaryExpression",
        attributes=dict(
            span=span_obj,
            condition=cond_obj,
            true=binary_obj,
            false=unary_obj,
        ),
    )

    expression = TernaryExpression(
        span=span_cls, condition=cond_cls, true=binary_cls, false=unary_cls
    )

    return obj, expression


@add_fixture_node
@pytest.fixture
def tuple_access(
    span_node, literals, construct_id
) -> tuple[dict, TupleAccessExpression]:
    text: str = "A.1"
    span_obj, span_cls = span_node
    one_obj, one_cls = literals(1)
    name_obj, name_cls = construct_id("A")

    obj = dict(
        cls_name="TupleAccessExpression",
        attributes=dict(
            span=span_obj,
            tuple_expression=name_obj,
            element_index=one_obj,
        ),
    )
    access = TupleAccessExpression(
        span=span_cls,
        tuple_expression=name_cls,
        element_index=one_cls,
    )

    return obj, access


@add_fixture_node
@pytest.fixture
def function_call(span_node, construct_id) -> tuple[dict, FunctionExpression]:
    text: str = "funky<>[]()"
    span_obj, span_cls = span_node
    name_obj, name_cls = construct_id("funky")

    obj = dict(
        cls_name="FunctionExpression",
        attributes=dict(
            span=span_obj, function=name_obj, template_types=[], indices=[], args=[]
        ),
    )
    function = FunctionExpression(
        span=span_cls, function=name_cls, template_types=[], indices=[], args=[]
    )

    return obj, function


@add_fixture_node
@pytest.fixture
def array_access(span_node, construct_id) -> tuple[dict, ArrayAccessExpression]:
    text: str = "array[j, k]"
    span_obj, span_cls = span_node
    array_id_obj, array_id_cls = construct_id("array")
    index_1_obj, index_1_cls = construct_id("j")
    index_2_obj, index_2_cls = construct_id("k")

    obj = dict(
        cls_name="ArrayAccessExpression",
        attributes=dict(
            span=span_obj,
            array_expression=array_id_obj,
            indices=[index_1_obj, index_2_obj],
        ),
    )
    array = ArrayAccessExpression(
        span=span_cls,
        array_expression=array_id_cls,
        indices=[index_1_cls, index_2_cls],
    )

    return obj, array


@add_fixture_node
@pytest.fixture
def tuple_express(span_node) -> tuple[dict, TupleExpression]:
    text: str = "(  )"
    span_obj, span_cls = span_node

    obj = dict(
        cls_name="TupleExpression",
        attributes=dict(
            span=span_obj,
            expressions=[],
        ),
    )
    array = TupleExpression(
        span=span_cls,
        expressions=[],
    )

    return obj, array


@add_fixture_node
@pytest.fixture
def id_express(span_node, construct_id) -> tuple[dict, IdentifierExpression]:
    text: str = "name"
    span_obj, span_cls = span_node
    id_obj, id_cls = construct_id("name")

    obj = dict(
        cls_name="IdentifierExpression",
        attributes=dict(
            span=span_obj,
            identifier=id_obj,
        ),
    )
    expression = IdentifierExpression(
        span=span_cls,
        identifier=id_cls,
    )

    return obj, expression


# STATEMENTS
@add_fixture_node
@pytest.fixture
def declaration(
    span_node, qualified, binary, construct_id
) -> tuple[dict, DeclarationStatement]:
    text: str = "input int32[m, n] bar = (5 + 10.0);"
    span_obj, span_cls = span_node
    qtype_obj, qtype_cls = qualified
    binary_obj, binary_cls = binary
    varname_obj, varname_id = construct_id("bar")

    obj = dict(
        cls_name="DeclarationStatement",
        attributes=dict(
            span=span_obj,
            variable_name=varname_obj,
            variable_type=qtype_obj,
            expression=binary_obj,
        ),
    )

    statement = DeclarationStatement(
        span=span_cls,
        variable_name=varname_id,
        variable_type=qtype_cls,
        expression=binary_cls,
    )

    return obj, statement


@add_fixture_node
@pytest.fixture
def express_state(span_node, unary, array_access) -> tuple[dict, ExpressionStatement]:
    text: str = "array[j, k] = -(5);"
    span_obj, span_cls = span_node
    array_obj, array_cls = array_access
    unary_obj, unary_cls = unary

    obj = dict(
        cls_name="ExpressionStatement",
        attributes=dict(
            span=span_obj,
            left=array_obj,
            right=unary_obj,
        ),
    )

    statement = ExpressionStatement(
        span=span_cls,
        left=array_cls,
        right=unary_cls,
    )

    return obj, statement


@add_fixture_node
@pytest.fixture
def iteration_state(span_node, construct_id) -> tuple[dict, ForAllStatement]:
    text: str = "forall (elements) {\n\n}"
    span_obj, span_cls = span_node
    index_obj, index_cls = construct_id("elements")

    obj = dict(
        cls_name="ForAllStatement",
        attributes=dict(span=span_obj, index=index_obj, body=[]),
    )

    statement = ForAllStatement(span=span_cls, index=index_cls, body=[])

    return obj, statement


@add_fixture_node
@pytest.fixture
def select_state(span_node, binary) -> tuple[dict, SelectionStatement]:
    text: str = "if (5 + 10.0) {\n\n}"
    span_obj, span_cls = span_node
    binary_obj, binary_cls = binary

    obj = dict(
        cls_name="SelectionStatement",
        attributes=dict(
            span=span_obj, condition=binary_obj, true_body=[], false_body=[]
        ),
    )

    statement = SelectionStatement(
        span=span_cls, condition=binary_cls, true_body=[], false_body=[]
    )

    return obj, statement


@add_fixture_node
@pytest.fixture
def return_state(span_node, unary) -> tuple[dict, ReturnStatement]:
    text: str = "return -(5);"
    span_obj, span_cls = span_node
    unary_obj, unary_cls = unary

    obj = dict(
        cls_name="ReturnStatement",
        attributes=dict(span=span_obj, expression=unary_obj),
    )

    statement = ReturnStatement(
        span=span_cls,
        expression=unary_cls,
    )

    return obj, statement


@add_fixture_node
@pytest.fixture
def import_node(span_node, construct_id) -> tuple[dict, Import]:
    text: str = "import x.y;"
    span_obj, span_cls = span_node
    id_obj, id_cls = construct_id("x.y")
    obj = dict(cls_name="Import", attributes=dict(span=span_obj, name=id_obj))
    import_statement = Import(span=span_cls, name=id_cls)

    return obj, import_statement


# FUNCTIONS
@add_fixture_node
@pytest.fixture
def operation(
    span_node, arg1, qualified, express_state, construct_id
) -> tuple[dict, Operation]:
    text: str = (
        "op foobar<>(input int32[m, n] rupaul) -> input int32[m, n]"
        " {\n  array[j, k] = -(5);\n}"
    )
    span_obj, span_cls = span_node
    arg1_obj, arg1_cls = arg1
    qtype_obj, qtype_cls = qualified
    name_id_obj, name_id = construct_id("foobar")
    e_state_obj, e_state_cls = express_state

    obj = dict(
        cls_name="Operation",
        attributes=dict(
            span=span_obj,
            name=name_id_obj,
            templates=[],
            args=[arg1_obj],
            body=[e_state_obj],
            return_type=qtype_obj,
        ),
    )

    op = Operation(
        span=span_cls,
        name=name_id,
        args=[arg1_cls],
        body=[e_state_cls],
        return_type=qtype_cls,
    )

    return obj, op


@add_fixture_node
@pytest.fixture
def procedure_with_templates(construct_id) -> tuple[dict, Procedure]:
    text: str = "proc mumu<T>() {\n\n}"
    name_id_obj, name_id = construct_id("mumu")
    tobj, tid = construct_id("T")

    obj = dict(
        cls_name="Procedure",
        attributes=dict(
            name=name_id_obj,
            templates=[
                dict(cls_name="TemplateDataType", attributes=dict(data_type=tobj))
            ],
            args=[],
            body=[],
        ),
    )
    proc = Procedure(
        span=None,
        name=name_id,
        templates=[TemplateDataType(data_type=tid)],
        args=[],
        body=[],
    )

    return obj, proc


@add_fixture_node
@pytest.fixture
def procedure(span_node, arg1, declaration, construct_id) -> tuple[dict, Procedure]:
    text: str = (
        "proc buzz<>(input int32[m, n] rupaul) "
        "{\n  input int32[m, n] bar = (5 + 10.0);\n}"
    )
    span_obj, span_cls = span_node
    arg1_obj, arg1_cls = arg1
    declare_obj, declare_cls = declaration
    name_id_obj, name_id = construct_id("buzz")

    obj = dict(
        cls_name="Procedure",
        attributes=dict(
            span=span_obj,
            name=name_id_obj,
            templates=[],
            args=[arg1_obj],
            body=[declare_obj],
        ),
    )

    proc = Procedure(
        span=span_cls, name=name_id, templates=[], args=[arg1_cls], body=[declare_cls]
    )

    return obj, proc


# MODULE
@add_fixture_node
@pytest.fixture
def module(span_node, operation, procedure) -> tuple[dict, Module]:
    text: str = (
        "op foobar<>(input int32[m, n] rupaul) -> input int32[m, n]"
        " {\n  array[j, k] = -(5);\n}\n"
        "proc buzz<>(input int32[m, n] rupaul) "
        "{\n  input int32[m, n] bar = (5 + 10.0);\n}"
    )
    span_obj, span_cls = span_node
    op_obj, op_cls = operation
    proc_obj, proc_cls = procedure

    obj = dict(
        cls_name="Module", attributes=dict(span=span_obj, statements=[op_obj, proc_obj])
    )

    module = Module(span=span_cls, statements=[op_cls, proc_cls])

    return obj, module
